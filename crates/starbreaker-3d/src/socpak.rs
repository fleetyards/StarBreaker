//! Socpak reader: opens ship interior containers from P4k, extracts geometry and lights.
//!
//! Flow: P4k → socpak (ZIP) → main .soc (CrCh) → IncludedObjects + CryXMLB → InteriorPayload

use std::collections::HashMap;

use crate::included_objects::IncludedObjects;
use quick_xml::Reader;
use quick_xml::events::Event;
use starbreaker_chunks::ChunkFile;
use starbreaker_chunks::known_types::crch;
use starbreaker_cryxml::CryXml;
use starbreaker_datacore::database::Database;
use starbreaker_datacore::query::value::Value;
use starbreaker_p4k::{MappedP4k, P4kArchive};

use crate::error::Error;
use crate::types::{InteriorMesh, InteriorPayload, LightInfo, LightStateInfo};

const LIGHT_AUTHORED_INTENSITY_SCALE: f32 = 1500.0;
const AMBIENT_PROXY_DIRECT_FACTOR: f32 = 0.25;

// ── DataCore query ──────────────────────────────────────────────────────────

/// Container reference from DataCore VehicleComponentParams.objectContainers[].
#[derive(Debug, Clone)]
pub struct ObjectContainerRef {
    pub bone_name: Option<String>,
    pub file_name: String,
    pub offset_position: [f32; 3],
    pub offset_rotation: [f32; 3], // Ang3 (degrees)
}

/// Query VehicleComponentParams.objectContainers from a ship entity record.
pub fn query_object_containers(
    db: &Database,
    record: &starbreaker_datacore::types::Record,
) -> Vec<ObjectContainerRef> {
    let vehicle_containers = db
        .compile_path::<Value>(
        record.struct_id(),
        "Components[VehicleComponentParams].objectContainers",
    )
        .ok()
        .and_then(|path| db.query::<Value>(&path, record).ok())
        .unwrap_or_default();
    let direct_object_container = db
        .compile_path::<String>(
            record.struct_id(),
            "Components[SObjectContainerComponentParams].objectContainer",
        )
        .ok()
        .and_then(|path| db.query_single::<String>(&path, record).ok().flatten());

    collect_object_container_refs(&vehicle_containers, direct_object_container.as_deref())
}

fn parse_container_ref(val: &Value) -> Option<ObjectContainerRef> {
    let Value::Object { fields, .. } = val else {
        return None;
    };
    // Dump all fields for debugging
    for (k, v) in fields.iter() {
        log::debug!("  container field: {k} = {v:?}");
    }

    let fields: HashMap<&str, &Value> = fields.iter().map(|(k, v)| (*k, v)).collect();

    let file_name = match fields.get("fileName") {
        Some(Value::String(s)) if !s.is_empty() => (*s).to_owned(),
        _ => return None,
    };
    let bone_name = match fields.get("boneName") {
        Some(Value::String(s)) if !s.is_empty() => Some((*s).to_owned()),
        _ => None,
    };
    let (offset_position, offset_rotation) = extract_offset(fields.get("Offset"));

    Some(ObjectContainerRef {
        bone_name,
        file_name,
        offset_position,
        offset_rotation,
    })
}

fn collect_object_container_refs(
    vehicle_containers: &[Value<'_>],
    direct_object_container: Option<&str>,
) -> Vec<ObjectContainerRef> {
    let mut containers = vehicle_containers
        .iter()
        .filter_map(parse_container_ref)
        .collect::<Vec<_>>();
    if let Some(file_name) = direct_object_container.filter(|file_name| !file_name.is_empty()) {
        containers.push(ObjectContainerRef {
            bone_name: None,
            file_name: file_name.to_owned(),
            offset_position: [0.0, 0.0, 0.0],
            offset_rotation: [0.0, 0.0, 0.0],
        });
    }
    containers
}

fn authored_light_intensity_to_candela(intensity_raw: f32) -> f32 {
    intensity_raw * LIGHT_AUTHORED_INTENSITY_SCALE
}

fn authored_light_intensity_to_candela_semantic(
    intensity_raw: f32,
    semantic_light_kind: &str,
    glow_multiplier: f32,
) -> f32 {
    let base = authored_light_intensity_to_candela(intensity_raw);
    if semantic_light_kind.eq_ignore_ascii_case("ambient_proxy") {
        return base * glow_multiplier.max(0.0) * AMBIENT_PROXY_DIRECT_FACTOR;
    }
    base
}

fn extract_offset(offset_val: Option<&&Value>) -> ([f32; 3], [f32; 3]) {
    let mut pos = [0.0f32; 3];
    let mut rot = [0.0f32; 3];

    if let Some(Value::Object { fields, .. }) = offset_val {
        let fields: HashMap<&str, &Value> = fields.iter().map(|(k, v)| (*k, v)).collect();

        if let Some(Value::Object { fields: pf, .. }) = fields.get("Position") {
            let pf: HashMap<&str, &Value> = pf.iter().map(|(k, v)| (*k, v)).collect();
            if let Some(Value::Float(x)) = pf.get("x") {
                pos[0] = *x;
            }
            if let Some(Value::Float(y)) = pf.get("y") {
                pos[1] = *y;
            }
            if let Some(Value::Float(z)) = pf.get("z") {
                pos[2] = *z;
            }
        }
        if let Some(Value::Object { fields: rf, .. }) = fields.get("Rotation") {
            let rf: HashMap<&str, &Value> = rf.iter().map(|(k, v)| (*k, v)).collect();
            if let Some(Value::Float(x)) = rf.get("x") {
                rot[0] = *x;
            }
            if let Some(Value::Float(y)) = rf.get("y") {
                rot[1] = *y;
            }
            if let Some(Value::Float(z)) = rf.get("z") {
                rot[2] = *z;
            }
        }
    }
    (pos, rot)
}

// ── Socpak loading ──────────────────────────────────────────────────────────

/// Load a single socpak and extract its interior geometry + lights.
pub fn load_interior_from_socpak(
    p4k: &MappedP4k,
    socpak_path: &str,
    container_transform: [[f32; 4]; 4],
    non_item_port_transform_delta: [[f32; 4]; 4],
    root_item_port_reference_candidates: &[[[f32; 4]; 4]],
) -> Result<InteriorPayload, Error> {
    let p4k_path = normalize_socpak_path(socpak_path);

    let entry = p4k
        .entry_case_insensitive(&p4k_path)
        .ok_or_else(|| Error::MissingSocpak(p4k_path.clone()))?;

    let socpak_data = p4k
        .read(entry)
        .map_err(|e| Error::P4kRead(format!("{p4k_path}: {e}")))?;

    let inner = P4kArchive::from_bytes(&socpak_data)
        .map_err(|e| Error::P4kRead(format!("ZIP parse {p4k_path}: {e}")))?;

    let name = socpak_path
        .rsplit(&['/', '\\'])
        .next()
        .unwrap_or(socpak_path)
        .strip_suffix(".socpak")
        .unwrap_or(socpak_path)
        .to_string();

    // Parse ALL .soc files in the socpak (main + children).
    // The main .soc has IncludedObjects geometry; child .socs have lights and VFX entities.
    let soc_entries: Vec<_> = inner
        .entries()
        .iter()
        .filter(|e| e.name.to_lowercase().ends_with(".soc"))
        .collect();

    if soc_entries.is_empty() {
        return Err(Error::MissingSocpak(format!("No .soc in {p4k_path}")));
    }

    let mut meshes = Vec::new();
    let mut lights = Vec::new();
    let mut tint_palette_names = Vec::new();

    for soc_entry in &soc_entries {
        let soc_data = match inner.read(soc_entry) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("failed to read {}: {e}", soc_entry.name);
                continue;
            }
        };

        match parse_soc(&soc_data, &soc_entry.name, container_transform) {
            Ok((payload, palette_names)) => {
                log::debug!(
                    "  .soc '{}' → {} meshes, {} lights",
                    soc_entry.name,
                    payload.meshes.len(),
                    payload.lights.len()
                );
                meshes.extend(
                    payload
                        .meshes
                        .into_iter()
                        .map(|mesh| transform_non_item_port_mesh(mesh, non_item_port_transform_delta)),
                );
                lights.extend(
                    payload
                        .lights
                        .into_iter()
                        .map(|light| transform_non_item_port_light(light, non_item_port_transform_delta)),
                );
                if tint_palette_names.is_empty() {
                    tint_palette_names = palette_names;
                }
            }
            Err(e) => {
                log::warn!("failed to parse {}: {e}", soc_entry.name);
            }
        }
    }

    meshes.extend(extract_root_item_port_meshes(
        &inner,
        &name,
        container_transform,
        root_item_port_reference_candidates,
    ));

    Ok(InteriorPayload {
        name,
        parent_entity_name: None,
        parent_node_name: None,
        meshes,
        lights,
        container_transform,
        tint_palette_names,
    })
}

fn normalize_socpak_path(path: &str) -> String {
    let normalized = path.replace('/', "\\");
    if normalized.to_lowercase().starts_with("data\\") {
        normalized
    } else {
        format!("Data\\{normalized}")
    }
}

/// Parse a .soc file's CrCh chunks. Returns meshes/lights + tint palette names.
fn parse_soc(
    data: &[u8],
    name: &str,
    container_transform: [[f32; 4]; 4],
) -> Result<(InteriorPayload, Vec<String>), Error> {
    let cf =
        ChunkFile::from_bytes(data).map_err(|e| Error::ChunkParse(format!("{name}.soc: {e}")))?;

    let ChunkFile::CrCh(crch_file) = &cf else {
        return Err(Error::ChunkParse(format!(
            "{name}.soc: expected CrCh, got IVO"
        )));
    };

    let mut meshes = Vec::new();
    let mut lights = Vec::new();
    let mut palette_names = Vec::new();

    for chunk in crch_file.chunks() {
        let chunk_data = crch_file.chunk_data(chunk);

        match chunk.chunk_type {
            crch::INCLUDED_OBJECTS => match IncludedObjects::from_bytes(chunk_data) {
                Ok(io) => {
                    meshes.extend(included_objects_to_meshes(&io));
                    if palette_names.is_empty() {
                        palette_names = io.tint_palette_paths.clone();
                    }
                }
                Err(e) => log::warn!("failed to parse IncludedObjects in {name}: {e}"),
            },
            crch::CRYXMLB => match starbreaker_cryxml::from_bytes(chunk_data) {
                Ok(xml) => {
                    let (entity_meshes, entity_lights) = extract_cryxml_entities(&xml);
                    meshes.extend(entity_meshes);
                    lights.extend(entity_lights);
                }
                Err(e) => log::warn!("failed to parse CryXMLB in {name}: {e}"),
            },
            _ => {}
        }
    }

    Ok((InteriorPayload {
        name: name.to_string(),
        parent_entity_name: None,
        parent_node_name: None,
        meshes,
        lights,
        container_transform,
        tint_palette_names: Vec::new(), // Set by caller from palette_names
    }, palette_names))
}

fn transform_non_item_port_mesh(
    mut mesh: InteriorMesh,
    non_item_port_transform_delta: [[f32; 4]; 4],
) -> InteriorMesh {
    let delta = glam::Mat4::from_cols_array_2d(&non_item_port_transform_delta);
    mesh.transform = (delta * glam::Mat4::from_cols_array_2d(&mesh.transform)).to_cols_array_2d();
    mesh
}

fn transform_non_item_port_light(
    mut light: LightInfo,
    non_item_port_transform_delta: [[f32; 4]; 4],
) -> LightInfo {
    let delta = glam::Mat4::from_cols_array_2d(&non_item_port_transform_delta);
    let (_, delta_rotation, _) = delta.to_scale_rotation_translation();
    let transformed_position = delta.transform_point3(glam::Vec3::new(
        light.position[0] as f32,
        light.position[1] as f32,
        light.position[2] as f32,
    ));
    light.position = [
        transformed_position.x as f64,
        transformed_position.y as f64,
        transformed_position.z as f64,
    ];
    let local_rotation = glam::Quat::from_xyzw(
        light.rotation[1] as f32,
        light.rotation[2] as f32,
        light.rotation[3] as f32,
        light.rotation[0] as f32,
    );
    let transformed_rotation = (delta_rotation * local_rotation).normalize();
    light.rotation = [
        transformed_rotation.w as f64,
        transformed_rotation.x as f64,
        transformed_rotation.y as f64,
        transformed_rotation.z as f64,
    ];
    let transformed_direction = (delta_rotation
        * glam::Vec3::new(
            light.direction_sc[0] as f32,
            light.direction_sc[1] as f32,
            light.direction_sc[2] as f32,
        ))
    .normalize_or_zero();
    light.direction_sc = [
        transformed_direction.x as f64,
        transformed_direction.y as f64,
        transformed_direction.z as f64,
    ];
    light
}

fn extract_root_item_port_meshes(
    inner: &P4kArchive,
    root_name: &str,
    container_transform: [[f32; 4]; 4],
    root_item_port_reference_candidates: &[[[f32; 4]; 4]],
) -> Vec<InteriorMesh> {
    let root_xml_name = format!("{}.xml", root_name.to_ascii_lowercase());
    let Some(entry) = inner.entries().iter().find(|entry| {
        entry.name.replace('/', "\\").to_ascii_lowercase() == root_xml_name
    }) else {
        return Vec::new();
    };

    let data = match inner.read(entry) {
        Ok(data) => data,
        Err(e) => {
            log::warn!("failed to read {}: {e}", entry.name);
            return Vec::new();
        }
    };

    let root_bounds = parse_editor_bounds(&data).or_else(|| read_root_editor_bounds(inner, root_name));
    let root_item_port_reference_transform = infer_root_item_port_reference_transform(
        inner,
        root_name,
        &data,
        root_item_port_reference_candidates,
    );
    let meshes = extract_item_port_meshes_from_container_xml(
        &data,
        container_transform,
        root_item_port_reference_transform,
    );
    let kept = filter_item_port_meshes_to_editor_bounds(meshes, root_bounds);
    if !kept.is_empty() {
        log::debug!(
            "  root container xml '{}' → {} item-port entities",
            entry.name,
            kept.len()
        );
    }
    kept
}

fn extract_item_port_meshes_from_container_xml(
    data: &[u8],
    container_transform: [[f32; 4]; 4],
    root_item_port_reference_transform: Option<[[f32; 4]; 4]>,
) -> Vec<InteriorMesh> {
    if let Ok(xml) = starbreaker_cryxml::from_bytes(data) {
        return extract_item_port_meshes_from_cryxml(
            &xml,
            container_transform,
            root_item_port_reference_transform,
        );
    }

    let Ok(text) = std::str::from_utf8(data) else {
        return Vec::new();
    };
    extract_item_port_meshes_from_text_xml(
        text,
        container_transform,
        root_item_port_reference_transform,
    )
}

fn extract_item_port_meshes_from_cryxml(
    xml: &CryXml,
    container_transform: [[f32; 4]; 4],
    root_item_port_reference_transform: Option<[[f32; 4]; 4]>,
) -> Vec<InteriorMesh> {
    fn walk(
        xml: &CryXml,
        node: &starbreaker_cryxml::CryXmlNode,
        container_transform: [[f32; 4]; 4],
        root_item_port_reference_transform: Option<[[f32; 4]; 4]>,
        meshes: &mut Vec<InteriorMesh>,
    ) {
        if xml.node_tag(node) == "ItemPort" {
            let attrs: HashMap<&str, &str> = xml.node_attributes(node).collect();
            push_item_port_mesh(
                &attrs,
                container_transform,
                root_item_port_reference_transform,
                meshes,
            );
        }
        for child in xml.node_children(node) {
            walk(
                xml,
                child,
                container_transform,
                root_item_port_reference_transform,
                meshes,
            );
        }
    }

    let mut meshes = Vec::new();
    walk(
        xml,
        xml.root(),
        container_transform,
        root_item_port_reference_transform,
        &mut meshes,
    );
    meshes
}

fn extract_item_port_meshes_from_text_xml(
    text: &str,
    container_transform: [[f32; 4]; 4],
    root_item_port_reference_transform: Option<[[f32; 4]; 4]>,
) -> Vec<InteriorMesh> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut meshes = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.name().as_ref() == b"ItemPort" => {
                let mut attrs: HashMap<String, String> = HashMap::new();
                for attr in e.attributes().flatten() {
                    let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
                    if let Ok(value) = attr.decode_and_unescape_value(reader.decoder()) {
                        attrs.insert(key, value.into_owned());
                    }
                }
                let borrowed: HashMap<&str, &str> = attrs
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.as_str()))
                    .collect();
                push_item_port_mesh(
                    &borrowed,
                    container_transform,
                    root_item_port_reference_transform,
                    &mut meshes,
                );
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => {
                log::warn!("failed to parse container item-port xml: {e}");
                break;
            }
        }
        buf.clear();
    }

    meshes
}

fn push_item_port_mesh(
    attrs: &HashMap<&str, &str>,
    container_transform: [[f32; 4]; 4],
    root_item_port_reference_transform: Option<[[f32; 4]; 4]>,
    meshes: &mut Vec<InteriorMesh>,
) {
    let Some(entity_class_name) = attrs
        .get("name")
        .and_then(|value| normalize_item_port_entity_name(value))
    else {
        return;
    };

    let pos = parse_csv_f64(
        attrs
            .get("interactionOffset")
            .copied()
            .unwrap_or("0,0,0"),
    );
    let rot = parse_csv_f64(attrs.get("rotation").copied().unwrap_or("1,0,0,0"));
    let scale = [1.0, 1.0, 1.0];
    let world_transform = glam::Mat4::from_cols_array_2d(&pos_rot_scale_to_4x4(&pos, &rot, &scale));
    // `resourceLinkToParent` describes resource linkage, not the space the
    // offset is authored in — the Corsair's tail container has `Port0` (link=1)
    // and `Port3` (link=0) sharing one `interactionOffset`. Every ItemPort in a
    // root container XML is authored against the root entity, so all of them
    // convert through the root reference when one was resolved. Treating the
    // unlinked ones as container-space left them carrying the container's own
    // bone offset twice: the Corsair's ramp door panels ended up 16.5m behind
    // the hull. `container_transform` is only correct when no root reference
    // exists.
    let reference_transform = root_item_port_reference_transform.unwrap_or(container_transform);
    let reference_transform = glam::Mat4::from_cols_array_2d(&reference_transform);
    let transform = (reference_transform.inverse() * world_transform).to_cols_array_2d();
    meshes.push(InteriorMesh {
        cgf_path: String::new(),
        material_path: None,
        transform,
        entity_class_guid: None,
        entity_class_name: Some(entity_class_name),
    });
}

fn normalize_item_port_entity_name(port_name: &str) -> Option<String> {
    let candidate = if let Some(rest) = port_name.strip_prefix("Port") {
        let digit_count = rest.chars().take_while(|ch| ch.is_ascii_digit()).count();
        if digit_count > 0 && rest.get(digit_count..digit_count + 1) == Some("_") {
            &rest[digit_count + 1..]
        } else {
            port_name
        }
    } else {
        port_name
    };

    let candidate = candidate.split('[').next()?.trim();
    if candidate.is_empty() {
        return None;
    }

    let candidate = if let Some((base, suffix)) = candidate.rsplit_once('-') {
        if !base.is_empty() && !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()) {
            base
        } else {
            candidate
        }
    } else {
        candidate
    };

    Some(candidate.to_string())
}

fn infer_root_item_port_reference_transform(
    inner: &P4kArchive,
    root_name: &str,
    root_xml_data: &[u8],
    root_item_port_reference_candidates: &[[[f32; 4]; 4]],
) -> Option<[[f32; 4]; 4]> {
    if root_item_port_reference_candidates.is_empty() {
        return None;
    }
    if root_item_port_reference_candidates.len() == 1 {
        return root_item_port_reference_candidates.first().copied();
    }
    let bounds = parse_editor_bounds(root_xml_data).or_else(|| read_root_editor_bounds(inner, root_name))?;
    let item_ports = extract_item_port_meshes_from_container_xml(
        root_xml_data,
        glam::Mat4::IDENTITY.to_cols_array_2d(),
        None,
    );
    if item_ports.is_empty() {
        return root_item_port_reference_candidates.first().copied();
    }
    root_item_port_reference_candidates
        .iter()
        .copied()
        .max_by(|left, right| {
            let left_score = score_reference_transform_bounds(*left, &item_ports, bounds);
            let right_score = score_reference_transform_bounds(*right, &item_ports, bounds);
            left_score
                .0
                .cmp(&right_score.0)
                .then_with(|| right_score.1.total_cmp(&left_score.1))
        })
}

fn read_root_editor_bounds(
    inner: &P4kArchive,
    root_name: &str,
) -> Option<([f32; 3], [f32; 3])> {
    let editor_xml_name = format!("{}_editor.xml", root_name.to_ascii_lowercase());
    let entry = inner.entries().iter().find(|entry| {
        entry.name.replace('/', "\\").to_ascii_lowercase() == editor_xml_name
    })?;
    let data = inner.read(entry).ok()?;
    parse_editor_bounds(&data)
}

fn parse_editor_bounds(data: &[u8]) -> Option<([f32; 3], [f32; 3])> {
    if let Ok(xml) = starbreaker_cryxml::from_bytes(data) {
        let attrs: HashMap<&str, &str> = xml.node_attributes(xml.root()).collect();
        return Some((
            parse_vec3_csv(attrs.get("minBounds").copied()?)?,
            parse_vec3_csv(attrs.get("maxBounds").copied()?)?,
        ));
    }
    let text = std::str::from_utf8(data).ok()?;
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) if e.name().as_ref() == b"ObjectContainer" => {
                let mut min_bounds = None;
                let mut max_bounds = None;
                for attr in e.attributes().flatten() {
                    let key = attr.key.as_ref();
                    let Ok(value) = attr.decode_and_unescape_value(reader.decoder()) else {
                        continue;
                    };
                    if key == b"minBounds" {
                        min_bounds = Some(parse_vec3_csv(value.as_ref()));
                    } else if key == b"maxBounds" {
                        max_bounds = Some(parse_vec3_csv(value.as_ref()));
                    }
                }
                return Some((min_bounds.flatten()?, max_bounds.flatten()?));
            }
            Ok(Event::Eof) => return None,
            Ok(_) => {}
            Err(_) => return None,
        }
        buf.clear();
    }
}

fn parse_vec3_csv(text: &str) -> Option<[f32; 3]> {
    let values = parse_csv_f64(text);
    Some([
        *values.first()? as f32,
        *values.get(1)? as f32,
        *values.get(2)? as f32,
    ])
}

fn filter_item_port_meshes_to_editor_bounds(
    meshes: Vec<InteriorMesh>,
    bounds: Option<([f32; 3], [f32; 3])>,
) -> Vec<InteriorMesh> {
    let Some((min_bounds, max_bounds)) = bounds else {
        return meshes;
    };
    meshes
        .into_iter()
        .filter(|mesh| {
            let translation =
                glam::Mat4::from_cols_array_2d(&mesh.transform).transform_point3(glam::Vec3::ZERO);
            let coords = [translation.x, translation.y, translation.z];
            (0..3).all(|axis| coords[axis] >= min_bounds[axis] && coords[axis] <= max_bounds[axis])
        })
        .collect()
}

fn score_reference_transform_bounds(
    reference_transform: [[f32; 4]; 4],
    item_ports: &[InteriorMesh],
    bounds: ([f32; 3], [f32; 3]),
) -> (usize, f32) {
    let reference_transform = glam::Mat4::from_cols_array_2d(&reference_transform);
    let (min_bounds, max_bounds) = bounds;
    let mut inside = 0usize;
    let mut overflow = 0.0f32;
    for port in item_ports {
        let world = glam::Mat4::from_cols_array_2d(&port.transform);
        let local = reference_transform.inverse() * world;
        let translation = local.transform_point3(glam::Vec3::ZERO);
        let coords = [translation.x, translation.y, translation.z];
        let mut within_bounds = true;
        for axis in 0..3 {
            if coords[axis] < min_bounds[axis] {
                within_bounds = false;
                overflow += min_bounds[axis] - coords[axis];
            } else if coords[axis] > max_bounds[axis] {
                within_bounds = false;
                overflow += coords[axis] - max_bounds[axis];
            }
        }
        if within_bounds {
            inside += 1;
        }
    }
    (inside, overflow)
}

// ── IncludedObjects → InteriorMesh ──────────────────────────────────────────

fn included_objects_to_meshes(io: &IncludedObjects) -> Vec<InteriorMesh> {
    log::debug!(
        "  IncludedObjects: {} CGFs, {} objects, {} materials, {} palettes",
        io.cgf_paths.len(),
        io.objects.len(),
        io.material_paths.len(),
        io.tint_palette_paths.len()
    );
    for (i, path) in io.cgf_paths.iter().enumerate() {
        log::debug!("    CGF[{i}]: {path}");
    }
    for (i, path) in io.tint_palette_paths.iter().enumerate() {
        log::debug!("    Palette[{i}]: {path}");
    }
    for obj in &io.objects {
        let path = io
            .cgf_paths
            .get(obj.cgf_index as usize)
            .map(|s| s.as_str())
            .unwrap_or("??");
        let filename = path.rsplit('/').next().unwrap_or(path);
        log::debug!(
            "    obj: cgf_index={}, unknown2={:#x}, v1=[{:.1},{:.1},{:.1}], v2=[{:.1},{:.1},{:.1}] → {}",
            obj.cgf_index,
            obj.unknown2,
            obj.vector1[0],
            obj.vector1[1],
            obj.vector1[2],
            obj.vector2[0],
            obj.vector2[1],
            obj.vector2[2],
            filename
        );
    }

    io.objects
        .iter()
        .filter_map(|obj| {
            let cgf_path = io.cgf_paths.get(obj.cgf_index as usize)?.clone();
            let transform = f64_3x4_to_f32_4x4(&obj.transform);
            Some(InteriorMesh {
                cgf_path,
                material_path: None,
                transform,
                entity_class_guid: None,
                entity_class_name: None,
            })
        })
        .collect()
}

fn f64_3x4_to_f32_4x4(m: &[[f64; 3]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32, 0.0],
        [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32, 0.0],
        [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32, 0.0],
        [m[3][0] as f32, m[3][1] as f32, m[3][2] as f32, 1.0],
    ]
}

// ── CryXMLB entity extraction ───────────────────────────────────────────────

/// Entity classes to skip (non-visual game logic).
const SKIP_ENTITY_CLASSES: &[&str] = &[
    "ActionArea",
    "AudioAreaAmbience",
    "AudioEnvironmentFeedbackPoint",
    "AudioTriggerSpot",
    "AreaShape",
    "CameraSource",
    "ColorGradient",
    "CommentEntity",
    "EditorCamera",
    "EnvironmentLight",
    "FlographEntity",
    "FogVolume",
    "GravityArea",
    "GravityBox",
    "GreenZone",
    "Hazard",
    "Hint",
    // NOTE: do NOT skip "Ladder" — in CryEngine the `Ladder` entity is an
    // interactive ladder with real visible geometry, not a non-visual area
    // trigger. Skipping it dropped `drak_clipper_lift_access_ladder_01.cgf`,
    // `drak_clipper_lift_access_ladder_hatch_01.cga`, and
    // `drak_clipper_cargo_hold_ladder_01.cgf` from the Clipper interior export.
    "LandingArea",
    "LedgeObject",
    "LocationManager",
    "MusicArea",
    "NavigationArea",
    "ParticleEffect",
    "ParticleField",
    "PlanetAreaEntity",
    "ProceduralPointOfInterestProxy",
    "Room",
    "RoomConnector",
    "SafeTeleportPoint",
    "SCShop",
    "SequenceObjectItem",
    "ShadowRegionEntity",
    "SurfaceRaindropsTarget",
    "TagPoint",
    "TransitDestination",
    "TransitGateway",
    "TransitManager",
    "TransitNavSpline",
    "VibrationAudioPoint",
    "VehicleAudioPoint",
];

fn extract_cryxml_entities(xml: &CryXml) -> (Vec<InteriorMesh>, Vec<LightInfo>) {
    let mut meshes = Vec::new();
    let mut lights = Vec::new();

    let root = xml.root();
    let root_tag = xml.node_tag(root);

    // Find <Entities> or <SCOC_Entities> container
    let entities_node = xml.node_children(root).find(|child| {
        let tag = xml.node_tag(child);
        tag == "Entities" || tag == "SCOC_Entities"
    });

    if let Some(container) = entities_node {
        process_entity_children(xml, container, &mut meshes, &mut lights);
    } else if root_tag == "Entities" || root_tag == "SCOC_Entities" {
        process_entity_children(xml, root, &mut meshes, &mut lights);
    }

    (meshes, lights)
}

fn process_entity_children(
    xml: &CryXml,
    parent: &starbreaker_cryxml::CryXmlNode,
    meshes: &mut Vec<InteriorMesh>,
    lights: &mut Vec<LightInfo>,
) {
    for entity in xml.node_children(parent) {
        if xml.node_tag(entity) != "Entity" {
            continue;
        }

        let attrs: HashMap<&str, &str> = xml.node_attributes(entity).collect();
        let entity_class = attrs.get("EntityClass").copied().unwrap_or("");

        // Skip non-visual entities
        if SKIP_ENTITY_CLASSES.contains(&entity_class)
            || entity_class.starts_with("Door_Ship_Sensor")
            || entity_class.starts_with("ChipSet_Light")
        {
            continue;
        }

        // Log all attributes for debugging
        let entity_name = attrs.get("Name").copied().unwrap_or("?");
        log::trace!(
            "  CryXML entity: class={entity_class} name={entity_name} attrs={:?}",
            attrs.keys().collect::<Vec<_>>()
        );

        let pos = parse_csv_f64(attrs.get("Pos").copied().unwrap_or("0,0,0"));
        let rot = parse_csv_f64(attrs.get("Rotate").copied().unwrap_or("1,0,0,0"));
        let scale = parse_csv_f64(attrs.get("Scale").copied().unwrap_or("1,1,1"));

        if entity_class == "Light"
            || entity_class == "LightBox"
            || entity_class == "LightGroup"
            || entity_class == "LightGroupPoweredItem"
        {
            let parsed_lights =
                parse_light_entities(xml, entity, &attrs, &pos, &rot, &scale, entity_class);
            lights.extend(parsed_lights);
            continue;
        }

        // Try to extract geometry path from inline PropertiesDataCore
        let transform = pos_rot_scale_to_4x4(&pos, &rot, &scale);
        let material_path = attrs.get("Material").map(|s| s.to_string());

        if let Some(geom_path) = extract_entity_geometry(xml, entity) {
            meshes.push(InteriorMesh {
                cgf_path: geom_path,
                material_path,
                transform,
                entity_class_guid: None,
                entity_class_name: None,
            });
        } else if let Some(guid) = attrs.get("EntityClassGUID") {
            // No inline geometry — resolve via DataCore using EntityClassGUID
            meshes.push(InteriorMesh {
                cgf_path: String::new(),
                material_path,
                transform,
                entity_class_guid: Some(guid.to_string()),
                entity_class_name: None,
            });
        }
    }
}

fn extract_entity_geometry(
    xml: &CryXml,
    entity: &starbreaker_cryxml::CryXmlNode,
) -> Option<String> {
    // PropertiesDataCore → EntityGeometryResource → Geometry → Geometry → Geometry → @path
    for child in xml.node_children(entity) {
        if xml.node_tag(child) != "PropertiesDataCore" {
            continue;
        }
        for prop in xml.node_children(child) {
            if xml.node_tag(prop) != "EntityGeometryResource" {
                continue;
            }
            for g1 in xml.node_children(prop) {
                if xml.node_tag(g1) != "Geometry" {
                    continue;
                }
                for g2 in xml.node_children(g1) {
                    if xml.node_tag(g2) != "Geometry" {
                        continue;
                    }
                    for g3 in xml.node_children(g2) {
                        if xml.node_tag(g3) != "Geometry" {
                            continue;
                        }
                        let inner_attrs: HashMap<&str, &str> = xml.node_attributes(g3).collect();
                        if let Some(path) = inner_attrs.get("path")
                            && !path.is_empty()
                        {
                            return Some(path.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Parse light properties from CryXML entity.
///
/// Handles both `Light` (single light in PropertiesDataCore/EntityComponentLight)
/// and `LightGroup` (multiple baked-in lights in EntityComponentLightGroup).
fn parse_light_entities(
    xml: &CryXml,
    entity: &starbreaker_cryxml::CryXmlNode,
    attrs: &HashMap<&str, &str>,
    pos: &[f64],
    rot: &[f64],
    scale: &[f64],
    entity_class: &str,
) -> Vec<LightInfo> {
    let base_name = attrs.get("Name").unwrap_or(&"Light").to_string();
    let base_pos = [
        pos.first().copied().unwrap_or(0.0),
        pos.get(1).copied().unwrap_or(0.0),
        pos.get(2).copied().unwrap_or(0.0),
    ];
    let base_rot = [
        rot.first().copied().unwrap_or(1.0),
        rot.get(1).copied().unwrap_or(0.0),
        rot.get(2).copied().unwrap_or(0.0),
        rot.get(3).copied().unwrap_or(0.0),
    ];
    let base_scale = [
        scale.first().copied().unwrap_or(1.0),
        scale.get(1).copied().unwrap_or(1.0),
        scale.get(2).copied().unwrap_or(1.0),
    ];

    if entity_class == "LightGroup" || has_light_group_component(xml, entity) {
        // LightGroup: EntityComponentLightGroup > BakedInLights > Light[]
        // Each Light child has its own EntityComponentLight
        return parse_light_group(xml, entity, &base_name, &base_pos, &base_rot, &base_scale);
    }

    // Single Light: PropertiesDataCore > EntityComponentLight
    if let Some(lc) = find_entity_component_light(xml, entity) {
        if let Some(light) =
            build_light_info_from_component(xml, &lc, &base_name, &base_pos, &base_rot)
        {
            return vec![light];
        }
    }

    // No light component/group payload means this entity is not an authored
    // runtime light source for export.
    Vec::new()
}

fn has_light_group_component(xml: &CryXml, entity: &starbreaker_cryxml::CryXmlNode) -> bool {
    for child in xml.node_children(entity) {
        if xml.node_tag(child) == "EntityComponentLightGroup" {
            return true;
        }
        if xml.node_tag(child) == "PropertiesDataCore"
            && xml
                .node_children(child)
                .any(|component| xml.node_tag(component) == "EntityComponentLightGroup")
        {
            return true;
        }
    }
    false
}

/// Find PropertiesDataCore > EntityComponentLight in a CryXML entity.
fn find_entity_component_light<'a>(
    xml: &'a CryXml,
    entity: &'a starbreaker_cryxml::CryXmlNode,
) -> Option<&'a starbreaker_cryxml::CryXmlNode> {
    for child in xml.node_children(entity) {
        if xml.node_tag(child) != "PropertiesDataCore" {
            continue;
        }
        for prop in xml.node_children(child) {
            if xml.node_tag(prop) == "EntityComponentLight" {
                return Some(prop);
            }
        }
    }
    None
}

/// Parse a LightGroup entity with baked-in lights.
fn parse_light_group(
    xml: &CryXml,
    entity: &starbreaker_cryxml::CryXmlNode,
    base_name: &str,
    base_pos: &[f64; 3],
    base_rot: &[f64; 4],
    base_scale: &[f64; 3],
) -> Vec<LightInfo> {
    let mut lights = Vec::new();

    // EntityComponentLightGroup > BakedInLights > Light[]
    // Also check direct children (PropertiesDataCore > EntityComponentLightGroup)
    for child in xml.node_children(entity) {
        let tag = xml.node_tag(child);
        let lg_node = if tag == "EntityComponentLightGroup" {
            child
        } else if tag == "PropertiesDataCore" {
            // Sometimes nested under PropertiesDataCore
            match xml
                .node_children(child)
                .find(|c| xml.node_tag(c) == "EntityComponentLightGroup")
            {
                Some(n) => n,
                None => continue,
            }
        } else {
            continue;
        };

        for baked in xml.node_children(lg_node) {
            if xml.node_tag(baked) != "BakedInLights" {
                continue;
            }
            let mut idx = 0;
            for light_node in xml.node_children(baked) {
                if xml.node_tag(light_node) != "Light" {
                    continue;
                }
                let light_name = format!("{base_name}-{idx:03}");

                // Each baked-in Light node has a RelativeXForm child with
                // per-light translation/rotation offsets relative to the group.
                let (rel_translation, rel_rotation, rel_scale) =
                    extract_relative_xform(xml, light_node);
                let rel_translation_scaled = [
                    rel_translation[0] * base_scale[0] * rel_scale[0],
                    rel_translation[1] * base_scale[1] * rel_scale[1],
                    rel_translation[2] * base_scale[2] * rel_scale[2],
                ];
                let rel_translation_world =
                    quat_rotate_vec(base_rot, &rel_translation_scaled);

                // Combine group position with per-light offset
                let light_pos = [
                    base_pos[0] + rel_translation_world[0],
                    base_pos[1] + rel_translation_world[1],
                    base_pos[2] + rel_translation_world[2],
                ];
                let light_rot = quat_mul(base_rot, &rel_rotation);

                // Each Light node has its own EntityComponentLight child
                for lc_child in xml.node_children(light_node) {
                    if xml.node_tag(lc_child) == "EntityComponentLight" {
                        if let Some(light) = build_light_info_from_component(
                            xml,
                            lc_child,
                            &light_name,
                            &light_pos,
                            &light_rot,
                        ) {
                            lights.push(light);
                        }
                        break;
                    }
                }
                idx += 1;
            }
        }
    }

    if lights.is_empty() {
        // Fallback
        lights.push(LightInfo {
            name: base_name.to_string(),
            position: *base_pos,
            transform_basis: "cryengine_z_up".to_string(),
            rotation: *base_rot,
            direction_sc: quat_rotate_vec(base_rot, &[1.0, 0.0, 0.0]),
            color: [1.0, 0.95, 0.9],
            light_type: "Omni".to_string(),
            semantic_light_kind: "point".to_string(),
            intensity_raw: 1.0,
            intensity_unit: "cryengine_authored_intensity".to_string(),
            intensity_candela_proxy: authored_light_intensity_to_candela(1.0),
            intensity: authored_light_intensity_to_candela(1.0),
            radius: 5.0,
            radius_m: 5.0,
            inner_angle: None,
            outer_angle: None,
            projector_texture: None,
            active_state: String::new(),
            states: std::collections::BTreeMap::new(),
        });
    }
    lights
}

/// Build a LightInfo from an EntityComponentLight CryXML node.
///
/// Expected structure:
/// ```xml
/// <EntityComponentLight lightType="..." intensity="..." radius="..." color="r,g,b" ...>
///   <projectorParams texture="..." fov="..." nearPlane="..." />
///   <colorParams colorTemperature="..." />
///   <optionalParams .../>
/// </EntityComponentLight>
/// ```
fn build_light_info_from_component(
    xml: &CryXml,
    component: &starbreaker_cryxml::CryXmlNode,
    name: &str,
    pos: &[f64; 3],
    rot: &[f64; 4],
) -> Option<LightInfo> {
    // The EntityComponentLight carries top-level fields (lightType,
    // useTemperature, etc.) directly. State-specific values (intensity,
    // temperature, and the <color r g b> child element) live on dedicated
    // state children: offState / defaultState / auxiliaryState /
    // emergencyState / cinematicState. Star Citizen renders the "default"
    // state for baked-in lights, so we read from <defaultState> only.
    let component_attrs: HashMap<&str, &str> = xml
        .node_attributes(component)
        .filter(|(k, _)| *k != "__type")
        .collect();

    let bool_truthy = |s: &str| matches!(s, "1" | "true" | "True" | "TRUE");
    let use_temperature = component_attrs
        .get("useTemperature")
        .map(|s| bool_truthy(s))
        .unwrap_or(false);
    let light_type = component_attrs
        .get("lightType")
        .copied()
        .unwrap_or("Omni")
        .to_string();
    let semantic_light_kind = semantic_light_kind_for_light(&light_type, None, None);

    // CryEngine light components expose several runtime states
    // (`offState` / `defaultState` / `auxiliaryState` / `emergencyState` /
    // `cinematicState`). Each carries its own intensity, temperature, and
    // <color r g b> child. Collect every authored state so downstream
    // tools can switch between them; then pick the first with
    // intensity > 0 (in fallback order) as the active state to expose on
    // the flat `color` / `intensity` fields.
    const ALL_STATES: &[&str] = &[
        "offState",
        "defaultState",
        "auxiliaryState",
        "emergencyState",
        "cinematicState",
    ];
    const STATE_PRIORITY: &[&str] = &[
        "defaultState",
        "auxiliaryState",
        "emergencyState",
        "cinematicState",
    ];

    let glow_multiplier = xml
        .node_children(component)
        .find(|c| xml.node_tag(c) == "miscParams")
        .and_then(|misc| {
            xml.node_attributes(misc)
                .find(|(k, _)| *k == "glowMultiplier")
                .and_then(|(_, v)| v.parse::<f32>().ok())
        })
        .unwrap_or(1.0);

    let read_state = |tag: &str| -> Option<LightStateInfo> {
        let node = xml
            .node_children(component)
            .find(|c| xml.node_tag(c) == tag)?;
        let a: HashMap<&str, &str> = xml
            .node_attributes(node)
            .filter(|(k, _)| *k != "__type")
            .collect();
        let intensity_raw = a
            .get("intensity")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let temperature = a
            .get("temperature")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(6500.0);
        let light_style = a
            .get("lightStyle")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let preset_tag = a.get("presetTag").copied().unwrap_or("").to_string();
        let (cr, cg, cb) = xml
            .node_children(node)
            .find(|c| xml.node_tag(c) == "color")
            .map(|c| {
                let ca: HashMap<&str, &str> = xml
                    .node_attributes(c)
                    .filter(|(k, _)| *k != "__type")
                    .collect();
                let f = |k: &str| {
                    ca.get(k)
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1.0)
                        .clamp(0.0, 1.0)
                };
                (f("r"), f("g"), f("b"))
            })
            .unwrap_or((1.0, 1.0, 1.0));
        Some(LightStateInfo {
            intensity_raw,
            intensity_unit: "cryengine_authored_intensity".to_string(),
            intensity_cd: authored_light_intensity_to_candela_semantic(
                intensity_raw,
                semantic_light_kind,
                glow_multiplier,
            ),
            intensity_candela_proxy: authored_light_intensity_to_candela_semantic(
                intensity_raw,
                semantic_light_kind,
                glow_multiplier,
            ),
            temperature,
            use_temperature,
            color: [cr, cg, cb],
            light_style,
            preset_tag,
        })
    };

    let mut states: std::collections::BTreeMap<String, LightStateInfo> =
        std::collections::BTreeMap::new();
    for tag in ALL_STATES {
        if let Some(s) = read_state(tag) {
            states.insert((*tag).to_string(), s);
        }
    }

    // Pick the active state via priority order.
    let active_state_name = STATE_PRIORITY
        .iter()
        .find(|tag| {
            states
                .get(**tag)
                .map(|s| s.intensity_raw > 0.0)
                .unwrap_or(false)
        })
        .copied()
        .unwrap_or("");
    let active = states.get(active_state_name);
    let intensity_raw = active.map(|s| s.intensity_raw).unwrap_or(0.0);
    let temperature = active.map(|s| s.temperature).unwrap_or(6500.0);
    let (color_r, color_g, color_b) = active
        .map(|s| (s.color[0], s.color[1], s.color[2]))
        .unwrap_or((1.0, 1.0, 1.0));

    let color = if use_temperature {
        kelvin_to_rgb(temperature.clamp(1000.0, 40000.0))
    } else {
        [color_r, color_g, color_b]
    };

    // sizeParams > lightRadius (attenuation radius).
    let radius = xml
        .node_children(component)
        .find(|c| xml.node_tag(c) == "sizeParams")
        .and_then(|sp| {
            xml.node_attributes(sp)
                .find(|(k, _)| *k == "lightRadius")
                .and_then(|(_, v)| v.parse::<f32>().ok())
        })
        .filter(|r| *r > 0.0)
        .unwrap_or(5.0);

    // projectorParams > texture, FOV
    let (projector_texture, fov) = xml
        .node_children(component)
        .find(|c| xml.node_tag(c) == "projectorParams")
        .map(|pp| {
            let a: HashMap<&str, &str> = xml
                .node_attributes(pp)
                .filter(|(k, _)| *k != "__type")
                .collect();
            let tex = a
                .get("texture")
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty());
            let fov = a
                .get("FOV")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0);
            (tex, fov)
        })
        .unwrap_or((None, 0.0));

    // Spot light half-angles
    let (inner_angle, outer_angle) = if light_type == "Projector" && fov > 0.0 {
        let outer = fov * 0.5;
        let inner = outer * 0.8;
        (Some(inner), Some(outer))
    } else {
        (None, None)
    };

    // CryEngine authored intensity → exported candela proxy.
    let semantic_light_kind = semantic_light_kind_for_light(&light_type, inner_angle, outer_angle);
    let candela = authored_light_intensity_to_candela_semantic(
        intensity_raw,
        semantic_light_kind,
        glow_multiplier,
    );
    let direction_sc = quat_rotate_vec(rot, &[1.0, 0.0, 0.0]);

    log::debug!(
        "  Light '{name}' type={light_type} useTemp={use_temperature} \
         temperature={temperature} intensity={intensity_raw} radius={radius} color={color:?}"
    );

    Some(LightInfo {
        name: name.to_string(),
        position: *pos,
        transform_basis: "cryengine_z_up".to_string(),
        rotation: *rot,
        direction_sc,
        color,
        light_type,
        semantic_light_kind: semantic_light_kind.to_string(),
        intensity_raw,
        intensity_unit: "cryengine_authored_intensity".to_string(),
        intensity_candela_proxy: candela,
        intensity: candela,
        radius,
        radius_m: radius,
        inner_angle,
        outer_angle,
        projector_texture,
        active_state: active_state_name.to_string(),
        states,
    })
}

fn semantic_light_kind_for_light(
    light_type: &str,
    inner_angle: Option<f32>,
    outer_angle: Option<f32>,
) -> &'static str {
    match light_type.to_ascii_lowercase().as_str() {
        "directional" | "sun" => "sun",
        "planar" | "area" => "area",
        "projector" | "spot" => "spot",
        "ambient" => "ambient_proxy",
        "omni" | "softomni" | "point" => "point",
        _ if inner_angle.unwrap_or(0.0) > 0.0 || outer_angle.unwrap_or(0.0) > 0.0 => "spot",
        _ => "point",
    }
}

/// Convert color temperature in Kelvin to linear sRGB [r, g, b] (0-1).
/// Uses Tanner Helland's algorithm (fast approximation).
fn kelvin_to_rgb(kelvin: f32) -> [f32; 3] {
    let temp = kelvin / 100.0;
    let r = if temp <= 66.0 {
        1.0
    } else {
        let x = temp - 60.0;
        (329.698727446 * x.powf(-0.1332047592) / 255.0).clamp(0.0, 1.0)
    };
    let g = if temp <= 66.0 {
        let x = temp;
        (99.4708025861 * x.ln() - 161.1195681661).clamp(0.0, 255.0) / 255.0
    } else {
        let x = temp - 60.0;
        (288.1221695283 * x.powf(-0.0755148492) / 255.0).clamp(0.0, 1.0)
    };
    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        let x = temp - 10.0;
        (138.5177312231 * x.ln() - 305.0447927307).clamp(0.0, 255.0) / 255.0
    };
    [r, g, b]
}

/// Extract translation and rotation from a baked-in Light node's RelativeXForm child.
/// Returns `([tx,ty,tz], [qw,qx,qy,qz], [sx,sy,sz])` — identity if absent.
fn extract_relative_xform(
    xml: &CryXml,
    light_node: &starbreaker_cryxml::CryXmlNode,
) -> ([f64; 3], [f64; 4], [f64; 3]) {
    for child in xml.node_children(light_node) {
        if xml.node_tag(child) != "RelativeXForm" {
            continue;
        }
        let attrs: HashMap<&str, &str> = xml.node_attributes(child).collect();
        let translation = parse_csv_f64(attrs.get("translation").copied().unwrap_or("0,0,0"));
        let rotation = parse_csv_f64(attrs.get("rotation").copied().unwrap_or("1,0,0,0"));
        let scale = parse_csv_f64(attrs.get("scale").copied().unwrap_or("1,1,1"));
        return (
            [
                translation.first().copied().unwrap_or(0.0),
                translation.get(1).copied().unwrap_or(0.0),
                translation.get(2).copied().unwrap_or(0.0),
            ],
            [
                rotation.first().copied().unwrap_or(1.0),
                rotation.get(1).copied().unwrap_or(0.0),
                rotation.get(2).copied().unwrap_or(0.0),
                rotation.get(3).copied().unwrap_or(0.0),
            ],
            [
                scale.first().copied().unwrap_or(1.0),
                scale.get(1).copied().unwrap_or(1.0),
                scale.get(2).copied().unwrap_or(1.0),
            ],
        );
    }
    ([0.0; 3], [1.0, 0.0, 0.0, 0.0], [1.0; 3])
}

// ── Math helpers ────────────────────────────────────────────────────────────

/// Multiply two quaternions [w, x, y, z].
fn quat_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn quat_rotate_vec(rotation: &[f64; 4], vector: &[f64; 3]) -> [f64; 3] {
    let quat = glam::DQuat::from_xyzw(rotation[1], rotation[2], rotation[3], rotation[0]);
    let rotated = quat * glam::DVec3::new(vector[0], vector[1], vector[2]);
    [rotated.x, rotated.y, rotated.z]
}

fn parse_csv_f64(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .collect()
}

/// Build a 4×4 column-major transform from position `[x,y,z]`, quaternion `[w,x,y,z]`, and scale `[x,y,z]`.
fn pos_rot_scale_to_4x4(pos: &[f64], rot: &[f64], scale: &[f64]) -> [[f32; 4]; 4] {
    let w = rot.first().copied().unwrap_or(1.0) as f32;
    let x = rot.get(1).copied().unwrap_or(0.0) as f32;
    let y = rot.get(2).copied().unwrap_or(0.0) as f32;
    let z = rot.get(3).copied().unwrap_or(0.0) as f32;
    let tx = pos.first().copied().unwrap_or(0.0) as f32;
    let ty = pos.get(1).copied().unwrap_or(0.0) as f32;
    let tz = pos.get(2).copied().unwrap_or(0.0) as f32;
    let sx = scale.first().copied().unwrap_or(1.0) as f32;
    let sy = scale.get(1).copied().unwrap_or(1.0) as f32;
    let sz = scale.get(2).copied().unwrap_or(1.0) as f32;

    let m = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(sx, sy, sz),
        glam::Quat::from_xyzw(x, y, z, w),
        glam::Vec3::new(tx, ty, tz),
    );
    m.to_cols_array_2d()
}


/// Build a 4×4 container transform from position offset and Ang3 rotation (degrees).
pub fn build_container_transform(pos: [f32; 3], rot_deg: [f32; 3]) -> [[f32; 4]; 4] {
    let px = rot_deg[0].to_radians();
    let py = rot_deg[1].to_radians();
    let pz = rot_deg[2].to_radians();
    let (sx, cx) = px.sin_cos();
    let (sy, cy) = py.sin_cos();
    let (sz, cz) = pz.sin_cos();

    // CryEngine Euler rotation order: Z * Y * X (yaw * pitch * roll)
    [
        [cy * cz, cy * sz, -sy, 0.0],
        [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy, 0.0],
        [cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy, 0.0],
        [pos[0], pos[1], pos[2], 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use crate::included_objects::{IncludedObject, IncludedObjects};
    use crate::types::InteriorMesh;
    use starbreaker_datacore::query::value::Value;

    use super::{
        build_container_transform, collect_object_container_refs,
        extract_item_port_meshes_from_text_xml, normalize_item_port_entity_name, parse_container_ref,
        filter_item_port_meshes_to_editor_bounds, quat_mul, quat_rotate_vec,
        semantic_light_kind_for_light,
    };

    fn approx_eq3(left: [f64; 3], right: [f64; 3]) {
        for index in 0..3 {
            assert!(
                (left[index] - right[index]).abs() < 1e-9,
                "component {} mismatch: left={} right={}",
                index,
                left[index],
                right[index]
            );
        }
    }

    #[test]
    fn light_group_relative_translation_respects_group_rotation() {
        let half_turn = std::f64::consts::FRAC_1_SQRT_2;
        let base_rotation = [half_turn, 0.0, 0.0, half_turn];
        let rel_translation = [5.0, 0.0, 0.0];
        let rotated = quat_rotate_vec(&base_rotation, &rel_translation);

        approx_eq3(rotated, [0.0, 5.0, 0.0]);
    }

    #[test]
    fn light_group_relative_rotation_still_composes_after_translation_fix() {
        let half_turn = std::f64::consts::FRAC_1_SQRT_2;
        let base_rotation = [half_turn, 0.0, 0.0, half_turn];
        let rel_rotation = [half_turn, half_turn, 0.0, 0.0];

        let combined = quat_mul(&base_rotation, &rel_rotation);

        approx_eq3([combined[1], combined[2], combined[3]], [0.5, 0.5, 0.5]);
        assert!((combined[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn semantic_light_kind_maps_planar_to_area() {
        assert_eq!(semantic_light_kind_for_light("Planar", None, None), "area");
    }

    #[test]
    fn semantic_light_kind_maps_unknown_angled_light_to_spot() {
        assert_eq!(semantic_light_kind_for_light("Unknown", Some(1.0), Some(2.0)), "spot");
    }

    #[test]
    fn authored_light_intensity_matches_max_script_scale() {
        assert_eq!(super::authored_light_intensity_to_candela(1.0), 1500.0);
        assert_eq!(super::authored_light_intensity_to_candela(2.5), 3750.0);
    }

    #[test]
    fn parse_container_ref_skips_blank_file_name() {
        let value = Value::Object {
            type_name: "SVehicleObjectContainerParams",
            fields: vec![("fileName", Value::String(""))],
            record_id: None,
        };
        assert!(parse_container_ref(&value).is_none());
    }

    #[test]
    fn collect_object_container_refs_includes_direct_object_container_component() {
        let vehicle_container = Value::Object {
            type_name: "SVehicleObjectContainerParams",
            fields: vec![
                ("fileName", Value::String("objectcontainers\\ships\\misc\\hull_c\\base_int_front_main.socpak")),
                ("boneName", Value::String("animated_front")),
            ],
            record_id: None,
        };

        let containers = collect_object_container_refs(
            &[vehicle_container],
            Some("ObjectContainers/Ships/MISC/Hull_C/base_int_back_main.socpak"),
        );

        assert_eq!(containers.len(), 2);
        assert_eq!(
            containers[0].file_name,
            "objectcontainers\\ships\\misc\\hull_c\\base_int_front_main.socpak"
        );
        assert_eq!(
            containers[1].file_name,
            "ObjectContainers/Ships/MISC/Hull_C/base_int_back_main.socpak"
        );
        assert_eq!(containers[1].bone_name, None);
        assert_eq!(containers[1].offset_position, [0.0, 0.0, 0.0]);
        assert_eq!(containers[1].offset_rotation, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn included_objects_use_geometry_authored_materials() {
        let io = IncludedObjects {
            cgf_paths: vec!["objects/props/toolbox.cgf".to_string()],
            material_paths: vec!["materials/container_default".to_string()],
            tint_palette_paths: Vec::new(),
            objects: vec![IncludedObject {
                cgf_index: 0,
                unknown2: 0,
                transform: [[0.0; 3]; 4],
                vector1: [0.0; 3],
                vector2: [0.0; 3],
            }],
        };

        let meshes = super::included_objects_to_meshes(&io);

        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].cgf_path, "objects/props/toolbox.cgf");
        assert_eq!(meshes[0].material_path, None);
    }

    #[test]
    fn ambient_proxy_intensity_applies_glow_and_proxy_factor() {
        let candela =
            super::authored_light_intensity_to_candela_semantic(10.0, "ambient_proxy", 0.2);
        assert_eq!(candela, 750.0);
    }

    #[test]
    fn normalize_item_port_entity_name_strips_port_prefix_and_instance_suffix() {
        assert_eq!(
            normalize_item_port_entity_name(
                "Port3_Door_RN_RoomConnector_Breachable_OpenReverse_Crew_Quarters[int_crew_quarter_a_SETUP]"
            ),
            Some("Door_RN_RoomConnector_Breachable_OpenReverse_Crew_Quarters".to_string())
        );
        assert_eq!(
            normalize_item_port_entity_name(
                "Port1_ControlPanel_Screen_DoorControl_Physical_Cutter_OpenNoneNone-005[int_crew_quarter_a_SETUP]"
            ),
            Some("ControlPanel_Screen_DoorControl_Physical_Cutter_OpenNoneNone".to_string())
        );
    }

    #[test]
    fn extract_item_port_meshes_from_text_xml_emits_named_entity_meshes() {
        let xml = r#"
            <ObjectContainer>
              <TileXmlEntry>
                <TileItemPortEntries>
                  <ItemPort
                    name="Port4_ControlPanel_Screen_DoorControl_Physical_Cutter_OpenLockLight-001[int_crew_quarter_a_SETUP]"
                    interactionOffset="6.5421228,-25.057394,3.61601"
                    rotation="1,0,0,0" />
                </TileItemPortEntries>
              </TileXmlEntry>
            </ObjectContainer>
        "#;

        let meshes = extract_item_port_meshes_from_text_xml(
            xml,
            glam::Mat4::IDENTITY.to_cols_array_2d(),
            None,
        );
        assert_eq!(meshes.len(), 1);
        assert_eq!(
            meshes[0].entity_class_name.as_deref(),
            Some("ControlPanel_Screen_DoorControl_Physical_Cutter_OpenLockLight")
        );
        assert_eq!(meshes[0].entity_class_guid, None);
        assert_eq!(meshes[0].cgf_path, "");
        assert!((meshes[0].transform[3][0] - 6.5421228).abs() < 1e-6);
        assert!((meshes[0].transform[3][1] + 25.057394).abs() < 1e-6);
        assert!((meshes[0].transform[3][2] - 3.61601).abs() < 1e-6);
    }

    #[test]
    fn extract_item_port_meshes_from_text_xml_localizes_against_current_container_transform() {
        let xml = r#"
            <ObjectContainer>
              <TileXmlEntry>
                <TileItemPortEntries>
                  <ItemPort
                    name="Port4_ControlPanel_Screen_DoorControl_Physical_Cutter_OpenLockLight-001[int_crew_quarter_a_SETUP]"
                    interactionOffset="6.5421228,-25.057394,3.61601"
                    rotation="1,0,0,0" />
                </TileItemPortEntries>
              </TileXmlEntry>
            </ObjectContainer>
        "#;

        let meshes = extract_item_port_meshes_from_text_xml(
            xml,
            build_container_transform([-3.875, -26.562, 3.625], [0.0, 0.0, 0.0]),
            None,
        );
        assert_eq!(meshes.len(), 1);
        assert!((meshes[0].transform[3][0] - 10.417123).abs() < 1e-6);
        assert!((meshes[0].transform[3][1] - 1.504606).abs() < 1e-6);
        assert!((meshes[0].transform[3][2] + 0.00899).abs() < 1e-6);
    }

    #[test]
    fn extract_item_port_meshes_from_text_xml_uses_reference_transform_for_linked_ports() {
        let xml = r#"
            <ObjectContainer>
              <TileXmlEntry>
                <TileItemPortEntries>
                  <ItemPort
                    name="Port0_ChipSet_LightControl_Crew_A[int_crew_quarter_a_SETUP]"
                    interactionOffset="7.625,-25.656252,3.6906581"
                    rotation="1,0,0,0"
                    resourceLinkToParent="1" />
                </TileItemPortEntries>
              </TileXmlEntry>
            </ObjectContainer>
        "#;

        let meshes = extract_item_port_meshes_from_text_xml(
            xml,
            build_container_transform([-3.875, -26.562, 3.625], [0.0, 0.0, 0.0]),
            Some(glam::Mat4::IDENTITY.to_cols_array_2d()),
        );
        assert_eq!(meshes.len(), 1);
        assert!((meshes[0].transform[3][0] - 7.625).abs() < 1e-6);
        assert!((meshes[0].transform[3][1] + 25.656252).abs() < 1e-6);
        assert!((meshes[0].transform[3][2] - 3.6906581).abs() < 1e-6);
    }

    #[test]
    fn extract_item_port_meshes_from_text_xml_uses_reference_transform_for_unlinked_ports() {
        // `resourceLinkToParent="0"` does not mean "authored in container space":
        // the Corsair's ramp door panels are unlinked yet still relative to the
        // root, and using the container transform left them offset by the
        // container's bone translation.
        let xml = r#"
            <ObjectContainer>
              <TileXmlEntry>
                <TileItemPortEntries>
                  <ItemPort
                    name="Port5_ControlPanel_Screen_DoorControl_Physical_Corsair_Ramp_Int_Left[int_hold_SETUP]"
                    interactionOffset="4.2099118,-20.362814,-0.25136301"
                    rotation="1,0,0,0"
                    resourceLinkToParent="0" />
                </TileItemPortEntries>
              </TileXmlEntry>
            </ObjectContainer>
        "#;

        let meshes = extract_item_port_meshes_from_text_xml(
            xml,
            glam::Mat4::IDENTITY.to_cols_array_2d(),
            Some(build_container_transform([0.0, -16.5, 0.0], [0.0, 0.0, 0.0])),
        );
        assert_eq!(meshes.len(), 1);
        assert!((meshes[0].transform[3][0] - 4.2099118).abs() < 1e-5);
        assert!((meshes[0].transform[3][1] - -3.862814).abs() < 1e-5);
        assert!((meshes[0].transform[3][2] - -0.25136301).abs() < 1e-5);
    }

    #[test]
    fn filter_item_port_meshes_to_editor_bounds_drops_out_of_bounds_ports() {
        let inside = InteriorMesh {
            cgf_path: String::new(),
            material_path: None,
            transform: glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0))
                .to_cols_array_2d(),
            entity_class_guid: None,
            entity_class_name: Some("InsideControl".to_string()),
        };
        let outside = InteriorMesh {
            cgf_path: String::new(),
            material_path: None,
            transform: glam::Mat4::from_translation(glam::Vec3::new(1.0, 53.0, 3.0))
                .to_cols_array_2d(),
            entity_class_guid: None,
            entity_class_name: Some("OutsideControl".to_string()),
        };

        let filtered = filter_item_port_meshes_to_editor_bounds(
            vec![inside, outside],
            Some(([-5.0, -3.0, -5.0], [5.0, 6.0, 5.0])),
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].entity_class_name.as_deref(), Some("InsideControl"));
    }

}
