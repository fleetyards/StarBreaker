//! Interior mesh discovery, loading, and preloading orchestration.
//!
//! Discovers interior CGF geometry from DataCore (`load_interiors`), builds
//! interior containers from socpak payloads (`build_interiors_from_payloads`),
//! preloads unique CGF meshes in parallel (`preload_interior_meshes`) and their
//! associated texture sets (`preload_interior_textures`). Also contains
//! `tint_palette_hash` (hash for texture cache keys) and
//! `expand_loadout_into_placements` (loadout→interior placement expansion).
//! Public types: `LoadedInteriors`, `InteriorCgfEntry`, `InteriorContainerData`.

use starbreaker_datacore::database::Database;
use starbreaker_datacore::query::value::Value;
use starbreaker_datacore::types::Record;
use starbreaker_p4k::MappedP4k;

use crate::mtl;
use crate::types::MaterialTextures;

use super::*;

pub(crate) struct LoadedInteriors {
    /// Unique CGF entries (deduplicated by path).
    pub unique_cgfs: Vec<InteriorCgfEntry>,
    /// Per-container data (one per socpak).
    pub containers: Vec<InteriorContainerData>,
}

impl Default for LoadedInteriors {
    fn default() -> Self {
        Self {
            unique_cgfs: Vec::new(),
            containers: Vec::new(),
        }
    }
}

/// Metadata for one unique interior CGF (no mesh data).
pub(crate) struct InteriorCgfEntry {
    pub cgf_path: String,
    pub material_path: Option<String>,
    pub name: String,
}

/// One interior container's placement data.
pub(crate) struct InteriorContainerData {
    pub name: String,
    /// Optional child scene instance that owns this interior container.
    pub parent_entity_name: Option<String>,
    /// Optional source node/bone within the parent scene instance.
    pub parent_node_name: Option<String>,
    /// 4×4 column-major transform positioning this container relative to the hull.
    pub container_transform: [[f32; 4]; 4],
    /// Each entry: (index into unique_cgfs, per-object local transform,
    /// optional per-placement tint palette override that takes precedence over
    /// the container's palette). Loadout-attached children resolve their own
    /// palette from the child entity's SGeometryResourceParams so each gadget
    /// tints independently of the parent socpak's tint palette.
    pub placements: Vec<(usize, [[f32; 4]; 4], Option<mtl::TintPalette>)>,
    pub lights: Vec<crate::types::LightInfo>,
    /// Tint palette resolved from the socpak's IncludedObjects tint_palette_paths.
    pub palette: Option<mtl::TintPalette>,
}

/// Discovery pass: parse socpaks to find unique CGF paths and placements.
/// No mesh data is loaded — that happens JIT during GLB packing.
pub(crate) fn load_interiors(
    db: &Database,
    p4k: &MappedP4k,
    record: &Record,
    opts: &ExportOptions,
) -> LoadedInteriors {
    use crate::socpak;

    let containers = socpak::query_object_containers(db, record);
    if containers.is_empty() {
        return LoadedInteriors::default();
    }

    log::info!("Discovering {} interior containers...", containers.len());

    let mut payloads = Vec::new();
    let root_entity_name = db.resolve_string2(record.name_offset).to_string();
    let root_geom_compiled = db
        .compile_path::<String>(
            record.struct_id(),
            "Components[SGeometryResourceParams].Geometry.Geometry.Geometry.path",
        )
        .ok();
    let root_geometry_path = root_geom_compiled
        .as_ref()
        .and_then(|path| db.query_single::<String>(path, record).ok().flatten())
        .unwrap_or_default();
    let root_nmc = (!root_geometry_path.is_empty())
        .then(|| load_nmc_for_cgf(p4k, &root_geometry_path))
        .flatten();
    let container_instances: Vec<_> = containers
        .iter()
        .map(|container| {
            let helper_transform =
                resolve_nmc_helper_transform(root_nmc.as_ref(), container.bone_name.as_deref());
            let container_transform = compose_helper_relative_container_transform(
                container.offset_position,
                container.offset_rotation,
                helper_transform,
            );
            let reference_transform = compose_root_container_transform(
                container.offset_position,
                container.offset_rotation,
                helper_transform,
            );
            let parent_entity_name = helper_transform
                .and(container.bone_name.as_ref().filter(|name| !name.is_empty()))
                .map(|_| root_entity_name.clone());
            let parent_node_name = helper_transform
                .and(container.bone_name.as_ref().filter(|name| !name.is_empty()).cloned());
            (
                container,
                container_transform,
                reference_transform,
                parent_entity_name,
                parent_node_name,
            )
        })
        .collect();
    let mut root_item_port_reference_candidates: std::collections::HashMap<String, Vec<[[f32; 4]; 4]>> =
        std::collections::HashMap::new();
    for (container, _, reference_transform, _, _) in &container_instances {
        root_item_port_reference_candidates
            .entry(container.file_name.to_ascii_lowercase())
            .or_default()
            .push(*reference_transform);
    }
    for (container, container_transform, _, parent_entity_name, parent_node_name) in &container_instances {
        let reference_candidates = root_item_port_reference_candidates
            .get(&container.file_name.to_ascii_lowercase())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        match socpak::load_interior_from_socpak(
            p4k,
            &container.file_name,
            *container_transform,
            glam::Mat4::IDENTITY.to_cols_array_2d(),
            reference_candidates,
        ) {
            Ok(mut payload) => {
                payload.container_transform = normalize_root_light_only_container_transform(
                    payload.container_transform,
                    parent_entity_name.is_some(),
                    &payload,
                );
                payload.parent_entity_name = parent_entity_name.clone();
                payload.parent_node_name = parent_node_name.clone();
                payloads.push(payload);
            }
            Err(e) => log::warn!("failed to load {}: {e}", container.file_name),
        }
    }

    let mut loaded =
        build_interiors_from_payloads(db, p4k, &payloads, opts.include_lights, opts.lod_level);
    let removed = remove_root_geometry_duplicate_interior_placements(
        &mut loaded,
        &root_geometry_path,
        &root_entity_name,
    );
    if removed > 0 {
        log::info!(
            "Skipped {removed} root-geometry duplicate interior placement(s) for '{}'",
            root_geometry_path
        );
    }
    loaded
}

fn helper_node_name_matches(node_name: &str, helper_name: &str) -> bool {
    let node_name = node_name.to_ascii_lowercase();
    let helper_name = helper_name.to_ascii_lowercase();
    node_name == helper_name || node_name.ends_with(&format!("_{helper_name}"))
}

fn interior_container_name_key(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .next()
        .unwrap_or(path)
        .strip_suffix(".socpak")
        .unwrap_or(path)
        .to_ascii_lowercase()
}

fn normalize_geometry_path_key(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

fn remove_root_geometry_duplicate_interior_placements(
    interiors: &mut LoadedInteriors,
    root_geometry_path: &str,
    root_entity_name: &str,
) -> usize {
    let root_key = normalize_geometry_path_key(root_geometry_path);
    if root_key.is_empty() {
        return 0;
    }

    let mut removed = 0usize;
    for container in &mut interiors.containers {
        // Containers bolted onto one of the root entity's own bones (e.g. the
        // Corsair's `nose` interior) still belong to the root hull, so a
        // placement of the root geometry inside them is the same authoring
        // duplicate as in an unparented container. Only a container owned by a
        // *different* entity may legitimately place that geometry.
        if container
            .parent_entity_name
            .as_deref()
            .is_some_and(|name| !name.eq_ignore_ascii_case(root_entity_name))
        {
            continue;
        }
        container.placements.retain(|(mesh_index, _, _)| {
            let duplicate = interiors
                .unique_cgfs
                .get(*mesh_index)
                .is_some_and(|entry| normalize_geometry_path_key(&entry.cgf_path) == root_key);
            if duplicate {
                removed += 1;
            }
            !duplicate
        });
    }

    if removed == 0 {
        return 0;
    }

    let mut used_indices = std::collections::HashSet::<usize>::new();
    for container in &interiors.containers {
        for (mesh_index, _, _) in &container.placements {
            used_indices.insert(*mesh_index);
        }
    }

    let mut remap = std::collections::HashMap::<usize, usize>::new();
    let mut compact = Vec::new();
    for (old_index, entry) in interiors.unique_cgfs.iter().enumerate() {
        if used_indices.contains(&old_index) {
            let new_index = compact.len();
            remap.insert(old_index, new_index);
            compact.push(InteriorCgfEntry {
                cgf_path: entry.cgf_path.clone(),
                material_path: entry.material_path.clone(),
                name: entry.name.clone(),
            });
        }
    }

    for container in &mut interiors.containers {
        for (mesh_index, _, _) in &mut container.placements {
            if let Some(new_index) = remap.get(mesh_index).copied() {
                *mesh_index = new_index;
            }
        }
    }
    interiors.unique_cgfs = compact;

    removed
}

fn compose_root_container_transform(
    offset_position: [f32; 3],
    offset_rotation: [f32; 3],
    helper_transform: Option<glam::Mat4>,
) -> [[f32; 4]; 4] {
    let offset = mat4_from_array(&crate::socpak::build_container_transform(
        offset_position,
        offset_rotation,
    ));
    let base = match helper_transform {
        Some(helper_transform) if helper_transform_duplicates_offset(helper_transform, offset) => offset,
        Some(helper_transform) => helper_transform * offset,
        None => offset,
    };
    mat4_to_array(base)
}

fn compose_helper_relative_container_transform(
    offset_position: [f32; 3],
    offset_rotation: [f32; 3],
    helper_transform: Option<glam::Mat4>,
) -> [[f32; 4]; 4] {
    let offset = mat4_from_array(&crate::socpak::build_container_transform(
        offset_position,
        offset_rotation,
    ));
    let local = match helper_transform {
        Some(helper_transform) if helper_transform_duplicates_offset(helper_transform, offset) => {
            glam::Mat4::IDENTITY
        }
        Some(_) | None => offset,
    };
    mat4_to_array(local)
}

fn helper_transform_duplicates_offset(helper_transform: glam::Mat4, offset_transform: glam::Mat4) -> bool {
    let (helper_scale, helper_rotation, helper_translation) =
        helper_transform.to_scale_rotation_translation();
    let (offset_scale, offset_rotation, offset_translation) =
        offset_transform.to_scale_rotation_translation();
    helper_scale.abs_diff_eq(offset_scale, 1e-3)
        && helper_translation.distance(offset_translation) <= 1.5e-2
        && helper_rotation.angle_between(offset_rotation).abs() < 1e-2
}

fn resolve_nmc_helper_transform(
    nmc: Option<&crate::nmc::NodeMeshCombo>,
    helper_name: Option<&str>,
) -> Option<glam::Mat4> {
    let helper_name = helper_name.filter(|name| !name.is_empty())?;
    let nmc = nmc?;
    let helper_index = nmc
        .nodes
        .iter()
        .position(|node| helper_node_name_matches(&node.name, helper_name))?;
    let helper_world = compute_nmc_world_transforms(nmc);
    let helper_world = helper_world[helper_index];
    let (_, rotation, translation) = helper_world.to_scale_rotation_translation();
    log::debug!(
        "resolved root hull helper '{}' from NMC node '{}' at translation {:?} rotation {:?}",
        helper_name,
        nmc.nodes[helper_index].name,
        translation,
        rotation
    );
    Some(helper_world)
}

fn child_interior_parent_target(
    child: &crate::types::ResolvedNode,
    scene_parent_entity_name: &str,
    override_attachment: Option<&str>,
    container_bone_name: Option<&str>,
) -> (Option<String>, Option<String>) {
    let child_creates_nodes = child.has_geometry || child.nmc.is_some();
    let inherited_attachment = override_attachment
        .filter(|name| !name.is_empty())
        .unwrap_or(&child.attachment_name);
    if child_creates_nodes {
        (
            Some(child.entity_name.clone()),
            container_bone_name
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned),
        )
    } else {
        (
            Some(scene_parent_entity_name.to_string()),
            container_bone_name
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned)
                .or_else(|| (!inherited_attachment.is_empty()).then(|| inherited_attachment.to_string())),
        )
    }
}

fn normalize_root_light_only_container_transform(
    container_transform: [[f32; 4]; 4],
    has_parent_helper: bool,
    payload: &crate::types::InteriorPayload,
) -> [[f32; 4]; 4] {
    if has_parent_helper && payload.meshes.is_empty() && !payload.lights.is_empty() {
        glam::Mat4::IDENTITY.to_cols_array_2d()
    } else {
        container_transform
    }
}

/// Discovery pass for object containers authored on child loadout entities.
pub(crate) fn load_child_interiors(
    db: &Database,
    p4k: &MappedP4k,
    root_entity_name: &str,
    children: &[crate::types::ResolvedNode],
    root_container_names: &std::collections::HashSet<String>,
    opts: &ExportOptions,
) -> LoadedInteriors {
    use crate::socpak;

    let mut payloads = Vec::new();
    fn collect(
        db: &Database,
        p4k: &MappedP4k,
        child: &crate::types::ResolvedNode,
        scene_parent_entity_name: &str,
        override_attachment: Option<&str>,
        root_container_names: &std::collections::HashSet<String>,
        payloads: &mut Vec<crate::types::InteriorPayload>,
    ) {
        let containers = if child.allows_child_object_containers {
            socpak::query_object_containers(db, &child.record)
        } else {
            Vec::new()
        };
        if !containers.is_empty() {
            log::info!(
                "Discovering {} interior containers for child {}...",
                containers.len(),
                child.entity_name
            );
            for container in &containers {
                let container_name = interior_container_name_key(&container.file_name);
                if root_container_names.contains(&container_name) {
                    log::debug!(
                        "skipping duplicate child interior container '{}' on child '{}'",
                        container.file_name,
                        child.entity_name,
                    );
                    continue;
                }
                let container_transform =
                    socpak::build_container_transform(container.offset_position, container.offset_rotation);
                match socpak::load_interior_from_socpak(
                    p4k,
                    &container.file_name,
                    container_transform,
                    glam::Mat4::IDENTITY.to_cols_array_2d(),
                    std::slice::from_ref(&container_transform),
                ) {
                    Ok(mut payload) => {
                        let (parent_entity_name, parent_node_name) = child_interior_parent_target(
                            child,
                            scene_parent_entity_name,
                            override_attachment,
                            container.bone_name.as_deref(),
                        );
                        payload.parent_entity_name = parent_entity_name;
                        payload.parent_node_name = parent_node_name;
                        payloads.push(payload);
                    }
                    Err(e) => log::warn!("failed to load {}: {e}", container.file_name),
                }
            }
        }
        for grandchild in &child.children {
            let child_creates_nodes = child.has_geometry || child.nmc.is_some();
            let inherited_attachment = override_attachment
                .filter(|name| !name.is_empty())
                .unwrap_or(&child.attachment_name);
            collect(
                db,
                p4k,
                grandchild,
                if child_creates_nodes {
                    &child.entity_name
                } else {
                    scene_parent_entity_name
                },
                if child_creates_nodes {
                    None
                } else {
                    Some(inherited_attachment)
                },
                root_container_names,
                payloads,
            );
        }
    }

    for child in children {
        collect(
            db,
            p4k,
            child,
            root_entity_name,
            None,
            root_container_names,
            &mut payloads,
        );
    }

    build_interiors_from_payloads(db, p4k, &payloads, opts.include_lights, opts.lod_level)
}

pub(crate) fn merge_interiors(target: &mut LoadedInteriors, source: LoadedInteriors) {
    use std::collections::HashMap;

    let mut index_by_key: HashMap<(String, Option<String>), usize> = target
        .unique_cgfs
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            (
                (entry.cgf_path.to_ascii_lowercase(), entry.material_path.clone()),
                index,
            )
        })
        .collect();

    for mut container in source.containers {
        for placement in &mut container.placements {
            let entry = &source.unique_cgfs[placement.0];
            let key = (entry.cgf_path.to_ascii_lowercase(), entry.material_path.clone());
            let merged_index = if let Some(index) = index_by_key.get(&key).copied() {
                index
            } else {
                let index = target.unique_cgfs.len();
                target.unique_cgfs.push(InteriorCgfEntry {
                    cgf_path: entry.cgf_path.clone(),
                    material_path: entry.material_path.clone(),
                    name: entry.name.clone(),
                });
                index_by_key.insert(key, index);
                index
            };
            placement.0 = merged_index;
        }
        target.containers.push(container);
    }
}

/// Shared interior building: dedup CGFs, resolve GUIDs, collect placements and lights.
/// Used by both `load_interiors` (from DataCore) and `socpaks_to_glb` (from explicit paths).
pub(crate) fn build_interiors_from_payloads(
    db: &Database,
    p4k: &MappedP4k,
    payloads: &[crate::types::InteriorPayload],
    include_lights: bool,
    lod_level: u32,
) -> LoadedInteriors {
    use std::collections::{HashMap, HashSet};
    use starbreaker_common::CigGuid;
    use std::str::FromStr;

    let guid_geom_compiled = db.compile_rooted::<String>(
        "EntityClassDefinition.Components[SGeometryResourceParams].Geometry.Geometry.Geometry.path",
    ).ok();
    let guid_mtl_compiled = db.compile_rooted::<String>(
        "EntityClassDefinition.Components[SGeometryResourceParams].Geometry.Geometry.Material.path",
    ).ok();

    let mut cgf_cache: HashMap<String, Option<usize>> = HashMap::new();
    let mut unique_cgfs = Vec::new();
    let mut container_data = Vec::new();
    // Cache of parent CGF NMC node tables for helper-bone resolution during
    // loadout expansion. Keyed by lowercase CGF path. Value of None means we
    // tried to load it and failed.
    let mut nmc_cache: HashMap<String, Option<crate::nmc::NodeMeshCombo>> = HashMap::new();
    // Built lazily — only entities that resolve via GUID trigger loadout walks.
    let mut entity_index: Option<starbreaker_datacore::loadout::EntityIndex> = None;

    for payload in payloads {
        log::debug!(
            "  {} → {} meshes, {} lights",
            payload.name,
            payload.meshes.len(),
            payload.lights.len()
        );

        let mut placements = Vec::new();
        let mut placement_keys: HashSet<(String, [u32; 16])> = HashSet::new();

        for im in &payload.meshes {
            let (cgf_path, mtl_path) = if !im.cgf_path.is_empty() {
                (im.cgf_path.clone(), im.material_path.clone())
            } else if let Some(guid_str) = &im.entity_class_guid {
                match resolve_guid_geometry(
                    db,
                    guid_str,
                    guid_geom_compiled.as_ref(),
                    guid_mtl_compiled.as_ref(),
                ) {
                    Some((geom, mtl)) => (geom, Some(mtl).filter(|s| !s.is_empty())),
                    None => {
                        log::debug!("  GUID {guid_str} → no geometry found");
                        continue;
                    }
                }
            } else if let Some(entity_class_name) = &im.entity_class_name {
                let Some(record) = resolve_named_entity_record(db, entity_class_name) else {
                    log::debug!("  entity class {entity_class_name} → no geometry found");
                    continue;
                };
                if is_automatic_door_portal_entity(db, record) {
                    log::debug!("  entity class {entity_class_name} → skipped automatic door portal helper");
                    continue;
                }
                match resolve_record_geometry(
                    db,
                    record,
                    guid_geom_compiled.as_ref(),
                    guid_mtl_compiled.as_ref(),
                ) {
                    Some((geom, mtl)) => (geom, Some(mtl).filter(|s| !s.is_empty())),
                    None => {
                        log::debug!("  entity class {entity_class_name} → no geometry found");
                        continue;
                    }
                }
            } else {
                continue;
            };
            if is_metadata_only_geometry(p4k, &cgf_path, lod_level) {
                log::debug!("  skipping metadata-only interior helper geometry {cgf_path}");
                continue;
            }

            let mesh_idx = *cgf_cache.entry(cgf_path.clone()).or_insert_with(|| {
                let idx = unique_cgfs.len();
                let name = cgf_path
                    .rsplit('/')
                    .next()
                    .unwrap_or(&cgf_path)
                    .strip_suffix(".cgf")
                    .unwrap_or(&cgf_path)
                    .to_string();
                unique_cgfs.push(InteriorCgfEntry {
                    cgf_path: cgf_path.clone(),
                    material_path: mtl_path.clone().or_else(|| im.material_path.clone()),
                    name,
                });
                Some(idx)
            });

            if let Some(idx) = mesh_idx {
                let placement_key = (
                    cgf_path.to_ascii_lowercase(),
                    transform_bits_key(&im.transform),
                );
                if !placement_keys.insert(placement_key) {
                    continue;
                }
                placements.push((idx, im.transform, None));

                // Expand entity loadout attachments. Many interior entities
                // (e.g. fire-extinguisher cabinets, kit lockers) carry their
                // visible body in a child loadout entry attached at a named
                // CryNode helper bone on the parent CGF, rather than on their
                // own SGeometryResourceParams.
                if let Some(guid_str) = &im.entity_class_guid {
                    if let Ok(guid) = CigGuid::from_str(guid_str) {
                        if let Some(parent_record) = db.record_by_id(&guid) {
                            let idx_ref = entity_index.get_or_insert_with(|| {
                                starbreaker_datacore::loadout::EntityIndex::new(db)
                            });
                            let tree = starbreaker_datacore::loadout::resolve_loadout_indexed(
                                idx_ref,
                                parent_record,
                            );
                            if !tree.root.children.is_empty() {
                                expand_loadout_into_placements(
                                    db,
                                    p4k,
                                    &tree.root.children,
                                    mat4_from_array(&im.transform),
                                    &cgf_path,
                                    &mut nmc_cache,
                                    &mut cgf_cache,
                                    &mut unique_cgfs,
                                    &mut placements,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Resolve tint palette from the socpak's IncludedObjects palette names.
        // These are DataCore TintPaletteTree record paths — extract the short name
        // (last path component) and look up the record.
        let palette = payload.tint_palette_names.first().and_then(|path| {
            let short_name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
            let tpt_si = db.struct_id("TintPaletteTree")?;
            let record = db.records_of_type(tpt_si).find(|r| {
                db.resolve_string2(r.name_offset).to_lowercase().ends_with(&short_name)
            })?;
            query_tint_from_record(
                db,
                record,
                Some(short_name),
            )
        });

        if let Some(ref p) = palette {
            log::debug!(
                "  {} palette: primary=[{:.2},{:.2},{:.2}] secondary=[{:.2},{:.2},{:.2}]",
                payload.name, p.primary[0], p.primary[1], p.primary[2],
                p.secondary[0], p.secondary[1], p.secondary[2],
            );
        }

        container_data.push(InteriorContainerData {
            name: payload.name.clone(),
            parent_entity_name: payload.parent_entity_name.clone(),
            parent_node_name: payload.parent_node_name.clone(),
            container_transform: payload.container_transform,
            placements,
            lights: if include_lights { payload.lights.clone() } else { Vec::new() },
            palette,
        });
    }

    log::info!(
        "  {} unique CGFs, {} containers",
        unique_cgfs.len(),
        container_data.len()
    );

    LoadedInteriors {
        unique_cgfs,
        containers: container_data,
    }
}

fn is_metadata_only_geometry(p4k: &MappedP4k, cgf_path: &str, lod_level: u32) -> bool {
    let p4k_geom_path = datacore_path_to_p4k(cgf_path);
    p4k.entry_case_insensitive(&p4k_geom_path).is_some()
        && p4k
            .entry_case_insensitive(&resolve_companion_path(p4k, &p4k_geom_path, lod_level))
            .is_none()
}

/// Resolve an EntityClassGUID to its geometry + material paths via DataCore.
pub(crate) fn resolve_guid_geometry(
    db: &Database,
    guid_str: &str,
    geom_compiled: Option<&starbreaker_datacore::query::compile::CompiledPath>,
    mtl_compiled: Option<&starbreaker_datacore::query::compile::CompiledPath>,
) -> Option<(String, String)> {
    use starbreaker_common::CigGuid;

    let guid = CigGuid::from_str(guid_str).ok()?;
    let record = db.record_by_id(&guid)?;
    let record_name = db.resolve_string2(record.name_offset);
    let struct_name = db.struct_name(record.struct_id());

    let geom_path = match geom_compiled
        .and_then(|compiled| db.query_single::<String>(compiled, record).ok().flatten())
        .or_else(|| {
            // Fallback: try compiling path for this specific struct type
            let compiled = db
                .compile_path::<String>(
                    record.struct_id(),
                    "Components[SGeometryResourceParams].Geometry.Geometry.Geometry.path",
                )
                .ok()?;
            db.query_single::<String>(&compiled, record).ok().flatten()
        }) {
        Some(p) => p,
        None => {
            log::debug!(
                "  GUID {guid_str} → {struct_name}.{record_name} has no geometry component"
            );
            return None;
        }
    };

    if geom_path.is_empty() {
        log::debug!("  GUID {guid_str} → {struct_name}.{record_name} has empty geometry path");
        return None;
    }

    let mtl_path = mtl_compiled
        .and_then(|compiled| db.query_single::<String>(compiled, record).ok().flatten())
        .or_else(|| {
            let compiled = db
                .compile_path::<String>(
                    record.struct_id(),
                    "Components[SGeometryResourceParams].Geometry.Geometry.Material.path",
                )
                .ok()?;
            db.query_single::<String>(&compiled, record).ok().flatten()
        })
        .unwrap_or_default();

    log::debug!("  GUID {guid_str} → {geom_path}");
    Some((geom_path, mtl_path))
}

fn transform_bits_key(transform: &[[f32; 4]; 4]) -> [u32; 16] {
    let mut key = [0u32; 16];
    let mut index = 0;
    for column in transform {
        for value in column {
            key[index] = value.to_bits();
            index += 1;
        }
    }
    key
}

fn entity_class_name_matches_record_short_name(entity_class_name: &str, record_short_name: &str) -> bool {
    let entity_class_name = entity_class_name.to_ascii_lowercase();
    let record_short_name = record_short_name.to_ascii_lowercase();
    entity_class_name == record_short_name
        || entity_class_name.starts_with(&(record_short_name + "_"))
}

fn resolve_named_entity_record<'db>(
    db: &'db Database,
    entity_class_name: &str,
) -> Option<&'db Record> {
    let normalized = entity_class_name.to_ascii_lowercase();
    let entity_si = db.struct_id("EntityClassDefinition")?;

    let mut exact_match = None;
    let mut prefix_match: Option<(&starbreaker_datacore::types::Record, String)> = None;

    for record in db.records_of_type(entity_si) {
        let record_name = db.resolve_string2(record.name_offset);
        let short_name = record_name.rsplit('.').next().unwrap_or(record_name);
        let short_lower = short_name.to_ascii_lowercase();

        if short_lower == normalized {
            exact_match = Some(record);
            break;
        }

        if entity_class_name_matches_record_short_name(entity_class_name, short_name) {
            let replace = prefix_match
                .as_ref()
                .map(|(_, current)| short_lower.len() > current.len())
                .unwrap_or(true);
            if replace {
                prefix_match = Some((record, short_lower));
            }
        }
    }

    exact_match.or_else(|| prefix_match.map(|(record, _)| record))
}

fn resolve_record_geometry(
    db: &Database,
    record: &Record,
    geom_compiled: Option<&starbreaker_datacore::query::compile::CompiledPath>,
    mtl_compiled: Option<&starbreaker_datacore::query::compile::CompiledPath>,
) -> Option<(String, String)> {
    let geom_path = match geom_compiled
        .and_then(|compiled| db.query_single::<String>(compiled, record).ok().flatten())
        .or_else(|| {
            let compiled = db
                .compile_path::<String>(
                    record.struct_id(),
                    "Components[SGeometryResourceParams].Geometry.Geometry.Geometry.path",
                )
                .ok()?;
            db.query_single::<String>(&compiled, record).ok().flatten()
        }) {
        Some(path) if !path.is_empty() => path,
        _ => return None,
    };

    let mtl_path = mtl_compiled
        .and_then(|compiled| db.query_single::<String>(compiled, record).ok().flatten())
        .or_else(|| {
            let compiled = db
                .compile_path::<String>(
                    record.struct_id(),
                    "Components[SGeometryResourceParams].Geometry.Geometry.Material.path",
                )
                .ok()?;
            db.query_single::<String>(&compiled, record).ok().flatten()
        })
        .unwrap_or_default();

    Some((geom_path, mtl_path))
}

fn is_automatic_door_portal_entity(db: &Database, record: &Record) -> bool {
    let Ok(compiled) = db.compile_path::<Value>(
        record.struct_id(),
        "Components[SCItemDoorParams].PortalMode",
    ) else {
        return false;
    };
    let Ok(values) = db.query::<Value>(&compiled, record) else {
        return false;
    };
    values.iter().any(|value| {
        matches!(
            value,
            Value::Object {
                type_name: "SCItemDoorPortalModeAutomaticParams",
                ..
            }
        )
    })
}

/// Walk a loadout subtree, emitting additional `(cgf_idx, transform)` placements
/// for each child entity that has a resolvable geometry path.
///
/// The transform for each child is composed as
/// `parent_world × helper_local_on_parent_cgf × port_offset`.
/// If the parent NMC is missing or the helper bone is not found, the child is
/// still placed using the parent's world transform plus any port offset, so
/// missing geometry is never silently dropped.
pub(crate) fn expand_loadout_into_placements(
    db: &Database,
    p4k: &MappedP4k,
    children: &[starbreaker_datacore::loadout::LoadoutNode],
    parent_world: glam::Mat4,
    parent_cgf_path: &str,
    nmc_cache: &mut std::collections::HashMap<String, Option<crate::nmc::NodeMeshCombo>>,
    cgf_cache: &mut std::collections::HashMap<String, Option<usize>>,
    unique_cgfs: &mut Vec<InteriorCgfEntry>,
    placements: &mut Vec<(usize, [[f32; 4]; 4], Option<mtl::TintPalette>)>,
) {
    if children.is_empty() {
        return;
    }
    // Look up parent NMC once for all children at this level.
    let parent_key = parent_cgf_path.to_ascii_lowercase();
    if !nmc_cache.contains_key(&parent_key) {
        let nmc = load_nmc_for_cgf(p4k, parent_cgf_path);
        nmc_cache.insert(parent_key.clone(), nmc);
    }
    // Clone the NMC out of the cache so we can release the borrow before
    // recursing (children resolve a different parent NMC).
    let parent_nmc: Option<crate::nmc::NodeMeshCombo> =
        nmc_cache.get(&parent_key).and_then(|v| v.clone());

    for child in children {
        let Some(child_geom) = child.geometry_path.as_deref() else {
            // No geometry on this node — but its grandchildren may still have
            // some (e.g. an empty container item that holds tools). Recurse
            // using the parent's transform and CGF as the attachment frame.
            if !child.children.is_empty() {
                expand_loadout_into_placements(
                    db,
                    p4k,
                    &child.children,
                    parent_world,
                    parent_cgf_path,
                    nmc_cache,
                    cgf_cache,
                    unique_cgfs,
                    placements,
                );
            }
            continue;
        };

        let helper_xform = compose_helper_transform(
            parent_nmc.as_ref(),
            child.helper_bone_name.as_deref(),
            child.offset_position,
            child.offset_rotation,
        );
        let child_world = parent_world * helper_xform;

        // Each loadout child resolves its own tint palette from the child
        // entity's SGeometryResourceParams (falling back to a name-matched
        // TintPaletteTree record). Gadgets like fire-extinguisher cabinets
        // need their own red/black palette regardless of the parent socpak's
        // palette.
        let child_palette = query_tint_palette(db, &child.record);

        let geom_owned = child_geom.to_string();
        let mtl_owned = child.material_path.clone();
        let child_idx = *cgf_cache.entry(geom_owned.clone()).or_insert_with(|| {
            let idx = unique_cgfs.len();
            let name = geom_owned
                .rsplit('/')
                .next()
                .unwrap_or(&geom_owned)
                .rsplit_once('.')
                .map(|(stem, _)| stem.to_string())
                .unwrap_or_else(|| geom_owned.clone());
            unique_cgfs.push(InteriorCgfEntry {
                cgf_path: geom_owned.clone(),
                material_path: mtl_owned.clone(),
                name,
            });
            Some(idx)
        });
        if let Some(idx) = child_idx {
            placements.push((idx, mat4_to_array(child_world), child_palette));
        }

        if !child.children.is_empty() {
            expand_loadout_into_placements(
                db,
                p4k,
                &child.children,
                child_world,
                &geom_owned,
                nmc_cache,
                cgf_cache,
                unique_cgfs,
                placements,
            );
        }
    }
}

/// Load a .cgf/.cgfm mesh from P4k by interior path.
/// When `use_model_bbox` is true, dequantizes positions using the model bounding
/// box instead of the scaling bbox (needed for interior CGF placement).
pub(crate) fn export_cgf_from_path(
    p4k: &MappedP4k,
    cgf_path: &str,
    material_path: Option<&str>,
    opts: &ExportOptions,
    png_cache: &mut PngCache,
    use_model_bbox: bool,
) -> Result<EntityPayload, Error> {
    // Strip "data/" prefix if present (CryXMLB paths sometimes include it)
    let clean_path = cgf_path.replace('\\', "/");
    let geometry_path = clean_path
        .strip_prefix("data/")
        .or_else(|| clean_path.strip_prefix("Data/"))
        .unwrap_or(&clean_path);
    let mtl_path = material_path.unwrap_or("");
    export_entity_from_paths_cached(p4k, geometry_path, mtl_path, opts, png_cache, use_model_bbox)
}

pub(crate) fn load_interior_mesh_asset(
    p4k: &MappedP4k,
    entry: &InteriorCgfEntry,
    opts: &ExportOptions,
    png_cache: &mut PngCache,
) -> Option<InteriorMeshAsset> {
    match export_cgf_from_path(
        p4k,
        &entry.cgf_path,
        entry.material_path.as_deref(),
        opts,
        png_cache,
        false,
    ) {
        Ok((mesh, mtl, _tex, nmc, _palette, _, _, _bones, _skeleton_source_path)) => {
            let needs_bake = mesh
                .scaling_min
                .iter()
                .zip(&mesh.model_min)
                .chain(mesh.scaling_max.iter().zip(&mesh.model_max))
                .any(|(s, m)| (s - m).abs() > 0.01);
            let mesh = if needs_bake {
                bake_nmc_into_mesh(mesh, nmc.as_ref(), false)
            } else {
                mesh
            };
            Some((mesh, mtl, nmc))
        }
        Err(e) => {
            log::warn!("failed to load CGF {}: {e}", entry.cgf_path);
            None
        }
    }
}

pub(crate) fn preload_interior_meshes(
    interiors: &LoadedInteriors,
    p4k: &MappedP4k,
    opts: &ExportOptions,
) -> Vec<Option<InteriorMeshAsset>> {
    use rayon::prelude::*;

    interiors
        .unique_cgfs
        .par_iter()
        .map(|entry| {
            let mut png_cache = PngCache::new();
            load_interior_mesh_asset(p4k, entry, opts, &mut png_cache)
        })
        .collect()
}

pub(crate) fn tint_palette_hash(palette: Option<&mtl::TintPalette>) -> u64 {
    use std::hash::{Hash, Hasher};

    let Some(palette) = palette else {
        return 0;
    };

    let mut hasher = std::hash::DefaultHasher::new();
    for color in [palette.primary, palette.secondary, palette.tertiary, palette.glass] {
        color[0].to_bits().hash(&mut hasher);
        color[1].to_bits().hash(&mut hasher);
        color[2].to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn collect_interior_palettes(
    interiors: &LoadedInteriors,
    fallback_palette: Option<&mtl::TintPalette>,
) -> Vec<(u64, Option<mtl::TintPalette>)> {
    let mut seen = std::collections::HashSet::new();
    let mut palettes = Vec::new();

    for container in &interiors.containers {
        let palette = container.palette.as_ref().or(fallback_palette).cloned();
        let palette_hash = tint_palette_hash(palette.as_ref());
        if seen.insert(palette_hash) {
            palettes.push((palette_hash, palette));
        }

        // Per-placement palette overrides (e.g. loadout-attached gadgets that
        // carry their own tint palette via SGeometryResourceParams).
        for (_, _, placement_palette) in &container.placements {
            if let Some(pal) = placement_palette {
                let h = tint_palette_hash(Some(pal));
                if seen.insert(h) {
                    palettes.push((h, Some(pal.clone())));
                }
            }
        }
    }

    palettes
}

pub(crate) fn preload_interior_textures(
    interiors: &LoadedInteriors,
    preloaded_meshes: &[Option<InteriorMeshAsset>],
    fallback_palette: Option<&mtl::TintPalette>,
    p4k: &MappedP4k,
    opts: &ExportOptions,
) -> std::collections::HashMap<PreloadedTextureKey, MaterialTextures> {
    use rayon::prelude::*;

    if !opts.material_mode.include_textures() {
        return std::collections::HashMap::new();
    }

    let palettes = collect_interior_palettes(interiors, fallback_palette);
    if palettes.is_empty() {
        return std::collections::HashMap::new();
    }

    let mut unique_materials = std::collections::HashMap::<String, mtl::MtlFile>::new();
    for asset in preloaded_meshes {
        let Some((_, Some(materials), _)) = asset else {
            continue;
        };
        let Some(source_path) = materials.source_path.as_ref() else {
            continue;
        };
        unique_materials
            .entry(source_path.to_ascii_lowercase())
            .or_insert_with(|| materials.clone());
    }

    if unique_materials.is_empty() {
        return std::collections::HashMap::new();
    }

    let jobs: Vec<(String, mtl::MtlFile, u64, Option<mtl::TintPalette>)> = unique_materials
        .into_iter()
        .flat_map(|(material_source, materials)| {
            palettes.iter().map(move |(palette_hash, palette)| {
                (
                    material_source.clone(),
                    materials.clone(),
                    *palette_hash,
                    palette.clone(),
                )
            })
        })
        .collect();

    jobs
        .into_par_iter()
        .map(|(material_source, materials, palette_hash, palette)| {
            let mut png_cache = PngCache::new();
            let textures = load_material_textures(
                p4k,
                &materials,
                palette.as_ref(),
                opts.texture_mip,
                &mut png_cache,
                opts.material_mode.include_normals(),
                opts.material_mode.experimental(),
            );
            (
                PreloadedTextureKey {
                    material_source,
                    palette_hash,
                },
                textures,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        child_interior_parent_target, compose_helper_relative_container_transform,
        normalize_root_light_only_container_transform,
        compose_root_container_transform, entity_class_name_matches_record_short_name,
        helper_node_name_matches, helper_transform_duplicates_offset,
        remove_root_geometry_duplicate_interior_placements, resolve_nmc_helper_transform,
        transform_bits_key, InteriorCgfEntry, InteriorContainerData, LoadedInteriors,
    };
    use crate::pipeline::nmc_bridge::mat4_from_array;

    #[test]
    fn helper_node_name_matches_prefixed_helper_suffix() {
        assert!(helper_node_name_matches(
            "body_260_helper_crew_quarter_d",
            "helper_crew_quarter_d"
        ));
        assert!(helper_node_name_matches("helper_crew_quarter_d", "helper_crew_quarter_d"));
        assert!(!helper_node_name_matches(
            "body_260_helper_crew_quarter_c",
            "helper_crew_quarter_d"
        ));
    }

    #[test]
    fn compose_root_container_transform_composes_full_helper_transform() {
        let helper_transform = glam::Mat4::from_rotation_translation(
            glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            glam::Vec3::new(1.0, 2.0, 3.0),
        );
        let transform = compose_root_container_transform(
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            Some(helper_transform),
        );
        let transform = mat4_from_array(&transform);
        let expected = helper_transform
            * mat4_from_array(&crate::socpak::build_container_transform(
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ));
        let actual = transform.to_cols_array();
        let expected = expected.to_cols_array();
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn compose_root_container_transform_skips_duplicate_helper_transform() {
        let offset = mat4_from_array(&crate::socpak::build_container_transform(
            [8.0625, -25.4375, 3.625],
            [0.0, 0.0, 180.0],
        ));
        assert!(helper_transform_duplicates_offset(offset, offset));
        let transform = compose_root_container_transform(
            [8.0625, -25.4375, 3.625],
            [0.0, 0.0, 180.0],
            Some(offset),
        );
        let transform = mat4_from_array(&transform);
        let actual = transform.to_cols_array();
        let expected = offset.to_cols_array();
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn compose_root_container_transform_skips_near_duplicate_helper_transform() {
        let helper = glam::Mat4::from_translation(glam::Vec3::new(
            -0.012000000104308128,
            -8.607000350952148,
            -3.332000732421875,
        ));

        let transform = compose_root_container_transform(
            [-0.0015102500328794122, -8.607000350952148, -3.3320000171661377],
            [0.0, 0.0, 0.0],
            Some(helper),
        );
        let transform = mat4_from_array(&transform);
        let expected = mat4_from_array(&crate::socpak::build_container_transform(
            [-0.0015102500328794122, -8.607000350952148, -3.3320000171661377],
            [0.0, 0.0, 0.0],
        ));

        let actual = transform.to_cols_array();
        let expected = expected.to_cols_array();
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn compose_helper_relative_container_transform_returns_identity_for_duplicate_helper() {
        let helper = glam::Mat4::from_translation(glam::Vec3::new(
            -0.012000000104308128,
            -8.607000350952148,
            -3.332000732421875,
        ));
        let transform = compose_helper_relative_container_transform(
            [-0.0015102500328794122, -8.607000350952148, -3.3320000171661377],
            [0.0, 0.0, 0.0],
            Some(helper),
        );
        let transform = mat4_from_array(&transform);
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        assert!(scale.abs_diff_eq(glam::Vec3::ONE, 1e-5));
        assert!(rotation.angle_between(glam::Quat::IDENTITY) < 1e-5);
        assert!(translation.abs_diff_eq(glam::Vec3::ZERO, 1e-5));
    }

    #[test]
    fn compose_helper_relative_container_transform_keeps_offset_for_distinct_helper() {
        let helper = glam::Mat4::from_translation(glam::Vec3::new(0.0, 20.0, 0.0));
        let transform = compose_helper_relative_container_transform(
            [0.0, 35.0, 0.0],
            [0.0, 0.0, 0.0],
            Some(helper),
        );
        let transform = mat4_from_array(&transform);
        let expected = mat4_from_array(&crate::socpak::build_container_transform(
            [0.0, 35.0, 0.0],
            [0.0, 0.0, 0.0],
        ));
        let actual = transform.to_cols_array();
        let expected = expected.to_cols_array();
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn resolve_nmc_helper_transform_reads_helper_node_transform() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let rotation = glam::Quat::from_rotation_z(std::f32::consts::PI);
        let transform = glam::Mat4::from_rotation_translation(rotation, glam::Vec3::new(4.0, 5.0, 6.0));
        let transformed = transform.to_cols_array_2d();
        let nmc = crate::nmc::NodeMeshCombo {
            nodes: vec![crate::nmc::NmcNode {
                name: "helper_crew_quarter_d".to_string(),
                parent_index: None,
                world_to_bone: identity,
                bone_to_world: [
                    [transformed[0][0], transformed[1][0], transformed[2][0], transformed[3][0]],
                    [transformed[0][1], transformed[1][1], transformed[2][1], transformed[3][1]],
                    [transformed[0][2], transformed[1][2], transformed[2][2], transformed[3][2]],
                ],
                scale: [1.0, 1.0, 1.0],
                geometry_type: 2,
                properties: std::collections::HashMap::new(),
            }],
            material_indices: Vec::new(),
        };
        let helper_transform =
            resolve_nmc_helper_transform(Some(&nmc), Some("helper_crew_quarter_d")).unwrap();
        let (scale, resolved_rotation, translation) =
            helper_transform.to_scale_rotation_translation();
        assert!(scale.abs_diff_eq(glam::Vec3::ONE, 1e-5));
        assert!(translation.abs_diff_eq(glam::Vec3::new(4.0, 5.0, 6.0), 1e-5));
        assert!(resolved_rotation.angle_between(rotation) < 1e-5);
    }

    #[test]
    fn entity_class_name_matches_record_short_name_allows_suffix_specialization() {
        assert!(entity_class_name_matches_record_short_name(
            "Door_RN_RoomConnector_Breachable_OpenReverse_Crew_Quarters",
            "Door_RN_RoomConnector_Breachable_OpenReverse"
        ));
        assert!(entity_class_name_matches_record_short_name(
            "ControlPanel_Screen_DoorControl_Physical_Cutter_OpenNoneNone",
            "ControlPanel_Screen_DoorControl_Physical_Cutter_OpenNoneNone"
        ));
        assert!(!entity_class_name_matches_record_short_name(
            "Door_RN_NoRoomConnector_Component",
            "Door_RN_RoomConnector_Breachable_OpenReverse"
        ));
    }

    #[test]
    fn transform_bits_key_matches_identical_transforms() {
        let transform = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)).to_cols_array_2d();
        assert_eq!(transform_bits_key(&transform), transform_bits_key(&transform));
    }

    #[test]
    fn root_geometry_duplicate_filter_only_removes_root_container_matches() {
        let identity = glam::Mat4::IDENTITY.to_cols_array_2d();
        let mut interiors = LoadedInteriors {
            unique_cgfs: vec![
                InteriorCgfEntry {
                    cgf_path: "Data/Objects/Ships/Test/root.cga".to_string(),
                    material_path: None,
                    name: "root".to_string(),
                },
                InteriorCgfEntry {
                    cgf_path: "Data/Objects/Ships/Test/door.cga".to_string(),
                    material_path: None,
                    name: "door".to_string(),
                },
            ],
            containers: vec![
                InteriorContainerData {
                    name: "root_container".to_string(),
                    parent_entity_name: None,
                    parent_node_name: None,
                    container_transform: identity,
                    placements: vec![(0, identity, None), (1, identity, None)],
                    lights: Vec::new(),
                    palette: None,
                },
                InteriorContainerData {
                    name: "child_container".to_string(),
                    parent_entity_name: Some("child".to_string()),
                    parent_node_name: Some("helper".to_string()),
                    container_transform: identity,
                    placements: vec![(0, identity, None)],
                    lights: Vec::new(),
                    palette: None,
                },
                InteriorContainerData {
                    name: "root_bone_container".to_string(),
                    parent_entity_name: Some("EntityClassDefinition.Test".to_string()),
                    parent_node_name: Some("nose".to_string()),
                    container_transform: identity,
                    placements: vec![(0, identity, None), (1, identity, None)],
                    lights: Vec::new(),
                    palette: None,
                },
            ],
        };

        let removed = remove_root_geometry_duplicate_interior_placements(
            &mut interiors,
            "Data/Objects/Ships/Test/root.cga",
            "EntityClassDefinition.Test",
        );

        assert_eq!(removed, 2);
        assert_eq!(interiors.containers[0].placements.len(), 1);
        assert_eq!(interiors.containers[1].placements.len(), 1);
        assert_eq!(interiors.containers[2].placements.len(), 1);
    }

    #[test]
    fn geometryless_child_interiors_use_inherited_scene_anchor() {
        let child = crate::types::ResolvedNode {
            entity_name: "MISC_Hull_C_Int_Rear".to_string(),
            attachment_name: "hardpoint_body_int_rear".to_string(),
            no_rotation: false,
            offset_position: [0.0; 3],
            offset_rotation: [0.0; 3],
            detach_direction: [0.0; 3],
            port_flags: String::new(),
            nmc: None,
            bones: Vec::new(),
            has_geometry: false,
            record: starbreaker_datacore::types::Record {
                name_offset: starbreaker_datacore::types::StringId2(-1),
                file_name_offset: starbreaker_datacore::types::StringId(0),
                tag_offset: starbreaker_datacore::types::StringId2(-1),
                struct_index: 0,
                id: starbreaker_common::CigGuid::EMPTY,
                instance_index: 0,
                struct_size: 0,
            },
            geometry_path: None,
            material_path: None,
            allows_child_object_containers: true,
            children: Vec::new(),
        };

        let (parent_entity_name, parent_node_name) = child_interior_parent_target(
            &child,
            "EntityClassDefinition.MISC_Hull_C",
            None,
            None,
        );

        assert_eq!(
            parent_entity_name,
            Some("EntityClassDefinition.MISC_Hull_C".to_string())
        );
        assert_eq!(
            parent_node_name,
            Some("hardpoint_body_int_rear".to_string())
        );
    }

    #[test]
    fn geometry_child_interiors_keep_child_entity_anchor() {
        let child = crate::types::ResolvedNode {
            entity_name: "MISC_Hull_C_CentralWalkway".to_string(),
            attachment_name: "hardpoint_CentralWalkway".to_string(),
            no_rotation: false,
            offset_position: [0.0; 3],
            offset_rotation: [0.0; 3],
            detach_direction: [0.0; 3],
            port_flags: String::new(),
            nmc: None,
            bones: Vec::new(),
            has_geometry: true,
            record: starbreaker_datacore::types::Record {
                name_offset: starbreaker_datacore::types::StringId2(-1),
                file_name_offset: starbreaker_datacore::types::StringId(0),
                tag_offset: starbreaker_datacore::types::StringId2(-1),
                struct_index: 0,
                id: starbreaker_common::CigGuid::EMPTY,
                instance_index: 0,
                struct_size: 0,
            },
            geometry_path: Some("Objects/Spaceships/Ships/MISC/Hull_C/Interior_Rooms/Walkway_Rigged/Hull_C_Int_CentralWalkway.cdf".to_string()),
            material_path: None,
            allows_child_object_containers: true,
            children: Vec::new(),
        };

        let (parent_entity_name, parent_node_name) = child_interior_parent_target(
            &child,
            "EntityClassDefinition.MISC_Hull_C",
            None,
            None,
        );

        assert_eq!(
            parent_entity_name,
            Some("MISC_Hull_C_CentralWalkway".to_string())
        );
        assert_eq!(parent_node_name, None);
    }

    #[test]
    fn helper_parented_light_only_root_container_zeros_local_transform() {
        let payload = crate::types::InteriorPayload {
            name: "ext_lights_rear".to_string(),
            parent_entity_name: None,
            parent_node_name: None,
            meshes: Vec::new(),
            lights: vec![crate::types::LightInfo {
                name: "Light".to_string(),
                position: [0.0, 0.0, 0.0],
                transform_basis: "cryengine_z_up".to_string(),
                rotation: [1.0, 0.0, 0.0, 0.0],
                direction_sc: [1.0, 0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                light_type: "Omni".to_string(),
                semantic_light_kind: "point".to_string(),
                intensity_raw: 1.0,
                intensity_unit: "cryengine_authored_intensity".to_string(),
                intensity_candela_proxy: 1.0,
                intensity: 1.0,
                radius: 1.0,
                radius_m: 1.0,
                inner_angle: None,
                outer_angle: None,
                projector_texture: None,
                active_state: String::new(),
                states: std::collections::BTreeMap::new(),
            }],
            container_transform: crate::socpak::build_container_transform([0.0, -35.0, 0.0], [0.0, 0.0, 0.0]),
            tint_palette_names: Vec::new(),
        };

        let transform = normalize_root_light_only_container_transform(
            payload.container_transform,
            true,
            &payload,
        );
        let transform = mat4_from_array(&transform);
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        assert!(scale.abs_diff_eq(glam::Vec3::ONE, 1e-5));
        assert!(rotation.angle_between(glam::Quat::IDENTITY) < 1e-5);
        assert!(translation.abs_diff_eq(glam::Vec3::ZERO, 1e-5));
    }
}
