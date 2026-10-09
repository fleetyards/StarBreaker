//! Child entity payload loading and caching helpers.
//!
//! Loads child mesh assets for loadout children and inline entity attachments.
//! `collect_child_payload_specs` gathers attachment specs from a loadout tree;
//! `load_child_payload_asset` exports a single child CGF; `load_child_payloads`
//! drives the full parallel load pipeline.
//! Also defines `LandingGearAsset` (cached landing gear geometry) and the path
//! normalisation helpers used for decomposed export deduplication.

use std::collections::HashSet;

use starbreaker_datacore::database::Database;
use starbreaker_p4k::MappedP4k;

use crate::mtl;
use crate::nmc;
use crate::types::MaterialTextures;

use super::*;

/// Try loading an entity's mesh data from its resolved geometry/material paths,
/// falling back to DataCore record lookup.
pub(crate) fn load_child_mesh(
    child: &crate::types::ResolvedNode,
    db: &Database,
    p4k: &MappedP4k,
    opts: &ExportOptions,
) -> Option<(
    crate::types::Mesh,
    Option<mtl::MtlFile>,
    Option<nmc::NodeMeshCombo>,
    Option<mtl::TintPalette>,
    Vec<crate::skeleton::Bone>,
    String,
    String,
    Option<String>,
)> {
    let result = if child.geometry_path.is_some() {
        let gp = child.geometry_path.as_deref().unwrap_or("");
        let mp = child.material_path.as_deref().unwrap_or("");
        export_entity_from_paths(p4k, gp, mp, opts)
            .map_err(|e| {
                log::warn!("  {} -> load from paths failed: {e}", child.entity_name);
                e
            })
            .or_else(|_| export_entity_payload(db, p4k, &child.record, opts))
    } else {
        export_entity_payload(db, p4k, &child.record, opts)
    };

    result
        .ok()
        .map(
            |(
                mesh,
                mtl,
                _tex,
                nmc,
                palette,
                geometry_path,
                material_path,
                bones,
                skeleton_source_path,
            )| {
                let resolved_palette = palette.or_else(|| query_tint_palette(db, &child.record));
                (
                    mesh,
                    mtl,
                    nmc,
                    resolved_palette,
                    bones,
                    geometry_path,
                    material_path,
                    skeleton_source_path,
                )
            },
        )
}

pub(crate) struct ChildPayloadSpec {
    pub(crate) child: crate::types::ResolvedNode,
    pub(crate) parent_entity_name: String,
    pub(crate) parent_node_name: String,
    pub(crate) no_rotation: bool,
    /// `(port name, record)` of each item directly attached to `child`, kept
    /// even for items without geometry so the child's docking collar port can
    /// be found once the database is at hand.
    pub(crate) port_items: Vec<(String, starbreaker_datacore::types::Record)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ChildPayloadCacheKey {
    pub(crate) record_id: starbreaker_datacore::types::CigGuid,
    pub(crate) geometry_path: Option<String>,
    pub(crate) material_path: Option<String>,
}

#[derive(Clone)]
pub(crate) struct LoadedChildPayload {
    pub(crate) mesh: crate::types::Mesh,
    pub(crate) materials: Option<mtl::MtlFile>,
    pub(crate) textures: Option<MaterialTextures>,
    pub(crate) nmc: Option<nmc::NodeMeshCombo>,
    pub(crate) palette: Option<mtl::TintPalette>,
    pub(crate) bones: Vec<crate::skeleton::Bone>,
    pub(crate) geometry_path: String,
    pub(crate) material_path: String,
    pub(crate) skeleton_source_path: Option<String>,
}

/// Cached result of loading a landing-gear CGF. Multiple gear hardpoints
/// often share the same geometry (e.g. mirrored gears), so this cache
/// dedupes P4k reads and texture decoding by gear_path.
#[derive(Clone)]
pub(crate) struct LandingGearAsset {
    pub(crate) mesh: crate::types::Mesh,
    pub(crate) materials: Option<mtl::MtlFile>,
    pub(crate) textures: Option<MaterialTextures>,
    pub(crate) nmc: Option<nmc::NodeMeshCombo>,
    pub(crate) geometry_path: String,
    pub(crate) material_path: String,
    pub(crate) bones: Vec<crate::skeleton::Bone>,
    pub(crate) skeleton_source_path: Option<String>,
}

pub(crate) fn collect_child_payload_specs(
    children: &[crate::types::ResolvedNode],
    parent_entity_name: &str,
    override_attachment: Option<(&str, bool)>,
    out: &mut Vec<ChildPayloadSpec>,
) {
    for child in children {
        let (attach_name, no_rotation) = match override_attachment {
            Some((name, parent_no_rot)) => (name.to_string(), child.no_rotation || parent_no_rot),
            None => (child.attachment_name.clone(), child.no_rotation),
        };

        let child_creates_nodes = child.has_geometry || child.nmc.is_some();
        if child_creates_nodes {
            out.push(ChildPayloadSpec {
                child: child.clone_payload_source(),
                parent_entity_name: parent_entity_name.to_string(),
                parent_node_name: attach_name,
                no_rotation,
                port_items: child
                    .children
                    .iter()
                    .map(|item| (item.attachment_name.clone(), item.record))
                    .collect(),
            });
            collect_child_payload_specs(&child.children, &child.entity_name, None, out);
        } else {
            collect_child_payload_specs(
                &child.children,
                parent_entity_name,
                Some((&child.attachment_name, child.no_rotation)),
                out,
            );
        }
    }
}

pub(crate) fn empty_child_mesh() -> crate::types::Mesh {
    crate::types::Mesh {
        positions: Vec::new(),
        indices: Vec::new(),
        uvs: None,
        secondary_uvs: None,
        normals: None,
        tangents: None,
        colors: None,
        submeshes: Vec::new(),
        model_min: [0.0; 3],
        model_max: [0.0; 3],
        scaling_min: [0.0; 3],
        scaling_max: [0.0; 3],
        skin_weights: None,
    }
}

pub(crate) fn normalize_decomposed_source_path(p4k: &MappedP4k, path: &str) -> String {
    let p4k_path = datacore_path_to_p4k(path);
    p4k.entry_case_insensitive(&p4k_path)
        .map(|entry| entry.name.replace('\\', "/"))
        .unwrap_or_else(|| p4k_path.replace('\\', "/"))
}

pub(crate) fn replace_extension(path: &str, new_extension: &str) -> String {
    let Some((stem, _)) = path.rsplit_once('.') else {
        return format!("{path}{new_extension}");
    };
    stem.to_string() + new_extension
}

pub(crate) fn decomposed_mesh_asset_path(p4k: &MappedP4k, geometry_path: &str) -> Option<String> {
    if geometry_path.is_empty() {
        None
    } else {
        Some(replace_extension(&normalize_decomposed_source_path(p4k, geometry_path), ".glb"))
    }
}

pub(crate) fn build_child_payload_cache_key(child: &crate::types::ResolvedNode) -> ChildPayloadCacheKey {
    ChildPayloadCacheKey {
        record_id: child.record.id,
        geometry_path: child.geometry_path.clone(),
        material_path: child.material_path.clone(),
    }
}

pub(crate) fn load_child_payload_asset(
    child: &crate::types::ResolvedNode,
    db: &Database,
    p4k: &MappedP4k,
    mesh_opts: &ExportOptions,
    final_material_mode: MaterialMode,
    existing_asset_paths: Option<&HashSet<String>>,
) -> Option<LoadedChildPayload> {
    if mesh_opts.kind == ExportKind::Decomposed {
        if let Some(geometry_path) = child.geometry_path.as_deref() {
            if let Some(mesh_asset_path) = decomposed_mesh_asset_path(p4k, geometry_path) {
                if existing_asset_paths
                    .is_some_and(|paths| paths.contains(&mesh_asset_path.to_ascii_lowercase()))
                {
                    let material_path = child.material_path.as_deref().unwrap_or("");
                    let (nmc, materials) = load_nmc_and_material(p4k, geometry_path, material_path);
                    let skeleton_source_path = resolve_geometry_files(p4k, geometry_path)
                        .ok()
                        .and_then(|resolved| {
                            skeleton_source_paths(resolved.skeleton_path.as_deref(), &resolved.parts[0].path)
                                .first()
                                .map(|path| (*path).to_string())
                        });
                    return Some(LoadedChildPayload {
                        mesh: empty_child_mesh(),
                        materials,
                        textures: None,
                        nmc,
                        palette: None,
                        bones: Vec::new(),
                        geometry_path: geometry_path.to_string(),
                        material_path: material_path.to_string(),
                        skeleton_source_path,
                    });
                }
            }
        }
    }

    let (mesh, mtl, nmc, palette, bones, geometry_path, material_path, skeleton_source_path) =
        load_child_mesh(child, db, p4k, mesh_opts)?;
    let textures = if final_material_mode.include_textures() {
        mtl.as_ref().map(|materials| {
            let mut png_cache = PngCache::new();
            load_material_textures(
                p4k,
                materials,
                palette.as_ref(),
                mesh_opts.texture_mip,
                &mut png_cache,
                final_material_mode.include_normals(),
                final_material_mode.experimental(),
            )
        })
    } else {
        None
    };

    Some(LoadedChildPayload {
        mesh,
        materials: mtl,
        textures,
        nmc,
        palette,
        bones,
        geometry_path,
        material_path,
        skeleton_source_path,
    })
}

pub(crate) fn load_child_payloads(
    specs: Vec<ChildPayloadSpec>,
    db: &Database,
    p4k: &MappedP4k,
    mesh_opts: &ExportOptions,
    final_material_mode: MaterialMode,
    existing_asset_paths: Option<&HashSet<String>>,
) -> Vec<crate::types::EntityPayload> {
    use rayon::prelude::*;

    let mut unique_children = Vec::new();
    let mut unique_child_indices = std::collections::HashMap::new();
    let mut spec_asset_indices = Vec::with_capacity(specs.len());

    for spec in &specs {
        let child = &spec.child;
        if !child.has_geometry {
            spec_asset_indices.push(None);
            continue;
        }

        let cache_key = build_child_payload_cache_key(child);
        let unique_index = if let Some(&index) = unique_child_indices.get(&cache_key) {
            index
        } else {
            let index = unique_children.len();
            unique_children.push(child.clone_payload_source());
            unique_child_indices.insert(cache_key, index);
            index
        };
        spec_asset_indices.push(Some(unique_index));
    }

    let loaded_assets: Vec<Option<LoadedChildPayload>> = unique_children
        .into_par_iter()
        .map(|child| {
            load_child_payload_asset(
                &child,
                db,
                p4k,
                mesh_opts,
                final_material_mode,
                existing_asset_paths,
            )
        })
        .collect();

    specs
        .into_iter()
        .enumerate()
        .filter_map(|(spec_index, spec)| {
            let child = &spec.child;
            let entity_category = entity_record_category(db, &child.record);
            let attach_def_type = entity_record_attach_def_type(db, &child.record);
            let docking_collar_port = spec
                .port_items
                .iter()
                .find(|(_, record)| {
                    entity_record_attach_def_type(db, record).as_deref() == Some("DockingCollar")
                })
                .map(|(port, _)| port.clone());
            if child.has_geometry {
                let asset_index = spec_asset_indices[spec_index]?;
                let loaded = loaded_assets.get(asset_index)?.as_ref()?;
                Some(crate::types::EntityPayload {
                    mesh: loaded.mesh.clone(),
                    materials: loaded.materials.clone(),
                    textures: loaded.textures.clone(),
                    nmc: loaded.nmc.clone(),
                    palette: loaded.palette.clone(),
                    geometry_path: loaded.geometry_path.clone(),
                    material_path: loaded.material_path.clone(),
                    bones: loaded.bones.clone(),
                    skeleton_source_path: loaded.skeleton_source_path.clone(),
                    entity_name: child.entity_name.clone(),
                    entity_category,
                    attach_def_type,
                    parent_node_name: spec.parent_node_name.clone(),
                    parent_entity_name: spec.parent_entity_name.clone(),
                    no_rotation: spec.no_rotation,
                    offset_position: child.offset_position,
                    offset_rotation: child.offset_rotation,
                    detach_direction: child.detach_direction,
                    port_flags: child.port_flags.clone(),
                    docking_collar_port: docking_collar_port.clone(),
                })
            } else if child.nmc.is_some() {
                Some(crate::types::EntityPayload {
                    mesh: empty_child_mesh(),
                    materials: None,
                    textures: None,
                    nmc: child.nmc.clone(),
                    palette: None,
                    geometry_path: child.geometry_path.clone().unwrap_or_default(),
                    material_path: child.material_path.clone().unwrap_or_default(),
                    bones: Vec::new(),
                    skeleton_source_path: None,
                    entity_name: child.entity_name.clone(),
                    entity_category,
                    attach_def_type,
                    parent_node_name: spec.parent_node_name.clone(),
                    parent_entity_name: spec.parent_entity_name.clone(),
                    no_rotation: spec.no_rotation,
                    offset_position: child.offset_position,
                    offset_rotation: child.offset_rotation,
                    detach_direction: child.detach_direction,
                    port_flags: child.port_flags.clone(),
                    docking_collar_port: docking_collar_port.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn entity_record_category(db: &Database, record: &starbreaker_datacore::types::Record) -> Option<String> {
    db.compile_path::<String>(record.struct_id(), "Category")
        .ok()
        .and_then(|compiled| db.query_single::<String>(&compiled, record).ok().flatten())
        .filter(|category| !category.is_empty())
}

fn entity_record_attach_def_type(
    db: &Database,
    record: &starbreaker_datacore::types::Record,
) -> Option<String> {
    db.compile_path::<String>(record.struct_id(), "Components[SAttachableComponentParams].AttachDef.Type")
        .ok()
        .and_then(|compiled| db.query_single::<String>(&compiled, record).ok().flatten())
        .filter(|attach_def_type| !attach_def_type.is_empty())
}
