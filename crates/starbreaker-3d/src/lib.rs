pub mod dequant;
pub(crate) mod decomposed;
pub mod error;
pub(crate) mod gltf;
pub(crate) mod included_objects;
pub mod ivo;
pub mod mtl;
pub mod nmc;
pub(crate) mod pipeline;
pub mod skeleton;
pub(crate) mod socpak;
pub(crate) mod socpak_children;
pub mod types;
pub mod animation;
pub mod chrparams;
pub mod validation;

pub use error::Error;
pub use included_objects::{IncludedObject, IncludedObjects};
pub use pipeline::{
    DecomposedExport, ExportFormat, ExportKind, ExportOptions, ExportResult, ExportedFile,
    ExportedFileKind, MaterialMode,
    assemble_glb_with_loadout, assemble_glb_with_loadout_with_progress, export_geometry_decomposed,
    dump_hierarchy, load_invisible_ports, query_animation_controller_source,
    resolve_loadout_meshes, socpaks_to_glb,
};
pub use types::Mesh;
pub use gltf::SkinnedGlb;
pub use validation::{ValidationReport, validate_decomposed_export};

use starbreaker_chunks::ChunkFile;

/// Parse a `.skin`/`.cgf` IVO file into a Mesh domain type.
/// Returns an error if the file uses CrCh format (not supported).
pub fn parse_skin(data: &[u8]) -> Result<Mesh, Error> {
    parse_skin_with_options(data, false)
}

/// Parse a `.skin`/`.cgf` IVO file, optionally dequantizing with model bbox.
/// Interior CGFs use `use_model_bbox = true` because IncludedObjects placements
/// are authored for model-bbox space.
pub(crate) fn parse_skin_with_options(data: &[u8], use_model_bbox: bool) -> Result<Mesh, Error> {
    let chunk_file = ChunkFile::from_bytes(data)?;
    let ivo = match &chunk_file {
        ChunkFile::Ivo(ivo) => ivo,
        ChunkFile::CrCh(_) => return Err(Error::UnsupportedFormat),
    };

    // Find and parse IvoSkin2 chunk (0xB8757777)
    let skin_entry = ivo
        .chunks()
        .iter()
        .find(|c| c.chunk_type == starbreaker_chunks::known_types::ivo::IVO_SKIN2)
        .ok_or(Error::MissingChunk {
            chunk_type: starbreaker_chunks::known_types::ivo::IVO_SKIN2,
        })?;
    let skin_mesh = ivo::skin::SkinMesh::read(ivo.chunk_data(skin_entry))?;

    // Find and parse MtlName chunks (0x83353333)
    let materials: Vec<ivo::material::MaterialName> = ivo
        .chunks()
        .iter()
        .filter(|c| c.chunk_type == starbreaker_chunks::known_types::ivo::MTL_NAME_IVO320)
        .map(|entry| ivo::material::MaterialName::read(ivo.chunk_data(entry)))
        .collect::<Result<_, _>>()?;

    Ok(types::build_mesh_with_bbox(&skin_mesh, &materials, use_model_bbox))
}

/// Convert a character `.skinm` (geometry) into a glTF skinned GLB bound to
/// the bones of its `.skin` (CompiledBones). Joint nodes carry the game bone
/// names and the bind pose, so body and armour pieces exported this way share
/// one skeleton by bone name. `name` labels the mesh, node and skin.
pub fn skin_to_rigged_glb(geometry: &[u8], skeleton: &[u8], name: Option<&str>) -> Result<SkinnedGlb, Error> {
    let bones = skeleton::parse_skeleton(skeleton)
        .ok_or_else(|| Error::NotRiggable("no CompiledBones chunk in the skeleton file".into()))?;
    animated_skeleton_glb(&bones, Some(geometry), &[], name)
}

/// The IVO_SKIN2 mesh and material names of a `.skinm`.
fn read_skin_geometry(geometry: &[u8]) -> Result<(ivo::skin::SkinMesh, Vec<ivo::material::MaterialName>), Error> {
    let chunk_file = ChunkFile::from_bytes(geometry)?;
    let ivo = match &chunk_file {
        ChunkFile::Ivo(ivo) => ivo,
        ChunkFile::CrCh(_) => return Err(Error::UnsupportedFormat),
    };
    let skin_entry = ivo
        .chunks()
        .iter()
        .find(|c| c.chunk_type == starbreaker_chunks::known_types::ivo::IVO_SKIN2)
        .ok_or(Error::MissingChunk {
            chunk_type: starbreaker_chunks::known_types::ivo::IVO_SKIN2,
        })?;
    let skin_mesh = ivo::skin::SkinMesh::read(ivo.chunk_data(skin_entry))?;
    let materials: Vec<ivo::material::MaterialName> = ivo
        .chunks()
        .iter()
        .filter(|c| c.chunk_type == starbreaker_chunks::known_types::ivo::MTL_NAME_IVO320)
        .map(|entry| ivo::material::MaterialName::read(ivo.chunk_data(entry)))
        .collect::<Result<_, _>>()?;
    Ok((skin_mesh, materials))
}

/// Vertices whose eight-influence bone map loses influences past the
/// strongest four, and the largest weight (0..1) lost.
fn eight_influence_truncation(skin_mesh: &ivo::skin::SkinMesh) -> (usize, f32) {
    if skin_mesh.streams.bone_maps.is_some() {
        return (0, 0.0);
    }
    let (mut vertices, mut max_weight) = (0, 0f32);
    for map in skin_mesh.streams.bone_maps8.iter().flatten() {
        let (_, dropped) = map.strongest_four();
        if dropped > 0 {
            vertices += 1;
            max_weight = max_weight.max(f32::from(dropped) / 255.0);
        }
    }
    (vertices, max_weight)
}

/// Write the rigged skeleton `bones` (joint nodes exactly as
/// [`skin_to_rigged_glb`] writes them) with one glTF animation per sampled
/// clip (see [`animation::sample_clip`]). `geometry`, a `.skinm` whose bone
/// maps index `bones`, is skinned into the same file when given; without it
/// the GLB carries only the skeleton (plus a mesh-less skin, so importers
/// build an armature) and the animations.
pub fn animated_skeleton_glb(
    bones: &[skeleton::Bone],
    geometry: Option<&[u8]>,
    clips: &[animation::SampledClip],
    name: Option<&str>,
) -> Result<SkinnedGlb, Error> {
    let Some(geometry) = geometry else {
        return gltf::write_animated_skeleton_glb(&[], bones, clips, name, &bundled_glb_options());
    };
    let (skin_mesh, materials) = read_skin_geometry(geometry)?;
    let mesh = types::build_skinned_mesh(&skin_mesh, &materials);
    let mut out = gltf::write_animated_skeleton_glb(&[(&mesh, name)], bones, clips, name, &bundled_glb_options())?;
    (out.truncated_vertices, out.max_truncated_weight) = eight_influence_truncation(&skin_mesh);
    Ok(out)
}

/// One `.skinm` piece for [`pieces_on_skeleton_glb`].
pub struct SkinPiece<'a> {
    /// The `.skinm` geometry.
    pub geometry: &'a [u8],
    /// CompiledBones of the piece's own `.skin`, which its bone maps index.
    pub bones: &'a [skeleton::Bone],
    /// Name of the piece's mesh and node.
    pub name: &'a str,
}

/// How one piece of [`pieces_on_skeleton_glb`] was bound to the skeleton.
#[derive(Debug, Clone, Default)]
pub struct PieceBinding {
    pub name: String,
    pub vertex_count: usize,
    /// Bones of the piece's `.skin`.
    pub piece_bones: usize,
    /// Piece bones found by name in the skeleton.
    pub matched_bones: usize,
    /// Piece bones missing from the skeleton, each with the skeleton bone
    /// (its nearest ancestor present there) that took over its influence.
    pub folded_bones: Vec<(String, String)>,
    /// Skeleton joints the piece's vertices are weighted to, by name.
    pub used_joints: Vec<String>,
    /// Largest bind-pose difference, over the used joints, between the
    /// piece's bone and the skeleton's: position (m) and rotation (degrees).
    /// The vertices are re-posed by it, so the piece sits on the skeleton's
    /// bind pose the way the game poses a skin attachment.
    pub max_bind_offset: f32,
    pub max_bind_angle_deg: f32,
    pub truncated_vertices: usize,
    pub max_truncated_weight: f32,
}

/// Result of [`pieces_on_skeleton_glb`].
pub struct SkeletonRiggedGlb {
    pub glb: SkinnedGlb,
    pub pieces: Vec<PieceBinding>,
}

/// Skin several character pieces (body, head, eyes, armour...) onto one
/// skeleton, typically the character's `.chr`, as one glTF skin: one joint
/// per skeleton bone, one skinned mesh node per piece.
///
/// Each piece's bone maps index its own `.skin` bone list; they are remapped
/// by bone name. A piece bone the skeleton lacks (the head's FACIAL_* rig
/// bones, absent from `bhm_skeleton_v7.chr`) hands its influence to its
/// nearest ancestor that the skeleton has (Head, Neck1...). That is exact
/// for the bind pose and for every pose of the skeleton's bones, as long as
/// the folded bones are not animated themselves (no facial animation).
/// Folding happens before eight-influence maps are cut to four, so it also
/// loses less weight.
pub fn pieces_on_skeleton_glb(
    skeleton: &[skeleton::Bone],
    pieces: &[SkinPiece],
    name: Option<&str>,
) -> Result<SkeletonRiggedGlb, Error> {
    use std::collections::HashMap;
    let by_name: HashMap<&str, usize> = skeleton.iter().enumerate().map(|(i, b)| (b.name.as_str(), i)).collect();
    let by_lower: HashMap<String, usize> = skeleton.iter().enumerate().map(|(i, b)| (b.name.to_lowercase(), i)).collect();
    let find = |bone: &str| by_name.get(bone).or_else(|| by_lower.get(&bone.to_lowercase())).copied();
    let rigid = |(rotation, translation): (glam::Quat, glam::Vec3)| glam::Mat4::from_rotation_translation(rotation, translation);

    let mut meshes = Vec::with_capacity(pieces.len());
    let mut bindings = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let mut binding = PieceBinding { name: piece.name.to_string(), piece_bones: piece.bones.len(), ..Default::default() };
        // Piece bone -> (skeleton bone, the piece bone carrying that name).
        let mut remap: Vec<(u16, usize)> = Vec::with_capacity(piece.bones.len());
        for index in 0..piece.bones.len() {
            let mut current = index;
            let target = loop {
                if let Some(target) = find(&piece.bones[current].name) {
                    break target;
                }
                current = skeleton::bone_parent(piece.bones, current).ok_or_else(|| {
                    Error::NotRiggable(format!("{}: bone {} has no ancestor in the skeleton", piece.name, piece.bones[index].name))
                })?;
            };
            if current == index {
                binding.matched_bones += 1;
            } else {
                binding.folded_bones.push((piece.bones[index].name.clone(), skeleton[target].name.clone()));
            }
            remap.push((target as u16, current));
        }
        let remap_joint = |joint: u16| remap.get(joint as usize).map_or(u16::MAX, |r| r.0);

        let (mut skin_mesh, materials) = read_skin_geometry(piece.geometry)?;
        for map in skin_mesh.streams.bone_maps.iter_mut().flatten() {
            map.joint_indices = map.joint_indices.map(remap_joint);
        }
        for map in skin_mesh.streams.bone_maps8.iter_mut().flatten() {
            map.joint_indices = map.joint_indices.map(remap_joint);
        }
        (binding.truncated_vertices, binding.max_truncated_weight) = eight_influence_truncation(&skin_mesh);
        let mut mesh = types::build_skinned_mesh(&skin_mesh, &materials);
        for submesh in &mut mesh.submeshes {
            submesh.node_parent_index = remap.get(submesh.node_parent_index as usize).map_or(0, |r| r.0);
        }

        // Per skeleton joint: skeleton bind * inverse(piece bind), which
        // re-poses the piece's vertices onto the skeleton's bind pose.
        let mut delta: HashMap<u16, glam::Mat4> = HashMap::new();
        for &(target, piece_bone) in &remap {
            delta.entry(target).or_insert_with(|| {
                rigid(skeleton::bone_world_transform(&skeleton[target as usize]))
                    * rigid(skeleton::bone_world_transform(&piece.bones[piece_bone])).inverse()
            });
        }
        let moved = delta.values().any(|m| !m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-6));
        let mut used = vec![false; skeleton.len()];
        if let Some(maps) = mesh.skin_weights.as_ref() {
            for (vertex, map) in maps.iter().enumerate() {
                let total: f32 = map.weights.iter().map(|&w| f32::from(w)).sum();
                if total <= 0.0 {
                    continue;
                }
                let mut blend = glam::Mat4::ZERO;
                for (&joint, &weight) in map.joint_indices.iter().zip(&map.weights) {
                    if weight == 0 || joint as usize >= skeleton.len() {
                        continue;
                    }
                    used[joint as usize] = true;
                    if let Some(m) = delta.get(&joint) {
                        blend += *m * (f32::from(weight) / total);
                    }
                }
                if !moved || blend == glam::Mat4::ZERO {
                    continue;
                }
                let p = &mut mesh.positions[vertex];
                *p = blend.transform_point3(glam::Vec3::from(*p)).to_array();
                if let Some(n) = mesh.normals.as_mut().and_then(|n| n.get_mut(vertex)) {
                    *n = blend.transform_vector3(glam::Vec3::from(*n)).normalize_or_zero().to_array();
                }
                if let Some(t) = mesh.tangents.as_mut().and_then(|t| t.get_mut(vertex)) {
                    let v = blend.transform_vector3(glam::Vec3::new(t[0], t[1], t[2])).normalize_or_zero();
                    *t = [v.x, v.y, v.z, t[3]];
                }
            }
        }
        for &(target, piece_bone) in &remap {
            if !std::mem::take(&mut used[target as usize]) {
                continue;
            }
            binding.used_joints.push(skeleton[target as usize].name.clone());
            let (piece_rotation, piece_position) = skeleton::bone_world_transform(&piece.bones[piece_bone]);
            let (rotation, position) = skeleton::bone_world_transform(&skeleton[target as usize]);
            binding.max_bind_offset = binding.max_bind_offset.max(piece_position.distance(position));
            binding.max_bind_angle_deg = binding.max_bind_angle_deg.max(piece_rotation.angle_between(rotation).to_degrees());
        }
        binding.vertex_count = mesh.positions.len();
        meshes.push(mesh);
        bindings.push(binding);
    }

    let named: Vec<(&types::Mesh, Option<&str>)> = meshes.iter().zip(pieces).map(|(m, p)| (m, Some(p.name))).collect();
    let mut glb = gltf::write_animated_skeleton_glb(&named, skeleton, &[], name, &bundled_glb_options())?;
    glb.truncated_vertices = bindings.iter().map(|b| b.truncated_vertices).sum();
    glb.max_truncated_weight = bindings.iter().map(|b| b.max_truncated_weight).fold(0.0, f32::max);
    Ok(SkeletonRiggedGlb { glb, pieces: bindings })
}

fn bundled_glb_options() -> gltf::GlbOptions {
    gltf::GlbOptions {
        material_mode: pipeline::MaterialMode::None,
        preserve_textureless_decal_primitives: false,
        include_vertex_colors: true,
        include_damage: true,
        metadata: gltf::GlbMetadata {
            entity_name: None,
            geometry_path: None,
            material_path: None,
            export_options: gltf::ExportOptionsMetadata {
                kind: "Bundled".to_string(),
                material_mode: "None".to_string(),
                format: "Glb".to_string(),
                lod_level: 0,
                texture_mip: 0,
                include_attachments: false,
                include_interior: false,
            },
        },
        fallback_palette: None,
    }
}

/// Parse a `.skin`/`.cgf` IVO file and convert to GLB in one step.
/// If `metadata` is provided (the primary `.cgf`/`.skin` bytes), NMC transforms
/// are parsed from it and applied to the scene graph.
pub fn skin_to_glb(data: &[u8], metadata: Option<&[u8]>) -> Result<Vec<u8>, Error> {
    let mesh = parse_skin(data)?;
    let root_nmc = metadata
        .and_then(nmc::parse_nmc_full)
        .map(|(nodes, mat_indices)| nmc::NodeMeshCombo {
            nodes,
            material_indices: mat_indices,
        });
    gltf::write_glb(
        gltf::GlbInput {
            root_mesh: Some(mesh),
            root_materials: None,
            root_textures: None,
            root_nmc,
            root_palette: None,
            skeleton_bones: Vec::new(),
            children: Vec::new(),
            interiors: pipeline::LoadedInteriors::default(),
        },
        &mut gltf::GlbLoaders {
            load_textures: &mut |_, _| None,
            load_interior_mesh: &mut |_| None,
        },
        &gltf::GlbOptions {
            material_mode: pipeline::MaterialMode::None,
            preserve_textureless_decal_primitives: false,
            include_vertex_colors: true,
            include_damage: true,
            metadata: gltf::GlbMetadata {
                entity_name: None,
                geometry_path: None,
                material_path: None,
                export_options: gltf::ExportOptionsMetadata {
                    kind: "Bundled".to_string(),
                    material_mode: "None".to_string(),
                    format: "Glb".to_string(),
                    lod_level: 0,
                    texture_mip: 0,
                    include_attachments: false,
                    include_interior: false,
                },
            },
            fallback_palette: None,
        },
    )
}
