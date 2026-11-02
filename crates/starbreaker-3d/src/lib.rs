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
    resolve_loadout_meshes, socpaks_to_glb, worn_model_paths, WornModelPaths,
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

/// A rigid model in its own model space, for [`rigid_on_bone_glb`].
pub struct RigidModel {
    pub mesh: types::Mesh,
    /// Model-space frames of the model's helper nodes / bones by lower-case
    /// name (`backpack_attach_1_loc`, ...), rotation and translation.
    pub helpers: std::collections::HashMap<String, (glam::Quat, glam::Vec3)>,
}

/// Load a `.cga`/`.cgf` model: `geometry` is its `.cgam`/`.cgfm`, `meta` the
/// `.cga`/`.cgf` with the node tree. Node transforms are baked into the
/// vertices (composed through the parent chain); every NMC node is listed in
/// `helpers` by its model-space frame.
pub fn load_rigid_cga(meta: &[u8], geometry: &[u8]) -> Result<RigidModel, Error> {
    let nmc = nmc::parse_nmc_full(meta).map(|(nodes, material_indices)| nmc::NodeMeshCombo { nodes, material_indices });
    let mut mesh = parse_skin(geometry)?;
    let mut helpers = std::collections::HashMap::new();
    if let Some(nmc) = &nmc {
        mesh = pipeline::bake_nmc_into_mesh(mesh, Some(nmc), false);
        for (node, world) in nmc.nodes.iter().zip(pipeline::compute_nmc_world_transforms(nmc)) {
            let (_, rotation, translation) = world.to_scale_rotation_translation();
            helpers.entry(node.name.to_lowercase()).or_insert((rotation.normalize(), translation));
        }
    }
    mesh.skin_weights = None;
    Ok(RigidModel { mesh, helpers })
}

/// Load the `.skinm` of a skinned model that is worn rigidly (a backpack
/// CDF's CA_SKIN on its own small `.chr`): bind-pose model space, its own
/// bone weights dropped. `bones` (the `.chr`'s) are listed in `helpers`.
pub fn load_rigid_skin(geometry: &[u8], bones: &[skeleton::Bone]) -> Result<RigidModel, Error> {
    let (skin_mesh, materials) = read_skin_geometry(geometry)?;
    let mut mesh = types::build_skinned_mesh(&skin_mesh, &materials);
    mesh.skin_weights = None;
    let helpers = bones.iter().map(|b| (b.name.to_lowercase(), skeleton::bone_world_transform(b))).collect();
    Ok(RigidModel { mesh, helpers })
}

/// Result of [`rigid_on_bone_glb`].
pub struct RigidGlb {
    pub glb: SkinnedGlb,
    /// The joints written: the attach bone and its ancestors, root first.
    pub joints: Vec<String>,
    /// Whether the offset helper was found in the model (else the model
    /// origin sits on the bone).
    pub helper_found: bool,
    /// Bind-pose world position of the attach bone (CryEngine Z-up, m).
    pub bone_position: [f32; 3],
}

/// Write rigid `models` as a glTF skin bound with weight 1 to `bone` of
/// `skeleton` (a core `.skin`, whose `backpack_attach_1_override` carries the
/// backpack). Placed like the game's item port: the model's `offset_helper`
/// frame (`backpack_attach_1_loc`) lands on the bone frame, so
/// vertex = bone_bind * inverse(helper) * model vertex. The joints are the
/// bone's ancestor chain from the skeleton root with the skeleton's bind
/// pose and names, so a viewer that drives joints by name from the
/// character (or grafts the bone from the worn core) moves it with the body.
pub fn rigid_on_bone_glb(
    models: &[RigidModel],
    skeleton_bones: &[skeleton::Bone],
    bone: &str,
    offset_helper: Option<&str>,
    name: Option<&str>,
) -> Result<RigidGlb, Error> {
    let target = skeleton_bones
        .iter()
        .position(|b| b.name.eq_ignore_ascii_case(bone))
        .ok_or_else(|| Error::NotRiggable(format!("attach bone {bone} not in the skeleton")))?;
    // Ancestor chain, root first.
    let mut chain = vec![target];
    while let Some(parent) = skeleton::bone_parent(skeleton_bones, *chain.last().unwrap()) {
        chain.push(parent);
    }
    chain.reverse();
    let bones: Vec<skeleton::Bone> = chain
        .iter()
        .enumerate()
        .map(|(k, &i)| skeleton::Bone {
            parent_index: (k > 0).then(|| (k - 1) as u16),
            ..skeleton_bones[i].clone()
        })
        .collect();
    let joint = (bones.len() - 1) as u16;
    let (bone_rotation, bone_position) = skeleton::bone_world_transform(&skeleton_bones[target]);
    let bone_world = glam::Mat4::from_rotation_translation(bone_rotation, bone_position);

    let mut helper_found = offset_helper.is_none();
    let mut meshes = Vec::with_capacity(models.len());
    for model in models {
        let helper = offset_helper.and_then(|h| model.helpers.get(&h.to_lowercase()));
        helper_found |= helper.is_some();
        let helper = helper.map_or(glam::Mat4::IDENTITY, |&(r, t)| glam::Mat4::from_rotation_translation(r, t));
        let place = bone_world * helper.inverse();
        let mut mesh = model.mesh.clone();
        for p in &mut mesh.positions {
            *p = place.transform_point3(glam::Vec3::from(*p)).to_array();
        }
        if let Some(normals) = mesh.normals.as_mut() {
            for n in normals {
                *n = place.transform_vector3(glam::Vec3::from(*n)).normalize_or_zero().to_array();
            }
        }
        if let Some(tangents) = mesh.tangents.as_mut() {
            for t in tangents {
                let v = place.transform_vector3(glam::Vec3::new(t[0], t[1], t[2])).normalize_or_zero();
                *t = [v.x, v.y, v.z, t[3]];
            }
        }
        for submesh in &mut mesh.submeshes {
            submesh.node_parent_index = joint;
        }
        mesh.skin_weights = Some(vec![
            ivo::skin::BoneMap12 { joint_indices: [joint, 0, 0, 0], weights: [255, 0, 0, 0] };
            mesh.positions.len()
        ]);
        meshes.push(mesh);
    }
    let named: Vec<(&types::Mesh, Option<&str>)> = meshes.iter().map(|m| (m, name)).collect();
    let glb = gltf::write_animated_skeleton_glb(&named, &bones, &[], name, &bundled_glb_options())?;
    Ok(RigidGlb {
        glb,
        joints: bones.iter().map(|b| b.name.clone()).collect(),
        helper_found,
        bone_position: bone_position.to_array(),
    })
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

#[cfg(test)]
mod rigid_tests {
    use super::*;

    fn bone(name: &str, parent: Option<u16>, position: [f32; 3], rotation_wxyz: [f32; 4]) -> skeleton::Bone {
        skeleton::Bone {
            name: name.into(),
            parent_index: parent,
            object_node_index: None,
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            world_position: position,
            world_rotation: rotation_wxyz,
        }
    }

    #[test]
    fn rigid_model_lands_on_the_bone_through_its_offset_helper() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let skeleton = vec![
            bone("World", None, [0.0; 3], [1.0, 0.0, 0.0, 0.0]),
            bone("Spine", Some(0), [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            bone("LeftArm", Some(1), [0.3, 0.0, 1.4], [1.0, 0.0, 0.0, 0.0]),
            // 90 degrees about Z, behind the spine.
            bone("backpack_attach_1_override", Some(1), [0.0, -0.2, 1.5], [half, 0.0, 0.0, half]),
        ];
        let mesh = types::Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
            indices: vec![0, 1, 2],
            uvs: None,
            secondary_uvs: None,
            normals: None,
            tangents: None,
            colors: None,
            submeshes: vec![types::SubMesh {
                material_name: None,
                material_id: 0,
                source_material_id: Some(0),
                first_index: 0,
                num_indices: 3,
                first_vertex: 0,
                num_vertices: 3,
                node_parent_index: 0,
            }],
            model_min: [0.0; 3],
            model_max: [1.0, 0.0, 0.1],
            scaling_min: [0.0; 3],
            scaling_max: [1.0, 0.0, 0.1],
            skin_weights: None,
        };
        // The helper sits 0.1 m up in the model: that point goes onto the bone.
        let helpers = [("backpack_attach_1_loc".to_string(), (glam::Quat::IDENTITY, glam::Vec3::new(0.0, 0.0, 0.1)))].into();
        let model = RigidModel { mesh, helpers };
        let out = rigid_on_bone_glb(&[model], &skeleton, "backpack_attach_1_override", Some("backpack_attach_1_loc"), Some("bp")).unwrap();
        assert_eq!(out.joints, ["World", "Spine", "backpack_attach_1_override"]);
        assert!(out.helper_found);
        assert_eq!(out.glb.vertex_count, 3);

        // Re-derive the placed vertices from the GLB's position accessor.
        let glb = &out.glb.glb;
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let root: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        let bin = &glb[20 + json_len + 8..];
        let prim = &root["meshes"][0]["primitives"][0];
        let acc = &root["accessors"][prim["attributes"]["POSITION"].as_u64().unwrap() as usize];
        let view = &root["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let off = view["byteOffset"].as_u64().unwrap_or(0) as usize + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
        let f = |i: usize| f32::from_le_bytes(bin[off + 4 * i..off + 4 * i + 4].try_into().unwrap());
        // Helper point -> bone origin; model +X -> bone +X = world +Y.
        let p2 = glam::Vec3::new(f(6), f(7), f(8));
        let p1 = glam::Vec3::new(f(3), f(4), f(5));
        let p0 = glam::Vec3::new(f(0), f(1), f(2));
        assert!(p2.abs_diff_eq(glam::Vec3::new(0.0, -0.2, 1.5), 1e-5), "{p2:?}");
        assert!(p0.abs_diff_eq(glam::Vec3::new(0.0, -0.2, 1.4), 1e-5), "{p0:?}");
        assert!((p1 - p0).abs_diff_eq(glam::Vec3::new(0.0, 1.0, 0.0), 1e-5), "{p1:?}");
        // Every vertex weighted fully to the attach joint (index 2).
        let joints = &root["accessors"][prim["attributes"]["JOINTS_0"].as_u64().unwrap() as usize];
        assert_eq!(joints["count"], 3);
    }
}
