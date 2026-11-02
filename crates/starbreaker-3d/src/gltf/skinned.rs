//! Skinned (rigged) GLB output for character `.skin` meshes.
//!
//! A character body and every armour piece are separate `.skin` files, each
//! with its own CompiledBones list whose bind pose is shared by bone name.
//! Each piece is written as a standard glTF skin: one joint node per bone
//! (named exactly like the game bone, local TRS in a parent hierarchy),
//! `inverseBindMatrices` = inverse of the bone's bind-pose world matrix, and
//! `JOINTS_0`/`WEIGHTS_0` on every primitive. Pieces loaded side by side line
//! up and deform together when their same-named joints are posed alike.
//!
//! Vertices and bind matrices stay in CryEngine model space (Z-up); the joint
//! roots sit under the usual `CryEngine_Z_up` wrapper, which carries the
//! skinned vertices to glTF Y-up. The skinned mesh node is a scene root with
//! no transform, as glTF ignores a skinned node's own transforms anyway.

use gltf_json as json;
use json::validation::{Checked, USize64};

use super::{GlbBuilder, GlbOptions};
use crate::error::Error;
use crate::ivo::skin::BoneMap12;
use crate::skeleton::Bone;
use crate::types::Mesh;

/// Result of [`write_skinned_glb`].
pub struct SkinnedGlb {
    pub glb: Vec<u8>,
    pub vertex_count: usize,
    pub joint_count: usize,
    /// Vertices with no usable influence, bound fully to their submesh's
    /// node bone (or the root bone).
    pub unweighted_vertices: usize,
    /// Influences naming a joint past the end of the bone list (dropped).
    pub out_of_range_influences: usize,
    /// Vertices whose eight-influence bone map lost influences past the
    /// strongest four (glTF/three.js read four), and the largest weight lost.
    pub truncated_vertices: usize,
    pub max_truncated_weight: f32,
}

/// Bone bind-pose world transform (model space) as rotation + translation.
fn bone_world(bone: &Bone) -> (glam::Quat, glam::Vec3) {
    let [w, x, y, z] = bone.world_rotation;
    (glam::Quat::from_xyzw(x, y, z, w).normalize(), glam::Vec3::from(bone.world_position))
}

/// Inverse of a rigid transform, built from its parts so the bottom row is
/// exactly (0, 0, 0, 1) as glTF requires of inverse bind matrices.
fn rigid_inverse((rotation, translation): (glam::Quat, glam::Vec3)) -> glam::Mat4 {
    let inverse = rotation.conjugate();
    glam::Mat4::from_rotation_translation(inverse, -(inverse * translation))
}

/// Parent index of `bones[index]`, `None` for a root or an invalid parent.
fn bone_parent(bones: &[Bone], index: usize) -> Option<usize> {
    bones[index]
        .parent_index
        .map(usize::from)
        .filter(|&parent| parent < bones.len() && parent != index)
}

/// Normalised influences of one vertex: a joint repeated in two slots is
/// summed, joints past `joint_count` are dropped, sorted by weight. Returns
/// the count of dropped influences, and `None` weights when nothing is left.
fn vertex_influences(map: &BoneMap12, joint_count: usize) -> (Option<([u16; 4], [f32; 4])>, usize) {
    let mut summed: Vec<(u16, f32)> = Vec::with_capacity(4);
    let mut dropped = 0;
    for (&joint, &weight) in map.joint_indices.iter().zip(&map.weights) {
        if weight == 0 {
            continue;
        }
        if joint as usize >= joint_count {
            dropped += 1;
            continue;
        }
        let weight = f32::from(weight) / 255.0;
        match summed.iter_mut().find(|(existing, _)| *existing == joint) {
            Some((_, total)) => *total += weight,
            None => summed.push((joint, weight)),
        }
    }
    let total: f32 = summed.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        return (None, dropped);
    }
    summed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut joints = [0u16; 4];
    let mut weights = [0f32; 4];
    for (slot, (joint, weight)) in summed.into_iter().enumerate() {
        joints[slot] = joint;
        weights[slot] = weight / total;
    }
    (Some((joints, weights)), dropped)
}

fn align4(bin: &mut Vec<u8>) {
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
}

fn push_accessor(
    builder: &mut GlbBuilder,
    offset: usize,
    count: usize,
    component: json::accessor::ComponentType,
    type_: json::accessor::Type,
    target: Option<json::buffer::Target>,
) -> u32 {
    let view = builder.buffer_views.len() as u32;
    builder.buffer_views.push(json::buffer::View {
        buffer: json::Index::new(0),
        byte_offset: Some(USize64(offset as u64)),
        byte_length: USize64((builder.bin.len() - offset) as u64),
        byte_stride: None,
        target: target.map(Checked::Valid),
        name: None,
        extensions: None,
        extras: Default::default(),
    });
    let accessor = builder.accessors.len() as u32;
    builder.accessors.push(json::Accessor {
        buffer_view: Some(json::Index::new(view)),
        byte_offset: Some(USize64(0)),
        count: USize64(count as u64),
        component_type: Checked::Valid(json::accessor::GenericComponentType(component)),
        type_: Checked::Valid(type_),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
        extensions: None,
        extras: Default::default(),
    });
    accessor
}

/// Write `mesh` as a glTF skinned mesh bound to `bones`.
///
/// `mesh.skin_weights` must hold a bone map for every vertex, with joint
/// indices into `bones` (see `types::build_skinned_mesh`).
pub fn write_skinned_glb(
    mesh: &Mesh,
    bones: &[Bone],
    name: Option<&str>,
    opts: &GlbOptions,
) -> Result<SkinnedGlb, Error> {
    if bones.is_empty() {
        return Err(Error::NotRiggable("skeleton has no bones".into()));
    }
    let Some(bone_maps) = mesh.skin_weights.as_ref() else {
        return Err(Error::NotRiggable("mesh has no bone map stream".into()));
    };
    let joint_count = bones.len();
    let vertex_count = mesh.positions.len();

    let mut builder = GlbBuilder::new();
    builder.include_vertex_colors = opts.include_vertex_colors;
    builder.include_damage = opts.include_damage;
    let packed = builder.pack_mesh(mesh, None, None, None, name, opts.material_mode, opts.preserve_textureless_decal_primitives);

    // Fallback joint for unweighted vertices: their submesh's node bone.
    let mut fallback_joint = vec![0u16; vertex_count];
    for submesh in &mesh.submeshes {
        let joint = submesh.node_parent_index;
        if joint as usize >= joint_count {
            continue;
        }
        let start = submesh.first_index as usize;
        let end = (start + submesh.num_indices as usize).min(mesh.indices.len());
        for &vertex in &mesh.indices[start..end] {
            if let Some(slot) = fallback_joint.get_mut(vertex as usize) {
                *slot = joint;
            }
        }
    }

    let mut joints = Vec::with_capacity(vertex_count);
    let mut weights = Vec::with_capacity(vertex_count);
    let mut unweighted_vertices = 0;
    let mut out_of_range_influences = 0;
    for vertex in 0..vertex_count {
        let (influences, dropped) = bone_maps
            .get(vertex)
            .map(|map| vertex_influences(map, joint_count))
            .unwrap_or((None, 0));
        out_of_range_influences += dropped;
        let (j, w) = influences.unwrap_or_else(|| {
            unweighted_vertices += 1;
            ([fallback_joint[vertex], 0, 0, 0], [1.0, 0.0, 0.0, 0.0])
        });
        joints.push(j);
        weights.push(w);
    }

    // JOINTS_0
    align4(&mut builder.bin);
    let joints_offset = builder.bin.len();
    let joints_component = if joint_count <= 256 {
        for j in &joints {
            builder.bin.extend(j.iter().map(|&joint| joint as u8));
        }
        json::accessor::ComponentType::U8
    } else {
        for j in &joints {
            for joint in j {
                builder.bin.extend_from_slice(&joint.to_le_bytes());
            }
        }
        json::accessor::ComponentType::U16
    };
    let joints_accessor = push_accessor(
        &mut builder,
        joints_offset,
        vertex_count,
        joints_component,
        json::accessor::Type::Vec4,
        Some(json::buffer::Target::ArrayBuffer),
    );

    // WEIGHTS_0
    align4(&mut builder.bin);
    let weights_offset = builder.bin.len();
    for w in &weights {
        for weight in w {
            builder.bin.extend_from_slice(&weight.to_le_bytes());
        }
    }
    let weights_accessor = push_accessor(
        &mut builder,
        weights_offset,
        vertex_count,
        json::accessor::ComponentType::F32,
        json::accessor::Type::Vec4,
        Some(json::buffer::Target::ArrayBuffer),
    );

    for primitive in &mut builder.meshes_json[packed.mesh_idx as usize].primitives {
        primitive
            .attributes
            .insert(Checked::Valid(json::mesh::Semantic::Joints(0)), json::Index::new(joints_accessor));
        primitive
            .attributes
            .insert(Checked::Valid(json::mesh::Semantic::Weights(0)), json::Index::new(weights_accessor));
    }

    // Joint nodes: local TRS = inverse(parent world) * world, so the chained
    // hierarchy reproduces each bone's bind-pose world matrix.
    let world: Vec<(glam::Quat, glam::Vec3)> = bones.iter().map(bone_world).collect();
    let first_joint_node = builder.nodes_json.len() as u32;
    let mut children: Vec<Vec<json::Index<json::Node>>> = vec![Vec::new(); joint_count];
    let mut roots = Vec::new();
    for index in 0..joint_count {
        match bone_parent(bones, index) {
            Some(parent) => children[parent].push(json::Index::new(first_joint_node + index as u32)),
            None => roots.push(json::Index::new(first_joint_node + index as u32)),
        }
    }
    for (index, bone) in bones.iter().enumerate() {
        let (rotation, translation) = match bone_parent(bones, index) {
            Some(parent) => {
                let parent_inverse = world[parent].0.conjugate();
                (
                    (parent_inverse * world[index].0).normalize(),
                    parent_inverse * (world[index].1 - world[parent].1),
                )
            }
            None => world[index],
        };
        let node_children = std::mem::take(&mut children[index]);
        builder.nodes_json.push(json::Node {
            name: Some(bone.name.clone()),
            translation: Some(translation.to_array()),
            rotation: Some(json::scene::UnitQuaternion(rotation.to_array())),
            children: (!node_children.is_empty()).then_some(node_children),
            ..Default::default()
        });
    }

    // inverseBindMatrices (no buffer view target: not a vertex attribute).
    align4(&mut builder.bin);
    let ibm_offset = builder.bin.len();
    for &transform in &world {
        for value in rigid_inverse(transform).to_cols_array() {
            builder.bin.extend_from_slice(&value.to_le_bytes());
        }
    }
    let ibm_accessor = push_accessor(
        &mut builder,
        ibm_offset,
        joint_count,
        json::accessor::ComponentType::F32,
        json::accessor::Type::Mat4,
        None,
    );

    builder.skins_json.push(json::Skin {
        extensions: None,
        extras: Default::default(),
        inverse_bind_matrices: Some(json::Index::new(ibm_accessor)),
        joints: (0..joint_count as u32).map(|i| json::Index::new(first_joint_node + i)).collect(),
        name: name.map(String::from),
        skeleton: (roots.len() == 1).then(|| roots[0]),
    });

    let mesh_node = builder.nodes_json.len() as u32;
    builder.nodes_json.push(json::Node {
        name: name.map(String::from),
        mesh: Some(json::Index::new(packed.mesh_idx)),
        skin: Some(json::Index::new(0)),
        ..Default::default()
    });
    builder.unwrapped_scene_nodes.push(json::Index::new(mesh_node));

    let glb = builder.finalize(roots, Vec::new(), &opts.metadata)?;
    Ok(SkinnedGlb {
        glb,
        vertex_count,
        joint_count,
        unweighted_vertices,
        out_of_range_influences,
        truncated_vertices: 0,
        max_truncated_weight: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SubMesh;

    fn bone(name: &str, parent: Option<u16>, position: [f32; 3], rotation_wxyz: [f32; 4]) -> Bone {
        Bone {
            name: name.into(),
            parent_index: parent,
            object_node_index: None,
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            world_position: position,
            world_rotation: rotation_wxyz,
        }
    }

    fn opts() -> GlbOptions {
        GlbOptions {
            material_mode: crate::pipeline::MaterialMode::None,
            preserve_textureless_decal_primitives: false,
            include_vertex_colors: false,
            include_damage: true,
            metadata: super::super::GlbMetadata {
                entity_name: None,
                geometry_path: None,
                material_path: None,
                export_options: super::super::ExportOptionsMetadata {
                    kind: "Bundled".into(),
                    material_mode: "None".into(),
                    format: "Glb".into(),
                    lod_level: 0,
                    texture_mip: 0,
                    include_attachments: false,
                    include_interior: false,
                },
            },
            fallback_palette: None,
        }
    }

    fn mesh() -> Mesh {
        let map = |joint_indices: [u16; 4], weights: [u8; 4]| BoneMap12 { joint_indices, weights };
        Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.5], [0.0, 0.0, 2.0]],
            indices: vec![0, 1, 2, 1, 2, 3],
            uvs: None,
            secondary_uvs: None,
            normals: None,
            tangents: None,
            colors: None,
            submeshes: vec![
                SubMesh {
                    material_name: None,
                    material_id: 0,
                    source_material_id: Some(0),
                    first_index: 0,
                    num_indices: 3,
                    first_vertex: 0,
                    num_vertices: 0,
                    node_parent_index: 0,
                },
                SubMesh {
                    material_name: None,
                    material_id: 1,
                    source_material_id: Some(1),
                    first_index: 3,
                    num_indices: 3,
                    first_vertex: 0,
                    num_vertices: 0,
                    node_parent_index: 1,
                },
            ],
            model_min: [0.0; 3],
            model_max: [1.0, 1.0, 2.0],
            scaling_min: [0.0; 3],
            scaling_max: [1.0, 1.0, 2.0],
            skin_weights: Some(vec![
                map([0, 0, 0, 0], [255, 0, 0, 0]),
                // Joint 1 named twice: summed.
                map([1, 0, 1, 0], [100, 55, 100, 0]),
                // Out-of-range joint dropped, rest renormalised.
                map([1, 9, 0, 0], [128, 127, 0, 0]),
                // Unweighted: bound to its submesh's node bone (1).
                map([0, 0, 0, 0], [0, 0, 0, 0]),
            ]),
        }
    }

    fn parse_glb(glb: &[u8]) -> (serde_json::Value, Vec<u8>) {
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let root = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        let bin_start = 20 + json_len;
        let bin_len = u32::from_le_bytes(glb[bin_start..bin_start + 4].try_into().unwrap()) as usize;
        (root, glb[bin_start + 8..bin_start + 8 + bin_len].to_vec())
    }

    fn accessor_bytes<'a>(root: &serde_json::Value, bin: &'a [u8], accessor: usize) -> &'a [u8] {
        let view = &root["bufferViews"][root["accessors"][accessor]["bufferView"].as_u64().unwrap() as usize];
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        &bin[offset..offset + view["byteLength"].as_u64().unwrap() as usize]
    }

    fn floats(root: &serde_json::Value, bin: &[u8], accessor: usize) -> Vec<f32> {
        accessor_bytes(root, bin, accessor)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn eight_influence_maps_keep_the_strongest_four() {
        let map = crate::ivo::skin::BoneMap24 {
            joint_indices: [3, 5, 3, 7, 9, 11, 13, 0],
            weights: [60, 50, 40, 40, 30, 20, 15, 0],
        };
        let (four, dropped) = map.strongest_four();
        assert_eq!(four.joint_indices, [3, 5, 7, 9]);
        assert_eq!(four.weights, [100, 50, 40, 30]);
        assert_eq!(dropped, 35);
    }

    #[test]
    fn vertex_influences_sum_repeats_and_normalise() {
        let (influences, dropped) =
            vertex_influences(&BoneMap12 { joint_indices: [1, 0, 1, 5], weights: [100, 55, 100, 10] }, 2);
        let (joints, weights) = influences.unwrap();
        assert_eq!(dropped, 1);
        assert_eq!(joints, [1, 0, 0, 0]);
        assert!((weights[0] - 200.0 / 255.0).abs() < 1e-6);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn write_skinned_glb_emits_skin_joints_and_weights() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let bones = vec![
            bone("Hips", None, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            // Rotated 90 degrees about Z and offset from its parent.
            bone("Spine", Some(0), [0.0, 0.2, 1.4], [half, 0.0, 0.0, half]),
        ];
        let out = write_skinned_glb(&mesh(), &bones, Some("piece"), &opts()).unwrap();
        assert_eq!(out.vertex_count, 4);
        assert_eq!(out.joint_count, 2);
        assert_eq!(out.unweighted_vertices, 1);
        assert_eq!(out.out_of_range_influences, 1);

        let (root, bin) = parse_glb(&out.glb);
        let node = |i: usize| &root["nodes"][i];
        let skin = &root["skins"][0];
        let joints: Vec<usize> = skin["joints"].as_array().unwrap().iter().map(|j| j.as_u64().unwrap() as usize).collect();
        let names: Vec<&str> = joints.iter().map(|&j| node(j)["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["Hips", "Spine"]);
        assert_eq!(skin["skeleton"].as_u64().unwrap() as usize, joints[0]);
        assert_eq!(node(joints[0])["children"][0].as_u64().unwrap() as usize, joints[1]);

        // The skinned mesh node is a scene root carrying the skin.
        let scene_roots: Vec<usize> = root["scenes"][0]["nodes"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect();
        let mesh_node = *scene_roots.iter().find(|&&n| node(n).get("mesh").is_some()).expect("mesh node at scene root");
        assert_eq!(node(mesh_node)["skin"], 0);
        assert!(node(mesh_node).get("matrix").is_none());

        // Chained local TRS times inverse bind matrix is identity per joint.
        let local = |n: &serde_json::Value| {
            let t: Vec<f32> = n["translation"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let r: Vec<f32> = n["rotation"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            glam::Mat4::from_rotation_translation(glam::Quat::from_xyzw(r[0], r[1], r[2], r[3]), glam::Vec3::new(t[0], t[1], t[2]))
        };
        let ibm = floats(&root, &bin, skin["inverseBindMatrices"].as_u64().unwrap() as usize);
        let hips = local(node(joints[0]));
        let spine = hips * local(node(joints[1]));
        assert!(spine.w_axis.truncate().abs_diff_eq(glam::Vec3::new(0.0, 0.2, 1.4), 1e-5));
        for (k, world) in [hips, spine].into_iter().enumerate() {
            let product = world * glam::Mat4::from_cols_slice(&ibm[k * 16..k * 16 + 16]);
            assert!(product.abs_diff_eq(glam::Mat4::IDENTITY, 1e-5), "{product:?}");
        }

        // Every primitive carries JOINTS_0 (u8) / WEIGHTS_0 summing to one.
        let primitives = root["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(primitives.len(), 2);
        for primitive in primitives {
            let joints_acc = primitive["attributes"]["JOINTS_0"].as_u64().unwrap() as usize;
            assert_eq!(root["accessors"][joints_acc]["componentType"], 5121);
            let joints: Vec<u8> = accessor_bytes(&root, &bin, joints_acc).to_vec();
            assert_eq!(joints, [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
            let weights = floats(&root, &bin, primitive["attributes"]["WEIGHTS_0"].as_u64().unwrap() as usize);
            for w in weights.chunks(4) {
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            }
            assert!((weights[4] - 200.0 / 255.0).abs() < 1e-6);
            assert!((weights[8] - 1.0).abs() < 1e-6);
            assert!((weights[12] - 1.0).abs() < 1e-6);
        }
    }
}
