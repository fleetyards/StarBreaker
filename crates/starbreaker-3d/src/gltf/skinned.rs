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
use crate::animation::SampledClip;
use crate::skeleton::{bind_local_transforms, bone_parent, bone_world_transform, Bone};
use crate::types::Mesh;

/// Result of [`write_animated_skeleton_glb`].
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

/// Inverse of a rigid transform, built from its parts so the bottom row is
/// exactly (0, 0, 0, 1) as glTF requires of inverse bind matrices.
fn rigid_inverse((rotation, translation): (glam::Quat, glam::Vec3)) -> glam::Mat4 {
    let inverse = rotation.conjugate();
    glam::Mat4::from_rotation_translation(inverse, -(inverse * translation))
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

/// Pack `values` as a float accessor of `count` elements (no buffer view
/// target: inverse bind matrices and animation data are not vertex data).
fn push_f32_accessor(builder: &mut GlbBuilder, values: &[f32], count: usize, type_: json::accessor::Type) -> u32 {
    align4(&mut builder.bin);
    let offset = builder.bin.len();
    for value in values {
        builder.bin.extend_from_slice(&value.to_le_bytes());
    }
    push_accessor(builder, offset, count, json::accessor::ComponentType::F32, type_, None)
}

/// Vertex skinning streams of a mesh, ready to attach to its primitives.
struct VertexSkinning {
    joints_accessor: u32,
    weights_accessor: u32,
    unweighted_vertices: usize,
    out_of_range_influences: usize,
}

/// Pack `JOINTS_0`/`WEIGHTS_0` for every vertex of `mesh`.
fn push_vertex_skinning(builder: &mut GlbBuilder, mesh: &Mesh, bone_maps: &[BoneMap12], joint_count: usize) -> VertexSkinning {
    let vertex_count = mesh.positions.len();
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
        builder,
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
        builder,
        weights_offset,
        vertex_count,
        json::accessor::ComponentType::F32,
        json::accessor::Type::Vec4,
        Some(json::buffer::Target::ArrayBuffer),
    );
    VertexSkinning { joints_accessor, weights_accessor, unweighted_vertices, out_of_range_influences }
}

/// Joint nodes (local TRS = bind pose, see `skeleton::bind_local_transforms`),
/// inverse bind matrices and the glTF skin. Returns the first joint node
/// index (joints are contiguous, in bone order) and the root joints.
fn push_joints(builder: &mut GlbBuilder, bones: &[Bone], name: Option<&str>) -> (u32, Vec<json::Index<json::Node>>) {
    let joint_count = bones.len();
    let world: Vec<(glam::Quat, glam::Vec3)> = bones.iter().map(bone_world_transform).collect();
    let local = bind_local_transforms(bones);
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
        let (rotation, translation) = local[index];
        let node_children = std::mem::take(&mut children[index]);
        builder.nodes_json.push(json::Node {
            name: Some(bone.name.clone()),
            translation: Some(translation.to_array()),
            rotation: Some(json::scene::UnitQuaternion(rotation.to_array())),
            children: (!node_children.is_empty()).then_some(node_children),
            ..Default::default()
        });
    }

    let ibm: Vec<f32> = world.iter().flat_map(|&transform| rigid_inverse(transform).to_cols_array()).collect();
    let ibm_accessor = push_f32_accessor(builder, &ibm, joint_count, json::accessor::Type::Mat4);

    builder.skins_json.push(json::Skin {
        extensions: None,
        extras: Default::default(),
        inverse_bind_matrices: Some(json::Index::new(ibm_accessor)),
        joints: (0..joint_count as u32).map(|i| json::Index::new(first_joint_node + i)).collect(),
        name: name.map(String::from),
        skeleton: (roots.len() == 1).then(|| roots[0]),
    });
    (first_joint_node, roots)
}

/// Bones animated by any clip, for rotation and for translation. Every clip
/// writes a track for each of them (bind pose where it has no channel), so
/// switching clips in a viewer that does not reset the pose leaves no
/// residue of the previous clip.
fn animated_bones(clips: &[SampledClip], joint_count: usize) -> (Vec<usize>, Vec<usize>) {
    let mut rotated = vec![false; joint_count];
    let mut translated = vec![false; joint_count];
    for clip in clips {
        for (bone, _) in &clip.rotations {
            rotated[*bone] = true;
        }
        for (bone, _) in &clip.translations {
            translated[*bone] = true;
        }
    }
    let collect = |flags: Vec<bool>| flags.into_iter().enumerate().filter(|(_, f)| *f).map(|(i, _)| i).collect();
    (collect(rotated), collect(translated))
}

/// One glTF animation per clip, channels on the joint nodes.
fn push_animations(builder: &mut GlbBuilder, bones: &[Bone], first_joint_node: u32, clips: &[SampledClip]) -> Result<(), Error> {
    let rest = bind_local_transforms(bones);
    let (rotated, translated) = animated_bones(clips, bones.len());
    for clip in clips {
        let frames = clip.frames.max(1);
        let times: Vec<f32> = (0..frames).map(|k| k as f32 / clip.fps).collect();
        let input = push_f32_accessor(builder, &times, frames, json::accessor::Type::Scalar);
        builder.accessors[input as usize].min = Some(serde_json::json!([times[0]]));
        builder.accessors[input as usize].max = Some(serde_json::json!([times[frames - 1]]));
        let interpolation = if frames == 1 {
            json::animation::Interpolation::Step
        } else {
            json::animation::Interpolation::Linear
        };

        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        let mut push_channel = |builder: &mut GlbBuilder, bone: usize, values: Vec<f32>, type_, path| {
            let output = push_f32_accessor(builder, &values, frames, type_);
            channels.push(json::animation::Channel {
                sampler: json::Index::new(samplers.len() as u32),
                target: json::animation::Target {
                    node: json::Index::new(first_joint_node + bone as u32),
                    path: Checked::Valid(path),
                    extensions: None,
                    extras: Default::default(),
                },
                extensions: None,
                extras: Default::default(),
            });
            samplers.push(json::animation::Sampler {
                input: json::Index::new(input),
                interpolation: Checked::Valid(interpolation),
                output: json::Index::new(output),
                extensions: None,
                extras: Default::default(),
            });
        };

        for &bone in &rotated {
            let mut samples: Vec<glam::Quat> = match clip.rotations.iter().find(|(b, _)| *b == bone) {
                Some((_, samples)) => samples.iter().map(|q| q.normalize()).collect(),
                None => vec![rest[bone].0; frames],
            };
            samples.resize(frames, *samples.last().unwrap_or(&rest[bone].0));
            for k in 1..frames {
                if samples[k - 1].dot(samples[k]) < 0.0 {
                    samples[k] = -samples[k];
                }
            }
            let values = samples.iter().flat_map(|q| q.to_array()).collect();
            push_channel(builder, bone, values, json::accessor::Type::Vec4, json::animation::Property::Rotation);
        }
        for &bone in &translated {
            let mut samples: Vec<glam::Vec3> = match clip.translations.iter().find(|(b, _)| *b == bone) {
                Some((_, samples)) => samples.clone(),
                None => vec![rest[bone].1; frames],
            };
            samples.resize(frames, *samples.last().unwrap_or(&rest[bone].1));
            let values = samples.iter().flat_map(|p| p.to_array()).collect();
            push_channel(builder, bone, values, json::accessor::Type::Vec3, json::animation::Property::Translation);
        }

        let extras = serde_json::json!({
            "source": clip.source,
            "fps": clip.fps,
            "frames": frames,
            "source_frames": clip.source_frames,
            "static": clip.is_static,
            "looping": clip.looping,
            "root_motion_removed_m": clip.root_motion_removed,
        });
        builder.animations_json.push(json::Animation {
            name: Some(clip.name.clone()),
            channels,
            samplers,
            extensions: None,
            extras: Some(serde_json::value::RawValue::from_string(extras.to_string())?),
        });
    }
    Ok(())
}

/// Write the rigged skeleton of `bones` with one glTF animation per clip,
/// and `meshes` (each with its node name) as skinned meshes bound to it, all
/// on the one glTF skin. Without meshes the skin is still written, so
/// importers build an armature from the joints.
///
/// Each `mesh.skin_weights` must hold a bone map for every vertex, with joint
/// indices into `bones` (see `types::build_skinned_mesh`).
pub fn write_animated_skeleton_glb(
    meshes: &[(&Mesh, Option<&str>)],
    bones: &[Bone],
    clips: &[SampledClip],
    name: Option<&str>,
    opts: &GlbOptions,
) -> Result<SkinnedGlb, Error> {
    if bones.is_empty() {
        return Err(Error::NotRiggable("skeleton has no bones".into()));
    }
    let joint_count = bones.len();

    let mut builder = GlbBuilder::new();
    builder.include_vertex_colors = opts.include_vertex_colors;
    builder.include_damage = opts.include_damage;
    let mut skinned_meshes = Vec::with_capacity(meshes.len());
    let (mut unweighted_vertices, mut out_of_range_influences) = (0, 0);
    for &(mesh, mesh_name) in meshes {
        let Some(bone_maps) = mesh.skin_weights.as_ref() else {
            return Err(Error::NotRiggable("mesh has no bone map stream".into()));
        };
        let packed = builder.pack_mesh(mesh, None, None, None, mesh_name, opts.material_mode, opts.preserve_textureless_decal_primitives);
        let skinning = push_vertex_skinning(&mut builder, mesh, bone_maps, joint_count);
        unweighted_vertices += skinning.unweighted_vertices;
        out_of_range_influences += skinning.out_of_range_influences;
        for primitive in &mut builder.meshes_json[packed.mesh_idx as usize].primitives {
            primitive
                .attributes
                .insert(Checked::Valid(json::mesh::Semantic::Joints(0)), json::Index::new(skinning.joints_accessor));
            primitive
                .attributes
                .insert(Checked::Valid(json::mesh::Semantic::Weights(0)), json::Index::new(skinning.weights_accessor));
        }
        skinned_meshes.push((packed.mesh_idx, mesh_name));
    }

    let (first_joint_node, roots) = push_joints(&mut builder, bones, name);

    for (mesh_idx, mesh_name) in skinned_meshes {
        let mesh_node = builder.nodes_json.len() as u32;
        builder.nodes_json.push(json::Node {
            name: mesh_name.map(String::from),
            mesh: Some(json::Index::new(mesh_idx)),
            skin: Some(json::Index::new(0)),
            ..Default::default()
        });
        builder.unwrapped_scene_nodes.push(json::Index::new(mesh_node));
    }
    push_animations(&mut builder, bones, first_joint_node, clips)?;

    let glb = builder.finalize(roots, Vec::new(), &opts.metadata)?;
    Ok(SkinnedGlb {
        glb,
        vertex_count: meshes.iter().map(|(m, _)| m.positions.len()).sum(),
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
        let out = write_animated_skeleton_glb(&[(&mesh(), Some("piece"))], &bones, &[], Some("piece"), &opts()).unwrap();
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

    fn sampled(name: &str, frames: usize, rotations: Vec<(usize, Vec<glam::Quat>)>, translations: Vec<(usize, Vec<glam::Vec3>)>) -> SampledClip {
        SampledClip {
            name: name.into(),
            source: format!("test.dba#{name}"),
            fps: 30.0,
            frames,
            source_frames: frames.saturating_sub(1) as f32,
            is_static: frames == 1,
            looping: false,
            rotations,
            translations,
            matched_channels: 1,
            unmatched_channels: 0,
            root_motion_removed: 0.0,
        }
    }

    #[test]
    fn skeleton_only_glb_carries_one_animation_per_clip_on_the_joints() {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let bones = vec![
            bone("Hips", None, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            bone("Spine", Some(0), [0.0, 0.2, 1.4], [half, 0.0, 0.0, half]),
        ];
        let pose = glam::Quat::from_rotation_x(0.5);
        let clips = vec![
            sampled("pose", 1, vec![(1, vec![pose])], vec![]),
            sampled(
                "bob",
                3,
                vec![],
                vec![(0, vec![glam::Vec3::new(0.0, 0.0, 1.0), glam::Vec3::new(0.0, 0.0, 1.05), glam::Vec3::new(0.0, 0.0, 1.0)])],
            ),
        ];
        let out = write_animated_skeleton_glb(&[], &bones, &clips, Some("rig"), &opts()).unwrap();
        assert_eq!(out.vertex_count, 0);
        let (root, bin) = parse_glb(&out.glb);
        assert!(root.get("meshes").is_none_or(|m| m.as_array().unwrap().is_empty()));
        let joints: Vec<u64> = root["skins"][0]["joints"].as_array().unwrap().iter().map(|j| j.as_u64().unwrap()).collect();
        let animations = root["animations"].as_array().unwrap();
        assert_eq!(animations.len(), 2);
        assert_eq!(animations[0]["name"], "pose");
        assert_eq!(animations[0]["extras"]["static"], true);
        assert_eq!(animations[1]["extras"]["source"], "test.dba#bob");

        for animation in animations {
            // Spine rotation and Hips translation in both clips: the clip
            // without a channel holds the bind pose.
            let channels = animation["channels"].as_array().unwrap();
            assert_eq!(channels.len(), 2);
            let mut paths: Vec<(u64, &str)> = channels
                .iter()
                .map(|c| (c["target"]["node"].as_u64().unwrap(), c["target"]["path"].as_str().unwrap()))
                .collect();
            paths.sort();
            assert_eq!(paths, [(joints[0], "translation"), (joints[1], "rotation")]);
        }

        let sampler = |a: usize, path: &str| {
            let animation = &animations[a];
            let channel = animation["channels"].as_array().unwrap().iter().find(|c| c["target"]["path"] == path).unwrap();
            &animation["samplers"][channel["sampler"].as_u64().unwrap() as usize]
        };
        let pose_rotation = sampler(0, "rotation");
        assert_eq!(pose_rotation["interpolation"], "STEP");
        let output = floats(&root, &bin, pose_rotation["output"].as_u64().unwrap() as usize);
        assert!(glam::Quat::from_slice(&output).abs_diff_eq(pose, 1e-6));
        // Hips translation in the static clip = the Hips rest translation.
        let rest = floats(&root, &bin, sampler(0, "translation")["output"].as_u64().unwrap() as usize);
        assert_eq!(rest, [0.0, 0.0, 1.0]);

        let bob = sampler(1, "translation");
        assert_eq!(bob["interpolation"], "LINEAR");
        let input = bob["input"].as_u64().unwrap() as usize;
        assert_eq!(floats(&root, &bin, input), [0.0, 1.0 / 30.0, 2.0 / 30.0]);
        assert_eq!(root["accessors"][input]["max"][0].as_f64().unwrap() as f32, 2.0 / 30.0);
        let bob_values = floats(&root, &bin, bob["output"].as_u64().unwrap() as usize);
        assert!((bob_values[5] - 1.05).abs() < 1e-6);
        // Spine rotation held at its bind local (parent Hips is unrotated).
        let spine_rest = floats(&root, &bin, sampler(1, "rotation")["output"].as_u64().unwrap() as usize);
        assert_eq!(spine_rest.len(), 12);
        let spine_node = &root["nodes"][joints[1] as usize]["rotation"];
        for k in 0..4 {
            assert!((spine_rest[k] - spine_node[k].as_f64().unwrap() as f32).abs() < 1e-6);
        }
    }
}
