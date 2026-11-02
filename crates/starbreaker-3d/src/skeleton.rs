//! Parse skeleton bones from `.chr` files (CompiledBones chunks).
//!
//! Skeleton bones provide attachment points (e.g., `magAttach`) that loadout
//! children reference by name when attaching to CDF weapon entities.
//!
//! Two chunk versions exist:
//! - v900 (`0xC2011111`): per-bone entries with inline transforms (68 bytes each)
//! - v901 (`0xC201973C`): compact entries (16 bytes) with separate transform blocks

use starbreaker_chunks::ChunkFile;
use starbreaker_chunks::known_types::ivo;
use starbreaker_common::reader::SpanReader;
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// A parsed skeleton bone with name and world transform.
#[derive(Debug, Clone)]
pub struct Bone {
    pub name: String,
    pub parent_index: Option<u16>,
    pub object_node_index: Option<u16>,
    /// Parent-relative position [x, y, z]
    pub local_position: [f32; 3],
    /// Parent-relative rotation quaternion [w, x, y, z]
    pub local_rotation: [f32; 4],
    /// World-space position [x, y, z]
    pub world_position: [f32; 3],
    /// World-space rotation quaternion [w, x, y, z]
    pub world_rotation: [f32; 4],
}

// ── Binary layouts ──────────────────────────────────────────────────────────

/// On-disk quaternion + translation (28 bytes). Quaternion stored as (x, y, z, w).
#[derive(Clone, Copy, FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawQuatTrans {
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    tx: f32,
    ty: f32,
    tz: f32,
}

/// v900 per-bone entry (68 bytes): IDs + relative transform + world transform.
#[derive(Clone, Copy, FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct BoneEntryV900 {
    controller_id: u32,
    limb_id: u32,
    parent_index: i32,
    relative: RawQuatTrans,
    world: RawQuatTrans,
}

/// v901 per-bone entry (16 bytes): compact metadata, transforms stored separately.
#[derive(Clone, Copy, FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct BoneEntryV901 {
    controller_id: u32,
    limb_id: u16,
    num_children: u16,
    parent_index: i16,
    _unknown1: i16,
    _unknown2: i16,
    object_node_index: i16,
}

// ── Size assertions ─────────────────────────────────────────────────────────

const _: () = {
    assert!(size_of::<RawQuatTrans>() == 28);
    assert!(size_of::<BoneEntryV900>() == 68);
    assert!(size_of::<BoneEntryV901>() == 16);
};

// ── Parsing ─────────────────────────────────────────────────────────────────

impl RawQuatTrans {
    fn rotation_wxyz(&self) -> [f32; 4] {
        [self.qw, self.qx, self.qy, self.qz]
    }

    fn position_xyz(&self) -> [f32; 3] {
        [self.tx, self.ty, self.tz]
    }
}

/// Resolve a list of rig-like node names from a CryEngine asset, trying
/// the CHR `CompiledBones` chunk first and falling back to the CGA/CGAM
/// NMC scene-graph node names.
///
/// CHR files (skeletons used by skinned meshes) expose names through
/// `parse_skeleton`. CGA/CGAM files (rigid scene graphs, e.g. Scorpius main
/// body) have no CompiledBones — their bone-equivalent names live in the
/// NMC chunk as scene-graph node names. Both name sources hash to the same
/// 32-bit space used by `.dba` channel hashes, so either can be used to
/// resolve animation channels by name.
///
/// Returns `None` if neither chunk is present or parseable. Order is
/// preserved from the source (bone order for CHR, node order for NMC) so
/// callers that care about stable output get deterministic results.
pub fn parse_rig_node_names(data: &[u8]) -> Option<Vec<String>> {
    if let Some(bones) = parse_skeleton(data) {
        if !bones.is_empty() {
            return Some(bones.into_iter().map(|b| b.name).collect());
        }
    }
    if let Some((nodes, _)) = crate::nmc::parse_nmc_full(data) {
        if !nodes.is_empty() {
            return Some(nodes.into_iter().map(|n| n.name).collect());
        }
    }
    None
}

/// Parse the CompiledBones chunk from a `.chr` skeleton file.
///
/// Returns `None` if the file can't be parsed or has no CompiledBones chunk.
pub fn parse_skeleton(data: &[u8]) -> Option<Vec<Bone>> {
    let chunk_file = ChunkFile::from_bytes(data).ok()?;
    let ivo_file = match &chunk_file {
        ChunkFile::Ivo(ivo_file) => ivo_file,
        ChunkFile::CrCh(_) => return None,
    };

    for entry in ivo_file.chunks() {
        let chunk = ivo_file.chunk_data(entry);
        if entry.chunk_type == ivo::COMPILED_BONES {
            return parse_compiled_bones_v901(chunk);
        }
        if entry.chunk_type == ivo::COMPILED_BONES_IVO320 {
            return parse_compiled_bones_v900(chunk);
        }
    }

    None
}

/// Parse null-separated bone name strings from a byte slice.
fn parse_bone_names(data: &[u8], expected: usize) -> Option<Vec<String>> {
    let names: Vec<String> = data
        .split(|&b| b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();

    if names.len() < expected {
        log::warn!("CompiledBones: expected {expected} names, got {}", names.len());
        return None;
    }
    Some(names)
}

fn log_bones(bones: &[Bone], version: &str) {
    log::debug!(
        "Parsed {} skeleton bones ({version}): {}",
        bones.len(),
        bones.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ")
    );

    if std::env::var("SB_DEBUG_RIGID_BIND").is_ok() {
        for bone in bones {
            log::debug!(
                "  bone {} parent={:?} object_node_index={:?} local_pos={:?} local_rot={:?}",
                bone.name,
                bone.parent_index,
                bone.object_node_index,
                bone.local_position,
                bone.local_rotation,
            );
        }
    }
}

/// Parse CompiledBones v900 (chunk type `0xC2011111`).
fn parse_compiled_bones_v900(data: &[u8]) -> Option<Vec<Bone>> {
    let mut r = SpanReader::new(data);
    let num_bones = r.read_u32().ok()? as usize;
    if num_bones == 0 || num_bones > 1000 {
        return None;
    }

    let entries = r.read_slice::<BoneEntryV900>(num_bones).ok()?;
    let names = parse_bone_names(r.remaining_bytes(), num_bones)?;

    let bones: Vec<Bone> = entries
        .iter()
        .zip(names)
        .map(|(e, name)| {
            let parent_index = (e.parent_index >= 0).then_some(e.parent_index as u16);
            Bone {
                name,
                parent_index,
                object_node_index: None,
                local_position: e.relative.position_xyz(),
                local_rotation: e.relative.rotation_wxyz(),
                world_position: e.world.position_xyz(),
                world_rotation: e.world.rotation_wxyz(),
            }
        })
        .collect();

    log_bones(&bones, "v900");
    Some(bones)
}

/// Parse CompiledBones v901 (chunk type `0xC201973C`).
fn parse_compiled_bones_v901(data: &[u8]) -> Option<Vec<Bone>> {
    let mut r = SpanReader::new(data);
    let num_bones = r.read_u32().ok()? as usize;
    let string_table_size = r.read_u32().ok()? as usize;
    let _flags1 = r.read_u32().ok()?;
    let _flags2 = r.read_u32().ok()?;
    r.advance(32).ok()?; // padding

    if num_bones == 0 || num_bones > 1000 {
        return None;
    }

    // Skip compact bone entries
    let entries = r.read_slice::<BoneEntryV901>(num_bones).ok()?;

    // Read string table
    let string_bytes = r.read_bytes(string_table_size).ok()?;
    let names = parse_bone_names(string_bytes, num_bones)?;

    let relative_transforms = r.read_slice::<RawQuatTrans>(num_bones).ok()?;
    let world_transforms = r.read_slice::<RawQuatTrans>(num_bones).ok()?;

    let bones: Vec<Bone> = relative_transforms
        .iter()
        .zip(world_transforms.iter())
        .zip(entries.iter().zip(names))
        .map(|((relative, world), (entry, name))| {
            let parent_index = (entry.parent_index >= 0).then_some(entry.parent_index as u16);
            Bone {
                name,
                parent_index,
                object_node_index: (entry.object_node_index >= 0)
                    .then_some(entry.object_node_index as u16),
                local_position: relative.position_xyz(),
                local_rotation: relative.rotation_wxyz(),
                world_position: world.position_xyz(),
                world_rotation: world.rotation_wxyz(),
            }
        })
        .collect();

    log_bones(&bones, "v901");
    Some(bones)
}

impl crate::animation::BoneTransforms for Bone {
    fn name(&self) -> &str {
        &self.name
    }
    fn parent_index(&self) -> Option<usize> {
        self.parent_index.map(|p| p as usize)
    }
    fn local_rotation_wxyz(&self) -> [f32; 4] {
        self.local_rotation
    }
    fn local_position(&self) -> [f32; 3] {
        self.local_position
    }
    fn set_local_rotation_wxyz(&mut self, q: [f32; 4]) {
        self.local_rotation = q;
    }
    fn set_local_position(&mut self, p: [f32; 3]) {
        self.local_position = p;
    }
    fn set_world_rotation_wxyz(&mut self, q: [f32; 4]) {
        self.world_rotation = q;
    }
    fn set_world_position(&mut self, p: [f32; 3]) {
        self.world_position = p;
    }
}

/// Parent index of `bones[index]`, `None` for a root or an invalid parent.
pub fn bone_parent(bones: &[Bone], index: usize) -> Option<usize> {
    bones[index]
        .parent_index
        .map(usize::from)
        .filter(|&parent| parent < bones.len() && parent != index)
}

/// Bind-pose world transform (model space) of a bone as rotation + translation.
pub fn bone_world_transform(bone: &Bone) -> (glam::Quat, glam::Vec3) {
    let [w, x, y, z] = bone.world_rotation;
    (glam::Quat::from_xyzw(x, y, z, w).normalize(), glam::Vec3::from(bone.world_position))
}

/// Parent-relative bind-pose transform of every bone, derived from the world
/// transforms (`inverse(parent world) * world`), so a chained hierarchy
/// reproduces each bone's bind-pose world matrix exactly. Roots keep their
/// world transform. This is the rest TRS of the rigged GLB joint nodes.
pub fn bind_local_transforms(bones: &[Bone]) -> Vec<(glam::Quat, glam::Vec3)> {
    let world: Vec<(glam::Quat, glam::Vec3)> = bones.iter().map(bone_world_transform).collect();
    (0..bones.len())
        .map(|index| match bone_parent(bones, index) {
            Some(parent) => {
                let parent_inverse = world[parent].0.conjugate();
                (
                    (parent_inverse * world[index].0).normalize(),
                    parent_inverse * (world[index].1 - world[parent].1),
                )
            }
            None => world[index],
        })
        .collect()
}

/// A bone [`graft_missing_bones`] added: its name, the name of the bone it
/// hangs under, and whether it is only an ancestor of a grafted helper (a
/// non-matching bone the donor chain needed because the target lacks it).
#[derive(Debug, Clone, PartialEq)]
pub struct GraftedBone {
    pub name: String,
    pub parent: String,
    pub ancestor: bool,
}

/// Copy the bones of `donor` whose name ends with `suffix` (case-insensitive,
/// e.g. `_override`: the item-port helpers of a worn `.skin`) and that
/// `bones` lacks into `bones`, appended after the existing ones (so the mesh
/// bone maps stay valid). Each keeps the donor's parent-relative bind
/// transform (`inverse(donor parent world) * donor world`) under the bone of
/// the same name in `bones`: the helper rides its parent the way it does in
/// the donor skeleton, whatever the two skeletons' bind poses. A donor
/// ancestor that `bones` lacks is grafted the same way first (marked
/// `ancestor`); a helper whose donor chain reaches the root without a bone
/// `bones` has is skipped. Returns the bones added, in order.
pub fn graft_missing_bones(bones: &mut Vec<Bone>, donor: &[Bone], suffix: &str) -> Vec<GraftedBone> {
    let suffix = suffix.to_lowercase();
    let mut index: std::collections::HashMap<String, usize> =
        bones.iter().enumerate().map(|(i, b)| (b.name.to_lowercase(), i)).collect();
    let mut added = Vec::new();
    for start in 0..donor.len() {
        let lower = donor[start].name.to_lowercase();
        if !lower.ends_with(&suffix) || index.contains_key(&lower) {
            continue;
        }
        // Donor chain up to the first bone the target has.
        let mut chain = vec![start];
        let anchor = loop {
            let Some(parent) = bone_parent(donor, *chain.last().unwrap()) else { break None };
            if chain.contains(&parent) {
                break None;
            }
            if let Some(&target) = index.get(&donor[parent].name.to_lowercase()) {
                break Some((parent, target));
            }
            chain.push(parent);
        };
        let Some((mut donor_parent, mut target_parent)) = anchor else { continue };
        for &d in chain.iter().rev() {
            let (dpr, dpt) = bone_world_transform(&donor[donor_parent]);
            let (dr, dt) = bone_world_transform(&donor[d]);
            let local_rotation = (dpr.conjugate() * dr).normalize();
            let local_position = dpr.conjugate() * (dt - dpt);
            let (pr, pt) = bone_world_transform(&bones[target_parent]);
            let rotation = (pr * local_rotation).normalize();
            let position = pt + pr * local_position;
            let parent_name = bones[target_parent].name.clone();
            bones.push(Bone {
                name: donor[d].name.clone(),
                parent_index: Some(target_parent as u16),
                object_node_index: None,
                local_position: local_position.to_array(),
                local_rotation: [local_rotation.w, local_rotation.x, local_rotation.y, local_rotation.z],
                world_position: position.to_array(),
                world_rotation: [rotation.w, rotation.x, rotation.y, rotation.z],
            });
            let new = bones.len() - 1;
            index.insert(donor[d].name.to_lowercase(), new);
            added.push(GraftedBone { name: donor[d].name.clone(), parent: parent_name, ancestor: d != start });
            donor_parent = d;
            target_parent = new;
        }
    }
    added
}

#[cfg(test)]
mod graft_tests {
    use super::*;

    fn bone(name: &str, parent: Option<u16>, rotation: glam::Quat, position: glam::Vec3) -> Bone {
        Bone {
            name: name.into(),
            parent_index: parent,
            object_node_index: None,
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            world_position: position.to_array(),
            world_rotation: [rotation.w, rotation.x, rotation.y, rotation.z],
        }
    }

    #[test]
    fn grafts_helpers_in_the_parent_frame() {
        let rz = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        // Donor: Spine at 1 m, rotated 90 deg about Z; Thigh under it; a helper
        // 0.1 m along the spine's +X, and one under a thigh twist the target lacks.
        let donor = vec![
            bone("Root", None, glam::Quat::IDENTITY, glam::Vec3::ZERO),
            bone("Spine", Some(0), rz, glam::Vec3::new(0.0, 0.0, 1.0)),
            bone("backpack_attach_1_override", Some(1), rz, glam::Vec3::new(0.0, 0.1, 1.0)),
            bone("Thigh_sIk", Some(0), glam::Quat::IDENTITY, glam::Vec3::new(0.2, 0.0, 0.5)),
            bone("wep_sidearm_attach_override", Some(3), glam::Quat::IDENTITY, glam::Vec3::new(0.3, 0.0, 0.5)),
            bone("Spine_override", Some(1), rz, glam::Vec3::new(0.0, 0.0, 1.0)),
        ];
        // Target: its Spine sits 0.9 m up with no rotation; it already has Spine_override.
        let mut target = vec![
            bone("Root", None, glam::Quat::IDENTITY, glam::Vec3::ZERO),
            bone("Spine", Some(0), glam::Quat::IDENTITY, glam::Vec3::new(0.0, 0.0, 0.9)),
            bone("spine_override", Some(1), glam::Quat::IDENTITY, glam::Vec3::new(0.0, 0.0, 0.9)),
        ];
        let added = graft_missing_bones(&mut target, &donor, "_override");
        assert_eq!(
            added.iter().map(|g| (g.name.as_str(), g.parent.as_str(), g.ancestor)).collect::<Vec<_>>(),
            [
                ("backpack_attach_1_override", "Spine", false),
                ("Thigh_sIk", "Root", true),
                ("wep_sidearm_attach_override", "Thigh_sIk", false),
            ]
        );
        assert_eq!(target.len(), 6);
        // 0.1 m along the spine's +X: the target spine is unrotated -> world +X.
        let (r, t) = bone_world_transform(&target[3]);
        assert!(t.abs_diff_eq(glam::Vec3::new(0.1, 0.0, 0.9), 1e-5), "{t:?}");
        assert!(r.abs_diff_eq(glam::Quat::IDENTITY, 1e-5));
        assert_eq!(target[3].parent_index, Some(1));
        assert_eq!(target[5].parent_index, Some(4));
        let (_, t) = bone_world_transform(&target[5]);
        assert!(t.abs_diff_eq(glam::Vec3::new(0.3, 0.0, 0.5), 1e-5), "{t:?}");
        // Re-deriving the locals reproduces the donor's.
        let local = bind_local_transforms(&target);
        assert!(local[3].1.abs_diff_eq(glam::Vec3::new(0.1, 0.0, 0.0), 1e-5));
    }
}
