use starbreaker_common::ParseError;
use starbreaker_common::reader::SpanReader;
use zerocopy::{FromBytes, Immutable, KnownLayout};

// ── Raw on-disk layouts ─────────────────────────────────────────────────────

/// Raw MeshInfo as stored on disk (76 bytes).
#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawMeshInfo {
    flags2: u32,
    num_vertices: u32,
    num_indices: u32,
    num_submeshes: u32,
    _unknown: u32,
    model_min: [f32; 3],
    model_max: [f32; 3],
    min_bound: [f32; 3],
    max_bound: [f32; 3],
    _vertex_format: u32,
    extra_count: u32,
}

/// Raw SubMeshDescriptor as stored on disk (48 bytes).
#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawSubMeshDescriptor {
    mat_id: u16,
    node_parent_index: u16,
    first_index: u32,
    num_indices: u32,
    first_vertex: u32,
    page_base: u32,
    num_vertices: u32,
    radius: f32,
    center: [f32; 3],
    _unknown0: u32,
    _unknown1: u32,
}

/// Quantized vertex: SNorm i16×3 + pad + BGRA + UV half×2 (16 bytes).
#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawQuantizedVertex {
    pos: [u16; 3],
    _pad: u16,
    color: [u8; 4],
    uv: [u16; 2],
}

/// Float vertex: f32×3 + BGRA + UV half×2 (20 bytes).
#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawFloatVertex {
    pos: [f32; 3],
    color: [u8; 4],
    uv: [u16; 2],
}

#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawBoneMap12 {
    joint_indices: [u16; 4],
    weights: [u8; 4],
}

/// IVOBONEMAP32 entry (24 bytes): eight u16 joints, eight u8 weights
/// summing to 255. Character bodies use it.
#[derive(Clone, Copy, FromBytes, KnownLayout, Immutable)]
#[repr(C, packed)]
struct RawBoneMap24 {
    joint_indices: [u16; 8],
    weights: [u8; 8],
}

const _: () = {
    assert!(size_of::<RawMeshInfo>() == 76);
    assert!(size_of::<RawSubMeshDescriptor>() == 48);
    assert!(size_of::<RawQuantizedVertex>() == 16);
    assert!(size_of::<RawFloatVertex>() == 20);
    assert!(size_of::<RawBoneMap12>() == 12);
};

/// Datastream type tag constants.
///
/// Names are from cryengine-converter/tmlaw templates. Some are misleading:
/// - IVOQTANGENTS is NOT quaternions — it's compressed tangent vectors (15-15-1-1 bits)
/// - IVONORMALS2 is dual-purpose: elem_size=12 → f32×3 normals, elem_size=4 → second UV set
///   (elem_size=4 maps to D3D SECOND_UV semantic, not normals)
/// - IVOTANGENTS IS quaternions — i16 SNorm XYZW (CryEngine SPipQTangents format)
pub mod stream_type {
    pub const IVOVERTSUVS: u32 = 0x91329AE9;
    pub const IVOVERTSUVS2: u32 = 0xB3A70D5E;
    pub const IVOINDICES: u32 = 0xEECDC168;
    pub const IVONORMALS: u32 = 0x9CF3F615;
    /// Dual-purpose stream: elem_size=12 → f32×3 normals, elem_size=4 → SECOND_UV (half×2).
    /// Name inherited from cryengine-converter; elem_size=4 variant is actually SECOND_UV
    /// per D3D semantic mapping (SECOND_UV → slot 0xe, flag 0x40).
    pub const IVONORMALS2: u32 = 0x38A581FE;
    pub const IVOTANGENTS: u32 = 0xB95E9A1B;
    pub const IVOQTANGENTS: u32 = 0xEE057252;
    pub const IVOBONEMAP: u32 = 0x677C7B23;
    pub const IVOBONEMAP32: u32 = 0x6ECA3708;
    pub const IVOCOLORS2: u32 = 0xD9EED421;
    pub const IVOUNKNOWN: u32 = 0x9D51C5EE;
}

#[derive(Debug, Clone)]
pub struct MeshInfo {
    pub flags2: u32,
    pub num_vertices: u32,
    pub num_indices: u32,
    pub num_submeshes: u32,
    /// First bounding box — original model-space extent.
    /// NMC scene graph transforms are in this coordinate system.
    pub model_min: [f32; 3],
    pub model_max: [f32; 3],
    /// Second bounding box ("scaling bbox") — the correct extent for SNorm dequantization.
    pub min_bound: [f32; 3],
    pub max_bound: [f32; 3],
    pub extra_count: u32,
}

#[derive(Debug, Clone)]
pub struct SubMeshDescriptor {
    /// Material index into the MTL submaterial list.
    pub mat_id: u16,
    /// Node index in the NodeMeshCombo (which node this submesh belongs to).
    pub node_parent_index: u16,
    pub first_index: u32,
    pub num_indices: u32,
    pub first_vertex: u32,
    /// Index page base for u16 index addressing on meshes with >65535 vertices.
    /// Add to each index in this submesh's range to get the absolute vertex index.
    pub page_base: u32,
    pub num_vertices: u32,
    pub radius: f32,
    pub center: [f32; 3],
    pub unknown0: u32,
    pub unknown1: u32,
}

#[derive(Debug)]
pub struct DataStreams {
    pub positions: PositionData,
    pub uvs: Vec<[u16; 2]>,
    pub secondary_uvs: Option<Vec<[u16; 2]>>,
    pub indices: Vec<u32>,
    pub bone_maps: Option<Vec<BoneMap12>>,
    /// Eight-influence IVOBONEMAP32 maps. Kept apart from `bone_maps` so the
    /// rigid-split mesh path (which only reads `bone_maps`) is unchanged;
    /// only the skinned glTF export reads them.
    pub bone_maps8: Option<Vec<BoneMap24>>,
    pub colors: Option<Vec<[u8; 4]>>,
    pub tangents: Option<TangentData>,
    pub normals: Option<NormalData>,
}

#[derive(Debug, Clone, Copy)]
pub struct BoneMap24 {
    pub joint_indices: [u16; 8],
    pub weights: [u8; 8],
}

impl BoneMap24 {
    /// The four strongest influences (a joint named in several slots summed
    /// first), and the weight (0..=255) of the dropped ones.
    pub fn strongest_four(self) -> (BoneMap12, u16) {
        let mut summed: Vec<(u16, u16)> = Vec::with_capacity(8);
        for (&joint, &weight) in self.joint_indices.iter().zip(&self.weights) {
            if weight == 0 {
                continue;
            }
            match summed.iter_mut().find(|(existing, _)| *existing == joint) {
                Some((_, total)) => *total += u16::from(weight),
                None => summed.push((joint, u16::from(weight))),
            }
        }
        summed.sort_by(|a, b| b.1.cmp(&a.1));
        let mut map = BoneMap12 { joint_indices: [0; 4], weights: [0; 4] };
        for (slot, (joint, weight)) in summed.iter().take(4).enumerate() {
            map.joint_indices[slot] = *joint;
            map.weights[slot] = (*weight).min(255) as u8;
        }
        let dropped = summed.iter().skip(4).map(|(_, weight)| weight).sum();
        (map, dropped)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoneMap12 {
    pub joint_indices: [u16; 4],
    pub weights: [u8; 4],
}

impl BoneMap12 {
    pub fn dominant_joint(self) -> Option<u16> {
        let (slot, weight) = self
            .weights
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, weight)| *weight)?;
        (weight > 0).then_some(self.joint_indices[slot])
    }
}

/// Raw normal data — format determined by which stream was present.
#[derive(Debug)]
pub enum NormalData {
    /// IVONORMALS — single u32 packed unit vector (15-15-1-1 bit layout).
    /// Same encoding as IVOQTANGENTS individual vectors.
    Packed(Vec<u32>),
    /// IVONORMALS2 — 3× f32
    Float(Vec<[f32; 3]>),
}

/// Raw tangent data — format determined by which stream was present.
#[derive(Debug)]
pub enum TangentData {
    /// IVOQTANGENTS — 4× i16 SNorm quaternion (XYZW) OR compressed tangent vectors (2× u32).
    /// Actual format depends on whether IVONORMALS2 is also present:
    ///   - With IVONORMALS2: compressed tangent vectors (decode_compressed_tangent)
    ///   - Without IVONORMALS2: QTangent quaternion (decode_qtangent_snorm)
    QTangents(Vec<[u16; 4]>),
    /// IVOTANGENTS — SNorm i16 quaternion (same decode as QTangent)
    Tangents(Vec<[u16; 4]>),
}

#[derive(Debug)]
pub enum PositionData {
    Quantized(Vec<[u16; 3]>),
    Float(Vec<[f32; 3]>),
}

#[derive(Debug)]
pub struct SkinMesh {
    pub flags: u32,
    pub info: MeshInfo,
    pub submeshes: Vec<SubMeshDescriptor>,
    /// Name hashes ([`node_name_hash`]) of the mesh's nodes, ascending; a
    /// submesh's `node_parent_index` indexes this list. Character skins split
    /// into clothing zone nodes (`l_hand_zone`, `torso01_zone`, ...) plus a
    /// node named after the file stem for everything outside a zone.
    pub extra_words: Vec<u32>,
    pub streams: DataStreams,
}

/// Hash of a skin mesh node name as stored in [`SkinMesh::extra_words`]:
/// 32-bit FNV-1a over the upper-cased name including its NUL terminator.
pub fn node_name_hash(name: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in name.bytes().map(|b| b.to_ascii_uppercase()).chain(std::iter::once(0)) {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
    }
    hash
}

/// Every `MeshChunk` name of DataCore's `SCItemClothingParams.Chunks`
/// (4.x): the clothing zones a worn mesh may be split into. Skin node names
/// are only stored hashed, so these resolve them.
pub const CLOTHING_ZONES: &[&str] = &[
    "hips_zone", "underwear_zone", "underwear_top_zone", "vneck_zone",
    "torso01_zone", "torso02_zone", "torso03_zone", "torso04_zone",
    "l_shoulder_zone", "l_arm01_zone", "l_arm02_zone", "l_arm03_zone", "l_arm04_zone", "l_arm05_zone",
    "l_arm05_torso0_zone", "l_hand_zone", "l_leg01_zone", "l_leg02_zone", "l_leg03_zone", "l_leg04_zone",
    "l_foot_zone",
    "r_shoulder_zone", "r_arm01_zone", "r_arm02_zone", "r_arm03_zone", "r_arm04_zone", "r_arm05_zone",
    "r_arm05_torso0_zone", "r_hand_zone", "r_leg01_zone", "r_leg02_zone", "r_leg03_zone", "r_leg04_zone",
    "r_foot_zone",
    "omega_arms_zone", "omega_core_zone", "omega_legs_zone",
    "theta_arms_zone", "theta_core_zone", "theta_head_zone", "theta_legs_zone",
    "theta_pouch01_zone", "theta_pouch02_zone", "theta_pouch03_zone", "theta_pouch04_zone",
    "theta_pouch05_zone", "theta_pouch06_zone",
    "pcg_eyebrows", "pcg_l_ear", "pcg_mouth", "pcg_nose", "pcg_r_ear",
];

/// Whether `name` is a clothing zone of [`CLOTHING_ZONES`].
pub fn is_clothing_zone(name: &str) -> bool {
    CLOTHING_ZONES.iter().any(|z| z.eq_ignore_ascii_case(name))
}

impl SkinMesh {
    /// Name of every node in [`Self::extra_words`] order: a clothing zone,
    /// `own_name` (the file stem, the node outside every zone) or
    /// `node_<hash>` when the hash matches neither.
    pub fn node_names(&self, own_name: Option<&str>) -> Vec<String> {
        let own = own_name.map(|n| (node_name_hash(n), n));
        self.extra_words
            .iter()
            .map(|&hash| {
                if let Some(zone) = CLOTHING_ZONES.iter().find(|z| node_name_hash(z) == hash) {
                    return zone.to_string();
                }
                match own {
                    Some((h, n)) if h == hash => n.to_string(),
                    _ => format!("node_{hash:08x}"),
                }
            })
            .collect()
    }

    /// Node name ([`Self::node_names`]) of each submesh, in submesh order.
    /// A mesh without a node list is one node: `own_name`.
    pub fn submesh_node_names(&self, own_name: Option<&str>) -> Vec<String> {
        let names = self.node_names(own_name);
        self.submeshes
            .iter()
            .map(|s| match names.get(s.node_parent_index as usize) {
                Some(name) => name.clone(),
                None if names.is_empty() => own_name.unwrap_or("node_0").to_string(),
                None => format!("node_index_{}", s.node_parent_index),
            })
            .collect()
    }
}

impl MeshInfo {
    /// Parse MeshInfo from a SpanReader (reads 76 bytes).
    pub fn read(reader: &mut SpanReader) -> Result<Self, ParseError> {
        let raw = reader.read_type::<RawMeshInfo>()?;
        Ok(Self {
            flags2: raw.flags2,
            num_vertices: raw.num_vertices,
            num_indices: raw.num_indices,
            num_submeshes: raw.num_submeshes,
            model_min: raw.model_min,
            model_max: raw.model_max,
            min_bound: raw.min_bound,
            max_bound: raw.max_bound,
            extra_count: raw.extra_count,
        })
    }
}

impl SubMeshDescriptor {
    fn from_raw(raw: &RawSubMeshDescriptor) -> Self {
        Self {
            mat_id: raw.mat_id,
            node_parent_index: raw.node_parent_index,
            first_index: raw.first_index,
            num_indices: raw.num_indices,
            first_vertex: raw.first_vertex,
            page_base: raw.page_base,
            num_vertices: raw.num_vertices,
            radius: raw.radius,
            center: raw.center,
            unknown0: raw._unknown0,
            unknown1: raw._unknown1,
        }
    }
}

use crate::error::Error;

impl SkinMesh {
    pub fn read(data: &[u8]) -> Result<Self, Error> {
        let mut reader = SpanReader::new(data);

        // Header: 4 bytes flags + 76 bytes MeshInfo + 88 bytes padding = 168
        let flags = reader.read_u32()?;
        let info = MeshInfo::read(&mut reader)?;
        reader.advance(88)?;

        // Submesh descriptors
        let raw_submeshes =
            reader.read_slice::<RawSubMeshDescriptor>(info.num_submeshes as usize)?;
        let submeshes: Vec<SubMeshDescriptor> =
            raw_submeshes.iter().map(SubMeshDescriptor::from_raw).collect();

        let extra_words = if info.extra_count == 0 {
            Vec::new()
        } else {
            reader.read_slice::<u32>(info.extra_count as usize)?.to_vec()
        };

        if std::env::var("SB_DEBUG_RIGID_BIND").is_ok() {
            log::debug!(
                "SkinMesh flags={} verts={} indices={} submeshes={} extra_words={:?}",
                flags,
                info.num_vertices,
                info.num_indices,
                info.num_submeshes,
                extra_words,
            );
            for (index, submesh) in submeshes.iter().enumerate() {
                log::debug!(
                    "  submesh[{index}] node_parent={} first_index={} num_indices={} first_vertex={} num_vertices={} center={:?} radius={} unknown0=0x{:#010X} unknown1=0x{:#010X}",
                    submesh.node_parent_index,
                    submesh.first_index,
                    submesh.num_indices,
                    submesh.first_vertex,
                    submesh.num_vertices,
                    submesh.center,
                    submesh.radius,
                    submesh.unknown0,
                    submesh.unknown1,
                );
            }
        }

        // Datastreams
        let streams = Self::read_streams(&mut reader, &info)?;

        Ok(Self {
            flags,
            info,
            submeshes,
            extra_words,
            streams,
        })
    }

    fn read_streams(reader: &mut SpanReader, info: &MeshInfo) -> Result<DataStreams, Error> {
        let mut positions: Option<PositionData> = None;
        let mut uvs: Option<Vec<[u16; 2]>> = None;
        let mut secondary_uvs: Option<Vec<[u16; 2]>> = None;
        let mut indices: Option<Vec<u32>> = None;
        let mut bone_maps: Option<Vec<BoneMap12>> = None;
        let mut bone_maps8: Option<Vec<BoneMap24>> = None;
        let mut colors: Option<Vec<[u8; 4]>> = None;
        let mut tangents: Option<TangentData> = None;
        let mut normals: Option<NormalData> = None;

        let num_verts = info.num_vertices as usize;
        let num_idx = info.num_indices as usize;

        while reader.remaining() >= 4 {
            let stream_type = reader.read_u32()?;

            // Zero-tag: inter-stream alignment padding, skip it
            if stream_type == 0 {
                continue;
            }

            // All real streams have an element_size u32 after the tag
            if reader.remaining() < 4 {
                break;
            }
            let element_size = reader.read_u32()?;

            // Record position before reading stream data (for alignment)
            let stream_data_start = reader.position();

            if std::env::var("SB_DEBUG_STREAMS").is_ok() {
                let name = match stream_type {
                    stream_type::IVOVERTSUVS => "IVOVERTSUVS",
                    stream_type::IVOVERTSUVS2 => "IVOVERTSUVS2",
                    stream_type::IVOINDICES => "IVOINDICES",
                    stream_type::IVONORMALS => "IVONORMALS",
                    stream_type::IVONORMALS2 => "IVONORMALS2",
                    stream_type::IVOTANGENTS => "IVOTANGENTS",
                    stream_type::IVOQTANGENTS => "IVOQTANGENTS",
                    stream_type::IVOBONEMAP => "IVOBONEMAP",
                    stream_type::IVOBONEMAP32 => "IVOBONEMAP32",
                    stream_type::IVOCOLORS2 => "IVOCOLORS2",
                    _ => "UNKNOWN",
                };
                eprintln!("  stream 0x{stream_type:08X} ({name}): elem_size={element_size}, count={num_verts}");
            }

            match stream_type {
                stream_type::IVOVERTSUVS | stream_type::IVOVERTSUVS2 => {
                    // element_size determines layout:
                    // 16 = SNorm i16×3 positions + pad(2) + RGBA(4) + UV half×2(4)
                    // 20 = float f32×3 positions + RGBA(4) + UV half×2(4)
                    // The on-disk color bytes are stored as BGRA.
                    if element_size == 16 {
                        let verts = reader.read_slice::<RawQuantizedVertex>(num_verts)?;
                        positions = Some(PositionData::Quantized(
                            verts.iter().map(|v| v.pos).collect(),
                        ));
                        colors = Some(verts.iter().map(|v| bgra_to_rgba(v.color)).collect());
                        uvs = Some(verts.iter().map(|v| v.uv).collect());
                    } else if element_size == 20 {
                        let verts = reader.read_slice::<RawFloatVertex>(num_verts)?;
                        positions = Some(PositionData::Float(
                            verts.iter().map(|v| v.pos).collect(),
                        ));
                        colors = Some(verts.iter().map(|v| bgra_to_rgba(v.color)).collect());
                        uvs = Some(verts.iter().map(|v| v.uv).collect());
                    } else {
                        return Err(Error::UnexpectedElementSize {
                            expected: 16,
                            got: element_size,
                        });
                    }
                }
                stream_type::IVOINDICES => {
                    if element_size == 2 {
                        let raw = reader.read_slice::<u16>(num_idx)?;
                        indices = Some(raw.iter().map(|&i| i as u32).collect());
                    } else {
                        indices = Some(reader.read_slice::<u32>(num_idx)?.to_vec());
                    }
                }
                stream_type::IVONORMALS => {
                    normals = Some(NormalData::Packed(
                        reader.read_slice::<u32>(num_verts)?.to_vec(),
                    ));
                }
                stream_type::IVONORMALS2 => {
                    if element_size == 12 {
                        normals = Some(NormalData::Float(
                            reader.read_slice::<[f32; 3]>(num_verts)?.to_vec(),
                        ));
                    } else if element_size == 4 {
                        secondary_uvs = Some(reader.read_slice::<[u16; 2]>(num_verts)?.to_vec());
                    } else {
                        reader.advance(element_size as usize * num_verts)?;
                    }
                }
                stream_type::IVOQTANGENTS => {
                    let raw = reader.read_slice::<[u16; 4]>(num_verts)?;
                    if std::env::var("SB_DUMP_QTANGENTS").is_ok() {
                        for (i, v) in raw.iter().take(20).enumerate() {
                            eprintln!(
                                "QTANG[{i}]: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",
                                v[0], v[1], v[2], v[3]
                            );
                        }
                    }
                    tangents = Some(TangentData::QTangents(raw.to_vec()));
                }
                stream_type::IVOTANGENTS => {
                    let raw = reader.read_slice::<[u16; 4]>(num_verts)?;
                    if std::env::var("SB_DEBUG_STREAMS").is_ok() {
                        for (i, v) in raw.iter().take(5).enumerate() {
                            eprintln!(
                                "    IVOTANGENTS[{i}]: 0x{:04X} 0x{:04X} 0x{:04X} 0x{:04X}",
                                v[0], v[1], v[2], v[3]
                            );
                        }
                    }
                    tangents = Some(TangentData::Tangents(raw.to_vec()));
                }
                stream_type::IVOBONEMAP => {
                    if element_size == 12 {
                        let raw = reader.read_slice::<RawBoneMap12>(num_verts)?;
                        bone_maps = Some(
                            raw.iter()
                                .map(|entry| BoneMap12 {
                                    joint_indices: entry.joint_indices,
                                    weights: entry.weights,
                                })
                                .collect(),
                        );
                    } else {
                        reader.advance(element_size as usize * num_verts)?;
                    }
                }
                stream_type::IVOBONEMAP32 if element_size == 24 => {
                    let raw = reader.read_slice::<RawBoneMap24>(num_verts)?;
                    bone_maps8 = Some(
                        raw.iter()
                            .map(|entry| BoneMap24 {
                                joint_indices: entry.joint_indices,
                                weights: entry.weights,
                            })
                            .collect(),
                    );
                }
                _ => {
                    let count = num_verts;
                    let skip_bytes = element_size as usize * count;
                    if reader.remaining() >= skip_bytes {
                        reader.advance(skip_bytes)?;
                    } else {
                        break;
                    }
                }
            }

            // Align to 8-byte boundary after each stream (matching cryengine-converter)
            let bytes_read = reader.position() - stream_data_start;
            let remainder = bytes_read % 8;
            if remainder != 0 {
                let pad = 8 - remainder;
                if reader.remaining() >= pad {
                    reader.advance(pad)?;
                }
            }
        }

        Ok(DataStreams {
            positions: positions.ok_or(Error::MissingChunk {
                chunk_type: stream_type::IVOVERTSUVS,
            })?,
            uvs: uvs.unwrap_or_default(),
            secondary_uvs,
            indices: indices.ok_or(Error::MissingChunk {
                chunk_type: stream_type::IVOINDICES,
            })?,
            bone_maps,
            bone_maps8,
            colors,
            tangents,
            normals,
        })
    }
}

fn bgra_to_rgba(color: [u8; 4]) -> [u8; 4] {
    [color[2], color[1], color[0], color[3]]
}

#[cfg(test)]
mod tests {
    use super::{bgra_to_rgba, node_name_hash};

    #[test]
    fn bgra_swizzles_to_rgba() {
        assert_eq!(bgra_to_rgba([0, 82, 255, 127]), [255, 82, 0, 127]);
    }

    #[test]
    fn node_name_hashes_match_the_skin_node_list() {
        // m_cds_undersuit_armor_01.skinm: extra_words [998859158, 1564911345, 3335329176].
        assert_eq!(node_name_hash("r_hand_zone"), 998_859_158);
        assert_eq!(node_name_hash("m_cds_undersuit_armor_01"), 1_564_911_345);
        assert_eq!(node_name_hash("L_HAND_ZONE"), 3_335_329_176);
    }
}
