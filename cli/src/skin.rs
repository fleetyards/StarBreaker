use std::path::PathBuf;
use std::collections::HashMap;
use parking_lot::Mutex;

use clap::Subcommand;
use rayon::prelude::*;

use crate::common::load_p4k;
use crate::error::{CliError, Result};

#[derive(Subcommand)]
pub enum SkinCommand {
    /// Export a .skin/.cgf file to GLB
    Export {
        /// P4k path substring (case-insensitive)
        path: String,
        /// Output .glb path
        output: Option<PathBuf>,
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
        /// Write a glTF skinned mesh: joints from the matching `.skin`'s
        /// CompiledBones (game bone names, bind pose), JOINTS_0/WEIGHTS_0 on
        /// every primitive. Pieces of one character share joints by name.
        #[arg(long)]
        rig: bool,
        /// Rig onto this skeleton (a local .chr/.skin or a P4k path substring,
        /// e.g. `male_v7/export/bhm_skeleton_v7.chr`) instead of the piece's
        /// own `.skin` bones; implies --rig. The joints are exactly the
        /// skeleton's bones, so `skin animations` clips on the same skeleton
        /// drive every joint. Piece bones the skeleton lacks (the head's
        /// FACIAL_* bones) hand their weight to their nearest ancestor that
        /// it has (Head, Neck1, ...): exact without facial animation.
        #[arg(long)]
        skeleton: Option<String>,
        /// With --skeleton: another `.skinm` (P4k path substring or local
        /// file) skinned into the same GLB on the same skin, one mesh node
        /// per piece. Repeatable, e.g. a head, its eyes and teeth on a body.
        #[arg(long = "add", requires = "skeleton")]
        add: Vec<String>,
    },
    /// Write a rigged skeleton GLB with one glTF animation per CAF/DBA clip.
    ///
    /// Joint nodes are written exactly like `skin export --rig` (game bone
    /// names, bind-pose local TRS under `CryEngine_Z_up`), so the clips play
    /// by bone name on the body and every rigged armour piece. Use the
    /// character's `.chr` skeleton (e.g. `male_v7/export/bhm_skeleton_v7.chr`)
    /// so the bones of every piece (Head, face, attachment parents) animate.
    ///
    /// Example: `skin animations --skeleton male_v7/export/bhm_skeleton_v7.chr
    /// --clip "male_v7/weapons/no_weapon/locomotion/stand.dba#nw_neutral_stand_runfast_forward=run_fast"
    /// animations.glb`
    Animations {
        /// Output .glb path
        output: PathBuf,
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
        /// Skeleton: a local .chr/.skin file, or a P4k path substring of one
        #[arg(long)]
        skeleton: String,
        /// Clip spec `<dba|caf>#<clip filter>[=<name>]`, repeatable. The file
        /// is a local path or a P4k path substring; the filter is a
        /// case-insensitive substring of the clip name (omit `#...` for every
        /// clip in the file). With `=<name>` the filter must pick one clip
        /// (an exact stem match wins); without it every match is written,
        /// named after its .caf stem. Additive clips (`*_add`) are skipped.
        #[arg(long = "clip", required = true)]
        clips: Vec<String>,
        /// Also skin the skeleton's mesh (the `.skinm` beside a `.skin`
        /// skeleton) into the GLB. Default: skeleton and animations only.
        #[arg(long)]
        mesh: bool,
        /// Keep the horizontal translation of the root bone (`World`). By
        /// default it is removed so locomotion cycles run in place; the
        /// vertical component, the root rotation and the Hips motion stay.
        #[arg(long)]
        keep_root_motion: bool,
    },
    /// Write many rigged GLBs in one process from a jobs JSON file
    /// (`{"jobs": [{"id", "output", "geometry", "attach"?}]}`): a worn
    /// `.skin` is rigged on its own bones like `export --rig`; a rigid model
    /// (`.cga`/`.cgf`/`.cdf`, e.g. a backpack) is bound with weight 1 to
    /// `attach.bone` of the `attach.skeleton` `.skin` (a core), its
    /// `attach.item_offset_helper` node placed on the bone. One JSON report
    /// line per job on stdout. Paths are DataCore paths (`Objects/...`).
    RigBatch {
        /// Jobs JSON file
        jobs: PathBuf,
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
        /// Worker threads (default: all cores)
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Inspect a .skinm/.skin/.cgfm/.chr file and print parsed metadata
    Inspect {
        /// P4k path substring (case-insensitive)
        path: String,
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
        /// For .skinm files: dump per-vertex bone weight statistics
        /// (max influence count, weight distribution, vertices weighted to
        /// multiple bones).
        #[arg(long)]
        bone_weights: bool,
    },
    /// Scan all mesh files and report stream/chunk type statistics
    ScanStreams {
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
    },
    /// Find all files containing a specific stream type ID
    FindStream {
        /// Stream type ID in hex (e.g. 9D51C5EE)
        stream_id: String,
        /// Path to Data.p4k
        #[arg(long, env = "SC_DATA_P4K")]
        p4k: Option<PathBuf>,
    },
}

impl SkinCommand {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Export { path, output, p4k, skeleton: Some(skeleton), add, .. } => {
                export_on_skeleton(path, add, output, p4k, skeleton)
            }
            Self::Export { path, output, p4k, rig, skeleton: None, .. } => export(path, output, p4k, rig),
            Self::Animations { output, p4k, skeleton, clips, mesh, keep_root_motion } => {
                animations(output, p4k, skeleton, clips, mesh, keep_root_motion)
            }
            Self::RigBatch { jobs, p4k, threads } => crate::rig_batch::run(jobs, p4k, threads),
            Self::Inspect { path, p4k, bone_weights } => inspect(path, p4k, bone_weights),
            Self::ScanStreams { p4k } => scan_streams(p4k),
            Self::FindStream { stream_id, p4k } => find_stream(stream_id, p4k),
        }
    }
}

fn stream_name(id: u32) -> &'static str {
    match id {
        0xEECDC168 => "IVOINDICES",
        0x91329AE9 => "IVOVERTSUVS",
        0xB3A70D5E => "IVOVERTSUVS2",
        0x9CF3F615 => "IVONORMALS",
        0x38A581FE => "IVONORMALS2",
        0xEE057252 => "IVOQTANGENTS",
        0xB95E9A1B => "IVOTANGENTS",
        0x677C7B23 => "IVOBONEMAP",
        0x6ECA3708 => "IVOBONEMAP32",
        0xD9EED421 => "IVOCOLORS2",
        0x9D51C5EE => "IVO_9D51C5EE",
        0xA596C0E6 => "IVO_A596C0E6",
        0x7E216CAB => "IVO_7E216CAB",
        0x69655CD8 => "IVO_69655CD8",
        0xDA86BE9C => "IVO_DA86BE9C",
        _ => "?",
    }
}

fn chunk_name(id: u32) -> &'static str {
    match id {
        0xBE5E493E => "RC_VERSION",
        0xC2011111 => "SKELETON",
        0x90C62222 => "PHYSICS",
        0x83353333 => "MATERIALINFO",
        0xA7169999 => "RBF",
        0x92914444 => "MESHINFO",
        0x0A2485B6 => "RIGLOGIC",
        0x7E035555 => "RIGINFO",
        0x2B8A2C29 => "SIMTOPOLOGY",
        0xF7086666 => "PROTOSINFO",
        0x9351756F => "LODDISTANCE",
        0x58DE1772 => "STATOBJ_PHYSICS",
        0x2B7ECF9F => "POSITIONBONEMAP",
        0x1E14A062 => "VFXHELPERS",
        0xC1C36AFE => "VISIBILITYGROUPS",
        0xC201973C => "SKELETON_v901",
        0x70697FDA => "SUBOBJECTS",
        0xB32459D2 => "MESH_CONTAINER",
        0xB8757777 => "IVO_SKIN2",
        0xE0181074 => "RAYTRACE_BVH",
        0x57A38888 => "BSHAPESGPU",
        0xFF277A9A => "SIMDEFORMATION",
        _ => "?",
    }
}

/// Scan stream headers from IVO_SKIN2 chunk data (already extracted from chunk wrapper).
fn scan_stream_types(chunk_data: &[u8]) -> Vec<(u32, u32)> {
    if chunk_data.len() < 168 { return vec![]; }

    // Header: 4B flags + 76B MeshInfo + 88B padding = 168
    let num_verts = u32::from_le_bytes(chunk_data[8..12].try_into().unwrap_or([0;4])) as usize;
    let num_idx = u32::from_le_bytes(chunk_data[16..20].try_into().unwrap_or([0;4])) as usize;
    let num_submeshes = u32::from_le_bytes(chunk_data[12..16].try_into().unwrap_or([0;4])) as usize;
    let extra_count = u32::from_le_bytes(chunk_data[76..80].try_into().unwrap_or([0;4])) as usize;

    // Sanity checks
    if num_submeshes > 1000 || num_verts > 10_000_000 || extra_count > 10000 { return vec![]; }

    let _sub_size = num_submeshes * 48; // SubMeshDescriptor is 48 bytes (36 data + 12 padding? let's try both)
    let start = 168 + num_submeshes * 36 + extra_count * 4;
    if start >= chunk_data.len() { return vec![]; }

    let mut streams = vec![];
    let mut pos = start;
    while pos + 8 <= chunk_data.len() {
        let tag = u32::from_le_bytes(chunk_data[pos..pos+4].try_into().unwrap_or([0;4]));
        if tag == 0 { pos += 4; continue; }
        let elem_size = u32::from_le_bytes(chunk_data[pos+4..pos+8].try_into().unwrap_or([0;4]));
        if elem_size == 0 || elem_size > 1024 { break; }

        streams.push((tag, elem_size));

        let count = if tag == 0xEECDC168 { num_idx } else { num_verts };
        let stream_bytes = elem_size as usize * count;
        pos += 8 + stream_bytes;
        // Align to 8 bytes
        let rem = stream_bytes % 8;
        if rem != 0 { pos += 8 - rem; }
    }
    streams
}

/// Extract chunk types and IVO stream types from a .cgfm/.skinm file using proper chunk parsing.
fn scan_file(data: &[u8]) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    use starbreaker_chunks::ChunkFile;

    let mut chunks = vec![];
    let mut streams = vec![];

    let Ok(chunk_file) = ChunkFile::from_bytes(data) else { return (chunks, streams) };
    let ChunkFile::Ivo(ivo) = &chunk_file else { return (chunks, streams) };

    for entry in ivo.chunks() {
        chunks.push((entry.chunk_type, entry.version));

        // If this is IVO_SKIN2, scan its stream types
        if entry.chunk_type == 0xB8757777 {
            let chunk_data = ivo.chunk_data(entry);
            streams.extend(scan_stream_types(chunk_data));
        }
    }
    (chunks, streams)
}

fn scan_streams(p4k_path: Option<PathBuf>) -> Result<()> {
    let p4k = load_p4k(p4k_path.as_deref())?;

    let mesh_entries: Vec<_> = p4k.entries().iter()
        .filter(|e| {
            let name = e.name.to_lowercase();
            name.ends_with(".skinm") || name.ends_with(".cgfm")
        })
        .collect();

    eprintln!("Scanning {} mesh files...", mesh_entries.len());

    // Collect: stream_id -> (count, elem_sizes_seen, example_files)
    type Stats = HashMap<u32, (usize, Vec<u32>, Vec<String>)>;
    let stream_stats: Mutex<Stats> = Mutex::new(HashMap::new());
    let chunk_stats: Mutex<Stats> = Mutex::new(HashMap::new());
    let errors = Mutex::new(0usize);
    let scanned = Mutex::new(0usize);

    mesh_entries.par_iter().for_each(|entry| {
        let Ok(data) = p4k.read(entry) else {
            *errors.lock() += 1;
            return;
        };
        let (chunks, streams) = scan_file(&data);

        {
            let mut s = chunk_stats.lock();
            for (tag, version) in chunks {
                let e = s.entry(tag).or_insert_with(|| (0, vec![], vec![]));
                e.0 += 1;
                if !e.1.contains(&version) { e.1.push(version); }
                if e.2.len() < 3 {
                    e.2.push(entry.name.rsplit(['/', '\\']).next().unwrap_or("").to_string());
                }
            }
        }
        {
            let mut s = stream_stats.lock();
            for (tag, elem_size) in streams {
                let e = s.entry(tag).or_insert_with(|| (0, vec![], vec![]));
                e.0 += 1;
                if !e.1.contains(&elem_size) { e.1.push(elem_size); }
                if e.2.len() < 3 {
                    e.2.push(entry.name.rsplit(['/', '\\']).next().unwrap_or("").to_string());
                }
            }
        }

        let mut n = scanned.lock();
        *n += 1;
        if *n % 5000 == 0 { eprint!("\r  {}/{} files...", *n, mesh_entries.len()); }
    });

    let stream_stats = stream_stats.into_inner();
    let chunk_stats = chunk_stats.into_inner();
    let errors = errors.into_inner();
    eprintln!("\r  Done. {} files scanned, {} read errors.", scanned.into_inner(), errors);

    // Chunk census
    let mut sorted: Vec<_> = chunk_stats.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    println!("\n=== CHUNK TYPE CENSUS ===\n");
    println!("{:<14} {:<22} {:>7}  {:<12}  Example", "Chunk ID", "Name", "Count", "Versions");
    println!("{}", "-".repeat(90));
    for (tag, (count, versions, examples)) in &sorted {
        let name = chunk_name(*tag);
        let ver: Vec<String> = versions.iter().map(|v| format!("0x{v:X}")).collect();
        let ex = examples.first().map(|s| s.as_str()).unwrap_or("");
        println!("0x{:08X}  {:<22} {:>7}  {:<12}  {}", tag, name, count, ver.join(","), ex);
    }

    // Stream census
    let mut sorted: Vec<_> = stream_stats.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    println!("\n=== STREAM TYPE CENSUS ===\n");
    println!("{:<14} {:<20} {:>7}  {:<15}  Example", "Stream ID", "Name", "Count", "Elem Sizes");
    println!("{}", "-".repeat(90));
    for (tag, (count, sizes, examples)) in &sorted {
        let name = stream_name(*tag);
        let sizes_str: Vec<String> = sizes.iter().map(|s| s.to_string()).collect();
        let ex = examples.first().map(|s| s.as_str()).unwrap_or("");
        println!("0x{:08X}  {:<20} {:>7}  {:<15}  {}", tag, name, count, sizes_str.join(","), ex);
    }

    Ok(())
}

fn find_stream(stream_id_hex: String, p4k_path: Option<PathBuf>) -> Result<()> {
    let target = u32::from_str_radix(&stream_id_hex, 16)?;
    let p4k = load_p4k(p4k_path.as_deref())?;

    let mesh_entries: Vec<_> = p4k.entries().iter()
        .filter(|e| {
            let name = e.name.to_lowercase();
            name.ends_with(".skinm") || name.ends_with(".cgfm")
        })
        .collect();

    eprintln!("Searching {} files for stream 0x{:08X}...", mesh_entries.len(), target);

    let results: Mutex<Vec<(String, u32)>> = Mutex::new(vec![]);

    mesh_entries.par_iter().for_each(|entry| {
        let Ok(data) = p4k.read(entry) else { return };
        let (_, streams) = scan_file(&data);
        for (tag, elem_size) in streams {
            if tag == target {
                results.lock().push((entry.name.clone(), elem_size));
                break;
            }
        }
    });

    let mut results = results.into_inner();
    results.sort();
    println!("\nFound {} files with stream 0x{:08X} ({}):\n",
        results.len(), target, stream_name(target));
    for (name, elem_size) in &results {
        println!("  elem_size={:<4}  {}", elem_size, name);
    }
    Ok(())
}

fn export(search: String, output: Option<PathBuf>, p4k_path: Option<PathBuf>, rig: bool) -> Result<()> {
    let p4k = load_p4k(p4k_path.as_deref())?;
    let search_lower = search.to_lowercase();

    let entry = p4k
        .entries()
        .iter()
        .find(|e| {
            let name = e.name.to_lowercase();
            name.contains(&search_lower) && (name.ends_with(".skinm") || name.ends_with(".cgfm"))
        })
        .ok_or_else(|| CliError::NotFound(format!("no .skinm/.cgfm file matching '{search}' in P4k")))?;

    eprintln!("Found: {}", entry.name);
    let data = p4k.read(entry)?;
    let glb = if rig {
        // The CompiledBones live in the `.skin` beside the `.skinm` geometry.
        let skeleton_name = entry
            .name
            .strip_suffix('m')
            .filter(|_| entry.name.to_lowercase().ends_with(".skinm"))
            .ok_or_else(|| CliError::NotFound(format!("--rig needs a .skinm, found {}", entry.name)))?;
        let skeleton_entry = p4k
            .entries()
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(skeleton_name))
            .ok_or_else(|| CliError::NotFound(format!("no skeleton file {skeleton_name} in P4k")))?;
        let skeleton = p4k.read(skeleton_entry)?;
        let stem = skeleton_name.rsplit(['/', '\\']).next().unwrap_or("skin");
        let stem = stem.rsplit_once('.').map_or(stem, |(base, _)| base);
        let rigged = starbreaker_3d::skin_to_rigged_glb(&data, &skeleton, Some(stem))?;
        eprintln!(
            "Rigged: {} vertices, {} joints, {} unweighted vertices, {} out-of-range influences, {} vertices cut to 4 influences (max weight lost {:.3})",
            rigged.vertex_count,
            rigged.joint_count,
            rigged.unweighted_vertices,
            rigged.out_of_range_influences,
            rigged.truncated_vertices,
            rigged.max_truncated_weight,
        );
        rigged.glb
    } else {
        starbreaker_3d::skin_to_glb(&data, None)?
    };

    let output = output.unwrap_or_else(|| {
        let stem = entry.name.rsplit(['/', '\\']).next().unwrap_or("output");
        PathBuf::from(format!("{stem}.glb"))
    });

    std::fs::write(&output, &glb)
        .map_err(|e| CliError::IoPath { source: e, path: output.display().to_string() })?;
    eprintln!("Written {} bytes to {}", glb.len(), output.display());
    Ok(())
}

/// `skin export --skeleton`: `path` and every `--add` piece skinned onto one
/// skeleton in one GLB.
fn export_on_skeleton(
    path: String,
    add: Vec<String>,
    output: Option<PathBuf>,
    p4k_path: Option<PathBuf>,
    skeleton_spec: String,
) -> Result<()> {
    let mut p4k = None;
    let skeleton = read_source(&skeleton_spec, &[".chr", ".skin"], &mut p4k, p4k_path.as_deref())?;
    let bones = starbreaker_3d::skeleton::parse_skeleton(&skeleton.data)
        .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {}", skeleton.path)))?;
    eprintln!("Skeleton: {} ({} bones)", skeleton.path, bones.len());

    struct Loaded {
        name: String,
        geometry: Vec<u8>,
        bones: Vec<starbreaker_3d::skeleton::Bone>,
    }
    let mut loaded = Vec::new();
    for spec in std::iter::once(&path).chain(&add) {
        let spec = if spec.to_lowercase().ends_with(".skin") { format!("{spec}m") } else { spec.clone() };
        let geometry = read_source(&spec, &[".skinm"], &mut p4k, p4k_path.as_deref())?;
        let skin_path = geometry.path.strip_suffix(['m', 'M']).unwrap_or(&geometry.path).to_string();
        let skin = read_source(&skin_path, &[".skin"], &mut p4k, p4k_path.as_deref())?;
        let piece_bones = starbreaker_3d::skeleton::parse_skeleton(&skin.data)
            .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {}", skin.path)))?;
        eprintln!("Piece: {}", geometry.path);
        loaded.push(Loaded { name: clip_stem(&skin.path).to_string(), geometry: geometry.data, bones: piece_bones });
    }
    let pieces: Vec<starbreaker_3d::SkinPiece> = loaded
        .iter()
        .map(|l| starbreaker_3d::SkinPiece { geometry: &l.geometry, bones: &l.bones, name: &l.name })
        .collect();
    let stem = clip_stem(&skeleton.path);
    let out = starbreaker_3d::pieces_on_skeleton_glb(&bones, &pieces, Some(stem))?;
    for piece in &out.pieces {
        let mut folded_into: Vec<(String, usize)> = Vec::new();
        for (_, target) in &piece.folded_bones {
            match folded_into.iter_mut().find(|(t, _)| t == target) {
                Some((_, n)) => *n += 1,
                None => folded_into.push((target.clone(), 1)),
            }
        }
        let folded: Vec<String> = folded_into.iter().map(|(t, n)| format!("{n} -> {t}")).collect();
        eprintln!(
            "  {:<32} {:>6} vertices, {}/{} bones matched{}, {} joints used ({}), bind offset {:.4} mm / {:.4} deg, {} vertices cut to 4 influences (max weight lost {:.3})",
            piece.name,
            piece.vertex_count,
            piece.matched_bones,
            piece.piece_bones,
            if folded.is_empty() { String::new() } else { format!(", folded {}", folded.join(", ")) },
            piece.used_joints.len(),
            piece.used_joints.join(" "),
            piece.max_bind_offset * 1000.0,
            piece.max_bind_angle_deg,
            piece.truncated_vertices,
            piece.max_truncated_weight,
        );
        let mut zones: Vec<&str> = Vec::new();
        for node in &piece.submesh_nodes {
            if starbreaker_3d::ivo::skin::is_clothing_zone(node) && !zones.contains(&node.as_str()) {
                zones.push(node);
            }
        }
        if !zones.is_empty() {
            eprintln!("    zones: {}", zones.join(" "));
        }
    }
    eprintln!(
        "Rigged: {} vertices, {} joints, {} unweighted vertices, {} out-of-range influences",
        out.glb.vertex_count, out.glb.joint_count, out.glb.unweighted_vertices, out.glb.out_of_range_influences,
    );
    let output = output.unwrap_or_else(|| PathBuf::from(format!("{}.glb", loaded[0].name)));
    std::fs::write(&output, &out.glb.glb)
        .map_err(|e| CliError::IoPath { source: e, path: output.display().to_string() })?;
    eprintln!("Written {} bytes to {}", out.glb.glb.len(), output.display());
    Ok(())
}

/// A file read from disk or from the P4k, with its display path.
struct SourceFile {
    path: String,
    data: Vec<u8>,
}

/// Read `spec` from disk when it names an existing file, else the P4k entry
/// whose path ends with (preferred) or contains `spec`, case-insensitively,
/// with one of `extensions`.
fn read_source(
    spec: &str,
    extensions: &[&str],
    p4k: &mut Option<starbreaker_p4k::MappedP4k>,
    p4k_path: Option<&std::path::Path>,
) -> Result<SourceFile> {
    let local = std::path::Path::new(spec);
    if local.is_file() {
        let data = std::fs::read(local).map_err(|e| CliError::IoPath { source: e, path: spec.to_string() })?;
        return Ok(SourceFile { path: local.display().to_string(), data });
    }
    if p4k.is_none() {
        *p4k = Some(load_p4k(p4k_path)?);
    }
    let p4k = p4k.as_ref().expect("p4k loaded");
    let needle = spec.replace('/', "\\").to_lowercase();
    let candidates: Vec<_> = p4k
        .entries()
        .iter()
        .filter(|e| {
            let name = e.name.to_lowercase();
            extensions.iter().any(|ext| name.ends_with(ext)) && name.contains(&needle)
        })
        .collect();
    let entry = candidates
        .iter()
        .find(|e| e.name.to_lowercase().ends_with(&needle))
        .or_else(|| candidates.first())
        .ok_or_else(|| CliError::NotFound(format!("no {} file matching '{spec}' on disk or in the P4k", extensions.join("/"))))?;
    if candidates.len() > 1 && !entry.name.to_lowercase().ends_with(&needle) {
        eprintln!("warning: '{spec}' matches {} files, using {}", candidates.len(), entry.name);
    }
    Ok(SourceFile { path: entry.name.clone(), data: p4k.read(entry)? })
}

/// File-name stem of a clip name (`.../nw_neutral_stand_runfast_forward.caf`
/// is `nw_neutral_stand_runfast_forward`).
fn clip_stem(name: &str) -> &str {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    file.rsplit_once('.').map_or(file, |(stem, _)| stem)
}

/// Parsed `--clip` spec.
struct ClipSpec<'a> {
    file: &'a str,
    filter: Option<&'a str>,
    name: Option<&'a str>,
}

fn parse_clip_spec(spec: &str) -> Result<ClipSpec<'_>> {
    let (file, rest) = match spec.split_once('#') {
        Some((file, rest)) => (file, Some(rest)),
        None => (spec, None),
    };
    let (filter, name) = match rest {
        Some(rest) => match rest.split_once('=') {
            Some((filter, name)) => (Some(filter), Some(name)),
            None => (Some(rest), None),
        },
        None => (None, None),
    };
    let filter = filter.filter(|f| !f.is_empty());
    let name = name.filter(|n| !n.is_empty());
    if file.is_empty() || (name.is_some() && filter.is_none()) {
        return Err(CliError::InvalidInput(format!("bad --clip '{spec}': expected <dba|caf>#<clip filter>[=<name>]")));
    }
    Ok(ClipSpec { file, filter, name })
}

/// Clips of `db` selected by `spec`, paired with their output names.
fn select_clips<'a>(
    db: &'a starbreaker_3d::animation::AnimationDatabase,
    spec: &ClipSpec,
    raw: &str,
) -> Result<Vec<(&'a starbreaker_3d::animation::AnimationClip, String)>> {
    let filter = spec.filter.map(str::to_lowercase);
    let matches: Vec<_> = db
        .clips
        .iter()
        .filter(|c| filter.as_ref().is_none_or(|f| c.name.to_lowercase().contains(f)))
        .collect();
    if matches.is_empty() {
        return Err(CliError::NotFound(format!("--clip '{raw}' matches no clip")));
    }
    let Some(name) = spec.name else {
        return Ok(matches.into_iter().map(|c| (c, clip_stem(&c.name).to_string())).collect());
    };
    let filter = filter.expect("a named clip spec has a filter");
    let exact: Vec<_> = matches
        .iter()
        .filter(|c| {
            let stem = clip_stem(&c.name).to_lowercase();
            stem == filter || format!("{stem}.caf") == filter || c.name.to_lowercase().ends_with(&filter)
        })
        .collect();
    let clip = match (matches.len(), exact.len()) {
        (1, _) => matches[0],
        (_, 1) => *exact[0],
        _ => {
            let names: Vec<&str> = matches.iter().take(8).map(|c| clip_stem(&c.name)).collect();
            return Err(CliError::InvalidInput(format!(
                "--clip '{raw}' matches {} clips ({}{}); make the filter unique",
                matches.len(),
                names.join(", "),
                if matches.len() > 8 { ", ..." } else { "" },
            )));
        }
    };
    Ok(vec![(clip, name.to_string())])
}

fn animations(
    output: PathBuf,
    p4k_path: Option<PathBuf>,
    skeleton_spec: String,
    clip_specs: Vec<String>,
    with_mesh: bool,
    keep_root_motion: bool,
) -> Result<()> {
    use starbreaker_3d::animation::{parse_caf, parse_dba, sample_clip, AnimationDatabase, SampleOptions, SampledClip};

    let mut p4k = None;
    let skeleton = read_source(&skeleton_spec, &[".chr", ".skin"], &mut p4k, p4k_path.as_deref())?;
    let bones = starbreaker_3d::skeleton::parse_skeleton(&skeleton.data)
        .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {}", skeleton.path)))?;
    eprintln!("Skeleton: {} ({} bones)", skeleton.path, bones.len());
    let geometry = if with_mesh {
        if !skeleton.path.to_lowercase().ends_with(".skin") {
            return Err(CliError::InvalidInput("--mesh needs a .skin skeleton (its .skinm holds the mesh)".into()));
        }
        let mesh_path = format!("{}m", skeleton.path);
        Some(read_source(&mesh_path, &[".skinm"], &mut p4k, p4k_path.as_deref())?.data)
    } else {
        None
    };

    let opts = SampleOptions { strip_root_translation: !keep_root_motion };
    let mut databases: HashMap<String, (String, AnimationDatabase)> = HashMap::new();
    let mut sampled: Vec<SampledClip> = Vec::new();
    for raw in &clip_specs {
        let spec = parse_clip_spec(raw)?;
        if !databases.contains_key(spec.file) {
            let source = read_source(spec.file, &[".dba", ".caf"], &mut p4k, p4k_path.as_deref())?;
            let db = if source.path.to_lowercase().ends_with(".caf") {
                let mut db = parse_caf(&source.data)?;
                for clip in &mut db.clips {
                    clip.name = source.path.clone();
                }
                db
            } else {
                parse_dba(&source.data)?
            };
            databases.insert(spec.file.to_string(), (source.path, db));
        }
        let (source_path, db) = &databases[spec.file];
        for (clip, name) in select_clips(db, &spec, raw)? {
            let stem = clip_stem(&clip.name);
            if stem.to_lowercase().ends_with("_add") {
                eprintln!("skip {stem}: additive clip (deltas on top of a base pose), not a standalone animation");
                continue;
            }
            if sampled.iter().any(|c| c.name == name) {
                return Err(CliError::InvalidInput(format!("two clips named '{name}'; name them with =<name>")));
            }
            let source = format!("{source_path}#{stem}");
            let clip = sample_clip(clip, &bones, &name, &source, opts)
                .ok_or_else(|| CliError::InvalidInput(format!("clip {stem} animates no bone of {}", skeleton.path)))?;
            eprintln!(
                "  {:<32} {:>4} frames @ {:>2} fps  {:<6} {} ({} channels, {} not in skeleton{})",
                clip.name,
                clip.frames,
                clip.fps,
                if clip.is_static { "static" } else if clip.looping { "loop" } else { "once" },
                stem,
                clip.matched_channels,
                clip.unmatched_channels,
                if clip.root_motion_removed > 0.001 {
                    format!(", {:.2} m root motion removed", clip.root_motion_removed)
                } else {
                    String::new()
                },
            );
            sampled.push(clip);
        }
    }
    if sampled.is_empty() {
        return Err(CliError::InvalidInput("no clips to write".into()));
    }

    let stem = clip_stem(&skeleton.path);
    let out = starbreaker_3d::animated_skeleton_glb(&bones, geometry.as_deref(), &sampled, Some(stem))?;
    std::fs::write(&output, &out.glb)
        .map_err(|e| CliError::IoPath { source: e, path: output.display().to_string() })?;
    eprintln!(
        "Written {} bytes to {} ({} joints, {} animations{})",
        out.glb.len(),
        output.display(),
        out.joint_count,
        sampled.len(),
        if geometry.is_some() { format!(", {} vertices", out.vertex_count) } else { String::new() },
    );
    Ok(())
}

fn inspect(search: String, p4k_path: Option<PathBuf>, bone_weights: bool) -> Result<()> {
    use starbreaker_chunks::ChunkFile;

    let p4k = load_p4k(p4k_path.as_deref())?;
    let search_lower = search.to_lowercase();

    let entry = p4k
        .entries()
        .iter()
        .find(|e| {
            let name = e.name.to_lowercase();
            // Prefer exact suffix match if search includes an extension
            if let Some(dot) = search_lower.rfind('.') {
                let search_ext = &search_lower[dot..];
                if !name.ends_with(search_ext) {
                    return false;
                }
            }
            name.contains(&search_lower)
                && (name.ends_with(".skinm")
                    || name.ends_with(".skin")
                    || name.ends_with(".cgfm")
                    || name.ends_with(".chr"))
        })
        .ok_or_else(|| CliError::NotFound(format!(
            "no .skinm/.skin/.cgfm/.chr file matching '{search}' in P4k"
        )))?;

    println!("Found: {}", entry.name);
    let data = p4k.read(entry)?;

    let lower_name = entry.name.to_lowercase();
    let is_skel_file = lower_name.ends_with(".chr") || lower_name.ends_with(".skin");
    if is_skel_file {
        if let Some(bones) = starbreaker_3d::skeleton::parse_skeleton(&data) {
            println!("Skeleton bones: {}", bones.len());
            for (index, bone) in bones.iter().enumerate() {
                println!(
                    "  [{index}] {} parent={:?} object_node_index={:?} local_pos={:?} local_rot={:?} world_pos={:?} world_rot={:?}",
                    bone.name,
                    bone.parent_index,
                    bone.object_node_index,
                    bone.local_position,
                    bone.local_rotation,
                    bone.world_position,
                    bone.world_rotation,
                );
            }
            return Ok(());
        }
        return Err(CliError::InvalidInput(format!(
            "failed to parse skeleton from {}",
            entry.name
        )));
    }

    let chunk_file = ChunkFile::from_bytes(&data)
        .map_err(|error| CliError::InvalidInput(format!("failed to parse chunk file {}: {error}", entry.name)))?;
    let ChunkFile::Ivo(ivo) = chunk_file else {
        return Err(CliError::InvalidInput(format!("unsupported non-IVO mesh file {}", entry.name)));
    };

    let skin_entry = ivo
        .chunks()
        .iter()
        .find(|chunk| chunk.chunk_type == starbreaker_chunks::known_types::ivo::IVO_SKIN2)
        .ok_or_else(|| CliError::InvalidInput(format!("no IVO_SKIN2 chunk in {}", entry.name)))?;

    let skin = starbreaker_3d::ivo::skin::SkinMesh::read(ivo.chunk_data(skin_entry))
        .map_err(|error| CliError::InvalidInput(format!("failed to parse skin mesh {}: {error}", entry.name)))?;

    println!(
        "SkinMesh flags={} verts={} indices={} submeshes={} extra_words={:?}",
        skin.flags,
        skin.info.num_vertices,
        skin.info.num_indices,
        skin.info.num_submeshes,
        skin.extra_words,
    );
    let own = entry.name.rsplit(['/', '\\']).next().and_then(|f| f.split('.').next());
    println!("Nodes: {}", skin.node_names(own).join(" "));
    for (index, submesh) in skin.submeshes.iter().enumerate() {
        println!(
            "  submesh[{index}] node_parent={} first_index={} num_indices={} first_vertex={} num_vertices={} center={:?} radius={} unknown0=0x{:08X} unknown1=0x{:08X}",
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

    if bone_weights {
        match &skin.streams.bone_maps {
            None => println!("\n[bone-weights] mesh has no bone_maps stream — purely rigid"),
            Some(maps) => {
                println!("\n[bone-weights] {} vertices total", maps.len());

                // Influence-count histogram: how many vertices have 1, 2, 3, 4 non-zero weights?
                let mut influence_hist = [0usize; 5];
                let mut joint_usage: HashMap<u16, usize> = HashMap::new();
                let mut multi_bone_examples: Vec<(usize, [u16; 4], [u8; 4])> = vec![];

                for (vid, bm) in maps.iter().enumerate() {
                    let n = bm.weights.iter().filter(|w| **w > 0).count();
                    influence_hist[n] += 1;
                    for slot in 0..4 {
                        if bm.weights[slot] > 0 {
                            *joint_usage.entry(bm.joint_indices[slot]).or_insert(0) += 1;
                        }
                    }
                    if n >= 2 && multi_bone_examples.len() < 16 {
                        multi_bone_examples.push((vid, bm.joint_indices, bm.weights));
                    }
                }

                println!("\nInfluence-count histogram (verts weighted to N non-zero bones):");
                for n in 0..=4 {
                    let pct = (influence_hist[n] as f64 / maps.len() as f64) * 100.0;
                    println!("  {n}-bone: {:>8} ({:>5.1}%)", influence_hist[n], pct);
                }

                println!("\nJoint usage (vertices weighted to each bone, sorted by count):");
                let mut sorted_joints: Vec<_> = joint_usage.into_iter().collect();
                sorted_joints.sort_by(|a, b| b.1.cmp(&a.1));
                for (joint, count) in sorted_joints.iter().take(20) {
                    println!("  joint[{:>3}]: {:>8} verts", joint, count);
                }

                if !multi_bone_examples.is_empty() {
                    println!("\nFirst {} vertices with multi-bone weights:", multi_bone_examples.len());
                    for (vid, joints, weights) in &multi_bone_examples {
                        println!(
                            "  vert[{:>5}] joints={:?} weights={:?}",
                            vid, joints, weights
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
