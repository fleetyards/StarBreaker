//! `skin rig-batch`: many rigged GLBs (worn armour, backpacks) in one process.
//!
//! Jobs file (JSON):
//! ```json
//! {"jobs": [
//!   {"id": "cds_armor_heavy_core_01_01_01", "output": "out/core.glb",
//!    "geometry": "Objects/Characters/Human/male_v7/armor/cds/m_cds_heavy_armor_01_core.skin"},
//!   {"id": "cds_combat_heavy_backpack_01_01_01", "output": "out/bp.glb",
//!    "geometry": "Objects/Characters/Human/backpack/cds/cds_combat_heavy_backpack_01.cga",
//!    "attach": {"bone": "backpack_attach_1_override", "item_offset_helper": "backpack_attach_1_loc",
//!               "skeleton": "Objects/Characters/Human/male_v7/armor/cds/m_cds_heavy_armor_01_core.skin",
//!               "extra_joints": ["wep_stocked_attach_2_override", "wep_stocked_attach_3_override",
//!                                "gadget_attach_1_override"]}}
//! ]}
//! ```
//! A `.skin` is written like `skin export --rig` (its own bones). With
//! `"graft_helpers_from": "<other .skin>"` (the male model of a female worn
//! item), every `*_override` helper bone of that skin the job's skin lacks is
//! added as a weight-less joint under the bone of the same name, with the
//! donor's parent-relative bind transform (missing donor ancestors grafted
//! too); the report lists them as `grafted`. A rigid
//! model (`.cga`/`.cgf`, or a `.cdf` whose CA_SKIN rides its own small
//! `.chr`) is bound with weight 1 to `attach.bone` of `attach.skeleton`, its
//! `item_offset_helper` on the bone, as the game's item port places it.
//! Each of `attach.extra_joints` the model has as a node / `.chr` bone (a
//! backpack's own weapon and gadget mounts) is added as a weight-less joint
//! under `attach.bone`, at its frame in the placed model, with its own name.
//! One JSON line per job goes to stdout.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::{Value as Json, json};
use starbreaker_p4k::MappedP4k;

use crate::common::load_p4k;
use crate::error::{CliError, Result};

pub fn run(jobs_file: PathBuf, p4k_path: Option<PathBuf>, threads: Option<usize>) -> Result<()> {
    let text = std::fs::read_to_string(&jobs_file)
        .map_err(|e| CliError::IoPath { source: e, path: jobs_file.display().to_string() })?;
    let root: Json = serde_json::from_str(&text)?;
    let jobs = root.get("jobs").and_then(Json::as_array).cloned().unwrap_or_default();
    let p4k = load_p4k(p4k_path.as_deref())?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.unwrap_or(0)).build()?;
    let failed = std::sync::atomic::AtomicUsize::new(0);
    pool.install(|| {
        jobs.par_iter().for_each(|job| {
            let started = std::time::Instant::now();
            let id = job.get("id").and_then(Json::as_str).unwrap_or("").to_string();
            let mut line = match run_job(&p4k, job) {
                Ok(report) => report,
                Err(e) => {
                    failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    json!({ "ok": false, "error": e.to_string() })
                }
            };
            line["id"] = json!(id);
            line["seconds"] = json!((started.elapsed().as_secs_f64() * 100.0).round() / 100.0);
            println!("{line}");
        });
    });
    let failed = failed.into_inner();
    eprintln!("{} jobs, {} failed", jobs.len(), failed);
    if failed > 0 {
        return Err(CliError::InvalidInput(format!("{failed} of {} jobs failed", jobs.len())));
    }
    Ok(())
}

/// P4k entry of a DataCore-style path (`Objects/...`, any case).
pub(crate) fn read(p4k: &MappedP4k, path: &str) -> Result<Vec<u8>> {
    let normalized = path.replace('/', "\\");
    let full = if normalized.to_lowercase().starts_with("data\\") { normalized } else { format!("Data\\{normalized}") };
    let entry = p4k.entry_case_insensitive(&full).ok_or_else(|| CliError::NotFound(format!("{full} not in the P4k")))?;
    Ok(p4k.read(entry)?)
}

fn stem(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.rsplit_once('.').map_or(file, |(s, _)| s).to_string()
}

fn ext(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

fn write_output(output: &str, glb: &[u8]) -> Result<()> {
    let output = PathBuf::from(output);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| CliError::IoPath { source: e, path: parent.display().to_string() })?;
    }
    let tmp = output.with_extension("glb.tmp");
    std::fs::write(&tmp, glb).map_err(|e| CliError::IoPath { source: e, path: tmp.display().to_string() })?;
    std::fs::rename(&tmp, &output).map_err(|e| CliError::IoPath { source: e, path: output.display().to_string() })?;
    Ok(())
}

fn run_job(p4k: &MappedP4k, job: &Json) -> Result<Json> {
    let geometry = job.get("geometry").and_then(Json::as_str).ok_or_else(|| CliError::InvalidInput("job without geometry".into()))?;
    let output = job.get("output").and_then(Json::as_str).ok_or_else(|| CliError::InvalidInput("job without output".into()))?;
    let name = job.get("name").and_then(Json::as_str).map(str::to_string).unwrap_or_else(|| stem(geometry));
    let kind = ext(geometry);
    if kind == "skin" && job.get("attach").is_none_or(Json::is_null) {
        let skin = read(p4k, geometry)?;
        let skinm = read(p4k, &format!("{geometry}m"))?;
        let mut bones = starbreaker_3d::skeleton::parse_skeleton(&skin)
            .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {geometry}")))?;
        let donor_path = job.get("graft_helpers_from").and_then(Json::as_str).filter(|p| !p.is_empty());
        let mut grafted = Vec::new();
        if let Some(donor_path) = donor_path {
            let donor = starbreaker_3d::skeleton::parse_skeleton(&read(p4k, donor_path)?)
                .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {donor_path}")))?;
            grafted = starbreaker_3d::skeleton::graft_missing_bones(&mut bones, &donor, "_override");
        }
        let rigged = starbreaker_3d::animated_skeleton_glb(&bones, Some(&skinm), &[], Some(&name))?;
        write_output(output, &rigged.glb)?;
        let mut report = json!({
            "ok": true, "output": output, "kind": "skin", "bytes": rigged.glb.len(),
            "vertices": rigged.vertex_count, "joints": rigged.joint_count,
            "unweighted_vertices": rigged.unweighted_vertices,
            "out_of_range_influences": rigged.out_of_range_influences,
            "truncated_vertices": rigged.truncated_vertices,
        });
        if let Some(donor_path) = donor_path {
            report["graft_helpers_from"] = json!(donor_path);
            report["grafted"] = grafted
                .iter()
                .map(|g| json!({ "name": g.name, "parent": g.parent, "ancestor": g.ancestor }))
                .collect();
        }
        return Ok(report);
    }

    let attach = job.get("attach").ok_or_else(|| CliError::InvalidInput(format!("rigid {kind} model needs \"attach\"")))?;
    let bone = attach.get("bone").and_then(Json::as_str).ok_or_else(|| CliError::InvalidInput("attach without bone".into()))?;
    let helper = attach.get("item_offset_helper").and_then(Json::as_str).filter(|h| !h.is_empty());
    let extra_joints: Vec<&str> = attach
        .get("extra_joints")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default();
    let skeleton_path = attach.get("skeleton").and_then(Json::as_str).ok_or_else(|| CliError::InvalidInput("attach without skeleton".into()))?;
    let skeleton_data = read(p4k, skeleton_path)?;
    let skeleton = starbreaker_3d::skeleton::parse_skeleton(&skeleton_data)
        .ok_or_else(|| CliError::InvalidInput(format!("no CompiledBones in {skeleton_path}")))?;

    let mut parts = Vec::new();
    let mut models = Vec::new();
    match kind.as_str() {
        "cga" | "cgf" => {
            let meta = read(p4k, geometry)?;
            let mesh = read(p4k, &format!("{geometry}m"))?;
            models.push(starbreaker_3d::load_rigid_cga(&meta, &mesh)?);
            parts.push(geometry.to_string());
        }
        "skin" => {
            let skin = read(p4k, geometry)?;
            let bones = starbreaker_3d::skeleton::parse_skeleton(&skin).unwrap_or_default();
            models.push(starbreaker_3d::load_rigid_skin(&read(p4k, &format!("{geometry}m"))?, &bones)?);
            parts.push(geometry.to_string());
        }
        "cdf" => {
            let (chr, skins) = cdf_parts(&read(p4k, geometry)?)?;
            let bones = match &chr {
                Some(chr) => starbreaker_3d::skeleton::parse_skeleton(&read(p4k, chr)?).unwrap_or_default(),
                None => Vec::new(),
            };
            for skin in &skins {
                let path = if ext(skin) == "skin" { format!("{skin}m") } else { skin.clone() };
                models.push(starbreaker_3d::load_rigid_skin(&read(p4k, &path)?, &bones)?);
                parts.push(skin.clone());
            }
            if models.is_empty() {
                return Err(CliError::InvalidInput(format!("{geometry}: no CA_SKIN attachment")));
            }
        }
        other => return Err(CliError::InvalidInput(format!("unsupported worn model .{other}"))),
    }
    let rigged = starbreaker_3d::rigid_on_bone_glb(&models, &skeleton, bone, helper, &extra_joints, Some(&name))?;
    write_output(output, &rigged.glb.glb)?;
    Ok(json!({
        "ok": true, "output": output, "kind": kind, "bytes": rigged.glb.glb.len(),
        "vertices": rigged.glb.vertex_count, "joints": rigged.joints,
        "bone": bone, "bone_position": rigged.bone_position, "skeleton": skeleton_path,
        "item_offset_helper": helper, "helper_found": rigged.helper_found, "parts": parts,
        "extra_joints": rigged.extra_joints.iter().map(|(n, p)| (n.clone(), json!(p))).collect::<serde_json::Map<_, _>>(),
    }))
}

/// The `.chr` model and CA_SKIN bindings of a character definition.
pub(crate) fn cdf_parts(data: &[u8]) -> Result<(Option<String>, Vec<String>)> {
    let text = if starbreaker_cryxml::is_cryxmlb(data) {
        format!("{}", starbreaker_cryxml::from_bytes(data)?)
    } else {
        String::from_utf8_lossy(data).into_owned()
    };
    let attr = |tag: &str, name: &str| -> Option<String> {
        let key = format!("{name}=\"");
        let start = tag.find(&key)? + key.len();
        Some(tag[start..].split('"').next()?.to_string())
    };
    let mut chr = None;
    let mut skins = Vec::new();
    for tag in text.split('<').skip(1) {
        let tag = tag.split('>').next().unwrap_or("");
        if tag.starts_with("Model ") {
            chr = attr(tag, "File").filter(|f| !f.is_empty());
        } else if tag.starts_with("Attachment ") && attr(tag, "Type").as_deref() == Some("CA_SKIN") {
            if let Some(binding) = attr(tag, "Binding").filter(|b| !b.is_empty()) {
                skins.push(binding);
            }
        }
    }
    Ok((chr, skins))
}
