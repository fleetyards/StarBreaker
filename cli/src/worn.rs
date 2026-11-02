//! `entity worn`: resolve the worn model of many items in one process.
//!
//! For each entity class (exact name, case-insensitive) and body (`Male`,
//! `Female`) it prints the file the character wears -- the tagged
//! SubGeometry's own `.skin`, not the `inventoryStoredEntity` prop that
//! `entity export --worn` prefers -- plus, for a rigid model (`.cga`, `.cgf`,
//! `.cdf` backpacks), the item port that carries it: the bone (`Helper`) on
//! the core's skeleton and the item's own offset helper node.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Value as Json, json};
use starbreaker_datacore::database::Database;
use starbreaker_datacore::types::Record;
use starbreaker_p4k::MappedP4k;

use crate::common::load_dcb_bytes;
use crate::error::{CliError, Result};

pub fn run(
    names: Vec<String>,
    list: Option<PathBuf>,
    bodies: Vec<String>,
    output: Option<PathBuf>,
    p4k_path: Option<PathBuf>,
) -> Result<()> {
    let mut wanted = names;
    if let Some(list) = &list {
        let text = std::fs::read_to_string(list).map_err(|e| CliError::IoPath { source: e, path: list.display().to_string() })?;
        wanted.extend(
            text.lines()
                .map(|l| l.split('#').next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.split_whitespace().next_back().unwrap_or(l).to_string()),
        );
    }
    if wanted.is_empty() {
        return Err(CliError::InvalidInput("name an entity class or pass --list".into()));
    }
    let (p4k, dcb_bytes) = load_dcb_bytes(p4k_path.as_deref(), None)?;
    let db = Database::from_bytes(&dcb_bytes)?;
    let entity_si = db
        .struct_id("EntityClassDefinition")
        .ok_or_else(|| CliError::NotFound("EntityClassDefinition struct not found in DCB".into()))?;
    let by_class: HashMap<String, &Record> = db
        .records_of_type(entity_si)
        .map(|r| {
            let name = db.resolve_string2(r.name_offset);
            (name.rsplit('.').next().unwrap_or(name).to_lowercase(), r)
        })
        .collect();

    let mut ports: Option<HashMap<String, Vec<PortHelper>>> = None;
    let mut items = serde_json::Map::new();
    for class in &wanted {
        let Some(record) = by_class.get(&class.to_lowercase()).copied() else {
            items.insert(class.clone(), json!({ "error": "no EntityClassDefinition record" }));
            continue;
        };
        let record_json = record_json(&db, record);
        let attach = component(&record_json, "SAttachableComponentParams").and_then(|c| c.get("AttachDef"));
        let item_type = attach.and_then(|a| a.get("Type")).and_then(Json::as_str).unwrap_or("").to_string();
        let sub_type = attach.and_then(|a| a.get("SubType")).and_then(Json::as_str).unwrap_or("").to_string();
        let mut entry = serde_json::Map::new();
        entry.insert("class".into(), json!(db.resolve_string2(record.name_offset).rsplit('.').next().unwrap_or("")));
        entry.insert("type".into(), json!(item_type));
        entry.insert("sub_type".into(), json!(sub_type));
        let mut rigid = false;
        for body in &bodies {
            let tag = match body.to_lowercase().as_str() {
                "male" => "Male",
                "female" => "Female",
                other => return Err(CliError::InvalidInput(format!("unknown body '{other}' (male, female)"))),
            };
            let resolved = starbreaker_3d::worn_model_paths(&db, record, tag, true);
            let value = match resolved {
                None => Json::Null,
                Some(paths) => {
                    let kind = extension(&paths.geometry);
                    rigid |= kind != "skin";
                    json!({
                        "geometry": paths.geometry,
                        "kind": kind,
                        "material": paths.material,
                        "tagged": paths.tagged,
                        "stored": paths.stored,
                        "in_p4k": p4k.as_ref().map(|p| in_p4k(p, &paths.geometry)),
                    })
                }
            };
            entry.insert(body.to_lowercase(), value);
        }
        if rigid && !item_type.is_empty() {
            let ports = ports.get_or_insert_with(|| port_helpers(&db, entity_si));
            let helpers: Vec<Json> = summarize(ports.get(&item_type.to_lowercase()).map(Vec::as_slice).unwrap_or(&[]));
            entry.insert("attach".into(), Json::Array(helpers));
        }
        items.insert(class.to_lowercase(), Json::Object(entry));
    }

    let text = serde_json::to_string_pretty(&json!({ "items": items }))?;
    match output {
        Some(path) => std::fs::write(&path, text).map_err(|e| CliError::IoPath { source: e, path: path.display().to_string() })?,
        None => println!("{text}"),
    }
    Ok(())
}

fn extension(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

fn in_p4k(p4k: &MappedP4k, path: &str) -> bool {
    let normalized = path.replace('/', "\\");
    let full = if normalized.to_lowercase().starts_with("data\\") { normalized } else { format!("Data\\{normalized}") };
    p4k.entry_case_insensitive(&full).is_some()
}

fn record_json(db: &Database, record: &Record) -> Json {
    starbreaker_datacore::export::to_json_compact(db, record)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(Json::Null)
}

fn component<'j>(record: &'j Json, type_name: &str) -> Option<&'j Json> {
    record
        .get("_RecordValue_")?
        .get("Components")?
        .as_array()?
        .iter()
        .find(|c| c.get("_Type_").and_then(Json::as_str) == Some(type_name))
}

/// One item port that accepts an item type, on some entity (a core).
#[derive(Clone)]
struct PortHelper {
    owner: String,
    port: String,
    helper: String,
    offset_helper: String,
    offset_position: [f64; 3],
    offset_rotation: [f64; 3],
}

/// Item type (lower case) -> the bone ports of every armour torso (core)
/// that accept it. Backpacks hang on the core's `backpack` port.
fn port_helpers(db: &Database, entity_si: starbreaker_datacore::types::StructId) -> HashMap<String, Vec<PortHelper>> {
    let mut out: HashMap<String, Vec<PortHelper>> = HashMap::new();
    let type_path = db.compile_path::<String>(entity_si, "Components[SAttachableComponentParams].AttachDef.Type").ok();
    for record in db.records_of_type(entity_si) {
        let item_type = type_path
            .as_ref()
            .and_then(|p| db.query_single::<String>(p, record).ok().flatten())
            .unwrap_or_default();
        if !item_type.eq_ignore_ascii_case("Char_Armor_Torso") {
            continue;
        }
        let json = record_json(db, record);
        let Some(ports) = component(&json, "SItemPortContainerComponentParams").and_then(|c| c.get("Ports")).and_then(Json::as_array) else {
            continue;
        };
        let owner = db.resolve_string2(record.name_offset).rsplit('.').next().unwrap_or("").to_string();
        for port in ports {
            let helper = port.pointer("/AttachmentImplementation/Helper/Helper");
            let Some(helper) = helper else { continue };
            let f = |v: Option<&Json>| v.and_then(Json::as_f64).unwrap_or(0.0);
            let offset = helper.get("Offset");
            let pos = offset.and_then(|o| o.get("Position"));
            let rot = offset.and_then(|o| o.get("Rotation"));
            let entry = PortHelper {
                owner: owner.clone(),
                port: port.get("Name").and_then(Json::as_str).unwrap_or("").to_string(),
                helper: helper.get("Name").and_then(Json::as_str).unwrap_or("").to_string(),
                offset_helper: helper.get("ItemOffsetHelperName").and_then(Json::as_str).unwrap_or("").to_string(),
                offset_position: [f(pos.and_then(|p| p.get("x"))), f(pos.and_then(|p| p.get("y"))), f(pos.and_then(|p| p.get("z")))],
                offset_rotation: [f(rot.and_then(|p| p.get("x"))), f(rot.and_then(|p| p.get("y"))), f(rot.and_then(|p| p.get("z")))],
            };
            for accepted in port.get("Types").and_then(Json::as_array).into_iter().flatten() {
                if let Some(t) = accepted.get("Type").and_then(Json::as_str) {
                    out.entry(t.to_lowercase()).or_default().push(entry.clone());
                }
            }
        }
    }
    out
}

/// Distinct (port, helper, offset helper, offset) combinations, each with
/// the number of cores using it and one example core.
fn summarize(helpers: &[PortHelper]) -> Vec<Json> {
    let mut groups: Vec<(String, usize, &PortHelper)> = Vec::new();
    for h in helpers {
        let key = format!("{}|{}|{}|{:?}|{:?}", h.port, h.helper, h.offset_helper, h.offset_position, h.offset_rotation);
        match groups.iter_mut().find(|(k, _, _)| *k == key) {
            Some((_, n, _)) => *n += 1,
            None => groups.push((key, 1, h)),
        }
    }
    groups.sort_by(|a, b| b.1.cmp(&a.1));
    groups
        .into_iter()
        .map(|(_, count, h)| {
            json!({
                "port": h.port,
                "bone": h.helper,
                "item_offset_helper": h.offset_helper,
                "offset_position": h.offset_position,
                "offset_rotation_deg": h.offset_rotation,
                "cores": count,
                "example_core": h.owner,
            })
        })
        .collect()
}
