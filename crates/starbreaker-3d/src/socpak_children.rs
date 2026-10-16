//! Nested object containers: a socpak's root XML can list further socpaks
//! under `<ChildObjectContainers>` (`<Child external="1" name=... pos=... rot=...>`),
//! placed relative to the parent container. The GS Constellation cockpits pull
//! the shared `rsi_constellation_gs_base_cockpit` in this way.
//!
//! `child_object_containers` reads those references from a socpak in the P4k;
//! `parse_child_object_containers` does the XML part (text or CryXmlB).

use quick_xml::events::Event;
use quick_xml::Reader;
use starbreaker_p4k::{MappedP4k, P4kArchive};

/// A socpak placed inside another one.
#[derive(Debug, Clone, PartialEq)]
pub struct ChildContainerRef {
    pub file_name: String,
    /// Placement relative to the parent container (column-major).
    pub local_transform: [[f32; 4]; 4],
}

/// Child containers listed by the socpak at `socpak_path`. Empty when the
/// socpak is missing or lists none.
pub fn child_object_containers(p4k: &MappedP4k, socpak_path: &str) -> Vec<ChildContainerRef> {
    let p4k_path = {
        let normalized = socpak_path.replace('/', "\\");
        if normalized.to_ascii_lowercase().starts_with("data\\") {
            normalized
        } else {
            format!("Data\\{normalized}")
        }
    };
    let Some(entry) = p4k.entry_case_insensitive(&p4k_path) else {
        return Vec::new();
    };
    let Ok(socpak_data) = p4k.read(entry) else {
        return Vec::new();
    };
    let Ok(inner) = P4kArchive::from_bytes(&socpak_data) else {
        return Vec::new();
    };
    let stem = socpak_path
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(socpak_path)
        .to_ascii_lowercase();
    let root_xml_name = format!("{}.xml", stem.strip_suffix(".socpak").unwrap_or(&stem));
    inner
        .entries()
        .iter()
        .find(|entry| entry.name.replace('/', "\\").to_ascii_lowercase() == root_xml_name)
        .and_then(|entry| inner.read(entry).ok())
        .map(|data| parse_child_object_containers(&data))
        .unwrap_or_default()
}

/// Parse the external `<Child>` entries of `<ChildObjectContainers>`.
pub fn parse_child_object_containers(data: &[u8]) -> Vec<ChildContainerRef> {
    if let Ok(xml) = starbreaker_cryxml::from_bytes(data) {
        let mut children = Vec::new();
        for list in xml.node_children(xml.root()) {
            if xml.node_tag(list) != "ChildObjectContainers" {
                continue;
            }
            for child in xml.node_children(list) {
                if xml.node_tag(child) == "Child" {
                    children.extend(child_ref(|key| {
                        xml.node_attributes(child)
                            .find(|(name, _)| *name == key)
                            .map(|(_, value)| value.to_string())
                    }));
                }
            }
        }
        return children;
    }

    let Ok(text) = std::str::from_utf8(data) else {
        return Vec::new();
    };
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut depth_in_list: Option<usize> = None;
    let mut depth = 0usize;
    let mut children = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                depth += 1;
                if e.name().as_ref() == b"ChildObjectContainers" {
                    depth_in_list = Some(depth);
                } else if e.name().as_ref() == b"Child" && depth_in_list == Some(depth - 1) {
                    children.extend(child_ref_from_attributes(&e, &reader));
                }
            }
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"Child" && depth_in_list == Some(depth) {
                    children.extend(child_ref_from_attributes(&e, &reader));
                }
            }
            Ok(Event::End(e)) => {
                if e.name().as_ref() == b"ChildObjectContainers" {
                    depth_in_list = None;
                }
                depth = depth.saturating_sub(1);
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => {
                log::warn!("failed to parse child object containers: {e}");
                break;
            }
        }
        buf.clear();
    }
    children
}

fn child_ref_from_attributes(
    element: &quick_xml::events::BytesStart,
    reader: &Reader<&[u8]>,
) -> Option<ChildContainerRef> {
    let attrs: Vec<(String, String)> = element
        .attributes()
        .flatten()
        .filter_map(|attr| {
            let value = attr.decode_and_unescape_value(reader.decoder()).ok()?;
            Some((String::from_utf8_lossy(attr.key.as_ref()).into_owned(), value.into_owned()))
        })
        .collect();
    child_ref(|key| attrs.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone()))
}

fn child_ref(attr: impl Fn(&str) -> Option<String>) -> Option<ChildContainerRef> {
    // A non-external child lives inside the parent socpak, whose .soc files
    // are all loaded already.
    if attr("external").as_deref() != Some("1") {
        return None;
    }
    let file_name = attr("name").filter(|name| name.to_ascii_lowercase().ends_with(".socpak"))?;
    let floats = |key: &str| -> Vec<f32> {
        attr(key)
            .map(|value| value.split(',').filter_map(|part| part.trim().parse().ok()).collect())
            .unwrap_or_default()
    };
    let translation = match floats("pos").as_slice() {
        [x, y, z] => glam::Vec3::new(*x, *y, *z),
        _ => glam::Vec3::ZERO,
    };
    // `rot` is a quaternion written w,x,y,z.
    let rotation = match floats("rot").as_slice() {
        [w, x, y, z] => glam::Quat::from_xyzw(*x, *y, *z, *w).normalize(),
        _ => glam::Quat::IDENTITY,
    };
    Some(ChildContainerRef {
        file_name,
        local_transform: glam::Mat4::from_rotation_translation(rotation, translation).to_cols_array_2d(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_external_children_with_their_placement() {
        let xml = br#"<ObjectContainer version="12" name="cockpit">
 <ChildObjectContainers>
  <Child external="1" name="Data/objectcontainers/ships/rsi/base_cockpit.socpak" pos="1,2,3" rot="0.70710677,0,0,0.70710677"/>
  <Child external="0" name="inline_child" pos="0,0,0" rot="1,0,0,0"/>
 </ChildObjectContainers>
 <StreamingOCs><OC name="cockpit.soc"/></StreamingOCs>
</ObjectContainer>"#;

        let children = parse_child_object_containers(xml);

        assert_eq!(children.len(), 1);
        assert_eq!(children[0].file_name, "Data/objectcontainers/ships/rsi/base_cockpit.socpak");
        let transform = glam::Mat4::from_cols_array_2d(&children[0].local_transform);
        let moved = transform.transform_point3(glam::Vec3::X);
        assert!((moved - glam::Vec3::new(1.0, 3.0, 3.0)).length() < 1e-5, "{moved:?}");
    }

    #[test]
    fn container_without_children_lists_none() {
        let xml = br#"<ObjectContainer name="tail"><StreamingOCs/></ObjectContainer>"#;
        assert!(parse_child_object_containers(xml).is_empty());
    }
}
