//! Bridge: convert a `starbreaker_3d::Mesh` into Blender 5.x `.blend` file bytes.
//!
//! The output is a single-scene `.blend` containing one OB_MESH object.
//! Material slots are allocated (null pointers) to match the submesh count.

use starbreaker_3d::types::SubMesh;
use starbreaker_3d::Mesh;
use starbreaker_blend::{
    bytes4_data, build_attribute, build_attribute_array, build_base,
    build_master_collection, build_collection_object, build_file_global, build_layer_collection,
    build_mat_ptr_array, build_mat_ptr_array_from_ptrs, build_material, build_matbits, build_mesh, build_object, build_scene,
    build_tool_settings, build_view_layer, compress_blend_bytes,
    floats2_data, floats3_data, ints2_data, ints_data, triangle_edge_topology,
    startup_ui_prefix_bytes, write_block, write_block_header, PtrAlloc,
    ATTR_DOMAIN_CORNER, ATTR_DOMAIN_FACE, ATTR_DOMAIN_POINT, ATTR_TYPE_BYTE_COLOR,
    ATTR_DOMAIN_EDGE, ATTR_TYPE_FLOAT2, ATTR_TYPE_FLOAT3, ATTR_TYPE_INT, ATTR_TYPE_INT32_2D,
    BLEND_MAGIC, DNA1_BYTES,
    SDNA_IDX_ATTRIBUTE, SDNA_IDX_ATTRIBUTE_ARRAY, SDNA_IDX_BASE, SDNA_IDX_COLLECTION, SDNA_IDX_COLLECTION_OBJECT,
    SDNA_IDX_DNA1, SDNA_IDX_FILE_GLOBAL, SDNA_IDX_LAYER_COLLECTION, SDNA_IDX_MATERIAL, SDNA_IDX_MESH,
    SDNA_IDX_OBJECT, SDNA_IDX_SCENE, SDNA_IDX_TOOL_SETTINGS, SDNA_IDX_VIEW_LAYER,
    STARTUP_UI_SCREEN_PTR,
};

/// Convert a `starbreaker_3d::Mesh` into `.blend` file bytes for Blender 5.1.x.
///
/// Produces a scene `.blend` containing a single OB_MESH object named `name`.
/// The mesh indices are expected to be triangles (3 indices per face).
///
/// Attributes written:
/// - `position`     (POINT / FLOAT3) — always
/// - `.edge_verts`  (EDGE / INT2)     — always
/// - `.corner_vert` (CORNER / INT)   — always
/// - `.corner_edge` (CORNER / INT)   — always
/// - `UVMap`        (CORNER / FLOAT2) — when `mesh.uvs` is `Some`
/// - `Color`        (CORNER / BYTE_COLOR) — when `mesh.colors` is `Some`
/// - `material_index` (FACE / INT)   — always (per-polygon material slot index from submeshes)
///
/// Per-vertex UVs and colors are expanded to per-loop (corner) data.
/// Per-face material indices are filled from `mesh.submeshes` ranges.
/// Custom normals are deferred to Phase 68.
pub(crate) fn mesh_to_blend(name: &str, mesh: &Mesh) -> Vec<u8> {
    let totvert = mesh.positions.len();
    let totloop = mesh.indices.len();
    let totpoly = totloop / 3;
    let (edge_verts, corner_edges) = triangle_edge_topology(&mesh.indices);
    let totedge = edge_verts.len();
    let material_names: Vec<String> = if mesh.submeshes.is_empty() {
        vec![format!("{name}:material_0")]
    } else {
        mesh.submeshes
            .iter()
            .enumerate()
            .map(|(slot, submesh)| {
                format!(
                    "{name}:{}",
                    submesh
                        .material_name
                        .as_deref()
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("material_{slot}"))
                )
            })
            .collect()
    };
    let mat_slots = material_names.len() as i16;

    let mut ptrs = PtrAlloc::new(0x1000);

    let _screen_ptr    = ptrs.alloc();
    let _wm_ptr        = ptrs.alloc();
    let object_ptr     = ptrs.alloc();
    let mesh_ptr       = ptrs.alloc();
    let mesh_mat_ptr   = ptrs.alloc();
    let obj_mat_ptr    = ptrs.alloc();
    let obj_matbits_ptr = ptrs.alloc();
    let material_ptrs: Vec<u64> = (0..material_names.len()).map(|_| ptrs.alloc()).collect();
    let scene_ptr      = ptrs.alloc();
    let view_layer_ptr = ptrs.alloc();
    let tool_settings_ptr = ptrs.alloc();
    let base_ptr       = ptrs.alloc();
    let collection_ptr = ptrs.alloc();
    let collection_object_ptr = ptrs.alloc();
    let layer_collection_ptr = ptrs.alloc();
    let poly_offs_ptr  = ptrs.alloc();
    let attrs_ptr      = ptrs.alloc();

    // Always-present attributes
    let name_pos_ptr  = ptrs.alloc();
    let name_ev_ptr   = ptrs.alloc();
    let name_cv_ptr   = ptrs.alloc();
    let name_ce_ptr   = ptrs.alloc();
    let array_pos_ptr = ptrs.alloc();
    let array_ev_ptr  = ptrs.alloc();
    let array_cv_ptr  = ptrs.alloc();
    let array_ce_ptr  = ptrs.alloc();
    let raw_pos_ptr   = ptrs.alloc();
    let raw_ev_ptr    = ptrs.alloc();
    let raw_cv_ptr    = ptrs.alloc();
    let raw_ce_ptr    = ptrs.alloc();

    // material_index (FACE domain)
    let name_matidx_ptr = ptrs.alloc();
    let array_matidx_ptr = ptrs.alloc();
    let raw_matidx_ptr = ptrs.alloc();

    // Optional: UV map
    let (name_uv_ptr, array_uv_ptr, raw_uv_ptr) = if mesh.uvs.is_some() {
        (ptrs.alloc(), ptrs.alloc(), ptrs.alloc())
    } else {
        (0, 0, 0)
    };

    // Optional: vertex colors
    let (name_col_ptr, array_col_ptr, raw_col_ptr) = if mesh.colors.is_some() {
        (ptrs.alloc(), ptrs.alloc(), ptrs.alloc())
    } else {
        (0, 0, 0)
    };

    // ── Geometry data ─────────────────────────────────────────────────────────

    // poly_offsets: [0, 3, 6, ..., totloop] — totpoly+1 entries (sentinel at end)
    let poly_offsets: Vec<i32> = (0..=totpoly as i32).map(|i| i * 3).collect();
    let raw_poly_offsets = ints_data(&poly_offsets);

    // corner_vert[i] = vertex index for loop corner i
    let corner_verts: Vec<i32> = mesh.indices.iter().map(|&i| i as i32).collect();

    let raw_position   = floats3_data(&mesh.positions);
    let raw_edge_verts = ints2_data(&edge_verts);
    let raw_corner_vert = ints_data(&corner_verts);
    let raw_corner_edge = ints_data(&corner_edges);

    // Per-polygon material index: fill from submesh ranges
    let mut material_indices: Vec<i32> = vec![0; totpoly];
    for (mat_idx, submesh) in mesh.submeshes.iter().enumerate() {
        let start_face = submesh.first_index / 3;
        let num_faces = submesh.num_indices / 3;
        for i in 0..num_faces {
            if (start_face + i) < totpoly as u32 {
                material_indices[(start_face + i) as usize] = mat_idx as i32;
            }
        }
    }
    let raw_material_index = ints_data(&material_indices);

    // Expand per-vertex UV → per-loop (CORNER domain requires one entry per loop corner)
    let raw_uv: Option<Vec<u8>> = mesh.uvs.as_ref().map(|uvs| {
        let expanded: Vec<[f32; 2]> = mesh.indices.iter().map(|&i| uvs[i as usize]).collect();
        floats2_data(&expanded)
    });

    // Expand per-vertex colors → per-loop
    let raw_color: Option<Vec<u8>> = mesh.colors.as_ref().map(|colors| {
        let expanded: Vec<[u8; 4]> = mesh.indices.iter().map(|&i| colors[i as usize]).collect();
        bytes4_data(&expanded)
    });

    // ── Attribute descriptor blob ─────────────────────────────────────────────

    let mut attr_blob: Vec<u8> = Vec::new();
    let mut num_attrs: u32 = 5; // position + edge_verts + corner_vert + corner_edge + material_index

    attr_blob.extend_from_slice(&build_attribute(
        name_pos_ptr, ATTR_TYPE_FLOAT3, ATTR_DOMAIN_POINT, array_pos_ptr,
    ));
    attr_blob.extend_from_slice(&build_attribute(
        name_ev_ptr, ATTR_TYPE_INT32_2D, ATTR_DOMAIN_EDGE, array_ev_ptr,
    ));
    attr_blob.extend_from_slice(&build_attribute(
        name_cv_ptr, ATTR_TYPE_INT, ATTR_DOMAIN_CORNER, array_cv_ptr,
    ));
    attr_blob.extend_from_slice(&build_attribute(
        name_ce_ptr, ATTR_TYPE_INT, ATTR_DOMAIN_CORNER, array_ce_ptr,
    ));
    attr_blob.extend_from_slice(&build_attribute(
        name_matidx_ptr, ATTR_TYPE_INT, ATTR_DOMAIN_FACE, array_matidx_ptr,
    ));
    if mesh.uvs.is_some() {
        attr_blob.extend_from_slice(&build_attribute(
            name_uv_ptr, ATTR_TYPE_FLOAT2, ATTR_DOMAIN_CORNER, array_uv_ptr,
        ));
        num_attrs += 1;
    }
    if mesh.colors.is_some() {
        attr_blob.extend_from_slice(&build_attribute(
            name_col_ptr, ATTR_TYPE_BYTE_COLOR, ATTR_DOMAIN_CORNER, array_col_ptr,
        ));
        num_attrs += 1;
    }

    // ── Datablocks ────────────────────────────────────────────────────────────

    let object_data = build_object(
        name, mesh_ptr, obj_mat_ptr, obj_matbits_ptr, mat_slots as i32, 0,
    );
    let scene_data = build_scene(name, view_layer_ptr, collection_ptr, tool_settings_ptr);
    let tool_settings_data = build_tool_settings();
    let view_layer_data = build_view_layer("ViewLayer", base_ptr, layer_collection_ptr);
    let base_data = build_base(object_ptr);
    let collection_data = build_master_collection(collection_object_ptr, collection_object_ptr, 0, 0);
    let collection_object_data = build_collection_object(object_ptr);
    let layer_collection_data = build_layer_collection(collection_ptr);
    let mesh_data = build_mesh(
        name, totvert, totedge, totpoly, totloop,
        poly_offs_ptr, attrs_ptr,
        mesh_mat_ptr, mat_slots,
        0, 0, 0,
        num_attrs,
    );
    let mesh_mat_array = build_mat_ptr_array_from_ptrs(&material_ptrs);
    let obj_mat_array  = build_mat_ptr_array(mat_slots as usize);
    let obj_matbits    = build_matbits(mat_slots as usize);

    let arr_pos = build_attribute_array(raw_pos_ptr,  totvert as i64);
    let arr_ev  = build_attribute_array(raw_ev_ptr,   totedge as i64);
    let arr_cv  = build_attribute_array(raw_cv_ptr,   totloop as i64);
    let arr_ce  = build_attribute_array(raw_ce_ptr,   totloop as i64);

    // ── Assemble file ─────────────────────────────────────────────────────────

    let mut out: Vec<u8> = Vec::with_capacity(512 * 1024);
    out.extend_from_slice(BLEND_MAGIC);

    let file_global = build_file_global(STARTUP_UI_SCREEN_PTR, scene_ptr, view_layer_ptr);
    write_block(&mut out, b"GLOB", SDNA_IDX_FILE_GLOBAL, 0x10, 1, &file_global);
    out.extend_from_slice(&startup_ui_prefix_bytes());

    // Minimal scene graph so Blender opens this as a normal scene, not library-only data.
    // CRITICAL: All DATA blocks for Scene must be consecutive immediately after SC\0\0.
    write_block(&mut out, b"SC\0\0", SDNA_IDX_SCENE, scene_ptr, 1, &scene_data);
    // SC DATA sequence:
    write_block(&mut out, b"DATA", SDNA_IDX_TOOL_SETTINGS, tool_settings_ptr, 1, &tool_settings_data);
    write_block(&mut out, b"DATA", SDNA_IDX_VIEW_LAYER, view_layer_ptr, 1, &view_layer_data);
    write_block(&mut out, b"DATA", SDNA_IDX_LAYER_COLLECTION, layer_collection_ptr, 1, &layer_collection_data);
    write_block(&mut out, b"DATA", SDNA_IDX_COLLECTION, collection_ptr, 1, &collection_data);  // embedded master_collection
    write_block(
        &mut out,
        b"DATA",
        SDNA_IDX_COLLECTION_OBJECT,
        collection_object_ptr,
        1,
        &collection_object_data,
    );
    write_block(&mut out, b"DATA", SDNA_IDX_BASE, base_ptr, 1, &base_data);

    // OB block + DATA blocks (gap=1 rule: mat** and matbits must immediately follow OB)
    write_block(&mut out, b"OB\0\0", SDNA_IDX_OBJECT, object_ptr, 1, &object_data);
    write_block(&mut out, b"DATA", 0, obj_mat_ptr,      1, &obj_mat_array);
    write_block(&mut out, b"DATA", 0, obj_matbits_ptr,  1, &obj_matbits);

    // ME block + DATA block (gap=1 rule: mesh mat** must immediately follow ME)
    write_block(&mut out, b"ME\0\0", SDNA_IDX_MESH, mesh_ptr, 1, &mesh_data);
    write_block(&mut out, b"DATA", 0, mesh_mat_ptr, 1, &mesh_mat_array);

    // Attribute descriptor block (all Attribute structs concatenated)
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE, attrs_ptr, num_attrs as u32, &attr_blob);

    // Attribute name strings
    write_block(&mut out, b"DATA", 0, name_pos_ptr, 1, b"position\0");
    write_block(&mut out, b"DATA", 0, name_ev_ptr,  1, b".edge_verts\0");
    write_block(&mut out, b"DATA", 0, name_cv_ptr,  1, b".corner_vert\0");
    write_block(&mut out, b"DATA", 0, name_ce_ptr,  1, b".corner_edge\0");
    write_block(&mut out, b"DATA", 0, name_matidx_ptr, 1, b"material_index\0");

    // Attribute array descriptors + raw data (topology, position, material_index)
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_pos_ptr, 1, &arr_pos);
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_ev_ptr,  1, &arr_ev);
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_cv_ptr,  1, &arr_cv);
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_ce_ptr,  1, &arr_ce);
    let arr_matidx = build_attribute_array(raw_matidx_ptr, totpoly as i64);
    write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_matidx_ptr, 1, &arr_matidx);
    write_block(&mut out, b"DATA", 0, raw_pos_ptr, 1, &raw_position);
    write_block(&mut out, b"DATA", 0, raw_ev_ptr,  1, &raw_edge_verts);
    write_block(&mut out, b"DATA", 0, raw_cv_ptr,  1, &raw_corner_vert);
    write_block(&mut out, b"DATA", 0, raw_ce_ptr,  1, &raw_corner_edge);
    write_block(&mut out, b"DATA", 0, raw_matidx_ptr, 1, &raw_material_index);

    // Optional: UV map
    if let Some(ref uv_data) = raw_uv {
        let arr_uv = build_attribute_array(raw_uv_ptr, totloop as i64);
        write_block(&mut out, b"DATA", 0,  name_uv_ptr,  1, b"UVMap\0");
        write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_uv_ptr, 1, &arr_uv);
        write_block(&mut out, b"DATA", 0,  raw_uv_ptr,   1, uv_data);
    }

    // Optional: vertex colors
    if let Some(ref color_data) = raw_color {
        let arr_col = build_attribute_array(raw_col_ptr, totloop as i64);
        write_block(&mut out, b"DATA", 0,  name_col_ptr,  1, b"Color\0");
        write_block(&mut out, b"DATA", SDNA_IDX_ATTRIBUTE_ARRAY, array_col_ptr, 1, &arr_col);
        write_block(&mut out, b"DATA", 0,  raw_col_ptr,   1, color_data);
    }

    // Polygon offset array
    write_block(&mut out, b"DATA", 0, poly_offs_ptr, 1, &raw_poly_offsets);

    for (material_ptr, material_name) in material_ptrs.iter().zip(material_names.iter()) {
        let material_data = build_material(material_name);
        write_block(&mut out, b"MA\0\0", SDNA_IDX_MATERIAL, *material_ptr, 1, &material_data);
    }

    write_block(&mut out, b"DNA1", SDNA_IDX_DNA1, 0x01, 1, DNA1_BYTES);
    write_block_header(&mut out, b"ENDB", 0, 0, 0, 0);

    compress_blend_bytes(&out)
}

/// Build a master assembly `.blend` that opens as a normal Blender scene.
///
/// A scene-only file can be interpreted by Blender as a library payload and
/// trigger "Library file, loading empty scene". To avoid that, we emit a tiny
/// placeholder mesh/object scene. The addon can replace/remove this placeholder
/// once it links real component assets.
#[allow(dead_code)]
pub(crate) fn create_master_blend(entity_name: &str) -> Vec<u8> {
    let placeholder = Mesh {
        positions: vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]],
        indices: vec![0, 1, 2],
        uvs: None,
        secondary_uvs: None,
        normals: None,
        tangents: None,
        colors: None,
        submeshes: vec![SubMesh {
            material_name: None,
            material_id: 0,
            source_material_id: None,
            first_index: 0,
            num_indices: 3,
            first_vertex: 0,
            num_vertices: 3,
            node_parent_index: 0,
        }],
        model_min: [0.0, 0.0, 0.0],
        model_max: [0.001, 0.001, 0.0],
        scaling_min: [0.0, 0.0, 0.0],
        scaling_max: [0.001, 0.001, 0.0],
        skin_weights: None,
    };

    mesh_to_blend(entity_name, &placeholder)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip test: load a real game mesh from the P4K, convert to `.blend`,
    /// and verify the output is structurally valid.
    ///
    /// Requires `SC_DATA_P4K` env var pointing to Data.p4k.
    /// Run with:
    ///   `cargo test --bin starbreaker blend_roundtrip -- --include-ignored`
    ///
    /// Set `BLEND_ROUNDTRIP_WRITE=1` to write the output to `/tmp/blend_roundtrip.blend`
    /// for manual inspection in Blender.
    #[test]
    #[ignore = "requires SC_DATA_P4K env var pointing to Data.p4k"]
    fn blend_roundtrip_from_p4k() {
        use starbreaker_p4k::MappedP4k;

        let p4k_path = std::env::var("SC_DATA_P4K")
            .expect("SC_DATA_P4K must be set to run this test");

        let p4k = MappedP4k::open(&p4k_path).expect("Failed to open P4K");

        // Find the first .skinm file (any game mesh; will vary by SC version)
        let entry = p4k
            .entries()
            .iter()
            .find(|e| e.name.to_lowercase().ends_with(".skinm"))
            .expect("No .skinm files found in P4K");

        eprintln!("Testing with: {}", entry.name);
        let data = p4k.read(entry).expect("Failed to read entry");

        let mesh = starbreaker_3d::parse_skin(&data).expect("Failed to parse skin mesh");

        let totvert = mesh.positions.len();
        let totloop = mesh.indices.len();
        let totpoly = totloop / 3;

        eprintln!(
            "Mesh: {} verts, {} loops, {} polys, {} submeshes, uvs={}, colors={}",
            totvert, totloop, totpoly, mesh.submeshes.len(),
            mesh.uvs.is_some(), mesh.colors.is_some(),
        );

        assert!(totvert > 0, "Mesh has no vertices");
        assert!(totloop > 0, "Mesh has no loops");
        assert_eq!(totloop % 3, 0, "Loop count not divisible by 3 (not pure triangles)");

        let blend_bytes = mesh_to_blend("TestMesh", &mesh);

        assert!(
            blend_bytes.starts_with(BLEND_MAGIC),
            "Output does not start with Blender 5.x magic header"
        );

        // Rough lower bound: position data (12 bytes/vert) + loop data (4 bytes/loop) + headers
        let min_expected = totvert * 12 + totloop * 4 + 2048;
        assert!(
            blend_bytes.len() >= min_expected,
            "Output file too small: {} bytes (expected at least {})",
            blend_bytes.len(),
            min_expected,
        );

        if std::env::var("BLEND_ROUNDTRIP_WRITE").is_ok() {
            std::fs::write("/tmp/blend_roundtrip.blend", &blend_bytes)
                .expect("Failed to write test output");
            eprintln!("Written {} bytes to /tmp/blend_roundtrip.blend", blend_bytes.len());
        }
    }
}
