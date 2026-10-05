//! Phase 4B tests for decal vertex group creation.

use super::*;

#[test]
fn test_identify_meshes_with_decals_excludes_pom_only_damage_material() {
    let mesh_materials = vec![(
        "pitbull_body".to_string(),
        vec![(
            "Damage_Internal_Structure".to_string(),
            "HardSurface".to_string(),
            "%PARALLAX_OCCLUSION_MAPPING%USE_SPECULAR_MAPS".to_string(),
            vec![
                "WearBlendBase".to_string(),
                "PomDisplacement".to_string(),
                "POMHeightBias".to_string(),
                "DamagePerObjectWear".to_string(),
            ],
        )],
    )];

    let result = identify_meshes_with_decals(&mesh_materials).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_identify_meshes_with_decals_keeps_non_meshdecal_stencil_material() {
    let mesh_materials = vec![(
        "marksman".to_string(),
        vec![(
            "graphic_decals_a".to_string(),
            "Illum".to_string(),
            "%VERTDATA%STENCIL_MAP%USE_DAMAGE_MAP".to_string(),
            vec![
                "StencilOpacity".to_string(),
                "StencilDiffuseColor".to_string(),
                "StencilBreakupTiling".to_string(),
            ],
        )],
    )];

    let result = identify_meshes_with_decals(&mesh_materials).unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].mesh_path, "marksman");
    assert_eq!(result[0].decal_materials.len(), 1);
    assert!(result[0].decal_materials[0].is_decal);
    assert!(!result[0].decal_materials[0].is_pom);
}

#[test]
fn test_identify_meshes_with_decals_excludes_meshdecal_glow_without_vertdata_or_stencil() {
    let mesh_materials = vec![(
        "pitbull_glow".to_string(),
        vec![(
            "Decal_Glow_Linked".to_string(),
            "MeshDecal".to_string(),
            "%DIFFUSE_MAP%USE_DAMAGE_MAP".to_string(),
            vec![
                "DecalDiffuseOpacity".to_string(),
                "DecalFalloff".to_string(),
                "DecalAlphaMultiplier".to_string(),
            ],
        )],
    )];

    let result = identify_meshes_with_decals(&mesh_materials).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_identify_meshes_with_decals_excludes_glass_shader_with_decal_params() {
    let mesh_materials = vec![(
        "pitbull_glass".to_string(),
        vec![(
            "Glass_Int".to_string(),
            "GlassPBR".to_string(),
            "%DECAL%DIRT%REFRACTION%PER_VERTEX_TINT".to_string(),
            vec![
                "DecalTiling".to_string(),
                "DecalSmoothness".to_string(),
                "DecalReflectance".to_string(),
            ],
        )],
    )];

    let result = identify_meshes_with_decals(&mesh_materials).unwrap();

    assert!(result.is_empty());
}

#[test]
fn test_map_faces_to_vertices_single_triangle() {
    let corner_verts = vec![0, 1, 2, 3, 4, 5];  // Two triangles
    let face_indices = vec![0];  // First triangle
    let vertices = map_faces_to_vertices(&face_indices, &corner_verts, 3).unwrap();
    
    assert_eq!(vertices, vec![0, 1, 2]);
}

#[test]
fn test_map_faces_to_vertices_multiple_faces() {
    let corner_verts = vec![0, 1, 2, 2, 3, 4, 4, 5, 6];  // Three triangles
    let face_indices = vec![0, 2];  // First and third triangles
    let vertices = map_faces_to_vertices(&face_indices, &corner_verts, 3).unwrap();
    
    assert_eq!(vertices, vec![0, 1, 2, 4, 5, 6]);
}

#[test]
fn test_map_faces_to_vertices_shared_vertices() {
    let corner_verts = vec![0, 1, 2, 1, 2, 3];  // Two triangles sharing edge 1-2
    let face_indices = vec![0, 1];
    let vertices = map_faces_to_vertices(&face_indices, &corner_verts, 3).unwrap();
    
    assert_eq!(vertices, vec![0, 1, 2, 3]);
}

#[test]
fn test_map_faces_to_vertices_empty() {
    let corner_verts = vec![0, 1, 2];
    let face_indices = vec![];
    let vertices = map_faces_to_vertices(&face_indices, &corner_verts, 3).unwrap();
    
    assert_eq!(vertices.len(), 0);
}

#[test]
fn test_map_faces_to_vertices_out_of_bounds() {
    let corner_verts = vec![0, 1, 2];  // Only one triangle
    let face_indices = vec![1];  // Second triangle (doesn't exist)
    let result = map_faces_to_vertices(&face_indices, &corner_verts, 3);
    
    assert!(result.is_err());
}

#[test]
fn test_collect_decal_vertices_single_face() {
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "Mesh_001".to_string(),
        decal_materials: vec![
            DecalMaterial {
                material_name: "Decal_001".to_string(),
                is_decal: true,
                is_pom: false,
                material_index: 0,
            },
        ],
        decal_face_indices: vec![0],
    };
    
    let corner_verts = vec![0, 1, 2, 3, 4, 5];  // Two triangles
    let face_indices = vec![0];  // First triangle
    
    let result = collect_decal_vertices(&mesh_with_decals, &face_indices, &corner_verts, 3).unwrap();
    assert_eq!(result.mesh_name, "Mesh_001");
    assert_eq!(result.total_vertices, 6);
    assert_eq!(result.vertex_groups.len(), 1);
    assert_eq!(result.vertex_groups[0].name, "starbreaker_decal_offset");
    assert_eq!(result.vertex_groups[0].vertex_indices, vec![0, 1, 2]);
}

#[test]
fn test_collect_decal_vertices_multiple_faces() {
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "Mesh_002".to_string(),
        decal_materials: vec![
            DecalMaterial {
                material_name: "Decal_001".to_string(),
                is_decal: true,
                is_pom: false,
                material_index: 0,
            },
            DecalMaterial {
                material_name: "POM_001".to_string(),
                is_decal: false,
                is_pom: true,
                material_index: 2,
            },
        ],
        decal_face_indices: vec![],
    };
    
    let corner_verts = vec![
        0, 1, 2,      // Face 0
        2, 3, 4,      // Face 1 (shares edge with Face 0)
        4, 5, 6,      // Face 2
    ];
    let face_indices = vec![0, 2];  // First and third faces
    
    let result = collect_decal_vertices(&mesh_with_decals, &face_indices, &corner_verts, 3).unwrap();
    assert_eq!(result.vertex_groups[0].name, "starbreaker_decal_offset");
    // Should contain vertices from faces 0 and 2: {0, 1, 2, 4, 5, 6}
    assert_eq!(result.vertex_groups[0].vertex_indices.len(), 6);
}

#[test]
fn test_collect_decal_vertices_empty_faces() {
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "Mesh_003".to_string(),
        decal_materials: vec![],
        decal_face_indices: vec![],
    };
    
    let corner_verts = vec![0, 1, 2, 3, 4, 5];
    let face_indices = vec![];
    
    let result = collect_decal_vertices(&mesh_with_decals, &face_indices, &corner_verts, 3).unwrap();
    assert_eq!(result.vertex_groups[0].name, "starbreaker_decal_offset");
    assert_eq!(result.vertex_groups[0].vertex_indices.len(), 0);
}

#[test]
fn test_decal_face_indices_use_source_material_id() {
    let mesh = Mesh {
        positions: vec![[0.0, 0.0, 0.0]; 6],
        indices: vec![0, 1, 2, 3, 4, 5],
        uvs: None,
        secondary_uvs: None,
        normals: None,
        tangents: None,
        colors: None,
        submeshes: vec![SubMesh {
            material_name: None,
            material_id: 0,
            source_material_id: Some(2),
            first_index: 0,
            num_indices: 6,
            first_vertex: 0,
            num_vertices: 6,
            node_parent_index: 0,
        }],
        model_min: [0.0, 0.0, 0.0],
        model_max: [0.0, 0.0, 0.0],
        scaling_min: [0.0, 0.0, 0.0],
        scaling_max: [0.0, 0.0, 0.0],
        skin_weights: None,
    };
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "source_id_mesh".to_string(),
        decal_materials: vec![DecalMaterial {
            material_name: "decal".to_string(),
            is_decal: true,
            is_pom: false,
            material_index: 2,
        }],
        decal_face_indices: Vec::new(),
    };

    let faces = decal_face_indices_for_mesh(&mesh, &mesh_with_decals);

    assert_eq!(faces, vec![0, 1]);
}

#[test]
fn test_decal_face_indices_fall_back_to_submesh_material_name() {
    let mesh = Mesh {
        positions: vec![[0.0, 0.0, 0.0]; 3],
        indices: vec![0, 1, 2],
        uvs: None,
        secondary_uvs: None,
        normals: None,
        tangents: None,
        colors: None,
        submeshes: vec![SubMesh {
            material_name: Some("Decal_POM".to_string()),
            material_id: 7,
            source_material_id: None,
            first_index: 0,
            num_indices: 3,
            first_vertex: 0,
            num_vertices: 3,
            node_parent_index: 0,
        }],
        model_min: [0.0, 0.0, 0.0],
        model_max: [0.0, 0.0, 0.0],
        scaling_min: [0.0, 0.0, 0.0],
        scaling_max: [0.0, 0.0, 0.0],
        skin_weights: None,
    };
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "name_fallback_mesh".to_string(),
        decal_materials: vec![DecalMaterial {
            material_name: "Decal_POM".to_string(),
            is_decal: true,
            is_pom: true,
            material_index: 24,
        }],
        decal_face_indices: Vec::new(),
    };

    let faces = decal_face_indices_for_mesh(&mesh, &mesh_with_decals);

    assert_eq!(faces, vec![0]);
}

#[test]
fn test_decal_face_indices_prefers_material_name_over_mismatched_source_material_id() {
    let mesh = Mesh {
        positions: vec![[0.0, 0.0, 0.0]; 6],
        indices: vec![0, 1, 2, 3, 4, 5],
        uvs: None,
        secondary_uvs: None,
        normals: None,
        tangents: None,
        colors: None,
        submeshes: vec![
            SubMesh {
                material_name: Some("Damage_Internal_Structure".to_string()),
                material_id: 0,
                source_material_id: Some(2),
                first_index: 0,
                num_indices: 3,
                first_vertex: 0,
                num_vertices: 3,
                node_parent_index: 0,
            },
            SubMesh {
                material_name: Some("Decal_POM_A".to_string()),
                material_id: 1,
                source_material_id: Some(7),
                first_index: 3,
                num_indices: 3,
                first_vertex: 3,
                num_vertices: 3,
                node_parent_index: 0,
            },
        ],
        model_min: [0.0, 0.0, 0.0],
        model_max: [0.0, 0.0, 0.0],
        scaling_min: [0.0, 0.0, 0.0],
        scaling_max: [0.0, 0.0, 0.0],
        skin_weights: None,
    };
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "prefer_name_mesh".to_string(),
        decal_materials: vec![DecalMaterial {
            material_name: "Decal_POM_A".to_string(),
            is_decal: true,
            is_pom: true,
            material_index: 2,
        }],
        decal_face_indices: Vec::new(),
    };

    let faces = decal_face_indices_for_mesh(&mesh, &mesh_with_decals);

    assert_eq!(faces, vec![1]);
}

#[test]
fn test_decal_vertex_group_is_valid_for_all_decal_meshes() {
    let mesh_with_decals = MeshWithDecals {
        mesh_path: "all_decal".to_string(),
        decal_materials: vec![DecalMaterial {
            material_name: "decal".to_string(),
            is_decal: true,
            is_pom: false,
            material_index: 0,
        }],
        decal_face_indices: Vec::new(),
    };

    let result = collect_decal_vertices(&mesh_with_decals, &[], &[], 3).unwrap();

    assert_eq!(result.vertex_groups.len(), 1);
    assert_eq!(result.vertex_groups[0].name, "starbreaker_decal_offset");
    assert!(result.vertex_groups[0].vertex_indices.is_empty());
}

#[test]
fn test_validate_vertex_groups_valid() {
    let vgroups = vec![
        VertexGroup {
            name: "starbreaker_decal_offset".to_string(),
            vertex_indices: vec![0, 1, 2, 3],
            weights: None,
        },
    ];
    
    assert!(validate_vertex_groups(&vgroups, 10).is_ok());
}

#[test]
fn test_validate_vertex_groups_empty_name() {
    let vgroups = vec![
        VertexGroup {
            name: "".to_string(),
            vertex_indices: vec![0, 1],
            weights: None,
        },
    ];
    
    assert!(validate_vertex_groups(&vgroups, 10).is_err());
}

#[test]
fn test_validate_vertex_groups_out_of_bounds() {
    let vgroups = vec![
        VertexGroup {
            name: "Group".to_string(),
            vertex_indices: vec![0, 1, 15],  // 15 is out of bounds
            weights: None,
        },
    ];
    
    assert!(validate_vertex_groups(&vgroups, 10).is_err());
}

fn skin_test_nmc(names: &[&str]) -> NodeMeshCombo {
    let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    NodeMeshCombo {
        nodes: names
            .iter()
            .enumerate()
            .map(|(index, name)| crate::nmc::NmcNode {
                name: (*name).to_string(),
                parent_index: index.checked_sub(1).map(|parent| parent as u16),
                world_to_bone: identity,
                bone_to_world: identity,
                scale: [1.0; 3],
                geometry_type: 0,
                properties: Default::default(),
            })
            .collect(),
        material_indices: Vec::new(),
    }
}

/// A soft-skinned piston vertex blends top and bottom bones; the groups are
/// named after the skeleton nodes so the addon can bind them to bone objects.
#[test]
fn skin_vertex_groups_emit_one_weighted_group_per_joint() {
    use crate::ivo::skin::BoneMap12;
    let mut mesh = empty_anchor_mesh();
    mesh.positions = vec![[0.0; 3]; 3];
    mesh.skin_weights = Some(vec![
        BoneMap12 { joint_indices: [1, 2, 0, 0], weights: [166, 89, 0, 0] },
        BoneMap12 { joint_indices: [0; 4], weights: [0; 4] },
        BoneMap12 { joint_indices: [2, 2, 0, 0], weights: [128, 127, 0, 0] },
    ]);
    let groups = skin_vertex_groups(&mesh, &skin_test_nmc(&["Root", "PistonTop", "PistonBottom"]));

    let summary: Vec<(&str, &[usize], Vec<f32>)> = groups
        .iter()
        .map(|g| {
            let weights = g.weights.as_ref().unwrap().iter().map(|w| (w * 255.0).round()).collect();
            (g.name.as_str(), g.vertex_indices.as_slice(), weights)
        })
        .collect();
    assert_eq!(
        summary,
        [("PistonTop", &[0][..], vec![166.0]), ("PistonBottom", &[0, 2][..], vec![89.0, 255.0])]
    );
}

#[test]
fn skin_vertex_groups_are_empty_without_skin_weights() {
    let mesh = empty_anchor_mesh();
    assert!(skin_vertex_groups(&mesh, &skin_test_nmc(&["Root"])).is_empty());
}

#[test]
fn remap_vertex_groups_keeps_weights_with_their_vertices() {
    let groups = vec![VertexGroup {
        name: "PistonTop".to_string(),
        vertex_indices: vec![4, 2, 9],
        weights: Some(vec![0.25, 0.5, 0.75]),
    }];
    let old_to_new: HashMap<u32, u32> = [(4, 1), (2, 0)].into_iter().collect();

    let remapped = remap_vertex_groups(Some(&groups), &old_to_new).unwrap();
    assert_eq!(remapped[0].vertex_indices, [0, 1]);
    assert_eq!(remapped[0].weights.as_deref(), Some(&[0.5, 0.25][..]));
}
