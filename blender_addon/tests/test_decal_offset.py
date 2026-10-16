from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import types
import unittest


ADDON_ROOT = Path(__file__).resolve().parents[1]
STARBREAKER_ROOT = ADDON_ROOT.parent
REPO_ROOT = STARBREAKER_ROOT.parent
VULTURE_ALT_A = REPO_ROOT / "ships/Data/Objects/Spaceships/Ships/DRAK/Vulture/drak_vulture_alt_a_TEX0.materials.json"

sys.path.insert(0, str(ADDON_ROOT))


if "starbreaker_addon" not in sys.modules:
    package = types.ModuleType("starbreaker_addon")
    package.__path__ = [str(ADDON_ROOT / "starbreaker_addon")]
    sys.modules["starbreaker_addon"] = package

if "starbreaker_addon.runtime" not in sys.modules:
    runtime_package = types.ModuleType("starbreaker_addon.runtime")
    runtime_package.__path__ = [str(ADDON_ROOT / "starbreaker_addon" / "runtime")]
    sys.modules["starbreaker_addon.runtime"] = runtime_package

if "starbreaker_addon.runtime.importer" not in sys.modules:
    importer_package = types.ModuleType("starbreaker_addon.runtime.importer")
    importer_package.__path__ = [str(ADDON_ROOT / "starbreaker_addon" / "runtime" / "importer")]
    sys.modules["starbreaker_addon.runtime.importer"] = importer_package


if "mathutils" not in sys.modules:
    mathutils = types.ModuleType("mathutils")

    class Matrix(tuple):
        def __new__(cls, rows):
            return tuple.__new__(cls, rows)

        def inverted(self):
            return self

    class Quaternion(tuple):
        def __new__(cls, values):
            return tuple.__new__(cls, values)

    mathutils.Matrix = Matrix
    mathutils.Quaternion = Quaternion
    sys.modules["mathutils"] = mathutils


if "bpy" not in sys.modules:
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        Context=object,
        Material=object,
        NodeLinks=object,
        Nodes=object,
        Object=object,
        ShaderNodeTexImage=object,
    )
    bpy.data = types.SimpleNamespace(node_groups=[], images=[])
    sys.modules["bpy"] = bpy


from starbreaker_addon.runtime.constants import (
    PROP_DECAL_HOST_CHANNEL,
    PROP_DECAL_HOST_RGB,
    PROP_HAS_POM,
    PROP_MATERIAL_IDENTITY,
    PROP_MATERIAL_SIDECAR,
    PROP_PALETTE_SCOPE,
    PROP_SUBMATERIAL_JSON,
    PROP_TEMPLATE_KEY,
)
from starbreaker_addon.manifest import MaterialSidecar, SubmaterialRecord
from starbreaker_addon.runtime.importer.builders import (
    BuildersMixin,
    _mesh_decal_neutral_breakup_default,
    _parallax_height_sampler_extension,
)
from starbreaker_addon.runtime.importer.decals import DecalsMixin
from starbreaker_addon.runtime.importer.materials import MaterialsMixin
from starbreaker_addon.runtime.importer.materials import _material_datablock_is_valid
from starbreaker_addon.runtime.importer.orchestration import OrchestrationMixin
from starbreaker_addon.material_contract import ContractInput, ShaderGroupContract
from starbreaker_addon.runtime.importer.utils import (
    _canonical_material_sidecar_path,
    _material_identity,
    _scene_attachment_offset_to_blender,
)
from starbreaker_addon.templates import template_plan_for_submaterial


class FakeNodeTree:
    def __init__(self):
        self.nodes = []
        self.links = []


class FakeMaterial(dict):
    def __init__(self, name: str, **props):
        library = props.pop("library", None)
        super().__init__(props)
        self.name = name
        self.node_tree = FakeNodeTree()
        self.use_nodes = True
        self.library = library

    def copy(self):
        return FakeMaterial(self.name, library=None, **dict(self))


class FakeMaterialsCollection(dict):
    def __iter__(self):
        return iter(self.values())

    def get(self, name: str, default=None):
        return super().get(name, default)

    def new(self, name: str):
        material = FakeMaterial(name)
        self[name] = material
        return material


class FakeMatrixWorld:
    def __init__(self, translation: tuple[float, float, float]):
        self._rows = (
            (1.0, 0.0, 0.0, translation[0]),
            (0.0, 1.0, 0.0, translation[1]),
            (0.0, 0.0, 1.0, translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )

    def __getitem__(self, index: int):
        return self._rows[index]


class FakeSlot:
    def __init__(self, material):
        self.material = material


class FakePolygon:
    def __init__(self, material_index: int, vertices: list[int]):
        self.material_index = material_index
        self.vertices = vertices


class FakeMesh:
    def __init__(self, polygons: list[FakePolygon], vertex_count: int):
        self.polygons = polygons

    def as_pointer(self) -> int:
        return id(self)


class FakeObject:
    def __init__(self, material_slots: list[FakeSlot], mesh: FakeMesh, **props):
        self.name = props.pop("name", "FakeObject")
        self.type = "MESH"
        self.material_slots = material_slots
        self.data = mesh
        self._props = dict(props)

    def get(self, name: str, default=None):
        return self._props.get(name, default)


class ImporterUnderTest(BuildersMixin):
    def __init__(self, *, channel: str | None = None, fallback_rgb: tuple[float, float, float] | None = None):
        self.channel = channel
        self.fallback_rgb = fallback_rgb
        self.illum_rgb_calls: list[tuple[float, float, float]] = []
        self.illum_specular_calls: list[float | None] = []

    def _mesh_decal_host_channel_for_object(self, obj):
        return self.channel

    def _mesh_decal_host_rgb_for_object(self, obj):
        return self.fallback_rgb

    def _ensure_illum_pom_host_rgb_variant(self, material, rgb, host_specular=None):
        self.illum_rgb_calls.append(rgb)
        self.illum_specular_calls.append(host_specular)
        return FakeMaterial(f"{material.name}__host_rgb", **dict(material))


class HostRoutingImporterUnderTest(BuildersMixin):
    def __init__(self):
        self.host_channel_cache = {}
        self.host_rgb_cache = {}


class FakePackage:
    def __init__(self, has_decal_texture: bool):
        self.has_decal_texture = has_decal_texture

    def resolve_path(self, relative_path):
        if self.has_decal_texture and relative_path:
            return Path("/tmp") / Path(relative_path).name
        return None


class DecalDefaultsImporterUnderTest(DecalsMixin):
    def __init__(self, has_decal_texture: bool):
        self.package = FakePackage(has_decal_texture)


class MaterialReuseImporterUnderTest(MaterialsMixin):
    def __init__(self):
        self.material_cache = {}
        self.material_identity_index = {}
        self.material_identity_index_ready = False
        self.package = None
        self.package_root = None
        self.rebuild_calls: list[str] = []

    def _palette_scope(self, palette=None) -> str:
        return "test-scope"

    def _ensure_material_identity_index(self) -> None:
        self.material_identity_index_ready = True

    def _build_managed_material(
        self,
        material,
        sidecar_path,
        sidecar,
        submaterial,
        palette,
        material_identity,
    ) -> None:
        self.rebuild_calls.append(material.name)
        material[PROP_TEMPLATE_KEY] = template_plan_for_submaterial(submaterial).template_key
        material[PROP_MATERIAL_IDENTITY] = material_identity
        material[PROP_MATERIAL_SIDECAR] = _canonical_material_sidecar_path(sidecar_path, sidecar)
        material[PROP_SUBMATERIAL_JSON] = json.dumps(submaterial.raw, sort_keys=True)
        material[PROP_PALETTE_SCOPE] = self._palette_scope(palette)


class ManagedMaterialBuildImporterUnderTest(BuildersMixin):
    def __init__(self):
        self.build_calls: list[str] = []

    def _build_nodraw_material(self, material) -> None:
        self.build_calls.append("nodraw")

    def _build_illum_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("illum")

    def _build_hard_surface_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("hard_surface")

    def _group_contract_for_submaterial(self, submaterial):
        return None

    def _build_contract_group_material(self, material, submaterial, palette, plan, group_contract) -> bool:
        return False

    def _build_glass_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("glass")

    def _build_screen_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("screen")

    def _build_effect_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("effects")

    def _build_principled_material(self, material, submaterial, palette, plan) -> None:
        self.build_calls.append("principled")

    def _apply_material_node_layout(self, material) -> None:
        return None

    def _sweep_unreachable_nodes(self, material) -> None:
        return None

    def _palette_scope(self, palette=None) -> str:
        return "test-scope"


class FakeSidecar:
    def __init__(self, submaterials):
        self.submaterials = submaterials


class FakeSubmaterial:
    def __init__(self, index: int, name: str):
        self.index = index
        self.submaterial_name = name


class TestMeshDecalNeutralBreakupDefault(unittest.TestCase):
    def test_returns_white_for_stencil_mesh_decal_without_breakup_texture(self) -> None:
        submaterial = SubmaterialRecord.from_value(
            {
                "shader_family": "MeshDecal",
                "decoded_feature_flags": {"has_stencil_map": True, "tokens": ["STENCIL_MAP"]},
            }
        )
        group_contract = ShaderGroupContract(
            name="SB_MeshDecal_v1",
            shader_families=["MeshDecal"],
            version=1,
            shader_output="Shader",
            inputs=[],
            metadata={},
            raw={},
        )
        contract_input = ContractInput(
            name="TexSlot8_GrimeBreakup",
            socket_type="NodeSocketColor",
            semantic="grime_breakup",
            source_slot="TexSlot8",
            required=False,
            default_value=None,
            raw={},
        )

        self.assertEqual(
            _mesh_decal_neutral_breakup_default(
                group_contract,
                submaterial,
                contract_input,
                source_socket=None,
            ),
            (1.0, 1.0, 1.0, 1.0),
        )

    def test_returns_white_for_pom_mesh_decal_without_breakup_texture(self) -> None:
        submaterial = SubmaterialRecord.from_value(
            {
                "shader_family": "MeshDecal",
                "decoded_feature_flags": {
                    "has_parallax_occlusion_mapping": True,
                    "tokens": ["PARALLAX_OCCLUSION_MAPPING"],
                },
            }
        )
        group_contract = ShaderGroupContract(
            name="SB_MeshDecal_v1",
            shader_families=["MeshDecal"],
            version=1,
            shader_output="Shader",
            inputs=[],
            metadata={},
            raw={},
        )
        contract_input = ContractInput(
            name="TexSlot8_GrimeBreakup",
            socket_type="NodeSocketColor",
            semantic="grime_breakup",
            source_slot="TexSlot8",
            required=False,
            default_value=None,
            raw={},
        )

        self.assertEqual(
            _mesh_decal_neutral_breakup_default(
                group_contract,
                submaterial,
                contract_input,
                source_socket=None,
            ),
            (1.0, 1.0, 1.0, 1.0),
        )

    def test_returns_none_when_breakup_texture_is_present(self) -> None:
        submaterial = SubmaterialRecord.from_value(
            {
                "shader_family": "MeshDecal",
                "decoded_feature_flags": {"has_stencil_map": True, "tokens": ["STENCIL_MAP"]},
            }
        )
        group_contract = ShaderGroupContract(
            name="SB_MeshDecal_v1",
            shader_families=["MeshDecal"],
            version=1,
            shader_output="Shader",
            inputs=[],
            metadata={},
            raw={},
        )
        contract_input = ContractInput(
            name="TexSlot8_GrimeBreakup",
            socket_type="NodeSocketColor",
            semantic="grime_breakup",
            source_slot="TexSlot8",
            required=False,
            default_value=None,
            raw={},
        )

        self.assertIsNone(
            _mesh_decal_neutral_breakup_default(
                group_contract,
                submaterial,
                contract_input,
                source_socket=object(),
            )
        )


class FakePackageWithSidecars:
    def __init__(self, sidecar):
        self.sidecar = sidecar
        self.palettes = {}
        self.scene = types.SimpleNamespace(root_entity=types.SimpleNamespace(palette_id=None))

    def load_material_sidecar(self, sidecar_path):
        return self.sidecar


class OrchestrationImporterUnderTest(OrchestrationMixin, BuildersMixin):
    def __init__(self, sidecar):
        self.package = FakePackageWithSidecars(sidecar)
        self.package_root = None
        self.import_palette_override = None
        self.import_paint_variant_sidecar = None
        self.exterior_material_sidecars = None
        self.mesh_polygon_counts_cache = {}
        self.slot_mapping_cache = {}
        self.sidecar_submaterials_by_index = {}
        self.sidecar_submaterials_by_name = {}
        self.sidecar_submaterials_by_name_all = {}

    def _ensure_runtime_shared_groups(self) -> None:
        return None

    def _effective_palette_id(self, palette_id: str | None) -> str | None:
        return palette_id

    def material_for_submaterial(self, sidecar_path, sidecar, submaterial, palette):
        return FakeMaterial(f"material_{submaterial.index}")

    def _remove_replaced_slot_material(self, material) -> None:
        return None

    def _rebind_mesh_decal_for_host(self, obj, palette, **_kwargs) -> int:
        return 0


class OrphanRemovalImporterUnderTest(OrchestrationMixin):
    def __init__(self):
        self._pending_orphan_materials = set()


class FakeHashableMaterial:
    library = None

    def __init__(self):
        self.users = 0

    def get(self, _key, default=None):
        return default


class FakeInvalidMaterial:
    @property
    def name(self):
        raise ReferenceError("StructRNA of type Material has been removed")


class DecalOffsetTests(unittest.TestCase):
    def test_invalid_material_datablock_is_detected(self) -> None:
        self.assertFalse(_material_datablock_is_valid(FakeInvalidMaterial()))

    def test_replaced_slot_material_cleanup_is_deferred(self) -> None:
        importer = OrphanRemovalImporterUnderTest()
        material = FakeHashableMaterial()

        importer._remove_replaced_slot_material(material)

        self.assertIn(material, importer._pending_orphan_materials)

    def test_rebuild_object_materials_skips_empty_unmapped_slots_without_warning(self) -> None:
        sidecar = FakeSidecar([
            FakeSubmaterial(0, "decal pom"),
            FakeSubmaterial(1, "tint_secondary"),
        ])
        importer = OrchestrationImporterUnderTest(sidecar)
        mesh = FakeMesh(polygons=[], vertex_count=0)
        importer.slot_mapping_cache[mesh.as_pointer()] = [0, 1, None, None]
        obj = FakeObject(
            material_slots=[FakeSlot(None), FakeSlot(None), FakeSlot(None), FakeSlot(None)],
            mesh=mesh,
            name="flair_poster_hook_mesh_005",
            starbreaker_material_sidecar="Data/Objects/props/flair/poster/flair_poster_1_a_TEX0.materials.json",
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            applied = importer.rebuild_object_materials(obj, None)

        self.assertEqual(applied, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIsNotNone(obj.material_slots[0].material)
        self.assertIsNotNone(obj.material_slots[1].material)
        self.assertIsNone(obj.material_slots[2].material)
        self.assertIsNone(obj.material_slots[3].material)

    def test_illum_pom_rebind_uses_palette_channel_rgb_when_no_authored_fallback_exists(self) -> None:
        decal = FakeMaterial(
            "drak_vulture:pom_decals",
            starbreaker_shader_family="Illum",
            **{
                PROP_HAS_POM: True,
                PROP_TEMPLATE_KEY: "decal_stencil",
            },
        )
        obj = FakeObject(
            material_slots=[FakeSlot(decal)],
            mesh=FakeMesh(polygons=[], vertex_count=0),
        )
        palette = types.SimpleNamespace(
            primary=(0.2, 0.3, 0.4),
            secondary=(0.5, 0.6, 0.7),
            tertiary=(0.8, 0.1, 0.2),
            glass=(0.9, 0.9, 0.95),
            raw={"finish": {"primary": {"specular": [0.03, 0.04, 0.05]}}},
        )
        importer = ImporterUnderTest(channel="primary", fallback_rgb=None)

        rebound = importer._rebind_mesh_decal_for_host(obj, palette)

        self.assertEqual(rebound, 1)
        self.assertEqual(importer.illum_rgb_calls, [palette.primary])
        self.assertAlmostEqual(importer.illum_specular_calls[0], 0.04)
        self.assertEqual(obj.material_slots[0].material.name, "drak_vulture:pom_decals__host_rgb")

    def test_illum_pom_rebind_prefers_palette_channel_rgb_over_fallback_rgb(self) -> None:
        decal = FakeMaterial(
            "drak_vulture:pom_decals",
            starbreaker_shader_family="Illum",
            **{
                PROP_HAS_POM: True,
                PROP_TEMPLATE_KEY: "decal_stencil",
            },
        )
        obj = FakeObject(
            material_slots=[FakeSlot(decal)],
            mesh=FakeMesh(polygons=[], vertex_count=0),
        )
        palette = types.SimpleNamespace(
            primary=(0.85, 0.72, 0.12),
            secondary=(0.5, 0.6, 0.7),
            tertiary=(0.8, 0.1, 0.2),
            glass=(0.9, 0.9, 0.95),
            raw={},
        )
        importer = ImporterUnderTest(
            channel="primary",
            fallback_rgb=(0.0627, 0.0627, 0.0627),
        )

        rebound = importer._rebind_mesh_decal_for_host(obj, palette)

        self.assertEqual(rebound, 1)
        self.assertEqual(importer.illum_rgb_calls, [palette.primary])
        self.assertEqual(obj.material_slots[0].material.name, "drak_vulture:pom_decals__host_rgb")

    def test_mesh_decal_host_variant_names_do_not_chain_existing_host_suffixes(self) -> None:
        bpy = sys.modules["bpy"]
        original_materials = getattr(bpy.data, "materials", None)
        materials = FakeMaterialsCollection()
        bpy.data.materials = materials
        try:
            decal = FakeMaterial(
                "drak_clipper_ext:Decal_POM_A#03e817cc__host_secondary",
                starbreaker_shader_family="MeshDecal",
                **{PROP_HAS_POM: True},
            )
            importer = ImporterUnderTest(channel="tertiary")
            palette = types.SimpleNamespace(
                primary=(0.2, 0.3, 0.4),
                secondary=(0.5, 0.6, 0.7),
                tertiary=(0.8, 0.1, 0.2),
                glass=(0.9, 0.9, 0.95),
            )

            variant = importer._ensure_mesh_decal_host_variant(decal, "tertiary", palette)

            self.assertEqual(
                variant.name,
                "drak_clipper_ext:Decal_POM_A#03e817cc__host_tertiary",
            )
        finally:
            bpy.data.materials = original_materials

    def test_host_channel_scan_returns_single_channel_without_polygon_counts(self) -> None:
        obj = FakeObject(
            material_slots=[FakeSlot(FakeMaterial("drak_clipper_ext_Paint_Secondary"))],
            mesh=FakeMesh(polygons=[], vertex_count=0),
        )
        importer = ImporterUnderTest()

        self.assertEqual(importer._scan_slots_for_host_channel(obj), "secondary")

    def test_host_channel_scan_uses_polygon_weight_when_channels_compete(self) -> None:
        obj = FakeObject(
            material_slots=[
                FakeSlot(FakeMaterial("drak_clipper_ext_Paint_Primary")),
                FakeSlot(FakeMaterial("drak_clipper_ext_Paint_Tertiary")),
            ],
            mesh=FakeMesh(
                polygons=[
                    FakePolygon(0, [0, 1, 2]),
                    FakePolygon(1, [3, 4, 5]),
                    FakePolygon(1, [6, 7, 8]),
                ],
                vertex_count=9,
            ),
        )
        importer = ImporterUnderTest()

        self.assertEqual(importer._scan_slots_for_host_channel(obj), "tertiary")

    def test_mesh_decal_host_channel_prefers_precomputed_object_property(self) -> None:
        obj = FakeObject(
            material_slots=[],
            mesh=FakeMesh(polygons=[], vertex_count=0),
            **{PROP_DECAL_HOST_CHANNEL: "glass"},
        )
        importer = HostRoutingImporterUnderTest()

        self.assertEqual(importer._mesh_decal_host_channel_for_object(obj), "glass")

    def test_mesh_decal_host_channel_falls_back_to_parent_precomputed_property(self) -> None:
        parent = FakeObject(
            material_slots=[],
            mesh=FakeMesh(polygons=[], vertex_count=0),
            **{PROP_DECAL_HOST_CHANNEL: "secondary"},
        )
        obj = FakeObject(
            material_slots=[],
            mesh=FakeMesh(polygons=[], vertex_count=0),
        )
        obj.parent = parent
        importer = HostRoutingImporterUnderTest()

        self.assertEqual(importer._mesh_decal_host_channel_for_object(obj), "secondary")

    def test_mesh_decal_host_rgb_prefers_precomputed_object_property(self) -> None:
        obj = FakeObject(
            material_slots=[],
            mesh=FakeMesh(polygons=[], vertex_count=0),
            **{PROP_DECAL_HOST_RGB: [0.2, 0.4, 0.6]},
        )
        importer = HostRoutingImporterUnderTest()

        self.assertEqual(importer._mesh_decal_host_rgb_for_object(obj), (0.2, 0.4, 0.6))

    def test_derive_decal_host_route_from_submaterials_uses_polygon_weight(self) -> None:
        obj = FakeObject(
            material_slots=[],
            mesh=FakeMesh(
                polygons=[
                    FakePolygon(0, [0, 1, 2]),
                    FakePolygon(1, [3, 4, 5]),
                    FakePolygon(1, [6, 7, 8]),
                ],
                vertex_count=9,
            ),
        )
        importer = HostRoutingImporterUnderTest()
        slot_submaterials = [
            SubmaterialRecord.from_value(
                {
                    "shader_family": "HardSurface",
                    "palette_routing": {"material_channel": {"name": "primary"}},
                }
            ),
            SubmaterialRecord.from_value(
                {
                    "shader_family": "HardSurface",
                    "palette_routing": {"material_channel": {"name": "tertiary"}},
                }
            ),
        ]

        channel, rgb = importer._derive_decal_host_route_from_submaterials(obj, slot_submaterials)

        self.assertEqual(channel, "tertiary")
        self.assertIsNone(rgb)

    def test_derive_decal_host_route_from_submaterials_falls_back_to_authored_tint(self) -> None:
        obj = FakeObject(
            material_slots=[],
            mesh=FakeMesh(polygons=[FakePolygon(0, [0, 1, 2])], vertex_count=3),
        )
        importer = HostRoutingImporterUnderTest()
        slot_submaterials = [
            SubmaterialRecord.from_value(
                {
                    "shader_family": "HardSurface",
                    "layer_manifest": [
                        {
                            "tint_color": [0.2, 0.4, 0.6],
                        }
                    ],
                }
            )
        ]

        channel, rgb = importer._derive_decal_host_route_from_submaterials(obj, slot_submaterials)

        self.assertIsNone(channel)
        self.assertEqual(rgb, (0.2, 0.4, 0.6))

    def test_parallax_bias_value_prefers_authored_height_bias(self) -> None:
        importer = ImporterUnderTest()
        submaterial = SubmaterialRecord.from_value(
            {
                "public_params": {
                    "HeightBias": 0.75,
                    "PomDisplacement": 0.04,
                }
            }
        )

        self.assertAlmostEqual(importer._parallax_bias_value(submaterial), 0.75)

    def test_parallax_height_sampler_extension_clips_default_uv_range(self) -> None:
        self.assertEqual(_parallax_height_sampler_extension(1.0), "CLIP")
        self.assertEqual(_parallax_height_sampler_extension(0.75), "CLIP")

    def test_parallax_height_sampler_extension_repeats_explicit_tiling(self) -> None:
        self.assertEqual(_parallax_height_sampler_extension(3.0), "REPEAT")

    def test_missing_mesh_decal_texture_defaults_alpha_to_zero(self) -> None:
        submaterial = SubmaterialRecord.from_value({"shader_family": "MeshDecal"})
        palette = types.SimpleNamespace(decal_texture=None)
        importer = DecalDefaultsImporterUnderTest(has_decal_texture=False)

        _, alpha = importer._virtual_tint_palette_decal_defaults(
            submaterial,
            palette,
            has_decal_texture=importer._has_palette_decal_texture(palette),
        )

        self.assertEqual(alpha, 0.0)

    def test_missing_stencil_map_texture_defaults_alpha_to_zero(self) -> None:
        submaterial = SubmaterialRecord.from_value(
            {
                "shader_family": "LayerBlend_V2",
                "decoded_feature_flags": {"has_stencil_map": True},
            }
        )
        palette = types.SimpleNamespace(decal_texture=None)
        importer = DecalDefaultsImporterUnderTest(has_decal_texture=False)

        _, alpha = importer._virtual_tint_palette_decal_defaults(
            submaterial,
            palette,
            has_decal_texture=importer._has_palette_decal_texture(palette),
        )

        self.assertEqual(alpha, 0.0)

    def test_mesh_decal_with_texture_keeps_existing_default_alpha(self) -> None:
        submaterial = SubmaterialRecord.from_value({"shader_family": "MeshDecal"})
        palette = types.SimpleNamespace(decal_texture="Data/Textures/paint/decal.png")
        importer = DecalDefaultsImporterUnderTest(has_decal_texture=True)

        _, alpha = importer._virtual_tint_palette_decal_defaults(
            submaterial,
            palette,
            has_decal_texture=importer._has_palette_decal_texture(palette),
        )

        self.assertEqual(alpha, 0.85)


class MaterialReuseTests(unittest.TestCase):
    @unittest.skipUnless(
        VULTURE_ALT_A.is_file(),
        "Vulture fixtures not present; skipping material reuse regression test",
    )
    def test_stale_template_key_forces_managed_material_rebuild(self) -> None:
        sidecar = MaterialSidecar.from_file(VULTURE_ALT_A)
        submaterial = next(
            candidate
            for candidate in sidecar.submaterials
            if candidate.submaterial_name == "livery_decal"
        )
        self.assertEqual(template_plan_for_submaterial(submaterial).template_key, "nodraw")

        bpy = sys.modules["bpy"]
        original_materials = getattr(bpy.data, "materials", None)
        materials = FakeMaterialsCollection()
        bpy.data.materials = materials
        try:
            sidecar_path = _canonical_material_sidecar_path("", sidecar)
            palette_scope = "test-scope"
            material_identity = _material_identity(sidecar_path, sidecar, submaterial, None, palette_scope)
            stale = FakeMaterial(
                submaterial.blender_material_name or "DRAK_Vulture:livery_decal",
                **{
                    PROP_TEMPLATE_KEY: "physical_surface",
                    PROP_MATERIAL_IDENTITY: material_identity,
                    PROP_MATERIAL_SIDECAR: sidecar_path,
                    PROP_SUBMATERIAL_JSON: json.dumps(submaterial.raw, sort_keys=True),
                    PROP_PALETTE_SCOPE: palette_scope,
                },
            )
            materials[stale.name] = stale

            importer = MaterialReuseImporterUnderTest()
            material = importer.material_for_submaterial(sidecar_path, sidecar, submaterial, None)

            self.assertIs(material, stale)
            self.assertEqual(importer.rebuild_calls, [stale.name])
            self.assertEqual(material[PROP_TEMPLATE_KEY], "nodraw")
        finally:
            bpy.data.materials = original_materials

    @unittest.skipUnless(
        VULTURE_ALT_A.is_file(),
        "Vulture fixtures not present; skipping material localization regression test",
    )
    def test_linked_reusable_material_is_copied_before_rebuild(self) -> None:
        sidecar = MaterialSidecar.from_file(VULTURE_ALT_A)
        submaterial = next(
            candidate
            for candidate in sidecar.submaterials
            if candidate.submaterial_name == "livery_decal"
        )

        bpy = sys.modules["bpy"]
        original_materials = getattr(bpy.data, "materials", None)
        materials = FakeMaterialsCollection()
        bpy.data.materials = materials
        try:
            sidecar_path = _canonical_material_sidecar_path("", sidecar)
            palette_scope = "test-scope"
            material_identity = _material_identity(sidecar_path, sidecar, submaterial, None, palette_scope)
            linked = FakeMaterial(
                submaterial.blender_material_name or "DRAK_Vulture:livery_decal",
                library=object(),
                **{
                    PROP_TEMPLATE_KEY: "wrong-template",
                    PROP_MATERIAL_IDENTITY: material_identity,
                    PROP_MATERIAL_SIDECAR: sidecar_path,
                    PROP_SUBMATERIAL_JSON: json.dumps(submaterial.raw, sort_keys=True),
                    PROP_PALETTE_SCOPE: palette_scope,
                },
            )
            materials[linked.name] = linked

            importer = MaterialReuseImporterUnderTest()
            material = importer.material_for_submaterial(sidecar_path, sidecar, submaterial, None)
            second = importer.material_for_submaterial(sidecar_path, sidecar, submaterial, None)

            self.assertIsNot(material, linked)
            self.assertIs(second, material)
            self.assertIsNone(material.library)
            self.assertEqual(material[PROP_MATERIAL_IDENTITY], material_identity)
            self.assertEqual(importer.rebuild_calls, [material.name])
        finally:
            bpy.data.materials = original_materials

    @unittest.skipUnless(
        VULTURE_ALT_A.is_file(),
        "Vulture fixtures not present; skipping managed material dispatch regression test",
    )
    def test_illum_nodraw_submaterial_uses_nodraw_builder(self) -> None:
        sidecar = MaterialSidecar.from_file(VULTURE_ALT_A)
        submaterial = next(
            candidate
            for candidate in sidecar.submaterials
            if candidate.submaterial_name == "livery_decal"
        )
        importer = ManagedMaterialBuildImporterUnderTest()
        material = FakeMaterial(submaterial.blender_material_name or "DRAK_Vulture:livery_decal")

        importer._build_managed_material(
            material,
            _canonical_material_sidecar_path("", sidecar),
            sidecar,
            submaterial,
            None,
            "identity",
        )

        self.assertEqual(importer.build_calls, ["nodraw"])
        self.assertEqual(material[PROP_TEMPLATE_KEY], "nodraw")


class SceneAttachmentOffsetTests(unittest.TestCase):
    def test_duplicate_no_rotation_helper_offset_is_suppressed(self) -> None:
        location = _scene_attachment_offset_to_blender(
            (0.0, -1.2599999904632568, 3.371000051498413),
            (0.0, 0.0, 0.0),
            no_rotation=True,
            parent_world_matrix=(
                (1.0000001192092896, 4.76837158203125e-7, 1.7484583736404602e-7, 0.0002016690996242687),
                (-1.7484549630353285e-7, -7.085781135174329e-7, 0.9999999403953552, -1.2602757215499878),
                (4.768372150465439e-7, -1.0000001192092896, -9.088388424061122e-7, 3.3709583282470703),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

        self.assertEqual(location, (0.0, 0.0, 0.0))

    def test_nonzero_rotation_no_rotation_attachment_keeps_offset(self) -> None:
        location = _scene_attachment_offset_to_blender(
            (0.9120000004768372, -1.2000000476837158, 1.0),
            (0.0, 0.0, 45.0),
            no_rotation=True,
            parent_world_matrix=(
                (1.0, 0.0, 0.0, 0.9120000004768372),
                (0.0, 1.0, 0.0, -1.2000000476837158),
                (0.0, 0.0, 1.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

        self.assertEqual(location, (0.9120000004768372, -1.0, -1.2000000476837158))


if __name__ == "__main__":
    unittest.main()
