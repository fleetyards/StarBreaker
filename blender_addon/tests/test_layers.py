from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


ADDON_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ADDON_ROOT))


if "starbreaker_addon" not in sys.modules:
    package = types.ModuleType("starbreaker_addon")
    package.__path__ = [str(ADDON_ROOT / "starbreaker_addon")]
    sys.modules["starbreaker_addon"] = package

if "starbreaker_addon.runtime" not in sys.modules:
    runtime_package = types.ModuleType("starbreaker_addon.runtime")
    runtime_package.__path__ = [str(ADDON_ROOT / "starbreaker_addon" / "runtime")]
    sys.modules["starbreaker_addon.runtime"] = runtime_package

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

    class Euler(tuple):
        def __new__(cls, values, order='XYZ'):
            return tuple.__new__(cls, values)

    mathutils.Matrix = Matrix
    mathutils.Quaternion = Quaternion
    mathutils.Euler = Euler
    sys.modules["mathutils"] = mathutils

if "bpy" not in sys.modules:
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Nodes=object, NodeLinks=object, Node=object)
    sys.modules["bpy"] = bpy


from starbreaker_addon.runtime.importer.builders import (
    _clamp_unit_float,
    _layered_wear_base_layer,
    _layered_wear_first_diffuse_layer,
    _layered_wear_metallic_values,
    _layered_wear_first_non_neutral_tint,
)
from starbreaker_addon.runtime.importer.layers import _detail_strength_or_zero, _stencil_override_selection


class LayerDetailTests(unittest.TestCase):
    def test_missing_detail_mask_forces_neutral_strength(self) -> None:
        self.assertEqual(_detail_strength_or_zero(1.0, None), 0.0)
        self.assertEqual(_detail_strength_or_zero(0.296667, None), 0.0)

    def test_present_detail_mask_preserves_authored_strength(self) -> None:
        self.assertEqual(_detail_strength_or_zero(1.0, object()), 1.0)
        self.assertEqual(_detail_strength_or_zero(0.296667, object()), 0.296667)

    def test_single_tint_override_selects_requested_slot(self) -> None:
        tint, specular, color_enable, tone_mode = _stencil_override_selection(
            2.0,
            is_virtual=False,
            tint_1=(1.0, 0.0, 0.0),
            tint_2=(0.0, 1.0, 0.0),
            tint_3=(0.0, 0.0, 1.0),
            specular_1=None,
            specular_2=(0.2, 0.2, 0.2),
            specular_3=None,
            stencil_glossiness=0.5,
        )
        self.assertEqual(tint, (0.0, 1.0, 0.0))
        self.assertEqual(specular, (0.2, 0.2, 0.2))
        self.assertEqual(color_enable, 1.0)
        self.assertEqual(tone_mode, 0.0)

    def test_neutral_non_virtual_override_can_disable_diffuse_color(self) -> None:
        tint, specular, color_enable, tone_mode = _stencil_override_selection(
            2.0,
            is_virtual=False,
            tint_1=(1.0, 1.0, 1.0),
            tint_2=(1.0, 1.0, 1.0),
            tint_3=(1.0, 1.0, 1.0),
            specular_1=None,
            specular_2=None,
            specular_3=None,
            stencil_glossiness=None,
        )
        self.assertEqual(tint, (1.0, 1.0, 1.0))
        self.assertIsNone(specular)
        self.assertEqual(color_enable, 0.0)
        self.assertEqual(tone_mode, 1.0)

    def test_virtual_override_keeps_diffuse_color_enabled(self) -> None:
        tint, specular, color_enable, tone_mode = _stencil_override_selection(
            2.0,
            is_virtual=True,
            tint_1=(1.0, 1.0, 1.0),
            tint_2=(1.0, 1.0, 1.0),
            tint_3=(1.0, 1.0, 1.0),
            specular_1=None,
            specular_2=None,
            specular_3=None,
            stencil_glossiness=None,
        )
        self.assertEqual(tint, (1.0, 1.0, 1.0))
        self.assertIsNone(specular)
        self.assertEqual(color_enable, 1.0)
        self.assertEqual(tone_mode, 0.0)

    def test_layered_wear_base_layer_returns_none_for_empty_manifest(self) -> None:
        submaterial = types.SimpleNamespace(layer_manifest=[])
        self.assertIsNone(_layered_wear_base_layer(submaterial))

    def test_layered_wear_base_layer_uses_first_manifest_entry(self) -> None:
        first = object()
        second = object()
        submaterial = types.SimpleNamespace(layer_manifest=[first, second])
        self.assertIs(_layered_wear_base_layer(submaterial), first)

    def test_layered_wear_base_layer_uses_single_layer_manifest_entry(self) -> None:
        only = object()
        submaterial = types.SimpleNamespace(layer_manifest=[only])
        self.assertIs(_layered_wear_base_layer(submaterial), only)

    def test_layered_wear_first_diffuse_layer_returns_first_with_diffuse(self) -> None:
        layer_a = types.SimpleNamespace(diffuse_export_path=None)
        layer_b = types.SimpleNamespace(diffuse_export_path="Data/foo.dds")
        layer_c = types.SimpleNamespace(diffuse_export_path="Data/bar.dds")
        submaterial = types.SimpleNamespace(layer_manifest=[layer_a, layer_b, layer_c])
        self.assertIs(_layered_wear_first_diffuse_layer(submaterial), layer_b)

    def test_layered_wear_first_non_neutral_tint_prefers_first_non_white(self) -> None:
        layer_a = types.SimpleNamespace(tint_color=(1.0, 1.0, 1.0))
        layer_b = types.SimpleNamespace(tint_color=(0.3, 0.4, 0.5))
        layer_c = types.SimpleNamespace(tint_color=(0.8, 0.7, 0.6))
        submaterial = types.SimpleNamespace(layer_manifest=[layer_a, layer_b, layer_c])
        self.assertEqual(
            _layered_wear_first_non_neutral_tint(submaterial),
            (0.3, 0.4, 0.5),
        )

    def test_layered_wear_first_non_neutral_tint_skips_wear_layers(self) -> None:
        # The Gallant's bare metal: white base, black wear layer.
        base = types.SimpleNamespace(name="BaseLayer1", tint_color=(1.0, 1.0, 1.0))
        wear = types.SimpleNamespace(name="WearLayer1", tint_color=(0.0, 0.0, 0.0))
        submaterial = types.SimpleNamespace(layer_manifest=[base, wear])
        self.assertIsNone(_layered_wear_first_non_neutral_tint(submaterial))

    def test_layered_wear_first_non_neutral_tint_returns_none_when_neutral(self) -> None:
        layer_a = types.SimpleNamespace(tint_color=(1.0, 1.0, 1.0))
        layer_b = types.SimpleNamespace(tint_color=None)
        submaterial = types.SimpleNamespace(layer_manifest=[layer_a, layer_b])
        self.assertIsNone(_layered_wear_first_non_neutral_tint(submaterial))

    def test_clamp_unit_float(self) -> None:
        self.assertEqual(_clamp_unit_float(-0.5), 0.0)
        self.assertEqual(_clamp_unit_float(0.25), 0.25)
        self.assertEqual(_clamp_unit_float(2.0), 1.0)

    def test_layered_wear_metallic_values_uses_both_layers(self) -> None:
        base_layer = types.SimpleNamespace(layer_snapshot={"metallic": 0.8})
        wear_layer = types.SimpleNamespace(layer_snapshot={"metallic": 0.2})
        self.assertEqual(
            _layered_wear_metallic_values(base_layer, wear_layer),
            (0.8, 0.2),
        )

    def test_layered_wear_metallic_values_falls_back_to_present_layer(self) -> None:
        wear_layer = types.SimpleNamespace(layer_snapshot={"metallic": 1.5})
        self.assertEqual(
            _layered_wear_metallic_values(None, wear_layer),
            (1.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
