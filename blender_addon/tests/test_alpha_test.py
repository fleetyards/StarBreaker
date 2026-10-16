"""Alpha-tested (cut-out) surfaces: authored ``AlphaTest`` becomes a hard clip."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))

from starbreaker_addon.manifest import SubmaterialRecord  # noqa: E402
from starbreaker_addon.templates import alpha_test_threshold, template_plan_for_submaterial  # noqa: E402


def _submaterial(shader_family: str, string_gen_mask: str, alpha_test: str | None) -> SubmaterialRecord:
    attributes = [{"name": "StringGenMask", "value": string_gen_mask}]
    if alpha_test is not None:
        attributes.append({"name": "AlphaTest", "value": alpha_test})
    tokens = [token for token in string_gen_mask.split("%") if token]
    return SubmaterialRecord.from_value(
        {
            "shader_family": shader_family,
            "activation_state": {"state": "active", "reason": "visible"},
            "authored_attributes": attributes,
            "decoded_feature_flags": {
                "tokens": tokens,
                "has_decal": "DECAL" in tokens,
                "has_parallax_occlusion_mapping": "PARALLAX_OCCLUSION_MAPPING" in tokens,
                "has_stencil_map": "STENCIL_MAP" in tokens,
            },
        }
    )


# Star Runner Ext_Grate: a POM floor grate cut out at alpha 0.61.
_GRATE_MASK = "%NORMAL_MAP%SPECULAR_MAP%USE_DAMAGE_MAP%VERTCOLORS%PARALLAX_OCCLUSION_MAPPING"


class AlphaTestTests(unittest.TestCase):
    def test_alpha_tested_pom_surface_clips_instead_of_blending(self) -> None:
        grate = _submaterial("Illum", _GRATE_MASK, "0.61000001")
        self.assertAlmostEqual(alpha_test_threshold(grate), 0.61, places=5)
        plan = template_plan_for_submaterial(grate)
        self.assertEqual(plan.template_key, "parallax_pom")
        self.assertEqual(plan.blend_method, "CLIP")
        self.assertTrue(plan.uses_alpha)

    def test_opaque_pom_surface_renders_opaque(self) -> None:
        # Star Runner Ext_Damage_Pom: an opaque POM hull panel showing
        # structure. Alpha-blending it by its diffuse alpha made the hull
        # see-through into the ship.
        damage = _submaterial("HardSurface", "%PARALLAX_OCCLUSION_MAPPING%USE_SPECULAR_MAPS", None)
        plan = template_plan_for_submaterial(damage)
        self.assertEqual(plan.template_key, "parallax_pom")
        self.assertEqual(plan.blend_method, "OPAQUE")
        self.assertFalse(plan.uses_alpha)

    def test_pom_decals_keep_their_alpha(self) -> None:
        pom_decal = _submaterial("Illum", "%DECAL%PARALLAX_OCCLUSION_MAPPING", None)
        plan = template_plan_for_submaterial(pom_decal)
        self.assertEqual(plan.template_key, "decal_stencil")
        self.assertTrue(plan.uses_alpha)

    def test_translucent_pom_surface_keeps_blending(self) -> None:
        translucent = _submaterial("Illum", _GRATE_MASK, None)
        translucent.raw["authored_attributes"].append({"name": "Opacity", "value": "0.5"})
        self.assertEqual(template_plan_for_submaterial(translucent).blend_method, "BLEND")

    def test_decals_keep_blending_over_their_host(self) -> None:
        pom_decal = _submaterial("Illum", "%DECAL%PARALLAX_OCCLUSION_MAPPING", "0.5")
        mesh_decal = _submaterial("MeshDecal", "%DIFFUSE_MAP", "0.17")
        stencil = _submaterial("Illum", "%STENCIL_MAP", "0.3")
        for submaterial in (pom_decal, mesh_decal, stencil):
            self.assertIsNone(alpha_test_threshold(submaterial))

    def test_missing_or_malformed_alpha_test_is_ignored(self) -> None:
        self.assertIsNone(alpha_test_threshold(_submaterial("Illum", _GRATE_MASK, None)))
        self.assertIsNone(alpha_test_threshold(_submaterial("Illum", _GRATE_MASK, "n/a")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
