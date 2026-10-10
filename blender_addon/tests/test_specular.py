from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


ADDON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ADDON_ROOT))


for package_name in (
    "starbreaker_addon",
    "starbreaker_addon.runtime",
    "starbreaker_addon.runtime.importer",
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ADDON_ROOT / Path(*package_name.split(".")))]
        sys.modules[package_name] = package


from starbreaker_addon.runtime.importer.specular import (
    reflectance_to_specular_ior_level,
    specular_ior_level_scale,
)


class SpecularIorLevelTests(unittest.TestCase):
    def test_dielectric_reflectance_maps_to_neutral_level(self) -> None:
        # F0 0.04 is what an IOR 1.5 dielectric reflects with no adjustment.
        self.assertAlmostEqual(reflectance_to_specular_ior_level(0.04), 0.5)

    def test_palette_spec_colours_keep_their_reflectance(self) -> None:
        # Constellation Mk5 Centaurus: spec 59/255 sRGB is ordinary paint,
        # spec 117/255 sRGB (linear 0.178) a dark metallic primary. Fed raw
        # into Specular IOR Level they reflected ~11x too little.
        self.assertAlmostEqual(reflectance_to_specular_ior_level(0.0437), 0.546, places=3)
        self.assertAlmostEqual(reflectance_to_specular_ior_level(0.178), 2.225, places=3)

    def test_scale_follows_the_bsdf_ior(self) -> None:
        self.assertAlmostEqual(specular_ior_level_scale(1.5), 12.5)
        self.assertGreater(specular_ior_level_scale(1.33), specular_ior_level_scale(1.5))


if __name__ == "__main__":
    unittest.main()
