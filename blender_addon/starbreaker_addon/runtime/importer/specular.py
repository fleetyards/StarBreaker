"""Specular helpers for runtime material reconstruction.

CryEngine authors specular as reflectance at normal incidence (F0): palette
``specColor`` and material ``Specular`` are sRGB colours whose linear value is
F0 directly -- 0.04 for ordinary paint, more for metallic finishes. Blender's
Principled BSDF instead takes ``Specular IOR Level``, where 0.5 reproduces the
F0 of its IOR and the reflectance scales linearly with the level::

    F0 = 2 * level * ((ior - 1) / (ior + 1)) ** 2

so an authored F0 has to be divided by that per-level reflectance before it
reaches the BSDF.
"""

from __future__ import annotations

DEFAULT_BSDF_IOR = 1.5


def specular_ior_level_scale(ior: float = DEFAULT_BSDF_IOR) -> float:
    """Factor turning an F0 reflectance into Principled ``Specular IOR Level``."""

    base_reflectance = ((ior - 1.0) / (ior + 1.0)) ** 2
    return 1.0 / (2.0 * base_reflectance)


def reflectance_to_specular_ior_level(reflectance: float, ior: float = DEFAULT_BSDF_IOR) -> float:
    """Principled ``Specular IOR Level`` that reflects ``reflectance`` at normal incidence."""

    return max(0.0, float(reflectance)) * specular_ior_level_scale(ior)
