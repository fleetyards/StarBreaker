from __future__ import annotations

from dataclasses import dataclass, replace

from .manifest import PaletteChannel, SubmaterialRecord, TextureReference


@dataclass(frozen=True)
class MaterialTemplatePlan:
    template_key: str
    label: str
    blend_method: str
    shadow_method: str
    cycles_first: bool
    uses_palette: bool
    uses_layers: bool
    uses_transmission: bool
    uses_emission: bool
    uses_alpha: bool


BASE_COLOR_ROLES = ("base_color", "alternate_base_color", "diffuse", "decal_sheet")
NORMAL_ROLES = ("normal_gloss",)
ROUGHNESS_ROLES = ("roughness",)
MASK_ROLES = ("wear_mask", "pattern_mask", "stencil", "tint_palette_decal", "hal_control")
HEIGHT_ROLES = ("height",)
OPACITY_ROLES = ("opacity",)


def active_submaterials(submaterials: list[SubmaterialRecord]) -> list[SubmaterialRecord]:
    return [submaterial for submaterial in submaterials if submaterial.activation_state == "active"]


def _is_empty_livery_decal(submaterial: SubmaterialRecord) -> bool:
    """A livery slot with nothing to draw until a paint fills it.

    Besides `livery_decal`, an untextured placeholder mesh decal: the F8C's
    `placeholder_livery_999th` covers the same stripes as its `livery` decal,
    with nothing to draw until a paint supplies its livery.
    """
    name = (submaterial.submaterial_name or "").lower()
    is_livery_slot = name == "livery_decal"
    if not is_livery_slot and not (submaterial.shader_family == "MeshDecal" and "placeholder" in name):
        return False
    # Only the livery slot counts a missing base colour as empty: other
    # livery decals (`livery`) draw from the palette's decal texture.
    if (
        is_livery_slot
        and submaterial.activation_state != "active"
        and submaterial.activation_reason == "missing_base_color_texture"
    ):
        return True
    return not [
        *submaterial.texture_slots,
        *submaterial.direct_textures,
        *submaterial.derived_textures,
        *submaterial.virtual_inputs,
    ]


def _uses_decal_or_stencil_template(submaterial: SubmaterialRecord) -> bool:
    family = submaterial.shader_family
    flags = submaterial.decoded_feature_flags
    return family == "MeshDecal" or flags.has_decal or (flags.has_stencil_map and family != "HardSurface")


def alpha_test_threshold(submaterial: SubmaterialRecord) -> float | None:
    """Authored ``AlphaTest`` cut-off of an alpha-tested surface, if any.

    CryEngine discards pixels whose alpha is below ``AlphaTest`` and draws
    the rest opaque, e.g. the Star Runner's POM floor grate (0.61). Decals
    and stencils blend over their host instead and are excluded.
    """
    flags = submaterial.decoded_feature_flags
    if submaterial.shader_family == "MeshDecal" or flags.has_decal or flags.has_stencil_map:
        return None
    value = _authored_float(submaterial, "AlphaTest")
    return value if value is not None and value > 0.0 else None


def _authored_float(submaterial: SubmaterialRecord, name: str) -> float | None:
    for attribute in submaterial.raw.get("authored_attributes", []):
        if attribute.get("name") != name:
            continue
        try:
            return float(attribute.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _is_opaque_surface(submaterial: SubmaterialRecord) -> bool:
    """Non-decal surface with no alpha test and full ``Opacity``.

    CryEngine draws such a surface opaque whatever its diffuse alpha holds,
    e.g. the Star Runner's ``Ext_Damage_Pom`` hull panels: a POM material
    whose diffuse alpha is not opacity. Blending it made the hull see-through.
    """
    flags = submaterial.decoded_feature_flags
    if submaterial.shader_family == "MeshDecal" or flags.has_decal or flags.has_stencil_map:
        return False
    if alpha_test_threshold(submaterial) is not None:
        return False
    opacity = _authored_float(submaterial, "Opacity")
    return opacity is None or opacity >= 1.0 - 1e-4


def template_plan_for_submaterial(submaterial: SubmaterialRecord) -> MaterialTemplatePlan:
    plan = _base_template_plan(submaterial)
    if plan.uses_alpha and plan.blend_method == "BLEND":
        if alpha_test_threshold(submaterial) is not None:
            # Alpha-tested: hard cut-out, not a translucent blend.
            return replace(plan, blend_method="CLIP")
        if plan.template_key == "parallax_pom" and _is_opaque_surface(submaterial):
            return replace(plan, blend_method="OPAQUE", uses_alpha=False)
    return plan


def _base_template_plan(submaterial: SubmaterialRecord) -> MaterialTemplatePlan:
    family = submaterial.shader_family
    flags = submaterial.decoded_feature_flags

    if family == "NoDraw":
        return MaterialTemplatePlan("nodraw", "NoDraw", "CLIP", "NONE", False, False, False, False, False, True)
    if _is_empty_livery_decal(submaterial):
        return MaterialTemplatePlan("nodraw", "NoDraw", "CLIP", "NONE", False, False, False, False, False, True)
    if submaterial.activation_state != "active" and not _uses_decal_or_stencil_template(submaterial):
        return MaterialTemplatePlan("nodraw", "NoDraw", "CLIP", "NONE", False, False, False, False, False, True)
    if family in {"Layer", "LayerBlend_V2"}:
        return MaterialTemplatePlan("layered_wear", "Layered Wear", "OPAQUE", "OPAQUE", True, True, True, False, False, False)
    if _uses_decal_or_stencil_template(submaterial):
        return MaterialTemplatePlan("decal_stencil", "Decal Or Stencil", "CLIP", "NONE", True, True, False, False, False, True)
    if flags.has_parallax_occlusion_mapping:
        return MaterialTemplatePlan("parallax_pom", "Parallax Or POM", "BLEND", "NONE", True, True, False, False, True, True)
    if family in {"DisplayScreen", "Monitor", "UIPlane"}:
        return MaterialTemplatePlan("screen_hud", "Screen Or HUD", "BLEND", "HASHED", True, False, False, False, True, True)
    if family in {"HumanSkin_V2", "Eye", "Organic"}:
        return MaterialTemplatePlan("biological", "Biological", "OPAQUE", "OPAQUE", True, True, False, False, False, False)
    if family == "HairPBR":
        return MaterialTemplatePlan("hair", "Hair", "HASHED", "HASHED", True, True, False, False, False, True)
    if family in {"Hologram", "HologramCIG", "Shield_Holo", "UIMesh"}:
        return MaterialTemplatePlan("effects", "Effects", "BLEND", "NONE", True, False, False, False, True, True)
    if family == "GlassPBR":
        return MaterialTemplatePlan("physical_surface", "Physical Surface", "BLEND", "HASHED", True, True, False, True, False, True)
    if family == "Illum":
        return MaterialTemplatePlan("physical_surface", "Physical Surface", "OPAQUE", "OPAQUE", True, True, False, False, True, False)
    return MaterialTemplatePlan("physical_surface", "Physical Surface", "OPAQUE", "OPAQUE", True, True, False, False, False, False)


def _all_texture_candidates(submaterial: SubmaterialRecord) -> list[TextureReference]:
    return [*submaterial.texture_slots, *submaterial.direct_textures, *submaterial.derived_textures]


def first_texture_export(submaterial: SubmaterialRecord, roles: tuple[str, ...]) -> str | None:
    for texture in _all_texture_candidates(submaterial):
        if texture.role in roles and texture.export_path:
            return texture.export_path
    return None


def representative_textures(submaterial: SubmaterialRecord) -> dict[str, str | None]:
    layer_base = next((layer.diffuse_export_path for layer in submaterial.layer_manifest if layer.diffuse_export_path), None)
    layer_normal = next((layer.normal_export_path for layer in submaterial.layer_manifest if layer.normal_export_path), None)
    layer_roughness = next((layer.roughness_export_path for layer in submaterial.layer_manifest if layer.roughness_export_path), None)
    return {
        "base_color": first_texture_export(submaterial, BASE_COLOR_ROLES) or layer_base,
        "normal": first_texture_export(submaterial, NORMAL_ROLES) or layer_normal,
        "roughness": first_texture_export(submaterial, ROUGHNESS_ROLES) or layer_roughness,
        "mask": first_texture_export(submaterial, MASK_ROLES),
        "height": first_texture_export(submaterial, HEIGHT_ROLES),
        "opacity": first_texture_export(submaterial, OPACITY_ROLES),
    }


def smoothness_texture_reference(submaterial: SubmaterialRecord) -> TextureReference | None:
    for texture in _all_texture_candidates(submaterial):
        if texture.alpha_semantic == "smoothness" and texture.export_path:
            return texture
    for layer in submaterial.layer_manifest:
        for texture in layer.texture_slots:
            if texture.alpha_semantic == "smoothness" and texture.export_path:
                return texture
    return None


def material_palette_channels(submaterial: SubmaterialRecord) -> list[PaletteChannel]:
    channels: list[PaletteChannel] = []
    if submaterial.palette_routing.material_channel is not None:
        channels.append(submaterial.palette_routing.material_channel)
    for binding in submaterial.palette_routing.layer_channels:
        if all((existing.index, existing.name) != (binding.channel.index, binding.channel.name) for existing in channels):
            channels.append(binding.channel)
    for layer in submaterial.layer_manifest:
        if layer.palette_channel is not None and all(
            (existing.index, existing.name) != (layer.palette_channel.index, layer.palette_channel.name)
            for existing in channels
        ):
            channels.append(layer.palette_channel)
    return channels


def has_virtual_input(submaterial: SubmaterialRecord, input_name: str) -> bool:
    return input_name in submaterial.virtual_inputs
