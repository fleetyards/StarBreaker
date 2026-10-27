"""Material-builder mixin for :class:`PackageImporter`.

Extracted in Phase 7.5 from ``runtime/_legacy.py``. Contains the
``_build_*_material`` dispatchers plus the small set of helpers they
rely on to resolve the template contract and per-submaterial group
contract (``_palette_scope``, ``_template_contract``,
``_group_contract_for_submaterial``, ``_ensure_contract_group``).

Each ``_build_*_material`` method owns the node-tree layout for one
shader family (hard_surface, illum, glass, principled, layered-wear,
nodraw, screen, effect, contract-group). They depend on the group
mixin (:class:`GroupsMixin`) for the shared ``_ensure_runtime_*_group``
node trees and on the per-material socket / wiring helpers still in
:class:`PackageImporter`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import bpy

from ..constants import (
    MATERIAL_IDENTITY_SCHEMA,
    NON_COLOR_INPUT_KEYWORDS,
    POM_DETAIL_DEFAULT,
    PROP_DECAL_HOST_CHANNEL,
    PROP_DECAL_HOST_RGB,
    PROP_MATERIAL_IDENTITY,
    PROP_MATERIAL_SIDECAR,
    PROP_PALETTE_ID,
    PROP_PALETTE_SCOPE,
    PROP_PALETTE_SCOPE_MAP,
    PROP_SHADER_FAMILY,
    SCENE_POM_DETAIL_PROP,
    PROP_SUBMATERIAL_JSON,
    PROP_SURFACE_SHADER_MODE,
    PROP_TEMPLATE_KEY,
    PROP_HAS_POM,
    SURFACE_SHADER_MODE_GLASS,
    SURFACE_SHADER_MODE_PRINCIPLED,
    pom_detail_settings,
)
from ..node_utils import _input_socket, _output_socket, _refresh_group_node_sockets, _set_group_input_default
from ..package_ops import _string_prop
from ..record_utils import (
    _authored_attribute_triplet,
    _emissive_opacity_factor,
    _float_authored_attribute,
    _float_public_param,
    _hard_surface_angle_shift_enabled,
    _layer_snapshot_float,
    _layer_texture_reference,
    _matching_texture_reference,
    _mean_triplet,
    _optional_float_public_param,
    _public_param_triplet,
    _resolve_public_param_default,
    _resolved_submaterial_palette_color,
    _routes_virtual_tint_palette_decal_alpha_to_decal_source,
    _routes_virtual_tint_palette_decal_to_decal_source,
    _triplet_from_any,
    _submaterial_texture_reference,
    _suppresses_virtual_tint_palette_stencil_input,
    _uses_vertex_color_tint,
    _uses_virtual_tint_palette_decal,
)
from ...manifest import LayerManifestEntry, MaterialSidecar, PaletteRecord, SubmaterialRecord, TextureReference
from ...material_contract import (
    ContractInput,
    ShaderGroupContract,
    TemplateContract,
    bundled_template_library_path,
    load_bundled_template_contract,
)
from ...palette import (
    palette_color,
    palette_finish_glossiness_factor,
    palette_finish_specular,
)
from ...templates import has_virtual_input, material_palette_channels, representative_textures, template_plan_for_submaterial
from ..palette_utils import _hard_surface_palette_iridescence_channel
from .types import LayerSurfaceSockets


def _canonical_material_sidecar_path(sidecar_path: str, sidecar: MaterialSidecar) -> str:
    return sidecar.normalized_export_relative_path or sidecar_path or sidecar.source_material_path or "material"


def _safe_identifier(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe.strip("_") or "value"


def _authored_emissive_triplet(submaterial: SubmaterialRecord) -> tuple[float, float, float] | None:
    for attribute in submaterial.raw.get("authored_attributes", []):
        if attribute.get("name") != "Emissive":
            continue
        return _triplet_from_any(attribute.get("value"))
    return None


def _layered_wear_base_layer(
    submaterial: SubmaterialRecord,
) -> LayerManifestEntry | None:
    if submaterial.layer_manifest:
        return submaterial.layer_manifest[0]
    return None


def _layered_wear_first_diffuse_layer(
    submaterial: SubmaterialRecord,
) -> LayerManifestEntry | None:
    # Never a wear layer's: the Gallant's polished-aluminium front rail
    # (H_bare_01_B) has no diffuse map of its own and took its wear layer's
    # grime map, a dirty green on bare metal.
    return next(
        (
            layer
            for layer in submaterial.layer_manifest
            if layer.diffuse_export_path
            and not (getattr(layer, "name", None) or "").lower().startswith("wearlayer")
        ),
        None,
    )


def _layered_wear_first_non_neutral_tint(
    submaterial: SubmaterialRecord,
) -> tuple[float, float, float] | None:
    for layer in submaterial.layer_manifest:
        if (getattr(layer, "name", None) or "").lower().startswith("wearlayer"):
            continue
        tint = layer.tint_color
        if tint is not None and any(abs(channel - 1.0) > 1e-6 for channel in tint):
            return tint
    return None


def _layer_metal_tint(layer: LayerManifestEntry) -> tuple[float, float, float] | None:
    """A layer's colour: its tint, times its specular colour when it is a
    metal. A metal's colour is its F0, and the layer file keeps that in
    Specular with a black Diffuse: the Gallant battery's weapon_copper_03
    (0.30, 0.18, 0.09) and coil bronze_01 rendered as white metal on the tint
    alone. A (1,1,1) specular is the spec-map placeholder and is left out."""
    tint = layer.tint_color
    if tint is None:
        return None
    if _layer_snapshot_float(layer, "metallic") < 0.5:
        return tint
    specular = _triplet_from_any(layer.layer_snapshot.get("specular"))
    if specular is None or all(abs(channel - 1.0) < 1e-6 for channel in specular):
        return tint
    # A near-black tint on a metal keeps only its hue: the Hurston
    # Attrition's copper rails (weapon_copper_striped_01, tint
    # 0.026/0.045/0.061) drew black at full strength and orange on the bare
    # specular; the tint's blue takes them to the rose copper of the shop
    # shots. A real tint is brighter -- the Zeus visor's 0.07/0.02/1.0 turns
    # its gold violet, the Ghost Mk II's 0.138 greys its aluminium.
    if max(tint) < 0.1:
        if max(tint) <= 1e-6:
            return specular
        return tuple(t / max(tint) * s for t, s in zip(tint, specular))  # type: ignore[return-value]
    return tuple(t * s for t, s in zip(tint, specular))  # type: ignore[return-value]


def _texture_reference_uv_scale(ref: TextureReference) -> tuple[float, float] | None:
    """The slot's authored TexMod TileU/TileV, or None without one."""
    scale = (ref.texture_transform or {}).get("scale")
    if isinstance(scale, (list, tuple)) and len(scale) == 2:
        try:
            u, v = float(scale[0]), float(scale[1])
        except (TypeError, ValueError):
            return None
        if u > 0.0 and v > 0.0:
            return u, v
    return None


def _illum_base_uv_scale(submaterial: SubmaterialRecord) -> tuple[float, float]:
    """Illum's base UV tiling: the diffuse slot's TexMod, else the normal's."""
    for slot in ("TexSlot1", "TexSlot2"):
        for ref in submaterial.texture_slots:
            if ref.slot == slot:
                scale = _texture_reference_uv_scale(ref)
                if scale is not None:
                    return scale
    return 1.0, 1.0


def _clamp_unit_float(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parallax_height_sampler_extension(uv_tile: float) -> str:
    """Mirror authored repeat intent for runtime POM height samplers.

    POM materials only need sampler wrap mode ``REPEAT`` when the authored
    material explicitly tiles the underlying surface beyond the base 0-1 UV
    range. The exported layer manifest preserves that as ``uv_tiling``; every
    other case should stay on ``CLIP`` to avoid sampling neighboring atlas
    islands at glancing angles.
    """

    return "REPEAT" if float(uv_tile) > 1.0 + 1e-4 else "CLIP"


def _layered_wear_metallic_values(
    base_layer: LayerManifestEntry | None,
    wear_layer: LayerManifestEntry | None,
) -> tuple[float, float] | None:
    base_metallic = _layer_snapshot_float(base_layer, "metallic") if base_layer is not None else None
    wear_metallic = _layer_snapshot_float(wear_layer, "metallic") if wear_layer is not None else None
    if base_metallic is None and wear_metallic is None:
        return None
    if base_metallic is None:
        base_metallic = wear_metallic
    if wear_metallic is None:
        wear_metallic = base_metallic
    if base_metallic is None or wear_metallic is None:
        return None
    return _clamp_unit_float(base_metallic), _clamp_unit_float(wear_metallic)


def _mesh_decal_neutral_breakup_default(
    group_contract: ShaderGroupContract,
    submaterial: SubmaterialRecord,
    contract_input: ContractInput,
    source_socket: Any,
) -> tuple[float, float, float, float] | None:
    if source_socket is not None:
        return None
    if group_contract.name != "SB_MeshDecal_v1":
        return None
    if contract_input.name != "TexSlot8_GrimeBreakup":
        return None
    if template_plan_for_submaterial(submaterial).template_key != "decal_stencil":
        return None
    return (1.0, 1.0, 1.0, 1.0)


def _tinted_specular(submaterial: Any) -> tuple[float, float, float] | None:
    """An Illum metal's coloured F0 from its authored Specular, else None.

    The 600i Executive Edition's gold trim is accent/grey metals with
    Specular 0.59,0.27,0.05 over a near-black diffuse; the Cutlasses' engine
    insulation is the same copper foil. Grey speculars stay on the default.
    """
    specular = _authored_attribute_triplet(submaterial, "Specular")
    if specular is None or max(specular) - min(specular) <= 0.05:
        return None
    # Read back as sRGB: as a linear F0 the gold came out dark orange copper,
    # the game shows it pale champagne.
    return tuple(c ** (1.0 / 2.2) for c in specular)


def _public_param_triplet(submaterial: Any, name: str) -> tuple[float, float, float] | None:
    raw = getattr(submaterial, "raw", None)
    params = raw.get("authored_public_params") if isinstance(raw, dict) else None
    for item in params or []:
        if isinstance(item, dict) and item.get("name") == name:
            try:
                values = [float(v) for v in str(item.get("value")).split(",")]
            except ValueError:
                return None
            return tuple(values[:3]) if len(values) >= 3 else None
    return None


def _scaled_specular(nodes: Any, socket: Any, specular: Any, *, x: int, y: int) -> Any:
    """An Illum specular map times the authored Specular, as the engine does.

    Read alone, the map (~0.6 on the 300 series' plastic panels) was the F0
    and the metallic: the 350r's near-black hull (Specular 0.045) rendered
    as brushed silver. The (1,1,1) metal placeholder keeps the map as is.
    """
    if socket is None or specular is None:
        return socket
    grey = sum(specular[:3]) / 3.0
    if grey >= 0.99 or max(specular) - min(specular) > 0.05:
        return socket
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.label = "Specular x authored"
    scale.location = (x, y)
    nodes.id_data.links.new(socket, scale.inputs[0])
    scale.inputs[1].default_value = grey
    return scale.outputs[0]


def _authored_shader_name(submaterial: Any) -> str:
    raw = getattr(submaterial, "raw", None)
    for item in (raw.get("authored_attributes") if isinstance(raw, dict) else None) or []:
        if isinstance(item, dict) and item.get("name") == "Shader":
            return str(item.get("value") or "").lower()
    return ""


class BuildersMixin:
    def _apply_uv_tiling(
        self,
        nodes: bpy.types.Nodes,
        links: bpy.types.NodeLinks,
        image_node: bpy.types.ShaderNodeTexImage | None,
        tile: float,
        *,
        x: int,
        y: int,
    ) -> None:
        """Phase 8: inject a Mapping + TexCoord pair before ``image_node`` to
        scale UVs by ``tile``. No-op when ``image_node`` is missing or the
        tiling factor is effectively 1.0. ``ShaderNodeMapping`` and
        ``ShaderNodeTexCoord`` are explicitly allowed at the top level by
        ``validators.MATERIAL_TOP_LEVEL_ALLOWED_BL_IDNAMES``.
        """
        if image_node is None:
            return
        if tile <= 0.0 or abs(tile - 1.0) < 1e-4:
            return
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (x - 220, y)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (x, y)
        mapping.vector_type = "POINT"
        scale_socket = mapping.inputs.get("Scale")
        if scale_socket is not None:
            scale_socket.default_value = (tile, tile, 1.0)
        links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
        vector_input = image_node.inputs.get("Vector")
        if vector_input is not None:
            links.new(mapping.outputs["Vector"], vector_input)

    def _sweep_unreachable_nodes(self, material: bpy.types.Material) -> None:
        """Remove nodes that cannot be reached by walking backwards from the
        ``ShaderNodeOutputMaterial`` output(s).

        VRAM optimization: after a material has been built, some sampler
        nodes (`ShaderNodeTexImage`) and their helpers
        (`ShaderNodeMapping`, `ShaderNodeTexCoord`, `ShaderNodeNormalMap`,
        …) may have been created by builders that then chose not to wire
        them — typically because a feature flag resolved to ``False``
        after the node was already created (or because a fallback path
        short-circuited wiring). Such nodes have no effect on the
        rendered output but still cause Cycles to load their images into
        VRAM.

        This pass performs a standard dead-code-elimination sweep: start
        from every ``ShaderNodeOutputMaterial``, walk *upstream* through
        ``node_tree.links``, mark every visited node as reachable, then
        remove every unmarked node.

        Safe for paint switching — ``rebuild_object_materials`` always
        calls ``nodes.clear()`` at the start of each builder, so no
        builder relies on finding pre-existing nodes from a prior build.
        """
        node_tree = material.node_tree
        if node_tree is None:
            return
        nodes = node_tree.nodes
        links = node_tree.links

        # Build reverse-adjacency: for each node, the set of nodes feeding it.
        incoming: dict[bpy.types.Node, set[bpy.types.Node]] = {}
        for link in links:
            incoming.setdefault(link.to_node, set()).add(link.from_node)

        reachable: set[bpy.types.Node] = set()
        stack: list[bpy.types.Node] = [n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            for predecessor in incoming.get(node, ()):
                if predecessor not in reachable:
                    stack.append(predecessor)

        for node in list(nodes):
            if node in reachable:
                continue
            # Frame nodes hold no logic but parent other nodes visually;
            # preserving them would anchor removed children, so drop them too.
            nodes.remove(node)

    def _wire_runtime_parallax(
        self,
        material: bpy.types.Material,
        height_node: bpy.types.Node,
        target_image_nodes: list[bpy.types.Node],
        scale_value: float,
        bias_value: float = 0.5,
        location: tuple[float, float] = (-1280, 720),
        uv_tile: float = 1.0,
    ) -> bpy.types.Node | None:
        """Insert the bundled ``POM_Vector`` production POM pipeline
        (30-step ray-march, authored in ``docs/StarBreaker/POM-test.blend``
        and bundled as ``resources/pom_library.blend``) between the
        material's UV source and ``target_image_nodes``' ``Vector``
        inputs.

        ``height_node`` must be a ``ShaderNodeTexImage`` whose ``image``
        slot holds the authored displacement map — its pixels drive the
        ray-march. Because ``POM_Vector``'s internal ``POM_disp`` /
        ``HeightMap`` groups contain sampler datablocks that Blender
        cannot override from outside the group, each unique displacement
        image gets its own appended copy of the whole POM chain (cached
        by image name, see ``_ensure_runtime_parallax_group``).

        Shared between ``_build_hard_surface_material`` and
        ``_build_contract_group_material`` (MeshDecal POM path). Returns
        the newly-created parallax group node, or ``None`` if the
        material has no node tree, no height image is available, or the
        POM library could not be appended.

        ``scale_value`` is the authored ``PomDisplacement`` public param
        (typically 0.02–0.1 in CryEngine's units). It is scaled up to
        POM-test's ``Scale`` range (≈1.0–3.0) so a 0.05 PomDisplacement
        reads as ≈1.5 POM scale — the reference file's hand-tuned
        default. ``Layers`` is fixed at 40 and ``Bias`` defaults to 0.5,
        but authored height-bias overrides are preserved when available.

        ``uv_tile`` is the layer's UVTiling factor (default 1.0 = no
        tiling). The POM group incorporates the scale internally via its
        ``UV Scale X`` / ``UV Scale Y`` inputs so that both the starting
        UV and the ray-march delta are scaled consistently. Height-sampler
        wrap mode also mirrors this authored repeat intent: explicit tiling
        uses ``REPEAT`` while default UVs stay on ``CLIP`` to avoid atlas
        bleed. Callers must NOT also call ``_apply_uv_tiling`` on the same
        target nodes; any pre-existing Mapping chain on a target's Vector
        socket is removed before POM wiring so the orphaned nodes can be
        swept later by ``_sweep_unreachable_nodes``.
        """
        node_tree = material.node_tree
        if node_tree is None or height_node is None:
            return None
        if height_node.bl_idname != "ShaderNodeTexImage" or height_node.image is None:
            return None

        pom_tree = self._ensure_runtime_parallax_group(
            height_image=height_node.image,
            sampler_extension=_parallax_height_sampler_extension(uv_tile),
        )
        if pom_tree is None:
            return None

        nodes = node_tree.nodes
        links = node_tree.links
        parallax_node = nodes.new("ShaderNodeGroup")
        parallax_node.node_tree = pom_tree
        _refresh_group_node_sockets(parallax_node)
        parallax_node.location = (location[0], location[1])
        parallax_node.label = "StarBreaker POM"

        # POM_Vector inputs: Scale (Float), Bias (Float), Non-planar
        # (Bool), UV Scale X/Y (Float). Layer count is controlled inside
        # the runtime POM root group based on the active scene profile.
        # Drive Scale from the authored PomDisplacement
        # (CryEngine-space ≈0.02–0.1) rescaled into POM-test's default
        # range (≈1.5 for 0.05 input) by multiplying by 30.
        self._set_socket_default(_input_socket(parallax_node, "Scale"), min(3.0, scale_value * 30.0))
        self._set_socket_default(_input_socket(parallax_node, "Bias"), max(0.0, min(1.0, bias_value)))
        self._set_socket_default(_input_socket(parallax_node, "Non-planar"), True)
        clamped_tile = max(0.001, uv_tile)
        self._set_socket_default(_input_socket(parallax_node, "UV Scale X"), clamped_tile)
        self._set_socket_default(_input_socket(parallax_node, "UV Scale Y"), clamped_tile)

        offset_vec = _output_socket(parallax_node, "Vector")
        if offset_vec is None:
            return parallax_node
        for tex_node in target_image_nodes:
            if tex_node is None:
                continue
            vector_input = tex_node.inputs.get("Vector")
            if vector_input is None:
                continue
            # Remove any existing link on the Vector socket (e.g. from
            # _apply_uv_tiling) so POM can take over.  UV scaling is
            # incorporated by the POM group internally via UV Scale X/Y;
            # orphaned Mapping nodes are swept by _sweep_unreachable_nodes.
            for existing_link in list(links):
                if existing_link.to_socket == vector_input:
                    links.remove(existing_link)
            links.new(offset_vec, vector_input)
        return parallax_node

    @staticmethod
    def _parallax_bias_value(submaterial: SubmaterialRecord) -> float:
        return max(
            0.0,
            min(
                1.0,
                _float_public_param(
                    submaterial,
                    "HeightBias",
                    "POMHeightBias",
                    "POM_HeightBias",
                )
                or 0.5,
            ),
        )

    def _build_managed_material(
        self,
        material: bpy.types.Material,
        sidecar_path: str,
        sidecar: MaterialSidecar,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        material_identity: str,
    ) -> None:
        palette_key = palette.id if palette is not None else "none"
        material.use_nodes = True
        plan = template_plan_for_submaterial(submaterial)
        surface_mode = SURFACE_SHADER_MODE_PRINCIPLED
        if plan.template_key == "nodraw":
            self._build_nodraw_material(material)
        elif submaterial.shader_family == "HardSurface":
            self._build_hard_surface_material(material, submaterial, palette, plan)
        elif submaterial.shader_family == "Illum" or _authored_shader_name(submaterial) in ("cloth", "layer"):
            # Cloth (the M50's seat mesh) and a Layer used as a surface (its
            # Rubber tyres) have Illum's inputs: diffuse and normal maps
            # tinted by the authored Diffuse. Built elsewhere, their
            # near-black fabric and rubber drew as the bare white maps.
            self._build_illum_material(material, submaterial, palette, plan)
        else:
            group_contract = None if plan.template_key == "layered_wear" else self._group_contract_for_submaterial(submaterial)
            if group_contract is not None and self._build_contract_group_material(material, submaterial, palette, plan, group_contract):
                if submaterial.shader_family == "GlassPBR":
                    surface_mode = SURFACE_SHADER_MODE_GLASS
            elif submaterial.shader_family == "GlassPBR":
                self._build_glass_material(material, submaterial, palette, plan)
                surface_mode = SURFACE_SHADER_MODE_GLASS
            elif plan.template_key == "screen_hud":
                self._build_screen_material(material, submaterial, palette, plan)
            elif plan.template_key == "effects":
                self._build_effect_material(material, submaterial, palette, plan)
            else:
                self._build_principled_material(material, submaterial, palette, plan)

        self._apply_material_node_layout(material)
        self._sweep_unreachable_nodes(material)

        material[PROP_SHADER_FAMILY] = submaterial.shader_family
        material[PROP_TEMPLATE_KEY] = plan.template_key
        material[PROP_PALETTE_ID] = palette_key
        material[PROP_PALETTE_SCOPE] = self._palette_scope(palette)
        material[PROP_MATERIAL_SIDECAR] = _canonical_material_sidecar_path(sidecar_path, sidecar)
        material[PROP_MATERIAL_IDENTITY] = material_identity
        material[PROP_SUBMATERIAL_JSON] = json.dumps(submaterial.raw, sort_keys=True)
        material[PROP_SURFACE_SHADER_MODE] = surface_mode
        material[PROP_HAS_POM] = bool(
            submaterial.decoded_feature_flags.has_parallax_occlusion_mapping
        )

    def _palette_scope(self, palette: PaletteRecord | None = None) -> str:
        """Return a stable per-``palette.id`` scope UUID for this package.

        Each distinct ``palette_id`` within a package gets its own UUID,
        persisted on the package root as a JSON map under
        ``PROP_PALETTE_SCOPE_MAP``. This is what lets the importer emit
        one ``StarBreaker Palette`` node group per palette scope (for
        example one for the exterior `palette/rsi_aurora_mk2` and one
        for the interior `palette/rsi_interior_default`).

        ``palette=None`` falls back to the legacy per-package scope
        stored under ``PROP_PALETTE_SCOPE``; callers that operate
        without a palette (glass, nodraw, etc.) keep working
        unchanged.
        """
        package_root = self.package_root
        if package_root is None:
            return _safe_identifier(self.package.package_name)

        palette_id = palette.id if palette is not None else None
        if palette_id:
            scope_map_json = _string_prop(package_root, PROP_PALETTE_SCOPE_MAP) or "{}"
            try:
                scope_map: dict[str, str] = json.loads(scope_map_json)
                if not isinstance(scope_map, dict):
                    scope_map = {}
            except (ValueError, TypeError):
                scope_map = {}
            scope = scope_map.get(palette_id)
            if not scope:
                scope = uuid.uuid4().hex
                scope_map[palette_id] = scope
                package_root[PROP_PALETTE_SCOPE_MAP] = json.dumps(scope_map, sort_keys=True)
            return scope

        # Fallback: single legacy scope for palette-less materials.
        palette_scope = _string_prop(package_root, PROP_PALETTE_SCOPE)
        if palette_scope:
            return palette_scope
        palette_scope = uuid.uuid4().hex
        package_root[PROP_PALETTE_SCOPE] = palette_scope
        return palette_scope

    def _template_contract(self) -> TemplateContract:
        if self.bundled_template_contract is None:
            self.bundled_template_contract = load_bundled_template_contract()
        return self.bundled_template_contract

    def _group_contract_for_submaterial(self, submaterial: SubmaterialRecord) -> ShaderGroupContract | None:
        return self._template_contract().group_for_shader_family(submaterial.shader_family)

    def _ensure_contract_group(self, group_contract: ShaderGroupContract) -> bpy.types.ShaderNodeTree | None:
        group = bpy.data.node_groups.get(group_contract.name)
        if group is None:
            library_path = bundled_template_library_path()
            if not library_path.is_file():
                return None
            with bpy.data.libraries.load(str(library_path), link=False) as (data_from, data_to):
                if group_contract.name not in data_from.node_groups:
                    return None
                data_to.node_groups = [group_contract.name]
            group = bpy.data.node_groups.get(group_contract.name)
        if group is not None and group_contract.name == "SB_GlassPBR_v1":
            self._patch_glass_template_lightpath(group)
        if group is not None and group_contract.name == "SB_MeshDecal_v1":
            self._patch_mesh_decal_template_emission(group)
            self._patch_mesh_decal_template_metal(group)
        return group

    def _patch_glass_template_lightpath(self, group: bpy.types.ShaderNodeTree) -> None:
        """Patch the bundled GlassPBR template to the live validated graph.

        Roughness comes from the authored ``RefractionGloss`` (1 = clear),
        with the wear gloss texture mixed in by ``WearBlendBase``. TexSlot6
        holds the gloss of the worn layer only: used as the glass gloss, the
        Constellation canopy (``glass_canopy_gloss``, ~0.5) rendered as
        frosted glass while it is authored clear.
        """
        if group.get("starbreaker_glass_lightpath_patch_version") == 5:
            return
        for socket_name, default in (("Param_RefractionGloss", 1.0), ("Param_WearBlendBase", 0.0)):
            existing = next(
                (
                    item
                    for item in group.interface.items_tree
                    if getattr(item, "item_type", "") == "SOCKET"
                    and item.in_out == "INPUT"
                    and item.name == socket_name
                ),
                None,
            )
            if existing is None:
                existing = group.interface.new_socket(name=socket_name, in_out="INPUT", socket_type="NodeSocketFloat")
            existing.default_value = default
        wear_gloss_item = next(
            (
                item
                for item in group.interface.items_tree
                if getattr(item, "item_type", "") == "SOCKET"
                and item.in_out == "INPUT"
                and item.name == "TexSlot6_WearGloss"
            ),
            None,
        )
        if wear_gloss_item is not None and hasattr(wear_gloss_item, "default_value"):
            wear_gloss_item.default_value = (1.0, 1.0, 1.0, 1.0)
        nodes = group.nodes
        links = group.links
        group_input = next((n for n in nodes if n.bl_idname == "NodeGroupInput"), None)
        group_output = next((n for n in nodes if n.bl_idname == "NodeGroupOutput"), None)
        if group_input is None or group_output is None:
            return
        for node in list(nodes):
            if node not in {group_input, group_output}:
                nodes.remove(node)
        group_input.location = (-132.14703369140625, -617.05126953125)
        group_output.location = (1639.01513671875, -298.6864929199219)
        shader_input = group_output.inputs.get("Shader") or (group_output.inputs[0] if group_output.inputs else None)
        if shader_input is None:
            return
        mix_legacy = nodes.new("ShaderNodeMixRGB")
        mix_legacy.name = "Mix (Legacy)"
        mix_legacy.location = (474.3709716796875, -582.9908447265625)
        mix_legacy.blend_type = "MULTIPLY"
        mix_legacy.inputs[0].default_value = 1.0

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.name = "Normal Map"
        normal_map.location = (775.9833984375, -803.2053833007812)
        normal_map.space = "TANGENT"
        normal_map.inputs["Strength"].default_value = 0.25

        gloss_mix = nodes.new("ShaderNodeMix")
        gloss_mix.name = "Wear Gloss Mix"
        gloss_mix.data_type = "FLOAT"
        gloss_mix.clamp_factor = True
        gloss_mix.location = (474.0, -230.0)

        gloss_to_roughness = nodes.new("ShaderNodeMath")
        gloss_to_roughness.name = "Gloss To Roughness"
        gloss_to_roughness.operation = "SUBTRACT"
        gloss_to_roughness.use_clamp = True
        gloss_to_roughness.inputs[0].default_value = 1.0
        gloss_to_roughness.location = (790.0, -230.0)

        hue_saturation = nodes.new("ShaderNodeHueSaturation")
        hue_saturation.name = "Hue/Saturation/Value"
        hue_saturation.location = (777.0719604492188, -579.3798828125)
        hue_saturation.inputs["Hue"].default_value = 0.57
        hue_saturation.inputs["Saturation"].default_value = 1.0
        hue_saturation.inputs["Value"].default_value = 1.0
        hue_saturation.inputs["Factor"].default_value = 1.0

        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = "Principled BSDF.001"
        principled.location = (1303.1201171875, -303.4607238769531)
        transmission_socket = _input_socket(principled, "Transmission Weight", "Transmission")
        if transmission_socket is not None:
            transmission_socket.default_value = 1.0
        principled_ior = _input_socket(principled, "IOR")
        if principled_ior is not None:
            principled_ior.default_value = 1.05

        maths = nodes.new("ShaderNodeMath")
        maths.name = "Maths"
        maths.location = (1040.0, -230.0)
        maths.operation = "MULTIPLY"
        maths.use_clamp = True

        tint_color_socket = _output_socket(group_input, "TexSlot4_TintColor")
        palette_glass_socket = _output_socket(group_input, "Palette_Glass")
        normal_socket = _output_socket(group_input, "TexSlot2_NormalGloss")
        smoothness_socket = _output_socket(group_input, "TexSlot6_WearGloss")
        dirt_socket = _output_socket(group_input, "TexSlot11_Dirt")
        refraction_gloss_socket = _output_socket(group_input, "Param_RefractionGloss")
        wear_blend_socket = _output_socket(group_input, "Param_WearBlendBase")
        shader_output_socket = _output_socket(principled, "BSDF") or principled.outputs[0]
        if (
            tint_color_socket is None
            or palette_glass_socket is None
            or normal_socket is None
            or smoothness_socket is None
            or dirt_socket is None
            or refraction_gloss_socket is None
            or wear_blend_socket is None
        ):
            return

        links.new(tint_color_socket, mix_legacy.inputs[1])
        links.new(palette_glass_socket, mix_legacy.inputs[2])
        links.new(normal_socket, normal_map.inputs["Color"])
        links.new(_output_socket(mix_legacy, "Color") or mix_legacy.outputs[0], hue_saturation.inputs["Color"])
        links.new(shader_output_socket, shader_input)
        links.new(wear_blend_socket, gloss_mix.inputs["Factor"])
        links.new(refraction_gloss_socket, gloss_mix.inputs["A"])
        links.new(smoothness_socket, gloss_mix.inputs["B"])
        links.new(gloss_mix.outputs["Result"], gloss_to_roughness.inputs[1])
        links.new(_output_socket(gloss_to_roughness, "Value") or gloss_to_roughness.outputs[0], maths.inputs[0])
        links.new(dirt_socket, maths.inputs[1])
        links.new(_output_socket(maths, "Value") or maths.outputs[0], _input_socket(principled, "Roughness"))
        links.new(_output_socket(hue_saturation, "Color") or hue_saturation.outputs[0], _input_socket(principled, "Base Color"))
        links.new(_output_socket(normal_map, "Normal") or normal_map.outputs[0], _input_socket(principled, "Normal"))
        group["starbreaker_glass_lightpath_patched"] = 1
        group["starbreaker_glass_lightpath_patch_version"] = 5

    @staticmethod
    def _patch_mesh_decal_template_emission(group: bpy.types.ShaderNodeTree) -> None:
        """Route MeshDecal emissive data into Principled emission."""
        if group.get("starbreaker_mesh_decal_emission_patch_version") == 4:
            return

        nodes = group.nodes
        links = group.links
        group_input = next((node for node in nodes if node.bl_idname == "NodeGroupInput"), None)
        principled = nodes.get("Principled BSDF")
        if group_input is None or principled is None:
            return

        emission_strength_socket = _output_socket(group_input, "Emission Strength")
        if emission_strength_socket is None:
            try:
                group.interface.new_socket(
                    name="Emission Strength",
                    in_out="INPUT",
                    socket_type="NodeSocketFloat",
                )
                emission_strength_socket = _output_socket(group_input, "Emission Strength")
            except Exception:
                emission_strength_socket = None

        use_vert_col_socket = _output_socket(group_input, "Use Vert Col")
        if use_vert_col_socket is None:
            try:
                group.interface.new_socket(
                    name="Use Vert Col",
                    in_out="INPUT",
                    socket_type="NodeSocketBool",
                )
                use_vert_col_socket = _output_socket(group_input, "Use Vert Col")
            except Exception:
                use_vert_col_socket = None

        decal_source_socket = _output_socket(group_input, "TexSlot1_DecalSource")
        apply_vc_tint = nodes.get("Apply VC Tint")
        emission_color_input = _input_socket(principled, "Emission Color", "Emission")
        emission_strength_input = _input_socket(principled, "Emission Strength")
        if (
            decal_source_socket is None
            or emission_color_input is None
            or emission_strength_socket is None
            or emission_strength_input is None
        ):
            return

        apply_vc_tint_factor = _input_socket(apply_vc_tint, "Factor") if apply_vc_tint is not None else None
        apply_vc_tint_result = _output_socket(apply_vc_tint, "Result") if apply_vc_tint is not None else None
        if use_vert_col_socket is not None and apply_vc_tint_factor is not None:
            for link in list(apply_vc_tint_factor.links):
                if link.from_socket is use_vert_col_socket:
                    continue
                links.remove(link)
            if not any(link.from_socket is use_vert_col_socket and link.to_socket is apply_vc_tint_factor for link in links):
                links.new(use_vert_col_socket, apply_vc_tint_factor)
        if apply_vc_tint_result is not None:
            for link in list(emission_color_input.links):
                if link.from_socket is apply_vc_tint_result:
                    continue
                links.remove(link)
            if not any(link.from_socket is apply_vc_tint_result and link.to_socket is emission_color_input for link in links):
                links.new(apply_vc_tint_result, emission_color_input)
        else:
            for link in list(emission_color_input.links):
                if link.from_socket is decal_source_socket:
                    continue
                links.remove(link)
            if not any(link.from_socket is decal_source_socket and link.to_socket is emission_color_input for link in links):
                links.new(decal_source_socket, emission_color_input)

        for link in list(emission_strength_input.links):
            if link.from_socket.name == "Value":
                continue
            links.remove(link)

        multiplier = nodes.get("Emission Strength x10")
        if multiplier is None or multiplier.bl_idname != "ShaderNodeMath":
            multiplier = nodes.new("ShaderNodeMath")
            multiplier.name = "Emission Strength x10"
            multiplier.label = "Emission Strength x10"
            multiplier.location = (principled.location.x - 240, principled.location.y - 220)
            multiplier.operation = "MULTIPLY"
            multiplier.inputs[1].default_value = 10.0
        if not any(link.from_socket is emission_strength_socket and link.to_socket is multiplier.inputs[0] for link in links):
            for link in list(multiplier.inputs[0].links):
                links.remove(link)
            links.new(emission_strength_socket, multiplier.inputs[0])
        if not any(link.from_socket is multiplier.outputs[0] and link.to_socket is emission_strength_input for link in links):
            for link in list(emission_strength_input.links):
                links.remove(link)
            links.new(multiplier.outputs[0], emission_strength_input)

        # Stencil breakup alpha routing (inside SB_MeshDecal_v1):
        # alpha = base_alpha * mix(1.0, grime_red, clamp(WearBlendFalloff * 2.0))
        grime_breakup_socket = _output_socket(group_input, "TexSlot8_GrimeBreakup")
        wear_falloff_socket = _output_socket(group_input, "Param_WearBlendFalloff")
        base_alpha_node = nodes.get("Maths.008")
        principled_alpha = _input_socket(principled, "Alpha")
        base_alpha_socket = _output_socket(base_alpha_node, "Value") if base_alpha_node is not None else None
        if (
            grime_breakup_socket is not None
            and wear_falloff_socket is not None
            and base_alpha_socket is not None
            and principled_alpha is not None
        ):
            separate = nodes.get("SB Stencil Breakup Separate")
            if separate is None or separate.bl_idname != "ShaderNodeSeparateColor":
                separate = nodes.new("ShaderNodeSeparateColor")
                separate.name = "SB Stencil Breakup Separate"
            separate.label = "Stencil Breakup Channel"
            separate.location = (base_alpha_node.location.x - 420, base_alpha_node.location.y - 280)

            strength_scale = nodes.get("SB Stencil Breakup Strength x2")
            if strength_scale is None or strength_scale.bl_idname != "ShaderNodeMath":
                strength_scale = nodes.new("ShaderNodeMath")
                strength_scale.name = "SB Stencil Breakup Strength x2"
            strength_scale.label = "Breakup Strength x2 (Clamp)"
            strength_scale.location = (base_alpha_node.location.x - 420, base_alpha_node.location.y - 120)
            strength_scale.operation = "MULTIPLY"
            strength_scale.use_clamp = True
            strength_scale.inputs[1].default_value = 2.0

            breakup_mix = nodes.get("SB Stencil Breakup Mix")
            if breakup_mix is None or breakup_mix.bl_idname != "ShaderNodeMix":
                breakup_mix = nodes.new("ShaderNodeMix")
                breakup_mix.name = "SB Stencil Breakup Mix"
            breakup_mix.label = "Stencil Breakup Strength"
            breakup_mix.location = (base_alpha_node.location.x - 180, base_alpha_node.location.y - 280)
            if hasattr(breakup_mix, "data_type"):
                breakup_mix.data_type = "FLOAT"
            breakup_mix.inputs[2].default_value = 1.0

            alpha_mul = nodes.get("SB Stencil Alpha Wear Multiply")
            if alpha_mul is None or alpha_mul.bl_idname != "ShaderNodeMath":
                alpha_mul = nodes.new("ShaderNodeMath")
                alpha_mul.name = "SB Stencil Alpha Wear Multiply"
            alpha_mul.label = "Alpha x Breakup"
            alpha_mul.location = (base_alpha_node.location.x + 220, base_alpha_node.location.y - 120)
            alpha_mul.operation = "MULTIPLY"

            for link in list(separate.inputs[0].links):
                links.remove(link)
            links.new(grime_breakup_socket, separate.inputs[0])
            for link in list(strength_scale.inputs[0].links):
                links.remove(link)
            links.new(wear_falloff_socket, strength_scale.inputs[0])
            for link in list(breakup_mix.inputs[0].links):
                links.remove(link)
            links.new(strength_scale.outputs[0], breakup_mix.inputs[0])
            for link in list(breakup_mix.inputs[3].links):
                links.remove(link)
            links.new(separate.outputs["Red"], breakup_mix.inputs[3])
            for link in list(alpha_mul.inputs[0].links):
                links.remove(link)
            links.new(base_alpha_socket, alpha_mul.inputs[0])
            for link in list(alpha_mul.inputs[1].links):
                links.remove(link)
            links.new(breakup_mix.outputs[0], alpha_mul.inputs[1])
            for link in list(principled_alpha.links):
                links.remove(link)
            links.new(alpha_mul.outputs[0], principled_alpha)

        group["starbreaker_mesh_decal_emission_patch_version"] = 4

    #: Decoded F0 range over which a MeshDecal texel turns from dielectric to
    #: metal. Dielectrics author F0 0.02-0.05 and metals >= 0.5; component POM
    #: atlases sit up to ~0.45 (sRGB ~186) and must stay dielectric.
    _MESH_DECAL_METAL_F0_RANGE = (0.40, 0.55)
    _MESH_DECAL_METAL_PATCH_VERSION = 1

    @classmethod
    def _patch_mesh_decal_template_metal(cls, group: bpy.types.ShaderNodeTree) -> None:
        """Render metal texels of a MeshDecal atlas as metal.

        CryEngine encodes metal as near-black diffuse plus a bright sRGB spec
        (F0) map. The template only fed the spec map into ``Specular IOR
        Level`` (scaled by the host's dielectric specular), so the dark
        polished-metal tiles of e.g. ``rsi_pom_B`` (diffuse ~0.03, F0 ~0.7)
        rendered as matte black patches on the Constellation hull.

        Texels whose decoded F0 passes ``_MESH_DECAL_METAL_F0_RANGE`` now get
        that F0 as base colour, full metallic and the decal's own roughness
        (``Decal Roughness``; -1 = unset, falls back to ``Host Roughness``).
        """
        if group.get("starbreaker_mesh_decal_metal_patch_version") == cls._MESH_DECAL_METAL_PATCH_VERSION:
            return

        nodes = group.nodes
        links = group.links
        group_input = next((node for node in nodes if node.bl_idname == "NodeGroupInput"), None)
        principled = nodes.get("Principled BSDF")
        host_tint = nodes.get("Apply Host Tint")
        roughness_mix = nodes.get("Mix.001")
        if group_input is None or principled is None or host_tint is None or roughness_mix is None:
            return

        if _output_socket(group_input, "Decal Roughness") is None:
            try:
                group.interface.new_socket(name="Decal Roughness", in_out="INPUT", socket_type="NodeSocketFloat")
            except Exception:
                return
        for item in group.interface.items_tree:
            if getattr(item, "item_type", "") == "SOCKET" and item.in_out == "INPUT" and item.name == "Decal Roughness":
                item.default_value = -1.0
                item.min_value = -1.0
                item.max_value = 1.0

        spec_socket = _output_socket(group_input, "TexSlot2_Specular")
        decal_roughness_socket = _output_socket(group_input, "Decal Roughness")
        host_roughness_socket = _output_socket(group_input, "Host Roughness")
        base_color_input = _input_socket(principled, "Base Color")
        metallic_input = _input_socket(principled, "Metallic")
        roughness_input = _input_socket(principled, "Roughness")
        if None in (
            spec_socket,
            decal_roughness_socket,
            host_roughness_socket,
            base_color_input,
            metallic_input,
            roughness_input,
        ):
            return

        def _node(name: str, bl_idname: str, label: str, location: tuple[float, float]) -> bpy.types.Node:
            node = nodes.get(name)
            if node is not None and node.bl_idname != bl_idname:
                nodes.remove(node)
                node = None
            if node is None:
                node = nodes.new(bl_idname)
                node.name = name
            node.label = label
            node.location = location
            return node

        def _link(from_socket: Any, to_socket: Any) -> None:
            for link in list(to_socket.links):
                links.remove(link)
            links.new(from_socket, to_socket)

        x, y = principled.location.x - 480, principled.location.y + 360

        # The spec image is sampled as Non-Color; CryEngine stores it in sRGB.
        f0 = _node("SB Decal Metal F0", "ShaderNodeGamma", "Spec sRGB -> F0", (x - 480, y))
        f0.inputs["Gamma"].default_value = 2.2
        _link(spec_socket, f0.inputs["Color"])

        f0_luma = _node("SB Decal Metal F0 Luma", "ShaderNodeRGBToBW", "F0 Luma", (x - 300, y))
        _link(f0.outputs[0], f0_luma.inputs[0])

        mask = _node("SB Decal Metal Mask", "ShaderNodeMapRange", "Metal Mask", (x - 120, y))
        mask.clamp = True
        mask.inputs["From Min"].default_value = cls._MESH_DECAL_METAL_F0_RANGE[0]
        mask.inputs["From Max"].default_value = cls._MESH_DECAL_METAL_F0_RANGE[1]
        mask.inputs["To Min"].default_value = 0.0
        mask.inputs["To Max"].default_value = 1.0
        _link(f0_luma.outputs[0], mask.inputs["Value"])
        mask_socket = mask.outputs["Result"]

        base_mix = _node("SB Decal Metal Base Color", "ShaderNodeMix", "Metal Base Color", (x + 120, y))
        base_mix.data_type = "RGBA"
        base_mix.blend_type = "MIX"
        _link(mask_socket, base_mix.inputs[0])
        _link(host_tint.outputs[2], base_mix.inputs[6])
        _link(f0.outputs[0], base_mix.inputs[7])
        _link(base_mix.outputs[2], base_color_input)
        _link(mask_socket, metallic_input)

        unset = _node("SB Decal Roughness Unset", "ShaderNodeMath", "Decal Roughness Unset", (x - 120, y - 200))
        unset.operation = "LESS_THAN"
        unset.inputs[1].default_value = 0.0
        _link(decal_roughness_socket, unset.inputs[0])

        own_rough = _node("SB Decal Own Roughness", "ShaderNodeMix", "Decal Own Roughness", (x + 120, y - 200))
        own_rough.data_type = "FLOAT"
        _link(unset.outputs[0], own_rough.inputs[0])
        _link(decal_roughness_socket, own_rough.inputs[2])
        _link(host_roughness_socket, own_rough.inputs[3])

        rough_mix = _node("SB Decal Metal Roughness", "ShaderNodeMix", "Metal Roughness", (x + 300, y - 200))
        rough_mix.data_type = "FLOAT"
        _link(mask_socket, rough_mix.inputs[0])
        _link(roughness_mix.outputs[0], rough_mix.inputs[2])
        _link(own_rough.outputs[0], rough_mix.inputs[3])
        _link(rough_mix.outputs[0], roughness_input)

        group["starbreaker_mesh_decal_metal_patch_version"] = cls._MESH_DECAL_METAL_PATCH_VERSION

    def _build_contract_group_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
        group_contract: ShaderGroupContract,
    ) -> bool:
        group_tree = self._ensure_contract_group(group_contract)
        if group_tree is None:
            return False

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (700, 0)
        group_node = nodes.new("ShaderNodeGroup")
        group_node.node_tree = group_tree
        group_node.location = (220, 0)

        shader_output = _output_socket(group_node, group_contract.shader_output)
        if shader_output is None:
            return False
        surface_shader = shader_output

        y = 280
        for contract_input in group_contract.inputs:
            target_socket = _input_socket(group_node, contract_input.name)
            if target_socket is None:
                continue
            semantic = (contract_input.semantic or contract_input.name).lower()
            if "disable" in semantic and "shadow" in semantic:
                if hasattr(target_socket, "default_value"):
                    target_socket.default_value = bool(self._plan_casts_no_shadows(plan, submaterial))
                source_socket = None
            elif semantic == "emission_strength" and hasattr(target_socket, "default_value"):
                if group_contract.name == "SB_MeshDecal_v1":
                    target_socket.default_value = self._mesh_decal_emission_strength(submaterial)
                else:
                    target_socket.default_value = self._illum_emission_strength(submaterial)
                source_socket = None
            elif semantic == "use_vertex_colors" and hasattr(target_socket, "default_value"):
                target_socket.default_value = _uses_vertex_color_tint(submaterial)
                source_socket = None
            elif semantic.startswith("public_param_"):
                # Generic authored-param default: the group input's semantic
                # is ``public_param_<lowercase param name>`` and the value
                # comes directly from ``submaterial.public_params`` (matched
                # case-insensitively). Scalars use the socket default_value
                # verbatim; the socket keeps its authored default when the
                # submaterial does not set the param.
                param_key = semantic.removeprefix("public_param_")
                if hasattr(target_socket, "default_value"):
                    resolved = _resolve_public_param_default(submaterial, param_key)
                    if (
                        resolved is None
                        and group_contract.name == "SB_MeshDecal_v1"
                        and template_plan_for_submaterial(submaterial).template_key == "decal_stencil"
                        and submaterial.decoded_feature_flags.has_stencil_map
                        and param_key == "decaldiffuseopacity"
                    ):
                        resolved = _resolve_public_param_default(submaterial, "stencilopacity")
                    if (
                        resolved is None
                        and group_contract.name == "SB_MeshDecal_v1"
                        and template_plan_for_submaterial(submaterial).template_key == "decal_stencil"
                        and submaterial.decoded_feature_flags.has_stencil_map
                        and param_key == "wearblendfalloff"
                    ):
                        # The group's breakup strength is WearBlendFalloff x 2;
                        # a stencil authors it as StencilDiffuseBreakup. At the
                        # group's default (0.5, full strength) the Hercules
                        # emblems (breakup 0.5 over a grunge mask) came out
                        # moth-eaten at ~60% opacity.
                        breakup = _resolve_public_param_default(submaterial, "stencildiffusebreakup")
                        if isinstance(breakup, (int, float)):
                            resolved = 0.5 * float(breakup)
                    if (
                        group_contract.name == "SB_MeshDecal_v1"
                        and param_key == "decalfalloff"
                        and isinstance(resolved, (int, float))
                    ):
                        # The template multiplies alpha by DecalFalloff, but it is the
                        # game's angle falloff, not an opacity: Polaris Decal_Dirt (128)
                        # turned faint dirt runs into opaque black rectangles.
                        resolved = min(float(resolved), 1.0)
                    if param_key == "damageperobjectwear":
                        # The authored value scales the object's wear state,
                        # which is 0 on a new item. Taken as the wear itself it
                        # mixed every MeshDecal 50% towards the white wear
                        # colour: the ATLS "08" plate printed light grey, not
                        # black, and the Nine Tails oni mask a white patch.
                        resolved = 0.0
                    if resolved is not None:
                        try:
                            target_socket.default_value = resolved
                        except Exception:
                            pass
                source_socket = None
            elif semantic == "host_tint":
                # Option E: wire the package palette's ``Decal Color``
                # output into the decal group's Host Tint input so decals
                # participate in livery-driven tinting. Leaves the default
                # white when the palette does not author real decal colour
                # data (``Decal Color`` output socket unlinked inside the
                # palette group), to avoid blackening decals on packages
                # without a decal palette layer.
                #
                # POM decals are the exception: they are projected onto a
                # host paint surface (primary/secondary/tertiary) and the
                # ``_rebind_mesh_decal_for_host`` pass produces cloned
                # ``__host_<channel>`` materials that wire ``Host Tint``
                # directly to the host channel colour. If we pre-wire the
                # base POM-decal material to ``Decal Color`` here, any
                # mesh the rebinder fails to pair with a host channel
                # still ends up tinted by the palette decal texture
                # rather than falling back to white. Skip the default
                # wiring for POM decals so unmatched hosts stay neutral.
                source_socket = None
                is_pom_decal = bool(
                    submaterial.decoded_feature_flags.has_parallax_occlusion_mapping
                )
                has_palette_routing = (
                    submaterial.palette_routing is not None
                    and submaterial.palette_routing.material_channel is not None
                )
                if (
                    palette is not None
                    and hasattr(self, "_palette_group_node")
                    and not is_pom_decal
                    and has_palette_routing
                ):
                    try:
                        palette_node = self._palette_group_node(nodes, links, palette, x=-420, y=y)
                    except Exception:
                        palette_node = None
                    if palette_node is not None:
                        palette_tree = getattr(palette_node, "node_tree", None)
                        provides_decal = False
                        if palette_tree is not None:
                            for subnode in palette_tree.nodes:
                                if subnode.type == "GROUP_OUTPUT":
                                    decal_input = subnode.inputs.get("Decal Color")
                                    if decal_input is not None and decal_input.is_linked:
                                        provides_decal = True
                                    break
                        if provides_decal:
                            source_socket = _output_socket(palette_node, "Decal Color")
            else:
                if (
                    group_contract.name == "SB_HardSurface_v1"
                    and semantic == "base_color"
                    and hasattr(target_socket, "default_value")
                ):
                    target_socket.default_value = (1.0, 1.0, 1.0, 1.0)
                elif (
                    group_contract.name == "SB_HardSurface_v1"
                    and semantic == "base_color_alpha"
                    and hasattr(target_socket, "default_value")
                ):
                    target_socket.default_value = 1.0
                elif ("alpha" in semantic or "opacity" in semantic) and hasattr(target_socket, "default_value"):
                    target_socket.default_value = 0.0
                elif (
                    group_contract.name == "SB_MeshDecal_v1"
                    and contract_input.name == "TexSlot1_DecalSource"
                    and template_plan_for_submaterial(submaterial).template_key == "decal_stencil"
                    and submaterial.decoded_feature_flags.has_stencil_map
                    and hasattr(target_socket, "default_value")
                ):
                    target_socket.default_value = (1.0, 1.0, 1.0, 1.0)
                source_socket = self._contract_input_source_socket(
                    nodes,
                    submaterial,
                    palette,
                    group_contract,
                    contract_input,
                    x=-220,
                    y=y,
                )
                if source_socket is not None:
                    source_socket = self._scaled_mesh_decal_source(
                        nodes,
                        links,
                        submaterial,
                        group_contract,
                        contract_input,
                        source_socket,
                        x=-60,
                        y=y,
                    )
            if source_socket is not None:
                links.new(source_socket, target_socket)
            else:
                neutral_breakup = _mesh_decal_neutral_breakup_default(
                    group_contract,
                    submaterial,
                    contract_input,
                    source_socket,
                )
                if neutral_breakup is not None and hasattr(target_socket, "default_value"):
                    target_socket.default_value = neutral_breakup
                elif "normal" in semantic and hasattr(target_socket, "default_value"):
                    target_socket.default_value = (0.5, 0.5, 1.0, 1.0)
            y -= 180

        group_handles_alpha = any(
            (contract_input.semantic or contract_input.name).lower() in {"alpha", "opacity"}
            or "alpha" in (contract_input.semantic or contract_input.name).lower()
            or "opacity" in (contract_input.semantic or contract_input.name).lower()
            for contract_input in group_contract.inputs
        )

        group_handles_emission = any(
            (contract_input.semantic or contract_input.name).lower() in {"emission_strength", "emission_color"}
            or "emission" in (contract_input.semantic or contract_input.name).lower()
            for contract_input in group_contract.inputs
        )

        if plan.uses_alpha and not group_handles_alpha:
            alpha_source = self._alpha_source_socket(
                nodes,
                submaterial,
                representative_textures(submaterial),
                x=-220,
                y=y,
            )
            if alpha_source is not None:
                transparent = nodes.new("ShaderNodeBsdfTransparent")
                transparent.location = (400, -180)
                mix = nodes.new("ShaderNodeMixShader")
                mix.location = (560, 0)
                links.new(alpha_source, mix.inputs[0])
                links.new(transparent.outputs[0], mix.inputs[1])
                links.new(surface_shader, mix.inputs[2])
                surface_shader = mix.outputs[0]

        emissive_triplet = _authored_emissive_triplet(submaterial)
        emissive_strength = max(
            _mean_triplet(emissive_triplet) or 0.0,
            _float_authored_attribute(submaterial, "Glow"),
        ) * _emissive_opacity_factor(submaterial)
        if emissive_strength > 0.0 and not group_handles_emission:
            emissive_color_path = self._texture_export_path(submaterial, "emissive") or representative_textures(
                submaterial
            ).get("base_color")
            emissive_color_node = self._image_node(nodes, emissive_color_path, x=-220, y=y, is_color=True)
            emissive_color_output = emissive_color_node.outputs[0] if emissive_color_node is not None else None
            if emissive_color_output is None and emissive_triplet is not None:
                emissive_color_output = self._value_color_socket(
                    nodes,
                    (*emissive_triplet, 1.0),
                    x=-220,
                    y=y,
                )
            if emissive_color_output is not None:
                emission = nodes.new("ShaderNodeEmission")
                emission.location = (400, -300)
                self._link_color_output(emissive_color_output, emission.inputs["Color"])
                emission.inputs["Strength"].default_value = emissive_strength
                add_shader = nodes.new("ShaderNodeAddShader")
                add_shader.location = (560, -40)
                links.new(surface_shader, add_shader.inputs[0])
                links.new(emission.outputs[0], add_shader.inputs[1])
                surface_shader = add_shader.outputs[0]

        # Mesh decals float up to a few cm over the hull (the F8C's livery
        # stripes sit 2 cm up); in game they are blended into the surface and
        # cast nothing, so keep them out of shadow rays like the other builders.
        self._wire_surface_shader_to_output(nodes, links, surface_shader, output, plan, submaterial)

        if submaterial.decoded_feature_flags.has_parallax_occlusion_mapping:
            # Phase 12 (POM plan, Phase 2 extension): contract-group
            # materials (notably MeshDecal, which ships authored height
            # samples in ``TexSlot4_Height``) get the same parallax
            # treatment as HardSurface. Find the tex image feeding the
            # height/displacement input of the group, then route all
            # other tex images feeding the group through the shared
            # ``StarBreaker Runtime Parallax`` node.
            height_node: bpy.types.Node | None = None
            targets: list[bpy.types.Node] = []
            group_node_name = group_node.name
            for link in material.node_tree.links:
                if link.to_node.name != group_node_name or link.from_node.bl_idname != "ShaderNodeTexImage":
                    continue
                socket_name = link.to_socket.name.lower()
                if "height" in socket_name or "displacement" in socket_name:
                    height_node = link.from_node
                elif all(t.name != link.from_node.name for t in targets):
                    targets.append(link.from_node)
            if height_node is not None and targets:
                pom_scale = _float_public_param(
                    submaterial,
                    "PomDisplacement",
                    "POMHeightBias",
                    "POM_HeightBias",
                    "POMDisplacement",
                )
                if pom_scale is None or pom_scale <= 0.0:
                    pom_scale = 0.05
                pom_scale = min(0.2, pom_scale)
                self._wire_runtime_parallax(
                    material,
                    height_node=height_node,
                    target_image_nodes=targets,
                    scale_value=pom_scale,
                    bias_value=self._parallax_bias_value(submaterial),
                    location=(-760, 320),
                )

        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)
        return True

    def _build_hard_surface_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        # Phase 8 notes on public params that cannot be mapped here:
        #   * SelfShadowStrength — no matching socket in the HardSurface
        #     runtime group today; adding one requires a group schema bump
        #     plus re-plumbing the self-shadow path. Deferred.
        #   * DamageTiling — there is no dedicated damage map sampler at
        #     this level; damage is composited via
        #     ``_layered_damage_factor_socket``. Deferred until a damage
        #     texture is sampled here directly.
        #   * FarGlowStartDistance / FarGlowEndDistance / FarGlowMultiplier
        #     — distance-based emissive falloff is a CryEngine post-process
        #     / HDR feature with no direct Blender shader equivalent. Not
        #     mapped intentionally.
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (700, 0)

        top_base = _submaterial_texture_reference(submaterial, slots=("TexSlot1",), roles=("base_color", "diffuse"))
        top_base_node = self._image_node(nodes, top_base.export_path if top_base is not None else None, x=-720, y=520, is_color=True)
        top_base_color = top_base_node.outputs[0] if top_base_node is not None else None
        # Diffuse-texture alpha channels in CryEngine are frequently
        # repurposed (gloss, detail mask, height) rather than opacity.
        # Only wire them to the shader's alpha inputs when the
        # material template explicitly opts into alpha handling.
        top_base_alpha = (
            _output_socket(top_base_node, "Alpha")
            if (top_base_node is not None and plan.uses_alpha)
            else None
        )
        material_channel = submaterial.palette_routing.material_channel.name if submaterial.palette_routing.material_channel is not None else None
        authored_angle_shift = _hard_surface_angle_shift_enabled(submaterial)
        palette_angle_shift_channel = _hard_surface_palette_iridescence_channel(
            palette,
            material_channel,
            authored_angle_shift=authored_angle_shift,
        )
        angle_shift_enabled = authored_angle_shift or (palette_angle_shift_channel is not None)
        iridescence_channel = palette_angle_shift_channel or "tertiary"

        primary_layer = submaterial.layer_manifest[0] if submaterial.layer_manifest else None
        secondary_layer = submaterial.layer_manifest[1] if len(submaterial.layer_manifest) > 1 else None
        primary = self._connect_manifest_layer_surface_group(
            nodes,
            links,
            submaterial,
            primary_layer,
            palette,
            x=-240,
            y=240,
            label="Primary Layer",
            detail_slots=("TexSlot7", "TexSlot13", "TexSlot6"),
            wire_diffuse_alpha=plan.uses_alpha,
        )
        secondary = self._connect_manifest_layer_surface_group(
            nodes,
            links,
            submaterial,
            secondary_layer,
            palette,
            x=-240,
            y=-120,
            label="Secondary Layer",
            detail_slots=("TexSlot7", "TexSlot13", "TexSlot6"),
            wire_diffuse_alpha=plan.uses_alpha,
        )
        wear_factor = self._layered_wear_factor_socket(nodes, links, submaterial, x=-720, y=-120)
        damage_factor = self._layered_damage_factor_socket(nodes, links, submaterial, x=-720, y=-240)
        iridescence_ramp_color = self._iridescence_ramp_color_socket(nodes, links, submaterial, x=-980, y=-1560)
        stencil = self._hard_surface_stencil_overlay_sockets(nodes, links, submaterial, x=-980, y=-1820)
        self._link_hard_surface_stencil_paint(nodes, links, submaterial, stencil, x=-700, y=520, palette=palette)

        macro_normal_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot3",), roles=("normal_gloss",))
        macro_normal_node = self._image_node(
            nodes,
            macro_normal_ref.export_path if macro_normal_ref is not None else None,
            x=-720,
            y=-420,
            is_color=False,
        )
        displacement_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot6",), roles=("height", "displacement"))
        displacement_node = self._image_node(
            nodes,
            displacement_ref.export_path if displacement_ref is not None else None,
            x=-720,
            y=-720,
            is_color=False,
        )
        emissive_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot14",))
        emissive_node = self._image_node(
            nodes,
            emissive_ref.export_path if emissive_ref is not None else None,
            x=-720,
            y=-1020,
            is_color=True,
        )

        # Phase 8: public-param UV tiling. MacroTiling scales the detail
        # macro-normal sampler; EmissiveTiling scales the emissive sampler.
        # Both are no-ops when the corresponding public param is absent or
        # equal to 1.0.
        macro_tiling = _float_public_param(submaterial, "MacroTiling", "MacroNormalTiling") or 1.0
        self._apply_uv_tiling(nodes, links, macro_normal_node, macro_tiling, x=-1040, y=-420)
        emissive_tiling = _float_public_param(submaterial, "EmissiveTiling") or 1.0
        self._apply_uv_tiling(nodes, links, emissive_node, emissive_tiling, x=-1040, y=-1020)

        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_hard_surface_group()
        _refresh_group_node_sockets(shader_group)
        shader_group.location = (140, 0)
        shader_group.label = "StarBreaker HardSurface"
        self._set_socket_default(_input_socket(shader_group, "Top Base Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Top Alpha"), 1.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Primary Alpha"), 1.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Roughness"), 0.45)
        self._set_socket_default(_input_socket(shader_group, "Primary Specular"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Specular Tint"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Primary Metallic"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Normal"), (0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Secondary Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Secondary Alpha"), 1.0)
        self._set_socket_default(_input_socket(shader_group, "Secondary Roughness"), 0.45)
        self._set_socket_default(_input_socket(shader_group, "Secondary Specular"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Secondary Specular Tint"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Secondary Metallic"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Secondary Normal"), (0.0, 0.0, 1.0))
        if angle_shift_enabled and palette is not None:
            # The palette encodes the shimmerscale pair with the two angle
            # endpoints swapped between the ``color`` and ``specular`` slots
            # of the tertiary entry (e.g. Aurora Mk II Shimmerscale stores
            # purple as ``tertiary.color`` and green as
            # ``tertiary.finish.specular``). Ground-truth screenshots show
            # the facing hit reading green and the grazing falloff reading
            # purple, so we feed the specular slot into Facing and the
            # color slot into Grazing.
            facing_socket = self._palette_specular_socket(
                nodes, palette, iridescence_channel, x=-720, y=-1320
            )
            grazing_socket = self._palette_color_socket(
                nodes, palette, iridescence_channel, x=-720, y=-1320
            )
            self._link_group_input(links, facing_socket, shader_group, "Iridescence Facing Color")
            self._link_group_input(links, grazing_socket, shader_group, "Iridescence Grazing Color")
        else:
            self._set_socket_default(_input_socket(shader_group, "Iridescence Facing Color"), (0.0, 0.0, 0.0, 1.0))
            self._set_socket_default(_input_socket(shader_group, "Iridescence Grazing Color"), (0.0, 0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Iridescence Ramp Color"), (0.0, 0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Iridescence Ramp Weight"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Iridescence Strength"), 1.0)
        iridescence_active = authored_angle_shift or (palette_angle_shift_channel is not None)
        self._set_socket_default(_input_socket(shader_group, "Iridescence Factor"), 1.0 if iridescence_active else 0.0)
        self._set_socket_default(_input_socket(shader_group, "Stencil Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "StencilDiffuseColor"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "StencilDiffuseColor2"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "StencilDiffuseColor3"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Stencil Tone Mode"), 0.0)
        self._link_group_input(links, iridescence_ramp_color, shader_group, "Iridescence Ramp Color")
        if iridescence_ramp_color is not None:
            self._set_socket_default(_input_socket(shader_group, "Iridescence Ramp Weight"), 1.0)
        iridescence_strength = _optional_float_public_param(submaterial, "IridescenceStrength")
        if iridescence_strength is not None and iridescence_strength > 0.0:
            self._set_socket_default(_input_socket(shader_group, "Iridescence Strength"), iridescence_strength)
        self._set_socket_default(_input_socket(shader_group, "Wear Factor"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Damage Factor"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Macro Normal Color"), (0.5, 0.5, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Macro Normal Strength"), 0.4)
        # Phase 8: read authored POM displacement from PomDisplacement /
        # POMHeightBias when %PARALLAX_OCCLUSION_MAPPING% is enabled; fall
        # back to 0.05 (the empirical bake-in value that matches non-POM
        # HardSurface materials).
        if submaterial.decoded_feature_flags.has_parallax_occlusion_mapping:
            pom_displacement = _float_public_param(
                submaterial,
                "PomDisplacement",
                "POMHeightBias",
                "POM_HeightBias",
                "POMDisplacement",
            )
            if pom_displacement is None or pom_displacement <= 0.0:
                pom_displacement = 0.08
            # A 0.2 ceiling prevents blowing out the geometry; no artificial floor
            # so very low authored values (e.g. 0.005 for subtle rubber tile) are
            # preserved rather than overridden.
            pom_displacement = min(0.2, pom_displacement)
        else:
            pom_displacement = 0.05
        self._set_socket_default(_input_socket(shader_group, "Displacement Strength"), pom_displacement)

        # Phase 12 (POM plan, Phase 2): when
        # ``%PARALLAX_OCCLUSION_MAPPING%`` is on and we have an authored
        # displacement sample to read, inject the
        # ``StarBreaker Runtime Parallax`` group between the UV source and
        # the base-colour sampler so the diffuse lookup reads from the
        # parallax-offset coordinates. TexSlot3 (macro normal) and
        # TexSlot14 (emissive) already have their own UV-mapping chain
        # via ``_apply_uv_tiling`` (public-param tiling) — we leave those
        # alone for now to avoid double-driving the ``Vector`` socket.
        # The height sample is taken at the *base* UV (not the offset
        # one) to match the standard offset-mapping algorithm.
        if (
            submaterial.decoded_feature_flags.has_parallax_occlusion_mapping
            and displacement_node is not None
            and top_base_node is not None
        ):
            self._wire_runtime_parallax(
                material,
                height_node=displacement_node,
                target_image_nodes=[top_base_node],
                scale_value=pom_displacement,
                bias_value=self._parallax_bias_value(submaterial),
            )
        # Phase 12 (POM follow-up): some HardSurface materials have no
        # top-level TexSlot6 displacement but DO carry a height map inside
        # their ``layer_manifest[0].texture_slots`` (TexSlot3 tagged with
        # the misleading ``alternate_base_color`` role, filename ending
        # in ``_displ``). This is the pattern used by tileable surfaces
        # like ``rsi_aurora_mk2:Tile_Grill_A``. Load the Primary-layer
        # height on demand and route parallax into the Primary layer's
        # base-colour + normal_gloss samplers. Without this 27 Aurora
        # Mk2 POM-flagged materials would render flat despite authored
        # height data existing in the sidecar.
        if (
            submaterial.decoded_feature_flags.has_parallax_occlusion_mapping
            and displacement_node is None
            and primary_layer is not None
        ):
            layer_height_ref = _layer_texture_reference(primary_layer, slots=("TexSlot3",))
            if (
                layer_height_ref is not None
                and layer_height_ref.export_path
                and "_displ" in (layer_height_ref.source_path or "").lower()
            ):
                layer_height_node = self._image_node(
                    nodes,
                    layer_height_ref.export_path,
                    x=-1480,
                    y=-720,
                    is_color=False,
                    reuse_any_existing=True,
                )
                if layer_height_node is not None and layer_height_node.image is not None:
                    layer_targets: list[bpy.types.ShaderNodeTexImage] = []
                    layer_base_ref = _layer_texture_reference(
                        primary_layer,
                        slots=("TexSlot1",),
                        roles=("base_color", "diffuse"),
                    )
                    layer_normal_ref = _layer_texture_reference(
                        primary_layer,
                        roles=("normal_gloss",),
                        alpha_semantic="smoothness",
                    )
                    for ref in (layer_base_ref, layer_normal_ref):
                        if ref is None or not ref.export_path:
                            continue
                        resolved = self.package.resolve_path(ref.export_path)
                        if resolved is None:
                            continue
                        resolved_str = str(resolved)
                        for node in nodes:
                            if node.bl_idname != "ShaderNodeTexImage" or node.image is None:
                                continue
                            if node is layer_height_node:
                                continue
                            if (
                                bpy.path.abspath(
                                    node.image.filepath, library=node.image.library
                                )
                                == resolved_str
                                and node not in layer_targets
                            ):
                                # Collect every sampler that resolves to this layer
                                # texture path (both primary + wear layer copies).
                                layer_targets.append(node)
                    if layer_targets:
                        self._wire_runtime_parallax(
                            material,
                            height_node=layer_height_node,
                            target_image_nodes=layer_targets,
                            scale_value=pom_displacement,
                            bias_value=self._parallax_bias_value(submaterial),
                            uv_tile=primary_layer.uv_tiling if primary_layer.uv_tiling is not None else 1.0,
                        )
        self._set_socket_default(_input_socket(shader_group, "Emission Color"), (0.0, 0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Emission Strength"), 0.0)
        shader_group["starbreaker_angle_shift_enabled"] = angle_shift_enabled
        shader_group["starbreaker_angle_shift_channel"] = iridescence_channel if angle_shift_enabled else ""

        self._link_group_input(links, top_base_color, shader_group, "Top Base Color")
        self._link_group_input(links, top_base_alpha, shader_group, "Top Alpha")
        self._link_group_input(links, primary.color, shader_group, "Primary Color")
        self._link_group_input(links, primary.alpha, shader_group, "Primary Alpha")
        self._link_group_input(links, primary.roughness, shader_group, "Primary Roughness")
        self._link_group_input(links, primary.specular, shader_group, "Primary Specular")
        self._link_group_input(links, primary.specular_tint, shader_group, "Primary Specular Tint")
        self._link_group_input(links, primary.metallic, shader_group, "Primary Metallic")
        self._link_group_input(links, primary.normal, shader_group, "Primary Normal")
        if secondary_layer is not None:
            self._link_group_input(links, secondary.color, shader_group, "Secondary Color")
            self._link_group_input(links, secondary.alpha, shader_group, "Secondary Alpha")
            self._link_group_input(links, secondary.roughness, shader_group, "Secondary Roughness")
            self._link_group_input(links, secondary.specular, shader_group, "Secondary Specular")
            self._link_group_input(links, secondary.specular_tint, shader_group, "Secondary Specular Tint")
            self._link_group_input(links, secondary.metallic, shader_group, "Secondary Metallic")
            self._link_group_input(links, secondary.normal, shader_group, "Secondary Normal")
        self._link_group_input(links, wear_factor, shader_group, "Wear Factor")
        self._link_group_input(links, damage_factor, shader_group, "Damage Factor")
        self._link_group_input(links, stencil.color, shader_group, "Stencil Color")
        self._set_socket_default(
            _input_socket(shader_group, "StencilDiffuseColor"),
            (*stencil.stencil_diffuse_color, 1.0),
        )
        self._set_socket_default(
            _input_socket(shader_group, "StencilDiffuseColor2"),
            (*stencil.stencil_diffuse_color_2, 1.0),
        )
        self._set_socket_default(
            _input_socket(shader_group, "StencilDiffuseColor3"),
            (*stencil.stencil_diffuse_color_3, 1.0),
        )
        self._set_socket_default(_input_socket(shader_group, "Stencil Tone Mode"), float(stencil.tone_mode))
        self._link_group_input(
            links,
            macro_normal_node.outputs[0] if macro_normal_node is not None else None,
            shader_group,
            "Macro Normal Color",
        )
        self._link_group_input(
            links,
            displacement_node.outputs[0] if displacement_node is not None else None,
            shader_group,
            "Displacement Height",
        )
        authored_emissive = _authored_emissive_triplet(submaterial)
        emission_color_source = emissive_node.outputs[0] if emissive_node is not None else None
        if emission_color_source is not None and authored_emissive is not None:
            tint = nodes.new("ShaderNodeRGB")
            tint.label = "Authored Emissive Tint"
            tint.location = (-460, -1220)
            tint.outputs[0].default_value = (*authored_emissive, 1.0)
            multiply = nodes.new("ShaderNodeMixRGB")
            multiply.label = "Emissive Texture x Authored Tint"
            multiply.location = (-220, -1100)
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1.0
            links.new(emission_color_source, multiply.inputs[1])
            links.new(tint.outputs[0], multiply.inputs[2])
            emission_color_source = multiply.outputs[0]
        self._link_group_input(
            links,
            emission_color_source,
            shader_group,
            "Emission Color",
        )
        if emissive_node is not None:
            self._set_socket_default(_input_socket(shader_group, "Emission Strength"), 0.0)
        else:
            if authored_emissive is not None and any(abs(component) > 1e-6 for component in authored_emissive):
                self._set_socket_default(
                    _input_socket(shader_group, "Emission Color"),
                    (*authored_emissive, 1.0),
                )
                self._set_socket_default(
                    _input_socket(shader_group, "Emission Strength"),
                    max(_float_authored_attribute(submaterial, "Glow"), 1.0),
                )

        surface_shader = _output_socket(shader_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, surface_shader, output, plan, submaterial)
        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

    def _illum_spec_map_metal(
        self, nodes: Any, links: Any, submaterial: SubmaterialRecord, base_color: Any, specular: Any
    ) -> tuple[Any, Any, Any]:
        """Where an Illum specular map (times the authored Specular) is a
        metal's F0, its colour is the surface colour.

        The LuxCore's heat-sink plate has a dark tan diffuse under a golden
        spec map (213/190/131 x 0.479): brass in game. Fed only the map's
        luma as a partial metallic, it kept the tan and mirrored the studio
        grey-white. F0 above ~0.12 counts as metal, fully from 0.3 on;
        painted panels (Specular ~0.045) stay as they were.
        """
        if specular is None:
            return base_color, None, None
        path = self._texture_path_for_slot(submaterial, "TexSlot4")
        spec_node = self._image_node(nodes, path, x=-1000, y=900, is_color=False) if path else None
        if spec_node is None:
            return base_color, None, None
        authored = _authored_attribute_triplet(submaterial, "Specular")
        grey = sum(authored[:3]) / 3.0 if authored else 1.0
        if grey >= 0.99:
            grey = 1.0
        # The surface group multiplies a metal's colour by the Diffuse tint;
        # F0 is not the paint's, so that is divided back out.
        diffuse = _authored_attribute_triplet(submaterial, "Diffuse")
        tint = sum(diffuse[:3]) / 3.0 if diffuse else 1.0
        if tint > 0.05:
            grey = min(grey / tint, 1.0)
        colour = nodes.new("ShaderNodeMixRGB")
        colour.blend_type = "MULTIPLY"
        colour.location = (-760, 1000)
        colour.inputs[0].default_value = 1.0
        links.new(spec_node.outputs["Color"], colour.inputs[1])
        colour.inputs[2].default_value = (grey, grey, grey, 1.0)
        mask = nodes.new("ShaderNodeMapRange")
        mask.interpolation_type = "SMOOTHSTEP"
        mask.location = (-760, 820)
        mask.inputs["From Min"].default_value = 0.12
        mask.inputs["From Max"].default_value = 0.3
        links.new(specular, mask.inputs["Value"])
        mixed = nodes.new("ShaderNodeMixRGB")
        mixed.blend_type = "MIX"
        mixed.location = (-580, 900)
        links.new(mask.outputs["Result"], mixed.inputs[0])
        links.new(base_color, mixed.inputs[1])
        links.new(colour.outputs[0], mixed.inputs[2])
        return mixed.outputs[0], mask.outputs["Result"], colour.outputs[0]

    def _legacy_blend_layer_mask(self, nodes: Any, links: Any, submaterial: SubmaterialRecord) -> Any:
        """CryEngine 3 Illum %BLENDLAYER: the blend map (TexSlot12, red) times
        the vertex alpha, to the power BlendFalloff.

        The vertex colour node names the exporter's `Color` attribute: the
        meshes have no active colour attribute, so an unnamed node read alpha
        0 and the blend layer never showed (the Origin 300i stayed its dark
        primary layer instead of white). BlendFactor is not multiplied in:
        the authored values (5.6 on the 300i, 9.7 on the Gladiator's blacks)
        saturate every vertex to the blend layer, which turned the Gladiator
        black; mask x alpha alone matches the Fleetyards references of both
        (white 300i, dark olive Gladiator). The real engine curve is not
        published; this is the empirical fit."""
        mask_node = self._image_node(nodes, self._texture_path_for_slot(submaterial, "TexSlot12"), x=-1100, y=160, is_color=False)
        if mask_node is None:
            return None
        tiling = _float_public_param(submaterial, "BlendMaskTiling") or 1.0
        if abs(tiling - 1.0) > 1e-6:
            uv = nodes.new("ShaderNodeTexCoord")
            uv.location = (-1500, 160)
            scale = nodes.new("ShaderNodeVectorMath")
            scale.operation = "SCALE"
            scale.location = (-1300, 160)
            scale.inputs["Scale"].default_value = tiling
            links.new(uv.outputs["UV"], scale.inputs[0])
            links.new(scale.outputs[0], mask_node.inputs["Vector"])
        split = nodes.new("ShaderNodeSeparateColor")
        split.location = (-900, 160)
        links.new(mask_node.outputs["Color"], split.inputs[0])
        vertex = nodes.new("ShaderNodeVertexColor")
        vertex.location = (-900, 0)
        vertex.layer_name = "Color"
        factor = split.outputs[0]
        alpha_socket = vertex.outputs["Alpha"]
        for operation, value, location in (
            ("MULTIPLY", None, -700),
            ("POWER", _float_public_param(submaterial, "BlendFalloff") or 1.0, -380),
        ):
            math = nodes.new("ShaderNodeMath")
            math.operation = operation
            math.use_clamp = operation == "POWER"
            math.location = (location, 160)
            links.new(factor, math.inputs[0])
            if value is None:
                links.new(alpha_socket, math.inputs[1])
            else:
                math.inputs[1].default_value = value
            factor = math.outputs[0]
        return factor

    def _build_illum_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (700, 0)

        blend_mask_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot12",), roles=("wear_mask", "pattern_mask", "blend_mask"))
        blend_mask_node = self._image_node(
            nodes,
            blend_mask_ref.export_path if blend_mask_ref is not None else None,
            x=-720,
            y=160,
            is_color=False,
        )
        blend_mask_socket = blend_mask_node.outputs[0] if blend_mask_node is not None else None

        material_channel = submaterial.palette_routing.material_channel.name if submaterial.palette_routing.material_channel is not None else None
        # Legacy Illum without any tint palette (the Holoviewer_Ships models of
        # concepts): the colour is the authored Diffuse on the primary layer
        # and BlendLayer2DiffuseColor on the blend layer, as CryEngine's old
        # Illum multiplied them into the diffuse maps.
        legacy_tints = palette is None and material_channel is None
        # An Illum %BLENDLAYER whose submaterial has no palette routing is the
        # same old shader even when the package carries a default palette
        # (the Origin 300 series: hull_a..d, blend map dust_blend_diff,
        # BlendFactor 4.5-15): its blend layer is BlendLayer2DiffuseColor over
        # the mask, in game the layer that dominates. Without this the 300i
        # and 315p rendered as the bare primary layer, dark grey.
        authored_blend = material_channel is None and submaterial.decoded_feature_flags.has_blend_layer
        primary_tint = _authored_attribute_triplet(submaterial, "Diffuse") if legacy_tints else None
        secondary_tint = _public_param_triplet(submaterial, "BlendLayer2DiffuseColor") if (legacy_tints or authored_blend) else None
        # ...and blended a surface with Opacity below 1 by its diffuse alpha
        # (the Kraken logo: white RGB, the emblem only in alpha).
        legacy_alpha = legacy_tints and 0.0 < _float_authored_attribute(submaterial, "Opacity") < 1.0 - 1e-4

        primary_color_node = self._image_node(
            nodes,
            self._texture_export_path(submaterial, "base_color", "diffuse") or self._texture_path_for_slot(submaterial, "TexSlot1"),
            x=-720,
            y=520,
            is_color=True,
        )
        # Only Illum materials that explicitly declare a virtual tint-palette
        # decal source should read the palette's ship-UV-space decal color.
        # Generic Illum materials like BEHR_marksman_S1:dull_metal_01 have a
        # real TexSlot1 authored diffuse and no decal source; routing Decal
        # Color there leaks unrelated livery/decal imagery into the base coat.
        # POM trims/details are also authored-texture driven and should bypass
        # the palette decal source entirely.
        if plan.template_key == "parallax_pom" or not _uses_virtual_tint_palette_decal(submaterial):
            decal_palette = type("_NoDecal", (), {"color": None, "alpha": None})()
        else:
            decal_palette = self._palette_decal_sockets(
                nodes,
                links,
                palette,
                material_channel,
                x=-420,
                y=520,
            )
        primary_normal_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot2",), roles=("normal_gloss",))
        primary_normal_node = self._image_node(
            nodes,
            primary_normal_ref.export_path if primary_normal_ref is not None else None,
            x=-720,
            y=-140,
            is_color=False,
        )
        primary_detail = self._detail_texture_channels(nodes, self._texture_path_for_slot(submaterial, "TexSlot6"), x=-720, y=-420)
        primary_roughness, _primary_roughness_is_smoothness = self._roughness_socket_for_texture_reference(nodes, primary_normal_ref, x=-460, y=-140)
        # Without a material channel palette_finish_* fall back to the primary
        # channel, which put the primary's specular on untinted trims (Hull B
        # engine nozzles, Trim_D: 0.768 -- flat white). Same rule as layers.
        illum_palette_glossiness = (
            palette_finish_glossiness_factor(palette, material_channel) if material_channel is not None else None
        )
        illum_palette_specular = (
            (_mean_triplet(palette_finish_specular(palette, material_channel)) or 0.0) if material_channel is not None else 0.0
        )
        primary_specular = self._specular_socket_for_texture_path(nodes, self._texture_path_for_slot(submaterial, "TexSlot4"), x=-720, y=760)
        primary_specular = _scaled_specular(nodes, primary_specular, _authored_attribute_triplet(submaterial, "Specular"), x=-520, y=760)
        # A coloured F0 is a metal: the layer colour goes to the Specular.
        tinted_specular = _tinted_specular(submaterial)
        primary_alpha_source = (
            (decal_palette.alpha if decal_palette.alpha is not None else (_output_socket(primary_color_node, "Alpha") if primary_color_node is not None else None))
            if plan.uses_alpha or legacy_alpha
            else None
        )
        # A blended decal draws at its authored Opacity: the Mk V's engine
        # pulse overlay (0.15) covered the thrusters in solid colour.
        decal_opacity = _float_authored_attribute(submaterial, "Opacity")
        # %DECAL also scales by DecalDiffuseOpacity and DecalAlphaMult: the
        # Valkyrie's leak runs (0.5 x 0.5) were drawn as solid dirt. An
        # opacity of 0 marks a normal/POM-only decal, which keeps its host
        # colour here; a multiplier above 1 only saturates in game.
        diffuse_opacity = _optional_float_public_param(submaterial, "DecalDiffuseOpacity")
        if submaterial.decoded_feature_flags.has_decal and diffuse_opacity is not None and diffuse_opacity > 0.0:
            alpha_mult = _optional_float_public_param(submaterial, "DecalAlphaMult")
            decal_opacity = (decal_opacity if decal_opacity > 0.0 else 1.0) * min(1.0, diffuse_opacity) * (min(1.0, alpha_mult) if alpha_mult is not None and alpha_mult > 0.0 else 1.0)
        if (
            primary_alpha_source is not None
            and submaterial.decoded_feature_flags.has_decal
            and 0.0 < decal_opacity < 1.0 - 1e-4
        ):
            scale = nodes.new("ShaderNodeMath")
            scale.operation = "MULTIPLY"
            scale.label = "Decal Opacity"
            scale.location = (-620, 380)
            links.new(primary_alpha_source, scale.inputs[0])
            scale.inputs[1].default_value = decal_opacity
            primary_alpha_source = scale.outputs[0]
        primary_base_color = decal_palette.color if decal_palette.color is not None else (primary_color_node.outputs[0] if primary_color_node is not None else None)
        primary_metal_mask = primary_metal_colour = None
        if tinted_specular is None and decal_palette.color is None and primary_base_color is not None:
            primary_base_color, primary_metal_mask, primary_metal_colour = self._illum_spec_map_metal(
                nodes, links, submaterial, primary_base_color, primary_specular
            )
        primary = self._connect_layer_surface_group(
            nodes,
            links,
            base_color_socket=primary_base_color,
            base_alpha_socket=self._alpha_tested_socket(
                nodes,
                links,
                submaterial,
                primary_alpha_source,
                x=-460,
                y=380,
            ),
            normal_color_socket=primary_normal_node.outputs[0] if primary_normal_node is not None else None,
            roughness_socket=primary_roughness,
            detail_channels=primary_detail,
            detail_diffuse_strength=0.35,
            detail_gloss_strength=0.35,
            detail_bump_strength=0.15,
            tint_color=primary_tint,
            palette=palette,
            palette_channel_name=material_channel,
            palette_finish_channel_name=material_channel,
            palette_glossiness=illum_palette_glossiness,
            specular_value=0.0,
            palette_specular_value=illum_palette_specular,
            metallic_value=1.0 if tinted_specular is not None else 0.0,
            specular_color=tinted_specular,
            specular_socket=primary_specular,
            metallic_socket=None if tinted_specular is not None else (primary_metal_mask or primary_specular),
            x=-180,
            y=220,
            label="Primary Layer",
        )
        
        if material_channel is None and not legacy_tints:
            for node in reversed(nodes):
                if node.type == 'GROUP' and 'LayerSurface' in node.node_tree.name:
                    palette_socket = _input_socket(node, "Palette Color")
                    if palette_socket is not None:
                        # No palette channel: the authored Diffuse tints the
                        # texture (the Spartan's tyres, the F8C's rockets were
                        # black). %DECAL too: the Heartseeker's pin-up, the
                        # Razor's sponsor logos and the Fortune's chevrons
                        # printed as black silhouettes. Glows stay dark: unlit,
                        # the Mk V's glow trims would print as flat blue paint.
                        unlit = (
                            (_float_authored_attribute(submaterial, "Glow") or 0.0) <= 0.0
                            and "glow" not in submaterial.submaterial_name.lower()
                        )
                        # A glow with no colour map has only its Diffuse for a
                        # colour: dark, the Demon Fang's red blade edges
                        # (I_energy_blade, Glow 0.3) drew black.
                        textureless = primary_color_node is None
                        diffuse = _authored_attribute_triplet(submaterial, "Diffuse") if unlit or textureless else None
                        palette_socket.default_value = (*diffuse, 1.0) if diffuse is not None else ((1.0, 1.0, 1.0, 1.0) if unlit else (0.0, 0.0, 0.0, 1.0))
                    break

        secondary_color_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot9",), roles=("alternate_base_color", "base_color", "diffuse"))
        if secondary_color_ref is not None:
            secondary_color_node = self._image_node(
                nodes,
                secondary_color_ref.export_path if secondary_color_ref is not None else None,
                x=-720,
                y=20,
                is_color=True,
            )
            secondary_normal_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot3",), roles=("normal_gloss",))
            secondary_normal_node = self._image_node(
                nodes,
                secondary_normal_ref.export_path if secondary_normal_ref is not None else None,
                x=-720,
                y=-700,
                is_color=False,
            )
            secondary_detail = self._detail_texture_channels(nodes, self._texture_path_for_slot(submaterial, "TexSlot13"), x=-720, y=-980)
            secondary_roughness, _secondary_roughness_is_smoothness = self._roughness_socket_for_texture_reference(nodes, secondary_normal_ref, x=-460, y=-700)
            secondary_specular = self._specular_socket_for_texture_path(nodes, self._texture_path_for_slot(submaterial, "TexSlot10"), x=-720, y=980)
            secondary_specular = _scaled_specular(nodes, secondary_specular, _public_param_triplet(submaterial, "BlendLayer2SpecularColor"), x=-520, y=980)
            secondary = self._connect_layer_surface_group(
                nodes,
                links,
                base_color_socket=secondary_color_node.outputs[0] if secondary_color_node is not None else None,
                base_alpha_socket=self._alpha_tested_socket(
                    nodes,
                    links,
                    submaterial,
                    (
                        _output_socket(secondary_color_node, "Alpha")
                        if (secondary_color_node is not None and plan.uses_alpha)
                        else None
                    ),
                    x=-460,
                    y=-420,
                ),
                normal_color_socket=secondary_normal_node.outputs[0] if secondary_normal_node is not None else None,
                roughness_socket=secondary_roughness,
                detail_channels=secondary_detail,
                detail_diffuse_strength=0.35,
                detail_gloss_strength=0.35,
                detail_bump_strength=0.15,
                tint_color=secondary_tint,
                palette=palette,
                palette_channel_name=material_channel,
                palette_finish_channel_name=material_channel,
                palette_glossiness=illum_palette_glossiness,
                specular_value=0.0,
                palette_specular_value=illum_palette_specular,
                metallic_value=0.0,
                specular_color=None,
                specular_socket=secondary_specular,
                metallic_socket=secondary_specular,
                x=-180,
                y=-140,
                label="Secondary Layer",
            )
        else:
            secondary = LayerSurfaceSockets()

        height_primary = self._mask_socket(nodes, self._texture_path_for_slot(submaterial, "TexSlot8"), x=-720, y=-1240)
        height_secondary = self._mask_socket(nodes, self._texture_path_for_slot(submaterial, "TexSlot11"), x=-720, y=-1400)

        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_illum_group()
        shader_group.location = (140, 0)
        shader_group.label = "StarBreaker Illum"
        self._set_socket_default(_input_socket(shader_group, "Primary Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Primary Alpha"), 1.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Roughness"), 0.35)
        self._set_socket_default(_input_socket(shader_group, "Primary Specular"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Primary Normal"), (0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Secondary Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Secondary Alpha"), 1.0)
        self._set_socket_default(_input_socket(shader_group, "Secondary Roughness"), 0.35)
        self._set_socket_default(_input_socket(shader_group, "Secondary Specular"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Secondary Normal"), (0.0, 0.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Blend Mask"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "POM Strength"), 0.0)
        self._set_socket_default(_input_socket(shader_group, "Emission Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Emission Strength"), self._illum_emission_strength(submaterial))
        self._set_socket_default(_input_socket(shader_group, "Primary Metallic"), 1.0 if tinted_specular is not None else 0.0)
        if primary_metal_mask is not None:
            # The Illum group reads metallic here, not from the surface: the
            # LuxCore's brass plate, F0 0.36, drew as white-glossy plastic.
            self._link_group_input(links, primary_metal_mask, shader_group, "Primary Metallic")
            # A metal's colour is the surface group's Specular Color (0.04
            # grey by default): the plate came out dark grey, not brass.
            surface = next((n for n in nodes if n.type == "GROUP" and n.label == "Primary Layer"), None)
            target = _input_socket(surface, "Specular Color") if surface is not None else None
            if target is not None and primary_metal_colour is not None:
                links.new(primary_metal_colour, target)

        self._link_group_input(links, primary.color, shader_group, "Primary Color")
        self._link_group_input(links, primary.alpha, shader_group, "Primary Alpha")
        self._link_group_input(links, primary.roughness, shader_group, "Primary Roughness")
        self._link_group_input(links, primary.specular, shader_group, "Primary Specular")
        self._link_group_input(links, primary.normal, shader_group, "Primary Normal")
        if secondary.color is not None:
            self._link_group_input(links, secondary.color, shader_group, "Secondary Color")
            self._link_group_input(links, secondary.alpha, shader_group, "Secondary Alpha")
            self._link_group_input(links, secondary.roughness, shader_group, "Secondary Roughness")
            self._link_group_input(links, secondary.specular, shader_group, "Secondary Specular")
            self._link_group_input(links, secondary.normal, shader_group, "Secondary Normal")
        if (legacy_tints or authored_blend) and blend_mask_socket is None and secondary.color is not None:
            blend_mask_socket = self._legacy_blend_layer_mask(nodes, links, submaterial)
        self._link_group_input(links, blend_mask_socket, shader_group, "Blend Mask")
        if plan.template_key == "parallax_pom":
            self._link_group_input(links, height_primary, shader_group, "Primary Height")
            self._link_group_input(links, height_secondary, shader_group, "Secondary Height")
            self._set_socket_default(
                _input_socket(shader_group, "POM Strength"),
                max(0.0, min(0.2, _float_public_param(submaterial, "PomDisplacement", "HeightBias") or 0.08)),
            )

        surface_shader = _output_socket(shader_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, surface_shader, output, plan, submaterial)
        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

        # Phase 12 (POM follow-up): Illum-family POM materials (typically
        # ``DECAL`` + ``PARALLAX_OCCLUSION_MAPPING``) carry a dedicated
        # ``TexSlot8`` height sampler (role=``height``, filename
        # ``*_displ*``). Route the bundled ``POM_Vector`` group so the
        # Primary TexSlot1 diffuse + TexSlot2 ddna samplers read from
        # offset UVs. Without this ``POM Strength`` alone can only drive
        # the cheap single-sample offset inside the illum shader group —
        # the unrolled ray-march in ``pom_library.blend`` is what
        # produces the perceptible depth.
        if submaterial.decoded_feature_flags.has_parallax_occlusion_mapping:
            height_ref = _submaterial_texture_reference(
                submaterial,
                slots=("TexSlot8",),
                roles=("height",),
            )
            if (
                height_ref is not None
                and height_ref.export_path
                and primary_color_node is not None
            ):
                height_image_node = self._image_node(
                    nodes,
                    height_ref.export_path,
                    x=-1480,
                    y=-1240,
                    is_color=False,
                    reuse_any_existing=True,
                )
                if height_image_node is not None and height_image_node.image is not None:
                    illum_targets: list[bpy.types.ShaderNodeTexImage] = [primary_color_node]
                    if primary_normal_node is not None:
                        illum_targets.append(primary_normal_node)
                    illum_pom_scale = _float_public_param(
                        submaterial, "PomDisplacement", "HeightBias"
                    )
                    if illum_pom_scale is None or illum_pom_scale <= 0.0:
                        illum_pom_scale = 0.08
                    illum_pom_scale = min(0.2, illum_pom_scale)
                    self._wire_runtime_parallax(
                        material,
                        height_node=height_image_node,
                        target_image_nodes=illum_targets,
                        scale_value=illum_pom_scale,
                        bias_value=self._parallax_bias_value(submaterial),
                        uv_tile=_illum_base_uv_scale(submaterial)[0],
                    )

        self._apply_illum_slot_tiling(material, submaterial)

    def _apply_illum_slot_tiling(self, material: bpy.types.Material, submaterial: SubmaterialRecord) -> None:
        """Scale the UVs of the Illum maps by their authored TexMod tiling.

        CryEngine's Illum reads every main map at the diffuse slot's TexMod
        tiling, the blend layer at that times BlendLayer2Tiling and the blend
        mask times BlendMaskTiling, unless a slot has its own TexMod. Read at
        1x, the Hull C's grav plates (TexSlot1 tiled 30x) drew their triangle
        pattern across the whole cargo grid. Samplers already fed (POM, which
        tiles internally) and the detail maps, tiled by their own path, stay.
        """
        base = _illum_base_uv_scale(submaterial)
        layer2 = _optional_float_public_param(submaterial, "BlendLayer2Tiling") or 1.0
        mask = _optional_float_public_param(submaterial, "BlendMaskTiling") or 1.0
        factors = {"TexSlot3": layer2, "TexSlot9": layer2, "TexSlot10": layer2, "TexSlot11": layer2, "TexSlot12": mask}
        scales: dict[str, tuple[float, float]] = {}
        for ref in submaterial.texture_slots:
            if not ref.export_path or ref.slot in (None, "TexSlot6", "TexSlot13"):
                continue
            own = _texture_reference_uv_scale(ref)
            factor = factors.get(ref.slot, 1.0)
            scale = own if own is not None else (base[0] * factor, base[1] * factor)
            resolved = self.package.resolve_path(ref.export_path)
            if resolved is not None:
                scales.setdefault(str(resolved), scale)
        if not scales:
            return
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        for node in list(nodes):
            if node.bl_idname != "ShaderNodeTexImage" or node.image is None:
                continue
            vector_input = node.inputs.get("Vector")
            if vector_input is None or vector_input.is_linked:
                continue
            scale = scales.get(bpy.path.abspath(node.image.filepath, library=node.image.library))
            if scale is None or (abs(scale[0] - 1.0) < 1e-4 and abs(scale[1] - 1.0) < 1e-4):
                continue
            tex_coord = nodes.new("ShaderNodeTexCoord")
            tex_coord.location = (node.location.x - 440, node.location.y)
            mapping = nodes.new("ShaderNodeMapping")
            mapping.location = (node.location.x - 220, node.location.y)
            mapping.vector_type = "POINT"
            mapping.inputs["Scale"].default_value = (scale[0], scale[1], 1.0)
            links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], vector_input)


    def _build_nodraw_material(self, material: bpy.types.Material) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (250, 0)
        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_nodraw_group()
        _refresh_group_node_sockets(shader_group)
        shader_group.location = (0, 0)
        shader_group.label = "StarBreaker NoDraw"
        surface = _output_socket(shader_group, "Shader")
        if surface is not None:
            links.new(surface, output.inputs[0])
        self._configure_material(material, blend_method="CLIP", shadow_method="NONE")

    def _build_screen_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (550, 0)

        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_screen_group()
        _refresh_group_node_sockets(shader_group)
        shader_group.location = (250, 0)
        shader_group.label = "StarBreaker Screen"
        self._set_socket_default(_input_socket(shader_group, "Base Color"), (0.5, 0.5, 0.5, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Emission Strength"), 3.0)
        self._set_socket_default(_input_socket(shader_group, "Mix Factor"), 0.12)
        self._set_socket_default(_input_socket(shader_group, "Use Checker"), 0.0)

        image_path = representative_textures(submaterial)["base_color"]
        color_source = self._color_source_socket(nodes, submaterial, palette, image_path, x=0, y=0)
        if color_source is not None:
            self._link_group_input(links, color_source, shader_group, "Base Color")
        elif has_virtual_input(submaterial, "$RenderToTexture"):
            self._set_socket_default(_input_socket(shader_group, "Use Checker"), 1.0)

        surface = _output_socket(shader_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, surface, output, plan, submaterial)
        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

    def _build_effect_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (550, 0)

        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_effect_group()
        _refresh_group_node_sockets(shader_group)
        shader_group.location = (250, 0)
        shader_group.label = "StarBreaker Effect"
        self._set_socket_default(_input_socket(shader_group, "Base Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Emission Strength"), 2.5)
        self._set_socket_default(_input_socket(shader_group, "Mix Factor"), 0.35)

        color_source = self._color_source_socket(nodes, submaterial, palette, representative_textures(submaterial)["base_color"], x=0, y=0)
        if color_source is not None:
            self._link_group_input(links, color_source, shader_group, "Base Color")

        surface = _output_socket(shader_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, surface, output, plan, submaterial)
        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

    def _build_layered_wear_principled_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        """Clean-top-level layered wear Principled builder.

        Top level is restricted to Material Output, Palette group nodes,
        Image Texture nodes, Wear Input helper group, LayeredInputs helper
        group, and Principled shader group. All BSDF/NormalMap/Bump/MixRGB/
        Mix/RGB nodes that the legacy ``_build_principled_material`` emitted
        at the material top level live inside the two new shader groups;
        per-layer tint, shadowless / emission / alpha flags, and roughness
        defaults are seeded as group-input socket defaults.

        Residual top-level helpers (``SeparateColor`` from the metallic-
        roughness split in ``_roughness_group_source_socket`` and the
        ``Math`` invert in ``_layer_roughness_socket``) are intentionally
        left in place and covered by the deferred LayerSurface detail-channel
        refactor.
        """
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (700, 0)

        principled_group = nodes.new("ShaderNodeGroup")
        principled_group.node_tree = self._ensure_runtime_principled_group()
        _refresh_group_node_sockets(principled_group)
        principled_group.location = (420, 0)
        principled_group.label = "StarBreaker Principled"

        layered_group = nodes.new("ShaderNodeGroup")
        layered_group.node_tree = self._ensure_runtime_layered_inputs_group()
        _refresh_group_node_sockets(layered_group)
        layered_group.location = (120, 0)
        layered_group.label = "StarBreaker LayeredInputs"

        textures = representative_textures(submaterial)

        # Base image (primary diffuse).
        # A layer's diffuse repeats at the layer's UV tiling. Sampled once,
        # the FR-66 shield's painted_surface_02 (12-66x) spread one grain
        # over each part and the paint looked washed out.
        base_source_layer = next(
            (
                layer
                for layer in submaterial.layer_manifest
                if textures["base_color"] and layer.diffuse_export_path == textures["base_color"]
                and not (layer.name or "").lower().startswith("wearlayer")
            ),
            None,
        )
        if base_source_layer is not None:
            base_image_node = self._layer_blend_image(
                nodes, links, textures["base_color"], float(base_source_layer.uv_tiling or 1.0), True, -280, 220
            )
        else:
            base_image_node = self._image_node(
                nodes, textures["base_color"], x=-280, y=220, is_color=True
            )
        if base_image_node is not None:
            base_image_socket = _input_socket(layered_group, "Base Image")
            if base_image_socket is not None:
                links.new(base_image_node.outputs[0], base_image_socket)

        # Base palette channel (optional).
        channels = material_palette_channels(submaterial)
        active_channel = submaterial.palette_routing.material_channel or (
            channels[0] if channels else None
        )
        base_palette_linked = False
        palette_metal = False
        if active_channel is not None and palette is not None:
            # A palette channel with a near-black colour and a strong
            # specular is a metal (the Gallant's tertiary: 0.004 diffuse,
            # gold 0.30/0.22/0.09 finish, its battery cell); its colour is
            # the finish specular, as on Illum.
            finish_specular = palette_finish_specular(palette, active_channel.name)
            palette_metal = (
                finish_specular is not None
                and sum(finish_specular) / 3.0 >= 0.15
                and sum(palette_color(palette, active_channel.name)) / 3.0 <= 0.05
            )
            base_palette_socket = (
                self._palette_specular_socket(nodes, palette, active_channel.name, x=-280, y=40)
                if palette_metal
                else self._palette_color_socket(nodes, palette, active_channel.name, x=-280, y=40)
            )
            if base_palette_socket is not None:
                target = _input_socket(layered_group, "Base Palette")
                if target is not None:
                    self._link_color_output(base_palette_socket, target)
                    base_palette_linked = True

        # Base layer tint / diffuse fallback. LayerBlend_V2 materials
        # encode the actual base paint colour in
        # ``layer_manifest[0].tint_color`` and the base art on
        # ``layer_manifest[0].diffuse_export_path``. When the parent
        # submaterial does not expose its own base_color texture (e.g.
        # KLWE weapon paints, whose only top-level slot is a wear mask)
        # the LayeredInputs group is left with white Base Image and
        # white Base Palette defaults and renders pure white. Source
        # the missing data from the base layer:
        #   - If no parent Base Image link exists, wire the base
        #     layer's diffuse texture into Base Image.
        #   - If no real palette channel routes into Base Palette,
        #     hijack that socket's default with the base layer's
        #     tint_color (Base Palette is multiplied with Base Image
        #     internally, so this works out as a base tint multiply).
        base_layer = _layered_wear_base_layer(submaterial)
        if base_image_node is None:
            image_source_layer = (
                base_layer
                if base_layer is not None and base_layer.diffuse_export_path
                else _layered_wear_first_diffuse_layer(submaterial)
            )
            if image_source_layer is not None and image_source_layer.diffuse_export_path:
                fallback_image_node = self._layer_blend_image(
                    nodes,
                    links,
                    image_source_layer.diffuse_export_path,
                    float(image_source_layer.uv_tiling or 1.0),
                    True,
                    -280,
                    220,
                )
                if fallback_image_node is not None:
                    fallback_target = _input_socket(layered_group, "Base Image")
                    if fallback_target is not None:
                        links.new(fallback_image_node.outputs[0], fallback_target)
        # A white base tint is the base colour, not a missing one: the
        # Gallant's bare-metal receiver (H_bare_01_A, base tint 1,1,1) took
        # its wear layer's black and rendered as black metal. Only a base
        # layer without any tint borrows the first tinted base layer's.
        fallback_tint = (
            _layer_metal_tint(base_layer)
            if base_layer is not None and base_layer.tint_color is not None
            else _layered_wear_first_non_neutral_tint(submaterial)
        )
        if not base_palette_linked and fallback_tint is not None:
            base_palette_target = _input_socket(layered_group, "Base Palette")
            if base_palette_target is not None and hasattr(
                base_palette_target, "default_value"
            ):
                base_palette_target.default_value = (
                    *fallback_tint,
                    1.0,
                )

        # Wear layer (tint + palette + diffuse).
        wear_layer = self._layered_wear_layer(submaterial)
        if wear_layer is None:
            wear_layer = next(
                (layer for layer in submaterial.layer_manifest if layer.diffuse_export_path),
                None,
            )
        if wear_layer is not None:
            if wear_layer.diffuse_export_path:
                layer_image_node = self._layer_blend_image(
                    nodes, links, wear_layer.diffuse_export_path, float(wear_layer.uv_tiling or 1.0), True, -280, -140
                )
                if layer_image_node is not None:
                    target = _input_socket(layered_group, "Layer Image")
                    if target is not None:
                        links.new(layer_image_node.outputs[0], target)
            if wear_layer.tint_color is not None and any(
                abs(channel - 1.0) > 1e-6 for channel in wear_layer.tint_color
            ):
                tint_socket = _input_socket(layered_group, "Layer Tint")
                if tint_socket is not None:
                    tint_socket.default_value = (*wear_layer.tint_color, 1.0)
            if wear_layer.palette_channel is not None and palette is not None:
                layer_palette_socket = self._palette_color_socket(
                    nodes, palette, wear_layer.palette_channel.name, x=-280, y=-320
                )
                if layer_palette_socket is not None:
                    target = _input_socket(layered_group, "Layer Palette")
                    if target is not None:
                        self._link_color_output(layer_palette_socket, target)

        # Wear factor (Wear Input helper group — already wrapped).
        wear_factor_socket = self._layered_wear_factor_socket(
            nodes, links, submaterial, x=-60, y=-460
        )
        if wear_factor_socket is not None:
            target = _input_socket(layered_group, "Wear Factor")
            if target is not None:
                links.new(wear_factor_socket, target)

        metallic_values = _layered_wear_metallic_values(base_layer, wear_layer)
        metallic_target = _input_socket(principled_group, "Metallic")
        if metallic_values is not None and metallic_target is not None:
            base_metallic, wear_metallic = metallic_values
            if wear_factor_socket is not None and abs(base_metallic - wear_metallic) > 1e-6:
                metallic_mix = nodes.new("ShaderNodeMix")
                metallic_mix.location = (140, -900)
                if hasattr(metallic_mix, "data_type"):
                    metallic_mix.data_type = "FLOAT"
                metallic_mix.inputs[2].default_value = base_metallic
                metallic_mix.inputs[3].default_value = wear_metallic
                links.new(wear_factor_socket, metallic_mix.inputs[0])
                links.new(metallic_mix.outputs[0], metallic_target)
            else:
                # Without a wear factor the base layer is all that shows.
                metallic_target.default_value = base_metallic if wear_factor_socket is None else wear_metallic
        if palette_metal and metallic_target is not None:
            for link in list(metallic_target.links):
                links.remove(link)
            metallic_target.default_value = 1.0

        # Roughness (base + wear layer).
        base_roughness_source = self._roughness_group_source_socket(
            nodes, submaterial, textures["roughness"], x=-280, y=-620
        )
        base_roughness_target = _input_socket(layered_group, "Base Roughness")
        if base_roughness_source is not None and base_roughness_target is not None:
            links.new(base_roughness_source, base_roughness_target)

        layer_roughness_source = self._layer_roughness_socket(
            nodes, submaterial, x=-280, y=-780
        )
        layer_roughness_target = _input_socket(layered_group, "Layer Roughness")
        if layer_roughness_source is not None and layer_roughness_target is not None:
            links.new(layer_roughness_source, layer_roughness_target)

        # LayeredInputs outputs → Principled group inputs.
        color_output = _output_socket(layered_group, "Color")
        roughness_output = _output_socket(layered_group, "Roughness")
        if color_output is not None:
            target = _input_socket(principled_group, "Base Color")
            if target is not None:
                links.new(color_output, target)
        if roughness_output is not None:
            target = _input_socket(principled_group, "Roughness")
            if target is not None:
                links.new(roughness_output, target)

        # LayerBlend_V2 with a blend map: all base layers, stacked by it.
        # The two-layer wear composition above stays for the rest; what it
        # fed is swept once these links replace it.
        blend_outputs = self._layer_blend_sockets(nodes, links, submaterial, palette, x=-280, y=1400)
        if blend_outputs is not None:
            color_output, roughness_output, metallic_output = blend_outputs
            for name, socket in (("Base Color", color_output), ("Roughness", roughness_output),
                                 ("Metallic", metallic_output)):
                target = _input_socket(principled_group, name)
                if target is not None:
                    links.new(socket, target)

        # Normal map.
        normal_path = textures["normal"]
        if normal_path:
            normal_node = self._image_node(
                nodes, normal_path, x=-280, y=-940, is_color=False
            )
            # A layer's DDNA repeats at the layer's tiling, as its colour
            # does: the Attrition's copper rails took pattern_ribbed_01 (tiling
            # 2.4) once across the UV sheet, and their ribbing, gloss and all,
            # stretched out of sight. The node also feeds the roughness.
            layer_tiling = next(
                (float(layer.uv_tiling or 1.0) for layer in submaterial.layer_manifest
                 if layer.normal_export_path == normal_path),
                1.0,
            )
            if normal_node is not None and not normal_node.inputs["Vector"].is_linked:
                self._apply_uv_tiling(nodes, links, normal_node, layer_tiling, x=-580, y=-940)
            if normal_node is not None:
                target = _input_socket(principled_group, "Normal Color")
                if target is not None:
                    links.new(normal_node.outputs[0], target)
                use_normal = _input_socket(principled_group, "Use Normal")
                if use_normal is not None:
                    use_normal.default_value = 1.0

        # Height / bump.
        height_path = textures["height"]
        if height_path:
            height_node = self._image_node(
                nodes, height_path, x=-280, y=-1100, is_color=False
            )
            if height_node is not None:
                target = _input_socket(principled_group, "Height")
                if target is not None:
                    links.new(height_node.outputs[0], target)
                use_bump = _input_socket(principled_group, "Use Bump")
                if use_bump is not None:
                    use_bump.default_value = 1.0

        # Alpha.
        if plan.uses_alpha:
            alpha_source = self._alpha_source_socket(
                nodes, submaterial, textures, x=-280, y=-1260
            )
            if alpha_source is not None:
                target = _input_socket(principled_group, "Alpha")
                if target is not None:
                    links.new(alpha_source, target)

        # Emission.
        if plan.uses_emission:
            strength_socket = _input_socket(principled_group, "Emission Strength")
            if strength_socket is not None:
                strength_socket.default_value = 2.0
            if color_output is not None:
                target = _input_socket(principled_group, "Emission Color")
                if target is not None:
                    links.new(color_output, target)
            elif palette is not None and plan.uses_palette:
                emissive = self._palette_color_socket(
                    nodes, palette, "primary", x=-280, y=360
                )
                if emissive is not None:
                    target = _input_socket(principled_group, "Emission Color")
                    if target is not None:
                        self._link_color_output(emissive, target)

        shader_out = _output_socket(principled_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, shader_out, output, plan, submaterial)

        self._configure_material(
            material, blend_method=plan.blend_method, shadow_method=plan.shadow_method
        )

    def _layer_blend_sockets(
        self,
        nodes: bpy.types.Nodes,
        links: bpy.types.NodeLinks,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        *,
        x: int,
        y: int,
    ) -> tuple[Any, Any, Any] | None:
        """Colour, roughness and metallic of a LayerBlend_V2 material whose
        base layers are chosen by a blend map (TexSlot12), or None.

        The layered-wear path reads only the first layer plus one more: the
        ADP-mk4 helmet's olive paint is its third base layer and never showed,
        and with the wear layers (metallic, black tint) over everything it
        rendered glossy black. Wear is left out here: a catalogue item is
        pristine, and the ORC-mkX core's white wear metal covered 40% of it.
        With a single base layer the material has nothing to blend.
        """
        # Slots by the number in the name: manifests skip some (the Zeus
        # helmet shell has BaseLayer2-4, its visor 3 and 4 only).
        numbered: dict[int, LayerManifestEntry] = {}
        for layer in submaterial.layer_manifest:
            name = (layer.name or "").lower()
            if name.startswith("baselayer") and name[9:].isdigit() and 1 <= int(name[9:]) <= 4:
                numbered.setdefault(int(name[9:]), layer)
        if len(numbered) < 2:
            return None
        blend_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot12",), roles=("blend_mask",))
        blend_node = self._image_node(
            nodes, blend_ref.export_path if blend_ref is not None else None, x=x - 300, y=y + 300, is_color=False
        )
        if blend_node is None:
            return None
        group = nodes.new("ShaderNodeGroup")
        group.node_tree = self._ensure_runtime_layer_blend_group()
        _refresh_group_node_sockets(group)
        group.location = (x + 200, y)
        group.label = "StarBreaker LayerBlend"
        blend_tiling = _float_public_param(submaterial, "BlendMapTiling") or 1.0
        self._apply_uv_tiling(nodes, links, blend_node, blend_tiling, x=x - 600, y=y + 300)
        # The game reads blend maps as sRGB: their DDS headers carry
        # CryEngine's EIF_SRGBRead (0x800; wear maps do not). Read as data,
        # a half-strength channel showed its layer at full half: the
        # Morozov-SH mouthpiece came out light grey where the shop shots
        # have it near its dark bottom layer. The image stays Non-Color --
        # other paths load the same file as data -- and is decoded here.
        decode = nodes.new("ShaderNodeGamma")
        decode.location = (x - 100, y + 250)
        decode.inputs["Gamma"].default_value = 2.2
        links.new(blend_node.outputs[0], decode.inputs["Color"])
        blend_color = decode.outputs[0]
        # Red, the top layer, is partly edge wear: the Morozov-SH helmet's
        # plate edges and seams pick its white layer 1 where its wear mask
        # (TexSlot11) is dark, and the shop shots have none of it. Weighting
        # red by the mask keeps the pristine item; on the ADP-mk4, whose
        # red is paint under a light mask, 87% of it stays.
        wear_ref = _submaterial_texture_reference(submaterial, slots=("TexSlot11",), roles=("wear_mask",))
        wear_node = self._image_node(
            nodes, wear_ref.export_path if wear_ref is not None else None, x=x - 300, y=y + 600, is_color=False
        )
        if wear_node is not None:
            if not wear_node.inputs["Vector"].is_linked:
                self._apply_uv_tiling(nodes, links, wear_node, blend_tiling, x=x - 600, y=y + 600)
            split = nodes.new("ShaderNodeSeparateColor")
            split.location = (x - 100, y + 400)
            links.new(blend_color, split.inputs[0])
            red = nodes.new("ShaderNodeMath")
            red.operation = "MULTIPLY"
            red.location = (x + 50, y + 500)
            links.new(split.outputs[0], red.inputs[0])
            # Only the darkest mask lets the layer below through: scaled
            # linearly, the Ace Interceptor's speckled mask (a quarter of it
            # near black) turned half the dark shell into white flecks. The
            # shop shot has fine scratches, which is what x4 leaves.
            sharp = nodes.new("ShaderNodeMath")
            sharp.operation = "MULTIPLY"
            sharp.use_clamp = True
            sharp.location = (x - 100, y + 600)
            sharp.inputs[1].default_value = 4.0
            links.new(wear_node.outputs["Color"], sharp.inputs[0])
            links.new(sharp.outputs[0], red.inputs[1])
            join = nodes.new("ShaderNodeCombineColor")
            join.location = (x + 50, y + 350)
            links.new(red.outputs[0], join.inputs[0])
            links.new(split.outputs[1], join.inputs[1])
            links.new(split.outputs[2], join.inputs[2])
            blend_color = join.outputs[0]
        links.new(blend_color, _input_socket(group, "Blend Color"))

        # An empty slot takes the layer underneath it (slot 4, at the bottom,
        # the highest one there is), so its channel changes nothing.
        slots: dict[int, LayerManifestEntry] = {}
        below = numbered[max(numbered)]
        for index in (4, 3, 2, 1):
            below = slots[index] = numbered.get(index, below)
        for index, layer in sorted(slots.items()):
            prefix = "Layer%d " % index
            ly = y - index * 400
            tiling = float(layer.uv_tiling or 1.0)
            image = self._layer_blend_image(nodes, links, layer.diffuse_export_path, tiling, True, x - 300, ly)
            if image is not None:
                links.new(image.outputs["Color"], _input_socket(group, prefix + "Image"))
            tint = _layer_metal_tint(layer)
            if tint is not None:
                self._set_socket_default(_input_socket(group, prefix + "Tint"), (*tint, 1.0))
            if layer.palette_channel is not None and palette is not None:
                palette_socket = self._palette_color_socket(nodes, palette, layer.palette_channel.name, x=x - 300, y=ly - 120)
                if palette_socket is not None:
                    self._link_color_output(palette_socket, _input_socket(group, prefix + "Palette"))
            smooth_ref = _layer_texture_reference(layer, alpha_semantic="smoothness")
            smooth_path = smooth_ref.export_path if smooth_ref is not None else layer.normal_export_path
            smooth = self._layer_blend_image(nodes, links, smooth_path, tiling, False, x - 300, ly - 240)
            if smooth is not None:
                links.new(smooth.outputs["Alpha"], _input_socket(group, prefix + "Smoothness"))
            else:
                # No gloss map: the layer file's own Shininess (0-255). The
                # Zeus visor's gold_polished has none and drew matte at the
                # 0.5 default; at its 255 it is the mirror it is in game.
                shininess = _layer_snapshot_float(layer, "shininess")
                if shininess > 0.0:
                    self._set_socket_default(_input_socket(group, prefix + "Smoothness"), min(1.0, shininess / 255.0))
            self._set_socket_default(_input_socket(group, prefix + "Gloss"), float(layer.gloss_mult if layer.gloss_mult is not None else 1.0))
            self._set_socket_default(_input_socket(group, prefix + "Metallic"), _layer_snapshot_float(layer, "metallic"))
        return (
            _output_socket(group, "Color"),
            _output_socket(group, "Roughness"),
            _output_socket(group, "Metallic"),
        )

    def _layer_blend_image(
        self,
        nodes: bpy.types.Nodes,
        links: bpy.types.NodeLinks,
        path: str | None,
        tiling: float,
        is_color: bool,
        x: int,
        y: int,
    ) -> bpy.types.ShaderNodeTexImage | None:
        """A layer texture at the layer's own tiling. `_image_node` hands back
        a node already in the tree for the same file; two layers sharing a
        texture at different tilings (the ORC-mkX gauntlets' paint at 20 and
        25) then need their own node."""
        node = self._image_node(nodes, path, x=x, y=y, is_color=is_color)
        if node is None:
            return None
        vector = node.inputs.get("Vector")
        if vector is not None and vector.is_linked:
            copy = nodes.new("ShaderNodeTexImage")
            copy.image = node.image
            copy.interpolation = node.interpolation
            copy.extension = node.extension
            copy.location = (x, y)
            node = copy
        self._apply_uv_tiling(nodes, links, node, tiling, x=x - 300, y=y)
        return node

    def _build_principled_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        if plan.template_key == "layered_wear":
            self._build_layered_wear_principled_material(material, submaterial, palette, plan)
            return

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (700, 0)
        principled = self._create_surface_bsdf(nodes)
        surface_shader = principled.outputs[0]

        textures = representative_textures(submaterial)
        base_socket = self._color_source_socket(nodes, submaterial, palette, textures["base_color"], x=40, y=140)
        if base_socket is None and palette is not None and plan.uses_palette:
            primary = self._palette_color_socket(nodes, palette, "primary", x=80, y=120)
            base_socket = primary

        if base_socket is not None:
            links.new(base_socket, _input_socket(principled, "Base Color"))

        if plan.uses_alpha:
            alpha_socket = _input_socket(principled, "Alpha")
            alpha_source = self._alpha_source_socket(nodes, submaterial, textures, x=80, y=20)
            if alpha_socket is not None:
                if alpha_source is not None:
                    links.new(alpha_source, alpha_socket)
                elif plan.template_key == "hair":
                    alpha_socket.default_value = 0.85

        roughness_socket = _input_socket(principled, "Roughness")
        roughness_default = 0.45 if submaterial.shader_family != "GlassPBR" else 0.08
        roughness_source = self._roughness_group_source_socket(
            nodes,
            submaterial,
            textures["roughness"],
            x=80,
            y=-120,
        )
        if roughness_socket is not None:
            if roughness_source is not None:
                links.new(roughness_source, roughness_socket)
            else:
                roughness_socket.default_value = roughness_default

        normal_input = _input_socket(principled, "Normal")
        normal_node = self._image_node(nodes, textures["normal"], x=80, y=-280, is_color=False)
        bump_node = None
        if textures["height"] or plan.template_key == "parallax_pom":
            bump_node = nodes.new("ShaderNodeBump")
            bump_node.location = (240, -320)
            bump_input = _input_socket(bump_node, "Height")
            bump_input.default_value = 0.02
            height_node = self._image_node(nodes, textures["height"] or textures["mask"], x=40, y=-420, is_color=False)
            if height_node is not None:
                links.new(height_node.outputs[0], bump_input)
        if normal_node is not None:
            normal_map = nodes.new("ShaderNodeNormalMap")
            normal_map.location = (240, -220)
            links.new(normal_node.outputs[0], _input_socket(normal_map, "Color"))
            if bump_node is not None:
                links.new(_output_socket(normal_map, "Normal"), _input_socket(bump_node, "Normal"))
            elif normal_input is not None:
                links.new(_output_socket(normal_map, "Normal"), normal_input)
        if bump_node is not None and normal_input is not None:
            links.new(_output_socket(bump_node, "Normal"), normal_input)

        if plan.uses_transmission:
            transmission = _input_socket(principled, "Transmission Weight", "Transmission")
            if transmission is not None:
                transmission.default_value = 1.0
            ior_socket = _input_socket(principled, "IOR")
            if ior_socket is not None:
                ior_socket.default_value = 1.45
            alpha_socket = _input_socket(principled, "Alpha")
            if alpha_socket is not None:
                alpha_socket.default_value = 0.2

        if plan.uses_emission:
            emission_color = _input_socket(principled, "Emission Color", "Emission")
            if emission_color is not None:
                if base_socket is not None:
                    links.new(base_socket, emission_color)
                elif palette is not None and plan.uses_palette:
                    emissive = self._palette_color_socket(nodes, palette, "primary", x=80, y=300)
                    links.new(emissive, emission_color)
            emission_strength = _input_socket(principled, "Emission Strength")
            if emission_strength is not None:
                emission_strength.default_value = 2.0

        if plan.template_key == "biological":
            subsurface = _input_socket(principled, "Subsurface Weight", "Subsurface")
            if subsurface is not None:
                subsurface.default_value = 0.15

        if plan.template_key == "hair":
            anisotropic = _input_socket(principled, "Anisotropic")
            if anisotropic is not None:
                anisotropic.default_value = 0.4

        self._wire_surface_shader_to_output(nodes, links, surface_shader, output, plan, submaterial)

        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

    def _build_glass_material(
        self,
        material: bpy.types.Material,
        submaterial: SubmaterialRecord,
        palette: PaletteRecord | None,
        plan: Any,
    ) -> None:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (620, 0)

        shader_group = nodes.new("ShaderNodeGroup")
        shader_group.node_tree = self._ensure_runtime_glass_group()
        _refresh_group_node_sockets(shader_group)
        shader_group.location = (360, 0)
        shader_group.label = "StarBreaker Glass"
        self._set_socket_default(_input_socket(shader_group, "Base Color"), (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Roughness"), 0.08)
        self._set_socket_default(_input_socket(shader_group, "IOR"), 1.05)
        self._set_socket_default(_input_socket(shader_group, "Normal Color"), (0.5, 0.5, 1.0, 1.0))
        self._set_socket_default(_input_socket(shader_group, "Normal Strength"), 0.25)
        self._set_socket_default(_input_socket(shader_group, "Use Normal"), 0.0)

        textures = representative_textures(submaterial)
        base_path = textures["base_color"]
        roughness_path = textures["roughness"] or self._texture_export_path(submaterial, "wear_gloss")
        normal_path = textures["normal"]

        base_socket = self._color_source_socket(nodes, submaterial, palette, base_path, x=40, y=140)
        if base_socket is None and palette is not None:
            base_socket = self._palette_color_socket(nodes, palette, "glass", x=80, y=120)
        if base_socket is None:
            base_socket = self._value_color_socket(nodes, (1.0, 1.0, 1.0, 1.0), x=80, y=120)
        if base_socket is not None:
            self._link_group_input(links, base_socket, shader_group, "Base Color")

        roughness_node = self._image_node(nodes, roughness_path, x=80, y=-120, is_color=False)
        if roughness_node is not None:
            self._link_group_input(links, roughness_node.outputs[0], shader_group, "Roughness")

        normal_node = self._image_node(nodes, normal_path, x=80, y=-280, is_color=False)
        if normal_node is not None:
            self._link_group_input(links, normal_node.outputs[0], shader_group, "Normal Color")
            self._set_socket_default(_input_socket(shader_group, "Use Normal"), 1.0)

        surface = _output_socket(shader_group, "Shader")
        self._wire_surface_shader_to_output(nodes, links, surface, output, plan, submaterial)

        self._configure_material(material, blend_method=plan.blend_method, shadow_method=plan.shadow_method)

    # ------------------------------------------------------------------
    # Option E2-Lite: per-(decal, host-channel) clones so each decal
    # object picks up the palette colour of its nearest paint material
    # instead of the palette's ship-UV-space ``Decal Color`` lookup.
    # ------------------------------------------------------------------

    _MESH_DECAL_HOST_CHANNEL_OUTPUT: dict[str, str] = {
        "primary": "Primary",
        "secondary": "Secondary",
        "tertiary": "Tertiary",
        "glass": "Glass Color",
    }

    # Suffix of the palette output for each channel's specular reflectance
    # colour and glossiness. Primary/Secondary/Tertiary expose
    # ``<Channel> SpecColor`` / ``<Channel> Glossiness``; glass exposes
    # ``Glass SpecColor`` / ``Glass Glossiness``.
    _MESH_DECAL_HOST_CHANNEL_SPEC: dict[str, str] = {
        "primary": "Primary SpecColor",
        "secondary": "Secondary SpecColor",
        "tertiary": "Tertiary SpecColor",
        "glass": "Glass SpecColor",
    }
    _MESH_DECAL_HOST_CHANNEL_GLOSS: dict[str, str] = {
        "primary": "Primary Glossiness",
        "secondary": "Secondary Glossiness",
        "tertiary": "Tertiary Glossiness",
        "glass": "Glass Glossiness",
    }
    _DECAL_HOST_VARIANT_MARKERS = ("__host_rgb_", "__host_")

    @classmethod
    def _decal_host_variant_base_name(cls, material: bpy.types.Material) -> str:
        base_name = material.name
        for marker in cls._DECAL_HOST_VARIANT_MARKERS:
            index = base_name.find(marker)
            if index >= 0:
                return base_name[:index]
        return base_name

    @classmethod
    def _decal_host_variant_base_material(cls, material: bpy.types.Material) -> bpy.types.Material:
        base_name = cls._decal_host_variant_base_name(material)
        if base_name == material.name:
            return material
        base_material = bpy.data.materials.get(base_name)
        if base_material is not None and getattr(base_material, "node_tree", None) is not None:
            return base_material
        return material

    @staticmethod
    def _id_pointer(data_block: object | None) -> int:
        if data_block is None:
            return 0
        as_pointer = getattr(data_block, "as_pointer", None)
        if callable(as_pointer):
            return int(as_pointer())
        return id(data_block)

    @classmethod
    def _object_material_signature(cls, obj: bpy.types.Object) -> tuple[int, tuple[int, ...]]:
        data_pointer = cls._id_pointer(getattr(obj, "data", None))
        material_pointers: list[int] = []
        for slot in getattr(obj, "material_slots", []) or []:
            mat = slot.material if slot is not None else None
            material_pointers.append(cls._id_pointer(mat))
        return data_pointer, tuple(material_pointers)

    @staticmethod
    def _stored_decal_host_channel(obj: bpy.types.Object | None) -> str | None:
        if obj is None or not hasattr(obj, "get"):
            return None
        channel = obj.get(PROP_DECAL_HOST_CHANNEL)
        if not isinstance(channel, str):
            return None
        normalized = channel.strip().lower()
        if normalized in {"primary", "secondary", "tertiary", "glass"}:
            return normalized
        return None

    @staticmethod
    def _stored_decal_host_rgb(obj: bpy.types.Object | None) -> tuple[float, float, float] | None:
        if obj is None or not hasattr(obj, "get"):
            return None
        rgb = obj.get(PROP_DECAL_HOST_RGB)
        if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
            return None
        try:
            return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _submaterial_palette_channel(submaterial: SubmaterialRecord) -> str | None:
        routing = getattr(submaterial, "palette_routing", None)
        material_channel = getattr(routing, "material_channel", None) if routing is not None else None
        if material_channel is not None:
            name = getattr(material_channel, "name", None)
            if isinstance(name, str) and name:
                return name.lower()
        for binding in getattr(routing, "layer_channels", []) or []:
            channel = getattr(binding, "channel", None)
            name = getattr(channel, "name", None) if channel is not None else None
            if isinstance(name, str) and name:
                return name.lower()
        return None

    @staticmethod
    def _submaterial_authored_tint_rgb(
        submaterial: SubmaterialRecord,
    ) -> tuple[float, float, float] | None:
        for layer in getattr(submaterial, "layer_manifest", []) or []:
            tint_color = getattr(layer, "tint_color", None)
            if isinstance(tint_color, (list, tuple)) and len(tint_color) >= 3:
                rgb = (float(tint_color[0]), float(tint_color[1]), float(tint_color[2]))
                if rgb != (1.0, 1.0, 1.0):
                    return rgb
            for attribute in getattr(layer, "authored_attributes", []) or []:
                name = str((attribute or {}).get("name", "")).lower()
                if name != "tintcolor":
                    continue
                value = (attribute or {}).get("value")
                if not isinstance(value, str):
                    continue
                parts = [part.strip() for part in value.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    rgb = (float(parts[0]), float(parts[1]), float(parts[2]))
                except ValueError:
                    continue
                if rgb != (1.0, 1.0, 1.0):
                    return rgb
        return None

    @staticmethod
    def _submaterial_is_host_candidate(submaterial: SubmaterialRecord) -> bool:
        shader_family = getattr(submaterial, "shader_family", None)
        if shader_family == "MeshDecal":
            return False
        try:
            if template_plan_for_submaterial(submaterial).template_key == "decal_stencil":
                return False
        except Exception:
            pass
        return True

    def _polygon_counts_by_material_index(self, obj: bpy.types.Object) -> dict[int, int]:
        cache = getattr(self, "mesh_polygon_counts_cache", None)
        data = getattr(obj, "data", None)
        data_pointer = self._id_pointer(data)
        if cache is not None and data_pointer in cache:
            return cache[data_pointer]
        counts: dict[int, int] = {}
        polygons = getattr(data, "polygons", None) if data is not None else None
        if polygons is None:
            return counts
        for poly in polygons:
            idx = int(getattr(poly, "material_index", 0))
            counts[idx] = counts.get(idx, 0) + 1
        if cache is not None:
            cache[data_pointer] = counts
        return counts

    def _derive_decal_host_route_from_submaterials(
        self,
        obj: bpy.types.Object,
        slot_submaterials: list[SubmaterialRecord | None],
    ) -> tuple[str | None, tuple[float, float, float] | None]:
        priorities = ("primary", "secondary", "tertiary", "glass")
        slot_channels: dict[int, str] = {}
        slot_tints: dict[int, tuple[float, float, float]] = {}
        host_slots: list[int] = []
        for slot_index, submaterial in enumerate(slot_submaterials):
            if submaterial is None or not self._submaterial_is_host_candidate(submaterial):
                continue
            host_slots.append(slot_index)
            channel = self._submaterial_palette_channel(submaterial)
            # Glass is never a decal's host (see _scan_slots_for_host_channel).
            if channel in priorities and channel != "glass":
                slot_channels[slot_index] = channel
            tint = self._submaterial_authored_tint_rgb(submaterial)
            if tint is not None:
                slot_tints[slot_index] = tint

        if slot_channels:
            unique_channels = set(slot_channels.values())
            if len(unique_channels) == 1:
                return next(iter(unique_channels)), None
            counts = self._polygon_counts_by_material_index(obj)
            channel_counts: dict[str, int] = {}
            for slot_index, channel in slot_channels.items():
                channel_counts[channel] = channel_counts.get(channel, 0) + max(1, counts.get(slot_index, 0))
            return (
                min(channel_counts, key=lambda channel: (-channel_counts[channel], priorities.index(channel))),
                None,
            )

        if slot_tints:
            counts = self._polygon_counts_by_material_index(obj)
            # Tinted trims on a mesh whose panels carry no tint (the 350r's
            # Illum hull, coloured by its palette input): leave the host to
            # the built material's colour instead of a minor trim's tint.
            tinted = sum(counts.get(i, 0) for i in slot_tints)
            if tinted * 2 < sum(counts.get(i, 0) for i in host_slots):
                return None, None
            if len(slot_tints) == 1:
                return None, next(iter(slot_tints.values()))
            weighted_tints = [
                (max(1, counts.get(slot_index, 0)), tint)
                for slot_index, tint in slot_tints.items()
            ]
            weighted_tints.sort(key=lambda item: item[0], reverse=True)
            return None, weighted_tints[0][1]

        return None, None

    def _mesh_decal_host_channel_for_object(self, obj: bpy.types.Object) -> str | None:
        """Scan ``obj``'s material slots for a non-decal paint material
        with a palette channel assignment. Returns the canonical channel
        name ("primary" / "secondary" / "tertiary" / "glass") of the
        dominant paint coverage on the object, or None if no host paint
        material is found. Prefers explicit palette_routing metadata on
        the submaterial JSON; falls back to a material name heuristic
        ("_Paint_Primary" etc.). Falls through to the parent object's
        material slots when ``obj`` itself carries only decal
        materials (typical of ``dec_*`` children split off their host
        ``geo_*`` geometry).
        """
        stored = self._stored_decal_host_channel(obj)
        if stored is not None:
            return stored
        channel = self._scan_slots_for_host_channel(obj)
        if channel is not None:
            return channel
        parent = getattr(obj, "parent", None)
        if parent is not None:
            stored = self._stored_decal_host_channel(parent)
            if stored is not None:
                return stored
            channel = self._scan_slots_for_host_channel(parent)
        return channel

    def _scan_slots_for_host_channel(self, obj: bpy.types.Object) -> str | None:
        cache = getattr(self, "host_channel_cache", None)
        cache_key = self._object_material_signature(obj) if cache is not None else None
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        priorities = ("primary", "secondary", "tertiary", "glass")
        slots = list(getattr(obj, "material_slots", []) or [])
        if not slots:
            if cache is not None and cache_key is not None:
                cache[cache_key] = None
            return None

        slot_channels: dict[int, str] = {}
        for index, slot in enumerate(slots):
            mat = slot.material if slot is not None else None
            if mat is None:
                continue
            family = mat.get("starbreaker_shader_family") if hasattr(mat, "get") else None
            if family == "MeshDecal":
                continue
            channel: str | None = None
            sj = mat.get("starbreaker_submaterial_json") if hasattr(mat, "get") else None
            if isinstance(sj, str):
                try:
                    parsed = json.loads(sj)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    routing = parsed.get("palette_routing") or {}
                    mc = routing.get("material_channel") or {}
                    name = (mc.get("name") if isinstance(mc, dict) else None) or ""
                    if name:
                        channel = str(name).lower()
                    else:
                        for binding in routing.get("layer_channels", []) or []:
                            ch = (binding or {}).get("channel") or {}
                            nm = ch.get("name") if isinstance(ch, dict) else None
                            if nm:
                                channel = str(nm).lower()
                                break
            if channel is None:
                lname = mat.name.lower()
                for key in priorities:
                    if f"_paint_{key}" in lname or f"paint_{key}" in lname:
                        channel = key
                        break
            # Glass is never a decal's host: on the 350r the canopy was the
            # only routed slot, so the seam decals took its white tint.
            if channel is None or channel == "glass":
                continue
            slot_channels[index] = channel

        if not slot_channels:
            if cache is not None and cache_key is not None:
                cache[cache_key] = None
            return None

        unique_channels = set(slot_channels.values())
        if len(unique_channels) == 1:
            result = next(iter(unique_channels))
            if cache is not None and cache_key is not None:
                cache[cache_key] = result
            return result

        counts: dict[int, int] = {}
        mesh = getattr(obj, "data", None)
        polygons = getattr(mesh, "polygons", None) if mesh is not None else None
        if polygons is not None:
            for poly in polygons:
                idx = int(getattr(poly, "material_index", 0))
                counts[idx] = counts.get(idx, 0) + 1

        channel_counts: dict[str, int] = {}
        for index, channel in slot_channels.items():
            channel_counts[channel] = channel_counts.get(channel, 0) + max(1, counts.get(index, 0))

        result = min(
            channel_counts,
            key=lambda channel: (-channel_counts[channel], priorities.index(channel)),
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
        return result

    def _mesh_decal_host_rgb_for_object(
        self, obj: bpy.types.Object
    ) -> tuple[float, float, float] | None:
        """Fallback for Phase 29: when an object carries only decal
        materials and neither it nor its parent exposes a palette
        channel, read the dominant non-decal paint material's authored
        tint and return it as an RGB triple. Picks the material that
        covers the most polygons on the source mesh (self → parent) to
        favour the main panel colour over structural accents. Returns
        None if no usable host colour can be read.
        """
        stored = self._stored_decal_host_rgb(obj)
        if stored is not None:
            return stored
        for candidate in (obj, getattr(obj, "parent", None)):
            if candidate is None:
                continue
            stored = self._stored_decal_host_rgb(candidate)
            if stored is not None:
                return stored
            rgb = self._dominant_paint_tint_for_object(candidate)
            if rgb is not None:
                return rgb
        return None

    def _dominant_paint_tint_for_object(
        self, obj: bpy.types.Object
    ) -> tuple[float, float, float] | None:
        cache = getattr(self, "host_rgb_cache", None)
        cache_key = self._object_material_signature(obj) if cache is not None else None
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        slots = list(getattr(obj, "material_slots", []) or [])
        if not slots:
            if cache is not None and cache_key is not None:
                cache[cache_key] = None
            return None
        # Tally polygons per slot where possible.
        counts: dict[int, int] = {}
        mesh = getattr(obj, "data", None)
        polygons = getattr(mesh, "polygons", None) if mesh is not None else None
        if polygons is not None:
            for poly in polygons:
                idx = int(getattr(poly, "material_index", 0))
                counts[idx] = counts.get(idx, 0) + 1
        order = sorted(
            range(len(slots)),
            key=lambda i: counts.get(i, 0),
            reverse=True,
        )
        candidates: list[tuple[int, tuple[float, float, float]]] = []
        for i in order:
            mat = slots[i].material
            if mat is None or mat.node_tree is None:
                continue
            if mat.get("starbreaker_shader_family") == "MeshDecal" or mat.get("starbreaker_decal_host_rgb_key"):
                continue
            rgb = self._read_paint_tint_rgb(mat)
            if rgb is not None:
                candidates.append((counts.get(i, 0), rgb))
        # Seams sit on the panels, not the trims: of the materials covering
        # at least half as much as the largest, take the darkest (the 350r
        # nose is mostly tan metal_dark trim around near-black hull panels).
        result = None
        if candidates:
            largest = candidates[0][0]
            close = [rgb for count, rgb in candidates if count * 2 >= largest]
            result = min(close, key=lambda rgb: 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
        return result

    @staticmethod
    def _read_paint_tint_rgb(
        material: bpy.types.Material,
    ) -> tuple[float, float, float] | None:
        """Read a paint material's authored tint from its runtime group
        nodes. Tries ``Tint Color`` first (LayerSurface / HardSurface
        carry the baked-in per-layer tint there) then ``Primary
        Color`` / ``Base Color``. Returns None if the material has no
        recognisable runtime group or all candidate inputs are still
        at their default white.
        """
        submaterial_json = material.get(PROP_SUBMATERIAL_JSON) if hasattr(material, "get") else None
        if isinstance(submaterial_json, str):
            try:
                parsed = json.loads(submaterial_json)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                for layer in parsed.get("layer_manifest", []) or []:
                    for attribute in layer.get("authored_attributes", []) or []:
                        if str((attribute or {}).get("name", "")).lower() != "tintcolor":
                            continue
                        value = (attribute or {}).get("value")
                        if not isinstance(value, str):
                            continue
                        parts = [part.strip() for part in value.split(",")]
                        if len(parts) < 3:
                            continue
                        try:
                            rgb = (float(parts[0]), float(parts[1]), float(parts[2]))
                        except ValueError:
                            continue
                        if rgb != (1.0, 1.0, 1.0):
                            return rgb
        # Palette Color last: an Illum hull keeps its colour there with a
        # white tint (the 350r's near-black panels), and without it the
        # seam decals on top fell back to the white atlas.
        preferred = ("Tint Color", "Primary Color", "Base Color", "Palette Color")
        fallback: tuple[float, float, float] | None = None
        for node in material.node_tree.nodes:
            if node.bl_idname != "ShaderNodeGroup":
                continue
            tree = getattr(node, "node_tree", None)
            if tree is None or not tree.name.startswith("StarBreaker Runtime"):
                continue
            for name in preferred:
                sock = node.inputs.get(name)
                if sock is None or not hasattr(sock, "default_value"):
                    continue
                try:
                    r, g, b, *_ = tuple(sock.default_value)
                except Exception:
                    continue
                if (r, g, b) == (1.0, 1.0, 1.0):
                    continue
                return (float(r), float(g), float(b))
        return fallback

    #: Bumped whenever the Illum POM host variant wiring changes, so clones
    #: saved by an older addon are rebuilt instead of reused.
    _ILLUM_POM_HOST_VARIANT_VERSION = 2

    #: Bumped whenever the MeshDecal palette-channel variant wiring changes.
    _MESH_DECAL_HOST_VARIANT_VERSION = 2

    def _ensure_illum_pom_host_rgb_variant(
        self,
        material: bpy.types.Material,
        rgb: tuple[float, float, float],
        host_specular: float | None = None,
    ) -> bpy.types.Material:
        """Clone an Illum POM decal material and give it the host panel's
        colour and reflectance.

        This mirrors the MeshDecal host variants for decal materials that
        are authored as ``Illum`` + ``DECAL`` + ``POM`` rather than
        ``MeshDecal``. The decal atlas is white with a metal-like spec map
        (F0 ~0.22): rendered at face value the recessed hull panels (e.g.
        the Constellation's VTOL hatch insets) show up as bright white
        patches. The clone keeps the decal's normal, gloss and parallax but
        takes the host tint and the host's dielectric reflectance.
        """
        if material is None or material.node_tree is None:
            return material
        rgb_key = self._rgb_variant_key(rgb)
        specular = 0.04 if host_specular is None else max(0.0, min(1.0, float(host_specular)))
        base_material = self._decal_host_variant_base_material(material)
        clone_name = f"{self._decal_host_variant_base_name(material)}__host_rgb_{rgb_key}"
        clone = bpy.data.materials.get(clone_name)
        if (
            clone is not None
            and clone.get("starbreaker_decal_host_rgb_key") == rgb_key
            and clone.get("starbreaker_decal_host_variant_version") == self._ILLUM_POM_HOST_VARIANT_VERSION
            and abs(float(clone.get("starbreaker_decal_host_specular", -1.0)) - specular) < 1e-6
        ):
            return clone
        if clone is None:
            clone = base_material.copy()
            clone.name = clone_name
        clone["starbreaker_decal_host_rgb_key"] = rgb_key
        clone["starbreaker_decal_host_variant_version"] = self._ILLUM_POM_HOST_VARIANT_VERSION
        clone["starbreaker_decal_host_specular"] = specular

        links = clone.node_tree.links

        def _set_unlinked(node: bpy.types.Node, name: str, value: Any) -> None:
            socket = node.inputs.get(name)
            if socket is None:
                return
            for link in list(socket.links):
                links.remove(link)
            try:
                socket.default_value = value
            except Exception:
                pass

        for node in clone.node_tree.nodes:
            if node.bl_idname != "ShaderNodeGroup":
                continue
            tree = getattr(node, "node_tree", None)
            if tree is None or tree.name != "StarBreaker Runtime LayerSurface":
                continue
            _set_unlinked(node, "Tint Color", (rgb[0], rgb[1], rgb[2], 1.0))
            # The host tint already carries the palette colour; a black
            # palette input (no material channel) would zero it out. A
            # white host means no host colour was found, so leave those
            # panels dark rather than rendering the white atlas.
            if tuple(round(float(c), 6) for c in rgb[:3]) != (1.0, 1.0, 1.0):
                _set_unlinked(node, "Palette Color", (1.0, 1.0, 1.0, 1.0))
            _set_unlinked(node, "Specular Value", 0.0)
            _set_unlinked(node, "Metallic", 0.0)
            _set_unlinked(node, "Palette Specular", specular)
        return clone

    def _ensure_mesh_decal_host_variant(
        self,
        material: bpy.types.Material,
        channel: str,
        palette: PaletteRecord | None,
    ) -> bpy.types.Material:
        """Return a cloned decal material keyed by ``channel`` whose
        ``Host Tint`` input is wired to the palette's per-channel colour
        output (``Primary`` / ``Secondary`` / ``Tertiary`` / ``Glass
        Color``) instead of the default ``Decal Color`` lookup. Cached
        in ``bpy.data.materials`` under ``<name>__host_<channel>``; the
        cache key is deterministic so repeat import calls reuse clones.
        """
        if palette is None or material is None or material.node_tree is None:
            return material
        output_name = self._MESH_DECAL_HOST_CHANNEL_OUTPUT.get(channel)
        if output_name is None:
            return material
        base_material = self._decal_host_variant_base_material(material)
        clone_name = f"{self._decal_host_variant_base_name(material)}__host_{channel}"
        clone = bpy.data.materials.get(clone_name)
        if (
            clone is not None
            and clone.get("starbreaker_decal_host_channel") == channel
            and clone.get("starbreaker_decal_host_variant_version") == self._MESH_DECAL_HOST_VARIANT_VERSION
        ):
            return clone
        # A clone saved by an older addon is rewired in place below.
        if clone is None:
            clone = base_material.copy()
            clone.name = clone_name
        clone["starbreaker_decal_host_channel"] = channel
        clone["starbreaker_decal_host_variant_version"] = self._MESH_DECAL_HOST_VARIANT_VERSION

        nodes = clone.node_tree.nodes
        links = clone.node_tree.links
        decal_group_node = next(
            (
                n
                for n in nodes
                if n.bl_idname == "ShaderNodeGroup"
                and getattr(n, "node_tree", None) is not None
                and n.node_tree.name.startswith("SB_MeshDecal")
            ),
            None,
        )
        if decal_group_node is None:
            return clone
        host_tint = decal_group_node.inputs.get("Host Tint")
        if host_tint is None:
            return clone
        palette_group_node = next(
            (
                n
                for n in nodes
                if n.bl_idname == "ShaderNodeGroup"
                and getattr(n, "node_tree", None) is not None
                and n.node_tree.name.startswith("StarBreaker Palette ")
            ),
            None,
        )
        if palette_group_node is None:
            # No palette group ever got instantiated inside this material
            # (``_build_contract_group_material`` only wires it when the
            # palette authors decal colour data). Build one now so we
            # can source the per-channel colour.
            try:
                palette_group_node = self._palette_group_node(nodes, links, palette, x=-420, y=0)
            except Exception:
                palette_group_node = None
        if palette_group_node is None:
            return clone
        new_source = _output_socket(palette_group_node, output_name)
        if new_source is None:
            return clone
        # Drop any existing link into Host Tint, then rewire.
        for link in list(host_tint.links):
            links.remove(link)
        links.new(new_source, host_tint)

        # Option E2-Lite metallic+roughness: also rewire Host Specular
        # Tint and Host Roughness from the matching palette outputs.
        spec_input = decal_group_node.inputs.get("Host Specular Tint")
        spec_output_name = self._MESH_DECAL_HOST_CHANNEL_SPEC.get(channel)
        if spec_input is not None and spec_output_name is not None:
            spec_source = _output_socket(palette_group_node, spec_output_name)
            if spec_source is not None:
                for link in list(spec_input.links):
                    links.remove(link)
                links.new(spec_source, spec_input)

        rough_input = decal_group_node.inputs.get("Host Roughness")
        # Metal texels keep the decal's own (DDNA) roughness; see
        # ``_patch_mesh_decal_template_metal``.
        decal_rough_input = decal_group_node.inputs.get("Decal Roughness")
        ddna_roughness = next(
            (
                _output_socket(n, "Roughness")
                for n in nodes
                if n.bl_idname == "ShaderNodeGroup"
                and getattr(n, "node_tree", None) is not None
                and n.node_tree.name.startswith("StarBreaker Runtime DDNA Roughness")
            ),
            None,
        )
        if decal_rough_input is not None and ddna_roughness is not None and not decal_rough_input.is_linked:
            links.new(ddna_roughness, decal_rough_input)
        gloss_output_name = self._MESH_DECAL_HOST_CHANNEL_GLOSS.get(channel)
        if rough_input is not None and gloss_output_name is not None:
            gloss_source = _output_socket(palette_group_node, gloss_output_name)
            if gloss_source is not None:
                for link in list(rough_input.links):
                    links.remove(link)
                # Invert glossiness to roughness via a math node cached
                # by name on the clone so repeat calls don't accumulate.
                inv_name = f"SB_DecalHostRoughInvert_{channel}"
                inv = nodes.get(inv_name)
                if inv is None or inv.bl_idname != "ShaderNodeMath":
                    inv = nodes.new("ShaderNodeMath")
                    inv.name = inv_name
                    inv.operation = "SUBTRACT"
                    inv.use_clamp = True
                    inv.label = "1 - glossiness (host)"
                    inv.location = (palette_group_node.location.x + 220.0, palette_group_node.location.y - 140.0)
                inv.inputs[0].default_value = 1.0
                # Clear any prior links into inputs[1] before rewiring.
                for link in list(inv.inputs[1].links):
                    links.remove(link)
                links.new(gloss_source, inv.inputs[1])
                links.new(inv.outputs[0], rough_input)
        return clone

    def _rebind_mesh_decal_for_host(
        self,
        obj: bpy.types.Object,
        palette: PaletteRecord | None,
        *,
        host_channel: str | None = None,
        fallback_rgb: tuple[float, float, float] | None = None,
    ) -> int:
        """Post-pass called after all slots on ``obj`` have been
        assigned. For each slot carrying a MeshDecal material, detect
        the object's nearest paint channel and swap the slot to a
        channel-keyed clone. Returns the number of slots rebinded.

        Phase 29 extensions:
        - Walks up to ``obj.parent`` when ``obj`` has no paint of its
          own (covers ``dec_*`` children split off their ``geo_*``
          host).
        - Falls back to an RGB variant (``__host_rgb_<hex>``) that
          wires Host Tint to the dominant host paint's authored tint
          when no palette channel can be identified.
        """
        decal_slots: list[tuple[bpy.types.MaterialSlot, bpy.types.Material, bool]] = []
        for slot in getattr(obj, "material_slots", []):
            mat = slot.material if slot is not None else None
            if mat is None:
                continue
            shader_family = mat.get("starbreaker_shader_family")
            is_mesh_decal = shader_family == "MeshDecal"
            is_relief = bool(mat.get(PROP_HAS_POM, False)) or self._is_illum_relief_decal(mat)
            is_illum_pom_decal = (
                shader_family == "Illum"
                and is_relief
                and mat.get(PROP_TEMPLATE_KEY) == "decal_stencil"
            )
            if not is_mesh_decal and not is_illum_pom_decal:
                continue
            # Host-tint rebinding is only meaningful for POM-family decal
            # overlays. Non-POM branding/text decals author their own
            # colour and must not be retinted by the host.
            if not is_relief:
                continue
            decal_slots.append((slot, mat, is_mesh_decal))
        if not decal_slots:
            return 0

        channel = host_channel if palette is not None else None
        if channel is None and palette is not None:
            channel = self._mesh_decal_host_channel_for_object(obj)
        resolved_fallback_rgb = fallback_rgb
        if channel is None and resolved_fallback_rgb is None:
            resolved_fallback_rgb = self._mesh_decal_host_rgb_for_object(obj)
        if channel is None and resolved_fallback_rgb is None:
            return 0
        rebound = 0
        for slot, mat, is_mesh_decal in decal_slots:
            if is_mesh_decal and channel is not None:
                if mat.get("starbreaker_decal_host_channel") == channel:
                    continue
                variant = self._ensure_mesh_decal_host_variant(mat, channel, palette)
            else:
                variant_rgb = None
                host_specular = None
                if channel is not None and palette is not None:
                    variant_rgb = palette_color(palette, channel)
                    host_specular = _mean_triplet(palette_finish_specular(palette, channel))
                if variant_rgb is None:
                    variant_rgb = resolved_fallback_rgb
                if variant_rgb is None:
                    continue
                key = mat.get("starbreaker_decal_host_rgb_key")
                rgb_key = self._rgb_variant_key(variant_rgb)
                if is_mesh_decal:
                    if key == rgb_key:
                        continue
                    variant = self._ensure_mesh_decal_host_rgb_variant(mat, variant_rgb)
                else:
                    variant = self._ensure_illum_pom_host_rgb_variant(mat, variant_rgb, host_specular)
            if variant is not mat:
                slot.material = variant
                rebound += 1
        return rebound

    @staticmethod
    def _is_illum_relief_decal(material: bpy.types.Material) -> bool:
        """An Illum %DECAL with a normal map but no POM flag: a relief atlas
        like the POM ones, its white diffuse meant to take the panel's colour.
        The M50's Poms_EXT drew its seams and inset outlines bright white.
        Branding decals carry no normal map and keep their own colours."""
        raw = material.get(PROP_SUBMATERIAL_JSON) if hasattr(material, "get") else None
        if not isinstance(raw, str):
            return False
        try:
            parsed = json.loads(raw)
        except Exception:
            return False
        mask = ""
        for attribute in parsed.get("authored_attributes", []) or []:
            if (attribute or {}).get("name") == "StringGenMask":
                mask = str(attribute.get("value") or "").upper()
        if not ("%DECAL" in mask and "%NORMAL_MAP" in mask and "STENCIL_MAP" not in mask):
            return False
        # A normal map is not enough: the RSI quantum drives' decal sheet
        # (qdrv_smal_decals_a, RSI logo, warning labels, maker logos) has one
        # too and printed near-black in the host colour. A relief atlas's
        # diffuse is grey; a printed one is coloured.
        if BuildersMixin._decal_diffuse_is_coloured(material):
            return False
        # Nor is a grey diffuse: the Animus's decal sheet (apar_decals: white
        # skull logo, lettering, stripes) is white with a flat normal map and
        # vanished into the host. A relief atlas's normal map has relief.
        slot2 = next((t for t in parsed.get("texture_slots", []) or [] if (t or {}).get("slot") == "TexSlot2"), None)
        return BuildersMixin._decal_normal_has_relief(material, (slot2 or {}).get("export_path"))

    _RELIEF_NORMAL_CACHE: dict[str, bool] = {}

    @staticmethod
    def _decal_normal_has_relief(material: bpy.types.Material, export_path: str | None) -> bool:
        """True when the decal's normal map bends its normals where the decal
        is opaque: mean |xy - 0.5| above 0.05 (the M50's Poms_EXT 0.155, the
        Animus's printed apar_decals 0.006). Unknown counts as relief, the
        rule's behaviour before this check."""
        if not export_path:
            return True
        name = export_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        cached = BuildersMixin._RELIEF_NORMAL_CACHE.get(name)
        if cached is not None:
            return cached
        normal = next((img for img in bpy.data.images
                       if bpy.path.basename(img.filepath).lower() == name), None)
        node_tree = getattr(material, "node_tree", None)
        colour = None
        for node in (node_tree.nodes if node_tree is not None else []):
            image = getattr(node, "image", None) if node.bl_idname == "ShaderNodeTexImage" else None
            if image is not None and image is not normal and image.colorspace_settings.name != "Non-Color":
                colour = image
                break
        result = True
        try:
            import numpy as np

            if normal is not None and normal.size[0] and normal.size[1]:
                def sampled(image):
                    pixels = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
                    image.pixels.foreach_get(pixels)
                    return pixels.reshape(image.size[1], image.size[0], 4)
                n = sampled(normal)
                step = max(1, max(n.shape[:2]) // 512)
                n = n[::step, ::step]
                dev = np.abs(n[..., :2] - 0.5).max(axis=-1)
                mask = None
                if colour is not None and tuple(colour.size) == tuple(normal.size):
                    mask = sampled(colour)[::step, ::step, 3] > 0.5
                values = dev[mask] if mask is not None and mask.any() else dev.reshape(-1)
                result = float(values.mean()) > 0.05
        except Exception:
            result = True
        BuildersMixin._RELIEF_NORMAL_CACHE[name] = result
        return result

    _COLOURED_DECAL_CACHE: dict[str, bool] = {}

    @staticmethod
    def _decal_diffuse_is_coloured(material: bpy.types.Material) -> bool:
        """True when the material's colour image has saturated texels: the
        mean chroma (max - min channel) over its opaque texels is above 0.08."""
        node_tree = getattr(material, "node_tree", None)
        if node_tree is None:
            return False
        for node in node_tree.nodes:
            image = getattr(node, "image", None) if node.bl_idname == "ShaderNodeTexImage" else None
            if image is None or image.colorspace_settings.name == "Non-Color":
                continue
            key = bpy.path.abspath(image.filepath, library=image.library) or image.name
            cached = BuildersMixin._COLOURED_DECAL_CACHE.get(key)
            if cached is None:
                cached = False
                try:
                    import numpy as np

                    width, height = image.size
                    if width and height:
                        pixels = np.empty(width * height * 4, dtype=np.float32)
                        image.pixels.foreach_get(pixels)
                        pixels = pixels.reshape(-1, 4)[:: max(1, (width * height) // 65536)]
                        opaque = pixels[pixels[:, 3] > 0.5]
                        if len(opaque):
                            chroma = opaque[:, :3].max(axis=1) - opaque[:, :3].min(axis=1)
                            cached = float(chroma.mean()) > 0.08
                except Exception:
                    cached = False
                BuildersMixin._COLOURED_DECAL_CACHE[key] = cached
            if cached:
                return True
        return False

    @staticmethod
    def _rgb_variant_key(rgb: tuple[float, float, float]) -> str:
        r, g, b = rgb
        return f"{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"

    def _ensure_mesh_decal_host_rgb_variant(
        self,
        material: bpy.types.Material,
        rgb: tuple[float, float, float],
    ) -> bpy.types.Material:
        """Clone a decal material and set its ``Host Tint`` input to a
        fixed RGB (no palette link). Used as a Phase 29 fallback when
        the host uses fixed-colour paint that isn't routed through any
        palette channel. Clones are cached in ``bpy.data.materials``
        under ``<name>__host_rgb_<hex>`` so repeat import calls reuse
        them.
        """
        if material is None or material.node_tree is None:
            return material
        rgb_key = self._rgb_variant_key(rgb)
        base_material = self._decal_host_variant_base_material(material)
        clone_name = f"{self._decal_host_variant_base_name(material)}__host_rgb_{rgb_key}"
        clone = bpy.data.materials.get(clone_name)
        if clone is not None and clone.get("starbreaker_decal_host_rgb_key") == rgb_key:
            return clone
        if clone is None:
            clone = base_material.copy()
            clone.name = clone_name
        clone["starbreaker_decal_host_rgb_key"] = rgb_key
        nodes = clone.node_tree.nodes
        links = clone.node_tree.links
        decal_group_node = next(
            (
                n
                for n in nodes
                if n.bl_idname == "ShaderNodeGroup"
                and getattr(n, "node_tree", None) is not None
                and n.node_tree.name.startswith("SB_MeshDecal")
            ),
            None,
        )
        if decal_group_node is None:
            return clone
        host_tint = decal_group_node.inputs.get("Host Tint")
        if host_tint is None:
            return clone
        for link in list(host_tint.links):
            links.remove(link)
        try:
            host_tint.default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        except Exception:
            pass
        return clone
