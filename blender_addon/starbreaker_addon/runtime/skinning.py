"""Armature rigs that deform soft-skinned meshes from animated bone objects.

The decomposed export animates one scene object per skeleton bone. Rigid
geometry is parented under its bone object, but a soft-skinned mesh (vertices
blended across several bones, e.g. telescoping landing-gear pistons) stays
whole on the skeleton root and carries one weighted vertex group per bone,
named after the bone's source node.

``ensure_skin_rigs`` gives each such mesh an armature whose bones copy the
world transform of the matching bone objects, plus an Armature modifier. Every
existing animation path (snap poses, NLA actions) then deforms the mesh
without knowing about skinning.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import bpy

SKIN_MODIFIER_NAME = "StarBreaker Skin"
SKIN_RIG_PROP = "starbreaker_skin_rig"
_SKIN_BONE_LENGTH = 0.05
_BONE_OBJECT_TYPES = {"EMPTY", "MESH"}
# Vertex groups the exporter writes for other purposes (`DECAL_OFFSET_GROUP_NAME`
# in blend_assembly.rs); never bone weights.
_NON_BONE_VERTEX_GROUPS = {"starbreaker_decal_offset"}

SourceNames = Callable[[Any], list[str]]
BindPose = Callable[[Any], "dict[str, Any] | None"]


def skin_joint_objects(mesh_obj: Any, source_names: SourceNames) -> dict[str, Any]:
    """Map a mesh's vertex-group names to the bone objects that drive them.

    Bones are looked up only in the mesh's own subtree (the mesh sits on the
    skeleton root), so mirrored copies of the same skeleton elsewhere in the
    scene -- e.g. the left and right landing-gear legs -- never cross-bind.
    """
    wanted = {group.name for group in getattr(mesh_obj, "vertex_groups", ())} - _NON_BONE_VERTEX_GROUPS
    if not wanted:
        return {}
    found: dict[str, Any] = {}
    for obj in [mesh_obj, *getattr(mesh_obj, "children_recursive", ())]:
        if getattr(obj, "type", None) not in _BONE_OBJECT_TYPES:
            continue
        for name in source_names(obj):
            if name in wanted and name not in found:
                found[name] = obj
                break
    return found


def skin_chain_to(mesh_obj: Any, joint_obj: Any) -> list[Any] | None:
    """Objects from just below ``mesh_obj`` down to ``joint_obj`` (inclusive).

    Returns an empty list for the mesh itself and ``None`` when ``joint_obj``
    is not a descendant of ``mesh_obj``.
    """
    chain: list[Any] = []
    current = joint_obj
    while current is not None and current is not mesh_obj:
        chain.append(current)
        current = getattr(current, "parent", None)
    if current is None:
        return None
    chain.reverse()
    return chain


def ensure_skin_rigs(
    context: bpy.types.Context,
    objects: Iterable[bpy.types.Object],
    source_names: SourceNames,
    bind_pose: BindPose,
) -> int:
    """Rig every soft-skinned mesh among ``objects`` that has no rig yet.

    Returns the number of rigs created.
    """
    created = 0
    for obj in list(objects):
        if getattr(obj, "type", None) != "MESH":
            continue
        joints = skin_joint_objects(obj, source_names)
        if not joints or obj.modifiers.get(SKIN_MODIFIER_NAME) is not None:
            continue
        rest = {}
        for name, joint in joints.items():
            matrix = _bind_matrix_in_mesh_space(obj, joint, bind_pose)
            if matrix is not None:
                rest[name] = matrix
        if rest:
            _build_rig(context, obj, joints, rest)
            created += 1
    return created


def _bind_matrix_in_mesh_space(mesh_obj: Any, joint_obj: Any, bind_pose: BindPose) -> Any:
    """Joint's bind transform relative to the mesh, from stored bind poses.

    Built from local transforms rather than ``matrix_world`` so the result does
    not depend on whatever pose the scene currently shows.
    """
    from mathutils import Matrix

    chain = skin_chain_to(mesh_obj, joint_obj)
    if chain is None:
        return None
    matrix = Matrix.Identity(4)
    for obj in chain:
        matrix = matrix @ obj.matrix_parent_inverse @ _bind_basis(obj, bind_pose)
    return matrix


def _bind_basis(obj: Any, bind_pose: BindPose) -> Any:
    from mathutils import Matrix, Quaternion, Vector

    data = bind_pose(obj)
    if not isinstance(data, dict):
        return obj.matrix_basis.copy()
    location = data.get("location")
    rotation = data.get("rotation_quaternion")
    if not (isinstance(location, list) and len(location) >= 3):
        return obj.matrix_basis.copy()
    if isinstance(rotation, list) and len(rotation) >= 4 and data.get("rotation_mode") == "QUATERNION":
        quaternion = Quaternion([float(value) for value in rotation[:4]])
    else:
        quaternion = obj.matrix_basis.to_quaternion()
    return Matrix.LocRotScale(Vector([float(value) for value in location[:3]]), quaternion, obj.scale)


def _build_rig(
    context: bpy.types.Context,
    mesh_obj: bpy.types.Object,
    joints: dict[str, bpy.types.Object],
    rest: dict[str, Any],
) -> bpy.types.Object:
    from mathutils import Matrix

    rig_name = f"{mesh_obj.name}_skin"
    armature = bpy.data.armatures.new(rig_name)
    rig = bpy.data.objects.new(rig_name, armature)
    collections = list(mesh_obj.users_collection) or [context.scene.collection]
    for collection in collections:
        collection.objects.link(rig)
    # Parent at identity so the rig's space is the mesh's space: rest bones
    # are bind transforms relative to the mesh, and moving the ship moves both.
    rig.parent = mesh_obj
    rig.matrix_parent_inverse = Matrix.Identity(4)
    rig.matrix_basis = Matrix.Identity(4)
    rig.hide_render = True
    rig.show_in_front = False
    rig[SKIN_RIG_PROP] = mesh_obj.name

    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    view_layer.objects.active = rig
    try:
        with context.temp_override(active_object=rig, object=rig, selected_objects=[rig]):
            bpy.ops.object.mode_set(mode="EDIT")
            try:
                for name, matrix in rest.items():
                    bone = armature.edit_bones.new(name)
                    bone.head = (0.0, 0.0, 0.0)
                    bone.tail = (0.0, _SKIN_BONE_LENGTH, 0.0)
                    translation = matrix.to_translation()
                    orientation = matrix.to_3x3().normalized().to_4x4()
                    orientation.translation = translation
                    bone.matrix = orientation
                    bone.use_deform = True
            finally:
                bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        view_layer.objects.active = previous_active

    for name in rest:
        constraint = rig.pose.bones[name].constraints.new("COPY_TRANSFORMS")
        constraint.name = "StarBreaker Joint"
        constraint.target = joints[name]
        constraint.target_space = "WORLD"
        constraint.owner_space = "WORLD"

    modifier = mesh_obj.modifiers.new(SKIN_MODIFIER_NAME, "ARMATURE")
    modifier.object = rig
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    # Deform before welds, normals and decal offsets see the geometry.
    mesh_obj.modifiers.move(len(mesh_obj.modifiers) - 1, 0)
    rig.hide_set(True)
    return rig
