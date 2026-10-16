"""Tests for `runtime.skinning`: binding soft-skinned vertex groups to bone objects."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ADDON_ROOT = Path(__file__).resolve().parent.parent / "starbreaker_addon"


def _load_skinning() -> types.ModuleType:
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location(
        "sb_skin_test_skinning", ADDON_ROOT / "runtime" / "skinning.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Obj:
    def __init__(self, name: str, type_: str = "EMPTY", parent: "_Obj | None" = None, groups=()):
        self.name = name
        self.type = type_
        self.parent = parent
        self.vertex_groups = [types.SimpleNamespace(name=group) for group in groups]
        self.children_recursive: list[_Obj] = []
        node = parent
        while node is not None:
            node.children_recursive.append(self)
            node = node.parent


def _source_names(obj: _Obj) -> list[str]:
    # Mirrors the scene naming the addon resolves: "<asset>_<index>_<bone>"
    # for bone empties and a numeric suffix on mirrored copies.
    names = [obj.name]
    parts = obj.name.split("_")
    for index, part in enumerate(parts[:-1]):
        if part.isdigit():
            names.append("_".join(parts[index + 1:]))
    if parts[-1].isdigit():
        names.append("_".join(parts[:-1]))
    return names


class SkinJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skinning = _load_skinning()

    def _leg(self, suffix: str = "") -> tuple[_Obj, _Obj, _Obj]:
        leg = _Obj(
            f"LG_Rear4_Root{suffix}",
            "MESH",
            groups=["starbreaker_decal_offset", "BONE_LG_Rear4_PistonTop", "BONE_LG_Rear4_Foot"],
        )
        root_bone = _Obj(f"LG_Rear4_Root{suffix}_1_BONE_LG_Rear4_Root", parent=leg)
        piston = _Obj(f"LG_Rear4_Root{suffix}_2_BONE_LG_Rear4_PistonTop", parent=root_bone)
        foot = _Obj(f"BONE_LG_Rear4_Foot{suffix}", "MESH", parent=root_bone)
        return leg, piston, foot

    def test_vertex_groups_bind_to_bone_objects_in_the_mesh_subtree(self) -> None:
        leg, piston, foot = self._leg()
        joints = self.skinning.skin_joint_objects(leg, _source_names)
        self.assertEqual(joints, {"BONE_LG_Rear4_PistonTop": piston, "BONE_LG_Rear4_Foot": foot})

    def test_mirrored_legs_bind_only_to_their_own_bones(self) -> None:
        left, _, _ = self._leg()
        right, right_piston, right_foot = self._leg("_001")
        joints = self.skinning.skin_joint_objects(right, _source_names)
        self.assertEqual(
            joints, {"BONE_LG_Rear4_PistonTop": right_piston, "BONE_LG_Rear4_Foot": right_foot}
        )
        self.assertNotIn(left, joints.values())

    def test_mesh_without_bone_groups_is_not_skinned(self) -> None:
        hull = _Obj("hull", "MESH", groups=["starbreaker_decal_offset"])
        _Obj("hardpoint", parent=hull)
        self.assertEqual(self.skinning.skin_joint_objects(hull, _source_names), {})

    def test_decal_offset_group_never_binds_to_a_bone(self) -> None:
        leg = _Obj("leg", "MESH", groups=["starbreaker_decal_offset"])
        _Obj("starbreaker_decal_offset", parent=leg)
        self.assertEqual(self.skinning.skin_joint_objects(leg, _source_names), {})

    def test_chain_runs_from_below_the_mesh_to_the_joint(self) -> None:
        leg, piston, _ = self._leg()
        chain = self.skinning.skin_chain_to(leg, piston)
        self.assertEqual([obj.name for obj in chain], [piston.parent.name, piston.name])
        self.assertEqual(self.skinning.skin_chain_to(leg, leg), [])

    def test_chain_is_none_for_objects_outside_the_mesh(self) -> None:
        leg, _, _ = self._leg()
        other, _, _ = self._leg("_001")
        self.assertIsNone(self.skinning.skin_chain_to(leg, other))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
