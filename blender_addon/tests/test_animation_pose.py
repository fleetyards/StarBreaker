"""Regression tests for the snap_first / snap_last pose-application math
used by `runtime.package_ops._apply_best_channel_transform`.

Pins:

* `_decode_animation_position("identity", …)` is a pass-through; the
  exporter writes Blender-frame XYZ already (per
  `crates/starbreaker-3d/src/animation/serialise.rs::clip_to_json`,
  which emits `(x, -z, y)`).
* The endpoint-policy selector picks the first sample at frame_index=0
  and the last sample otherwise (literal mode).
* The position-track delta is computed against the keyframe whose
  decoded position is closest to the bone's bind location, then added
  to the bind location. A bone whose entire position track is constant
  must therefore stay at bind, regardless of the absolute offset
  between the clip's authored values and the bone's bind position.

This last property is the key contract the `wings_deploy` X-shape
investigation hinges on: rotators `Wing_Rotator_Top_Left` and
`Wing_Rotator_Bottom_Right` have constant position tracks in the
sidecar and therefore MUST land at bind position. If a future change
breaks that property, the X-shape failure mode reappears regardless
of any parent-frame composition fix.

See `docs/StarBreaker/animation-research.md` (Scorpius wing-deploy
kinematics → Deployed-pose verification in Blender) for the data the
constants below are pinned against.
"""

from __future__ import annotations

import json
import math
import sys
import types
import unittest
import zlib
from pathlib import Path


ADDON_ROOT = Path(__file__).resolve().parent.parent / "starbreaker_addon"


class _StubObject:
    """Minimal stand-in for `bpy.types.Object` for pose-math tests."""

    def __init__(
        self,
        name: str,
        location=(0.0, 0.0, 0.0),
        rotation_quaternion=(1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.name = name
        self.location = tuple(float(v) for v in location)
        self.rotation_mode = "QUATERNION"
        self.rotation_quaternion = tuple(float(v) for v in rotation_quaternion)
        self.type = "MESH"
        self.parent = None
        self.children_recursive: list[object] = []
        self._props: dict[str, object] = {}

    # dict-like custom-properties access mirroring the bpy API surface
    # the production code uses.
    def __getitem__(self, key: str) -> object:
        return self._props[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def get(self, key: str, default: object = None) -> object:
        return self._props.get(key, default)


def _load_package_ops() -> types.ModuleType:
    bpy = sys.modules.get("bpy")
    if bpy is None:
        bpy = types.ModuleType("bpy")
        sys.modules["bpy"] = bpy
    bpy.types = types.SimpleNamespace(Context=object, Object=object, ID=object, Light=object)
    bpy.data = types.SimpleNamespace(objects=[], lights=[])

    mathutils = sys.modules.get("mathutils")
    if mathutils is None:
        mathutils = types.ModuleType("mathutils")
        sys.modules["mathutils"] = mathutils

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

    runtime_pkg = sys.modules.get("sb_anim_test_runtime")
    if runtime_pkg is None:
        runtime_pkg = types.ModuleType("sb_anim_test_runtime")
        runtime_pkg.__path__ = [str(ADDON_ROOT / "runtime")]
        sys.modules["sb_anim_test_runtime"] = runtime_pkg

    addon_pkg = sys.modules.get("sb_anim_test_addon")
    if addon_pkg is None:
        addon_pkg = types.ModuleType("sb_anim_test_addon")
        addon_pkg.__path__ = [str(ADDON_ROOT)]
        sys.modules["sb_anim_test_addon"] = addon_pkg

    manifest_stub = types.ModuleType("sb_anim_test_addon.manifest")
    manifest_stub.PackageBundle = type("PackageBundle", (), {"load": staticmethod(lambda p: None)})
    manifest_stub.SceneInstanceRecord = type("SceneInstanceRecord", (), {})
    sys.modules["sb_anim_test_addon.manifest"] = manifest_stub

    palette_stub = types.ModuleType("sb_anim_test_addon.palette")
    palette_stub.palette_id_for_livery_instance = lambda *a, **kw: None
    palette_stub.resolved_palette_id = lambda package, requested, inherited: requested or inherited
    sys.modules["sb_anim_test_addon.palette"] = palette_stub

    validators_stub = types.ModuleType("sb_anim_test_runtime.validators")
    validators_stub._purge_orphaned_file_backed_images = lambda: 0
    validators_stub._purge_orphaned_managed_materials = lambda: 0
    validators_stub._purge_orphaned_runtime_actions = lambda: 0
    validators_stub._purge_orphaned_runtime_groups = lambda: 0
    sys.modules["sb_anim_test_runtime.validators"] = validators_stub

    importer_stub = types.ModuleType("sb_anim_test_runtime.importer")
    importer_stub.PackageImporter = type("PackageImporter", (), {})
    sys.modules["sb_anim_test_runtime.importer"] = importer_stub

    constants_path = ADDON_ROOT / "runtime" / "constants.py"
    constants = types.ModuleType("sb_anim_test_runtime.constants")
    constants.__file__ = str(constants_path)
    spec = __import__("importlib.util").util.spec_from_file_location(
        "sb_anim_test_runtime.constants", str(constants_path)
    )
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(constants)
    sys.modules["sb_anim_test_runtime.constants"] = constants

    source = (ADDON_ROOT / "runtime" / "package_ops.py").read_text()
    source = source.replace("from ..manifest import", "from sb_anim_test_addon.manifest import")
    source = source.replace("from ..palette import", "from sb_anim_test_addon.palette import")
    source = source.replace("from .constants import", "from sb_anim_test_runtime.constants import")
    source = source.replace("from .validators import", "from sb_anim_test_runtime.validators import")
    source = source.replace(
        "from .importer import PackageImporter",
        "from sb_anim_test_runtime.importer import PackageImporter",
    )
    module = types.ModuleType("sb_anim_test_runtime.package_ops")
    module.__file__ = str(ADDON_ROOT / "runtime" / "package_ops.py")
    module.__package__ = "sb_anim_test_runtime"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


# Pinned values from the live sidecar
# `ships/Packages/RSI Scorpius_LOD0_TEX0/scene.json`,
# clip "wings_deploy" (verified 2026-04-27).
# Position values are already in exported Blender-frame XYZ.
_TOP_LEFT_POS_FIRST = [0.023793935775756836, 0.8021461367607117, -1.3102056980133057]
_TOP_LEFT_POS_LAST = [0.023793935775756836, 0.8021461367607117, -1.3102056980133057]
_TOP_RIGHT_POS_FIRST = [-0.5459997653961182, 0.8021460771560669, 1.3102059364318848]
_TOP_RIGHT_POS_LAST = [0.023352086544036865, 0.8021460771560669, 1.6394926309585571]

# Bind positions in Blender local frame; see
# docs/StarBreaker/animation-research.md -> bind pose tables.
_TOP_LEFT_BIND = (-0.546, 0.802, -1.310)
_TOP_RIGHT_BIND = (-0.546, 0.802, 1.310)

_NATIVE_BLEND_ROTATOR_BIND = (-1.3102056980133057, -0.5459997653961182, -0.8021461367607117)
_NATIVE_BLEND_ROTATOR_BIND_ROT = (0.999048, 0.0, 0.0, -0.043619)
_NATIVE_BLEND_ROTATOR_SIDECAR_FIRST = [-1.3102056980133057, 0.8021461367607117, -0.5459997653961182]
_NATIVE_BLEND_ROTATOR_SIDECAR_ROT_FIRST = [0.999048, 0.000014, 0.043620, 0.000014]


class AnimationPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package_ops = _load_package_ops()

    # ---- pure decoder ------------------------------------------------

    def test_decode_identity_is_passthrough(self) -> None:
        decoded = self.package_ops._decode_animation_position([1.5, -2.25, 3.0], "identity")
        self.assertEqual(decoded, (1.5, -2.25, 3.0))

    def test_decode_legacy_swaps_y_and_z(self) -> None:
        # Legacy decoder retained for older sidecars: (x, z, -y) → (x, -z, y)
        decoded = self.package_ops._decode_animation_position([1.0, 2.0, 3.0], "legacy")
        self.assertEqual(decoded, (1.0, -3.0, 2.0))

    def test_animation_overlap_warnings_detects_shared_channel_time_overlap(self) -> None:
        warnings = self.package_ops._animation_overlap_warnings(
            "landing_gear_extend",
            10.0,
            20.0,
            {"hash_a", "hash_b"},
            [
                {
                    "id": "existing-1",
                    "animation_name": "landing_gear_retract",
                    "start_frame": 20.0,
                    "duration_frames": 15.0,
                    "driven_hashes": ["hash_b", "hash_c"],
                }
            ],
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("landing_gear_retract", warnings[0])
        self.assertIn("shared channels", warnings[0])

    def test_animation_overlap_warnings_skips_non_overlapping_or_disjoint_channels(self) -> None:
        warnings = self.package_ops._animation_overlap_warnings(
            "wings_deploy",
            1.0,
            10.0,
            {"hash_a"},
            [
                {
                    "id": "existing-1",
                    "animation_name": "other",
                    "start_frame": 20.0,
                    "duration_frames": 10.0,
                    "driven_hashes": ["hash_a"],
                },
                {
                    "id": "existing-2",
                    "animation_name": "other2",
                    "start_frame": 4.0,
                    "duration_frames": 10.0,
                    "driven_hashes": ["hash_z"],
                },
            ],
        )

        self.assertEqual(warnings, [])

    def test_animation_instance_from_value_parses_and_normalizes_payload(self) -> None:
        parsed = self.package_ops._animation_instance_from_value(
            {
                "id": "abc123",
                "animation_name": "landing_gear_extend",
                "start_frame": "12",
                "duration_frames": 30,
                "driven_hashes": ["h2", "h1", "h1"],
            }
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["id"], "abc123")
        self.assertEqual(parsed["animation_name"], "landing_gear_extend")
        self.assertEqual(parsed["start_frame"], 12.0)
        self.assertEqual(parsed["duration_frames"], 30.0)
        self.assertEqual(parsed["driven_hashes"], ["h1", "h2"])

    def test_animation_instance_from_value_rejects_missing_required_fields(self) -> None:
        self.assertIsNone(self.package_ops._animation_instance_from_value({"id": "only-id"}))
        self.assertIsNone(self.package_ops._animation_instance_from_value({"animation_name": "clip"}))

    # ---- _apply_best_channel_transform endpoint policy ---------------

    def _make_object(self, name: str, bind_loc: tuple[float, float, float]) -> _StubObject:
        return _StubObject(name=name, location=bind_loc)

    def _bind_data(self, bind_loc: tuple[float, float, float]) -> dict[str, object]:
        return {
            "location": list(bind_loc),
            "rotation_mode": "QUATERNION",
            "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
            "parent_distance": None,
        }

    def test_constant_position_track_keeps_bone_at_verbatim_sample(self) -> None:
        """Wing_Rotator_Top_Left has a constant position track in the
        wings_deploy sidecar. With verbatim position, the endpoint lands
        at the sampled position directly (CAF stores absolute parent-local
        values). The first and last samples are equal so the bone stays at
        the clip's authored resting position, which happens to be very
        close to bind for this bone.
        """
        obj = self._make_object("Wing_Rotator_Top_Left", _TOP_LEFT_BIND)
        channel = {
            "rotation": [[0.999, 0.0, 0.044, 0.0], [0.866, 0.0, 0.500, 0.0]],
            "position": [_TOP_LEFT_POS_FIRST, _TOP_LEFT_POS_LAST],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data(_TOP_LEFT_BIND),
            channel,
            frame_index=1,
            endpoint_policy="literal",
        )

        for axis, (got, want) in enumerate(zip(obj.location, _TOP_LEFT_POS_LAST)):
            self.assertAlmostEqual(
                got, want, places=5,
                msg=f"axis {axis}: constant track must use verbatim sample",
            )

    def test_authored_position_track_lands_at_verbatim_last_sample(self) -> None:
        """Wing_Rotator_Top_Right's clip moves the bone between frame 0
        and the last frame. With verbatim position apply, the endpoint
        lands at the last sample directly (CAF stores absolute parent-local
        values, not deltas relative to any anchor).
        """
        obj = self._make_object("Wing_Rotator_Top_Right", _TOP_RIGHT_BIND)
        channel = {
            "rotation": [[0.999, 0.0, -0.044, 0.0], [0.866, 0.0, -0.500, 0.0]],
            "position": [_TOP_RIGHT_POS_FIRST, _TOP_RIGHT_POS_LAST],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data(_TOP_RIGHT_BIND),
            channel,
            frame_index=1,
            endpoint_policy="literal",
        )

        for axis, (got, want) in enumerate(zip(obj.location, _TOP_RIGHT_POS_LAST)):
            self.assertAlmostEqual(got, want, places=5, msg=f"axis {axis}")

    def test_native_blend_source_basis_decoder_matches_bind_endpoint(self) -> None:
        obj = self._make_object("Wing_Rotator_Top_Left", _NATIVE_BLEND_ROTATOR_BIND)
        channel = {
            "rotation": [_NATIVE_BLEND_ROTATOR_SIDECAR_ROT_FIRST, [0.984808, 0.0, 0.173648, 0.0]],
            "position": [_NATIVE_BLEND_ROTATOR_SIDECAR_FIRST, _NATIVE_BLEND_ROTATOR_SIDECAR_FIRST],
        }
        bind_data = self._bind_data(_NATIVE_BLEND_ROTATOR_BIND)
        bind_data["rotation_quaternion"] = list(_NATIVE_BLEND_ROTATOR_BIND_ROT)

        self.package_ops._apply_best_channel_transform(
            obj,
            bind_data,
            channel,
            frame_index=0,
            endpoint_policy="literal",
            sample_decoder="source",
        )

        for axis, (got, want) in enumerate(zip(obj.location, _NATIVE_BLEND_ROTATOR_BIND)):
            self.assertAlmostEqual(got, want, places=5, msg=f"axis {axis}")
        for axis, (got, want) in enumerate(zip(obj.rotation_quaternion, _NATIVE_BLEND_ROTATOR_BIND_ROT)):
            self.assertAlmostEqual(got, want, places=4, msg=f"quat {axis}")

    def test_animation_sample_decoder_uses_bind_pose_evidence(self) -> None:
        obj = self._make_object("Wing_Rotator_Top_Left", _NATIVE_BLEND_ROTATOR_BIND)
        bind_data = self._bind_data(_NATIVE_BLEND_ROTATOR_BIND)
        bind_data["rotation_quaternion"] = list(_NATIVE_BLEND_ROTATOR_BIND_ROT)
        channel = {
            "rotation": [_NATIVE_BLEND_ROTATOR_SIDECAR_ROT_FIRST, [0.984808, 0.0, 0.173648, 0.0]],
            "position": [_NATIVE_BLEND_ROTATOR_SIDECAR_FIRST, _NATIVE_BLEND_ROTATOR_SIDECAR_FIRST],
        }

        decoder = self.package_ops._select_animation_sample_decoder(
            [(obj, "0xDEADBEEF", bind_data, channel)]
        )

        self.assertEqual(decoder, "source")

    def test_prefixed_native_blend_object_matches_source_node_suffix_hash(self) -> None:
        root = self._make_object("StarBreaker RSI Aurora Mk2_LOD0_TEX0", (0.0, 0.0, 0.0))
        child = self._make_object("geo_nose_167_anim_wing_top_left", (1.0, 2.0, 3.0))
        root.children_recursive = [child]
        source_hash = f"0x{zlib.crc32(b'anim_wing_top_left') & 0xFFFFFFFF:08X}"
        bones = {
            source_hash: [
                {
                    "rotation": [[1.0, 0.0, 0.0, 0.0]],
                }
            ]
        }

        candidates, _policy, _decoder = self.package_ops._animation_bone_candidates(root, bones)

        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0][0], child)
        self.assertEqual(candidates[0][1], source_hash)

    def test_trailing_instance_suffix_matches_source_node_hash(self) -> None:
        root = self._make_object("StarBreaker DRAK Clipper_LOD0_TEX0", (0.0, 0.0, 0.0))
        child = self._make_object("door_upper_anim_001", (2.304258, 0.0, 0.327204))
        root.children_recursive = [child]
        source_hash = f"0x{zlib.crc32(b'door_upper_anim') & 0xFFFFFFFF:08X}"
        bones = {
            source_hash: [
                {
                    "position": [[2.304258, -0.327204, 0.0], [1.360307, 0.015211, 0.0]],
                    "rotation": [[-0.300719, 0.0, 0.0, 0.953713], [1.0, 0.0, 0.0, 0.0]],
                }
            ]
        }

        candidates, _policy, _decoder = self.package_ops._animation_bone_candidates(root, bones)

        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0][0], child)
        self.assertEqual(candidates[0][1], source_hash)

    def test_channel_variant_selection_uses_parent_hierarchy_provenance(self) -> None:
        rear_parent = self._make_object("door_upper_anim_001_0_drak_clipper_landing_gear_rear", (0, 0, 0))
        rear_obj = self._make_object("door_upper_anim_001_14_swingarm_big_anim", (0, 0, 0))
        rear_obj.parent = rear_parent
        front = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/landing_gear/drak_clipper_landing_gear_front.cga",
            "position": [[0.0, 0.0, 0.0]],
        }
        rear = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/landing_gear/drak_clipper_landing_gear_rear.cga",
            "rotation": [[1.0, 0.0, 0.0, 0.0]],
        }

        self.assertIs(self.package_ops._select_channel_variant_for_object(rear_obj, [front, rear]), rear)

    def test_channel_variant_confidence_reports_decisive_provenance_match(self) -> None:
        left_parent = self._make_object("body_105_hardpoint_tail_landing_gear_left", (0, 0, 0))
        left_obj = self._make_object("BONE_foot_001", (0.0, 0.5282, -2.7299))
        left_obj.parent = left_parent
        front = {
            "source_skeleton_path": "Objects/.../DRAK_Corsair_Landing_Gear_Front_CHR.chr",
            "position": [[0.0, 0.5282, -2.7299]],
        }
        left = {
            "source_skeleton_path": "Objects/.../DRAK_Corsair_Landing_Gear_Left_CHR.chr",
            "position": [[0.0, 0.158, -2.746]],
        }

        channel, decisive = self.package_ops._select_channel_variant_with_confidence(
            left_obj, [front, left]
        )
        self.assertIs(channel, left)
        self.assertTrue(decisive)

    def test_channel_variant_confidence_is_false_when_variants_tie(self) -> None:
        obj = self._make_object("BONE_foot_001", (0.0, 0.5282, -2.7299))
        first = {"position": [[0.0, 0.0, 0.0]]}
        second = {"position": [[1.0, 0.0, 0.0]]}

        channel, decisive = self.package_ops._select_channel_variant_with_confidence(
            obj, [first, second]
        )
        self.assertIs(channel, first)
        self.assertFalse(decisive)

    def test_channel_variant_selection_prefers_direct_mesh_asset_over_parent_tokens(self) -> None:
        rear_parent = self._make_object("body_7_drak_clipper_landing_gear_rear_door_right", (0, 0, 0))
        exterior_obj = self._make_object("door_upper_anim", (0, 0, 0))
        exterior_obj.parent = rear_parent
        exterior_obj["starbreaker_mesh_asset"] = (
            "Data/Objects/Spaceships/Ships/DRAK/clipper/exterior/drak_clipper_LOD0.blend"
        )
        exterior = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/exterior/drak_clipper.cga",
            "position": [[1.330354, -0.673419, 0.0]],
        }
        rear = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/landing_gear/drak_clipper_landing_gear_rear.cga",
            "position": [[2.304258, -0.327204, 0.0]],
        }

        self.assertIs(
            self.package_ops._select_channel_variant_for_object(exterior_obj, [exterior, rear]),
            exterior,
        )

    def test_channel_variant_direction_uses_parent_not_bone_function_name(self) -> None:
        rear_parent = self._make_object("door_upper_anim_001_23_Foot_joint_anim", (0, 0, 0))
        rear_grandparent = self._make_object("door_upper_anim_001_0_drak_clipper_landing_gear_rear", (0, 0, 0))
        rear_parent.parent = rear_grandparent
        rear_step = self._make_object("door_upper_anim_001_28_front_step_anim", (0, 0, 0))
        rear_step.parent = rear_parent
        front = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/landing_gear/drak_clipper_landing_gear_front.cga",
            "position": [[0.0, 0.298622, -0.439994]],
        }
        rear = {
            "source_skeleton_path": "Objects/Spaceships/Ships/DRAK/clipper/landing_gear/drak_clipper_landing_gear_rear.cga",
            "position": [[1.325272, 0.440688, 0.0]],
        }

        self.assertIs(
            self.package_ops._select_channel_variant_for_object(rear_step, [front, rear]),
            rear,
        )

    def test_override_blend_mode_uses_sample_verbatim(self) -> None:
        """Phase 38 override path. When the per-bone `blend_mode` is
        marked as `override` (CHR-bind sits outside the AABB of CAF
        position samples), the addon must use the sampled position
        verbatim and ignore the bind. The canonical real-world case
        is Scorpius `BONE_Front_Landing_Gear_Foot`, whose CHR-bind is
        ~1.81m off any clip sample. With the additive pathway that
        bone lands far off the gear; with the override pathway it
        lands at the sample exactly.
        """
        bind = (10.0, 0.0, 0.0)
        sample = [3.5, 1.25, -0.75]
        obj = self._make_object("BONE_Front_Landing_Gear_Foot", bind)
        channel = {
            "rotation": [[1.0, 0.0, 0.0, 0.0]],
            "position": [sample],
            "blend_mode": "override",
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data(bind),
            channel,
            frame_index=0,
            endpoint_policy="transition_end",
        )
        for axis, (got, want) in enumerate(zip(obj.location, sample)):
            self.assertAlmostEqual(
                got, want, places=5,
                msg=f"axis {axis}: override mode must use sample verbatim",
            )

    def test_position_track_matches_bind_detects_incompatible_duplicate(self) -> None:
        positions = [
            [0.0, 0.099973, -1.002515],
            [0.0, 0.099973, 0.530196],
        ]

        self.assertTrue(
            self.package_ops._position_track_matches_bind((0.0, 0.1, 0.53), positions),
            "front swingarm bind should match the shared track",
        )
        self.assertFalse(
            self.package_ops._position_track_matches_bind((0.788171, -0.460876, -0.0005), positions),
            "rear swingarm bind should be rejected for the shared track",
        )

    def test_shared_hash_position_policy_suppresses_incompatible_duplicates(self) -> None:
        channel = {
            "position": [
                [0.0, 0.099973, -1.002515],
                [0.0, 0.099973, 0.530196],
            ]
        }
        front = self._make_object("swingarm_big_anim.001", (0.0, 0.1, 0.53))
        rear = self._make_object("swingarm_big_anim.003", (0.788171, -0.460876, -0.0005))

        policy = self.package_ops._shared_hash_position_policy(
            {
                "0x2522C378": [
                    (front, self._bind_data((0.0, 0.1, 0.53))),
                    (rear, self._bind_data((0.788171, -0.460876, -0.0005))),
                ]
            },
            {"0x2522C378": channel},
        )

        self.assertTrue(policy[id(front)])
        self.assertFalse(policy[id(rear)])

    def test_transform_application_skips_position_but_preserves_rotation_for_mismatched_duplicate(self) -> None:
        bind = (2.304258, -0.327204, 0.0)
        obj = _StubObject(
            "door_upper_anim.004",
            location=bind,
            rotation_quaternion=(0.300706, 0.0, 0.0, -0.953717),
        )
        bind_data = {
            "location": list(bind),
            "rotation_mode": "QUATERNION",
            "rotation_quaternion": [0.300706, 0.0, 0.0, -0.953717],
            "parent_distance": None,
        }
        channel = {
            "rotation": [[1.0, 0.0, 0.0, 0.0], [-0.300719, 0.0, 0.0, 0.953713]],
            "position": [[0.386403, -0.331004, -0.000001], [1.330354, -0.673419, -0.000001]],
        }

        self.package_ops._apply_best_channel_transform(
            obj,
            bind_data,
            channel,
            frame_index=0,
            endpoint_policy="literal",
            allow_rotation=True,
            allow_position=False,
        )

        for axis, (got, want) in enumerate(zip(obj.location, bind)):
            self.assertAlmostEqual(got, want, places=5, msg=f"location axis {axis}")
        expected_rot = (1.0, 0.0, 0.0, 0.0)
        for axis, (got, want) in enumerate(zip(obj.rotation_quaternion, expected_rot)):
            self.assertAlmostEqual(got, want, places=5, msg=f"rotation axis {axis}")

    def test_apply_animation_pose_does_not_position_incompatible_duplicate_hash(self) -> None:
        root = self._make_object("StarBreaker Test", (0.0, 0.0, 0.0))
        incompatible = self._make_object("shared_node", (10.0, 0.0, 0.0))
        compatible = self._make_object("shared_node_001", (0.0, 0.0, 0.0))
        root.children_recursive = [incompatible, compatible]
        source_hash = f"0x{zlib.crc32(b'shared_node') & 0xFFFFFFFF:08X}"
        clip = {
            "bones": {
                source_hash: {
                    "position": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "rotation": [[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]],
                }
            }
        }

        self.package_ops._apply_animation_pose(root, clip, 1)

        self.assertEqual(incompatible.location, (10.0, 0.0, 0.0))
        self.assertEqual(compatible.location, (1.0, 0.0, 0.0))
        for got, want in zip(incompatible.rotation_quaternion, (0.707, 0.707, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=5)

    # ---- Phase 39: layered-Action helpers ---------------------------

    def test_action_fcurves_handles_legacy_action(self) -> None:
        """Pre-Blender-4.4 Actions expose `Action.fcurves` directly.
        The helper must return that collection unchanged.
        """
        legacy_fcurves = ["fc_loc_x", "fc_loc_y", "fc_loc_z", "fc_quat_w"]
        action = types.SimpleNamespace(fcurves=legacy_fcurves)
        result = self.package_ops._action_fcurves(action)
        self.assertEqual(result, legacy_fcurves)

    def test_action_fcurves_walks_layered_channelbag(self) -> None:
        """Blender 5.1 Actions have no `Action.fcurves`; fcurves live on
        `action.layers[*].strips[*].channelbag(slot).fcurves`. The helper
        must enumerate them via the layered API.
        """
        slot = object()
        cb_fcurves = ["fc_loc_x", "fc_loc_y", "fc_quat_w"]
        channelbag = types.SimpleNamespace(fcurves=cb_fcurves)
        strip = types.SimpleNamespace(
            channelbag=lambda s, ensure=False: channelbag if s is slot else None
        )
        layer = types.SimpleNamespace(strips=[strip])
        action = types.SimpleNamespace(layers=[layer], slots=[slot])
        # Make sure attempting to access `.fcurves` does NOT yield the
        # legacy attribute.
        self.assertFalse(hasattr(action, "fcurves"))
        result = self.package_ops._action_fcurves(action)
        self.assertEqual(result, cb_fcurves)

    def test_action_groups_collection_returns_layered_channelbag_groups(self) -> None:
        """The Phase 39 regression: `Action.groups` is removed in Blender
        5.1 and grouping must come from the layered channelbag instead.
        Looking up `action.groups` directly used to abort the Insert
        Action loop after the first bone; the helper must transparently
        return the channelbag's groups collection.
        """

        class _Groups:
            def __init__(self) -> None:
                self._items: dict[str, object] = {}

            def get(self, name: str) -> object | None:
                return self._items.get(name)

            def new(self, name: str) -> object:
                grp = object()
                self._items[name] = grp
                return grp

        cb_groups = _Groups()
        slot = object()
        channelbag = types.SimpleNamespace(groups=cb_groups)
        strip = types.SimpleNamespace(
            channelbag=lambda s, ensure=False: channelbag if s is slot else None
        )
        layer = types.SimpleNamespace(strips=[strip])
        action = types.SimpleNamespace(layers=[layer], slots=[slot])
        self.assertFalse(hasattr(action, "groups"))
        result = self.package_ops._action_groups_collection(action)
        self.assertIs(result, cb_groups)
        # The helper must support the regular Insert-Action call sequence:
        # caller does `groups.get(name)` → falsy → `groups.new(name)`.
        self.assertIsNone(result.get("BoneA"))
        new_group = result.new("BoneA")
        self.assertIs(result.get("BoneA"), new_group)

    def test_action_groups_collection_returns_none_when_no_data(self) -> None:
        """A freshly-created layered Action with no keyframes inserted
        yet has no slots/channelbags. The helper must return None
        instead of raising — callers then skip grouping silently.
        """
        action = types.SimpleNamespace(layers=[], slots=[])
        self.assertIsNone(self.package_ops._action_groups_collection(action))

    def test_endpoint_policy_literal_picks_first_at_frame_zero(self) -> None:
        obj = self._make_object("Wing_Rotator_Top_Right", _TOP_RIGHT_BIND)
        channel = {
            "rotation": [[0.999, 0.0, -0.044, 0.0], [0.866, 0.0, -0.500, 0.0]],
            "position": [_TOP_RIGHT_POS_FIRST, _TOP_RIGHT_POS_LAST],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data(_TOP_RIGHT_BIND),
            channel,
            frame_index=0,
            endpoint_policy="literal",
        )
        # At frame 0 the literal policy picks pos[0]. This endpoint is
        # bind-equivalent, so CAF quantization residual is snapped back
        # to the stored bind pose.
        for axis, (got, want) in enumerate(zip(obj.location, _TOP_RIGHT_BIND)):
            self.assertAlmostEqual(got, want, places=5, msg=f"axis {axis}")
        # Rotation at frame 0 is bind-equivalent (~5° tilt).
        self.assertAlmostEqual(obj.rotation_quaternion[0], 0.999, places=3)

    def test_rotation_sample_writes_quaternion_unchanged(self) -> None:
        obj = self._make_object("Wing_Mechanism_Top_Left", (0.0, 0.0, 0.0))
        channel = {
            "rotation": [[1.0, 0.0, 0.0, 0.0], [0.985, 0.174, 0.0, 0.0]],
            "position": [],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data((0.0, 0.0, 0.0)),
            channel,
            frame_index=1,
            endpoint_policy="literal",
        )
        self.assertEqual(obj.rotation_mode, "QUATERNION")
        for got, want in zip(obj.rotation_quaternion, (0.985, 0.174, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=5)

    def test_fragment_tagged_cyclic_clip_targets_mid_transition_time(self) -> None:
        clip = {
            "fragments": [{"frag_tags": ["Open"], "animations": [{"name": "canopy_open"}]}],
            "bones": {
                "0x00000001": {
                    "position": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.1, 0.0]],
                    "position_time": [0.0, 36.5, 75.0],
                },
                "0x00000002": {
                    "position": [[1.0, 0.0, 0.0], [1.0, -3.0, 0.0], [1.0, -0.1, 0.0]],
                    "position_time": [0.0, 36.5, 75.0],
                },
            },
        }

        self.assertEqual(self.package_ops._clip_cyclic_transition_target_frame(clip), 36.5)

    def test_cyclic_target_requires_source_fragment_metadata(self) -> None:
        clip = {
            "bones": {
                "0x00000001": {
                    "position": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.1, 0.0]],
                    "position_time": [0.0, 36.5, 75.0],
                }
            }
        }

        self.assertIsNone(self.package_ops._clip_cyclic_transition_target_frame(clip))

    def test_mixed_cyclic_clip_keeps_literal_endpoint(self) -> None:
        clip = {
            "fragments": [{"frag_tags": ["Deploy"], "animations": [{"name": "landing_gear_extend"}]}],
            "bones": {
                "0x00000001": {
                    "position": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.1, 0.0]],
                    "position_time": [0.0, 50.0, 100.0],
                },
                "0x00000002": {
                    "position": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    "position_time": [0.0, 50.0, 100.0],
                },
                "0x00000003": {
                    "position": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    "position_time": [0.0, 50.0, 100.0],
                },
            },
        }

        self.assertIsNone(self.package_ops._clip_cyclic_transition_target_frame(clip))

    def test_fragment_endpoint_policy_maps_state_tags_to_transition(self) -> None:
        deploy = {
            "fragment": "Landing_Gear",
            "frag_tags": ["Deploy"],
            "animations": [{"name": "landing_gear_extend"}],
        }
        retract = {
            "fragment": "Landing_Gear",
            "frag_tags": ["Retract"],
            "animations": [{"name": "landing_gear_extend", "speed": -1}],
        }

        # Forward fragment (Deploy): snap_first -> start, snap_last -> end.
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy, "snap_first"),
            "transition_start",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy, "snap_last"),
            "transition_end",
        )
        # Reverse-playback fragment (Retract, speed=-1): mapping flips.
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(retract, "snap_first"),
            "transition_end",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(retract, "snap_last"),
            "transition_start",
        )

    def test_fragment_endpoint_policy_uses_semantic_reverse_when_speed_missing(self) -> None:
        deploy_with_swapped_clip = {
            "fragment": "Wings",
            "frag_tags": ["Deploy"],
            "animations": [{"name": "drak_corsair_wings_retract"}],
        }
        retract_with_swapped_clip = {
            "fragment": "Wings",
            "frag_tags": ["Retract"],
            "animations": [{"name": "drak_corsair_wings_deploy"}],
        }

        # Missing speed metadata, but semantic mismatch indicates reverse.
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy_with_swapped_clip, "snap_first"),
            "transition_end",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy_with_swapped_clip, "snap_last"),
            "transition_start",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(retract_with_swapped_clip, "snap_first"),
            "transition_end",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(retract_with_swapped_clip, "snap_last"),
            "transition_start",
        )

    def test_fragment_endpoint_policy_prefers_explicit_speed_over_semantics(self) -> None:
        deploy_with_explicit_forward_speed = {
            "fragment": "Wings",
            "frag_tags": ["Deploy"],
            "animations": [{"name": "drak_corsair_wings_retract", "speed": 1.0}],
        }

        # Explicit speed metadata is authoritative and keeps forward mapping.
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy_with_explicit_forward_speed, "snap_first"),
            "transition_start",
        )
        self.assertEqual(
            self.package_ops._fragment_endpoint_policy(deploy_with_explicit_forward_speed, "snap_last"),
            "transition_end",
        )

    def test_action_mode_uses_semantic_reverse_playback_fallback(self) -> None:
        clip = {"name": "drak_corsair_wings_retract", "bones": {}}
        fragment = {
            "fragment": "Wings",
            "frag_tags": ["Deploy"],
            "animations": [{"name": "drak_corsair_wings_retract"}],
        }
        package_root = _StubObject("package-root")
        package = object()

        captured: dict[str, object] = {}
        original_load = self.package_ops._load_package_from_root
        original_find = self.package_ops._find_animation_selection
        original_apply = self.package_ops._apply_animation_mode_for_clip
        original_mode_map = self.package_ops.package_animation_mode_map
        try:
            self.package_ops._load_package_from_root = lambda _root: package
            self.package_ops._find_animation_selection = lambda _package, _name: (clip, fragment)
            self.package_ops.package_animation_mode_map = lambda _root: {}

            def _fake_apply(
                context,
                root,
                package_bundle,
                clip_data,
                mode,
                animation_name=None,
                fragment=None,
                reverse_playback=False,
                fps_policy="adapt_scene",
            ):
                captured["reverse_playback"] = reverse_playback
                captured["mode"] = mode
                captured["animation_name"] = animation_name
                return 1

            self.package_ops._apply_animation_mode_for_clip = _fake_apply
            updated = self.package_ops.apply_animation_mode_to_package_root(
                context=types.SimpleNamespace(scene=types.SimpleNamespace(frame_set=lambda *_a, **_kw: None)),
                package_root=package_root,
                animation_name="fragment:0:drak_corsair_wings_retract",
                mode="action",
            )
        finally:
            self.package_ops._load_package_from_root = original_load
            self.package_ops._find_animation_selection = original_find
            self.package_ops._apply_animation_mode_for_clip = original_apply
            self.package_ops.package_animation_mode_map = original_mode_map

        self.assertEqual(updated, 1)
        self.assertEqual(captured.get("mode"), "action")
        self.assertEqual(captured.get("animation_name"), "fragment:0:drak_corsair_wings_retract")
        self.assertTrue(captured.get("reverse_playback"))

    def test_reverse_snap_first_uses_cyclic_target_for_transition_end(self) -> None:
        clip = {
            "name": "landing_gear_extend",
            "source_fragment": {
                "fragment": "Landing_Gear",
                "frag_tags": ["Retract"],
                "animations": [{"name": "landing_gear_extend", "speed": -1}],
            },
        }
        capture: dict[str, object] = {}
        original_apply = self.package_ops._apply_animation_pose
        original_target = self.package_ops._clip_cyclic_transition_target_frame
        try:
            self.package_ops._clip_cyclic_transition_target_frame = lambda _clip: 36.5

            def _fake_apply(
                package_root,
                clip_data,
                frame_index,
                endpoint_policy,
                target_frame=None,
                anchor_frame=None,
            ):
                capture["frame_index"] = frame_index
                capture["endpoint_policy"] = endpoint_policy
                capture["target_frame"] = target_frame
                capture["anchor_frame"] = anchor_frame
                return 1

            self.package_ops._apply_animation_pose = _fake_apply
            updated = self.package_ops._apply_animation_mode_for_clip(
                context=None,
                package_root=object(),
                package=object(),
                clip=clip,
                mode="snap_first",
                fragment=clip["source_fragment"],
                reverse_playback=True,
            )
        finally:
            self.package_ops._apply_animation_pose = original_apply
            self.package_ops._clip_cyclic_transition_target_frame = original_target

        self.assertEqual(updated, 1)
        self.assertEqual(capture["frame_index"], 0)
        self.assertEqual(capture["endpoint_policy"], "transition_end")
        self.assertEqual(capture["target_frame"], 36.5)
        self.assertEqual(capture["anchor_frame"], 36.5)

    def test_paired_clip_fallback_uses_fragment_transition_policy(self) -> None:
        primary_clip = {"name": "landing_gear_retract.caf", "bones": {}}
        paired_fragment = {
            "frag_tags": ["Retract"],
            "animations": [{"name": "landing_gear_deploy", "speed": -1}],
        }
        paired_clip = {
            "name": "landing_gear_deploy.caf",
            "bones": {"001": {"rotation": [[1, 0, 0, 0]]}},
            "source_fragment": paired_fragment,
        }

        calls: list[dict[str, object]] = []
        original_apply = self.package_ops._apply_animation_pose
        original_pair = self.package_ops._paired_clip_for_snap
        original_target = self.package_ops._clip_cyclic_transition_target_frame
        try:
            self.package_ops._paired_clip_for_snap = lambda package, clip, frame: (paired_clip, -1)
            self.package_ops._clip_cyclic_transition_target_frame = (
                lambda clip_data: 24.0 if clip_data is paired_clip else None
            )

            def _fake_apply(
                package_root,
                clip_data,
                frame_index,
                endpoint_policy,
                target_frame=None,
                anchor_frame=None,
            ):
                calls.append(
                    {
                        "clip": clip_data,
                        "frame_index": frame_index,
                        "endpoint_policy": endpoint_policy,
                        "target_frame": target_frame,
                        "anchor_frame": anchor_frame,
                    }
                )
                return 0 if len(calls) == 1 else 1

            self.package_ops._apply_animation_pose = _fake_apply
            updated = self.package_ops._apply_animation_mode_for_clip(
                context=None,
                package_root=object(),
                package=object(),
                clip=primary_clip,
                mode="snap_first",
                fragment=paired_fragment,
                reverse_playback=True,
            )
        finally:
            self.package_ops._apply_animation_pose = original_apply
            self.package_ops._paired_clip_for_snap = original_pair
            self.package_ops._clip_cyclic_transition_target_frame = original_target

        self.assertEqual(updated, 1)
        self.assertEqual(len(calls), 2)
        fallback = calls[1]
        self.assertIs(fallback["clip"], paired_clip)
        self.assertEqual(fallback["endpoint_policy"], "transition_end")
        self.assertEqual(fallback["frame_index"], -1)
        self.assertEqual(fallback["target_frame"], 24.0)
        self.assertEqual(fallback["anchor_frame"], 24.0)

    def test_target_frame_snap_opens_from_bind_anchored_start(self) -> None:
        # snap_last on a cyclic clip whose first sample is
        # bind-aligned (the bone's resting state in clip-frame) and
        # whose target frame holds the deployed pose: anchor must be
        # the bind-nearer endpoint (here the first sample), so
        # `result = bind + (target - first)` lands at the target.
        obj = self._make_object("Canopy_Front", (0.0, 0.0, 0.0))
        channel = {
            "position": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.1, 0.0]],
            "position_time": [0.0, 36.5, 75.0],
        }

        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data((0.0, 0.0, 0.0)),
            channel,
            frame_index=-1,
            endpoint_policy="literal",
            target_frame=36.5,
            anchor_frame=36.5,
        )

        self.assertEqual(obj.location, (0.0, 2.0, 0.0))

    def test_target_frame_snap_closes_to_bind_when_target_is_reference(self) -> None:
        obj = self._make_object("Canopy_Front", (0.0, 0.0, 0.0))
        channel = {
            "position": [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            "position_time": [0.0, 36.5, 75.0],
        }

        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data((0.0, 0.0, 0.0)),
            channel,
            frame_index=-1,
            endpoint_policy="literal",
            target_frame=36.5,
            anchor_frame=36.5,
        )

        self.assertEqual(obj.location, (0.0, 0.0, 0.0))

    def test_snap_first_can_use_target_frame_as_anchor_reference(self) -> None:
        obj = self._make_object("Canopy_Front", (0.0, 0.0, 0.0))
        channel = {
            "position": [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            "position_time": [0.0, 36.5, 75.0],
        }

        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data((0.0, 0.0, 0.0)),
            channel,
            frame_index=0,
            endpoint_policy="literal",
            anchor_frame=36.5,
        )

        self.assertEqual(obj.location, (0.0, 2.0, 0.0))

    # ---- Phase 53 hotfix: clip-root-frame DBA fields are research-only --

    def test_clip_start_position_does_not_displace_per_bone_anchor(self) -> None:
        """Regression: Phase 53 originally consumed DBA `start_position`
        as a per-bone anchor, which displaced bones whose first
        authored sample was not at the clip-root origin (Scorpius
        landing_gear_deploy: bones "completely separated"). The DBA
        fields are clip-root-frame, not parent-local, so they are
        plumbed through scene.json but not used per-bone. Anchor
        selection must remain a bind-distance choice between
        `first_sample` and the `anchor_frame` sample.
        """
        obj = self._make_object("BONE_Back_Piston_Lower", (0.0, 0.0, 0.0))
        channel = {
            # First sample is bind-aligned; target deploys the bone.
            "position": [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            "position_time": [0.0, 36.5, 75.0],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            self._bind_data((0.0, 0.0, 0.0)),
            channel,
            frame_index=-1,
            endpoint_policy="literal",
            target_frame=36.5,
            anchor_frame=36.5,
        )
        # bind + (target - first) = (0, 0.5, 0) — not (0, 0, 0)+(0, 0.5, 0)
        # piled on a non-zero clip-root start.
        self.assertEqual(obj.location, (0.0, 0.5, 0.0))

    def test_drak_door_cyclic_returns_to_bind_via_first_sample_anchor(self) -> None:
        """DRAK Clipper shape: cyclic rotation that returns to identity.
        Without the retired `_channel_other_endpoint` heuristic, the
        first-sample anchor leaves snap_last at bind even when a
        mid-clip extreme is present.
        """
        obj = self._make_object("door_lower_anim", (0.0, 0.0, 0.0))
        bind = self._bind_data((0.0, 0.0, 0.0))
        channel = {
            "rotation": [
                [1.0, 0.0, 0.0, 0.0],
                [0.707, 0.707, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            "rotation_time": [0.0, 36.5, 75.0],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            bind,
            channel,
            frame_index=-1,
            endpoint_policy="transition_end",
        )
        for actual, expected in zip(obj.rotation_quaternion, (1.0, 0.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_literal_snap_last_picks_last_sample_as_anchor_when_last_is_nearer_bind(self) -> None:
        """Regression guard for DRAK Clipper swingarm pattern.
        snap_last on a literal clip where `last_pos ≈ bind` and
        `first_pos` is far away: the bind-distance pick must choose
        `last` as anchor so the result is bind + (last - last) = bind,
        not bind + (last - first) = displaced.
        """
        obj = self._make_object("swingarm_big_anim", (0.0, 0.1, 0.53))
        bind = self._bind_data((0.0, 0.1, 0.53))
        channel = {
            # first ≈ retracted (far), last ≈ bind (deployed rest pose)
            "position": [[0.0, 0.1, -1.0], [0.0, 0.1, 0.53]],
            "position_time": [0.0, 75.0],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            bind,
            channel,
            frame_index=-1,
            endpoint_policy="literal",
        )
        # verbatim last = (0, 0.1, 0.53) = bind → no displacement
        for actual, expected in zip(obj.location, (0.0, 0.1, 0.53)):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_caf_only_clip_uses_first_sample_anchor(self) -> None:
        """CAF-only clips have no DBA metadata. Positions use verbatim
        sample (absolute parent-local); rotations use verbatim sample.
        """
        obj = self._make_object("wing_pivot", (0.0, 0.0, 0.0))
        bind = self._bind_data((0.0, 0.0, 0.0))
        channel = {
            "rotation": [[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]],
            "position": [[0.0, 1.5, 0.0], [0.0, 5.5, 0.0]],
        }
        self.package_ops._apply_best_channel_transform(
            obj,
            bind,
            channel,
            frame_index=-1,
            endpoint_policy="transition_end",
        )
        # verbatim last position = (0, 5.5, 0) regardless of bind or first
        self.assertEqual(obj.location, (0.0, 5.5, 0.0))
        for actual, expected in zip(obj.rotation_quaternion, (0.707, 0.0, 0.707, 0.0)):
            self.assertAlmostEqual(actual, expected, places=5)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
