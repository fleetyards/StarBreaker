//! Resample a CAF/DBA clip onto a skeleton as per-frame local TRS, ready
//! for glTF animation channels on the rigged joint nodes.
//!
//! Character clip keys are parent-relative local transforms in the same
//! CryEngine model frame as the `.skin`/`.chr` CompiledBones (no axis swap):
//! a bone's key at a frame replaces its bind-pose local rotation/position.
//! Bones a clip has no channel for keep their bind pose. Additive clips
//! (`*_add`) store deltas instead and are not handled here.

use std::collections::HashMap;

use glam::{Quat, Vec3};

use super::{bone_name_hash, AnimationClip, Keyframe};
use crate::skeleton::{bind_local_transforms, bone_parent, Bone};

/// Options for [`sample_clip`].
#[derive(Debug, Clone, Copy)]
pub struct SampleOptions {
    /// Zero the horizontal (X/Y) translation of the skeleton root bones
    /// (`World`, the locomotion root), so a locomotion cycle plays in
    /// place. The vertical component and the root rotation are kept; the
    /// body's own bob and sway live on `Hips` and stay untouched.
    pub strip_root_translation: bool,
}

/// One clip resampled at its frame rate onto a skeleton.
#[derive(Debug, Clone)]
pub struct SampledClip {
    pub name: String,
    /// Where the clip came from (DBA/CAF path and clip name), for extras.
    pub source: String,
    pub fps: f32,
    /// Number of samples per track (1 for a static pose).
    pub frames: usize,
    /// Source span in frames (`last key - first key`).
    pub source_frames: f32,
    /// Every channel is constant: written as a single key.
    pub is_static: bool,
    /// First and last sample agree (pose within 1 degree / 5 mm), ignoring
    /// helper leaves (IK targets, PlaneWeight/PlaneTarget foot planting).
    pub looping: bool,
    /// `(bone index, local rotation per sample)`; only animated bones.
    pub rotations: Vec<(usize, Vec<Quat>)>,
    /// `(bone index, local translation per sample)`; only animated bones.
    pub translations: Vec<(usize, Vec<Vec3>)>,
    /// Clip channels that matched a skeleton bone / that did not.
    pub matched_channels: usize,
    pub unmatched_channels: usize,
    /// Largest horizontal root offset removed by `strip_root_translation`.
    pub root_motion_removed: f32,
}

impl SampledClip {
    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        (self.frames.saturating_sub(1)) as f32 / self.fps
    }
}

fn sorted<T: Clone>(keys: &[Keyframe<T>]) -> Vec<Keyframe<T>> {
    let mut keys = keys.to_vec();
    keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    keys
}

/// Index pair and blend factor bracketing `time` in sorted `keys`.
fn bracket<T>(keys: &[Keyframe<T>], time: f32) -> (usize, usize, f32) {
    let after = keys.partition_point(|k| k.time <= time);
    if after == 0 {
        return (0, 0, 0.0);
    }
    if after >= keys.len() {
        let last = keys.len() - 1;
        return (last, last, 0.0);
    }
    let (a, b) = (after - 1, after);
    let span = keys[b].time - keys[a].time;
    let t = if span > 0.0 { (time - keys[a].time) / span } else { 0.0 };
    (a, b, t)
}

fn eval_rotation(keys: &[Keyframe<[f32; 4]>], time: f32) -> Quat {
    let (a, b, t) = bracket(keys, time);
    let qa = Quat::from_array(keys[a].value).normalize();
    if a == b || t == 0.0 {
        return qa;
    }
    let qb = Quat::from_array(keys[b].value).normalize();
    qa.slerp(qb, t).normalize()
}

fn eval_position(keys: &[Keyframe<[f32; 3]>], time: f32) -> Vec3 {
    let (a, b, t) = bracket(keys, time);
    let pa = Vec3::from(keys[a].value);
    if a == b || t == 0.0 {
        return pa;
    }
    pa.lerp(Vec3::from(keys[b].value), t)
}

/// Flip quaternions onto the hemisphere of their predecessor so linear
/// interpolation between samples never takes the long way round.
fn make_sign_continuous(samples: &mut [Quat]) {
    for i in 1..samples.len() {
        if samples[i - 1].dot(samples[i]) < 0.0 {
            samples[i] = -samples[i];
        }
    }
}

/// Resample `clip` onto `bones` at integer frames over its key span.
///
/// Returns `None` when no channel of the clip names a skeleton bone.
pub fn sample_clip(
    clip: &AnimationClip,
    bones: &[Bone],
    name: &str,
    source: &str,
    opts: SampleOptions,
) -> Option<SampledClip> {
    let by_hash: HashMap<u32, usize> =
        bones.iter().enumerate().map(|(i, b)| (bone_name_hash(&b.name), i)).collect();
    let rest = bind_local_transforms(bones);

    let times = clip
        .channels
        .iter()
        .flat_map(|c| c.rotations.iter().map(|k| k.time).chain(c.positions.iter().map(|k| k.time)));
    let (start, end) = times.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), t| (lo.min(t), hi.max(t)));
    let (start, end) = if start.is_finite() { (start, end) } else { (0.0, 0.0) };
    let source_frames = end - start;
    let sample_count = source_frames.ceil().max(0.0) as usize + 1;
    let sample_time = |k: usize| (start + k as f32).min(end);

    let mut rotations = Vec::new();
    let mut translations = Vec::new();
    let mut matched = 0;
    let mut root_motion_removed = 0.0f32;
    for channel in &clip.channels {
        let Some(&bone) = by_hash.get(&channel.bone_hash) else {
            continue;
        };
        matched += 1;
        if !channel.rotations.is_empty() {
            let keys = sorted(&channel.rotations);
            let mut samples: Vec<Quat> = (0..sample_count).map(|k| eval_rotation(&keys, sample_time(k))).collect();
            make_sign_continuous(&mut samples);
            rotations.push((bone, samples));
        }
        if !channel.positions.is_empty() {
            let keys = sorted(&channel.positions);
            let mut samples: Vec<Vec3> = (0..sample_count).map(|k| eval_position(&keys, sample_time(k))).collect();
            if opts.strip_root_translation && bone_parent(bones, bone).is_none() {
                let rest_position = rest[bone].1;
                for sample in &mut samples {
                    let removed = (sample.truncate() - rest_position.truncate()).length();
                    root_motion_removed = root_motion_removed.max(removed);
                    sample.x = rest_position.x;
                    sample.y = rest_position.y;
                }
            }
            translations.push((bone, samples));
        }
    }
    if matched == 0 {
        return None;
    }
    rotations.sort_by_key(|(bone, _)| *bone);
    translations.sort_by_key(|(bone, _)| *bone);

    let rotation_close = |a: Quat, b: Quat, tolerance: f32| 1.0 - a.dot(b).abs().min(1.0) <= tolerance;
    let is_static = rotations.iter().all(|(_, s)| s.iter().all(|&q| rotation_close(q, s[0], 1e-7)))
        && translations.iter().all(|(_, s)| s.iter().all(|p| p.distance(s[0]) <= 1e-5));
    let mut frames = sample_count;
    if is_static {
        frames = 1;
        for (_, samples) in &mut rotations {
            samples.truncate(1);
        }
        for (_, samples) in &mut translations {
            samples.truncate(1);
        }
    }
    // Engine helper leaves (IK targets, PlaneWeight/PlaneTarget foot-plant
    // parameters) are not part of the visible pose and need not close.
    let mut has_children = vec![false; bones.len()];
    for index in 0..bones.len() {
        if let Some(parent) = bone_parent(bones, index) {
            has_children[parent] = true;
        }
    }
    let pose_bone = |bone: usize| {
        let name = &bones[bone].name;
        let under_root = bone_parent(bones, bone).is_some_and(|parent| bone_parent(bones, parent).is_none());
        has_children[bone] || !(under_root || name.starts_with("Plane") || name.contains("IK"))
    };
    // 1 degree: 1 - cos(0.5 deg) ~= 3.8e-5.
    let looping = frames > 1
        && rotations.iter().filter(|(b, _)| pose_bone(*b)).all(|(_, s)| rotation_close(s[0], s[frames - 1], 3.8e-5))
        && translations.iter().filter(|(b, _)| pose_bone(*b)).all(|(_, s)| s[0].distance(s[frames - 1]) <= 0.005);

    Some(SampledClip {
        name: name.to_string(),
        source: source.to_string(),
        fps: if clip.fps > 0.0 { clip.fps } else { 30.0 },
        frames,
        source_frames,
        is_static,
        looping,
        rotations,
        translations,
        matched_channels: matched,
        unmatched_channels: clip.channels.len() - matched,
        root_motion_removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::BoneChannel;

    fn bone(name: &str, parent: Option<u16>, position: [f32; 3]) -> Bone {
        Bone {
            name: name.into(),
            parent_index: parent,
            object_node_index: None,
            local_position: [0.0; 3],
            local_rotation: [1.0, 0.0, 0.0, 0.0],
            world_position: position,
            world_rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn channel(name: &str, rotations: Vec<(f32, [f32; 4])>, positions: Vec<(f32, [f32; 3])>) -> BoneChannel {
        BoneChannel {
            bone_hash: bone_name_hash(name),
            rotations: rotations.into_iter().map(|(time, value)| Keyframe { time, value }).collect(),
            positions: positions.into_iter().map(|(time, value)| Keyframe { time, value }).collect(),
            rot_format_flags: 0,
            pos_format_flags: 0,
        }
    }

    fn clip(channels: Vec<BoneChannel>) -> AnimationClip {
        AnimationClip { name: "test.caf".into(), fps: 30.0, channels, start_rotation: None, start_position: None }
    }

    fn skeleton() -> Vec<Bone> {
        vec![
            bone("World", None, [0.0; 3]),
            bone("Hips", Some(0), [0.0, 0.0, 1.0]),
            bone("Spine", Some(1), [0.0, 0.0, 1.1]),
            bone("PlaneWeightLeft", Some(0), [0.0; 3]),
        ]
    }

    #[test]
    fn constant_two_frame_pose_collapses_to_one_key() {
        let q = Quat::from_rotation_z(0.3).to_array();
        let clip = clip(vec![
            channel("Spine", vec![(0.0, q), (2.0, q)], vec![]),
            channel("Unknown", vec![(0.0, q)], vec![]),
        ]);
        let sampled = sample_clip(&clip, &skeleton(), "pose", "src", SampleOptions { strip_root_translation: true }).unwrap();
        assert!(sampled.is_static);
        assert!(!sampled.looping);
        assert_eq!(sampled.frames, 1);
        assert_eq!(sampled.matched_channels, 1);
        assert_eq!(sampled.unmatched_channels, 1);
        assert_eq!(sampled.rotations.len(), 1);
        let (bone, samples) = &sampled.rotations[0];
        assert_eq!((*bone, samples.len()), (2, 1));
        assert!(samples[0].abs_diff_eq(Quat::from_array(q), 1e-6));
    }

    #[test]
    fn samples_every_frame_with_slerp_and_continuous_signs() {
        let a = Quat::from_rotation_z(0.0);
        let b = Quat::from_rotation_z(1.0);
        // Second key stored on the opposite hemisphere.
        let clip = clip(vec![channel("Hips", vec![(0.0, a.to_array()), (4.0, (-b).to_array())], vec![])]);
        let sampled = sample_clip(&clip, &skeleton(), "turn", "src", SampleOptions { strip_root_translation: true }).unwrap();
        assert_eq!(sampled.frames, 5);
        assert!((sampled.duration() - 4.0 / 30.0).abs() < 1e-6);
        let samples = &sampled.rotations[0].1;
        for (k, q) in samples.iter().enumerate() {
            let expected = Quat::from_rotation_z(k as f32 * 0.25);
            assert!(q.dot(expected) > 0.99999, "frame {k}: {q:?}");
            assert!((q.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn root_horizontal_motion_is_stripped_and_loop_detected() {
        let clip = clip(vec![
            channel("World", vec![], vec![(0.0, [0.0, 0.0, 0.0]), (2.0, [0.0, 3.0, 0.02])]),
            // Hips bob goes up and back down: a loop.
            channel("Hips", vec![], vec![(0.0, [0.0, 0.1, 0.9]), (1.0, [0.0, 0.1, 0.95]), (2.0, [0.0, 0.1, 0.9])]),
        ]);
        let skeleton = skeleton();
        let stripped = sample_clip(&clip, &skeleton, "run", "src", SampleOptions { strip_root_translation: true }).unwrap();
        let world = &stripped.translations[0];
        assert_eq!(world.0, 0);
        assert!(world.1.iter().all(|p| p.x == 0.0 && p.y == 0.0));
        assert!((world.1[2].z - 0.02).abs() < 1e-6, "vertical root motion is kept");
        assert!((stripped.root_motion_removed - 3.0).abs() < 1e-5);
        assert!(stripped.translations[1].1.iter().all(|p| (p.y - 0.1).abs() < 1e-6), "Hips untouched");
        // 2 cm vertical drift on the root is beyond the loop tolerance.
        assert!(!stripped.looping);

        let kept = sample_clip(&clip, &skeleton, "run", "src", SampleOptions { strip_root_translation: false }).unwrap();
        assert!((kept.translations[0].1[1].y - 1.5).abs() < 1e-6);
        assert_eq!(kept.root_motion_removed, 0.0);
    }

    #[test]
    fn matching_first_and_last_pose_is_a_loop() {
        let q = Quat::from_rotation_x(0.4);
        let clip = clip(vec![
            channel(
                "Spine",
                vec![(0.0, Quat::IDENTITY.to_array()), (3.0, q.to_array()), (6.0, Quat::IDENTITY.to_array())],
                vec![],
            ),
            // A helper leaf under the root that does not close the loop.
            channel("PlaneWeightLeft", vec![], vec![(0.0, [0.0; 3]), (6.0, [0.4, 0.0, 0.0])]),
        ]);
        let sampled = sample_clip(&clip, &skeleton(), "idle", "src", SampleOptions { strip_root_translation: true }).unwrap();
        assert!(sampled.looping);
        assert!(!sampled.is_static);
        assert_eq!(sampled.frames, 7);
    }

    #[test]
    fn clip_without_skeleton_bones_is_rejected() {
        let clip = clip(vec![channel("Tail", vec![(0.0, Quat::IDENTITY.to_array())], vec![])]);
        assert!(sample_clip(&clip, &skeleton(), "x", "src", SampleOptions { strip_root_translation: true }).is_none());
    }
}
