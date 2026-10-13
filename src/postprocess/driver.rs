//! The correction driver — port of upstream
//! `MotionCorrection/src/cpp/AnimProcessing/Utility.cpp` (contact interval
//! bookkeeping, hip/root correction, per-bone rotation correction,
//! effector + contact IK, and the `CorrectMotion` top level).

use std::collections::BTreeMap;

use super::ik::{Pose, one_bone_ik, two_bone_ik};
use super::math::{Quat, Transform, Vec3};
use super::trajectory::TrajectoryCorrector;

/// Upstream weights (`Utility.cpp` file constants).
const POS_WEIGHT: f32 = 0.001;
const VEL_WEIGHT: f32 = 1.0;
const ACC_WEIGHT: f32 = 10.0;

/// `Animation::IKType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IkType {
    /// `kOneBone` — rotate the parent joint toward the target.
    OneBone,
    /// `kTwoBone` — bend the parent chain (two joints) toward the target.
    TwoBone,
}

/// `Animation::ContactInfo`.
#[derive(Debug, Clone)]
pub struct ContactInfo {
    /// IK contact joint index.
    pub joint_index: i32,
    /// Per-frame contact strength (0..1) for this joint.
    pub contact_mask: Vec<f32>,
    /// Solver type.
    pub contact_type: IkType,
    /// IK hint offset (world-space pole target seed).
    pub hint_offset: Vec3,
    /// Floor height for contact clamping.
    pub min_height: f32,
}

/// `Animation::JointLocalToGlobal` — compose from `index` up to the root,
/// then apply `root_tx`.
pub fn joint_local_to_global(
    parents: &[i32],
    index: i32,
    pose: &[Transform],
    root_tx: &Transform,
) -> Transform {
    let mut world = Transform::IDENTITY;
    let mut idx = index;
    while idx > -1 {
        world = world * pose[idx as usize];
        idx = parents[idx as usize];
    }
    world * *root_tx
}

/// `Animation::JointLocalToGlobal` with the default identity root transform.
pub fn joint_local_to_global_identity(
    parents: &[i32],
    index: i32,
    pose: &[Transform],
) -> Transform {
    joint_local_to_global(parents, index, pose, &Transform::IDENTITY)
}

/// `FilterContactIntervals` — drop intervals that cannot be corrected
/// without popping: bounded by constraints on *both* ends for two-bone
/// contacts, on *either* end for one-bone.
fn filter_contact_intervals(
    intervals: &mut Vec<(usize, usize)>,
    mask: &[f32],
    one_bone_contact: bool,
) {
    intervals.retain(|interval| {
        let start_constrained = interval.0 != 0 && mask[interval.0 - 1] != 0.0;
        let end_constrained = interval.1 != mask.len() && mask[interval.1] != 0.0;
        if one_bone_contact {
            !(start_constrained || end_constrained)
        } else {
            !(start_constrained && end_constrained)
        }
    });
}

/// `ComputeContactIntervals` — half-open `[start, end)` frame ranges above
/// the contact threshold, ignoring masked frames.
fn compute_contact_intervals(
    contacts: &[f32],
    mask: &[f32],
    threshold: f32,
) -> Vec<(usize, usize)> {
    // turn off the contacts for all frames that are constrained/masked
    let contacts_no_mask: Vec<f32> = contacts
        .iter()
        .zip(mask)
        .map(|(&c, &m)| if m != 0.0 { 0.0 } else { c })
        .collect();

    let mut intervals = Vec::new();
    let mut start: i64 = -1;
    for (frame, &c) in contacts_no_mask.iter().enumerate() {
        let is_contact = c > threshold;
        if is_contact && start == -1 {
            start = frame as i64;
        } else if !is_contact && start != -1 {
            intervals.push((start as usize, frame));
            start = -1;
        }
    }
    if start != -1 {
        intervals.push((start as usize, mask.len()));
    }
    intervals
}

/// `FindContactPoints` — a representative world-space point per contact
/// interval (the constrained neighbor frame when one exists, the midpoint
/// otherwise), with per-frame floor clamping in the free case.
// (argument count mirrors the upstream C++ signature)
#[allow(clippy::too_many_arguments)]
fn find_contact_points(
    points: &mut Vec<Vec3>,
    in_contact: &mut Vec<i32>,
    parents: &[i32],
    joint_index: i32,
    poses: &[Pose],
    intervals: &[(usize, usize)],
    mask: &[f32],
    min_height: f32,
) {
    let frame_count = mask.len();
    in_contact.clear();
    in_contact.resize(frame_count, 0);
    points.clear();
    points.resize(frame_count, Vec3::ZERO);

    for interval in intervals {
        let start_constrained = interval.0 != 0 && mask[interval.0 - 1] != 0.0;
        let end_constrained = interval.1 != mask.len() && mask[interval.1] != 0.0;

        let frame = if start_constrained {
            interval.0 - 1
        } else if end_constrained {
            interval.1
        } else {
            (interval.0 + interval.1) / 2
        };

        let target = joint_local_to_global_identity(parents, joint_index, &poses[frame]).t;
        for i in interval.0..interval.1 {
            let frame_pt = joint_local_to_global_identity(parents, joint_index, &poses[i]).t;
            in_contact[i] = 1;
            points[i] = target;
            if !start_constrained && !end_constrained {
                points[i].y = frame_pt.y.max(min_height);
            }
        }
    }
}

/// `TargetReachFalloff` — gaussian weight that fades the contact IK out as
/// the target leaves the limb's reach (based on the *default* pose lengths).
fn target_reach_falloff(
    parents: &[i32],
    default_pose: &[Transform],
    mut joint_index: usize,
    ik_type: IkType,
    target: Vec3,
    pose: &[Transform],
    root_tx: &Transform,
) -> f32 {
    let mut max_reach = default_pose[joint_index].t.length();
    if ik_type == IkType::TwoBone {
        joint_index = parents[joint_index] as usize;
        max_reach += default_pose[joint_index].t.length();
    }
    // base joint world transform
    joint_index = parents[joint_index] as usize;
    let world_tx = joint_local_to_global(parents, joint_index as i32, pose, root_tx);

    let target_dist = target.distance(world_tx.t);
    let mut tmp = (target_dist / max_reach - 0.99).max(0.0) / 0.01;
    tmp *= tmp;
    (-2.0 * tmp * tmp).exp()
}

/// `SmoothChannels` — replace constrained frames adjacent to unconstrained
/// ones with the neighbor average (in place on a row-major `N × cols`).
fn smooth_channels(x: &mut [f64], cols: usize, mask: &[f32]) {
    let n = mask.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
        let i_prev = if i == 0 { 0 } else { i - 1 };
        let i_next = (i + 1).min(n - 1);
        if i > 0 && mask[i] > 0.0 && mask[i_prev] == 0.0 {
            for j in 0..cols {
                x[i * cols + j] = 0.5 * (x[i_prev * cols + j] + x[i_next * cols + j]);
            }
        }
        if mask[i] > 0.0 && mask[i_next] == 0.0 {
            for j in 0..cols {
                x[i * cols + j] = 0.5 * (x[i_prev * cols + j] + x[i_next * cols + j]);
            }
        }
    }
}

/// `CorrectHipsY` — warp the root Y trajectory onto the target's root Y on
/// constrained frames (soft weights: `pos×10, vel, acc×0.1`).
fn correct_hips_y(poses: &mut [Pose], target_poses: &[Pose], full_body_mask: &[f32]) {
    let margins: Vec<f64> = full_body_mask
        .iter()
        .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
        .collect();
    let x: Vec<f64> = poses.iter().map(|p| p[0].t.y as f64).collect();
    let observations: Vec<f64> = target_poses.iter().map(|p| p[0].t.y as f64).collect();

    let corrector = TrajectoryCorrector::new(
        &margins,
        POS_WEIGHT * 10.0,
        VEL_WEIGHT,
        ACC_WEIGHT * 0.1,
        None,
        100,
    );
    let fixed = corrector.interpolate(&observations, &x, 1);
    for (frame, out) in fixed.into_iter().enumerate() {
        poses[frame][0].t.y = out as f32;
    }
}

/// `CorrectHipsXZ` — warp the root XZ trajectory: pinned to the target on
/// full-body frames, held within `root_margin` on root-masked frames and on
/// end-effector contact frames.
fn correct_hips_xz(
    poses: &mut [Pose],
    target_poses: &[Pose],
    full_body_mask: &[f32],
    root_mask: &[f32],
    end_effector_pins: &[ContactInfo],
    velocity_weights: &[f64],
    root_margin: f32,
) {
    let n = poses.len();
    let mut margins: Vec<f64> = full_body_mask
        .iter()
        .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
        .collect();

    let mut root_combined_mask = vec![0.0f32; n];
    for i in 0..n {
        root_combined_mask[i] = ((full_body_mask[i] > 0.0) || (root_mask[i] > 0.0)) as u8 as f32;
        if root_mask[i] > 0.0 && margins[i] != 0.0 {
            margins[i] = root_margin as f64;
        }
        for c in end_effector_pins {
            if c.contact_mask[i] != 0.0 && margins[i] != 0.0 {
                margins[i] = root_margin as f64;
            }
        }
    }

    let corrector = TrajectoryCorrector::new(
        &margins,
        POS_WEIGHT,
        VEL_WEIGHT,
        ACC_WEIGHT,
        Some(velocity_weights),
        100,
    );

    let mut x = vec![0.0f64; n * 2];
    let mut observations = vec![0.0f64; n * 2];
    for frame in 0..n {
        x[frame * 2] = poses[frame][0].t.x as f64;
        x[frame * 2 + 1] = poses[frame][0].t.z as f64;
        observations[frame * 2] = target_poses[frame][0].t.x as f64;
        observations[frame * 2 + 1] = target_poses[frame][0].t.z as f64;
    }

    smooth_channels(&mut x, 2, &root_combined_mask);
    let fixed = corrector.interpolate(&observations, &x, 2);
    for frame in 0..n {
        poses[frame][0].t.x = fixed[frame * 2] as f32;
        poses[frame][0].t.z = fixed[frame * 2 + 1] as f32;
    }
}

/// `CorrectRotationsForBone` — correct one bone's rotation channel: each
/// quaternion is expanded to its 6-D forward/up representation, all six
/// channels are run through the corrector, and the result is rebuilt with
/// `LookRotation`.
fn correct_rotations_for_bone(
    poses: &mut [Pose],
    target_poses: &[Pose],
    mask: &[f32],
    corrector: &TrajectoryCorrector,
    bone_idx: usize,
    perform_channel_smoothing: bool,
) {
    let n = poses.len();

    // quaternion components flip through 180° rotations, so work in the
    // forward/up vector representation instead
    let mut forward_up = vec![0.0f32; 6 * n];
    let mut target_forward_up = vec![0.0f32; 6 * n];
    for frame in 0..n {
        let q = poses[frame][bone_idx].rot;
        let forward = q.z_axis();
        let up = q.y_axis();
        forward_up[frame] = forward.x;
        forward_up[n + frame] = forward.y;
        forward_up[2 * n + frame] = forward.z;
        forward_up[3 * n + frame] = up.x;
        forward_up[4 * n + frame] = up.y;
        forward_up[5 * n + frame] = up.z;

        let q = target_poses[frame][bone_idx].rot;
        let forward = q.z_axis();
        let up = q.y_axis();
        target_forward_up[frame] = forward.x;
        target_forward_up[n + frame] = forward.y;
        target_forward_up[2 * n + frame] = forward.z;
        target_forward_up[3 * n + frame] = up.x;
        target_forward_up[4 * n + frame] = up.y;
        target_forward_up[5 * n + frame] = up.z;
    }

    let mut x = vec![0.0f64; n];
    let mut observations = vec![0.0f64; n];
    for dim in 0..6 {
        for frame in 0..n {
            x[frame] = forward_up[dim * n + frame] as f64;
            observations[frame] = (mask[frame] * target_forward_up[dim * n + frame]) as f64;
        }
        if perform_channel_smoothing {
            smooth_channels(&mut x, 1, mask);
        }
        let fixed = corrector.interpolate(&observations, &x, 1);
        for frame in 0..n {
            forward_up[dim * n + frame] = fixed[frame] as f32;
        }
    }

    for frame in 0..n {
        let forward = Vec3::new(
            forward_up[frame],
            forward_up[n + frame],
            forward_up[2 * n + frame],
        )
        .normalized();
        let up = Vec3::new(
            forward_up[3 * n + frame],
            forward_up[4 * n + frame],
            forward_up[5 * n + frame],
        )
        .normalized();
        poses[frame][bone_idx].rot = Quat::look_rotation(forward, up);
    }
}

/// `CorrectJointRotations` — warp every bone's rotations onto the target on
/// full-body-constrained frames.
fn correct_joint_rotations(
    poses: &mut [Pose],
    target_poses: &[Pose],
    full_body_mask: &[f32],
    velocity_weights: &[f64],
) {
    let margins: Vec<f64> = full_body_mask
        .iter()
        .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
        .collect();
    let corrector = TrajectoryCorrector::new(
        &margins,
        POS_WEIGHT * 10.0,
        VEL_WEIGHT,
        ACC_WEIGHT,
        Some(velocity_weights),
        100,
    );
    let bone_count = poses[0].len();
    for bone_idx in 0..bone_count {
        correct_rotations_for_bone(
            poses,
            target_poses,
            full_body_mask,
            &corrector,
            bone_idx,
            true,
        );
    }
}

/// `DoEffectorIK` — apply two/one-bone IK to every end-effector-pinned
/// frame (restoring each joint's global rotation afterwards), then smooth
/// every touched bone back onto the IK result.
fn do_effector_ik(
    poses: &mut [Pose],
    target_poses: &[Pose],
    full_body_mask: &[f32],
    end_effector_pins: &[ContactInfo],
    parents: &[i32],
) {
    // C++ std::map iterates in key order; BTreeMap reproduces it.
    let mut joint_correction_masks: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut ik_fixed_poses = poses.to_vec();

    for c in end_effector_pins {
        let joint = c.joint_index;
        if joint < 0 || joint as usize >= parents.len() {
            continue;
        }
        let joint = joint as usize;
        let parent = parents[joint];
        if parent < 0 || parent as usize >= parents.len() {
            continue;
        }
        let parent = parent as usize;

        joint_correction_masks
            .entry(joint)
            .or_insert_with(|| full_body_mask.to_vec());
        joint_correction_masks
            .entry(parent)
            .or_insert_with(|| full_body_mask.to_vec());
        let parent_parent = parents[parent];
        let two_bone = c.contact_type == IkType::TwoBone;
        if two_bone {
            joint_correction_masks
                .entry(parent_parent as usize)
                .or_insert_with(|| full_body_mask.to_vec());
        }

        for fix_frame in 0..full_body_mask.len() {
            if c.contact_mask[fix_frame] == 0.0 {
                continue;
            }
            let target_global =
                joint_local_to_global_identity(parents, c.joint_index, &target_poses[fix_frame]);

            // flag the parent joint as fixed in its correction mask
            joint_correction_masks.get_mut(&parent).unwrap()[fix_frame] = 1.0;
            match c.contact_type {
                IkType::OneBone => {
                    one_bone_ik(
                        &mut ik_fixed_poses[fix_frame],
                        &Transform::IDENTITY,
                        joint as u32,
                        1.0,
                        target_global.t,
                        parents,
                    );
                }
                IkType::TwoBone => {
                    joint_correction_masks
                        .get_mut(&(parent_parent as usize))
                        .unwrap()[fix_frame] = 1.0;
                    two_bone_ik(
                        &mut ik_fixed_poses[fix_frame],
                        &Transform::IDENTITY,
                        joint as u32,
                        1.0,
                        target_global.t,
                        parents,
                        c.hint_offset,
                    );
                }
            }

            // fix the global rotation of the joint to match the input
            joint_correction_masks.get_mut(&joint).unwrap()[fix_frame] = 1.0;
            let parent_global =
                joint_local_to_global_identity(parents, parent as i32, &ik_fixed_poses[fix_frame]);
            ik_fixed_poses[fix_frame][joint].rot = target_global.rot * parent_global.rot.conj();
        }
    }

    // IK introduces popping; re-line the touched trajectories
    for (&bone, msk) in &joint_correction_masks {
        let margins: Vec<f64> = msk
            .iter()
            .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
            .collect();
        let corrector = TrajectoryCorrector::new(
            &margins,
            POS_WEIGHT * 10.0,
            VEL_WEIGHT,
            ACC_WEIGHT,
            None,
            100,
        );
        correct_rotations_for_bone(poses, &ik_fixed_poses, msk, &corrector, bone, false);
    }
}

/// `DoContactIK` — stabilize limbs on ground contacts: two-bone IK with a
/// reach falloff for feet, floor-projection one-bone IK for toes, and (on
/// rigs with double ankles) the second-ankle correction phases.
// (argument count mirrors the upstream C++ signature)
#[allow(clippy::too_many_arguments)]
fn do_contact_ik(
    poses: &mut [Pose],
    full_body_mask: &[f32],
    contacts: &[ContactInfo],
    end_effector_pins: &[ContactInfo],
    parents: &[i32],
    default_pose: &[Transform],
    contact_threshold: f32,
    has_double_ankle_joints: bool,
) {
    let mut joint_correction_masks: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut ik_fixed_poses = poses.to_vec();
    let original_poses = poses.to_vec();
    let mut two_bone_contact_frames: BTreeMap<usize, Vec<bool>> = BTreeMap::new();

    // fills `joint_mask` from the pin covering this joint (or its parent —
    // we may be correcting a toe joint)
    let add_end_effector_mask = |joint_idx: usize, parent_idx: i32, joint_mask: &mut Vec<f32>| {
        let pin = end_effector_pins
            .iter()
            .find(|c| joint_idx as i32 == c.joint_index)
            .or_else(|| {
                end_effector_pins
                    .iter()
                    .find(|c| parent_idx == c.joint_index)
            });
        if let Some(msk) = pin {
            for (i, &m) in msk.contact_mask.iter().enumerate() {
                if m != 0.0 {
                    joint_mask[i] = 1.0;
                }
            }
        }
    };

    // ---- two-bone contacts first ----
    for c in contacts {
        if c.contact_type != IkType::TwoBone {
            continue;
        }
        let joint = c.joint_index as usize;
        let parent = parents[joint];
        if parent < 0 {
            continue;
        }
        let parent = parent as usize;
        let parent_parent = parents[parent] as usize;

        let mut joint_mask = full_body_mask.to_vec();
        add_end_effector_mask(joint, parents[joint], &mut joint_mask);

        joint_correction_masks
            .entry(parent)
            .or_insert_with(|| joint_mask.clone());
        joint_correction_masks
            .entry(parent_parent)
            .or_insert_with(|| joint_mask.clone());
        joint_correction_masks
            .entry(joint)
            .or_insert_with(|| joint_mask.clone());

        let mut contact_intervals =
            compute_contact_intervals(&c.contact_mask, &joint_mask, contact_threshold);
        filter_contact_intervals(&mut contact_intervals, &joint_mask, false);

        let mut contact_points: Vec<Vec3> = Vec::new();
        let mut in_contact: Vec<i32> = Vec::new();
        find_contact_points(
            &mut contact_points,
            &mut in_contact,
            parents,
            c.joint_index,
            poses,
            &contact_intervals,
            &joint_mask,
            c.min_height,
        );

        for fix_frame in 0..full_body_mask.len() {
            if in_contact[fix_frame] == 0 {
                continue;
            }
            let target = contact_points[fix_frame];
            joint_correction_masks.get_mut(&parent).unwrap()[fix_frame] = 1.0;
            joint_correction_masks.get_mut(&parent_parent).unwrap()[fix_frame] = 1.0;
            joint_correction_masks.get_mut(&joint).unwrap()[fix_frame] = 1.0;

            if has_double_ankle_joints {
                let entry = two_bone_contact_frames
                    .entry(joint)
                    .or_insert_with(|| vec![false; full_body_mask.len()]);
                entry[fix_frame] = true;
            }

            // save the original global rotation of the joint
            let joint_global_rotation =
                joint_local_to_global_identity(parents, c.joint_index, &ik_fixed_poses[fix_frame])
                    .rot;

            let w = target_reach_falloff(
                parents,
                default_pose,
                joint,
                c.contact_type,
                target,
                &ik_fixed_poses[fix_frame],
                &Transform::IDENTITY,
            );

            let orig_parent_rotation = ik_fixed_poses[fix_frame][parent].rot;
            let orig_parent_parent_rotation = ik_fixed_poses[fix_frame][parent_parent].rot;
            two_bone_ik(
                &mut ik_fixed_poses[fix_frame],
                &Transform::IDENTITY,
                joint as u32,
                1.0,
                target,
                parents,
                c.hint_offset,
            );
            ik_fixed_poses[fix_frame][parent].rot = Quat::slerp(
                orig_parent_rotation,
                ik_fixed_poses[fix_frame][parent].rot,
                w,
            );
            ik_fixed_poses[fix_frame][parent_parent].rot = Quat::slerp(
                orig_parent_parent_rotation,
                ik_fixed_poses[fix_frame][parent_parent].rot,
                w,
            );

            // restore the previous global rotation of this joint
            let parent_global_rotation =
                joint_local_to_global_identity(parents, parent as i32, &ik_fixed_poses[fix_frame])
                    .rot;
            ik_fixed_poses[fix_frame][joint].rot =
                joint_global_rotation * parent_global_rotation.conj();
        }
    }

    for (&bone, msk) in &joint_correction_masks {
        let margins: Vec<f64> = msk
            .iter()
            .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
            .collect();
        let corrector = TrajectoryCorrector::new(
            &margins,
            POS_WEIGHT * 10.0,
            VEL_WEIGHT,
            ACC_WEIGHT,
            None,
            100,
        );
        correct_rotations_for_bone(poses, &ik_fixed_poses, msk, &corrector, bone, false);
    }
    joint_correction_masks.clear();

    // ---- then one-bone contacts ----
    for c in contacts {
        if c.contact_type != IkType::OneBone {
            continue;
        }
        let joint = c.joint_index as usize;
        let parent = parents[joint];
        if parent < 0 {
            continue;
        }
        let parent = parent as usize;

        let mut joint_mask = full_body_mask.to_vec();
        add_end_effector_mask(joint, parents[joint], &mut joint_mask);

        joint_correction_masks
            .entry(parent)
            .or_insert_with(|| joint_mask.clone());

        let mut contact_intervals =
            compute_contact_intervals(&c.contact_mask, &joint_mask, contact_threshold);
        filter_contact_intervals(&mut contact_intervals, &joint_mask, true);
        for interval in &contact_intervals {
            // (index loop mirrors upstream; the mask write below needs the
            // absolute frame number)
            #[allow(clippy::needless_range_loop)]
            for fix_frame in interval.0..interval.1 {
                // stick the joint to the floor, letting it slide side to
                // side: iteratively project onto the floor sphere
                let parent_pos = joint_local_to_global_identity(
                    parents,
                    parent as i32,
                    &ik_fixed_poses[fix_frame],
                )
                .t;
                let mut target = joint_local_to_global_identity(
                    parents,
                    c.joint_index,
                    &ik_fixed_poses[fix_frame],
                )
                .t;
                let joint_length = (target - parent_pos).length();
                for _ in 0..10 {
                    target.y = c.min_height;
                    let dir = (target - parent_pos).normalized();
                    target = parent_pos + dir * joint_length;
                }

                one_bone_ik(
                    &mut ik_fixed_poses[fix_frame],
                    &Transform::IDENTITY,
                    joint as u32,
                    1.0,
                    target,
                    parents,
                );
                joint_correction_masks.get_mut(&parent).unwrap()[fix_frame] = 1.0;
            }
        }
    }

    for (&bone, msk) in &joint_correction_masks {
        let margins: Vec<f64> = msk
            .iter()
            .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
            .collect();
        let corrector = TrajectoryCorrector::new(
            &margins,
            POS_WEIGHT * 10.0,
            VEL_WEIGHT,
            ACC_WEIGHT,
            None,
            100,
        );
        correct_rotations_for_bone(poses, &ik_fixed_poses, msk, &corrector, bone, false);
    }

    if has_double_ankle_joints {
        let mut saved_first_ankle_targets: BTreeMap<usize, BTreeMap<usize, Vec3>> = BTreeMap::new();
        let mut saved_toe_targets: BTreeMap<usize, BTreeMap<usize, Vec3>> = BTreeMap::new();
        let mut contact_to_toe_idx: BTreeMap<usize, i32> = BTreeMap::new();

        // toe joints for each leg (the parent of a toe is the first ankle)
        for tc in contacts {
            if tc.contact_type == IkType::OneBone {
                let parent_idx = parents[tc.joint_index as usize];
                if parent_idx >= 0 {
                    contact_to_toe_idx.insert(parent_idx as usize, tc.joint_index);
                }
            }
        }

        // phase 1: correct the second ankle via hip->knee->second-ankle IK
        for c in contacts {
            if c.contact_type != IkType::TwoBone {
                continue;
            }
            let first_ankle = c.joint_index as usize;
            let second_ankle = parents[first_ankle] as usize;
            let knee = parents[second_ankle] as usize;
            let hip = parents[knee];
            if hip < 0 {
                continue;
            }
            let hip = hip as usize;

            let Some(contact_frames) = two_bone_contact_frames.get(&first_ankle) else {
                continue;
            };

            let mut joint_mask = full_body_mask.to_vec();
            add_end_effector_mask(first_ankle, parents[first_ankle], &mut joint_mask);

            joint_correction_masks
                .entry(knee)
                .or_insert_with(|| joint_mask.clone());
            joint_correction_masks
                .entry(hip)
                .or_insert_with(|| joint_mask.clone());

            for fix_frame in 0..full_body_mask.len() {
                if !contact_frames[fix_frame] {
                    continue;
                }

                saved_first_ankle_targets
                    .entry(first_ankle)
                    .or_default()
                    .insert(
                        fix_frame,
                        joint_local_to_global_identity(
                            parents,
                            first_ankle as i32,
                            &ik_fixed_poses[fix_frame],
                        )
                        .t,
                    );
                if let Some(&toe) = contact_to_toe_idx.get(&first_ankle) {
                    saved_toe_targets.entry(first_ankle).or_default().insert(
                        fix_frame,
                        joint_local_to_global_identity(parents, toe, &ik_fixed_poses[fix_frame]).t,
                    );
                }

                let original_first_ankle_global = joint_local_to_global_identity(
                    parents,
                    first_ankle as i32,
                    &original_poses[fix_frame],
                );
                let original_second_ankle_global = joint_local_to_global_identity(
                    parents,
                    second_ankle as i32,
                    &original_poses[fix_frame],
                );

                let delta_first_to_second =
                    original_first_ankle_global.delta_to_other(&original_second_ankle_global);
                let corrected_first_ankle_global = joint_local_to_global_identity(
                    parents,
                    first_ankle as i32,
                    &ik_fixed_poses[fix_frame],
                );
                let target = (delta_first_to_second * corrected_first_ankle_global).t;

                two_bone_ik(
                    &mut ik_fixed_poses[fix_frame],
                    &Transform::IDENTITY,
                    second_ankle as u32,
                    1.0,
                    target,
                    parents,
                    c.hint_offset,
                );

                joint_correction_masks.get_mut(&knee).unwrap()[fix_frame] = 1.0;
                joint_correction_masks.get_mut(&hip).unwrap()[fix_frame] = 1.0;
            }
        }

        for (&bone, msk) in &joint_correction_masks {
            let margins: Vec<f64> = msk
                .iter()
                .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
                .collect();
            let corrector = TrajectoryCorrector::new(
                &margins,
                POS_WEIGHT * 10.0,
                VEL_WEIGHT,
                ACC_WEIGHT,
                None,
                100,
            );
            correct_rotations_for_bone(poses, &ik_fixed_poses, msk, &corrector, bone, false);
        }

        // phase 2: one-bone IK to restore first ankle and toe
        joint_correction_masks.clear();
        for c in contacts {
            if c.contact_type != IkType::TwoBone {
                continue;
            }
            let first_ankle = c.joint_index as usize;
            let second_ankle = parents[first_ankle] as usize;

            let Some(contact_frames) = two_bone_contact_frames.get(&first_ankle) else {
                continue;
            };

            let mut joint_mask = full_body_mask.to_vec();
            add_end_effector_mask(first_ankle, parents[first_ankle], &mut joint_mask);

            joint_correction_masks
                .entry(second_ankle)
                .or_insert_with(|| joint_mask.clone());
            joint_correction_masks
                .entry(first_ankle)
                .or_insert_with(|| joint_mask.clone());

            for fix_frame in 0..full_body_mask.len() {
                if !contact_frames[fix_frame] {
                    continue;
                }

                // rotate the second ankle so the first ankle reaches its
                // saved target
                let first_target = saved_first_ankle_targets
                    .get(&first_ankle)
                    .and_then(|m| m.get(&fix_frame))
                    .copied();
                if let Some(target) = first_target {
                    one_bone_ik(
                        &mut ik_fixed_poses[fix_frame],
                        &Transform::IDENTITY,
                        first_ankle as u32,
                        1.0,
                        target,
                        parents,
                    );
                    joint_correction_masks.get_mut(&second_ankle).unwrap()[fix_frame] = 1.0;
                }

                // rotate the first ankle so the toe reaches its saved target
                let toe_target = contact_to_toe_idx
                    .get(&first_ankle)
                    .and_then(|_| saved_toe_targets.get(&first_ankle))
                    .and_then(|m| m.get(&fix_frame))
                    .copied();
                if let (Some(toe), Some(target)) =
                    (contact_to_toe_idx.get(&first_ankle).copied(), toe_target)
                {
                    one_bone_ik(
                        &mut ik_fixed_poses[fix_frame],
                        &Transform::IDENTITY,
                        toe as u32,
                        1.0,
                        target,
                        parents,
                    );
                    joint_correction_masks.get_mut(&first_ankle).unwrap()[fix_frame] = 1.0;
                }
            }
        }

        for (&bone, msk) in &joint_correction_masks {
            let margins: Vec<f64> = msk
                .iter()
                .map(|&m| if m != 0.0 { 0.0 } else { -1.0 })
                .collect();
            let corrector = TrajectoryCorrector::new(
                &margins,
                POS_WEIGHT * 10.0,
                VEL_WEIGHT,
                ACC_WEIGHT,
                None,
                100,
            );
            correct_rotations_for_bone(poses, &ik_fixed_poses, msk, &corrector, bone, false);
        }
    }
}

/// `Animation::CorrectMotion` — the full correction pass over `poses` in
/// place, toward `target_poses`, honoring the constraint masks, contact
/// pins and effector pins.
#[allow(clippy::too_many_arguments)]
pub fn correct_motion(
    poses: &mut [Pose],
    target_poses: &[Pose],
    full_body_mask: &[f32],
    root_mask: &[f32],
    contacts: &[ContactInfo],
    end_effector_pins: &[ContactInfo],
    parents: &[i32],
    default_pose: &[Transform],
    contact_threshold: f32,
    root_margin: f32,
    has_double_ankle_joints: bool,
) {
    let n = poses.len();
    // preserve velocities more strongly on frames where the root velocity
    // is low (so a standing character doesn't drift)
    let mut velocity_weights = vec![0.0f32; n];
    for frame in 1..n {
        let xdiff = poses[frame][0].t.x - poses[frame - 1][0].t.x;
        let zdiff = poses[frame][0].t.z - poses[frame - 1][0].t.z;
        let mut v_mag = (xdiff * xdiff + zdiff * zdiff).sqrt() / 0.05;
        v_mag = v_mag.max(1.0 / 1000.0);
        velocity_weights[frame] = 1.0 / v_mag;
    }
    if n > 1 {
        velocity_weights[0] = velocity_weights[1];
    }
    let velocity_weights: Vec<f64> = velocity_weights.iter().map(|&w| w as f64).collect();

    // 1. root y
    correct_hips_y(poses, target_poses, full_body_mask);
    // 2. root xz
    correct_hips_xz(
        poses,
        target_poses,
        full_body_mask,
        root_mask,
        end_effector_pins,
        &velocity_weights,
        root_margin,
    );
    // 3. joint rotations
    correct_joint_rotations(poses, target_poses, full_body_mask, &velocity_weights);
    // 4. effector pins
    do_effector_ik(
        poses,
        target_poses,
        full_body_mask,
        end_effector_pins,
        parents,
    );
    // 5. ground contacts
    do_contact_ik(
        poses,
        full_body_mask,
        contacts,
        end_effector_pins,
        parents,
        default_pose,
        contact_threshold,
        has_double_ankle_joints,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight 3-joint chain.
    fn chain(n_frames: usize) -> (Vec<Pose>, Vec<i32>, Vec<Transform>) {
        let parents = vec![-1, 0, 1];
        let default_pose = vec![
            Transform::IDENTITY,
            Transform {
                rot: Quat::IDENTITY,
                t: Vec3::new(0.0, 1.0, 0.0),
            },
            Transform {
                rot: Quat::IDENTITY,
                t: Vec3::new(0.0, 1.0, 0.0),
            },
        ];
        let poses: Vec<Pose> = (0..n_frames)
            .map(|f| {
                vec![
                    Transform {
                        rot: Quat::IDENTITY,
                        t: Vec3::new(f as f32 * 0.05, 1.0, 0.0),
                    },
                    default_pose[1],
                    default_pose[2],
                ]
            })
            .collect();
        (poses, parents, default_pose)
    }

    /// With no constraints and no contacts the whole pass is the identity.
    #[test]
    fn no_constraints_is_identity() {
        let (mut poses, parents, _dp) = chain(8);
        let before = poses.clone();
        let empty: [ContactInfo; 0] = [];
        correct_motion(
            &mut poses, &before, &[0.0; 8], &[0.0; 8], &empty, &empty, &parents, &_dp, 0.5, 0.05,
            false,
        );
        for (a, b) in poses.iter().zip(&before) {
            for (ta, tb) in a.iter().zip(b) {
                assert!((ta.t.x - tb.t.x).abs() < 1e-6);
                assert!((ta.t.y - tb.t.y).abs() < 1e-6);
                assert!((ta.rot.w - tb.rot.w).abs() < 1e-5);
            }
        }
    }

    /// A pinned root y on all frames pulls every root y to the target.
    #[test]
    fn full_constraint_warp() {
        let (mut poses, parents, dp) = chain(8);
        let mut targets = poses.clone();
        for t in &mut targets {
            t[0].t.y = 0.5; // pin the root down
        }
        let ones = vec![1.0; 8];
        let zeros = vec![0.0; 8];
        let empty: [ContactInfo; 0] = [];
        correct_motion(
            &mut poses, &targets, &ones, &zeros, &empty, &empty, &parents, &dp, 0.5, 0.05, false,
        );
        for p in &poses {
            assert!((p[0].t.y - 0.5).abs() < 1e-4, "root y {}", p[0].t.y);
        }
    }

    /// A foot contact near the floor stabilizes the foot: with the contact
    /// pinned on the middle frames, the foot joint's world y stays close to
    /// a constant across the contact interval.
    #[test]
    fn contact_ik_stabilizes_foot() {
        let (mut poses, parents, dp) = chain(16);
        // target poses: identical to input (nothing constrained)
        let targets = poses.clone();
        let zeros = vec![0.0; 16];
        // foot contact on frames 4..12 (joint 2 is the "foot")
        let mut contacts = vec![ContactInfo {
            joint_index: 2,
            contact_mask: vec![0.0; 16],
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, 1.0),
            min_height: 0.0,
        }];
        for i in 4..12 {
            contacts[0].contact_mask[i] = 1.0;
        }
        correct_motion(
            &mut poses,
            &targets,
            &zeros,
            &zeros,
            &contacts,
            &[], // no effector pins
            &parents,
            &dp,
            0.5,
            0.05,
            false,
        );
        // during the contact the foot's world y should be roughly constant
        // (it was already constant in this straight-chain input; the pass
        // must not blow it up)
        let ys: Vec<f32> = (4..12)
            .map(|f| joint_local_to_global_identity(&parents, 2, &poses[f]).t.y)
            .collect();
        let spread = ys.iter().fold(0.0f32, |m, &y| m.max((y - ys[0]).abs()));
        assert!(spread < 0.1, "foot y spread during contact: {spread}");
    }
}
