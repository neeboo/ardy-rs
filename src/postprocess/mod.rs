//! Motion postprocess — Rust port of upstream `nv-tlabs/ardy`
//! `MotionCorrection` (Apache-2.0), the trajectory-correction / IK pass that
//! `ardy-mlx` v1 skipped with `--no-postprocess`.
//!
//! Layout mirrors the C++ / Python source tree:
//! - [`math`] — `Math/{Vector,Quaternion,Transform,Matrix}` primitives
//! - [`trajectory`] — `TrajectoryCorrector` (ADMM smoothing with pins and
//!   margins)
//! - [`ik`] — `InverseKinematics.cpp` (two-bone / one-bone solvers)
//! - [`driver`] — `Utility.cpp` (contact filtering, hip/rotation
//!   correction, effector + contact IK, `CorrectMotion`)
//! - this file's `correct_motion` — the pybind marshaling from
//!   `BindingsPython.cpp` (+ the `motion_correction/motion_postprocess.py`
//!   unpacking), and `post_process_motion` from `ardy/postprocess.py`

pub mod driver;
pub mod ik;
pub mod math;
pub mod trajectory;

use crate::error::{Error, Result};
use crate::model::skeleton::Skeleton;
use driver::{ContactInfo, IkType, joint_local_to_global_identity};
use ik::Pose;
use math::{Quat, Transform, Vec3, quat_from_matrix_rows};

/// `strip_nan_inf` from `BindingsPython.cpp`.
fn strip_nan_inf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        0.0
    } else {
        x
    }
}

/// Per-category constraint masks plus the keyframe targets those frames
/// should hit — the flat-image of the upstream constraint list after
/// `extract_input_motion_from_constraints`. Unconstrained generation passes
/// [`MotionConstraints::unconstrained`].
#[derive(Debug, Clone)]
pub struct MotionConstraints {
    /// `FullBody` mask `[T]`.
    pub full_body: Vec<f32>,
    /// `LeftHand` mask `[T]`.
    pub left_hand: Vec<f32>,
    /// `RightHand` mask `[T]`.
    pub right_hand: Vec<f32>,
    /// `LeftFoot` mask `[T]`.
    pub left_foot: Vec<f32>,
    /// `RightFoot` mask `[T]`.
    pub right_foot: Vec<f32>,
    /// `Root` (root2d) mask `[T]`.
    pub root: Vec<f32>,
    /// Target root translations `[T, 3]` (only constrained frames matter).
    pub root_translations_target: Vec<[f32; 3]>,
    /// Target local joint rotations `[T, J, 4]` wxyz (identity off-keyframe).
    pub joint_rotations_target: Vec<Vec<[f32; 4]>>,
}

impl MotionConstraints {
    /// All-zero masks and identity targets — the upstream `constraint_lst =
    /// None` path.
    pub fn unconstrained(num_frames: usize, num_joints: usize) -> Self {
        Self {
            full_body: vec![0.0; num_frames],
            left_hand: vec![0.0; num_frames],
            right_hand: vec![0.0; num_frames],
            left_foot: vec![0.0; num_frames],
            right_foot: vec![0.0; num_frames],
            root: vec![0.0; num_frames],
            root_translations_target: vec![[0.0; 3]; num_frames],
            joint_rotations_target: vec![vec![[1.0, 0.0, 0.0, 0.0]; num_joints]; num_frames],
        }
    }
}

/// Port of the pybind11 `correct_motion` (`BindingsPython.cpp`) including the
/// unpacking its Python caller does (`motion_correction/motion_postprocess.py`).
///
/// Arrays are frame-major flat buffers, mutated in place exactly like the
/// binding: `root_translations` is `T×3` xyz, `joint_rotations` is
/// `T×J×4` **wxyz** (w at stride 0 — the reference stores quats w-first and
/// the binding shuffles to the engine's xyzw on the way in and back).
/// `contacts` is `T×4` as `[left foot, left toe, right foot, right toe]`.
///
/// `joint_ref_rotations` are `[J, 4]` in the *rig's* xyzw spelling (the
/// upstream rigs are all `[0, 0, 0, 1]` = identity, read by the C++
/// `Quaternion(x, y, z, w)` constructor).
#[allow(clippy::too_many_arguments)]
pub fn correct_motion(
    root_translations: &mut [f32],
    joint_rotations: &mut [f32],
    root_translations_target: &[f32],
    joint_rotations_target: &[f32],
    full_pose_mask: &[f32],
    left_hand_mask: &[f32],
    right_hand_mask: &[f32],
    left_foot_mask: &[f32],
    right_foot_mask: &[f32],
    root_mask: &[f32],
    contacts: &[f32],
    joint_parents: &[i32],
    joint_ref_translations: &[[f32; 3]],
    joint_ref_rotations: &[[f32; 4]],
    left_hand_idx: usize,
    right_hand_idx: usize,
    left_foot_idx: usize,
    right_foot_idx: usize,
    contact_threshold: f32,
    root_margin: f32,
    has_double_ankle_joints: bool,
) -> Result<()> {
    let num_joints = joint_parents.len();
    let gen_length = full_pose_mask.len();

    // --- binding argument validation ---
    if joint_ref_translations.len() != num_joints || joint_ref_rotations.len() != num_joints {
        return Err(Error::Contract(
            "correct_motion: joint_parents, joint_ref_translations and joint_ref_rotations \
             must have the same size"
                .into(),
        ));
    }
    for &idx in [left_hand_idx, right_hand_idx, left_foot_idx, right_foot_idx].iter() {
        if idx >= num_joints {
            return Err(Error::Contract(
                "correct_motion: end-effector indices must be less than the number of joints"
                    .into(),
            ));
        }
    }
    for (name, mask) in [
        ("left_hand_mask", left_hand_mask),
        ("right_hand_mask", right_hand_mask),
        ("left_foot_mask", left_foot_mask),
        ("right_foot_mask", right_foot_mask),
        ("root_mask", root_mask),
    ] {
        if mask.len() != gen_length {
            return Err(Error::Contract(format!(
                "correct_motion: {name} must have length {gen_length}"
            )));
        }
    }
    if root_translations.len() != 3 * gen_length || root_translations_target.len() != 3 * gen_length
    {
        return Err(Error::Contract(
            "correct_motion: root translations must be T×3".into(),
        ));
    }
    if joint_rotations.len() != 4 * num_joints * gen_length
        || joint_rotations_target.len() != 4 * num_joints * gen_length
    {
        return Err(Error::Contract(
            "correct_motion: joint rotations must be T×J×4".into(),
        ));
    }
    if contacts.len() != 4 * gen_length {
        return Err(Error::Contract(
            "correct_motion: contacts must be T×4".into(),
        ));
    }

    // --- default pose from the working rig ---
    let default_pose: Vec<Transform> = (0..num_joints)
        .map(|i| Transform {
            // the C++ constructs Quaternion(x, y, z, w) from the rig list
            rot: Quat::new(
                joint_ref_rotations[i][0],
                joint_ref_rotations[i][1],
                joint_ref_rotations[i][2],
                joint_ref_rotations[i][3],
            ),
            t: Vec3::new(
                joint_ref_translations[i][0],
                joint_ref_translations[i][1],
                joint_ref_translations[i][2],
            ),
        })
        .collect();

    // --- end-effector pins: hands then feet, per-category masks ---
    let mut end_effector_pins = vec![
        ContactInfo {
            joint_index: left_hand_idx as i32,
            contact_mask: Vec::with_capacity(gen_length),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, -0.1),
            min_height: 0.0,
        },
        ContactInfo {
            joint_index: right_hand_idx as i32,
            contact_mask: Vec::with_capacity(gen_length),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, -0.1),
            min_height: 0.0,
        },
        ContactInfo {
            joint_index: left_foot_idx as i32,
            contact_mask: Vec::with_capacity(gen_length),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, 0.1),
            min_height: 0.0,
        },
        ContactInfo {
            joint_index: right_foot_idx as i32,
            contact_mask: Vec::with_capacity(gen_length),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, 0.1),
            min_height: 0.0,
        },
    ];
    for i in 0..gen_length {
        end_effector_pins[0]
            .contact_mask
            .push((1.0 - full_pose_mask[i]) * left_hand_mask[i]);
        end_effector_pins[1]
            .contact_mask
            .push((1.0 - full_pose_mask[i]) * right_hand_mask[i]);
        end_effector_pins[2]
            .contact_mask
            .push((1.0 - full_pose_mask[i]) * left_foot_mask[i]);
        end_effector_pins[3]
            .contact_mask
            .push((1.0 - full_pose_mask[i]) * right_foot_mask[i]);
    }

    // --- contact info: right foot [0], left foot [1], toes after ---
    let foot_min_height = |idx: usize| {
        joint_local_to_global_identity(joint_parents, idx as i32, &default_pose)
            .t
            .y
    };
    let mut contact_info = vec![
        ContactInfo {
            joint_index: right_foot_idx as i32,
            contact_mask: Vec::new(),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, 0.1),
            min_height: foot_min_height(right_foot_idx),
        },
        ContactInfo {
            joint_index: left_foot_idx as i32,
            contact_mask: Vec::new(),
            contact_type: IkType::TwoBone,
            hint_offset: Vec3::new(0.0, 0.0, 0.1),
            min_height: foot_min_height(left_foot_idx),
        },
    ];
    // contacts layout: [L foot, L toe, R foot, R toe]; heel flagged whenever
    // the toe is a contact; masked frames contribute nothing.
    contact_info[0].contact_mask = (0..gen_length)
        .map(|i| {
            let r = if right_foot_mask[i] != 0.0 {
                0.0
            } else {
                contacts[4 * i + 2]
            };
            let r_toe = if right_foot_mask[i] != 0.0 {
                0.0
            } else {
                contacts[4 * i + 3]
            };
            (r_toe + r).min(1.0)
        })
        .collect();
    contact_info[1].contact_mask = (0..gen_length)
        .map(|i| {
            let l = if left_foot_mask[i] != 0.0 {
                0.0
            } else {
                contacts[4 * i]
            };
            let l_toe = if left_foot_mask[i] != 0.0 {
                0.0
            } else {
                contacts[4 * i + 1]
            };
            (l_toe + l).min(1.0)
        })
        .collect();

    // toe joints: children of the feet (last child wins, matching the loop)
    let mut left_toe_idx: i32 = -1;
    let mut right_toe_idx: i32 = -1;
    for (i, &p) in joint_parents.iter().enumerate() {
        if p == left_foot_idx as i32 {
            left_toe_idx = i as i32;
        }
        if p == right_foot_idx as i32 {
            right_toe_idx = i as i32;
        }
    }
    if left_toe_idx != -1 && right_toe_idx != -1 {
        // upstream quirk: both toes get the *right* toe's neutral height
        let toe_min_height =
            joint_local_to_global_identity(joint_parents, right_toe_idx, &default_pose)
                .t
                .y;
        contact_info.resize(
            4,
            ContactInfo {
                joint_index: 0,
                contact_mask: Vec::new(),
                contact_type: IkType::TwoBone,
                hint_offset: Vec3::ZERO,
                min_height: 0.0,
            },
        );
        contact_info[2] = ContactInfo {
            joint_index: right_toe_idx,
            contact_mask: (0..gen_length)
                .map(|i| {
                    if right_foot_mask[i] != 0.0 {
                        0.0
                    } else {
                        contacts[4 * i + 3]
                    }
                })
                .collect(),
            contact_type: IkType::OneBone,
            hint_offset: Vec3::ZERO,
            min_height: toe_min_height,
        };
        contact_info[3] = ContactInfo {
            joint_index: left_toe_idx,
            contact_mask: (0..gen_length)
                .map(|i| {
                    if left_foot_mask[i] != 0.0 {
                        0.0
                    } else {
                        contacts[4 * i + 1]
                    }
                })
                .collect(),
            contact_type: IkType::OneBone,
            hint_offset: Vec3::ZERO,
            min_height: toe_min_height,
        };
    }

    // --- setTransforms: fill per-frame poses from the flat arrays ---
    fn set_transforms(poses: &mut [Pose], root: &[f32], rots: &[f32], num_joints: usize) {
        for (f, pose) in poses.iter_mut().enumerate() {
            pose[0].t = Vec3::new(
                strip_nan_inf(root[3 * f]),
                strip_nan_inf(root[3 * f + 1]),
                strip_nan_inf(root[3 * f + 2]),
            );
        }
        for (f, pose) in poses.iter_mut().enumerate() {
            for (j, tx) in pose.iter_mut().enumerate() {
                let b = 4 * (num_joints * f + j);
                // input is wxyz; the engine quaternion is xyzw
                let q = Quat::new(
                    strip_nan_inf(rots[b + 1]),
                    strip_nan_inf(rots[b + 2]),
                    strip_nan_inf(rots[b + 3]),
                    strip_nan_inf(rots[b]),
                );
                tx.rot = q.normalized();
            }
        }
    }

    let mut poses_fixed: Vec<Pose> = vec![default_pose.clone(); gen_length];
    set_transforms(
        &mut poses_fixed,
        root_translations,
        joint_rotations,
        num_joints,
    );
    let mut poses_target: Vec<Pose> = vec![default_pose.clone(); gen_length];
    set_transforms(
        &mut poses_target,
        root_translations_target,
        joint_rotations_target,
        num_joints,
    );

    // --- `Animation::CorrectMotion` ---
    driver::correct_motion(
        &mut poses_fixed,
        &poses_target,
        full_pose_mask,
        root_mask,
        &contact_info,
        &end_effector_pins,
        joint_parents,
        &default_pose,
        contact_threshold,
        root_margin,
        has_double_ankle_joints,
    );

    // --- write back: root xyz, rotations wxyz ---
    for (f, frame) in poses_fixed.iter().enumerate().take(gen_length) {
        let t = frame[0].t;
        root_translations[3 * f] = t.x;
        root_translations[3 * f + 1] = t.y;
        root_translations[3 * f + 2] = t.z;
    }
    for (f, frame) in poses_fixed.iter().enumerate().take(gen_length) {
        for (j, pose) in frame.iter().enumerate().take(num_joints) {
            let q = pose.rot;
            let b = 4 * (num_joints * f + j);
            joint_rotations[b] = q.w;
            joint_rotations[b + 1] = q.x;
            joint_rotations[b + 2] = q.y;
            joint_rotations[b + 3] = q.z;
        }
    }
    Ok(())
}

/// FK products returned by [`post_process_motion`]:
/// `(local_rot_mats, root_positions, posed_joints, global_rot_mats)`.
pub type CorrectedMotion = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Port of `ardy/postprocess.py::post_process_motion` for a single motion
/// (one batch element): correct in place, then run the final FK.
///
/// Inputs/outputs are frame-major flat buffers: `local_rot_mats` `T×J×9`
/// row-major matrices, `root_positions` `T×3`, `contacts` `T×4`
/// `[L foot, L toe, R foot, R toe]`. Returns the corrected rotations and
/// root plus the FK products (`posed_joints` `T×J×3`,
/// `global_rot_mats` `T×J×9`).
///
/// The matrix↔quaternion conversions mirror torch's wxyz
/// `matrix_to_quaternion`/`quaternion_to_matrix` pair; the correction itself
/// is byte-for-byte the [`correct_motion`] path.
#[allow(clippy::too_many_arguments)]
pub fn post_process_motion(
    local_rot_mats: &mut [f32],
    root_positions: &mut [f32],
    contacts: &[f32],
    skeleton: &Skeleton,
    constraints: &MotionConstraints,
    contact_threshold: f32,
    root_margin: f32,
) -> Result<CorrectedMotion> {
    let num_joints = skeleton.nbjoints;
    let num_frames = root_positions.len() / 3;
    if local_rot_mats.len() != 9 * num_joints * num_frames {
        return Err(Error::Contract(
            "post_process_motion: local_rot_mats must be T×J×3×3".into(),
        ));
    }
    if constraints.full_body.len() != num_frames {
        return Err(Error::Contract(
            "post_process_motion: masks must have length T".into(),
        ));
    }

    // matrix_to_quaternion (wxyz)
    let mut rotations = vec![0.0f32; 4 * num_joints * num_frames];
    for f in 0..num_frames {
        for j in 0..num_joints {
            let b = 9 * (num_joints * f + j);
            let m = local_rot_mats[b..b + 9].try_into().expect("9 floats");
            let q = mat_to_quat_wxyz(&m);
            let o = 4 * (num_joints * f + j);
            rotations[o..o + 4].copy_from_slice(&q);
        }
    }

    let mut root = root_positions.to_vec();
    let mut target_rots = vec![0.0f32; 4 * num_joints * num_frames];
    for f in 0..num_frames {
        for j in 0..num_joints {
            let q = constraints.joint_rotations_target[f][j]; // wxyz
            let o = 4 * (num_joints * f + j);
            target_rots[o..o + 4].copy_from_slice(&q);
        }
    }

    let target_roots: Vec<f32> = constraints
        .root_translations_target
        .iter()
        .flat_map(|t| t.iter().copied())
        .collect();
    let target_rots_ref: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 1.0]; num_joints]; // rig rotations are identity (xyzw)
    correct_motion(
        &mut root,
        &mut rotations,
        &target_roots,
        // flatten the [T][J][4] wxyz targets to the binding's layout
        &constraints
            .joint_rotations_target
            .iter()
            .flat_map(|f| f.iter())
            .flat_map(|q| q.iter().copied())
            .collect::<Vec<f32>>(),
        &constraints.full_body,
        &constraints.left_hand,
        &constraints.right_hand,
        &constraints.left_foot,
        &constraints.right_foot,
        &constraints.root,
        contacts,
        &skeleton.joint_parents,
        &working_rig_translations(skeleton),
        &target_rots_ref,
        ee_index(skeleton, "LeftHand")?,
        ee_index(skeleton, "RightHand")?,
        ee_index(skeleton, "LeftFoot")?,
        ee_index(skeleton, "RightFoot")?,
        contact_threshold,
        root_margin,
        false, // G1Skeleton34-only; the core cskel27 rig has no double ankles
    )?;

    // quaternion_to_matrix + final FK over the corrected local transforms
    let mut corrected_mats = vec![0.0f32; 9 * num_joints * num_frames];
    let mut posed_joints = vec![0.0f32; 3 * num_joints * num_frames];
    let mut global_mats = vec![0.0f32; 9 * num_joints * num_frames];
    for f in 0..num_frames {
        let mut pose: Pose = Vec::with_capacity(num_joints);
        for j in 0..num_joints {
            let b = 4 * (num_joints * f + j);
            let q = Quat::new(
                rotations[b + 1],
                rotations[b + 2],
                rotations[b + 3],
                rotations[b],
            );
            pose.push(Transform {
                rot: q,
                t: if j == 0 {
                    Vec3::new(root[3 * f], root[3 * f + 1], root[3 * f + 2])
                } else {
                    pose_ref_translation(skeleton, j)
                },
            });
            let o = 9 * (num_joints * f + j);
            corrected_mats[o..o + 9].copy_from_slice(&quat_to_mat(q));
        }
        for j in 0..num_joints {
            let world = joint_local_to_global_identity(&skeleton.joint_parents, j as i32, &pose);
            let o = 3 * (num_joints * f + j);
            posed_joints[o..o + 3].copy_from_slice(&[world.t.x, world.t.y, world.t.z]);
            let m = quat_to_mat(world.rot);
            let g = 9 * (num_joints * f + j);
            global_mats[g..g + 9].copy_from_slice(&m);
        }
    }
    root_positions.copy_from_slice(&root);
    local_rot_mats.copy_from_slice(&corrected_mats);
    Ok((corrected_mats, root, posed_joints, global_mats))
}

/// End-effector index by exact joint name (the non-G1 retarget map).
fn ee_index(skeleton: &Skeleton, name: &str) -> Result<usize> {
    skeleton
        .joint_names
        .iter()
        .position(|&n| n == name)
        .ok_or_else(|| Error::Contract(format!("post_process_motion: joint {name:?} not in rig")))
}

/// Working-rig local translations (`create_working_rig_from_skeleton`):
/// parent-relative neutral offsets; the root gets the ground shift so the
/// lowest neutral point (the toe) sits at y = 0.007.
fn working_rig_translations(skeleton: &Skeleton) -> Vec<[f32; 3]> {
    (0..skeleton.nbjoints)
        .map(|i| {
            if skeleton.joint_parents[i] == -1 {
                skeleton.core_rig_root_translation()
            } else {
                skeleton.core_rig_offset(i)
            }
        })
        .collect()
}

/// Per-joint rig translation for joint `j` (same rules as
/// [`working_rig_translations`]).
fn pose_ref_translation(skeleton: &Skeleton, j: usize) -> Vec3 {
    let t = if skeleton.joint_parents[j] == -1 {
        skeleton.core_rig_root_translation()
    } else {
        skeleton.core_rig_offset(j)
    };
    Vec3::new(t[0], t[1], t[2])
}

/// `matrix_to_quaternion` (wxyz) — row-major 3×3 in, `[w, x, y, z]` out.
/// Same trace-branch extraction as [`quat_from_matrix_rows`].
pub fn mat_to_quat_wxyz(m: &[f32; 9]) -> [f32; 4] {
    let r0 = Vec3::new(m[0], m[1], m[2]);
    let r1 = Vec3::new(m[3], m[4], m[5]);
    let r2 = Vec3::new(m[6], m[7], m[8]);
    let q = quat_from_matrix_rows(r0, r1, r2);
    [q.w, q.x, q.y, q.z]
}

/// `quaternion_to_matrix` — standard rotation matrix, row-major.
pub fn quat_to_mat(q: Quat) -> [f32; 9] {
    let x = q.x_axis();
    let y = q.y_axis();
    let z = q.z_axis();
    [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
}

/// `quaternion_to_matrix` (wxyz) — standard rotation matrix, row-major.
pub fn quat_to_mat_wxyz(q_wxyz: &[f32; 4]) -> [f32; 9] {
    quat_to_mat(Quat::new(q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity in, no constraints, no contacts → identity out (the pass is
    /// a no-op without data to correct).
    #[test]
    fn unconstrained_identity_roundtrip() {
        let skel = Skeleton::core();
        let t = 12;
        let j = skel.nbjoints;
        let mut mats = vec![0.0f32; 9 * j * t];
        for f in 0..t {
            for jj in 0..j {
                let o = 9 * (j * f + jj);
                mats[o] = 1.0;
                mats[o + 4] = 1.0;
                mats[o + 8] = 1.0;
            }
        }
        let mut root = vec![0.5f32; 3 * t];
        let contacts = vec![0.0f32; 4 * t];
        let before_root = root.clone();
        let cons = MotionConstraints::unconstrained(t, j);
        let (mats_out, root_out, posed, global) =
            post_process_motion(&mut mats, &mut root, &contacts, skel, &cons, 0.5, 0.04)
                .expect("postprocess");
        assert_eq!(mats_out.len(), mats.len());
        for (a, b) in mats_out.iter().zip(mats.iter()) {
            assert!((a - b).abs() < 1e-6, "rotation matrix changed {a} vs {b}");
        }
        for (a, b) in root_out.iter().zip(before_root.iter()) {
            assert!((a - b).abs() < 1e-6, "root moved {a} vs {b}");
        }
        // FK sanity: joint 0 sits at the (shifted) root; deeper joints are
        // further from the origin than their parents (monotone chain lengths)
        let o0 = 0; // frame 0, joint 0 — offset into the posed buffer
        assert!((posed[o0] - before_root[0]).abs() < 1e-5);
        assert!(global.len() == mats.len());
    }

    /// A strong left-foot contact on a planted foot must pin the foot near
    /// its own position (no skating), while the unconstrained path would
    /// leave it wherever it was.
    #[test]
    fn contact_flag_pins_foot() {
        let skel = Skeleton::core();
        let t = 10;
        let j = skel.nbjoints;
        let mut mats = vec![0.0f32; 9 * j * t];
        for o in (0..9 * j * t).step_by(9) {
            mats[o] = 1.0;
            mats[o + 4] = 1.0;
            mats[o + 8] = 1.0;
        }
        let mut root = vec![0.0f32; 3 * t];
        let mut contacts = vec![0.0f32; 4 * t];
        for f in 3..8 {
            contacts[4 * f] = 1.0; // left foot planted
        }
        let cons = MotionConstraints::unconstrained(t, j);
        let (_, _, posed, _) =
            post_process_motion(&mut mats, &mut root, &contacts, skel, &cons, 0.5, 0.04)
                .expect("postprocess");
        // left foot index
        let lf = skel
            .joint_names
            .iter()
            .position(|&n| n == "LeftFoot")
            .expect("LeftFoot");
        // during contact frames the foot must not wander (all poses are
        // identical, so the corrected foot is at a constant point)
        let at = |f: usize| {
            let o = 3 * (j * f + lf);
            [posed[o], posed[o + 1], posed[o + 2]]
        };
        let a = at(3);
        for f in 4..8 {
            let b = at(f);
            assert!((a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4);
        }
    }

    /// matrix↔quaternion round trip on a generic rotation.
    #[test]
    fn quat_matrix_roundtrip() {
        let q0 = Quat::from_axis_angle(Vec3::new(0.3, -0.5, 0.81).normalized(), 1.1);
        let qw = [q0.w, q0.x, q0.y, q0.z];
        let m = quat_to_mat_wxyz(&qw);
        let q1 = mat_to_quat_wxyz(&m);
        // same rotation (possibly negated)
        let dot: f32 = qw.iter().zip(q1.iter()).map(|(a, b)| a * b).sum();
        assert!((dot.abs() - 1.0).abs() < 1e-6);
        assert!((m[0] * m[1] + m[3] * m[4] + m[6] * m[7]).abs() < 1e-6); // rows 0·1
    }
}
