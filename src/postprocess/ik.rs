//! Two-bone and one-bone IK — port of upstream
//! `MotionCorrection/src/cpp/AnimProcessing/InverseKinematics.cpp`.

use super::math::{Quat, Transform, Vec3};

/// A posed skeleton: one `Transform` per joint, in joint order.
pub type Pose = Vec<Transform>;

fn get_angle_with_two_side_vectors(vec_left: Vec3, vec_right: Vec3) -> f32 {
    let l_norm = vec_left.normalized();
    let r_norm = vec_right.normalized();
    let cosine = l_norm.dot(r_norm);
    let sine = l_norm.cross(r_norm).length();
    sine.atan2(cosine) // radians
}

fn get_angle_with_cosine_rule(l_side_left: f32, l_side_right: f32, l_side_across: f32) -> f32 {
    let val = (l_side_right * l_side_right + l_side_left * l_side_left
        - l_side_across * l_side_across)
        / (2.0 * l_side_left * l_side_right);
    let val = val.clamp(-1.0, 1.0); // numerical stability; also impossible triangles
    val.acos() // radians
}

/// Parent walk shared by both solvers: the world transform *above* `a_idx`.
fn parent_world(pose: &[Transform], parents: &[i32], a_idx: usize, root: &Transform) -> Transform {
    let mut acc = Transform::IDENTITY;
    let mut idx = parents[a_idx];
    while idx >= 0 {
        acc = acc * pose[idx as usize];
        idx = parents[idx as usize];
    }
    acc * *root
}

/// Two-bone IK (`IK::TwoBoneIk`): bend joints `a` and `b` so `|ac| = |at|`
/// while `vec_ac` keeps its direction, then rotate `a` so `c` lands on
/// `target`. `c_idx` is the end joint; `b` and `a` are its parent and
/// grandparent.
pub fn two_bone_ik(
    pose: &mut Pose,
    root_transform: &Transform,
    c_idx: u32,
    weight: f32,
    target: Vec3,
    parents: &[i32],
    hint_offset: Vec3,
) {
    let weight = weight.clamp(0.0, 1.0);
    // NaN-faithful: `weight <= 0.0` would differ on NaN; mirrors C++ `!(weight > 0)`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(weight > 0.0) {
        return;
    }

    let c_idx = c_idx as usize;
    let b_idx = match usize::try_from(parents[c_idx]).ok() {
        Some(v) => v,
        None => return,
    };
    let a_idx = match usize::try_from(parents[b_idx]).ok() {
        Some(v) => v,
        None => return,
    };

    let a_parent_world = parent_world(pose, parents, a_idx, root_transform);
    let a_world = pose[a_idx] * a_parent_world;
    let b_world = pose[b_idx] * a_world;
    let c_world = pose[c_idx] * b_world;

    let a = a_world.t;
    let b = b_world.t;
    let c = c_world.t;
    let t = Vec3::lerp(c, target, weight);

    // stage 1: extend / contract the joint chain to match the distance
    let eps: f32 = 1e-4;
    let l_ab = (b - a).length();
    let l_bc = (c - b).length();
    let l_at = (a - t).length();
    let l_at = l_at.clamp(eps, (l_ab + l_bc) * 0.999); // unreachable: max reach

    let theta_bac_current = get_angle_with_two_side_vectors(a - b, a - c);
    let theta_abc_current = get_angle_with_two_side_vectors(b - a, b - c);
    if l_ab < eps || l_bc < eps || l_at < eps {
        return; // too short; rejecting potentially unstable requests
    }
    let theta_bac_desired = get_angle_with_cosine_rule(l_ab, l_at, l_bc);
    let theta_abc_desired = get_angle_with_cosine_rule(l_ab, l_bc, l_at);

    // in joint[0]'s parent's space
    let mut rotation_axis = (c - a).cross(b_world.transform_point(hint_offset) - a);
    let l = rotation_axis.length();
    if l == 0.0 {
        rotation_axis = Vec3::new(0.0, 0.0, 1.0);
    } else {
        rotation_axis = rotation_axis.scale(1.0 / l);
    }

    // rotation with the axis expressed in a's / b's local space
    let rotation_axis_local_in_b = b_world.rot.rotate_vector_inverse(rotation_axis);
    let rotate_in_b = Transform {
        rot: Quat::from_axis_angle(
            rotation_axis_local_in_b,
            theta_abc_desired - theta_abc_current,
        ),
        t: Vec3::ZERO,
    };
    pose[b_idx] = rotate_in_b * pose[b_idx];

    let rotation_axis_local_in_a = a_world.rot.rotate_vector_inverse(rotation_axis);
    let rotate_in_a = Transform {
        rot: Quat::from_axis_angle(
            rotation_axis_local_in_a,
            theta_bac_desired - theta_bac_current,
        ),
        t: Vec3::ZERO,
    };
    pose[a_idx] = rotate_in_a * pose[a_idx];

    let a_world = pose[a_idx] * a_parent_world;

    // stage 2: rotate joint a so the end joint c matches the target
    let ac_local = a_world.rot.rotate_vector_inverse(c - a_world.t);
    let at_local = a_world.rot.rotate_vector_inverse(target - a_world.t);
    let rotate_stage_two = Transform {
        rot: Quat::from_rotation_between_vectors(ac_local, at_local),
        t: Vec3::ZERO,
    };
    pose[a_idx] = rotate_stage_two * pose[a_idx];
}

/// One-bone IK (`IK::OneBoneIk`): rotate `a` so `b` leans toward `target`,
/// blended by `weight`.
pub fn one_bone_ik(
    pose: &mut Pose,
    root_transform: &Transform,
    b_idx: u32,
    weight: f32,
    target: Vec3,
    parents: &[i32],
) {
    let weight = weight.clamp(0.0, 1.0);
    // NaN-faithful: `weight <= 0.0` would differ on NaN; mirrors C++ `!(weight > 0)`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(weight > 0.0) {
        return;
    }

    let b_idx = b_idx as usize;
    let a_idx = match usize::try_from(parents[b_idx]).ok() {
        Some(v) => v,
        None => return,
    };

    let a_parent_world = parent_world(pose, parents, a_idx, root_transform);
    let a_world = pose[a_idx] * a_parent_world;
    let b_world = pose[b_idx] * a_world;

    let ab_local = a_world.rot.rotate_vector_inverse(b_world.t - a_world.t);
    let at_local = a_world.rot.rotate_vector_inverse(target - a_world.t);

    let delta_r_local = Quat::nlerp(
        Quat::IDENTITY,
        Quat::from_rotation_between_vectors(ab_local, at_local),
        weight,
    );
    pose[a_idx].rot = delta_r_local * pose[a_idx].rot;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planar two-bone chain: root at origin, b at (1,0,0), c at (2,0,0).
    /// Reaching t = (1,0,1) (exactly reachable: |at| = √2 = l_ab = l_bc)
    /// must put c on the target and keep a at the origin.
    #[test]
    fn two_bone_ik_reaches_exact_target() {
        let parents = vec![-1, 0, 1];
        let mut pose = vec![Transform::IDENTITY; 3];
        // b bends -90° about z so c = (2,0,0)? Instead keep the chain straight:
        // local translations a->b = (1,0,0), b->c = (1,0,0).
        pose[0].t = Vec3::ZERO;
        pose[1].t = Vec3::new(1.0, 0.0, 0.0);
        pose[2].t = Vec3::new(1.0, 0.0, 0.0);

        let target = Vec3::new(1.0, 0.0, 1.0);
        two_bone_ik(
            &mut pose,
            &Transform::IDENTITY,
            2,
            1.0,
            target,
            &parents,
            Vec3::new(0.0, 0.0, 1.0),
        );

        // recompute world positions
        let pw = parent_world(&pose, &parents, 0, &Transform::IDENTITY);
        let aw = pose[0] * pw;
        let bw = pose[1] * aw;
        let cw = pose[2] * bw;
        assert!((aw.t.x).abs() < 1e-5 && (aw.t.y).abs() < 1e-5 && (aw.t.z).abs() < 1e-5);
        let err = cw.t.distance(target);
        assert!(err < 1e-4, "end joint missed target by {err}");
    }

    /// An unreachable target clamps to max reach — direction preserved,
    /// length = l_ab + l_bc (times the 0.999 clamp).
    #[test]
    fn two_bone_ik_clamps_unreachable() {
        let parents = vec![-1, 0, 1];
        let mut pose = vec![
            Transform::IDENTITY,
            Transform {
                rot: Quat::IDENTITY,
                t: Vec3::new(1.0, 0.0, 0.0),
            },
            Transform {
                rot: Quat::IDENTITY,
                t: Vec3::new(1.0, 0.0, 0.0),
            },
        ];
        let target = Vec3::new(10.0, 0.0, 0.0);
        two_bone_ik(
            &mut pose,
            &Transform::IDENTITY,
            2,
            1.0,
            target,
            &parents,
            Vec3::new(0.0, 1.0, 0.0),
        );
        let pw = parent_world(&pose, &parents, 0, &Transform::IDENTITY);
        let aw = pose[0] * pw;
        let bw = pose[1] * aw;
        let cw = pose[2] * bw;
        let reach = cw.t.length();
        // max reach 2·0.999 = 1.998; straight chain should stay straight
        assert!((reach - 1.998).abs() < 1e-3, "reach {reach}");
    }

    /// One-bone IK with weight 1 puts b exactly on the target (when
    /// reachable by rotation about a).
    #[test]
    fn one_bone_ik_reaches() {
        let parents = vec![-1, 0];
        let mut pose = vec![
            Transform::IDENTITY,
            Transform {
                rot: Quat::IDENTITY,
                t: Vec3::new(0.5, 0.0, 0.0),
            },
        ];
        // target at the same radius, different direction
        let target = Vec3::new(0.0, 0.5, 0.0);
        one_bone_ik(&mut pose, &Transform::IDENTITY, 1, 1.0, target, &parents);
        let pw = parent_world(&pose, &parents, 0, &Transform::IDENTITY);
        let aw = pose[0] * pw;
        let bw = pose[1] * aw;
        let err = bw.t.distance(target);
        assert!(err < 1e-5, "b missed target by {err}");
    }

    /// Weight 0 (and negative) are no-ops.
    #[test]
    fn zero_weight_is_noop() {
        let parents = vec![-1, 0, 1];
        let mut pose = vec![Transform::IDENTITY; 3];
        pose[1].t = Vec3::new(1.0, 0.0, 0.0);
        pose[2].t = Vec3::new(1.0, 0.0, 0.0);
        let before = pose.clone();
        two_bone_ik(
            &mut pose,
            &Transform::IDENTITY,
            2,
            0.0,
            Vec3::new(3.0, 3.0, 3.0),
            &parents,
            Vec3::ZERO,
        );
        one_bone_ik(
            &mut pose,
            &Transform::IDENTITY,
            1,
            0.0,
            Vec3::new(3.0, 3.0, 3.0),
            &parents,
        );
        for (a, b) in pose.iter().zip(&before) {
            assert!((a.t.x - b.t.x).abs() < 1e-12);
            assert!((a.rot.x - b.rot.x).abs() < 1e-12);
        }
    }
}
