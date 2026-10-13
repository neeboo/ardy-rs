//! The Core (27-joint) skeleton, 6D rotation conversion, and forward
//! kinematics.
//!
//! Port of `ardy_mlx/model/skeleton.py`, itself a line-by-line port of the
//! upstream Apache-2.0 source (`nv-tlabs/ardy` @ `693f74d`):
//! `skeleton/kinematics.py::batch_rigid_transform` (the canonical SMPL-style
//! FK), `geometry.py::cont6d_to_matrix`, and
//! `skeleton/transforms.py::global_rots_to_local_rots`. The joint table is
//! transcribed from `CoreSkeleton27.bone_order_names_with_parents` and the
//! neutral rest pose from `ardy/assets/skeletons/cskel27/joints.p` — source
//! data, not checkpoint data.
//!
//! ardy-mlx measured this port against the golden runs: driven from the
//! golden `local_rot_mats` + `root_positions` it reproduces `posed_joints` to
//! `1.7e-07`, and driven from the feature vector's `global_rot_data` to
//! `4.3e-07` (float32 storage precision). `tests/decode.rs` re-validates the
//! Rust port the same way.
//!
//! Units / frame: world coordinates, right-handed, Y up, metres, ground =
//! XZ plane. The root joint is `Hips` and `root_positions` is its world
//! position, so `posed_joints[..., 0, :] == root_positions` exactly.

use std::sync::OnceLock;

use mlx_rs::Array;
use mlx_rs::ops::{
    self,
    indexing::{Ellipsis, TryIndexOp},
};

use crate::{Error, Result};

/// `CoreSkeleton27.bone_order_names_with_parents` (`skeleton/definitions.py`),
/// in order. Index 0 is the root.
pub const CORE_JOINT_NAMES: [&str; 27] = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
];

/// Parent index per joint (`-1` = root). Same source as [`CORE_JOINT_NAMES`].
pub const CORE_JOINT_PARENTS: [i32; 27] = [
    -1, // Hips
    0,  // Spine            <- Hips
    1,  // Spine1           <- Spine
    2,  // Spine2           <- Spine1
    3,  // Spine3           <- Spine2
    4,  // Neck             <- Spine3
    5,  // Head             <- Neck
    4,  // RightShoulder    <- Spine3
    7,  // RightArm         <- RightShoulder
    8,  // RightForeArm     <- RightArm
    9,  // RightHand        <- RightForeArm
    10, // RightHandEnd     <- RightHand
    10, // RightHandThumb1  <- RightHand
    4,  // LeftShoulder     <- Spine3
    13, // LeftArm          <- LeftShoulder
    14, // LeftForeArm      <- LeftArm
    15, // LeftHand         <- LeftForeArm
    16, // LeftHandEnd      <- LeftHand
    16, // LeftHandThumb1   <- LeftHand
    0,  // RightUpLeg       <- Hips
    19, // RightLeg         <- RightUpLeg
    20, // RightFoot        <- RightLeg
    21, // RightToeBase     <- RightFoot
    0,  // LeftUpLeg        <- Hips
    23, // LeftLeg          <- LeftUpLeg
    24, // LeftFoot         <- LeftLeg
    25, // LeftToeBase      <- LeftFoot
];

/// `neutral_joints` from `ardy/assets/skeletons/cskel27/joints.p` (27 x 3,
/// float64 in the asset, cast to float32 for compute like the reference
/// does). Full precision so the transcription is exact, not rounded.
pub const CORE_NEUTRAL_JOINTS: [[f64; 3]; 27] = [
    [0.0, 0.0, 0.0],                                                // Hips
    [0.0, 0.070_989_1, -0.047_326_099_999_999_93],                  // Spine
    [0.0, 0.164_203_318_195_104_6, -0.063_762_289_180_484_34],      // Spine1
    [0.0, 0.25849533092620636, -0.072_011_822_543_843_85],          // Spine2
    [0.0, 0.35314752658315185, -0.072_011_917_118_271_2],           // Spine3
    [0.0, 0.601_609_550_648_070_1, -0.036_517_567_004_764_68],      // Neck
    [0.0, 0.729_779_338_948_321_4, -0.013917896912177378],          // Head
    [-0.0319949, 0.525_919_571_936_606, -0.018_687_292_195_083_79], // RightShoulder
    [
        -0.190_902_9,
        0.525_919_516_889_253_6,
        -0.018_687_292_195_028_79,
    ], // RightArm
    [
        -0.486_338_9,
        0.525_919_414_547_209_3,
        -0.018_687_292_194_824_19,
    ], // RightForeArm
    [
        -0.7189909000000001,
        0.525919333954183,
        -0.018687292194582476,
    ], // RightHand
    [
        -0.788_602_400_000_000_1,
        0.5259193098400492,
        -0.01868729219448604,
    ], // RightHandEnd
    [
        -0.746_835_493_569_504_4,
        0.507_356_271_529_853_6,
        0.027_720_424_223_916_38,
    ], // RightHandThumb1
    [
        0.031_994_900_000_000_01,
        0.525_919_571_936_606,
        -0.018_687_292_195_083_79,
    ], // LeftShoulder
    [
        0.190_902_9,
        0.525_919_571_936_606,
        -0.018_687_292_195_083_79,
    ], // LeftArm
    [
        0.486_338_9,
        0.525_919_571_936_606,
        -0.018_687_292_195_083_79,
    ], // LeftForeArm
    [
        0.7189909000000001,
        0.525_919_571_936_606,
        -0.018_687_292_195_083_79,
    ], // LeftHand
    [
        0.788_602_4,
        0.525_919_571_936_606,
        -0.018_687_292_195_083_79,
    ], // LeftHandEnd
    [0.746_835_5, 0.507_356_519_157_93, 0.027_720_424_223_376_49],  // LeftHandThumb1
    [-0.094_918_2, -0.027_728_9, 0.0],                              // RightUpLeg
    [-0.094_918_2, -0.439_846_9, 0.0],                              // RightLeg
    [
        -0.094_918_200_000_000_09,
        -0.895_937_9,
        1.0127254590486245e-16,
    ], // RightFoot
    [
        -0.094_918_200_000_000_1,
        -0.954_412_825_233_483_3,
        0.16065826618182677,
    ], // RightToeBase
    [0.094_918_200_000_000_02, -0.027_728_9, 0.0],                  // LeftUpLeg
    [0.094_918_200_000_000_02, -0.43984690000000004, 0.0],          // LeftLeg
    [
        0.094_918_199_999_999_92,
        -0.895_937_900_000_000_1,
        1.0127254590486246e-16,
    ], // LeftFoot
    [
        0.094_918_199_999_999_91,
        -0.954_412_825_233_483_3,
        0.16065826618182655,
    ], // LeftToeBase
];

/// Root joint (`SkeletonBase.root_idx`; the only joint with parent `-1`).
pub const CORE_ROOT_INDEX: usize = 0;

/// `left_foot_joint_names + right_foot_joint_names` (`skeleton/base.py`),
/// the order the 4 `foot_contacts` channels refer to.
pub const CORE_FOOT_JOINT_NAMES: [&str; 4] =
    ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"];

/// `hip_joint_names` -- `[right, left]` (`definitions.py`); the ordering
/// `get_root_heading_angle` relies on.
pub const CORE_HIP_JOINT_NAMES: [&str; 2] = ["RightUpLeg", "LeftUpLeg"];

/// A skeleton: hierarchy + neutral joints + FK.
///
/// `joint_parents` is ordered so that every parent index is smaller than its
/// children's, which is what lets [`Skeleton::fk`] walk joints once, in
/// order, instead of the reference's per-level scatter. The multiplication
/// chain per joint is identical either way (`G_j = G_parent @ T_j`).
#[derive(Debug)]
pub struct Skeleton {
    /// Skeleton identifier, e.g. `"cskel27"`.
    pub name: &'static str,
    /// Joint names in index order.
    pub joint_names: Vec<&'static str>,
    /// Parent index per joint (`-1` for the root).
    pub joint_parents: Vec<i32>,
    /// Number of joints.
    pub nbjoints: usize,
    /// Index of the root joint (0).
    pub root_index: usize,
    /// Indices of the four foot-contact joints.
    pub foot_joint_idx: Vec<usize>,
    /// Indices of the two hip joints, `[right, left]`.
    pub hip_joint_idx: Vec<usize>,
    /// Relative joint offsets in the parent frame: `neutral[j] -
    /// neutral[parent[j]]` (`kinematics.py`), flattened `[J, 3]` float32. The
    /// root keeps its own offset (zero for Core).
    rel_joints: Vec<f32>,
}

impl Skeleton {
    /// Build a skeleton and validate the FK preconditions: one root, and
    /// every parent index smaller than its child's.
    pub fn new(
        name: &'static str,
        joint_names: &[&'static str],
        joint_parents: &[i32],
        neutral_joints: &[[f64; 3]],
        root_index: usize,
        foot_joint_names: &[&str],
        hip_joint_names: &[&str],
    ) -> Result<Self> {
        let nbjoints = joint_names.len();
        if joint_parents.len() != nbjoints || neutral_joints.len() != nbjoints {
            return Err(Error::Contract(format!(
                "skeleton {name:?}: {} names, {} parents, neutral {}",
                nbjoints,
                joint_parents.len(),
                neutral_joints.len()
            )));
        }
        let roots: Vec<usize> = joint_parents
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p < 0)
            .map(|(i, _)| i)
            .collect();
        if roots != [root_index] {
            return Err(Error::Contract(format!(
                "skeleton {name:?}: expected root {root_index}, got {roots:?}"
            )));
        }
        for (j, &p) in joint_parents.iter().enumerate() {
            if j != root_index && p >= j as i32 {
                return Err(Error::Contract(format!(
                    "skeleton {name:?}: joint {j} ({}) has parent {p} >= {j}; \
                     FK relies on parents coming first",
                    joint_names[j]
                )));
            }
        }

        // The reference casts the rest pose to the compute dtype *before*
        // subtracting, so the float32 rest pose is the one that matters.
        let neutral32: Vec<[f32; 3]> = neutral_joints
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let mut rel: Vec<f32> = Vec::with_capacity(nbjoints * 3);
        for (j, p) in neutral32.iter().enumerate() {
            let q = if j == root_index {
                [0.0f32; 3]
            } else {
                let par = neutral32[joint_parents[j] as usize];
                [p[0] - par[0], p[1] - par[1], p[2] - par[2]]
            };
            rel.extend_from_slice(&q);
        }

        let foot_joint_idx = foot_joint_names
            .iter()
            .map(|n| {
                joint_names
                    .iter()
                    .position(|m| m == n)
                    .ok_or_else(|| Error::Contract(format!("skeleton {name:?}: no joint {n:?}")))
            })
            .collect::<Result<Vec<usize>>>()?;
        let hip_joint_idx = hip_joint_names
            .iter()
            .map(|n| {
                joint_names
                    .iter()
                    .position(|m| m == n)
                    .ok_or_else(|| Error::Contract(format!("skeleton {name:?}: no joint {n:?}")))
            })
            .collect::<Result<Vec<usize>>>()?;

        Ok(Self {
            name,
            joint_names: joint_names.to_vec(),
            joint_parents: joint_parents.to_vec(),
            nbjoints,
            root_index,
            foot_joint_idx,
            hip_joint_idx,
            rel_joints: rel,
        })
    }

    /// Working-rig parent-relative offset for joint `j > 0`
    /// (`postprocess.py::create_working_rig_from_skeleton`:
    /// `neutral[j] - neutral[parent]`, cast to float32).
    pub fn core_rig_offset(&self, j: usize) -> [f32; 3] {
        let p = self.joint_parents[j] as usize;
        let n = &CORE_NEUTRAL_JOINTS;
        [
            (n[j][0] - n[p][0]) as f32,
            (n[j][1] - n[p][1]) as f32,
            (n[j][2] - n[p][2]) as f32,
        ]
    }

    /// Working-rig root translation: `neutral[0] + [0, -toe_height + 0.007,
    /// 0]` with `toe_height = min neutral y` — the rig sits just above the
    /// ground so contact IK's `minHeight` is positive.
    pub fn core_rig_root_translation(&self) -> [f32; 3] {
        let n = &CORE_NEUTRAL_JOINTS;
        let toe_height = n.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        [
            n[0][0] as f32,
            (n[0][1] - toe_height + 0.007) as f32,
            n[0][2] as f32,
        ]
    }

    /// The Core (27-joint) skeleton used by every released Core checkpoint.
    pub fn core() -> &'static Skeleton {
        static CORE: OnceLock<Skeleton> = OnceLock::new();
        CORE.get_or_init(|| {
            // The const tables above are compile-time validated by unit test;
            // construction cannot fail.
            Skeleton::new(
                "cskel27",
                &CORE_JOINT_NAMES,
                &CORE_JOINT_PARENTS,
                &CORE_NEUTRAL_JOINTS,
                CORE_ROOT_INDEX,
                &CORE_FOOT_JOINT_NAMES,
                &CORE_HIP_JOINT_NAMES,
            )
            .expect("core skeleton tables are valid")
        })
    }

    /// `batch_rigid_transform` (`skeleton/kinematics.py`).
    ///
    /// `local_rot_mats` is `(..., J, 3, 3)`, `root_positions` `(..., 3)`.
    /// Returns `(global_rot_mats, posed_joints, posed_joints_norootpos)` with
    /// shapes `(..., J, 3, 3)`, `(..., J, 3)`, `(..., J, 3)`.
    ///
    /// `root_positions_is_global=True` in the reference subtracts
    /// `neutral_joints[root_idx]` from the rest pose so that `root_positions`
    /// is the root's *world* position; on this skeleton that offset is
    /// exactly zero, but the subtraction is kept for faithfulness (the
    /// `rel_joints` construction already bakes it in).
    pub fn fk(
        &self,
        local_rot_mats: &Array,
        root_positions: &Array,
    ) -> Result<(Array, Array, Array)> {
        let shape = local_rot_mats.shape();
        let rank = shape.len();
        if rank < 3 || shape[rank - 1] != 3 || shape[rank - 2] != 3 {
            return Err(Error::Contract(format!(
                "local_rot_mats must be (..., J, 3, 3), got {shape:?}"
            )));
        }
        if shape[rank - 3] != self.nbjoints as i32 {
            return Err(Error::Contract(format!(
                "expected {} joints, got {}",
                self.nbjoints,
                shape[rank - 3]
            )));
        }
        let lead = &shape[..rank - 3];
        let t: i32 = lead.iter().product::<i32>().max(1);
        let j = self.nbjoints as i32;
        let rots = local_rot_mats.reshape(&[t, j, 3, 3])?;
        let root_pos = root_positions.reshape(&[t, 3])?;

        // T_j = [[R_j, rel_j], [0, 0, 0, 1]]
        let rel = Array::from_slice(&self.rel_joints, &[1, j, 3, 1]);
        let rel_col = ops::broadcast_to(&rel, &[t, j, 3, 1])?;
        let top = ops::concatenate(&[&rots, &rel_col], -1)?; // [t, J, 3, 4]
        let bottom = ops::concatenate(
            &[
                &Array::zeros::<f32>(&[t, j, 1, 3])?,
                &Array::ones::<f32>(&[t, j, 1, 1])?,
            ],
            -1,
        )?; // [t, J, 1, 4]
        let tm = ops::concatenate(&[&top, &bottom], -2)?; // [t, J, 4, 4]

        // G_root = T_root; G_j = G_parent @ T_j  (parents always come first)
        let mut transforms: Vec<Array> = Vec::with_capacity(self.nbjoints);
        transforms.push(tm.try_index((0i32.., 0i32))?); // [t, 4, 4]
        for jf in 1..self.nbjoints {
            let parent_t = &transforms[self.joint_parents[jf] as usize]; // [t, 4, 4]
            let tj = tm.try_index((0i32.., jf as i32))?; // [t, 4, 4]
            transforms.push(ops::matmul(parent_t, &tj)?);
        }
        let glob = ops::stack(&transforms.iter().collect::<Vec<_>>(), 1)?; // [t, J, 4, 4]

        let posed_norootpos = glob
            .try_index((Ellipsis, 0..3, 3..4))?
            .reshape(&[t, j, 3])?;
        let posed_joints = ops::add(&posed_norootpos, &root_pos.expand_dims(1)?)?;
        let global_rot_mats = glob.try_index((Ellipsis, 0..3, 0..3))?;

        // squeeze the folded leading axis back out (`t` may be 1)
        let squeeze = |a: Array, extra: &[i32]| -> Result<Array> {
            let mut s: Vec<i32> = lead.to_vec();
            s.extend_from_slice(extra);
            Ok(a.reshape(&s)?)
        };
        Ok((
            squeeze(global_rot_mats, &[j, 3, 3])?,
            squeeze(posed_joints, &[j, 3])?,
            squeeze(posed_norootpos, &[j, 3])?,
        ))
    }
}

/// Batched 3-vector cross product keeping the last axis (`MLX has no cross`).
fn cross(a: &Array, b: &Array) -> Result<Array> {
    // python `a[..., i]`: int index on the last axis, dropping it, so the
    // stacked result is `[..., 3]` (no keepdims anywhere)
    let ch = |v: &Array, i: usize| v.try_index((Ellipsis, i as i32));
    Ok(ops::stack(
        &[
            &ops::subtract(
                &ops::multiply(&ch(a, 1)?, &ch(b, 2)?)?,
                &ops::multiply(&ch(a, 2)?, &ch(b, 1)?)?,
            )?,
            &ops::subtract(
                &ops::multiply(&ch(a, 2)?, &ch(b, 0)?)?,
                &ops::multiply(&ch(a, 0)?, &ch(b, 2)?)?,
            )?,
            &ops::subtract(
                &ops::multiply(&ch(a, 0)?, &ch(b, 1)?)?,
                &ops::multiply(&ch(a, 1)?, &ch(b, 0)?)?,
            )?,
        ],
        -1,
    )?)
}

fn l2_keepdims(v: &Array) -> Result<Array> {
    Ok(ops::sqrt(&ops::sum_axes(
        &ops::multiply(v, v)?,
        &[-1],
        true,
    )?)?)
}

/// `cont6d_to_matrix` (`geometry.py`), Gram-Schmidt on two vectors.
///
/// `cont6d` is `(..., 6)` = two 3-vectors that are *approximately* the first
/// two basis vectors of the rotation; the Gram-Schmidt pass is what makes
/// them orthonormal, and it is not cosmetic: on the golden clip the raw 6D
/// block is `0.026` away from the matrix it reconstructs. The result is
/// `(..., 3, 3)` with x/y/z as **columns** -- the convention the FK chain
/// and `global_rots_to_local_rots` expect.
pub fn cont6d_to_matrix(cont6d: &Array) -> Result<Array> {
    let shape = cont6d.shape();
    if shape[shape.len() - 1] != 6 {
        return Err(Error::Contract(format!(
            "the last dimension must be 6, got {}",
            shape[shape.len() - 1]
        )));
    }
    let x_raw = cont6d.try_index((Ellipsis, 0..3))?;
    let y_raw = cont6d.try_index((Ellipsis, 3..6))?;

    let x = ops::divide(&x_raw, &l2_keepdims(&x_raw)?)?;
    let z = ops::divide(&cross(&x, &y_raw)?, &l2_keepdims(&cross(&x, &y_raw)?)?)?;
    let y = cross(&z, &x)?;

    Ok(ops::stack(&[&x, &y, &z], -1)?)
}

/// `matrix_to_cont6d` (`geometry.py`): first two **columns** of the trailing
/// 3x3, concatenated. `cont6d_to_matrix` inverts this exactly.
pub fn matrix_to_cont6d(matrix: &Array) -> Result<Array> {
    // python `matrix[..., :, i]`: int index on the last axis, dropping it
    Ok(ops::concatenate(
        &[
            &matrix.try_index((Ellipsis, 0i32))?,
            &matrix.try_index((Ellipsis, 1i32))?,
        ],
        -1,
    )?)
}

/// `global_rots_to_local_rots` (`skeleton/transforms.py`).
///
/// `local_j = R_parent_global^T @ R_j_global`, with the root's parent treated
/// as identity. This is what the reference's `inverse()` uses to turn the
/// stored `global_rot_data` back into the local rotations FK consumes.
pub fn global_rots_to_local_rots(global_rot_mats: &Array, skel: &Skeleton) -> Result<Array> {
    let shape = global_rot_mats.shape();
    if shape[shape.len() - 3] != skel.nbjoints as i32 {
        return Err(Error::Contract(format!(
            "expected {} joints, got {}",
            skel.nbjoints,
            shape[shape.len() - 3]
        )));
    }
    let idx = Array::from_slice(&skel.joint_parents, &[skel.nbjoints as i32]);
    let parent = global_rot_mats.take_axis(&idx, -3)?;
    // the root's parent is identity
    let lead = &parent.shape()[..parent.shape().len() - 3];
    let mut eye_shape = lead.to_vec();
    eye_shape.extend_from_slice(&[1, 3, 3]);
    let eye = ops::broadcast_to(&ops::eye::<f32>(3, None, None)?, &eye_shape)?;
    let rest = parent.try_index((Ellipsis, 1..skel.nbjoints as i32, 0..3, 0..3))?;
    let parent = ops::concatenate(&[&eye, &rest], -3)?;
    Ok(ops::matmul(
        &ops::swap_axes(&parent, -1, -2)?,
        global_rot_mats,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parents-first precondition FK relies on, and the foot/hip index
    /// lookups.
    #[test]
    fn core_skeleton_tables_valid() {
        let skel = Skeleton::core();
        assert_eq!(skel.nbjoints, 27);
        assert_eq!(skel.joint_parents.len(), 27);
        assert_eq!(skel.rel_joints.len(), 27 * 3);
        assert_eq!(skel.foot_joint_idx, vec![25, 26, 21, 22]);
        assert_eq!(skel.hip_joint_idx, vec![19, 23]);
        // every parent index smaller than its child's
        for (j, &p) in skel.joint_parents.iter().enumerate() {
            assert!(p < j as i32 || j == skel.root_index);
        }
    }

    /// `cont6d_to_matrix` inverts `matrix_to_cont6d` exactly for a rotation.
    #[test]
    fn cont6d_roundtrip() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // a rotation about y by 30 degrees (c^2 + s^2 == 1), batched
        let (c, s) = (0.75f32.sqrt(), 0.5f32);
        let r = Array::from_slice(
            &[
                c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c, c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c,
            ],
            &[2, 3, 3],
        );
        let c6d = matrix_to_cont6d(&r).unwrap();
        assert_eq!(c6d.shape(), &[2, 6]);
        let back = cont6d_to_matrix(&c6d).unwrap();
        assert_eq!(back.shape(), &[2, 3, 3]);
        let diff = ops::max(&ops::abs(&(back - r))?, None)?;
        let diff = diff.reshape(&[-1])?.as_slice::<f32>()[0];
        assert!(diff < 1e-6, "cont6d roundtrip off by {diff}");
        Ok(())
    }
}
