//! Motion representation: feature slicing, normalisation stats, root/local-root.
//!
//! Ports the parts of `ardy/motion_rep/reps/base.py` and
//! `ardy/motion_rep/reps/ardy_motionrep.py` that the batch generation path
//! actually touches, with the reference line numbers in the docstrings.
//!
//! The one genuinely surprising thing here is the **statistics slicing**. The
//! shipped `stats/motion/{mean,std}.npy` vectors are 334 long, but the feature
//! vector that flows through the model is 330 long:
//!
//! ```text
//!     stats layout : [ global_root 5 ][ local_root 4 ][ body 325 ]   = 334
//!     feature vec  : [ global_root 5 ][          body 325 ]          = 330
//! ```
//!
//! so the 330-wide normalisation uses indices `0..5` plus `9..334`
//! (`base.py:66-72`), skipping the four local-root slots.

use mlx_rs::{
    Array, Dtype, ops,
    ops::indexing::{Ellipsis, TryIndexOp},
};

use crate::error::{Error, Result};

use super::nn::scalar;

/// `Stats.__init__` default (ardy/motion_rep/stats.py:27).
pub const STATS_EPS: f32 = 1e-5;

/// `(data - mean) / sqrt(std**2 + eps)` (ardy/motion_rep/stats.py:15-82).
///
/// The eps-adjusted denominator is precomputed once, because the reference
/// does the same (`stats.py:70-75`) and the two must agree bit for bit.
#[derive(Clone, Debug)]
pub struct Stats {
    mean: Array,
    std_eps: Array,
}

impl Stats {
    /// Build from a raw mean/std pair (both 1-D, equal length).
    pub fn from_pair(mean: &[f32], std: &[f32], eps: f32) -> Result<Self> {
        if mean.len() != std.len() {
            return Err(Error::Contract(format!(
                "stats mean len {} != std len {}",
                mean.len(),
                std.len()
            )));
        }
        let std_eps: Vec<f32> = std.iter().map(|s| (s * s + eps).sqrt()).collect();
        let n = mean.len() as i32;
        Ok(Self {
            mean: Array::from_slice(mean, &[n]),
            std_eps: Array::from_slice(&std_eps, &[n]),
        })
    }

    /// Build from the bundle's parsed `mean.npy`/`std.npy` pair.
    pub fn from_npy(
        mean: &crate::io::npy::NpyArray,
        std: &crate::io::npy::NpyArray,
    ) -> Result<Self> {
        if mean.shape.len() != 1 {
            return Err(Error::Contract(format!(
                "stats mean must be 1-D, got shape {:?}",
                mean.shape
            )));
        }
        Self::from_pair(&mean.data, &std.data, STATS_EPS)
    }

    /// `Stats.sliced` (stats.py:36-43) — select feature indices.
    pub fn sliced(&self, indices: &[usize]) -> Result<Self> {
        let idx: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let idx = Array::from_slice(&idx, &[idx.len() as i32]);
        Ok(Self {
            mean: self.mean.take_axis(&idx, 0)?,
            std_eps: self.std_eps.take_axis(&idx, 0)?,
        })
    }

    /// Feature width `D`.
    pub fn dim(&self) -> usize {
        self.mean.shape()[0] as usize
    }

    /// The precomputed `(mean, sqrt(std**2 + eps))` pair, for tests.
    pub fn parts(&self) -> (&Array, &Array) {
        (&self.mean, &self.std_eps)
    }

    /// `(data - mean) / std_eps`.
    pub fn normalize(&self, data: &Array) -> Result<Array> {
        Ok(ops::divide(
            &ops::subtract(data, &self.mean)?,
            &self.std_eps,
        )?)
    }

    /// `data * std_eps + mean`.
    pub fn unnormalize(&self, data: &Array) -> Result<Array> {
        Ok(ops::add(&ops::multiply(data, &self.std_eps)?, &self.mean)?)
    }
}

/// `diff_angles` (ardy/motion_rep/tools.py:13-31): fps-scaled wrapped difference.
pub fn diff_angles(angles: &Array, fps: f32) -> Result<Array> {
    let cos = ops::cos(angles)?;
    let sin = ops::sin(angles)?;
    let t = angles.shape()[angles.ndim() - 1];
    let cos_prev = cos.try_index((Ellipsis, ..(t - 1)))?;
    let cos_next = cos.try_index((Ellipsis, 1..))?;
    let sin_prev = sin.try_index((Ellipsis, ..(t - 1)))?;
    let sin_next = sin.try_index((Ellipsis, 1..))?;
    let cos_diff = ops::add(
        &ops::multiply(&cos_next, &cos_prev)?,
        &ops::multiply(&sin_next, &sin_prev)?,
    )?;
    let sin_diff = ops::subtract(
        &ops::multiply(&sin_next, &cos_prev)?,
        &ops::multiply(&cos_next, &sin_prev)?,
    )?;
    Ok(ops::multiply(
        &ops::atan2(&sin_diff, &cos_diff)?,
        scalar(fps),
    )?)
}

/// `compute_vel_xyz` / `compute_vel_angle`'s scatter (tools.py:66-79, 100-110).
///
/// The reference pads the finite-difference sequence with one zero **and then**
/// writes `v[lengths-1] = v[clamp(lengths-2, min=0)]` with a gather/scatter.
/// `lengths` is *not* the tensor's time extent in general: the denoiser passes
/// `lengths = history_len + generation_len` while the tensor spans
/// `history_len + generation_len + future_len` frames. On a single-window clip
/// the two coincide, which is exactly why the difference is invisible until a
/// second window exists (window 0 of the 4.0 s run: `lengths = 40` over an
/// 80-frame tensor).
///
/// `lengths` is `None` means "the whole sequence is valid", i.e. the last entry
/// is replaced — which is what every call site that omits `lengths` gets.
fn repeat_diff_at_last(velocity: &Array, lengths: Option<&Array>) -> Result<Array> {
    let b = velocity.shape()[0];
    let t = velocity.shape()[1];
    if t < 2 {
        return Ok(velocity.clone());
    }
    let idx: Array = match lengths {
        None => ops::full::<i32>(&[b], Array::from_slice(&[t - 1], &[]))?,
        Some(l) => {
            let li = l.as_dtype(Dtype::Int32)?.reshape(&[b])?;
            ops::maximum(
                &ops::subtract(&li, Array::from_slice(&[1], &[]))?,
                Array::from_slice(&[0], &[]),
            )?
        }
    };
    let src = ops::maximum(
        &ops::subtract(&idx, Array::from_slice(&[1], &[]))?,
        Array::from_slice(&[0], &[]),
    )?;
    // velocity[b, src[b]] for every batch row: take_along_axis with the index
    // expanded to velocity's rank.
    let mut src_nd = src.reshape(&[b])?;
    while (src_nd.ndim() as usize) < velocity.ndim() {
        src_nd = src_nd.expand_dims(-1)?;
    }
    let src_vals = velocity.take_along_axis(&src_nd, 1)?;
    let ar = ops::arange::<i32, i32>(None, t, None)?;
    // [B, T] bool — explicit shapes: with B > 1 the bare [T] vs [B] pair
    // cannot broadcast.
    let hit = ar.reshape(&[1, t])?.eq(&idx.reshape(&[b, 1])?)?;
    let mut hit_shape: Vec<i32> = vec![b, t];
    while hit_shape.len() < velocity.ndim() {
        hit_shape.push(1);
    }
    let hit = hit.reshape(&hit_shape)?;
    Ok(ops::select(&hit, &src_vals, velocity)?)
}

/// `compute_vel_xyz` (ardy/motion_rep/tools.py:34-80).
///
/// `v[t] = fps * (p[t+1] - p[t])`, zero-padded at the end, then `v[lengths-1]`
/// is overwritten with `v[lengths-2]` (so the final valid frame repeats the
/// previous velocity rather than staying zero). See [`repeat_diff_at_last`]
/// for why `lengths` matters.
pub fn compute_vel_xyz(positions: &Array, fps: f32, lengths: Option<&Array>) -> Result<Array> {
    let t = positions.shape()[1];
    // python: positions[:, 1:] - positions[:, :-1] — slice axis 1, keep the rest.
    let next = positions.try_index(((0i32..), (1i32..)))?;
    let prev = positions.try_index(((0i32..), (..(t - 1))))?;
    let diff = ops::multiply(&ops::subtract(&next, &prev)?, scalar(fps))?;
    let b = positions.shape()[0];
    let mut pad_shape: Vec<i32> = vec![b, 1];
    pad_shape.extend_from_slice(&positions.shape()[2..]);
    let pad = ops::zeros::<f32>(&pad_shape)?;
    let velocity = ops::concatenate(&[&diff, &pad], 1)?;
    repeat_diff_at_last(&velocity, lengths)
}

/// Zero-pad one step at the end, then repeat `x[lengths-2]` at `x[lengths-1]`.
///
/// Mirrors the `einops.pack` + gather/scatter tail of `compute_vel_angle`
/// (tools.py:88-110). `lengths=None` means the whole sequence is valid, so the
/// repeated index is the last frame — the single-window behaviour.
fn pad_last(x: &Array, lengths: Option<&Array>) -> Result<Array> {
    let mut pad_shape: Vec<i32> = x.shape().to_vec();
    let last = pad_shape.len() - 1;
    pad_shape[last] = 1;
    let pad = ops::zeros::<f32>(&pad_shape)?;
    let padded = ops::concatenate(&[x, &pad], -1)?;
    repeat_diff_at_last(&padded, lengths)
}

/// Minimal port of `ArdyMotionRep` for the Core skeleton (J=27, 20 fps).
///
/// Only the block layout, the statistics, and the root/local-root conversions
/// used by the batch path are implemented. Block order is
/// `docs/motion-contract.md` S1.1:
///
/// | block | width | slice |
/// |---|---|---|
/// | `root_pos` | 3 | 0..3 |
/// | `global_root_heading` | 2 | 3..5 |
/// | `local_joints_positions` | (J-1)·3 | 5..83 |
/// | `global_rot_data` | J·6 | 83..245 |
/// | `velocities` | J·3 | 245..326 |
/// | `foot_contacts` | 4 | 326..330 |
#[derive(Clone, Debug)]
pub struct MotionRep {
    nbjoints: usize,
    fps: f32,
    global_root_dim: usize,
    local_root_dim: usize,
    body_dim: usize,
    motion_rep_dim: usize,
    full_stats: Stats,
    global_root_stats: Stats,
    local_root_stats: Stats,
    body_stats: Stats,
}

impl MotionRep {
    /// Build from the full `(mean, std)` stats pair and the skeleton shape.
    ///
    /// Validates the stats width against the block layout, then builds the
    /// 330/5/4/325-wide slices exactly as `ArdyMotionRep._from_blocks` does.
    pub fn new(
        mean: &crate::io::npy::NpyArray,
        std: &crate::io::npy::NpyArray,
        nbjoints: usize,
        fps: f32,
    ) -> Result<Self> {
        let all_stats = Stats::from_npy(mean, std)?;
        let global_root_dim = 3 + 2; // root_pos + global_root_heading
        let local_root_dim = 4; // rot_vel(1) + planar vel(2) + root y(1)
        let body_dim = (nbjoints - 1) * 3 + nbjoints * 6 + nbjoints * 3 + 4;
        let motion_rep_dim = global_root_dim + body_dim;
        if all_stats.dim() != global_root_dim + local_root_dim + body_dim {
            return Err(Error::Contract(format!(
                "stats dim {} != global_root {global_root_dim} + local_root {local_root_dim} + body {body_dim}",
                all_stats.dim()
            )));
        }
        let mut sel: Vec<usize> = (0..global_root_dim).collect();
        sel.extend(global_root_dim + local_root_dim..all_stats.dim());
        Ok(Self {
            nbjoints,
            fps,
            global_root_dim,
            local_root_dim,
            body_dim,
            motion_rep_dim,
            full_stats: all_stats.sliced(&sel)?,
            global_root_stats: all_stats.sliced(&(0..global_root_dim).collect::<Vec<_>>())?,
            local_root_stats: all_stats
                .sliced(&(global_root_dim..global_root_dim + local_root_dim).collect::<Vec<_>>())?,
            body_stats: all_stats
                .sliced(&(global_root_dim + local_root_dim..all_stats.dim()).collect::<Vec<_>>())?,
        })
    }

    /// Build from a loaded [`crate::io::bundle::Bundle`] (`stats/motion`).
    pub fn from_bundle(bundle: &crate::io::bundle::Bundle) -> Result<Self> {
        let raw = bundle.stats("motion")?;
        let fps = bundle.layout().fps as f32;
        Self::new(&raw.mean, &raw.std, bundle.layout().nbjoints, fps)
    }

    /// Joint count (27 for Core).
    pub fn nbjoints(&self) -> usize {
        self.nbjoints
    }

    /// Sample rate in Hz.
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Width of the full feature vector (`root_pos 3 + heading 2 + body 325`).
    pub fn motion_rep_dim(&self) -> usize {
        self.motion_rep_dim
    }

    /// Width of the root block (`root_pos` + `global_root_heading` = 5).
    pub fn global_root_dim(&self) -> usize {
        self.global_root_dim
    }

    /// Alias for [`Self::global_root_dim`] — the contract calls this the root block.
    pub fn root_block_dim(&self) -> usize {
        self.global_root_dim
    }

    /// Width of the local-root representation (`rot_vel, vx, vz, y` = 4).
    pub fn local_root_dim(&self) -> usize {
        self.local_root_dim
    }

    /// Width of the body block (`motion_rep_dim - global_root_dim` = 325).
    pub fn body_dim(&self) -> usize {
        self.body_dim
    }

    /// Width of the tokenizer's per-frame output (`local_root 4 + body 325`).
    pub fn local_motion_rep_dim(&self) -> usize {
        self.local_root_dim + self.body_dim
    }

    /// `extract_root` (base.py:150-154 context): `features[..., :5]`.
    pub fn extract_root(&self, features: &Array) -> Result<Array> {
        Ok(features.try_index((Ellipsis, ..(self.global_root_dim as i32)))?)
    }

    /// `extract_body`: `features[..., 5:]`.
    pub fn extract_body(&self, features: &Array) -> Result<Array> {
        Ok(features.try_index((Ellipsis, (self.global_root_dim as i32)..))?)
    }

    /// `concat_root_body`: `[root | body]` on the last axis.
    pub fn concat_root_body(&self, root: &Array, body: &Array) -> Result<Array> {
        Ok(ops::concatenate(&[root, body], -1)?)
    }

    /// Normalise the 330-wide feature vector with the sliced stats.
    pub fn normalize(&self, features: &Array) -> Result<Array> {
        self.full_stats.normalize(features)
    }

    /// Un-normalise the 330-wide feature vector.
    pub fn unnormalize(&self, features: &Array) -> Result<Array> {
        self.full_stats.unnormalize(features)
    }

    /// Sliced stats access, for tests and callers that need them directly.
    pub fn stats(
        &self,
    ) -> (
        /*full*/ &Stats,
        /*global root*/ &Stats,
        /*local root*/ &Stats,
        /*body*/ &Stats,
    ) {
        (
            &self.full_stats,
            &self.global_root_stats,
            &self.local_root_stats,
            &self.body_stats,
        )
    }

    /// Translate root x/z so `center_frame_index` becomes the local origin
    /// (`ardy_motionrep.py:112-134`).
    ///
    /// The y component of the centre is forced to 0, so recentring never moves
    /// the character vertically.
    pub fn recenter_root_motion(
        &self,
        root_motion: &Array,
        center_frame_index: usize,
        is_normalized: bool,
        to_normalize: bool,
    ) -> Result<Array> {
        Ok(self
            .recenter_root_motion_with_center(
                root_motion,
                center_frame_index,
                is_normalized,
                to_normalize,
            )?
            .0)
    }

    /// [`Self::recenter_root_motion`] returning the centre position too.
    pub fn recenter_root_motion_with_center(
        &self,
        root_motion: &Array,
        center_frame_index: usize,
        is_normalized: bool,
        to_normalize: bool,
    ) -> Result<(Array, Array)> {
        let root = if is_normalized {
            self.global_root_stats.unnormalize(root_motion)?
        } else {
            root_motion.clone()
        };
        let mut center_pos = root.try_index((
            (0i32..),
            ((center_frame_index as i32)..(center_frame_index as i32 + 1)),
            (0i32..3),
        ))?; // [B, 1, 3]
        center_pos = center_pos.reshape(&[-1, 3])?; // [B, 3]
        let cp0 = center_pos.try_index((Ellipsis, ..1))?;
        let cp2 = center_pos.try_index((Ellipsis, 2..3))?;
        let zero = ops::zeros::<f32>(&[cp0.shape()[0], 1])?;
        let center_pos = ops::concatenate(&[&cp0, &zero, &cp2], -1)?;

        let root_pos = root.try_index((Ellipsis, ..3))?;
        let rp0 = root_pos.try_index((Ellipsis, ..1))?;
        let rp1 = root_pos.try_index((Ellipsis, 1..2))?;
        let rp2 = root_pos.try_index((Ellipsis, 2..3))?;
        let shifted = ops::concatenate(
            &[
                &ops::subtract(&rp0, &cp0.expand_dims(1)?)?,
                &rp1,
                &ops::subtract(&rp2, &cp2.expand_dims(1)?)?,
            ],
            -1,
        )?;
        let tail = root.try_index((Ellipsis, 3..))?;
        let out = ops::concatenate(&[&shifted, &tail], -1)?;
        let out = if to_normalize {
            self.global_root_stats.normalize(&out)?
        } else {
            out
        };
        Ok((out, center_pos))
    }

    /// `translate_normalized_root_motion` (ardy_model.py:29-38).
    ///
    /// The translation's y component must be zero (asserted in the reference).
    pub fn translate_normalized_root_motion(
        &self,
        root_motion: &Array,
        translation: &Array,
    ) -> Result<Array> {
        if *translation.shape().last().unwrap() != 3 {
            return Err(Error::Contract(
                "translation must be a 3-vector".to_string(),
            ));
        }
        let root = self.global_root_stats.unnormalize(root_motion)?;
        let t = translation.expand_dims(1)?; // [B, 1, 3]
        let root_pos = ops::add(&root.try_index((Ellipsis, ..3))?, &t)?;
        let tail = root.try_index((Ellipsis, 3..))?;
        self.global_root_stats
            .normalize(&ops::concatenate(&[&root_pos, &tail], -1)?)
    }

    /// `get_root_heading_angle` (base.py:150-154): `atan2(h1, h0)`.
    pub fn get_root_heading_angle(&self, features: &Array) -> Result<Array> {
        let heading = features.try_index((Ellipsis, (self.global_root_dim as i32 - 2)..))?;
        let h1 = heading.try_index((Ellipsis, 1))?;
        let h0 = heading.try_index((Ellipsis, 0))?;
        Ok(ops::atan2(&h1, &h0)?)
    }

    /// `global_root_to_local_root` (base.py:115-148).
    ///
    /// Returns `[B, T, 4]` = `(dtheta/dt, vx, vz, root_y)` where the angles
    /// come from `atan2(sin, cos)` of the stored heading.
    ///
    /// **Easy to get wrong:** `normalized` governs *both* ends. The global
    /// root block is un-normalised on the way in, and the resulting 4-vector is
    /// re-normalised with `local_root_stats` on the way out (`base.py:145-146`).
    /// So the local-root condition handed to the Stage-A decoder and to the
    /// Stage-B body stage is in *normalised* space, not metres and m/s.
    ///
    /// `lengths` is forwarded to the two velocity helpers, where it decides
    /// **which** frame receives the repeated velocity (`lengths - 1`, not the
    /// tensor's last frame). Omitting it means "the whole sequence is valid".
    pub fn global_root_to_local_root(
        &self,
        root_features: &Array,
        normalized: bool,
        lengths: Option<&Array>,
    ) -> Result<Array> {
        let root = if normalized {
            self.global_root_stats.unnormalize(root_features)?
        } else {
            root_features.clone()
        };
        let root_pos = root.try_index((Ellipsis, ..3))?;
        let heading = root.try_index((Ellipsis, 3..5))?;
        // python: cos = heading[..., 0]; sin = heading[..., 1] (int indices,
        // dropping the channel axis -> [B, T])
        let cos = heading.try_index((Ellipsis, 0))?;
        let sin = heading.try_index((Ellipsis, 1))?;
        let heading_angle = ops::atan2(&sin, &cos)?;
        let rot_vel = pad_last(&diff_angles(&heading_angle, self.fps)?, lengths)?;
        let rot_vel = rot_vel.expand_dims(-1)?;
        // reference: compute_vel_xyz(root_pos[..., None, :], ...)[..., 0, [0, 2]]
        let pos4 = root_pos.expand_dims(2)?; // [B, T, 1, 3]
        let vel4 = compute_vel_xyz(&pos4, self.fps, lengths)?;
        let b = vel4.shape()[0];
        let t = vel4.shape()[1];
        let vel3 = vel4.reshape(&[b, t, 3])?; // [B, T, 3]
        let vel = ops::stack(
            &[
                // python indexes with ints here, dropping the channel axis
                &vel3.try_index((Ellipsis, 0))?,
                &vel3.try_index((Ellipsis, 2))?,
            ],
            -1,
        )?; // planar (x, z) [B, T, 2]
        let global_root_y = root_pos.try_index((Ellipsis, 1..2))?;
        let local_root = ops::concatenate(&[&rot_vel, &vel, &global_root_y], -1)?;
        if normalized {
            Ok(self.local_root_stats.normalize(&local_root)?)
        } else {
            Ok(local_root)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_pair(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mean: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let std: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.1).collect();
        (mean, std)
    }

    #[test]
    fn stats_normalize_unnormalize_roundtrip() {
        let (mean, std) = stats_pair(8);
        let s = Stats::from_pair(&mean, &std, STATS_EPS).unwrap();
        let data = Array::from_slice(&[0.1, -2.0, 3.5, 4.0, -1.0, 0.0, 2.2, 7.7], &[8]);
        let norm = s.normalize(&data).unwrap();
        let back = s.unnormalize(&norm).unwrap();
        let nd = norm.as_slice::<f32>().to_vec();
        let bd = back.as_slice::<f32>().to_vec();
        for i in 0..8 {
            let want = (data.as_slice::<f32>()[i] - mean[i]) / (std[i] * std[i] + STATS_EPS).sqrt();
            assert!((nd[i] - want).abs() < 1e-6, "norm[{i}] {} vs {want}", nd[i]);
            assert!((bd[i] - data.as_slice::<f32>()[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn stats_sliced_selects_indices() {
        let (mean, std) = stats_pair(10);
        let s = Stats::from_pair(&mean, &std, STATS_EPS).unwrap();
        let s2 = s.sliced(&[0, 1, 2, 7, 8, 9]).unwrap();
        assert_eq!(s2.dim(), 6);
        let (m, se) = s2.parts();
        assert_eq!(m.as_slice::<f32>(), [0.0, 0.5, 1.0, 3.5, 4.0, 4.5]);
        // std_eps must be recomputed-from-source equivalent: slicing std_eps
        // elementwise is the same operation.
        for (i, j) in [0usize, 1, 2, 7, 8, 9].iter().enumerate() {
            let want = (std[*j] * std[*j] + STATS_EPS).sqrt();
            assert!((se.as_slice::<f32>()[i] - want).abs() < 1e-6);
        }
    }

    #[test]
    fn diff_angles_wraps_and_scales() {
        // angles: 0, pi/2, pi, 3pi/2 -> consecutive wrapped diffs are all pi/2
        let vals: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].to_vec();
        let a = Array::from_slice(&vals, &[1, 4]);
        let d = diff_angles(&a, 20.0).unwrap();
        assert_eq!(d.shape(), &[1, 3]);
        for k in 0..3 {
            let got = d.as_slice::<f32>()[k];
            assert!((got - 20.0).abs() < 1e-5, "diff[{k}] = {got}");
        }
        // wrap-around: 3.0 -> -3.0 is a step of -6 rad; atan2(sin(-6), cos(-6))
        // folds it back into (-pi, pi] as -6 + 2*pi ~= 0.2832.
        let vals: Vec<f32> = [3.0, -3.0].to_vec();
        let a = Array::from_slice(&vals, &[1, 2]);
        let d = diff_angles(&a, 1.0).unwrap();
        let want = -6.0f32 + 2.0 * std::f32::consts::PI;
        assert!((d.as_slice::<f32>()[0] - want).abs() < 1e-5);
    }

    #[test]
    fn compute_vel_xyz_pads_and_repeats() {
        // positions along x: 0, 1, 3 -> v = [20, 40] with fps 20, then pad -> [20, 40, 0]
        // lengths=None: last entry (0) is replaced by v[1]=40 -> [20, 40, 40]
        let p = Array::from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0], &[1, 3, 3]);
        let v = compute_vel_xyz(&p, 20.0, None).unwrap();
        assert_eq!(v.shape(), &[1, 3, 3]);
        assert_eq!(
            v.as_slice::<f32>(),
            [20.0, 0.0, 0.0, 40.0, 0.0, 0.0, 40.0, 0.0, 0.0]
        );
    }

    #[test]
    fn repeat_diff_at_last_honours_lengths() {
        // Same clip but lengths = [2]: idx = 1, src = 0 -> v[1] = v[0] = 20,
        // and v[2] keeps its padded zero. This is the multi-window trap.
        let p = Array::from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0], &[1, 3, 3]);
        let lengths = Array::from_slice(&[2.0], &[1]);
        let v = compute_vel_xyz(&p, 20.0, Some(&lengths)).unwrap();
        assert_eq!(
            v.as_slice::<f32>(),
            [20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn motion_rep_slices_and_dims() {
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean.clone(),
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std.clone(),
            },
            27,
            20.0,
        )
        .unwrap();
        assert_eq!(mr.motion_rep_dim(), 330);
        assert_eq!(mr.global_root_dim(), 5);
        assert_eq!(mr.local_root_dim(), 4);
        assert_eq!(mr.body_dim(), 325);
        assert_eq!(mr.local_motion_rep_dim(), 329);
        let (full, gr, lr, body) = mr.stats();
        assert_eq!(full.dim(), 330);
        assert_eq!(gr.dim(), 5);
        assert_eq!(lr.dim(), 4);
        assert_eq!(body.dim(), 325);
        // body stats must equal stats indices 9..334
        let (m, _) = gr.parts();
        assert_eq!(m.as_slice::<f32>(), &mean[0..5]);
        let (bm, _) = body.parts();
        assert_eq!(bm.as_slice::<f32>(), &mean[9..334]);
        // full stats: 0..5 then 9..334
        let (fm, _) = full.parts();
        assert_eq!(fm.as_slice::<f32>()[..5], mean[0..5]);
        assert_eq!(&fm.as_slice::<f32>()[5..], &mean[9..334]);
    }

    /// Materialise an array to a contiguous f32 vector (slices of larger
    /// arrays are strided views and cannot be read with `as_slice`).
    fn flat(a: &Array) -> Vec<f32> {
        let f = a.reshape(&[-1]).unwrap();
        f.eval().unwrap();
        f.as_slice::<f32>().to_vec()
    }

    #[test]
    fn extract_root_body_slices() {
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean,
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std,
            },
            27,
            20.0,
        )
        .unwrap();
        let data: Vec<f32> = (0..10 * 330).map(|i| i as f32).collect();
        let f = Array::from_slice(&data, &[1, 10, 330]);
        let root = mr.extract_root(&f).unwrap();
        let body = mr.extract_body(&f).unwrap();
        assert_eq!(root.shape(), &[1, 10, 5]);
        assert_eq!(body.shape(), &[1, 10, 325]);
        let r = flat(&root);
        for t in 0..10 {
            for c in 0..5 {
                assert_eq!(r[t * 5 + c], (t * 330 + c) as f32);
            }
        }
        let b = flat(&body);
        assert_eq!(b[0], 5.0);
        // body element 324 is global feature 5 + 324 = 329 (frame 0's last)
        assert_eq!(b[324], 329.0);
        assert_eq!(b[10 * 325 - 1], (9 * 330 + 329) as f32);
        // concat restores the original
        let joined = mr.concat_root_body(&root, &body).unwrap();
        assert_eq!(flat(&joined), data);
    }

    #[test]
    fn global_root_to_local_root_matches_hand_math() {
        // root: x moves 1 m per frame for 3 frames, y = 0.5, heading = 0 (cos=1, sin=0).
        // rot_vel = 0 everywhere; vel = [20, 0]; y = 0.5.
        let root: Vec<f32> = [
            [0.0, 0.5, 0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0, 1.0, 0.0],
            [2.0, 0.5, 0.0, 1.0, 0.0],
        ]
        .concat();
        let r = Array::from_slice(&root, &[1, 3, 5]);
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean,
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std,
            },
            27,
            20.0,
        )
        .unwrap();
        let local = mr.global_root_to_local_root(&r, false, None).unwrap();
        assert_eq!(local.shape(), &[1, 3, 4]);
        let l = local.as_slice::<f32>();
        for t in 0..3 {
            assert!(l[t * 4].abs() < 1e-6, "rot_vel[{t}] = {}", l[t * 4]);
            // every frame moves +1 m/frame, and the final frame repeats the
            // previous velocity rather than dropping to zero
            assert!((l[t * 4 + 1] - 20.0).abs() < 1e-4);
            assert!(l[t * 4 + 2].abs() < 1e-6);
            assert!((l[t * 4 + 3] - 0.5).abs() < 1e-6);
        }
        // normalized=True expects NORMALISED global root on the way in; feed it
        // gr.normalize(root_raw) and the un-normalised result must agree with
        // the raw call above.
        let (_, gr, lr_stats, _) = mr.stats();
        let r_norm = gr.normalize(&r).unwrap();
        let local_n = mr.global_root_to_local_root(&r_norm, true, None).unwrap();
        let back = lr_stats.unnormalize(&local_n).unwrap();
        for (a, b) in flat(&back).iter().zip(flat(&local)) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn recenter_root_motion_shifts_xz_only() {
        let root: Vec<f32> = [
            [1.0, 1.0, 2.0, 1.0, 0.0],
            [3.0, 1.0, 4.0, 1.0, 0.0],
            [5.0, 1.0, 6.0, 1.0, 0.0],
        ]
        .concat();
        let r = Array::from_slice(&root, &[1, 3, 5]);
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean,
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std,
            },
            27,
            20.0,
        )
        .unwrap();
        let (out, center) = mr
            .recenter_root_motion_with_center(&r, 1, false, false)
            .unwrap();
        assert_eq!(center.shape(), &[1, 3]);
        assert_eq!(center.as_slice::<f32>(), [3.0, 0.0, 4.0]);
        let o = out.as_slice::<f32>();
        // frame 1 becomes the origin; y untouched
        assert_eq!(&o[0..5], &[-2.0, 1.0, -2.0, 1.0, 0.0]);
        assert_eq!(&o[5..10], &[0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(&o[10..15], &[2.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn get_root_heading_angle_uses_atan2() {
        let root: Vec<f32> = [[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0]].concat();
        let r = Array::from_slice(&root, &[1, 2, 5]);
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean,
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std,
            },
            27,
            20.0,
        )
        .unwrap();
        let a = mr.get_root_heading_angle(&r).unwrap();
        assert_eq!(a.shape(), &[1, 2]);
        assert!((a.as_slice::<f32>()[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((a.as_slice::<f32>()[1] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn translate_normalized_root_motion_adds_translation() {
        // translation of exactly one metre along x must add 1 to every x.
        let (mean, std) = stats_pair(334);
        let mr = MotionRep::new(
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: mean,
            },
            &crate::io::npy::NpyArray {
                shape: vec![334],
                data: std,
            },
            27,
            20.0,
        )
        .unwrap();
        let root: Vec<f32> = [[1.0, 1.0, 2.0, 1.0, 0.0], [3.0, 1.0, 4.0, 1.0, 0.0]].concat();
        let r = Array::from_slice(&root, &[1, 2, 5]);
        let t = Array::from_slice(&[1.0, 0.0, 0.0], &[1, 3]);
        let out = mr.translate_normalized_root_motion(&r, &t).unwrap();
        let (_, gr, _, _) = mr.stats();
        // In raw space the result must be unnormalize(root) + t exactly.
        let back = gr.unnormalize(&out).unwrap();
        let base = gr.unnormalize(&r).unwrap();
        let bb = flat(&back);
        let bs = flat(&base);
        for i in 0..10 {
            let want = bs[i] + if i % 5 == 0 { 1.0 } else { 0.0 };
            assert!((bb[i] - want).abs() < 1e-4, "[{i}] {} vs {want}", bb[i]);
        }
    }
}
