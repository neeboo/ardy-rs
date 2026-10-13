//! Decoding a hybrid token sequence into explicit motion features, and those
//! into world-space joints.
//!
//! Two stages, both ported from `ardy_mlx/model/ardy.py` /
//! `ardy-motionrep`:
//!
//! 1. **hybrid -> features** (`explicit_from_hybrid_autoregressive`,
//!    `explicit_from_hybrid_whole`): split the 148-wide hybrid into root
//!    motion and body latent, convert the global root to a local root, run
//!    the Stage-A decoder, and stitch. The *autoregressive* variant detokenises
//!    each generation horizon **with only its own (cropped) history** because
//!    that is what `generate()` does when a history crop is active; the Stage-A
//!    decoder is causal, so the two variants agree up to float32 reduction
//!    order.
//! 2. **features -> joints** ([`inverse`], port of
//!    `ArdyMotionRep.inverse`): unnormalise, unpack the feature blocks,
//!    `global_rot_data -> global_rot_mats -> local_rot_mats -> fk`.

use mlx_rs::Array;
use mlx_rs::ops::{
    self,
    indexing::{Ellipsis, TryIndexOp},
};

use super::fsq::FsqVaeTransformer;
use super::motion_rep::MotionRep;
use super::nn::scalar;
use super::skeleton::{Skeleton, cont6d_to_matrix, global_rots_to_local_rots};
use crate::{Error, Result};

/// Chunk plan of `get_explicit_motion_from_hybrid_autoregressive`
/// (`latent_utils.py` via `ardy.py`).
///
/// Returns `(history_token_start, generation_token_start,
/// generation_token_end)` per generation horizon. `crop_tokens` is
/// `crop_history_length / num_frames_per_token`; with the Core default
/// (160 frames = 40 tokens) a 2-window clip never truncates, but the
/// arithmetic is the same one that does.
pub fn autoregressive_chunk_slices(
    token_len: usize,
    gen_tokens: usize,
    crop_tokens: Option<usize>,
) -> Result<Vec<(usize, usize, usize)>> {
    if gen_tokens < 1 {
        return Err(Error::Contract(format!(
            "gen_tokens must be >= 1, got {gen_tokens}"
        )));
    }
    let mut out = Vec::new();
    let mut token_idx = 0;
    while token_idx < token_len {
        let generation_token_end = (token_idx + gen_tokens).min(token_len);
        let history_token_start = match crop_tokens {
            None => 0,
            Some(c) => token_idx.saturating_sub(c),
        };
        out.push((history_token_start, token_idx, generation_token_end));
        token_idx += gen_tokens;
    }
    Ok(out)
}

/// `explicit_from_hybrid_autoregressive` (`ardy.py:363-395`).
///
/// `latent` is the body latent `[B, T, 128]` and `root_motion` the
/// *normalised* global root `[B, T, 5]` (both from splitting the translated
/// hybrid sequence); the result is the normalised explicit feature vector
/// `[B, T, 330]`. `motion_pad_mask` is `None` on the whole no-constraint
/// batch path, so the local-root conversion runs with `lengths=None`.
#[allow(clippy::too_many_arguments)]
pub fn explicit_from_hybrid_autoregressive(
    latent: &Array,
    root_motion: &Array,
    tokenizer: &FsqVaeTransformer,
    motion_rep: &MotionRep,
    num_frames_per_token: usize,
    gen_horizon_len: usize,
    crop_history_length: Option<usize>,
) -> Result<Array> {
    let f = num_frames_per_token;
    let gen_tokens = gen_horizon_len / f;
    let token_len = latent.shape()[1] as usize;
    let crop_tokens = crop_history_length.map(|c| c / f);
    let mut pieces = Vec::new();
    for (h_start, g_start, g_end) in
        autoregressive_chunk_slices(token_len, gen_tokens, crop_tokens)?
    {
        let (h_fs, g_fs, g_fe) = (h_start * f, g_start * f, g_end * f);
        let sub_root = root_motion.try_index(((0i32..), (h_fs as i32..g_fe as i32), (0i32..)))?;
        let sub_latent = latent.try_index(((0i32..), (h_start as i32..g_end as i32), (0i32..)))?;
        let local_root = motion_rep.global_root_to_local_root(&sub_root, true, None)?;
        let (_root, body) = tokenizer.detokenize(&sub_latent, &local_root, None)?;
        let explicit = motion_rep.concat_root_body(&sub_root, &body)?;
        pieces.push(explicit.try_index((
            (0i32..),
            ((g_fs - h_fs) as i32..(g_fe - h_fs) as i32),
            (0i32..),
        ))?);
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    Ok(ops::concatenate(&refs, 1)?)
}

/// `explicit_from_hybrid_whole` (`ardy.py:395-408`): one detokenise pass over
/// the whole sequence instead of per-horizon chunks.
pub fn explicit_from_hybrid_whole(
    latent: &Array,
    root_motion: &Array,
    tokenizer: &FsqVaeTransformer,
    motion_rep: &MotionRep,
) -> Result<Array> {
    let local_root = motion_rep.global_root_to_local_root(root_motion, true, None)?;
    let (_root, body) = tokenizer.detokenize(latent, &local_root, None)?;
    motion_rep.concat_root_body(root_motion, &body)
}

/// `generate()`'s final decode dispatch (`ardy.py:514-518`): split the
/// translated hybrid, decode (autoregressive when a history crop is active),
/// and trim to `num_frames`.
#[allow(clippy::too_many_arguments)]
pub fn motion_from_hybrid(
    hybrid_translated: &Array,
    tokenizer: &FsqVaeTransformer,
    motion_rep: &MotionRep,
    num_frames_per_token: usize,
    nframe_root_dim: usize,
    gen_horizon_len: usize,
    crop_history_length: Option<usize>,
    num_frames: usize,
) -> Result<Array> {
    let (root_motion, latent) = super::denoiser::split_hybrid_free(
        hybrid_translated,
        num_frames_per_token,
        nframe_root_dim,
        motion_rep.root_block_dim(),
    )?;
    let motion = match crop_history_length {
        Some(crop) => explicit_from_hybrid_autoregressive(
            &latent,
            &root_motion,
            tokenizer,
            motion_rep,
            num_frames_per_token,
            gen_horizon_len,
            Some(crop),
        )?,
        None => explicit_from_hybrid_whole(&latent, &root_motion, tokenizer, motion_rep)?,
    };
    let total = motion.shape()[1] as usize;
    Ok(motion.try_index(((0i32..), (0..num_frames.min(total) as i32), (0i32..)))?)
}

/// Output of [`inverse`] — the tensors `ArdyMotionRep.inverse` returns.
///
/// `smooth_root_pos` is an alias of `root_positions` in the reference and is
/// not duplicated here.
pub struct DecodedMotion {
    /// Local (parent-relative) joint rotation matrices `[B, T, J, 3, 3]`.
    pub local_rot_mats: Array,
    /// Global joint rotation matrices `[B, T, J, 3, 3]`.
    pub global_rot_mats: Array,
    /// World joint positions `[B, T, J, 3]`; joint 0 is the root at
    /// `root_positions`.
    pub posed_joints: Array,
    /// World root position `[B, T, 3]`.
    pub root_positions: Array,
    /// Foot contact flags `[B, T, 4]` (bool, `contact > 0.5`).
    pub foot_contacts: Array,
    /// Root heading `[B, T, 2]` as `(cos theta, sin theta)`.
    pub global_root_heading: Array,
}

/// `ArdyMotionRep.inverse` (`ardy_motionrep.py:237-282`).
///
/// `features` are the explicit motion features `[B, T, 330]`, normalised or
/// not per `is_normalized`. The chain: unnormalise -> unpack blocks ->
/// `cont6d_to_matrix` -> `global_rots_to_local_rots` -> `fk`.
pub fn inverse(
    motion_rep: &MotionRep,
    skel: &Skeleton,
    features: &Array,
    is_normalized: bool,
) -> Result<DecodedMotion> {
    if skel.nbjoints != motion_rep.nbjoints() {
        return Err(Error::Contract(format!(
            "skeleton has {} joints, motion rep expects {}",
            skel.nbjoints,
            motion_rep.nbjoints()
        )));
    }
    let feats = if is_normalized {
        motion_rep.unnormalize(features)?
    } else {
        features.clone()
    };

    // block layout (`ardy_motionrep.py:57-64`): root_pos[3] |
    // heading[2] | local_joints[(J-1)*3] | global_rot_data[J*6] |
    // velocities[J*3] | foot_contacts[4]
    let j = skel.nbjoints;
    let rot_start = 5 + (j - 1) * 3;
    let rot_end = rot_start + j * 6;
    let vel_end = rot_end + j * 3;
    if feats.shape()[feats.shape().len() - 1] != (vel_end + 4) as i32 {
        return Err(Error::Contract(format!(
            "expected the {j}-joint feature vector of width {}, got {}",
            vel_end + 4,
            feats.shape()[feats.shape().len() - 1]
        )));
    }

    let root_positions = feats.try_index((Ellipsis, 0..3))?;
    let global_root_heading = feats.try_index((Ellipsis, 3..5))?;
    let lead: Vec<i32> = feats.shape()[..feats.shape().len() - 1].to_vec();
    let mut rot_shape = lead.clone();
    rot_shape.extend_from_slice(&[j as i32, 6]);
    let global_rot_data = feats
        .try_index((Ellipsis, rot_start as i32..rot_end as i32))?
        .reshape(&rot_shape)?;
    let foot_contacts = ops::gt(
        &feats.try_index((Ellipsis, vel_end as i32..(vel_end + 4) as i32))?,
        scalar(0.5),
    )?;

    let global_rot_mats = cont6d_to_matrix(&global_rot_data)?;
    let local_rot_mats = global_rots_to_local_rots(&global_rot_mats, skel)?;
    let (_g, posed_joints, _n) = skel.fk(&local_rot_mats, &root_positions)?;

    Ok(DecodedMotion {
        local_rot_mats,
        global_rot_mats,
        posed_joints,
        root_positions,
        foot_contacts,
        global_root_heading,
    })
}
