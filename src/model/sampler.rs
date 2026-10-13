//! The autoregressive sampling chain: window assembly, the per-window DDIM
//! loop, and the inter-window recenter / requantize / heading refresh.
//!
//! Port of the no-constraint batch path in `ardy_mlx/model/ardy.py`
//! (`_generate_window` / `generate`; reference:
//! `ardy/model/ardy_model.py:415-695`). The B-side `sampling/` package is
//! deliberately **not** ported: its window geometry differs from the golden
//! reference (see the M4 notes in the repo history).
//!
//! Frame/token conventions (Core checkpoint): one token = 4 frames, a window
//! generates 40 frames = 10 tokens, and `total_frames_window` is a multiple of
//! 4. The frame masks are f32 (`1.0 = valid`) exactly as the reference
//! multiplies by them; the denoiser's token masks are bool.
//!
//! Two indexing spaces stay distinct inside the window loop (`ardy.py:338-342`):
//! the **loop index** `i` (which runs `num_steps-1 ..= 0` and indexes the
//! sampler buffers) and the **mapped timestep** `map_tensor[i]` (which the
//! denoiser embeds). They coincide for the released Core checkpoint only
//! because `num_steps == num_base_steps == 10`.

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::TryIndexOp;

use super::denoiser::{AutoLatentTwostageDenoiser, split_hybrid_free};
use super::diffusion::{Diffusion, denoising_step};
use super::fsq::FsqVaeTransformer;
use super::motion_rep::MotionRep;
use crate::{Error, Result};

/// `timestep_indices` (`ardy.py:79-88`): `range(num_steps)[::-1]`.
pub fn timestep_indices(num_steps: usize) -> Result<Vec<usize>> {
    if num_steps < 1 {
        return Err(Error::Contract(format!(
            "num_steps must be >= 1, got {num_steps}"
        )));
    }
    Ok((0..num_steps).rev().collect())
}

/// `get_three_mask_from_len` (`ardy_model.py:15-26`): f32 `[B, total_len]`
/// history / generation / future masks from frame lengths.
///
/// `future_len` is kept in the signature for parity with the reference (the
/// generation end already encodes everything the f32 masks need).
pub fn get_three_mask_from_len(
    history_len: &Array,
    generation_len: &Array,
    future_len: &Array,
    total_len: usize,
) -> Result<(Array, Array, Array)> {
    let _ = future_len;
    let total_len = total_len as i32;
    let idx = ops::arange::<f32, f32>(None, total_len as f32, None)?; // [total_len]
    let idx = idx.expand_dims(0)?; // [1, total_len]
    let history_end = history_len.expand_dims(1)?;
    let generation_end = ops::add(history_len, generation_len)?.expand_dims(1)?;
    let f32_of = |b: Array| b.as_dtype(mlx_rs::Dtype::Float32);
    let history_mask = f32_of(ops::lt(&idx, &history_end)?)?;
    let generation_mask = f32_of(ops::logical_and(
        &ops::ge(&idx, &history_end)?,
        &ops::lt(&idx, &generation_end)?,
    )?)?;
    let future_mask = f32_of(ops::ge(&idx, &generation_end)?)?;
    Ok((history_mask, generation_mask, future_mask))
}

/// `convert_frame_mask_to_token_mask` (`latent_utils.py:192-220`).
///
/// Token masks are bool (`all()` over the `num_frames_per_token` frames of each
/// token). **The trap this function exists to pin** (`latent_utils.py:207`):
/// when `motion_mask` is `None` -- the whole no-constraint batch path --
/// `future_token_mask` is `zeros_like(history_token_mask)`, *not* `future_mask`
/// reduced to tokens. Window 0 of an 80-frame clip therefore has
/// `future_mask.sum() == 40` frames but `future_token_mask.sum() == 0` tokens;
/// the token mask feeds the denoiser's attention padding, so getting this wrong
/// silently changes window 0.
pub fn convert_frame_mask_to_token_mask(
    history_mask: &Array,
    generation_mask: &Array,
    future_mask: &Array,
    motion_mask: Option<&Array>,
    num_frames_per_token: usize,
) -> Result<(Array, Array, Array)> {
    let fpt = num_frames_per_token as i32;
    let bs = history_mask.shape()[0];
    let nf = history_mask.shape()[1];
    let nt = nf / fpt;
    let to_tokens = |m: &Array| m.reshape(&[bs, nt, fpt])?.all_axis(-1, false);
    let history_token_mask = to_tokens(history_mask)?;
    let generation_token_mask = to_tokens(generation_mask)?;
    let future_token_mask = match motion_mask {
        None => ops::zeros_like(&history_token_mask)?,
        Some(mm) => {
            let base = to_tokens(future_mask)?;
            let has_observation = mm
                .any_axis(-1, false)?
                .reshape(&[bs, nt, fpt])?
                .any_axis(-1, false)?;
            ops::logical_and(&base, &has_observation)?
        }
    };
    Ok((history_token_mask, generation_token_mask, future_token_mask))
}

/// `assemble_window_latent` (`ardy_model.py:481-490`): the window tensor is
/// `zeros(total_tokens)` with the *clean* history written into the first
/// `history_token_end` tokens and the fresh noise `x_t` into the generation
/// span. Tokens after the generation span (masked future) stay zero. Adding to
/// zeros is the payload, so the assembly is one concatenation.
pub fn assemble_window_latent(
    history_hybrid: Option<&Array>,
    x_t: &Array,
    history_token_end: usize,
    generation_token_end: usize,
    total_tokens: usize,
) -> Result<Array> {
    let bs = x_t.shape()[0];
    let d = x_t.shape()[2];
    let after_w = (total_tokens - generation_token_end) as i32;
    if history_token_end > 0 {
        let h = history_hybrid.ok_or_else(|| {
            Error::Contract("history_token_end > 0 but no history_hybrid was supplied".to_string())
        })?;
        let tail = h.try_index((
            (0i32..),
            ((h.shape()[1] - history_token_end as i32)..),
            (0i32..),
        ))?;
        return if after_w > 0 {
            Ok(ops::concatenate(
                &[&tail, x_t, &Array::zeros::<f32>(&[bs, after_w, d])?],
                1,
            )?)
        } else {
            Ok(ops::concatenate(&[&tail, x_t], 1)?)
        };
    }
    if after_w > 0 {
        Ok(ops::concatenate(
            &[x_t, &Array::zeros::<f32>(&[bs, after_w, d])?],
            1,
        )?)
    } else {
        Ok(x_t.clone())
    }
}

/// One window of the autoregressive loop (`ArdyBatch._generate_window`,
/// `ardy.py:244-371`).
///
/// Runs the `num_steps` DDIM iterations and returns the updated sequence: for
/// the first window (`history_hybrid == None`) the generation tokens only,
/// otherwise the incoming history followed by the new generation tokens. The
/// window's own token coordinates are `[0, total_frames - history_start_frame)`;
/// `future_len` spans whatever remains of `total_frames`.
///
/// `on_step`, when supplied, observes `(loop_index i, x after the update)` for
/// each iteration -- the seam the reference fills with `trace["steps"]`.
#[allow(clippy::too_many_arguments)]
pub fn generate_window<S: FnMut(usize, &Array)>(
    denoiser: &AutoLatentTwostageDenoiser,
    diffusion: &Diffusion,
    x_t: &Array,
    history_hybrid: Option<&Array>,
    text_feat: &Array,
    heading: &Array,
    total_frames: usize,
    history_start_frame: usize,
    history_end_frame: usize,
    gen_horizon_len: usize,
    num_frames_per_token: usize,
    num_steps: usize,
    cfg_weight: (f32, f32),
    mut on_step: Option<S>,
) -> Result<Array> {
    let f = num_frames_per_token;
    let bs = text_feat.shape()[0];
    let nt = gen_horizon_len / f;
    let history_token_end = (history_end_frame - history_start_frame) / f;
    let generation_token_end = history_token_end + nt;
    let total_tokens = (total_frames - history_start_frame) / f;

    // lengths are f32 in the reference (`mx.full(..., dtype=mx.float32)`)
    let hist_len = Array::from_slice(
        &vec![history_token_end as f32 * f as f32; bs as usize],
        &[bs],
    );
    let gen_len = Array::from_slice(&vec![gen_horizon_len as f32; bs as usize], &[bs]);
    let future_len_val = total_frames as f32 - gen_horizon_len as f32 - history_end_frame as f32;
    let fut_len = Array::from_slice(&vec![future_len_val; bs as usize], &[bs]);

    let (history_mask, generation_mask, future_mask) = get_three_mask_from_len(
        &hist_len,
        &gen_len,
        &fut_len,
        total_frames - history_start_frame,
    )?;
    // no constraints: future_token_mask stays zeros (the latent_utils.py:207 trap)
    let (history_token_mask, generation_token_mask, future_token_mask) =
        convert_frame_mask_to_token_mask(&history_mask, &generation_mask, &future_mask, None, f)?;

    let mut x = assemble_window_latent(
        history_hybrid,
        x_t,
        history_token_end,
        generation_token_end,
        total_tokens,
    )?;

    for i in timestep_indices(num_steps)? {
        // the *mapped* timestep feeds the denoiser's embedder; i indexes the
        // sampler buffers inside denoising_step
        let t0 = diffusion.map_tensor.try_index((i as i32,))?;
        let timesteps = ops::broadcast_to(&t0, &[bs])?;
        let w = super::denoiser::Window {
            x: x.clone(),
            history_len: hist_len.clone(),
            generation_len: gen_len.clone(),
            future_len: fut_len.clone(),
            history_mask: history_mask.clone(),
            generation_mask: generation_mask.clone(),
            future_mask: future_mask.clone(),
            history_token_mask: history_token_mask.clone(),
            generation_token_mask: generation_token_mask.clone(),
            future_token_mask: future_token_mask.clone(),
            text_feat: text_feat.clone(),
            timesteps,
            first_heading_angle: Some(heading.clone()),
            motion_mask: None,
            observed_motion: None,
        };
        x = denoising_step(
            denoiser,
            diffusion,
            &w,
            history_token_end,
            generation_token_end,
            i,
            cfg_weight.0,
            cfg_weight.1,
        )?;
        if let Some(cb) = on_step.as_mut() {
            cb(i, &x);
        }
    }

    match history_hybrid {
        None => Ok(x.try_index(((0i32..), (0i32..generation_token_end as i32), (0i32..)))?),
        Some(h) => Ok(ops::concatenate(
            &[
                h,
                &x.try_index((
                    (0i32..),
                    (history_token_end as i32..generation_token_end as i32),
                    (0i32..),
                ))?,
            ],
            1,
        )?),
    }
}

/// The inter-window update (`ArdyBatch.generate`, `ardy.py:478-513`): recenter
/// the accumulated hybrid on its last generated frame, requantize the body
/// latent, re-join, and refresh the heading from the recentered root.
///
/// `history_end_frame` is this window's end; the recentering center is
/// `history_end_frame + gen_horizon_len - 1` clamped to the sequence.
pub struct RecenterOutput {
    /// Requantized, recentered history to feed the next window.
    pub history: Array,
    /// Center position `[B, 3]` to fold into the accumulated `global_transl`.
    pub center_pos: Array,
    /// Refreshed `first_heading_angle` `[B]` (a scalar frame index into the
    /// cropped sequence); `None` when refreshing is off or no crop is set.
    pub heading: Option<Array>,
}

/// Per-step observer of [`generate`]: `(window index, loop index, x after the
/// DDIM update)`.
pub type StepHook<'a> = &'a mut dyn FnMut(usize, usize, &Array);

/// See [`RecenterOutput`]; the reference block this ports is
/// `ArdyBatch.generate` (`ardy.py:494-513`).
#[allow(clippy::too_many_arguments)]
pub fn recenter_step(
    history: &Array,
    motion_rep: &MotionRep,
    tokenizer: &FsqVaeTransformer,
    num_frames_per_token: usize,
    nframe_root_dim: usize,
    history_end_frame: usize,
    gen_horizon_len: usize,
    crop_history_length: Option<usize>,
    refresh_first_heading: bool,
) -> Result<RecenterOutput> {
    let bs = history.shape()[0];
    let (root_motion, latent) = split_hybrid_free(
        history,
        num_frames_per_token,
        nframe_root_dim,
        motion_rep.root_block_dim(),
    )?;
    let center_frame_index =
        (history_end_frame + gen_horizon_len - 1).min(root_motion.shape()[1] as usize - 1);
    let (new_root_motion, center_pos) = motion_rep.recenter_root_motion_with_center(
        &root_motion,
        center_frame_index,
        true,
        true,
    )?;
    let latent = tokenizer.requantize(&latent)?;
    let history = AutoLatentTwostageDenoiser::join_hybrid(
        &new_root_motion.reshape(&[bs, -1, nframe_root_dim as i32])?,
        &latent,
    )?;

    let heading = match (crop_history_length, refresh_first_heading) {
        (Some(crop), true) => {
            let (_, global_root_stats, _, _) = motion_rep.stats();
            let un = global_root_stats.unnormalize(&new_root_motion)?;
            let heading = motion_rep.get_root_heading_angle(&un)?;
            let cropped_start = (new_root_motion.shape()[1] as usize).saturating_sub(crop);
            // scalar frame index -> [B]
            Some(heading.try_index(((0i32..), cropped_start as i32))?)
        }
        _ => None,
    };
    Ok(RecenterOutput {
        history,
        center_pos,
        heading,
    })
}

/// The full no-constraint autoregressive generate (`ArdyBatch.generate`,
/// `ardy.py:395-560`): windows of pure noise denoised in sequence, each
/// followed by the recenter/requantize/heading step, then the accumulated
/// translation folded back into the root.
///
/// `noise` supplies each window's deterministic `x_T` (`[B, nt, 148]`) -- MLX
/// cannot reproduce the reference CUDA RNG, and the golden run seeds one noise
/// tensor per window. `on_step` mirrors [`generate_window`]'s hook with
/// `(window, loop_index, x)`.
///
/// Returns the translated hybrid sequence (the M4 endpoint,
/// `final_hybrid_translated`); decoding to motion is the M5 stage.
///
/// [`StepHook`] is the `(window, loop_index, x after the update)` observer.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    denoiser: &AutoLatentTwostageDenoiser,
    tokenizer: &FsqVaeTransformer,
    motion_rep: &MotionRep,
    diffusion: &Diffusion,
    text_feat: &Array,
    num_frames: usize,
    num_steps: usize,
    cfg_weight: (f32, f32),
    first_heading_angle: Option<Array>,
    gen_horizon_len: usize,
    crop_history_length: Option<usize>,
    noise: &dyn Fn(usize) -> Result<Array>,
    mut on_step: Option<StepHook<'_>>,
) -> Result<GenerateOutput> {
    let f = denoiser.num_frames_per_token();
    let nframe_root_dim = denoiser.nframe_root_dim();
    let bs = text_feat.shape()[0];
    let heading = match first_heading_angle {
        Some(h) => h,
        None => Array::zeros::<f32>(&[bs])?,
    };

    let num_auto_steps = num_frames.div_ceil(gen_horizon_len);
    let num_gen_frames = num_auto_steps * gen_horizon_len;

    let mut history: Option<Array> = None;
    let mut global_transl = Array::zeros::<f32>(&[bs, 3])?;
    let mut heading = heading;
    let mut offsets = vec![0usize];

    for auto_step in 0..num_auto_steps {
        let history_end_frame = auto_step * gen_horizon_len;
        let history_start_frame = match crop_history_length {
            None => 0,
            Some(crop) => history_end_frame.saturating_sub(crop),
        };
        let x_t = noise(auto_step)?;

        let window_index = auto_step;
        let step_adapter = on_step
            .as_deref_mut()
            .map(|cb| move |i: usize, x: &Array| cb(window_index, i, x));
        history = Some(generate_window(
            denoiser,
            diffusion,
            &x_t,
            history.as_ref(),
            text_feat,
            &heading,
            num_gen_frames,
            history_start_frame,
            history_end_frame,
            gen_horizon_len,
            f,
            num_steps,
            cfg_weight,
            step_adapter,
        )?);
        offsets.push(history.as_ref().unwrap().shape()[1] as usize * f);

        let rec = recenter_step(
            history.as_ref().unwrap(),
            motion_rep,
            tokenizer,
            f,
            nframe_root_dim,
            history_end_frame,
            gen_horizon_len,
            crop_history_length,
            true,
        )?;
        global_transl = ops::add(&global_transl, &rec.center_pos)?;
        if crop_history_length.is_some()
            && let Some(h) = rec.heading
        {
            heading = h;
        }
        history = Some(rec.history);
    }

    // fold the accumulated translation back into the root
    let history = history.unwrap();
    let (root_motion, latent) =
        split_hybrid_free(&history, f, nframe_root_dim, motion_rep.root_block_dim())?;
    let new_root_motion =
        motion_rep.translate_normalized_root_motion(&root_motion, &global_transl)?;
    let hybrid_translated = AutoLatentTwostageDenoiser::join_hybrid(
        &new_root_motion.reshape(&[bs, -1, nframe_root_dim as i32])?,
        &latent,
    )?;

    Ok(GenerateOutput {
        hybrid_translated,
        global_transl,
        offsets,
    })
}

/// Output of [`generate`].
pub struct GenerateOutput {
    /// The accumulated hybrid sequence with `global_transl` folded into the
    /// root (the reference's `hybrid_translated`, `[B, T, 148]`).
    pub hybrid_translated: Array,
    /// Accumulated recentering translation `[B, 3]`.
    pub global_transl: Array,
    /// Cumulative token-count -> frame offsets after each window
    /// (`offsets[0] == 0`).
    pub offsets: Vec<usize>,
}
