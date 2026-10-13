//! Stage B: [`AutoLatentTwostageDenoiser`] -- one denoising step.
//!
//! Port of `ardy_mlx/model/denoiser.py` (reference:
//! `ardy/model/auto_latent_twostage_denoiser.py` and `cfg.py`).
//!
//! The input and output are both the **hybrid token** of width 148:
//!
//! ```text
//! token = [ global root block, flattened over 4 frames (20) | body latent (128) ]
//! ```
//!
//! Stage 1 predicts the global root for every token; its output is converted
//! to a *local* root (which re-normalises through `local_root_stats` -- see
//! `motion_rep.global_root_to_local_root`) and concatenated with the body
//! latent to form the stage-2 input.
//!
//! Conditioning is by **prefix concatenation**, not cross-attention: the
//! backbone prepends `[text, timestep, first-heading-angle]` tokens.
//! Classifier-free guidance runs the *same* network three times batched
//! (`cfg_type: separated`):
//!
//! ```text
//! pass 0: real text,  zero constraints
//! pass 1: zero text,  real constraints
//! pass 2: zero text,  zero constraints
//! out = uncond + w_text * (text - uncond) + w_cstr * (cstr - uncond)
//! ```
//!
//! With no constraints at all (the batch path with no `--constraints`), the
//! `observed_motion` / `motion_mask` tensors are all zeros, so the constraint
//! projections are still *evaluated* -- they are simply fed zeros. Do not skip
//! them: they are part of the compute graph that produced the reference
//! numbers.

use mlx_rs::Array;
use mlx_rs::ops::{
    self,
    indexing::{Ellipsis, TryIndexOp},
};

use super::backbone::TransformerEncoderBlock;
use super::motion_rep::MotionRep;
use super::nn::{Linear, WeightStore};
use crate::{Error, Result};

/// One denoiser call's worth of window tensors.
///
/// Masks are bool (`true = valid`); the frame-level masks are `[B, nf]`, the
/// token-level masks `[B, nt]`, lengths `[B]`. `motion_mask` /
/// `observed_motion` are `[B, nf, motion_rep_dim]` when present -- the
/// reference substitutes zeros when absent.
pub struct Window {
    /// Hybrid tokens `[B, nt, 148]`.
    pub x: Array,
    /// History length in frames `[B]`.
    pub history_len: Array,
    /// Generation length in frames `[B]`.
    pub generation_len: Array,
    /// Future (horizon tail) length in frames `[B]`.
    pub future_len: Array,
    /// Frame-level history mask `[B, nf]` (bool, `true = valid`).
    pub history_mask: Array,
    /// Frame-level generation mask `[B, nf]` (bool, `true = valid`).
    pub generation_mask: Array,
    /// Frame-level future mask `[B, nf]` (bool, `true = valid`).
    pub future_mask: Array,
    /// Token-level history mask `[B, nt]` (bool, `true = valid`).
    pub history_token_mask: Array,
    /// Token-level generation mask `[B, nt]` (bool, `true = valid`).
    pub generation_token_mask: Array,
    /// Token-level future mask `[B, nt]` (bool, `true = valid`).
    pub future_token_mask: Array,
    /// Precomputed text embedding `[B, L, 4096]`.
    pub text_feat: Array,
    /// Integer diffusion step index `[B]`.
    pub timesteps: Array,
    /// First-frame global heading angle `[B]`; consumed by the backbone
    /// prefix when `input_first_heading_angle` is set (the Core checkpoint).
    pub first_heading_angle: Option<Array>,
    /// Constraint validity `[B, nf, motion_rep_dim]` (float 0/1); zeros when
    /// absent.
    pub motion_mask: Option<Array>,
    /// Constraint motion `[B, nf, motion_rep_dim]`; zeros when absent.
    pub observed_motion: Option<Array>,
}

impl Window {
    /// The reference's CFG batch-3 assembly, shared verbatim by
    /// `forward_batch3` and the cfg wrapper.
    ///
    /// ```text
    /// batch rows: [real text, no constraints | zero text, constraints | zero text, no constraints]
    /// ```
    ///
    /// The `text_feat_pad_mask` triple the reference also builds is ignored by
    /// the backbone (`use_text_mask == False`), so it is not materialized.
    pub fn tripled(&self) -> Result<Window> {
        fn triple(t: &Array) -> Result<Array> {
            Ok(ops::concatenate(&[t, t, t], 0)?)
        }
        fn zero_then_t(t: &Array) -> Result<Array> {
            Ok(ops::concatenate(
                &[&ops::zeros_like(t)?, t, &ops::zeros_like(t)?],
                0,
            )?)
        }
        // text branch order: row 0 keeps the prompt, rows 1-2 get zeros
        // (denoiser.py:316-321). Note this is *not* `zero_then_t`, which puts
        // the payload in the middle (constraints branch).
        fn text_then_zeros(t: &Array) -> Result<Array> {
            let z = ops::zeros_like(t)?;
            Ok(ops::concatenate(&[t, &z, &z], 0)?)
        }
        Ok(Window {
            x: triple(&self.x)?,
            history_len: triple(&self.history_len)?,
            generation_len: triple(&self.generation_len)?,
            future_len: triple(&self.future_len)?,
            history_mask: triple(&self.history_mask)?,
            generation_mask: triple(&self.generation_mask)?,
            future_mask: triple(&self.future_mask)?,
            history_token_mask: triple(&self.history_token_mask)?,
            generation_token_mask: triple(&self.generation_token_mask)?,
            future_token_mask: zero_then_t(&self.future_token_mask)?,
            text_feat: text_then_zeros(&self.text_feat)?,
            timesteps: triple(&self.timesteps)?,
            first_heading_angle: self.first_heading_angle.as_ref().map(triple).transpose()?,
            motion_mask: self.motion_mask.as_ref().map(zero_then_t).transpose()?,
            observed_motion: self.observed_motion.as_ref().map(zero_then_t).transpose()?,
        })
    }
}

/// `AutoLatentTwostageDenoiser` (`auto_latent_twostage_denoiser.py:40-400`).
pub struct AutoLatentTwostageDenoiser<'a> {
    motion_rep: &'a MotionRep,
    num_frames_per_token: usize,
    nframe_root_dim: usize,
    latent_embedding_dim: usize,
    nframe_local_root_dim: usize,

    global_root_hybrid_proj: Linear,
    global_root_hybrid_constraints_proj: Linear,
    local_root_hybrid_proj: Linear,
    local_root_hybrid_constraints_proj: Linear,
    future_constraints_proj: Linear,

    root_model: TransformerEncoderBlock,
    body_model: TransformerEncoderBlock,
}

impl<'a> AutoLatentTwostageDenoiser<'a> {
    /// Load the released-Core architecture from `prefix` (default
    /// `"denoiser"`): width 1024, 8 layers / 8 heads, 4 frames/token,
    /// root-flat 20, body latent 128, one text token, learned-prefix PE.
    pub fn new(store: &mut WeightStore, motion_rep: &'a MotionRep, prefix: &str) -> Result<Self> {
        let latent_dim = 1024;
        let num_frames_per_token = 4;
        let nframe_root_dim = 20;
        let latent_embedding_dim = 128;
        let num_layers = 8;
        let num_heads = 8;
        let num_text_tokens = 1;
        let input_first_heading_angle = true;
        let nframe_local_root_dim = motion_rep.local_root_dim() * num_frames_per_token;

        let global_root_hybrid_proj =
            Linear::new(store, &format!("{prefix}.global_root_hybrid_proj"), true)?;
        let global_root_hybrid_constraints_proj = Linear::new(
            store,
            &format!("{prefix}.global_root_hybrid_constraints_proj"),
            true,
        )?;
        let local_root_hybrid_proj =
            Linear::new(store, &format!("{prefix}.local_root_hybrid_proj"), true)?;
        let local_root_hybrid_constraints_proj = Linear::new(
            store,
            &format!("{prefix}.local_root_hybrid_constraints_proj"),
            true,
        )?;
        let future_constraints_proj =
            Linear::new(store, &format!("{prefix}.future_constraints_proj"), true)?;

        let root_model = TransformerEncoderBlock::new(
            store,
            &format!("{prefix}.backbone.root_model"),
            latent_dim,
            num_layers,
            num_heads,
            nframe_root_dim,
            num_text_tokens,
            input_first_heading_angle,
        )?;
        let body_model = TransformerEncoderBlock::new(
            store,
            &format!("{prefix}.backbone.body_model"),
            latent_dim,
            num_layers,
            num_heads,
            latent_embedding_dim,
            num_text_tokens,
            input_first_heading_angle,
        )?;

        // The projection widths encode the layout, so a mismatch means the
        // checkpoint is not the one this port was written for.
        let per_frame_ext = motion_rep.motion_rep_dim() + motion_rep.body_dim();
        let exp_gr = nframe_root_dim + latent_embedding_dim + per_frame_ext * num_frames_per_token;
        let exp_lr =
            nframe_local_root_dim + latent_embedding_dim + per_frame_ext * num_frames_per_token;
        let exp_fc = motion_rep.motion_rep_dim() * 2 * num_frames_per_token;
        let check = |l: &Linear, expected: usize, name: &str| -> Result<()> {
            let got = l.in_features;
            if got != expected {
                return Err(Error::Contract(format!(
                    "{name} expects {expected} inputs, checkpoint has {got}"
                )));
            }
            Ok(())
        };
        check(
            &global_root_hybrid_constraints_proj,
            exp_gr,
            "global_root_hybrid_constraints_proj",
        )?;
        check(
            &local_root_hybrid_constraints_proj,
            exp_lr,
            "local_root_hybrid_constraints_proj",
        )?;
        check(&future_constraints_proj, exp_fc, "future_constraints_proj")?;

        Ok(Self {
            motion_rep,
            num_frames_per_token,
            nframe_root_dim,
            latent_embedding_dim,
            nframe_local_root_dim,
            global_root_hybrid_proj,
            global_root_hybrid_constraints_proj,
            local_root_hybrid_proj,
            local_root_hybrid_constraints_proj,
            future_constraints_proj,
            root_model,
            body_model,
        })
    }

    /// `get_root_and_latent_body_motion_from_hybrid` (`latent_utils.py:67-76`).
    ///
    /// -> `(root [B, nf, root_block_dim], latent [B, nt, 128])`.
    pub fn split_hybrid(&self, hybrid: &Array) -> Result<(Array, Array)> {
        let shape = hybrid.shape();
        let (b, num_tokens, _) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let num_frames = num_tokens * self.num_frames_per_token;
        let root = hybrid
            .try_index((Ellipsis, 0..self.nframe_root_dim as i32))?
            .reshape(&[
                b as i32,
                num_frames as i32,
                self.motion_rep.root_block_dim() as i32,
            ])?;
        let latent = hybrid.try_index((Ellipsis, self.nframe_root_dim as i32..))?;
        Ok((root, latent))
    }

    /// `get_hybrid_motion_from_root_and_latent_body_motion`
    /// (`latent_utils.py:143-157`).
    pub fn join_hybrid(root_flat: &Array, latent: &Array) -> Result<Array> {
        Ok(ops::concatenate(&[root_flat, latent], -1)?)
    }

    /// `AutoLatentTwostageDenoiser.forward`
    /// (`auto_latent_twostage_denoiser.py:147-400`).
    pub fn forward(&self, w: &Window) -> Result<Array> {
        let shape = w.x.shape();
        let (bs, num_tokens, _) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let f = self.num_frames_per_token;
        let num_frames = num_tokens * f;
        let mr = self.motion_rep;

        let (global_root_motion, latent_body_motion) = self.split_hybrid(&w.x)?; // [B,T,5], [B,nt,128]

        let token_index = ops::broadcast_to(
            &ops::arange::<i32, i32>(None, num_tokens as i32, None)?,
            &[bs as i32, num_tokens as i32],
        )?;
        // positional_encoding_mode == "learned_prefix_zero_at_first_generation"
        let origin_index = ops::floor_divide(&w.history_len, scalar_i(f as i32))?;
        let token_index = ops::subtract(&token_index, &origin_index.expand_dims(1)?)?;

        // Constraint tensors; the reference substitutes zeros when absent.
        let (motion_mask, observed_motion) = match (&w.motion_mask, &w.observed_motion) {
            (Some(mm), Some(om)) => (mm.clone(), om.clone()),
            _ => (
                Array::zeros::<f32>(&[bs as i32, num_frames as i32, mr.motion_rep_dim() as i32])?,
                Array::zeros::<f32>(&[bs as i32, num_frames as i32, mr.motion_rep_dim() as i32])?,
            ),
        };

        let observed_root = mr.extract_root(&observed_motion)?;
        let observed_body = mr.extract_body(&observed_motion)?;
        let root_motion_mask = mr.extract_root(&motion_mask)?;
        let keep = ops::subtract(super::nn::scalar(1.0), &root_motion_mask)?;
        let global_root_motion = ops::add(
            &ops::multiply(&global_root_motion, &keep)?,
            &ops::multiply(&observed_root, &root_motion_mask)?,
        )?;

        let root_flat = global_root_motion.reshape(&[
            bs as i32,
            num_tokens as i32,
            self.nframe_root_dim as i32,
        ])?;
        let x_infilled_extended = ops::concatenate(
            &[
                &latent_body_motion,
                &root_flat,
                &observed_body.reshape(&[bs as i32, num_tokens as i32, -1])?,
                &motion_mask.reshape(&[bs as i32, num_tokens as i32, -1])?,
            ],
            -1,
        )?;

        let future_gate = w.future_token_mask.expand_dims(-1)?;
        let future_token_proj = ops::multiply(
            &self.future_constraints_proj.forward(
                &x_infilled_extended.try_index((Ellipsis, self.latent_embedding_dim as i32..))?,
            )?,
            &future_gate,
        )?;
        let gen_gate = w.generation_token_mask.expand_dims(-1)?;
        let hist_gate = w.history_token_mask.expand_dims(-1)?;
        let root_stage_input = ops::add(
            &ops::add(
                &ops::multiply(
                    &self
                        .global_root_hybrid_constraints_proj
                        .forward(&x_infilled_extended)?,
                    &gen_gate,
                )?,
                &ops::multiply(&self.global_root_hybrid_proj.forward(&w.x)?, &hist_gate)?,
            )?,
            &future_token_proj,
        )?;
        let root_stage_pad_mask = ops::logical_or(
            &ops::logical_or(&w.history_token_mask, &w.generation_token_mask)?,
            &w.future_token_mask,
        )?;

        let global_root_motion_pred = self.root_model.forward(
            &root_stage_input,
            &root_stage_pad_mask,
            &w.text_feat,
            &w.timesteps,
            w.first_heading_angle.as_ref(),
            Some(&token_index),
        )?; // [B, nt, 20]
        let global_root_motion_pred = global_root_motion_pred.reshape(&[
            bs as i32,
            num_frames as i32,
            mr.root_block_dim() as i32,
        ])?;
        let gen_frame_gate = w.generation_mask.expand_dims(-1)?;
        let hist_frame_gate = w.history_mask.expand_dims(-1)?;
        let global_root_motion_pred = ops::add(
            &ops::multiply(&global_root_motion_pred, &gen_frame_gate)?,
            &ops::multiply(&global_root_motion, &hist_frame_gate)?,
        )?;
        let lengths = ops::add(&w.history_len, &w.generation_len)?;

        // Stage 1 -> stage 2: convert the predicted global root to a local
        // root. `lengths` must be forwarded: it selects which frame gets the
        // repeated velocity (`tools.py:66-79`). On a single-window clip
        // `lengths` equals the tensor length and the distinction is invisible;
        // on window 0 of a 4.0 s clip it is 40 against an 80-frame tensor (the
        // trailing 40 frames are the zeroed *future* region), and ignoring it
        // corrupts the local-root condition for every generation token.
        let local_root_motion =
            mr.global_root_to_local_root(&global_root_motion_pred, true, Some(&lengths))?;
        let local_root_flat = local_root_motion.reshape(&[
            bs as i32,
            num_tokens as i32,
            self.nframe_local_root_dim as i32,
        ])?;
        let x_new = ops::concatenate(&[&local_root_flat, &latent_body_motion], -1)?; // 144

        let x_new_extended = ops::concatenate(
            &[
                &x_new,
                &observed_body.reshape(&[bs as i32, num_tokens as i32, -1])?,
                &motion_mask.reshape(&[bs as i32, num_tokens as i32, -1])?,
            ],
            -1,
        )?;
        let body_stage_input = ops::add(
            &ops::add(
                &ops::multiply(
                    &self
                        .local_root_hybrid_constraints_proj
                        .forward(&x_new_extended)?,
                    &gen_gate,
                )?,
                &ops::multiply(&self.local_root_hybrid_proj.forward(&x_new)?, &hist_gate)?,
            )?,
            &future_token_proj,
        )?;
        let body_stage_pad_mask = ops::logical_or(
            &ops::logical_or(&w.history_token_mask, &w.generation_token_mask)?,
            &w.future_token_mask,
        )?;

        let latent_body_motion_pred = self.body_model.forward(
            &body_stage_input,
            &body_stage_pad_mask,
            &w.text_feat,
            &w.timesteps,
            w.first_heading_angle.as_ref(),
            Some(&token_index),
        )?;
        let latent_body_motion_pred = ops::add(
            &ops::multiply(&latent_body_motion_pred, &gen_gate)?,
            &ops::multiply(&latent_body_motion, &hist_gate)?,
        )?;

        Ok(ops::concatenate(
            &[
                &global_root_motion_pred.reshape(&[
                    bs as i32,
                    num_tokens as i32,
                    self.nframe_root_dim as i32,
                ])?,
                &latent_body_motion_pred,
            ],
            -1,
        )?)
    }
}

/// `AutoLatentClassifierFreeGuidedModelSeparated` (`cfg.py:22-149`).
///
/// Free function rather than a wrapper struct: the reference wrapper holds no
/// state beyond the model.
pub fn cfg_guided(
    model: &AutoLatentTwostageDenoiser,
    cfg_weight_text: f32,
    cfg_weight_cstr: f32,
    w: &Window,
) -> Result<Array> {
    let out_3 = model.forward(&w.tripled()?)?;
    let parts = out_3.split_equal(3, 0)?;
    let (out_text, out_cstr, out_uncond) = (&parts[0], &parts[1], &parts[2]);
    Ok(ops::add(
        &ops::add(
            out_uncond,
            &ops::multiply(
                &ops::subtract(out_text, out_uncond)?,
                super::nn::scalar(cfg_weight_text),
            )?,
        )?,
        &ops::multiply(
            &ops::subtract(out_cstr, out_uncond)?,
            super::nn::scalar(cfg_weight_cstr),
        )?,
    )?)
}

/// Integer scalar helper.
fn scalar_i(v: i32) -> Array {
    Array::from_slice(&[v], &[])
}
