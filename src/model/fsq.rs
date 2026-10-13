//! Stage A: [`FsqVaeTransformer`] -- the motion tokenizer (encode/decode).
//!
//! Port of `ardy_mlx/model/fsq.py` (reference: `ardy/model/autoencoder/fsq.py`
//! and `transformer.py`). Both sub-transformers are **causal** for the released
//! Core checkpoint (`causal_encoder: true`, `causal_decoder: true`), so the
//! attention mask is causal *and* padding-masked.
//!
//! Two details that silently break alignment if guessed:
//!
//! 1. `latent_embedding_dim` is **overwritten** by `num_fsq_levels` upstream --
//!    both are 128 here, but the argument is not what is used.
//! 2. The FSQ quantiser is **not** a plain `round(z*32)/32`. `FSQ.bound`
//!    first squeezes `z` through `tanh` with a shift derived from `levels`
//!    and `eps = 1e-3`:
//!
//!    ```text
//!    half_l = (levels - 1) * (1 + eps) / 2
//!    offset = 0.5                       # levels is even
//!    shift  = atanh(offset / half_l)
//!    zhat   = round(tanh(z + shift) * half_l - offset) / (levels / 2)
//!    ```
//!
//!    [`detokenize`](FsqVaeTransformer::detokenize) then re-quantises
//!    differently -- it assumes the value is already bounded and does
//!    `round(clamp(x, -1, 1) * 32) / 32` ([`re_round`]).

use mlx_rs::Array;
use mlx_rs::ops::{self, indexing::Ellipsis, indexing::TryIndexOp};

use super::motion_rep::{MotionRep, Stats};
use super::nn::{
    Linear, PositionalEncoding, TransformerEncoder, WeightStore, build_additive_mask, scalar,
};
use crate::Result;

/// `FSQ(levels=[64]*128)` and `FSQ.bound`'s default eps.
pub const FSQ_LEVELS: usize = 64;
/// Width of one quantized token (`levels = [64] * 128`).
pub const FSQ_TOKEN_DIM: usize = 128;
/// `FSQ.bound`'s default eps.
pub const FSQ_EPS: f64 = 1e-3;

/// `(shift, half_l)` from `FSQ.bound`.
///
/// Computed in `f64` to match the Python reference (`math.atanh` on Python
/// floats); the results are narrowed to `f32` only at the point of use, as the
/// scalars enter MLX ops.
pub fn fsq_shift(levels: usize, eps: f64) -> (f64, f64) {
    let half_l = (levels - 1) as f64 * (1.0 + eps) / 2.0;
    let offset = if levels.is_multiple_of(2) { 0.5 } else { 0.0 };
    let shift = (offset / half_l).atanh();
    (shift, half_l)
}

/// `FSQ.bound` at eval.
pub fn fsq_bound(z: &Array, levels: usize, eps: f64) -> Result<Array> {
    let (shift, half_l) = fsq_shift(levels, eps);
    let offset = if levels.is_multiple_of(2) {
        0.5f32
    } else {
        0.0
    };
    let bounded = ops::tanh(&ops::add(z, scalar(shift as f32))?)?;
    let bounded = ops::subtract(
        &ops::multiply(&bounded, scalar(half_l as f32))?,
        scalar(offset),
    )?;
    Ok(ops::divide(
        &ops::round(&bounded, None)?,
        scalar((levels / 2) as f32),
    )?)
}

/// The inline re-rounding step inside `detokenize`.
///
/// Assumes `z` is already *un-normalised*; `round(clamp(z, -1, 1) * 32) / 32`.
/// Note this is **not** the same as [`FsqVaeTransformer::requantize`], which
/// wraps this step in the `post_quantization` stats.
pub fn re_round(z: &Array, levels: usize) -> Result<Array> {
    let half_width = (levels / 2) as f32;
    let clipped = ops::clip(z, (-1.0f32, 1.0f32))?;
    Ok(ops::divide(
        &ops::round(&ops::multiply(&clipped, scalar(half_width))?, None)?,
        scalar(half_width),
    )?)
}

/// `torch.nn.Transformer.generate_square_subsequent_mask` with `dtype=bool`.
///
/// `true` means "not allowed to attend" (the diagonal and below are `false`).
pub fn causal_mask(seq_len: usize) -> Array {
    let n = seq_len as i32;
    let mut v = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            v[i * seq_len + j] = j > i;
        }
    }
    Array::from_slice::<bool>(&v, &[n, n])
}

/// Motion tokenizer: 4 frames <-> 1 latent token.
///
/// Encode: `body(325) x 4` -> `input_proj(512)` -> 8 causal layers ->
/// `output_proj(128)` -> FSQ -> `post_quantization` normalisation.
/// Decode: unnormalise -> re-round -> `input_proj(128->512)` -> concat local
/// root (`4 x 4 = 16`) -> `external_cond_blocks(528->512)` -> PE -> 8 causal
/// layers -> `output_proj(512->1316)` -> `(root 4 | body 325)` per frame.
pub struct FsqVaeTransformer<'a> {
    motion_rep: &'a MotionRep,
    num_frames_per_token: usize,
    causal_encoder: bool,
    causal_decoder: bool,

    enc_input_proj: Linear,
    enc_layers: TransformerEncoder,
    enc_output_proj: Linear,

    dec_input_proj: Linear,
    dec_external_cond_blocks: Linear,
    dec_layers: TransformerEncoder,
    dec_output_proj: Linear,

    /// The reference passes no `target_cond` on this path, so
    /// `target_cond_blocks` is intentionally not wired up. Flagged loudly if
    /// someone later needs it: the weights are present but unused here.
    has_target_cond: bool,

    /// Shared PE tables (identical construction in both transformers).
    pe: PositionalEncoding,
    post_quantization_stats: Option<Stats>,
    pre_quantization_stats: Option<Stats>,
}

impl<'a> FsqVaeTransformer<'a> {
    /// Load the released-Core architecture from `prefix` (default
    /// `"pose_net"`): 4 frames/token, width 512, 8 layers, 4 heads,
    /// non-pre-norm (`TransformerEncoder`'s Python default), causal both ways.
    pub fn new(store: &mut WeightStore, motion_rep: &'a MotionRep, prefix: &str) -> Result<Self> {
        let num_frames_per_token = 4;
        let width = 512;
        let num_layers = 8;
        let num_heads = 4;
        let norm_first = false;

        // --- encoder -----------------------------------------------------
        let enc_input_proj = Linear::new(store, &format!("{prefix}.encoder.input_proj"), true)?;
        let enc_layers = TransformerEncoder::new(
            store,
            &format!("{prefix}.encoder.seqTransEncoder"),
            num_layers,
            width,
            num_heads,
            norm_first,
        )?;
        let enc_output_proj = Linear::new(store, &format!("{prefix}.encoder.output_proj"), true)?;

        // --- decoder -----------------------------------------------------
        let dec_input_proj = Linear::new(store, &format!("{prefix}.decoder.input_proj"), true)?;
        let dec_external_cond_blocks = Linear::new(
            store,
            &format!("{prefix}.decoder.external_cond_blocks.0"),
            true,
        )?;
        let dec_layers = TransformerEncoder::new(
            store,
            &format!("{prefix}.decoder.seqTransEncoder"),
            num_layers,
            width,
            num_heads,
            norm_first,
        )?;
        let dec_output_proj = Linear::new(store, &format!("{prefix}.decoder.output_proj"), true)?;

        let has_target_cond = store.has(&format!("{prefix}.decoder.target_cond_blocks.0.weight"));

        Ok(Self {
            motion_rep,
            num_frames_per_token,
            causal_encoder: true,
            causal_decoder: true,
            enc_input_proj,
            enc_layers,
            enc_output_proj,
            dec_input_proj,
            dec_external_cond_blocks,
            dec_layers,
            dec_output_proj,
            has_target_cond,
            pe: PositionalEncoding::new(width, 5000),
            post_quantization_stats: None,
            pre_quantization_stats: None,
        })
    }

    /// Width of a quantized token embedding (`FSQ_TOKEN_DIM`, 128).
    pub fn latent_dim(&self) -> usize {
        FSQ_TOKEN_DIM
    }

    /// Frames folded into one token (4).
    pub fn num_frames_per_token(&self) -> usize {
        self.num_frames_per_token
    }

    /// `true` when the checkpoint carries decoder `target_cond_blocks`
    /// weights; they are present but intentionally unwired (no `target_cond`
    /// on the reference inference path).
    pub fn has_target_cond(&self) -> bool {
        self.has_target_cond
    }

    /// Attach `post_quantization` + `pre_quantization` stats.
    pub fn load_stats(&mut self, post: Stats, pre: Stats) {
        self.post_quantization_stats = Some(post);
        self.pre_quantization_stats = Some(pre);
    }

    fn stats(&self) -> Result<&Stats> {
        self.post_quantization_stats.as_ref().ok_or_else(|| {
            crate::Error::Contract(
                "post_quantization stats not loaded; call load_stats()".to_string(),
            )
        })
    }

    /// Encoder transformer only: body `[B,T,325]` -> `[B,T/4,128]`.
    pub fn encode_body(&self, body: &Array, motion_pad_mask: Option<&Array>) -> Result<Array> {
        let shape = body.shape();
        let (b, t, d) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let f = self.num_frames_per_token;
        // rearrange(x, "b (t f) d -> b t (f d)")
        let x = body.reshape(&[b as i32, (t / f) as i32, (f * d) as i32])?;
        let x = self.enc_input_proj.forward(&x)?;
        let x = self.pe.forward(&x)?;
        let attn_mask = if self.causal_encoder {
            Some(causal_mask(x.shape()[1] as usize))
        } else {
            None
        };
        let kpm = match motion_pad_mask {
            Some(m) => {
                let token_pad = m
                    .reshape(&[b as i32, (t / f) as i32, f as i32])?
                    .all_axis(-1, false)?;
                Some(token_pad.logical_not()?)
            }
            None => None,
        };
        let mask = build_additive_mask(kpm.as_ref(), attn_mask.as_ref())?;
        let x = self.enc_layers.forward(&x, mask.as_ref())?;
        self.enc_output_proj.forward(&x)
    }

    /// `FSQVAETransformer.tokenize`.
    ///
    /// `features` are **normalised** 330-dim features; only the body block is
    /// used (`encoder_input_feature_mode == "body"`).
    pub fn tokenize(&self, features: &Array, motion_pad_mask: &Array) -> Result<Array> {
        let body = self.motion_rep.extract_body(features)?;
        let x_encoder = self.encode_body(&body, Some(motion_pad_mask))?;
        let token_embeddings = fsq_bound(&x_encoder, FSQ_LEVELS, FSQ_EPS)?;
        self.stats()?.normalize(&token_embeddings)
    }

    /// Decoder transformer only: `[B,T/4,128]` + local root -> `[B,T,329]`.
    pub fn decode_latent(
        &self,
        token_embeddings: &Array,
        external_cond: &Array,
        motion_pad_mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = token_embeddings.shape();
        let (b, num_tokens, _) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let f = self.num_frames_per_token;
        let num_frames = num_tokens * f;

        let h = self.dec_input_proj.forward(token_embeddings)?;
        let ext = external_cond.reshape(&[b as i32, num_tokens as i32, -1])?;
        let h = ops::concatenate(&[&h, &ext], -1)?;
        let h = ops::maximum(&self.dec_external_cond_blocks.forward(&h)?, scalar(0.0))?;
        let h = self.pe.forward(&h)?;

        let attn_mask = if self.causal_decoder {
            Some(causal_mask(num_tokens))
        } else {
            None
        };
        let kpm = match motion_pad_mask {
            Some(m) => {
                let token_pad = m
                    .reshape(&[b as i32, num_tokens as i32, f as i32])?
                    .all_axis(-1, false)?;
                Some(token_pad.logical_not()?)
            }
            None => None,
        };
        let mask = build_additive_mask(kpm.as_ref(), attn_mask.as_ref())?;
        let h = self.dec_layers.forward(&h, mask.as_ref())?;
        let h = self.dec_output_proj.forward(&h)?;
        Ok(h.reshape(&[b as i32, num_frames as i32, -1])?)
    }

    /// `FSQVAETransformer.detokenize` -> `(root, body)`.
    pub fn detokenize(
        &self,
        token_embeddings: &Array,
        external_cond: &Array,
        motion_pad_mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let stats = self.stats()?;
        let t = stats.unnormalize(token_embeddings)?;
        let t = re_round(&t, FSQ_LEVELS)?;
        let x_decoder = self.decode_latent(&t, external_cond, motion_pad_mask)?;
        let lr = self.motion_rep.local_root_dim() as i32;
        Ok((
            x_decoder.try_index((Ellipsis, 0..lr))?,
            x_decoder.try_index((Ellipsis, lr..))?,
        ))
    }

    /// `FSQVAETransformer.requantize`.
    ///
    /// Note this *is* stats-wrapped (un-normalise -> re-round -> normalise),
    /// unlike the inline step inside [`Self::detokenize`].
    pub fn requantize(&self, token_embeddings: &Array) -> Result<Array> {
        let stats = self.stats()?;
        stats.normalize(&re_round(
            &stats.unnormalize(token_embeddings)?,
            FSQ_LEVELS,
        )?)
    }

    /// The reference rebinds `forward = detokenize`.
    pub fn forward(
        &self,
        token_embeddings: &Array,
        external_cond: &Array,
        motion_pad_mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        self.detokenize(token_embeddings, external_cond, motion_pad_mask)
    }

    /// Encode -> decode on the same features; used by the alignment harness.
    pub fn roundtrip(
        &self,
        features: &Array,
        motion_pad_mask: &Array,
    ) -> Result<(Array, Array, Array)> {
        let latent = self.tokenize(features, motion_pad_mask)?;
        let local_root = self.motion_rep.global_root_to_local_root(
            &self.motion_rep.extract_root(features)?,
            true,
            None,
        )?;
        let (root, body) = self.detokenize(&latent, &local_root, Some(motion_pad_mask))?;
        Ok((latent, root, body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Materialize a (possibly strided) view as a flat f32 vector.
    fn flat(a: &Array) -> Vec<f32> {
        let f = a.reshape(&[-1]).unwrap();
        f.eval().unwrap();
        f.as_slice::<f32>().to_vec()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "{ctx}[{i}]: {x} vs {y} (tol {tol})");
        }
    }

    #[test]
    fn fsq_shift_core_values() {
        // hand-computed: half_l = 63 * 1.001 / 2, shift = atanh(0.5 / half_l)
        let (shift, half_l) = fsq_shift(FSQ_LEVELS, FSQ_EPS);
        assert!((half_l - 31.5315).abs() < 1e-9, "half_l = {half_l}");
        assert!((shift - 0.015_858_488).abs() < 1e-8, "shift = {shift}");
        // odd level counts use offset 0
        let (shift_odd, half_l_odd) = fsq_shift(63, FSQ_EPS);
        assert_eq!(half_l_odd, 62.0_f64 * (1.0 + FSQ_EPS) / 2.0);
        assert_eq!(shift_odd, 0.0);
    }

    #[test]
    fn fsq_bound_matches_hand_computed_grid() {
        // tanh(1.0158585)*31.5315 - 0.5 = 23.72 -> round 24 -> 24/32
        // symmetric for -1; tanh(0.3159)*31.5315 - 0.5 = 9.14 -> 9/32
        // tanh(5.0159)*31.5315 - 0.5 = 31.03 -> 31/32
        let z = Array::from_slice(&[1.0f32, -1.0, 0.3, 5.0], &[1, 1, 4]);
        let out = fsq_bound(&z, FSQ_LEVELS, FSQ_EPS).unwrap();
        assert_close(
            &flat(&out),
            &[0.75, -0.75, 0.28125, 0.96875],
            1e-5,
            "fsq_bound",
        );
    }

    #[test]
    fn re_round_clips_and_quantizes() {
        let z = Array::from_slice(&[0.031f32, -0.049, 0.5, 1.0, -2.0, 0.0], &[2, 3]);
        let out = re_round(&z, FSQ_LEVELS).unwrap();
        assert_close(
            &flat(&out),
            &[0.03125, -0.0625, 0.5, 1.0, -1.0, 0.0],
            1e-6,
            "re_round",
        );
    }

    #[test]
    fn causal_mask_blocks_strict_future() {
        let m = causal_mask(4);
        let expected = [
            false, true, true, true, //
            false, false, true, true, //
            false, false, false, true, //
            false, false, false, false,
        ];
        assert_eq!(m.as_slice::<bool>(), expected);
    }
}
