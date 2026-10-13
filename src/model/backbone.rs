//! `TransformerEncoderBlock` -- the denoiser's shared transformer backbone.
//!
//! Port of `ardy_mlx/model/backbone.py` (reference: `ardy/model/backbone.py`).
//! One block is instantiated twice (once for the root stage, once for the body
//! stage); both share the structure but have their own weights under
//! `denoiser.backbone.{root,body}_model`.
//!
//! The released Core checkpoint uses
//! `positional_encoding_mode: learned_prefix_zero_at_first_generation`, which
//! means:
//!
//! * the prefix is `[text(1), timestep(1), first_heading_angle(1)]` and gets a
//!   **learned** positional embedding (3 rows, `learned_prefix_embedding`);
//! * the motion tokens get the sinusoidal table indexed by `token_index`,
//!   which is *relative* to the first generation token and may be negative;
//! * the plain sinusoidal `sequence_pos_encoder` is therefore **unused** in
//!   the forward pass -- it is not materialized here.
//!
//! There is **no causal mask** here: `seqTransEncoder` runs with only a key
//! padding mask, so attention is bidirectional.
//!
//! The reference's `text_feat_pad_mask` argument is ignored on this path
//! (`use_text_mask == False` forces every text token to be unmasked), so this
//! port does not take it.

use mlx_rs::Array;
use mlx_rs::ops::{self, indexing::TryIndexOp};

use super::nn::{
    LearnedPositionalEncoding, Linear, PositionalEncodingNegativeIndex, TimestepEmbedder,
    TransformerEncoder, WeightStore, build_additive_mask,
};
use crate::{Error, Result};

/// One denoiser stage: prefix conditioning + bidirectional transformer.
pub struct TransformerEncoderBlock {
    input_first_heading_angle: bool,
    learned_mode: bool,
    num_text_tokens: usize,
    prefix_length: Option<usize>,
    embed_text: Linear,
    embed_timestep: TimestepEmbedder,
    output_linear: Linear,
    linear_first_heading_angle: Option<Linear>,
    learned_prefix_embedding: Option<LearnedPositionalEncoding>,
    motion_token_embedding: Option<PositionalEncodingNegativeIndex>,
    seq_trans_encoder: TransformerEncoder,
}

impl TransformerEncoderBlock {
    #[allow(clippy::too_many_arguments)]
    /// Load the released-Core configuration from `prefix`: width 1024,
    /// 8 layers / 8 heads, learned-prefix positional encoding (the `"default"`
    /// mode exists upstream but is not exercised by this checkpoint).
    pub fn new(
        store: &mut WeightStore,
        prefix: &str,
        latent_dim: usize,
        num_layers: usize,
        num_heads: usize,
        output_dim: usize,
        num_text_tokens: usize,
        input_first_heading_angle: bool,
    ) -> Result<Self> {
        let embed_text = Linear::new(store, &format!("{prefix}.embed_text"), true)?;
        let embed_timestep =
            TimestepEmbedder::new(store, &format!("{prefix}.embed_timestep"), latent_dim)?;
        let output_linear = Linear::new(store, &format!("{prefix}.output_linear"), true)?;
        if output_linear.out_features != output_dim {
            return Err(Error::Contract(format!(
                "{prefix}.output_linear produces {} features, expected {output_dim}",
                output_linear.out_features
            )));
        }

        // positional_encoding_mode == "learned_prefix_zero_at_first_generation"
        let mut prefix_length = num_text_tokens + 1;
        let linear_first_heading_angle = if input_first_heading_angle {
            prefix_length += 1;
            Some(Linear::new(
                store,
                &format!("{prefix}.linear_first_heading_angle"),
                true,
            )?)
        } else {
            None
        };
        let learned_prefix_encoding = LearnedPositionalEncoding::new(
            store,
            &format!("{prefix}.learned_prefix_embedding.embedding"),
        )?;
        let motion_token_embedding = PositionalEncodingNegativeIndex::new(latent_dim, 5000);

        let seq_trans_encoder = TransformerEncoder::new(
            store,
            &format!("{prefix}.seqTransEncoder"),
            num_layers,
            latent_dim,
            num_heads,
            false,
        )?;

        Ok(Self {
            input_first_heading_angle,
            learned_mode: true,
            num_text_tokens,
            prefix_length: Some(prefix_length),
            embed_text,
            embed_timestep,
            output_linear,
            linear_first_heading_angle,
            learned_prefix_embedding: Some(learned_prefix_encoding),
            motion_token_embedding: Some(motion_token_embedding),
            seq_trans_encoder,
        })
    }

    /// Prefix length (`[text, timestep, heading]` = 3 for the Core
    /// checkpoint).
    pub fn prefix_length(&self) -> Option<usize> {
        self.prefix_length
    }

    /// `TransformerEncoderBlock.forward` (`backbone.py:300-396`).
    ///
    /// `x_pad_mask` is a bool `[B, nt]` mask, `true = valid`; `token_index` is
    /// the int `[B, nt]` PE index relative to the first generation token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        x_pad_mask: &Array,
        text_feat: &Array,
        timesteps: &Array,
        first_heading_angle: Option<&Array>,
        token_index: Option<&Array>,
    ) -> Result<Array> {
        if !self.learned_mode {
            return Err(Error::Contract(
                "positional_encoding_mode \"default\" is not exercised by the \
                 released Core checkpoint and is not implemented"
                    .to_string(),
            ));
        }
        let batch_size = x.shape()[0] as usize;

        // The reference slices the *sequence* axis: `text_feat[:, :num_text_tokens]`
        // (`backbone.py:329`) — the feature dim stays intact.
        let text_feat =
            text_feat.try_index(((0i32..), (0i32..self.num_text_tokens as i32), (0i32..)))?;
        let emb_text = self.embed_text.forward(&text_feat)?; // [B, 1, D]
        let emb_time = self.embed_timestep.forward(timesteps)?; // [B, 1, D]
        let time_mask = Array::ones::<bool>(&[batch_size as i32, 1])?;

        let mut prefix_feats = ops::concatenate(&[&emb_text, &emb_time], 1)?;
        // use_text_mask == False -> every text token is unmasked
        let text_mask = Array::ones::<bool>(&[batch_size as i32, emb_text.shape()[1]])?;
        let mut prefix_mask = ops::concatenate(&[&text_mask, &time_mask], 1)?;

        if self.input_first_heading_angle {
            let lin = self
                .linear_first_heading_angle
                .as_ref()
                .expect("heading linear present when input_first_heading_angle");
            let angle = first_heading_angle.ok_or_else(|| {
                Error::Contract(
                    "input_first_heading_angle is set but no first_heading_angle was passed"
                        .to_string(),
                )
            })?;
            let feats = ops::stack(&[ops::cos(angle)?, ops::sin(angle)?], -1)?; // [B, 2]
            let feats = lin.forward(&feats)?;
            let feats = feats.expand_dims(1)?; // [B, 1, D]
            prefix_feats = ops::concatenate(&[&prefix_feats, &feats], 1)?;
            prefix_mask = ops::concatenate(
                &[&prefix_mask, &Array::ones::<bool>(&[batch_size as i32, 1])?],
                1,
            )?;
        }

        let pose_start_ind = prefix_feats.shape()[1];

        let prefix_feats = self
            .learned_prefix_embedding
            .as_ref()
            .expect("learned prefix embedding in learned mode")
            .forward(&prefix_feats)?;
        let tok_idx = token_index.ok_or_else(|| {
            Error::Contract(
                "learned_prefix_zero_at_first_generation requires token_index".to_string(),
            )
        })?;
        let x_pe = self
            .motion_token_embedding
            .as_ref()
            .expect("motion token embedding in learned mode")
            .forward(x, tok_idx)?;
        let xseq = ops::concatenate(&[&prefix_feats, &x_pe], 1)?;

        let full_mask = ops::concatenate(&[&prefix_mask, x_pad_mask], 1)?;
        let src_key_padding_mask = full_mask.logical_not()?;
        let mask = build_additive_mask(Some(&src_key_padding_mask), None)?;
        let output = self.seq_trans_encoder.forward(&xseq, mask.as_ref())?;
        let output = output.try_index(((0i32..), (pose_start_ind..)))?;
        self.output_linear.forward(&output)
    }
}
