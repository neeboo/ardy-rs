//! Checkpoint bundle loading: `config.yaml` + converted weights + stats.
//!
//! A bundle is the on-disk layout produced by `scripts/ardy_convert.py`:
//!
//! ```text
//! ARDY-Core-RP-20FPS-Horizon40/
//!   config.yaml
//!   mlx/tokenizer.mlx.safetensors
//!   mlx/denoiser.mlx.safetensors
//!   stats/{motion,pre_quantization,post_quantization}/{mean,std}.npy
//! ```

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::io::npy::NpyArray;
use crate::io::weights::WeightMap;
use crate::io::FeatureLayout;

/// A loaded checkpoint bundle: parsed config, weights, and stats access.
#[derive(Clone, Debug)]
pub struct Bundle {
    root: PathBuf,
    config: BundleConfig,
    layout: FeatureLayout,
    tokenizer: WeightMap,
    denoiser: WeightMap,
}

impl Bundle {
    /// Open a bundle directory, eagerly parsing `config.yaml` and loading
    /// both converted safetensors files.
    pub fn open(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.yaml");
        let config: BundleConfig = parse_yaml(&config_path)?;
        let layout = config.layout().ok_or_else(|| {
            Error::Bundle(format!(
                "{}: skeleton/fps/horizon do not match a released ARDY preset",
                dir.display()
            ))
        })?;

        let tokenizer = WeightMap::load(&dir.join("mlx/tokenizer.mlx.safetensors"))?;
        let denoiser = WeightMap::load(&dir.join("mlx/denoiser.mlx.safetensors"))?;

        let bundle = Self {
            root: dir.to_path_buf(),
            config,
            layout,
            tokenizer,
            denoiser,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// Bundle root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Parsed `config.yaml`.
    pub fn config(&self) -> &BundleConfig {
        &self.config
    }

    /// Feature layout derived from the config's skeleton/fps/horizon.
    pub fn layout(&self) -> &FeatureLayout {
        &self.layout
    }

    /// Converted Stage A tokenizer weights.
    pub fn tokenizer(&self) -> &WeightMap {
        &self.tokenizer
    }

    /// Converted Stage B denoiser weights.
    pub fn denoiser(&self) -> &WeightMap {
        &self.denoiser
    }

    /// Load one stats directory (`motion`, `pre_quantization`,
    /// `post_quantization`) as mean/std pair.
    pub fn stats(&self, kind: &str) -> Result<Stats> {
        let dir = self.root.join("stats").join(kind);
        let mean = crate::io::npy::read_npy_f32(&dir.join("mean.npy"))?;
        let std = crate::io::npy::read_npy_f32(&dir.join("std.npy"))?;
        if mean.shape != std.shape {
            return Err(Error::Bundle(format!(
                "{}: mean shape {:?} != std shape {:?}",
                dir.display(),
                mean.shape,
                std.shape
            )));
        }
        Ok(Stats { mean, std })
    }

    /// Cross-check the parsed config against the frozen contract.
    fn validate(&self) -> Result<()> {
        fn expect(what: &str, got: impl std::fmt::Debug, want: impl std::fmt::Debug) -> Result<()> {
            if format!("{got:?}") != format!("{want:?}") {
                Err(Error::Contract(format!(
                    "{what} is {got:?}, contract says {want:?}"
                )))
            } else {
                Ok(())
            }
        }

        let cfg = &self.config;
        expect("num_base_steps", cfg.num_base_steps, crate::io::NUM_BASE_STEPS)?;
        expect("cfg_type", cfg.cfg_type.as_str(), crate::io::CFG_TYPE)?;
        expect(
            "autoencoder.num_frames_per_token",
            cfg.autoencoder.num_frames_per_token,
            crate::io::NUM_FRAMES_PER_TOKEN,
        )?;
        expect(
            "autoencoder.latent_embedding_dim",
            cfg.autoencoder.latent_embedding_dim,
            crate::io::LATENT_BODY_DIM,
        )?;
        expect(
            "denoiser.nframe_root_dim",
            cfg.denoiser.nframe_root_dim,
            self.layout.nframe_root_dim(),
        )?;
        expect(
            "denoiser token width",
            cfg.denoiser.nframe_root_dim + cfg.denoiser.latent_embedding_dim,
            self.layout.dim_token(),
        )?;

        // The converted weights must contain the tensors the models will ask
        // for; the per-model loaders do the deep checks, this catches a
        // truncated or swapped bundle early.
        if self.tokenizer.is_empty() || self.denoiser.is_empty() {
            return Err(Error::Bundle(format!(
                "{}: tokenizer/denoiser safetensors loaded 0 tensors",
                self.root.display()
            )));
        }
        Ok(())
    }
}

/// Normalization statistics (one `mean`/`std` pair), shape `(D,)` or `(rows, D)`.
#[derive(Clone, Debug)]
pub struct Stats {
    /// Per-feature mean.
    pub mean: NpyArray,
    /// Per-feature standard deviation (nonzero).
    pub std: NpyArray,
}

impl Stats {
    /// Feature width `D`.
    pub fn dim(&self) -> usize {
        self.mean.row_len()
    }
}

/// `config.yaml` top level.
#[derive(Clone, Debug, Deserialize)]
pub struct BundleConfig {
    /// Diffusion schedule length (both stages share one schedule).
    pub num_base_steps: usize,
    /// CFG mode; the released checkpoints use `"separated"`.
    pub cfg_type: String,
    /// Generation horizon in frames per window.
    pub gen_horizon_len: usize,
    /// Stage A tokenizer (FSQ VAE transformer) settings.
    pub autoencoder: TokenizerConfig,
    /// Stage B denoiser settings.
    pub denoiser: DenoiserConfig,
}

impl BundleConfig {
    /// Feature layout for this config, or `None` if the skeleton/fps/horizon
    /// combination is not a released preset.
    pub fn layout(&self) -> Option<FeatureLayout> {
        let skeleton = skeleton_name(&self.autoencoder.motion_rep.skeleton.target)?;
        Some(FeatureLayout {
            nbjoints: joint_count(skeleton)?,
            fps: self.autoencoder.motion_rep.fps,
            horizon: self.gen_horizon_len,
            skeleton,
        })
    }
}

/// The `motion_rep:` block shared by both stages (fps + skeleton reference;
/// `stats_path` is Hydra-interpolated and not meaningful here).
#[derive(Clone, Debug, Deserialize)]
pub struct MotionRepRef {
    /// Sample rate in Hz.
    pub fps: usize,
    /// Skeleton constructor reference.
    pub skeleton: SkeletonRef,
}

/// The `skeleton:` block (`_target_: ardy.skeleton.CoreSkeleton27`).
#[derive(Clone, Debug, Deserialize)]
pub struct SkeletonRef {
    /// Skeleton constructor reference, e.g. `ardy.skeleton.CoreSkeleton27`.
    #[serde(rename = "_target_")]
    pub target: String,
}

/// Stage A tokenizer (FSQ VAE transformer) settings.
#[derive(Clone, Debug, Deserialize)]
pub struct TokenizerConfig {
    /// `pose` for the released checkpoints.
    pub pose_root_mode: String,
    /// Normalize motion features before encoding.
    pub encode_with_normalization: bool,
    /// Quantize the encoded latent through FSQ.
    pub encode_with_quantization: bool,
    /// Stream order of the four encode/decode passes, e.g.
    /// `[body, pose, body, root]`.
    pub feature_mode: Vec<String>,
    /// Number of FSQ channels (== `latent_embedding_dim`).
    pub num_fsq_levels: usize,
    /// FSQ level count per channel (int expands to `[level; channels]`).
    pub fsq_level_list: i64,
    /// Body latent width.
    pub latent_embedding_dim: usize,
    /// Frames per token ("patch size").
    pub num_frames_per_token: usize,
    /// Transformer width.
    pub latent_dim: usize,
    /// Feed-forward width.
    pub ff_size: usize,
    /// Transformer depth.
    pub num_layers: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Activation name (`gelu` for the released checkpoints).
    pub activation: String,
    /// Pre-norm vs post-norm blocks.
    pub norm_first: bool,
    /// Causal masking on the encoder.
    pub causal_encoder: bool,
    /// Causal masking on the decoder.
    pub causal_decoder: bool,
    /// Motion representation reference (fps + skeleton).
    pub motion_rep: MotionRepRef,
}

/// Stage B denoiser settings.
#[derive(Clone, Debug, Deserialize)]
pub struct DenoiserConfig {
    /// `concat` for the released checkpoints.
    pub motion_mask_mode: String,
    /// Frames per token ("patch size").
    pub num_frames_per_token: usize,
    /// Root block flattened over a token (20 for both released skeletons).
    pub nframe_root_dim: usize,
    /// Body latent width.
    pub latent_embedding_dim: usize,
    /// Text feature shape `[B, L]` (hidden size is the contract's 4096).
    pub llm_shape: Vec<usize>,
    /// Whether text masks are used (false for the released checkpoints).
    pub use_text_mask: bool,
    /// Transformer width.
    pub latent_dim: usize,
    /// Feed-forward width.
    pub ff_size: usize,
    /// Transformer depth.
    pub num_layers: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Activation name (`gelu`).
    pub activation: String,
    /// Pre-norm vs post-norm blocks.
    pub norm_first: bool,
    /// Whether the first heading angle enters as its own input.
    pub input_first_heading_angle: bool,
    /// Learned positional encoding with a zeroed prefix at the first
    /// generation frame.
    pub positional_encoding_mode: String,
    /// Motion representation reference (fps + skeleton).
    pub motion_rep: MotionRepRef,
}

/// Joint count for a skeleton `_target_` name (`CoreSkeleton27` => 27,
/// `G1Skeleton34` => 34).
pub fn joint_count(skeleton_target: &str) -> Option<usize> {
    match skeleton_target.rsplit('.').next()? {
        "CoreSkeleton27" => Some(27),
        "G1Skeleton34" => Some(34),
        _ => None,
    }
}

/// Short skeleton name (`CoreSkeleton27`) for a skeleton `_target_`.
pub fn skeleton_name(skeleton_target: &str) -> Option<&'static str> {
    match skeleton_target.rsplit('.').next()? {
        "CoreSkeleton27" => Some("CoreSkeleton27"),
        "G1Skeleton34" => Some("G1Skeleton34"),
        _ => None,
    }
}

fn parse_yaml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_yaml::from_str(&text)
        .map_err(|e| Error::Bundle(format!("{}: {e}", path.display())))
}
