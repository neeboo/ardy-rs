//! Contract constants and feature layout for ARDY motion data.
//!
//! Port of `ardy_mlx/io/__init__.py`, which is itself transcribed from
//! `nv-tlabs/ardy` @ `693f74d` (see the citations on every constant). The
//! single source of truth is ardy-mlx's `docs/motion-contract.md`; do not
//! invent values here.

use std::collections::BTreeMap;

pub mod bundle;
pub mod npy;
pub mod weights;

/// Feature block order — the concatenation order of `frames (T, D)`.
///
/// `ardy/motion_rep/reps/ardy_motionrep.py:57-64` (dict insertion order) and
/// `:169-181` (einops.pack).
pub const BLOCK_ORDER: [&str; 6] = [
    "root_pos",               // [3] metres, world root position (x, y, z)
    "global_root_heading",    // [2] (cos theta, sin theta)
    "local_joints_positions", // [J-1, 3] metres, root-local, ground-aligned
    "global_rot_data",        // [J, 6] 6D continuous rotation of GLOBAL joint rotations
    "velocities",             // [J, 3] metres/second, global
    "foot_contacts",          // [4] binary
];

/// `last_root_feature == "global_root_heading"` (`ardy_motionrep.py:65`) =>
/// the root block is these first two blocks.
pub const ROOT_BLOCK_NAMES: [&str; 2] = ["root_pos", "global_root_heading"];

/// Everything after the root block (`base.py:48`).
pub const BODY_BLOCK_NAMES: [&str; 4] = [
    "local_joints_positions",
    "global_rot_data",
    "velocities",
    "foot_contacts",
];

/// `feet.py:23-25,52` — `torch.cat((feet_l, feet_r))` => left first.
pub const FOOT_CONTACT_ORDER: [&str; 4] = ["left_heel", "left_toe", "right_heel", "right_toe"];

/// `config.yaml:25` — 4 frames per token ("patch size").
pub const NUM_FRAMES_PER_TOKEN: usize = 4;

/// `config.yaml:22-23` — FSQ channels; `latent_embedding_dim = num_fsq_levels`
/// (`fsq.py:85`).
pub const FSQ_CHANNELS: usize = 128;

/// An int `fsq_level_list` expands to `[64] * num_fsq_levels`
/// (`fsq.py:117-118`).
pub const FSQ_LEVELS: i64 = 64;

/// The body latent width == FSQ channels. Corroborated by
/// `stats/post_quantization/mean.npy` shape `(128,)`.
pub const LATENT_BODY_DIM: usize = FSQ_CHANNELS;

/// `local_root_size_dict` (`ardy_motionrep.py:66-70`):
/// rot_vel[1] + vel[2] + global_root_y[1].
pub const LOCAL_ROOT_DIM: usize = 4;

/// The root block flattened over a token; 20 for both released skeletons.
pub const ROOT_BLOCK_DIM: usize = 5;

/// `ardy_model.py:479` — `nframe_root_dim + latent_embedding_dim` == 20 + 128.
pub const HYBRID_TOKEN_DIM: usize = ROOT_BLOCK_DIM * NUM_FRAMES_PER_TOKEN + LATENT_BODY_DIM;

/// `config.yaml:2` — diffusion schedule length (both stages share one).
pub const NUM_BASE_STEPS: usize = 10;

/// `config.yaml:3` — three-pass "separated" classifier-free guidance.
pub const CFG_TYPE: &str = "separated";

/// `scripts/generate.py:90-96` — default `--cfg_weight` as (text, constraint).
pub const DEFAULT_CFG_WEIGHT: (f64, f64) = (2.0, 2.0);

/// `config.yaml:49-51` `llm_shape` — Llama-3-8B hidden size; text features are
/// `[B, L, 4096]`.
pub const TEXT_FEAT_DIM: usize = 4096;

/// `scripts/generate.py:106-115` — "ARDY is trained on windows of at most 10 s".
pub const MAX_WINDOW_SECONDS: f64 = 10.0;

/// Longest history that still fits the model's trained window.
///
/// Mirror of `scripts/generate.py:106-115`. This is the *batch* default (the
/// maximum); the interactive GUI slider defaults to the minimum instead.
pub fn default_history_frames(
    fps: f64,
    gen_horizon_len: usize,
    num_frames_per_token: usize,
) -> usize {
    let max_window_len =
        ((MAX_WINDOW_SECONDS * fps) as usize / num_frames_per_token) * num_frames_per_token;
    ((max_window_len - gen_horizon_len) / num_frames_per_token) * num_frames_per_token
}

/// Slice arithmetic for `frames (T, D)` for one skeleton.
///
/// All values are derived, not hardcoded, so a new skeleton cannot silently
/// disagree with the contract: [`FeatureLayout::motion_rep_dim`] is the sum of
/// the block sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureLayout {
    /// Number of skeleton joints (root included).
    pub nbjoints: usize,
    /// Sample rate in Hz.
    pub fps: usize,
    /// Generation horizon in frames per window.
    pub horizon: usize,
    /// Skeleton identifier, e.g. `"CoreSkeleton27"`.
    pub skeleton: &'static str,
}

impl FeatureLayout {
    /// Per-block feature count (for one frame), in [`BLOCK_ORDER`].
    pub fn block_dims(&self) -> BTreeMap<&'static str, usize> {
        BTreeMap::from([
            ("root_pos", 3),
            ("global_root_heading", 2),
            ("local_joints_positions", (self.nbjoints - 1) * 3),
            ("global_rot_data", self.nbjoints * 6),
            ("velocities", self.nbjoints * 3),
            ("foot_contacts", 4),
        ])
    }

    /// Half-open `[start, stop)` slice of every block inside a `(T, D)` row.
    pub fn slices(&self) -> BTreeMap<&'static str, (usize, usize)> {
        let dims = self.block_dims();
        let mut out = BTreeMap::new();
        let mut start = 0;
        for name in BLOCK_ORDER {
            let stop = start + dims[name];
            out.insert(name, (start, stop));
            start = stop;
        }
        out
    }

    /// `D` of `frames (T, D)` (`base.py:46`).
    pub fn motion_rep_dim(&self) -> usize {
        self.block_dims().values().sum()
    }

    /// `global_root_dim` (`base.py:50`).
    pub fn root_block_dim(&self) -> usize {
        ROOT_BLOCK_NAMES.iter().map(|n| self.block_dims()[n]).sum()
    }

    /// `body_dim` (`base.py:49`).
    pub fn body_dim(&self) -> usize {
        self.motion_rep_dim() - self.root_block_dim()
    }

    /// `local_root_dim + body_dim` (`base.py:53`).
    pub fn local_motion_rep_dim(&self) -> usize {
        LOCAL_ROOT_DIM + self.body_dim()
    }

    /// Root block slice `[0, root_block_dim)`.
    pub fn root_slice(&self) -> (usize, usize) {
        (0, self.root_block_dim())
    }

    /// Body slice `[root_block_dim, motion_rep_dim)`.
    pub fn body_slice(&self) -> (usize, usize) {
        (self.root_block_dim(), self.motion_rep_dim())
    }

    /// Root block flattened over [`NUM_FRAMES_PER_TOKEN`] frames.
    pub fn nframe_root_dim(&self) -> usize {
        self.root_block_dim() * NUM_FRAMES_PER_TOKEN
    }

    /// Denoiser hybrid token width: root block flattened + body latent.
    pub fn dim_token(&self) -> usize {
        self.nframe_root_dim() + LATENT_BODY_DIM
    }

    /// Trained window budget, floored to a whole number of tokens
    /// (`generate.py:114`).
    pub fn max_window_frames(&self) -> usize {
        ((MAX_WINDOW_SECONDS * self.fps as f64) as usize / NUM_FRAMES_PER_TOKEN)
            * NUM_FRAMES_PER_TOKEN
    }

    /// Batch-path `--history_frames` default (`generate.py:115`).
    pub fn default_history_frames(&self) -> usize {
        default_history_frames(self.fps as f64, self.horizon, NUM_FRAMES_PER_TOKEN)
    }
}

/// The four released checkpoints (`README.md:67-70` upstream). Only the first
/// ships locally today.
pub fn preset(name: &str) -> Option<FeatureLayout> {
    match name {
        "ARDY-Core-RP-20FPS-Horizon40" => Some(FeatureLayout {
            nbjoints: 27,
            fps: 20,
            horizon: 40,
            skeleton: "CoreSkeleton27",
        }),
        "ARDY-Core-RP-20FPS-Horizon8" => Some(FeatureLayout {
            nbjoints: 27,
            fps: 20,
            horizon: 8,
            skeleton: "CoreSkeleton27",
        }),
        "ARDY-G1-RP-25FPS-Horizon52" => Some(FeatureLayout {
            nbjoints: 34,
            fps: 25,
            horizon: 52,
            skeleton: "G1Skeleton34",
        }),
        "ARDY-G1-RP-25FPS-Horizon8" => Some(FeatureLayout {
            nbjoints: 34,
            fps: 25,
            horizon: 8,
            skeleton: "G1Skeleton34",
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Self-check against the contract's independently verified numbers
    /// (`docs/motion-contract.md` S1.3). D=330 for Core was confirmed three
    /// ways: block sum, tokenizer io dims, and the denoiser constraints
    /// projection dims.
    #[test]
    fn contract_numbers_match() {
        let core = preset("ARDY-Core-RP-20FPS-Horizon40").unwrap();
        assert_eq!(core.motion_rep_dim(), 330);
        assert_eq!(core.body_dim(), 325);
        assert_eq!(core.local_motion_rep_dim(), 329);
        assert_eq!(core.dim_token(), 148);
        assert_eq!(core.default_history_frames(), 160);

        let g1 = preset("ARDY-G1-RP-25FPS-Horizon52").unwrap();
        assert_eq!(g1.motion_rep_dim(), 414);
        assert_eq!(g1.body_dim(), 409);

        assert_eq!(HYBRID_TOKEN_DIM, 148);
        assert_eq!(default_history_frames(20.0, 40, 4), 160);
    }
}
