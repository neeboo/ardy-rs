中文: [README.zh-CN.md](README.zh-CN.md)

# ardy-rs

A Rust implementation of **ARDY** (NVIDIA, SIGGRAPH 2026 — autoregressive
diffusion with a hybrid motion representation) on Apple Silicon, ported from
[`ardy-mlx`](https://github.com/neeboo/ardy-mlx) — the MLX implementation
aligned against the reference `nv-tlabs/ardy` runs to float32-roundoff level
(~2e-05) — and running on [MLX](https://github.com/ml-explore/mlx) through
[`mlx-rs`](https://crates.io/crates/mlx-rs).

**Status: port in progress.** Milestones:

| milestone | scope | state |
|---|---|---|
| M0 | checkpoint bundle I/O: contract constants, safetensors weights, stats `.npy` | **done** |
| M1 | nn operator layer (linear, layernorm, gelu/silu, PE, masks, attention) | in progress |
| M2 | Stage A tokenizer (FSQ VAE transformer) | |
| M3 | Stage B denoiser (twostage, 3-pass CFG batch) | |
| M4 | sampling: cosine schedule, DDIM `eta=0`, autoregressive windows | |
| M5 | motion_rep decode + skeleton FK + CLI, end-to-end vs golden clips | |
| M6 | MotionCorrection port (trajectory correction + IK postprocess) | |
| M7 | docs, attribution, publish | |

Out of scope, same as ardy-mlx v1: the 8B text encoder (precomputed
embeddings in), CUDA/TensorRT, real-time streaming.

## Checkpoints

The converted MLX weights (`checkpoints/ARDY-Core-RP-20FPS-Horizon40/mlx/`)
are **not committed** and neither are golden outputs (NVIDIA terms; see
`NOTICE.md`). Point the tests at a local ardy-mlx checkout — the default
discovery is the sibling directory `../ardy-mlx/checkpoints/...`, overridable
with `ARDY_BUNDLE`.

## License

Apache-2.0. See `NOTICE.md` for attribution and the checkpoint-terms record.
