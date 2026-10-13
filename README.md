中文: [README.zh-CN.md](README.zh-CN.md)

# ardy-rs

A Rust implementation of **ARDY** (NVIDIA, SIGGRAPH 2026 — autoregressive
diffusion with a hybrid motion representation) on Apple Silicon, ported from
[`ardy-mlx`](https://github.com/neeboo/ardy-mlx) — the MLX implementation
aligned against the reference `nv-tlabs/ardy` runs to float32-roundoff level
(~2e-05) — and running on [MLX](https://github.com/ml-explore/mlx) through
[`mlx-rs`](https://crates.io/crates/mlx-rs).

**Status: complete** against the `ARDY-Core-RP-20FPS-Horizon40` checkpoint
(`cskel27` core rig, `has_double_ankle_joints = false`):

| milestone | scope | state |
|---|---|---|
| M0 | checkpoint bundle I/O: contract constants, safetensors weights, stats `.npy` | done |
| M1 | nn operator layer (linear, layernorm, gelu/silu, PE, masks, attention) | done |
| M2 | Stage A tokenizer (FSQ VAE transformer) — golden-replay exact | done |
| M3 | Stage B denoiser (twostage, 3-pass CFG batch) — golden-aligned to 1.4e-6 | done |
| M4 | sampling: cosine schedule, DDIM `eta=0`, autoregressive windows | done |
| M5 | motion_rep decode + skeleton FK + CLI, end-to-end vs golden clips (5.2e-6) | done |
| M6 | MotionCorrection port (trajectory correction + IK postprocess), diff'd vs the real C++ at ≤ 1e-5 | done |
| M7 | docs, attribution | done |

Out of scope, same as ardy-mlx v1: the 8B text encoder (precomputed
embeddings in), CUDA/TensorRT, real-time streaming.

## CLI

```sh
cargo run --release -- \
    --bundle ../ardy-mlx/checkpoints/ARDY-Core-RP-20FPS-Horizon40 \
    --text-embedding ../ardy-mlx/artifacts/golden-run-4s/raw/text_feat.npy \
    --seed 0 --frames 80 \
    --out /tmp/ardy-out \
    --postprocess
```

Writes one `.npy` per output tensor (`motion_normalized`, `local_rot_mats`,
`global_rot_mats`, `posed_joints`, `root_positions`, `smooth_root_pos`,
`foot_contacts`, `global_root_heading`) plus an `offsets.json` sidecar.

Key flags (see `--help` for all):

- `--noise a.npy b.npy …` — per-window `x_T` tensors, replacing the
  `--seed` RNG (the golden runs' CUDA noise, which MLX cannot reproduce).
- `--golden FILE.npz` — compare outputs 1:1 against a golden archive.
- `--postprocess` — run the MotionCorrection IK/trajectory postpass on the
  decoded joints (upstream defaults, unconstrained). Off by default; the
  golden contract runs without it.

## Tests

```sh
cargo test
```

The golden-gated suites replay intermediate tensors and final clips from a
local ardy-mlx checkout (default discovery `../ardy-mlx`, override
`ARDY_BUNDLE`); they skip without the bundle.

The MotionCorrection port is additionally verified **differentially against
the real upstream C++**: `tests/postprocess.rs` feeds synthetic LCG records
(no NVIDIA data) through both the Rust port and a probe binary compiled from
the unmodified upstream sources. See `scripts/mc_probe/README.md` to build
the probe; without it those tests pass vacuously.

## A verified upstream quirk

The differential probe proved that upstream's
`TrajectoryCorrector::computeDiffMats` velocity weighting is **dead code**:
`multVelWeights` computes weighted triplets into a local vector but never
calls `setFromTriplets`, so the velocity-difference matrix `V` stays
unweighted. The Rust port mirrors the observed behavior and documents it in
`DiffMats`'s doc comment — constraining frames match the C++ exactly
(≤ 1e-5), free frames included.

## Checkpoints

The converted MLX weights (`checkpoints/ARDY-Core-RP-20FPS-Horizon40/mlx/`)
are **not committed** and neither are golden outputs (NVIDIA terms; see
`NOTICE.md`). Point the tests at a local ardy-mlx checkout — the default
discovery is the sibling directory `../ardy-mlx/checkpoints/...`, overridable
with `ARDY_BUNDLE`.

## License

Apache-2.0. See `NOTICE.md` for attribution and the checkpoint-terms record.
