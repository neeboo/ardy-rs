# NOTICE and attribution

## Upstream lineage

- **ARDY** — NVIDIA, "Autoregressive Diffusion with Hybrid Representation for
  Interactive Human Motion Generation" (SIGGRAPH 2026). Reference code:
  `nv-tlabs/ardy` @ `693f74d` (Apache-2.0). The MotionCorrection port
  (milestone M6) is derived from that repository's C++
  `MotionCorrection/AnimProcessing` sources; it mirrors one verified upstream
  quirk (dead velocity weighting in `TrajectoryCorrector::computeDiffMats`,
  see `README.md`) rather than "fixing" it, so constrained output stays
  bit-comparable with the reference.
- **ardy-mlx** — `neeboo/ardy-mlx` (Apache-2.0), the MLX implementation this
  repository ports. Its weight conversion is byte-exact against the reference
  checkpoints (428/428 tensors); this repository consumes those converted
  safetensors files unchanged.

## Checkpoint terms

The ARDY checkpoints are released under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/):
local inference, conversion, fine-tuning and commercial use are permitted;
redistribution requires the agreement and attribution. Following ardy-mlx's
recorded decision (`docs/decisions/0002-checkpoint-terms.md`), **this
repository does not redistribute NVIDIA weights or outputs derived from
them**. Converted weights (`*.mlx.safetensors`), golden outputs and
normalization statistics stay local; the test suite skips without them.

## Text encoder

ARDY's text conditioning is a gated Llama-3-8B encoder. This repository (like
ardy-mlx v1) consumes precomputed text embeddings only and never runs or
redistributes the encoder.
