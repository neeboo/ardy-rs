//! Rust implementation of **ARDY** (NVIDIA, SIGGRAPH 2026 — autoregressive
//! diffusion with a hybrid motion representation) on Apple Silicon.
//!
//! This is a port of [`ardy-mlx`](https://github.com/neeboo/ardy-mlx), the
//! MLX implementation aligned against the reference `nv-tlabs/ardy` runs to
//! float32-roundoff level, plus the upstream `MotionCorrection`
//! trajectory/IK postprocess pass.
//!
//! Scope mirrors ardy-mlx v1: the motion model and sampling (precomputed text
//! embeddings in, motions out). The 8B text encoder is out of scope, and
//! NVIDIA weights are never redistributed.
//!
//! ```no_run
//! use std::path::Path;
//! use ardy::io::bundle::Bundle;
//!
//! let bundle = Bundle::open(Path::new("checkpoints/ARDY-Core-RP-20FPS-Horizon40"))?;
//! println!("{} tokenizer tensors, {} denoiser tensors",
//!          bundle.tokenizer().len(), bundle.denoiser().len());
//! # Ok::<(), ardy::Error>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod io;

pub use error::{Error, Result};
