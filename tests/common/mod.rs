//! Shared helpers for the bundle-gated golden replay tests.
//!
//! Include with `mod common; use common::*;` from each `tests/*.rs` binary.
//! Tests skip with a message when the bundle (`ARDY_BUNDLE`) or the golden
//! run (`ARDY_GOLDEN`) is absent.

// helpers are shared across test binaries; not every binary uses every helper
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ardy::io::npy::{NpyArray, read_npy_f32, read_npz_f32};
use mlx_rs::Array;

pub fn bundle_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ARDY_BUNDLE") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ardy-mlx/checkpoints/ARDY-Core-RP-20FPS-Horizon40");
    p.is_dir().then_some(p)
}

pub fn golden_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ARDY_GOLDEN") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ardy-mlx/artifacts/golden-run-4s/raw");
    p.is_dir().then_some(p)
}

pub fn npy(dir: &Path, name: &str) -> NpyArray {
    read_npy_f32(&dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Every member of an `.npz` golden dump, in archive order.
pub fn npz(dir: &Path, name: &str) -> Vec<(String, NpyArray)> {
    read_npz_f32(&dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// One `.npz` member by key (name without the `.npy` suffix).
pub fn npz_member<'a>(members: &'a [(String, NpyArray)], key: &str) -> &'a NpyArray {
    members
        .iter()
        .find(|(name, _)| name.strip_suffix(".npy") == Some(key))
        .map(|(_, arr)| arr)
        .unwrap_or_else(|| panic!("npz member {key:?}.npy not found"))
}

/// Host-side copy of an `.npy` payload as an unevaluated-shaped MLX array.
pub fn arr(n: &NpyArray) -> Array {
    let dims: Vec<i32> = n.shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&n.data, &dims)
}

/// The golden hook stores every tensor as f32; masks were bool 0/1 at
/// runtime. Reconstruct the bools exactly (`> 0.5`).
pub fn bool_arr(n: &NpyArray) -> Array {
    let v: Vec<bool> = n.data.iter().map(|&x| x > 0.5).collect();
    let dims: Vec<i32> = n.shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&v, &dims)
}

/// Same as [`bool_arr`] but for exact integer scalars/vectors (lengths,
/// timesteps) dumped as f32.
pub fn i32_arr(n: &NpyArray) -> Array {
    let v: Vec<i32> = n.data.iter().map(|&x| x as i32).collect();
    let dims: Vec<i32> = n.shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&v, &dims)
}

/// Materialize a (possibly strided) view as a flat f32 vector.
pub fn flat(a: &Array) -> Vec<f32> {
    let f = a.reshape(&[-1]).unwrap();
    f.eval().unwrap();
    f.as_slice::<f32>().to_vec()
}

pub fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length {} vs {}", a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{ctx}: max abs diff = {max:.3e}");
    assert!(max <= tol, "{ctx}: max abs diff {max:.3e} > {tol:.0e}");
}
