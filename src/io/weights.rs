//! Safetensors weight loading for the MLX-converted checkpoints.
//!
//! The converted bundles (`mlx/tokenizer.mlx.safetensors`,
//! `mlx/denoiser.mlx.safetensors`) are all-F32 containers produced by
//! `scripts/ardy_convert.py` (428/428 tensors byte-identical to the
//! reference). Parsing uses the `safetensors` crate; this module adds the
//! F32-only decode and the name->tensor map the model code walks.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Error, Result};

/// One F32 tensor: row-major shape plus flat payload.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightTensor {
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Flat row-major f32 payload, `shape` product long.
    pub data: Vec<f32>,
}

impl WeightTensor {
    /// Number of rows (everything but the last dimension folded together).
    pub fn rows(&self) -> usize {
        if self.shape.len() <= 1 {
            1
        } else {
            self.shape[..self.shape.len() - 1].iter().product()
        }
    }

    /// Columns (the last dimension).
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&0)
    }
}

/// A whole safetensors file decoded as f32, ordered by tensor name.
#[derive(Clone, Debug, Default)]
pub struct WeightMap {
    tensors: BTreeMap<String, WeightTensor>,
}

impl WeightMap {
    /// Load a safetensors file, decoding every tensor as f32.
    ///
    /// The converted MLX bundles are all F32; F16/F64/I32 widen or convert
    /// losslessly where possible and anything else is rejected.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(path, &bytes)
    }

    /// Parse safetensors bytes (see [`WeightMap::load`]).
    pub fn parse(path: &Path, bytes: &[u8]) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| Error::Safetensors(format!("{}: {e}", path.display())))?;
        let mut tensors = BTreeMap::new();
        for (name, view) in st.iter() {
            let shape: Vec<usize> = view.shape().to_vec();
            let data = decode_f32(path, name, view.dtype(), view.data())?;
            tensors.insert(name.to_string(), WeightTensor { shape, data });
        }
        Ok(Self { tensors })
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the file had no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Look up one tensor by name.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.tensors.get(name).ok_or_else(|| {
            Error::Safetensors(format!(
                "no tensor named {name:?} (have {} tensors)",
                self.tensors.len()
            ))
        })
    }

    /// Look up one tensor by name, with caller-supplied context in the error.
    pub fn get_in(&self, context: &str, name: &str) -> Result<&WeightTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Safetensors(format!("{context}: no tensor named {name:?}")))
    }

    /// Tensor names sharing a dotted prefix, sorted.
    pub fn prefixed<'s>(&'s self, prefix: &str) -> impl Iterator<Item = &'s String> + 's {
        let with_dot = format!("{prefix}.");
        self.tensors
            .keys()
            .filter(move |n| n.starts_with(with_dot.as_str()))
    }
}

fn decode_f32(path: &Path, name: &str, dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    let bail = |msg: String| Error::Safetensors(format!("{}: {name}: {msg}", path.display()));
    let expected = |n: usize, got: usize| {
        bail(format!(
            "dtype {dtype:?} needs {n} bytes per element, payload has {got} total"
        ))
    };
    let mut out = Vec::with_capacity(data.len() / 4);
    match dtype {
        safetensors::Dtype::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(expected(4, data.len()));
            }
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        safetensors::Dtype::F64 => {
            if !data.len().is_multiple_of(8) {
                return Err(expected(8, data.len()));
            }
            for chunk in data.chunks_exact(8) {
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
        safetensors::Dtype::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(expected(2, data.len()));
            }
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_bits_to_f32(bits));
            }
        }
        safetensors::Dtype::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(expected(2, data.len()));
            }
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f32::from_bits((bits as u32) << 16));
            }
        }
        other => {
            return Err(bail(format!("dtype {other:?} is not a float type")));
        }
    }
    Ok(out)
}

/// IEEE 754 binary16 -> f32, without the `half` crate (same core as
/// `io::npy::half_bits_to_f32`).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut e = 113u32;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x3ff;
                sign | (e << 23) | (f << 13)
            }
        }
        0x1f => sign | (0xff << 23) | (frac << 13),
        e => sign | ((e + 112) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal one-tensor safetensors file: u64 header length, JSON
    /// header, payload.
    fn one_tensor_f32() -> Vec<u8> {
        let payload: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"layer.w": {{"dtype": "F32", "shape": [1, 3], "data_offsets": [0, {}]}}}}"#,
            payload.len()
        );
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn parses_f32() {
        let map = WeightMap::parse(Path::new("t.safetensors"), &one_tensor_f32()).unwrap();
        assert_eq!(map.len(), 1);
        let w = map.get("layer.w").unwrap();
        assert_eq!(w.shape, vec![1, 3]);
        assert_eq!(w.data, vec![1.0, -2.5, 3.25]);
        assert_eq!(w.rows(), 1);
        assert_eq!(w.cols(), 3);
    }

    #[test]
    fn missing_tensor_errors() {
        let map = WeightMap::parse(Path::new("t.safetensors"), &one_tensor_f32()).unwrap();
        assert!(map.get("nope").is_err());
        assert!(map.get_in("ctx", "nope").is_err());
        assert_eq!(map.prefixed("layer").count(), 1);
        assert_eq!(map.prefixed("layer.x").count(), 0);
    }
}
