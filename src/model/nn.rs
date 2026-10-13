//! Neural-network primitives for the ARDY port.
//!
//! Rust port of `ardy_mlx/model/nn.py`. Every primitive here mirrors exactly
//! one PyTorch module used by the reference implementation, because the goal
//! is numerical alignment, not a nice API. Where PyTorch's default is
//! surprising, the docstring says so.
//!
//! Conventions, unchanged from the Python port:
//!
//! * All tensors are [`mlx_rs::Array`] in float32, laid out `[B, T, D]`.
//! * [`Linear`] weights are stored in PyTorch's `[out_features,
//!   in_features]` layout, so the forward is `x @ W.T + b` with no
//!   transposes at load time.
//! * Masks follow PyTorch's `src_key_padding_mask` convention: `true` means
//!   "ignore this position".

use std::collections::BTreeSet;

use mlx_rs::{Array, Dtype, ops};

use crate::error::Result;
use crate::io::weights::{WeightMap, WeightTensor};

/// Read-only view over a converted checkpoint with use tracking.
///
/// Keys are the *reference* state-dict names (`pose_net.encoder...` /
/// `denoiser.backbone...`), so a missing or renamed tensor fails loudly at
/// lookup time instead of silently becoming zeros. Every [`WeightStore::get`]
/// records the name; [`WeightStore::unused`] afterwards is a non-empty list
/// means the port is incomplete.
pub struct WeightStore {
    tensors: WeightMap,
    used: BTreeSet<String>,
}

impl WeightStore {
    /// Wrap an already-loaded weight map.
    pub fn new(tensors: WeightMap) -> Self {
        Self {
            tensors,
            used: BTreeSet::new(),
        }
    }

    /// Fetch one tensor as an f32 MLX array, recording the lookup.
    pub fn get(&mut self, name: &str) -> Result<Array> {
        self.used.insert(name.to_string());
        let t = self.tensors.get(name)?;
        Ok(array_from_tensor(t))
    }

    /// Fetch one tensor as an f32 MLX array, with context in the error.
    pub fn get_in(&mut self, context: &str, name: &str) -> Result<Array> {
        self.used.insert(name.to_string());
        let t = self.tensors.get_in(context, name)?;
        Ok(array_from_tensor(t))
    }

    /// Raw f32 view of one tensor, recording the lookup (shape checks and
    /// similar introspection that does not need an MLX array).
    pub fn tensor(&mut self, name: &str) -> Result<&WeightTensor> {
        self.used.insert(name.to_string());
        self.tensors.get(name)
    }

    /// True when the checkpoint contains `name` (does not mark it used).
    pub fn has(&self, name: &str) -> bool {
        self.tensors.names().any(|n| n == name)
    }

    /// Tensors never looked up — a non-empty list means the port is
    /// incomplete.
    pub fn unused(&self) -> Vec<&str> {
        self.tensors
            .names()
            .filter(|n| !self.used.contains(n.as_str()))
            .map(|n| n.as_str())
            .collect()
    }
}

/// Wrap a row-major f32 tensor as an MLX array.
fn array_from_tensor(t: &WeightTensor) -> Array {
    let shape: Vec<i32> = t.shape.iter().map(|&d| d as i32).collect();
    Array::from_slice(&t.data, &shape)
}

/// A scalar f32 array (`shape == []`) for broadcast use.
pub fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[])
}

/// `nn.Linear`: `y = x @ W.T + b` (weights kept in PyTorch layout).
pub struct Linear {
    weight: Array,
    bias: Option<Array>,
    /// Input feature count (`weight.shape[1]`).
    pub in_features: usize,
    /// Output feature count (`weight.shape[0]`).
    pub out_features: usize,
}

impl Linear {
    /// Load `prefix.weight` (and `prefix.bias` when `bias`).
    pub fn new(store: &mut WeightStore, prefix: &str, bias: bool) -> Result<Self> {
        let weight = store.get_in(prefix, &format!("{prefix}.weight"))?;
        let bias = if bias {
            Some(store.get_in(prefix, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        let in_features = weight.shape()[1] as usize;
        let out_features = weight.shape()[0] as usize;
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Apply the affine transform.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = x.matmul(&self.weight.transpose()?)?;
        Ok(match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }
}

/// `nn.LayerNorm`: biased variance, `eps = 1e-5` by default.
///
/// PyTorch normalises by `sqrt(var + eps)` with the *biased* variance
/// (divide by N, not N-1) and then applies `weight`/`bias`.
pub struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    /// Load `prefix.weight`/`prefix.bias`.
    pub fn new(store: &mut WeightStore, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: store.get_in(prefix, &format!("{prefix}.weight"))?,
            bias: store.get_in(prefix, &format!("{prefix}.bias"))?,
            eps: 1e-5,
        })
    }

    /// Normalise the last dimension.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mean = x.mean_axis(-1, true)?;
        let centred = ops::subtract(x, &mean)?;
        let var = ops::multiply(&centred, &centred)?.mean_axis(-1, true)?;
        let denom = ops::sqrt(&ops::add(&var, scalar(self.eps))?)?;
        let norm = ops::divide(&centred, &denom)?;
        Ok(ops::add(&ops::multiply(&norm, &self.weight)?, &self.bias)?)
    }
}

/// Exact (erf-based) GELU — `torch.nn.functional.gelu`'s default.
///
/// Deliberately *not* the tanh approximation (which `mx.nn.gelu` may use).
pub fn gelu(x: &Array) -> Result<Array> {
    let scaled = ops::divide(x, scalar(std::f32::consts::SQRT_2))?;
    let half = ops::multiply(x, scalar(0.5))?;
    let erf = ops::erf(&scaled)?;
    Ok(ops::multiply(&half, &ops::add(&erf, scalar(1.0))?)?)
}

/// `torch.nn.SiLU` == `x * sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(ops::multiply(x, &ops::sigmoid(x)?)?)
}

/// The reference `PositionalEncoding.pe` table, built in float32.
///
/// Mirrors `ardy/model/backbone.py:414-431` (and the identical copy in
/// `ardy/model/autoencoder/transformer.py`):
///
/// ```text
/// position = arange(max_len, float32)[:, None]
/// div_term = pow(10000.0, -arange(0, d_model, 2, float32) / d_model)
/// pe[:, 0::2] = sin(position * div_term)
/// pe[:, 1::2] = cos(position * div_term)
/// ```
///
/// Row-major `[max_len, d_model]`. Built on the CPU side so the float32
/// arithmetic matches the numpy reference element for element.
pub fn sinusoidal_pe_table(d_model: usize, max_len: usize) -> Vec<f32> {
    let div = pe_div_term(d_model);
    let mut pe = vec![0.0f32; max_len * d_model];
    for pos in 0..max_len {
        fill_pe_row(
            &mut pe[pos * d_model..(pos + 1) * d_model],
            pos as f32,
            &div,
        );
    }
    pe
}

/// `PositionalEncodingNegativeIndex.pe` — row-major `[2*max_len-1, d_model]`.
///
/// Rows `0..max_len-1` are the non-negative table; the rest are the reversed
/// negative table, so a Python-style negative index `-k` addresses row
/// `len - k` (`backbone.py:520-540`).
pub fn negative_pe_table(d_model: usize, max_len: usize) -> Vec<f32> {
    let div = pe_div_term(d_model);
    let mut pe = sinusoidal_pe_table(d_model, max_len);
    // pe_negative[:, 0::2] = sin(-position[1:] * div); rows appended in
    // reverse so the final row addresses index -1.
    for k in (1..max_len).rev() {
        let start = pe.len();
        pe.resize(start + d_model, 0.0);
        fill_pe_row(&mut pe[start..], -(k as f32), &div);
    }
    pe
}

/// `pow(10000.0, -arange(0, d_model, 2) / d_model)` in float32 — one entry
/// per even/odd column pair.
fn pe_div_term(d_model: usize) -> Vec<f32> {
    (0..d_model)
        .step_by(2)
        .map(|i| 10000.0f32.powf(-(i as f32) / (d_model as f32)))
        .collect()
}

/// One table row: `[sin(pos*div[0]), cos(pos*div[0]), sin(pos*div[1]), ...]`.
fn fill_pe_row(row: &mut [f32], position: f32, div: &[f32]) {
    for (k, &d) in div.iter().enumerate() {
        let arg = position * d;
        row[2 * k] = arg.sin();
        if 2 * k + 1 < row.len() {
            row[2 * k + 1] = arg.cos();
        }
    }
}

/// Non-learned sinusoidal PE added to a `[B, T, D]` sequence.
pub struct PositionalEncoding {
    pe: Array,
}

impl PositionalEncoding {
    /// Build the table (`d_model`, `max_len = 5000` in the reference).
    pub fn new(d_model: usize, max_len: usize) -> Self {
        Self {
            pe: Array::from_slice(
                &sinusoidal_pe_table(d_model, max_len),
                &[max_len as i32, d_model as i32],
            ),
        }
    }

    /// The underlying `[max_len, d_model]` table.
    pub fn table(&self) -> &Array {
        &self.pe
    }

    /// Add `pe[:T]` to every batch element.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let t = x.shape()[1];
        let positions = ops::arange::<i32, i32>(None, t, None)?;
        let rows = self.pe.take_axis(&positions, 0)?.expand_dims(0)?;
        Ok(ops::add(x, &rows)?)
    }
}

/// Sinusoidal PE whose indices may be negative (`backbone.py:482-556`).
pub struct PositionalEncodingNegativeIndex {
    pe: Array,
    table_len: i32,
}

impl PositionalEncodingNegativeIndex {
    /// Build the `[2*max_len-1, d_model]` table.
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let pe = negative_pe_table(d_model, max_len);
        let table_len = (2 * max_len - 1) as i32;
        Self {
            pe: Array::from_slice(&pe, &[table_len, d_model as i32]),
            table_len,
        }
    }

    /// Add the rows addressed by `index` (Python-style negative indexing).
    pub fn forward(&self, x: &Array, index: &Array) -> Result<Array> {
        let zero = Array::from_slice(&[0i32], &[]);
        let non_neg = ops::ge(index, &zero)?;
        let shifted = ops::add(index, Array::from_slice(&[self.table_len], &[]))?;
        let safe = ops::select(&non_neg, index, &shifted)?.as_dtype(Dtype::Int32)?;
        let rows = self.pe.take_axis(&safe, 0)?;
        Ok(ops::add(x, &rows)?)
    }
}

/// `nn.Embedding(max_len, d_model)` added position-wise
/// (`backbone.py:463-480`).
pub struct LearnedPositionalEncoding {
    weight: Array,
}

impl LearnedPositionalEncoding {
    /// Load `prefix.weight` (`[max_len, d_model]`).
    pub fn new(store: &mut WeightStore, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: store.get_in(prefix, &format!("{prefix}.weight"))?,
        })
    }

    /// Add the first `T` embedding rows to every batch element.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let t = x.shape()[1];
        let positions = ops::arange::<i32, i32>(None, t, None)?;
        let rows = self.weight.take_axis(&positions, 0)?.expand_dims(0)?;
        Ok(ops::add(x, &rows)?)
    }
}

/// Timestep embedder (`backbone.py:439-461`).
///
/// Subtle and easy to get wrong: the timestep is **not** embedded
/// sinusoidally from its value. `F.embedding(timesteps.int(), pe_table)`
/// *looks up row `t` of the sinusoidal table*, and that row is then passed
/// through `Linear -> SiLU -> Linear`. `timesteps` is therefore an integer
/// index into `[0, num_base_steps)`, not a continuous noise level.
pub struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    pe: Array,
}

impl TimestepEmbedder {
    /// Load `prefix.time_embed.{0,2}` and build the PE table for
    /// `latent_dim`.
    pub fn new(store: &mut WeightStore, prefix: &str, latent_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(store, &format!("{prefix}.time_embed.0"), true)?,
            linear2: Linear::new(store, &format!("{prefix}.time_embed.2"), true)?,
            pe: PositionalEncoding::new(latent_dim, 5000).table().clone(),
        })
    }

    /// Embed integer timesteps `[B]` into `[B, 1, D]`.
    pub fn forward(&self, timesteps: &Array) -> Result<Array> {
        let idx = timesteps.as_dtype(Dtype::Int32)?;
        let emb = self.pe.take_axis(&idx, 0)?;
        let h = self.linear1.forward(&emb)?;
        let h = self.linear2.forward(&silu(&h)?)?;
        Ok(h.expand_dims(1)?)
    }
}

/// Boolean -> additive float conversion: `true` becomes `-inf`.
fn additive(m: &Array) -> Result<Array> {
    Ok(ops::select(m, scalar(f32::NEG_INFINITY), scalar(0.0))?)
}

/// Replicate `SDPATransformerEncoderLayer._build_attn_mask`
/// (`backbone.py:111-139`).
///
/// Returns an additive float mask broadcasting to `[B, nhead, L, S]` with
/// `-inf` at ignored positions. Boolean masks use the PyTorch convention:
/// `true == ignore`. The padding mask contributes `[B, 1, 1, S]` and the
/// attention mask `[L, S]`, exactly as in the reference. `None` when both
/// inputs are `None`.
pub fn build_additive_mask(
    key_padding_mask: Option<&Array>,
    attn_mask: Option<&Array>,
) -> Result<Option<Array>> {
    let mut parts: Vec<Array> = Vec::new();
    if let Some(kpm) = key_padding_mask {
        parts.push(additive(kpm)?.expand_dims_axes(&[1, 2])?);
    }
    if let Some(am) = attn_mask {
        parts.push(additive(am)?);
    }
    match parts.len() {
        0 => Ok(None),
        _ => {
            let mut out = parts[0].clone();
            for extra in &parts[1..] {
                out += extra;
            }
            Ok(Some(out))
        }
    }
}

/// Scaled dot-product attention, `[B, H, L, D]` inputs.
///
/// Written out explicitly rather than delegating to a fused kernel so the
/// mask semantics and the `1/sqrt(head_dim)` scaling are unambiguous.
pub fn sdpa(q: &Array, k: &Array, v: &Array, mask: Option<&Array>) -> Result<Array> {
    let head_dim = q.shape()[3] as f32;
    let scale = 1.0 / head_dim.sqrt();
    let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)? * scale;
    let scores = match mask {
        Some(m) => scores + m,
        None => scores,
    };
    let weights = ops::softmax_axis(&scores, -1, None)?;
    Ok(weights.matmul(v)?)
}

/// `SDPATransformerEncoderLayer` (`backbone.py:58-175`).
///
/// Parameter names are shared with `nn.TransformerEncoderLayer`; the fused
/// `in_proj_weight` `[3*D, D]` is chunked into q, k, v exactly as PyTorch's
/// `MultiheadAttention` does.
pub struct TransformerEncoderLayer {
    d_model: usize,
    nhead: usize,
    head_dim: usize,
    norm_first: bool,
    in_proj_weight: Array,
    in_proj_bias: Array,
    out_proj_weight: Array,
    out_proj_bias: Array,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    /// Load the layer from `prefix`.
    pub fn new(
        store: &mut WeightStore,
        prefix: &str,
        d_model: usize,
        nhead: usize,
        norm_first: bool,
    ) -> Result<Self> {
        Ok(Self {
            d_model,
            nhead,
            head_dim: d_model / nhead,
            norm_first,
            in_proj_weight: store.get_in(prefix, &format!("{prefix}.self_attn.in_proj_weight"))?,
            in_proj_bias: store.get_in(prefix, &format!("{prefix}.self_attn.in_proj_bias"))?,
            out_proj_weight: store
                .get_in(prefix, &format!("{prefix}.self_attn.out_proj.weight"))?,
            out_proj_bias: store.get_in(prefix, &format!("{prefix}.self_attn.out_proj.bias"))?,
            linear1: Linear::new(store, &format!("{prefix}.linear1"), true)?,
            linear2: Linear::new(store, &format!("{prefix}.linear2"), true)?,
            norm1: LayerNorm::new(store, &format!("{prefix}.norm1"))?,
            norm2: LayerNorm::new(store, &format!("{prefix}.norm2"))?,
        })
    }

    fn sa_block(&self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let b = x.shape()[0];
        let seq_len = x.shape()[1];
        let qkv = ops::add(
            &x.matmul(&self.in_proj_weight.transpose()?)?,
            &self.in_proj_bias,
        )?;
        let qkv = qkv.split_equal(3, -1)?;
        let heads = [b, seq_len, self.nhead as i32, self.head_dim as i32];
        let shape_heads =
            |a: &Array| -> Result<Array> { Ok(a.reshape(&heads)?.transpose_axes(&[0, 2, 1, 3])?) };
        let attn = sdpa(
            &shape_heads(&qkv[0])?,
            &shape_heads(&qkv[1])?,
            &shape_heads(&qkv[2])?,
            mask,
        )?;
        let merged = [b, seq_len, self.d_model as i32];
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&merged)?;
        Ok(ops::add(
            &attn.matmul(&self.out_proj_weight.transpose()?)?,
            &self.out_proj_bias,
        )?)
    }

    fn ff_block(&self, x: &Array) -> Result<Array> {
        self.linear2.forward(&gelu(&self.linear1.forward(x)?)?)
    }

    /// Run the layer.
    pub fn forward(&self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        if self.norm_first {
            let x = ops::add(x, &self.sa_block(&self.norm1.forward(x)?, mask)?)?;
            return Ok(ops::add(&x, &self.ff_block(&self.norm2.forward(&x)?)?)?);
        }
        let x = self
            .norm1
            .forward(&ops::add(x, &self.sa_block(x, mask)?)?)?;
        self.norm2.forward(&ops::add(&x, &self.ff_block(&x)?)?)
    }
}

/// `SDPATransformerEncoder`: N identical layers, optional final norm.
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    /// Load `num_layers` layers from `prefix.layers.{i}`.
    pub fn new(
        store: &mut WeightStore,
        prefix: &str,
        num_layers: usize,
        d_model: usize,
        nhead: usize,
        norm_first: bool,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|i| {
                TransformerEncoderLayer::new(
                    store,
                    &format!("{prefix}.layers.{i}"),
                    d_model,
                    nhead,
                    norm_first,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// Run every layer in order.
    pub fn forward(&self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Build a WeightStore from (name, shape, data) specs via a synthesized
    /// safetensors buffer.
    fn store_from(specs: &[(&str, Vec<usize>, Vec<f32>)]) -> WeightStore {
        let mut header = String::from("{");
        let mut payload: Vec<u8> = Vec::new();
        for (i, (name, shape, data)) in specs.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let start = payload.len();
            for v in data {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            let shape_json: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header.push_str(&format!(
                "{:?}: {{\"dtype\": \"F32\", \"shape\": [{}], \"data_offsets\": [{}, {}]}}",
                name,
                shape_json.join(","),
                start,
                payload.len()
            ));
        }
        header.push('}');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        let map = WeightMap::parse(Path::new("test.safetensors"), &bytes).unwrap();
        WeightStore::new(map)
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gelu_and_silu_match_reference_values() {
        let x = Array::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[4]);
        let g = gelu(&x).unwrap();
        let g = g.as_slice::<f32>();
        // Exact-erf GELU: gelu(0)=0, gelu(1)=0.8413447, gelu(-1)=-0.1586553,
        // gelu(2)=1.9544997.
        for (got, want) in g.iter().zip([0.0, 0.8413447, -0.1586553, 1.9544997]) {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
        let s = silu(&x).unwrap();
        let s = s.as_slice::<f32>();
        // SiLU: silu(0)=0, silu(1)=0.7310586, silu(-1)=-0.2689414,
        // silu(2)=1.7615942.
        for (got, want) in s.iter().zip([0.0, 0.7310586, -0.2689414, 1.7615942]) {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
    }

    #[test]
    fn sinusoidal_pe_table_matches_reference_layout() {
        // d_model=4: div_term = [1.0, 0.01]; rows are [sin, cos, sin', cos'].
        let pe = sinusoidal_pe_table(4, 3);
        assert_eq!(pe.len(), 12);
        let row = |i: usize| &pe[i * 4..(i + 1) * 4];
        assert_eq!(row(0), &[0.0, 1.0, 0.0, 1.0]);
        let r1 = row(1);
        for (got, want) in r1
            .iter()
            .zip([0.841_471, 0.540_302_3, 0.009_999_834, 0.999_950_05])
        {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
        let r2 = row(2);
        for (got, want) in
            r2.iter()
                .zip([0.909_297_4f32, -0.416_146_87, 0.019_998_667, 0.999_800_1])
        {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
    }

    #[test]
    fn negative_pe_table_addresses_python_negative_indices() {
        let d = 4;
        let max_len = 5;
        let pe = negative_pe_table(d, max_len);
        assert_eq!(pe.len(), (2 * max_len - 1) * d);
        let row = |i: usize| &pe[i * d..(i + 1) * d];
        // Row 0 is position 0; row len-1 is position -1; row len-2 is -2.
        let r_neg1 = row(2 * max_len - 2);
        for (got, want) in
            r_neg1
                .iter()
                .zip([-0.841_471, 0.540_302_3, -0.009_999_834, 0.999_950_05])
        {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
        let r_neg2 = row(2 * max_len - 3);
        for (got, want) in
            r_neg2
                .iter()
                .zip([-0.909_297_4f32, -0.416_146_87, -0.019_998_667, 0.999_800_1])
        {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }
        // Forward pass: index [-1, 0, 1] picks rows len-1, 0, 1.
        let enc = PositionalEncodingNegativeIndex::new(d, max_len);
        let x = Array::from_slice(&vec![0.0f32; 3 * d], &[1, 3, d as i32]);
        let index = Array::from_slice(&[-1i32, 0, 1], &[3]);
        let out = enc.forward(&x, &index).unwrap();
        let out = out.as_slice::<f32>();
        for (k, want_row) in [row(2 * max_len - 2), row(0), row(1)]
            .into_iter()
            .enumerate()
        {
            for j in 0..d {
                assert!(close(out[k * d + j], want_row[j], 1e-6));
            }
        }
    }

    #[test]
    fn linear_forward_hand_example() {
        // W [2,3] (PyTorch layout), b [2], x [1,3]: y = x @ W.T + b.
        let mut store = store_from(&[
            ("l.weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("l.bias", vec![2], vec![0.5, -0.5]),
        ]);
        let lin = Linear::new(&mut store, "l", true).unwrap();
        assert_eq!(lin.in_features, 3);
        assert_eq!(lin.out_features, 2);
        let x = Array::from_slice(&[1.0f32, 0.0, -1.0], &[1, 3]);
        let y = lin.forward(&x).unwrap();
        // x@W.T = [1*1 + 0*2 + -1*3, 1*4 + 0*5 + -1*6] = [-2, -2]; +b.
        let y = y.as_slice::<f32>();
        assert!(close(y[0], -1.5, 1e-6));
        assert!(close(y[1], -2.5, 1e-6));

        // No-bias variant.
        let lin = Linear::new(
            &mut store_from(&[("l.weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]),
            "l",
            false,
        )
        .unwrap();
        let y = lin.forward(&x).unwrap();
        let y = y.as_slice::<f32>();
        assert!(close(y[0], -2.0, 1e-6));
        assert!(close(y[1], -2.0, 1e-6));
    }

    #[test]
    fn layernorm_biased_variance_and_constant_row() {
        // Constant row: variance 0 -> output is exactly the bias.
        let mut store = store_from(&[
            ("n.weight", vec![3], vec![2.0, 2.0, 2.0]),
            ("n.bias", vec![3], vec![10.0, 20.0, 30.0]),
        ]);
        let ln = LayerNorm::new(&mut store, "n").unwrap();
        let x = Array::from_slice(&[7.0f32, 7.0, 7.0], &[1, 3]);
        let y = ln.forward(&x).unwrap();
        let y = y.as_slice::<f32>();
        for (got, want) in y.iter().zip([10.0f32, 20.0, 30.0]) {
            assert!(close(*got, want, 1e-5), "{got} vs {want}");
        }

        // Non-trivial row, computed in f64 with the biased variance.
        let x0 = [1.0f32, 2.0, 9.0];
        let mean = 4.0;
        let var = ((1.0 - 4.0f64).powi(2) + 4.0 + 25.0) / 3.0;
        let w = [1.5f32, -0.5, 2.0];
        let b = [0.1f32, 0.2, 0.3];
        let mut store = store_from(&[
            ("n.weight", vec![3], w.to_vec()),
            ("n.bias", vec![3], b.to_vec()),
        ]);
        let ln = LayerNorm::new(&mut store, "n").unwrap();
        let x = Array::from_slice(&x0, &[1, 3]);
        let y = ln.forward(&x).unwrap();
        let y = y.as_slice::<f32>();
        for j in 0..3 {
            let want = ((x0[j] as f64 - mean) / (var + 1e-5).sqrt()) * w[j] as f64 + b[j] as f64;
            assert!((y[j] as f64 - want).abs() < 1e-5, "{} vs {want}", y[j]);
        }
    }

    #[test]
    fn sdpa_matches_manual_softmax_and_honours_mask() {
        // One head, L=2, D=2; verify against an f64 manual computation.
        let q = Array::from_slice(&[0.5f32, -0.5, 1.0, 0.0], &[1, 1, 2, 2]);
        let k = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let v = Array::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[1, 1, 2, 2]);
        let out = sdpa(&q, &k, &v, None).unwrap();
        // scores = q@k.T / sqrt(2) = [[0.5, -0.5], [1.0, 0.0]] / sqrt(2).
        let s0 = 0.5f64 / (2.0f64).sqrt();
        let s1 = -0.5f64 / (2.0f64).sqrt();
        let s2 = 1.0f64 / (2.0f64).sqrt();
        let s3 = 0.0f64;
        let exp = |x: f64| x.exp();
        let row = |a: f64, b: f64| {
            let z = exp(a) + exp(b);
            (exp(a) / z, exp(b) / z)
        };
        let (w00, w01) = row(s0, s1);
        let (w10, w11) = row(s2, s3);
        let out = out.as_slice::<f32>();
        // out[q][d] = sum_k w[q][k] * v[k][d], layout row-major [1,1,2,2].
        let want = [
            (w00 * 2.0 + w01 * 6.0) as f32,
            (w00 * 4.0 + w01 * 8.0) as f32,
            (w10 * 2.0 + w11 * 6.0) as f32,
            (w10 * 4.0 + w11 * 8.0) as f32,
        ];
        for (got, want) in out.iter().zip(want) {
            assert!(close(*got, want, 1e-6), "{got} vs {want}");
        }

        // -inf on key 1 -> all weight on key 0 -> out == v[0] everywhere.
        let mask = additive(
            &Array::from_slice(&[0.0f32, 1.0], &[1, 1, 1, 2])
                .as_dtype(Dtype::Bool)
                .unwrap(),
        )
        .unwrap();
        let out = sdpa(&q, &k, &v, Some(&mask)).unwrap();
        let out = out.as_slice::<f32>();
        assert!(close(out[0], 2.0, 1e-6));
        assert!(close(out[1], 4.0, 1e-6));
        assert!(close(out[2], 2.0, 1e-6));
        assert!(close(out[3], 4.0, 1e-6));
    }

    #[test]
    fn additive_mask_broadcasts_like_the_reference() {
        // [B,S] padding + [L,S] attention -> [B,1,L,S] additive mask.
        // Self-attention is square (L == S). b0 clean; b1 ignores step 1.
        let kpm = Array::from_slice(&[0.0f32, 0.0, 0.0, 1.0], &[2, 2])
            .as_dtype(Dtype::Bool)
            .unwrap();
        let am = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[2, 2])
            .as_dtype(Dtype::Bool)
            .unwrap();
        let m = build_additive_mask(Some(&kpm), Some(&am)).unwrap().unwrap();
        assert_eq!(m.shape(), &[2, 1, 2, 2]);
        let m = m.as_slice::<f32>();
        // Layout [B,1,L,S]: idx = (b*L + q)*S + s. kpm adds -inf at b1/step-1;
        // am adds -inf at key-1 for every query.
        let idx = |b: usize, q: usize, s: usize| (b * 2 + q) * 2 + s;
        assert_eq!(m[idx(0, 0, 0)], 0.0);
        assert!(m[idx(0, 0, 1)].is_infinite()); // attn
        assert!(m[idx(1, 1, 1)].is_infinite()); // kpm + attn
        assert_eq!(m[idx(1, 0, 0)], 0.0); // kpm only at s1, not s0

        assert!(build_additive_mask(None, None).unwrap().is_none());
    }

    #[test]
    fn encoder_layer_runs_both_norm_orders() {
        // D=4, nhead=2, ff=8, one layer; shape-only + finiteness contract.
        let d = 4usize;
        let mut specs: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            (
                "e.self_attn.in_proj_weight".into(),
                vec![3 * d, d],
                vec![0.1; 3 * d * d],
            ),
            (
                "e.self_attn.in_proj_bias".into(),
                vec![3 * d],
                vec![0.0; 3 * d],
            ),
            (
                "e.self_attn.out_proj.weight".into(),
                vec![d, d],
                vec![0.1; d * d],
            ),
            ("e.self_attn.out_proj.bias".into(), vec![d], vec![0.0; d]),
            ("e.linear1.weight".into(), vec![8, d], vec![0.1; 8 * d]),
            ("e.linear1.bias".into(), vec![8], vec![0.0; 8]),
            ("e.linear2.weight".into(), vec![d, 8], vec![0.1; d * 8]),
            ("e.linear2.bias".into(), vec![d], vec![0.0; d]),
            ("e.norm1.weight".into(), vec![d], vec![1.0; d]),
            ("e.norm1.bias".into(), vec![d], vec![0.0; d]),
            ("e.norm2.weight".into(), vec![d], vec![1.0; d]),
            ("e.norm2.bias".into(), vec![d], vec![0.0; d]),
        ];
        let specs: Vec<(&str, Vec<usize>, Vec<f32>)> = specs
            .iter_mut()
            .map(|(n, s, v)| (n.as_str(), s.clone(), std::mem::take(v)))
            .collect();
        for norm_first in [false, true] {
            let mut store = store_from(&specs);
            let layer = TransformerEncoderLayer::new(&mut store, "e", d, 2, norm_first).unwrap();
            let x = Array::from_slice(
                &(0..20).map(|i| (i as f32) * 0.25 - 2.0).collect::<Vec<_>>(),
                &[1, 5, d as i32],
            );
            let y = layer.forward(&x, None).unwrap();
            assert_eq!(y.shape(), &[1, 5, d as i32]);
            assert!(y.as_slice::<f32>().iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn weightstore_tracks_unused_and_missing() {
        let mut store = store_from(&[
            ("a.w", vec![1, 2], vec![1.0, 2.0]),
            ("b.w", vec![2], vec![3.0, 4.0]),
        ]);
        assert!(store.has("a.w"));
        assert!(!store.has("c.w"));
        assert!(store.get("a.w").is_ok());
        assert!(store.get("zzz").is_err());
        assert_eq!(store.unused(), vec!["b.w"]);
    }
}
