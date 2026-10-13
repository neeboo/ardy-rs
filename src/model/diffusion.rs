//! Cosine-schedule diffusion and the deterministic (eta = 0) DDIM step.
//!
//! Port of `ardy_mlx/model/diffusion.py` (reference: `ardy/model/diffusion.py`).
//!
//! Two index spaces are easy to conflate and the reference is careful about
//! them (`ardy.py:340-342`):
//!
//! * the **loop index** `i` runs `num_steps-1 ..= 0` and indexes the
//!   *subsampled* schedule (`use_timesteps`);
//! * the **mapped timestep** `map_tensor[i]` indexes the *base* buffers and is
//!   what the denoiser's `TimestepEmbedder` consumes.
//!
//! For the released Core checkpoint `num_steps == num_base_steps == 10`, so the
//! two coincide (`space_timesteps` is the identity) -- exactly why conflating
//! them is easy to miss.

use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::TryIndexOp;

use super::denoiser::{AutoLatentTwostageDenoiser, Window, cfg_guided};
use crate::{Error, Result};

/// Cosine `alpha_bar` (`diffusion.py:16-18`), computed in f64 like the Python
/// reference before the f32 store.
fn alpha_bar(t: f64) -> f64 {
    ((t + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2)
        .cos()
        .powi(2)
}

/// Cosine beta schedule (`get_beta_schedule`, `diffusion.py:12-26`).
fn beta_schedule(num_diffusion_timesteps: usize, max_beta: f64) -> Vec<f32> {
    let n = num_diffusion_timesteps as f64;
    (0..num_diffusion_timesteps)
        .map(|i| {
            let t1 = i as f64 / n;
            let t2 = (i + 1) as f64 / n;
            (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta) as f32
        })
        .collect()
}

/// The diffusion process: base cosine schedule plus the derived per-step
/// variables for the active subsampled schedule.
pub struct Diffusion {
    /// Base schedule length (`num_base_steps`, 10 for the Core checkpoint).
    pub num_base_steps: usize,
    /// Base betas `[T]`.
    pub betas_base: Array,
    /// Cumulative product of `1 - betas_base` `[T]`.
    pub alphas_cumprod_base: Array,
    /// Betas of the active schedule `[S]`.
    pub betas: Array,
    /// `1 - betas` `[S]`.
    pub alphas: Array,
    /// Cumulative product of `alphas`, clamped at 1e-9 `[S]`.
    pub alphas_cumprod: Array,
    /// `[1, alphas_cumprod[:-1]]` `[S]`.
    pub alphas_cumprod_prev: Array,
    /// `rsqrt(alphas_cumprod)` `[S]`.
    pub sqrt_recip_alphas_cumprod: Array,
    /// `rsqrt(alphas_cumprod / (1 - alphas_cumprod))` `[S]`.
    pub sqrt_recipm1_alphas_cumprod: Array,
    /// `betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)` `[S]`.
    pub posterior_variance: Array,
    /// `rsqrt(1 / alphas_cumprod)` `[S]`.
    pub sqrt_alphas_cumprod: Array,
    /// `rsqrt(1 / (1 - alphas_cumprod))` `[S]`.
    pub sqrt_one_minus_alphas_cumprod: Array,
    /// Loop-index -> base-timestep map of the active schedule `[S]`.
    pub map_tensor: Array,
}

impl Diffusion {
    /// Build the base schedule and immediately activate the full-length
    /// schedule (the reference constructor does the same).
    pub fn new(num_base_steps: usize) -> Result<Self> {
        let betas_base = Array::from_slice(
            &beta_schedule(num_base_steps, 0.999),
            &[num_base_steps as i32],
        );
        let alphas_cumprod_base =
            ops::cumprod(&ops::subtract(scalar(1.0), &betas_base)?, 0, None, None)?;
        let mut d = Self {
            num_base_steps,
            betas_base,
            alphas_cumprod_base,
            betas: Array::zeros::<f32>(&[0])?,
            alphas: Array::zeros::<f32>(&[0])?,
            alphas_cumprod: Array::zeros::<f32>(&[0])?,
            alphas_cumprod_prev: Array::zeros::<f32>(&[0])?,
            sqrt_recip_alphas_cumprod: Array::zeros::<f32>(&[0])?,
            sqrt_recipm1_alphas_cumprod: Array::zeros::<f32>(&[0])?,
            posterior_variance: Array::zeros::<f32>(&[0])?,
            sqrt_alphas_cumprod: Array::zeros::<f32>(&[0])?,
            sqrt_one_minus_alphas_cumprod: Array::zeros::<f32>(&[0])?,
            map_tensor: Array::zeros::<i32>(&[0])?,
        };
        let (use_timesteps, map_tensor) = d.space_timesteps(num_base_steps)?;
        d.calc_diffusion_vars(&use_timesteps)?;
        d.map_tensor = map_tensor;
        Ok(d)
    }

    /// `space_timesteps` (`diffusion.py:50-58`): evenly spaced loop indices for
    /// a `num_denoising_steps`-step schedule, plus the loop->base map.
    pub fn space_timesteps(&self, num_denoising_steps: usize) -> Result<(Array, Array)> {
        let n = self.num_base_steps;
        if num_denoising_steps < 1 {
            return Err(Error::Contract(format!(
                "num_denoising_steps must be >= 1, got {num_denoising_steps}"
            )));
        }
        // (nsteps_train - 1) / max(1, num_denoising_steps - 1), f32 like the
        // reference's float32 arange multiply.
        let frac_stride = (n as f32 - 1.0) / (num_denoising_steps as f32 - 1.0).max(1.0);
        let idx = ops::arange::<f32, f32>(None, n as f32, None)?;
        let use_timesteps = ops::round(&ops::multiply(&idx, scalar(frac_stride))?, None)?;
        let use_timesteps = ops::minimum(
            &use_timesteps.as_dtype(mlx_rs::Dtype::Int32)?,
            scalar_i(n as i32 - 1),
        )?;
        let map_tensor = ops::arange::<i32, i32>(None, n as i32, None)?.take(&use_timesteps)?;
        Ok((use_timesteps, map_tensor))
    }

    /// `calc_diffusion_vars` (`diffusion.py:60-92`): derive the active
    /// schedule's variables from `alphas_cumprod_base[use_timesteps]`.
    pub fn calc_diffusion_vars(&mut self, use_timesteps: &Array) -> Result<()> {
        let ones = Array::from_slice(&[1.0f32], &[1]);
        let alphas_cumprod_base = self.alphas_cumprod_base.take(use_timesteps)?;
        // [:-1] on the 1-D schedule vectors
        let s = alphas_cumprod_base.shape()[0];
        let ac_tail = alphas_cumprod_base.try_index((0i32..s - 1,))?;
        let last_alpha_cumprod = ops::concatenate(&[&ones, &ac_tail], 0)?;
        self.betas = ops::subtract(
            scalar(1.0),
            &ops::divide(&alphas_cumprod_base, &last_alpha_cumprod)?,
        )?;

        self.alphas = ops::subtract(scalar(1.0), &self.betas)?;
        let alphas_cumprod = ops::clip(
            &ops::cumprod(&self.alphas, 0, None, None)?,
            (scalar(1e-9), ()),
        )?;
        let n = alphas_cumprod.shape()[0] as i32;
        let ac_prev_tail = alphas_cumprod.try_index((0i32..n - 1,))?;
        self.alphas_cumprod = alphas_cumprod;
        self.alphas_cumprod_prev = ops::concatenate(&[&ones, &ac_prev_tail], 0)?;
        self.sqrt_recip_alphas_cumprod = ops::rsqrt(self.alphas_cumprod.clone())?;
        self.sqrt_recipm1_alphas_cumprod = ops::rsqrt(&ops::divide(
            self.alphas_cumprod.clone(),
            &ops::subtract(scalar(1.0), self.alphas_cumprod.clone())?,
        )?)?;
        self.posterior_variance = ops::multiply(
            self.betas.clone(),
            &ops::divide(
                &ops::subtract(scalar(1.0), self.alphas_cumprod_prev.clone())?,
                &ops::subtract(scalar(1.0), self.alphas_cumprod.clone())?,
            )?,
        )?;
        self.sqrt_alphas_cumprod =
            ops::rsqrt(&ops::divide(scalar(1.0), self.alphas_cumprod.clone())?)?;
        self.sqrt_one_minus_alphas_cumprod = ops::rsqrt(&ops::divide(
            scalar(1.0),
            &ops::subtract(scalar(1.0), self.alphas_cumprod.clone())?,
        )?)?;
        Ok(())
    }
}

/// `DDIMSampler` (`diffusion.py:113-133`): one deterministic eta=0 update.
///
/// `t` is the **loop index** into the active schedule's buffers.
pub fn ddim_step(
    diffusion: &Diffusion,
    x_t: &Array,
    pred_xstart: &Array,
    t: usize,
) -> Result<Array> {
    let t = t as i32;
    let rec = &diffusion.sqrt_recip_alphas_cumprod;
    let recm1 = &diffusion.sqrt_recipm1_alphas_cumprod;
    let eps = ops::divide(
        &ops::subtract(&ops::multiply(&rec.try_index((t,))?, x_t)?, pred_xstart)?,
        &recm1.try_index((t,))?,
    )?;
    let alpha_bar_prev = diffusion.alphas_cumprod_prev.try_index((t,))?;
    Ok(ops::add(
        &ops::multiply(pred_xstart, &ops::sqrt(&alpha_bar_prev)?)?,
        &ops::multiply(
            &eps,
            &ops::sqrt(&ops::subtract(scalar(1.0), &alpha_bar_prev)?)?,
        )?,
    )?)
}

/// `Ardy.denoising_step` (`ardy_model.py:231-313`): one full denoising step.
///
/// Composes [`cfg_guided`] (the batch-3 CFG wrapper) with the DDIM update
/// applied **to the generation tokens only**; history and future tokens pass
/// through unchanged (`ardy.py:344-348`).
///
/// `w.timesteps` must already carry the *mapped* timestep the denoiser embeds;
/// `sampler_t` is the loop index the sampler indexes the buffers with. The
/// generation span is `[history_token_end, generation_token_end)` tokens
/// (contiguous in every window the released pipeline builds, so the reference's
/// boolean-mask gather reduces to this slice).
#[allow(clippy::too_many_arguments)]
pub fn denoising_step(
    denoiser: &AutoLatentTwostageDenoiser,
    diffusion: &Diffusion,
    w: &Window,
    history_token_end: usize,
    generation_token_end: usize,
    sampler_t: usize,
    cfg_weight_text: f32,
    cfg_weight_cstr: f32,
) -> Result<Array> {
    let clean = cfg_guided(denoiser, cfg_weight_text, cfg_weight_cstr, w)?;
    let x = &w.x;
    let shape = x.shape();
    let gen_t = x.try_index((
        (0i32..),
        (history_token_end as i32..generation_token_end as i32),
        (0i32..),
    ))?;
    let gen_clean = clean.try_index((
        (0i32..),
        (history_token_end as i32..generation_token_end as i32),
        (0i32..),
    ))?;
    let gen_next = ddim_step(diffusion, &gen_t, &gen_clean, sampler_t)?;
    Ok(ops::concatenate(
        &[
            &x.try_index(((0i32..), (0i32..history_token_end as i32), (0i32..)))?,
            &gen_next,
            &x.try_index(((0i32..), (generation_token_end as i32..shape[1]), (0i32..)))?,
        ],
        1,
    )?)
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[])
}

fn scalar_i(v: i32) -> Array {
    Array::from_slice(&[v], &[])
}
