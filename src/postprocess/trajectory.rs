//! Trajectory correction — port of upstream
//! `MotionCorrection/src/cpp/AnimProcessing/TrajectoryCorrector.cpp`.
//!
//! ADMM interpolation (Overby'17 equations 8–10) of a 1-D-per-column
//! trajectory that hits pinned observations exactly, stays within margin
//! radii around margin observations, and otherwise stays close to the
//! reference in position, velocity and acceleration. All math in `f64`,
//! matching the upstream `Eigen` dense/s `Sparse<double>` pipeline; the
//! sparse operators are small enough (N = frame count) that dense rows and
//! a dense LU with partial pivoting reproduce `SparseLU` to ~1e-12.

/// Difference operators `V = T - I` (row 0..N-2 kept) and `A = 2T - I - T²`
/// (rows 1..N-3 kept), as dense row-major `(rows × N)`.
///
/// NOTE: upstream's `multVelWeights` computes weighted triplets but never
/// writes them back to `V` (no `setFromTriplets` — verified against the real
/// Eigen binary: `computeDiffMats(V, A, N, vw=(2,3,5,7,11,13))` returns V
/// unweighted). The velocity weighting is therefore dead code upstream and
/// `velocity_weights` is accepted here only for signature fidelity.
struct DiffMats {
    /// `V` rows (row-major, `v_nrows × N`).
    v: Vec<f64>,
    /// Kept row count of `V`.
    v_nrows: usize,
    /// `A` rows (row-major, `a_nrows × N`).
    a: Vec<f64>,
    /// Kept row count of `A`.
    a_nrows: usize,
}

impl DiffMats {
    fn new(n: usize, velocity_weights: Option<&[f64]>) -> Self {
        let _ = velocity_weights; // upstream no-op, see struct doc
        // T: (i, i+1) = 1 for i in 0..N-1; V = T - I keeps rows with >= 2
        // nonzeros => rows 0..N-2 (row N-1 has only its diagonal).
        let v_nrows = n.saturating_sub(1);
        let mut v = vec![0.0; v_nrows * n];
        for i in 0..v_nrows {
            v[i * n + i] = -1.0;
            v[i * n + i + 1] = 1.0;
        }

        // A = 2T - I - T² keeps rows with >= 3 nonzeros => rows 0..N-3
        // (verified against Eigen: row i has nnz 3 for i ≤ N-3; rows
        // N-2 and N-1 have 2 and 1).
        let a_nrows = n.saturating_sub(2);
        let mut a = vec![0.0; a_nrows * n];
        for i in 0..a_nrows {
            a[i * n + i] = -1.0;
            a[i * n + i + 1] = 2.0;
            a[i * n + i + 2] = -1.0;
        }
        Self {
            v,
            v_nrows,
            a,
            a_nrows,
        }
    }

    /// `Mᵀ M` accumulated into an `N × N` accumulator.
    fn add_atam(&self, acc: &mut [f64], m: &[f64], m_nrows: usize, n: usize, weight: f64) {
        // (Aᵀ A)[i][j] = Σ_k M[k][i] M[k][j]
        for k in 0..m_nrows {
            let row = &m[k * n..(k + 1) * n];
            for i in 0..n {
                if row[i] == 0.0 {
                    continue;
                }
                for j in 0..n {
                    acc[i * n + j] += weight * row[i] * row[j];
                }
            }
        }
    }
}

/// The ported `TrajectoryCorrector`.
pub struct TrajectoryCorrector {
    /// `pos_weight·I + vel_weight·VᵀV + acc_weight·AᵀA` (row-major N×N).
    m_n: Vec<f64>,
    /// ADMM step size: `0.5·sqrt(max diag of m_N)`.
    admm_stepsize: f64,
    /// Iteration count (upstream default 100).
    admm_iters: u32,
    /// Frames with `margin > 0`.
    margin_locs: Vec<u32>,
    /// Their margin radii.
    margin_vals: Vec<f64>,
    /// Frames with `margin != 0` (unconstrained + margin).
    unconstrained_locs: Vec<u32>,
    /// Frames with `margin == 0` (pinned).
    constrained_locs: Vec<u32>,
    /// LU of the reduced `(S M Sᵀ)` system, with the pivot permutation.
    lu: Option<DenseLu>,
}

/// Dense LU with partial pivoting: `P A = L U`.
struct DenseLu {
    n: usize,
    /// Row-major LU factors in place (L unit-diagonal implicit).
    a: Vec<f64>,
    /// Row pivot order.
    perm: Vec<usize>,
}

impl DenseLu {
    fn factor(mut a: Vec<f64>, n: usize) -> Option<Self> {
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            // pivot: largest |value| in column k at or below row k
            let mut p = k;
            let mut best = a[k * n + k].abs();
            for i in k + 1..n {
                let v = a[i * n + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best == 0.0 {
                return None; // singular
            }
            perm.swap(k, p);
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            let pivot = a[k * n + k];
            for i in k + 1..n {
                let f = a[i * n + k] / pivot;
                a[i * n + k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= f * a[k * n + j];
                    }
                }
            }
        }
        Some(Self { n, a, perm })
    }

    /// Solve `A x = b` for a column block (`b` = rows × 1 here; the caller
    /// loops over columns).
    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        // apply permutation
        let mut x = vec![0.0; n];
        for (i, &pi) in self.perm.iter().enumerate() {
            x[i] = b[pi];
        }
        // forward: L y = Pb
        // (index loops over the flat LU storage are the clearest kernel here)
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let mut s = x[i];
            for j in 0..i {
                s -= self.a[i * n + j] * x[j];
            }
            x[i] = s;
        }
        // back: U x = y
        #[allow(clippy::needless_range_loop)]
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in i + 1..n {
                s -= self.a[i * n + j] * x[j];
            }
            x[i] = s / self.a[i * n + i];
        }
        x
    }
}

impl TrajectoryCorrector {
    /// Port of the upstream constructor. `margins[i] < 0` = unconstrained,
    /// `== 0` = pinned, `> 0` = bounded within that radius.
    pub fn new(
        margins: &[f64],
        pos_weight: f32,
        vel_weight: f32,
        acc_weight: f32,
        velocity_weights: Option<&[f64]>,
        admm_iters: u32,
    ) -> Self {
        let n = margins.len();
        let mut margin_locs = Vec::new();
        let mut margin_vals = Vec::new();
        let mut unconstrained_locs = Vec::new();
        let mut constrained_locs = Vec::new();
        for (i, &m) in margins.iter().enumerate() {
            if m > 0.0 {
                margin_locs.push(i as u32);
                margin_vals.push(m);
            }
            if m == 0.0 {
                constrained_locs.push(i as u32);
            } else {
                unconstrained_locs.push(i as u32);
            }
        }

        let diff = DiffMats::new(n, velocity_weights);
        let mut m_n = vec![0.0; n * n];
        diff.add_atam(&mut m_n, &diff.v, diff.v_nrows, n, vel_weight as f64);
        diff.add_atam(&mut m_n, &diff.a, diff.a_nrows, n, acc_weight as f64);
        for i in 0..n {
            m_n[i * n + i] += pos_weight as f64;
        }

        let diag_max = (0..n).fold(0.0f64, |mx, i| m_n[i * n + i].max(mx));
        let admm_stepsize = 0.5 * diag_max.sqrt();

        // M = m_N + stepsize·diag(margin_locs), reduced by the
        // unconstrained selection S: lhs = S M Sᵀ.
        let nu = unconstrained_locs.len();
        let mut full_m = m_n.clone();
        for &loc in &margin_locs {
            full_m[loc as usize * n + loc as usize] += admm_stepsize;
        }
        let mut lhs = vec![0.0; nu * nu];
        for (ri, &ui) in unconstrained_locs.iter().enumerate() {
            for (ci, &uj) in unconstrained_locs.iter().enumerate() {
                lhs[ri * nu + ci] = full_m[ui as usize * n + uj as usize];
            }
        }
        let lu = (nu > 0).then(|| DenseLu::factor(lhs, nu)).flatten();

        Self {
            m_n,
            admm_stepsize,
            admm_iters,
            margin_locs,
            margin_vals,
            unconstrained_locs,
            constrained_locs,
            lu,
        }
    }

    /// `Interpolate(ret, observations, ref_positions)` — `ret` and
    /// `ref_positions` are `N × C` row-major; `observations` likewise.
    pub fn interpolate(
        &self,
        observations: &[f64],
        ref_positions: &[f64],
        cols: usize,
    ) -> Vec<f64> {
        let n = ref_positions.len() / cols;
        if self.constrained_locs.is_empty() && self.margin_locs.is_empty() {
            return ref_positions.to_vec();
        }

        let mut x = vec![0.0; n * cols];
        if self.margin_locs.is_empty() {
            self.x_update(&mut x, &[], &[], ref_positions, observations, cols);
        } else {
            x.copy_from_slice(ref_positions);
            // z, z_t, u: (margin_locs × C)
            let nm = self.margin_locs.len();
            let mut z = vec![0.0; nm * cols];
            let mut u = vec![0.0; nm * cols];
            let mut z_t = vec![0.0; nm * cols];
            for (i, &loc) in self.margin_locs.iter().enumerate() {
                for j in 0..cols {
                    z_t[i * cols + j] = observations[loc as usize * cols + j];
                    z[i * cols + j] = ref_positions[loc as usize * cols + j];
                }
            }
            for _ in 0..self.admm_iters {
                self.x_update(&mut x, &z, &u, ref_positions, observations, cols);
                self.z_update(&mut z, &x, &z_t, &u, cols);
                self.u_update(&mut u, &x, &z, cols);
            }
        }
        x
    }

    /// `x_update(x, z, u, x_t, x_o)` — writes the solved trajectory into
    /// `x`.
    fn x_update(&self, x: &mut [f64], z: &[f64], u: &[f64], x_t: &[f64], x_o: &[f64], cols: usize) {
        let n = x_t.len() / cols;
        // x_diffs = x_t, with pinned frames replaced by (x_t - x_o)
        let mut x_diffs = x_t.to_vec();
        for &loc in &self.constrained_locs {
            for j in 0..cols {
                x_diffs[loc as usize * cols + j] -= x_o[loc as usize * cols + j];
            }
        }
        // r = m_N · x_diffs (+ stepsize·(z - u) at margin locs)
        let mut r = vec![0.0; n * cols];
        for i in 0..n {
            let mrow = &self.m_n[i * n..(i + 1) * n];
            for j in 0..cols {
                let mut s = 0.0;
                for k in 0..n {
                    s += mrow[k] * x_diffs[k * cols + j];
                }
                r[i * cols + j] = s;
            }
        }
        for (i, &loc) in self.margin_locs.iter().enumerate() {
            for j in 0..cols {
                r[loc as usize * cols + j] +=
                    self.admm_stepsize * (z[i * cols + j] - u[i * cols + j]);
            }
        }

        // reduce to unconstrained set, solve, scatter back (`lu` is `Some`
        // exactly when there are unconstrained locs, so one `if let` covers it)
        if let Some(lu) = &self.lu {
            for j in 0..cols {
                let rhs: Vec<f64> = self
                    .unconstrained_locs
                    .iter()
                    .map(|&loc| r[loc as usize * cols + j])
                    .collect();
                let sol = lu.solve(&rhs);
                for (i, &loc) in self.unconstrained_locs.iter().enumerate() {
                    x[loc as usize * cols + j] = sol[i];
                }
            }
        }
        for &loc in &self.constrained_locs {
            for j in 0..cols {
                x[loc as usize * cols + j] = x_o[loc as usize * cols + j];
            }
        }
    }

    /// `z_update(z, x, z_t, u)` — clamp the residual norm to the margin.
    fn z_update(&self, z: &mut [f64], x: &[f64], z_t: &[f64], u: &[f64], cols: usize) {
        for (i, &loc) in self.margin_locs.iter().enumerate() {
            for j in 0..cols {
                z[i * cols + j] = x[loc as usize * cols + j] + u[i * cols + j] - z_t[i * cols + j];
            }
            let mut norm = 0.0;
            for j in 0..cols {
                let v = z[i * cols + j];
                norm += v * v;
            }
            let norm = norm.sqrt();
            if norm > self.margin_vals[i] {
                for j in 0..cols {
                    z[i * cols + j] *= self.margin_vals[i] / norm;
                }
            }
            for j in 0..cols {
                z[i * cols + j] += z_t[i * cols + j];
            }
        }
    }

    /// `u_update(u, x, z)`
    fn u_update(&self, u: &mut [f64], x: &[f64], z: &[f64], cols: usize) {
        for (i, &loc) in self.margin_locs.iter().enumerate() {
            for j in 0..cols {
                u[i * cols + j] += x[loc as usize * cols + j] - z[i * cols + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-unconstrained corrector early-outs to the reference.
    #[test]
    fn unconstrained_is_identity() {
        let margins = vec![-1.0; 16];
        let c = TrajectoryCorrector::new(&margins, 0.001, 1.0, 10.0, None, 100);
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.37).sin()).collect();
        let zeros = vec![0.0; 16];
        let out = c.interpolate(&zeros, &x, 1);
        assert!(out.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-15));
    }

    /// Pinned endpoints must be hit exactly and the middle must smooth
    /// toward the reference.
    #[test]
    fn pinned_endpoints_are_hit() {
        let n = 12;
        let mut margins = vec![-1.0; n];
        margins[0] = 0.0;
        margins[n - 1] = 0.0;
        let c = TrajectoryCorrector::new(&margins, 0.001, 1.0, 10.0, None, 100);
        // reference: a bump; observations: flat at the endpoints
        let reference: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.3).sin()).collect();
        let mut obs = vec![0.0; n];
        obs[0] = reference[0];
        obs[n - 1] = reference[n - 1];
        let out = c.interpolate(&obs, &reference, 1);
        assert!((out[0] - obs[0]).abs() < 1e-9);
        assert!((out[n - 1] - obs[n - 1]).abs() < 1e-9);
        // interior follows the reference closely (obs == ref there)
        for i in 1..n - 1 {
            assert!((out[i] - reference[i]).abs() < 1e-6, "frame {i}");
        }
    }

    /// A margin observation pulls the trajectory only within the radius.
    #[test]
    fn margin_bounds_are_respected() {
        let n = 10;
        let mut margins = vec![-1.0; n];
        margins[5] = 0.05; // allowed to deviate ≤ 0.05 from the target
        let c = TrajectoryCorrector::new(&margins, 0.001, 1.0, 10.0, None, 100);
        let reference = vec![0.0; n];
        let mut obs = vec![0.0; n];
        obs[5] = 10.0; // target far outside the margin
        let out = c.interpolate(&obs, &reference, 1);
        let dev = (out[5] - obs[5]).abs();
        assert!(dev <= 0.05 + 1e-9, "deviation {dev} exceeds margin");
    }

    /// Two columns interpolate independently — each column of the output
    /// equals the single-column solve of the same data.
    #[test]
    fn multi_column() {
        let n = 8;
        let margins = {
            let mut m = vec![-1.0; n];
            m[0] = 0.0;
            m
        };
        let c = TrajectoryCorrector::new(&margins, 0.001, 1.0, 10.0, None, 100);
        let reference = vec![1.0; n * 2];
        let mut obs = vec![1.0; n * 2];
        obs[0] = 4.0;
        obs[1] = -2.0;
        let out = c.interpolate(&obs, &reference, 2);
        assert!((out[0] - 4.0).abs() < 1e-9);
        assert!((out[1] - (-2.0)).abs() < 1e-9);

        // column 0 solves (ref = 1s, pin to 4); column 1 pin to −2.
        for (col, pin) in [(0usize, 4.0f64), (1, -2.0)] {
            let mut single_obs = vec![0.0; n];
            single_obs[0] = pin;
            let single_ref = vec![1.0; n];
            let single = c.interpolate(&single_obs, &single_ref, 1);
            for i in 0..n {
                assert!(
                    (out[i * 2 + col] - single[i]).abs() < 1e-12,
                    "frame {i} col {col}"
                );
            }
        }
    }
}

#[cfg(test)]
mod oracle {
    use super::*;

    /// Hand-transcribed from the numpy ADMM oracle (/tmp/admm_oracle.py):
    /// N=12, margins[i]=−1 except [3]=0.4, [8]=0; obs/ref ~ N(0,1) seed 7.
    /// The margin loc may deviate from its observation by at most the margin
    /// radius (out[3] = obs[3] − 0.4) and the pin is hit exactly.
    #[test]
    fn matches_numpy_oracle_mixed() {
        let margins: Vec<f64> = {
            let mut m = vec![-1.0; 12];
            m[3] = 0.4;
            m[8] = 0.0;
            m
        };
        let c = TrajectoryCorrector::new(&margins, 0.001, 1.0, 10.0, None, 100);
        let ref_: Vec<f64> = vec![
            0.0012301533574825742,
            0.2987455375084699,
            -0.2741378553622176,
            -0.8905918387572742,
            -0.45467078517172255,
            -0.9916465549964624,
            0.060143602597438485,
            1.3402152455545335,
            -0.49220651855132963,
            -0.6204748998199404,
            0.4898420501851982,
            0.35688700816006075,
        ];
        let obs: Vec<f64> = vec![
            0.10541424899789856,
            -0.9304680447082047,
            -0.02925182246327349,
            0.6953031944582878,
            -1.344214547285082,
            -0.45761576104021817,
            -1.901222739800844,
            -1.289537739784976,
            -1.8417350377917323,
            -0.23509113107468127,
            -1.2674464814437032,
            0.2712643588217015,
        ];
        let out = c.interpolate(&obs, &ref_, 1);
        let expected: Vec<f64> = vec![
            2.155518148841,
            2.169361203777,
            1.284222819970,
            0.295303194458,
            0.260743064101,
            -0.801813344021,
            -0.293368678428,
            0.461131750786,
            -1.841735037792,
            -2.342369815971,
            -1.544184624378,
            -1.960685122039,
        ];
        for (i, (a, b)) in out.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-9, "frame {i}: {a} vs {b}");
        }
    }
}
