//! Scalar (non-MLX) math for the MotionCorrection port.
//!
//! Literal port of the upstream C++ `MotionCorrection/src/cpp/Math`
//! primitives (`Quaternion`, `Vector`, `Transform`), preserving the
//! upstream engine's conventions exactly (decoded from the SSE sources):
//!
//! * quaternions store `(x, y, z, w)`;
//! * `Quat::mul` is the standard Hamilton product, but the C++ `operator*`
//!   is the *reversed* DirectXMath order — `a * b` = Hamilton `b ⊗ a` — so
//!   the `Mul` impl here mirrors that reversal and every `a * b` from the
//!   C++ translates literally;
//! * `rotate_vector` applies `q ⊗ v ⊗ q*` (the standard matrix sense) and
//!   `rotate_vector_inverse` applies `q* ⊗ v ⊗ q`;
//! * `Transform::mul(a, b)` is the standard rigid composition `B ∘ A`
//!   (apply `a` first, then `b`): `q = normalize(b.q ⊗ a.q)`,
//!   `t = R_b·a.t + b.t`;
//! * `delta_to_other` is `self⁻¹ ∘ target` (a local-frame delta), so
//!   `mul(delta, other) = other ∘ delta` composes back.
//!
//! The differential probe tests (`tests/postprocess.rs`) pin every one of
//! these formulas against the real C++ compiled from upstream, so any
//! convention slip here fails loudly instead of silently bending poses.

/// 3-vector of `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {i} out of range"),
        }
    }
}

impl Vec3 {
    /// `Vector::Zero` — `(0, 0, 0)`.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    // `add`/`sub` are inherent C++-mirror names (`Vector::Add`/`Sub`), not the
    // std ops traits — silence the trait-method confusion lint.

    /// `(x, y, z)`
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Componentwise sum.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    /// Componentwise difference.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    /// Scalar product.
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// `Dot3`
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// `Cross3`
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// `GetLength3`
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// `GetDistance3`
    pub fn distance(self, o: Self) -> f32 {
        self.sub(o).length()
    }

    /// Linear interpolation `Lerp(a, b, t)`.
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        a.add(b.sub(a).scale(t))
    }

    /// `GetNormalized3` — divide by the 3-length; zero/inf length yields
    /// zeros (matching the upstream SSE failsafes).
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 || !len.is_finite() {
            return Self::default();
        }
        self.scale(1.0 / len)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        self.add(o)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self.sub(o)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        self.scale(s)
    }
}

/// Quaternion stored `(x, y, z, w)` — the upstream memory order.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quat {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W (real) component.
    pub w: f32,
}

impl Quat {
    /// Identity `(0, 0, 0, 1)`.
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// `(x, y, z, w)`
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// From an (already normalized) axis and an angle in radians —
    /// `Quaternion(axis, angle)`: `(axis·sin(θ/2), cos(θ/2))`.
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = 0.5 * angle;
        let (s, c) = half.sin_cos();
        Self::new(axis.x * s, axis.y * s, axis.z * s, c)
    }

    /// Standard Hamilton product `a ⊗ b`. (The upstream `operator*` is the
    /// reversed DirectXMath order and lives in the `Mul` impl below.)
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, b: Self) -> Self {
        Self::new(
            self.w * b.x + self.x * b.w + self.y * b.z - self.z * b.y,
            self.w * b.y - self.x * b.z + self.y * b.w + self.z * b.x,
            self.w * b.z + self.x * b.y - self.y * b.x + self.z * b.w,
            self.w * b.w - self.x * b.x - self.y * b.y - self.z * b.z,
        )
    }

    /// `GetConjugate` — `(-x, -y, -z, w)`.
    pub const fn conj(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// 4-component dot.
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }

    /// `GetNormalized`
    pub fn normalized(self) -> Self {
        let len = self.dot(self).sqrt();
        if len == 0.0 || !len.is_finite() {
            return Self::default();
        }
        Self::new(self.x / len, self.y / len, self.z / len, self.w / len)
    }

    /// `GetInverse` — `Invert()`: the conjugate divided by the 4-length
    /// (≤ Epsilon length yields the zero quaternion, upstream failsafe).
    pub fn inverse(self) -> Self {
        let len = self.dot(self).sqrt();
        if len <= 1e-6 || !len.is_finite() {
            return Self::default();
        }
        Self::new(-self.x / len, -self.y / len, -self.z / len, self.w / len)
    }

    /// `RotateVector` — `q ⊗ (v, 0) ⊗ q*` (the standard matrix sense).
    pub fn rotate_vector(self, v: Vec3) -> Vec3 {
        let p = Self::new(v.x, v.y, v.z, 0.0);
        let r = self.mul(p).mul(self.conj());
        Vec3::new(r.x, r.y, r.z)
    }

    /// `RotateVectorInverse` — `q* ⊗ (v, 0) ⊗ q`.
    pub fn rotate_vector_inverse(self, v: Vec3) -> Vec3 {
        let p = Self::new(v.x, v.y, v.z, 0.0);
        let r = self.conj().mul(p).mul(self);
        Vec3::new(r.x, r.y, r.z)
    }

    /// `XAxis` — first column of the standard rotation matrix `M(q)`.
    pub fn x_axis(self) -> Vec3 {
        let s = 2.0 * self.w;
        let x2 = 2.0 * self.x;
        Vec3::new(
            x2 * self.x + s * self.w - 1.0,
            x2 * self.y + s * self.z,
            x2 * self.z + s * -self.y,
        )
    }

    /// `YAxis` — second column of `M(q)`.
    pub fn y_axis(self) -> Vec3 {
        let s = 2.0 * self.w;
        let y2 = 2.0 * self.y;
        Vec3::new(
            y2 * self.x + s * -self.z,
            y2 * self.y + s * self.w - 1.0,
            y2 * self.z + s * self.x,
        )
    }

    /// `ZAxis` — third column of `M(q)` (the engine's "forward").
    pub fn z_axis(self) -> Vec3 {
        let s = 2.0 * self.w;
        let z2 = 2.0 * self.z;
        Vec3::new(
            self.x * z2 + s * self.y,
            self.y * z2 + s * -self.x,
            self.z * z2 + s * self.w - 1.0,
        )
    }

    /// `SLerp(from, to, t)` — sign-flips `to` when the dot is negative,
    /// falls back to `(1-t, t)` lerp weights when `cos ≥ 1 - 1e-5`, and
    /// does *not* renormalize the result (matching upstream).
    pub fn slerp(from: Self, to: Self, t: f32) -> Self {
        let mut cos = from.dot(to);
        let sign = if cos < 0.0 { -1.0 } else { 1.0 };
        cos *= sign;

        let (s0, s1) = if cos < 1.0 - 1e-5 {
            let sin = (1.0 - cos * cos).sqrt();
            let omega = sin.atan2(cos);
            (((1.0 - t) * omega).sin() / sin, (t * omega).sin() / sin)
        } else {
            (1.0 - t, t)
        };

        Self::new(
            from.x * s0 + to.x * s1 * sign,
            from.y * s0 + to.y * s1 * sign,
            from.z * s0 + to.z * s1 * sign,
            from.w * s0 + to.w * s1 * sign,
        )
    }

    /// `NLerp(from, to, t)` — flip `from` on negative dot, lerp, normalize.
    pub fn nlerp(from: Self, to: Self, t: f32) -> Self {
        let from = if from.dot(to) < 0.0 { from.neg() } else { from };
        Self::new(
            from.x + (to.x - from.x) * t,
            from.y + (to.y - from.y) * t,
            from.z + (to.z - from.z) * t,
            from.w + (to.w - from.w) * t,
        )
        .normalized()
    }

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }

    /// `FromRotationBetweenVectors` — shortest-arc rotation aligning
    /// `source` onto `target` (both normalized internally).
    pub fn from_rotation_between_vectors(source: Vec3, target: Vec3) -> Self {
        let a = source.normalized();
        let b = target.normalized();
        let d = a.dot(b);
        // upstream thresholds: Vector::OneMinusEpsilon / EpsilonMinusOne
        // (= ±(1 - Math::Epsilon) with Epsilon = 1e-6)
        const ONE_MINUS_EPS: f32 = 1.0 - 1e-6;

        // Parallel — zero rotation.
        if d >= ONE_MINUS_EPS {
            return Self::IDENTITY;
        }
        // Opposite — 180° around any axis orthogonal to `from`
        // (upstream picks `(-z, y, x, 0)`).
        if d <= -ONE_MINUS_EPS {
            return Self::new(-a.z, a.y, a.x, 0.0).normalized();
        }
        // General case: `(cross, dot + |q|)` normalized.
        let cross = a.cross(b);
        let mut q = Self::new(cross.x, cross.y, cross.z, d);
        q.w += q.dot(q).sqrt();
        q.normalized()
    }

    /// `LookRotation(forward, up)` — matrix rows
    /// `(normalize(up×forward), forward×that, forward)` through the RTM
    /// matrix→quaternion extraction.
    pub fn look_rotation(forward: Vec3, up: Vec3) -> Self {
        let t = up.cross(forward).normalized();
        let r0 = t;
        let r1 = forward.cross(t);
        let r2 = forward;
        quat_from_matrix_rows(r0, r1, r2)
    }
}

/// The upstream `operator*` is the *reversed* DirectXMath product:
/// `operator*(q1, q2)` = Hamilton `q2 ⊗ q1` (decoded from the SSE lanes).
/// Keeping the reversal here makes every `a * b` in the C++ sources
/// translate literally as `a * b` in Rust.
impl std::ops::Mul for Quat {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        b.mul(self)
    }
}

/// `Matrix::GetRotation` (RTM-based) over a rotation stored as rows.
pub fn quat_from_matrix_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Quat {
    let trace = r0.x + r1.y + r2.z;
    if trace > 0.0 {
        let inv_trace = 1.0 / (trace + 1.0).sqrt();
        let half = inv_trace * 0.5;
        Quat::new(
            (r1.z - r2.y) * half,
            (r2.x - r0.z) * half,
            (r0.y - r1.x) * half,
            (1.0 / inv_trace) * 0.5,
        )
        .normalized()
    } else {
        // Find the axis with the highest diagonal value.
        let mut idx0 = 0usize;
        if r1.y > r0.x {
            idx0 = 1;
        }
        if r2.z > [r0.x, r1.y, r2.z][idx0] {
            idx0 = 2;
        }
        let idx1 = (idx0 + 1) % 3;
        let idx2 = (idx1 + 1) % 3;
        let rows = [r0, r1, r2];
        let diag = [r0.x, r1.y, r2.z];
        let pseudo_trace = 1.0 + diag[idx0] - diag[idx1] - diag[idx2];
        let inv_pt = 1.0 / pseudo_trace.sqrt();
        let half = inv_pt * 0.5;

        let mut q = [0.0f32; 4];
        q[idx0] = (1.0 / inv_pt) * 0.5;
        q[idx1] = (rows[idx0][idx1] + rows[idx1][idx0]) * half;
        q[idx2] = (rows[idx0][idx2] + rows[idx2][idx0]) * half;
        q[3] = (rows[idx1][idx2] - rows[idx2][idx1]) * half;
        Quat::new(q[0], q[1], q[2], q[3]).normalized()
    }
}

/// Rigid transform — the upstream `Math::Transform` at unit scale (the
/// correction pass never touches scale).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Transform {
    /// Rotation.
    pub rot: Quat,
    /// Translation.
    pub t: Vec3,
}

impl Transform {
    /// `Transform::Identity`.
    pub const IDENTITY: Self = Self {
        rot: Quat::IDENTITY,
        t: Vec3::new(0.0, 0.0, 0.0),
    };

    /// `operator*` at unit scale — the standard rigid composition `B ∘ A`
    /// (apply `a` first, then `b`): `q = normalize(b.q ⊗ a.q)` (the C++
    /// writes `b.q * a.q` under the reversed operator), `t = R_b·a.t + b.t`.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Self, b: Self) -> Self {
        Self {
            rot: b.rot.mul(a.rot).normalized(),
            t: b.rot.rotate_vector(a.t).add(b.t),
        }
    }

    /// `TransformPoint` at unit scale: `t + rotate_vector(q, p)`.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        self.rot.rotate_vector(p).add(self.t)
    }

    /// `GetDeltaToOther(target)` — `Delta(*self, target)` at unit scale:
    /// `self⁻¹ ∘ target`, i.e. `fi = q.inverse(); q' = fi ⊗ target.q` (C++
    /// `target.q * fi` under the reversed operator); `t' = fi ⊗ Δt ⊗ fi*`.
    /// Then `delta * other` = `other ∘ delta` composes back to `target`
    /// whenever `other == self`.
    pub fn delta_to_other(&self, target: &Self) -> Self {
        let fi = self.rot.inverse();
        Self {
            rot: fi.mul(target.rot),
            t: fi.rotate_vector(target.t.sub(self.t)),
        }
    }
}

impl std::ops::Mul for Transform {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Transform::mul(self, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn hamilton_mul_matches_matrix() {
        // 90° about Z: q = (0, 0, √2/2, √2/2). M(q) maps x̂ → ŷ.
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let x = q.x_axis();
        assert!(close(x.x, 0.0, 1e-6) && close(x.y, 1.0, 1e-6) && close(x.z, 0.0, 1e-6));
        // q ⊗ q = 180° about Z: M maps x̂ → −x̂.
        let q2 = q.mul(q);
        let x = q2.x_axis();
        assert!(close(x.x, -1.0, 1e-5) && close(x.y, 0.0, 1e-5));
    }

    #[test]
    fn inverse_is_conjugate_over_length() {
        let q = Quat::from_axis_angle(Vec3::new(0.1, 0.9, 0.2).normalized(), 0.7);
        let fi = q.inverse();
        assert!(close(fi.x, -q.x, 1e-7) && close(fi.y, -q.y, 1e-7) && close(fi.z, -q.z, 1e-7));
        // unit-length: inverse * q = identity (Hamilton)
        let back = Quat::mul(q, fi);
        assert!(close(back.x, 0.0, 1e-6) && close(back.w, 1.0, 1e-6));
        // non-unit: upstream GetInverse = conj/|q| is NOT renormalized, so
        // q ⊗ q⁻¹ = (0, 0, 0, |q|), not identity.
        let q2 = Quat::new(0.4, -0.8, 0.2, 1.6); // |q2| = sqrt(3.4) ≈ 1.84
        let back2 = Quat::mul(q2, q2.inverse());
        assert!(close(back2.x, 0.0, 1e-6) && close(back2.w, 3.4f32.sqrt(), 1e-6));
    }

    #[test]
    fn rotate_vector_is_standard_sense() {
        // rotate_vector applies q ⊗ v ⊗ q*: +90°-about-Z maps x̂ → +ŷ.
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let v = q.rotate_vector(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(v.x, 0.0, 1e-6) && close(v.y, 1.0, 1e-6) && close(v.z, 0.0, 1e-6));
        let w = q.rotate_vector_inverse(v);
        assert!(close(w.x, 1.0, 1e-6) && close(w.y, 0.0, 1e-6));
    }

    #[test]
    fn mul_operator_is_reversed_product() {
        // C++ operator*(q1, q2) = Hamilton q2 ⊗ q1 (DirectXMath lane order).
        let qz = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7);
        let qy = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.3);
        assert_eq!(qz * qy, Quat::mul(qy, qz));
        assert_ne!(qz * qy, Quat::mul(qz, qy));
    }

    #[test]
    fn slerp_endpoints_and_halfway() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
        let s0 = Quat::slerp(a, b, 0.0);
        assert!(close(s0.x, a.x, 1e-7) && close(s0.w, a.w, 1e-7));
        let s1 = Quat::slerp(a, b, 1.0);
        assert!(close(s1.y, b.y, 1e-7) && close(s1.w, b.w, 1e-7));
        // Halfway must be 45° about y on the sign-fixed path.
        let half = Quat::slerp(a, b, 0.5).normalized();
        let want = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_4);
        assert!(close(half.y, want.y, 1e-6) && close(half.w, want.w, 1e-6));
    }

    #[test]
    fn slerp_sign_flip_takes_short_path() {
        let a = Quat::IDENTITY;
        // −b represents the same rotation as b; slerp must not swing the
        // long way (which would land at 135° at t=0.5).
        let b = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2).neg();
        let half = Quat::slerp(a, b, 0.5).normalized();
        let angle = 2.0 * half.w.acos();
        assert!(close(angle, std::f32::consts::FRAC_PI_4, 1e-4));
    }

    #[test]
    fn rotation_between_vectors_shortest_arc() {
        let q =
            Quat::from_rotation_between_vectors(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        // Standard sense of q maps x̂ → ŷ: rotate_vector(x̂) = ŷ ...
        let v = q.rotate_vector(Vec3::new(1.0, 0.0, 0.0));
        assert!(close(v.x, 0.0, 1e-6) && close(v.y, 1.0, 1e-6));
        // ... and rotate_vector_inverse(ŷ) = x̂.
        let x = q.rotate_vector_inverse(Vec3::new(0.0, 1.0, 0.0));
        assert!(close(x.x, 1.0, 1e-6) && close(x.y, 0.0, 1e-6));
        // Opposite vectors: the (−z, y, x, 0) branch stays a unit quaternion.
        let q = Quat::from_rotation_between_vectors(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        );
        let n = (q.dot(q)).sqrt();
        assert!(close(n, 1.0, 1e-6));
        // Parallel: identity.
        let q =
            Quat::from_rotation_between_vectors(Vec3::new(0.3, 0.4, 0.5), Vec3::new(0.6, 0.8, 1.0));
        assert!(close(q.w, 1.0, 1e-6));
    }

    #[test]
    fn look_rotation_is_orthonormal_with_forward_as_z() {
        let forward = Vec3::new(0.2, -0.3, 0.9).normalized();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let q = Quat::look_rotation(forward, up);
        let z = q.z_axis();
        assert!(close(z.x, forward.x, 1e-6) && close(z.y, forward.y, 1e-6));
        let (x, y) = (q.x_axis(), q.y_axis());
        assert!(close(x.dot(y), 0.0, 1e-6) && close(y.dot(z), 0.0, 1e-6));
        assert!(close(x.length(), 1.0, 1e-6) && close(y.length(), 1.0, 1e-6));
    }

    #[test]
    fn transform_mul_translation_chain() {
        // a = (rot 90°z, t=0), b = (identity, t=(1,0,0)):
        // a*b = b ∘ a: q = b.q ⊗ a.q = a.q, t = R_b·0 + (1,0,0).
        let qa = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let a = Transform {
            rot: qa,
            t: Vec3::new(0.0, 0.0, 0.0),
        };
        let b = Transform {
            rot: Quat::IDENTITY,
            t: Vec3::new(1.0, 0.0, 0.0),
        };
        let ab = Transform::mul(a, b);
        assert!(close(ab.t.x, 1.0, 1e-7) && close(ab.t.y, 0.0, 1e-7));
        // b*a = a ∘ b: t = R_a·(1,0,0) + 0 = (0,1,0) (standard matrix sense).
        let ba = Transform::mul(b, a);
        assert!(close(ba.t.x, 0.0, 1e-6) && close(ba.t.y, 1.0, 1e-6));
    }

    #[test]
    fn delta_to_other_composes_back() {
        // delta = from → to; delta * from must reproduce `to`.
        let from = Transform {
            rot: Quat::from_axis_angle(Vec3::new(0.1, 0.9, 0.2).normalized(), 0.7),
            t: Vec3::new(1.0, -2.0, 0.5),
        };
        let to = Transform {
            rot: Quat::from_axis_angle(Vec3::new(-0.4, 0.2, 0.9).normalized(), 2.1),
            t: Vec3::new(-3.0, 4.0, 2.0),
        };
        let delta = from.delta_to_other(&to);
        let back = Transform::mul(delta, from);
        assert!(close(back.t.x, to.t.x, 1e-4));
        assert!(close(back.t.y, to.t.y, 1e-4));
        assert!(close(back.t.z, to.t.z, 1e-4));
        assert!(close(back.rot.x, to.rot.x, 1e-4) && close(back.rot.w, to.rot.w, 1e-4));
    }
}
