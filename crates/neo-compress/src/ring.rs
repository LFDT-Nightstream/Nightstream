//! Ring equations of `K[X]/(X^64 + 1)` as inner products (SPEC.md section 6).
//!
//! The equation `Σ_j P_j · z_j = Y` holds exactly when both sides have the
//! same value at each point `δ`, where the value of a ring element is the
//! value of its reduced polynomial of degree below 64. Block `j` of the left
//! side has the value `Σ_h z_j[h] · (X^h · P_j)(δ)`, so the equation at `δ` is
//! the inner product of the witness with the weights `(X^h · P_j)(δ)`.

use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

/// The powers `δ^0, …, δ^63` of the point at which the equations are read.
pub struct Point {
    powers: [K; D],
}

impl Point {
    pub fn new(delta: K) -> Self {
        let mut powers = [K::ONE; D];
        for lane in 1..D {
            powers[lane] = powers[lane - 1] * delta;
        }
        Self { powers }
    }

    /// The value at `δ` of the ring element with the coefficients `element`.
    pub fn value(&self, element: &[K]) -> K {
        element
            .iter()
            .zip(&self.powers)
            .map(|(&coefficient, &power)| coefficient * power)
            .sum()
    }

    pub fn value_base(&self, element: &[F]) -> K {
        element
            .iter()
            .zip(&self.powers)
            .map(|(&coefficient, &power)| power * coefficient)
            .sum()
    }

    /// `(X^h · P)(δ)` for each lane `h`. With `X^64 = −1`, the terms of `P`
    /// below `64 − h` keep their sign and shift up by `h`, and the other terms
    /// change sign and shift down by `64 − h`.
    ///
    /// The pairing `(e, P) ↦ Σ_h e[h] · lanes(P)[h]` is `(e · P)(δ)`, which is
    /// symmetric, so `lanes(e)` is also the linear form `P ↦ (e · P)(δ)`.
    pub fn lanes(&self, element: &[K; D]) -> [K; D] {
        let delta = self.powers[1];
        // low[m] = Σ_{l < m} P[l] δ^l, and high[m] = Σ_{l ≥ m} P[l] δ^(l − m).
        let mut low = [K::ZERO; D + 1];
        let mut high = [K::ZERO; D + 1];
        for lane in 0..D {
            low[lane + 1] = low[lane] + element[lane] * self.powers[lane];
            high[D - 1 - lane] = element[D - 1 - lane] + delta * high[D - lane];
        }
        std::array::from_fn(|lane| self.powers[lane] * low[D - lane] - high[D - lane])
    }
}

/// The conjugate `X ↦ X^(−1)` of SPEC.md section 2.2, on coefficients in `K`:
/// `conjugate(a) · z` has the constant term `Σ_h a[h] · z[h]`.
pub fn conjugate(element: &[K; D]) -> [K; D] {
    std::array::from_fn(|lane| if lane == 0 { element[0] } else { -element[D - lane] })
}

pub(crate) fn dot(left: &[K], right: &[K]) -> K {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * right)
        .sum()
}
