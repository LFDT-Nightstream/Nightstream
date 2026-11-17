//! Ring equations of `K[X]/(X^64 + 1)` as inner products (SPEC.md section 6).

use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

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

    pub fn lanes(&self, element: &[K; D]) -> [K; D] {
        let delta = self.powers[1];
        let mut low = [K::ZERO; D + 1];
        let mut high = [K::ZERO; D + 1];
        for lane in 0..D {
            low[lane + 1] = low[lane] + element[lane] * self.powers[lane];
            high[D - 1 - lane] = element[D - 1 - lane] + delta * high[D - lane];
        }
        std::array::from_fn(|lane| self.powers[lane] * low[D - lane] - high[D - lane])
    }
}

/// `conj` of SPEC.md Algorithm 9.
pub fn conjugate(element: &[K; D]) -> [K; D] {
    std::array::from_fn(|lane| if lane == 0 { element[0] } else { -element[D - lane] })
}

pub(crate) fn dot(left: &[K], right: &[K]) -> K {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * right)
        .sum()
}
