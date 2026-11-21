//! Ring equations of `K[X]/(X^64 + 1)` as inner products (SPEC.md section 6).

use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

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

pub(crate) struct LaneTransform<'a> {
    point: &'a Point,
    scaled: Option<([K; D], K)>,
}

impl<'a> LaneTransform<'a> {
    pub(crate) fn new(point: &'a Point) -> Self {
        let delta_to_d = point.powers[D - 1] * point.powers[1];
        let scaled = delta_to_d.try_inverse().and_then(|inverse| {
            let factor = K::ONE + inverse;
            Some((point.powers.map(|p| p * factor), factor.try_inverse()?))
        });
        Self { point, scaled }
    }

    pub(crate) fn lanes(&self, element: &[K; D]) -> [K; D] {
        let Some((scaled, inverse)) = &self.scaled else {
            return self.point.lanes(element);
        };
        let mut terms = [K::ZERO; D];
        let mut sum = K::ZERO;
        for lane in 0..D {
            let product = element[lane] * scaled[lane];
            terms[D - 1 - lane] = product;
            sum += product;
        }
        let mut remaining = sum * *inverse;
        let mut result = [K::ZERO; D];
        result[0] = remaining;
        for lane in 1..D {
            remaining -= terms[lane - 1];
            result[lane] = remaining * self.point.powers[lane];
        }
        result
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
