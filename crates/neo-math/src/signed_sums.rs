//! Exact sums of signed monomial shifts of ring elements. Each coefficient
//! enters as its two 32-bit halves, so every term is one i64 addition.
//! Reduction modulo Phi_81 and Goldilocks runs once, after all terms.

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{ring::D, F};

/// A ring element split into the 32-bit halves of its canonical coefficients.
pub struct SplitRing {
    low: [i64; D],
    high: [i64; D],
}

impl SplitRing {
    pub fn new(coefficients: &[F; D]) -> Self {
        let canonical = coefficients.map(|value| value.as_canonical_u64());
        Self {
            low: canonical.map(|value| (value & 0xffff_ffff) as i64),
            high: canonical.map(|value| (value >> 32) as i64),
        }
    }
}

/// Raw convolution sums of signed-unit shifts, before any reduction.
///
/// One `add_signed_units` call adds at most `D < 2^6` terms below `2^32` to
/// each raw coefficient. Fewer than `2^25` calls into one accumulator,
/// merged sums included, keep every sum exact in i64.
#[derive(Clone)]
pub struct SignedShiftSums {
    low: [i64; 2 * D - 1],
    high: [i64; 2 * D - 1],
}

impl SignedShiftSums {
    pub fn zero() -> Self {
        Self {
            low: [0; 2 * D - 1],
            high: [0; 2 * D - 1],
        }
    }

    /// Add `element * X^lane` for every positive lane and subtract it for
    /// every negative lane.
    #[inline]
    pub fn add_signed_units(&mut self, element: &SplitRing, mut positive: u64, mut negative: u64) {
        debug_assert_eq!(
            (positive | negative) >> D,
            0,
            "signed-unit mask exceeds the ring dimension"
        );
        while positive != 0 {
            let shift = positive.trailing_zeros() as usize;
            for lane in 0..D {
                self.low[shift + lane] += element.low[lane];
                self.high[shift + lane] += element.high[lane];
            }
            positive &= positive - 1;
        }
        while negative != 0 {
            let shift = negative.trailing_zeros() as usize;
            for lane in 0..D {
                self.low[shift + lane] -= element.low[lane];
                self.high[shift + lane] -= element.high[lane];
            }
            negative &= negative - 1;
        }
    }

    pub fn add(&mut self, other: &Self) {
        for degree in 0..2 * D - 1 {
            self.low[degree] += other.low[degree];
            self.high[degree] += other.high[degree];
        }
    }

    /// The reduced ring element. `X^54 = -X^27 - 1` and `X^81 = 1`.
    pub fn reduce(&self) -> [F; D] {
        let raw: [F; 2 * D - 1] = core::array::from_fn(|degree| {
            let value = (i128::from(self.high[degree]) << 32) + i128::from(self.low[degree]);
            F::from_u64(value.rem_euclid(i128::from(F::ORDER_U64)) as u64)
        });
        core::array::from_fn(|lane| {
            if lane < D / 2 {
                let wrapped = raw.get(lane + 81).copied().unwrap_or(F::ZERO);
                raw[lane] - raw[lane + D] + wrapped
            } else {
                raw[lane] - raw[lane + D / 2]
            }
        })
    }
}
