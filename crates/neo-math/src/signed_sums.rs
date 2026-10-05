//! Exact sums of signed monomial shifts of ring elements. Each coefficient
//! enters as two bounded integer limbs, so every term is one i64 addition.
//! Reduction modulo Phi_81 and Goldilocks runs once, after all terms.

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{ring::D, F};

/// Integer coefficient limbs representing a ring element modulo Goldilocks.
/// Each limb has magnitude below 2^32 + 3; it need not be canonical.
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

    /// Lift little-endian 256-bit coefficients without field reductions.
    /// For x = 2^32, x^2 = x - 1 and x^6 = 1 modulo Goldilocks.
    /// Normalizing the two limbs subtracts a multiple of p and keeps their
    /// magnitudes below 2^32 + 3, including negative intermediate carries.
    pub fn from_wide256(coefficients: [[u32; 8]; D]) -> Self {
        let mut low = [0i64; D];
        let mut high = [0i64; D];
        for (lane, words) in coefficients.into_iter().enumerate() {
            let w = words.map(i64::from);
            let a = w[0] - w[2] - w[3] + w[5] + w[6];
            let b = w[1] + w[2] - w[4] - w[5] + w[7] + (a >> 32);
            let carry = b >> 32;
            low[lane] = (a & 0xffff_ffff) - carry;
            high[lane] = (b & 0xffff_ffff) + carry;
        }
        Self { low, high }
    }
}

/// Raw convolution sums of signed-unit shifts, before any reduction.
///
/// One `add_signed_units` call adds at most D terms below `2^32 + 3` to
/// each raw coefficient. Fewer than `2^25` calls into one accumulator,
/// merged sums included, keep every sum exact in i64.
#[derive(Clone)]
pub struct SignedShiftSums {
    low: [i64; 2 * D - 1],
    high: [i64; 2 * D - 1],
}

const _: () = assert!((D as u128) * ((1u128 << 32) + 3) * (1u128 << 25) < (1u128 << 63));

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
        let cancelled = positive & negative;
        positive ^= cancelled;
        negative ^= cancelled;
        while positive != 0 {
            let shift = positive.trailing_zeros() as usize;
            for (sum, value) in self.low[shift..shift + D].iter_mut().zip(&element.low) {
                *sum += value;
            }
            for (sum, value) in self.high[shift..shift + D].iter_mut().zip(&element.high) {
                *sum += value;
            }
            positive &= positive - 1;
        }
        while negative != 0 {
            let shift = negative.trailing_zeros() as usize;
            for (sum, value) in self.low[shift..shift + D].iter_mut().zip(&element.low) {
                *sum -= value;
            }
            for (sum, value) in self.high[shift..shift + D].iter_mut().zip(&element.high) {
                *sum -= value;
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
