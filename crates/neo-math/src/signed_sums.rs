use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{ring::D, F};

pub const LIMB_BOUND: u128 = 1 << 32;
pub const MAX_EXACT_TERMS: u64 = 1 << 24;

const _: () = assert!((D as u128) * LIMB_BOUND * (MAX_EXACT_TERMS as u128) < (1u128 << 63));
const _: () = assert!(D == u64::BITS as usize);

pub struct SplitRing {
    low: [i64; D],
    high: [i64; D],
}

impl SplitRing {
    pub fn new(coefficients: &[F; D]) -> Self {
        Self::from_u64(&coefficients.map(|value| value.as_canonical_u64()))
    }

    pub fn from_u64(coefficients: &[u64; D]) -> Self {
        Self {
            low: coefficients.map(|value| (value & 0xffff_ffff) as i64),
            high: coefficients.map(|value| (value >> 32) as i64),
        }
    }
}

#[derive(Clone)]
pub struct SignedShiftSums {
    low: [i64; 2 * D - 1],
    high: [i64; 2 * D - 1],
    #[cfg(debug_assertions)]
    terms: u64,
}

impl SignedShiftSums {
    pub fn zero() -> Self {
        Self {
            low: [0; 2 * D - 1],
            high: [0; 2 * D - 1],
            #[cfg(debug_assertions)]
            terms: 0,
        }
    }

    #[inline]
    pub fn add_signed_units(&mut self, element: &SplitRing, mut positive: u64, mut negative: u64) {
        #[cfg(debug_assertions)]
        {
            self.terms += 1;
            debug_assert!(
                self.terms <= MAX_EXACT_TERMS,
                "signed shift sums exceed the exact i64 term bound"
            );
        }
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
        #[cfg(debug_assertions)]
        {
            self.terms += other.terms;
            debug_assert!(
                self.terms <= MAX_EXACT_TERMS,
                "signed shift sums exceed the exact i64 term bound"
            );
        }
        for degree in 0..2 * D - 1 {
            self.low[degree] += other.low[degree];
            self.high[degree] += other.high[degree];
        }
    }

    pub fn reduce(&self) -> [F; D] {
        let raw = |degree: usize| F::from_i128((i128::from(self.high[degree]) << 32) + i128::from(self.low[degree]));
        core::array::from_fn(|lane| {
            let direct = raw(lane);
            if lane + D < 2 * D - 1 {
                let wrapped_past_x_d = raw(lane + D);
                direct - wrapped_past_x_d
            } else {
                direct
            }
        })
    }
}
