//! Exact sums of signed monomial shifts of ring elements. Each coefficient
//! enters as two bounded integer limbs, so every term is one i64 addition.
//! Reduction modulo Phi_81 and Goldilocks runs once, after all terms.

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{
    ring::{D, ETA, PHI_MID_DEGREE},
    F,
};

pub const LIMB_BOUND: u128 = (1 << 32) + 3;
const MAX_TERMS: u64 = 1 << 25;
const X_POWERS: [(i64, i64); 8] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)];

#[inline(always)]
pub fn wide256_signed_halves(words: [u32; 8]) -> (i64, i64) {
    words
        .iter()
        .zip(X_POWERS)
        .fold((0, 0), |(low, high), (&word, (one, x))| {
            (low + one * i64::from(word), high + x * i64::from(word))
        })
}

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

    pub fn from_wide256(coefficients: [[u32; 8]; D]) -> Self {
        let mut low = [0i64; D];
        let mut high = [0i64; D];
        for (lane, words) in coefficients.into_iter().enumerate() {
            let (a, b) = wide256_signed_halves(words);
            let b = b + (a >> 32);
            let carry = b >> 32;
            low[lane] = (a & 0xffff_ffff) - carry;
            high[lane] = (b & 0xffff_ffff) + carry;
        }
        Self { low, high }
    }
}

#[derive(Clone)]
pub struct SignedShiftSums {
    low: [i64; 2 * D - 1],
    high: [i64; 2 * D - 1],
    terms: u64,
}

const _: () = assert!((D as u128) * LIMB_BOUND * (MAX_TERMS as u128) < (1u128 << 63));

impl SignedShiftSums {
    pub fn zero() -> Self {
        Self {
            low: [0; 2 * D - 1],
            high: [0; 2 * D - 1],
            terms: 0,
        }
    }

    #[inline]
    pub fn add_signed_units(&mut self, element: &SplitRing, mut positive: u64, mut negative: u64) {
        debug_assert_eq!(
            (positive | negative) >> D,
            0,
            "signed-unit mask exceeds the ring dimension"
        );
        self.terms += 1;
        assert!(
            self.terms <= MAX_TERMS,
            "signed shift sums exceed the exact i64 term bound"
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
        self.terms += other.terms;
        assert!(
            self.terms <= MAX_TERMS,
            "signed shift sums exceed the exact i64 term bound"
        );
        for degree in 0..2 * D - 1 {
            self.low[degree] += other.low[degree];
            self.high[degree] += other.high[degree];
        }
    }

    pub fn reduce(&self) -> [F; D] {
        let raw: [F; 2 * D - 1] = core::array::from_fn(|degree| {
            let value = (i128::from(self.high[degree]) << 32) + i128::from(self.low[degree]);
            F::from_u64(value.rem_euclid(i128::from(F::ORDER_U64)) as u64)
        });
        core::array::from_fn(|lane| {
            if lane < PHI_MID_DEGREE {
                let wrapped = raw.get(lane + ETA).copied().unwrap_or(F::ZERO);
                raw[lane] - raw[lane + D] + wrapped
            } else {
                raw[lane] - raw[lane + PHI_MID_DEGREE]
            }
        })
    }
}
