//! Exact sums of signed monomial shifts of ring elements. Each coefficient
//! enters as two bounded integer limbs, so every term is one i64 addition.
//! Reduction modulo `X^D + 1` and `q` runs once, after all terms.

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{field::FOLD, ring::D, F};

/// `2^(64·i) mod q` for the four 64-bit words of a 256-bit integer.
const WORD_WEIGHTS: [u64; 4] = [1, FOLD, FOLD * FOLD, FOLD * FOLD * FOLD];
const WEIGHT_SUM: u64 = WORD_WEIGHTS[0] + WORD_WEIGHTS[1] + WORD_WEIGHTS[2] + WORD_WEIGHTS[3];

/// Each limb of a `SplitRing` is in `[0, LIMB_BOUND)`.
///
/// `SplitRing::new` gives limbs below `2^32`. In `from_wide256`, each half
/// is at most `WEIGHT_SUM·(2^32 − 1)`. The carry of the low half into the
/// high half is then at most `WEIGHT_SUM − 1`, so the high half stays below
/// `WEIGHT_SUM·2^32` and its own carry is at most `WEIGHT_SUM − 1`. That
/// carry has weight `2^64 ≡ 59`, so the low limb is below
/// `2^32 + 59·WEIGHT_SUM`.
pub const LIMB_BOUND: u128 = (1 << 32) + (FOLD * WEIGHT_SUM) as u128;
const MAX_TERMS: u64 = 1 << 24;

/// A coefficient position receives at most `D` shifts of each term.
const _: () = assert!((D as u128) * LIMB_BOUND * (MAX_TERMS as u128) < (1u128 << 63));
/// A `u64` mask has one bit for each coefficient.
const _: () = assert!(D == u64::BITS as usize);

/// Two integers `(a, b)` with `a + b·2^32` congruent modulo `q` to the
/// little-endian 256-bit integer `words`. Each is at most
/// `WEIGHT_SUM·(2^32 − 1)`, which is below `2^50`.
#[inline(always)]
pub fn wide256_halves(words: [u32; 8]) -> (u64, u64) {
    WORD_WEIGHTS
        .iter()
        .enumerate()
        .fold((0, 0), |(low, high), (word, weight)| {
            (
                low + weight * u64::from(words[2 * word]),
                high + weight * u64::from(words[2 * word + 1]),
            )
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
            let (a, b) = wide256_halves(words);
            let b = b + (a >> 32);
            low[lane] = ((a & 0xffff_ffff) + FOLD * (b >> 32)) as i64;
            high[lane] = (b & 0xffff_ffff) as i64;
        }
        Self { low, high }
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
                self.terms <= MAX_TERMS,
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
                self.terms <= MAX_TERMS,
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
        // X^D = −1.
        core::array::from_fn(|lane| {
            if lane + D < 2 * D - 1 {
                raw(lane) - raw(lane + D)
            } else {
                raw(lane)
            }
        })
    }
}
