//! Exact sums of signed monomial shifts of ring elements. Each coefficient
//! enters as two bounded integer limbs, so every term is one i64 addition.
//! Reduction modulo `X^D + 1` and `q` runs once, after all terms.
//! A product with a small-digit ring element also splits each coefficient
//! into two limbs. Each output coefficient is then an exact i64 dot product,
//! with one reduction modulo `q`.

use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{field::Q, ring::D, F};

/// Each limb of a `SplitRing` is in `[0, LIMB_BOUND)`.
pub const LIMB_BOUND: u128 = 1 << 32;
/// The most terms that one `SignedShiftSums` holds exactly.
pub const MAX_TERMS: u64 = 1 << 24;

/// A coefficient position receives at most `D` shifts of each term.
const _: () = assert!((D as u128) * LIMB_BOUND * (MAX_TERMS as u128) < (1u128 << 63));
/// A `u64` mask has one bit for each coefficient.
const _: () = assert!(D == u64::BITS as usize);

/// Coefficient `j` is `limbs[2j] + 2^32 · limbs[2j + 1]`.
pub struct SplitRing {
    limbs: [u32; 2 * D],
}

impl SplitRing {
    pub fn new(coefficients: &[F; D]) -> Self {
        Self::from_u64(&coefficients.map(|value| value.as_canonical_u64()))
    }

    /// The ring element with these integer coefficients: each limb is 32 bits of one.
    pub fn from_u64(coefficients: &[u64; D]) -> Self {
        Self {
            limbs: core::array::from_fn(|limb| (coefficients[limb / 2] >> (32 * (limb % 2))) as u32),
        }
    }
}

/// The alignment and the limb order put each 16-byte vector access of
/// `add_signed_units` on a 16-byte boundary, at every shift. A vector access
/// across that boundary is several times slower.
#[derive(Clone)]
#[repr(align(16))]
pub struct SignedShiftSums {
    /// The two limb sums of degree `k` at `2k` and `2k + 1`, as in `SplitRing`.
    limbs: [i64; 2 * (2 * D - 1)],
    #[cfg(debug_assertions)]
    terms: u64,
}

impl SignedShiftSums {
    pub fn zero() -> Self {
        Self {
            limbs: [0; 2 * (2 * D - 1)],
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
            for (sum, &limb) in self.limbs[2 * shift..][..2 * D]
                .iter_mut()
                .zip(&element.limbs)
            {
                *sum += i64::from(limb);
            }
            positive &= positive - 1;
        }
        while negative != 0 {
            let shift = negative.trailing_zeros() as usize;
            for (sum, &limb) in self.limbs[2 * shift..][..2 * D]
                .iter_mut()
                .zip(&element.limbs)
            {
                *sum -= i64::from(limb);
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
        for (sum, part) in self.limbs.iter_mut().zip(&other.limbs) {
            *sum += part;
        }
    }

    pub fn reduce(&self) -> [F; D] {
        let raw = |degree: usize| {
            F::from_i128((i128::from(self.limbs[2 * degree + 1]) << 32) + i128::from(self.limbs[2 * degree]))
        };
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

/// The largest digit magnitude that `SmallDigits` holds.
pub const SMALL_DIGIT_BOUND: u64 = 1 << 25;

/// A product coefficient sums `D` products of a signed 32-bit limb and a digit.
const _: () = assert!((D as u128) * (1 << 31) * (SMALL_DIGIT_BOUND as u128) < (1u128 << 63));

/// A ring element as `high · 2^32 + low`, with each limb in `[−2^31, 2^31)`.
pub struct SignedLimbs {
    low: [i32; D],
    high: [i32; D],
}

impl SignedLimbs {
    pub fn new(coefficients: &[F; D]) -> Self {
        // Two signed limbs hold [−2^63 − 2^31, 2^63 − 2^31). That range is wider
        // than q, so each residue has a representative in it.
        let shifted = coefficients.map(|value| {
            let value = value.as_canonical_u64();
            let representative = if value < (1 << 63) - (1 << 31) {
                value
            } else {
                value.wrapping_sub(Q)
            };
            representative.wrapping_add(1 << 31) as i64
        });
        Self {
            low: shifted.map(|value| ((value & 0xffff_ffff) - (1 << 31)) as i32),
            high: shifted.map(|value| (value >> 32) as i32),
        }
    }
}

/// A ring element with integer coefficients in `[−SMALL_DIGIT_BOUND, SMALL_DIGIT_BOUND]`.
pub struct SmallDigits {
    /// `d_j` at `D − 1 − j` and `−d_j` at `2D − 1 − j`, because `X^D = −1`.
    /// Coefficient `k` of `c · d` is then the dot product of `c` with the
    /// window that starts at `D − 1 − k`.
    windows: [i32; 2 * D - 1],
}

impl SmallDigits {
    /// `None` when a coefficient is outside the bound.
    pub fn new(coefficients: &[F; D]) -> Option<Self> {
        let mut windows = [0; 2 * D - 1];
        for (j, value) in coefficients.iter().enumerate() {
            let value = value.as_canonical_u64();
            let digit = if value <= SMALL_DIGIT_BOUND {
                value as i32
            } else if Q - value <= SMALL_DIGIT_BOUND {
                -((Q - value) as i32)
            } else {
                return None;
            };
            windows[D - 1 - j] = digit;
            if j > 0 {
                windows[2 * D - 1 - j] = -digit;
            }
        }
        Some(Self { windows })
    }

    /// The masks of the coefficients `1` and `−1`, when no other value occurs.
    pub fn signed_unit_masks(&self) -> Option<(u64, u64)> {
        let mut masks = (0, 0);
        for j in 0..D {
            match self.windows[D - 1 - j] {
                0 => {}
                1 => masks.0 |= 1 << j,
                -1 => masks.1 |= 1 << j,
                _ => return None,
            }
        }
        Some(masks)
    }

    /// The product `form · self` in `R_q`, exact in `i64` before one reduction.
    #[inline]
    pub fn mul(&self, form: &SignedLimbs) -> [F; D] {
        core::array::from_fn(|k| {
            let window = &self.windows[D - 1 - k..][..D];
            let dot = |limbs: &[i32; D]| -> i64 {
                limbs
                    .iter()
                    .zip(window)
                    .map(|(&limb, &digit)| i64::from(limb) * i64::from(digit))
                    .sum()
            };
            F::from_i128((i128::from(dot(&form.high)) << 32) + i128::from(dot(&form.low)))
        })
    }
}
