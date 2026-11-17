use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{field::Q, ring::D, F};

pub const LIMB_BOUND: u128 = 1 << 32;
pub const MAX_EXACT_TERMS: u64 = 1 << 24;

const _: () = assert!((D as u128) * LIMB_BOUND * (MAX_EXACT_TERMS as u128) < (1u128 << 63));
const _: () = assert!(D == u64::BITS as usize);

pub struct SplitRing {
    limbs: [u32; 2 * D],
}

impl SplitRing {
    pub fn new(coefficients: &[F; D]) -> Self {
        Self::from_u64(&coefficients.map(|value| value.as_canonical_u64()))
    }

    pub fn from_u64(coefficients: &[u64; D]) -> Self {
        Self {
            limbs: core::array::from_fn(|limb| (coefficients[limb / 2] >> (32 * (limb % 2))) as u32),
        }
    }
}

// A 16-byte vector access across a 16-byte boundary is several times slower.
#[derive(Clone)]
#[repr(align(16))]
pub struct SignedShiftSums {
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
                self.terms <= MAX_EXACT_TERMS,
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
                self.terms <= MAX_EXACT_TERMS,
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

pub const SMALL_DIGIT_BOUND: u64 = 1 << 25;

const _: () = assert!((D as u128) * (1 << 31) * (SMALL_DIGIT_BOUND as u128) < (1u128 << 63));

pub struct SignedLimbs {
    low: [i32; D],
    high: [i32; D],
}

impl SignedLimbs {
    pub fn new(coefficients: &[F; D]) -> Self {
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

pub struct SmallDigits {
    windows: [i32; 2 * D - 1],
}

impl SmallDigits {
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
