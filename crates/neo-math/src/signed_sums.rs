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

const DIGITS_PER_GROUP: usize = 4;
const _: () = assert!(D.is_multiple_of(DIGITS_PER_GROUP));
const GROUP_MASK: usize = (1 << DIGITS_PER_GROUP) - 1;
const PATTERNS: usize = 3usize.pow((DIGITS_PER_GROUP - 1) as u32);
const PATTERN_LIMBS: usize = 2 * (2 * D - 1);
const BUCKET_LIMBS: usize = PATTERNS * PATTERN_LIMBS;
const SIGNED_PATTERNS: [u8; 1 << (2 * DIGITS_PER_GROUP)] = {
    let mut patterns = [0; 1 << (2 * DIGITS_PER_GROUP)];
    let mut index = 0;
    while index < patterns.len() {
        let positive = index & GROUP_MASK;
        let negative = index >> DIGITS_PER_GROUP;
        let first_is_negative = (negative & 1) as u8;
        let sign = 1 - 2 * first_is_negative as i8;
        let mut pattern = 0;
        let mut lane = 1;
        while lane < DIGITS_PER_GROUP {
            let digit = ((positive >> lane) & 1) as i8 - ((negative >> lane) & 1) as i8;
            pattern += (1 + sign * digit) as u8 * 3u8.pow((lane - 1) as u32);
            lane += 1;
        }
        patterns[index] = 2 * pattern + first_is_negative;
        index += 1;
    }
    patterns
};

pub struct SignedUnitBuckets {
    patterns: Box<[i64; BUCKET_LIMBS]>,
    #[cfg(debug_assertions)]
    terms: u64,
}

impl SignedUnitBuckets {
    pub fn min_saved_shifts(occupied: u64) -> usize {
        let pairs = occupied | (occupied >> 1);
        let groups = (pairs | (pairs >> 2)) & (u64::MAX / GROUP_MASK as u64);
        (occupied.count_ones() - groups.count_ones()) as usize
    }

    pub fn zero() -> Self {
        Self {
            patterns: vec![0; BUCKET_LIMBS]
                .into_boxed_slice()
                .try_into()
                .expect("fixed bucket length"),
            #[cfg(debug_assertions)]
            terms: 0,
        }
    }

    #[inline]
    pub fn add_signed_units(&mut self, element: &SplitRing, positive: u64, negative: u64) {
        #[cfg(debug_assertions)]
        {
            self.terms += 1;
            debug_assert!(self.terms <= MAX_EXACT_TERMS);
        }
        let patterns = self.patterns.as_chunks_mut::<PATTERN_LIMBS>().0;
        let mut live = positive ^ negative;
        while live != 0 {
            let degree = live.trailing_zeros() as usize;
            let plus = ((positive >> degree) as usize) & GROUP_MASK;
            let minus = ((negative >> degree) as usize) & GROUP_MASK;
            let index = plus | (minus << DIGITS_PER_GROUP);
            let pattern = SIGNED_PATTERNS[index];
            let accum = &mut patterns[usize::from(pattern >> 1)][2 * degree..][..2 * D];
            if pattern & 1 == 0 {
                for (sum, &limb) in accum.iter_mut().zip(&element.limbs) {
                    *sum += i64::from(limb);
                }
            } else {
                for (sum, &limb) in accum.iter_mut().zip(&element.limbs) {
                    *sum -= i64::from(limb);
                }
            }
            live &= !((GROUP_MASK as u64) << degree);
        }
    }

    pub fn add_to(&self, total: &mut SignedShiftSums) {
        #[cfg(debug_assertions)]
        {
            total.terms += self.terms;
            debug_assert!(total.terms <= MAX_EXACT_TERMS);
        }
        let patterns = self.patterns.as_chunks::<PATTERN_LIMBS>().0;
        for (index, values) in patterns.iter().enumerate() {
            for (sum, value) in total.limbs.iter_mut().zip(values) {
                *sum += value;
            }
            let mut pattern = index;
            for offset in 1..DIGITS_PER_GROUP {
                let digit = pattern % 3;
                pattern /= 3;
                let target = &mut total.limbs[2 * offset..];
                match digit {
                    2 => {
                        for (sum, value) in target.iter_mut().zip(values) {
                            *sum += value;
                        }
                    }
                    0 => {
                        for (sum, value) in target.iter_mut().zip(values) {
                            *sum -= value;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

const MAX_EVENT_DIGIT: u64 = 2048;

pub struct SmallEventDigits([i32; D]);

impl SmallEventDigits {
    pub fn from_i8(digits: &[i8; D]) -> Self {
        Self(digits.map(i32::from))
    }
    pub fn new(digits: &[F; D]) -> Option<Self> {
        let mut small = [0i32; D];
        for (out, input) in small.iter_mut().zip(digits) {
            let value = input.as_canonical_u64();
            if value <= MAX_EVENT_DIGIT {
                *out = value as i32;
            } else if Q - value <= MAX_EVENT_DIGIT {
                *out = -((Q - value) as i32);
            } else {
                return None;
            }
        }
        Some(Self(small))
    }

    pub fn from_centered(digits: &[F; D]) -> Option<Self> {
        let mut small = [0i32; D];
        let mut valid = true;
        for (out, input) in small.iter_mut().zip(digits) {
            let signed = input.as_canonical_u64() as i64;
            let centered = signed + ((signed >> 63) & Q.wrapping_neg() as i64);
            valid &= centered.unsigned_abs() <= MAX_EVENT_DIGIT;
            *out = centered as i32;
        }
        valid.then_some(Self(small))
    }
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
        self.count_term();
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

    #[inline]
    pub fn add_signed_event_pair(
        output: [&mut Self; 2],
        coefficients: [&[F; D]; 2],
        mut events: u64,
        positive: u64,
        negative: u64,
    ) {
        let [real, imaginary] = output;
        real.count_term();
        imaginary.count_term();
        let (positive, negative) = (positive & !negative, negative & !positive);
        while events != 0 {
            let degree = events.trailing_zeros() as usize;
            events &= events - 1;
            let [a, b] = coefficients.map(|form| form[degree].as_canonical_u64());
            if a | b == 0 {
                continue;
            }
            let [a_low, a_high, b_low, b_high] =
                [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32].map(i64::from);
            let re = &mut real.limbs.as_chunks_mut::<2>().0[degree..][..D];
            let im = &mut imaginary.limbs.as_chunks_mut::<2>().0[degree..][..D];
            let mut mask = positive;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                re[lane][0] += a_low;
                re[lane][1] += a_high;
                im[lane][0] += b_low;
                im[lane][1] += b_high;
                mask &= mask - 1;
            }
            let mut mask = negative;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                re[lane][0] -= a_low;
                re[lane][1] -= a_high;
                im[lane][0] -= b_low;
                im[lane][1] -= b_high;
                mask &= mask - 1;
            }
        }
    }

    #[inline]
    fn count_term(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.terms += 1;
            debug_assert!(
                self.terms <= MAX_EXACT_TERMS,
                "signed shift sums exceed the exact i64 term bound"
            );
        }
    }

    pub fn add_small_event_pair(output: [&mut Self; 2], forms: [&[F; D]; 2], events: u64, digits: &[F; D]) -> bool {
        let Some(small) = SmallEventDigits::new(digits) else {
            return false;
        };
        Self::add_prepared_event_pair(output, forms, events, &small);
        true
    }

    pub fn add_prepared_event_pair(
        output: [&mut Self; 2],
        forms: [&[F; D]; 2],
        mut events: u64,
        digits: &SmallEventDigits,
    ) {
        let [real, imaginary] = output;
        #[cfg(debug_assertions)]
        for accum in [&mut *real, &mut *imaginary] {
            accum.terms += MAX_EVENT_DIGIT;
            debug_assert!(accum.terms <= MAX_EXACT_TERMS);
        }
        while events != 0 {
            let degree = events.trailing_zeros() as usize;
            events &= events - 1;
            let [x, y] = forms.map(|f| f[degree].as_canonical_u64());
            if x | y == 0 {
                continue;
            }
            let parts = [x as u32, (x >> 32) as u32, y as u32, (y >> 32) as u32].map(i64::from);
            let re = &mut real.limbs.as_chunks_mut::<2>().0[degree..][..D];
            let im = &mut imaginary.limbs.as_chunks_mut::<2>().0[degree..][..D];
            for ((a, b), s) in re.iter_mut().zip(im).zip(digits.0) {
                let d = i64::from(s);
                a[0] += parts[0] * d;
                a[1] += parts[1] * d;
                b[0] += parts[2] * d;
                b[1] += parts[3] * d;
            }
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
