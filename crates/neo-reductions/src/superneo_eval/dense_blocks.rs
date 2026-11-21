use std::borrow::Cow;

use super::{as_base_field, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub(super) enum DenseBlocks {
    Wide(Vec<Rq>),
    Centered(Vec<[i8; D]>),
}

fn centered(value: F) -> i64 {
    let signed = value.as_canonical_u64() as i64;
    signed + ((signed >> 63) & neo_math::field::Q.wrapping_neg() as i64)
}

impl DenseBlocks {
    pub(super) fn try_centered<Ff>(input: &[Ff], blocks: usize) -> Option<(Self, Vec<bool>)>
    where
        Ff: Field + Copy,
        K: From<Ff>,
    {
        if blocks < 2048 {
            return None;
        }
        let fits = |&value: &Ff| i8::try_from(centered(as_base_field(value))).is_ok();
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let fits = input.par_iter().with_min_len(8192).all(fits);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let fits = input.iter().all(fits);
        if !fits {
            return None;
        }
        let column = |index| {
            let row: [i8; D] = std::array::from_fn(|lane| centered(as_base_field(input[lane * blocks + index])) as i8);
            let nonzero = row.iter().any(|&value| value != 0);
            (row, nonzero)
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let (data, nonzero) = {
            let (mut rows, mut flags) = (Vec::new(), Vec::new());
            (0..blocks)
                .into_par_iter()
                .with_min_len(512)
                .map(column)
                .unzip_into_vecs(&mut rows, &mut flags);
            (rows, flags)
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let (data, nonzero) = (0..blocks).map(column).unzip();
        Some((Self::Centered(data), nonzero))
    }

    pub(super) fn event_digits(&self, block: usize) -> Option<neo_math::signed_sums::SmallEventDigits> {
        use neo_math::signed_sums::SmallEventDigits;
        match self {
            Self::Wide(rows) => SmallEventDigits::from_centered(&rows[block].0),
            Self::Centered(rows) => Some(SmallEventDigits::from_i8(&rows[block])),
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Wide(r) => r.len(),
            Self::Centered(r) => r.len(),
        }
    }
    #[inline]
    pub(super) fn coefficient(&self, block: usize, lane: usize) -> F {
        match self {
            Self::Wide(r) => r[block].0[lane],
            Self::Centered(r) => F::from_i8(r[block][lane]),
        }
    }
    #[inline]
    pub(super) fn ring(&self, block: usize) -> Cow<'_, Rq> {
        match self {
            Self::Wide(r) => Cow::Borrowed(&r[block]),
            Self::Centered(r) => Cow::Owned(Rq(r[block].map(F::from_i8))),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/centered_blocks.rs"]
mod tests;
