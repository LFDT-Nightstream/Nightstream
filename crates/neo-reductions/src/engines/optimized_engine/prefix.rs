//! Multilinear prefixes with an implicit zero suffix over the full cube.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

pub(super) fn pair(values: &[K], index: usize) -> (K, K) {
    (
        values.get(2 * index).copied().unwrap_or(K::ZERO),
        values.get(2 * index + 1).copied().unwrap_or(K::ZERO),
    )
}

pub(super) fn interpolate(low: K, high: K, point: K) -> K {
    low + (high - low) * point
}

pub(super) fn fold(values: &mut Vec<K>, challenge: K) {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let next = values
        .par_chunks(2)
        .map(|pair| interpolate(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), challenge))
        .collect();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let next = values
        .chunks(2)
        .map(|pair| interpolate(pair[0], pair.get(1).copied().unwrap_or(K::ZERO), challenge))
        .collect();
    *values = next;
}

/// The input stays in the caller's existing packed, constant or dense Mat.
/// Extension-field storage is allocated only after the first fold halves it.
pub(super) enum Assignment<'a> {
    Input { witness: &'a Mat<F>, len: usize },
    Folded(Vec<K>),
}

impl<'a> Assignment<'a> {
    pub(super) fn new(witness: &'a Mat<F>, width: usize) -> Self {
        let len = if witness
            .virtual_constant_value()
            .is_some_and(|value| *value == F::ZERO)
        {
            0
        } else if let Some((positive, negative)) = witness.packed_signed_unit_column_masks() {
            positive
                .iter()
                .zip(negative)
                .rposition(|(&p, &n)| p | n != 0)
                .map_or(0, |block| {
                    block * D + (u64::BITS - (positive[block] | negative[block]).leading_zeros()) as usize
                })
        } else {
            let mut len = width;
            while len != 0 && witness[((len - 1) % D, (len - 1) / D)] == F::ZERO {
                len -= 1;
            }
            len
        };
        Self::Input { witness, len }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Input { len, .. } => *len,
            Self::Folded(values) => values.len(),
        }
    }

    pub(super) fn get(&self, index: usize) -> K {
        match self {
            Self::Input { witness, len } if index < *len => K::from(witness[(index % D, index / D)]),
            Self::Input { .. } => K::ZERO,
            Self::Folded(values) => values.get(index).copied().unwrap_or(K::ZERO),
        }
    }

    pub(super) fn pair(&self, index: usize) -> (K, K) {
        (self.get(2 * index), self.get(2 * index + 1))
    }

    pub(super) fn fold(&mut self, challenge: K) {
        if let Self::Folded(values) = self {
            fold(values, challenge);
            return;
        }
        let pairs = self.len().div_ceil(2);
        let value = |index| {
            let (low, high) = self.pair(index);
            interpolate(low, high, challenge)
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let next = (0..pairs).into_par_iter().map(value).collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let next = (0..pairs).map(value).collect();
        *self = Self::Folded(next);
    }
}

pub(super) fn norm_pair(low: K, high: K) -> [K; 4] {
    let delta = high - low;
    let three = K::from(F::from_u64(3));
    let low_squared = low * low;
    let delta_squared = delta * delta;
    [
        low_squared * low - low,
        delta * (three * low_squared - K::ONE),
        three * low * delta_squared,
        delta_squared * delta,
    ]
}
