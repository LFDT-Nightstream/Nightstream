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

/// Signed-unit prefixes share their possible field values. Early folds store
/// u16 indices into that table instead of one extension-field value per entry.
/// Other inputs use the ordinary dense fold.
pub(super) enum Assignment<'a> {
    Input {
        witness: &'a Mat<F>,
        len: usize,
    },
    Encoded {
        codes: Vec<u16>,
        values: Vec<K>,
        zero: u16,
    },
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
            Self::Encoded { codes, .. } => codes.len(),
            Self::Folded(values) => values.len(),
        }
    }

    pub(super) fn get(&self, index: usize) -> K {
        match self {
            Self::Input { witness, len } if index < *len => K::from(witness[(index % D, index / D)]),
            Self::Input { .. } => K::ZERO,
            Self::Encoded { codes, values, .. } => codes
                .get(index)
                .map_or(K::ZERO, |&code| values[usize::from(code)]),
            Self::Folded(values) => values.get(index).copied().unwrap_or(K::ZERO),
        }
    }

    pub(super) fn pair(&self, index: usize) -> (K, K) {
        (self.get(2 * index), self.get(2 * index + 1))
    }

    pub(super) fn fold(&mut self, challenge: K) {
        let encoding = match self {
            Self::Input { witness, len } if *len != 0 && witness.is_packed_signed_unit() => {
                Some((vec![-K::ONE, K::ZERO, K::ONE], 1u16))
            }
            Self::Encoded { values, zero, .. } if values.len() * values.len() <= usize::from(u16::MAX) + 1 => {
                Some((values.clone(), *zero))
            }
            _ => None,
        };
        if let Some((values, zero)) = encoding {
            let base = values.len();
            let next_values = (0..base * base)
                .map(|code| interpolate(values[code % base], values[code / base], challenge))
                .collect();
            let current: &Self = self;
            let code = |index| match current {
                Self::Input { witness, len } if index < *len => {
                    let value = witness[(index % D, index / D)];
                    if value == -F::ONE {
                        0
                    } else if value == F::ZERO {
                        1
                    } else {
                        2
                    }
                }
                Self::Input { .. } => zero,
                Self::Encoded { codes, .. } => codes.get(index).copied().unwrap_or(zero),
                Self::Folded(_) => unreachable!("dense values have no signed-unit code"),
            };
            let pair_code = |index| code(2 * index) + base as u16 * code(2 * index + 1);
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let codes = (0..self.len().div_ceil(2))
                .into_par_iter()
                .map(pair_code)
                .collect();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let codes = (0..self.len().div_ceil(2)).map(pair_code).collect();
            *self = Self::Encoded {
                codes,
                values: next_values,
                zero: zero + base as u16 * zero,
            };
            return;
        }
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

#[cfg(test)]
#[path = "../../../tests/unit/assignment_prefix.rs"]
mod tests;

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
