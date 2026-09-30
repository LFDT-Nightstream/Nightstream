//! Multilinear prefixes with an implicit zero suffix over the full cube.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use crate::superneo_eval::EqualityWeights;

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
    fold_pairs(values, K::ZERO, |low, high| interpolate(low, high, challenge));
}

fn fold_pairs<T: Copy + Send + Sync>(values: &mut Vec<T>, zero: T, combine: impl Fn(T, T) -> T + Sync) {
    let previous_len = values.len();
    if previous_len == 0 {
        *values = Vec::new();
        return;
    }
    let next_len = previous_len.div_ceil(2);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let workers = rayon::current_num_threads();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let workers = 1;
    // Even chunks keep pairs together. Each worker writes only inside its
    // own chunk; compaction starts after every worker has finished reading.
    let chunk_len = 2 * next_len.div_ceil(workers);
    let fold_chunk = |chunk: &mut [T]| {
        for index in 0..chunk.len().div_ceil(2) {
            let low = chunk[2 * index];
            let high = chunk.get(2 * index + 1).copied().unwrap_or(zero);
            chunk[index] = combine(low, high);
        }
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    values.par_chunks_mut(chunk_len).for_each(fold_chunk);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    values.chunks_mut(chunk_len).for_each(fold_chunk);
    for start in (chunk_len..previous_len).step_by(chunk_len) {
        let count = (previous_len - start).min(chunk_len).div_ceil(2);
        values.copy_within(start..start + count, start / 2);
    }
    values.truncate(next_len);
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

    /// The pair codes of a prefix whose pairs take few distinct values.
    pub(super) fn early_pairs(&self) -> Option<EarlyPairs<'_>> {
        match self {
            Self::Input { witness, len } if *len != 0 => witness
                .packed_signed_unit_column_masks()
                .map(|(positive, negative)| EarlyPairs::Masks { positive, negative }),
            Self::Encoded { codes, values, zero } if values.len() <= MAX_EARLY_VALUES => Some(EarlyPairs::Encoded {
                codes,
                values,
                zero: *zero,
            }),
            _ => None,
        }
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
            if let Self::Encoded {
                codes,
                values,
                zero: current_zero,
            } = self
            {
                fold_pairs(codes, zero, |low, high| low + base as u16 * high);
                *values = next_values;
                *current_zero = zero + base as u16 * zero;
                return;
            }
            let current: &Self = self;
            let masks = current.early_pairs();
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
                Self::Encoded { .. } | Self::Folded(_) => unreachable!("only initial codes need an allocation"),
            };
            let pair_code = |index| match &masks {
                Some(pairs) => pairs.code(index) as u16,
                None => code(2 * index) + base as u16 * code(2 * index + 1),
            };
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

/// At most 81 values give at most 6,561 distinct pairs.
const MAX_EARLY_VALUES: usize = 81;

/// Pairs of an early prefix. Pair `index` is `(values[code % base],
/// values[code / base])` for `code = self.code(index)` and `base = values.len()`.
pub(super) enum EarlyPairs<'a> {
    /// Signed-unit input columns; codes 0, 1, 2 are -1, 0, 1.
    Masks {
        positive: &'a [u64],
        negative: &'a [u64],
    },
    Encoded {
        codes: &'a [u16],
        values: &'a [K],
        zero: u16,
    },
}

impl EarlyPairs<'_> {
    pub(super) fn values(&self) -> &[K] {
        const SIGNED_UNITS: [K; 3] = [K::NEG_ONE, K::ZERO, K::ONE];
        match self {
            Self::Masks { .. } => &SIGNED_UNITS,
            Self::Encoded { values, .. } => values,
        }
    }

    #[inline]
    pub(super) fn code(&self, index: usize) -> usize {
        match self {
            Self::Masks { positive, negative } => {
                // D is even, so both coordinates of a pair share one column.
                let (column, lane) = ((2 * index) / D, (2 * index) % D);
                let (Some(&positive), Some(&negative)) = (positive.get(column), negative.get(column)) else {
                    return 4;
                };
                let unit = |lane: usize| (1 + (positive >> lane & 1) - (negative >> lane & 1)) as usize;
                unit(lane) + 3 * unit(lane + 1)
            }
            Self::Encoded { codes, values, zero } => {
                let code = |index: usize| usize::from(codes.get(index).copied().unwrap_or(*zero));
                code(2 * index) + values.len() * code(2 * index + 1)
            }
        }
    }
}

/// `sum_index weight(offset + index) * norm_pair(pair(index))` over the
/// first `pairs` pairs. Weights are summed per pair code first; field sums
/// are exact, so each distinct pair's norm is evaluated once.
pub(super) fn early_norm_coefficients(
    early: &EarlyPairs<'_>,
    pairs: usize,
    weights: &EqualityWeights,
    offset: usize,
) -> [K; 4] {
    let values = early.values();
    let base = values.len();
    let zero = || vec![K::ZERO; base * base];
    let add = |mut buckets: Vec<K>, index: usize| {
        buckets[early.code(index)] += weights.at(offset + index);
        buckets
    };
    let merge = |mut left: Vec<K>, right: Vec<K>| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let buckets = (0..pairs)
        .into_par_iter()
        .with_min_len(1 << 14)
        .fold(zero, add)
        .reduce(zero, merge);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let buckets = (0..pairs).fold(zero(), add);
    let mut result = [K::ZERO; 4];
    for (code, weight) in buckets.into_iter().enumerate() {
        if weight == K::ZERO {
            continue;
        }
        for (total, value) in result
            .iter_mut()
            .zip(norm_pair(values[code % base], values[code / base]))
        {
            *total += value * weight;
        }
    }
    result
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
