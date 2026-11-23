#[path = "packed_norm/deferred.rs"]
mod deferred;

use neo_math::{D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{add_coefficients, interpolate, norm_pair, Assignment};
use crate::superneo_eval::EqualityWeights;

pub(in crate::engines::optimized_engine) const PACKED_NORM_ROUNDS: usize = 3;
const GROUP_LANES: usize = 1 << PACKED_NORM_ROUNDS;
const PATTERNS: usize = 3usize.pow(GROUP_LANES as u32);
const ZERO: usize = (PATTERNS - 1) / 2;
const _: () = assert!(D.is_multiple_of(GROUP_LANES) && PATTERNS <= u16::MAX as usize + 1);
const MASK_CODES: [usize; 1 << GROUP_LANES] = {
    let mut codes = [0; 1 << GROUP_LANES];
    let mut mask = 1usize;
    while mask < codes.len() {
        codes[mask] = codes[mask & (mask - 1)] + 3usize.pow(mask.trailing_zeros());
        mask += 1;
    }
    codes
};

struct PackedSource<'a> {
    positive: &'a [u64],
    negative: &'a [u64],
    groups: usize,
}

impl PackedSource<'_> {
    fn histogram(&self, weights: &EqualityWeights) -> Vec<K> {
        let zero = || vec![K::ZERO; PATTERNS];
        let add = |mut sums: Vec<K>, index: usize| {
            let pattern = self.pattern(index);
            if pattern != 0 && pattern != ZERO && pattern + 1 != PATTERNS {
                sums[pattern] += weights.at(index);
            }
            sums
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let sums = (0..self.groups)
            .into_par_iter()
            .with_min_len(1 << 14)
            .fold(zero, add)
            .reduce(zero, |mut left, right| {
                for (total, value) in left.iter_mut().zip(right) {
                    *total += value;
                }
                left
            });
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let sums = (0..self.groups).fold(zero(), add);
        sums
    }

    fn pattern(&self, index: usize) -> usize {
        let block = index / (D / GROUP_LANES);
        let shift = (index % (D / GROUP_LANES)) * GROUP_LANES;
        let mask = (1 << GROUP_LANES) - 1;
        let p = (self.positive[block] >> shift) & mask;
        let n = (self.negative[block] >> shift) & mask;
        ZERO + MASK_CODES[p as usize] - MASK_CODES[n as usize]
    }

    fn into_assignment(self, point: &[K; PACKED_NORM_ROUNDS]) -> Assignment<'static> {
        let code = |index| self.pattern(index) as u16;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let codes = (0..self.groups)
            .into_par_iter()
            .with_min_len(1 << 12)
            .map(code)
            .collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let codes = (0..self.groups).map(code).collect();
        Assignment::Encoded {
            codes,
            values: codebook(point),
            zero: ZERO as u16,
        }
    }
}

pub(in crate::engines::optimized_engine) struct PackedNorm<'a> {
    sources: Vec<(usize, PackedSource<'a>)>,
    histogram: Vec<K>,
}

impl<'a> PackedNorm<'a> {
    pub(in crate::engines::optimized_engine) fn new(
        assignments: &[Assignment<'a>],
        alpha: &[K],
        gamma: K,
    ) -> Option<Self> {
        let sources = assignments
            .iter()
            .enumerate()
            .filter(|(_, a)| a.len() != 0)
            .map(|(index, assignment)| {
                let Assignment::Input { witness, len } = assignment else {
                    return None;
                };
                let (positive, negative) = witness.packed_signed_unit_column_masks()?;
                Some((
                    index,
                    PackedSource {
                        positive,
                        negative,
                        groups: len.div_ceil(GROUP_LANES),
                    },
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        let weights = EqualityWeights::new(&alpha[PACKED_NORM_ROUNDS..]);
        let histogram = sources
            .iter()
            .map(|(index, source)| {
                let mut bins = source.histogram(&weights);
                let weight = crate::engines::pi_ccs_joint::gamma_power(gamma, *index);
                if weight != K::ONE {
                    for value in &mut bins {
                        *value *= weight;
                    }
                }
                bins
            })
            .reduce(|mut total, bins| {
                for (sum, contribution) in total.iter_mut().zip(bins) {
                    *sum += contribution;
                }
                total
            })?;
        Some(Self { sources, histogram })
    }

    pub(in crate::engines::optimized_engine) fn coefficients(&self, point: &[K], tail: &[K]) -> [K; 4] {
        let values = codebook(point);
        let base = values.len();
        let polynomial = |pattern: usize| norm_pair(values[pattern % base], values[pattern / base]);
        if tail.is_empty() {
            return self
                .histogram
                .iter()
                .enumerate()
                .filter(|(_, weight)| **weight != K::ZERO)
                .fold([K::ZERO; 4], |total, (pattern, &weight)| {
                    add_coefficients(total, polynomial(pattern).map(|coefficient| coefficient * weight))
                });
        }
        let radix = base * base;
        let polys: Vec<_> = (0..radix).map(polynomial).collect();
        let weights = EqualityWeights::new(tail);
        self.histogram
            .iter()
            .enumerate()
            .filter(|(_, weight)| **weight != K::ZERO)
            .fold([K::ZERO; 4], |mut total, (mut pattern, &weight)| {
                for face in 0..(1 << tail.len()) {
                    let weight = weight * weights.at(face);
                    total = add_coefficients(total, polys[pattern % radix].map(|coefficient| coefficient * weight));
                    pattern /= radix;
                }
                total
            })
    }

    pub(in crate::engines::optimized_engine) fn bind(
        self,
        assignments: &mut [Assignment<'a>],
        point: &[K; PACKED_NORM_ROUNDS],
    ) {
        for table in assignments.iter_mut() {
            *table = Assignment::Folded(Vec::new());
        }
        for (index, source) in self.sources {
            assignments[index] = source.into_assignment(point);
        }
    }
}

fn codebook(point: &[K]) -> Vec<K> {
    let mut values = vec![-K::ONE, K::ZERO, K::ONE];
    for &challenge in point {
        let base = values.len();
        values = (0..base * base)
            .map(|code| interpolate(values[code % base], values[code / base], challenge))
            .collect();
    }
    values
}

#[cfg(test)]
#[path = "../../../tests/unit/packed_norm.rs"]
mod tests;

pub(in crate::engines::optimized_engine) fn encoded_moments(
    codes: &[u16],
    values: &[K],
    zero: u16,
    weights: &EqualityWeights,
) -> [K; 3] {
    let residuals: Vec<_> = values
        .iter()
        .map(|&value| value * value * value - value)
        .collect();
    let (low, high) = weights.factors();
    let group = |(index, codes): (usize, &[u16])| {
        let mut products = [neo_math::ProductSum::default(); 3];
        for (pair, &weight) in codes.chunks(2).zip(low) {
            let l = usize::from(pair[0]);
            let h = usize::from(pair.get(1).copied().unwrap_or(zero));
            let delta = values[h] - values[l];
            if delta != K::ZERO {
                let square = delta * delta;
                let terms = [
                    residuals[h] - residuals[l],
                    square * (values[l] + values[l] + values[h]),
                    square * delta,
                ];
                for (product, value) in products.iter_mut().zip(terms) {
                    *product = product.add(value, weight);
                }
            }
        }
        products.map(|product| product.value() * high[index])
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let totals = codes
        .par_chunks(low.len() * 2)
        .with_min_len(super::MIN_JOB_PAIRS.div_ceil(low.len()))
        .enumerate()
        .map(group)
        .reduce(|| [K::ZERO; 3], add_coefficients);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let totals = codes
        .chunks(low.len() * 2)
        .enumerate()
        .map(group)
        .fold([K::ZERO; 3], add_coefficients);
    let [endpoints, quadratic, cubic] = totals;
    [endpoints - quadratic, quadratic - cubic, cubic]
}

pub(in crate::engines::optimized_engine) fn bind_encoded(
    codes: &[u16],
    values: &[K],
    zero: u16,
    challenge: K,
    weights: &EqualityWeights,
) -> (Vec<K>, [K; 3]) {
    let mut dense = vec![K::ZERO; codes.len().div_ceil(2)];
    let (low, high) = weights.factors();
    let group = |(index, group): (usize, &mut [K])| {
        let get = |index| values[usize::from(codes.get(index).copied().unwrap_or(zero))];
        let mut products = [neo_math::ProductSum::default(); 3];
        for (local, (pair, &weight)) in group.chunks_mut(2).zip(low).enumerate() {
            let at = 4 * (index * low.len() + local);
            let a = interpolate(get(at), get(at + 1), challenge);
            let b = interpolate(get(at + 2), get(at + 3), challenge);
            pair[0] = a;
            if pair.len() == 2 {
                pair[1] = b;
            }
            for (product, value) in products
                .iter_mut()
                .zip(super::norm_higher_coefficients(a, b))
            {
                *product = product.add(value, weight);
            }
        }
        products.map(|product| product.value() * high[index])
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let totals = dense
        .par_chunks_mut(low.len() * 2)
        .with_min_len(super::MIN_JOB_PAIRS.div_ceil(low.len()))
        .enumerate()
        .map(group)
        .reduce(|| [K::ZERO; 3], add_coefficients);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let totals = dense
        .chunks_mut(low.len() * 2)
        .enumerate()
        .map(group)
        .fold([K::ZERO; 3], add_coefficients);
    (dense, totals)
}
