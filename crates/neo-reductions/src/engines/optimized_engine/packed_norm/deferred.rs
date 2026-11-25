use neo_math::{ProductSum, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::super::{add_coefficients, from_norm_moments, norm_moments, Assignment};
use super::{codebook, PackedNorm, PackedSource, GROUP_LANES, PATTERNS, ZERO};
use crate::engines::pi_ccs_joint::gamma_power;
use crate::superneo_eval::EqualityWeights;

#[derive(Clone, Copy)]
struct CubicPattern {
    value: K,
    residual: K,
    triple_square: K,
    cube: K,
}

impl PackedSource<'_> {
    fn padded_pattern(&self, index: usize) -> usize {
        if index < self.groups {
            self.pattern(index)
        } else {
            ZERO
        }
    }

    fn face_value<const BYTES: usize>(&self, index: usize, lookup: &[[K; 256]; BYTES]) -> K {
        let byte = index * BYTES;
        if byte >= self.groups {
            return K::ZERO;
        }
        let column = byte / size_of::<u64>();
        let shift = byte % size_of::<u64>() * u8::BITS as usize;
        let positive = (self.positive[column] >> shift).to_le_bytes();
        let negative = (self.negative[column] >> shift).to_le_bytes();
        let mut value = K::ZERO;
        for (i, table) in lookup.iter().enumerate() {
            value += table[usize::from(positive[i])] - table[usize::from(negative[i])];
        }
        value
    }
}

impl PackedNorm<'_> {
    pub(in crate::engines::optimized_engine) fn defer(&self) -> bool {
        let pattern_bytes = self.sources.iter().fold(0usize, |bytes, (_, source)| {
            bytes.saturating_add(source.groups.saturating_mul(size_of::<u16>()))
        });
        self.sources.len() >= 2 && pattern_bytes >= PATTERNS * (size_of::<CubicPattern>() + size_of::<K>())
    }

    pub(in crate::engines::optimized_engine) fn discard_histogram(&mut self) {
        self.histogram = Vec::new();
    }

    pub(in crate::engines::optimized_engine) fn moments(
        &self,
        point: &[K],
        weights: &EqualityWeights,
        gamma: K,
    ) -> [K; 3] {
        let patterns: Vec<_> = codebook(point)
            .into_iter()
            .map(|value| {
                let square = value.square();
                let cube = square * value;
                CubicPattern {
                    value,
                    residual: cube - value,
                    triple_square: square.double() + square,
                    cube,
                }
            })
            .collect();
        let (low_weights, high_weights) = weights.factors();
        let source_moments = |(index, source): &(usize, PackedSource<'_>)| {
            let pairs = source.groups.div_ceil(2);
            let group_moments = |group: usize| {
                let start = group * low_weights.len();
                let mut sums = [ProductSum::default(); 3];
                for (i, &weight) in low_weights.iter().take(pairs - start).enumerate() {
                    let offset = 2 * (start + i);
                    let left = source.padded_pattern(offset);
                    let right = source.padded_pattern(offset + 1);
                    if left != right {
                        let low = patterns[left];
                        let high = patterns[right];
                        let mixed_low = low.triple_square * high.value;
                        let mixed_high = high.triple_square * low.value;
                        let quadratic_and_cubic = low.cube.double() + high.cube - mixed_low;
                        let cubic = high.cube - low.cube + mixed_low - mixed_high;
                        for (sum, term) in
                            sums.iter_mut()
                                .zip([high.residual - low.residual, quadratic_and_cubic, cubic])
                        {
                            *sum = sum.add(term, weight);
                        }
                    }
                }
                sums.map(|sum| sum.value() * high_weights[group])
            };
            let groups = 0..pairs.div_ceil(low_weights.len());
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let sums = groups
                .into_par_iter()
                .with_min_len(4)
                .map(group_moments)
                .reduce(|| [K::ZERO; 3], add_coefficients);
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let sums = groups
                .map(group_moments)
                .fold([K::ZERO; 3], add_coefficients);
            let weight = gamma_power(gamma, *index);
            sums.map(|v| v * weight)
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let totals = self
            .sources
            .par_iter()
            .map(source_moments)
            .reduce(|| [K::ZERO; 3], add_coefficients);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let totals = self
            .sources
            .iter()
            .map(source_moments)
            .fold([K::ZERO; 3], add_coefficients);
        let [endpoints, quadratic_and_cubic, cubic] = totals;
        [endpoints - quadratic_and_cubic, quadratic_and_cubic - cubic, cubic]
    }

    pub(in crate::engines::optimized_engine) fn face<const BYTES: usize>(
        &self,
        point: &[K],
        weights: &EqualityWeights,
        gamma: K,
        assignments: &mut [Assignment<'_>],
    ) -> [K; 3] {
        const { assert!(BYTES == 2 || BYTES == 4 || BYTES == 8) };
        let face_weights = EqualityWeights::new(point);
        let lookup: [[K; 256]; BYTES] = std::array::from_fn(|byte| {
            let mut table = [K::ZERO; 256];
            for mask in 1usize..256 {
                let lane = byte * GROUP_LANES + mask.trailing_zeros() as usize;
                table[mask] = table[mask & (mask - 1)] + face_weights.at(lane);
            }
            table
        });
        let (low_weights, high_weights) = weights.factors();
        let span = 2 * low_weights.len();
        let project = |(index, source): &(usize, PackedSource<'_>)| {
            let len = source.groups.div_ceil(BYTES);
            let group_moments = |group: usize, mut output: Option<&mut [K]>| {
                let start = group * span;
                let mut sums = [ProductSum::default(); 3];
                for (pair, &weight) in low_weights
                    .iter()
                    .take((len - start).div_ceil(2))
                    .enumerate()
                {
                    let at = pair * 2;
                    let low = source.face_value(start + at, &lookup);
                    let high = source.face_value(start + at + 1, &lookup);
                    if let Some(values) = output.as_deref_mut() {
                        values[at] = low;
                        if let Some(last) = values.get_mut(at + 1) {
                            *last = high;
                        }
                    }
                    norm_moments(&mut sums, low, high, weight);
                }
                sums.map(|sum| sum.value() * high_weights[group])
            };
            let (moments, dense) = if BYTES == size_of::<u64>() {
                let mut dense = vec![K::ZERO; len];
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let sums = dense
                    .par_chunks_mut(span)
                    .with_min_len(4)
                    .enumerate()
                    .map(|(i, values)| group_moments(i, Some(values)))
                    .reduce(|| [K::ZERO; 3], add_coefficients);
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let sums = dense
                    .chunks_mut(span)
                    .enumerate()
                    .map(|(i, values)| group_moments(i, Some(values)))
                    .fold([K::ZERO; 3], add_coefficients);
                (sums, dense)
            } else {
                let groups = 0..len.div_ceil(span);
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let sums = groups
                    .into_par_iter()
                    .with_min_len(4)
                    .map(|i| group_moments(i, None))
                    .reduce(|| [K::ZERO; 3], add_coefficients);
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let sums = groups
                    .map(|i| group_moments(i, None))
                    .fold([K::ZERO; 3], add_coefficients);
                (sums, Vec::new())
            };
            let weight = gamma_power(gamma, *index);
            (*index, from_norm_moments(moments).map(|v| v * weight), dense)
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let results: Vec<_> = self.sources.par_iter().map(project).collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let results: Vec<_> = self.sources.iter().map(project).collect();
        if BYTES == size_of::<u64>() {
            for assignment in assignments.iter_mut() {
                *assignment = Assignment::Folded(Vec::new());
            }
        }
        let mut total = [K::ZERO; 3];
        for (index, coefficients, dense) in results {
            if BYTES == size_of::<u64>() {
                assignments[index] = Assignment::Folded(dense);
            }
            total = add_coefficients(total, coefficients);
        }
        total
    }
}
