use neo_math::{sum_of_scaled, ProductSum, D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{EqualityWeights, RealBlockStorage, SuperneoZBlocks};

pub(crate) const LANE_BITS: usize = D.ilog2() as usize;
const TASK_BLOCKS: usize = 2048;
const _: () = assert!(D.is_power_of_two());

pub(crate) struct PadProjection([K; D]);

impl PadProjection {
    pub(crate) fn new(ratio: K) -> Self {
        let mut power = K::ONE;
        let powers = std::array::from_fn(|_| {
            let next = power;
            power *= ratio;
            next
        });
        Self(bar(powers))
    }

    pub(crate) fn contract_blocks(&self, witness: &SuperneoZBlocks, high_point: &[K]) -> [K; D] {
        let high = EqualityWeights::new(high_point);
        let task = |part: usize| {
            let start = part * TASK_BLOCKS;
            let end = (start + TASK_BLOCKS).min(witness.block_len());
            match &witness.re {
                RealBlockStorage::Dense { blocks, nonzero } => {
                    let mut sums = [ProductSum::default(); D];
                    for block in start..end {
                        if !nonzero[block] {
                            continue;
                        }
                        let weight = high.at(block);
                        let digits = blocks.ring(block);
                        for (sum, &value) in sums.iter_mut().zip(&digits.0) {
                            *sum = sum.add_scaled(value, weight);
                        }
                    }
                    sums.map(ProductSum::value)
                }
                RealBlockStorage::SignedUnit { positive, negative } => {
                    let mut sums = [K::ZERO; D];
                    for block in start..end {
                        let weight = high.at(block);
                        let mut signs = positive[block];
                        while signs != 0 {
                            sums[signs.trailing_zeros() as usize] += weight;
                            signs &= signs - 1;
                        }
                        let mut signs = negative[block];
                        while signs != 0 {
                            sums[signs.trailing_zeros() as usize] -= weight;
                            signs &= signs - 1;
                        }
                    }
                    sums
                }
                RealBlockStorage::Zero { .. } => [K::ZERO; D],
            }
        };
        let tasks = 0..witness.block_len().div_ceil(TASK_BLOCKS);
        let add = |left: [K; D], right: [K; D]| std::array::from_fn(|i| left[i] + right[i]);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let lanes = tasks.into_par_iter().map(task).reduce(|| [K::ZERO; D], add);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let lanes = tasks.map(task).fold([K::ZERO; D], add);
        ring_product(&lanes, &self.0)
    }

    pub(crate) fn add_contracted_lanes(&self, witness: &SuperneoZBlocks, low_point: &[K; LANE_BITS], output: &mut [K]) {
        let low = EqualityWeights::new(low_point);
        let point = std::array::from_fn(|lane| low.at(lane));
        let form = bar(ring_product(&bar(point), &self.0));
        let add = |(block, value): (usize, &mut K)| {
            if witness.real_nonzero(block) {
                *value += match &witness.re {
                    RealBlockStorage::Dense { blocks, .. } => sum_of_scaled(blocks.ring(block).0.into_iter().zip(form)),
                    _ => sum_of_scaled((0..D).map(|lane| (witness.real_coefficient(block, lane), form[lane]))),
                };
            }
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        output
            .par_iter_mut()
            .with_min_len(TASK_BLOCKS)
            .enumerate()
            .for_each(add);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        output.iter_mut().enumerate().for_each(add);
    }
}

fn bar(input: [K; D]) -> [K; D] {
    std::array::from_fn(|lane| if lane == 0 { input[0] } else { -input[D - lane] })
}

fn ring_product(a: &[K; D], b: &[K; D]) -> [K; D] {
    std::array::from_fn(|lane| {
        let mut sum = ProductSum::default();
        for (i, &value) in a.iter().enumerate() {
            let degree = lane + D - i;
            let coefficient = if degree >= D { b[degree - D] } else { -b[degree] };
            sum = sum.add(value, coefficient);
        }
        sum.value()
    })
}
