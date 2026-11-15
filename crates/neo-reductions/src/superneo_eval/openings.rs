//! Complete real-witness openings shared by the normal PiCCS and PiDEC paths.

use neo_ccs::V1_2Evaluations;
use neo_math::{superneo_bar_block, KExtensions, D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::block_sums::TASK_BLOCKS;
use super::{EqualityWeights, SuperneoEvalCache, SuperneoZBlocks};
use crate::PiCcsError;

impl SuperneoEvalCache {
    pub fn eval_real_v1_2_openings(
        &self,
        point: &[K],
        witnesses: &[SuperneoZBlocks],
    ) -> Result<Vec<V1_2Evaluations<K>>, PiCcsError> {
        self.eval_real_v1_2_openings_reusing(point, witnesses, Vec::new())
    }

    pub(crate) fn eval_real_v1_2_openings_reusing(
        &self,
        point: &[K],
        witnesses: &[SuperneoZBlocks],
        storage: Vec<K>,
    ) -> Result<Vec<V1_2Evaluations<K>>, PiCcsError> {
        #[cfg(feature = "perf-timers")]
        let started = std::time::Instant::now();
        let (rows, width, matrix_count) = self
            .relation_shape()
            .ok_or_else(|| PiCcsError::InvalidInput("opening cache shape is inconsistent".into()))?;
        let variables = (usize::BITS - rows.max(width).saturating_sub(1).leading_zeros()) as usize;
        if width % D != 0
            || point.len() < variables
            || witnesses
                .iter()
                .any(|witness| witness.block_len() != width / D || !witness.imag_all_zero)
        {
            return Err(PiCcsError::InvalidInput("real opening point or witness shape".into()));
        }
        if witnesses.iter().all(SuperneoZBlocks::real_is_zero) {
            return Ok((0..witnesses.len())
                .map(|_| V1_2Evaluations {
                    eval_k: vec![K::ZERO; D],
                    eval_a: vec![vec![K::ZERO; D]; matrix_count],
                })
                .collect());
        }
        let weights = EqualityWeights::new(point);
        let row_weights = (0..rows).map(|row| weights.at(row)).collect::<Vec<_>>();
        let matrices = self.eval_ring_linear_forms_reusing(&row_weights, rows, witnesses, storage);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "[real-openings] matrices elapsed={:.3}s witnesses={}",
            started.elapsed().as_secs_f64(),
            witnesses.len()
        );
        let result = pad_openings(witnesses, point)
            .into_iter()
            .zip(matrices)
            .map(|(eval_k, eval_a)| V1_2Evaluations {
                eval_k: eval_k.to_vec(),
                eval_a: eval_a
                    .into_iter()
                    .map(|coefficients| coefficients.to_vec())
                    .collect(),
            })
            .collect();
        #[cfg(feature = "perf-timers")]
        eprintln!("[real-openings] pad elapsed={:.3}s", started.elapsed().as_secs_f64());
        Ok(result)
    }
}

/// `D` is a power of two, so ring element `block` covers the low cube at the high index `block`.
const _: () = assert!(D.is_power_of_two());

pub(super) fn pad_openings(witnesses: &[SuperneoZBlocks], point: &[K]) -> Vec<[K; D]> {
    let active: Vec<_> = witnesses
        .iter()
        .filter(|witness| !witness.real_is_zero())
        .collect();
    let Some(blocks) = active.first().map(|witness| witness.block_len()) else {
        return vec![[K::ZERO; D]; witnesses.len()];
    };
    let (low_point, high_point) = point.split_at(D.trailing_zeros() as usize);
    let low = EqualityWeights::new(low_point);
    let high = EqualityWeights::new(high_point);
    let template: [K; D] = {
        let values: [K; D] = std::array::from_fn(|lane| low.at(lane));
        let real = superneo_bar_block(values.map(|value| value.as_coeffs()[0]));
        let imaginary = superneo_bar_block(values.map(|value| value.as_coeffs()[1]));
        std::array::from_fn(|lane| K::from_coeffs([real[lane], imaginary[lane]]))
    };
    let zero = || vec![[K::ZERO; D]; active.len()];
    let task = |task: usize| {
        let mut sums = zero();
        for block in task * TASK_BLOCKS..((task + 1) * TASK_BLOCKS).min(blocks) {
            let weight = high.at(block);
            if weight == K::ZERO {
                continue;
            }
            for (sum, witness) in sums.iter_mut().zip(&active) {
                if !witness.real_nonzero(block) {
                    continue;
                }
                if let Some((positive, negative)) = witness.signed_unit_masks() {
                    let mut mask = positive[block];
                    while mask != 0 {
                        sum[mask.trailing_zeros() as usize] += weight;
                        mask &= mask - 1;
                    }
                    mask = negative[block];
                    while mask != 0 {
                        sum[mask.trailing_zeros() as usize] -= weight;
                        mask &= mask - 1;
                    }
                } else {
                    for (lane, value) in sum.iter_mut().enumerate() {
                        *value += weight * K::from(witness.real_coefficient(block, lane));
                    }
                }
            }
        }
        sums
    };
    let combine = |mut left: Vec<[K; D]>, right: Vec<[K; D]>| {
        for (left, right) in left.iter_mut().zip(right) {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
        }
        left
    };
    let tasks = 0..blocks.div_ceil(TASK_BLOCKS);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let sums = tasks.into_par_iter().map(task).reduce(zero, combine);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let sums = tasks.map(task).fold(zero(), combine);
    let mut sums = sums.iter().map(|sum| {
        let mut result = [K::ZERO; D];
        add_ring_product(&mut result, sum, &template);
        result
    });
    witnesses
        .iter()
        .map(|witness| {
            if witness.real_is_zero() {
                return [K::ZERO; D];
            }
            sums.next().expect("one sum per active witness")
        })
        .collect()
}

fn add_ring_product(output: &mut [K; D], left: &[K; D], right: &[K; D]) {
    // X^D = −1 in the ring of SPEC.md section 2.2.
    for (i, &left) in left.iter().enumerate() {
        if left != K::ZERO {
            for (j, &right) in right.iter().enumerate() {
                if i + j < D {
                    output[i + j] += left * right;
                } else {
                    output[i + j - D] -= left * right;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/pad_opening.rs"]
mod tests;
