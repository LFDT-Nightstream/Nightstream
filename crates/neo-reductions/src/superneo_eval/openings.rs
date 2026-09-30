//! Complete real-witness openings shared by the normal PiCCS and PiDEC paths.

use neo_ccs::V1_1Evaluations;
use neo_math::{superneo_bar_block, KExtensions, Rq, D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::block_sums::{add_pair_sums, to_extension, zero_pair_sums, TaskSums, TASK_BLOCKS};
use super::{EqualityWeights, SuperneoEvalCache, SuperneoZBlocks};
use crate::PiCcsError;

impl SuperneoEvalCache {
    /// Evaluate all D Pad coefficients and all genuine matrix coefficients
    /// from concrete real witness blocks over the complete carrier.
    /// The cache and witnesses are prover data, not verifier authority.
    pub fn eval_real_v1_1_openings(
        &self,
        point: &[K],
        witnesses: &[SuperneoZBlocks],
    ) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
        self.eval_real_v1_1_openings_reusing(point, witnesses, Vec::new())
    }

    pub(crate) fn eval_real_v1_1_openings_reusing(
        &self,
        point: &[K],
        witnesses: &[SuperneoZBlocks],
        storage: Vec<K>,
    ) -> Result<Vec<V1_1Evaluations<K>>, PiCcsError> {
        #[cfg(feature = "perf-timers")]
        let started = std::time::Instant::now();
        let (rows, width, matrix_count) = self
            .relation_shape()
            .ok_or_else(|| PiCcsError::InvalidInput("opening cache shape is inconsistent".into()))?;
        let variables = (usize::BITS - rows.max(width).saturating_sub(1).leading_zeros()) as usize;
        if width % D != 0
            || point.len() != variables
            || witnesses
                .iter()
                .any(|witness| witness.block_len() != width / D || !witness.imag_all_zero)
        {
            return Err(PiCcsError::InvalidInput("real opening point or witness shape".into()));
        }
        if witnesses.iter().all(SuperneoZBlocks::real_is_zero) {
            return Ok((0..witnesses.len())
                .map(|_| V1_1Evaluations {
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
        let result = pad_openings(witnesses, &weights)
            .into_iter()
            .zip(matrices)
            .map(|(eval_k, eval_a)| V1_1Evaluations {
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

/// Pad openings of real witnesses with equal block counts. Each block's
/// weight forms are computed once and shared by every witness.
pub(super) fn pad_openings(witnesses: &[SuperneoZBlocks], weights: &EqualityWeights) -> Vec<[K; D]> {
    let active: Vec<_> = witnesses
        .iter()
        .filter(|witness| !witness.real_is_zero())
        .collect();
    let Some(blocks) = active.first().map(|witness| witness.block_len()) else {
        return vec![[K::ZERO; D]; witnesses.len()];
    };
    let forms = |block: usize| {
        // Pad covers every lane of the block, including scalar zero-tail
        // lanes: its higher ring coefficients need those equality weights.
        let weights: [K; D] = std::array::from_fn(|lane| weights.at(block * D + lane));
        (
            Rq(superneo_bar_block(weights.map(|value| value.as_coeffs()[0]))),
            Rq(superneo_bar_block(weights.map(|value| value.as_coeffs()[1]))),
        )
    };
    let task = |task: usize| {
        let mut sums = TaskSums::new(active.len());
        for block in task * TASK_BLOCKS..((task + 1) * TASK_BLOCKS).min(blocks) {
            sums.add_block(&active, block, || forms(block));
        }
        sums.finish()
    };
    let tasks = 0..blocks.div_ceil(TASK_BLOCKS);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let sums = tasks
        .into_par_iter()
        .map(task)
        .reduce(|| zero_pair_sums(active.len()), add_pair_sums);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let sums = tasks
        .map(task)
        .fold(zero_pair_sums(active.len()), add_pair_sums);
    let mut sums = to_extension(sums).into_iter();
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

#[cfg(test)]
#[path = "../../tests/unit/pad_opening.rs"]
mod tests;
