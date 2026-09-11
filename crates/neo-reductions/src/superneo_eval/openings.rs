//! Complete real-witness openings shared by the normal PiCCS and PiDEC paths.

use neo_ccs::V1_1Evaluations;
use neo_math::{superneo_bar_block, KExtensions, Rq, D, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{EqualityWeights, RealBlockStorage, SuperneoEvalCache, SuperneoZBlocks};
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
        let (rows, width, _) = self
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
        let weights = EqualityWeights::new(point);
        let row_weights = (0..rows).map(|row| weights.at(row)).collect::<Vec<_>>();
        let matrices = self.eval_ring_linear_forms_for_real_z_blocks(&row_weights, rows, witnesses);
        Ok(witnesses
            .iter()
            .zip(matrices)
            .map(|(witness, eval_a)| V1_1Evaluations {
                eval_k: pad_opening(witness, &weights).to_vec(),
                eval_a: eval_a
                    .into_iter()
                    .map(|coefficients| coefficients.to_vec())
                    .collect(),
            })
            .collect())
    }
}

fn pad_opening(witness: &SuperneoZBlocks, weights: &EqualityWeights) -> [K; D] {
    if matches!(&witness.re, RealBlockStorage::Zero { .. }) {
        return [K::ZERO; D];
    }
    let block = |block: usize| {
        if !witness.real_nonzero(block) {
            return [K::ZERO; D];
        }
        let value = Rq(std::array::from_fn(|lane| witness.real_coefficient(block, lane)));
        // Pad covers every lane of the block, including scalar zero-tail
        // lanes: its higher ring coefficients need those equality weights.
        let weights: [K; D] = std::array::from_fn(|lane| weights.at(block * D + lane));
        let real = Rq(superneo_bar_block(weights.map(|value| value.as_coeffs()[0]))).mul(&value);
        let imaginary = Rq(superneo_bar_block(weights.map(|value| value.as_coeffs()[1]))).mul(&value);
        std::array::from_fn(|lane| K::from_coeffs([real.0[lane], imaginary.0[lane]]))
    };
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        (0..witness.block_len()).into_par_iter().map(block).reduce(
            || [K::ZERO; D],
            |left, right| std::array::from_fn(|lane| left[lane] + right[lane]),
        )
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    {
        (0..witness.block_len())
            .map(block)
            .fold([K::ZERO; D], |left, right| {
                std::array::from_fn(|lane| left[lane] + right[lane])
            })
    }
}
