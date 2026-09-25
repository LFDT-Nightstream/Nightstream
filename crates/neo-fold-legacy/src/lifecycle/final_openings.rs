//! Arithmetic acceleration contract for final CE witness openings.

use neo_math::{D, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::lifecycle::Error;
use crate::paper::relations::{CeClaim, WitnessMat};

/// One final witness opening in exact SuperNeo v1_1 form.
#[allow(non_camel_case_types)]
pub struct V1_1WitnessOpenings {
    pub eval_k: [K; D],
    pub eval_a: Vec<[K; D]>,
}

/// Arithmetic backend for the final witness-opening check.
///
/// This backend is part of the verifier's trusted computing base. It returns
/// separate Pad and matrix opening values, and the host compares them with the
/// proof claims without recomputing them. A faulty or malicious backend can
/// therefore invalidate the verification result. Use the default CPU path
/// when the accelerator is not trusted.
pub trait FinalWitnessOpeningBackend {
    fn final_witness_openings(
        &mut self,
        cache: &OptimizedStructureCache,
        witnesses: &[WitnessMat],
        point: &[K],
        assignment_width: usize,
    ) -> Result<Option<Vec<V1_1WitnessOpenings>>, String>;
}

pub(super) fn validate_opening_shape(
    openings: &[V1_1WitnessOpenings],
    witness_count: usize,
    matrix_count: usize,
) -> Result<(), Error> {
    if openings.len() == witness_count
        && openings
            .iter()
            .all(|value| value.eval_a.len() == matrix_count)
    {
        return Ok(());
    }
    Err(Error::FinalAccumulatorOpeningBackend {
        reason: format!("expected {witness_count} witnesses with one Eval_K and {matrix_count} Eval_A rows"),
    })
}

pub(super) fn check_claim_openings(
    index: usize,
    claim: &CeClaim,
    ell_d: usize,
    openings: &V1_1WitnessOpenings,
) -> Result<(), Error> {
    if claim.eval_a.len() != openings.eval_a.len() {
        return Err(Error::FinalAccumulatorCeRelationViolation {
            index,
            matrix_index: openings.eval_a.len().min(claim.eval_a.len()),
        });
    }
    let d_pad = 1usize << ell_d;
    let pad_matches = claim.eval_k.len() == d_pad
        && claim
            .eval_k
            .iter()
            .take(D)
            .zip(&openings.eval_k)
            .all(|(a, b)| a == b)
        && claim.eval_k.iter().skip(D).all(|&value| value == K::ZERO);
    if !pad_matches {
        return Err(Error::FinalAccumulatorCeRelationViolation { index, matrix_index: 0 });
    }
    for (matrix, (recorded, expected)) in claim.eval_a.iter().zip(&openings.eval_a).enumerate() {
        let matches = recorded.len() == d_pad
            && recorded.iter().take(D).zip(expected).all(|(a, b)| a == b)
            && recorded.iter().skip(D).all(|&value| value == K::ZERO);
        if !matches {
            return Err(Error::FinalAccumulatorCeRelationViolation {
                index,
                matrix_index: matrix + 1,
            });
        }
    }
    Ok(())
}
