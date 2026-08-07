//! Arithmetic acceleration contract for final CE witness openings.

use neo_math::{D, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::lifecycle::Error;
use crate::paper::relations::{CeClaim, WitnessMat};

/// Arithmetic backend for the final witness-opening check.
///
/// This backend is part of the verifier's trusted computing base. It returns
/// identity-first opening values, and the host compares those values with the
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
    ) -> Result<Option<Vec<Vec<[K; D]>>>, String>;
}

pub(super) fn validate_opening_shape(
    openings: &[Vec<[K; D]>],
    witness_count: usize,
    matrix_count: usize,
) -> Result<(), Error> {
    if openings.len() == witness_count && openings.iter().all(|rows| rows.len() == matrix_count) {
        return Ok(());
    }
    Err(Error::FinalAccumulatorOpeningBackend {
        reason: format!("expected {witness_count} witnesses with {matrix_count} identity-first rows"),
    })
}

pub(super) fn check_claim_openings(
    index: usize,
    claim: &CeClaim,
    ell_d: usize,
    openings: &[[K; D]],
) -> Result<(), Error> {
    if claim.y_ring.len() != openings.len() {
        return Err(Error::FinalAccumulatorCeRelationViolation {
            index,
            matrix_index: openings.len().min(claim.y_ring.len()),
        });
    }
    let d_pad = 1usize << ell_d;
    for (matrix_index, (recorded, expected)) in claim.y_ring.iter().zip(openings).enumerate() {
        let matches = recorded.len() == d_pad
            && recorded.iter().take(D).zip(expected).all(|(a, b)| a == b)
            && recorded.iter().skip(D).all(|&value| value == K::ZERO);
        if !matches {
            return Err(Error::FinalAccumulatorCeRelationViolation { index, matrix_index });
        }
    }
    if claim.ct.len() != openings.len() {
        return Err(Error::FinalAccumulatorCtMismatch {
            index,
            matrix_index: openings.len().min(claim.ct.len()),
        });
    }
    for (matrix_index, (recorded, expected)) in claim.ct.iter().zip(openings).enumerate() {
        if recorded != &expected[0] {
            return Err(Error::FinalAccumulatorCtMismatch { index, matrix_index });
        }
    }
    Ok(())
}
