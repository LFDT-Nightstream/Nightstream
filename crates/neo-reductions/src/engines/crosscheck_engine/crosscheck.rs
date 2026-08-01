//! Exact cross-check wrapper for canonical rectangular `Pi_CCS`.
//!
//! On native targets, the wrapper starts the optimized engine and PaperExact
//! concurrently from the same transcript state. It compares both transcript
//! checkpoints, each proof surface, all outputs, and the canonical proof
//! bytes before it returns the optimized result. Single-threaded Wasm uses a
//! sequential fallback.

#![allow(non_snake_case)]

use crate::engines::optimized_engine::{PiCcsProof, PiCcsProofVariant};
use crate::engines::PiCcsEngine;
use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};

#[cfg(not(target_arch = "wasm32"))]
fn run_engine_pair<Optimized, Reference, RunOptimized, RunReference>(
    run_optimized: RunOptimized,
    run_reference: RunReference,
) -> Result<(Optimized, Reference), PiCcsError>
where
    Reference: Send,
    RunOptimized: FnOnce() -> Optimized,
    RunReference: FnOnce() -> Reference + Send,
{
    std::thread::scope(|scope| {
        let reference = scope.spawn(run_reference);
        let optimized = run_optimized();
        let reference = reference
            .join()
            .map_err(|_| PiCcsError::ProtocolError("PaperExact cross-check worker panicked".into()))?;
        Ok((optimized, reference))
    })
}

#[cfg(target_arch = "wasm32")]
fn run_engine_pair<Optimized, Reference, RunOptimized, RunReference>(
    run_optimized: RunOptimized,
    run_reference: RunReference,
) -> Result<(Optimized, Reference), PiCcsError>
where
    RunOptimized: FnOnce() -> Optimized,
    RunReference: FnOnce() -> Reference,
{
    Ok((run_optimized(), run_reference()))
}

fn transcript_checkpoint(transcript: &Poseidon2Transcript) -> [u8; 32] {
    let mut checkpoint = transcript.clone();
    checkpoint.digest32()
}

/// Surfaces checked by `OptimizedWithCrosscheck`.
#[derive(Clone, Debug)]
pub struct CrosscheckCfg {
    pub fail_fast: bool,
    pub initial_sum: bool,
    pub per_round: bool,
    pub terminal: bool,
    pub outputs: bool,
    pub byte_exact: bool,
}

impl Default for CrosscheckCfg {
    fn default() -> Self {
        Self {
            fail_fast: true,
            initial_sum: true,
            per_round: true,
            terminal: true,
            outputs: true,
            byte_exact: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CrossCheckEngine<I, R> {
    pub inner: I,
    pub ref_oracle: R,
    pub cfg: CrosscheckCfg,
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_prove<I, R, L>(
    inner: &I,
    reference: &R,
    cfg: &CrosscheckCfg,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    I: PiCcsEngine,
    R: PiCcsEngine + Sync,
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt> + Sync,
{
    let mut reference_transcript = transcript.clone();
    let (optimized, reference) = run_engine_pair(
        || {
            inner.prove(
                transcript,
                params,
                structure,
                fresh_claims,
                fresh_witnesses,
                running_claims,
                running_witnesses,
                commitment,
            )
        },
        || {
            reference.prove(
                &mut reference_transcript,
                params,
                structure,
                fresh_claims,
                fresh_witnesses,
                running_claims,
                running_witnesses,
                commitment,
            )
        },
    )?;
    let ((optimized_outputs, optimized_proof), (reference_outputs, reference_proof)) = match (optimized, reference) {
        (Ok(optimized), Ok(reference)) => (optimized, reference),
        (Err(optimized), Ok(_)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "optimized prover failed while PaperExact succeeded: {optimized}"
            )))
        }
        (Ok(_), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "PaperExact prover failed while optimized succeeded: {reference}"
            )))
        }
        (Err(optimized), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "both cross-check provers failed (optimized: {optimized}; PaperExact: {reference})"
            )))
        }
    };
    if transcript_checkpoint(transcript) != transcript_checkpoint(&reference_transcript) {
        return Err(PiCcsError::ProtocolError(
            "cross-check prover transcript states differ".into(),
        ));
    }

    if optimized_proof.variant != PiCcsProofVariant::PaperRectangularV1
        || reference_proof.variant != PiCcsProofVariant::PaperRectangularV1
    {
        return Err(PiCcsError::ProtocolError(
            "cross-check requires PaperRectangularV1 from both engines".into(),
        ));
    }
    if cfg.initial_sum
        && (optimized_proof.sc_initial_sum != reference_proof.sc_initial_sum
            || optimized_proof.sc_initial_sum_nc != reference_proof.sc_initial_sum_nc)
    {
        return Err(PiCcsError::ProtocolError("cross-check initial claims differ".into()));
    }
    if cfg.per_round
        && (optimized_proof.sumcheck_rounds != reference_proof.sumcheck_rounds
            || optimized_proof.sumcheck_rounds_nc != reference_proof.sumcheck_rounds_nc
            || optimized_proof.sumcheck_challenges != reference_proof.sumcheck_challenges
            || optimized_proof.sumcheck_challenges_nc != reference_proof.sumcheck_challenges_nc)
    {
        return Err(PiCcsError::ProtocolError(
            "cross-check SumCheck rounds or folds differ".into(),
        ));
    }
    if cfg.terminal
        && (optimized_proof.sumcheck_final != reference_proof.sumcheck_final
            || optimized_proof.sumcheck_final_nc != reference_proof.sumcheck_final_nc)
    {
        return Err(PiCcsError::ProtocolError("cross-check terminal claims differ".into()));
    }
    if cfg.outputs && optimized_outputs != reference_outputs {
        return Err(PiCcsError::ProtocolError("cross-check output claims differ".into()));
    }
    if cfg.byte_exact
        && optimized_proof
            .canonical_bytes()
            .map_err(|error| PiCcsError::ProtocolError(format!("cannot serialize optimized proof: {error}")))?
            != reference_proof
                .canonical_bytes()
                .map_err(|error| PiCcsError::ProtocolError(format!("cannot serialize reference proof: {error}")))?
    {
        return Err(PiCcsError::ProtocolError(
            "cross-check canonical proof bytes differ".into(),
        ));
    }

    let _ = cfg.fail_fast;
    Ok((optimized_outputs, optimized_proof))
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_verify<I, R>(
    inner: &I,
    reference: &R,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError>
where
    I: PiCcsEngine,
    R: PiCcsEngine + Sync,
{
    let mut reference_transcript = transcript.clone();
    let (optimized, checked) = run_engine_pair(
        || {
            inner.verify(
                transcript,
                params,
                structure,
                fresh_claims,
                running_claims,
                outputs,
                proof,
            )
        },
        || {
            reference.verify(
                &mut reference_transcript,
                params,
                structure,
                fresh_claims,
                running_claims,
                outputs,
                proof,
            )
        },
    )?;
    let (optimized, checked) = match (optimized, checked) {
        (Ok(optimized), Ok(checked)) => (optimized, checked),
        (Err(optimized), Ok(_)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "optimized verifier failed while PaperExact succeeded: {optimized}"
            )))
        }
        (Ok(_), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "PaperExact verifier failed while optimized succeeded: {reference}"
            )))
        }
        (Err(optimized), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "both cross-check verifiers failed (optimized: {optimized}; PaperExact: {reference})"
            )))
        }
    };
    if transcript_checkpoint(transcript) != transcript_checkpoint(&reference_transcript) {
        return Err(PiCcsError::ProtocolError(
            "cross-check verifier transcript states differ".into(),
        ));
    }
    if optimized != checked {
        return Err(PiCcsError::ProtocolError("cross-check verifier results differ".into()));
    }
    Ok(optimized)
}
