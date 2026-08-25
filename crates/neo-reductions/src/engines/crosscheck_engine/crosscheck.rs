//! Complete optimized-versus-PaperExact PiCCS comparison.
//!
//! Every check is mandatory. The wrapper compares the independent event
//! traces, challenges, rounds, terminal value, outputs, transcript state, and
//! versioned proof bytes before it returns the optimized result.

use crate::engines::optimized_engine::{OptimizedStructureCache, PiCcsProof};
use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::engines::paper_exact_engine::PaperTranscriptBinding;
use crate::engines::pi_ccs_joint_protocol::TranscriptBinding;

#[cfg(not(target_arch = "wasm32"))]
fn run_pair<Optimized, Reference, RunOptimized, RunReference>(
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
            .map_err(|_| PiCcsError::ProtocolError("PaperExact crosscheck worker panicked".into()))?;
        Ok((optimized, reference))
    })
}

#[cfg(target_arch = "wasm32")]
fn run_pair<Optimized, Reference, RunOptimized, RunReference>(
    run_optimized: RunOptimized,
    run_reference: RunReference,
) -> Result<(Optimized, Reference), PiCcsError>
where
    RunOptimized: FnOnce() -> Optimized,
    RunReference: FnOnce() -> Reference,
{
    Ok((run_optimized(), run_reference()))
}

fn checkpoint(transcript: &Poseidon2Transcript) -> [u8; 32] {
    transcript.clone().digest32()
}

#[derive(Clone, Debug)]
pub struct CrossCheckEngine<I, R> {
    pub inner: I,
    pub ref_oracle: R,
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_prove<I, R, L>(
    _inner: &I,
    _reference: &R,
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
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt> + Sync,
{
    crosscheck_prove_with_binding(
        _inner,
        _reference,
        transcript,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        commitment,
        TranscriptBinding::digest_only(),
    )
}

fn reference_binding(binding: TranscriptBinding) -> PaperTranscriptBinding {
    let _ = binding;
    PaperTranscriptBinding::digest_only()
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_prove_with_binding<I, R, L>(
    _inner: &I,
    _reference: &R,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    commitment: &L,
    binding: TranscriptBinding,
) -> Result<(Vec<CeClaim<Cmt, F, K>>, PiCcsProof), PiCcsError>
where
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt> + Sync,
{
    let cache = OptimizedStructureCache::build(structure)?;
    let mut reference_transcript = transcript.clone();
    let paper_binding = reference_binding(binding);
    let (optimized, reference) = run_pair(
        || {
            crate::engines::optimized_engine::paper_joint::prove_with_trace(
                transcript,
                params,
                structure,
                fresh_claims,
                fresh_witnesses,
                running_claims,
                running_witnesses,
                commitment,
                &cache,
                binding,
            )
        },
        || {
            crate::engines::paper_exact_engine::prove::paper_exact_prove_with_trace_and_binding(
                &mut reference_transcript,
                params,
                structure,
                fresh_claims,
                fresh_witnesses,
                running_claims,
                running_witnesses,
                commitment,
                paper_binding,
            )
        },
    )?;
    let (
        (optimized_outputs, optimized_proof, _, optimized_trace),
        (reference_outputs, reference_proof, reference_trace),
    ) = match (optimized, reference) {
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
                "both crosscheck provers failed (optimized: {optimized}; PaperExact: {reference})"
            )))
        }
    };

    if optimized_trace != reference_trace {
        return Err(PiCcsError::ProtocolError(
            "crosscheck protocol event traces differ".into(),
        ));
    }
    if checkpoint(transcript) != checkpoint(&reference_transcript) {
        return Err(PiCcsError::ProtocolError("crosscheck transcript states differ".into()));
    }
    if optimized_outputs != reference_outputs || optimized_proof != reference_proof {
        return Err(PiCcsError::ProtocolError(
            "crosscheck proof or output values differ".into(),
        ));
    }
    if optimized_proof.canonical_bytes()
        != crate::engines::paper_exact_engine::encode_proof(&reference_proof)
            .map_err(|error| PiCcsError::ProtocolError(format!("PaperExact codec failed: {error}")))?
    {
        return Err(PiCcsError::ProtocolError(
            "crosscheck canonical proof bytes differ".into(),
        ));
    }
    Ok((optimized_outputs, optimized_proof))
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_verify<I, R>(
    _inner: &I,
    _reference: &R,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    crosscheck_verify_with_binding(
        _inner,
        _reference,
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        TranscriptBinding::digest_only(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn crosscheck_verify_with_binding<I, R>(
    _inner: &I,
    _reference: &R,
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    binding: TranscriptBinding,
) -> Result<bool, PiCcsError> {
    let mut reference_transcript = transcript.clone();
    let paper_binding = reference_binding(binding);
    let (optimized, reference) = run_pair(
        || {
            crate::engines::pi_ccs_joint_protocol::verify_with_trace(
                transcript,
                params,
                structure,
                fresh_claims,
                running_claims,
                outputs,
                proof,
                binding,
                None,
            )
        },
        || {
            crate::engines::paper_exact_engine::verify::paper_exact_verify_with_trace_and_binding(
                &mut reference_transcript,
                params,
                structure,
                fresh_claims,
                running_claims,
                outputs,
                proof,
                paper_binding,
            )
        },
    )?;
    let ((optimized_valid, optimized_trace), (reference_valid, reference_trace)) = match (optimized, reference) {
        (Ok(optimized), Ok(reference)) => (optimized, reference),
        (Err(optimized), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "both crosscheck verifiers failed (optimized: {optimized}; PaperExact: {reference})"
            )))
        }
        (Err(optimized), Ok(_)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "optimized verifier failed while PaperExact completed: {optimized}"
            )))
        }
        (Ok(_), Err(reference)) => {
            return Err(PiCcsError::ProtocolError(format!(
                "PaperExact verifier failed while optimized completed: {reference}"
            )))
        }
    };
    if optimized_trace != reference_trace
        || checkpoint(transcript) != checkpoint(&reference_transcript)
        || optimized_valid != reference_valid
    {
        return Err(PiCcsError::ProtocolError(
            "crosscheck verifier traces or decisions differ".into(),
        ));
    }
    Ok(optimized_valid)
}
