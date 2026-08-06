//! Optimized verifier entrypoints for the selected one-joint PiCCS protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::engines::pi_ccs_joint_protocol::TranscriptBinding;
use crate::error::PiCcsError;

use super::{OptimizedStructureCache, PiCcsProof, PiCcsVerifyPerf};

pub fn optimized_verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    let cache = OptimizedStructureCache::build(structure)?;
    optimized_verify_with_cache(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        &cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_verify_with_cache(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
) -> Result<bool, PiCcsError> {
    Ok(optimized_verify_with_cache_and_perf(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        cache,
    )?
    .0)
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_verify_with_cache_and_perf(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    cache.validate_structure(structure)?;
    verify_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        cache.matrix_digest(),
        TranscriptBinding::claims(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_verify_with_cache_and_instance_digest_and_perf(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: [F; 4],
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    cache.validate_structure(structure)?;
    verify_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        cache.matrix_digest(),
        TranscriptBinding::digest(public_instance_digest),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_verify_with_cache_and_instance_digest_and_me_input_handle_and_perf(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: [F; 4],
    running_accumulator_handle: [F; 4],
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    cache.validate_structure(structure)?;
    verify_with_binding(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        cache.matrix_digest(),
        TranscriptBinding::digest_and_handle(public_instance_digest, running_accumulator_handle),
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_with_binding(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    expected_matrix_digest: &[F; 4],
    binding: TranscriptBinding,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    let started = std::time::Instant::now();
    let valid = crate::engines::pi_ccs_joint_protocol::verify_with_binding_and_matrix_digest(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        binding,
        expected_matrix_digest,
    )?;
    Ok((
        valid,
        PiCcsVerifyPerf {
            total_ms: started.elapsed().as_secs_f64() * 1_000.0,
            ..PiCcsVerifyPerf::default()
        },
    ))
}
