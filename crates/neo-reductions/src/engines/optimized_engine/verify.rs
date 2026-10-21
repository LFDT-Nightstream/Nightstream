//! Optimized verifier entrypoints for the selected one-joint PiCCS protocol.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::engines::pi_ccs_joint::ProtocolTrace;
use crate::error::PiCcsError;

use super::PiCcsProof;

/// Verify PiCCS and return every verifier-computed conformance value.
///
/// The [caller contract](crate::engines::PiCcsEngine::verify) applies.
pub fn optimized_verify_with_trace(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<(bool, ProtocolTrace), PiCcsError> {
    crate::engines::pi_ccs_joint_protocol::verify_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
    )
}

/// Replay the public protocol against the caller-selected relation header.
/// Matrix evaluator caches belong to preprocessing and proving.
///
/// The [caller contract](crate::engines::PiCcsEngine::verify) applies.
pub fn optimized_verify(
    transcript: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
) -> Result<bool, PiCcsError> {
    Ok(optimized_verify_with_trace(
        transcript,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
    )?
    .0)
}
