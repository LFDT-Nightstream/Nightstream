//! Production-only entry points for the delayed 19-block/6-lane NC variant.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;

use crate::error::PiCcsError;

use super::backend::BackendTranscriptMode;
use super::oracle::BlockLaneNcPending;
use super::proof_assembly::proof_from_terminal_state;
use super::prove::{owned_rounds, run_optimized_replay_with_cache_and_perf, ReplayTraceMode};
use super::replay_binding::ReplayBinding;
use super::verify::optimized_verify_with_cache_and_public_instance_digest_impl;
use super::{OptimizedStructureCache, PiCcsProof, PiCcsProvePerf, PiCcsVerifyPerf, PiDecProverPrecompute};

#[allow(clippy::too_many_arguments)]
pub fn optimized_prove_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf<
    L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>,
>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    fresh_witnesses: &[CcsWitness<F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    running_witnesses: &[Mat<F>],
    public_instance_digest: [F; 4],
    accumulator_handle: [F; 4],
    pending: Option<BlockLaneNcPending>,
    log: &L,
    cache: &OptimizedStructureCache,
) -> Result<
    (
        Vec<CeClaim<Cmt, F, K>>,
        PiCcsProof,
        PiCcsProvePerf,
        PiDecProverPrecompute,
    ),
    PiCcsError,
> {
    if params.b != 2 {
        return Err(PiCcsError::InvalidInput(
            "block-lane delayed Π_CCS requires the strict base-two norm relation".into(),
        ));
    }
    let (terminal, rounds) = run_optimized_replay_with_cache_and_perf(
        tr,
        params,
        structure,
        fresh_claims,
        fresh_witnesses,
        running_claims,
        running_witnesses,
        log,
        cache,
        ReplayBinding::block_lane_handle(public_instance_digest, accumulator_handle, pending),
        ReplayTraceMode::Prove,
        true,
        None,
        None,
        None,
        BackendTranscriptMode::Replay,
    )?;
    let rounds = owned_rounds(rounds.expect("block-lane prove must capture proof rounds"))?;
    let proof = proof_from_terminal_state(&terminal, rounds);
    let pi_dec_precompute = terminal
        .pi_dec_precompute
        .clone()
        .ok_or_else(|| PiCcsError::InvalidInput("block-lane prove did not produce Pi_DEC precomputation".into()))?;
    Ok((terminal.me_outputs, proof, terminal.perf, pi_dec_precompute))
}

#[allow(clippy::too_many_arguments)]
pub fn optimized_verify_block_lane_delayed_with_cache_and_instance_digest_and_me_input_handle_and_perf(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_claims: &[CcsClaim<Cmt, F>],
    running_claims: &[CeClaim<Cmt, F, K>],
    outputs: &[CeClaim<Cmt, F, K>],
    proof: &PiCcsProof,
    cache: &OptimizedStructureCache,
    public_instance_digest: [F; 4],
    accumulator_handle: [F; 4],
    pending: Option<BlockLaneNcPending>,
) -> Result<(bool, PiCcsVerifyPerf), PiCcsError> {
    optimized_verify_with_cache_and_public_instance_digest_impl(
        tr,
        params,
        structure,
        fresh_claims,
        running_claims,
        outputs,
        proof,
        cache,
        ReplayBinding::block_lane_handle(public_instance_digest, accumulator_handle, pending),
    )
}
