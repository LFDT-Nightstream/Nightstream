//! Owns the chunk NIFS.V body bridge inside the recursive-step circuit.
//!
//! This module reuses the staged inner verifier body directly and does not call
//! the outer chunk-theorem public-IO wrapper.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use spartan2::provider::goldi::F as SpartanF;

use super::super::recursive_cover::{
    alloc_recursive_cover_claims, recursive_accumulator_instance_digest_circuit_from_claims,
    Rv64imRecursiveCoverStateVar,
};
use super::super::{synthesize_rv64im_chunk_nifs_verifier_body, Rv64imClaimBundle};
use crate::rv64im::final_relation::RV64IM_CHUNK_DONE_RAW_TAG;
use crate::rv64im::kernel::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
use crate::rv64im::main_recursion::Rv64imMainRecursionFPrimeAdvice;
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imMainRecursionFPrimePayload;

pub(super) struct Rv64imMainRecursionStepChunkReplayOutput {
    pub(super) live_folded_accumulator_out_digest: [AllocatedNum<SpartanF>; 4],
}

fn mark_unsatisfied<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, label: &str) -> Result<(), SynthesisError> {
    cs.enforce(|| label, |lc| lc + CS::one(), |lc| lc + CS::one(), |lc| lc);
    Ok(())
}

pub(super) fn synthesize_rv64im_main_recursion_step_chunk_replay<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv64imMainRecursionFPrimeAdvice,
    payload: &Rv64imMainRecursionFPrimePayload,
    state_in_var: &Rv64imRecursiveCoverStateVar,
    state_out_var: &Rv64imRecursiveCoverStateVar,
) -> Result<Rv64imMainRecursionStepChunkReplayOutput, SynthesisError> {
    let (params, _, structure) = rv64im_cached_root_main_lane_context().map_err(|_| SynthesisError::Unsatisfiable)?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache().map_err(|_| SynthesisError::Unsatisfiable)?;
    let dims = build_dims_and_policy(params, structure).map_err(|_| SynthesisError::Unsatisfiable)?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let replay_chunk = payload
        .effective_chunk_replay_surface(
            &witness.running_state().transcript,
            &witness.running_state().carry.main.claims,
        )
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )?;
    let live_state_in_claims = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        &witness.running_state().carry.main.claims,
        "state_in_live_claims",
    )?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let replayed_next_claims = synthesize_rv64im_chunk_nifs_verifier_body(
        params,
        structure,
        dims,
        &mat_digest,
        &witness.fresh_state_out().carry.main.claims,
        &mut cs.namespace(|| "payload_chunk_step"),
        witness.chunk_index() as usize,
        &payload.chunk_cover,
        &replay_chunk,
        &mut replayed_transcript,
        carried_claims,
        payload.boundary_plan,
    )?;
    if replayed_next_claims.effective_count() != witness.fresh_state_out().carry.main.claims.len() {
        mark_unsatisfied(
            &mut cs.namespace(|| "payload_replayed_effective_claim_count_mismatch"),
            "payload_replayed_effective_claim_count_mismatch",
        )?;
    }
    let live_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "live_folded_accumulator_out_digest"),
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "live_folded_accumulator_out_digest",
    )?;
    replayed_transcript.append_const_fields_raw(
        cs.namespace(|| "payload_chunk_done"),
        &[
            SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
            SpartanF::from_canonical_u64(1),
        ],
    )?;
    let replayed_transcript_out = replayed_transcript.state_fields(cs.namespace(|| "payload_transcript_out"))?;
    for (lane_index, (replayed_lane, state_out_lane)) in replayed_transcript_out
        .iter()
        .zip(state_out_var.transcript_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("payload_transcript_out_lane_{lane_index}"),
            |lc| lc + replayed_lane.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + state_out_lane.get_variable(),
        );
    }
    let replayed_absorbed = SpartanF::from_canonical_u64(replayed_transcript.absorbed() as u64);
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (replayed_absorbed, CS::one()),
    );

    Ok(Rv64imMainRecursionStepChunkReplayOutput {
        live_folded_accumulator_out_digest,
    })
}
