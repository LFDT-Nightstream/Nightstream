//! Owns the chunk NIFS.V body bridge inside the recursive-step circuit.
//!
//! This module reuses the staged inner verifier body, then absorbs the
//! authoritative chunk-relation digest with a synthetic public input so the
//! carried transcript matches native state_out before `chunk_done`.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use p3_goldilocks::Goldilocks;

use super::super::recursive_cover::{
    alloc_recursive_carried_projection_claims, alloc_recursive_carried_x_r_only_claims,
    carried_projection_claims_have_zero_public_tail,
    recursive_accumulator_instance_digest_circuit_from_projection_digests, Rv64imRecursiveCoverStateVar,
};
use super::super::{synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io, Rv64imClaimBundle};
use crate::rv64im::final_relation::RV64IM_CHUNK_DONE_RAW_TAG;
use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::kernel::rv64im_cached_root_main_lane_optimized_cache;
use crate::rv64im::main_recursion::Rv64imMainRecursionFPrimeAdvice;
use crate::rv64im::main_relation_circuit::claim::enforce_claim_projection_eq_native;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::Rv64imMainRecursionFPrimePayload;

pub(super) struct Rv64imMainRecursionStepChunkReplayOutput {
    pub(super) live_folded_accumulator_out_digest: [AllocatedNum<SpartanF>; 4],
}

pub(super) fn synthesize_rv64im_main_recursion_step_chunk_replay<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &Rv64imMainRecursionFPrimeAdvice,
    payload: &Rv64imMainRecursionFPrimePayload,
    state_in_var: &Rv64imRecursiveCoverStateVar,
    state_out_var: &Rv64imRecursiveCoverStateVar,
    trace_prefix: Option<&str>,
) -> Result<Rv64imMainRecursionStepChunkReplayOutput, SynthesisError> {
    let (params, _, structure) = crate::rv64im::kernel::rv64im_root_main_lane_context_for_claim_count(
        witness.running_state().carry.main.claims.len(),
    )
    .map_err(|_| SynthesisError::Unsatisfiable)?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache().map_err(|_| SynthesisError::Unsatisfiable)?;
    let dims = build_dims_and_policy(&params, structure).map_err(|_| SynthesisError::Unsatisfiable)?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|_| SynthesisError::Unsatisfiable)?;
    let mut replayed_transcript = super::super::import_chunk_fold_transcript_in(
        &mut cs.namespace(|| "transcript_in_import"),
        state_in_var,
        &witness.running_state().transcript,
        payload.initial_transcript_in,
        "transcript_in_import",
    )?;
    let live_state_in_claims =
        if payload.initial_transcript_in && carried_projection_claims_have_zero_public_tail(&payload.state_in_claims) {
            alloc_recursive_carried_x_r_only_claims(
                &mut cs.namespace(|| "state_in_live_claims"),
                &payload.state_in_claims,
                "state_in_live_claims",
            )?
        } else {
            alloc_recursive_carried_projection_claims(
                &mut cs.namespace(|| "state_in_live_claims"),
                &payload.state_in_claims,
                "state_in_live_claims",
            )?
        };
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let replayed_next_claims = synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io(
        &params,
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
        // HyperNova §6.3 requires a single compiled F' circuit reused across
        // values. The live recursive-step path must therefore bind ME inputs
        // from the authenticated carried digest slice, not from a native
        // logical-claim slice that would re-specialize transcript constants.
        None,
        Some(&witness.running_state().carry.main_projection_digests),
        payload.boundary_plan,
        payload.rlc_zero_commit_suffix_len,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
        trace_prefix,
    )?;
    if replayed_next_claims.effective_count() != witness.fresh_state_out().carry.main.claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (claim_index, (actual, expected)) in replayed_next_claims
        .effective_claims()
        .iter()
        .zip(witness.fresh_state_out().carry.main.claims.iter())
        .enumerate()
    {
        enforce_claim_projection_eq_native(
            &mut cs.namespace(|| format!("state_out_claim_{claim_index}")),
            actual,
            expected,
            &format!("state_out_claim_{claim_index}"),
        )?;
    }
    let expected_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_projection_digests(
        &mut cs.namespace(|| "expected_folded_accumulator_out_digest"),
        &witness.fresh_state_out().carry.main_projection_digests,
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
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
    let replayed_absorbed_var =
        AllocatedNum::alloc(cs.namespace(|| "payload_transcript_absorbed_out_expected"), || {
            Ok(replayed_absorbed)
        })?;
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + replayed_absorbed_var.get_variable(),
    );

    Ok(Rv64imMainRecursionStepChunkReplayOutput {
        live_folded_accumulator_out_digest: expected_folded_accumulator_out_digest,
    })
}
