use std::io::{self, Write};
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_math::{KExtensions, K};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use neo_reductions::engines::utils::{
    PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG, PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use super::*;
use crate::rv32im::f_prime::Rv32imMainRecursionFPrimeAdvice;
use crate::rv32im::final_relation::RV32IM_CHUNK_DONE_RAW_TAG;
use crate::rv32im::kernel::{rv32im_cached_root_main_lane_context, rv32im_cached_root_main_lane_optimized_cache};
use crate::rv32im::main_relation_spartan::fingerprint_cs::FingerprintCS;
use crate::rv32im::main_relation_spartan::recursive_cover::{
    alloc_recursive_carried_projection_claims, alloc_recursive_cover_claims, alloc_recursive_cover_state,
    debug_measure_recursive_accumulator_instance_digest_circuit_from_claims_aux,
    recursive_accumulator_instance_digest_circuit_from_claims, Rv32imRecursiveCoverClaimVar,
};
use crate::rv32im::main_relation_spartan::Rv32imMainRecursionFPrimePayload;
use crate::rv32im::main_relation_spartan::{rv32im_main_relation_delta, Rv32imClaimBundle};
use crate::spartan_backend::{Rv32imDeciderEngine, ShapeCS, SpartanCircuit, SpartanF, SpartanShape, SplitR1CSShape};
use crate::superneo_circuit::claim::{enforce_claim_projection_eq_native, me_digest_poseidon};
use crate::superneo_circuit::transcript::Poseidon2TranscriptCircuit;

fn stage_err(stage: &str, err: impl ToString) -> Rv32imMainRecursionStepSpartanError {
    Rv32imMainRecursionStepSpartanError::Prepare(format!("{stage}: {}", err.to_string()))
}

fn emit_trace(trace_prefix: &str, label: &str, elapsed_ms: f64) {
    eprintln!("{trace_prefix}.{label}={elapsed_ms:.2}ms");
    let _ = io::stderr().flush();
}

fn push_constraint_delta(
    stages: &mut Vec<Rv32imNamedConstraintDelta>,
    previous: &mut usize,
    current: usize,
    name: impl Into<String>,
) {
    let delta = current.saturating_sub(*previous);
    *previous = current;
    stages.push(Rv32imNamedConstraintDelta {
        name: name.into(),
        delta,
    });
}

fn ensure_stage_satisfied(
    cs: &TestConstraintSystem<SpartanF>,
    stage: &str,
) -> Result<(), Rv32imMainRecursionStepSpartanError> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(stage_err(
            stage,
            cs.which_is_unsatisfied().unwrap_or("unknown constraint"),
        ))
    }
}

fn alloc_live_state_in_projection_claims<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    _witness: &Rv32imMainRecursionFPrimeAdvice,
    payload: &Rv32imMainRecursionFPrimePayload,
    label: &str,
) -> Result<Vec<Rv32imRecursiveCoverClaimVar>, SynthesisError> {
    alloc_recursive_carried_projection_claims(cs, &payload.state_in_claims, label)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rv32imMainRecursionStepSpartanShapeSynthesisMetrics {
    pub shared_ms: f64,
    pub precommitted_ms: f64,
    pub synthesize_ms: f64,
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayFingerprint {
    pub after_state_cover: String,
    pub after_chunk_meta: String,
    pub after_pi_ccs: String,
    pub after_synthetic_relation_io: String,
    pub after_pi_rlc_parent_claim: String,
    pub after_pi_rlc_rhos: String,
    pub after_pi_rlc_rho_mats: String,
    pub after_pi_rlc_public: String,
    pub after_pi_rlc: String,
    pub after_chunk_body: String,
    pub after_chunk_replay: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepStageAuxCounts {
    pub after_private_witness_inputs: usize,
    pub after_alloc_cover_states: usize,
    pub after_bind_state_and_pc: usize,
    pub after_chunk_replay: usize,
    pub after_inactive_side_lane_and_x_out: usize,
    pub after_public_output_eq: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayAuxCounts {
    pub after_state_cover: usize,
    pub after_chunk_meta: usize,
    pub after_pi_ccs: usize,
    pub after_synthetic_relation_io: usize,
    pub after_pi_rlc_parent_claim: usize,
    pub after_pi_rlc_rhos: usize,
    pub after_pi_rlc_rho_mats: usize,
    pub after_pi_rlc_public: usize,
    pub after_pi_rlc: usize,
    pub after_chunk_body: usize,
    pub after_chunk_replay: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayTailAuxCounts {
    pub after_state_out_projection_eq: usize,
    pub after_expected_digest: usize,
    pub after_chunk_done: usize,
    pub after_transcript_state_eq: usize,
    pub after_transcript_absorbed_eq: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageAuxCounts {
    pub after_bind_header: usize,
    pub after_bind_me_inputs: usize,
    pub after_sample_challenges: usize,
    pub after_alloc_fresh_claims: usize,
    pub after_fe_sumcheck: usize,
    pub after_nc_sumcheck: usize,
    pub after_fold_digest: usize,
    pub after_alloc_outputs: usize,
    pub after_output_binding: usize,
    pub after_terminal_fe: usize,
    pub after_terminal_nc: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageConstraintCounts {
    pub after_bind_header: usize,
    pub after_bind_me_inputs: usize,
    pub after_sample_challenges: usize,
    pub after_alloc_fresh_claims: usize,
    pub after_fe_sumcheck: usize,
    pub after_nc_sumcheck: usize,
    pub after_fold_digest: usize,
    pub after_alloc_outputs: usize,
    pub after_output_binding: usize,
    pub after_terminal_fe: usize,
    pub after_terminal_nc: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsBindMeInputsAuxBreakdown {
    pub after_bind_header: usize,
    pub after_claim_digests: Vec<usize>,
    pub after_bind_digests: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsSumcheckConstraintBreakdown {
    pub fe_cover_round_lengths: Vec<u64>,
    pub fe_effective_round_lengths: Vec<usize>,
    pub fe_stages: Vec<Rv32imNamedConstraintDelta>,
    pub nc_cover_round_lengths: Vec<u64>,
    pub nc_effective_round_lengths: Vec<usize>,
    pub nc_stages: Vec<Rv32imNamedConstraintDelta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiCcsStageFingerprint {
    pub after_bind_header: String,
    pub after_bind_me_inputs: String,
    pub after_sample_challenges: String,
    pub after_alloc_fresh_claims: String,
    pub after_fe_sumcheck: String,
    pub after_nc_sumcheck: String,
    pub after_fold_digest: String,
    pub after_alloc_outputs: String,
    pub after_output_binding: String,
    pub after_terminal_fe: String,
    pub after_terminal_nc: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiRlcPublicConstraintBreakdown {
    pub shared_point_constraints: usize,
    pub x_constraints: usize,
    pub c_constraints: usize,
    pub y_ring_constraints: usize,
    pub y_zcol_constraints: usize,
    pub aux_constraints: usize,
    pub total_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imNamedConstraintDelta {
    pub name: String,
    pub delta: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imPiRlcPublicStageBreakdown {
    pub stages: Vec<Rv32imNamedConstraintDelta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainRecursionStepChunkReplayTailDigestAuxBreakdown {
    pub after_header: usize,
    pub claim_after_digests: Vec<usize>,
    pub after_outer_hash: usize,
}

pub fn debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepChunkReplayFingerprint, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("chunk_replay_context", err))?;
    let optimized_cache =
        rv32im_cached_root_main_lane_optimized_cache().map_err(|err| stage_err("chunk_replay_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("chunk_replay_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("chunk_replay_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("chunk_replay_state_in", err))?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("chunk_replay_state_out", err))?;
    let after_state_cover = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("chunk_replay_surface", err))?;
    let mut replayed_transcript = super::super::import_chunk_fold_transcript_in(
        &mut cs.namespace(|| "chunk_replay_transcript"),
        &state_in_var,
        &witness.running_state().transcript,
        payload.initial_transcript_in,
        "chunk_replay_transcript",
    )
    .map_err(|err| stage_err("chunk_replay_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("chunk_replay_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    )
    .map_err(|err| stage_err("chunk_replay_chunk_meta", err))?;
    let after_chunk_meta = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };
    let pi_ccs = super::super::synthesize_pi_ccs_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_ccs"),
        &mut replayed_transcript,
        &carried_claims,
        None,
    )
    .map_err(|err| stage_err("chunk_replay_pi_ccs", err))?;
    let after_pi_ccs = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let bridge_handoff_digest = super::super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        replay_chunk.handoff.bridge_handoff_digest,
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("chunk_replay_bridge_handoff_digest", err))?;
    super::super::enforce_synthetic_outer_chunk_relation_public_io(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_synthetic_relation_io"),
        &pi_ccs.fold_digest,
        &pi_ccs.public_chunk_digest,
        &bridge_handoff_digest,
        "payload_chunk_synthetic_relation_io",
    )
    .map_err(|err| stage_err("chunk_replay_synthetic_relation_io", err))?;
    let after_synthetic_relation_io = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims,
            super::super::Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = super::super::cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))
            .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::alloc_ce_claim(
            &mut cs.namespace(|| "payload_chunk_pi_rlc_parent_claim"),
            &claim,
            "payload_chunk_pi_rlc_parent_claim",
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_alloc", err))?
    } else {
        let claim = super::super::cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| "payload_chunk_pi_rlc_parent_claim"),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            "payload_chunk_pi_rlc_parent_claim",
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_alloc", err))?
    };
    let after_pi_rlc_parent_claim = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let child_claim_source = match ctx.boundary_plan.child_claim_source {
        super::super::Rv32imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let rho_vars = super::super::sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| "payload_chunk_pi_rlc_rhos"),
        &mut replayed_transcript,
        pi_ccs.padded_ccs_outputs.len(),
        "payload_chunk_pi_rlc_rhos",
    )
    .map_err(|err| stage_err("chunk_replay_pi_rlc_rhos", err))?;
    let after_pi_rlc_rhos = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let after_pi_rlc_rho_mats;
    match ctx.boundary_plan.rlc_mode {
        super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
            after_pi_rlc_rho_mats = after_pi_rlc_rhos.clone();
            super::super::enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                child_claim_source,
                &rho_vars,
                ctx.params.b,
                "payload_chunk_pi_rlc_public",
            )
            .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
        }
        super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => {
            if constant_child_prefix == pi_ccs.padded_ccs_outputs.len() {
                after_pi_rlc_rho_mats = after_pi_rlc_rhos.clone();
                crate::superneo_circuit::pi_rlc::enforce_rlc_public_with_rho_coeffs_for_constant_children(
                    &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_vars,
                    "payload_chunk_pi_rlc_public",
                )
                .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
            } else {
                let active_dense_children_len = pi_ccs
                    .padded_ccs_outputs
                    .len()
                    .saturating_sub(payload.rlc_zero_commit_suffix_len);
                let rho_mats = if constant_child_prefix > 0 && constant_child_prefix < active_dense_children_len {
                    super::super::materialize_goldilocks_rot_matrices(
                        &mut cs.namespace(|| "payload_chunk_pi_rlc_rho_mats"),
                        &rho_vars[..active_dense_children_len],
                        "payload_chunk_pi_rlc_rho_mats",
                    )
                    .map_err(|err| stage_err("chunk_replay_pi_rlc_rho_mats", err))?
                } else {
                    Vec::new()
                };
                after_pi_rlc_rho_mats = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));
                crate::superneo_circuit::pi_rlc::enforce_rlc_public_with_split_rho_views_constant_prefix_zero_commit_suffix(
                    &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_vars,
                    &rho_mats,
                    constant_child_prefix,
                    payload.rlc_zero_commit_suffix_len,
                    "payload_chunk_pi_rlc_public",
                )
                .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
            }
        }
    }
    let after_pi_rlc_public = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));
    let pi_rlc = super::super::Rv32imPiRlcStageOutput { parent_claim };
    let after_pi_rlc = after_pi_rlc_public.clone();

    let replayed_next_claims = super::super::synthesize_pi_dec_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_dec"),
        carried_claims,
        &pi_ccs,
        pi_rlc,
    )
    .map_err(|err| stage_err("chunk_replay_body", err))?;
    let after_chunk_body = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    if replayed_next_claims.effective_count() != witness.fresh_state_out().carry.main.claims.len() {
        return Err(stage_err(
            "chunk_replay_state_out_claim_count",
            "state_out claim count mismatch",
        ));
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
        )
        .map_err(|err| stage_err("chunk_replay_state_out_claim_eq", err))?;
    }
    let _expected_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "expected_folded_accumulator_out_digest"),
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_replay_expected_digest", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| "payload_chunk_done"),
            &[
                SpartanF::from_canonical_u64(RV32IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )
        .map_err(|err| stage_err("chunk_replay_chunk_done", err))?;
    let replayed_transcript_out = replayed_transcript
        .state_fields(cs.namespace(|| "payload_transcript_out"))
        .map_err(|err| stage_err("chunk_replay_transcript_out", err))?;
    let one = <FingerprintCS as ConstraintSystem<SpartanF>>::one();
    for (lane_index, (replayed_lane, state_out_lane)) in replayed_transcript_out
        .iter()
        .zip(state_out_var.transcript_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("payload_transcript_out_lane_{lane_index}"),
            |lc| lc + replayed_lane.get_variable(),
            |lc| lc + one,
            |lc| lc + state_out_lane.get_variable(),
        );
    }
    let replayed_absorbed = SpartanF::from_canonical_u64(replayed_transcript.absorbed() as u64);
    let replayed_absorbed_var =
        AllocatedNum::alloc(cs.namespace(|| "payload_transcript_absorbed_out_expected"), || {
            Ok(replayed_absorbed)
        })
        .map_err(|err| stage_err("chunk_replay_transcript_absorbed_expected", err))?;
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + one,
        |lc| lc + replayed_absorbed_var.get_variable(),
    );

    Ok(Rv32imMainRecursionStepChunkReplayFingerprint {
        after_state_cover,
        after_chunk_meta,
        after_pi_ccs,
        after_synthetic_relation_io,
        after_pi_rlc_parent_claim,
        after_pi_rlc_rhos,
        after_pi_rlc_rho_mats,
        after_pi_rlc_public,
        after_pi_rlc,
        after_chunk_body,
        after_chunk_replay: super::format_spartan_digest_hex(cs.finish_digest32(0)),
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiCcsStageAuxCounts, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_ccs_context", err))?;
    let optimized_cache =
        rv32im_cached_root_main_lane_optimized_cache().map_err(|err| stage_err("pi_ccs_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_ccs_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_ccs_mat_digest", "invalid matrix digest width"))?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_ccs_surface", err))?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("pi_ccs_state_in", err))?;
    let mut replayed_transcript = super::super::import_chunk_fold_transcript_in(
        &mut cs.namespace(|| "pi_ccs_transcript"),
        &state_in_var,
        &witness.running_state().transcript,
        payload.initial_transcript_in,
        "pi_ccs_transcript",
    )
    .map_err(|err| stage_err("pi_ccs_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_ccs_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };

    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_chunk_meta", err))?;

    super::super::bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.params,
        ctx.structure.n,
        ctx.structure.m,
        ctx.structure.t(),
        &ctx.structure.f,
        ctx.dims,
        ctx.mat_digest,
        &ctx.chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| stage_err("pi_ccs_bind_header", err))?;
    let after_bind_header = cs.num_aux();

    let (accumulator_handle, accumulator_handle_values) = ctx
        .me_input_accumulator_handle
        .ok_or_else(|| stage_err("pi_ccs_bind_me_inputs", "missing accumulator handle"))?;
    crate::superneo_circuit::pi_ccs::bind_me_inputs_accumulator_handle(
        &mut cs.namespace(|| format!("chunk_{}_bind_me_input_accumulator", ctx.chunk_index)),
        &mut replayed_transcript,
        witness.running_state().carry.main.claims.len(),
        accumulator_handle,
        &accumulator_handle_values,
    )
    .map_err(|err| stage_err("pi_ccs_bind_me_inputs", err))?;
    let after_bind_me_inputs = cs.num_aux();

    let public_challenges = crate::superneo_circuit::pi_ccs::sample_challenges_with_native(
        &mut cs.namespace(|| format!("chunk_{}_sample_challenges", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.dims,
        &ctx.chunk.pi_ccs.public_challenges,
        &format!("chunk_{}_sample_challenges", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sample_challenges", err))?;
    let after_sample_challenges = cs.num_aux();

    let cover_fresh_claim_count = ctx.cover_chunk.fresh_claim_count as usize;
    let covered_fresh_claims = ctx
        .cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| super::super::cover_ccs_claim(shape, ctx.chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_cover_fresh_claims", err))?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{}_fresh_claim_{fresh_index}", ctx.chunk_index)),
                fresh,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_alloc_fresh_claims", err))?;
    let after_alloc_fresh_claims = cs.num_aux();

    let (initial_sum_fe, initial_sum_fe_value) =
        crate::superneo_circuit::initial_sum::claimed_initial_sum_from_me_inputs(
            &mut cs.namespace(|| format!("chunk_{}_initial_sum_fe", ctx.chunk_index)),
            ctx.structure,
            &public_challenges.alpha,
            &ctx.chunk.pi_ccs.public_challenges.alpha,
            &public_challenges.gamma,
            ctx.chunk.pi_ccs.public_challenges.gamma,
            cover_fresh_claim_count,
            carried_claims.effective_claims(),
            ctx.rlc_zero_commit_suffix_len,
            rv32im_main_relation_delta(),
            &format!("chunk_{}_initial_sum_fe", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_initial_sum_fe", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_fe_sumcheck_domain", err))?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_tag", ctx.chunk_index)),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| stage_err("pi_ccs_fe_sumcheck_initial_tag", err))?;
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_append", ctx.chunk_index)),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| stage_err("pi_ccs_fe_sumcheck_initial_append", err))?;
    } else {
        super::super::append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            &mut replayed_transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_fe_sumcheck_initial", err))?;
    }
    let padded_fe_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_alloc_fe_rounds", err))?;
    let fe_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fe_sumcheck", err))?;
    let (r_prime_vars, alpha_prime_vars) =
        super::super::split_vec(&fe_challenges, ctx.dims.ell_n).map_err(|err| stage_err("pi_ccs_fe_split", err))?;
    let after_fe_sumcheck = cs.num_aux();

    let zero_nc = crate::superneo_circuit::k_field::alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index)),
        crate::superneo_circuit::k_field::KNum::from_neo_k(K::ZERO),
        &format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_initial_sum_nc_zero", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_nc_sumcheck_domain", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_tag", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_nc_sumcheck_initial_tag", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_append", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| stage_err("pi_ccs_nc_sumcheck_initial_append", err))?;
    let padded_nc_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_alloc_nc_rounds", err))?;
    let nc_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_nc_sumcheck", err))?;
    let (s_col_prime_vars, alpha_prime_nc_vars) =
        super::super::split_vec(&nc_challenges, ctx.dims.ell_m).map_err(|err| stage_err("pi_ccs_nc_split", err))?;
    let after_nc_sumcheck = cs.num_aux();

    let _fold_digest = replayed_transcript
        .digest32(cs.namespace(|| format!("chunk_{}_fold_digest", ctx.chunk_index)))
        .map_err(|err| stage_err("pi_ccs_fold_digest", err))?;
    let after_fold_digest = cs.num_aux();

    let effective_output_count = ctx.chunk.pi_ccs.ccs_outputs.len();
    let constant_child_prefix = match ctx.boundary_plan.rlc_mode {
        super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => constant_child_prefix,
        super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => 0,
    };
    let zero_output_suffix_start = effective_output_count.saturating_sub(ctx.rlc_zero_commit_suffix_len);
    let mut padded_ccs_outputs = Vec::with_capacity(ctx.cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in ctx.cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = ctx.chunk.pi_ccs.ccs_outputs.get(output_index);
        let claim = if output_index < effective_output_count {
            super::super::cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &ctx.chunk.pi_ccs.row_chals,
                &ctx.chunk.pi_ccs.s_col,
            )
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = ctx.chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = ctx.chunk.pi_ccs.s_col.clone();
            Ok(padded_claim)
        }
        .map_err(|err| stage_err("pi_ccs_cover_output", err))?;
        let output = if output_index < constant_child_prefix {
            if output_index < cover_fresh_claim_count {
                let fresh = covered_fresh_claims
                    .get(output_index)
                    .ok_or_else(|| stage_err("pi_ccs_alloc_output", "fresh output missing"))?;
                let fresh_x_values = crate::superneo_circuit::output_binding::embedded_fresh_x_values(fresh);
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &fresh.c.data,
                    &fresh_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
            } else {
                let me_input_index = output_index
                    .checked_sub(cover_fresh_claim_count)
                    .ok_or_else(|| stage_err("pi_ccs_alloc_output", "me-input underflow"))?;
                let me_input = carried_claims
                    .effective_claims()
                    .get(me_input_index)
                    .ok_or_else(|| stage_err("pi_ccs_alloc_output", "me-input missing"))?;
                let me_input_x_values = crate::superneo_circuit::output_binding::embedded_me_input_x_values(me_input)
                    .map_err(|err| stage_err("pi_ccs_alloc_output", err))?;
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &me_input_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
            }
        } else if output_index < cover_fresh_claim_count {
            let fresh = covered_fresh_claim_vars
                .get(output_index)
                .ok_or_else(|| stage_err("pi_ccs_alloc_output", "fresh output vars missing"))?;
            crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point_compact_x(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &fresh.c_data,
                &fresh.c_data_values,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
        } else if output_index < effective_output_count {
            let me_input_index = output_index
                .checked_sub(cover_fresh_claim_count)
                .ok_or_else(|| stage_err("pi_ccs_alloc_output", "me-input underflow"))?;
            let me_input = carried_claims
                .effective_claims()
                .get(me_input_index)
                .ok_or_else(|| stage_err("pi_ccs_alloc_output", "me-input missing"))?;
            if output_index >= zero_output_suffix_start {
                crate::superneo_circuit::claim::alloc_ce_claim_x_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
            } else {
                crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
            }
        } else {
            super::super::alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_alloc_output", err))?
        };
        padded_ccs_outputs.push(output);
    }
    let after_alloc_outputs = cs.num_aux();

    crate::superneo_circuit::output_binding::enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| format!("chunk_{}_output_binding", ctx.chunk_index)),
        ctx.structure,
        ctx.params,
        &covered_fresh_claim_vars,
        carried_claims.effective_claims(),
        &padded_ccs_outputs,
        ctx.rlc_zero_commit_suffix_len,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &format!("chunk_{}_output_binding", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_output_binding", err))?;
    let after_output_binding = cs.num_aux();

    let me_inputs_r_vars = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r.as_slice());
    let me_inputs_r_values = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r_values.as_slice());
    let effective_me_output_count = effective_output_count
        .checked_sub(cover_fresh_claim_count)
        .ok_or_else(|| stage_err("pi_ccs_effective_me_output_count", "underflow"))?;
    let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
        .iter()
        .cloned()
        .chain(
            padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                .iter()
                .cloned(),
        )
        .collect::<Vec<_>>();
    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_fe(
        &mut cs.namespace(|| format!("chunk_{}_terminal_fe", ctx.chunk_index)),
        &sumcheck_final_fe,
        ctx.structure,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.alpha,
        &public_challenges.beta_a,
        &public_challenges.beta_r,
        &public_challenges.gamma,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &alpha_prime_vars,
        &ctx.chunk.pi_ccs.alpha_prime,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        me_inputs_r_vars,
        me_inputs_r_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_fe", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_terminal_fe", err))?;
    let after_terminal_fe = cs.num_aux();

    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_nc(
        &mut cs.namespace(|| format!("chunk_{}_terminal_nc", ctx.chunk_index)),
        &sumcheck_final_nc,
        ctx.params,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.beta_a,
        &public_challenges.beta_m,
        &public_challenges.gamma,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &alpha_prime_nc_vars,
        &ctx.chunk.pi_ccs.alpha_prime_nc,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_nc", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_terminal_nc", err))?;
    let after_terminal_nc = cs.num_aux();

    Ok(Rv32imPiCcsStageAuxCounts {
        after_bind_header,
        after_bind_me_inputs,
        after_sample_challenges,
        after_alloc_fresh_claims,
        after_fe_sumcheck,
        after_nc_sumcheck,
        after_fold_digest,
        after_alloc_outputs,
        after_output_binding,
        after_terminal_fe,
        after_terminal_nc,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiCcsStageConstraintCounts, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_ccs_constraints_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_ccs_constraints_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_ccs_constraints_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_ccs_constraints_mat_digest", "invalid matrix digest width"))?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_ccs_constraints_surface", err))?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("pi_ccs_constraints_state_in", err))?;
    let mut replayed_transcript = super::super::import_chunk_fold_transcript_in(
        &mut cs.namespace(|| "pi_ccs_constraints_transcript"),
        &state_in_var,
        &witness.running_state().transcript,
        payload.initial_transcript_in,
        "pi_ccs_constraints_transcript",
    )
    .map_err(|err| stage_err("pi_ccs_constraints_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_ccs_constraints_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };

    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_constraints_chunk_meta", err))?;

    super::super::bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.params,
        ctx.structure.n,
        ctx.structure.m,
        ctx.structure.t(),
        &ctx.structure.f,
        ctx.dims,
        ctx.mat_digest,
        &ctx.chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_bind_header", err))?;
    let after_bind_header = cs.num_constraints();

    let (accumulator_handle, accumulator_handle_values) = ctx
        .me_input_accumulator_handle
        .ok_or_else(|| stage_err("pi_ccs_constraints_bind_me_inputs", "missing accumulator handle"))?;
    crate::superneo_circuit::pi_ccs::bind_me_inputs_accumulator_handle(
        &mut cs.namespace(|| format!("chunk_{}_bind_me_input_accumulator", ctx.chunk_index)),
        &mut replayed_transcript,
        witness.running_state().carry.main.claims.len(),
        accumulator_handle,
        &accumulator_handle_values,
    )
    .map_err(|err| stage_err("pi_ccs_constraints_bind_me_inputs", err))?;
    let after_bind_me_inputs = cs.num_constraints();

    let public_challenges = crate::superneo_circuit::pi_ccs::sample_challenges_with_native(
        &mut cs.namespace(|| format!("chunk_{}_sample_challenges", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.dims,
        &ctx.chunk.pi_ccs.public_challenges,
        &format!("chunk_{}_sample_challenges", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_sample_challenges", err))?;
    let after_sample_challenges = cs.num_constraints();

    let cover_fresh_claim_count = ctx.cover_chunk.fresh_claim_count as usize;
    let covered_fresh_claims = ctx
        .cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| super::super::cover_ccs_claim(shape, ctx.chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_constraints_cover_fresh_claims", err))?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{}_fresh_claim_{fresh_index}", ctx.chunk_index)),
                fresh,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_constraints_alloc_fresh_claims", err))?;
    let after_alloc_fresh_claims = cs.num_constraints();

    let (initial_sum_fe, initial_sum_fe_value) =
        crate::superneo_circuit::initial_sum::claimed_initial_sum_from_me_inputs(
            &mut cs.namespace(|| format!("chunk_{}_initial_sum_fe", ctx.chunk_index)),
            ctx.structure,
            &public_challenges.alpha,
            &ctx.chunk.pi_ccs.public_challenges.alpha,
            &public_challenges.gamma,
            ctx.chunk.pi_ccs.public_challenges.gamma,
            cover_fresh_claim_count,
            carried_claims.effective_claims(),
            ctx.rlc_zero_commit_suffix_len,
            rv32im_main_relation_delta(),
            &format!("chunk_{}_initial_sum_fe", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_constraints_initial_sum_fe", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_constraints_fe_sumcheck_domain", err))?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_tag", ctx.chunk_index)),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| stage_err("pi_ccs_constraints_fe_sumcheck_initial_tag", err))?;
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_append", ctx.chunk_index)),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| stage_err("pi_ccs_constraints_fe_sumcheck_initial_append", err))?;
    } else {
        super::super::append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            &mut replayed_transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_constraints_fe_sumcheck_initial", err))?;
    }
    let padded_fe_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_alloc_fe_rounds", err))?;
    let fe_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_constraints_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_fe_sumcheck", err))?;
    let (r_prime_vars, alpha_prime_vars) = super::super::split_vec(&fe_challenges, ctx.dims.ell_n)
        .map_err(|err| stage_err("pi_ccs_constraints_fe_split", err))?;
    let after_fe_sumcheck = cs.num_constraints();

    let zero_nc = crate::superneo_circuit::k_field::alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index)),
        crate::superneo_circuit::k_field::KNum::from_neo_k(K::ZERO),
        &format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_initial_sum_nc_zero", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_constraints_nc_sumcheck_domain", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_tag", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_constraints_nc_sumcheck_initial_tag", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_append", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| stage_err("pi_ccs_constraints_nc_sumcheck_initial_append", err))?;
    let padded_nc_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_alloc_nc_rounds", err))?;
    let nc_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_constraints_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_nc_sumcheck", err))?;
    let (s_col_prime_vars, alpha_prime_nc_vars) = super::super::split_vec(&nc_challenges, ctx.dims.ell_m)
        .map_err(|err| stage_err("pi_ccs_constraints_nc_split", err))?;
    let after_nc_sumcheck = cs.num_constraints();

    let _fold_digest = replayed_transcript
        .digest32(cs.namespace(|| format!("chunk_{}_fold_digest", ctx.chunk_index)))
        .map_err(|err| stage_err("pi_ccs_constraints_fold_digest", err))?;
    let after_fold_digest = cs.num_constraints();

    let effective_output_count = ctx.chunk.pi_ccs.ccs_outputs.len();
    let constant_child_prefix = match ctx.boundary_plan.rlc_mode {
        super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => constant_child_prefix,
        super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => 0,
    };
    let zero_output_suffix_start = effective_output_count.saturating_sub(ctx.rlc_zero_commit_suffix_len);
    let mut padded_ccs_outputs = Vec::with_capacity(ctx.cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in ctx.cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = ctx.chunk.pi_ccs.ccs_outputs.get(output_index);
        let claim = if output_index < effective_output_count {
            super::super::cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &ctx.chunk.pi_ccs.row_chals,
                &ctx.chunk.pi_ccs.s_col,
            )
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = ctx.chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = ctx.chunk.pi_ccs.s_col.clone();
            Ok(padded_claim)
        }
        .map_err(|err| stage_err("pi_ccs_constraints_cover_output", err))?;
        let output = if output_index < constant_child_prefix {
            if output_index < cover_fresh_claim_count {
                let fresh = covered_fresh_claims
                    .get(output_index)
                    .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "fresh output missing"))?;
                let fresh_x_values = crate::superneo_circuit::output_binding::embedded_fresh_x_values(fresh);
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &fresh.c.data,
                    &fresh_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
            } else {
                let me_input_index = output_index
                    .checked_sub(cover_fresh_claim_count)
                    .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "me-input underflow"))?;
                let me_input = carried_claims
                    .effective_claims()
                    .get(me_input_index)
                    .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "me-input missing"))?;
                let me_input_x_values = crate::superneo_circuit::output_binding::embedded_me_input_x_values(me_input)
                    .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?;
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &me_input_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
            }
        } else if output_index < cover_fresh_claim_count {
            let fresh = covered_fresh_claim_vars
                .get(output_index)
                .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "fresh output vars missing"))?;
            crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point_compact_x(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &fresh.c_data,
                &fresh.c_data_values,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
        } else if output_index < effective_output_count {
            let me_input_index = output_index
                .checked_sub(cover_fresh_claim_count)
                .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "me-input underflow"))?;
            let me_input = carried_claims
                .effective_claims()
                .get(me_input_index)
                .ok_or_else(|| stage_err("pi_ccs_constraints_alloc_output", "me-input missing"))?;
            if output_index >= zero_output_suffix_start {
                crate::superneo_circuit::claim::alloc_ce_claim_x_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
            } else {
                crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
            }
        } else {
            super::super::alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_constraints_alloc_output", err))?
        };
        padded_ccs_outputs.push(output);
    }
    let after_alloc_outputs = cs.num_constraints();

    crate::superneo_circuit::output_binding::enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| format!("chunk_{}_output_binding", ctx.chunk_index)),
        ctx.structure,
        ctx.params,
        &covered_fresh_claim_vars,
        carried_claims.effective_claims(),
        &padded_ccs_outputs,
        ctx.rlc_zero_commit_suffix_len,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &format!("chunk_{}_output_binding", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_output_binding", err))?;
    let after_output_binding = cs.num_constraints();

    let me_inputs_r_vars = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r.as_slice());
    let me_inputs_r_values = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r_values.as_slice());
    let effective_me_output_count = effective_output_count
        .checked_sub(cover_fresh_claim_count)
        .ok_or_else(|| stage_err("pi_ccs_constraints_effective_me_output_count", "underflow"))?;
    let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
        .iter()
        .cloned()
        .chain(
            padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                .iter()
                .cloned(),
        )
        .collect::<Vec<_>>();
    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_fe(
        &mut cs.namespace(|| format!("chunk_{}_terminal_fe", ctx.chunk_index)),
        &sumcheck_final_fe,
        ctx.structure,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.alpha,
        &public_challenges.beta_a,
        &public_challenges.beta_r,
        &public_challenges.gamma,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &alpha_prime_vars,
        &ctx.chunk.pi_ccs.alpha_prime,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        me_inputs_r_vars,
        me_inputs_r_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_fe", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_terminal_fe", err))?;
    let after_terminal_fe = cs.num_constraints();

    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_nc(
        &mut cs.namespace(|| format!("chunk_{}_terminal_nc", ctx.chunk_index)),
        &sumcheck_final_nc,
        ctx.params,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.beta_a,
        &public_challenges.beta_m,
        &public_challenges.gamma,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &alpha_prime_nc_vars,
        &ctx.chunk.pi_ccs.alpha_prime_nc,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_nc", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_constraints_terminal_nc", err))?;
    let after_terminal_nc = cs.num_constraints();

    Ok(Rv32imPiCcsStageConstraintCounts {
        after_bind_header,
        after_bind_me_inputs,
        after_sample_challenges,
        after_alloc_fresh_claims,
        after_fe_sumcheck,
        after_nc_sumcheck,
        after_fold_digest,
        after_alloc_outputs,
        after_output_binding,
        after_terminal_fe,
        after_terminal_nc,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiCcsBindMeInputsAuxBreakdown, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_ccs_bind_breakdown_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_ccs_bind_breakdown_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_ccs_bind_breakdown_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_ccs_bind_breakdown_mat_digest", "invalid matrix digest width"))?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_ccs_bind_breakdown_surface", err))?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_state_in", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state,
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_live_state_in_claims", err))?;
    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };

    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_chunk_meta", err))?;
    super::super::bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.params,
        ctx.structure.n,
        ctx.structure.m,
        ctx.structure.t(),
        &ctx.structure.f,
        ctx.dims,
        ctx.mat_digest,
        &ctx.chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_bind_header", err))?;
    let after_bind_header = cs.num_aux();

    if live_state_in_claims.len() != witness.running_state().carry.main.claims.len() {
        return Err(stage_err(
            "pi_ccs_bind_breakdown_claim_arity",
            "live state-in claims and native logical claims disagree",
        ));
    }
    crate::superneo_circuit::pi_ccs::bind_me_inputs_accumulator_handle(
        &mut cs.namespace(|| "me_input_accumulator_handle"),
        &mut replayed_transcript,
        witness.running_state().carry.main.claims.len(),
        &state_in_var.folded_accumulator_digest,
        &digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_bind_accumulator", err))?;
    let after_bind_digests = cs.num_aux();

    Ok(Rv32imPiCcsBindMeInputsAuxBreakdown {
        after_bind_header,
        after_claim_digests: Vec::new(),
        after_bind_digests,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiCcsSumcheckConstraintBreakdown, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_ccs_sumcheck_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_ccs_sumcheck_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_ccs_sumcheck_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_ccs_sumcheck_mat_digest", "invalid matrix digest width"))?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_ccs_sumcheck_surface", err))?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_state_in", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state,
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };

    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_chunk_meta", err))?;
    super::super::bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.params,
        ctx.structure.n,
        ctx.structure.m,
        ctx.structure.t(),
        &ctx.structure.f,
        ctx.dims,
        ctx.mat_digest,
        &ctx.chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_bind_header", err))?;
    let (accumulator_handle, accumulator_handle_values) = ctx
        .me_input_accumulator_handle
        .ok_or_else(|| stage_err("pi_ccs_sumcheck_bind_me_inputs", "missing accumulator handle"))?;
    crate::superneo_circuit::pi_ccs::bind_me_inputs_accumulator_handle(
        &mut cs.namespace(|| format!("chunk_{}_bind_me_input_accumulator", ctx.chunk_index)),
        &mut replayed_transcript,
        witness.running_state().carry.main.claims.len(),
        accumulator_handle,
        &accumulator_handle_values,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_bind_me_inputs", err))?;
    let public_challenges = crate::superneo_circuit::pi_ccs::sample_challenges_with_native(
        &mut cs.namespace(|| format!("chunk_{}_sample_challenges", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.dims,
        &ctx.chunk.pi_ccs.public_challenges,
        &format!("chunk_{}_sample_challenges", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_sample_challenges", err))?;

    let cover_fresh_claim_count = ctx.cover_chunk.fresh_claim_count as usize;

    let mut fe_stages = Vec::new();
    let mut fe_previous = cs.num_constraints();
    let fe_cover_round_lengths = ctx.cover_chunk.fe_round_lengths.clone();
    let fe_effective_round_lengths = ctx
        .chunk
        .pi_ccs
        .replay_proof
        .sumcheck_rounds
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let (initial_sum_fe, initial_sum_fe_value) =
        crate::superneo_circuit::initial_sum::claimed_initial_sum_from_me_inputs(
            &mut cs.namespace(|| format!("chunk_{}_initial_sum_fe", ctx.chunk_index)),
            ctx.structure,
            &public_challenges.alpha,
            &ctx.chunk.pi_ccs.public_challenges.alpha,
            &public_challenges.gamma,
            ctx.chunk.pi_ccs.public_challenges.gamma,
            cover_fresh_claim_count,
            carried_claims.effective_claims(),
            ctx.rlc_zero_commit_suffix_len,
            rv32im_main_relation_delta(),
            &format!("chunk_{}_initial_sum_fe", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_initial_sum_fe", err))?;
    push_constraint_delta(
        &mut fe_stages,
        &mut fe_previous,
        cs.num_constraints(),
        "initial_sum_claim",
    );
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_fe_domain", err))?;
    push_constraint_delta(&mut fe_stages, &mut fe_previous, cs.num_constraints(), "domain");
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_tag", ctx.chunk_index)),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| stage_err("pi_ccs_sumcheck_fe_initial_tag", err))?;
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_append", ctx.chunk_index)),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| stage_err("pi_ccs_sumcheck_fe_initial_append", err))?;
    } else {
        super::super::append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            &mut replayed_transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_fe_initial", err))?;
    }
    push_constraint_delta(
        &mut fe_stages,
        &mut fe_previous,
        cs.num_constraints(),
        "initial_transcript",
    );
    let padded_fe_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_alloc_fe_rounds", err))?;
    push_constraint_delta(&mut fe_stages, &mut fe_previous, cs.num_constraints(), "alloc_rounds");
    let fe_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, _) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds_with_trace(
        &mut cs,
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
        |cs, stage| push_constraint_delta(&mut fe_stages, &mut fe_previous, cs.num_constraints(), stage),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_fe", err))?;
    let _ = super::super::split_vec(&fe_challenges, ctx.dims.ell_n)
        .map_err(|err| stage_err("pi_ccs_sumcheck_fe_split", err))?;
    push_constraint_delta(
        &mut fe_stages,
        &mut fe_previous,
        cs.num_constraints(),
        "split_challenges",
    );

    let mut nc_stages = Vec::new();
    let mut nc_previous = cs.num_constraints();
    let nc_cover_round_lengths = ctx.cover_chunk.nc_round_lengths.clone();
    let nc_effective_round_lengths = ctx
        .chunk
        .pi_ccs
        .replay_proof
        .sumcheck_rounds_nc
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let zero_nc = crate::superneo_circuit::k_field::alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index)),
        crate::superneo_circuit::k_field::KNum::from_neo_k(K::ZERO),
        &format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_initial_sum_nc_zero", err))?;
    push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), "initial_zero");
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_nc_domain", err))?;
    push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), "domain");
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_tag", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_nc_initial_tag", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_append", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| stage_err("pi_ccs_sumcheck_nc_initial_append", err))?;
    push_constraint_delta(
        &mut nc_stages,
        &mut nc_previous,
        cs.num_constraints(),
        "initial_transcript",
    );
    let padded_nc_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_alloc_nc_rounds", err))?;
    push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), "alloc_rounds");
    let nc_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, _) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds_with_trace(
        &mut cs,
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
        |cs, stage| push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), stage),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_nc", err))?;
    let _ = super::super::split_vec(&nc_challenges, ctx.dims.ell_m)
        .map_err(|err| stage_err("pi_ccs_sumcheck_nc_split", err))?;
    push_constraint_delta(
        &mut nc_stages,
        &mut nc_previous,
        cs.num_constraints(),
        "split_challenges",
    );

    Ok(Rv32imPiCcsSumcheckConstraintBreakdown {
        fe_cover_round_lengths,
        fe_effective_round_lengths,
        fe_stages,
        nc_cover_round_lengths,
        nc_effective_round_lengths,
        nc_stages,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepChunkReplayTailAuxCounts, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("chunk_tail_context", err))?;
    let optimized_cache =
        rv32im_cached_root_main_lane_optimized_cache().map_err(|err| stage_err("chunk_tail_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("chunk_tail_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("chunk_tail_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("chunk_tail_state_in", err))?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("chunk_tail_state_out", err))?;
    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("chunk_tail_surface", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("chunk_tail_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("chunk_tail_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let bridge_handoff_digest = super::super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        replay_chunk.handoff.bridge_handoff_digest,
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("chunk_tail_bridge_handoff_digest", err))?;
    let replayed_next_claims =
        super::super::synthesize_rv32im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io(
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
            None,
            payload.boundary_plan,
            payload.rlc_zero_commit_suffix_len,
            payload
                .initial_transcript_in
                .then_some(payload.chunk_cover.fresh_claim_count as usize),
            Some(&bridge_handoff_digest),
            None,
        )
        .map_err(|err| stage_err("chunk_tail_body", err))?;

    let after_state_out_projection_eq = cs.num_aux();

    let _expected_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "expected_folded_accumulator_out_digest"),
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_tail_expected_digest", err))?;
    let after_expected_digest = cs.num_aux();

    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| "payload_chunk_done"),
            &[
                SpartanF::from_canonical_u64(RV32IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )
        .map_err(|err| stage_err("chunk_tail_chunk_done", err))?;
    let after_chunk_done = cs.num_aux();

    let replayed_transcript_out = replayed_transcript
        .state_fields(cs.namespace(|| "payload_transcript_out"))
        .map_err(|err| stage_err("chunk_tail_transcript_out", err))?;
    let one = <FingerprintCS as ConstraintSystem<SpartanF>>::one();
    for (lane_index, (replayed_lane, state_out_lane)) in replayed_transcript_out
        .iter()
        .zip(state_out_var.transcript_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("payload_transcript_out_lane_{lane_index}"),
            |lc| lc + replayed_lane.get_variable(),
            |lc| lc + one,
            |lc| lc + state_out_lane.get_variable(),
        );
    }
    let after_transcript_state_eq = cs.num_aux();

    let replayed_absorbed = SpartanF::from_canonical_u64(replayed_transcript.absorbed() as u64);
    let replayed_absorbed_var =
        AllocatedNum::alloc(cs.namespace(|| "payload_transcript_absorbed_out_expected"), || {
            Ok(replayed_absorbed)
        })
        .map_err(|err| stage_err("chunk_tail_transcript_absorbed_expected", err))?;
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + one,
        |lc| lc + replayed_absorbed_var.get_variable(),
    );
    let after_transcript_absorbed_eq = cs.num_aux();

    Ok(Rv32imMainRecursionStepChunkReplayTailAuxCounts {
        after_state_out_projection_eq,
        after_expected_digest,
        after_chunk_done,
        after_transcript_state_eq,
        after_transcript_absorbed_eq,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepChunkReplayTailDigestAuxBreakdown, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("chunk_tail_digest_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("chunk_tail_digest_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("chunk_tail_digest_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("chunk_tail_digest_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("chunk_tail_digest_state_in", err))?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("chunk_tail_digest_state_out", err))?;
    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("chunk_tail_digest_surface", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("chunk_tail_digest_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("chunk_tail_digest_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let bridge_handoff_digest = super::super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        replay_chunk.handoff.bridge_handoff_digest,
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("chunk_tail_digest_bridge_handoff_digest", err))?;
    let replayed_next_claims =
        super::super::synthesize_rv32im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io(
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
            None,
            payload.boundary_plan,
            payload.rlc_zero_commit_suffix_len,
            payload
                .initial_transcript_in
                .then_some(payload.chunk_cover.fresh_claim_count as usize),
            Some(&bridge_handoff_digest),
            None,
        )
        .map_err(|err| stage_err("chunk_tail_digest_body", err))?;

    let breakdown = debug_measure_recursive_accumulator_instance_digest_circuit_from_claims_aux(
        &mut cs,
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_tail_digest_breakdown", err))?;
    Ok(Rv32imMainRecursionStepChunkReplayTailDigestAuxBreakdown {
        after_header: breakdown.after_header,
        claim_after_digests: breakdown.after_claim_digests,
        after_outer_hash: breakdown.after_outer_hash,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiCcsStageFingerprint, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_ccs_fingerprint_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_ccs_fingerprint_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_ccs_fingerprint_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_ccs_fingerprint_mat_digest", "invalid matrix digest width"))?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_ccs_fingerprint_surface", err))?;
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_state_in", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state,
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };

    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_chunk_meta", err))?;

    super::super::bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.params,
        ctx.structure.n,
        ctx.structure.m,
        ctx.structure.t(),
        &ctx.structure.f,
        ctx.dims,
        ctx.mat_digest,
        &ctx.chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_bind_header", err))?;
    let after_bind_header = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let (accumulator_handle, accumulator_handle_values) = ctx
        .me_input_accumulator_handle
        .ok_or_else(|| stage_err("pi_ccs_fingerprint_bind_me_inputs", "missing accumulator handle"))?;
    crate::superneo_circuit::pi_ccs::bind_me_inputs_accumulator_handle(
        &mut cs.namespace(|| format!("chunk_{}_bind_me_input_accumulator", ctx.chunk_index)),
        &mut replayed_transcript,
        witness.running_state().carry.main.claims.len(),
        accumulator_handle,
        &accumulator_handle_values,
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_bind_me_inputs", err))?;
    let after_bind_me_inputs = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let public_challenges = crate::superneo_circuit::pi_ccs::sample_challenges_with_native(
        &mut cs.namespace(|| format!("chunk_{}_sample_challenges", ctx.chunk_index)),
        &mut replayed_transcript,
        ctx.dims,
        &ctx.chunk.pi_ccs.public_challenges,
        &format!("chunk_{}_sample_challenges", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_sample_challenges", err))?;
    let after_sample_challenges = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let cover_fresh_claim_count = ctx.cover_chunk.fresh_claim_count as usize;
    let covered_fresh_claims = ctx
        .cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| super::super::cover_ccs_claim(shape, ctx.chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_fingerprint_cover_fresh_claims", err))?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{}_fresh_claim_{fresh_index}", ctx.chunk_index)),
                fresh,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_fresh_claims", err))?;
    let after_alloc_fresh_claims = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let (initial_sum_fe, initial_sum_fe_value) =
        crate::superneo_circuit::initial_sum::claimed_initial_sum_from_me_inputs(
            &mut cs.namespace(|| format!("chunk_{}_initial_sum_fe", ctx.chunk_index)),
            ctx.structure,
            &public_challenges.alpha,
            &ctx.chunk.pi_ccs.public_challenges.alpha,
            &public_challenges.gamma,
            ctx.chunk.pi_ccs.public_challenges.gamma,
            cover_fresh_claim_count,
            carried_claims.effective_claims(),
            ctx.rlc_zero_commit_suffix_len,
            rv32im_main_relation_delta(),
            &format!("chunk_{}_initial_sum_fe", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_initial_sum_fe", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_fe_sumcheck_domain", err))?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_tag", ctx.chunk_index)),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| stage_err("pi_ccs_fingerprint_fe_sumcheck_initial_tag", err))?;
        replayed_transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_append", ctx.chunk_index)),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| stage_err("pi_ccs_fingerprint_fe_sumcheck_initial_append", err))?;
    } else {
        super::super::append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            &mut replayed_transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_fe_sumcheck_initial", err))?;
    }
    let padded_fe_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_fe_rounds", err))?;
    let fe_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_fe_sumcheck", err))?;
    let (r_prime_vars, alpha_prime_vars) = super::super::split_vec(&fe_challenges, ctx.dims.ell_n)
        .map_err(|err| stage_err("pi_ccs_fingerprint_fe_split", err))?;
    let after_fe_sumcheck = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let zero_nc = crate::superneo_circuit::k_field::alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index)),
        crate::superneo_circuit::k_field::KNum::from_neo_k(K::ZERO),
        &format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_initial_sum_nc_zero", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_domain", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_nc_sumcheck_domain", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_tag", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_nc_sumcheck_initial_tag", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_append", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| stage_err("pi_ccs_fingerprint_nc_sumcheck_initial_append", err))?;
    let padded_nc_rounds = super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_nc_rounds", err))?;
    let nc_round_values = super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_nc_sumcheck", err))?;
    let (s_col_prime_vars, alpha_prime_nc_vars) = super::super::split_vec(&nc_challenges, ctx.dims.ell_m)
        .map_err(|err| stage_err("pi_ccs_fingerprint_nc_split", err))?;
    let after_nc_sumcheck = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let _fold_digest = replayed_transcript
        .digest32(cs.namespace(|| format!("chunk_{}_fold_digest", ctx.chunk_index)))
        .map_err(|err| stage_err("pi_ccs_fingerprint_fold_digest", err))?;
    let after_fold_digest = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let effective_output_count = ctx.chunk.pi_ccs.ccs_outputs.len();
    let constant_child_prefix = match ctx.boundary_plan.rlc_mode {
        super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => constant_child_prefix,
        super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => 0,
    };
    let zero_output_suffix_start = effective_output_count.saturating_sub(ctx.rlc_zero_commit_suffix_len);
    let mut padded_ccs_outputs = Vec::with_capacity(ctx.cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in ctx.cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = ctx.chunk.pi_ccs.ccs_outputs.get(output_index);
        let claim = if output_index < effective_output_count {
            super::super::cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &ctx.chunk.pi_ccs.row_chals,
                &ctx.chunk.pi_ccs.s_col,
            )
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = ctx.chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = ctx.chunk.pi_ccs.s_col.clone();
            Ok(padded_claim)
        }
        .map_err(|err| stage_err("pi_ccs_fingerprint_cover_output", err))?;
        let output = if output_index < constant_child_prefix {
            if output_index < cover_fresh_claim_count {
                let fresh = covered_fresh_claims
                    .get(output_index)
                    .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "fresh output missing"))?;
                let fresh_x_values = crate::superneo_circuit::output_binding::embedded_fresh_x_values(fresh);
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &fresh.c.data,
                    &fresh_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
            } else {
                let me_input_index = output_index
                    .checked_sub(cover_fresh_claim_count)
                    .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "me-input underflow"))?;
                let me_input = carried_claims
                    .effective_claims()
                    .get(me_input_index)
                    .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "me-input missing"))?;
                let me_input_x_values = crate::superneo_circuit::output_binding::embedded_me_input_x_values(me_input)
                    .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?;
                crate::superneo_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &me_input_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
            }
        } else if output_index < cover_fresh_claim_count {
            let fresh = covered_fresh_claim_vars
                .get(output_index)
                .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "fresh output vars missing"))?;
            crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point_compact_x(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &fresh.c_data,
                &fresh.c_data_values,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
        } else if output_index < effective_output_count {
            let me_input_index = output_index
                .checked_sub(cover_fresh_claim_count)
                .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "me-input underflow"))?;
            let me_input = carried_claims
                .effective_claims()
                .get(me_input_index)
                .ok_or_else(|| stage_err("pi_ccs_fingerprint_alloc_output", "me-input missing"))?;
            if output_index >= zero_output_suffix_start {
                crate::superneo_circuit::claim::alloc_ce_claim_x_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
            } else {
                crate::superneo_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.commitment.data,
                    &me_input.commitment.data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )
                .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
            }
        } else {
            super::super::alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &claim,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )
            .map_err(|err| stage_err("pi_ccs_fingerprint_alloc_output", err))?
        };
        padded_ccs_outputs.push(output);
    }
    let after_alloc_outputs = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    crate::superneo_circuit::output_binding::enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| format!("chunk_{}_output_binding", ctx.chunk_index)),
        ctx.structure,
        ctx.params,
        &covered_fresh_claim_vars,
        carried_claims.effective_claims(),
        &padded_ccs_outputs,
        ctx.rlc_zero_commit_suffix_len,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &format!("chunk_{}_output_binding", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_output_binding", err))?;
    let after_output_binding = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let me_inputs_r_vars = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r.as_slice());
    let me_inputs_r_values = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r_values.as_slice());
    let effective_me_output_count = effective_output_count
        .checked_sub(cover_fresh_claim_count)
        .ok_or_else(|| stage_err("pi_ccs_fingerprint_effective_me_output_count", "underflow"))?;
    let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
        .iter()
        .cloned()
        .chain(
            padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                .iter()
                .cloned(),
        )
        .collect::<Vec<_>>();
    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_fe(
        &mut cs.namespace(|| format!("chunk_{}_terminal_fe", ctx.chunk_index)),
        &sumcheck_final_fe,
        ctx.structure,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.alpha,
        &public_challenges.beta_a,
        &public_challenges.beta_r,
        &public_challenges.gamma,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &alpha_prime_vars,
        &ctx.chunk.pi_ccs.alpha_prime,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        me_inputs_r_vars,
        me_inputs_r_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_fe", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_terminal_fe", err))?;
    let after_terminal_fe = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let _ = crate::superneo_circuit::terminal_identity::enforce_terminal_identity_nc(
        &mut cs.namespace(|| format!("chunk_{}_terminal_nc", ctx.chunk_index)),
        &sumcheck_final_nc,
        ctx.params,
        &ctx.chunk.pi_ccs.public_challenges,
        &public_challenges.beta_a,
        &public_challenges.beta_m,
        &public_challenges.gamma,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &alpha_prime_nc_vars,
        &ctx.chunk.pi_ccs.alpha_prime_nc,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        ctx.rlc_zero_commit_suffix_len,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_terminal_nc", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fingerprint_terminal_nc", err))?;
    let after_terminal_nc = super::format_spartan_digest_hex(cs.finish_digest32(0));

    Ok(Rv32imPiCcsStageFingerprint {
        after_bind_header,
        after_bind_me_inputs,
        after_sample_challenges,
        after_alloc_fresh_claims,
        after_fe_sumcheck,
        after_nc_sumcheck,
        after_fold_digest,
        after_alloc_outputs,
        after_output_binding,
        after_terminal_fe,
        after_terminal_nc,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_stage_aux_counts(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepStageAuxCounts, Rv32imMainRecursionStepSpartanError> {
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let witness = &circuit.backend_relation.f_prime_advice;
    let payload = &circuit.backend_relation.payload;
    let mut cs = FingerprintCS::new();

    let public_inputs = circuit
        .expected_public_values()
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| stage_err("stage_aux_counts_public_inputs", err))?;
    let mut public_cursor = 0usize;
    let x_out_input = next_public_digest(&public_inputs, &mut public_cursor, "x_out")
        .map_err(|err| stage_err("stage_aux_counts_x_out", err))?;
    let folded_accumulator_out_digest_input =
        next_public_digest(&public_inputs, &mut public_cursor, "folded_accumulator_out_digest")
            .map_err(|err| stage_err("stage_aux_counts_folded_accumulator_out_digest", err))?;

    let next_chunk_count = witness.chunk_count_in() + 1;
    let chunk_index_halves = private_u64_halves(
        &mut cs.namespace(|| "chunk_index_halves"),
        next_chunk_count,
        "chunk_index_halves",
    )
    .map_err(|err| stage_err("stage_aux_counts_chunk_index_halves", err))?;
    let z_0_input = private_digest_inputs(&mut cs.namespace(|| "z_0"), *payload.z_0(), "z_0")
        .map_err(|err| stage_err("stage_aux_counts_z_0", err))?;
    let z_i_input = private_digest_inputs(&mut cs.namespace(|| "z_i"), *payload.z_i(), "z_i")
        .map_err(|err| stage_err("stage_aux_counts_z_i", err))?;
    let z_next_input = private_digest_inputs(&mut cs.namespace(|| "z_next"), *payload.z_next(), "z_next")
        .map_err(|err| stage_err("stage_aux_counts_z_next", err))?;
    let pc_next_halves = private_u64_halves(
        &mut cs.namespace(|| "pc_next_halves"),
        payload.pc_next(),
        "pc_next_halves",
    )
    .map_err(|err| stage_err("stage_aux_counts_pc_next_halves", err))?;
    let step_handle_meta_values = [
        SpartanF::from_canonical_u64(payload.handoff.public_chunk.start_index as u64),
        SpartanF::from_canonical_u64(payload.handoff.public_chunk.steps.len() as u64),
    ];
    let step_handle_meta = alloc_private_field_values(
        &mut cs.namespace(|| "step_handle_meta"),
        &step_handle_meta_values,
        "step_handle_meta",
    )
    .map_err(|err| stage_err("stage_aux_counts_step_handle_meta", err))?;
    let after_private_witness_inputs = cs.num_aux();

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("stage_aux_counts_state_in", err))?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("stage_aux_counts_state_out", err))?;
    let after_alloc_cover_states = cs.num_aux();

    let canonical_initial_z = digest_const_inputs(
        &mut cs.namespace(|| "canonical_initial_z"),
        crate::rv32im::chunk_step_ivc::rv32im_chunk_step_ivc_initial_state_for_step_cap(
            witness
                .verifier_key_fs()
                .step_cap()
                .map_err(|err| stage_err("stage_aux_counts_step_cap", err))?,
        )
        .carry
        .terminal_handle
        .0,
        "canonical_initial_z",
    )
    .map_err(|err| stage_err("stage_aux_counts_canonical_initial_z", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_0_eq_initial"),
        &z_0_input,
        &canonical_initial_z,
        "z_0_eq_initial",
    )
    .map_err(|err| stage_err("stage_aux_counts_z_0_eq_initial", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_i_eq_state_in_terminal_handle"),
        &z_i_input,
        &state_in_var.terminal_handle,
        "z_i_eq_state_in_terminal_handle",
    )
    .map_err(|err| stage_err("stage_aux_counts_z_i_eq_state_in_terminal_handle", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "z_next_eq_state_out_terminal_handle"),
        &z_next_input,
        &state_out_var.terminal_handle,
        "z_next_eq_state_out_terminal_handle",
    )
    .map_err(|err| stage_err("stage_aux_counts_z_next_eq_state_out_terminal_handle", err))?;
    super::ensure_unit_program_counter(payload.pc_i()).map_err(|err| stage_err("stage_aux_counts_pc_i_unit", err))?;
    super::ensure_unit_program_counter(payload.pc_next())
        .map_err(|err| stage_err("stage_aux_counts_pc_next_unit", err))?;
    let exact_initial_x_out_prefix = if payload.initial_transcript_in {
        let prefix = super::exact_initial_x_out_prefix(
            witness
                .verifier_key_fs()
                .step_cap()
                .map_err(|err| stage_err("stage_aux_counts_step_cap", err))?,
        );
        super::enforce_u64_halves_eq_constant(
            &mut cs.namespace(|| "chunk_index_eq_exact_initial"),
            &chunk_index_halves,
            prefix.next_chunk_count,
            "chunk_index_eq_exact_initial",
        );
        super::enforce_u64_halves_eq_constant(
            &mut cs.namespace(|| "pc_next_halves_eq_exact_initial"),
            &pc_next_halves,
            prefix.pc_next,
            "pc_next_halves_eq_exact_initial",
        );
        Some(prefix)
    } else {
        None
    };
    let after_bind_state_and_pc = cs.num_aux();

    let bridge_handoff_digest = super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        witness.bridge_handoff_digest(),
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("stage_aux_counts_bridge_handoff_digest", err))?;
    let live_folded_accumulator_out_digest = synthesize_rv32im_main_recursion_step_chunk_replay(
        &mut cs.namespace(|| "payload_chunk_replay"),
        witness,
        payload,
        &state_in_var,
        &state_out_var,
        &bridge_handoff_digest,
        None,
    )
    .map_err(|err| stage_err("stage_aux_counts_chunk_replay", err))?
    .live_folded_accumulator_out_digest;
    let after_chunk_replay = cs.num_aux();

    let expected_step_handle = super::fixed_shape_recursive_step_handle_digest_circuit(
        &mut cs.namespace(|| "expected_step_handle"),
        "expected_step_handle",
        &state_in_var.terminal_handle,
        &digest32_as_spartan_fields(witness.running_state().carry.terminal_handle.0),
        &chunk_index_halves,
        witness.chunk_count_in() + 1,
        &step_handle_meta[0],
        step_handle_meta_values[0],
        &step_handle_meta[1],
        step_handle_meta_values[1],
        &payload.handoff.chunk_relation_digest,
    )
    .map_err(|err| stage_err("stage_aux_counts_expected_step_handle", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "state_out_terminal_handle_eq_expected_step_handle"),
        &state_out_var.terminal_handle,
        &expected_step_handle,
        "state_out_terminal_handle_eq_expected_step_handle",
    )
    .map_err(|err| {
        stage_err(
            "stage_aux_counts_state_out_terminal_handle_eq_expected_step_handle",
            err,
        )
    })?;

    enforce_inactive_side_lane_constraints(
        &mut cs.namespace(|| "inactive_side_lane"),
        "inactive_side_lane",
        witness.side_witness().claim_count(),
        payload.phi_side_commitment_words.len() as u64,
    )
    .map_err(|err| stage_err("stage_aux_counts_inactive_side_lane", err))?;
    let live_folded_accumulator_out_digest_values = digest32_as_spartan_fields(
        crate::rv32im::final_relation::rv32im_chunk_fold_carry_recursive_accumulator_digest(
            &witness.fresh_state_out().carry,
        ),
    );
    let x_out_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "x_out_digest"),
        "x_out_digest",
        witness.verifier_key_fs().expected_digest(),
        &chunk_index_halves,
        &u64_halves_as_spartan_fields(next_chunk_count),
        &z_0_input,
        &digest32_as_spartan_fields(*payload.z_0()),
        &z_next_input,
        &digest32_as_spartan_fields(*payload.z_next()),
        &pc_next_halves,
        &u64_halves_as_spartan_fields(payload.pc_next()),
        &live_folded_accumulator_out_digest,
        &live_folded_accumulator_out_digest_values,
        exact_initial_x_out_prefix,
    )
    .map_err(|err| stage_err("stage_aux_counts_x_out_digest", err))?;
    let after_inactive_side_lane_and_x_out = cs.num_aux();

    enforce_digest_eq(
        &mut cs.namespace(|| "x_out_eq"),
        &x_out_input,
        &x_out_digest,
        "x_out_eq",
    )
    .map_err(|err| stage_err("stage_aux_counts_x_out_eq", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "folded_accumulator_out_digest_eq"),
        &folded_accumulator_out_digest_input,
        &live_folded_accumulator_out_digest,
        "folded_accumulator_out_digest_eq",
    )
    .map_err(|err| stage_err("stage_aux_counts_folded_accumulator_out_digest_eq", err))?;
    let after_public_output_eq = cs.num_aux();

    Ok(Rv32imMainRecursionStepStageAuxCounts {
        after_private_witness_inputs,
        after_alloc_cover_states,
        after_bind_state_and_pc,
        after_chunk_replay,
        after_inactive_side_lane_and_x_out,
        after_public_output_eq,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiRlcPublicConstraintBreakdown, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_rlc_public_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_rlc_public_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_rlc_public_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_rlc_public_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "pi_rlc_public_state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "pi_rlc_public_state_in",
    )
    .map_err(|err| stage_err("pi_rlc_public_state_in", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("pi_rlc_public_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "pi_rlc_public_state_in_live_claims"),
        witness,
        payload,
        "pi_rlc_public_state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_rlc_public_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_rlc_public_chunk_replay_surface", err))?;
    let checkpoints = super::super::debug_measure_rv32im_rlc_public_stage_ranges(
        params,
        structure,
        dims,
        &mat_digest,
        &witness.fresh_state_out().carry.main.claims,
        &mut cs,
        witness.chunk_index() as usize,
        &payload.chunk_cover,
        &replay_chunk,
        &mut replayed_transcript,
        carried_claims,
        payload.boundary_plan,
        payload.rlc_zero_commit_suffix_len,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    )
    .map_err(|err| stage_err("pi_rlc_public_stage_ranges", err))?;

    let mut shared_point_constraints = 0usize;
    let mut x_constraints = 0usize;
    let mut c_constraints = 0usize;
    let mut y_ring_constraints = 0usize;
    let mut y_zcol_constraints = 0usize;
    let mut aux_constraints = 0usize;
    let mut prev = 0usize;
    for (name, end) in checkpoints.stage_ends() {
        let delta = end.saturating_sub(prev);
        prev = *end;
        if name.starts_with("shared_point_") {
            shared_point_constraints += delta;
        } else if name == "x" {
            x_constraints += delta;
        } else if name == "c" {
            c_constraints += delta;
        } else if name.starts_with("y_ring_") {
            y_ring_constraints += delta;
        } else if name == "y_zcol" {
            y_zcol_constraints += delta;
        } else if name.starts_with("aux_") {
            aux_constraints += delta;
        }
    }
    Ok(Rv32imPiRlcPublicConstraintBreakdown {
        shared_point_constraints,
        x_constraints,
        c_constraints,
        y_ring_constraints,
        y_zcol_constraints,
        aux_constraints,
        total_constraints: prev,
    })
}

pub fn debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imPiRlcPublicStageBreakdown, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("pi_rlc_public_stage_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("pi_rlc_public_stage_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("pi_rlc_public_stage_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("pi_rlc_public_stage_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "pi_rlc_public_state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "pi_rlc_public_state_in",
    )
    .map_err(|err| stage_err("pi_rlc_public_stage_state_in", err))?;
    let transcript_in_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_in_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("pi_rlc_public_stage_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "pi_rlc_public_state_in_live_claims"),
        witness,
        payload,
        "pi_rlc_public_state_in_live_claims",
    )
    .map_err(|err| stage_err("pi_rlc_public_stage_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("pi_rlc_public_stage_chunk_replay_surface", err))?;
    let checkpoints = super::super::debug_measure_rv32im_rlc_public_stage_ranges(
        params,
        structure,
        dims,
        &mat_digest,
        &witness.fresh_state_out().carry.main.claims,
        &mut cs,
        witness.chunk_index() as usize,
        &payload.chunk_cover,
        &replay_chunk,
        &mut replayed_transcript,
        carried_claims,
        payload.boundary_plan,
        payload.rlc_zero_commit_suffix_len,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    )
    .map_err(|err| stage_err("pi_rlc_public_stage_ranges", err))?;

    let mut stages = Vec::with_capacity(checkpoints.stage_ends().len());
    let mut prev = 0usize;
    for (name, end) in checkpoints.stage_ends() {
        let delta = end.saturating_sub(prev);
        prev = *end;
        stages.push(Rv32imNamedConstraintDelta {
            name: name.clone(),
            delta,
        });
    }
    Ok(Rv32imPiRlcPublicStageBreakdown { stages })
}

pub fn debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepChunkReplayAuxCounts, Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("chunk_replay_context", err))?;
    let optimized_cache =
        rv32im_cached_root_main_lane_optimized_cache().map_err(|err| stage_err("chunk_replay_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("chunk_replay_dims", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("chunk_replay_mat_digest", "invalid matrix digest width"))?;

    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("chunk_replay_state_in", err))?;
    let state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("chunk_replay_state_out", err))?;
    let after_state_cover = cs.num_aux();

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("chunk_replay_surface", err))?;
    let mut replayed_transcript = super::super::import_chunk_fold_transcript_in(
        &mut cs.namespace(|| "chunk_replay_transcript"),
        &state_in_var,
        &witness.running_state().transcript,
        payload.initial_transcript_in,
        "chunk_replay_transcript",
    )
    .map_err(|err| stage_err("chunk_replay_transcript", err))?;
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("chunk_replay_live_state_in_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    )
    .map_err(|err| stage_err("chunk_replay_chunk_meta", err))?;
    let after_chunk_meta = cs.num_aux();

    let ctx = super::super::Rv32imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        me_input_accumulator_handle: Some((
            &state_in_var.folded_accumulator_digest,
            digest32_as_spartan_fields(witness.folded_accumulator_in_digest()),
        )),
        boundary_plan: payload.boundary_plan,
        rlc_zero_commit_suffix_len: payload.rlc_zero_commit_suffix_len,
        exact_initial_chunk_step_count: payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    };
    let pi_ccs = super::super::synthesize_pi_ccs_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_ccs"),
        &mut replayed_transcript,
        &carried_claims,
        None,
    )
    .map_err(|err| stage_err("chunk_replay_pi_ccs", err))?;
    let after_pi_ccs = cs.num_aux();

    let bridge_handoff_digest = super::super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        replay_chunk.handoff.bridge_handoff_digest,
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("chunk_replay_bridge_handoff_digest", err))?;
    super::super::enforce_synthetic_outer_chunk_relation_public_io(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_synthetic_relation_io"),
        &pi_ccs.fold_digest,
        &pi_ccs.public_chunk_digest,
        &bridge_handoff_digest,
        "payload_chunk_synthetic_relation_io",
    )
    .map_err(|err| stage_err("chunk_replay_synthetic_relation_io", err))?;
    let after_synthetic_relation_io = cs.num_aux();

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims,
            super::super::Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = super::super::cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))
            .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::alloc_ce_claim(
            &mut cs.namespace(|| "payload_chunk_pi_rlc_parent_claim"),
            &claim,
            "payload_chunk_pi_rlc_parent_claim",
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_alloc", err))?
    } else {
        let claim = super::super::cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| "payload_chunk_pi_rlc_parent_claim"),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            "payload_chunk_pi_rlc_parent_claim",
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_alloc", err))?
    };
    let after_pi_rlc_parent_claim = cs.num_aux();

    let child_claim_source = match ctx.boundary_plan.child_claim_source {
        super::super::Rv32imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let rho_vars = super::super::sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| "payload_chunk_pi_rlc_rhos"),
        &mut replayed_transcript,
        pi_ccs.padded_ccs_outputs.len(),
        "payload_chunk_pi_rlc_rhos",
    )
    .map_err(|err| stage_err("chunk_replay_pi_rlc_rhos", err))?;
    let after_pi_rlc_rhos = cs.num_aux();

    let after_pi_rlc_rho_mats;
    match ctx.boundary_plan.rlc_mode {
        super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
            after_pi_rlc_rho_mats = cs.num_aux();
            super::super::enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                child_claim_source,
                &rho_vars,
                ctx.params.b,
                "payload_chunk_pi_rlc_public",
            )
            .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
        }
        super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => {
            if constant_child_prefix == pi_ccs.padded_ccs_outputs.len() {
                after_pi_rlc_rho_mats = cs.num_aux();
                crate::superneo_circuit::pi_rlc::enforce_rlc_public_with_rho_coeffs_for_constant_children(
                    &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_vars,
                    "payload_chunk_pi_rlc_public",
                )
                .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
            } else {
                let active_dense_children_len = pi_ccs
                    .padded_ccs_outputs
                    .len()
                    .saturating_sub(payload.rlc_zero_commit_suffix_len);
                let rho_mats = if constant_child_prefix > 0 && constant_child_prefix < active_dense_children_len {
                    super::super::materialize_goldilocks_rot_matrices(
                        &mut cs.namespace(|| "payload_chunk_pi_rlc_rho_mats"),
                        &rho_vars[..active_dense_children_len],
                        "payload_chunk_pi_rlc_rho_mats",
                    )
                    .map_err(|err| stage_err("chunk_replay_pi_rlc_rho_mats", err))?
                } else {
                    Vec::new()
                };
                after_pi_rlc_rho_mats = cs.num_aux();
                crate::superneo_circuit::pi_rlc::enforce_rlc_public_with_split_rho_views_constant_prefix_zero_commit_suffix(
                    &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_vars,
                    &rho_mats,
                    constant_child_prefix,
                    payload.rlc_zero_commit_suffix_len,
                    "payload_chunk_pi_rlc_public",
                )
                .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
            }
        }
    }
    let after_pi_rlc_public = cs.num_aux();
    let pi_rlc = super::super::Rv32imPiRlcStageOutput { parent_claim };
    let after_pi_rlc = after_pi_rlc_public;

    let replayed_next_claims = super::super::synthesize_pi_dec_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_dec"),
        carried_claims,
        &pi_ccs,
        pi_rlc,
    )
    .map_err(|err| stage_err("chunk_replay_body", err))?;
    let after_chunk_body = cs.num_aux();

    let _expected_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "expected_folded_accumulator_out_digest"),
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_replay_expected_digest", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| "payload_chunk_done"),
            &[
                SpartanF::from_canonical_u64(RV32IM_CHUNK_DONE_RAW_TAG),
                SpartanF::from_canonical_u64(1),
            ],
        )
        .map_err(|err| stage_err("chunk_replay_chunk_done", err))?;
    let replayed_transcript_out = replayed_transcript
        .state_fields(cs.namespace(|| "payload_transcript_out"))
        .map_err(|err| stage_err("chunk_replay_transcript_out", err))?;
    let one = <FingerprintCS as ConstraintSystem<SpartanF>>::one();
    for (lane_index, (replayed_lane, state_out_lane)) in replayed_transcript_out
        .iter()
        .zip(state_out_var.transcript_state.iter())
        .enumerate()
    {
        cs.enforce(
            || format!("payload_transcript_out_lane_{lane_index}"),
            |lc| lc + replayed_lane.get_variable(),
            |lc| lc + one,
            |lc| lc + state_out_lane.get_variable(),
        );
    }
    let replayed_absorbed = SpartanF::from_canonical_u64(replayed_transcript.absorbed() as u64);
    let replayed_absorbed_var =
        AllocatedNum::alloc(cs.namespace(|| "payload_transcript_absorbed_out_expected"), || {
            Ok(replayed_absorbed)
        })
        .map_err(|err| stage_err("chunk_replay_transcript_absorbed_expected", err))?;
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + one,
        |lc| lc + replayed_absorbed_var.get_variable(),
    );
    Ok(Rv32imMainRecursionStepChunkReplayAuxCounts {
        after_state_cover,
        after_chunk_meta,
        after_pi_ccs,
        after_synthetic_relation_io,
        after_pi_rlc_parent_claim,
        after_pi_rlc_rhos,
        after_pi_rlc_rho_mats,
        after_pi_rlc_public,
        after_pi_rlc,
        after_chunk_body,
        after_chunk_replay: cs.num_aux(),
    })
}

pub fn debug_measure_rv32im_main_recursion_step_spartan_commitment_key(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<f64, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let shape =
        ShapeCS::<Rv32imDeciderEngine>::r1cs_shape(&circuit).map_err(|err| stage_err("first_step_shape", err))?;
    let _ = SplitR1CSShape::commitment_key(&[&shape]).map_err(|err| stage_err("first_step_commitment_key", err))?;
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
}

fn emit_optional_trace(trace_prefix: Option<&str>, label: &str, elapsed_ms: f64) {
    if let Some(trace_prefix) = trace_prefix {
        emit_trace(trace_prefix, label, elapsed_ms);
    }
}

fn measure_circuit_shape_with_trace(
    circuit: &Rv32imMainRecursionStepCircuit,
    trace_prefix: Option<&str>,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let mut cs = FingerprintCS::new();
    let started = Instant::now();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| stage_err("step_shape_shared", err))?;
    emit_optional_trace(trace_prefix, "shape_shared", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| stage_err("step_shape_precommitted", err))?;
    emit_optional_trace(
        trace_prefix,
        "shape_precommitted",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| stage_err("step_shape_synthesize", err))?;
    emit_optional_trace(
        trace_prefix,
        "shape_synthesize",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    let num_inputs = cs.public_input_count(circuit.num_challenges());
    let num_aux = cs.num_aux();
    let num_constraints = cs.num_constraints();
    let started = Instant::now();
    let shape_digest = cs.finish_digest32(circuit.num_challenges());
    emit_optional_trace(
        trace_prefix,
        "shape_finish_digest",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(Rv32imMainRecursionStepSpartanCircuitShape {
        num_inputs,
        num_aux,
        num_constraints,
        constraint_fingerprint: format_spartan_digest_hex(shape_digest),
    })
}

fn measure_circuit_shape(
    circuit: &Rv32imMainRecursionStepCircuit,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    measure_circuit_shape_with_trace(circuit, None)
}

pub fn debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let circuit = build_rv32im_main_recursion_step_shape_only_circuit(spartan_shape)?;
    measure_circuit_shape(&circuit)
}

pub fn debug_trace_rv32im_main_recursion_step_spartan_circuit_shape_measurement(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    emit_trace(
        trace_prefix,
        "build_live_circuit",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    measure_circuit_shape_with_trace(&circuit, Some(trace_prefix))
}

pub fn debug_trace_rv32im_main_recursion_step_shape_only_circuit_shape_measurement(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_shape_only_circuit(spartan_shape)?;
    emit_trace(
        trace_prefix,
        "build_shape_only_circuit",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    measure_circuit_shape_with_trace(&circuit, Some(trace_prefix))
}

pub fn debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRecursionStepSpartanShapeSynthesisMetrics, Rv32imMainRecursionStepSpartanError> {
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();

    let started = Instant::now();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| stage_err("first_step_shape_shared", err))?;
    let shared_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| stage_err("first_step_shape_precommitted", err))?;
    let precommitted_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let started = Instant::now();
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| stage_err("first_step_shape_synthesize", err))?;
    let synthesize_ms = started.elapsed().as_secs_f64() * 1_000.0;

    Ok(Rv32imMainRecursionStepSpartanShapeSynthesisMetrics {
        shared_ms,
        precommitted_ms,
        synthesize_ms,
        num_inputs: cs.num_inputs(),
        num_aux: cs.num_aux(),
        num_constraints: cs.num_constraints(),
    })
}

pub fn debug_trace_rv32im_main_recursion_step_spartan_shape_synthesis(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionStepSpartanShapeSynthesisMetrics, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    emit_trace(trace_prefix, "build_circuit", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let mut cs = ShapeCS::<Rv32imDeciderEngine>::new();
    emit_trace(trace_prefix, "shape_cs_new", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| stage_err("first_step_shape_shared", err))?;
    let shared_ms = started.elapsed().as_secs_f64() * 1_000.0;
    emit_trace(trace_prefix, "shared", shared_ms);

    let started = Instant::now();
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| stage_err("first_step_shape_precommitted", err))?;
    let precommitted_ms = started.elapsed().as_secs_f64() * 1_000.0;
    emit_trace(trace_prefix, "precommitted", precommitted_ms);

    let started = Instant::now();
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| stage_err("first_step_shape_synthesize", err))?;
    let synthesize_ms = started.elapsed().as_secs_f64() * 1_000.0;
    emit_trace(trace_prefix, "synthesize", synthesize_ms);

    let metrics = Rv32imMainRecursionStepSpartanShapeSynthesisMetrics {
        shared_ms,
        precommitted_ms,
        synthesize_ms,
        num_inputs: cs.num_inputs(),
        num_aux: cs.num_aux(),
        num_constraints: cs.num_constraints(),
    };
    eprintln!(
        "{trace_prefix}.sizes=num_inputs:{} num_aux:{} num_constraints:{}",
        metrics.num_inputs, metrics.num_aux, metrics.num_constraints
    );
    let _ = io::stderr().flush();
    Ok(metrics)
}

pub fn debug_trace_rv32im_main_recursion_step_fingerprint_synthesize(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    emit_trace(
        trace_prefix,
        "build_live_circuit",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let mut cs = FingerprintCS::new();
    emit_trace(
        trace_prefix,
        "fingerprint_cs_new",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let expected_public_values = circuit.expected_public_values();
    emit_trace(
        trace_prefix,
        "expected_public_values",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let public_inputs = alloc_private_field_values(
        &mut cs.namespace(|| "fingerprint_public_inputs"),
        &expected_public_values,
        "fingerprint_public_inputs",
    )
    .map_err(|err| stage_err("fingerprint_public_inputs", err))?;
    emit_trace(
        trace_prefix,
        "alloc_public_inputs",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let mut public_cursor = 0usize;
    synthesize_rv32im_main_recursion_step_body(
        &circuit,
        &mut cs.namespace(|| "fingerprint_synthesize"),
        &public_inputs,
        &mut public_cursor,
        Some(trace_prefix),
    )
    .map_err(|err| stage_err("fingerprint_synthesize", err))?;
    emit_trace(trace_prefix, "body_total", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let shape = Rv32imMainRecursionStepSpartanCircuitShape {
        num_inputs: cs.public_input_count(circuit.num_challenges()),
        num_aux: cs.num_aux(),
        num_constraints: cs.num_constraints(),
        constraint_fingerprint: format_spartan_digest_hex(cs.finish_digest32(circuit.num_challenges())),
    };
    emit_trace(trace_prefix, "finish_digest", started.elapsed().as_secs_f64() * 1_000.0);
    Ok(shape)
}

pub fn debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize(
    spartan_shape: &Rv32imMainRecursionStepSpartanShape,
    trace_prefix: &str,
) -> Result<Rv32imMainRecursionStepSpartanCircuitShape, Rv32imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv32im_main_recursion_step_shape_only_circuit(spartan_shape)?;
    emit_trace(
        trace_prefix,
        "build_shape_only_circuit",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let mut cs = FingerprintCS::new();
    emit_trace(
        trace_prefix,
        "fingerprint_cs_new",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let expected_public_values = circuit.expected_public_values();
    emit_trace(
        trace_prefix,
        "expected_public_values",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let public_inputs = alloc_private_field_values(
        &mut cs.namespace(|| "fingerprint_public_inputs"),
        &expected_public_values,
        "fingerprint_public_inputs",
    )
    .map_err(|err| stage_err("fingerprint_public_inputs", err))?;
    emit_trace(
        trace_prefix,
        "alloc_public_inputs",
        started.elapsed().as_secs_f64() * 1_000.0,
    );

    let started = Instant::now();
    let mut public_cursor = 0usize;
    synthesize_rv32im_main_recursion_step_body(
        &circuit,
        &mut cs.namespace(|| "fingerprint_synthesize"),
        &public_inputs,
        &mut public_cursor,
        Some(trace_prefix),
    )
    .map_err(|err| stage_err("fingerprint_synthesize", err))?;
    emit_trace(trace_prefix, "body_total", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let shape = Rv32imMainRecursionStepSpartanCircuitShape {
        num_inputs: cs.public_input_count(circuit.num_challenges()),
        num_aux: cs.num_aux(),
        num_constraints: cs.num_constraints(),
        constraint_fingerprint: format_spartan_digest_hex(cs.finish_digest32(circuit.num_challenges())),
    };
    emit_trace(trace_prefix, "finish_digest", started.elapsed().as_secs_f64() * 1_000.0);
    Ok(shape)
}

pub fn debug_profile_rv32im_main_recursion_step_chunk_replay_stages(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv32imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| stage_err("cached_root_main_lane_context", err))?;
    let optimized_cache = rv32im_cached_root_main_lane_optimized_cache()
        .map_err(|err| stage_err("cached_root_main_lane_optimized_cache", err))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| stage_err("build_dims_and_policy", err))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| stage_err("digest_ccs_matrices_with_sparse_cache", "matrix digest length mismatch"))?;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    eprintln!("n2-step-chunk|start|state_in");
    let _ = io::stderr().flush();
    let started = Instant::now();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("state_in", err))?;
    eprintln!(
        "n2-step-chunk|done|state_in|{:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );
    let _ = io::stderr().flush();
    eprintln!("n2-step-chunk|start|state_out");
    let _ = io::stderr().flush();
    let started = Instant::now();
    let _state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("state_out", err))?;
    eprintln!(
        "n2-step-chunk|done|state_out|{:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );
    let _ = io::stderr().flush();
    ensure_stage_satisfied(&cs, "state_alloc")?;

    let replay_chunk = payload
        .padded_chunk_replay_surface()
        .map_err(|err| stage_err("padded_chunk_replay_surface", err))?;
    let synthetic_chunk_relation_digest = alloc_const_field_values(
        &mut cs.namespace(|| "synthetic_chunk_relation_digest"),
        &digest32_as_spartan_fields(payload.handoff.chunk_relation_digest),
        "synthetic_chunk_relation_digest",
    )
    .map_err(|err| stage_err("synthetic_chunk_relation_digest", err))?;
    let mut synthetic_chunk_relation_cursor = 0usize;
    let transcript_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
    let mut replayed_transcript = Poseidon2TranscriptCircuit::from_state(
        state_in_var.transcript_state.clone(),
        transcript_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| stage_err("transcript_state_import", err))?;
    eprintln!(
        "n2-step-chunk|info|absorbed_before_chunk_meta|{}",
        replayed_transcript.absorbed()
    );
    let _ = io::stderr().flush();
    eprintln!(
        "n2-step-chunk|info|chunk_meta_words|{}",
        if replay_chunk.handoff.public_chunk.steps.len() == 1 {
            2
        } else {
            3
        }
    );
    let _ = io::stderr().flush();
    let live_state_in_claims = alloc_live_state_in_projection_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        witness,
        payload,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("state_in_live_claims", err))?;
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    crate::rv32im::main_relation_spartan::debug_profile_rv32im_main_relation_chunk_stage_progress(
        params,
        structure,
        dims,
        &mat_digest,
        &witness.fresh_state_out().carry.main.claims,
        &mut cs,
        witness.chunk_index() as usize,
        &payload.chunk_cover,
        &replay_chunk,
        &synthetic_chunk_relation_digest,
        &mut synthetic_chunk_relation_cursor,
        &mut replayed_transcript,
        carried_claims,
        // HyperNova §6.3 requires a single compiled recursive-step family.
        // The profiler must therefore follow the live padded path and bind ME
        // inputs from the allocated carried claims themselves.
        None,
        payload.boundary_plan,
        false,
    )
    .map_err(|err| stage_err("chunk_replay_profile", err))?;
    Ok(())
}

pub fn debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv32imMainRecursionStepSpartanError> {
    let claims = &backend_relation
        .f_prime_advice
        .running_state()
        .carry
        .main
        .claims;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let live_claims = alloc_recursive_cover_claims(&mut cs.namespace(|| "live_claims"), claims, "live_claims")
        .map_err(|err| stage_err("live_claims", err))?;
    ensure_stage_satisfied(&cs, "live_claims")?;

    let mut scratch = Vec::<F>::with_capacity(2048);
    for (claim_index, (native_claim, live_claim)) in claims.iter().zip(live_claims.iter()).enumerate() {
        let digest = me_digest_poseidon(
            &mut cs.namespace(|| format!("live_claim_digest_{claim_index}")),
            &live_claim.claim,
            &format!("live_claim_digest_{claim_index}"),
        )
        .map_err(|err| stage_err("live_claim_digest", err))?;
        ensure_stage_satisfied(&cs, &format!("live_claim_digest[{claim_index}]"))?;
        let actual =
            allocated_digest_field_values(&digest).map_err(|err| stage_err("live_claim_digest_values", err))?;
        let expected = me_digest_poseidon_into(&mut scratch, native_claim)
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));
        if actual != expected {
            return Err(stage_err(
                "live_claim_digest_parity",
                format!("claim {claim_index} digest mismatch"),
            ));
        }
    }

    Ok(())
}

pub fn debug_check_rv32im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv32imMainRecursionStepSpartanError> {
    let claims = &backend_relation
        .f_prime_advice
        .fresh_state_out()
        .carry
        .main
        .claims;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let output_claims = alloc_recursive_cover_claims(&mut cs.namespace(|| "output_claims"), claims, "output_claims")
        .map_err(|err| stage_err("output_claims", err))?;
    let output_terminal_handle = digest_const_inputs(
        &mut cs.namespace(|| "output_terminal_handle"),
        backend_relation
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
        "output_terminal_handle",
    )
    .map_err(|err| stage_err("output_terminal_handle", err))?;
    ensure_stage_satisfied(&cs, "output_claims")?;

    let output_claim_vars = output_claims
        .into_iter()
        .map(|claim| claim.claim)
        .collect::<Vec<_>>();
    let digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "output_accumulator_digest"),
        &output_claim_vars,
        &output_terminal_handle,
        "output_accumulator_digest",
    )
    .map_err(|err| stage_err("output_accumulator_digest", err))?;
    ensure_stage_satisfied(&cs, "output_accumulator_digest")?;

    let actual = allocated_digest_field_values(&digest).map_err(|err| stage_err("output_digest_values", err))?;
    let expected = digest32_as_spartan_fields(
        crate::rv32im::final_relation::rv32im_chunk_fold_carry_recursive_accumulator_digest(
            &backend_relation.f_prime_advice.fresh_state_out().carry,
        ),
    );
    if actual != expected {
        return Err(stage_err(
            "fresh_output_accumulator_digest_parity",
            "fresh output accumulator digest mismatch",
        ));
    }

    Ok(())
}
