use std::io::{self, Write};
use std::time::Instant;

use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_reductions::engines::utils::me_digest_poseidon_into;
use neo_reductions::engines::utils::{build_dims_and_policy, digest_ccs_matrices_with_sparse_cache};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use spartan2::traits::circuit::SpartanCircuit;
use spartan2::{
    bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS},
    provider::goldi::F as SpartanF,
    SplitR1CSShape,
};

use super::*;
use crate::rv64im::final_relation::RV64IM_CHUNK_DONE_RAW_TAG;
use crate::rv64im::kernel::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
use crate::rv64im::main_relation_circuit::claim::{enforce_claim_eq_native, me_digest_poseidon};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::rv64im_chunk_step_recursive_carry_state_digest;
use crate::rv64im::main_relation_spartan::fingerprint_cs::FingerprintCS;
use crate::rv64im::main_relation_spartan::recursive_cover::{
    alloc_recursive_cover_claims, alloc_recursive_cover_state,
    recursive_accumulator_instance_digest_circuit_from_claims,
};
use crate::rv64im::main_relation_spartan::Rv64imClaimBundle;

fn stage_err(stage: &str, err: impl ToString) -> Rv64imMainRecursionStepSpartanError {
    Rv64imMainRecursionStepSpartanError::Prepare(format!("{stage}: {}", err.to_string()))
}

fn emit_trace(trace_prefix: &str, label: &str, elapsed_ms: f64) {
    eprintln!("{trace_prefix}.{label}={elapsed_ms:.2}ms");
    let _ = io::stderr().flush();
}

fn ensure_stage_satisfied(
    cs: &TestConstraintSystem<SpartanF>,
    stage: &str,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(stage_err(
            stage,
            cs.which_is_unsatisfied().unwrap_or("unknown constraint"),
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rv64imMainRecursionStepSpartanShapeSynthesisMetrics {
    pub shared_ms: f64,
    pub precommitted_ms: f64,
    pub synthesize_ms: f64,
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imMainRecursionStepChunkReplayFingerprint {
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

pub fn debug_measure_rv64im_main_recursion_step_chunk_replay_fingerprint(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepChunkReplayFingerprint, Rv64imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let mut cs = FingerprintCS::new();
    let (params, _, structure) =
        rv64im_cached_root_main_lane_context().map_err(|err| stage_err("chunk_replay_context", err))?;
    let optimized_cache =
        rv64im_cached_root_main_lane_optimized_cache().map_err(|err| stage_err("chunk_replay_optimized_cache", err))?;
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
        .effective_chunk_replay_surface(
            &witness.running_state().transcript,
            &witness.running_state().carry.main.claims,
        )
        .map_err(|err| stage_err("chunk_replay_surface", err))?;
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
    .map_err(|err| stage_err("chunk_replay_transcript", err))?;
    let live_state_in_claims = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        &payload.state_in_claims,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("chunk_replay_live_state_in_claims", err))?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    super::super::append_chunk_meta(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
    )
    .map_err(|err| stage_err("chunk_replay_chunk_meta", err))?;
    let after_chunk_meta = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let ctx = super::super::Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest: &mat_digest,
        terminal_final_claims: &witness.fresh_state_out().carry.main.claims,
        chunk_index: witness.chunk_index() as usize,
        cover_chunk: &payload.chunk_cover,
        chunk: &replay_chunk,
        logical_me_input_claims: None,
        boundary_plan: payload.boundary_plan,
    };
    let pi_ccs = super::super::synthesize_pi_ccs_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_ccs"),
        &mut replayed_transcript,
        &carried_claims,
    )
    .map_err(|err| stage_err("chunk_replay_pi_ccs", err))?;
    let after_pi_ccs = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    super::super::enforce_synthetic_outer_chunk_relation_public_io(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_synthetic_relation_io"),
        &mut replayed_transcript,
        "payload_chunk_synthetic_relation_io",
    )
    .map_err(|err| stage_err("chunk_replay_synthetic_relation_io", err))?;
    let after_synthetic_relation_io = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            super::super::Rv64imChunkChildClaimSource::TerminalFinalClaims,
            super::super::Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
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
        super::super::Rv64imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        super::super::Rv64imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let padded_rho_count = pi_ccs
        .padded_ccs_outputs
        .len()
        .saturating_sub(pi_ccs.effective_output_count);
    let mut rho_vars = super::super::sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| "payload_chunk_pi_rlc_rhos"),
        &mut replayed_transcript,
        pi_ccs.effective_output_count,
        "payload_chunk_pi_rlc_rhos",
    )
    .map_err(|err| stage_err("chunk_replay_pi_rlc_rhos", err))?;
    if padded_rho_count > 0 {
        rho_vars.extend(
            super::super::alloc_zero_rot_rhos(
                &mut cs.namespace(|| "payload_chunk_pi_rlc_rhos_pad"),
                padded_rho_count,
                "payload_chunk_pi_rlc_rhos_pad",
            )
            .map_err(|err| stage_err("chunk_replay_pi_rlc_rhos_pad", err))?,
        );
    }
    let after_pi_rlc_rhos = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let after_pi_rlc_rho_mats;
    match ctx.boundary_plan.rlc_mode {
        super::super::Rv64imChunkRlcMode::TerminalLastChunkShortcut => {
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
        super::super::Rv64imChunkRlcMode::Standard { constant_child_prefix } => {
            let mut rho_mats = super::super::materialize_goldilocks_rot_matrices(
                &mut cs.namespace(|| "payload_chunk_pi_rlc_rho_mats"),
                &rho_vars[..pi_ccs.effective_output_count],
                "payload_chunk_pi_rlc_rho_mats",
            )
            .map_err(|err| stage_err("chunk_replay_pi_rlc_rho_mats", err))?;
            if padded_rho_count > 0 {
                rho_mats.extend(
                    super::super::alloc_zero_rot_rho_matrices(
                        &mut cs.namespace(|| "payload_chunk_pi_rlc_rho_mats_pad"),
                        padded_rho_count,
                        "payload_chunk_pi_rlc_rho_mats_pad",
                    )
                    .map_err(|err| stage_err("chunk_replay_pi_rlc_rho_mats_pad", err))?,
                );
            }
            after_pi_rlc_rho_mats = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));
            super::super::enforce_rlc_public_with_rho_vars_constant_prefix(
                &mut cs.namespace(|| "payload_chunk_pi_rlc_public"),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                &rho_mats,
                constant_child_prefix,
                "payload_chunk_pi_rlc_public",
            )
            .map_err(|err| stage_err("chunk_replay_pi_rlc_public", err))?;
        }
    }
    let after_pi_rlc_public = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));
    let pi_rlc = super::super::Rv64imPiRlcStageOutput { parent_claim };
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
            "chunk_replay_effective_count",
            "replayed effective-count mismatch",
        ));
    }
    for (claim_index, (replayed_claim, expected_claim)) in replayed_next_claims
        .effective_claims()
        .iter()
        .zip(witness.fresh_state_out().carry.main.claims.iter())
        .enumerate()
    {
        enforce_claim_eq_native(
            &mut cs.namespace(|| format!("payload_state_out_claim_eq_{claim_index}")),
            replayed_claim,
            expected_claim,
            &format!("payload_state_out_claim_eq_{claim_index}"),
        )
        .map_err(|err| stage_err("chunk_replay_state_out_claim_eq", err))?;
    }
    let expected_state_out_claims = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "state_out_expected_claims"),
        &payload.state_out_claims,
        "state_out_expected_claims",
    )
    .map_err(|err| stage_err("chunk_replay_expected_state_out_claims", err))?;
    let expected_state_out_claim_vars = expected_state_out_claims
        .into_iter()
        .map(|claim| claim.claim)
        .collect::<Vec<_>>();
    let live_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "live_folded_accumulator_out_digest"),
        replayed_next_claims.effective_claims(),
        &state_out_var.terminal_handle,
        "live_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_replay_live_digest", err))?;
    let expected_folded_accumulator_out_digest = recursive_accumulator_instance_digest_circuit_from_claims(
        &mut cs.namespace(|| "expected_folded_accumulator_out_digest"),
        &expected_state_out_claim_vars,
        &state_out_var.terminal_handle,
        "expected_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("chunk_replay_expected_digest", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "payload_state_out_digest_eq"),
        &live_folded_accumulator_out_digest,
        &expected_folded_accumulator_out_digest,
        "payload_state_out_digest_eq",
    )
    .map_err(|err| stage_err("chunk_replay_digest_eq", err))?;
    replayed_transcript
        .append_const_fields_raw(
            cs.namespace(|| "payload_chunk_done"),
            &[
                SpartanF::from_canonical_u64(RV64IM_CHUNK_DONE_RAW_TAG),
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
    cs.enforce(
        || "payload_transcript_absorbed_out",
        |lc| lc + state_out_var.transcript_absorbed.get_variable(),
        |lc| lc + one,
        |lc| lc + (replayed_absorbed, one),
    );

    Ok(Rv64imMainRecursionStepChunkReplayFingerprint {
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

pub fn debug_measure_rv64im_main_recursion_step_spartan_commitment_key(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<f64, Rv64imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let shape = ShapeCS::<Rv64imSpartan2DeciderEngine>::r1cs_shape(&circuit)
        .map_err(|err| stage_err("first_step_shape", err))?;
    let _ = SplitR1CSShape::commitment_key(&[&shape]).map_err(|err| stage_err("first_step_commitment_key", err))?;
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
}

pub fn debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanShapeSynthesisMetrics, Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = ShapeCS::<Rv64imSpartan2DeciderEngine>::new();

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

    Ok(Rv64imMainRecursionStepSpartanShapeSynthesisMetrics {
        shared_ms,
        precommitted_ms,
        synthesize_ms,
        num_inputs: cs.num_inputs(),
        num_aux: cs.num_aux(),
        num_constraints: cs.num_constraints(),
    })
}

pub fn debug_trace_rv64im_main_recursion_step_spartan_shape_synthesis(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
    trace_prefix: &str,
) -> Result<Rv64imMainRecursionStepSpartanShapeSynthesisMetrics, Rv64imMainRecursionStepSpartanError> {
    let started = Instant::now();
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    emit_trace(trace_prefix, "build_circuit", started.elapsed().as_secs_f64() * 1_000.0);

    let started = Instant::now();
    let mut cs = ShapeCS::<Rv64imSpartan2DeciderEngine>::new();
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

    let metrics = Rv64imMainRecursionStepSpartanShapeSynthesisMetrics {
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

pub fn debug_profile_rv64im_main_recursion_step_chunk_replay_stages(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let (params, _, structure) =
        rv64im_cached_root_main_lane_context().map_err(|err| stage_err("cached_root_main_lane_context", err))?;
    let optimized_cache = rv64im_cached_root_main_lane_optimized_cache()
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
        .effective_chunk_replay_surface(
            &witness.running_state().transcript,
            &witness.running_state().carry.main.claims,
        )
        .map_err(|err| stage_err("effective_chunk_replay_surface", err))?;
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
    let live_state_in_claims = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        &witness.running_state().carry.main.claims,
        "state_in_live_claims",
    )
    .map_err(|err| stage_err("state_in_live_claims", err))?;
    let carried_claims = Rv64imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    crate::rv64im::main_relation_spartan::debug_profile_rv64im_main_relation_chunk_stage_progress(
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
        Some(&witness.running_state().carry.main.claims),
        payload.boundary_plan,
        false,
    )
    .map_err(|err| stage_err("chunk_replay_profile", err))?;
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
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

pub fn debug_check_rv64im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
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
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(
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

fn alloc_const_u64(
    cs: &mut TestConstraintSystem<SpartanF>,
    label: &str,
    value: u64,
) -> Result<AllocatedNum<SpartanF>, Rv64imMainRecursionStepSpartanError> {
    alloc_const_field_values(
        &mut cs.namespace(|| label.to_string()),
        &[SpartanF::from_canonical_u64(value)],
        label,
    )
    .map_err(|err| stage_err(label, err))?
    .into_iter()
    .next()
    .ok_or_else(|| stage_err(label, "missing u64 allocation"))
}

fn alloc_wrapper_step_public_var(
    cs: &mut TestConstraintSystem<SpartanF>,
    step_index: usize,
    relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<
    (
        Rv64imMainRecursionStepPublicVar,
        Rv64imMainRecursionStepSpartanStatement,
    ),
    Rv64imMainRecursionStepSpartanError,
> {
    let statement = relation.spartan_statement.clone();
    let carry_state_in_digest_value = rv64im_chunk_step_recursive_carry_state_digest(
        &relation.payload.state_in_claims,
        &relation.f_prime_advice.running_state().transcript,
        relation
            .f_prime_advice
            .running_state()
            .carry
            .terminal_handle
            .0,
    );
    let carry_state_out_digest_value = rv64im_chunk_step_recursive_carry_state_digest(
        &relation.payload.state_out_claims,
        &relation.payload.fixed_transcript_out,
        relation
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
    );
    let folded_accumulator_in_digest_value =
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(
            &relation.f_prime_advice.running_state().carry,
        );
    let chunk_index = alloc_const_u64(
        cs,
        &format!("wrapper_step_{step_index}_chunk_index"),
        relation.f_prime_advice.chunk_index(),
    )?;
    let carry_state_in_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_carry_state_in_digest")),
        carry_state_in_digest_value,
        &format!("wrapper_step_{step_index}_carry_state_in_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_carry_state_in_digest", err))?;
    let folded_accumulator_in_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_folded_accumulator_in_digest")),
        folded_accumulator_in_digest_value,
        &format!("wrapper_step_{step_index}_folded_accumulator_in_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_folded_accumulator_in_digest", err))?;
    let carry_state_out_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_carry_state_out_digest")),
        carry_state_out_digest_value,
        &format!("wrapper_step_{step_index}_carry_state_out_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_carry_state_out_digest", err))?;
    let x_out = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_x_out")),
        statement.x_out.bytes(),
        &format!("wrapper_step_{step_index}_x_out"),
    )
    .map_err(|err| stage_err("wrapper_step_x_out", err))?;
    let folded_accumulator_out_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_folded_accumulator_out_digest")),
        statement.folded_accumulator_digest,
        &format!("wrapper_step_{step_index}_folded_accumulator_out_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_folded_accumulator_out_digest", err))?;
    Ok((
        Rv64imMainRecursionStepPublicVar {
            chunk_index,
            carry_state_in_digest,
            folded_accumulator_in_digest,
            carry_state_out_digest,
            x_out,
            folded_accumulator_out_digest,
        },
        statement,
    ))
}

pub fn debug_check_rv64im_main_recursion_step_spartan_compressed_chain_wrapper_only(
    chain_shape: &Rv64imMainRecursionStepSpartanCompressedChainShape,
    backend_relations: &[Rv64imMainRecursionFPrimeBackendRelation],
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let statement = build_rv64im_main_recursion_step_spartan_statement(backend_relations)?;
    let _ = super::compressed_chain::build_rv64im_main_recursion_step_compressed_chain_circuit_from_relations(
        chain_shape,
        backend_relations,
    )?;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let x_out_input = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_x_out"),
        statement.x_out.bytes(),
        "wrapper_x_out",
    )
    .map_err(|err| stage_err("wrapper_x_out", err))?;
    let folded_accumulator_out_digest_input = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_folded_accumulator_out_digest"),
        statement.folded_accumulator_digest,
        "wrapper_folded_accumulator_out_digest",
    )
    .map_err(|err| stage_err("wrapper_folded_accumulator_out_digest", err))?;

    let initial_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let initial_state_claims = backend_relations
        .first()
        .map(|relation| relation.payload.state_in_claims.clone())
        .unwrap_or_else(|| initial_state.carry.main.claims.clone());
    let initial_carry_state_digest = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_carry_state_digest"),
        rv64im_chunk_step_recursive_carry_state_digest(
            &initial_state_claims,
            &initial_state.transcript,
            initial_state.carry.terminal_handle.0,
        ),
        "wrapper_initial_carry_state_digest",
    )
    .map_err(|err| stage_err("wrapper_initial_carry_state_digest", err))?;
    let initial_x_out = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_x_out"),
        statement.x_out.bytes(),
        "wrapper_initial_x_out",
    )
    .map_err(|err| stage_err("wrapper_initial_x_out", err))?;
    let initial_folded_accumulator_digest = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_folded_accumulator_digest"),
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry),
        "wrapper_initial_folded_accumulator_digest",
    )
    .map_err(|err| stage_err("wrapper_initial_folded_accumulator_digest", err))?;

    let mut previous_step: Option<Rv64imMainRecursionStepPublicVar> = None;
    for (step_index, relation) in backend_relations.iter().enumerate() {
        let (step_public, _statement) = alloc_wrapper_step_public_var(&mut cs, step_index, relation)?;
        ensure_stage_satisfied(&cs, &format!("wrapper_step_alloc[{step_index}]"))?;

        if let Some(previous) = previous_step.as_ref() {
            enforce_digest_eq(
                &mut cs.namespace(|| format!("wrapper_step_{step_index}_accumulator_chain")),
                &previous.carry_state_out_digest,
                &step_public.carry_state_in_digest,
                &format!("wrapper_step_{step_index}_accumulator_chain"),
            )
            .map_err(|err| stage_err("wrapper_accumulator_chain", err))?;
            enforce_digest_eq(
                &mut cs.namespace(|| format!("wrapper_step_{step_index}_folded_accumulator_chain")),
                &previous.folded_accumulator_out_digest,
                &step_public.folded_accumulator_in_digest,
                &format!("wrapper_step_{step_index}_folded_accumulator_chain"),
            )
            .map_err(|err| stage_err("wrapper_folded_accumulator_chain", err))?;
        } else {
            if relation.f_prime_advice.chunk_index() != 0 {
                return Err(stage_err(
                    "wrapper_initial_chunk_index_eq",
                    format!(
                        "expected first chunk index 0, found {}",
                        relation.f_prime_advice.chunk_index()
                    ),
                ));
            }
            enforce_digest_eq(
                &mut cs.namespace(|| "wrapper_initial_carry_state_chain"),
                &initial_carry_state_digest,
                &step_public.carry_state_in_digest,
                "wrapper_initial_carry_state_chain",
            )
            .map_err(|err| stage_err("wrapper_initial_carry_state_chain", err))?;
            enforce_digest_eq(
                &mut cs.namespace(|| "wrapper_initial_folded_accumulator_chain"),
                &initial_folded_accumulator_digest,
                &step_public.folded_accumulator_in_digest,
                "wrapper_initial_folded_accumulator_chain",
            )
            .map_err(|err| stage_err("wrapper_initial_folded_accumulator_chain", err))?;
        }

        previous_step = Some(step_public);
    }

    let final_x_out = previous_step
        .as_ref()
        .map(|step| step.x_out.clone())
        .unwrap_or(initial_x_out);
    let final_folded_accumulator_digest = previous_step
        .as_ref()
        .map(|step| step.folded_accumulator_out_digest.clone())
        .unwrap_or(initial_folded_accumulator_digest);
    enforce_digest_eq(
        &mut cs.namespace(|| "wrapper_x_out_eq"),
        &x_out_input,
        &final_x_out,
        "wrapper_x_out_eq",
    )
    .map_err(|err| stage_err("wrapper_x_out_eq", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "wrapper_folded_accumulator_output_eq"),
        &folded_accumulator_out_digest_input,
        &final_folded_accumulator_digest,
        "wrapper_folded_accumulator_output_eq",
    )
    .map_err(|err| stage_err("wrapper_folded_accumulator_output_eq", err))?;

    ensure_stage_satisfied(&cs, "compressed_chain_wrapper_only")
}
