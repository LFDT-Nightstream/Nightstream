//! Owns audit helpers for native F', NIFS, and recursive-step Spartan surfaces.

use crate::rv32im::chunk::step_ivc::Rv32imChunkStepIvcRelation;
use crate::rv32im::construction2::{
    audit_rv32im_main_recursion_construction2_pi_rlc_rho_mats,
    build_rv32im_main_recursion_construction2_default_fresh_instance,
    build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i,
    build_rv32im_main_recursion_construction2_input_state_image, build_rv32im_main_recursion_construction2_nifs_bridge,
    build_rv32im_main_recursion_construction2_output_state_image,
    build_rv32im_main_recursion_construction2_verified_step_statement_from_relation,
    verify_rv32im_main_recursion_construction2_nifs_step,
};
use crate::rv32im::f_prime::{
    build_rv32im_main_recursion_backend_statement_from_advice, build_rv32im_main_recursion_base_case_default_carry,
    build_rv32im_main_recursion_x_hash_from_advice,
};
use crate::rv32im::main_relation_trace::build_rv32im_main_circuit_chunk_trace_from_authoritative_parts;
use crate::rv32im::nifs::{
    prove_rv32im_nifs_step, verify_rv32im_nifs_step, Rv32imNifsFreshInstance, Rv32imNifsFreshWitness,
    Rv32imNifsRunningWitness,
};
use crate::rv32im::recursion_spartan::build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs;
use crate::rv32im::SimpleKernelError;
use neo_ccs::{check_ccs_rowwise_zero, check_ce_consistency, CeWitness};
use neo_math::{F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

pub use crate::rv32im::f_prime::{
    build_rv32im_main_recursion_f_prime_advices, build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_advices_single_step_with_perf,
    build_rv32im_main_recursion_f_prime_advices_with_perf, build_rv32im_main_recursion_f_prime_public_output,
    debug_trace_rv32im_main_recursion_f_prime_advices_single_step_build, evaluate_rv32im_main_recursion_f_prime_advice,
    verify_rv32im_main_recursion_f_prime_public_output, Rv32imEncodedPublicInput, Rv32imMainRecursionFPrimeAdvice,
    Rv32imMainRecursionFPrimeAdviceBuildPerf, Rv32imMainRecursionFPrimeAdviceStepBuildPerf,
    Rv32imMainRecursionFPrimePublicOutput, Rv32imMainRecursionFPrimeStepImage, Rv32imMainRecursionSideClaim,
    Rv32imMainRecursionSideLaneWitness,
};
pub use crate::rv32im::main_relation_spartan::{
    build_rv32im_main_recursion_f_prime_backend_relations,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf,
    build_rv32im_main_recursion_f_prime_claim_cover, build_rv32im_main_recursion_f_prime_payload,
    build_rv32im_main_recursion_f_prime_payloads, build_rv32im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv32im_main_recursion_step_authoritative_chunk_surface,
    build_rv32im_main_recursion_step_spartan_published_target, build_rv32im_main_recursion_step_spartan_shape,
    debug_check_rv32im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics,
    debug_check_rv32im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv32im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv32im_main_recursion_step_spartan_circuit,
    debug_check_rv32im_main_recursion_step_spartan_embedded_body,
    debug_check_rv32im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity,
    debug_check_rv32im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    debug_check_rv32im_main_recursion_x_out_gadget_parity,
    debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint,
    debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown,
    debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis,
    debug_measure_rv32im_main_recursion_step_stage_aux_counts,
    debug_profile_rv32im_main_recursion_step_chunk_replay_stages,
    debug_trace_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_trace_rv32im_main_recursion_step_fingerprint_synthesize,
    debug_trace_rv32im_main_recursion_step_shape_only_circuit_shape_measurement,
    debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize,
    debug_trace_rv32im_main_recursion_step_spartan_circuit_shape_measurement,
    debug_trace_rv32im_main_recursion_step_spartan_shape_synthesis, Rv32imCcsClaimShape, Rv32imCcsWitnessShape,
    Rv32imCeClaimDigestShape, Rv32imChunkStepIvcShape, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionFPrimeBackendRelationBuildPerf, Rv32imMainRecursionFPrimeClaimCover,
    Rv32imMainRecursionFPrimePayload, Rv32imMainRecursionStepAuthoritativeChunkSurface,
    Rv32imMainRecursionStepChunkReplayAuxCounts, Rv32imMainRecursionStepChunkReplayTailAuxCounts,
    Rv32imMainRecursionStepChunkReplayTailDigestAuxBreakdown, Rv32imMainRecursionStepSpartanCircuitShape,
    Rv32imMainRecursionStepSpartanError, Rv32imMainRecursionStepSpartanPublishedTarget,
    Rv32imMainRecursionStepSpartanShape, Rv32imMainRecursionStepSpartanStatement,
    Rv32imMainRecursionStepStageAuxCounts, Rv32imNamedConstraintDelta, Rv32imPiCcsBindMeInputsAuxBreakdown,
    Rv32imPiCcsStageAuxCounts, Rv32imPiCcsStageConstraintCounts, Rv32imPiCcsStageFingerprint,
    Rv32imPiCcsSumcheckConstraintBreakdown, Rv32imPiRlcPublicConstraintBreakdown, Rv32imPiRlcPublicStageBreakdown,
};
pub fn debug_trace_rv32im_main_recursion_construction2_default_pair_for_full_width(
    vk_fs: &crate::rv32im::f_prime::Rv32imVerifierKeyFs,
    full_width: usize,
    trace_prefix: &str,
) -> Result<crate::rv32im::Rv32imMainRecursionConstruction2DefaultPair, SimpleKernelError> {
    crate::rv32im::construction2::default::debug_trace_build_rv32im_main_recursion_construction2_default_pair_for_full_width(
        vk_fs,
        full_width,
        trace_prefix,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv32imMainCircuitChunkTraceAuthoritativeSummary {
    pub step_lo: u64,
    pub step_hi: u64,
    pub chunk_relation_digest: [u8; 32],
}

pub fn audit_build_rv32im_main_circuit_chunk_trace_authoritative_summary(
    relation: &Rv32imChunkStepIvcRelation,
) -> Result<Rv32imMainCircuitChunkTraceAuthoritativeSummary, SimpleKernelError> {
    let chunk_trace = build_rv32im_main_circuit_chunk_trace_from_authoritative_parts(
        relation.witness.handoff.bridge_handoff.chunk_index as usize,
        &relation.witness.handoff,
        &relation.statement.chunk_summary,
        &relation.witness.state_in.carry,
        &relation.witness.state_out.carry,
        &relation.witness.state_in.transcript,
        &relation.witness.state_out.transcript,
        &relation.witness.replay_witness,
    )?;
    Ok(Rv32imMainCircuitChunkTraceAuthoritativeSummary {
        step_lo: chunk_trace.step_lo(),
        step_hi: chunk_trace.step_hi(),
        chunk_relation_digest: chunk_trace.handoff.chunk_relation_digest,
    })
}

pub fn audit_rv32im_nifs_round_trip_from_chunk_step_relation(
    relation: &Rv32imChunkStepIvcRelation,
) -> Result<(), SimpleKernelError> {
    let running = Rv32imNifsRunningWitness {
        state: relation.witness.state_in.clone(),
    };
    let fresh_instance = Rv32imNifsFreshInstance {
        step_public: relation.statement.step_public.clone(),
        chunk_summary: relation.statement.chunk_summary.clone(),
    };
    let fresh_witness = Rv32imNifsFreshWitness {
        handoff: relation.witness.handoff.clone(),
        state_out: relation.witness.state_out.clone(),
    };
    let proof = prove_rv32im_nifs_step(&running, &fresh_instance, &fresh_witness)?;
    let next_running = verify_rv32im_nifs_step(&running, &fresh_instance, &fresh_witness, &proof)?;

    if next_running.state.carry.terminal_handle != relation.witness.state_out.carry.terminal_handle {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit NIFS round-trip terminal handle does not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.transcript != relation.witness.state_out.transcript {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit NIFS round-trip transcript does not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.carry.main.claims != relation.witness.state_out.carry.main.claims {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit NIFS round-trip carried claims do not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.carry.main.witnesses != relation.witness.state_out.carry.main.witnesses {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit NIFS round-trip carried witnesses do not match the chunk-step relation witness".into(),
        ));
    }
    Ok(())
}

pub fn audit_rv32im_main_recursion_backend_statement_matches_native_f_prime(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    let step_image = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
    let rebuilt_statement = build_rv32im_main_recursion_backend_statement_from_advice(advice)?;
    if rebuilt_statement.x_out != *step_image.x_out()
        || rebuilt_statement.folded_accumulator_digest != step_image.folded_accumulator_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit main-recursion backend statement does not match the native F' step image".into(),
        ));
    }
    Ok(())
}

pub fn audit_rv32im_main_recursion_default_carry_satisfies_r1_literally(
    template_state: &crate::rv32im::final_relation::Rv32imChunkFoldState,
) -> Result<(), SimpleKernelError> {
    let default_carry = build_rv32im_main_recursion_base_case_default_carry(template_state)?;
    audit_rv32im_main_recursion_default_carry_claims_and_witnesses_satisfy_r1(&default_carry, "canonical default carry")
}

fn audit_rv32im_main_recursion_default_carry_claims_and_witnesses_satisfy_r1(
    carry: &crate::proof::Carry,
    label: &str,
) -> Result<(), SimpleKernelError> {
    if carry.claims.len() != carry.witnesses.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM audit default-carry R1 check requires one witness per carried CE claim in the {label}"
        )));
    }
    let (params, log, structure) = crate::rv32im::kernel::rv32im_cached_root_main_lane_context()?;
    for (claim_index, (claim, witness)) in carry.claims.iter().zip(carry.witnesses.iter()).enumerate() {
        let zero_x = vec![neo_math::F::ZERO; claim.m_in];
        let zero_w = vec![neo_math::F::ZERO; structure.m.saturating_sub(claim.m_in)];
        check_ccs_rowwise_zero(structure, &zero_x, &zero_w).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit default-carry R1 check failed CCS row-wise zero for {label} claim {claim_index}: {err}"
            ))
        })?;
        check_ce_consistency(params, structure, log, claim, &CeWitness { Z: witness.clone() }).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit default-carry R1 check failed CE consistency for {label} claim {claim_index}: {err}"
            ))
        })?;
        if witness
            .as_slice()
            .iter()
            .any(|value| *value != neo_math::F::ZERO)
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM audit default-carry R1 check requires {label} claim {claim_index} to use the canonical zero witness"
            )));
        }
    }
    Ok(())
}

pub fn audit_rv32im_main_recursion_construction2_state_images_match_native_f_prime(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    let input_state_image = build_rv32im_main_recursion_construction2_input_state_image(advice);
    let input_x_i = input_state_image.encoded_public_input();
    if input_x_i != *advice.x_i() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit Construction-2 input state image did not encode the carried native x_i".into(),
        ));
    }

    let Some(construction2_input_u_i) = advice.construction2_input_fresh_instance() else {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit Construction-2 state-image parity requires a threaded input fresh instance u_i".into(),
        ));
    };
    if advice.chunk_count_in() == 0 {
        let canonical_full_width =
            crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_canonical_full_width(
                advice.verifier_key_fs(),
                advice.phi_side(),
            )?;
        let expected_default = build_rv32im_main_recursion_construction2_default_fresh_instance(
            advice.verifier_key_fs(),
            canonical_full_width,
        )?;
        if construction2_input_u_i != &expected_default {
            return Err(SimpleKernelError::Bridge(
                "RV32IM audit Construction-2 base-case input fresh instance is not the canonical default witness-backed u_perp"
                    .into(),
            ));
        }
    } else if construction2_input_u_i.x_i() != &input_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit Construction-2 input fresh instance x_i drifted from the canonical input state image".into(),
        ));
    }

    let step_image = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
    let output_state_image = build_rv32im_main_recursion_construction2_output_state_image(advice)?;
    let output_x_i = output_state_image.encoded_public_input();
    if output_x_i != *step_image.x_out() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit Construction-2 output state image did not encode the native F' x_{i+1}".into(),
        ));
    }
    if step_image.construction2_u_next().x_i() != &output_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit Construction-2 output fresh instance x_i drifted from the canonical output state image"
                .into(),
        ));
    }

    Ok(())
}

pub fn audit_rv32im_main_recursion_step_spartan_published_target_matches_construction2_state_images(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let advice = &backend_relation.f_prime_advice;
    audit_rv32im_main_recursion_construction2_state_images_match_native_f_prime(advice)?;

    let published_target =
        build_rv32im_main_recursion_step_spartan_published_target(backend_relation).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit failed to build recursive-step published target from backend relation: {err}"
            ))
        })?;
    let step_image = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
    let output_state_image = build_rv32im_main_recursion_construction2_output_state_image(advice)?;
    let output_x_i = output_state_image.encoded_public_input();

    if published_target.x_out != output_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit recursive-step published target x_{i+1} drifted from the canonical Construction-2 output state image"
                .into(),
        ));
    }
    if published_target.folded_accumulator_out_digest != step_image.folded_accumulator_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit recursive-step published target folded-accumulator output drifted from the native F' image"
                .into(),
        ));
    }

    Ok(())
}

pub fn audit_rv32im_main_recursion_step_spartan_fixed_shape_across_chain(
    relations: &[Rv32imChunkStepIvcRelation],
) -> Result<
    (
        Rv32imMainRecursionStepSpartanCircuitShape,
        Rv32imMainRecursionStepSpartanCircuitShape,
    ),
    SimpleKernelError,
> {
    let (spartan_shape, backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape(relations).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit recursive-step backend relation build failed: {err}"
            ))
        })?;
    let Some(first) = backend_relations.first() else {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit recursive-step fixed-shape contract requires at least one backend relation".into(),
        ));
    };
    let last = backend_relations
        .last()
        .expect("non-empty backend relations must have last");
    let first_shape =
        debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit failed to measure first recursive-step circuit shape: {err}"
            ))
        })?;
    let last_shape =
        debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, last).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV32IM audit failed to measure last recursive-step circuit shape: {err}"
            ))
        })?;
    if first_shape.num_inputs != last_shape.num_inputs
        || first_shape.num_aux != last_shape.num_aux
        || first_shape.num_constraints != last_shape.num_constraints
    {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM audit recursive-step fixed-shape contract failed: first={:?} last={:?}",
            first_shape, last_shape
        )));
    }
    Ok((first_shape, last_shape))
}

fn retag_rv32im_main_recursion_advice_chunk_position(
    template: &Rv32imMainRecursionFPrimeAdvice,
    chunk_count_in: u64,
) -> Result<Rv32imMainRecursionFPrimeAdvice, SimpleKernelError> {
    let mut advice = template.clone();
    *advice.chunk_count_in_mut() = chunk_count_in;
    *advice.chunk_index_mut() = chunk_count_in;
    {
        let handoff = advice.verified_kernel_handoff_mut();
        handoff.bridge_handoff.chunk_index = chunk_count_in;
        handoff.bridge_handoff.digest = handoff.bridge_handoff.expected_digest();
    }
    rv32im_main_recursion_advice_retarget_x_hash_to_current_accumulator(&mut advice);
    if chunk_count_in == 0 {
        let canonical_full_width =
            crate::rv32im::construction2::default::build_rv32im_main_recursion_construction2_canonical_full_width(
                advice.verifier_key_fs(),
                advice.phi_side(),
            )?;
        let canonical_u_perp =
            crate::rv32im::construction2::build_rv32im_main_recursion_construction2_default_fresh_instance(
                advice.verifier_key_fs(),
                canonical_full_width,
            )?;
        let construction2_u_i = advice
            .construction2_input_fresh_instance_mut()
            .ok_or_else(|| {
                SimpleKernelError::Bridge(
                    "RV32IM audit fixed-shape position probe requires a threaded Construction-2 fresh input".into(),
                )
            })?;
        *construction2_u_i = canonical_u_perp;
    }
    Ok(advice)
}

pub fn audit_rv32im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(
    relations: &[Rv32imChunkStepIvcRelation],
    chunk_positions: &[u64],
) -> Result<Vec<(u64, [u8; 32], Rv32imMainRecursionStepSpartanCircuitShape)>, SimpleKernelError> {
    if relations.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit recursive-step fixed-shape position probe requires at least one relation".into(),
        ));
    }
    if chunk_positions.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM audit recursive-step fixed-shape position probe requires at least one chunk position".into(),
        ));
    }

    let template_advice = build_rv32im_main_recursion_f_prime_advices(&relations[..1])?
        .into_iter()
        .next()
        .expect("single relation must yield one recursive-step advice");

    let mut out = Vec::with_capacity(chunk_positions.len());
    for &chunk_count_in in chunk_positions {
        let synthetic_advice = retag_rv32im_main_recursion_advice_chunk_position(&template_advice, chunk_count_in)?;
        let (spartan_shape, backend_relations) =
            build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
                &relations[..1],
                &[synthetic_advice],
            )
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV32IM audit fixed-shape position probe failed to build backend relation at chunk {chunk_count_in}: {err}"
                ))
            })?;
        let backend_relation = backend_relations
            .first()
            .expect("single synthetic advice must yield one backend relation");
        let circuit_shape =
            debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, backend_relation).map_err(
                |err| {
                    SimpleKernelError::Bridge(format!(
                        "RV32IM audit fixed-shape position probe failed to measure chunk {chunk_count_in}: {err}"
                    ))
                },
            )?;
        out.push((chunk_count_in, spartan_shape.expected_digest(), circuit_shape));
    }

    let (_, baseline_shape_digest, baseline_shape) = out
        .first()
        .cloned()
        .expect("non-empty position probe must have baseline");
    for (chunk_count_in, shape_digest, circuit_shape) in out.iter().skip(1) {
        if *shape_digest != baseline_shape_digest
            || circuit_shape.num_inputs != baseline_shape.num_inputs
            || circuit_shape.num_aux != baseline_shape.num_aux
            || circuit_shape.num_constraints != baseline_shape.num_constraints
            || circuit_shape.constraint_fingerprint != baseline_shape.constraint_fingerprint
        {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM audit recursive-step fixed-shape position probe drifted at chunk {chunk_count_in}: baseline_digest={baseline_shape_digest:?} baseline_shape={baseline_shape:?} actual_digest={shape_digest:?} actual_shape={circuit_shape:?}"
            )));
        }
    }

    Ok(out)
}

pub fn audit_build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &crate::rv32im::f_prime::Rv32imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &crate::rv32im::final_relation::Rv32imRecursiveAccumulator,
) -> Result<Rv32imEncodedPublicInput, SimpleKernelError> {
    build_rv32im_main_recursion_x_last_from_accumulator_with_vk_fs(vk_fs, chunk_count, accumulator_final)
}

pub fn rv32im_main_recursion_advice_tamper_chunk_index(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    *advice.chunk_index_mut() ^= 1;
}

pub fn rv32im_main_recursion_backend_relation_tamper_payload_chunk_digest_shell(
    relation: &mut Rv32imMainRecursionFPrimeBackendRelation,
) {
    relation.payload.handoff.public_chunk_digest[0] ^= 1;
    relation.payload.handoff.chunk_relation_digest[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_z_i_first_byte(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    advice.z_i_mut()[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_pc_i(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    *advice.pc_i_mut() = 0;
}

pub fn rv32im_main_recursion_advice_tamper_side_witness_nonzero(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    *advice.side_witness_mut() = Rv32imMainRecursionSideLaneWitness {
        claims: vec![Rv32imMainRecursionSideClaim {
            schema: crate::rv32im::kernel::FamilyEvalSchemaId::Stage1Rows,
            slot: 0,
            point_words: vec![0],
            payload_words: vec![0],
        }],
    };
}

pub fn rv32im_main_recursion_advice_tamper_x_hash_first_byte(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    advice.x_i_mut().bytes_mut()[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_folded_accumulator_input_digest_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.folded_accumulator_in_digest_mut()[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_construction2_input_fresh_instance_x_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice
        .construction2_input_fresh_instance_mut()
        .expect("Construction-2 input fresh instance must be present on native F' advice")
        .x_i_mut()
        .bytes_mut()[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_running_state_terminal_handle_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().carry.terminal_handle.0[0] ^= 1;
    advice.z_i_mut()[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_running_state_terminal_handle_only_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().carry.terminal_handle.0[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_running_state_first_claim_commitment_first_word(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let claim = advice
        .running_state_mut()
        .carry
        .main
        .claims
        .first_mut()
        .expect("native F' advice must carry at least one running CE claim in the current single-slot specialization");
    let first_word = claim
        .c
        .data
        .first_mut()
        .expect("running CE claim commitment must carry at least one word");
    *first_word += neo_math::F::from_u64(1);
}

pub fn rv32im_main_recursion_advice_tamper_running_state_transcript_state_first_field(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().transcript.state[0] += neo_math::F::from_u64(1);
}

pub fn rv32im_main_recursion_advice_tamper_terminal_step(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    let terminal_step = advice.terminal_step_mut();
    *terminal_step = !*terminal_step;
}

pub fn rv32im_main_recursion_advice_tamper_fresh_state_out_terminal_handle_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.fresh_state_out_mut().carry.terminal_handle.0[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_fresh_state_out_transcript_absorbed(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let absorbed = &mut advice.fresh_state_out_mut().transcript.absorbed;
    *absorbed = if *absorbed == 0 { 1 } else { 0 };
}

pub fn rv32im_main_recursion_advice_tamper_prepared_step_digest_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let digest = advice
        .verified_kernel_handoff_mut()
        .prepared_step_digests
        .first_mut()
        .expect("at least one prepared-step digest");
    digest[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_bridge_handoff_digest_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.verified_kernel_handoff_mut().bridge_handoff.digest[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_bridge_binding_digest_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let binding = advice
        .verified_kernel_handoff_mut()
        .bridge_handoff
        .step_bindings
        .first_mut()
        .expect("at least one bridge binding");
    binding.digest[0] ^= 1;
}

pub fn audit_rv32im_main_recursion_construction2_bridge_next_running(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<crate::rv32im::final_relation::Rv32imChunkFoldState, SimpleKernelError> {
    let construction2_u_i = advice.construction2_input_fresh_instance().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV32IM audit bridge-next-running requires the threaded HyperNova Construction-2 input u_i".into(),
        )
    })?;
    let bridge = build_rv32im_main_recursion_construction2_nifs_bridge(advice, construction2_u_i)?;
    Ok(verify_rv32im_main_recursion_construction2_nifs_step(&bridge)?.state)
}

pub fn audit_rv32im_main_recursion_construction2_verified_step_statement_digest(
    relation: &Rv32imChunkStepIvcRelation,
) -> Result<[u8; 32], SimpleKernelError> {
    Ok(build_rv32im_main_recursion_construction2_verified_step_statement_from_relation(relation)?.expected_digest())
}

pub fn audit_rv32im_main_recursion_construction2_pi_rlc_rho_digests(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> Result<Vec<[u8; 32]>, SimpleKernelError> {
    let construction2_u_i = advice.construction2_input_fresh_instance().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV32IM audit Pi_RLC rho digests require the threaded HyperNova Construction-2 input u_i".into(),
        )
    })?;
    let bridge = build_rv32im_main_recursion_construction2_nifs_bridge(advice, construction2_u_i)?;
    let rho_mats = audit_rv32im_main_recursion_construction2_pi_rlc_rho_mats(&bridge)?;
    Ok(rho_mats
        .into_iter()
        .map(|rho| {
            let mut tr = Poseidon2Transcript::new(b"neo.fold.next/tests/rv32im_main_recursion_pi_rlc_rho");
            tr.append_u64s(
                b"neo.fold.next/tests/rv32im_main_recursion_pi_rlc_rho/shape",
                &[rho.rows() as u64, rho.cols() as u64],
            );
            let mut values = Vec::with_capacity(rho.rows() * rho.cols());
            for row in 0..rho.rows() {
                for col in 0..rho.cols() {
                    values.push(rho[(row, col)]);
                }
            }
            tr.append_fields_raw(&values);
            tr.digest32()
        })
        .collect())
}

pub fn audit_rv32im_main_recursion_construction2_pi_fold_debug_dump(
    advice: &Rv32imMainRecursionFPrimeAdvice,
) -> String {
    format!("{:#?}", advice.construction2_pi_fold())
}

pub fn audit_build_rv32im_main_recursion_construction2_fresh_instance_with_explicit_x_i(
    advice: &Rv32imMainRecursionFPrimeAdvice,
    current_input_fresh_instance: &crate::rv32im::Rv32imMainRecursionConstruction2FreshInstance,
    x_i: crate::rv32im::Rv32imEncodedPublicInput,
) -> Result<crate::rv32im::Rv32imMainRecursionConstruction2FreshInstance, SimpleKernelError> {
    build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i(
        advice,
        current_input_fresh_instance,
        x_i,
    )
}

pub fn rv32im_main_recursion_advice_retarget_x_hash_to_current_accumulator(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let rebuilt_x_i = build_rv32im_main_recursion_x_hash_from_advice(advice);
    *advice.x_i_mut() = rebuilt_x_i.clone();
    if let Some(construction2_u_i) = advice.construction2_input_fresh_instance_mut() {
        *construction2_u_i.x_i_mut() = rebuilt_x_i;
    }
}

pub fn rv32im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    advice.verifier_key_fs_mut().main_lane_shape_digest[0] ^= 1;
}

pub fn rv32im_main_recursion_advice_tamper_ccs_replay_first_round_coeff(advice: &mut Rv32imMainRecursionFPrimeAdvice) {
    advice
        .construction2_pi_fold_mut()
        .tamper_ccs_replay_first_round_coeff()
        .expect("Construction-2 Pi_CCS replay payload must carry at least one sumcheck coefficient");
}

pub fn rv32im_main_recursion_advice_tamper_authoritative_ccs_replay_first_round_coeff(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
) {
    let replay_proof = &mut advice.main_circuit_replay_witness_mut().ccs_replay_proof;
    let coeff = if let Some(coeff) = replay_proof
        .sumcheck_rounds
        .first_mut()
        .and_then(|round| round.first_mut())
    {
        coeff
    } else {
        replay_proof
            .sumcheck_rounds_nc
            .first_mut()
            .and_then(|round| round.first_mut())
            .expect("authoritative replay witness must carry at least one sumcheck coefficient")
    };
    *coeff += K::ONE;
}

pub fn rv32im_main_recursion_advice_tamper_dec_child_commitment_first_word(
    advice: &mut Rv32imMainRecursionFPrimeAdvice,
    child_index: usize,
) {
    advice
        .fresh_state_out_mut()
        .carry
        .main
        .claims
        .get_mut(child_index)
        .expect("valid Construction-2 DEC child index")
        .c
        .data[0] += F::ONE;
}
