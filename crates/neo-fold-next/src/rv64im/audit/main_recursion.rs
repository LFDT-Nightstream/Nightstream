//! Owns audit helpers for native F', NIFS, and recursive-step Spartan surfaces.

use crate::rv64im::chunk_step_ivc::Rv64imChunkStepIvcRelation;
use crate::rv64im::construction2::{
    audit_rv64im_main_recursion_construction2_pi_rlc_rho_mats,
    build_rv64im_main_recursion_construction2_input_state_image, build_rv64im_main_recursion_construction2_nifs_bridge,
    build_rv64im_main_recursion_construction2_output_state_image,
    build_rv64im_main_recursion_construction2_verified_step_statement_from_relation,
    verify_rv64im_main_recursion_construction2_nifs_step,
};
use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_backend_statement_from_parts, build_rv64im_main_recursion_base_case_default_slot,
    build_rv64im_main_recursion_x_hash_from_advice,
};
use crate::rv64im::main_relation_trace::build_rv64im_main_circuit_chunk_trace_from_authoritative_parts;
use crate::rv64im::nifs::{
    prove_rv64im_nifs_step, verify_rv64im_nifs_step, Rv64imNifsFreshInstance, Rv64imNifsFreshWitness,
    Rv64imNifsRunningWitness,
};
use crate::rv64im::recursion_spartan::{
    audit_rv64im_main_recursion_final_relation_public_images_match,
    audit_rv64im_main_recursion_terminal_published_target_matches_native_witness,
    build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs, build_rv64im_recursion_proof_from_parts,
    validate_rv64im_main_recursion_public_surface_against_published_statement,
    validate_rv64im_recursion_verifier_key_against_published_statement, Rv64imRecursionProof,
};
use crate::rv64im::SimpleKernelError;
use neo_ccs::{check_ccs_rowwise_zero, check_ce_consistency, CeWitness};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

pub use crate::rv64im::main_recursion::{
    build_rv64im_main_recursion_f_prime_advices, build_rv64im_main_recursion_f_prime_advices_single_step,
    build_rv64im_main_recursion_f_prime_public_output, evaluate_rv64im_main_recursion_f_prime_advice,
    verify_rv64im_main_recursion_f_prime_public_output, Rv64imEncodedPublicInput, Rv64imMainRecursionFPrimeAdvice,
    Rv64imMainRecursionFPrimePublicOutput, Rv64imMainRecursionFPrimeStepImage, Rv64imMainRecursionSideClaim,
    Rv64imMainRecursionSideLaneWitness,
};
pub use crate::rv64im::main_relation_spartan::{
    build_rv64im_main_recursion_f_prime_backend_relations,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv64im_main_recursion_f_prime_claim_cover, build_rv64im_main_recursion_f_prime_payload,
    build_rv64im_main_recursion_f_prime_payloads, build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv64im_main_recursion_step_authoritative_chunk_surface,
    build_rv64im_main_recursion_step_spartan_compressed_chain_shape,
    build_rv64im_main_recursion_step_spartan_published_target, build_rv64im_main_recursion_step_spartan_shape,
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv64im_main_recursion_f_prime_backend_relation_semantics,
    debug_check_rv64im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv64im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv64im_main_recursion_step_spartan_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_parity,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_circuit,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_shape_only_setup,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_statement_binding,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_wrapper_only,
    debug_check_rv64im_main_recursion_step_spartan_embedded_body,
    debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_check_rv64im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv64im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    debug_check_rv64im_main_recursion_step_spartan_shape_only_chain_parity,
    debug_check_rv64im_main_recursion_x_out_gadget_parity,
    debug_compare_rv64im_main_recursion_step_spartan_circuit_shapes,
    debug_compare_rv64im_main_recursion_step_spartan_shape_only_skeleton,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
    debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages,
    prove_rv64im_main_recursion_step_spartan, prove_rv64im_main_recursion_step_spartan_chain,
    prove_rv64im_main_recursion_step_spartan_compressed_chain, setup_rv64im_main_recursion_step_spartan_cached,
    setup_rv64im_main_recursion_step_spartan_shape_cached, validate_rv64im_main_recursion_step_spartan_chain_shape,
    verify_rv64im_main_recursion_step_spartan, verify_rv64im_main_recursion_step_spartan_and_extract_published_target,
    verify_rv64im_main_recursion_step_spartan_chain,
    verify_rv64im_main_recursion_step_spartan_chain_and_extract_published_targets,
    verify_rv64im_main_recursion_step_spartan_compressed_chain,
    verify_rv64im_main_recursion_step_spartan_published_target,
    verify_rv64im_main_recursion_step_spartan_published_target_chain, Rv64imCcsClaimShape, Rv64imCcsWitnessShape,
    Rv64imCeClaimDigestShape, Rv64imChunkStepIvcShape, Rv64imMainRecursionFPrimeBackendRelation,
    Rv64imMainRecursionFPrimeClaimCover, Rv64imMainRecursionFPrimePayload,
    Rv64imMainRecursionStepAuthoritativeChunkSurface, Rv64imMainRecursionStepSpartanChainProof,
    Rv64imMainRecursionStepSpartanCircuitShape, Rv64imMainRecursionStepSpartanCompressedChainProof,
    Rv64imMainRecursionStepSpartanCompressedChainProveMetrics, Rv64imMainRecursionStepSpartanCompressedChainShape,
    Rv64imMainRecursionStepSpartanError, Rv64imMainRecursionStepSpartanKeyPair, Rv64imMainRecursionStepSpartanProof,
    Rv64imMainRecursionStepSpartanProverKey, Rv64imMainRecursionStepSpartanPublishedTarget,
    Rv64imMainRecursionStepSpartanShape, Rv64imMainRecursionStepSpartanStatement,
    Rv64imMainRecursionStepSpartanVerifierKey,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rv64imMainCircuitChunkTraceAuthoritativeSummary {
    pub step_lo: u64,
    pub step_hi: u64,
    pub chunk_relation_digest: [u8; 32],
}

pub fn audit_build_rv64im_main_circuit_chunk_trace_authoritative_summary(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<Rv64imMainCircuitChunkTraceAuthoritativeSummary, SimpleKernelError> {
    let chunk_trace = build_rv64im_main_circuit_chunk_trace_from_authoritative_parts(
        relation.witness.handoff.bridge_handoff.chunk_index as usize,
        &relation.witness.handoff,
        &relation.statement.chunk_summary,
        &relation.witness.state_in.carry,
        &relation.witness.state_out.carry,
        &relation.witness.state_in.transcript,
        &relation.witness.state_out.transcript,
        &relation.witness.replay_witness,
    )?;
    Ok(Rv64imMainCircuitChunkTraceAuthoritativeSummary {
        step_lo: chunk_trace.step_lo(),
        step_hi: chunk_trace.step_hi(),
        chunk_relation_digest: chunk_trace.handoff.chunk_relation_digest,
    })
}

pub fn rv64im_main_recursion_proof_x_last_mut(proof: &mut Rv64imRecursionProof) -> &mut [u8; 32] {
    proof.x_last_bytes_mut()
}

pub fn audit_rv64im_nifs_round_trip_from_chunk_step_relation(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<(), SimpleKernelError> {
    let running = Rv64imNifsRunningWitness {
        state: relation.witness.state_in.clone(),
    };
    let fresh_instance = Rv64imNifsFreshInstance {
        step_public: relation.statement.step_public.clone(),
        chunk_summary: relation.statement.chunk_summary.clone(),
    };
    let fresh_witness = Rv64imNifsFreshWitness {
        handoff: relation.witness.handoff.clone(),
        state_out: relation.witness.state_out.clone(),
    };
    let proof = prove_rv64im_nifs_step(&running, &fresh_instance, &fresh_witness)?;
    let next_running = verify_rv64im_nifs_step(&running, &fresh_instance, &fresh_witness, &proof)?;

    if next_running.state.carry.terminal_handle != relation.witness.state_out.carry.terminal_handle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit NIFS round-trip terminal handle does not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.transcript != relation.witness.state_out.transcript {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit NIFS round-trip transcript does not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.carry.main.claims != relation.witness.state_out.carry.main.claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit NIFS round-trip carried claims do not match the chunk-step relation witness".into(),
        ));
    }
    if next_running.state.carry.main.witnesses != relation.witness.state_out.carry.main.witnesses {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit NIFS round-trip carried witnesses do not match the chunk-step relation witness".into(),
        ));
    }
    Ok(())
}

pub fn audit_rv64im_main_recursion_backend_statement_matches_native_f_prime(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)?;
    let rebuilt_statement = build_rv64im_main_recursion_backend_statement_from_parts(
        step_image.chunk_count(),
        step_image.folded_accumulator_digest(),
        step_image.step_statement_chain_digest(),
        step_image.bridge_handoff_chain_digest(),
        step_image.terminal_handle_digest(),
    )?;
    if rebuilt_statement.x_out != *step_image.x_out()
        || rebuilt_statement.folded_accumulator_digest != step_image.folded_accumulator_digest()
        || rebuilt_statement.step_statement_chain_digest != step_image.step_statement_chain_digest()
        || rebuilt_statement.bridge_handoff_chain_digest != step_image.bridge_handoff_chain_digest()
        || rebuilt_statement.terminal_handle_digest != step_image.terminal_handle_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit main-recursion backend statement does not match the native F' step image".into(),
        ));
    }
    Ok(())
}

pub fn audit_rv64im_main_recursion_default_slot_satisfies_r1_literally(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    let carried_main = &advice.running_state().carry.main;
    if carried_main.claims.len() != 1 || carried_main.witnesses.len() != 1 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit default-slot R1 check requires exactly one carried CE slot in the current single-PC specialization"
                .into(),
        ));
    }
    audit_rv64im_main_recursion_default_slot_claim_and_witness_satisfy_r1(
        &carried_main.claims[0],
        &carried_main.witnesses[0],
        "carried base-case slot",
    )
}

pub fn audit_rv64im_main_recursion_canonical_default_slot_satisfies_r1_literally() -> Result<(), SimpleKernelError> {
    let canonical_state = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state();
    let default_slot = build_rv64im_main_recursion_base_case_default_slot(&canonical_state)?;
    audit_rv64im_main_recursion_default_slot_claim_and_witness_satisfy_r1(
        default_slot.claim(),
        default_slot.witness(),
        "canonical default slot",
    )
}

fn audit_rv64im_main_recursion_default_slot_claim_and_witness_satisfy_r1(
    claim: &neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
    witness: &neo_ccs::Mat<neo_math::F>,
    label: &str,
) -> Result<(), SimpleKernelError> {
    let (params, log, structure) = crate::rv64im::kernel::rv64im_cached_root_main_lane_context()?;
    let zero_x = vec![neo_math::F::ZERO; claim.m_in];
    let zero_w = vec![neo_math::F::ZERO; structure.m.saturating_sub(claim.m_in)];
    check_ccs_rowwise_zero(structure, &zero_x, &zero_w).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM audit default-slot R1 check failed CCS row-wise zero for the {label}: {err}"
        ))
    })?;
    check_ce_consistency(params, structure, log, claim, &CeWitness { Z: witness.clone() }).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM audit default-slot R1 check failed CE consistency for the {label}: {err}"
        ))
    })?;
    if witness
        .as_slice()
        .iter()
        .any(|value| *value != neo_math::F::ZERO)
    {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM audit default-slot R1 check requires the {label} witness to be the canonical zero witness"
        )));
    }
    Ok(())
}

pub fn audit_rv64im_main_recursion_construction2_state_images_match_native_f_prime(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<(), SimpleKernelError> {
    let input_state_image = build_rv64im_main_recursion_construction2_input_state_image(advice);
    let input_x_i = input_state_image.encoded_public_input();
    if input_x_i != *advice.x_i() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit Construction-2 input state image did not encode the carried native x_i".into(),
        ));
    }

    let Some(construction2_input_u_i) = advice.construction2_input_fresh_instance() else {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit Construction-2 state-image parity requires a threaded input fresh instance u_i".into(),
        ));
    };
    if construction2_input_u_i.x_i() != &input_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit Construction-2 input fresh instance x_i drifted from the canonical input state image".into(),
        ));
    }

    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)?;
    let output_state_image = build_rv64im_main_recursion_construction2_output_state_image(advice)?;
    let output_x_i = output_state_image.encoded_public_input();
    if output_x_i != *step_image.x_out() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit Construction-2 output state image did not encode the native F' x_{i+1}".into(),
        ));
    }
    if step_image.construction2_u_next().x_i() != &output_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit Construction-2 output fresh instance x_i drifted from the canonical output state image"
                .into(),
        ));
    }

    Ok(())
}

pub fn audit_rv64im_main_recursion_step_spartan_published_target_matches_construction2_state_images(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let advice = &backend_relation.f_prime_advice;
    audit_rv64im_main_recursion_construction2_state_images_match_native_f_prime(advice)?;

    let published_target =
        build_rv64im_main_recursion_step_spartan_published_target(backend_relation).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM audit failed to build recursive-step published target from backend relation: {err}"
            ))
        })?;
    let step_image = evaluate_rv64im_main_recursion_f_prime_advice(advice)?;
    let output_state_image = build_rv64im_main_recursion_construction2_output_state_image(advice)?;
    let output_x_i = output_state_image.encoded_public_input();

    if published_target.chunk_index != advice.chunk_index() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target chunk_index drifted from the authoritative verified step"
                .into(),
        ));
    }
    if published_target.folded_accumulator_in_digest != advice.folded_accumulator_in_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target folded-accumulator input drifted from the carried native input state"
                .into(),
        ));
    }
    if published_target.step_statement_chain_digest_in != advice.step_statement_chain_digest_in() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target step-statement chain input drifted from the carried native input state"
                .into(),
        ));
    }
    if published_target.bridge_handoff_chain_digest_in != advice.bridge_handoff_chain_digest_in() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target bridge-handoff chain input drifted from the carried native input state"
                .into(),
        ));
    }
    if published_target.x_out != output_x_i {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target x_{i+1} drifted from the canonical Construction-2 output state image"
                .into(),
        ));
    }
    if published_target.folded_accumulator_out_digest != step_image.folded_accumulator_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target folded-accumulator output drifted from the native F' image"
                .into(),
        ));
    }
    if published_target.step_statement_chain_digest != step_image.step_statement_chain_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target step-statement chain output drifted from the native F' image"
                .into(),
        ));
    }
    if published_target.bridge_handoff_chain_digest != step_image.bridge_handoff_chain_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target bridge-handoff chain output drifted from the native F' image"
                .into(),
        ));
    }
    if published_target.terminal_handle_digest != step_image.terminal_handle_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step published target terminal handle drifted from the native F' image".into(),
        ));
    }

    Ok(())
}

pub fn audit_rv64im_main_recursion_step_spartan_fixed_shape_across_chain(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<
    (
        Rv64imMainRecursionStepSpartanCircuitShape,
        Rv64imMainRecursionStepSpartanCircuitShape,
    ),
    SimpleKernelError,
> {
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(relations).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM audit recursive-step backend relation build failed: {err}"
            ))
        })?;
    let Some(first) = backend_relations.first() else {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step fixed-shape contract requires at least one backend relation".into(),
        ));
    };
    let last = backend_relations
        .last()
        .expect("non-empty backend relations must have last");
    let first_shape =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM audit failed to measure first recursive-step circuit shape: {err}"
            ))
        })?;
    let last_shape =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, last).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM audit failed to measure last recursive-step circuit shape: {err}"
            ))
        })?;
    if first_shape.num_inputs != last_shape.num_inputs
        || first_shape.num_aux != last_shape.num_aux
        || first_shape.num_constraints != last_shape.num_constraints
    {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM audit recursive-step fixed-shape contract failed: first={:?} last={:?}",
            first_shape, last_shape
        )));
    }
    if let Some(delta) = debug_compare_rv64im_main_recursion_step_spartan_circuit_shapes(&spartan_shape, first, last)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM audit failed to compare first/last recursive-step circuits: {err}"
            ))
        })?
    {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM audit recursive-step fixed-shape skeleton drifted across the chain: {delta}"
        )));
    }
    Ok((first_shape, last_shape))
}

fn retag_rv64im_main_recursion_advice_chunk_position(
    template: &Rv64imMainRecursionFPrimeAdvice,
    chunk_count_in: u64,
) -> Rv64imMainRecursionFPrimeAdvice {
    let mut advice = template.clone();
    *advice.chunk_count_in_mut() = chunk_count_in;
    *advice.chunk_index_mut() = chunk_count_in;
    {
        let handoff = advice.verified_kernel_handoff_mut();
        handoff.bridge_handoff.chunk_index = chunk_count_in;
        handoff.bridge_handoff.digest = handoff.bridge_handoff.expected_digest();
    }
    rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator(&mut advice);
    advice
}

pub fn audit_rv64im_main_recursion_step_spartan_fixed_shape_at_chunk_positions(
    relations: &[Rv64imChunkStepIvcRelation],
    chunk_positions: &[u64],
) -> Result<Vec<(u64, [u8; 32], Rv64imMainRecursionStepSpartanCircuitShape)>, SimpleKernelError> {
    if relations.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step fixed-shape position probe requires at least one relation".into(),
        ));
    }
    if chunk_positions.is_empty() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM audit recursive-step fixed-shape position probe requires at least one chunk position".into(),
        ));
    }

    let template_advice = build_rv64im_main_recursion_f_prime_advices(&relations[..1])?
        .into_iter()
        .next()
        .expect("single relation must yield one recursive-step advice");

    let mut out = Vec::with_capacity(chunk_positions.len());
    for &chunk_count_in in chunk_positions {
        let synthetic_advice = retag_rv64im_main_recursion_advice_chunk_position(&template_advice, chunk_count_in);
        let (spartan_shape, backend_relations) =
            build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
                &relations[..1],
                &[synthetic_advice],
            )
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM audit fixed-shape position probe failed to build backend relation at chunk {chunk_count_in}: {err}"
                ))
            })?;
        let backend_relation = backend_relations
            .first()
            .expect("single synthetic advice must yield one backend relation");
        let circuit_shape =
            debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, backend_relation).map_err(
                |err| {
                    SimpleKernelError::Bridge(format!(
                        "RV64IM audit fixed-shape position probe failed to measure chunk {chunk_count_in}: {err}"
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
                "RV64IM audit recursive-step fixed-shape position probe drifted at chunk {chunk_count_in}: baseline_digest={baseline_shape_digest:?} baseline_shape={baseline_shape:?} actual_digest={shape_digest:?} actual_shape={circuit_shape:?}"
            )));
        }
    }

    Ok(out)
}

pub fn audit_build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
    vk_fs: &crate::rv64im::main_recursion::Rv64imVerifierKeyFs,
    chunk_count: u64,
    accumulator_final: &crate::rv64im::final_relation::Rv64imRecursiveAccumulator,
    step_statement_chain_digest: [u8; 32],
    bridge_handoff_chain_digest: [u8; 32],
) -> Result<Rv64imEncodedPublicInput, SimpleKernelError> {
    build_rv64im_main_recursion_x_last_from_accumulator_with_vk_fs(
        vk_fs,
        chunk_count,
        accumulator_final,
        step_statement_chain_digest,
        bridge_handoff_chain_digest,
    )
}

pub fn build_rv64im_main_recursion_proof_surface_stub_from_relations(
    relations: &[Rv64imChunkStepIvcRelation],
) -> Result<Rv64imRecursionProof, SimpleKernelError> {
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(relations).map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM audit recursion proof surface stub build failed: {err}"))
        })?;
    validate_rv64im_main_recursion_step_spartan_chain_shape(&spartan_shape, &backend_relations).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM audit recursion proof surface stub shared step-shape validation failed: {err}"
        ))
    })?;
    let chain_proof = crate::rv64im::recursion_spartan::audit_empty_step_proof_chain();
    build_rv64im_recursion_proof_from_parts(spartan_shape, chain_proof)
}

pub fn audit_rv64im_main_recursion_proof_matches_published_statement(
    published_statement: &crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement,
    proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    let _ = validate_rv64im_main_recursion_public_surface_against_published_statement(published_statement, proof)?;
    Ok(())
}

pub fn audit_rv64im_main_recursion_final_relation_public_images_match_against_published_statement(
    published_statement: &crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement,
    accumulator_witness: &crate::rv64im::recursion_spartan::Rv64imMainRecursionAccumulatorWitness,
    proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    audit_rv64im_main_recursion_final_relation_public_images_match(published_statement, accumulator_witness, proof)
}

pub fn audit_rv64im_main_recursion_terminal_published_target_matches_native_witness_against_published_statement(
    published_statement: &crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement,
    accumulator_witness: &crate::rv64im::recursion_spartan::Rv64imMainRecursionAccumulatorWitness,
    proof: &Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    audit_rv64im_main_recursion_terminal_published_target_matches_native_witness(
        published_statement,
        accumulator_witness,
        proof,
    )
}

pub fn audit_rv64im_recursion_verifier_key_matches_published_statement(
    recursion_vk: &crate::rv64im::recursion_spartan::Rv64imRecursionVerifierKey,
    published_statement: &crate::rv64im::main_proof::Rv64imAccumulatorPublicStatement,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_recursion_verifier_key_against_published_statement(recursion_vk, published_statement)
}

pub fn rv64im_main_recursion_proof_first_step_snark_bytes_mut(proof: &mut Rv64imRecursionProof) -> &mut Vec<u8> {
    proof.first_step_proof_snark_bytes_mut()
}

pub fn rv64im_main_recursion_accumulator_witness_final_fold_witness_mut(
    witness: &mut crate::rv64im::recursion_spartan::Rv64imMainRecursionAccumulatorWitness,
) -> &mut crate::chunk_relation::ChunkReplayWitness {
    witness.final_fold_witness_mut()
}

pub fn rv64im_main_recursion_accumulator_witness_running_final_mut(
    witness: &mut crate::rv64im::recursion_spartan::Rv64imMainRecursionAccumulatorWitness,
) -> &mut crate::rv64im::chunk_fold_step::Rv64imChunkFoldCarry {
    witness.running_final_mut()
}

pub fn rv64im_main_recursion_proof_pop_last_step_proof(
    proof: &mut Rv64imRecursionProof,
) -> Result<(), SimpleKernelError> {
    proof.pop_last_step_proof()
}

pub fn rv64im_main_recursion_advice_tamper_chunk_index(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    *advice.chunk_index_mut() ^= 1;
}

pub fn rv64im_main_recursion_backend_relation_tamper_payload_chunk_digest_shell(
    relation: &mut Rv64imMainRecursionFPrimeBackendRelation,
) {
    relation.payload.handoff.public_chunk_digest[0] ^= 1;
    relation.payload.handoff.chunk_relation_digest[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_z_i_first_byte(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    advice.z_i_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_pc_i(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    *advice.pc_i_mut() = 0;
}

pub fn rv64im_main_recursion_advice_tamper_side_witness_nonzero(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    *advice.side_witness_mut() = Rv64imMainRecursionSideLaneWitness {
        claims: vec![Rv64imMainRecursionSideClaim {
            schema: crate::rv64im::kernel::FamilyEvalSchemaId::Stage1Rows,
            slot: 0,
            point_words: vec![0],
            payload_words: vec![0],
        }],
    };
}

pub fn rv64im_main_recursion_advice_tamper_x_hash_first_byte(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    advice.x_i_mut().bytes_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_construction2_input_fresh_instance_x_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice
        .construction2_input_fresh_instance_mut()
        .expect("Construction-2 input fresh instance must be present on native F' advice")
        .x_i_mut()
        .bytes_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_step_statement_chain_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.step_statement_chain_digest_in_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_running_state_terminal_handle_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().carry.terminal_handle.0[0] ^= 1;
    advice.z_i_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_running_state_terminal_handle_only_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().carry.terminal_handle.0[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_running_state_first_claim_commitment_first_word(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
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

pub fn rv64im_main_recursion_advice_tamper_running_state_transcript_state_first_field(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.running_state_mut().transcript.state[0] += neo_math::F::from_u64(1);
}

pub fn rv64im_main_recursion_advice_tamper_bridge_handoff_chain_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.bridge_handoff_chain_digest_in_mut()[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_terminal_step(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    let terminal_step = advice.terminal_step_mut();
    *terminal_step = !*terminal_step;
}

pub fn rv64im_main_recursion_advice_tamper_fresh_state_out_terminal_handle_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.fresh_state_out_mut().carry.terminal_handle.0[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_fresh_state_out_transcript_absorbed(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    let absorbed = &mut advice.fresh_state_out_mut().transcript.absorbed;
    *absorbed = if *absorbed == 0 { 1 } else { 0 };
}

pub fn rv64im_main_recursion_advice_tamper_legacy_prepared_step_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    let digest = advice
        .verified_kernel_handoff_mut()
        .prepared_step_digests
        .first_mut()
        .expect("at least one prepared-step digest");
    digest[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_legacy_bridge_handoff_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.verified_kernel_handoff_mut().bridge_handoff.digest[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_legacy_bridge_binding_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    let binding = advice
        .verified_kernel_handoff_mut()
        .bridge_handoff
        .step_bindings
        .first_mut()
        .expect("at least one bridge binding");
    binding.digest[0] ^= 1;
}

pub fn audit_rv64im_main_recursion_construction2_bridge_next_running(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<crate::rv64im::final_relation::Rv64imChunkFoldState, SimpleKernelError> {
    let construction2_u_i = advice.construction2_input_fresh_instance().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV64IM audit bridge-next-running requires the threaded HyperNova Construction-2 input u_i".into(),
        )
    })?;
    let bridge = build_rv64im_main_recursion_construction2_nifs_bridge(advice, construction2_u_i)?;
    Ok(verify_rv64im_main_recursion_construction2_nifs_step(&bridge)?.state)
}

pub fn audit_rv64im_main_recursion_construction2_verified_step_statement_digest(
    relation: &Rv64imChunkStepIvcRelation,
) -> Result<[u8; 32], SimpleKernelError> {
    Ok(build_rv64im_main_recursion_construction2_verified_step_statement_from_relation(relation)?.expected_digest())
}

pub fn audit_rv64im_main_recursion_construction2_pi_rlc_rho_digests(
    advice: &Rv64imMainRecursionFPrimeAdvice,
) -> Result<Vec<[u8; 32]>, SimpleKernelError> {
    let construction2_u_i = advice.construction2_input_fresh_instance().ok_or_else(|| {
        SimpleKernelError::Bridge(
            "RV64IM audit Pi_RLC rho digests require the threaded HyperNova Construction-2 input u_i".into(),
        )
    })?;
    let bridge = build_rv64im_main_recursion_construction2_nifs_bridge(advice, construction2_u_i)?;
    let rho_mats = audit_rv64im_main_recursion_construction2_pi_rlc_rho_mats(&bridge)?;
    Ok(rho_mats
        .into_iter()
        .map(|rho| {
            let mut tr = Poseidon2Transcript::new(b"neo.fold.next/tests/rv64im_main_recursion_pi_rlc_rho");
            tr.append_u64s(
                b"neo.fold.next/tests/rv64im_main_recursion_pi_rlc_rho/shape",
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

pub fn rv64im_main_recursion_advice_retarget_x_hash_to_current_accumulator(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    let rebuilt_x_i = build_rv64im_main_recursion_x_hash_from_advice(advice);
    *advice.x_i_mut() = rebuilt_x_i.clone();
    if let Some(construction2_u_i) = advice.construction2_input_fresh_instance_mut() {
        *construction2_u_i.x_i_mut() = rebuilt_x_i;
    }
}

pub fn rv64im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
) {
    advice.verifier_key_fs_mut().main_lane_shape_digest[0] ^= 1;
}

pub fn rv64im_main_recursion_advice_tamper_ccs_replay_first_round_coeff(advice: &mut Rv64imMainRecursionFPrimeAdvice) {
    advice
        .construction2_pi_fold_mut()
        .tamper_ccs_replay_first_round_coeff()
        .expect("Construction-2 Pi_CCS replay payload must carry at least one sumcheck coefficient");
}

pub fn rv64im_main_recursion_advice_tamper_dec_child_commitment_first_word(
    advice: &mut Rv64imMainRecursionFPrimeAdvice,
    child_index: usize,
) {
    advice
        .construction2_pi_fold_mut()
        .tamper_dec_child_commitment_first_word(child_index)
        .expect("valid Construction-2 DEC child index");
}
