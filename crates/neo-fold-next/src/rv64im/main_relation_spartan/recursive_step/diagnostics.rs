use std::time::Instant;

use bellpepper_core::test_cs::TestConstraintSystem;
use neo_reductions::engines::utils::me_digest_poseidon_into;
use spartan2::traits::circuit::SpartanCircuit;
use spartan2::{
    bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS},
    provider::goldi::F as SpartanF,
    SplitR1CSShape,
};

use super::*;
use crate::rv64im::main_relation_circuit::claim::me_digest_poseidon;
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::main_relation_spartan::chunk_step_recursive::rv64im_chunk_step_recursive_carry_state_digest;
use crate::rv64im::main_relation_spartan::recursive_cover::{
    alloc_recursive_cover_claims, alloc_recursive_cover_state,
};
use crate::rv64im::main_relation_spartan::Rv64imClaimBundle;

fn stage_err(stage: &str, err: impl ToString) -> Rv64imMainRecursionStepSpartanError {
    Rv64imMainRecursionStepSpartanError::Prepare(format!("{stage}: {}", err.to_string()))
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
    println!("n2-step-chunk|start|state_in");
    let started = Instant::now();
    let state_in_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_in"),
        &payload.state_in_claims,
        &witness.running_state().transcript,
        witness.running_state().carry.terminal_handle.0,
        "state_in",
    )
    .map_err(|err| stage_err("state_in", err))?;
    println!(
        "n2-step-chunk|done|state_in|{:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );
    println!("n2-step-chunk|start|state_out");
    let started = Instant::now();
    let _state_out_var = alloc_recursive_cover_state(
        &mut cs.namespace(|| "state_out"),
        &payload.state_out_claims,
        &payload.fixed_transcript_out,
        witness.fresh_state_out().carry.terminal_handle.0,
        "state_out",
    )
    .map_err(|err| stage_err("state_out", err))?;
    println!(
        "n2-step-chunk|done|state_out|{:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );
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
    let carry_state_in_digest = rv64im_chunk_step_recursive_carry_state_digest(
        &relation.payload.state_in_claims,
        &relation.f_prime_advice.running_state().transcript,
        relation
            .f_prime_advice
            .running_state()
            .carry
            .terminal_handle
            .0,
    );
    let carry_state_out_digest = rv64im_chunk_step_recursive_carry_state_digest(
        &relation.payload.state_out_claims,
        &relation.payload.fixed_transcript_out,
        relation
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
    );
    let chunk_index = alloc_const_u64(
        cs,
        &format!("wrapper_step_{step_index}_chunk_index"),
        relation.f_prime_advice.chunk_index(),
    )?;
    let carry_state_in_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_carry_state_in_digest")),
        carry_state_in_digest,
        &format!("wrapper_step_{step_index}_carry_state_in_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_carry_state_in_digest", err))?;
    let step_statement_chain_digest_in = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_step_statement_chain_digest_in")),
        relation.f_prime_advice.step_statement_chain_digest_in(),
        &format!("wrapper_step_{step_index}_step_statement_chain_digest_in"),
    )
    .map_err(|err| stage_err("wrapper_step_step_statement_chain_digest_in", err))?;
    let step_statement_chain_digest_out = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_step_statement_chain_digest_out")),
        statement.step_statement_chain_digest,
        &format!("wrapper_step_{step_index}_step_statement_chain_digest_out"),
    )
    .map_err(|err| stage_err("wrapper_step_step_statement_chain_digest_out", err))?;
    let bridge_handoff_chain_digest_in = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_bridge_handoff_chain_digest_in")),
        relation.f_prime_advice.bridge_handoff_chain_digest_in(),
        &format!("wrapper_step_{step_index}_bridge_handoff_chain_digest_in"),
    )
    .map_err(|err| stage_err("wrapper_step_bridge_handoff_chain_digest_in", err))?;
    let bridge_handoff_chain_digest_out = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_bridge_handoff_chain_digest_out")),
        statement.bridge_handoff_chain_digest,
        &format!("wrapper_step_{step_index}_bridge_handoff_chain_digest_out"),
    )
    .map_err(|err| stage_err("wrapper_step_bridge_handoff_chain_digest_out", err))?;
    let carry_state_out_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_carry_state_out_digest")),
        carry_state_out_digest,
        &format!("wrapper_step_{step_index}_carry_state_out_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_carry_state_out_digest", err))?;
    let folded_accumulator_out_digest = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_folded_accumulator_out_digest")),
        statement.folded_accumulator_digest,
        &format!("wrapper_step_{step_index}_folded_accumulator_out_digest"),
    )
    .map_err(|err| stage_err("wrapper_step_folded_accumulator_out_digest", err))?;
    let terminal_handle_digest_out = digest_const_inputs(
        &mut cs.namespace(|| format!("wrapper_step_{step_index}_terminal_handle_digest_out")),
        statement.terminal_handle_digest,
        &format!("wrapper_step_{step_index}_terminal_handle_digest_out"),
    )
    .map_err(|err| stage_err("wrapper_step_terminal_handle_digest_out", err))?;
    Ok((
        Rv64imMainRecursionStepPublicVar {
            chunk_index,
            carry_state_in_digest,
            step_statement_chain_digest_in,
            step_statement_chain_digest_out,
            bridge_handoff_chain_digest_in,
            bridge_handoff_chain_digest_out,
            carry_state_out_digest,
            folded_accumulator_out_digest,
            terminal_handle_digest_out,
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
    let step_statement_chain_out_input = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_step_statement_chain_out"),
        statement.step_statement_chain_digest,
        "wrapper_step_statement_chain_out",
    )
    .map_err(|err| stage_err("wrapper_step_statement_chain_out", err))?;
    let bridge_handoff_chain_out_input = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_bridge_handoff_chain_out"),
        statement.bridge_handoff_chain_digest,
        "wrapper_bridge_handoff_chain_out",
    )
    .map_err(|err| stage_err("wrapper_bridge_handoff_chain_out", err))?;
    let terminal_handle_out_input = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_terminal_handle_out"),
        statement.terminal_handle_digest,
        "wrapper_terminal_handle_out",
    )
    .map_err(|err| stage_err("wrapper_terminal_handle_out", err))?;

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
    let initial_folded_accumulator_digest = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_folded_accumulator_digest"),
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry),
        "wrapper_initial_folded_accumulator_digest",
    )
    .map_err(|err| stage_err("wrapper_initial_folded_accumulator_digest", err))?;
    let initial_step_statement_chain = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_step_statement_chain_seed"),
        crate::rv64im::chunk_step_ivc::rv64im_step_statement_chain_digest_init(),
        "wrapper_initial_step_statement_chain_seed",
    )
    .map_err(|err| stage_err("wrapper_initial_step_statement_chain", err))?;
    let initial_bridge_handoff_chain = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_bridge_handoff_chain_seed"),
        crate::rv64im::chunk_step_ivc::rv64im_bridge_handoff_chain_digest_init(),
        "wrapper_initial_bridge_handoff_chain_seed",
    )
    .map_err(|err| stage_err("wrapper_initial_bridge_handoff_chain", err))?;
    let initial_terminal_handle = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_initial_terminal_handle"),
        initial_state.carry.terminal_handle.0,
        "wrapper_initial_terminal_handle",
    )
    .map_err(|err| stage_err("wrapper_initial_terminal_handle", err))?;

    let mut final_folded_accumulator_digest_value = digest32_as_spartan_fields(
        crate::rv64im::final_relation::rv64im_chunk_fold_carry_recursive_accumulator_digest(&initial_state.carry),
    );
    let mut final_terminal_handle_value = digest32_as_spartan_fields(initial_state.carry.terminal_handle.0);

    let mut previous_step: Option<Rv64imMainRecursionStepPublicVar> = None;
    for (step_index, relation) in backend_relations.iter().enumerate() {
        let (step_public, statement) = alloc_wrapper_step_public_var(&mut cs, step_index, relation)?;
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
                &mut cs.namespace(|| format!("wrapper_step_{step_index}_statement_chain")),
                &previous.step_statement_chain_digest_out,
                &step_public.step_statement_chain_digest_in,
                &format!("wrapper_step_{step_index}_statement_chain"),
            )
            .map_err(|err| stage_err("wrapper_statement_chain", err))?;
            enforce_digest_eq(
                &mut cs.namespace(|| format!("wrapper_step_{step_index}_bridge_chain")),
                &previous.bridge_handoff_chain_digest_out,
                &step_public.bridge_handoff_chain_digest_in,
                &format!("wrapper_step_{step_index}_bridge_chain"),
            )
            .map_err(|err| stage_err("wrapper_bridge_chain", err))?;
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
                &mut cs.namespace(|| "wrapper_initial_step_statement_chain"),
                &initial_step_statement_chain,
                &step_public.step_statement_chain_digest_in,
                "wrapper_initial_step_statement_chain",
            )
            .map_err(|err| stage_err("wrapper_initial_step_statement_chain", err))?;
            enforce_digest_eq(
                &mut cs.namespace(|| "wrapper_initial_bridge_handoff_chain"),
                &initial_bridge_handoff_chain,
                &step_public.bridge_handoff_chain_digest_in,
                "wrapper_initial_bridge_handoff_chain",
            )
            .map_err(|err| stage_err("wrapper_initial_bridge_handoff_chain", err))?;
        }

        final_folded_accumulator_digest_value = digest32_as_spartan_fields(statement.folded_accumulator_digest);
        final_terminal_handle_value = digest32_as_spartan_fields(statement.terminal_handle_digest);
        previous_step = Some(step_public);
    }

    let final_folded_accumulator_digest = previous_step
        .as_ref()
        .map(|step| step.folded_accumulator_out_digest.clone())
        .unwrap_or(initial_folded_accumulator_digest);
    let final_step_statement_chain = previous_step
        .as_ref()
        .map(|step| step.step_statement_chain_digest_out.clone())
        .unwrap_or(initial_step_statement_chain);
    let final_bridge_handoff_chain = previous_step
        .as_ref()
        .map(|step| step.bridge_handoff_chain_digest_out.clone())
        .unwrap_or(initial_bridge_handoff_chain);
    let final_terminal_handle = previous_step
        .as_ref()
        .map(|step| step.terminal_handle_digest_out.clone())
        .unwrap_or(initial_terminal_handle);
    let initial_z = crate::rv64im::chunk_step_ivc::rv64im_chunk_step_ivc_initial_state()
        .carry
        .terminal_handle
        .0;
    let final_z_0 = digest_const_inputs(
        &mut cs.namespace(|| "wrapper_final_z_0"),
        initial_z,
        "wrapper_final_z_0",
    )
    .map_err(|err| stage_err("wrapper_final_z_0", err))?;
    let final_x_out = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "wrapper_final_x_out"),
        "wrapper_final_x_out",
        chain_shape.step_shapes.len() as u64,
        &final_z_0,
        &digest32_as_spartan_fields(initial_z),
        &final_terminal_handle,
        &final_terminal_handle_value,
        crate::rv64im::main_recursion::RV64IM_MAIN_RECURSION_TRIVIAL_PC,
        &final_folded_accumulator_digest,
        &final_folded_accumulator_digest_value,
    )
    .map_err(|err| stage_err("wrapper_final_x_out", err))?;
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
    enforce_digest_eq(
        &mut cs.namespace(|| "wrapper_step_statement_chain_output_eq"),
        &step_statement_chain_out_input,
        &final_step_statement_chain,
        "wrapper_step_statement_chain_output_eq",
    )
    .map_err(|err| stage_err("wrapper_step_statement_chain_output_eq", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "wrapper_bridge_handoff_chain_output_eq"),
        &bridge_handoff_chain_out_input,
        &final_bridge_handoff_chain,
        "wrapper_bridge_handoff_chain_output_eq",
    )
    .map_err(|err| stage_err("wrapper_bridge_handoff_chain_output_eq", err))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "wrapper_terminal_handle_output_eq"),
        &terminal_handle_out_input,
        &final_terminal_handle,
        "wrapper_terminal_handle_output_eq",
    )
    .map_err(|err| stage_err("wrapper_terminal_handle_output_eq", err))?;

    ensure_stage_satisfied(&cs, "compressed_chain_wrapper_only")
}
