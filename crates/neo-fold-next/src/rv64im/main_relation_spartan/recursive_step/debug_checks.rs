use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_reductions::engines::utils::build_dims_and_policy;

use super::*;
use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::kernel::rv64im_root_main_lane_context_for_step_cap;
use crate::rv64im::main_relation_spartan::fingerprint_cs::FingerprintCS;

fn ensure_debug_cs_satisfied(
    cs: &TestConstraintSystem<SpartanF>,
    stage: &str,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(Rv64imMainRecursionStepSpartanError::Prepare(format!(
            "{stage}: {}",
            cs.which_is_unsatisfied().unwrap_or("unknown constraint")
        )))
    }
}

pub fn debug_check_rv64im_main_recursion_step_spartan_circuit(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied recursive-step constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_embedded_body(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let relation_public_inputs = alloc_private_field_values(
        &mut cs.namespace(|| "embedded_public_inputs"),
        &circuit.expected_public_values(),
        "embedded_public_inputs",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let mut relation_public_cursor = 0usize;
    synthesize_rv64im_main_recursion_step_body(
        &circuit,
        &mut cs.namespace(|| "embedded_body"),
        &relation_public_inputs,
        &mut relation_public_cursor,
        None,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if relation_public_cursor != relation_public_inputs.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion embedded step body did not consume all expected public values".into(),
        ));
    }
    if !cs.is_satisfied() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            cs.which_is_unsatisfied()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "unknown unsatisfied embedded recursive-step constraint".to_string()),
        ));
    }
    Ok(())
}

pub fn debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(
    spartan_shape: &Rv64imMainRecursionStepSpartanShape,
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imMainRecursionStepSpartanCircuitShape, Rv64imMainRecursionStepSpartanError> {
    let circuit = build_rv64im_main_recursion_step_circuit(spartan_shape, backend_relation)?;
    let mut cs = FingerprintCS::new();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let num_inputs = cs.public_input_count(circuit.num_challenges());
    let num_aux = cs.num_aux();
    let num_constraints = cs.num_constraints();
    let shape_digest = cs.finish_digest32(circuit.num_challenges());
    Ok(Rv64imMainRecursionStepSpartanCircuitShape {
        num_inputs,
        num_aux,
        num_constraints,
        constraint_fingerprint: format_spartan_digest_hex(shape_digest),
    })
}

pub fn debug_check_rv64im_main_recursion_step_spartan_inactive_side_lane_constraints(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    enforce_inactive_side_lane_constraints(
        &mut cs.namespace(|| "inactive_side_lane"),
        "inactive_side_lane",
        backend_relation.f_prime_advice.side_witness().claim_count(),
        backend_relation.payload.phi_side_commitment_words.len() as u64,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    ensure_debug_cs_satisfied(&cs, "inactive_side_lane")
}

pub fn debug_check_rv64im_main_recursion_x_out_gadget_parity(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    ensure_main_recursion_step_spartan_statement_binding(backend_relation)?;
    let statement = &backend_relation.spartan_statement;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let folded_accumulator_digest = digest_const_inputs(
        &mut cs.namespace(|| "folded_accumulator_digest"),
        statement.folded_accumulator_digest,
        "folded_accumulator_digest",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let z_0 = digest_const_inputs(&mut cs.namespace(|| "z_0"), *backend_relation.payload.z_0(), "z_0")
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let z_next = digest_const_inputs(
        &mut cs.namespace(|| "z_next"),
        *backend_relation.payload.z_next(),
        "z_next",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let expected_x_out = digest_const_inputs(
        &mut cs.namespace(|| "expected_x_out"),
        statement.x_out.bytes(),
        "expected_x_out",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let chunk_count = backend_relation.f_prime_advice.chunk_count_in() + 1;
    let chunk_count_halves = private_u64_halves(
        &mut cs.namespace(|| "chunk_count_halves"),
        chunk_count,
        "chunk_count_halves",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let pc_next_halves = private_u64_halves(
        &mut cs.namespace(|| "pc_next_halves"),
        backend_relation.payload.pc_next(),
        "pc_next_halves",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let x_out_digest = main_recursion_x_out_circuit(
        &mut cs.namespace(|| "x_out_digest"),
        "x_out_digest",
        backend_relation
            .f_prime_advice
            .verifier_key_fs()
            .expected_digest(),
        &chunk_count_halves,
        &u64_halves_as_spartan_fields(chunk_count),
        &z_0,
        &digest32_as_spartan_fields(*backend_relation.payload.z_0()),
        &z_next,
        &digest32_as_spartan_fields(*backend_relation.payload.z_next()),
        &pc_next_halves,
        &u64_halves_as_spartan_fields(backend_relation.payload.pc_next()),
        &folded_accumulator_digest,
        &digest32_as_spartan_fields(statement.folded_accumulator_digest),
        None,
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "x_out_eq"),
        &x_out_digest,
        &expected_x_out,
        "x_out_eq",
    )
    .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    ensure_debug_cs_satisfied(&cs, "x_out")
}

pub fn debug_check_rv64im_main_recursion_step_spartan_chunk_replay_surface(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let replay_chunk = backend_relation
        .payload
        .effective_chunk_replay_surface()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    if !backend_relation
        .payload
        .chunk_cover
        .covers_replay_surface(&replay_chunk)
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step payload replay surface is not dominated by the carried chunk cover".into(),
        ));
    }
    if replay_chunk.pi_ccs.ccs_outputs.len() < replay_chunk.fresh_claims.len() {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step replay surface has fewer CCS outputs than fresh claims".into(),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_main_recursion_step_spartan_pi_ccs_replay_lengths(
    backend_relation: &Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<(), Rv64imMainRecursionStepSpartanError> {
    let replay_chunk = backend_relation
        .payload
        .effective_chunk_replay_surface()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let step_cap = backend_relation
        .f_prime_advice
        .verifier_key_fs()
        .step_cap()
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let (params, _, structure) = rv64im_root_main_lane_context_for_step_cap(step_cap)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;
    let dims = build_dims_and_policy(&params, structure)
        .map_err(|err| Rv64imMainRecursionStepSpartanError::Prepare(err.to_string()))?;

    if replay_chunk.pi_ccs.replay_proof.sumcheck_rounds.len()
        != replay_chunk.pi_ccs.row_chals.len() + replay_chunk.pi_ccs.alpha_prime.len()
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS FE replay round count does not match row_chals + alpha_prime".into(),
        ));
    }
    if replay_chunk.pi_ccs.replay_proof.sumcheck_rounds_nc.len()
        != replay_chunk.pi_ccs.s_col.len() + replay_chunk.pi_ccs.alpha_prime_nc.len()
    {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS NC replay round count does not match s_col + alpha_prime_nc".into(),
        ));
    }
    if replay_chunk.pi_ccs.row_chals.len() != dims.ell_n {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS row challenge count does not match ell_n".into(),
        ));
    }
    if replay_chunk.pi_ccs.s_col.len() != dims.ell_m {
        return Err(Rv64imMainRecursionStepSpartanError::Prepare(
            "rv64im main recursion step Pi_CCS column challenge count does not match ell_m".into(),
        ));
    }
    Ok(())
}
