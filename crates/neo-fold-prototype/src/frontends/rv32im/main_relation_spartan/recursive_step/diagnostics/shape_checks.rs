use super::*;
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
