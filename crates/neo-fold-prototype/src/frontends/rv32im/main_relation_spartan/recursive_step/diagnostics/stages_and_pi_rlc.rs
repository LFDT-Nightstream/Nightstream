use super::*;
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
        crate::rv32im::chunk::step_ivc::rv32im_chunk_step_ivc_initial_state_for_step_cap(
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
    let checkpoints = super::super::super::debug_measure_rv32im_rlc_public_stage_ranges(
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
    let checkpoints = super::super::super::debug_measure_rv32im_rlc_public_stage_ranges(
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
    let mut replayed_transcript = super::super::super::import_chunk_fold_transcript_in(
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
    super::super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        payload
            .initial_transcript_in
            .then_some(payload.chunk_cover.fresh_claim_count as usize),
    )
    .map_err(|err| stage_err("chunk_replay_chunk_meta", err))?;
    let after_chunk_meta = cs.num_aux();

    let ctx = super::super::super::Rv32imChunkNifsVerifierCtx {
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
    let pi_ccs = super::super::super::synthesize_pi_ccs_stage(
        &ctx,
        &mut cs.namespace(|| "payload_chunk_pi_ccs"),
        &mut replayed_transcript,
        &carried_claims,
        None,
    )
    .map_err(|err| stage_err("chunk_replay_pi_ccs", err))?;
    let after_pi_ccs = cs.num_aux();

    let bridge_handoff_digest = super::super::super::digest_const_inputs(
        &mut cs.namespace(|| "payload_chunk_bridge_handoff_digest"),
        replay_chunk.handoff.bridge_handoff_digest,
        "payload_chunk_bridge_handoff_digest",
    )
    .map_err(|err| stage_err("chunk_replay_bridge_handoff_digest", err))?;
    super::super::super::enforce_synthetic_outer_chunk_relation_public_io(
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
            super::super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims,
            super::super::super::Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim =
            super::super::super::cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))
                .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::super::alloc_ce_claim(
            &mut cs.namespace(|| "payload_chunk_pi_rlc_parent_claim"),
            &claim,
            "payload_chunk_pi_rlc_parent_claim",
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_alloc", err))?
    } else {
        let claim = super::super::super::cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )
        .map_err(|err| stage_err("chunk_replay_pi_rlc_parent_cover", err))?;
        super::super::super::alloc_ce_claim_public_surface_with_shared_point(
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
        super::super::super::Rv32imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        super::super::super::Rv32imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let rho_vars = super::super::super::sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| "payload_chunk_pi_rlc_rhos"),
        &mut replayed_transcript,
        pi_ccs.padded_ccs_outputs.len(),
        "payload_chunk_pi_rlc_rhos",
    )
    .map_err(|err| stage_err("chunk_replay_pi_rlc_rhos", err))?;
    let after_pi_rlc_rhos = cs.num_aux();

    let after_pi_rlc_rho_mats;
    match ctx.boundary_plan.rlc_mode {
        super::super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
            after_pi_rlc_rho_mats = cs.num_aux();
            super::super::super::enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
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
        super::super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => {
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
                    super::super::super::materialize_goldilocks_rot_matrices(
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
    let pi_rlc = super::super::super::Rv32imPiRlcStageOutput { parent_claim };
    let after_pi_rlc = after_pi_rlc_public;

    let replayed_next_claims = super::super::super::synthesize_pi_dec_stage(
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
