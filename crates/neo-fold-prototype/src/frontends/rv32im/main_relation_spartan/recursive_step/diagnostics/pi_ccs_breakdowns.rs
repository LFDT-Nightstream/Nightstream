use super::*;
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

    super::super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_bind_breakdown_chunk_meta", err))?;
    super::super::super::bind_header_and_instance_digest(
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

    super::super::super::append_chunk_meta_with_exact_initial_constants(
        &mut cs.namespace(|| "payload_chunk_meta"),
        &mut replayed_transcript,
        &replay_chunk.handoff,
        ctx.exact_initial_chunk_step_count,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_chunk_meta", err))?;
    super::super::super::bind_header_and_instance_digest(
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
        super::super::super::append_k_to_transcript(
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
    let padded_fe_rounds = super::super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_alloc_fe_rounds", err))?;
    push_constraint_delta(&mut fe_stages, &mut fe_previous, cs.num_constraints(), "alloc_rounds");
    let fe_round_values = super::super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, _) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds_with_trace(
        &mut cs,
        &mut replayed_transcript,
        super::super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
        |cs, stage| push_constraint_delta(&mut fe_stages, &mut fe_previous, cs.num_constraints(), stage),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_fe", err))?;
    let _ = super::super::super::split_vec(&fe_challenges, ctx.dims.ell_n)
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
    let padded_nc_rounds = super::super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_alloc_nc_rounds", err))?;
    push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), "alloc_rounds");
    let nc_round_values = super::super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, _) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds_with_trace(
        &mut cs,
        &mut replayed_transcript,
        super::super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
        |cs, stage| push_constraint_delta(&mut nc_stages, &mut nc_previous, cs.num_constraints(), stage),
    )
    .map_err(|err| stage_err("pi_ccs_sumcheck_nc", err))?;
    let _ = super::super::super::split_vec(&nc_challenges, ctx.dims.ell_m)
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
