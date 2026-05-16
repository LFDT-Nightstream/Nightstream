use super::*;
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
    let mut replayed_transcript = super::super::super::import_chunk_fold_transcript_in(
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
    .map_err(|err| stage_err("pi_ccs_chunk_meta", err))?;

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
        .map(|(claim_index, shape)| {
            super::super::super::cover_ccs_claim(shape, ctx.chunk.fresh_claims.get(claim_index))
        })
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
        super::super::super::append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            &mut replayed_transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )
        .map_err(|err| stage_err("pi_ccs_fe_sumcheck_initial", err))?;
    }
    let padded_fe_rounds = super::super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_alloc_fe_rounds", err))?;
    let fe_round_values = super::super::super::pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| stage_err("pi_ccs_pad_fe_round_values", err))?;
    let fe_challenge_values =
        super::super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_fe_sumcheck", err))?;
    let (r_prime_vars, alpha_prime_vars) = super::super::super::split_vec(&fe_challenges, ctx.dims.ell_n)
        .map_err(|err| stage_err("pi_ccs_fe_split", err))?;
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
    let padded_nc_rounds = super::super::super::alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_alloc_nc_rounds", err))?;
    let nc_round_values = super::super::super::pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| stage_err("pi_ccs_pad_nc_round_values", err))?;
    let nc_challenge_values =
        super::super::super::chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = crate::superneo_circuit::sumcheck_replay::verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_sumcheck", ctx.chunk_index)),
        &mut replayed_transcript,
        super::super::super::max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
    )
    .map_err(|err| stage_err("pi_ccs_nc_sumcheck", err))?;
    let (s_col_prime_vars, alpha_prime_nc_vars) = super::super::super::split_vec(&nc_challenges, ctx.dims.ell_m)
        .map_err(|err| stage_err("pi_ccs_nc_split", err))?;
    let after_nc_sumcheck = cs.num_aux();

    let _fold_digest = replayed_transcript
        .digest32(cs.namespace(|| format!("chunk_{}_fold_digest", ctx.chunk_index)))
        .map_err(|err| stage_err("pi_ccs_fold_digest", err))?;
    let after_fold_digest = cs.num_aux();

    let effective_output_count = ctx.chunk.pi_ccs.ccs_outputs.len();
    let constant_child_prefix = match ctx.boundary_plan.rlc_mode {
        super::super::super::Rv32imChunkRlcMode::Standard { constant_child_prefix } => constant_child_prefix,
        super::super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => 0,
    };
    let zero_output_suffix_start = effective_output_count.saturating_sub(ctx.rlc_zero_commit_suffix_len);
    let mut padded_ccs_outputs = Vec::with_capacity(ctx.cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in ctx.cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = ctx.chunk.pi_ccs.ccs_outputs.get(output_index);
        let claim = if output_index < effective_output_count {
            super::super::super::cover_ce_claim_with_shared_point(
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
            super::super::super::alloc_ce_claim_public_surface_with_shared_point(
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
