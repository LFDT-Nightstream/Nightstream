use super::*;
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
    let after_chunk_meta = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

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
    let after_pi_ccs = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

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
    let after_synthetic_relation_io = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

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
    let after_pi_rlc_parent_claim = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

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
    let after_pi_rlc_rhos = super::format_spartan_digest_hex(cs.clone().finish_digest32(0));

    let after_pi_rlc_rho_mats;
    match ctx.boundary_plan.rlc_mode {
        super::super::super::Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
            after_pi_rlc_rho_mats = after_pi_rlc_rhos.clone();
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
                    super::super::super::materialize_goldilocks_rot_matrices(
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
    let pi_rlc = super::super::super::Rv32imPiRlcStageOutput { parent_claim };
    let after_pi_rlc = after_pi_rlc_public.clone();

    let replayed_next_claims = super::super::super::synthesize_pi_dec_stage(
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
