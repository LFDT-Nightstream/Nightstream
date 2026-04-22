//! Owns explicit Π_CCS / Π_RLC / Π_DEC stage helpers for the chunk verifier body.
//!
//! This module keeps the inner verifier stages separate from the outer
//! chunk-theorem wrapper so recursive F' can progressively shed wrapper cargo
//! without rewriting the arithmetic gadgets again.

use std::io::{self, Write};
use std::time::Instant;

use super::*;

pub(super) struct Rv64imChunkNifsVerifierCtx<'a> {
    pub(super) params: &'a NeoParams,
    pub(super) structure: &'a CcsStructure<F>,
    pub(super) dims: Dims,
    pub(super) mat_digest: &'a [Goldilocks; 4],
    pub(super) terminal_final_claims: &'a [neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    pub(super) chunk_index: usize,
    pub(super) cover_chunk: &'a Rv64imMainCircuitChunkCover,
    pub(super) chunk: &'a Rv64imMainCircuitChunkReplaySurface,
    pub(super) logical_me_input_claims: Option<&'a [neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>]>,
    pub(super) logical_me_input_digests: Option<&'a [[F; 4]]>,
    pub(super) boundary_plan: Rv64imChunkBoundaryPlan,
    pub(super) rlc_zero_commit_suffix_len: usize,
}

pub(super) struct Rv64imPiCcsStageOutput {
    pub(super) effective_output_count: usize,
    pub(super) padded_ccs_outputs: Vec<CeClaimVar>,
    pub(super) r_prime_vars: Vec<KNumVar>,
    pub(super) s_col_prime_vars: Vec<KNumVar>,
    pub(super) fold_digest: [AllocatedNum<SpartanF>; 4],
}

pub(super) struct Rv64imPiRlcStageOutput {
    pub(super) parent_claim: CeClaimVar,
}

pub(super) struct Rv64imChunkNifsVerifierBodyOutput {
    pub(super) next_claims: Rv64imClaimBundle,
    pub(super) pi_ccs_fold_digest: [AllocatedNum<SpartanF>; 4],
}

fn emit_nifs_stage_trace(trace_prefix: Option<&str>, label: &str, started: Instant) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={:.2}ms", started.elapsed().as_secs_f64() * 1_000.0);
        let _ = io::stderr().flush();
    }
}

pub(super) fn synthesize_rv64im_chunk_nifs_verifier_body_with_outer_relation_mode<CS: ConstraintSystem<SpartanF>>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut CS,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    logical_me_input_claims: Option<&[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>]>,
    logical_me_input_digests: Option<&[[F; 4]]>,
    boundary_plan: Rv64imChunkBoundaryPlan,
    rlc_zero_commit_suffix_len: usize,
    absorb_synthetic_chunk_relation_io: bool,
    trace_prefix: Option<&str>,
) -> Result<Rv64imChunkNifsVerifierBodyOutput, SynthesisError> {
    if !cover_chunk.covers_replay_surface(chunk) {
        return Err(SynthesisError::Unsatisfiable);
    }
    if chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let started = Instant::now();
    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )?;
    emit_nifs_stage_trace(trace_prefix, "chunk_meta", started);
    let ctx = Rv64imChunkNifsVerifierCtx {
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        chunk_index,
        cover_chunk,
        chunk,
        logical_me_input_claims,
        logical_me_input_digests,
        boundary_plan,
        rlc_zero_commit_suffix_len,
    };
    let started = Instant::now();
    let pi_ccs = synthesize_pi_ccs_stage(&ctx, cs, transcript, &carried_claims, trace_prefix)?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs", started);
    if absorb_synthetic_chunk_relation_io {
        let started = Instant::now();
        enforce_synthetic_outer_chunk_relation_public_io(
            &ctx,
            &mut cs.namespace(|| format!("chunk_{}_synthetic_relation_digest", chunk_index)),
            &pi_ccs.fold_digest,
            &format!("chunk_{}_synthetic_relation_digest", chunk_index),
        )?;
        emit_nifs_stage_trace(trace_prefix, "synthetic_relation_io", started);
    }
    let started = Instant::now();
    let pi_rlc = synthesize_pi_rlc_stage(&ctx, cs, transcript, &pi_ccs)?;
    emit_nifs_stage_trace(trace_prefix, "pi_rlc", started);
    let started = Instant::now();
    let replayed_next_claims = synthesize_pi_dec_stage(&ctx, cs, carried_claims, &pi_ccs, pi_rlc)?;
    emit_nifs_stage_trace(trace_prefix, "pi_dec", started);
    Ok(Rv64imChunkNifsVerifierBodyOutput {
        next_claims: replayed_next_claims,
        pi_ccs_fold_digest: pi_ccs.fold_digest,
    })
}

pub(super) fn synthesize_rv64im_chunk_nifs_verifier_body_with_synthetic_chunk_relation_io<
    CS: ConstraintSystem<SpartanF>,
>(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut CS,
    chunk_index: usize,
    cover_chunk: &Rv64imMainCircuitChunkCover,
    chunk: &Rv64imMainCircuitChunkReplaySurface,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv64imClaimBundle,
    logical_me_input_claims: Option<&[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>]>,
    logical_me_input_digests: Option<&[[F; 4]]>,
    boundary_plan: Rv64imChunkBoundaryPlan,
    rlc_zero_commit_suffix_len: usize,
    trace_prefix: Option<&str>,
) -> Result<Rv64imClaimBundle, SynthesisError> {
    Ok(synthesize_rv64im_chunk_nifs_verifier_body_with_outer_relation_mode(
        params,
        structure,
        dims,
        mat_digest,
        terminal_final_claims,
        cs,
        chunk_index,
        cover_chunk,
        chunk,
        transcript,
        carried_claims,
        logical_me_input_claims,
        logical_me_input_digests,
        boundary_plan,
        rlc_zero_commit_suffix_len,
        true,
        trace_prefix,
    )?
    .next_claims)
}

pub(super) fn synthesize_pi_ccs_stage<CS: ConstraintSystem<SpartanF>>(
    ctx: &Rv64imChunkNifsVerifierCtx<'_>,
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: &Rv64imClaimBundle,
    trace_prefix: Option<&str>,
) -> Result<Rv64imPiCcsStageOutput, SynthesisError> {
    let started = Instant::now();
    bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{}_bind_header", ctx.chunk_index)),
        transcript,
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
    )?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.bind_header", started);
    let started = Instant::now();
    if let Some(logical_me_input_digests) = ctx.logical_me_input_digests {
        crate::rv64im::main_relation_circuit::pi_ccs::bind_me_inputs_with_projection_digests(
            &mut cs.namespace(|| format!("chunk_{}_bind_me_inputs", ctx.chunk_index)),
            transcript,
            logical_me_input_digests,
            trace_prefix,
        )?;
    } else if let Some(logical_me_input_claims) = ctx.logical_me_input_claims {
        crate::rv64im::main_relation_circuit::pi_ccs::bind_me_inputs_with_native_claims(
            &mut cs.namespace(|| format!("chunk_{}_bind_me_inputs", ctx.chunk_index)),
            transcript,
            carried_claims.effective_claims(),
            logical_me_input_claims,
            trace_prefix,
        )?;
    } else {
        bind_me_inputs(
            &mut cs.namespace(|| format!("chunk_{}_bind_me_inputs", ctx.chunk_index)),
            transcript,
            carried_claims.effective_claims(),
            trace_prefix,
        )?;
    }
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.bind_me_inputs", started);
    let started = Instant::now();
    let public_challenges = crate::rv64im::main_relation_circuit::pi_ccs::sample_challenges_with_native(
        &mut cs.namespace(|| format!("chunk_{}_sample_challenges", ctx.chunk_index)),
        transcript,
        ctx.dims,
        &ctx.chunk.pi_ccs.public_challenges,
        &format!("chunk_{}_sample_challenges", ctx.chunk_index),
    )?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.sample_challenges", started);

    let started = Instant::now();
    let active_fresh_claim_count = ctx.chunk.handoff.public_chunk.steps.len();
    let cover_fresh_claim_count = ctx.cover_chunk.fresh_claim_count as usize;
    if active_fresh_claim_count > ctx.chunk.fresh_claims.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if cover_fresh_claim_count > ctx.cover_chunk.fresh_claim_shapes.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let covered_fresh_claims = ctx
        .cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| cover_ccs_claim(shape, ctx.chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::rv64im::main_relation_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{}_fresh_claim_{fresh_index}", ctx.chunk_index)),
                fresh,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.alloc_fresh_claims", started);

    let started = Instant::now();
    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_fe", ctx.chunk_index)),
        ctx.structure,
        &public_challenges.alpha,
        &ctx.chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        ctx.chunk.pi_ccs.public_challenges.gamma,
        cover_fresh_claim_count,
        carried_claims.effective_claims(),
        ctx.rlc_zero_commit_suffix_len,
        rv64im_main_relation_delta(),
        &format!("chunk_{}_initial_sum_fe", ctx.chunk_index),
    )?;
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{}_fe_sumcheck_domain", ctx.chunk_index)),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
    )?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        transcript.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_tag", ctx.chunk_index)),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )?;
        transcript.append_const_fields_raw(
            cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial_append", ctx.chunk_index)),
            &[
                SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
            ],
        )?;
    } else {
        append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index)),
            transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{}_fe_sumcheck_initial", ctx.chunk_index),
        )?;
    }
    let padded_fe_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{}_fe_round", ctx.chunk_index),
    )?;
    let fe_round_values = pad_round_values(
        &ctx.cover_chunk.fe_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )?;
    let fe_challenge_values = chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.row_chals, &ctx.chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_fe_sumcheck", ctx.chunk_index)),
        transcript,
        max_degree_from_cover_round_lengths(&ctx.cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{}_fe_sumcheck", ctx.chunk_index),
    )?;
    let (r_prime_vars, alpha_prime_vars) = split_vec(&fe_challenges, ctx.dims.ell_n)?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.fe_sumcheck", started);

    let started = Instant::now();
    let zero_nc = alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index)),
        KNum::from_neo_k(K::ZERO),
        &format!("chunk_{}_initial_sum_nc_zero", ctx.chunk_index),
    )?;
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{}_nc_sumcheck_domain", ctx.chunk_index)),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
    )?;
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_tag", ctx.chunk_index)),
        &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
    )?;
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("chunk_{}_nc_sumcheck_initial_append", ctx.chunk_index)),
        &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
    )?;
    let padded_nc_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_rounds", ctx.chunk_index)),
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{}_nc_round", ctx.chunk_index),
    )?;
    let nc_round_values = pad_round_values(
        &ctx.cover_chunk.nc_round_lengths,
        &ctx.chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )?;
    let nc_challenge_values = chunk_sumcheck_challenges(&ctx.chunk.pi_ccs.s_col, &ctx.chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{}_nc_sumcheck", ctx.chunk_index)),
        transcript,
        max_degree_from_cover_round_lengths(&ctx.cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv64im_main_relation_delta(),
        &format!("chunk_{}_nc_sumcheck", ctx.chunk_index),
    )?;
    let (s_col_prime_vars, alpha_prime_nc_vars) = split_vec(&nc_challenges, ctx.dims.ell_m)?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.nc_sumcheck", started);

    let started = Instant::now();
    let fold_digest = transcript.digest32(cs.namespace(|| format!("chunk_{}_fold_digest", ctx.chunk_index)))?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.fold_digest", started);

    let started = Instant::now();
    let effective_output_count = ctx.chunk.pi_ccs.ccs_outputs.len();
    let constant_child_prefix = match ctx.boundary_plan.rlc_mode {
        Rv64imChunkRlcMode::Standard { constant_child_prefix } => constant_child_prefix,
        Rv64imChunkRlcMode::TerminalLastChunkShortcut => 0,
    };
    let mut padded_ccs_outputs = Vec::with_capacity(ctx.cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in ctx.cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = ctx.chunk.pi_ccs.ccs_outputs.get(output_index);
        let output = if output_index < active_fresh_claim_count {
            let claim = cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &ctx.chunk.pi_ccs.row_chals,
                &ctx.chunk.pi_ccs.s_col,
            )?;
            if output_index < constant_child_prefix {
                let fresh = covered_fresh_claims
                    .get(output_index)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let fresh_x_values =
                    crate::rv64im::main_relation_circuit::output_binding::embedded_fresh_x_values(fresh);
                crate::rv64im::main_relation_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &fresh.c.data,
                    &fresh_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )?
            } else {
                let fresh = covered_fresh_claim_vars
                    .get(output_index)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                crate::rv64im::main_relation_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &fresh.c_data,
                    &fresh.c_data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )?
            }
        } else if output_index < effective_output_count {
            let claim = cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &ctx.chunk.pi_ccs.row_chals,
                &ctx.chunk.pi_ccs.s_col,
            )?;
            let me_input_index = output_index
                .checked_sub(cover_fresh_claim_count)
                .ok_or(SynthesisError::Unsatisfiable)?;
            let me_input = carried_claims
                .effective_claims()
                .get(me_input_index)
                .ok_or(SynthesisError::Unsatisfiable)?;
            if output_index < constant_child_prefix {
                let me_input_x_values =
                    crate::rv64im::main_relation_circuit::output_binding::embedded_me_input_x_values(me_input)?;
                crate::rv64im::main_relation_circuit::claim::alloc_ce_claim_without_f_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.c_data_values,
                    &me_input_x_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )?
            } else {
                crate::rv64im::main_relation_circuit::claim::alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                    &claim,
                    &me_input.c_data,
                    &me_input.c_data_values,
                    &r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
                )?
            }
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = ctx.chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = ctx.chunk.pi_ccs.s_col.clone();
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index)),
                &padded_claim,
                &r_prime_vars,
                &ctx.chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &ctx.chunk.pi_ccs.s_col,
                &format!("chunk_{}_ccs_output_{output_index}", ctx.chunk_index),
            )?
        };
        padded_ccs_outputs.push(output);
    }
    let ccs_outputs = padded_ccs_outputs.clone();
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.alloc_outputs", started);
    let started = Instant::now();
    enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| format!("chunk_{}_output_binding", ctx.chunk_index)),
        ctx.structure,
        ctx.params,
        &covered_fresh_claim_vars,
        carried_claims.effective_claims(),
        &ccs_outputs,
        &r_prime_vars,
        &ctx.chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &ctx.chunk.pi_ccs.s_col,
        &format!("chunk_{}_output_binding", ctx.chunk_index),
    )?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.output_binding", started);
    let me_inputs_r_vars = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.r.as_slice());
    let me_inputs_r_values = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.r_values.as_slice());
    let effective_me_output_count = effective_output_count
        .checked_sub(cover_fresh_claim_count)
        .ok_or(SynthesisError::Unsatisfiable)?;
    let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
        .iter()
        .cloned()
        .chain(
            padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                .iter()
                .cloned(),
        )
        .collect::<Vec<_>>();
    let started = Instant::now();
    let _ = enforce_terminal_identity_fe(
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
        rv64im_main_relation_delta(),
        &format!("chunk_{}_terminal_fe", ctx.chunk_index),
    )?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.terminal_fe", started);
    let started = Instant::now();
    let _ = enforce_terminal_identity_nc(
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
        rv64im_main_relation_delta(),
        &format!("chunk_{}_terminal_nc", ctx.chunk_index),
    )?;
    emit_nifs_stage_trace(trace_prefix, "pi_ccs.terminal_nc", started);

    Ok(Rv64imPiCcsStageOutput {
        effective_output_count,
        padded_ccs_outputs,
        r_prime_vars,
        s_col_prime_vars,
        fold_digest,
    })
}

pub(super) fn enforce_outer_chunk_relation_public_io<CS: ConstraintSystem<SpartanF>>(
    ctx: &Rv64imChunkNifsVerifierCtx<'_>,
    cs: &mut CS,
    fold_digest: &[AllocatedNum<SpartanF>; 4],
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
) -> Result<(), SynthesisError> {
    let chunk_relation_digest_input = next_public_digest(
        public_inputs,
        public_cursor,
        &format!("chunk_{}_relation_digest", ctx.chunk_index),
    )?;
    let chunk_relation_digest = chunk_relation_digest_circuit(
        &mut cs.namespace(|| format!("chunk_{}_relation_digest", ctx.chunk_index)),
        ctx.chunk.handoff.public_chunk_digest,
        fold_digest,
        ctx.chunk.handoff.bridge_handoff_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("chunk_{}_relation_digest_eq", ctx.chunk_index)),
        &chunk_relation_digest,
        &chunk_relation_digest_input,
        &format!("chunk_{}_relation_digest_eq", ctx.chunk_index),
    )?;
    Ok(())
}

pub(super) fn enforce_synthetic_outer_chunk_relation_public_io<CS: ConstraintSystem<SpartanF>>(
    ctx: &Rv64imChunkNifsVerifierCtx<'_>,
    cs: &mut CS,
    fold_digest: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<(), SynthesisError> {
    let synthetic_chunk_relation_digest = alloc_private_field_values(
        &mut cs.namespace(|| format!("{label}_synthetic_chunk_relation_digest")),
        &digest32_as_spartan_fields(ctx.chunk.handoff.chunk_relation_digest),
        &format!("{label}_synthetic_chunk_relation_digest"),
    )?;
    let mut synthetic_chunk_relation_cursor = 0usize;
    enforce_outer_chunk_relation_public_io(
        ctx,
        cs,
        fold_digest,
        &synthetic_chunk_relation_digest,
        &mut synthetic_chunk_relation_cursor,
    )?;
    if synthetic_chunk_relation_cursor != synthetic_chunk_relation_digest.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn synthesize_pi_rlc_stage<CS: ConstraintSystem<SpartanF>>(
    ctx: &Rv64imChunkNifsVerifierCtx<'_>,
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    pi_ccs: &Rv64imPiCcsStageOutput,
) -> Result<Rv64imPiRlcStageOutput, SynthesisError> {
    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            Rv64imChunkChildClaimSource::TerminalFinalClaims,
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = cover_ce_claim(&ctx.cover_chunk.parent_claim_shape, Some(&ctx.chunk.pi_rlc.parent))?;
        alloc_ce_claim(
            &mut cs.namespace(|| format!("chunk_{}_terminal_parent_claim", ctx.chunk_index)),
            &claim,
            &format!("chunk_{}_terminal_parent_claim", ctx.chunk_index),
        )?
    } else {
        let claim = cover_ce_claim_with_shared_point(
            &ctx.cover_chunk.parent_claim_shape,
            Some(&ctx.chunk.pi_rlc.parent),
            &ctx.chunk.pi_ccs.row_chals,
            &ctx.chunk.pi_ccs.s_col,
        )?;
        alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| format!("chunk_{}_parent_claim", ctx.chunk_index)),
            &claim,
            &pi_ccs.r_prime_vars,
            &ctx.chunk.pi_ccs.row_chals,
            &pi_ccs.s_col_prime_vars,
            &ctx.chunk.pi_ccs.s_col,
            &format!("chunk_{}_parent_claim", ctx.chunk_index),
        )?
    };
    let child_claim_source = match ctx.boundary_plan.child_claim_source {
        Rv64imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        Rv64imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let rho_vars = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{}_rlc_rhos", ctx.chunk_index)),
        transcript,
        pi_ccs.padded_ccs_outputs.len(),
        &format!("chunk_{}_rlc_rhos", ctx.chunk_index),
    )?;
    match ctx.boundary_plan.rlc_mode {
        Rv64imChunkRlcMode::TerminalLastChunkShortcut => {
            enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                &mut cs.namespace(|| format!("chunk_{}_rlc_public", ctx.chunk_index)),
                &parent_claim,
                &pi_ccs.padded_ccs_outputs,
                child_claim_source,
                &rho_vars,
                ctx.params.b,
                &format!("chunk_{}_rlc_public", ctx.chunk_index),
            )?;
        }
        Rv64imChunkRlcMode::Standard { constant_child_prefix } => {
            if constant_child_prefix == pi_ccs.padded_ccs_outputs.len() {
                crate::rv64im::main_relation_circuit::pi_rlc::enforce_rlc_public_with_rho_coeffs_for_constant_children(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_public", ctx.chunk_index)),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_vars,
                    &format!("chunk_{}_rlc_public", ctx.chunk_index),
                )?;
            } else {
                let rho_mats = materialize_goldilocks_rot_matrices(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_rho_mats", ctx.chunk_index)),
                    &rho_vars,
                    &format!("chunk_{}_rlc_rho_mats", ctx.chunk_index),
                )?;
                crate::rv64im::main_relation_circuit::pi_rlc::enforce_rlc_public_with_rho_vars_constant_prefix_zero_commit_suffix(
                    &mut cs.namespace(|| format!("chunk_{}_rlc_public", ctx.chunk_index)),
                    &parent_claim,
                    &pi_ccs.padded_ccs_outputs,
                    &rho_mats,
                    constant_child_prefix,
                    ctx.rlc_zero_commit_suffix_len,
                    &format!("chunk_{}_rlc_public", ctx.chunk_index),
                )?;
            }
        }
    }

    Ok(Rv64imPiRlcStageOutput { parent_claim })
}

pub(super) fn synthesize_pi_dec_stage<CS: ConstraintSystem<SpartanF>>(
    ctx: &Rv64imChunkNifsVerifierCtx<'_>,
    cs: &mut CS,
    carried_claims: Rv64imClaimBundle,
    pi_ccs: &Rv64imPiCcsStageOutput,
    pi_rlc: Rv64imPiRlcStageOutput,
) -> Result<Rv64imClaimBundle, SynthesisError> {
    let carry_terminal_state = matches!(
        (ctx.boundary_plan.child_claim_source, ctx.boundary_plan.next_carry_mode),
        (
            Rv64imChunkChildClaimSource::TerminalFinalClaims,
            Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let child_claim_source = match ctx.boundary_plan.child_claim_source {
        Rv64imChunkChildClaimSource::ReplayedChildren => &ctx.chunk.pi_dec.children,
        Rv64imChunkChildClaimSource::TerminalFinalClaims => ctx.terminal_final_claims,
    };
    let effective_child_count = child_claim_source.len();
    let padded_child_claims = ctx
        .cover_chunk
        .child_claim_shapes
        .iter()
        .enumerate()
        .map(|(child_index, shape)| {
            if carry_terminal_state {
                let claim = cover_ce_claim(shape, child_claim_source.get(child_index))?;
                alloc_ce_claim(
                    &mut cs.namespace(|| format!("chunk_{}_terminal_child_claim_{child_index}", ctx.chunk_index)),
                    &claim,
                    &format!("chunk_{}_terminal_child_claim_{child_index}", ctx.chunk_index),
                )
            } else {
                let claim = cover_ce_claim_with_shared_point(
                    shape,
                    child_claim_source.get(child_index),
                    &ctx.chunk.pi_ccs.row_chals,
                    &ctx.chunk.pi_ccs.s_col,
                )?;
                alloc_ce_claim_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{}_child_claim_{child_index}", ctx.chunk_index)),
                    &claim,
                    &pi_ccs.r_prime_vars,
                    &ctx.chunk.pi_ccs.row_chals,
                    &pi_ccs.s_col_prime_vars,
                    &ctx.chunk.pi_ccs.s_col,
                    &format!("chunk_{}_child_claim_{child_index}", ctx.chunk_index),
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    match ctx.boundary_plan.next_carry_mode {
        Rv64imChunkNextCarryMode::ReplaceWithEffectiveChildren => {
            enforce_dec_public(
                &mut cs.namespace(|| format!("chunk_{}_dec_public", ctx.chunk_index)),
                &pi_rlc.parent_claim,
                &padded_child_claims[..effective_child_count],
                ctx.params.b,
                &format!("chunk_{}_dec_public", ctx.chunk_index),
            )?;
            Ok(Rv64imClaimBundle::from_padded_claims(
                padded_child_claims,
                effective_child_count,
            ))
        }
        Rv64imChunkNextCarryMode::PreserveIncoming => {
            if !matches!(
                ctx.boundary_plan.rlc_mode,
                Rv64imChunkRlcMode::TerminalLastChunkShortcut
            ) {
                crate::rv64im::main_relation_circuit::pi_dec::enforce_dec_public_with_constant_children(
                    &mut cs.namespace(|| format!("chunk_{}_dec_public", ctx.chunk_index)),
                    &pi_rlc.parent_claim,
                    child_claim_source,
                    ctx.params.b,
                    &format!("chunk_{}_dec_public", ctx.chunk_index),
                )?;
            }
            Ok(carried_claims)
        }
    }
}
