use std::io::{self, Write};
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem};
use p3_field::PrimeField64;

use super::*;
use crate::spartan_backend::SpartanF;

mod prefix_fingerprints;
pub(crate) mod stage_ranges;

pub use prefix_fingerprints::debug_measure_rv32im_main_relation_state_in_prefix_fingerprints;
pub(crate) use stage_ranges::{
    debug_measure_rv32im_rlc_public_stage_ranges, debug_synthesize_rv32im_main_relation_chunk_with_stage_ranges,
    Rv32imMainRelationChunkStageCheckpoints,
};

fn chunk_stage_err(cs: &TestConstraintSystem<SpartanF>, stage: &str) -> String {
    let unsat = cs.which_is_unsatisfied().unwrap_or("unknown");
    format!("{stage}: {unsat}")
}

fn checkpoint(cs: &TestConstraintSystem<SpartanF>, stage: &str) -> Result<(), String> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(chunk_stage_err(cs, stage))
    }
}

fn profile_stage<F>(stage: &str, f: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String>,
{
    eprintln!("n2-step-chunk|start|{stage}");
    let _ = io::stderr().flush();
    let started = Instant::now();
    let result = f();
    eprintln!(
        "n2-step-chunk|done|{stage}|{:.3}",
        started.elapsed().as_secs_f64() * 1_000.0
    );
    let _ = io::stderr().flush();
    result
}

fn emit_aux_count(stage: &str, aux_count: usize) {
    eprintln!("n2-step-chunk|aux|{stage}|{aux_count}");
    let _ = io::stderr().flush();
}

pub(crate) fn debug_locate_rv32im_main_relation_chunk_stage(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut TestConstraintSystem<SpartanF>,
    chunk_index: usize,
    cover_chunk: &Rv32imMainCircuitChunkCover,
    chunk: &Rv32imMainCircuitChunkReplaySurface,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv32imClaimBundle,
    boundary_plan: Rv32imChunkBoundaryPlan,
    append_chunk_done: bool,
) -> Result<Rv32imClaimBundle, String> {
    if !cover_chunk.covers_replay_surface(chunk) {
        return Err("covers_replay_surface".into());
    }
    if chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err("ccs_outputs_lt_fresh_claims".into());
    }
    append_chunk_meta(
        &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
        transcript,
        &chunk.handoff,
    )
    .map_err(|err| format!("append_chunk_meta: {err}"))?;
    checkpoint(cs, "append_chunk_meta")?;

    bind_header_and_instance_digest(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_bind_header")),
        transcript,
        params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        mat_digest,
        &chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| format!("bind_header_and_instance_digest: {err}"))?;
    checkpoint(cs, "bind_header_and_instance_digest")?;

    bind_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_bind_me_inputs")),
        transcript,
        carried_claims.effective_claims(),
        None,
    )
    .map_err(|err| format!("bind_me_inputs: {err}"))?;
    checkpoint(cs, "bind_me_inputs")?;

    let public_challenges = sample_challenges(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_sample_challenges")),
        transcript,
        dims,
    )
    .map_err(|err| format!("sample_challenges: {err}"))?;
    checkpoint(cs, "sample_challenges")?;

    let active_fresh_claim_count = chunk.handoff.public_chunk.steps.len();
    let cover_fresh_claim_count = cover_chunk.fresh_claim_count as usize;
    if active_fresh_claim_count > chunk.fresh_claims.len() {
        return Err("active_fresh_claim_count_gt_fresh_claims".into());
    }
    if cover_fresh_claim_count > cover_chunk.fresh_claim_shapes.len() {
        return Err("active_fresh_claim_count_gt_fresh_claims".into());
    }
    let covered_fresh_claims = cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| cover_ccs_claim(shape, chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("cover_ccs_claim: {err}"))?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_fresh_claim_{fresh_index}")),
                fresh,
            )
            .map_err(|err| format!("alloc_fresh_claim_{fresh_index}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_initial_sum_fe")),
        structure,
        &public_challenges.alpha,
        &chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        chunk.pi_ccs.public_challenges.gamma,
        cover_fresh_claim_count,
        carried_claims.effective_claims(),
        0,
        rv32im_main_relation_delta(),
        &format!("chunk_{chunk_index}_initial_sum_fe"),
    )
    .map_err(|err| format!("claimed_initial_sum_from_me_inputs: {err}"))?;
    checkpoint(cs, "claimed_initial_sum_from_me_inputs")?;

    transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_domain")),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| format!("fe_sumcheck_domain: {err}"))?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial_tag")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| format!("fe_sumcheck_initial_tag: {err}"))?;
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial_append")),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| format!("fe_sumcheck_initial_append: {err}"))?;
    } else {
        append_k_to_transcript(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial")),
            transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            &format!("chunk_{chunk_index}_fe_sumcheck_initial"),
        )
        .map_err(|err| format!("fe_sumcheck_initial: {err}"))?;
    }
    checkpoint(cs, "fe_sumcheck_initial")?;

    let padded_fe_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_rounds")),
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
        &format!("chunk_{chunk_index}_fe_round"),
    )
    .map_err(|err| format!("alloc_fe_rounds: {err}"))?;
    let fe_round_values = pad_round_values(
        &cover_chunk.fe_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| format!("pad_fe_round_values: {err}"))?;
    let fe_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.row_chals, &chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck")),
        transcript,
        max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{chunk_index}_fe_sumcheck"),
    )
    .map_err(|err| format!("fe_sumcheck: {err}"))?;
    checkpoint(cs, "fe_sumcheck")?;
    let (r_prime_vars, alpha_prime_vars) =
        split_vec(&fe_challenges, dims.ell_n).map_err(|err| format!("split_fe_challenges: {err}"))?;

    let zero_nc = alloc_constant_k(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_initial_sum_nc_zero")),
        KNum::from_neo_k(K::ZERO),
        &format!("chunk_{chunk_index}_initial_sum_nc_zero"),
    )
    .map_err(|err| format!("initial_sum_nc_zero: {err}"))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_domain")),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| format!("nc_sumcheck_domain: {err}"))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_initial_tag")),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| format!("nc_sumcheck_initial_tag: {err}"))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_initial_append")),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| format!("nc_sumcheck_initial_append: {err}"))?;
    checkpoint(cs, "nc_sumcheck_initial")?;

    let padded_nc_rounds = alloc_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_nc_rounds")),
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        &format!("chunk_{chunk_index}_nc_round"),
    )
    .map_err(|err| format!("alloc_nc_rounds: {err}"))?;
    let nc_round_values = pad_round_values(
        &cover_chunk.nc_round_lengths,
        &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| format!("pad_nc_round_values: {err}"))?;
    let nc_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.s_col, &chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = verify_sumcheck_rounds(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck")),
        transcript,
        max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{chunk_index}_nc_sumcheck"),
    )
    .map_err(|err| format!("nc_sumcheck: {err}"))?;
    checkpoint(cs, "nc_sumcheck")?;
    let (s_col_prime_vars, alpha_prime_nc_vars) =
        split_vec(&nc_challenges, dims.ell_m).map_err(|err| format!("split_nc_challenges: {err}"))?;

    let fold_digest = transcript
        .digest32(cs.namespace(|| format!("chunk_{chunk_index}_fold_digest")))
        .map_err(|err| format!("fold_digest: {err}"))?;
    let chunk_relation_digest_input = next_public_digest(
        public_inputs,
        public_cursor,
        &format!("chunk_{chunk_index}_relation_digest"),
    )
    .map_err(|err| format!("chunk_relation_digest_input: {err}"))?;
    let chunk_relation_digest = chunk_relation_digest_circuit(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_relation_digest")),
        chunk.handoff.public_chunk_digest,
        &fold_digest,
        chunk.handoff.bridge_handoff_digest,
    )
    .map_err(|err| format!("chunk_relation_digest_circuit: {err}"))?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_relation_digest_eq")),
        &chunk_relation_digest,
        &chunk_relation_digest_input,
        &format!("chunk_{chunk_index}_relation_digest_eq"),
    )
    .map_err(|err| format!("chunk_relation_digest_eq: {err}"))?;
    checkpoint(cs, "relation_digest")?;

    let effective_output_count = chunk.pi_ccs.ccs_outputs.len();
    let mut padded_ccs_outputs = Vec::with_capacity(cover_chunk.ccs_output_shapes.len());
    for (output_index, shape) in cover_chunk.ccs_output_shapes.iter().enumerate() {
        let effective_claim = chunk.pi_ccs.ccs_outputs.get(output_index);
        let output = if output_index < active_fresh_claim_count {
            let claim =
                cover_ce_claim_with_shared_point(shape, effective_claim, &chunk.pi_ccs.row_chals, &chunk.pi_ccs.s_col)
                    .map_err(|err| format!("cover_fresh_output_{output_index}: {err}"))?;
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                &claim,
                &r_prime_vars,
                &chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &chunk.pi_ccs.s_col,
                &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
            )
            .map_err(|err| format!("alloc_fresh_output_{output_index}: {err}"))?
        } else if output_index < effective_output_count {
            let claim =
                cover_ce_claim_with_shared_point(shape, effective_claim, &chunk.pi_ccs.row_chals, &chunk.pi_ccs.s_col)
                    .map_err(|err| format!("cover_ccs_output_{output_index}: {err}"))?;
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                &claim,
                &r_prime_vars,
                &chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &chunk.pi_ccs.s_col,
                &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
            )
            .map_err(|err| format!("alloc_ccs_output_{output_index}: {err}"))?
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = chunk.pi_ccs.s_col.clone();
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                &padded_claim,
                &r_prime_vars,
                &chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &chunk.pi_ccs.s_col,
                &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
            )
            .map_err(|err| format!("alloc_padded_ccs_output_{output_index}: {err}"))?
        };
        padded_ccs_outputs.push(output);
        checkpoint(cs, &format!("ccs_output_{output_index}"))?;
    }
    let ccs_outputs = padded_ccs_outputs.clone();
    enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_output_binding")),
        structure,
        params,
        &covered_fresh_claim_vars,
        carried_claims.effective_claims(),
        &ccs_outputs,
        0,
        &r_prime_vars,
        &chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &chunk.pi_ccs.s_col,
        &format!("chunk_{chunk_index}_output_binding"),
    )
    .map_err(|err| format!("output_binding: {err}"))?;
    checkpoint(cs, "output_binding")?;

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
        .ok_or_else(|| "effective_me_output_count_underflow".to_string())?;
    let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
        .iter()
        .cloned()
        .chain(
            padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                .iter()
                .cloned(),
        )
        .collect::<Vec<_>>();
    let _ = enforce_terminal_identity_fe(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_fe")),
        &sumcheck_final_fe,
        structure,
        &chunk.pi_ccs.public_challenges,
        &public_challenges.alpha,
        &public_challenges.beta_a,
        &public_challenges.beta_r,
        &public_challenges.gamma,
        &r_prime_vars,
        &chunk.pi_ccs.row_chals,
        &alpha_prime_vars,
        &chunk.pi_ccs.alpha_prime,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        0,
        me_inputs_r_vars,
        me_inputs_r_values,
        rv32im_main_relation_delta(),
        &format!("chunk_{chunk_index}_terminal_fe"),
    )
    .map_err(|err| format!("terminal_fe: {err}"))?;
    checkpoint(cs, "terminal_fe")?;

    let _ = enforce_terminal_identity_nc(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_nc")),
        &sumcheck_final_nc,
        params,
        &chunk.pi_ccs.public_challenges,
        &public_challenges.beta_a,
        &public_challenges.beta_m,
        &public_challenges.gamma,
        &s_col_prime_vars,
        &chunk.pi_ccs.s_col,
        &alpha_prime_nc_vars,
        &chunk.pi_ccs.alpha_prime_nc,
        &effective_terminal_outputs,
        cover_fresh_claim_count,
        0,
        rv32im_main_relation_delta(),
        &format!("chunk_{chunk_index}_terminal_nc"),
    )
    .map_err(|err| format!("terminal_nc: {err}"))?;
    checkpoint(cs, "terminal_nc")?;

    let carry_terminal_state = matches!(
        (boundary_plan.child_claim_source, boundary_plan.next_carry_mode),
        (
            Rv32imChunkChildClaimSource::TerminalFinalClaims,
            Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );
    let parent_claim = if carry_terminal_state {
        let claim = cover_ce_claim(&cover_chunk.parent_claim_shape, Some(&chunk.pi_rlc.parent))
            .map_err(|err| format!("cover_terminal_parent_claim: {err}"))?;
        alloc_ce_claim(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_parent_claim")),
            &claim,
            &format!("chunk_{chunk_index}_terminal_parent_claim"),
        )
        .map_err(|err| format!("alloc_terminal_parent_claim: {err}"))?
    } else {
        let claim = cover_ce_claim_with_shared_point(
            &cover_chunk.parent_claim_shape,
            Some(&chunk.pi_rlc.parent),
            &chunk.pi_ccs.row_chals,
            &chunk.pi_ccs.s_col,
        )
        .map_err(|err| format!("cover_parent_claim: {err}"))?;
        alloc_ce_claim_public_surface_with_shared_point(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_parent_claim")),
            &claim,
            &r_prime_vars,
            &chunk.pi_ccs.row_chals,
            &s_col_prime_vars,
            &chunk.pi_ccs.s_col,
            &format!("chunk_{chunk_index}_parent_claim"),
        )
        .map_err(|err| format!("alloc_parent_claim: {err}"))?
    };
    checkpoint(cs, "parent_claim")?;

    let child_claim_source = match boundary_plan.child_claim_source {
        Rv32imChunkChildClaimSource::ReplayedChildren => &chunk.pi_dec.children,
        Rv32imChunkChildClaimSource::TerminalFinalClaims => terminal_final_claims,
    };
    let effective_child_count = child_claim_source.len();
    let padded_child_claims = cover_chunk
        .child_claim_shapes
        .iter()
        .enumerate()
        .map(|(child_index, shape)| {
            if carry_terminal_state {
                let claim = cover_ce_claim(shape, child_claim_source.get(child_index))
                    .map_err(|err| format!("cover_terminal_child_claim_{child_index}: {err}"))?;
                alloc_ce_claim_dec_surface(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_child_claim_{child_index}")),
                    &claim,
                    &format!("chunk_{chunk_index}_terminal_child_claim_{child_index}"),
                )
                .map_err(|err| format!("alloc_terminal_child_claim_{child_index}: {err}"))
            } else {
                let claim = cover_ce_claim_with_shared_point(
                    shape,
                    child_claim_source.get(child_index),
                    &chunk.pi_ccs.row_chals,
                    &chunk.pi_ccs.s_col,
                )
                .map_err(|err| format!("cover_child_claim_{child_index}: {err}"))?;
                alloc_ce_claim_dec_surface_with_shared_r(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_child_claim_{child_index}")),
                    &claim,
                    &r_prime_vars,
                    &chunk.pi_ccs.row_chals,
                    &format!("chunk_{chunk_index}_child_claim_{child_index}"),
                )
                .map_err(|err| format!("alloc_child_claim_{child_index}: {err}"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    checkpoint(cs, "child_claims")?;

    let rho_vars = sample_goldilocks_rot_rhos(
        &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_rhos")),
        transcript,
        padded_ccs_outputs.len(),
        &format!("chunk_{chunk_index}_rlc_rhos"),
    )
    .map_err(|err| format!("sample_rlc_rhos: {err}"))?;
    checkpoint(cs, "sample_rlc_rhos")?;

    match boundary_plan.rlc_mode {
        Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
            enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_public")),
                &parent_claim,
                &padded_ccs_outputs,
                child_claim_source,
                &rho_vars,
                params.b,
                &format!("chunk_{chunk_index}_rlc_public"),
            )
            .map_err(|err| format!("rlc_public_last_chunk: {err}"))?;
        }
        Rv32imChunkRlcMode::Standard { constant_child_prefix } => {
            let active_dense_children_len = padded_ccs_outputs.len();
            let rho_mats = materialize_goldilocks_rot_matrices(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_rho_mats")),
                &rho_vars[..active_dense_children_len],
                &format!("chunk_{chunk_index}_rlc_rho_mats"),
            )
            .map_err(|err| format!("materialize_rlc_rho_mats: {err}"))?;
            checkpoint(cs, "materialize_rlc_rho_mats")?;
            crate::superneo_circuit::pi_rlc::debug_locate_rlc_public_with_split_rho_views_stage(
                cs,
                &parent_claim,
                &padded_ccs_outputs,
                &rho_vars,
                &rho_mats,
                constant_child_prefix,
                0,
                &format!("chunk_{chunk_index}_rlc_public"),
            )
            .map_err(|err| format!("rlc_public: {err}"))?;
        }
    }
    checkpoint(cs, "rlc_public")?;

    let next_carried_claims = match boundary_plan.next_carry_mode {
        Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren => {
            enforce_dec_public(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_dec_public")),
                &parent_claim,
                &padded_child_claims[..effective_child_count],
                params.b,
                &format!("chunk_{chunk_index}_dec_public"),
            )
            .map_err(|err| format!("dec_public: {err}"))?;
            checkpoint(cs, "dec_public")?;
            Rv32imClaimBundle::from_padded_claims(padded_child_claims, effective_child_count)
        }
        Rv32imChunkNextCarryMode::PreserveIncoming => {
            if !matches!(boundary_plan.rlc_mode, Rv32imChunkRlcMode::TerminalLastChunkShortcut) {
                crate::superneo_circuit::pi_dec::enforce_dec_public_with_constant_children(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_dec_public")),
                    &parent_claim,
                    child_claim_source,
                    params.b,
                    &format!("chunk_{chunk_index}_dec_public"),
                )
                .map_err(|err| format!("dec_public_constant_children: {err}"))?;
                checkpoint(cs, "dec_public_constant_children")?;
            }
            carried_claims
        }
    };
    if append_chunk_done {
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_done_{chunk_index}")),
                &[
                    SpartanF::from_canonical_u64(RV32IM_CHUNK_DONE_RAW_TAG),
                    SpartanF::from_canonical_u64(1),
                ],
            )
            .map_err(|err| format!("chunk_done: {err}"))?;
        checkpoint(cs, "chunk_done")?;
    }
    Ok(next_carried_claims)
}

pub(crate) fn debug_profile_rv32im_main_relation_chunk_stage_progress(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    dims: Dims,
    mat_digest: &[Goldilocks; 4],
    terminal_final_claims: &[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>],
    cs: &mut TestConstraintSystem<SpartanF>,
    chunk_index: usize,
    cover_chunk: &Rv32imMainCircuitChunkCover,
    chunk: &Rv32imMainCircuitChunkReplaySurface,
    public_inputs: &[AllocatedNum<SpartanF>],
    public_cursor: &mut usize,
    transcript: &mut Poseidon2TranscriptCircuit,
    carried_claims: Rv32imClaimBundle,
    logical_me_input_claims: Option<&[neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>]>,
    boundary_plan: Rv32imChunkBoundaryPlan,
    append_chunk_done: bool,
) -> Result<Rv32imClaimBundle, String> {
    if !cover_chunk.covers_replay_surface(chunk) {
        return Err("covers_replay_surface".into());
    }
    if chunk.pi_ccs.ccs_outputs.len() < chunk.fresh_claims.len() {
        return Err("ccs_outputs_lt_fresh_claims".into());
    }

    profile_stage("append_chunk_meta_raw", || {
        append_chunk_meta(
            &mut cs.namespace(|| format!("chunk_meta_{chunk_index}")),
            transcript,
            &chunk.handoff,
        )
        .map_err(|err| format!("append_chunk_meta: {err}"))
    })?;
    profile_stage("append_chunk_meta_checkpoint", || checkpoint(cs, "append_chunk_meta"))?;

    profile_stage("bind_header_and_instance_digest", || {
        bind_header_and_instance_digest(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_bind_header")),
            transcript,
            params,
            structure.n,
            structure.m,
            structure.t(),
            &structure.f,
            dims,
            mat_digest,
            &chunk
                .handoff
                .public_chunk_instance_digest
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
        )
        .map_err(|err| format!("bind_header_and_instance_digest: {err}"))?;
        checkpoint(cs, "bind_header_and_instance_digest")
    })?;

    let mut me_input_digests = Vec::new();
    let mut me_input_digest_values = Vec::new();
    profile_stage("bind_me_input_digests_compute", || {
        me_input_digests.reserve(carried_claims.effective_count());
        me_input_digest_values.reserve(carried_claims.effective_count());
        if let Some(logical_me_input_claims) = logical_me_input_claims {
            if carried_claims.effective_count() != logical_me_input_claims.len() {
                return Err("bind_me_input_digests_compute_len_mismatch".into());
            }
            for (idx, (claim, native_claim)) in carried_claims
                .effective_claims()
                .iter()
                .zip(logical_me_input_claims.iter())
                .enumerate()
            {
                me_input_digests.push(
                    crate::superneo_circuit::claim::me_digest_poseidon_with_native_claim(
                        &mut cs.namespace(|| format!("chunk_{chunk_index}_me_input_digest_{idx}")),
                        claim,
                        native_claim,
                        &format!("chunk_{chunk_index}_me_input_digest_{idx}"),
                    )
                    .map_err(|err| format!("me_digest_poseidon_with_native_claim: {err}"))?,
                );
                me_input_digest_values
                    .push(crate::superneo_circuit::claim::me_digest_poseidon_values_from_native_claim(native_claim));
            }
        } else {
            for (idx, claim) in carried_claims.effective_claims().iter().enumerate() {
                me_input_digests.push(
                    crate::superneo_circuit::claim::me_digest_poseidon(
                        &mut cs.namespace(|| format!("chunk_{chunk_index}_me_input_digest_{idx}")),
                        claim,
                        &format!("chunk_{chunk_index}_me_input_digest_{idx}"),
                    )
                    .map_err(|err| format!("me_digest_poseidon: {err}"))?,
                );
                me_input_digest_values.push(crate::superneo_circuit::claim::me_digest_poseidon_values(claim));
            }
        }
        checkpoint(cs, "bind_me_input_digests_compute")
    })?;
    profile_stage("bind_me_input_digests_transcript", || {
        crate::superneo_circuit::pi_ccs::bind_me_input_digests(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_bind_me_inputs")),
            transcript,
            &me_input_digests,
            &me_input_digest_values,
        )
        .map_err(|err| format!("bind_me_input_digests: {err}"))?;
        checkpoint(cs, "bind_me_input_digests_transcript")
    })?;

    let mut public_challenges = None;
    profile_stage("sample_challenges", || {
        public_challenges = Some(
            sample_challenges(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_sample_challenges")),
                transcript,
                dims,
            )
            .map_err(|err| format!("sample_challenges: {err}"))?,
        );
        checkpoint(cs, "sample_challenges")
    })?;
    let public_challenges = public_challenges.ok_or_else(|| "sample_challenges missing".to_string())?;

    let active_fresh_claim_count = chunk.handoff.public_chunk.steps.len();
    let cover_fresh_claim_count = cover_chunk.fresh_claim_count as usize;
    if active_fresh_claim_count > chunk.fresh_claims.len() {
        return Err("active_fresh_claim_count_gt_fresh_claims".into());
    }
    if cover_fresh_claim_count > cover_chunk.fresh_claim_shapes.len() {
        return Err("active_fresh_claim_count_gt_fresh_claims".into());
    }
    let covered_fresh_claims = cover_chunk
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| cover_ccs_claim(shape, chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("cover_ccs_claim: {err}"))?;
    let covered_fresh_claim_vars = covered_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_fresh_claim_{fresh_index}")),
                fresh,
            )
            .map_err(|err| format!("alloc_fresh_claim_{fresh_index}: {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut initial_sum_fe = None;
    let mut initial_sum_fe_value = None;
    profile_stage("claimed_initial_sum_from_me_inputs", || {
        let (sum, value) = claimed_initial_sum_from_me_inputs(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_initial_sum_fe")),
            structure,
            &public_challenges.alpha,
            &chunk.pi_ccs.public_challenges.alpha,
            &public_challenges.gamma,
            chunk.pi_ccs.public_challenges.gamma,
            cover_fresh_claim_count,
            carried_claims.effective_claims(),
            0,
            rv32im_main_relation_delta(),
            &format!("chunk_{chunk_index}_initial_sum_fe"),
        )
        .map_err(|err| format!("claimed_initial_sum_from_me_inputs: {err}"))?;
        initial_sum_fe = Some(sum);
        initial_sum_fe_value = Some(value);
        checkpoint(cs, "claimed_initial_sum_from_me_inputs")?;
        emit_aux_count("claimed_initial_sum_from_me_inputs", cs.scalar_aux().len());
        Ok(())
    })?;
    let initial_sum_fe = initial_sum_fe.ok_or_else(|| "claimed_initial_sum_from_me_inputs missing".to_string())?;
    let initial_sum_fe_value =
        initial_sum_fe_value.ok_or_else(|| "claimed_initial_sum_from_me_inputs value missing".to_string())?;

    profile_stage("fe_sumcheck_initial", || {
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_domain")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
            )
            .map_err(|err| format!("fe_sumcheck_domain: {err}"))?;
        if carried_claims.effective_count() == 0 {
            let coeffs = initial_sum_fe_value.as_coeffs();
            transcript
                .append_const_fields_raw(
                    cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial_tag")),
                    &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
                )
                .map_err(|err| format!("fe_sumcheck_initial_tag: {err}"))?;
            transcript
                .append_const_fields_raw(
                    cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial_append")),
                    &[
                        SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                        SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                    ],
                )
                .map_err(|err| format!("fe_sumcheck_initial_append: {err}"))?;
        } else {
            append_k_to_transcript(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck_initial")),
                transcript,
                PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
                &initial_sum_fe,
                initial_sum_fe_value,
                &format!("chunk_{chunk_index}_fe_sumcheck_initial"),
            )
            .map_err(|err| format!("fe_sumcheck_initial: {err}"))?;
        }
        checkpoint(cs, "fe_sumcheck_initial")?;
        emit_aux_count("fe_sumcheck_initial", cs.scalar_aux().len());
        Ok(())
    })?;

    let mut r_prime_vars = None;
    let mut s_col_prime_vars = None;
    let mut alpha_prime_vars = None;
    let mut alpha_prime_nc_vars = None;
    let mut sumcheck_final_fe = None;
    let mut sumcheck_final_nc = None;

    profile_stage("fe_sumcheck", || {
        let padded_fe_rounds = alloc_rounds(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_rounds")),
            &cover_chunk.fe_round_lengths,
            &chunk.pi_ccs.replay_proof.sumcheck_rounds,
            &format!("chunk_{chunk_index}_fe_round"),
        )
        .map_err(|err| format!("alloc_fe_rounds: {err}"))?;
        let fe_round_values = pad_round_values(
            &cover_chunk.fe_round_lengths,
            &chunk.pi_ccs.replay_proof.sumcheck_rounds,
        )
        .map_err(|err| format!("pad_fe_round_values: {err}"))?;
        let fe_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.row_chals, &chunk.pi_ccs.alpha_prime);
        let (fe_challenges, final_fe) = verify_sumcheck_rounds(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_fe_sumcheck")),
            transcript,
            max_degree_from_cover_round_lengths(&cover_chunk.fe_round_lengths),
            &initial_sum_fe,
            &padded_fe_rounds,
            &fe_round_values,
            &fe_challenge_values,
            rv32im_main_relation_delta(),
            &format!("chunk_{chunk_index}_fe_sumcheck"),
        )
        .map_err(|err| format!("fe_sumcheck: {err}"))?;
        let (r_prime, alpha_prime) =
            split_vec(&fe_challenges, dims.ell_n).map_err(|err| format!("split_fe_challenges: {err}"))?;
        r_prime_vars = Some(r_prime);
        alpha_prime_vars = Some(alpha_prime);
        sumcheck_final_fe = Some(final_fe);
        checkpoint(cs, "fe_sumcheck")?;
        emit_aux_count("fe_sumcheck", cs.scalar_aux().len());
        Ok(())
    })?;

    profile_stage("nc_sumcheck", || {
        let zero_nc = alloc_constant_k(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_initial_sum_nc_zero")),
            KNum::from_neo_k(K::ZERO),
            &format!("chunk_{chunk_index}_initial_sum_nc_zero"),
        )
        .map_err(|err| format!("initial_sum_nc_zero: {err}"))?;
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_domain")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
            )
            .map_err(|err| format!("nc_sumcheck_domain: {err}"))?;
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_initial_tag")),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| format!("nc_sumcheck_initial_tag: {err}"))?;
        transcript
            .append_const_fields_raw(
                cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck_initial_append")),
                &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
            )
            .map_err(|err| format!("nc_sumcheck_initial_append: {err}"))?;
        let padded_nc_rounds = alloc_rounds(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_nc_rounds")),
            &cover_chunk.nc_round_lengths,
            &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
            &format!("chunk_{chunk_index}_nc_round"),
        )
        .map_err(|err| format!("alloc_nc_rounds: {err}"))?;
        let nc_round_values = pad_round_values(
            &cover_chunk.nc_round_lengths,
            &chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        )
        .map_err(|err| format!("pad_nc_round_values: {err}"))?;
        let nc_challenge_values = chunk_sumcheck_challenges(&chunk.pi_ccs.s_col, &chunk.pi_ccs.alpha_prime_nc);
        let (nc_challenges, final_nc) = verify_sumcheck_rounds(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_nc_sumcheck")),
            transcript,
            max_degree_from_cover_round_lengths(&cover_chunk.nc_round_lengths),
            &zero_nc,
            &padded_nc_rounds,
            &nc_round_values,
            &nc_challenge_values,
            rv32im_main_relation_delta(),
            &format!("chunk_{chunk_index}_nc_sumcheck"),
        )
        .map_err(|err| format!("nc_sumcheck: {err}"))?;
        let (s_col_prime, alpha_prime_nc) =
            split_vec(&nc_challenges, dims.ell_m).map_err(|err| format!("split_nc_challenges: {err}"))?;
        s_col_prime_vars = Some(s_col_prime);
        alpha_prime_nc_vars = Some(alpha_prime_nc);
        sumcheck_final_nc = Some(final_nc);
        checkpoint(cs, "nc_sumcheck")?;
        emit_aux_count("nc_sumcheck", cs.scalar_aux().len());
        Ok(())
    })?;

    let r_prime_vars = r_prime_vars.ok_or_else(|| "fe_sumcheck r_prime missing".to_string())?;
    let s_col_prime_vars = s_col_prime_vars.ok_or_else(|| "nc_sumcheck s_col_prime missing".to_string())?;
    let alpha_prime_vars = alpha_prime_vars.ok_or_else(|| "fe_sumcheck alpha_prime missing".to_string())?;
    let alpha_prime_nc_vars = alpha_prime_nc_vars.ok_or_else(|| "nc_sumcheck alpha_prime_nc missing".to_string())?;
    let sumcheck_final_fe = sumcheck_final_fe.ok_or_else(|| "fe_sumcheck final missing".to_string())?;
    let sumcheck_final_nc = sumcheck_final_nc.ok_or_else(|| "nc_sumcheck final missing".to_string())?;

    profile_stage("relation_digest", || {
        let fold_digest = transcript
            .digest32(cs.namespace(|| format!("chunk_{chunk_index}_fold_digest")))
            .map_err(|err| format!("fold_digest: {err}"))?;
        let chunk_relation_digest_input = next_public_digest(
            public_inputs,
            public_cursor,
            &format!("chunk_{chunk_index}_relation_digest"),
        )
        .map_err(|err| format!("chunk_relation_digest_input: {err}"))?;
        let chunk_relation_digest = chunk_relation_digest_circuit(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_relation_digest")),
            chunk.handoff.public_chunk_digest,
            &fold_digest,
            chunk.handoff.bridge_handoff_digest,
        )
        .map_err(|err| format!("chunk_relation_digest_circuit: {err}"))?;
        enforce_digest_eq(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_relation_digest_eq")),
            &chunk_relation_digest,
            &chunk_relation_digest_input,
            &format!("chunk_{chunk_index}_relation_digest_eq"),
        )
        .map_err(|err| format!("chunk_relation_digest_eq: {err}"))?;
        checkpoint(cs, "relation_digest")?;
        emit_aux_count("relation_digest", cs.scalar_aux().len());
        Ok(())
    })?;

    let mut padded_ccs_outputs = Vec::with_capacity(cover_chunk.ccs_output_shapes.len());
    profile_stage("ccs_outputs_and_binding", || {
        let effective_output_count = chunk.pi_ccs.ccs_outputs.len();
        for (output_index, shape) in cover_chunk.ccs_output_shapes.iter().enumerate() {
            let effective_claim = chunk.pi_ccs.ccs_outputs.get(output_index);
            let output = if output_index < active_fresh_claim_count {
                let claim = cover_ce_claim_with_shared_point(
                    shape,
                    effective_claim,
                    &chunk.pi_ccs.row_chals,
                    &chunk.pi_ccs.s_col,
                )
                .map_err(|err| format!("cover_fresh_output_{output_index}: {err}"))?;
                alloc_ce_claim_public_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                    &claim,
                    &r_prime_vars,
                    &chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &chunk.pi_ccs.s_col,
                    &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
                )
                .map_err(|err| format!("alloc_fresh_output_{output_index}: {err}"))?
            } else if output_index < effective_output_count {
                let claim = cover_ce_claim_with_shared_point(
                    shape,
                    effective_claim,
                    &chunk.pi_ccs.row_chals,
                    &chunk.pi_ccs.s_col,
                )
                .map_err(|err| format!("cover_ccs_output_{output_index}: {err}"))?;
                alloc_ce_claim_public_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                    &claim,
                    &r_prime_vars,
                    &chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &chunk.pi_ccs.s_col,
                    &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
                )
                .map_err(|err| format!("alloc_ccs_output_{output_index}: {err}"))?
            } else {
                let mut padded_claim = shape.zero_claim();
                padded_claim.r = chunk.pi_ccs.row_chals.clone();
                padded_claim.s_col = chunk.pi_ccs.s_col.clone();
                alloc_ce_claim_public_surface_with_shared_point(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_ccs_output_{output_index}")),
                    &padded_claim,
                    &r_prime_vars,
                    &chunk.pi_ccs.row_chals,
                    &s_col_prime_vars,
                    &chunk.pi_ccs.s_col,
                    &format!("chunk_{chunk_index}_ccs_output_{output_index}"),
                )
                .map_err(|err| format!("alloc_padded_ccs_output_{output_index}: {err}"))?
            };
            padded_ccs_outputs.push(output);
        }
        let ccs_outputs = padded_ccs_outputs.clone();
        enforce_me_outputs_against_inputs(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_output_binding")),
            structure,
            params,
            &covered_fresh_claim_vars,
            carried_claims.effective_claims(),
            &ccs_outputs,
            0,
            &r_prime_vars,
            &chunk.pi_ccs.row_chals,
            &s_col_prime_vars,
            &chunk.pi_ccs.s_col,
            &format!("chunk_{chunk_index}_output_binding"),
        )
        .map_err(|err| format!("output_binding: {err}"))?;
        checkpoint(cs, "output_binding")?;
        emit_aux_count("ccs_outputs_and_binding", cs.scalar_aux().len());
        Ok(())
    })?;

    profile_stage("terminal_identities", || {
        let effective_output_count = chunk.pi_ccs.ccs_outputs.len();
        let cover_fresh_claim_count = cover_chunk.fresh_claim_count as usize;
        let effective_me_output_count = effective_output_count
            .checked_sub(cover_fresh_claim_count)
            .ok_or_else(|| "effective_me_output_count_underflow".to_string())?;
        let effective_terminal_outputs = padded_ccs_outputs[..cover_fresh_claim_count]
            .iter()
            .cloned()
            .chain(
                padded_ccs_outputs[cover_fresh_claim_count..cover_fresh_claim_count + effective_me_output_count]
                    .iter()
                    .cloned(),
            )
            .collect::<Vec<_>>();
        let me_inputs_r_vars = carried_claims
            .effective_claims()
            .first()
            .map(|claim| claim.openings.r.as_slice());
        let me_inputs_r_values = carried_claims
            .effective_claims()
            .first()
            .map(|claim| claim.openings.r_values.as_slice());
        let _ = enforce_terminal_identity_fe(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_fe")),
            &sumcheck_final_fe,
            structure,
            &chunk.pi_ccs.public_challenges,
            &public_challenges.alpha,
            &public_challenges.beta_a,
            &public_challenges.beta_r,
            &public_challenges.gamma,
            &r_prime_vars,
            &chunk.pi_ccs.row_chals,
            &alpha_prime_vars,
            &chunk.pi_ccs.alpha_prime,
            &effective_terminal_outputs,
            cover_fresh_claim_count,
            0,
            me_inputs_r_vars,
            me_inputs_r_values,
            rv32im_main_relation_delta(),
            &format!("chunk_{chunk_index}_terminal_fe"),
        )
        .map_err(|err| format!("terminal_fe: {err}"))?;
        checkpoint(cs, "terminal_fe")?;
        let _ = enforce_terminal_identity_nc(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_nc")),
            &sumcheck_final_nc,
            params,
            &chunk.pi_ccs.public_challenges,
            &public_challenges.beta_a,
            &public_challenges.beta_m,
            &public_challenges.gamma,
            &s_col_prime_vars,
            &chunk.pi_ccs.s_col,
            &alpha_prime_nc_vars,
            &chunk.pi_ccs.alpha_prime_nc,
            &effective_terminal_outputs,
            cover_fresh_claim_count,
            0,
            rv32im_main_relation_delta(),
            &format!("chunk_{chunk_index}_terminal_nc"),
        )
        .map_err(|err| format!("terminal_nc: {err}"))?;
        checkpoint(cs, "terminal_nc")?;
        emit_aux_count("terminal_identities", cs.scalar_aux().len());
        Ok(())
    })?;

    let carry_terminal_state = matches!(
        (boundary_plan.child_claim_source, boundary_plan.next_carry_mode),
        (
            Rv32imChunkChildClaimSource::TerminalFinalClaims,
            Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren
        )
    );

    let mut parent_claim = None;
    let mut padded_child_claims = None;
    profile_stage("parent_and_child_claims", || {
        let parent = if carry_terminal_state {
            let claim = cover_ce_claim(&cover_chunk.parent_claim_shape, Some(&chunk.pi_rlc.parent))
                .map_err(|err| format!("cover_terminal_parent_claim: {err}"))?;
            alloc_ce_claim(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_parent_claim")),
                &claim,
                &format!("chunk_{chunk_index}_terminal_parent_claim"),
            )
            .map_err(|err| format!("alloc_terminal_parent_claim: {err}"))?
        } else {
            let claim = cover_ce_claim_with_shared_point(
                &cover_chunk.parent_claim_shape,
                Some(&chunk.pi_rlc.parent),
                &chunk.pi_ccs.row_chals,
                &chunk.pi_ccs.s_col,
            )
            .map_err(|err| format!("cover_parent_claim: {err}"))?;
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("chunk_{chunk_index}_parent_claim")),
                &claim,
                &r_prime_vars,
                &chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &chunk.pi_ccs.s_col,
                &format!("chunk_{chunk_index}_parent_claim"),
            )
            .map_err(|err| format!("alloc_parent_claim: {err}"))?
        };
        let child_claim_source = match boundary_plan.child_claim_source {
            Rv32imChunkChildClaimSource::ReplayedChildren => &chunk.pi_dec.children,
            Rv32imChunkChildClaimSource::TerminalFinalClaims => terminal_final_claims,
        };
        let children = cover_chunk
            .child_claim_shapes
            .iter()
            .enumerate()
            .map(|(child_index, shape)| {
                if carry_terminal_state {
                    let claim = cover_ce_claim(shape, child_claim_source.get(child_index))
                        .map_err(|err| format!("cover_terminal_child_claim_{child_index}: {err}"))?;
                    alloc_ce_claim_dec_surface(
                        &mut cs.namespace(|| format!("chunk_{chunk_index}_terminal_child_claim_{child_index}")),
                        &claim,
                        &format!("chunk_{chunk_index}_terminal_child_claim_{child_index}"),
                    )
                    .map_err(|err| format!("alloc_terminal_child_claim_{child_index}: {err}"))
                } else {
                    let claim = cover_ce_claim_with_shared_point(
                        shape,
                        child_claim_source.get(child_index),
                        &chunk.pi_ccs.row_chals,
                        &chunk.pi_ccs.s_col,
                    )
                    .map_err(|err| format!("cover_child_claim_{child_index}: {err}"))?;
                    alloc_ce_claim_dec_surface_with_shared_r(
                        &mut cs.namespace(|| format!("chunk_{chunk_index}_child_claim_{child_index}")),
                        &claim,
                        &r_prime_vars,
                        &chunk.pi_ccs.row_chals,
                        &format!("chunk_{chunk_index}_child_claim_{child_index}"),
                    )
                    .map_err(|err| format!("alloc_child_claim_{child_index}: {err}"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        parent_claim = Some(parent);
        padded_child_claims = Some(children);
        checkpoint(cs, "child_claims")
    })?;
    let parent_claim = parent_claim.ok_or_else(|| "parent claim missing".to_string())?;
    let padded_child_claims = padded_child_claims.ok_or_else(|| "child claims missing".to_string())?;

    let effective_child_count = match boundary_plan.child_claim_source {
        Rv32imChunkChildClaimSource::ReplayedChildren => chunk.pi_dec.children.len(),
        Rv32imChunkChildClaimSource::TerminalFinalClaims => terminal_final_claims.len(),
    };

    profile_stage("rlc_and_dec", || {
        let rho_vars = sample_goldilocks_rot_rhos(
            &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_rhos")),
            transcript,
            padded_ccs_outputs.len(),
            &format!("chunk_{chunk_index}_rlc_rhos"),
        )
        .map_err(|err| format!("sample_rlc_rhos: {err}"))?;
        match boundary_plan.rlc_mode {
            Rv32imChunkRlcMode::TerminalLastChunkShortcut => {
                let child_claim_source = match boundary_plan.child_claim_source {
                    Rv32imChunkChildClaimSource::ReplayedChildren => &chunk.pi_dec.children,
                    Rv32imChunkChildClaimSource::TerminalFinalClaims => terminal_final_claims,
                };
                enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_public")),
                    &parent_claim,
                    &padded_ccs_outputs,
                    child_claim_source,
                    &rho_vars,
                    params.b,
                    &format!("chunk_{chunk_index}_rlc_public"),
                )
                .map_err(|err| format!("rlc_public_last_chunk: {err}"))?;
            }
            Rv32imChunkRlcMode::Standard { constant_child_prefix } => {
                let active_dense_children_len = padded_ccs_outputs.len();
                let rho_mats = materialize_goldilocks_rot_matrices(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_rlc_rho_mats")),
                    &rho_vars[..active_dense_children_len],
                    &format!("chunk_{chunk_index}_rlc_rho_mats"),
                )
                .map_err(|err| format!("materialize_rlc_rho_mats: {err}"))?;
                crate::superneo_circuit::pi_rlc::debug_locate_rlc_public_with_split_rho_views_stage(
                    cs,
                    &parent_claim,
                    &padded_ccs_outputs,
                    &rho_vars,
                    &rho_mats,
                    constant_child_prefix,
                    0,
                    &format!("chunk_{chunk_index}_rlc_public"),
                )
                .map_err(|err| format!("rlc_public: {err}"))?;
            }
        }
        checkpoint(cs, "rlc_public")?;

        match boundary_plan.next_carry_mode {
            Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren => {
                enforce_dec_public(
                    &mut cs.namespace(|| format!("chunk_{chunk_index}_dec_public")),
                    &parent_claim,
                    &padded_child_claims[..effective_child_count],
                    params.b,
                    &format!("chunk_{chunk_index}_dec_public"),
                )
                .map_err(|err| format!("dec_public: {err}"))?;
                checkpoint(cs, "dec_public")?;
            }
            Rv32imChunkNextCarryMode::PreserveIncoming => {
                if !matches!(boundary_plan.rlc_mode, Rv32imChunkRlcMode::TerminalLastChunkShortcut) {
                    let child_claim_source = match boundary_plan.child_claim_source {
                        Rv32imChunkChildClaimSource::ReplayedChildren => &chunk.pi_dec.children,
                        Rv32imChunkChildClaimSource::TerminalFinalClaims => terminal_final_claims,
                    };
                    crate::superneo_circuit::pi_dec::enforce_dec_public_with_constant_children(
                        &mut cs.namespace(|| format!("chunk_{chunk_index}_dec_public")),
                        &parent_claim,
                        child_claim_source,
                        params.b,
                        &format!("chunk_{chunk_index}_dec_public"),
                    )
                    .map_err(|err| format!("dec_public_constant_children: {err}"))?;
                    checkpoint(cs, "dec_public_constant_children")?;
                }
            }
        }
        Ok(())
    })?;

    let next_carried_claims = match boundary_plan.next_carry_mode {
        Rv32imChunkNextCarryMode::ReplaceWithEffectiveChildren => {
            Rv32imClaimBundle::from_padded_claims(padded_child_claims, effective_child_count)
        }
        Rv32imChunkNextCarryMode::PreserveIncoming => carried_claims,
    };

    if append_chunk_done {
        profile_stage("chunk_done", || {
            transcript
                .append_const_fields_raw(
                    cs.namespace(|| format!("chunk_done_{chunk_index}")),
                    &[
                        SpartanF::from_canonical_u64(RV32IM_CHUNK_DONE_RAW_TAG),
                        SpartanF::from_canonical_u64(1),
                    ],
                )
                .map_err(|err| format!("chunk_done: {err}"))?;
            checkpoint(cs, "chunk_done")
        })?;
    }

    Ok(next_carried_claims)
}
