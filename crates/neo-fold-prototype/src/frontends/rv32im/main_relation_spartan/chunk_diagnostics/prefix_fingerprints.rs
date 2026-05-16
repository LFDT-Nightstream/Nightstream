use std::fmt::Write as _;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;

use super::super::*;
use crate::rv32im::main_relation_spartan::fingerprint_cs::FingerprintCS;
use crate::rv32im::main_relation_spartan::recursive_cover::alloc_recursive_cover_claims;
use crate::rv32im::SimpleKernelError;
use crate::spartan_backend::SpartanF;

#[derive(Clone, Debug)]
pub struct Rv32imMainRelationStateInPrefixFingerprints {
    pub after_live_state_in_claim_alloc: String,
    pub after_live_state_in_claim_alloc_aux: usize,
    pub per_claim_compute: Vec<String>,
    pub bind_me_input_digests_compute: String,
    pub bind_me_input_digests_compute_aux: usize,
    pub bind_me_input_digests_transcript: String,
    pub bind_me_input_digests_transcript_aux: usize,
    pub claimed_initial_sum_from_me_inputs: String,
    pub claimed_initial_sum_from_me_inputs_aux: usize,
    pub fe_sumcheck_initial: String,
    pub fe_sumcheck_initial_aux: usize,
    pub fe_sumcheck: String,
    pub fe_sumcheck_aux: usize,
    pub nc_sumcheck_initial: String,
    pub nc_sumcheck_initial_aux: usize,
    pub nc_sumcheck: String,
    pub nc_sumcheck_aux: usize,
    pub relation_digest: String,
    pub relation_digest_aux: usize,
    pub ccs_outputs_and_binding: String,
    pub ccs_outputs_and_binding_aux: usize,
    pub terminal_identities: String,
    pub terminal_identities_aux: usize,
}

fn digest_hex(digest: [u8; 32]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn debug_measure_rv32im_main_relation_state_in_prefix_fingerprints(
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imMainRelationStateInPrefixFingerprints, SimpleKernelError> {
    let (params, _, structure) =
        rv32im_cached_root_main_lane_context().map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let optimized_cache =
        rv32im_cached_root_main_lane_optimized_cache().map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let dims = build_dims_and_policy(params, structure).map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let mat_digest: [Goldilocks; 4] = digest_ccs_matrices_with_sparse_cache(structure, Some(optimized_cache.sparse()))
        .try_into()
        .map_err(|_| {
            SimpleKernelError::Bridge("rv32im chunk prefix fingerprint requires 4-word matrix digest".into())
        })?;

    let witness = &backend_relation.f_prime_advice;
    let payload = &backend_relation.payload;
    let replay_chunk = payload.effective_chunk_replay_surface()?;

    let mut cs = FingerprintCS::new();
    let transcript_state = witness
        .running_state()
        .transcript
        .state
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| format!("transcript_state_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let transcript_state: [AllocatedNum<SpartanF>; neo_params::poseidon2_goldilocks::WIDTH] = transcript_state
        .try_into()
        .map_err(|_| SimpleKernelError::Bridge("rv32im chunk prefix fingerprint invalid transcript width".into()))?;

    let transcript_values = witness
        .running_state()
        .transcript
        .state
        .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()));

    let mut transcript = Poseidon2TranscriptCircuit::from_state(
        transcript_state,
        transcript_values,
        witness.running_state().transcript.absorbed,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    append_chunk_meta(
        &mut cs.namespace(|| "chunk_meta"),
        &mut transcript,
        &replay_chunk.handoff,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    bind_header_and_instance_digest(
        &mut cs.namespace(|| "bind_header"),
        &mut transcript,
        params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        &mat_digest,
        &replay_chunk
            .handoff
            .public_chunk_instance_digest
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;

    let live_state_in_claims = alloc_recursive_cover_claims(
        &mut cs.namespace(|| "state_in_live_claims"),
        &payload.state_in_claims,
        "state_in_live_claims",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let after_live_state_in_claim_alloc = digest_hex(cs.clone().finish_digest32(0));
    let carried_claims = Rv32imClaimBundle::from_effective_claims(
        live_state_in_claims
            .into_iter()
            .map(|claim| claim.claim)
            .collect(),
    );
    let after_live_state_in_claim_alloc_aux = cs.num_aux();

    let mut me_input_digests = Vec::with_capacity(carried_claims.effective_count());
    let mut me_input_digest_values = Vec::with_capacity(carried_claims.effective_count());
    let mut per_claim_compute = Vec::with_capacity(carried_claims.effective_count());
    for (idx, claim) in carried_claims.effective_claims().iter().enumerate() {
        me_input_digests.push(
            crate::superneo_circuit::claim::me_digest_poseidon(
                &mut cs.namespace(|| format!("me_input_digest_{idx}")),
                claim,
                &format!("me_input_digest_{idx}"),
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?,
        );
        me_input_digest_values.push(crate::superneo_circuit::claim::me_digest_poseidon_values(claim));
        per_claim_compute.push(digest_hex(cs.clone().finish_digest32(0)));
    }
    let bind_me_input_digests_compute = digest_hex(cs.clone().finish_digest32(0));
    let bind_me_input_digests_compute_aux = cs.num_aux();

    crate::superneo_circuit::pi_ccs::bind_me_input_digests(
        &mut cs.namespace(|| "bind_me_inputs"),
        &mut transcript,
        &me_input_digests,
        &me_input_digest_values,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let bind_me_input_digests_transcript = digest_hex(cs.clone().finish_digest32(0));
    let bind_me_input_digests_transcript_aux = cs.num_aux();

    let public_challenges = sample_challenges(&mut cs.namespace(|| "sample_challenges"), &mut transcript, dims)
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let effective_fresh_claim_count = replay_chunk.fresh_claims.len();
    let _ = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| "claimed_initial_sum_from_me_inputs"),
        structure,
        &public_challenges.alpha,
        &replay_chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        replay_chunk.pi_ccs.public_challenges.gamma,
        effective_fresh_claim_count,
        carried_claims.effective_claims(),
        0,
        rv32im_main_relation_delta(),
        "claimed_initial_sum_from_me_inputs",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let claimed_initial_sum_from_me_inputs_fingerprint = digest_hex(cs.clone().finish_digest32(0));
    let claimed_initial_sum_from_me_inputs_aux = cs.num_aux();

    let effective_fresh_claim_count = replay_chunk.fresh_claims.len();
    let covered_fresh_claims = payload
        .chunk_cover
        .fresh_claim_shapes
        .iter()
        .enumerate()
        .map(|(claim_index, shape)| cover_ccs_claim(shape, replay_chunk.fresh_claims.get(claim_index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let effective_fresh_claims = &covered_fresh_claims[..effective_fresh_claim_count];
    let effective_fresh_claim_vars = effective_fresh_claims
        .iter()
        .enumerate()
        .map(|(fresh_index, fresh)| {
            crate::superneo_circuit::output_binding::alloc_fresh_ccs_claim(
                &mut cs.namespace(|| format!("fresh_claim_{fresh_index}")),
                fresh,
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (initial_sum_fe, initial_sum_fe_value) = claimed_initial_sum_from_me_inputs(
        &mut cs.namespace(|| "initial_sum_fe"),
        structure,
        &public_challenges.alpha,
        &replay_chunk.pi_ccs.public_challenges.alpha,
        &public_challenges.gamma,
        replay_chunk.pi_ccs.public_challenges.gamma,
        effective_fresh_claim_count,
        carried_claims.effective_claims(),
        0,
        rv32im_main_relation_delta(),
        "initial_sum_fe",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;

    transcript
        .append_const_fields_raw(
            cs.namespace(|| "fe_sumcheck_domain"),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    if carried_claims.effective_count() == 0 {
        let coeffs = initial_sum_fe_value.as_coeffs();
        transcript
            .append_const_fields_raw(
                cs.namespace(|| "fe_sumcheck_initial_tag"),
                &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
        transcript
            .append_const_fields_raw(
                cs.namespace(|| "fe_sumcheck_initial_append"),
                &[
                    SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                ],
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    } else {
        append_k_to_transcript(
            &mut cs.namespace(|| "fe_sumcheck_initial"),
            &mut transcript,
            PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
            &initial_sum_fe,
            initial_sum_fe_value,
            "fe_sumcheck_initial",
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    }
    let fe_sumcheck_initial = digest_hex(cs.clone().finish_digest32(0));
    let fe_sumcheck_initial_aux = cs.num_aux();

    let padded_fe_rounds = alloc_rounds(
        &mut cs.namespace(|| "fe_rounds"),
        &payload.chunk_cover.fe_round_lengths,
        &replay_chunk.pi_ccs.replay_proof.sumcheck_rounds,
        "fe_round",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let fe_round_values = pad_round_values(
        &payload.chunk_cover.fe_round_lengths,
        &replay_chunk.pi_ccs.replay_proof.sumcheck_rounds,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let fe_challenge_values =
        chunk_sumcheck_challenges(&replay_chunk.pi_ccs.row_chals, &replay_chunk.pi_ccs.alpha_prime);
    let (fe_challenges, sumcheck_final_fe) = verify_sumcheck_rounds(
        &mut cs.namespace(|| "fe_sumcheck"),
        &mut transcript,
        max_degree_from_cover_round_lengths(&payload.chunk_cover.fe_round_lengths),
        &initial_sum_fe,
        &padded_fe_rounds,
        &fe_round_values,
        &fe_challenge_values,
        rv32im_main_relation_delta(),
        "fe_sumcheck",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let fe_sumcheck = digest_hex(cs.clone().finish_digest32(0));
    let fe_sumcheck_aux = cs.num_aux();
    let (r_prime_vars, alpha_prime_vars) =
        split_vec(&fe_challenges, dims.ell_n).map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;

    let zero_nc = alloc_constant_k(
        &mut cs.namespace(|| "initial_sum_nc_zero"),
        KNum::from_neo_k(K::ZERO),
        "initial_sum_nc_zero",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| "nc_sumcheck_domain"),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG)],
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| "nc_sumcheck_initial_tag"),
            &[SpartanF::from_canonical_u64(PI_CCS_SUMCHECK_INITIAL_RAW_TAG)],
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    transcript
        .append_const_fields_raw(
            cs.namespace(|| "nc_sumcheck_initial_append"),
            &[SpartanF::from_canonical_u64(0), SpartanF::from_canonical_u64(0)],
        )
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let nc_sumcheck_initial = digest_hex(cs.clone().finish_digest32(0));
    let nc_sumcheck_initial_aux = cs.num_aux();

    let padded_nc_rounds = alloc_rounds(
        &mut cs.namespace(|| "nc_rounds"),
        &payload.chunk_cover.nc_round_lengths,
        &replay_chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
        "nc_round",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let nc_round_values = pad_round_values(
        &payload.chunk_cover.nc_round_lengths,
        &replay_chunk.pi_ccs.replay_proof.sumcheck_rounds_nc,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let nc_challenge_values =
        chunk_sumcheck_challenges(&replay_chunk.pi_ccs.s_col, &replay_chunk.pi_ccs.alpha_prime_nc);
    let (nc_challenges, sumcheck_final_nc) = verify_sumcheck_rounds(
        &mut cs.namespace(|| "nc_sumcheck"),
        &mut transcript,
        max_degree_from_cover_round_lengths(&payload.chunk_cover.nc_round_lengths),
        &zero_nc,
        &padded_nc_rounds,
        &nc_round_values,
        &nc_challenge_values,
        rv32im_main_relation_delta(),
        "nc_sumcheck",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let nc_sumcheck = digest_hex(cs.clone().finish_digest32(0));
    let nc_sumcheck_aux = cs.num_aux();
    let (s_col_prime_vars, alpha_prime_nc_vars) =
        split_vec(&nc_challenges, dims.ell_m).map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;

    let fold_digest = transcript
        .digest32(cs.namespace(|| "fold_digest"))
        .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let chunk_relation_digest_input = alloc_private_field_values(
        &mut cs.namespace(|| "synthetic_chunk_relation_digest"),
        &digest32_as_spartan_fields(replay_chunk.handoff.chunk_relation_digest),
        "synthetic_chunk_relation_digest",
    )
    .and_then(|values| {
        values
            .try_into()
            .map_err(|_| bellpepper_core::SynthesisError::Unsatisfiable)
    })
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let chunk_relation_digest = chunk_relation_digest_circuit(
        &mut cs.namespace(|| "chunk_relation_digest"),
        replay_chunk.handoff.public_chunk_digest,
        &fold_digest,
        replay_chunk.handoff.bridge_handoff_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "chunk_relation_digest_eq"),
        &chunk_relation_digest,
        &chunk_relation_digest_input,
        "chunk_relation_digest_eq",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let relation_digest = digest_hex(cs.clone().finish_digest32(0));
    let relation_digest_aux = cs.num_aux();

    let effective_output_count = replay_chunk.pi_ccs.ccs_outputs.len();
    let mut padded_ccs_outputs = Vec::with_capacity(payload.chunk_cover.ccs_output_shapes.len());
    for (output_index, shape) in payload.chunk_cover.ccs_output_shapes.iter().enumerate() {
        let effective_claim = replay_chunk.pi_ccs.ccs_outputs.get(output_index);
        let output = if output_index < effective_fresh_claim_count {
            let claim = cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &replay_chunk.pi_ccs.row_chals,
                &replay_chunk.pi_ccs.s_col,
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("ccs_output_{output_index}")),
                &claim,
                &r_prime_vars,
                &replay_chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &replay_chunk.pi_ccs.s_col,
                &format!("ccs_output_{output_index}"),
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?
        } else if output_index < effective_output_count {
            let claim = cover_ce_claim_with_shared_point(
                shape,
                effective_claim,
                &replay_chunk.pi_ccs.row_chals,
                &replay_chunk.pi_ccs.s_col,
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("ccs_output_{output_index}")),
                &claim,
                &r_prime_vars,
                &replay_chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &replay_chunk.pi_ccs.s_col,
                &format!("ccs_output_{output_index}"),
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?
        } else {
            let mut padded_claim = shape.zero_claim();
            padded_claim.r = replay_chunk.pi_ccs.row_chals.clone();
            padded_claim.s_col = replay_chunk.pi_ccs.s_col.clone();
            alloc_ce_claim_public_surface_with_shared_point(
                &mut cs.namespace(|| format!("ccs_output_{output_index}")),
                &padded_claim,
                &r_prime_vars,
                &replay_chunk.pi_ccs.row_chals,
                &s_col_prime_vars,
                &replay_chunk.pi_ccs.s_col,
                &format!("ccs_output_{output_index}"),
            )
            .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?
        };
        padded_ccs_outputs.push(output);
    }
    let ccs_outputs = padded_ccs_outputs[..effective_output_count].to_vec();
    enforce_me_outputs_against_inputs(
        &mut cs.namespace(|| "output_binding"),
        structure,
        params,
        &effective_fresh_claim_vars,
        carried_claims.effective_claims(),
        &ccs_outputs,
        0,
        &r_prime_vars,
        &replay_chunk.pi_ccs.row_chals,
        &s_col_prime_vars,
        &replay_chunk.pi_ccs.s_col,
        "output_binding",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let ccs_outputs_and_binding = digest_hex(cs.clone().finish_digest32(0));
    let ccs_outputs_and_binding_aux = cs.num_aux();

    let me_inputs_r_vars = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r.as_slice());
    let me_inputs_r_values = carried_claims
        .effective_claims()
        .first()
        .map(|claim| claim.openings.r_values.as_slice());
    let _ = enforce_terminal_identity_fe(
        &mut cs.namespace(|| "terminal_fe"),
        &sumcheck_final_fe,
        structure,
        &replay_chunk.pi_ccs.public_challenges,
        &public_challenges.alpha,
        &public_challenges.beta_a,
        &public_challenges.beta_r,
        &public_challenges.gamma,
        &r_prime_vars,
        &replay_chunk.pi_ccs.row_chals,
        &alpha_prime_vars,
        &replay_chunk.pi_ccs.alpha_prime,
        &ccs_outputs,
        effective_fresh_claim_count,
        0,
        me_inputs_r_vars,
        me_inputs_r_values,
        rv32im_main_relation_delta(),
        "terminal_fe",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let _ = enforce_terminal_identity_nc(
        &mut cs.namespace(|| "terminal_nc"),
        &sumcheck_final_nc,
        params,
        &replay_chunk.pi_ccs.public_challenges,
        &public_challenges.beta_a,
        &public_challenges.beta_m,
        &public_challenges.gamma,
        &s_col_prime_vars,
        &replay_chunk.pi_ccs.s_col,
        &alpha_prime_nc_vars,
        &replay_chunk.pi_ccs.alpha_prime_nc,
        &ccs_outputs,
        effective_fresh_claim_count,
        0,
        rv32im_main_relation_delta(),
        "terminal_nc",
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))?;
    let terminal_identities = digest_hex(cs.clone().finish_digest32(0));
    let terminal_identities_aux = cs.num_aux();

    Ok(Rv32imMainRelationStateInPrefixFingerprints {
        after_live_state_in_claim_alloc,
        after_live_state_in_claim_alloc_aux,
        per_claim_compute,
        bind_me_input_digests_compute,
        bind_me_input_digests_compute_aux,
        bind_me_input_digests_transcript,
        bind_me_input_digests_transcript_aux,
        claimed_initial_sum_from_me_inputs: claimed_initial_sum_from_me_inputs_fingerprint,
        claimed_initial_sum_from_me_inputs_aux,
        fe_sumcheck_initial,
        fe_sumcheck_initial_aux,
        fe_sumcheck,
        fe_sumcheck_aux,
        nc_sumcheck_initial,
        nc_sumcheck_initial_aux,
        nc_sumcheck,
        nc_sumcheck_aux,
        relation_digest,
        relation_digest_aux,
        ccs_outputs_and_binding,
        ccs_outputs_and_binding_aux,
        terminal_identities,
        terminal_identities_aux,
    })
}
