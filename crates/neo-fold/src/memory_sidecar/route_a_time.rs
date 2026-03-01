use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_memory::ts_common as ts;
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::memory_sidecar::claim_plan::{poseidon_local_time_claim_metas, RouteATimeClaimPlan};
use crate::memory_sidecar::memory::{RouteAMemoryOracles, ShoutRouteAProtocol, TimeBatchedClaims, TwistRouteAProtocol};
use crate::memory_sidecar::route_a_compiler::shadow_assert_compiled_schedule_matches_metas;
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::bind_batched_dynamic_claims;
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::shard_proof_types::BatchedTimeProof;
use crate::PiCcsError;

pub struct RouteABatchedTimeProverOutput {
    pub r_time: Vec<K>,
    pub proof: BatchedTimeProof,
}

pub struct ExtraBatchedTimeClaim {
    pub oracle: Box<dyn RoundOracle>,
    pub claimed_sum: K,
    pub label: &'static [u8],
}

fn split_extra_claim(
    claim: Option<ExtraBatchedTimeClaim>,
) -> (Option<Box<dyn RoundOracle>>, Option<&'static [u8]>, Option<K>) {
    match claim {
        Some(extra) => (Some(extra.oracle), Some(extra.label), Some(extra.claimed_sum)),
        None => (None, None, None),
    }
}

fn append_optional_claim<'a>(
    oracle: &'a mut Option<Box<dyn RoundOracle>>,
    label: Option<&'static [u8]>,
    claimed_sum: Option<K>,
    is_dynamic: bool,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
    missing_label_msg: &'static str,
    missing_claimed_sum_msg: &'static str,
) {
    if let Some(oracle) = oracle.as_deref_mut() {
        let label = label.expect(missing_label_msg);
        let claimed_sum = claimed_sum.expect(missing_claimed_sum_msg);
        claimed_sums.push(claimed_sum);
        degree_bounds.push(oracle.degree_bound());
        labels.push(label);
        claim_is_dynamic.push(is_dynamic);
        claims.push(BatchedClaim {
            oracle,
            claimed_sum,
            label,
        });
    } else {
        debug_assert!(label.is_none(), "label present without oracle");
    }
}

pub fn prove_route_a_batched_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_t: usize,
    mem_oracles: &mut RouteAMemoryOracles,
    step: &StepWitnessBundle<Cmt, F, K>,
    twist_read_claims: Vec<K>,
    twist_write_claims: Vec<K>,
    wb_time_claim: Option<ExtraBatchedTimeClaim>,
    wp_time_claim: Option<ExtraBatchedTimeClaim>,
    decode_decode_fields_claim: Option<ExtraBatchedTimeClaim>,
    decode_decode_immediates_claim: Option<ExtraBatchedTimeClaim>,
    width_bitness_claim: Option<ExtraBatchedTimeClaim>,
    width_quiescence_claim: Option<ExtraBatchedTimeClaim>,
    width_selector_linkage_claim: Option<ExtraBatchedTimeClaim>,
    width_load_semantics_claim: Option<ExtraBatchedTimeClaim>,
    width_store_semantics_claim: Option<ExtraBatchedTimeClaim>,
    control_next_pc_linear_claim: Option<ExtraBatchedTimeClaim>,
    control_next_pc_control_claim: Option<ExtraBatchedTimeClaim>,
    control_branch_semantics_claim: Option<ExtraBatchedTimeClaim>,
    control_control_writeback_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_io_link_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_bitness_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_canonical_u64_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_sidecar_link_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_mode_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_link_cycle_inv_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_link_cycle_sum_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_cont_inv_claim: Option<ExtraBatchedTimeClaim>,
    poseidon_cont_sum_claim: Option<ExtraBatchedTimeClaim>,
    ob_inc_total: Option<ExtraBatchedTimeClaim>,
) -> Result<RouteABatchedTimeProverOutput, PiCcsError> {
    let mut claimed_sums: Vec<K> = Vec::new();
    let mut degree_bounds: Vec<usize> = Vec::new();
    let mut labels: Vec<&'static [u8]> = Vec::new();
    let mut claim_is_dynamic: Vec<bool> = Vec::new();
    let mut claims: Vec<BatchedClaim<'_>> = Vec::new();

    let mut shout_protocol =
        ShoutRouteAProtocol::new(&mut mem_oracles.shout, &mut mem_oracles.shout_gamma_groups, ell_t);
    shout_protocol.append_time_claims(
        ell_t,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
    )?;

    // Optional: event-table Shout linkage trace hash claim.
    let shout_event_trace_hash_claim = mem_oracles.shout_event_trace_hash.as_ref().map(|o| o.claim);
    let mut shout_event_trace_hash_prefix = mem_oracles
        .shout_event_trace_hash
        .as_mut()
        .map(|o| RoundOraclePrefix::new(o.oracle.as_mut(), ell_t));
    if let (Some(claim), Some(prefix)) = (shout_event_trace_hash_claim, shout_event_trace_hash_prefix.as_mut()) {
        claimed_sums.push(claim);
        degree_bounds.push(prefix.degree_bound());
        labels.push(b"shout/event_trace_hash");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: prefix,
            claimed_sum: claim,
            label: b"shout/event_trace_hash",
        });
    }

    let mut twist_protocol =
        TwistRouteAProtocol::new(&mut mem_oracles.twist, ell_t, twist_read_claims, twist_write_claims)?;
    twist_protocol.append_time_claims(
        ell_t,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
    )?;
    macro_rules! append_zero_optional_claim {
        ($claim_opt:ident, $degree_bound:ident, $oracle:ident, $label:ident, $missing_label_msg:literal, $missing_claimed_sum_msg:literal) => {
            let $degree_bound = $claim_opt.as_ref().map(|extra| extra.oracle.degree_bound());
            let (mut $oracle, $label, _claimed_sum) = split_extra_claim($claim_opt);
            append_optional_claim(
                &mut $oracle,
                $label,
                Some(K::ZERO),
                false,
                &mut claimed_sums,
                &mut degree_bounds,
                &mut labels,
                &mut claim_is_dynamic,
                &mut claims,
                $missing_label_msg,
                $missing_claimed_sum_msg,
            );
        };
    }

    macro_rules! append_dynamic_optional_claim {
        ($claim_opt:ident, $degree_bound:ident, $oracle:ident, $label:ident, $claimed_sum:ident, $missing_label_msg:literal, $missing_claimed_sum_msg:literal) => {
            let $degree_bound = $claim_opt.as_ref().map(|extra| extra.oracle.degree_bound());
            let (mut $oracle, $label, $claimed_sum) = split_extra_claim($claim_opt);
            append_optional_claim(
                &mut $oracle,
                $label,
                $claimed_sum,
                true,
                &mut claimed_sums,
                &mut degree_bounds,
                &mut labels,
                &mut claim_is_dynamic,
                &mut claims,
                $missing_label_msg,
                $missing_claimed_sum_msg,
            );
        };
    }

    append_zero_optional_claim!(
        wb_time_claim,
        wb_time_degree_bound,
        wb_time_oracle,
        wb_time_label,
        "missing wb_time label",
        "missing wb_time claimed_sum"
    );
    append_zero_optional_claim!(
        wp_time_claim,
        wp_time_degree_bound,
        wp_time_oracle,
        wp_time_label,
        "missing wp_time label",
        "missing wp_time claimed_sum"
    );
    append_zero_optional_claim!(
        decode_decode_fields_claim,
        decode_decode_fields_degree_bound,
        decode_decode_fields_oracle,
        decode_decode_fields_label,
        "missing decode_fields label",
        "missing decode_fields claimed_sum"
    );
    append_zero_optional_claim!(
        decode_decode_immediates_claim,
        decode_decode_immediates_degree_bound,
        decode_decode_immediates_oracle,
        decode_decode_immediates_label,
        "missing decode_immediates label",
        "missing decode_immediates claimed_sum"
    );
    append_zero_optional_claim!(
        width_bitness_claim,
        width_bitness_degree_bound,
        width_bitness_oracle,
        width_bitness_label,
        "missing width_bitness label",
        "missing width_bitness claimed_sum"
    );
    append_zero_optional_claim!(
        width_quiescence_claim,
        width_quiescence_degree_bound,
        width_quiescence_oracle,
        width_quiescence_label,
        "missing width_quiescence label",
        "missing width_quiescence claimed_sum"
    );
    append_zero_optional_claim!(
        width_selector_linkage_claim,
        width_selector_linkage_degree_bound,
        width_selector_linkage_oracle,
        width_selector_linkage_label,
        "missing width_selector_linkage label",
        "missing width_selector_linkage claimed_sum"
    );
    append_zero_optional_claim!(
        width_load_semantics_claim,
        width_load_semantics_degree_bound,
        width_load_semantics_oracle,
        width_load_semantics_label,
        "missing width_load_semantics label",
        "missing width_load_semantics claimed_sum"
    );
    append_zero_optional_claim!(
        width_store_semantics_claim,
        width_store_semantics_degree_bound,
        width_store_semantics_oracle,
        width_store_semantics_label,
        "missing width_store_semantics label",
        "missing width_store_semantics claimed_sum"
    );
    append_zero_optional_claim!(
        control_next_pc_linear_claim,
        control_next_pc_linear_degree_bound,
        control_next_pc_linear_oracle,
        control_next_pc_linear_label,
        "missing control_next_pc_linear label",
        "missing control_next_pc_linear claimed_sum"
    );
    append_zero_optional_claim!(
        control_next_pc_control_claim,
        control_next_pc_control_degree_bound,
        control_next_pc_control_oracle,
        control_next_pc_control_label,
        "missing control_next_pc_control label",
        "missing control_next_pc_control claimed_sum"
    );
    append_zero_optional_claim!(
        control_branch_semantics_claim,
        control_branch_semantics_degree_bound,
        control_branch_semantics_oracle,
        control_branch_semantics_label,
        "missing control_branch_semantics label",
        "missing control_branch_semantics claimed_sum"
    );
    append_zero_optional_claim!(
        control_control_writeback_claim,
        control_control_writeback_degree_bound,
        control_control_writeback_oracle,
        control_control_writeback_label,
        "missing control_writeback label",
        "missing control_writeback claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_io_link_claim,
        poseidon_io_link_degree_bound,
        poseidon_io_link_oracle,
        poseidon_io_link_label,
        "missing poseidon/io_link label",
        "missing poseidon/io_link claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_bitness_claim,
        poseidon_bitness_degree_bound,
        poseidon_bitness_oracle,
        poseidon_bitness_label,
        "missing poseidon/bitness label",
        "missing poseidon/bitness claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_canonical_u64_claim,
        poseidon_canonical_u64_degree_bound,
        poseidon_canonical_u64_oracle,
        poseidon_canonical_u64_label,
        "missing poseidon/canonical_u64 label",
        "missing poseidon/canonical_u64 claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_sidecar_link_claim,
        poseidon_sidecar_link_degree_bound,
        poseidon_sidecar_link_oracle,
        poseidon_sidecar_link_label,
        "missing poseidon/sidecar_link label",
        "missing poseidon/sidecar_link claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_mode_claim,
        poseidon_mode_degree_bound,
        poseidon_mode_oracle,
        poseidon_mode_label,
        "missing poseidon/mode label",
        "missing poseidon/mode claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_link_cycle_inv_claim,
        poseidon_link_cycle_inv_degree_bound,
        poseidon_link_cycle_inv_oracle,
        poseidon_link_cycle_inv_label,
        "missing poseidon/link_cycle_inv label",
        "missing poseidon/link_cycle_inv claimed_sum"
    );
    append_dynamic_optional_claim!(
        poseidon_link_cycle_sum_claim,
        poseidon_link_cycle_sum_degree_bound,
        poseidon_link_cycle_sum_oracle,
        poseidon_link_cycle_sum_label,
        poseidon_link_cycle_sum_claimed_sum,
        "missing poseidon/link_cycle_sum label",
        "missing poseidon/link_cycle_sum claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_cont_inv_claim,
        poseidon_cont_inv_degree_bound,
        poseidon_cont_inv_oracle,
        poseidon_cont_inv_label,
        "missing poseidon/cont_inv label",
        "missing poseidon/cont_inv claimed_sum"
    );
    append_zero_optional_claim!(
        poseidon_cont_sum_claim,
        poseidon_cont_sum_degree_bound,
        poseidon_cont_sum_oracle,
        poseidon_cont_sum_label,
        "missing poseidon/cont_sum label",
        "missing poseidon/cont_sum claimed_sum"
    );
    append_dynamic_optional_claim!(
        ob_inc_total,
        ob_inc_total_degree_bound,
        ob_inc_total_oracle,
        ob_inc_total_label,
        ob_inc_total_claimed_sum,
        "missing ob_inc_total label",
        "missing ob_inc_total claimed_sum"
    );

    let metas = RouteATimeClaimPlan::time_claim_metas_for_instances(
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
        wb_time_degree_bound.is_some(),
        wp_time_degree_bound.is_some(),
        decode_decode_fields_degree_bound.is_some() || decode_decode_immediates_degree_bound.is_some(),
        width_bitness_degree_bound.is_some()
            || width_quiescence_degree_bound.is_some()
            || width_selector_linkage_degree_bound.is_some()
            || width_load_semantics_degree_bound.is_some()
            || width_store_semantics_degree_bound.is_some(),
        control_next_pc_linear_degree_bound.is_some()
            || control_next_pc_control_degree_bound.is_some()
            || control_branch_semantics_degree_bound.is_some()
            || control_control_writeback_degree_bound.is_some(),
        poseidon_io_link_degree_bound.is_some()
            || poseidon_bitness_degree_bound.is_some()
            || poseidon_canonical_u64_degree_bound.is_some()
            || poseidon_sidecar_link_degree_bound.is_some()
            || poseidon_mode_degree_bound.is_some()
            || poseidon_link_cycle_inv_degree_bound.is_some()
            || poseidon_link_cycle_sum_degree_bound.is_some()
            || poseidon_cont_inv_degree_bound.is_some()
            || poseidon_cont_sum_degree_bound.is_some(),
        ob_inc_total_degree_bound,
    );
    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let expected_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();

    // Phase-1 Route-A claim compiler shadow check:
    // validate that legacy assembled schedule matches canonical compiled schedule.
    shadow_assert_compiled_schedule_matches_metas(
        &labels,
        &degree_bounds,
        &claim_is_dynamic,
        &metas,
        "prove/route_a_batched_time",
    )?;

    if degree_bounds != expected_degree_bounds {
        return Err(PiCcsError::ProtocolError("batched time degree bounds drift".into()));
    }
    if labels != expected_labels {
        return Err(PiCcsError::ProtocolError("batched time labels drift".into()));
    }
    if claim_is_dynamic != expected_dynamic {
        return Err(PiCcsError::ProtocolError("batched time dynamic-flag drift".into()));
    }

    let (r_time, per_claim_results) = if claims.is_empty() {
        (
            ts::sample_ext_point(
                tr,
                b"shard/batched_time/empty/r_time",
                b"shard/batched_time/empty/r_time/0",
                b"shard/batched_time/empty/r_time/1",
                ell_t,
            ),
            Vec::new(),
        )
    } else {
        // Run batched sum-check prover (shared r_time challenges).
        bind_batched_dynamic_claims(tr, &claimed_sums, &labels, &degree_bounds, &claim_is_dynamic);
        let (r_time, per_claim_results) =
            run_batched_sumcheck_prover_ds(tr, b"shard/batched_time", step_idx, claims.as_mut_slice())?;
        (r_time, per_claim_results)
    };

    if r_time.len() != ell_t {
        return Err(PiCcsError::ProtocolError(format!(
            "batched sumcheck returned r_time.len()={}, expected ell_t={ell_t}",
            r_time.len()
        )));
    }

    let proof = BatchedTimeProof {
        claimed_sums: claimed_sums.clone(),
        degree_bounds: degree_bounds.clone(),
        labels: labels.iter().map(|label| label.to_vec()).collect(),
        round_polys: per_claim_results
            .into_iter()
            .map(|r| r.round_polys)
            .collect(),
    };

    Ok(RouteABatchedTimeProverOutput { r_time, proof })
}

pub struct RouteABatchedTimeVerifyOutput {
    pub r_time: Vec<K>,
    pub final_values: Vec<K>,
}

pub fn verify_route_a_batched_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_t: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    proof: &BatchedTimeProof,
    wb_enabled: bool,
    wp_enabled: bool,
    decode_stage_enabled: bool,
    width_stage_enabled: bool,
    control_stage_enabled: bool,
    poseidon_cycle_enabled: bool,
    ob_inc_total_degree_bound: Option<usize>,
) -> Result<RouteABatchedTimeVerifyOutput, PiCcsError> {
    let metas = RouteATimeClaimPlan::time_claim_metas_for_step(
        step,
        wb_enabled,
        wp_enabled,
        decode_stage_enabled,
        width_stage_enabled,
        control_stage_enabled,
        poseidon_cycle_enabled,
        ob_inc_total_degree_bound,
    );
    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let claim_is_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();
    let proof_labels_static: Vec<&'static [u8]> = proof
        .labels
        .iter()
        .map(|label| {
            metas
                .iter()
                .find(|meta| meta.label == label.as_slice())
                .map(|meta| meta.label)
                .ok_or_else(|| {
                    PiCcsError::ProtocolError(format!("verify/route_a_batched_time: unknown proof label {:?}", label))
                })
        })
        .collect::<Result<_, _>>()?;

    // Phase-1 Route-A claim compiler shadow check:
    // verifier-side canonical schedule reconstruction must match proof-carried schedule metadata.
    shadow_assert_compiled_schedule_matches_metas(
        &proof_labels_static,
        &proof.degree_bounds,
        &claim_is_dynamic,
        &metas,
        "verify/route_a_batched_time",
    )?;

    let expected_claims = claim_is_dynamic.len();
    if proof.round_polys.len() != expected_claims {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: batched_time claim count mismatch (expected {}, got {})",
            step_idx,
            expected_claims,
            proof.round_polys.len()
        )));
    }
    if proof.claimed_sums.len() != expected_claims {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: batched_time claimed_sums.len() mismatch (expected {}, got {})",
            step_idx,
            expected_claims,
            proof.claimed_sums.len()
        )));
    }
    for (i, (&sum, &dyn_ok)) in proof
        .claimed_sums
        .iter()
        .zip(claim_is_dynamic.iter())
        .enumerate()
    {
        if !dyn_ok && sum != K::ZERO {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time claimed_sums[{}] must be 0 (label {:?})",
                step_idx, i, expected_labels[i]
            )));
        }
    }
    if proof.degree_bounds != expected_degree_bounds {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: batched_time degree_bounds mismatch",
            step_idx
        )));
    }
    if proof.labels.len() != expected_labels.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: batched_time labels length mismatch",
            step_idx
        )));
    }
    for (i, (got, exp)) in proof.labels.iter().zip(expected_labels.iter()).enumerate() {
        if got.as_slice() != *exp {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time label mismatch at claim {}",
                step_idx, i
            )));
        }
    }

    let (r_time, final_values) = if expected_claims == 0 {
        (
            ts::sample_ext_point(
                tr,
                b"shard/batched_time/empty/r_time",
                b"shard/batched_time/empty/r_time/0",
                b"shard/batched_time/empty/r_time/1",
                ell_t,
            ),
            Vec::new(),
        )
    } else {
        // Verify the batched time sumcheck rounds (derives shared r_time).
        bind_batched_dynamic_claims(
            tr,
            &proof.claimed_sums,
            &expected_labels,
            &expected_degree_bounds,
            &claim_is_dynamic,
        );
        let (r_time, final_values, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"shard/batched_time",
            step_idx,
            &proof.round_polys,
            &proof.claimed_sums,
            &expected_labels,
            &expected_degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "batched time sumcheck verification failed".into(),
            ));
        }
        (r_time, final_values)
    };

    if r_time.len() != ell_t {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: r_time length mismatch (got {}, expected ell_t={})",
            step_idx,
            r_time.len(),
            ell_t
        )));
    }
    if final_values.len() != expected_claims {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: batched final_values length mismatch",
            step_idx
        )));
    }

    Ok(RouteABatchedTimeVerifyOutput { r_time, final_values })
}

pub struct PoseidonLocalTimeProverOutput {
    pub r_local: Vec<K>,
    pub proof: BatchedTimeProof,
}

pub struct PoseidonLocalTimeVerifyOutput {
    pub r_local: Vec<K>,
    pub final_values: Vec<K>,
}

pub fn prove_poseidon_local_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_local: usize,
    round_claim: Option<ExtraBatchedTimeClaim>,
    transition_claim: Option<ExtraBatchedTimeClaim>,
    cycle_local_link_claim: Option<ExtraBatchedTimeClaim>,
    link_local_inv_claim: Option<ExtraBatchedTimeClaim>,
    link_local_sum_claim: Option<ExtraBatchedTimeClaim>,
) -> Result<PoseidonLocalTimeProverOutput, PiCcsError> {
    let metas = poseidon_local_time_claim_metas();
    let mut claimed_sums: Vec<K> = Vec::new();
    let mut degree_bounds: Vec<usize> = Vec::new();
    let mut labels: Vec<&'static [u8]> = Vec::new();
    let mut claim_is_dynamic: Vec<bool> = Vec::new();
    let mut claims: Vec<BatchedClaim<'_>> = Vec::new();

    let (mut round_oracle, round_label, _) = split_extra_claim(round_claim);
    append_optional_claim(
        &mut round_oracle,
        round_label,
        Some(K::ZERO),
        false,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
        "missing poseidon/round label",
        "missing poseidon local claimed_sum",
    );

    let (mut transition_oracle, transition_label, _) = split_extra_claim(transition_claim);
    append_optional_claim(
        &mut transition_oracle,
        transition_label,
        Some(K::ZERO),
        false,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
        "missing poseidon/transition label",
        "missing poseidon local claimed_sum",
    );

    let (mut link_oracle, link_label, _) = split_extra_claim(cycle_local_link_claim);
    append_optional_claim(
        &mut link_oracle,
        link_label,
        Some(K::ZERO),
        false,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
        "missing poseidon/cycle_local_link label",
        "missing poseidon local claimed_sum",
    );

    let (mut link_local_inv_oracle, link_local_inv_label, _) = split_extra_claim(link_local_inv_claim);
    append_optional_claim(
        &mut link_local_inv_oracle,
        link_local_inv_label,
        Some(K::ZERO),
        false,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
        "missing poseidon/link_local_inv label",
        "missing poseidon local claimed_sum",
    );

    let (mut link_local_sum_oracle, link_local_sum_label, link_local_sum_claimed_sum) =
        split_extra_claim(link_local_sum_claim);
    append_optional_claim(
        &mut link_local_sum_oracle,
        link_local_sum_label,
        link_local_sum_claimed_sum,
        true,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
        "missing poseidon/link_local_sum label",
        "missing poseidon local claimed_sum",
    );

    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let expected_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();
    if degree_bounds != expected_degree_bounds {
        return Err(PiCcsError::ProtocolError("poseidon local degree bounds drift".into()));
    }
    if labels != expected_labels {
        return Err(PiCcsError::ProtocolError("poseidon local labels drift".into()));
    }
    if claim_is_dynamic != expected_dynamic {
        return Err(PiCcsError::ProtocolError("poseidon local dynamic-flag drift".into()));
    }

    bind_batched_dynamic_claims(tr, &claimed_sums, &labels, &degree_bounds, &claim_is_dynamic);
    let (r_local, per_claim_results) =
        run_batched_sumcheck_prover_ds(tr, b"shard/poseidon_local_time", step_idx, claims.as_mut_slice())?;
    if r_local.len() != ell_local {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local sumcheck returned r_local.len()={}, expected ell_local={ell_local}",
            r_local.len()
        )));
    }

    let proof = BatchedTimeProof {
        claimed_sums: claimed_sums.clone(),
        degree_bounds: degree_bounds.clone(),
        labels: labels.iter().map(|label| label.to_vec()).collect(),
        round_polys: per_claim_results
            .iter()
            .map(|r| r.round_polys.clone())
            .collect(),
    };

    Ok(PoseidonLocalTimeProverOutput { r_local, proof })
}

pub fn verify_poseidon_local_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_local: usize,
    proof: &BatchedTimeProof,
) -> Result<PoseidonLocalTimeVerifyOutput, PiCcsError> {
    let metas = poseidon_local_time_claim_metas();
    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let claim_is_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();
    let expected_claims = metas.len();

    if proof.round_polys.len() != expected_claims {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: poseidon_local_time claim count mismatch (expected {}, got {})",
            step_idx,
            expected_claims,
            proof.round_polys.len()
        )));
    }
    if proof.claimed_sums.len() != expected_claims {
        return Err(PiCcsError::InvalidInput(format!(
            "step {}: poseidon_local_time claimed_sums.len() mismatch (expected {}, got {})",
            step_idx,
            expected_claims,
            proof.claimed_sums.len()
        )));
    }
    if proof.degree_bounds != expected_degree_bounds {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: poseidon_local_time degree_bounds mismatch",
            step_idx
        )));
    }
    if proof.labels.len() != expected_labels.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: poseidon_local_time labels length mismatch",
            step_idx
        )));
    }
    for (i, (got, exp)) in proof.labels.iter().zip(expected_labels.iter()).enumerate() {
        if got.as_slice() != *exp {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: poseidon_local_time label mismatch at claim {}",
                step_idx, i
            )));
        }
    }
    for (i, (&sum, &dyn_ok)) in proof
        .claimed_sums
        .iter()
        .zip(claim_is_dynamic.iter())
        .enumerate()
    {
        if !dyn_ok && sum != K::ZERO {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: poseidon_local_time claimed_sums[{}] must be 0 (label {:?})",
                step_idx, i, expected_labels[i]
            )));
        }
    }

    bind_batched_dynamic_claims(
        tr,
        &proof.claimed_sums,
        &expected_labels,
        &expected_degree_bounds,
        &claim_is_dynamic,
    );
    let (r_local, final_values, ok) = verify_batched_sumcheck_rounds_ds(
        tr,
        b"shard/poseidon_local_time",
        step_idx,
        &proof.round_polys,
        &proof.claimed_sums,
        &expected_labels,
        &expected_degree_bounds,
    );
    if !ok {
        return Err(PiCcsError::SumcheckError(
            "poseidon local time sumcheck verification failed".into(),
        ));
    }
    if r_local.len() != ell_local {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: poseidon_local_time r_local length mismatch (got {}, expected {})",
            step_idx,
            r_local.len(),
            ell_local
        )));
    }
    if final_values.len() != expected_claims {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: poseidon_local_time final_values length mismatch",
            step_idx
        )));
    }

    Ok(PoseidonLocalTimeVerifyOutput { r_local, final_values })
}
