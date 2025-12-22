use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_reductions::sumcheck::{BatchedClaim, BatchedClaimResult, RoundOracle};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::memory::{RouteAMemoryOracles, ShoutRouteAProtocol, TimeBatchedClaims, TwistRouteAProtocol};
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::bind_batched_dynamic_claims;
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::shard_proof_types::BatchedTimeProof;
use crate::PiCcsError;

pub struct RouteABatchedTimeProverOutput {
    pub r_time: Vec<K>,
    pub per_claim_results: Vec<BatchedClaimResult>,
    pub proof: BatchedTimeProof,
}

pub fn prove_route_a_batched_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_n: usize,
    ccs_time_degree_bound: usize,
    ccs_initial_sum: K,
    ccs_oracle: &mut dyn RoundOracle,
    mem_oracles: &mut RouteAMemoryOracles,
    step: &StepWitnessBundle<Cmt, F, K>,
    twist_read_claims: Vec<K>,
    twist_write_claims: Vec<K>,
) -> Result<RouteABatchedTimeProverOutput, PiCcsError> {
    let mut claimed_sums: Vec<K> = Vec::new();
    let mut degree_bounds: Vec<usize> = Vec::new();
    let mut labels: Vec<&'static [u8]> = Vec::new();
    let mut claim_is_dynamic: Vec<bool> = Vec::new();
    let mut claims: Vec<BatchedClaim<'_>> = Vec::new();

    // CCS claim (time/row rounds only).
    let mut ccs_time = RoundOraclePrefix::new(ccs_oracle, ell_n);
    claimed_sums.push(ccs_initial_sum);
    degree_bounds.push(ccs_time.degree_bound());
    labels.push(b"ccs/time");
    // Keep CCS/time claimed sum in the dynamic-claim registry for transcript consistency.
    claim_is_dynamic.push(true);
    claims.push(BatchedClaim {
        oracle: &mut ccs_time,
        claimed_sum: ccs_initial_sum,
        label: b"ccs/time",
    });

    let mut shout_protocol = ShoutRouteAProtocol::new(&mut mem_oracles.shout, ell_n);
    shout_protocol.append_time_claims(
        ell_n,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
    );

    let mut twist_protocol =
        TwistRouteAProtocol::new(&mut mem_oracles.twist, ell_n, twist_read_claims, twist_write_claims);
    twist_protocol.append_time_claims(
        ell_n,
        &mut claimed_sums,
        &mut degree_bounds,
        &mut labels,
        &mut claim_is_dynamic,
        &mut claims,
    );

    let metas = RouteATimeClaimPlan::time_claim_metas_for_instances(
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
        ccs_time_degree_bound,
    );
    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let expected_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();

    if degree_bounds != expected_degree_bounds {
        return Err(PiCcsError::ProtocolError("batched time degree bounds drift".into()));
    }
    if labels != expected_labels {
        return Err(PiCcsError::ProtocolError("batched time labels drift".into()));
    }
    if claim_is_dynamic != expected_dynamic {
        return Err(PiCcsError::ProtocolError("batched time dynamic-flag drift".into()));
    }

    // Run batched sum-check prover (shared r_time challenges).
    bind_batched_dynamic_claims(tr, &claimed_sums, &labels, &degree_bounds, &claim_is_dynamic);
    let (r_time, per_claim_results) =
        run_batched_sumcheck_prover_ds(tr, b"shard/batched_time", step_idx, claims.as_mut_slice())?;

    if r_time.len() != ell_n {
        return Err(PiCcsError::ProtocolError(format!(
            "batched sumcheck returned r_time.len()={}, expected ell_n={ell_n}",
            r_time.len()
        )));
    }

    let proof = BatchedTimeProof {
        claimed_sums: claimed_sums.clone(),
        degree_bounds: degree_bounds.clone(),
        labels: labels.clone(),
        round_polys: per_claim_results
            .iter()
            .map(|r| r.round_polys.clone())
            .collect(),
    };

    Ok(RouteABatchedTimeProverOutput {
        r_time,
        per_claim_results,
        proof,
    })
}

pub struct RouteABatchedTimeVerifyOutput {
    pub r_time: Vec<K>,
    pub final_values: Vec<K>,
}

pub fn verify_route_a_batched_time(
    tr: &mut Poseidon2Transcript,
    step_idx: usize,
    ell_n: usize,
    ccs_time_degree_bound: usize,
    claimed_initial_sum: K,
    step: &StepInstanceBundle<Cmt, F, K>,
    proof: &BatchedTimeProof,
) -> Result<RouteABatchedTimeVerifyOutput, PiCcsError> {
    let metas = RouteATimeClaimPlan::time_claim_metas_for_step(step, ccs_time_degree_bound);
    let expected_degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
    let expected_labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
    let claim_is_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();

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
    if proof.claimed_sums.is_empty() || proof.claimed_sums[0] != claimed_initial_sum {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: batched_time claimed_sums[0] (CCS/time) != public initial sum",
            step_idx
        )));
    }
    for (i, (&sum, &dyn_ok)) in proof
        .claimed_sums
        .iter()
        .zip(claim_is_dynamic.iter())
        .enumerate()
    {
        if i == 0 {
            continue;
        }
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
        if (*got as &[u8]) != *exp {
            return Err(PiCcsError::ProtocolError(format!(
                "step {}: batched_time label mismatch at claim {}",
                step_idx, i
            )));
        }
    }

    // Verify the batched time/row sumcheck rounds (derives shared r_time).
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
    if r_time.len() != ell_n {
        return Err(PiCcsError::ProtocolError(format!(
            "step {}: r_time length mismatch (got {}, expected ell_n={})",
            step_idx,
            r_time.len(),
            ell_n
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
