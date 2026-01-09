//! Route-A batched time verification (Phase 2 scaffolding).
//!
//! Mirrors `neo_fold::memory_sidecar::route_a_time::verify_route_a_batched_time`.

use bellpepper_core::ConstraintSystem;

use crate::circuit::fold_circuit_helpers as helpers;
use crate::error::{Result, SpartanBridgeError};
use crate::gadgets::k_field::KNumVar;
use crate::gadgets::sumcheck::verify_batched_sumcheck_rounds_ds;
use crate::gadgets::transcript::Poseidon2TranscriptVar;
use crate::CircuitF;

use neo_fold::shard::BatchedTimeProof;
use neo_math::{K as NeoK, KExtensions};
use p3_field::PrimeCharacteristicRing;

pub struct RouteABatchedTimeOutVars {
    pub r_time: Vec<KNumVar>,
    pub final_values: Vec<KNumVar>,
    pub claimed_sums_vars: Vec<KNumVar>,
    /// Allocated per-claim round polynomials as K variables (for in-circuit consistency checks).
    pub round_polys_vars: Vec<Vec<Vec<KNumVar>>>,
}

fn bind_batched_dynamic_claims<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    claimed_sums_vars: &[KNumVar],
    claimed_sums_vals: &[NeoK],
    labels: &[&'static [u8]],
    degree_bounds: &[usize],
    claim_is_dynamic: &[bool],
    ctx: &str,
) -> Result<()> {
    if claimed_sums_vars.len() != claimed_sums_vals.len()
        || claimed_sums_vars.len() != labels.len()
        || claimed_sums_vars.len() != degree_bounds.len()
        || claimed_sums_vars.len() != claim_is_dynamic.len()
    {
        return Err(SpartanBridgeError::InvalidInput(
            "bind_batched_dynamic_claims: length mismatch".into(),
        ));
    }

    tr.append_message(
        cs,
        b"batched/dynamic_bind/len",
        &(claimed_sums_vars.len() as u64).to_le_bytes(),
        ctx,
    )?;

    for (idx, (((sum_var, &sum_val), label), (&deg, &dyn_ok))) in claimed_sums_vars
        .iter()
        .zip(claimed_sums_vals.iter())
        .zip(labels.iter())
        .zip(degree_bounds.iter().zip(claim_is_dynamic.iter()))
        .enumerate()
    {
        tr.append_message(
            cs,
            b"batched/dynamic_bind/claim_label",
            label,
            &format!("{ctx}_dyn_bind_{idx}"),
        )?;
        tr.append_message(
            cs,
            b"batched/dynamic_bind/claim_idx",
            &(idx as u64).to_le_bytes(),
            &format!("{ctx}_dyn_bind_{idx}"),
        )?;
        tr.append_message(
            cs,
            b"batched/dynamic_bind/degree_bound",
            &(deg as u64).to_le_bytes(),
            &format!("{ctx}_dyn_bind_{idx}"),
        )?;
        tr.append_message(
            cs,
            b"batched/dynamic_bind/is_dynamic",
            &[dyn_ok as u8],
            &format!("{ctx}_dyn_bind_{idx}"),
        )?;

        if !dyn_ok {
            continue;
        }

        let coeffs = sum_val.as_coeffs();
        tr.append_fields_vars(
            cs,
            b"batched/dynamic_bind/claimed_sum",
            &[sum_var.c0, sum_var.c1],
            &[
                helpers::neo_f_to_circuit(&coeffs[0]),
                helpers::neo_f_to_circuit(&coeffs[1]),
            ],
            &format!("{ctx}_dyn_bind_{idx}_claimed_sum"),
        )?;
    }

    Ok(())
}

/// Verify the Route-A batched time proof inside the circuit, producing the shared `r_time` point.
///
/// This is a Phase 2 building block; higher-level step verification additionally checks:
/// - `r_time` matches the Π-CCS time-round challenges prefix
/// - CCS Ajtai rounds and terminal identity after the time batch
pub fn verify_route_a_batched_time_step<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    step_idx: usize,
    ell_n: usize,
    delta: CircuitF,
    claimed_initial_sum_var: &KNumVar,
    claimed_initial_sum_val: NeoK,
    proof: &BatchedTimeProof,
    expected_labels: &[&'static [u8]],
    expected_degree_bounds: &[usize],
    claim_is_dynamic: &[bool],
    ctx: &str,
) -> Result<RouteABatchedTimeOutVars> {
    if proof.round_polys.len() != expected_labels.len()
        || proof.claimed_sums.len() != expected_labels.len()
        || expected_degree_bounds.len() != expected_labels.len()
        || claim_is_dynamic.len() != expected_labels.len()
    {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: claim meta length mismatch"
        )));
    }

    if proof.claimed_sums.is_empty() || proof.claimed_sums[0] != claimed_initial_sum_val {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: claimed_sums[0] != claimed_initial_sum"
        )));
    }

    // Allocate claimed sums and bind claim0 to the computed initial sum.
    let mut claimed_sums_vars = Vec::with_capacity(proof.claimed_sums.len());
    for (i, &sum) in proof.claimed_sums.iter().enumerate() {
        let var = helpers::alloc_k_from_neo(cs, sum, &format!("{ctx}_claimed_sum_{i}"))?;
        claimed_sums_vars.push(var);
    }
    helpers::enforce_k_eq(
        cs,
        &claimed_sums_vars[0],
        claimed_initial_sum_var,
        &format!("{ctx}_claimed_sum0_matches_T"),
    );

    // Host-side check for non-dynamic claims (avoid polluting R1CS with policy logic).
    for (i, (&dyn_ok, &sum)) in claim_is_dynamic.iter().zip(proof.claimed_sums.iter()).enumerate() {
        if i == 0 {
            continue;
        }
        if !dyn_ok && sum != NeoK::ZERO {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: claimed_sums[{i}] must be 0 (label {:?})",
                expected_labels[i]
            )));
        }
    }

    // Host-side check that proof metadata matches expected claim plan.
    if proof.degree_bounds != expected_degree_bounds {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: degree_bounds mismatch"
        )));
    }
    if proof.labels.len() != expected_labels.len() {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: labels length mismatch"
        )));
    }
    for (i, (got, exp)) in proof.labels.iter().zip(expected_labels.iter()).enumerate() {
        if (*got as &[u8]) != *exp {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: label mismatch at claim {i}"
            )));
        }
    }

    // Allocate batched round polys.
    let mut round_polys_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(proof.round_polys.len());
    for (claim_idx, claim_rounds) in proof.round_polys.iter().enumerate() {
        let mut claim_vars = Vec::with_capacity(claim_rounds.len());
        for (round_idx, round_poly) in claim_rounds.iter().enumerate() {
            let mut coeff_vars = Vec::with_capacity(round_poly.len());
            for (coeff_idx, &coeff) in round_poly.iter().enumerate() {
                coeff_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    coeff,
                    &format!("{ctx}_claim_{claim_idx}_round_{round_idx}_coeff_{coeff_idx}"),
                )?);
            }
            claim_vars.push(coeff_vars);
        }
        round_polys_vars.push(claim_vars);
    }

    // Bind dynamic claimed sums to the transcript (must happen before the batched sumcheck).
    bind_batched_dynamic_claims(
        cs,
        tr,
        &claimed_sums_vars,
        &proof.claimed_sums,
        expected_labels,
        expected_degree_bounds,
        claim_is_dynamic,
        &format!("{ctx}_dyn_bind"),
    )?;

    // Verify the batched sumcheck rounds (derives shared r_time challenges).
    let (r_time, final_values) = verify_batched_sumcheck_rounds_ds(
        cs,
        tr,
        b"shard/batched_time",
        step_idx,
        delta,
        expected_degree_bounds,
        &claimed_sums_vars,
        expected_labels,
        &round_polys_vars,
        &proof.round_polys,
        &format!("{ctx}_batched_sumcheck"),
    )?;

    if r_time.len() != ell_n {
        return Err(SpartanBridgeError::InvalidInput(format!(
            "{ctx}: r_time length mismatch (got {}, expected ell_n={ell_n})",
            r_time.len()
        )));
    }

    Ok(RouteABatchedTimeOutVars {
        r_time,
        final_values,
        claimed_sums_vars,
        round_polys_vars,
    })
}
