//! Transcript-bound sumcheck gadgets (single + batched).
//!
//! These mirror:
//! - `neo_fold::memory_sidecar::sumcheck_ds::{verify_sumcheck_rounds_ds, verify_batched_sumcheck_rounds_ds}`
//! - `neo_reductions::sumcheck::{verify_sumcheck_rounds, verify_batched_sumcheck_rounds}`
//!
//! The goal is to keep transcript order *exactly* the same as native.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_math::{F as NeoF, K as NeoK, KExtensions};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::gadgets::k_field::KNumVar;
use crate::gadgets::pi_ccs::{sumcheck_eval_gadget, sumcheck_round_gadget};
use crate::gadgets::transcript::Poseidon2TranscriptVar;
use crate::CircuitF;

fn k_to_circuit_coeffs(k: NeoK) -> [CircuitF; 2] {
    let coeffs = k.as_coeffs();
    [
        CircuitF::from(coeffs[0].as_canonical_u64()),
        CircuitF::from(coeffs[1].as_canonical_u64()),
    ]
}

fn k_from_allocated(c0: &bellpepper_core::num::AllocatedNum<CircuitF>, c1: &bellpepper_core::num::AllocatedNum<CircuitF>) -> NeoK {
    let c0_u64 = c0
        .get_value()
        .unwrap_or(CircuitF::from(0u64))
        .to_canonical_u64();
    let c1_u64 = c1
        .get_value()
        .unwrap_or(CircuitF::from(0u64))
        .to_canonical_u64();
    neo_math::from_complex(NeoF::from_u64(c0_u64), NeoF::from_u64(c1_u64))
}

fn sc_start<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    domain: &'static [u8],
    inst_idx: usize,
    ctx: &str,
) -> Result<(), SynthesisError> {
    tr.append_message(cs, b"sc/domain", domain, ctx)?;
    tr.append_message(cs, b"sc/inst_idx", &(inst_idx as u64).to_le_bytes(), ctx)?;
    Ok(())
}

fn sc_end<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    domain: &'static [u8],
    inst_idx: usize,
    ctx: &str,
) -> Result<(), SynthesisError> {
    tr.append_message(cs, b"sc/domain_end", domain, ctx)?;
    tr.append_message(cs, b"sc/inst_idx_end", &(inst_idx as u64).to_le_bytes(), ctx)?;
    Ok(())
}

/// In-circuit verifier for a single (non-batched) sumcheck with DS framing.
///
/// Returns (per-round challenges, final running sum).
pub fn verify_sumcheck_rounds_ds<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    domain: &'static [u8],
    inst_idx: usize,
    degree_bound: usize,
    delta: CircuitF,
    claimed_sum_var: &KNumVar,
    claimed_sum_val: NeoK,
    rounds_vars: &[Vec<KNumVar>],
    rounds_vals: &[Vec<NeoK>],
    ctx: &str,
) -> Result<(Vec<KNumVar>, KNumVar), SynthesisError> {
    if rounds_vars.len() != rounds_vals.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    sc_start(cs, tr, domain, inst_idx, ctx)?;
    let [c0_val, c1_val] = k_to_circuit_coeffs(claimed_sum_val);
    tr.append_fields_vars(
        cs,
        b"sc/claimed_sum",
        &[claimed_sum_var.c0, claimed_sum_var.c1],
        &[c0_val, c1_val],
        &format!("{ctx}_claimed_sum"),
    )?;

    let mut running_sum = claimed_sum_var.clone();
    let mut challenges = Vec::with_capacity(rounds_vars.len());

    for (round_idx, (coeffs_vars, coeffs_vals)) in rounds_vars.iter().zip(rounds_vals.iter()).enumerate() {
        if coeffs_vars.len() != coeffs_vals.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        if coeffs_vars.len() > degree_bound + 1 {
            return Err(SynthesisError::Unsatisfiable);
        }

        sumcheck_round_gadget(
            cs,
            coeffs_vars,
            coeffs_vals,
            &running_sum,
            delta,
            &format!("{ctx}_round_{round_idx}_invariant"),
        )?;

        // Commit coefficients to transcript (same order/labels as native verifier).
        for (coeff_idx, (coeff_var, &coeff_val)) in coeffs_vars.iter().zip(coeffs_vals.iter()).enumerate() {
            let [c0, c1] = k_to_circuit_coeffs(coeff_val);
            tr.append_fields_vars(
                cs,
                b"sumcheck/round/coeff",
                &[coeff_var.c0, coeff_var.c1],
                &[c0, c1],
                &format!("{ctx}_round_{round_idx}_coeff_{coeff_idx}"),
            )?;
        }

        // Sample per-round challenge as an extension field element (two base-field challenges).
        let c0 = tr.challenge_field(cs, b"sumcheck/challenge/0", &format!("{ctx}_round_{round_idx}"))?;
        let c1 = tr.challenge_field(cs, b"sumcheck/challenge/1", &format!("{ctx}_round_{round_idx}"))?;
        let challenge_var = KNumVar {
            c0: c0.get_variable(),
            c1: c1.get_variable(),
        };
        let challenge_val = k_from_allocated(&c0, &c1);
        challenges.push(challenge_var.clone());

        // Advance: running_sum := p(challenge)
        running_sum = sumcheck_eval_gadget(
            cs,
            coeffs_vars,
            coeffs_vals,
            &challenge_var,
            challenge_val,
            delta,
            &format!("{ctx}_round_{round_idx}_eval"),
        )?;
    }

    sc_end(cs, tr, domain, inst_idx, ctx)?;
    Ok((challenges, running_sum))
}

/// In-circuit verifier for a batched sumcheck with shared challenges and DS framing.
///
/// Returns (shared challenges, final values per claim).
pub fn verify_batched_sumcheck_rounds_ds<CS: ConstraintSystem<CircuitF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptVar,
    domain: &'static [u8],
    inst_idx: usize,
    delta: CircuitF,
    degree_bounds: &[usize],
    claimed_sums: &[KNumVar],
    claim_labels: &[&[u8]],
    round_polys_vars: &[Vec<Vec<KNumVar>>], // [claim][round][coeff]
    round_polys_vals: &[Vec<Vec<NeoK>>],
    ctx: &str,
) -> Result<(Vec<KNumVar>, Vec<KNumVar>), SynthesisError> {
    if round_polys_vars.len() != round_polys_vals.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if round_polys_vars.len() != claimed_sums.len()
        || round_polys_vars.len() != claim_labels.len()
        || round_polys_vars.len() != degree_bounds.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    if round_polys_vars.is_empty() {
        sc_start(cs, tr, domain, inst_idx, ctx)?;
        sc_end(cs, tr, domain, inst_idx, ctx)?;
        return Ok((Vec::new(), Vec::new()));
    }

    let num_rounds = round_polys_vars[0].len();
    for rounds in round_polys_vars {
        if rounds.len() != num_rounds {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for rounds in round_polys_vals {
        if rounds.len() != num_rounds {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    sc_start(cs, tr, domain, inst_idx, ctx)?;

    let mut running_sums: Vec<KNumVar> = claimed_sums.to_vec();
    let mut shared_challenges: Vec<KNumVar> = Vec::with_capacity(num_rounds);

    for round_idx in 0..num_rounds {
        tr.append_message(
            cs,
            b"batched/round_idx",
            &(round_idx as u64).to_le_bytes(),
            &format!("{ctx}_batched_round_{round_idx}"),
        )?;

        // Enforce invariants for all claims for this round.
        for claim_idx in 0..round_polys_vars.len() {
            let round_poly_vars = &round_polys_vars[claim_idx][round_idx];
            let round_poly_vals = &round_polys_vals[claim_idx][round_idx];

            if round_poly_vars.len() != round_poly_vals.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            if round_poly_vars.len() > degree_bounds[claim_idx] + 1 {
                return Err(SynthesisError::Unsatisfiable);
            }

            sumcheck_round_gadget(
                cs,
                round_poly_vars,
                round_poly_vals,
                &running_sums[claim_idx],
                delta,
                &format!("{ctx}_claim_{claim_idx}_round_{round_idx}_invariant"),
            )?;
        }

        // Append ALL round polynomials to transcript (same order as native verifier).
        for claim_idx in 0..round_polys_vars.len() {
            tr.append_message(
                cs,
                b"batched/claim_label",
                claim_labels[claim_idx],
                &format!("{ctx}_batched_round_{round_idx}_claim_{claim_idx}"),
            )?;
            tr.append_message(
                cs,
                b"batched/claim_idx",
                &(claim_idx as u64).to_le_bytes(),
                &format!("{ctx}_batched_round_{round_idx}_claim_{claim_idx}"),
            )?;

            let round_poly_vars = &round_polys_vars[claim_idx][round_idx];
            let round_poly_vals = &round_polys_vals[claim_idx][round_idx];
            for (coeff_idx, (coeff_var, &coeff_val)) in round_poly_vars.iter().zip(round_poly_vals.iter()).enumerate()
            {
                let [c0, c1] = k_to_circuit_coeffs(coeff_val);
                tr.append_fields_vars(
                    cs,
                    b"batched/round/coeff",
                    &[coeff_var.c0, coeff_var.c1],
                    &[c0, c1],
                    &format!("{ctx}_claim_{claim_idx}_round_{round_idx}_coeff_{coeff_idx}"),
                )?;
            }
        }

        // Sample one shared challenge from transcript.
        let c0 = tr.challenge_field(cs, b"batched/challenge/0", &format!("{ctx}_batched_round_{round_idx}"))?;
        let c1 = tr.challenge_field(cs, b"batched/challenge/1", &format!("{ctx}_batched_round_{round_idx}"))?;
        let shared_challenge_var = KNumVar {
            c0: c0.get_variable(),
            c1: c1.get_variable(),
        };
        let shared_challenge_val = k_from_allocated(&c0, &c1);
        shared_challenges.push(shared_challenge_var.clone());

        // Fold each claim: running_sum := p(shared_challenge).
        for claim_idx in 0..round_polys_vars.len() {
            running_sums[claim_idx] = sumcheck_eval_gadget(
                cs,
                &round_polys_vars[claim_idx][round_idx],
                &round_polys_vals[claim_idx][round_idx],
                &shared_challenge_var,
                shared_challenge_val,
                delta,
                &format!("{ctx}_claim_{claim_idx}_round_{round_idx}_eval"),
            )?;
        }
    }

    sc_end(cs, tr, domain, inst_idx, ctx)?;
    Ok((shared_challenges, running_sums))
}
