//! Owns in-circuit sumcheck replay over the RV64IM main-relation transcript.
//!
//! This mirrors the native `neo_reductions::sumcheck::verify_sumcheck_rounds`
//! path: absorb round coefficients, sample transcript challenges, and enforce
//! the running-sum invariant in-circuit.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_math::{KExtensions, K as NeoK};
use neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG;
use p3_field::PrimeField64;
use spartan2::provider::goldi::F as SpartanF;

use super::k_field::{alloc_k, enforce_k_eq, KNum, KNumVar};
use super::sumcheck::{sumcheck_eval_gadget, sumcheck_round_gadget};
use super::transcript::Poseidon2TranscriptCircuit;

pub fn verify_sumcheck_rounds<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    degree_bound: usize,
    initial_sum: &KNumVar,
    rounds: &[Vec<KNumVar>],
    round_values: &[Vec<NeoK>],
    challenge_values: &[NeoK],
    delta: SpartanF,
    label: &str,
) -> Result<(Vec<KNumVar>, KNumVar), SynthesisError> {
    if rounds.len() != round_values.len() || rounds.len() != challenge_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut running_sum = initial_sum.clone();

    tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_transcript_v3")),
        &[SpartanF::from_canonical_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)],
    )?;

    for (round_idx, ((round_vars, round_vals), challenge_value)) in rounds
        .iter()
        .zip(round_values.iter())
        .zip(challenge_values.iter())
        .enumerate()
    {
        if round_vars.len() != round_vals.len() || round_vars.len() > degree_bound + 1 {
            return Err(SynthesisError::Unsatisfiable);
        }
        sumcheck_round_gadget(
            cs,
            round_vars,
            round_vals,
            &running_sum,
            &format!("{label}_round_{round_idx}"),
        )?;
        append_round_coeffs(
            cs.namespace(|| format!("{label}_append_round_{round_idx}")),
            tr,
            round_vals,
        )?;
        let challenge = sample_sumcheck_challenge(cs.namespace(|| format!("{label}_challenge_{round_idx}")), tr)?;
        let expected_challenge = alloc_k(
            cs,
            Some(KNum::from_neo_k(*challenge_value)),
            &format!("{label}_challenge_expected_{round_idx}"),
        )?;
        enforce_k_eq(
            cs,
            &challenge,
            &expected_challenge,
            &format!("{label}_challenge_match_{round_idx}"),
        );
        running_sum = sumcheck_eval_gadget(
            cs,
            round_vars,
            round_vals,
            &challenge,
            *challenge_value,
            delta,
            &format!("{label}_eval_{round_idx}"),
        )?;
        challenges.push(challenge);
    }

    Ok((challenges, running_sum))
}

fn append_round_coeffs<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    tr: &mut Poseidon2TranscriptCircuit,
    coeff_values: &[NeoK],
) -> Result<(), SynthesisError> {
    let mut packed_values = Vec::with_capacity(coeff_values.len() * 2);
    for coeff_value in coeff_values {
        let coeff_parts = coeff_value.as_coeffs();
        packed_values.push(SpartanF::from_canonical_u64(coeff_parts[0].as_canonical_u64()));
        packed_values.push(SpartanF::from_canonical_u64(coeff_parts[1].as_canonical_u64()));
    }
    let packed_vars = packed_values
        .iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc(cs.namespace(|| format!("round_coeff_{idx}")), || Ok(*value)))
        .collect::<Result<Vec<_>, _>>()?;
    let mut field_terms = Vec::with_capacity(packed_vars.len());
    let mut field_constants = Vec::with_capacity(packed_vars.len());
    for value in &packed_vars {
        field_terms.push(vec![(value.get_variable(), SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
    }
    tr.append_field_linear_combinations_raw(
        cs.namespace(|| "round_coeffs"),
        &field_terms,
        &field_constants,
        &packed_values,
    )?;
    Ok(())
}

fn sample_sumcheck_challenge<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    tr: &mut Poseidon2TranscriptCircuit,
) -> Result<KNumVar, SynthesisError> {
    let pair = tr.challenge_fields_raw(cs.namespace(|| "pair"), 2)?;
    if pair.len() != 2 {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(KNumVar {
        c0: pair[0].get_variable(),
        c1: pair[1].get_variable(),
    })
}
