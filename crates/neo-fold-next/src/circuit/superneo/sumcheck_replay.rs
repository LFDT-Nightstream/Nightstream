//! Owns in-circuit sumcheck replay over the RV32IM main-relation transcript.
//!
//! This mirrors the native `neo_reductions::sumcheck::verify_sumcheck_rounds`
//! path: absorb round coefficients, sample transcript challenges, and enforce
//! the running-sum invariant in-circuit.

use crate::spartan_backend::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::Field;
use neo_math::{KExtensions, K as NeoK};
use neo_reductions::sumcheck::SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG;
use p3_field::PrimeField64;

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
    verify_sumcheck_rounds_with_trace(
        cs,
        tr,
        degree_bound,
        initial_sum,
        rounds,
        round_values,
        challenge_values,
        delta,
        label,
        |_, _| {},
    )
}

pub(crate) fn verify_sumcheck_rounds_with_trace<CS, Trace>(
    cs: &mut CS,
    tr: &mut Poseidon2TranscriptCircuit,
    degree_bound: usize,
    initial_sum: &KNumVar,
    rounds: &[Vec<KNumVar>],
    round_values: &[Vec<NeoK>],
    challenge_values: &[NeoK],
    delta: SpartanF,
    label: &str,
    mut trace: Trace,
) -> Result<(Vec<KNumVar>, KNumVar), SynthesisError>
where
    CS: ConstraintSystem<SpartanF>,
    Trace: FnMut(&mut CS, &str),
{
    if rounds.len() != round_values.len() || rounds.len() != challenge_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut running_sum = initial_sum.clone();

    tr.append_const_fields_raw(
        cs.namespace(|| format!("{label}_transcript_v3")),
        &[SpartanF::from_canonical_u64(SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG)],
    )?;
    trace(cs, "transcript_v3");

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
        let round_check = format!("round_{round_idx}.round_check");
        trace(cs, &round_check);
        append_round_coeffs(
            cs.namespace(|| format!("{label}_append_round_{round_idx}")),
            tr,
            round_vars,
            round_vals,
        )?;
        let append_round = format!("round_{round_idx}.append_round");
        trace(cs, &append_round);
        let challenge = sample_sumcheck_challenge(cs.namespace(|| format!("{label}_challenge_{round_idx}")), tr)?;
        let challenge_sample = format!("round_{round_idx}.challenge_sample");
        trace(cs, &challenge_sample);
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
        let challenge_match = format!("round_{round_idx}.challenge_match");
        trace(cs, &challenge_match);
        running_sum = sumcheck_eval_gadget(
            cs,
            round_vars,
            round_vals,
            &challenge,
            *challenge_value,
            delta,
            &format!("{label}_eval_{round_idx}"),
        )?;
        let eval = format!("round_{round_idx}.eval");
        trace(cs, &eval);
        challenges.push(challenge);
    }

    Ok((challenges, running_sum))
}

fn append_round_coeffs<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    tr: &mut Poseidon2TranscriptCircuit,
    coeff_vars: &[KNumVar],
    coeff_values: &[NeoK],
) -> Result<(), SynthesisError> {
    if coeff_vars.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut packed_values = Vec::with_capacity(coeff_values.len() * 2);
    let mut field_terms = Vec::with_capacity(coeff_values.len() * 2);
    let mut field_constants = Vec::with_capacity(coeff_values.len() * 2);
    for (coeff_var, coeff_value) in coeff_vars.iter().zip(coeff_values.iter()) {
        let coeff_parts = coeff_value.as_coeffs();
        packed_values.push(SpartanF::from_canonical_u64(coeff_parts[0].as_canonical_u64()));
        packed_values.push(SpartanF::from_canonical_u64(coeff_parts[1].as_canonical_u64()));
        field_terms.push(vec![(coeff_var.c0, SpartanF::ONE)]);
        field_constants.push(SpartanF::ZERO);
        field_terms.push(vec![(coeff_var.c1, SpartanF::ONE)]);
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
