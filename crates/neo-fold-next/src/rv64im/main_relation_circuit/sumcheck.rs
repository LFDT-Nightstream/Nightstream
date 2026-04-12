//! Owns sumcheck bellpepper gadgets for the RV64IM main relation circuit.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::PrimeField;
use neo_math::K as NeoK;
use p3_field::PrimeCharacteristicRing;

use super::k_field::{
    alloc_k_constant_k_linear_combination, enforce_k_eq_constant_f_linear_combination, k_add, k_mul, KNum, KNumVar,
};

pub fn sumcheck_round_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    coeffs: &[KNumVar],
    coeff_values: &[NeoK],
    claimed_sum: &KNumVar,
    label: &str,
) -> Result<(), SynthesisError> {
    if coeffs.is_empty() || coeffs.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut terms = Vec::with_capacity(coeffs.len());
    for (idx, coeff) in coeffs.iter().enumerate() {
        let scale = if idx == 0 { F::ONE + F::ONE } else { F::ONE };
        terms.push((scale, coeff.c0, coeff.c1));
    }
    let _ = coeff_values;
    enforce_k_eq_constant_f_linear_combination(cs, claimed_sum, &terms, &format!("{label}_claimed"));
    Ok(())
}

pub fn sumcheck_eval_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    coeffs: &[KNumVar],
    coeff_values: &[NeoK],
    challenge: &KNumVar,
    challenge_value: NeoK,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    if coeffs.is_empty() || coeffs.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut result = coeffs[coeffs.len() - 1].clone();
    let mut result_value = coeff_values[coeffs.len() - 1];
    for (step_idx, (coeff, coeff_value)) in coeffs[..coeffs.len() - 1]
        .iter()
        .rev()
        .zip(coeff_values[..coeff_values.len() - 1].iter().rev())
        .enumerate()
    {
        let mul_value = result_value * challenge_value;
        let mul = k_mul(
            cs,
            &result,
            challenge,
            KNum::from_neo_k(result_value),
            KNum::from_neo_k(challenge_value),
            KNum::from_neo_k(mul_value),
            delta,
            &format!("{label}_mul_{step_idx}"),
        )?;
        let add_value = mul_value + *coeff_value;
        result = k_add(
            cs,
            &mul,
            coeff,
            Some(KNum::from_neo_k(add_value)),
            &format!("{label}_add_{step_idx}"),
        )?;
        result_value = add_value;
    }
    Ok(result)
}

pub fn sumcheck_eval_gadget_constant_challenge<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    coeffs: &[KNumVar],
    coeff_values: &[NeoK],
    challenge_value: NeoK,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    if coeffs.is_empty() || coeffs.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let mut power_value = NeoK::ONE;
    let mut result_value = NeoK::ZERO;
    let mut terms = Vec::with_capacity(coeffs.len());
    for (coeff, coeff_value) in coeffs.iter().zip(coeff_values.iter()) {
        result_value += power_value * *coeff_value;
        terms.push((KNum::from_neo_k(power_value), coeff.c0, coeff.c1));
        power_value *= challenge_value;
    }

    alloc_k_constant_k_linear_combination(cs, &terms, KNum::from_neo_k(result_value), delta, label)
}
