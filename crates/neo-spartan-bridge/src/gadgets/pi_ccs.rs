//! Π-CCS circuit gadgets used by the FoldRun circuit.
//!
//! Currently this module exposes:
//! - Sumcheck round verification
//! - Sumcheck evaluation via Horner’s method
//! - A base-b recomposition helper (legacy; no longer used by the main circuit)

use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::PrimeField;
use crate::gadgets::k_field::{KNum, KNumVar, k_add, k_mul};
use neo_math::K as NeoK;

/// Recompose K-field digits in base b: Σ b^ℓ * y[ℓ]
///
/// y: vector of K-field elements (digits)
/// b: base
/// delta: δ constant for K multiplication
pub fn base_b_recompose_k<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    y_digits: &[KNumVar],
    b: u32,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    if y_digits.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }

    // Start with y[0]
    let mut result = y_digits[0].clone();
    
    // Compute powers of b in F
    let b_f = F::from(b as u64);
    let mut b_power = b_f;

    for (ell, y_ell) in y_digits.iter().enumerate().skip(1) {
        // Lift b^ell to K as (b^ell, 0)
        let b_pow_c0 = cs.alloc(
            || format!("{}_b_pow_{}_c0", label, ell),
            || Ok(b_power),
        )?;
        
        cs.enforce(
            || format!("{}_b_pow_{}_c0_constraint", label, ell),
            |lc| lc + (b_power, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + b_pow_c0,
        );
        
        let b_pow_k = KNumVar {
            c0: b_pow_c0,
            c1: cs.alloc(|| format!("{}_b_pow_{}_c1_zero", label, ell), || Ok(F::ZERO))?,
        };
        
        // Enforce c1 = 0
        cs.enforce(
            || format!("{}_b_pow_{}_c1_zero_constraint", label, ell),
            |lc| lc + b_pow_k.c1,
            |lc| lc + CS::one(),
            |lc| lc,
        );

        // term = b^ell * y[ell]
        let term = k_mul(cs, &b_pow_k, y_ell, delta, None, &format!("{}_term_{}", label, ell))?;
        
        // result += term
        result = k_add(cs, &result, &term, None, &format!("{}_add_{}", label, ell))?;
        
        // Update b_power for next iteration
        b_power = b_power * b_f;
    }

    Ok(result)
}

/// Verify a single sumcheck round: p(0) + p(1) = claimed_sum
///
/// coeffs: polynomial coefficients [c0, c1, c2, ...] where p(X) = Σ c_i X^i
/// coeff_values: native K values for the same coefficients (for witness hints)
/// claimed_sum: the value that p(0) + p(1) should equal
pub fn sumcheck_round_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    coeffs: &[KNumVar],
    coeff_values: &[NeoK],
    claimed_sum: &KNumVar,
    _delta: F,
    label: &str,
) -> Result<(), SynthesisError> {
    if coeffs.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if coeffs.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    // p(0) = c0 (first coefficient)
    let p_at_0 = coeffs[0].clone();

    // p(1) = Σ c_i, built incrementally with native K hints to keep the witness consistent.
    let mut p1_val = coeff_values[0];
    let mut p_at_1 = coeffs[0].clone();
    for (i, c_i) in coeffs.iter().enumerate().skip(1) {
        p1_val += coeff_values[i];
        let hint = KNum::<F>::from_neo_k(p1_val);
        p_at_1 = k_add(
            cs,
            &p_at_1,
            c_i,
            Some(hint),
            &format!("{}_p1_add_{}", label, i),
        )?;
    }

    // sum = p(0) + p(1) with native hint as well.
    let sum_val = coeff_values[0] + p1_val;
    let sum_hint = KNum::<F>::from_neo_k(sum_val);
    let sum = k_add(
        cs,
        &p_at_0,
        &p_at_1,
        Some(sum_hint),
        &format!("{}_sum", label),
    )?;

    // Enforce sum == claimed_sum
    // sum.c0 == claimed_sum.c0
    cs.enforce(
        || format!("{}_sum_c0_check", label),
        |lc| lc + sum.c0,
        |lc| lc + CS::one(),
        |lc| lc + claimed_sum.c0,
    );

    // sum.c1 == claimed_sum.c1
    cs.enforce(
        || format!("{}_sum_c1_check", label),
        |lc| lc + sum.c1,
        |lc| lc + CS::one(),
        |lc| lc + claimed_sum.c1,
    );

    Ok(())
}

/// Verify sumcheck evaluation: p(challenge) = next_claimed_sum
///
/// Evaluates p(X) = Σ c_i X^i at X = challenge using Horner's method.
/// Uses native K hints for every intermediate multiplication/addition so that
/// the prover's witness can satisfy all constraints in nontrivial cases.
fn k_mul_with_hint<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    a: &KNumVar,
    b: &KNumVar,
    a_hint: KNum<F>,
    b_hint: KNum<F>,
    prod_hint: KNum<F>,
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    // Allocate result components with known product hint.
    let c0 = cs.alloc(
        || format!("{}_prod_c0", label),
        || Ok(prod_hint.c0),
    )?;

    let c1 = cs.alloc(
        || format!("{}_prod_c1", label),
        || Ok(prod_hint.c1),
    )?;

    // Intermediate products in the base field, derived from operand hints.
    let a0b0_val = a_hint.c0 * b_hint.c0;
    let a1b1_val = a_hint.c1 * b_hint.c1;
    let a0b1_val = a_hint.c0 * b_hint.c1;
    let a1b0_val = a_hint.c1 * b_hint.c0;

    let a0_b0 = cs.alloc(
        || format!("{}_a0b0", label),
        || Ok(a0b0_val),
    )?;

    let a1_b1 = cs.alloc(
        || format!("{}_a1b1", label),
        || Ok(a1b1_val),
    )?;

    let a0_b1 = cs.alloc(
        || format!("{}_a0b1", label),
        || Ok(a0b1_val),
    )?;

    let a1_b0 = cs.alloc(
        || format!("{}_a1b0", label),
        || Ok(a1b0_val),
    )?;

    // Enforce intermediate products.
    cs.enforce(
        || format!("{}_a0b0_constraint", label),
        |lc| lc + a.c0,
        |lc| lc + b.c0,
        |lc| lc + a0_b0,
    );

    cs.enforce(
        || format!("{}_a1b1_constraint", label),
        |lc| lc + a.c1,
        |lc| lc + b.c1,
        |lc| lc + a1_b1,
    );

    cs.enforce(
        || format!("{}_a0b1_constraint", label),
        |lc| lc + a.c0,
        |lc| lc + b.c1,
        |lc| lc + a0_b1,
    );

    cs.enforce(
        || format!("{}_a1b0_constraint", label),
        |lc| lc + a.c1,
        |lc| lc + b.c0,
        |lc| lc + a1_b0,
    );

    // Enforce c0 = a0_b0 + δ * a1_b1
    cs.enforce(
        || format!("{}_c0_constraint", label),
        |lc| lc + a0_b0 + (delta, a1_b1),
        |lc| lc + CS::one(),
        |lc| lc + c0,
    );

    // Enforce c1 = a0_b1 + a1_b0
    cs.enforce(
        || format!("{}_c1_constraint", label),
        |lc| lc + a0_b1 + a1_b0,
        |lc| lc + CS::one(),
        |lc| lc + c1,
    );

    Ok(KNumVar { c0, c1 })
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
    if coeffs.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if coeffs.len() != coeff_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    // Evaluate p(challenge) using Horner's method with native hints:
    // result_val starts from the highest-degree coefficient and is updated as
    //   result_val <- result_val * challenge_value + c_i
    // at each step. We mirror this in-circuit with k_mul / k_add using hints.
    let n = coeffs.len();
    let mut result = coeffs[n - 1].clone();
    let mut result_val = coeff_values[n - 1];

    // Walk coefficients from c_{n-2} down to c_0.
    for (step_idx, (c_var, c_val)) in coeffs[..n - 1]
        .iter()
        .rev()
        .zip(coeff_values[..n - 1].iter().rev())
        .enumerate()
    {
        // result_tmp = result * challenge
        let mul_val = result_val * challenge_value;
        let a_hint = KNum::<F>::from_neo_k(result_val);
        let b_hint = KNum::<F>::from_neo_k(challenge_value);
        let prod_hint = KNum::<F>::from_neo_k(mul_val);
        let result_tmp = k_mul_with_hint(
            cs,
            &result,
            challenge,
            a_hint,
            b_hint,
            prod_hint,
            delta,
            &format!("{}_horner_mul_{}", label, step_idx + 1),
        )?;

        // result_next = result_tmp + c_i
        let add_val = mul_val + *c_val;
        let add_hint = KNum::<F>::from_neo_k(add_val);
        result = k_add(
            cs,
            &result_tmp,
            c_var,
            Some(add_hint),
            &format!("{}_horner_add_{}", label, step_idx + 1),
        )?;

        result_val = add_val;
    }

    Ok(result)
}
