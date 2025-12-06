//! Common gadgets for Π-CCS verification
//!
//! This module provides:
//! - eq gadget: χ evaluation over boolean/field points
//! - Range product gadget: ∏(y - t) for t in range
//! - MLE evaluation gadget
//!
//! NOTE: eq_gadget and mle_eval_gadget currently use cs.get() which is unsafe for production.
//! They are gated behind #[cfg(feature = "unsafe-gadgets")] until rewritten.

use crate::gadgets::k_field::KNumVar;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::PrimeField;

#[cfg(feature = "unsafe-gadgets")]
use bellpepper_core::Variable;

#[cfg(feature = "unsafe-gadgets")]
/// Compute eq(x, a) = ∏_i ((1 - x_i)(1 - a_i) + x_i * a_i) in the circuit
///
/// **WARNING**: This gadget uses cs.get() which is not available in production constraint systems.
/// It is only intended for TestConstraintSystem and must be rewritten before production use.
///
/// x: boolean variables (bits)
/// a: field elements
pub fn eq_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    x_bits: &[Variable],
    a_fields: &[Variable],
    label: &str,
) -> Result<Variable, SynthesisError> {
    if x_bits.len() != a_fields.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    if x_bits.is_empty() {
        // eq of empty vectors is 1
        return Ok(CS::one());
    }

    // Start with chi = 1
    let mut chi = CS::one();

    for (i, (x_i, a_i)) in x_bits.iter().zip(a_fields.iter()).enumerate() {
        // term = (1 - x_i)(1 - a_i) + x_i * a_i
        // Expand: 1 - x_i - a_i + x_i*a_i + x_i*a_i
        //       = 1 - x_i - a_i + 2*x_i*a_i

        let term = cs.alloc(
            || format!("{}_eq_term_{}", label, i),
            || {
                let x_val = cs.get(*x_i)?;
                let a_val = cs.get(*a_i)?;
                let one_minus_x = F::ONE - x_val;
                let one_minus_a = F::ONE - a_val;
                Ok(one_minus_x * one_minus_a + x_val * a_val)
            },
        )?;

        // Allocate product x_i * a_i
        let x_a = cs.alloc(
            || format!("{}_xa_{}", label, i),
            || {
                let x_val = cs.get(*x_i)?;
                let a_val = cs.get(*a_i)?;
                Ok(x_val * a_val)
            },
        )?;

        // Enforce x_a = x_i * a_i
        cs.enforce(
            || format!("{}_xa_constraint_{}", label, i),
            |lc| lc + *x_i,
            |lc| lc + *a_i,
            |lc| lc + x_a,
        );

        // Enforce term = 1 - x_i - a_i + 2*x_a
        cs.enforce(
            || format!("{}_term_constraint_{}", label, i),
            |lc| lc + CS::one() - *x_i - *a_i + (F::from(2u64), x_a),
            |lc| lc + CS::one(),
            |lc| lc + term,
        );

        // chi_next = chi * term
        let chi_next = cs.alloc(
            || format!("{}_chi_{}", label, i + 1),
            || {
                let chi_val = cs.get(chi)?;
                let term_val = cs.get(term)?;
                Ok(chi_val * term_val)
            },
        )?;

        cs.enforce(
            || format!("{}_chi_mul_{}", label, i + 1),
            |lc| lc + chi,
            |lc| lc + term,
            |lc| lc + chi_next,
        );

        chi = chi_next;
    }

    Ok(chi)
}

/// Compute range product over K: ∏_{t=-(b-1)}^{b-1} (y - t)
///
/// This is used in Π-CCS to verify the range constraint on Ajtai digits.
///
/// y: K-field element
/// b: base (defines range [-(b-1), b-1])
/// delta: δ constant for K multiplication (u^2 = δ)
///
/// **NOTE**: This is currently a placeholder. Real implementation would compute
/// the product over the full range without cs.get(). For now, just returns y.
pub fn range_product_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    _cs: &mut CS,
    y: &KNumVar,
    _b: u32,
    _delta: F,
    _label: &str,
) -> Result<KNumVar, SynthesisError> {
    // Placeholder: just return y for now (not a real range check)
    // TODO: Implement full range product without cs.get()
    // Just return y unchanged
    Ok(y.clone())
}

#[cfg(feature = "unsafe-gadgets")]
/// Evaluate MLE at a point using χ weights
///
/// **WARNING**: This gadget uses cs.get() which is not available in production constraint systems.
/// It is only intended for TestConstraintSystem and must be rewritten before production use.
///
/// Given a table y[ρ] of K-field elements and a boolean point α (as bits),
/// compute: Σ_ρ y[ρ] * χ_α[ρ]
///
/// Where χ_α[ρ] = ∏_i ((1 - α_i)(1 - ρ_i) + α_i * ρ_i)
///
/// This is used in Π-CCS to evaluate Ajtai MLEs.
pub fn mle_eval_gadget<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    alpha_bits: &[Variable],
    y_table: &[KNumVar],
    delta: F,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let num_vars = alpha_bits.len();
    let expected_len = 1usize << num_vars;

    if y_table.len() != expected_len {
        return Err(SynthesisError::Unsatisfiable);
    }

    // Allocate χ_α table (in F, then lift to K)
    let mut chi_table = Vec::with_capacity(expected_len);

    for rho in 0..expected_len {
        // Compute χ_α[ρ] = ∏_i ((1 - α_i)(1 - ρ_i) + α_i * ρ_i)
        let mut chi = CS::one();

        for (i, alpha_i) in alpha_bits.iter().enumerate() {
            let rho_i = (rho >> i) & 1;
            let rho_i_f = F::from(rho_i as u64);

            // term = (1 - α_i)(1 - ρ_i) + α_i * ρ_i
            let term = cs.alloc(
                || format!("{}_chi_term_rho{}_bit{}", label, rho, i),
                || {
                    let alpha_val = cs.get(*alpha_i)?;
                    let one_minus_alpha = F::ONE - alpha_val;
                    let one_minus_rho = F::ONE - rho_i_f;
                    Ok(one_minus_alpha * one_minus_rho + alpha_val * rho_i_f)
                },
            )?;

            // Enforce term constraint (similar to eq_gadget)
            // term = 1 - α_i - ρ_i + 2*α_i*ρ_i
            let alpha_rho = cs.alloc(
                || format!("{}_alpha_rho_rho{}_bit{}", label, rho, i),
                || {
                    let alpha_val = cs.get(*alpha_i)?;
                    Ok(alpha_val * rho_i_f)
                },
            )?;

            cs.enforce(
                || format!("{}_alpha_rho_constraint_rho{}_bit{}", label, rho, i),
                |lc| lc + *alpha_i,
                |lc| lc + (rho_i_f, CS::one()),
                |lc| lc + alpha_rho,
            );

            cs.enforce(
                || format!("{}_term_constraint_rho{}_bit{}", label, rho, i),
                |lc| lc + CS::one() - *alpha_i - (rho_i_f, CS::one()) + (F::from(2u64), alpha_rho),
                |lc| lc + CS::one(),
                |lc| lc + term,
            );

            // chi *= term
            let chi_next = cs.alloc(
                || format!("{}_chi_rho{}_step{}", label, rho, i + 1),
                || {
                    let chi_val = cs.get(chi)?;
                    let term_val = cs.get(term)?;
                    Ok(chi_val * term_val)
                },
            )?;

            cs.enforce(
                || format!("{}_chi_mul_rho{}_step{}", label, rho, i + 1),
                |lc| lc + chi,
                |lc| lc + term,
                |lc| lc + chi_next,
            );

            chi = chi_next;
        }

        chi_table.push(chi);
    }

    // Compute weighted sum: Σ_ρ y[ρ] * χ[ρ]
    // Start with zero in K
    let zero_c0 = cs.alloc(|| format!("{}_sum_init_c0", label), || Ok(F::ZERO))?;
    let zero_c1 = cs.alloc(|| format!("{}_sum_init_c1", label), || Ok(F::ZERO))?;

    cs.enforce(
        || format!("{}_zero_c0_constraint", label),
        |lc| lc + zero_c0,
        |lc| lc + CS::one(),
        |lc| lc,
    );

    cs.enforce(
        || format!("{}_zero_c1_constraint", label),
        |lc| lc + zero_c1,
        |lc| lc + CS::one(),
        |lc| lc,
    );

    let mut sum = KNumVar {
        c0: zero_c0,
        c1: zero_c1,
    };

    for (rho, (y_rho, chi_rho)) in y_table.iter().zip(chi_table.iter()).enumerate() {
        // chi_rho is in F, lift to K
        let chi_k = k_lift_from_f(cs, *chi_rho, &format!("{}_chi_lift_{}", label, rho))?;

        // term = y[rho] * χ[rho] (in K)
        let term = k_mul(cs, y_rho, &chi_k, delta, None, &format!("{}_term_{}", label, rho))?;

        // sum += term
        sum = k_add(cs, &sum, &term, None, &format!("{}_add_{}", label, rho))?;
    }

    Ok(sum)
}
