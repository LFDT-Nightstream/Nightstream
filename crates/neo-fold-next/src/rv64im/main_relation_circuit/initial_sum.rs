//! Owns the FE initial-sum gadget for RV64IM main-relation chunk circuits.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::CcsStructure;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{alloc_constant_k, k_add, k_mul, k_scalar_mul, KNum, KNumVar};

pub fn claimed_initial_sum_from_me_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    alpha_vars: &[KNumVar],
    alpha_values: &[K],
    gamma_var: &KNumVar,
    gamma_value: K,
    k_mcs: usize,
    me_inputs: &[CeClaimVar],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if alpha_vars.len() != alpha_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if me_inputs.is_empty() {
        let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
        return Ok((zero, K::ZERO));
    }

    let k_total = k_mcs
        .checked_add(me_inputs.len())
        .ok_or(SynthesisError::Unsatisfiable)?;
    if k_total < 2 {
        let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
        return Ok((zero, K::ZERO));
    }

    for (idx, claim) in me_inputs.iter().enumerate() {
        if claim.y_ring.len() < structure.t() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for row in claim.y_ring.iter().take(structure.t()) {
            if row.len() < (1usize << alpha_vars.len()) {
                return Err(SynthesisError::Unsatisfiable);
            }
        }
        if idx > 0 && claim.r_values != me_inputs[0].r_values {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    let (chi_alpha, chi_alpha_values) = chi_table(cs, alpha_vars, alpha_values, delta, &format!("{label}_chi_alpha"))?;
    let (gamma_to_k, gamma_to_k_value) = pow_k(
        cs,
        gamma_var,
        gamma_value,
        k_total,
        delta,
        &format!("{label}_gamma_to_k"),
    )?;

    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_inner_zero"))?;
    let mut inner = zero;
    let mut inner_value = K::ZERO;
    for matrix_idx in 0..structure.t() {
        for (me_idx, claim) in me_inputs.iter().enumerate() {
            let absolute_slot = k_mcs + me_idx + 1;
            let (y_eval, y_eval_value) = dot_k_row(
                cs,
                &claim.y_ring[matrix_idx],
                &claim.y_ring_values[matrix_idx],
                &chi_alpha,
                &chi_alpha_values,
                delta,
                &format!("{label}_y_eval_{matrix_idx}_{me_idx}"),
            )?;
            let (weight, weight_value) = weight_for_input(
                cs,
                gamma_var,
                gamma_value,
                &gamma_to_k,
                gamma_to_k_value,
                absolute_slot - 1,
                matrix_idx,
                delta,
                &format!("{label}_weight_{matrix_idx}_{me_idx}"),
            )?;
            let contrib_value = weight_value * y_eval_value;
            let contrib = k_mul(
                cs,
                &weight,
                &y_eval,
                KNum::from_neo_k(weight_value),
                KNum::from_neo_k(y_eval_value),
                KNum::from_neo_k(contrib_value),
                delta,
                &format!("{label}_contrib_{matrix_idx}_{me_idx}"),
            )?;
            inner_value += contrib_value;
            inner = k_add(
                cs,
                &inner,
                &contrib,
                Some(KNum::from_neo_k(inner_value)),
                &format!("{label}_inner_acc_{matrix_idx}_{me_idx}"),
            )?;
        }
    }

    let total_value = gamma_to_k_value * inner_value;
    let total = k_mul(
        cs,
        &gamma_to_k,
        &inner,
        KNum::from_neo_k(gamma_to_k_value),
        KNum::from_neo_k(inner_value),
        KNum::from_neo_k(total_value),
        delta,
        &format!("{label}_total"),
    )?;
    Ok((total, total_value))
}

fn dot_k_row<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    row: &[KNumVar],
    row_values: &[K],
    chi: &[KNumVar],
    chi_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if row.len() != row_values.len() || row.len() != chi.len() || row.len() != chi_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc = zero;
    let mut acc_value = K::ZERO;
    for (idx, ((row_var, row_value), (chi_var, chi_value))) in row
        .iter()
        .zip(row_values.iter())
        .zip(chi.iter().zip(chi_values.iter()))
        .enumerate()
    {
        let term_value = *row_value * *chi_value;
        let term = k_mul(
            cs,
            row_var,
            chi_var,
            KNum::from_neo_k(*row_value),
            KNum::from_neo_k(*chi_value),
            KNum::from_neo_k(term_value),
            delta,
            &format!("{label}_term_{idx}"),
        )?;
        acc_value += term_value;
        acc = k_add(
            cs,
            &acc,
            &term,
            Some(KNum::from_neo_k(acc_value)),
            &format!("{label}_acc_{idx}"),
        )?;
    }
    Ok((acc, acc_value))
}

fn weight_for_input<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    gamma_var: &KNumVar,
    gamma_value: K,
    gamma_to_k: &KNumVar,
    gamma_to_k_value: K,
    gamma_prefix_exp: usize,
    matrix_idx: usize,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let (prefix, prefix_value) = pow_k(
        cs,
        gamma_var,
        gamma_value,
        gamma_prefix_exp,
        delta,
        &format!("{label}_prefix"),
    )?;
    let (outer, outer_value) = pow_k(
        cs,
        gamma_to_k,
        gamma_to_k_value,
        matrix_idx,
        delta,
        &format!("{label}_outer"),
    )?;
    let weight_value = prefix_value * outer_value;
    let weight = k_mul(
        cs,
        &prefix,
        &outer,
        KNum::from_neo_k(prefix_value),
        KNum::from_neo_k(outer_value),
        KNum::from_neo_k(weight_value),
        delta,
        &format!("{label}_mul"),
    )?;
    Ok((weight, weight_value))
}

fn chi_table<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    point_vars: &[KNumVar],
    point_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(Vec<KNumVar>, Vec<K>), SynthesisError> {
    if point_vars.len() != point_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let one = alloc_constant_k(cs, KNum::from_neo_k(K::ONE), &format!("{label}_one"))?;
    let mut out_vars = vec![one.clone()];
    let mut out_values = vec![K::ONE];

    for (bit, (bit_var, bit_value)) in point_vars.iter().zip(point_values.iter()).enumerate() {
        let neg = k_scalar_mul(
            cs,
            -SpartanF::ONE,
            bit_var,
            Some(KNum::from_neo_k(-*bit_value)),
            &format!("{label}_neg_{bit}"),
        )?;
        let one_minus_value = K::ONE - *bit_value;
        let one_minus = k_add(
            cs,
            &one,
            &neg,
            Some(KNum::from_neo_k(one_minus_value)),
            &format!("{label}_one_minus_{bit}"),
        )?;

        let prior_len = out_vars.len();
        let mut next_vars = Vec::with_capacity(prior_len * 2);
        let mut next_values = Vec::with_capacity(prior_len * 2);

        for idx in 0..prior_len {
            let next_value = out_values[idx] * one_minus_value;
            let next_var = k_mul(
                cs,
                &out_vars[idx],
                &one_minus,
                KNum::from_neo_k(out_values[idx]),
                KNum::from_neo_k(one_minus_value),
                KNum::from_neo_k(next_value),
                delta,
                &format!("{label}_zero_branch_{bit}_{idx}"),
            )?;
            next_vars.push(next_var);
            next_values.push(next_value);
        }
        for idx in 0..prior_len {
            let next_value = out_values[idx] * *bit_value;
            let next_var = k_mul(
                cs,
                &out_vars[idx],
                bit_var,
                KNum::from_neo_k(out_values[idx]),
                KNum::from_neo_k(*bit_value),
                KNum::from_neo_k(next_value),
                delta,
                &format!("{label}_one_branch_{bit}_{idx}"),
            )?;
            next_vars.push(next_var);
            next_values.push(next_value);
        }

        out_vars = next_vars;
        out_values = next_values;
    }
    Ok((out_vars, out_values))
}

fn pow_k<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    base_var: &KNumVar,
    base_value: K,
    exponent: usize,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let one = alloc_constant_k(cs, KNum::from_neo_k(K::ONE), &format!("{label}_one"))?;
    let mut acc = one;
    let mut acc_value = K::ONE;
    for idx in 0..exponent {
        let next_acc_value = acc_value * base_value;
        acc = k_mul(
            cs,
            &acc,
            base_var,
            KNum::from_neo_k(acc_value),
            KNum::from_neo_k(base_value),
            KNum::from_neo_k(next_acc_value),
            delta,
            &format!("{label}_step_{idx}"),
        )?;
        acc_value = next_acc_value;
    }
    Ok((acc, acc_value))
}
