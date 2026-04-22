//! Owns reusable K-field gadgets needed by Π_CCS terminal identities.

use crate::rv64im::ivc_snark::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ccs::SparsePoly;
use neo_math::K as NeoK;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::k_field::{alloc_constant_k, k_add, k_mul, k_scalar_mul, KNum, KNumVar};

pub fn eq_points<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    p: &[KNumVar],
    q: &[KNumVar],
    p_values: &[NeoK],
    q_values: &[NeoK],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, NeoK), SynthesisError> {
    if p.len() != q.len() || p.len() != p_values.len() || q.len() != q_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let one = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ONE), &format!("{label}_one"))?;
    let mut acc = one.clone();
    let mut acc_value = NeoK::ONE;

    for (idx, ((p_var, q_var), (p_value, q_value))) in p
        .iter()
        .zip(q.iter())
        .zip(p_values.iter().zip(q_values.iter()))
        .enumerate()
    {
        let neg_p = k_scalar_mul(
            cs,
            -SpartanF::ONE,
            p_var,
            Some(KNum::from_neo_k(-*p_value)),
            &format!("{label}_neg_p_{idx}"),
        )?;
        let neg_q = k_scalar_mul(
            cs,
            -SpartanF::ONE,
            q_var,
            Some(KNum::from_neo_k(-*q_value)),
            &format!("{label}_neg_q_{idx}"),
        )?;
        let one_minus_p = k_add(
            cs,
            &one,
            &neg_p,
            Some(KNum::from_neo_k(NeoK::ONE - *p_value)),
            &format!("{label}_one_minus_p_{idx}"),
        )?;
        let one_minus_q = k_add(
            cs,
            &one,
            &neg_q,
            Some(KNum::from_neo_k(NeoK::ONE - *q_value)),
            &format!("{label}_one_minus_q_{idx}"),
        )?;
        let prod1_value = (NeoK::ONE - *p_value) * (NeoK::ONE - *q_value);
        let prod1 = k_mul(
            cs,
            &one_minus_p,
            &one_minus_q,
            KNum::from_neo_k(NeoK::ONE - *p_value),
            KNum::from_neo_k(NeoK::ONE - *q_value),
            KNum::from_neo_k(prod1_value),
            delta,
            &format!("{label}_prod1_{idx}"),
        )?;
        let prod2_value = *p_value * *q_value;
        let prod2 = k_mul(
            cs,
            p_var,
            q_var,
            KNum::from_neo_k(*p_value),
            KNum::from_neo_k(*q_value),
            KNum::from_neo_k(prod2_value),
            delta,
            &format!("{label}_prod2_{idx}"),
        )?;
        let term_value = prod1_value + prod2_value;
        let term = k_add(
            cs,
            &prod1,
            &prod2,
            Some(KNum::from_neo_k(term_value)),
            &format!("{label}_term_{idx}"),
        )?;
        let next_acc_value = acc_value * term_value;
        acc = k_mul(
            cs,
            &acc,
            &term,
            KNum::from_neo_k(acc_value),
            KNum::from_neo_k(term_value),
            KNum::from_neo_k(next_acc_value),
            delta,
            &format!("{label}_acc_{idx}"),
        )?;
        acc_value = next_acc_value;
    }

    Ok((acc, acc_value))
}

pub fn range_product<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &KNumVar,
    value_native: NeoK,
    base_b: u32,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, NeoK), SynthesisError> {
    if base_b == 2 {
        let square_value = value_native * value_native;
        let square = k_mul(
            cs,
            value,
            value,
            KNum::from_neo_k(value_native),
            KNum::from_neo_k(value_native),
            KNum::from_neo_k(square_value),
            delta,
            &format!("{label}_square"),
        )?;
        let cube_value = square_value * value_native;
        let cube = k_mul(
            cs,
            &square,
            value,
            KNum::from_neo_k(square_value),
            KNum::from_neo_k(value_native),
            KNum::from_neo_k(cube_value),
            delta,
            &format!("{label}_cube"),
        )?;
        let neg_value = -value_native;
        let neg = k_scalar_mul(
            cs,
            -SpartanF::ONE,
            value,
            Some(KNum::from_neo_k(neg_value)),
            &format!("{label}_neg"),
        )?;
        let out_value = cube_value + neg_value;
        let out = k_add(
            cs,
            &cube,
            &neg,
            Some(KNum::from_neo_k(out_value)),
            &format!("{label}_out"),
        )?;
        return Ok((out, out_value));
    }

    let mut acc = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ONE), &format!("{label}_one"))?;
    let mut acc_value = NeoK::ONE;

    for t in -((base_b as i64) - 1)..=((base_b as i64) - 1) {
        let t_native = NeoK::from(neo_math::F::from_i64(t));
        let neg_t = alloc_constant_k(cs, KNum::from_neo_k(-t_native), &format!("{label}_neg_t_{t}"))?;
        let diff_value = value_native - t_native;
        let diff = k_add(
            cs,
            value,
            &neg_t,
            Some(KNum::from_neo_k(diff_value)),
            &format!("{label}_diff_{t}"),
        )?;
        let next_acc_value = acc_value * diff_value;
        acc = k_mul(
            cs,
            &acc,
            &diff,
            KNum::from_neo_k(acc_value),
            KNum::from_neo_k(diff_value),
            KNum::from_neo_k(next_acc_value),
            delta,
            &format!("{label}_acc_{t}"),
        )?;
        acc_value = next_acc_value;
    }

    Ok((acc, acc_value))
}

pub fn eval_sparse_poly_in_k<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    poly: &SparsePoly<neo_math::F>,
    inputs: &[KNumVar],
    input_values: &[NeoK],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, NeoK), SynthesisError> {
    if inputs.len() != input_values.len() || poly.arity() != inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
    }

    let zero = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ZERO), &format!("{label}_zero"))?;
    let one = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ONE), &format!("{label}_one"))?;
    let mut acc = zero;
    let mut acc_value = NeoK::ZERO;

    for (term_idx, term) in poly.terms().iter().enumerate() {
        let mut term_var = one.clone();
        let mut term_value = NeoK::ONE;

        for (input_idx, exp) in term.exps.iter().copied().enumerate() {
            if exp == 0 {
                continue;
            }
            for pow_idx in 0..exp {
                let next_value = term_value * input_values[input_idx];
                term_var = k_mul(
                    cs,
                    &term_var,
                    &inputs[input_idx],
                    KNum::from_neo_k(term_value),
                    KNum::from_neo_k(input_values[input_idx]),
                    KNum::from_neo_k(next_value),
                    delta,
                    &format!("{label}_term_{term_idx}_var_{input_idx}_pow_{pow_idx}"),
                )?;
                term_value = next_value;
            }
        }

        let coeff_value = NeoK::from(term.coeff);
        let scaled_value = coeff_value * term_value;
        let scaled = k_scalar_mul(
            cs,
            SpartanF::from_canonical_u64(term.coeff.as_canonical_u64()),
            &term_var,
            Some(KNum::from_neo_k(scaled_value)),
            &format!("{label}_term_{term_idx}_scale"),
        )?;
        let next_acc_value = acc_value + scaled_value;
        acc = k_add(
            cs,
            &acc,
            &scaled,
            Some(KNum::from_neo_k(next_acc_value)),
            &format!("{label}_acc_{term_idx}"),
        )?;
        acc_value = next_acc_value;
    }

    Ok((acc, acc_value))
}

pub fn chi_table_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    point_vars: &[KNumVar],
    point_values: &[NeoK],
    delta: SpartanF,
    label: &str,
) -> Result<(Vec<KNumVar>, Vec<NeoK>), SynthesisError> {
    if point_vars.len() != point_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let one = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ONE), &format!("{label}_one"))?;
    let mut out_vars = vec![one.clone()];
    let mut out_values = vec![NeoK::ONE];

    for (bit, (bit_var, bit_value)) in point_vars.iter().zip(point_values.iter()).enumerate() {
        let neg = k_scalar_mul(
            cs,
            -SpartanF::ONE,
            bit_var,
            Some(KNum::from_neo_k(-*bit_value)),
            &format!("{label}_neg_{bit}"),
        )?;
        let one_minus_value = NeoK::ONE - *bit_value;
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

pub fn pow_k_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    base_var: &KNumVar,
    base_value: NeoK,
    exponent: usize,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, NeoK), SynthesisError> {
    let one = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ONE), &format!("{label}_one"))?;
    let mut acc = one;
    let mut acc_value = NeoK::ONE;
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

pub fn dot_k_var_rows<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left_vars: &[KNumVar],
    left_values: &[NeoK],
    right_vars: &[KNumVar],
    right_values: &[NeoK],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, NeoK), SynthesisError> {
    if left_vars.len() != left_values.len()
        || right_vars.len() != right_values.len()
        || left_vars.len() != right_vars.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let zero = alloc_constant_k(cs, KNum::from_neo_k(NeoK::ZERO), &format!("{label}_zero"))?;
    let mut acc = zero;
    let mut acc_value = NeoK::ZERO;
    for (idx, ((left_var, left_value), (right_var, right_value))) in left_vars
        .iter()
        .zip(left_values.iter())
        .zip(right_vars.iter().zip(right_values.iter()))
        .enumerate()
    {
        let product_value = *left_value * *right_value;
        let product = k_mul(
            cs,
            left_var,
            right_var,
            KNum::from_neo_k(*left_value),
            KNum::from_neo_k(*right_value),
            KNum::from_neo_k(product_value),
            delta,
            &format!("{label}_product_{idx}"),
        )?;
        acc_value += product_value;
        acc = k_add(
            cs,
            &acc,
            &product,
            Some(KNum::from_neo_k(acc_value)),
            &format!("{label}_acc_{idx}"),
        )?;
    }
    Ok((acc, acc_value))
}
