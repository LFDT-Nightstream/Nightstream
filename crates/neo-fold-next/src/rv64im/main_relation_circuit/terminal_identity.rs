//! Owns FE/NC terminal-identity gadgets for the RV64IM main relation circuit.
//!
//! These gadgets mirror the native optimized-engine RHS formulas over
//! authoritative claim fields. They do not own transcript binding, sumcheck
//! replay, or CE witness-opening checks.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{alloc_constant_k, enforce_k_eq, k_add, k_mul, KNum, KNumVar};
use super::terminal_common::{eq_points, eval_sparse_poly_in_k, range_product};

pub fn rhs_terminal_identity_fe<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    public_challenges: &neo_reductions::optimized_engine::Challenges,
    alpha_vars: &[KNumVar],
    beta_a_vars: &[KNumVar],
    beta_r_vars: &[KNumVar],
    gamma_var: &KNumVar,
    r_prime_vars: &[KNumVar],
    r_prime_values: &[K],
    alpha_prime_vars: &[KNumVar],
    alpha_prime_values: &[K],
    me_outputs: &[CeClaimVar],
    k_mcs: usize,
    me_inputs_r_vars: Option<&[KNumVar]>,
    me_inputs_r_values: Option<&[K]>,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let k_total = me_outputs.len();
    if k_total == 0 || k_mcs == 0 || k_mcs > k_total {
        return Err(SynthesisError::Unsatisfiable);
    }
    if alpha_vars.len() != public_challenges.alpha.len()
        || beta_a_vars.len() != public_challenges.beta_a.len()
        || beta_r_vars.len() != public_challenges.beta_r.len()
        || r_prime_vars.len() != r_prime_values.len()
        || alpha_prime_vars.len() != alpha_prime_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let (eq_alpha_prime_beta_a, eq_alpha_prime_beta_a_value) = eq_points(
        &mut cs.namespace(|| format!("{label}_eq_alpha_prime_beta_a")),
        alpha_prime_vars,
        beta_a_vars,
        alpha_prime_values,
        &public_challenges.beta_a,
        delta,
        &format!("{label}_eq_alpha_prime_beta_a"),
    )?;
    let (eq_r_prime_beta_r, eq_r_prime_beta_r_value) = eq_points(
        &mut cs.namespace(|| format!("{label}_eq_r_prime_beta_r")),
        r_prime_vars,
        beta_r_vars,
        r_prime_values,
        &public_challenges.beta_r,
        delta,
        &format!("{label}_eq_r_prime_beta_r"),
    )?;
    let eq_beta_value = eq_alpha_prime_beta_a_value * eq_r_prime_beta_r_value;
    let eq_beta = k_mul(
        &mut cs.namespace(|| format!("{label}_eq_beta")),
        &eq_alpha_prime_beta_a,
        &eq_r_prime_beta_r,
        KNum::from_neo_k(eq_alpha_prime_beta_a_value),
        KNum::from_neo_k(eq_r_prime_beta_r_value),
        KNum::from_neo_k(eq_beta_value),
        delta,
        &format!("{label}_eq_beta"),
    )?;

    let eq_ar = if k_total > k_mcs {
        let me_inputs_r_vars = me_inputs_r_vars.ok_or(SynthesisError::Unsatisfiable)?;
        let me_inputs_r_values = me_inputs_r_values.ok_or(SynthesisError::Unsatisfiable)?;
        if me_inputs_r_vars.len() != me_inputs_r_values.len() || me_inputs_r_vars.len() != r_prime_vars.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (eq_alpha_prime_alpha, eq_alpha_prime_alpha_value) = eq_points(
            &mut cs.namespace(|| format!("{label}_eq_alpha_prime_alpha")),
            alpha_prime_vars,
            alpha_vars,
            alpha_prime_values,
            &public_challenges.alpha,
            delta,
            &format!("{label}_eq_alpha_prime_alpha"),
        )?;
        let (eq_r_prime_r, eq_r_prime_r_value) = eq_points(
            &mut cs.namespace(|| format!("{label}_eq_r_prime_r")),
            r_prime_vars,
            me_inputs_r_vars,
            r_prime_values,
            me_inputs_r_values,
            delta,
            &format!("{label}_eq_r_prime_r"),
        )?;
        let eq_ar_value = eq_alpha_prime_alpha_value * eq_r_prime_r_value;
        Some((
            k_mul(
                &mut cs.namespace(|| format!("{label}_eq_ar")),
                &eq_alpha_prime_alpha,
                &eq_r_prime_r,
                KNum::from_neo_k(eq_alpha_prime_alpha_value),
                KNum::from_neo_k(eq_r_prime_r_value),
                KNum::from_neo_k(eq_ar_value),
                delta,
                &format!("{label}_eq_ar"),
            )?,
            eq_ar_value,
        ))
    } else {
        None
    };

    let (gamma_to_k, gamma_to_k_value) = pow_k_var(
        &mut cs.namespace(|| format!("{label}_gamma_to_k")),
        gamma_var,
        public_challenges.gamma,
        k_total,
        delta,
        &format!("{label}_gamma_to_k"),
    )?;

    let (f_prime, f_prime_value) = compute_f_prime(
        cs,
        structure,
        me_outputs,
        k_mcs,
        gamma_var,
        public_challenges.gamma,
        delta,
        label,
    )?;

    let (chi_alpha_prime, chi_alpha_prime_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_alpha_prime")),
        alpha_prime_vars,
        alpha_prime_values,
        delta,
        &format!("{label}_chi_alpha_prime"),
    )?;

    let (eval_sum, eval_sum_value) = if k_total > k_mcs {
        compute_eval_sum(
            cs,
            structure.t(),
            me_outputs,
            k_mcs,
            gamma_var,
            public_challenges.gamma,
            &gamma_to_k,
            gamma_to_k_value,
            &chi_alpha_prime,
            &chi_alpha_prime_values,
            delta,
            &format!("{label}_eval_sum"),
        )?
    } else {
        (
            alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_eval_sum_zero"))?,
            K::ZERO,
        )
    };

    let left_value = eq_beta_value * f_prime_value;
    let left = k_mul(
        &mut cs.namespace(|| format!("{label}_left")),
        &eq_beta,
        &f_prime,
        KNum::from_neo_k(eq_beta_value),
        KNum::from_neo_k(f_prime_value),
        KNum::from_neo_k(left_value),
        delta,
        &format!("{label}_left"),
    )?;

    let (rhs, rhs_value) = if let Some((eq_ar, eq_ar_value)) = eq_ar {
        let right_coeff_value = eq_ar_value * gamma_to_k_value;
        let right_coeff = k_mul(
            &mut cs.namespace(|| format!("{label}_right_coeff")),
            &eq_ar,
            &gamma_to_k,
            KNum::from_neo_k(eq_ar_value),
            KNum::from_neo_k(gamma_to_k_value),
            KNum::from_neo_k(right_coeff_value),
            delta,
            &format!("{label}_right_coeff"),
        )?;
        let right_value = right_coeff_value * eval_sum_value;
        let right = k_mul(
            &mut cs.namespace(|| format!("{label}_right")),
            &right_coeff,
            &eval_sum,
            KNum::from_neo_k(right_coeff_value),
            KNum::from_neo_k(eval_sum_value),
            KNum::from_neo_k(right_value),
            delta,
            &format!("{label}_right"),
        )?;
        let rhs_value = left_value + right_value;
        let rhs = k_add(
            &mut cs.namespace(|| format!("{label}_rhs")),
            &left,
            &right,
            Some(KNum::from_neo_k(rhs_value)),
            &format!("{label}_rhs"),
        )?;
        (rhs, rhs_value)
    } else {
        (left, left_value)
    };

    Ok((rhs, rhs_value))
}

pub fn enforce_terminal_identity_fe<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    sumcheck_final: &KNumVar,
    structure: &CcsStructure<F>,
    public_challenges: &neo_reductions::optimized_engine::Challenges,
    alpha_vars: &[KNumVar],
    beta_a_vars: &[KNumVar],
    beta_r_vars: &[KNumVar],
    _gamma_var: &KNumVar,
    r_prime_vars: &[KNumVar],
    r_prime_values: &[K],
    alpha_prime_vars: &[KNumVar],
    alpha_prime_values: &[K],
    me_outputs: &[CeClaimVar],
    k_mcs: usize,
    me_inputs_r_vars: Option<&[KNumVar]>,
    me_inputs_r_values: Option<&[K]>,
    delta: SpartanF,
    label: &str,
) -> Result<K, SynthesisError> {
    let (rhs, rhs_value) = rhs_terminal_identity_fe(
        cs,
        structure,
        public_challenges,
        alpha_vars,
        beta_a_vars,
        beta_r_vars,
        _gamma_var,
        r_prime_vars,
        r_prime_values,
        alpha_prime_vars,
        alpha_prime_values,
        me_outputs,
        k_mcs,
        me_inputs_r_vars,
        me_inputs_r_values,
        delta,
        label,
    )?;
    enforce_k_eq(cs, sumcheck_final, &rhs, &format!("{label}_matches"));
    Ok(rhs_value)
}

pub fn rhs_terminal_identity_nc<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    params: &NeoParams,
    public_challenges: &neo_reductions::optimized_engine::Challenges,
    beta_a_vars: &[KNumVar],
    beta_m_vars: &[KNumVar],
    gamma_var: &KNumVar,
    s_col_prime_vars: &[KNumVar],
    s_col_prime_values: &[K],
    alpha_prime_vars: &[KNumVar],
    alpha_prime_values: &[K],
    me_outputs: &[CeClaimVar],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if me_outputs.is_empty()
        || beta_a_vars.len() != public_challenges.beta_a.len()
        || beta_m_vars.len() != public_challenges.beta_m.len()
        || s_col_prime_vars.len() != s_col_prime_values.len()
        || alpha_prime_vars.len() != alpha_prime_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let (eq_alpha_prime_beta_a, eq_alpha_prime_beta_a_value) = eq_points(
        &mut cs.namespace(|| format!("{label}_eq_alpha_prime_beta_a")),
        alpha_prime_vars,
        beta_a_vars,
        alpha_prime_values,
        &public_challenges.beta_a,
        delta,
        &format!("{label}_eq_alpha_prime_beta_a"),
    )?;
    let (eq_s_col_beta_m, eq_s_col_beta_m_value) = eq_points(
        &mut cs.namespace(|| format!("{label}_eq_s_col_beta_m")),
        s_col_prime_vars,
        beta_m_vars,
        s_col_prime_values,
        &public_challenges.beta_m,
        delta,
        &format!("{label}_eq_s_col_beta_m"),
    )?;
    let eq_beta_value = eq_alpha_prime_beta_a_value * eq_s_col_beta_m_value;
    let eq_beta = k_mul(
        &mut cs.namespace(|| format!("{label}_eq_beta")),
        &eq_alpha_prime_beta_a,
        &eq_s_col_beta_m,
        KNum::from_neo_k(eq_alpha_prime_beta_a_value),
        KNum::from_neo_k(eq_s_col_beta_m_value),
        KNum::from_neo_k(eq_beta_value),
        delta,
        &format!("{label}_eq_beta"),
    )?;
    let (chi_alpha_prime, chi_alpha_prime_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_alpha_prime")),
        alpha_prime_vars,
        alpha_prime_values,
        delta,
        &format!("{label}_chi_alpha_prime"),
    )?;

    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_nc_sum_zero"))?;
    let mut nc_sum = zero;
    let mut nc_sum_value = K::ZERO;

    for (output_idx, output) in me_outputs.iter().enumerate() {
        if output.y_zcol.len() < chi_alpha_prime_values.len()
            || output.y_zcol_values.len() < chi_alpha_prime_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (y_eval, y_eval_value) = dot_k_row_var(
            &mut cs.namespace(|| format!("{label}_y_eval_ns_{output_idx}")),
            &output.y_zcol,
            &output.y_zcol_values,
            &chi_alpha_prime,
            &chi_alpha_prime_values,
            delta,
            &format!("{label}_y_eval_{output_idx}"),
        )?;
        let (n_i, n_i_value) = range_product(
            &mut cs.namespace(|| format!("{label}_range_ns_{output_idx}")),
            &y_eval,
            y_eval_value,
            params.b,
            delta,
            &format!("{label}_range_{output_idx}"),
        )?;
        let (gamma_i, gamma_i_value) = pow_k_var(
            &mut cs.namespace(|| format!("{label}_gamma_{output_idx}")),
            gamma_var,
            public_challenges.gamma,
            output_idx + 1,
            delta,
            &format!("{label}_gamma_{output_idx}"),
        )?;
        let weighted_value = gamma_i_value * n_i_value;
        let weighted = k_mul(
            &mut cs.namespace(|| format!("{label}_weighted_{output_idx}")),
            &gamma_i,
            &n_i,
            KNum::from_neo_k(gamma_i_value),
            KNum::from_neo_k(n_i_value),
            KNum::from_neo_k(weighted_value),
            delta,
            &format!("{label}_weighted_{output_idx}"),
        )?;
        nc_sum_value += weighted_value;
        nc_sum = k_add(
            &mut cs.namespace(|| format!("{label}_nc_sum_acc_{output_idx}")),
            &nc_sum,
            &weighted,
            Some(KNum::from_neo_k(nc_sum_value)),
            &format!("{label}_nc_sum_acc_{output_idx}"),
        )?;
    }

    let rhs_value = eq_beta_value * nc_sum_value;
    let rhs = k_mul(
        &mut cs.namespace(|| format!("{label}_rhs_ns")),
        &eq_beta,
        &nc_sum,
        KNum::from_neo_k(eq_beta_value),
        KNum::from_neo_k(nc_sum_value),
        KNum::from_neo_k(rhs_value),
        delta,
        &format!("{label}_rhs"),
    )?;
    Ok((rhs, rhs_value))
}

pub fn enforce_terminal_identity_nc<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    sumcheck_final_nc: &KNumVar,
    params: &NeoParams,
    public_challenges: &neo_reductions::optimized_engine::Challenges,
    beta_a_vars: &[KNumVar],
    beta_m_vars: &[KNumVar],
    gamma_var: &KNumVar,
    s_col_prime_vars: &[KNumVar],
    s_col_prime_values: &[K],
    alpha_prime_vars: &[KNumVar],
    alpha_prime_values: &[K],
    me_outputs: &[CeClaimVar],
    delta: SpartanF,
    label: &str,
) -> Result<K, SynthesisError> {
    let (rhs, rhs_value) = rhs_terminal_identity_nc(
        cs,
        params,
        public_challenges,
        beta_a_vars,
        beta_m_vars,
        gamma_var,
        s_col_prime_vars,
        s_col_prime_values,
        alpha_prime_vars,
        alpha_prime_values,
        me_outputs,
        delta,
        label,
    )?;
    enforce_k_eq(cs, sumcheck_final_nc, &rhs, &format!("{label}_matches"));
    Ok(rhs_value)
}

fn compute_f_prime<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    me_outputs: &[CeClaimVar],
    k_mcs: usize,
    gamma_var: &KNumVar,
    gamma_value: K,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut acc = zero;

    for (idx, claim) in me_outputs.iter().take(k_mcs).enumerate() {
        if claim.ct.len() < structure.t() || claim.ct_values.len() < structure.t() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (f_i, f_i_value) = eval_sparse_poly_in_k(
            cs,
            &structure.f,
            &claim.ct[..structure.t()],
            &claim.ct_values[..structure.t()],
            delta,
            &format!("{label}_poly_{idx}"),
        )?;
        let (gamma_i, gamma_i_value) = pow_k_var(
            &mut cs.namespace(|| format!("{label}_gamma_{idx}")),
            gamma_var,
            gamma_value,
            idx,
            delta,
            &format!("{label}_gamma_{idx}"),
        )?;
        let weighted_value = gamma_i_value * f_i_value;
        let weighted = k_mul(
            &mut cs.namespace(|| format!("{label}_weighted_{idx}")),
            &gamma_i,
            &f_i,
            KNum::from_neo_k(gamma_i_value),
            KNum::from_neo_k(f_i_value),
            KNum::from_neo_k(weighted_value),
            delta,
            &format!("{label}_weighted_{idx}"),
        )?;
        acc_value += weighted_value;
        acc = k_add(
            &mut cs.namespace(|| format!("{label}_acc_{idx}")),
            &acc,
            &weighted,
            Some(KNum::from_neo_k(acc_value)),
            &format!("{label}_acc_{idx}"),
        )?;
    }

    Ok((acc, acc_value))
}

fn compute_eval_sum<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    t: usize,
    me_outputs: &[CeClaimVar],
    k_mcs: usize,
    gamma_var: &KNumVar,
    gamma_value: K,
    gamma_to_k_var: &KNumVar,
    gamma_to_k_value: K,
    chi_alpha_prime: &[KNumVar],
    chi_alpha_prime_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut acc = zero;

    for j in 0..t {
        let (gamma_k_j, gamma_k_j_value) = pow_k_var(
            &mut cs.namespace(|| format!("{label}_gamma_to_k_{j}")),
            gamma_to_k_var,
            gamma_to_k_value,
            j,
            delta,
            &format!("{label}_gamma_to_k_{j}"),
        )?;
        for (i_abs, output) in me_outputs.iter().enumerate().skip(k_mcs) {
            if output.y_ring.len() <= j {
                return Err(SynthesisError::Unsatisfiable);
            }
            let row = &output.y_ring[j];
            let row_values = &output.y_ring_values[j];
            if row.len() < chi_alpha_prime_values.len() || row_values.len() < chi_alpha_prime_values.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            let (y_eval, y_eval_value) = dot_k_row_var(
                &mut cs.namespace(|| format!("{label}_y_eval_ns_j{j}_i{i_abs}")),
                row,
                row_values,
                chi_alpha_prime,
                chi_alpha_prime_values,
                delta,
                &format!("{label}_y_eval_j{j}_i{i_abs}"),
            )?;
            let (gamma_i, gamma_i_value) = pow_k_var(
                &mut cs.namespace(|| format!("{label}_gamma_{i_abs}")),
                gamma_var,
                gamma_value,
                i_abs,
                delta,
                &format!("{label}_gamma_{i_abs}"),
            )?;
            let gamma_pair_value = gamma_i_value * gamma_k_j_value;
            let gamma_pair = k_mul(
                &mut cs.namespace(|| format!("{label}_gamma_pair_j{j}_i{i_abs}")),
                &gamma_i,
                &gamma_k_j,
                KNum::from_neo_k(gamma_i_value),
                KNum::from_neo_k(gamma_k_j_value),
                KNum::from_neo_k(gamma_pair_value),
                delta,
                &format!("{label}_gamma_pair_j{j}_i{i_abs}"),
            )?;
            let weight_value = gamma_pair_value;
            let contrib_value = weight_value * y_eval_value;
            let contrib = k_mul(
                &mut cs.namespace(|| format!("{label}_contrib_j{j}_i{i_abs}")),
                &gamma_pair,
                &y_eval,
                KNum::from_neo_k(gamma_pair_value),
                KNum::from_neo_k(y_eval_value),
                KNum::from_neo_k(contrib_value),
                delta,
                &format!("{label}_contrib_j{j}_i{i_abs}"),
            )?;
            acc_value += contrib_value;
            acc = k_add(
                &mut cs.namespace(|| format!("{label}_acc_j{j}_i{i_abs}")),
                &acc,
                &contrib,
                Some(KNum::from_neo_k(acc_value)),
                &format!("{label}_acc_j{j}_i{i_abs}"),
            )?;
        }
    }

    Ok((acc, acc_value))
}

fn dot_k_row_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    row: &[KNumVar],
    row_values: &[K],
    chi: &[KNumVar],
    chi_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if row.len() < chi_values.len() || row_values.len() < chi_values.len() || chi.len() < chi_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut acc = zero;
    for idx in 0..chi_values.len() {
        let term_value = row_values[idx] * chi_values[idx];
        acc_value += term_value;
        let term = k_mul(
            &mut cs.namespace(|| format!("{label}_term_{idx}")),
            &row[idx],
            &chi[idx],
            KNum::from_neo_k(row_values[idx]),
            KNum::from_neo_k(chi_values[idx]),
            KNum::from_neo_k(term_value),
            delta,
            &format!("{label}_term_{idx}"),
        )?;
        acc = k_add(
            &mut cs.namespace(|| format!("{label}_acc_{idx}")),
            &acc,
            &term,
            Some(KNum::from_neo_k(acc_value)),
            &format!("{label}_acc_{idx}"),
        )?;
    }
    Ok((acc, acc_value))
}

fn chi_table_var<CS: ConstraintSystem<SpartanF>>(
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
        let neg = super::k_field::k_scalar_mul(
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

fn pow_k_var<CS: ConstraintSystem<SpartanF>>(
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

#[allow(dead_code)]
pub fn dummy_claim(
    y_ring: Vec<Vec<K>>,
    ct: Vec<K>,
    y_zcol: Vec<K>,
    r: Vec<K>,
    s_col: Vec<K>,
) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(neo_math::D, 1),
        X: neo_ccs::Mat::zero(neo_math::D, 1, F::ZERO),
        r,
        s_col,
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 1,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}
