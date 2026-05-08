//! Owns the NC terminal identity RHS and equality check.

use super::*;

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
    me_outputs: &[CircuitCeClaim],
    k_mcs: usize,
    zero_y_zcol_suffix_len: usize,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if me_outputs.is_empty()
        || beta_a_vars.len() != public_challenges.beta_a.len()
        || beta_m_vars.len() != public_challenges.beta_m.len()
        || s_col_prime_vars.len() != s_col_prime_values.len()
        || alpha_prime_vars.len() != alpha_prime_values.len()
        || k_mcs > me_outputs.len()
        || zero_y_zcol_suffix_len > me_outputs.len().saturating_sub(k_mcs)
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
    let zero_suffix_start = me_outputs.len() - zero_y_zcol_suffix_len;

    for (output_idx, output) in me_outputs.iter().enumerate() {
        if output_idx >= zero_suffix_start {
            if !claim_has_zero_y_zcol(output) {
                return Err(SynthesisError::Unsatisfiable);
            }
            continue;
        }
        if output.norm_check.y_zcol.len() < chi_alpha_prime_values.len()
            || output.norm_check.y_zcol_values.len() < chi_alpha_prime_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (y_eval, y_eval_value) = dot_k_var_rows(
            &mut cs.namespace(|| format!("{label}_y_eval_ns_{output_idx}")),
            &output.norm_check.y_zcol,
            &output.norm_check.y_zcol_values,
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
    me_outputs: &[CircuitCeClaim],
    k_mcs: usize,
    zero_y_zcol_suffix_len: usize,
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
        k_mcs,
        zero_y_zcol_suffix_len,
        delta,
        label,
    )?;
    enforce_k_eq(cs, sumcheck_final_nc, &rhs, &format!("{label}_matches"));
    Ok(rhs_value)
}
