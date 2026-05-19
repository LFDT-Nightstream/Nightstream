//! Owns the FE terminal identity RHS and equality check.

use super::*;

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
    me_outputs: &[CircuitCeClaim],
    k_mcs: usize,
    zero_y_ring_suffix_len: usize,
    me_inputs_r_vars: Option<&[KNumVar]>,
    me_inputs_r_values: Option<&[K]>,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let k_total = me_outputs.len();
    if k_total == 0 || k_mcs == 0 || k_mcs > k_total {
        return Err(SynthesisError::Unsatisfiable);
    }
    if zero_y_ring_suffix_len > k_total.saturating_sub(k_mcs) {
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
            zero_y_ring_suffix_len,
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
    me_outputs: &[CircuitCeClaim],
    k_mcs: usize,
    zero_y_ring_suffix_len: usize,
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
        zero_y_ring_suffix_len,
        me_inputs_r_vars,
        me_inputs_r_values,
        delta,
        label,
    )?;
    enforce_k_eq(cs, sumcheck_final, &rhs, &format!("{label}_matches"));
    Ok(rhs_value)
}
