//! Owns FE/NC terminal-identity gadgets for the RV64IM main relation circuit.
//!
//! These gadgets mirror the native optimized-engine RHS formulas over
//! authoritative claim fields. They do not own transcript binding, sumcheck
//! replay, or CE witness-opening checks.

use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

use super::claim::CeClaimVar;
use super::k_field::{alloc_constant_k, alloc_k_constant_k_linear_combination, enforce_k_eq, KNum, KNumVar};
use super::terminal_common::{eval_sparse_poly_in_k, range_product};

pub fn rhs_terminal_identity_fe<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
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

    let eq_alpha_prime_beta_a_value = eq_alpha_prime_values(alpha_prime_values, &public_challenges.beta_a);
    let eq_r_prime_beta_r_value = eq_alpha_prime_values(r_prime_values, &public_challenges.beta_r);
    let eq_beta_value = eq_alpha_prime_beta_a_value * eq_r_prime_beta_r_value;

    let eq_ar_value = if k_total > k_mcs {
        let me_inputs_r_vars = me_inputs_r_vars.ok_or(SynthesisError::Unsatisfiable)?;
        let me_inputs_r_values = me_inputs_r_values.ok_or(SynthesisError::Unsatisfiable)?;
        if me_inputs_r_vars.len() != me_inputs_r_values.len() || me_inputs_r_vars.len() != r_prime_vars.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let eq_alpha_prime_alpha_value = eq_alpha_prime_values(alpha_prime_values, &public_challenges.alpha);
        let eq_r_prime_r_value = eq_alpha_prime_values(r_prime_values, me_inputs_r_values);
        Some(eq_alpha_prime_alpha_value * eq_r_prime_r_value)
    } else {
        None
    };

    let gamma_to_k_value = public_challenges.gamma.exp_u64(k_total as u64);

    let (f_prime, f_prime_value) =
        compute_f_prime(cs, structure, me_outputs, k_mcs, public_challenges.gamma, delta, label)?;

    let chi_alpha_prime_values = chi_table_constant(alpha_prime_values);

    let (eval_sum, eval_sum_value) = if k_total > k_mcs {
        compute_eval_sum(
            cs,
            structure.t(),
            me_outputs,
            k_mcs,
            public_challenges.gamma,
            gamma_to_k_value,
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
    let right_coeff_value = eq_ar_value.unwrap_or(K::ZERO) * gamma_to_k_value;
    let right_value = right_coeff_value * eval_sum_value;
    let rhs_value = left_value + right_value;
    let mut rhs_terms = vec![(KNum::from_neo_k(eq_beta_value), f_prime.c0, f_prime.c1)];
    if right_coeff_value != K::ZERO {
        rhs_terms.push((KNum::from_neo_k(right_coeff_value), eval_sum.c0, eval_sum.c1));
    }
    let rhs = alloc_k_constant_k_linear_combination(
        cs,
        &rhs_terms,
        KNum::from_neo_k(rhs_value),
        delta,
        &format!("{label}_rhs"),
    )?;

    Ok((rhs, rhs_value))
}

fn eq_alpha_prime_values(left: &[K], right: &[K]) -> K {
    let mut acc = K::ONE;
    for (left, right) in left.iter().zip(right.iter()) {
        acc *= (K::ONE - *left) * (K::ONE - *right) + *left * *right;
    }
    acc
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
    _gamma_var: &KNumVar,
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

    let eq_alpha_prime_beta_a_value = eq_alpha_prime_values(alpha_prime_values, &public_challenges.beta_a);
    let eq_s_col_beta_m_value = eq_alpha_prime_values(s_col_prime_values, &public_challenges.beta_m);
    let eq_beta_value = eq_alpha_prime_beta_a_value * eq_s_col_beta_m_value;
    let chi_alpha_prime_values = chi_table_constant(alpha_prime_values);

    let mut nc_terms = Vec::new();
    let mut nc_sum_value = K::ZERO;

    for (output_idx, output) in me_outputs.iter().enumerate() {
        if output.y_zcol.len() < chi_alpha_prime_values.len()
            || output.y_zcol_values.len() < chi_alpha_prime_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (y_eval, y_eval_value) = eval_k_row_with_constant_chi(
            &mut cs.namespace(|| format!("{label}_y_eval_ns_{output_idx}")),
            &output.y_zcol,
            &output.y_zcol_values,
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
        let weighted_value = public_challenges.gamma.exp_u64((output_idx + 1) as u64) * n_i_value;
        nc_sum_value += weighted_value;
        if weighted_value != K::ZERO {
            nc_terms.push((
                KNum::from_neo_k(public_challenges.gamma.exp_u64((output_idx + 1) as u64)),
                n_i.c0,
                n_i.c1,
            ));
        }
    }

    let rhs_value = eq_beta_value * nc_sum_value;
    let nc_sum = alloc_k_constant_k_linear_combination(
        &mut cs.namespace(|| format!("{label}_nc_sum_ns")),
        &nc_terms,
        KNum::from_neo_k(nc_sum_value),
        delta,
        &format!("{label}_nc_sum"),
    )?;
    let rhs = alloc_k_constant_k_linear_combination(
        &mut cs.namespace(|| format!("{label}_rhs_ns")),
        &[(KNum::from_neo_k(eq_beta_value), nc_sum.c0, nc_sum.c1)],
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
    gamma_value: K,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut terms = Vec::new();

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
        let gamma_i_value = public_challenges_gamma_pow(gamma_value, idx);
        let weighted_value = gamma_i_value * f_i_value;
        acc_value += weighted_value;
        if gamma_i_value != K::ZERO {
            terms.push((KNum::from_neo_k(gamma_i_value), f_i.c0, f_i.c1));
        }
    }

    let acc = if terms.is_empty() {
        zero
    } else {
        alloc_k_constant_k_linear_combination(cs, &terms, KNum::from_neo_k(acc_value), delta, &format!("{label}_acc"))?
    };
    Ok((acc, acc_value))
}

fn compute_eval_sum<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    t: usize,
    me_outputs: &[CeClaimVar],
    k_mcs: usize,
    gamma_value: K,
    gamma_to_k_value: K,
    chi_alpha_prime_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut terms = Vec::new();

    for j in 0..t {
        let gamma_k_j_value = public_challenges_gamma_pow(gamma_to_k_value, j);
        for (i_abs, output) in me_outputs.iter().enumerate().skip(k_mcs) {
            if output.y_ring.len() <= j {
                return Err(SynthesisError::Unsatisfiable);
            }
            let row = &output.y_ring[j];
            let row_values = &output.y_ring_values[j];
            if row.len() < chi_alpha_prime_values.len() || row_values.len() < chi_alpha_prime_values.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            let (y_eval, y_eval_value) = eval_k_row_with_constant_chi(
                cs,
                row,
                row_values,
                chi_alpha_prime_values,
                delta,
                &format!("{label}_y_eval_j{j}_i{i_abs}"),
            )?;
            let gamma_i_value = public_challenges_gamma_pow(gamma_value, i_abs);
            let weight_value = gamma_i_value * gamma_k_j_value;
            let contrib_value = weight_value * y_eval_value;
            acc_value += contrib_value;
            if weight_value != K::ZERO {
                terms.push((KNum::from_neo_k(weight_value), y_eval.c0, y_eval.c1));
            }
        }
    }

    let acc = if terms.is_empty() {
        zero
    } else {
        alloc_k_constant_k_linear_combination(cs, &terms, KNum::from_neo_k(acc_value), delta, &format!("{label}_acc"))?
    };
    Ok((acc, acc_value))
}

fn eval_k_row_with_constant_chi<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    row: &[KNumVar],
    row_values: &[K],
    chi_values: &[K],
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    if row.len() < chi_values.len() || row_values.len() < chi_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut terms = Vec::new();
    for idx in 0..chi_values.len() {
        if row_values[idx] == K::ZERO || chi_values[idx] == K::ZERO {
            continue;
        }
        let term_value = row_values[idx] * chi_values[idx];
        acc_value += term_value;
        terms.push((KNum::from_neo_k(chi_values[idx]), row[idx].c0, row[idx].c1));
    }
    let acc = if terms.is_empty() {
        zero
    } else {
        alloc_k_constant_k_linear_combination(cs, &terms, KNum::from_neo_k(acc_value), delta, &format!("{label}_acc"))?
    };
    Ok((acc, acc_value))
}

fn chi_table_constant(point_values: &[K]) -> Vec<K> {
    let mut out_values = vec![K::ONE];

    for bit_value in point_values.iter() {
        let one_minus_value = K::ONE - *bit_value;
        let prior_len = out_values.len();
        let mut next_values = Vec::with_capacity(prior_len * 2);

        for idx in 0..prior_len {
            let next_value = out_values[idx] * one_minus_value;
            next_values.push(next_value);
        }
        for idx in 0..prior_len {
            let next_value = out_values[idx] * *bit_value;
            next_values.push(next_value);
        }

        out_values = next_values;
    }
    out_values
}

fn public_challenges_gamma_pow(base: K, exponent: usize) -> K {
    base.exp_u64(exponent as u64)
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
