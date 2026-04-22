//! Owns FE/NC terminal-identity gadgets for the RV64IM main relation circuit.
//!
//! These gadgets mirror the native optimized-engine RHS formulas over
//! authoritative claim fields. They do not own transcript binding, sumcheck
//! replay, or CE witness-opening checks.

use crate::rv64im::ivc_snark::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use super::claim::CeClaimVar;
use super::k_field::{alloc_constant_k, enforce_k_eq, k_add, k_mul, KNum, KNumVar};
use super::terminal_common::{
    chi_table_var, dot_k_var_rows, eq_points, eval_sparse_poly_in_k, pow_k_var, range_product,
};

fn claim_has_zero_y_ring(claim: &CeClaimVar, t: usize) -> bool {
    claim
        .y_ring_values
        .iter()
        .take(t)
        .all(|row| row.iter().all(|value| *value == K::ZERO))
}

fn claim_has_zero_y_zcol(claim: &CeClaimVar) -> bool {
    claim.y_zcol_values.iter().all(|value| *value == K::ZERO)
}

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
    me_outputs: &[CeClaimVar],
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
        if output.y_zcol.len() < chi_alpha_prime_values.len()
            || output.y_zcol_values.len() < chi_alpha_prime_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        let (y_eval, y_eval_value) = dot_k_var_rows(
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
        if claim.y_ring.len() < structure.t() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let ct_values = claim
            .y_ring_values
            .iter()
            .take(structure.t())
            .map(|row| row.first().copied().ok_or(SynthesisError::Unsatisfiable))
            .collect::<Result<Vec<_>, _>>()?;
        let ct_vars = claim
            .y_ring
            .iter()
            .take(structure.t())
            .map(|row| row.first().cloned().ok_or(SynthesisError::Unsatisfiable))
            .collect::<Result<Vec<_>, _>>()?;
        let (f_i, f_i_value) = eval_sparse_poly_in_k(
            &mut cs.namespace(|| format!("{label}_f_i_{idx}")),
            &structure.f,
            &ct_vars,
            &ct_values,
            delta,
            &format!("{label}_f_i_{idx}"),
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
    zero_y_ring_suffix_len: usize,
    delta: SpartanF,
    label: &str,
) -> Result<(KNumVar, K), SynthesisError> {
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    let mut acc_value = K::ZERO;
    let mut acc = zero;
    let zero_suffix_start = me_outputs.len() - zero_y_ring_suffix_len;

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
            if i_abs >= zero_suffix_start {
                if !claim_has_zero_y_ring(output, t) {
                    return Err(SynthesisError::Unsatisfiable);
                }
                continue;
            }
            if output.y_ring.len() <= j {
                return Err(SynthesisError::Unsatisfiable);
            }
            let row_vars = &output.y_ring[j];
            let row_values = &output.y_ring_values[j];
            if row_vars.len() < chi_alpha_prime_values.len() || row_values.len() < chi_alpha_prime_values.len() {
                return Err(SynthesisError::Unsatisfiable);
            }
            let (y_eval, y_eval_value) = dot_k_var_rows(
                &mut cs.namespace(|| format!("{label}_y_eval_ns_j{j}_i{i_abs}")),
                row_vars,
                row_values,
                chi_alpha_prime,
                chi_alpha_prime_values,
                delta,
                &format!("{label}_y_eval_j{j}_i{i_abs}"),
            )?;
            let (gamma_i, gamma_i_value) = pow_k_var(
                &mut cs.namespace(|| format!("{label}_gamma_j{j}_i{i_abs}")),
                gamma_var,
                gamma_value,
                i_abs,
                delta,
                &format!("{label}_gamma_j{j}_i{i_abs}"),
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
