//! Owns shared FE/NC terminal identity RHS helper sums.

use super::*;

pub(super) fn compute_f_prime<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    me_outputs: &[CircuitCeClaim],
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
        if claim.openings.y_ring.len() < structure.t() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let ct_values = claim
            .openings
            .y_ring_values
            .iter()
            .take(structure.t())
            .map(|row| row.first().copied().ok_or(SynthesisError::Unsatisfiable))
            .collect::<Result<Vec<_>, _>>()?;
        let ct_vars = claim
            .openings
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

pub(super) fn compute_eval_sum<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    t: usize,
    me_outputs: &[CircuitCeClaim],
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
            if output.openings.y_ring.len() <= j {
                return Err(SynthesisError::Unsatisfiable);
            }
            let row_vars = &output.openings.y_ring[j];
            let row_values = &output.openings.y_ring_values[j];
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
