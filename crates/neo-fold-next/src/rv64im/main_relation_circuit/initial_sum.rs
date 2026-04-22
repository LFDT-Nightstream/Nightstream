//! Owns the FE initial-sum gadget for RV64IM main-relation chunk circuits.

use crate::rv64im::ivc_snark::SpartanF;
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ccs::CcsStructure;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use super::claim::CeClaimVar;
use super::k_field::{alloc_constant_k, k_add, k_mul, KNum, KNumVar};
use super::terminal_common::{chi_table_var, dot_k_var_rows, pow_k_var};

fn claim_has_zero_y_ring(claim: &CeClaimVar, t: usize) -> bool {
    claim
        .y_ring_values
        .iter()
        .take(t)
        .all(|row| row.iter().all(|value| *value == K::ZERO))
}

pub fn claimed_initial_sum_from_me_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    structure: &CcsStructure<F>,
    alpha_vars: &[KNumVar],
    alpha_values: &[K],
    gamma_var: &KNumVar,
    gamma_value: K,
    k_mcs: usize,
    me_inputs: &[CeClaimVar],
    zero_y_ring_suffix_len: usize,
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
    if zero_y_ring_suffix_len > me_inputs.len() {
        return Err(SynthesisError::Unsatisfiable);
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

    let (chi_alpha, chi_alpha_values) = chi_table_var(
        &mut cs.namespace(|| format!("{label}_chi_alpha")),
        alpha_vars,
        alpha_values,
        delta,
        &format!("{label}_chi_alpha"),
    )?;
    let (gamma_to_k, gamma_to_k_value) = pow_k_var(
        &mut cs.namespace(|| format!("{label}_gamma_to_k")),
        gamma_var,
        gamma_value,
        k_total,
        delta,
        &format!("{label}_gamma_to_k"),
    )?;
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_acc_zero"))?;
    let mut total = zero;
    let mut total_value = K::ZERO;
    let zero_suffix_start = me_inputs.len() - zero_y_ring_suffix_len;

    for matrix_idx in 0..structure.t() {
        let (outer, outer_value) = pow_k_var(
            &mut cs.namespace(|| format!("{label}_outer_{matrix_idx}")),
            &gamma_to_k,
            gamma_to_k_value,
            matrix_idx,
            delta,
            &format!("{label}_outer_{matrix_idx}"),
        )?;
        for (me_idx, claim) in me_inputs.iter().enumerate() {
            if me_idx >= zero_suffix_start {
                if !claim_has_zero_y_ring(claim, structure.t()) {
                    return Err(SynthesisError::Unsatisfiable);
                }
                continue;
            }
            let absolute_slot = k_mcs + me_idx + 1;
            let row_vars = &claim.y_ring[matrix_idx];
            let row_values = &claim.y_ring_values[matrix_idx];
            let (y_eval, y_eval_value) = dot_k_var_rows(
                &mut cs.namespace(|| format!("{label}_row_eval_{matrix_idx}_{me_idx}")),
                row_vars,
                row_values,
                &chi_alpha,
                &chi_alpha_values,
                delta,
                &format!("{label}_row_eval_{matrix_idx}_{me_idx}"),
            )?;
            let (gamma_abs, gamma_abs_value) = pow_k_var(
                &mut cs.namespace(|| format!("{label}_gamma_abs_{matrix_idx}_{me_idx}")),
                gamma_var,
                gamma_value,
                absolute_slot - 1,
                delta,
                &format!("{label}_gamma_abs_{matrix_idx}_{me_idx}"),
            )?;
            let gamma_slot_value = gamma_to_k_value * gamma_abs_value;
            let gamma_slot = k_mul(
                &mut cs.namespace(|| format!("{label}_gamma_slot_{matrix_idx}_{me_idx}")),
                &gamma_to_k,
                &gamma_abs,
                KNum::from_neo_k(gamma_to_k_value),
                KNum::from_neo_k(gamma_abs_value),
                KNum::from_neo_k(gamma_slot_value),
                delta,
                &format!("{label}_gamma_slot_{matrix_idx}_{me_idx}"),
            )?;
            let slot_weight_value = gamma_slot_value * outer_value;
            let slot_weight = k_mul(
                &mut cs.namespace(|| format!("{label}_slot_weight_{matrix_idx}_{me_idx}")),
                &gamma_slot,
                &outer,
                KNum::from_neo_k(gamma_slot_value),
                KNum::from_neo_k(outer_value),
                KNum::from_neo_k(slot_weight_value),
                delta,
                &format!("{label}_slot_weight_{matrix_idx}_{me_idx}"),
            )?;
            let contrib_value = slot_weight_value * y_eval_value;
            let contrib = k_mul(
                &mut cs.namespace(|| format!("{label}_contrib_{matrix_idx}_{me_idx}")),
                &slot_weight,
                &y_eval,
                KNum::from_neo_k(slot_weight_value),
                KNum::from_neo_k(y_eval_value),
                KNum::from_neo_k(contrib_value),
                delta,
                &format!("{label}_contrib_{matrix_idx}_{me_idx}"),
            )?;
            total_value += contrib_value;
            total = k_add(
                &mut cs.namespace(|| format!("{label}_acc_{matrix_idx}_{me_idx}")),
                &total,
                &contrib,
                Some(KNum::from_neo_k(total_value)),
                &format!("{label}_acc_{matrix_idx}_{me_idx}"),
            )?;
        }
    }

    Ok((total, total_value))
}
