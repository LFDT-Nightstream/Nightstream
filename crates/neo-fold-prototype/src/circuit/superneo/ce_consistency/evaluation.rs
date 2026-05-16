//! Owns CE opening/evaluation checks over `r`, `s_col`, `y_ring`, and `y_zcol`.

use super::*;

pub(crate) fn debug_paper_dec_child_y_ring_formula_mismatch(
    structure: &CcsStructure<F>,
    witness: &CcsWitness<F>,
    claim: &CeClaim<Commitment, F, K>,
) -> Result<Option<String>, String> {
    validate_superneo_witness_mat(&witness.Z, structure.m).map_err(|err| err.to_string())?;
    let chi_r = tensor_point::<K>(&claim.r);
    for (matrix_idx, matrix) in structure.matrices.iter().enumerate() {
        let row_cap = core::cmp::min(core::cmp::min(matrix.rows(), structure.n), chi_r.len());
        for rho in 0..D {
            let mut acc = K::ZERO;
            for (row, weight) in chi_r.iter().copied().enumerate().take(row_cap) {
                let row_terms = row_ring_projection_terms(matrix, row, witness.Z.cols().saturating_mul(D), rho)
                    .map_err(|err| err.to_string())?;
                let mut row_component = F::ZERO;
                for (logical_col, coeff) in row_terms {
                    let z_coeff = if logical_col < structure.m {
                        witness_mat_get_f(&witness.Z, structure.m, logical_col % D, logical_col)
                    } else if witness.Z.cols() == structure.m.div_ceil(D) {
                        witness.Z[(logical_col % D, logical_col / D)]
                    } else {
                        F::ZERO
                    };
                    row_component += coeff * z_coeff;
                }
                acc += weight.scale_base(row_component);
            }
            let target = claim
                .y_ring
                .get(matrix_idx)
                .and_then(|row| row.get(rho))
                .copied()
                .unwrap_or(K::ZERO);
            if acc != target {
                return Ok(Some(format!(
                    "circuit CE y_ring formula mismatch: matrix={matrix_idx}, rho={rho}, claim={}, formula={}",
                    format_k(target),
                    format_k(acc),
                )));
            }
        }
    }
    Ok(None)
}

pub(super) fn enforce_claim_y_ring_from_point_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    witness: &PackedWitnessVar,
    expected_m: usize,
    active_row_count: usize,
    matrix: &CcsMatrix<F>,
    chi_r: &[KNumVar],
    chi_r_values: &[K],
    active_rho_count: usize,
    target: &[KNumVar],
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    if witness.rows() != D || active_rho_count > D || target.len() < active_rho_count {
        return Err(SynthesisError::Unsatisfiable);
    }
    let row_cap = core::cmp::min(core::cmp::min(matrix.rows(), active_row_count), chi_r.len());
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    for (rho, target) in target.iter().take(active_rho_count).enumerate() {
        let mut acc = zero.clone();
        let mut acc_value = K::ZERO;
        for row in 0..row_cap {
            let row_terms = row_ring_projection_terms(matrix, row, witness_y_ring_width(witness, expected_m)?, rho)?;
            let affine_terms = row_terms
                .iter()
                .map(|(logical_col, coeff)| {
                    let z_var = witness.logical_entry(expected_m, *logical_col)?;
                    let z_value = witness.logical_value(expected_m, *logical_col)?;
                    Ok((
                        z_var,
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        SpartanF::from_canonical_u64(z_value.as_canonical_u64()),
                    ))
                })
                .collect::<Result<Vec<_>, SynthesisError>>()?;
            let row_component = alloc_affine_base(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_component")),
                &affine_terms,
                SpartanF::ZERO,
            )?;
            let row_component_value = row_terms
                .iter()
                .try_fold(F::ZERO, |acc, (logical_col, coeff)| {
                    Ok::<_, SynthesisError>(acc + *coeff * witness.logical_value(expected_m, *logical_col)?)
                })?;
            let term_value = chi_r_values[row].scale_base(row_component_value);
            let term = k_base_mul_var(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_term")),
                &chi_r[row],
                row_component.get_variable(),
                KNum::from_neo_k(chi_r_values[row]),
                SpartanF::from_canonical_u64(row_component_value.as_canonical_u64()),
                KNum::from_neo_k(term_value),
                &format!("{label}_row_{row}_rho_{rho}_term"),
            )?;
            acc_value += term_value;
            acc = k_add(
                &mut cs.namespace(|| format!("{label}_row_{row}_rho_{rho}_acc")),
                &acc,
                &term,
                Some(KNum::from_neo_k(acc_value)),
                &format!("{label}_row_{row}_rho_{rho}_acc"),
            )?;
        }
        let _ = delta;
        enforce_k_eq(cs, target, &acc, &format!("{label}_{rho}"));
    }
    Ok(())
}

fn row_ring_projection_terms(
    matrix: &CcsMatrix<F>,
    row: usize,
    effective_m: usize,
    rho: usize,
) -> Result<Vec<(usize, F)>, SynthesisError> {
    if rho >= D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let block_count = effective_m.div_ceil(D);
    let mut terms = Vec::new();
    for blk in 0..block_count {
        let base = blk * D;
        let mut a = [F::ZERO; D];
        for (off, coeff) in a.iter_mut().enumerate() {
            *coeff = matrix_entry_base_f(matrix, row, base + off);
        }
        if a.iter().all(|value| *value == F::ZERO) {
            continue;
        }
        let a_bar = Rq(superneo_bar_block(a));
        for off in 0..D {
            let logical_col = base + off;
            if logical_col >= effective_m {
                break;
            }
            let mut basis = [F::ZERO; D];
            basis[off] = F::ONE;
            let coeff = a_bar.mul(&Rq(basis)).0[rho];
            if coeff != F::ZERO {
                terms.push((logical_col, coeff));
            }
        }
    }
    Ok(terms)
}

fn witness_y_ring_width(witness: &PackedWitnessVar, expected_m: usize) -> Result<usize, SynthesisError> {
    if expected_m == 0 || witness.rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    if witness.cols() == expected_m.div_ceil(D) {
        return witness
            .cols()
            .checked_mul(D)
            .ok_or(SynthesisError::Unsatisfiable);
    }
    Err(SynthesisError::Unsatisfiable)
}

fn matrix_entry_base_f(matrix: &CcsMatrix<F>, row: usize, col: usize) -> F {
    if row >= matrix.rows() || col >= matrix.cols() {
        return F::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == col {
                F::ONE
            } else {
                F::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let start = csc.col_ptr[col];
            let end = csc.col_ptr[col + 1];
            let mut acc = F::ZERO;
            for idx in start..end {
                if csc.row_idx[idx] == row {
                    acc += csc.vals[idx];
                }
            }
            acc
        }
    }
}

fn format_k(value: K) -> String {
    let [re, im] = value.as_coeffs();
    format!("({}, {})", re.as_canonical_u64(), im.as_canonical_u64())
}

fn alloc_affine_base<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(bellpepper_core::num::AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    constant: SpartanF,
) -> Result<bellpepper_core::num::AllocatedNum<SpartanF>, SynthesisError> {
    let mut value = constant;
    for (_, coeff, term_value) in terms {
        value += *coeff * *term_value;
    }
    let out = bellpepper_core::num::AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc + (constant, CS::one());
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}

pub(super) fn enforce_claim_y_zcol_from_digits_var<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digits: &BalancedDigitWitnessVar,
    expected_m: usize,
    chi_s: &[KNumVar],
    chi_s_values: &[K],
    target: &[KNumVar],
    delta: SpartanF,
    label: &str,
) -> Result<(), SynthesisError> {
    if digits.logical_cols() != expected_m
        || target.len() < D
        || chi_s.len() < expected_m
        || chi_s_values.len() < expected_m
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero = alloc_constant_k(cs, KNum::from_neo_k(K::ZERO), &format!("{label}_zero"))?;
    for (rho, target) in target.iter().enumerate() {
        let mut acc = zero.clone();
        let mut acc_value = K::ZERO;
        if rho < D {
            for logical_col in 0..expected_m {
                let weight = chi_s
                    .get(logical_col)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let weight_value = *chi_s_values
                    .get(logical_col)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let digit = digits
                    .digit_vars(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let digit_value = *digits
                    .digit_values(logical_col)?
                    .get(rho)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let term_value = weight_value * K::from(digit_value);
                let term = k_base_mul_var(
                    &mut cs.namespace(|| format!("{label}_term_{logical_col}_{rho}")),
                    weight,
                    digit.get_variable(),
                    KNum::from_neo_k(weight_value),
                    SpartanF::from_canonical_u64(digit_value.as_canonical_u64()),
                    KNum::from_neo_k(term_value),
                    &format!("{label}_term_{logical_col}_{rho}"),
                )?;
                acc_value += term_value;
                acc = k_add(
                    &mut cs.namespace(|| format!("{label}_acc_{logical_col}_{rho}")),
                    &acc,
                    &term,
                    Some(KNum::from_neo_k(acc_value)),
                    &format!("{label}_acc_{logical_col}_{rho}"),
                )?;
            }
        }
        let _ = delta;
        enforce_k_eq(cs, target, &acc, &format!("{label}_{rho}"));
    }
    Ok(())
}

pub(super) fn chi_table_var<CS: ConstraintSystem<SpartanF>>(
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
            &mut cs.namespace(|| format!("{label}_neg_{bit}")),
            -SpartanF::ONE,
            bit_var,
            Some(KNum::from_neo_k(-*bit_value)),
            &format!("{label}_neg_{bit}"),
        )?;
        let one_minus_value = K::ONE - *bit_value;
        let one_minus = k_add(
            &mut cs.namespace(|| format!("{label}_one_minus_{bit}")),
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
                &mut cs.namespace(|| format!("{label}_zero_branch_{bit}_{idx}")),
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
                &mut cs.namespace(|| format!("{label}_one_branch_{bit}_{idx}")),
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
