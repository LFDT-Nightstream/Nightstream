use super::basis::*;
use super::constraints::*;
use super::*;

pub(super) fn enforce_y_row_rlc_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CircuitCeClaim],
    rho_mats: &[Mat<F>],
    row_idx: usize,
    d_pad: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (dst_row, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        if dst_row < D {
            for (child, rho) in children.iter().zip(rho_mats.iter()) {
                for src_row in 0..D {
                    let coeff = rho[(dst_row, src_row)];
                    terms.push((
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        child.openings.y_ring[row_idx][src_row].c0,
                        child.openings.y_ring[row_idx][src_row].c1,
                    ));
                }
            }
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{dst_row}"));
    }
    Ok(())
}

pub(super) fn enforce_y_row_rlc_target_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CircuitCeClaim],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    row_idx: usize,
    d_pad: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero_commit_suffix_start = children.len().saturating_sub(zero_commit_suffix_len);
    for (dst_row, target) in target.iter().enumerate() {
        let mut linear_terms = Vec::new();
        let mut terms = Vec::new();
        if dst_row < D {
            for (child_idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
                if child_idx >= zero_commit_suffix_start {
                    if child
                        .openings
                        .y_ring_values
                        .get(row_idx)
                        .map(|row| row.iter().any(|value| *value != K::ZERO))
                        .unwrap_or(false)
                    {
                        return Err(SynthesisError::Unsatisfiable);
                    }
                    continue;
                }
                for src_row in 0..D {
                    let coeff_var = rho.entry(dst_row, src_row)?;
                    let value = child.openings.y_ring_values[row_idx][src_row];
                    if child_idx < constant_child_prefix {
                        let coeffs = value.as_coeffs();
                        linear_terms.push((
                            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                            coeff_var.get_variable(),
                        ));
                        continue;
                    }
                    let coeff_value = rho.entry_value(dst_row, src_row)?;
                    let term = scale_k_by_f_var(
                        cs.namespace(|| format!("{label}_term_{child_idx}_{src_row}_{dst_row}")),
                        &coeff_var,
                        coeff_value,
                        &child.openings.y_ring[row_idx][src_row],
                        value,
                        &format!("{label}_term_{child_idx}_{src_row}_{dst_row}"),
                    )?;
                    terms.push(term);
                }
            }
        }
        enforce_k_affine_sum_eq(cs, target, &linear_terms, &terms, &format!("{label}_{dst_row}"));
    }
    Ok(())
}

pub(super) fn enforce_y_row_rlc_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CircuitCeClaim],
    rhos: &[RotRhoVar],
    row_idx: usize,
    d_pad: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad || children.len() != rhos.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (dst_row, target) in target.iter().enumerate() {
        let mut linear_terms = Vec::new();
        if dst_row < D {
            for (child_idx, child) in children.iter().enumerate() {
                for coeff_idx in 0..D {
                    let (coeff_c0, coeff_c1) =
                        basis_k_row_scale(dst_row, &child.openings.y_ring_values[row_idx], coeff_idx);
                    linear_terms.push((
                        SpartanF::from_canonical_u64(coeff_c0.as_canonical_u64()),
                        SpartanF::from_canonical_u64(coeff_c1.as_canonical_u64()),
                        rhos[child_idx].coeffs[coeff_idx].get_variable(),
                    ));
                }
            }
        }
        enforce_k_affine_sum_eq(cs, target, &linear_terms, &[], &format!("{label}_{dst_row}"));
    }
    Ok(())
}

pub(super) fn enforce_y_row_rlc_eq_dec_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    rlc_children: &[CircuitCeClaim],
    dec_children: &[CeClaim<Commitment, F, K>],
    rhos: &[RotRhoVar],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad || rlc_children.len() != rhos.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    for (dst_row, target) in target.iter().enumerate() {
        let mut linear_terms = Vec::new();
        if dst_row < D {
            for (child_idx, child) in rlc_children.iter().enumerate() {
                for coeff_idx in 0..D {
                    let (coeff_c0, coeff_c1) =
                        basis_k_row_scale(dst_row, &child.openings.y_ring_values[row_idx], coeff_idx);
                    linear_terms.push((
                        SpartanF::from_canonical_u64(coeff_c0.as_canonical_u64()),
                        SpartanF::from_canonical_u64(coeff_c1.as_canonical_u64()),
                        rhos[child_idx].coeffs[coeff_idx].get_variable(),
                    ));
                }
            }
        }
        enforce_k_affine_sum_eq(cs, target, &linear_terms, &[], &format!("{label}_{dst_row}_rlc"));
        let mut pow = K::ONE;
        let mut expected = K::ZERO;
        for child in dec_children {
            let value = child
                .y_ring
                .get(row_idx)
                .and_then(|row| row.get(dst_row))
                .copied()
                .unwrap_or(K::ZERO);
            expected += pow * value;
            pow *= b;
        }
        enforce_k_eq_native(cs, target, expected, &format!("{label}_{dst_row}_dec"));
    }
    Ok(())
}
