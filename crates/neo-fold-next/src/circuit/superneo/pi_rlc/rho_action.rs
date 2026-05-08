use super::basis::*;
use super::constraints::*;
use super::*;

pub(super) fn enforce_rho_left_action_on_dense_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    cols: usize,
    children: &[Vec<AllocatedNum<SpartanF>>],
    column_major: bool,
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    if parent.len() != D * cols {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.len() != D * cols {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, column_major);
            cs.enforce(
                || format!("{label}_{row}_{col}"),
                |lc| {
                    let mut acc = lc;
                    for (child, rho) in children.iter().zip(rho_mats.iter()) {
                        for k in 0..D {
                            let coeff = SpartanF::from_canonical_u64(rho[(row, k)].as_canonical_u64());
                            let child_idx = dense_index(k, col, cols, column_major);
                            acc = acc + (coeff, child[child_idx].get_variable());
                        }
                    }
                    acc
                },
                |lc| lc + CS::one(),
                |lc| lc + parent[parent_idx].get_variable(),
            );
        }
    }
    Ok(())
}

pub(super) fn enforce_rho_left_action_on_dense_f_slices_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    cols: usize,
    children: &[Vec<AllocatedNum<SpartanF>>],
    child_native_values: &[Vec<F>],
    column_major: bool,
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if parent.len() != D * cols
        || children.is_empty()
        || children.len() != rho_mats.len()
        || child_native_values.len() != children.len()
        || constant_child_prefix > children.len()
        || zero_commit_suffix_len > children.len().saturating_sub(constant_child_prefix)
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let zero_commit_suffix_start = children.len().saturating_sub(zero_commit_suffix_len);
    for (child_idx, ((_child, native_child), rho)) in children
        .iter()
        .zip(child_native_values.iter())
        .zip(rho_mats.iter())
        .enumerate()
    {
        let native_child_ok = if child_idx < constant_child_prefix {
            native_child.len() <= D * cols
        } else {
            native_child.len() == D * cols
        };
        if !native_child_ok || rho.entry_value(0, 0).is_err() {
            return Err(SynthesisError::Unsatisfiable);
        }
        if child_idx >= zero_commit_suffix_start && native_child.iter().any(|value| *value != F::ZERO) {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, column_major);
            let mut linear_terms = Vec::new();
            let mut products = Vec::new();
            for (child_idx, ((child, native_child), rho)) in children
                .iter()
                .zip(child_native_values.iter())
                .zip(rho_mats.iter())
                .enumerate()
            {
                if child_idx >= constant_child_prefix && child.len() != D * cols {
                    return Err(SynthesisError::Unsatisfiable);
                }
                if child_idx >= zero_commit_suffix_start {
                    continue;
                }
                for k in 0..D {
                    let coeff = rho.entry(row, k)?;
                    let child_idx_flat = dense_index(k, col, cols, column_major);
                    let child_value = if child_idx < constant_child_prefix {
                        native_child.get(child_idx_flat).copied().unwrap_or(F::ZERO)
                    } else {
                        native_child[child_idx_flat]
                    };
                    if child_idx < constant_child_prefix {
                        linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                        continue;
                    }
                    let product = coeff.mul(
                        cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}_{k}")),
                        &child[child_idx_flat],
                    )?;
                    products.push(product);
                }
            }
            enforce_field_affine_sum_eq(
                cs,
                &parent[parent_idx],
                &linear_terms,
                &products,
                &format!("{label}_eq_{row}_{col}"),
            );
        }
    }
    Ok(())
}

pub(super) fn enforce_rho_left_action_on_canonical_embedded_x_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    cols: usize,
    children: &[Vec<AllocatedNum<SpartanF>>],
    child_native_values: &[Vec<F>],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if parent.len() != D * cols
        || children.is_empty()
        || children.len() != rho_mats.len()
        || child_native_values.len() != children.len()
        || constant_child_prefix > children.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (child_idx, ((_child, native_child), rho)) in children
        .iter()
        .zip(child_native_values.iter())
        .zip(rho_mats.iter())
        .enumerate()
    {
        let native_child_ok = if child_idx < constant_child_prefix {
            native_child.len() == cols || native_child.len() == D * cols
        } else {
            native_child.len() == cols || native_child.len() == D * cols
        };
        if !native_child_ok || rho.entry_value(0, 0).is_err() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, false);
            let mut linear_terms = Vec::new();
            let mut products = Vec::new();
            for (child_idx, ((child, native_child), rho)) in children
                .iter()
                .zip(child_native_values.iter())
                .zip(rho_mats.iter())
                .enumerate()
            {
                let compact_child = native_child.len() == cols;
                if compact_child {
                    let logical_start = col.checked_mul(D).ok_or(SynthesisError::Unsatisfiable)?;
                    let logical_end = logical_start.saturating_add(D).min(cols);
                    for logical_col in logical_start..logical_end {
                        let active_lane = logical_col % D;
                        let coeff = rho.entry(row, active_lane)?;
                        let child_value = native_child[logical_col];
                        if child_idx < constant_child_prefix {
                            linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                            continue;
                        }
                        let child_var = child
                            .get(logical_col)
                            .ok_or(SynthesisError::Unsatisfiable)?;
                        let product = coeff.mul(
                            cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}_{logical_col}")),
                            child_var,
                        )?;
                        products.push(product);
                    }
                } else {
                    for in_lane in 0..D {
                        let coeff = rho.entry(row, in_lane)?;
                        let child_idx_flat = dense_index(in_lane, col, cols, false);
                        let child_value = native_child[child_idx_flat];
                        if child_idx < constant_child_prefix {
                            linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                            continue;
                        }
                        let child_var = child
                            .get(child_idx_flat)
                            .ok_or(SynthesisError::Unsatisfiable)?;
                        let product = coeff.mul(
                            cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}_{in_lane}")),
                            child_var,
                        )?;
                        products.push(product);
                    }
                }
            }
            enforce_field_affine_sum_eq(
                cs,
                &parent[parent_idx],
                &linear_terms,
                &products,
                &format!("{label}_eq_{row}_{col}"),
            );
        }
    }
    Ok(())
}

pub(super) fn enforce_rho_coeff_left_action_on_canonical_embedded_x_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    cols: usize,
    children: &[Vec<AllocatedNum<SpartanF>>],
    child_native_values: &[Vec<F>],
    rhos: &[RotRhoVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    let active_children_len = children.len().saturating_sub(zero_commit_suffix_len);
    if parent.len() != D * cols
        || children.is_empty()
        || children.len() != rhos.len()
        || child_native_values.len() != children.len()
        || constant_child_prefix > active_children_len
        || zero_commit_suffix_len > children.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (child_idx, ((_child, native_child), rho)) in children
        .iter()
        .take(active_children_len)
        .zip(child_native_values.iter())
        .zip(rhos.iter().take(active_children_len))
        .enumerate()
    {
        let native_child_ok = if child_idx < constant_child_prefix {
            native_child.len() == cols || native_child.len() == D * cols
        } else {
            native_child.len() == cols || native_child.len() == D * cols
        };
        if !native_child_ok || rho.coeffs.len() != D || rho.coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, false);
            let mut linear_terms = Vec::new();
            let mut products = Vec::new();
            let mut native_expected = F::ZERO;
            for (child_idx, ((child, native_child), rho)) in children
                .iter()
                .take(active_children_len)
                .zip(child_native_values.iter())
                .zip(rhos.iter().take(active_children_len))
                .enumerate()
            {
                let compact_child = native_child.len() == cols;
                if compact_child {
                    let logical_start = col.checked_mul(D).ok_or(SynthesisError::Unsatisfiable)?;
                    let logical_end = logical_start.saturating_add(D).min(cols);
                    for logical_col in logical_start..logical_end {
                        let active_lane = logical_col % D;
                        let coeff = alloc_rot_rho_entry_from_coeffs(
                            cs.namespace(|| format!("{label}_coeff_{row}_{col}_{child_idx}_{logical_col}")),
                            rho,
                            row,
                            active_lane,
                            &format!("{label}_coeff_{row}_{col}_{child_idx}_{logical_col}"),
                        )?;
                        let child_value = native_child[logical_col];
                        let mut native_coeff = F::ZERO;
                        for coeff_idx in 0..D {
                            native_coeff +=
                                GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, active_lane)] * rho.coeff_values[coeff_idx];
                        }
                        native_expected += native_coeff * child_value;
                        if child_idx < constant_child_prefix {
                            linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                            continue;
                        }
                        let child_var = child
                            .get(logical_col)
                            .ok_or(SynthesisError::Unsatisfiable)?;
                        let product = coeff.mul(
                            cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}_{logical_col}")),
                            child_var,
                        )?;
                        products.push(product);
                    }
                } else {
                    for in_lane in 0..D {
                        let coeff = alloc_rot_rho_entry_from_coeffs(
                            cs.namespace(|| format!("{label}_coeff_{row}_{col}_{child_idx}_{in_lane}")),
                            rho,
                            row,
                            in_lane,
                            &format!("{label}_coeff_{row}_{col}_{child_idx}_{in_lane}"),
                        )?;
                        let child_idx_flat = dense_index(in_lane, col, cols, false);
                        let child_value = native_child[child_idx_flat];
                        let mut native_coeff = F::ZERO;
                        for coeff_idx in 0..D {
                            native_coeff +=
                                GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, in_lane)] * rho.coeff_values[coeff_idx];
                        }
                        native_expected += native_coeff * child_value;
                        if child_idx < constant_child_prefix {
                            linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                            continue;
                        }
                        let child_var = child
                            .get(child_idx_flat)
                            .ok_or(SynthesisError::Unsatisfiable)?;
                        let product = coeff.mul(
                            cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}_{in_lane}")),
                            child_var,
                        )?;
                        products.push(product);
                    }
                }
            }
            enforce_field_affine_sum_eq(
                cs,
                &parent[parent_idx],
                &linear_terms,
                &products,
                &format!("{label}_eq_{row}_{col}"),
            );
        }
    }
    Ok(())
}

pub(super) fn enforce_rho_coeff_left_action_on_dense_constant_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    parent_values: &[F],
    cols: usize,
    child_native_values: &[Vec<F>],
    column_major: bool,
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if child_native_values.is_empty()
        || child_native_values.len() != rhos.len()
        || (!parent.is_empty() && parent.len() != D * cols)
        || parent_values.len() != D * cols
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (native_child, rho) in child_native_values.iter().zip(rhos.iter()) {
        if native_child.len() != D * cols || rho.coeffs.len() != D || rho.coeff_values.len() != D {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, column_major);
            let mut linear_terms = Vec::new();
            for (child_idx, native_child) in child_native_values.iter().enumerate() {
                for coeff_idx in 0..D {
                    let coeff = basis_dense_f_scale(row, col, cols, column_major, native_child, coeff_idx);
                    linear_terms.push((
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        rhos[child_idx].coeffs[coeff_idx].clone(),
                    ));
                }
            }
            if parent.is_empty() {
                cs.enforce(
                    || format!("{label}_eq_{row}_{col}"),
                    |lc| {
                        let mut acc = lc;
                        for (coeff, var) in &linear_terms {
                            acc = acc + (*coeff, var.get_variable());
                        }
                        acc
                    },
                    |lc| lc + CS::one(),
                    |lc| {
                        lc + (
                            SpartanF::from_canonical_u64(parent_values[parent_idx].as_canonical_u64()),
                            CS::one(),
                        )
                    },
                );
            } else {
                enforce_field_affine_sum_eq(
                    cs,
                    &parent[parent_idx],
                    &linear_terms,
                    &[],
                    &format!("{label}_eq_{row}_{col}"),
                );
            }
        }
    }
    Ok(())
}

pub(super) fn enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children<
    CS: ConstraintSystem<SpartanF>,
>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    parent_values: &[F],
    cols: usize,
    child_native_values: &[Vec<F>],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if child_native_values.is_empty()
        || child_native_values.len() != rhos.len()
        || (!parent.is_empty() && parent.len() != D * cols)
        || parent_values.len() != D * cols
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (native_child, rho) in child_native_values.iter().zip(rhos.iter()) {
        if (native_child.len() != cols && native_child.len() != D * cols)
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    for row in 0..D {
        for col in 0..cols {
            let parent_idx = dense_index(row, col, cols, false);
            let mut linear_terms = Vec::new();
            for (child_idx, native_child) in child_native_values.iter().enumerate() {
                let compact_child = native_child.len() == cols;
                if compact_child {
                    let logical_start = col.checked_mul(D).ok_or(SynthesisError::Unsatisfiable)?;
                    let logical_end = logical_start.saturating_add(D).min(cols);
                    for logical_col in logical_start..logical_end {
                        let active_lane = logical_col % D;
                        let value = native_child[logical_col];
                        if value == F::ZERO {
                            continue;
                        }
                        for coeff_idx in 0..D {
                            let basis_coeff = GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, active_lane)];
                            if basis_coeff == F::ZERO {
                                continue;
                            }
                            linear_terms.push((
                                SpartanF::from_canonical_u64((basis_coeff * value).as_canonical_u64()),
                                rhos[child_idx].coeffs[coeff_idx].clone(),
                            ));
                        }
                    }
                } else {
                    for in_lane in 0..D {
                        let value = native_child[dense_index(in_lane, col, cols, false)];
                        if value == F::ZERO {
                            continue;
                        }
                        for coeff_idx in 0..D {
                            let basis_coeff = GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, in_lane)];
                            if basis_coeff == F::ZERO {
                                continue;
                            }
                            linear_terms.push((
                                SpartanF::from_canonical_u64((basis_coeff * value).as_canonical_u64()),
                                rhos[child_idx].coeffs[coeff_idx].clone(),
                            ));
                        }
                    }
                }
            }
            if parent.is_empty() {
                cs.enforce(
                    || format!("{label}_eq_{row}_{col}"),
                    |lc| {
                        let mut acc = lc;
                        for (coeff, var) in &linear_terms {
                            acc = acc + (*coeff, var.get_variable());
                        }
                        acc
                    },
                    |lc| lc + CS::one(),
                    |lc| {
                        lc + (
                            SpartanF::from_canonical_u64(parent_values[parent_idx].as_canonical_u64()),
                            CS::one(),
                        )
                    },
                );
            } else {
                enforce_field_affine_sum_eq(
                    cs,
                    &parent[parent_idx],
                    &linear_terms,
                    &[],
                    &format!("{label}_eq_{row}_{col}"),
                );
            }
        }
    }
    Ok(())
}
