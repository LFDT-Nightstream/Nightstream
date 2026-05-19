use super::constraints::*;
use super::rho_action::*;
use super::y_rows::*;
use super::*;

pub fn enforce_rlc_public<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || rho.rows() != D
            || rho.cols() != D
            || child.openings.r_values != parent.openings.r_values
            || child.openings.y_ring.len() != parent.openings.y_ring.len()
            || child.norm_check.y_zcol.len() != parent.norm_check.y_zcol.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }

    enforce_rho_left_action_on_dense_f_slices(
        cs,
        &parent.public_input.x,
        parent.public_input.cols,
        &children
            .iter()
            .map(|child| child.public_input.x.clone())
            .collect::<Vec<_>>(),
        false,
        rho_mats,
        &format!("{label}_x"),
    )?;

    let commitment_cols = parent.commitment.data.len() / D;
    if commitment_cols * D != parent.commitment.data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.commitment.data.len() != parent.commitment.data.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    enforce_rho_left_action_on_dense_f_slices(
        cs,
        &parent.commitment.data,
        commitment_cols,
        &children
            .iter()
            .map(|child| child.commitment.data.clone())
            .collect::<Vec<_>>(),
        true,
        rho_mats,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len());
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_rlc_target(
            cs,
            &parent.openings.y_ring[idx],
            children,
            rho_mats,
            idx,
            d_pad,
            &format!("{label}_y_{idx}"),
        )?;
    }

    Ok(())
}

pub fn enforce_rlc_public_with_rho_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars_constant_prefix_zero_commit_suffix(cs, parent, children, rho_mats, 0, 0, label)
}

pub fn enforce_rlc_public_with_rho_vars_constant_prefix<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars_constant_prefix_zero_commit_suffix(
        cs,
        parent,
        children,
        rho_mats,
        constant_child_prefix,
        0,
        label,
    )
}

pub fn enforce_rlc_public_with_rho_vars_constant_prefix_zero_commit_suffix<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || constant_child_prefix > children.len()
        || zero_commit_suffix_len > children.len().saturating_sub(constant_child_prefix)
        || parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        let zero_commit_suffix = idx >= children.len().saturating_sub(zero_commit_suffix_len);
        let child_c_data_ok = if idx < constant_child_prefix || zero_commit_suffix {
            child.commitment.data.is_empty() || child.commitment.data.len() == parent.commitment.data.len()
        } else {
            child.commitment.data.len() == parent.commitment.data.len()
        };
        let child_y_ring_ok = if zero_commit_suffix {
            child.openings.y_ring.is_empty() || child.openings.y_ring.len() == parent.openings.y_ring.len()
        } else {
            child.openings.y_ring.len() == parent.openings.y_ring.len()
        };
        let child_y_zcol_ok = if zero_commit_suffix {
            child.norm_check.y_zcol.is_empty() || child.norm_check.y_zcol.len() == parent.norm_check.y_zcol.len()
        } else {
            child.norm_check.y_zcol.len() == parent.norm_check.y_zcol.len()
        };
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || child.openings.r_values != parent.openings.r_values
            || !child_y_ring_ok
            || !child_y_zcol_ok
            || !child_c_data_ok
            || rho.entry_value(0, 0).is_err()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }

    enforce_rho_left_action_on_canonical_embedded_x_with_vars(
        cs,
        &parent.public_input.x,
        parent.public_input.cols,
        &children
            .iter()
            .map(|child| child.public_input.x.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.public_input.x_values.clone())
            .collect::<Vec<_>>(),
        rho_mats,
        constant_child_prefix,
        &format!("{label}_x"),
    )?;

    enforce_rho_left_action_on_dense_f_slices_with_vars(
        cs,
        &parent.commitment.data,
        parent.commitment.data.len() / D,
        &children
            .iter()
            .map(|child| child.commitment.data.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.commitment.data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rho_mats,
        constant_child_prefix,
        zero_commit_suffix_len,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len());
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_rlc_target_with_vars(
            cs,
            &parent.openings.y_ring[idx],
            children,
            rho_mats,
            constant_child_prefix,
            zero_commit_suffix_len,
            idx,
            d_pad,
            &format!("{label}_y_{idx}"),
        )?;
    }

    Ok(())
}

pub fn enforce_rlc_public_with_split_rho_views_constant_prefix_zero_commit_suffix<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rhos: &[RotRhoVar],
    rho_mats_active: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    let active_children_len = children.len().saturating_sub(zero_commit_suffix_len);
    let mixed_constant_prefix = constant_child_prefix > 0 && constant_child_prefix < active_children_len;
    if children.is_empty()
        || children.len() != rhos.len()
        || (mixed_constant_prefix && rho_mats_active.len() != active_children_len)
        || (!mixed_constant_prefix && !rho_mats_active.is_empty())
        || constant_child_prefix > active_children_len
        || parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        let zero_commit_suffix = idx >= active_children_len;
        let child_c_data_ok = if idx < constant_child_prefix || zero_commit_suffix {
            child.commitment.data.is_empty() || child.commitment.data.len() == parent.commitment.data.len()
        } else {
            child.commitment.data.len() == parent.commitment.data.len()
        };
        let child_y_ring_ok = if zero_commit_suffix {
            child.openings.y_ring.is_empty() || child.openings.y_ring.len() == parent.openings.y_ring.len()
        } else {
            child.openings.y_ring.len() == parent.openings.y_ring.len()
        };
        let child_y_zcol_ok = if zero_commit_suffix {
            child.norm_check.y_zcol.is_empty() || child.norm_check.y_zcol.len() == parent.norm_check.y_zcol.len()
        } else {
            child.norm_check.y_zcol.len() == parent.norm_check.y_zcol.len()
        };
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || child.openings.r_values != parent.openings.r_values
            || !child_y_ring_ok
            || !child_y_zcol_ok
            || !child_c_data_ok
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }
    if mixed_constant_prefix {
        for rho in rho_mats_active {
            if rho.entry_value(0, 0).is_err() {
                return Err(SynthesisError::Unsatisfiable);
            }
        }
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_with_vars(
        cs,
        &parent.public_input.x,
        parent.public_input.cols,
        &children
            .iter()
            .map(|child| child.public_input.x.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.public_input.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        constant_child_prefix,
        0,
        &format!("{label}_x"),
    )?;

    ensure_zero_commit_suffix(children, zero_commit_suffix_len)?;

    let active_children = &children[..active_children_len];
    let active_c_vars = active_children
        .iter()
        .map(|child| child.commitment.data.clone())
        .collect::<Vec<_>>();
    let active_c_values = active_children
        .iter()
        .map(|child| child.commitment.data_values.clone())
        .collect::<Vec<_>>();
    if active_children.is_empty() {
        for (idx, entry) in parent.commitment.data.iter().enumerate() {
            enforce_field_affine_sum_eq(cs, entry, &[], &[], &format!("{label}_c_eq_{idx}"));
        }
    } else if constant_child_prefix == 0 {
        ring_action::enforce_rho_coeff_left_action_on_dense_commitment_columns_toom3_with_vars(
            cs,
            &parent.commitment.data,
            parent.commitment.data.len() / D,
            &active_c_vars,
            &active_c_values,
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else if constant_child_prefix == active_children_len {
        enforce_rho_coeff_left_action_on_dense_constant_f_slices(
            cs,
            &parent.commitment.data,
            &parent.commitment.data_values,
            parent.commitment.data.len() / D,
            &active_c_values,
            true,
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else {
        enforce_rho_left_action_on_dense_f_slices_with_vars(
            cs,
            &parent.commitment.data,
            parent.commitment.data.len() / D,
            &active_c_vars,
            &active_c_values,
            true,
            rho_mats_active,
            constant_child_prefix,
            0,
            &format!("{label}_c"),
        )?;
    }

    let d_pad = parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len());
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        if active_children.is_empty() {
            for (dst_row, target) in parent.openings.y_ring[idx].iter().enumerate() {
                enforce_k_affine_sum_eq(cs, target, &[], &[], &format!("{label}_y_{idx}_{dst_row}"));
            }
        } else if constant_child_prefix == 0 {
            ring_action::enforce_rho_coeff_left_action_on_y_row_toom3_with_vars(
                cs,
                &parent.openings.y_ring[idx],
                &active_children
                    .iter()
                    .map(|child| child.openings.y_ring[idx].clone())
                    .collect::<Vec<_>>(),
                &active_children
                    .iter()
                    .map(|child| child.openings.y_ring_values[idx].clone())
                    .collect::<Vec<_>>(),
                &rhos[..active_children_len],
                &format!("{label}_y_{idx}"),
            )?;
        } else if constant_child_prefix == active_children_len {
            enforce_y_row_rlc_target_with_rho_coeffs(
                cs,
                &parent.openings.y_ring[idx],
                active_children,
                &rhos[..active_children_len],
                idx,
                d_pad,
                &format!("{label}_y_{idx}"),
            )?;
        } else {
            enforce_y_row_rlc_target_with_vars(
                cs,
                &parent.openings.y_ring[idx],
                active_children,
                rho_mats_active,
                constant_child_prefix,
                0,
                idx,
                d_pad,
                &format!("{label}_y_{idx}"),
            )?;
        }
    }

    Ok(())
}

pub fn enforce_rlc_public_with_rho_coeffs_for_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rhos.len()
        || parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || child.openings.r_values != parent.openings.r_values
            || child.openings.y_ring.len() != parent.openings.y_ring.len()
            || child.norm_check.y_zcol.len() != parent.norm_check.y_zcol.len()
            || child.public_input.x_values.len() != D * parent.public_input.m_in
            || child.commitment.data_values.len() != parent.commitment.data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children(
        cs,
        &parent.public_input.x,
        &parent.public_input.x_values,
        parent.public_input.cols,
        &children
            .iter()
            .map(|child| child.public_input.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        &format!("{label}_x"),
    )?;

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.commitment.data,
        &parent.commitment.data_values,
        parent.commitment.data_values.len() / D,
        &children
            .iter()
            .map(|child| child.commitment.data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rhos,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len());
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_rlc_target_with_rho_coeffs(
            cs,
            &parent.openings.y_ring[idx],
            children,
            rhos,
            idx,
            d_pad,
            &format!("{label}_y_{idx}"),
        )?;
    }

    Ok(())
}

pub fn enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    rlc_children: &[CircuitCeClaim],
    dec_children: &[CeClaim<Commitment, F, K>],
    rhos: &[RotRhoVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if rlc_children.is_empty()
        || rlc_children.len() != rhos.len()
        || dec_children.is_empty()
        || parent.public_input.rows != D
        || parent.public_input.cols != parent.public_input.m_in
        || parent.openings.r.len() != parent.openings.r_values.len()
        || parent.norm_check.y_zcol.len() != parent.norm_check.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in rlc_children.iter().zip(rhos.iter()).enumerate() {
        if child.public_input.m_in != parent.public_input.m_in
            || child.public_input.rows != D
            || child.public_input.cols != parent.public_input.m_in
            || child.openings.r_values != parent.openings.r_values
            || child.openings.y_ring_values.len() != parent.openings.y_ring_values.len()
            || child.public_input.x_values.len() != D * parent.public_input.m_in
            || child.commitment.data_values.len() != parent.commitment.data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.openings.r, &child.openings.r, &format!("{label}_r_{idx}"))?;
    }
    for child in dec_children {
        if child.m_in != parent.public_input.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.public_input.m_in
            || child.r != parent.openings.r_values
            || child.y_ring.len() != parent.openings.y_ring_values.len()
            || child.y_zcol.len() != parent.norm_check.y_zcol_values.len()
            || child.c.data.len() != parent.commitment.data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children(
        cs,
        &parent.public_input.x,
        &parent.public_input.x_values,
        parent.public_input.cols,
        &rlc_children
            .iter()
            .map(|child| child.public_input.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        &format!("{label}_x"),
    )?;

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.commitment.data,
        &parent.commitment.data_values,
        parent.commitment.data_values.len() / D,
        &rlc_children
            .iter()
            .map(|child| child.commitment.data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rhos,
        &format!("{label}_c"),
    )?;

    let d_pad = parent
        .openings
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.norm_check.y_zcol_values.len());
    for (idx, row) in parent.openings.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_rlc_eq_dec_target_with_rho_coeffs(
            cs,
            &parent.openings.y_ring[idx],
            rlc_children,
            dec_children,
            rhos,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
    }
    Ok(())
}

pub fn enforce_rlc_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public(cs, parent, children, rho_mats, label)
}

pub fn enforce_rlc_public_non_commitment_with_rho_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CircuitCeClaim,
    children: &[CircuitCeClaim],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars(cs, parent, children, rho_mats, label)
}
