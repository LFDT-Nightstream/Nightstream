#![allow(dead_code)]

use crate::rv64im::ivc_snark::{Rv64imDeciderEngine, ShapeCS, SpartanF};
use bellpepper_core::test_cs::TestConstraintSystem;

use super::*;

#[derive(Clone, Debug)]
pub(crate) struct RlcPublicStageCheckpoints {
    stage_ends: Vec<(String, usize)>,
}

impl RlcPublicStageCheckpoints {
    fn push(&mut self, name: String, end: usize) {
        self.stage_ends.push((name, end));
    }

    pub(crate) fn stage_ends(&self) -> &[(String, usize)] {
        &self.stage_ends
    }

    pub(crate) fn phase_for_row(&self, row: usize) -> Option<(String, usize)> {
        let mut start = 0usize;
        for (name, end) in &self.stage_ends {
            if row < *end {
                return Some((name.clone(), row - start));
            }
            start = *end;
        }
        None
    }
}

fn rlc_stage_err(cs: &TestConstraintSystem<SpartanF>, stage: &str) -> String {
    let unsat = cs.which_is_unsatisfied().unwrap_or("unknown");
    format!("{stage}: {unsat}")
}

fn checkpoint(cs: &TestConstraintSystem<SpartanF>, stage: &str) -> Result<(), String> {
    if cs.is_satisfied() {
        Ok(())
    } else {
        Err(rlc_stage_err(cs, stage))
    }
}

pub(crate) fn debug_locate_rlc_public_with_split_rho_views_stage(
    cs: &mut TestConstraintSystem<SpartanF>,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
    rho_mats_active: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), String> {
    let active_children_len = children.len().saturating_sub(zero_commit_suffix_len);
    if children.is_empty()
        || children.len() != rhos.len()
        || rho_mats_active.len() != active_children_len
        || constant_child_prefix > active_children_len
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err("preflight".into());
    }

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        let zero_commit_suffix = idx >= active_children_len;
        let child_c_data_ok = if idx < constant_child_prefix || zero_commit_suffix {
            child.c_data.is_empty() || child.c_data.len() == parent.c_data.len()
        } else {
            child.c_data.len() == parent.c_data.len()
        };
        let child_y_ring_ok = if zero_commit_suffix {
            child.y_ring.is_empty() || child.y_ring.len() == parent.y_ring.len()
        } else {
            child.y_ring.len() == parent.y_ring.len()
        };
        let child_y_zcol_ok = if zero_commit_suffix {
            child.y_zcol.is_empty() || child.y_zcol.len() == parent.y_zcol.len()
        } else {
            child.y_zcol.len() == parent.y_zcol.len()
        };
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || !child_y_ring_ok
            || !child_y_zcol_ok
            || !child_c_data_ok
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(format!("preflight_child_{idx}"));
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))
            .map_err(|err| format!("r_{idx}: {err}"))?;
        checkpoint(cs, &format!("shared_point_{idx}"))?;
    }

    for rho in rho_mats_active {
        if rho.entry_value(0, 0).is_err() {
            return Err("preflight_rho_mat".into());
        }
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_with_vars(
        cs,
        &parent.x,
        parent.x_cols,
        &children
            .iter()
            .map(|child| child.x.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        constant_child_prefix,
        zero_commit_suffix_len,
        &format!("{label}_x"),
    )
    .map_err(|err| format!("x: {err}"))?;
    checkpoint(cs, "x")?;

    ensure_zero_commit_suffix(children, zero_commit_suffix_len).map_err(|err| format!("c_suffix: {err}"))?;
    let active_children = &children[..active_children_len];
    if active_children.is_empty() {
        for (idx, entry) in parent.c_data.iter().enumerate() {
            enforce_field_affine_sum_eq(cs, entry, &[], &[], &format!("{label}_c_eq_{idx}"));
        }
    } else if constant_child_prefix == 0 {
        super::ring_action::enforce_rho_coeff_left_action_on_dense_commitment_columns_toom3_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
            &active_children
                .iter()
                .map(|child| child.c_data.clone())
                .collect::<Vec<_>>(),
            &active_children
                .iter()
                .map(|child| child.c_data_values.clone())
                .collect::<Vec<_>>(),
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )
        .map_err(|err| format!("c: {err}"))?;
    } else {
        enforce_rho_left_action_on_dense_f_slices_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
            &active_children
                .iter()
                .map(|child| child.c_data.clone())
                .collect::<Vec<_>>(),
            &active_children
                .iter()
                .map(|child| child.c_data_values.clone())
                .collect::<Vec<_>>(),
            true,
            rho_mats_active,
            constant_child_prefix,
            0,
            &format!("{label}_c"),
        )
        .map_err(|err| format!("c: {err}"))?;
    }
    checkpoint(cs, "c")?;

    let d_pad = parent
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.y_zcol_values.len());
    for (idx, row) in parent.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(format!("y_ring_preflight_{idx}"));
        }
        if active_children.is_empty() {
            for (dst_row, target) in parent.y_ring[idx].iter().enumerate() {
                enforce_k_affine_sum_eq(cs, target, &[], &[], &format!("{label}_y_{idx}_{dst_row}"));
            }
        } else if constant_child_prefix == 0 {
            super::ring_action::enforce_rho_coeff_left_action_on_y_row_toom3_with_vars(
                cs,
                &parent.y_ring[idx],
                &active_children
                    .iter()
                    .map(|child| child.y_ring[idx].clone())
                    .collect::<Vec<_>>(),
                &active_children
                    .iter()
                    .map(|child| child.y_ring_values[idx].clone())
                    .collect::<Vec<_>>(),
                &rhos[..active_children_len],
                &format!("{label}_y_{idx}"),
            )
            .map_err(|err| format!("y_ring_{idx}: {err}"))?;
        } else {
            enforce_y_row_rlc_target_with_vars(
                cs,
                &parent.y_ring[idx],
                active_children,
                rho_mats_active,
                constant_child_prefix,
                0,
                idx,
                d_pad,
                &format!("{label}_y_{idx}"),
            )
            .map_err(|err| format!("y_ring_{idx}: {err}"))?;
        }
        checkpoint(cs, &format!("y_ring_{idx}"))?;
    }

    Ok(())
}

pub(crate) fn debug_measure_rlc_public_with_split_rho_views_stage_ranges(
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
    rho_mats_active: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<RlcPublicStageCheckpoints, SynthesisError> {
    let active_children_len = children.len().saturating_sub(zero_commit_suffix_len);
    let mixed_constant_prefix = constant_child_prefix > 0 && constant_child_prefix < active_children_len;
    if children.is_empty()
        || children.len() != rhos.len()
        || (mixed_constant_prefix && rho_mats_active.len() != active_children_len)
        || (!mixed_constant_prefix && !rho_mats_active.is_empty())
        || constant_child_prefix > active_children_len
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let stage_start = cs.num_constraints();
    let mut checkpoints = RlcPublicStageCheckpoints { stage_ends: Vec::new() };

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        let zero_commit_suffix = idx >= active_children_len;
        let child_c_data_ok = if idx < constant_child_prefix || zero_commit_suffix {
            child.c_data.is_empty() || child.c_data.len() == parent.c_data.len()
        } else {
            child.c_data.len() == parent.c_data.len()
        };
        let child_y_ring_ok = if zero_commit_suffix {
            child.y_ring.is_empty() || child.y_ring.len() == parent.y_ring.len()
        } else {
            child.y_ring.len() == parent.y_ring.len()
        };
        let child_y_zcol_ok = if zero_commit_suffix {
            child.y_zcol.is_empty() || child.y_zcol.len() == parent.y_zcol.len()
        } else {
            child.y_zcol.len() == parent.y_zcol.len()
        };
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || !child_y_ring_ok
            || !child_y_zcol_ok
            || !child_c_data_ok
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        checkpoints.push(format!("shared_point_{idx}"), cs.num_constraints() - stage_start);
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_with_vars(
        cs,
        &parent.x,
        parent.x_cols,
        &children
            .iter()
            .map(|child| child.x.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        constant_child_prefix,
        zero_commit_suffix_len,
        &format!("{label}_x"),
    )?;
    checkpoints.push("x".into(), cs.num_constraints() - stage_start);

    ensure_zero_commit_suffix(children, zero_commit_suffix_len)?;
    let active_children = &children[..active_children_len];
    if active_children.is_empty() {
        for (idx, entry) in parent.c_data.iter().enumerate() {
            enforce_field_affine_sum_eq(cs, entry, &[], &[], &format!("{label}_c_eq_{idx}"));
        }
    } else if constant_child_prefix == 0 {
        super::ring_action::enforce_rho_coeff_left_action_on_dense_commitment_columns_toom3_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
            &active_children
                .iter()
                .map(|child| child.c_data.clone())
                .collect::<Vec<_>>(),
            &active_children
                .iter()
                .map(|child| child.c_data_values.clone())
                .collect::<Vec<_>>(),
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else if constant_child_prefix == active_children_len {
        enforce_rho_coeff_left_action_on_dense_constant_f_slices(
            cs,
            &parent.c_data,
            &parent.c_data_values,
            parent.c_data.len() / D,
            &active_children
                .iter()
                .map(|child| child.c_data_values.clone())
                .collect::<Vec<_>>(),
            true,
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else {
        enforce_rho_left_action_on_dense_f_slices_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
            &active_children
                .iter()
                .map(|child| child.c_data.clone())
                .collect::<Vec<_>>(),
            &active_children
                .iter()
                .map(|child| child.c_data_values.clone())
                .collect::<Vec<_>>(),
            true,
            rho_mats_active,
            constant_child_prefix,
            0,
            &format!("{label}_c"),
        )?;
    }
    checkpoints.push("c".into(), cs.num_constraints() - stage_start);

    let d_pad = parent
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.y_zcol_values.len());
    for (idx, row) in parent.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        if active_children.is_empty() {
            for (dst_row, target) in parent.y_ring[idx].iter().enumerate() {
                enforce_k_affine_sum_eq(cs, target, &[], &[], &format!("{label}_y_{idx}_{dst_row}"));
            }
        } else if constant_child_prefix == 0 {
            super::ring_action::enforce_rho_coeff_left_action_on_y_row_toom3_with_vars(
                cs,
                &parent.y_ring[idx],
                &active_children
                    .iter()
                    .map(|child| child.y_ring[idx].clone())
                    .collect::<Vec<_>>(),
                &active_children
                    .iter()
                    .map(|child| child.y_ring_values[idx].clone())
                    .collect::<Vec<_>>(),
                &rhos[..active_children_len],
                &format!("{label}_y_{idx}"),
            )?;
        } else if constant_child_prefix == active_children_len {
            enforce_y_row_rlc_target_with_rho_coeffs(
                cs,
                &parent.y_ring[idx],
                active_children,
                &rhos[..active_children_len],
                idx,
                d_pad,
                &format!("{label}_y_{idx}"),
            )?;
        } else {
            enforce_y_row_rlc_target_with_vars(
                cs,
                &parent.y_ring[idx],
                active_children,
                rho_mats_active,
                constant_child_prefix,
                0,
                idx,
                d_pad,
                &format!("{label}_y_{idx}"),
            )?;
        }
        checkpoints.push(format!("y_ring_{idx}"), cs.num_constraints() - stage_start);
    }

    Ok(checkpoints)
}

pub(crate) fn debug_measure_rlc_public_with_rho_coeffs_for_constant_children_stage_ranges(
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<RlcPublicStageCheckpoints, SynthesisError> {
    if children.is_empty()
        || children.len() != rhos.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let stage_start = cs.num_constraints();
    let mut checkpoints = RlcPublicStageCheckpoints { stage_ends: Vec::new() };

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.x_values.len() != D * parent.m_in
            || child.c_data_values.len() != parent.c_data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        checkpoints.push(format!("shared_point_{idx}"), cs.num_constraints() - stage_start);
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children(
        cs,
        &parent.x,
        &parent.x_values,
        parent.x_cols,
        &children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        &format!("{label}_x"),
    )?;
    checkpoints.push("x".into(), cs.num_constraints() - stage_start);

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.c_data,
        &parent.c_data_values,
        parent.c_data_values.len() / D,
        &children
            .iter()
            .map(|child| child.c_data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rhos,
        &format!("{label}_c"),
    )?;
    checkpoints.push("c".into(), cs.num_constraints() - stage_start);

    let d_pad = parent
        .y_ring_values
        .first()
        .map(|row| row.len())
        .unwrap_or(0)
        .max(parent.y_zcol_values.len());
    for (idx, row) in parent.y_ring_values.iter().enumerate() {
        if row.len() != d_pad {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_y_row_rlc_target_with_rho_coeffs(
            cs,
            &parent.y_ring[idx],
            children,
            rhos,
            idx,
            d_pad,
            &format!("{label}_y_{idx}"),
        )?;
        checkpoints.push(format!("y_ring_{idx}"), cs.num_constraints() - stage_start);
    }

    Ok(checkpoints)
}
