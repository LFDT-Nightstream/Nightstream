//! Owns public Π_RLC arithmetic checks over CE claims for the RV64IM main-relation circuit.
//!
//! This module owns rho-driven claim folding and the last-chunk shortcut that still lives on
//! the Π_RLC side of the bridge theorem boundary. Pure b-ary Π_DEC checks live in `pi_dec.rs`.

#[path = "rlc_dec/diagnostics.rs"]
mod diagnostics;
#[path = "pi_rlc/ring_action.rs"]
mod ring_action;

use crate::rv64im::ivc_snark::SpartanF;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::sync::LazyLock;

use super::claim::CeClaimVar;
use super::k_field::{enforce_k_eq, enforce_k_eq_constant_f_linear_combination, k_base_mul_var, KNum, KNumVar};
use super::rho_sampling::{RotRhoMatrixVar, RotRhoVar};

static GOLDILOCKS_ROT_BASIS_MATS: LazyLock<Vec<Mat<F>>> = LazyLock::new(build_goldilocks_rot_basis_mats);

pub(crate) use diagnostics::{
    debug_locate_rlc_public_with_split_rho_views_stage,
    debug_measure_rlc_public_with_rho_coeffs_for_constant_children_stage_ranges,
    debug_measure_rlc_public_with_split_rho_views_stage_ranges, RlcPublicStageCheckpoints,
};

pub fn enforce_rlc_public<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || rho.rows() != D
            || rho.cols() != D
            || child.r_values != parent.r_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.y_zcol.len() != parent.y_zcol.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
    }

    enforce_rho_left_action_on_dense_f_slices(
        cs,
        &parent.x,
        parent.x_cols,
        &children
            .iter()
            .map(|child| child.x.clone())
            .collect::<Vec<_>>(),
        false,
        rho_mats,
        &format!("{label}_x"),
    )?;

    let commitment_cols = parent.c_data.len() / D;
    if commitment_cols * D != parent.c_data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.c_data.len() != parent.c_data.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    enforce_rho_left_action_on_dense_f_slices(
        cs,
        &parent.c_data,
        commitment_cols,
        &children
            .iter()
            .map(|child| child.c_data.clone())
            .collect::<Vec<_>>(),
        true,
        rho_mats,
        &format!("{label}_c"),
    )?;

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
        enforce_y_row_rlc_target(
            cs,
            &parent.y_ring[idx],
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
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars_constant_prefix_zero_commit_suffix(cs, parent, children, rho_mats, 0, 0, label)
}

pub fn enforce_rlc_public_with_rho_vars_constant_prefix<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
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
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || constant_child_prefix > children.len()
        || zero_commit_suffix_len > children.len().saturating_sub(constant_child_prefix)
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        let zero_commit_suffix = idx >= children.len().saturating_sub(zero_commit_suffix_len);
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
            || rho.entry_value(0, 0).is_err()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
    }

    enforce_rho_left_action_on_canonical_embedded_x_with_vars(
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
        rho_mats,
        constant_child_prefix,
        &format!("{label}_x"),
    )?;

    enforce_rho_left_action_on_dense_f_slices_with_vars(
        cs,
        &parent.c_data,
        parent.c_data.len() / D,
        &children
            .iter()
            .map(|child| child.c_data.clone())
            .collect::<Vec<_>>(),
        &children
            .iter()
            .map(|child| child.c_data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rho_mats,
        constant_child_prefix,
        zero_commit_suffix_len,
        &format!("{label}_c"),
    )?;

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
        enforce_y_row_rlc_target_with_vars(
            cs,
            &parent.y_ring[idx],
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
    parent: &CeClaimVar,
    children: &[CeClaimVar],
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
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
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
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
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
        0,
        &format!("{label}_x"),
    )?;

    ensure_zero_commit_suffix(children, zero_commit_suffix_len)?;

    let active_children = &children[..active_children_len];
    let active_c_vars = active_children
        .iter()
        .map(|child| child.c_data.clone())
        .collect::<Vec<_>>();
    let active_c_values = active_children
        .iter()
        .map(|child| child.c_data_values.clone())
        .collect::<Vec<_>>();
    if active_children.is_empty() {
        for (idx, entry) in parent.c_data.iter().enumerate() {
            enforce_field_affine_sum_eq(cs, entry, &[], &[], &format!("{label}_c_eq_{idx}"));
        }
    } else if constant_child_prefix == 0 {
        ring_action::enforce_rho_coeff_left_action_on_dense_commitment_columns_toom3_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
            &active_c_vars,
            &active_c_values,
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else if constant_child_prefix == active_children_len {
        enforce_rho_coeff_left_action_on_dense_constant_f_slices(
            cs,
            &parent.c_data,
            &parent.c_data_values,
            parent.c_data.len() / D,
            &active_c_values,
            true,
            &rhos[..active_children_len],
            &format!("{label}_c"),
        )?;
    } else {
        enforce_rho_left_action_on_dense_f_slices_with_vars(
            cs,
            &parent.c_data,
            parent.c_data.len() / D,
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
            ring_action::enforce_rho_coeff_left_action_on_y_row_toom3_with_vars(
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
    }

    Ok(())
}

pub fn enforce_rlc_public_with_rho_coeffs_for_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rhos.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

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
    }

    Ok(())
}

pub fn enforce_rlc_dec_public_with_rho_coeffs_for_last_chunk<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    rlc_children: &[CeClaimVar],
    dec_children: &[CeClaim<Commitment, F, K>],
    rhos: &[RotRhoVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if rlc_children.is_empty()
        || rlc_children.len() != rhos.len()
        || dec_children.is_empty()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in rlc_children.iter().zip(rhos.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.y_ring_values.len() != parent.y_ring_values.len()
            || child.x_values.len() != D * parent.m_in
            || child.c_data_values.len() != parent.c_data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
    }
    for child in dec_children {
        if child.m_in != parent.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.m_in
            || child.r != parent.r_values
            || child.y_ring.len() != parent.y_ring_values.len()
            || child.y_zcol.len() != parent.y_zcol_values.len()
            || child.c.data.len() != parent.c_data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children(
        cs,
        &parent.x,
        &parent.x_values,
        parent.x_cols,
        &rlc_children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        rhos,
        &format!("{label}_x"),
    )?;

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.c_data,
        &parent.c_data_values,
        parent.c_data_values.len() / D,
        &rlc_children
            .iter()
            .map(|child| child.c_data_values.clone())
            .collect::<Vec<_>>(),
        true,
        rhos,
        &format!("{label}_c"),
    )?;

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
        enforce_y_row_rlc_eq_dec_target_with_rho_coeffs(
            cs,
            &parent.y_ring[idx],
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

fn enforce_rho_left_action_on_dense_f_slices<CS: ConstraintSystem<SpartanF>>(
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

fn enforce_rho_left_action_on_dense_f_slices_with_vars<CS: ConstraintSystem<SpartanF>>(
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

fn enforce_rho_left_action_on_canonical_embedded_x_with_vars<CS: ConstraintSystem<SpartanF>>(
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
                    let active_lane = col % D;
                    let coeff = rho.entry(row, active_lane)?;
                    let child_value = native_child[col];
                    if child_idx < constant_child_prefix {
                        linear_terms.push((SpartanF::from_canonical_u64(child_value.as_canonical_u64()), coeff));
                        continue;
                    }
                    let child_var = child.get(col).ok_or(SynthesisError::Unsatisfiable)?;
                    let product = coeff.mul(
                        cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}")),
                        child_var,
                    )?;
                    products.push(product);
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

fn enforce_rho_coeff_left_action_on_canonical_embedded_x_with_vars<CS: ConstraintSystem<SpartanF>>(
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
                    let active_lane = col % D;
                    let coeff = alloc_rot_rho_entry_from_coeffs(
                        cs.namespace(|| format!("{label}_coeff_{row}_{col}_{child_idx}")),
                        rho,
                        row,
                        active_lane,
                        &format!("{label}_coeff_{row}_{col}_{child_idx}"),
                    )?;
                    let child_value = native_child[col];
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
                    let child_var = child.get(col).ok_or(SynthesisError::Unsatisfiable)?;
                    let product = coeff.mul(
                        cs.namespace(|| format!("{label}_mul_{row}_{col}_{child_idx}")),
                        child_var,
                    )?;
                    products.push(product);
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

fn enforce_rho_coeff_left_action_on_dense_constant_f_slices<CS: ConstraintSystem<SpartanF>>(
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

fn enforce_rho_coeff_left_action_on_canonical_embedded_x_constant_children<CS: ConstraintSystem<SpartanF>>(
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
                    let active_lane = col % D;
                    let value = native_child[col];
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

fn dense_index(row: usize, col: usize, cols: usize, column_major: bool) -> usize {
    if column_major {
        col * D + row
    } else {
        row * cols + col
    }
}

pub fn enforce_rlc_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public(cs, parent, children, rho_mats, label)
}

pub fn enforce_rlc_public_non_commitment_with_rho_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars(cs, parent, children, rho_mats, label)
}

fn enforce_y_row_rlc_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
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
                        child.y_ring[row_idx][src_row].c0,
                        child.y_ring[row_idx][src_row].c1,
                    ));
                }
            }
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{dst_row}"));
    }
    Ok(())
}

fn enforce_y_row_rlc_target_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
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
                    let value = child.y_ring_values[row_idx][src_row];
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
                        &child.y_ring[row_idx][src_row],
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

fn enforce_y_row_rlc_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
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
                    let (coeff_c0, coeff_c1) = basis_k_row_scale(dst_row, &child.y_ring_values[row_idx], coeff_idx);
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

fn enforce_y_row_rlc_eq_dec_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    rlc_children: &[CeClaimVar],
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
                    let (coeff_c0, coeff_c1) = basis_k_row_scale(dst_row, &child.y_ring_values[row_idx], coeff_idx);
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

fn enforce_equal_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    left: &[KNumVar],
    right: &[KNumVar],
    label: &str,
) -> Result<(), SynthesisError> {
    if left.len() != right.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        enforce_k_eq(cs, l, r, &format!("{label}_{idx}"));
    }
    Ok(())
}

fn enforce_k_eq_native<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, actual: &KNumVar, expected: K, label: &str) {
    let coeffs = expected.as_coeffs();
    cs.enforce(
        || format!("{label}_c0_eq"),
        |lc| lc + actual.c0,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()), CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_eq"),
        |lc| lc + actual.c1,
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()), CS::one()),
    );
}

fn enforce_field_affine_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &AllocatedNum<SpartanF>,
    linear_terms: &[(SpartanF, AllocatedNum<SpartanF>)],
    product_terms: &[AllocatedNum<SpartanF>],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_sum"),
        |lc| {
            let mut acc = lc;
            for (scale, term) in linear_terms {
                acc = acc + (*scale, term.get_variable());
            }
            for term in product_terms {
                acc = acc + term.get_variable();
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.get_variable(),
    );
}

fn enforce_k_affine_sum_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    linear_terms: &[(SpartanF, SpartanF, bellpepper_core::Variable)],
    product_terms: &[KNumVar],
    label: &str,
) {
    cs.enforce(
        || format!("{label}_c0_sum"),
        |lc| {
            let mut acc = lc;
            for (coeff_c0, _, variable) in linear_terms {
                acc = acc + (*coeff_c0, *variable);
            }
            for term in product_terms {
                acc = acc + term.c0;
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c0,
    );
    cs.enforce(
        || format!("{label}_c1_sum"),
        |lc| {
            let mut acc = lc;
            for (_, coeff_c1, variable) in linear_terms {
                acc = acc + (*coeff_c1, *variable);
            }
            for term in product_terms {
                acc = acc + term.c1;
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + target.c1,
    );
}

fn scale_k_by_f_var<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    scalar: &AllocatedNum<SpartanF>,
    scalar_value: F,
    value: &KNumVar,
    value_native: K,
    label: &str,
) -> Result<KNumVar, SynthesisError> {
    let term_value = K::from(scalar_value) * value_native;
    k_base_mul_var(
        &mut cs,
        value,
        scalar.get_variable(),
        KNum::from_neo_k(value_native),
        SpartanF::from_canonical_u64(scalar_value.as_canonical_u64()),
        KNum::from_neo_k(term_value),
        label,
    )
}

fn ensure_zero_commit_suffix(children: &[CeClaimVar], zero_commit_suffix_len: usize) -> Result<(), SynthesisError> {
    if zero_commit_suffix_len == 0 {
        return Ok(());
    }
    let zero_commit_suffix_start = children.len().saturating_sub(zero_commit_suffix_len);
    for child in &children[zero_commit_suffix_start..] {
        if child.c_data_values.iter().any(|value| *value != F::ZERO) {
            return Err(SynthesisError::Unsatisfiable);
        }
        if child
            .y_ring_values
            .iter()
            .any(|row| row.iter().any(|value| *value != K::ZERO))
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    Ok(())
}

fn alloc_rot_rho_entry_from_coeffs<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    rho: &RotRhoVar,
    row: usize,
    col: usize,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    if row >= D || col >= D || rho.coeffs.len() != D || rho.coeff_values.len() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    if col == 0 {
        return Ok(rho.coeffs[row].clone());
    }
    let mut value = F::ZERO;
    let mut terms = Vec::new();
    for coeff_idx in 0..D {
        let basis_coeff = GOLDILOCKS_ROT_BASIS_MATS[coeff_idx][(row, col)];
        if basis_coeff == F::ZERO {
            continue;
        }
        value += basis_coeff * rho.coeff_values[coeff_idx];
        terms.push((
            rho.coeffs[coeff_idx].clone(),
            SpartanF::from_canonical_u64(basis_coeff.as_canonical_u64()),
            SpartanF::from_canonical_u64(rho.coeff_values[coeff_idx].as_canonical_u64()),
        ));
    }
    alloc_affine_field_terms(
        cs.namespace(|| format!("{label}_affine")),
        &terms,
        SpartanF::from_canonical_u64(value.as_canonical_u64()),
    )
}

fn alloc_affine_field_terms<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    terms: &[(AllocatedNum<SpartanF>, SpartanF, SpartanF)],
    value: SpartanF,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(value))?;
    cs.enforce(
        || "affine",
        |lc| lc + CS::one(),
        |lc| lc + out.get_variable(),
        |lc| {
            let mut rhs = lc;
            for (term, coeff, _) in terms {
                rhs = rhs + (*coeff, term.get_variable());
            }
            rhs
        },
    );
    Ok(out)
}

fn build_goldilocks_rot_basis_mats() -> Vec<Mat<F>> {
    let neg_phi = neo_reductions::RotRing::goldilocks()
        .phi_coeffs
        .iter()
        .map(|coeff| {
            if *coeff >= 0 {
                F::ZERO - F::from_u64(*coeff as u64)
            } else {
                F::from_u64((-*coeff) as u64)
            }
        })
        .collect::<Vec<_>>();
    let mut mats = Vec::with_capacity(D);
    for coeff_idx in 0..D {
        let mut col = vec![F::ZERO; D];
        col[coeff_idx] = F::ONE;
        let mut mat = Mat::zero(D, D, F::ZERO);
        for j in 0..D {
            for row in 0..D {
                mat[(row, j)] = col[row];
            }
            let tail = col[D - 1];
            let mut next = vec![F::ZERO; D];
            next[0] = tail * neg_phi[0];
            for row in 1..D {
                next[row] = col[row - 1] + tail * neg_phi[row];
            }
            col = next;
        }
        mats.push(mat);
    }
    mats
}

fn basis_dense_f_scale(row: usize, col: usize, cols: usize, column_major: bool, child: &[F], coeff_idx: usize) -> F {
    let basis = &GOLDILOCKS_ROT_BASIS_MATS[coeff_idx];
    let mut acc = F::ZERO;
    for src_row in 0..D {
        let basis_coeff = basis[(row, src_row)];
        if basis_coeff == F::ZERO {
            continue;
        }
        let child_idx = dense_index(src_row, col, cols, column_major);
        let value = child[child_idx];
        acc += basis_coeff * value;
    }
    acc
}

fn basis_k_row_scale(row: usize, child: &[K], coeff_idx: usize) -> (F, F) {
    let basis = &GOLDILOCKS_ROT_BASIS_MATS[coeff_idx];
    let mut acc_c0 = F::ZERO;
    let mut acc_c1 = F::ZERO;
    for src_row in 0..D {
        let basis_coeff = basis[(row, src_row)];
        if basis_coeff == F::ZERO {
            continue;
        }
        let coeffs = child[src_row].as_coeffs();
        acc_c0 += basis_coeff * coeffs[0];
        acc_c1 += basis_coeff * coeffs[1];
    }
    (acc_c0, acc_c1)
}
