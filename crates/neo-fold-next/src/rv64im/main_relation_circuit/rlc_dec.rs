//! Owns public Π_RLC / Π_DEC arithmetic checks over CE claims for the RV64IM main-relation circuit.
//!
//! These checks follow the native bridge theorem boundary: public RLC/DEC equalities cover
//! `c`, `X`, `r`, `s_col`, `y_ring`, `ct`, `aux_openings`, and `y_zcol`, while direct CE
//! consistency is enforced only for authoritative Π_CCS outputs.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{KExtensions, D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;
use std::sync::LazyLock;

use super::claim::CeClaimVar;
use super::k_field::{enforce_k_eq, enforce_k_eq_constant_f_linear_combination, k_base_mul_var, KNum, KNumVar};
use super::rho_sampling::{RotRhoMatrixVar, RotRhoVar};

static GOLDILOCKS_ROT_BASIS_MATS: LazyLock<Vec<Mat<F>>> = LazyLock::new(build_goldilocks_rot_basis_mats);

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
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
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
            || child.s_col_values != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
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
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    if !parent.y_zcol.is_empty() {
        enforce_y_zcol_rlc_target(
            cs,
            &parent.y_zcol,
            children,
            rho_mats,
            d_pad,
            &format!("{label}_y_zcol"),
        )?;
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_rlc_target(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            rho_mats,
            aux_idx,
            &format!("{label}_aux_{aux_idx}"),
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
    enforce_rlc_public_with_rho_vars_constant_prefix(cs, parent, children, rho_mats, 0, label)
}

pub fn enforce_rlc_public_with_rho_vars_constant_prefix<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || constant_child_prefix > children.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.s_col_values != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.c_data.len() != parent.c_data.len()
            || rho.entry_value(0, 0).is_err()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
    }

    enforce_rho_left_action_on_dense_f_slices_with_vars(
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
        false,
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
            idx,
            d_pad,
            &format!("{label}_y_{idx}"),
        )?;
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    if !parent.y_zcol.is_empty() {
        enforce_y_zcol_rlc_target_with_vars(
            cs,
            &parent.y_zcol,
            children,
            rho_mats,
            constant_child_prefix,
            d_pad,
            &format!("{label}_y_zcol"),
        )?;
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_rlc_target_with_vars(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            rho_mats,
            constant_child_prefix,
            aux_idx,
            &format!("{label}_aux_eq_{aux_idx}"),
        )?;
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
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in children.iter().zip(rhos.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.s_col_values != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.x_values.len() != D * parent.m_in
            || child.c_data_values.len() != parent.c_data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
    }

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.x,
        &parent.x_values,
        parent.x_cols,
        &children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        false,
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
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    if !parent.y_zcol.is_empty() {
        enforce_y_zcol_rlc_target_with_rho_coeffs(
            cs,
            &parent.y_zcol,
            children,
            rhos,
            d_pad,
            &format!("{label}_y_zcol"),
        )?;
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_rlc_target_with_rho_coeffs(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            rhos,
            aux_idx,
            &format!("{label}_aux_{aux_idx}"),
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
        || parent.s_col.len() != parent.s_col_values.len()
        || !parent.y_ring.is_empty()
        || !parent.ct.is_empty()
        || !parent.aux_openings.is_empty()
        || !parent.y_zcol.is_empty()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, (child, rho)) in rlc_children.iter().zip(rhos.iter()).enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.s_col_values != parent.s_col_values
            || child.y_ring_values.len() != parent.y_ring_values.len()
            || child.x_values.len() != D * parent.m_in
            || child.c_data_values.len() != parent.c_data_values.len()
            || rho.coeffs.len() != D
            || rho.coeff_values.len() != D
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
    }
    for child in dec_children {
        if child.m_in != parent.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.m_in
            || child.r != parent.r_values
            || child.s_col != parent.s_col_values
            || child.y_ring.len() != parent.y_ring_values.len()
            || child.y_zcol.len() != parent.y_zcol_values.len()
            || child.c.data.len() != parent.c_data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    enforce_rho_coeff_left_action_on_dense_constant_f_slices(
        cs,
        &parent.x,
        &parent.x_values,
        parent.x_cols,
        &rlc_children
            .iter()
            .map(|child| child.x_values.clone())
            .collect::<Vec<_>>(),
        false,
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

pub fn enforce_dec_public<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for (idx, child) in children.iter().enumerate() {
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.s_col_values != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.c_data.len() != parent.c_data.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        enforce_equal_k_slice(cs, &parent.s_col, &child.s_col, &format!("{label}_s_col_{idx}"))?;
    }

    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.x,
        &children
            .iter()
            .map(|child| child.x.clone())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )?;

    enforce_scalar_power_sum_on_dense_f_slices(
        cs,
        &parent.c_data,
        &children
            .iter()
            .map(|child| child.c_data.clone())
            .collect::<Vec<_>>(),
        base_b,
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
        enforce_y_row_dec_target(
            cs,
            &parent.y_ring[idx],
            children,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_dec_target(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            aux_idx,
            base_b,
            &format!("{label}_aux_{aux_idx}"),
        )?;
    }

    Ok(())
}

pub fn enforce_dec_public_with_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaim<Commitment, F, K>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.s_col.len() != parent.s_col_values.len()
        || parent.ct.len() != parent.ct_values.len()
        || parent.aux_openings.len() != parent.aux_openings_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    for child in children {
        if child.m_in != parent.m_in
            || child.X.rows() != D
            || child.X.cols() != parent.m_in
            || child.r != parent.r_values
            || child.s_col != parent.s_col_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.aux_openings.len() != parent.aux_openings.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || child.c.data.len() != parent.c_data_values.len()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
    }

    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.x,
        &parent.x_values,
        &children
            .iter()
            .map(|child| child.X.as_slice().to_vec())
            .collect::<Vec<_>>(),
        base_b,
        &format!("{label}_x"),
    )?;

    enforce_scalar_power_sum_on_dense_constant_f_slices(
        cs,
        &parent.c_data,
        &parent.c_data_values,
        &children
            .iter()
            .map(|child| child.c.data.clone())
            .collect::<Vec<_>>(),
        base_b,
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
        enforce_y_row_dec_target_constant_children(
            cs,
            &parent.y_ring[idx],
            children,
            idx,
            d_pad,
            base_b,
            &format!("{label}_y_{idx}"),
        )?;
        if idx < parent.ct.len() {
            enforce_k_eq(
                cs,
                &parent.y_ring[idx][0],
                &parent.ct[idx],
                &format!("{label}_ct_eq_{idx}"),
            );
        }
    }

    for aux_idx in 0..parent.aux_openings.len() {
        enforce_aux_dec_target_constant_children(
            cs,
            &parent.aux_openings[aux_idx],
            children,
            aux_idx,
            base_b,
            &format!("{label}_aux_{aux_idx}"),
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
                            if coeff != SpartanF::ZERO {
                                let child_idx = dense_index(k, col, cols, column_major);
                                acc = acc + (coeff, child[child_idx].get_variable());
                            }
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
    for ((child, native_child), rho) in children
        .iter()
        .zip(child_native_values.iter())
        .zip(rho_mats.iter())
    {
        if child.len() != D * cols || native_child.len() != D * cols || rho.entry_value(0, 0).is_err() {
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
                for k in 0..D {
                    let coeff = rho.entry(row, k)?;
                    let coeff_value = rho.entry_value(row, k)?;
                    let child_idx_flat = dense_index(k, col, cols, column_major);
                    let child_value = native_child[child_idx_flat];
                    if coeff_value == F::ZERO || child_value == F::ZERO {
                        continue;
                    }
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
                    if coeff == F::ZERO {
                        continue;
                    }
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

fn dense_index(row: usize, col: usize, cols: usize, column_major: bool) -> usize {
    if column_major {
        col * D + row
    } else {
        row * cols + col
    }
}

fn enforce_scalar_power_sum_on_dense_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    children: &[Vec<AllocatedNum<SpartanF>>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.len() != parent.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    let b = SpartanF::from_canonical_u64(base_b as u64);
    for idx in 0..parent.len() {
        cs.enforce(
            || format!("{label}_{idx}"),
            |lc| {
                let mut acc = lc;
                let mut pow = SpartanF::ONE;
                for child in children {
                    acc = acc + (pow, child[idx].get_variable());
                    pow *= b;
                }
                acc
            },
            |lc| lc + CS::one(),
            |lc| lc + parent[idx].get_variable(),
        );
    }
    Ok(())
}

fn enforce_scalar_power_sum_on_dense_constant_f_slices<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &[AllocatedNum<SpartanF>],
    parent_values: &[F],
    children: &[Vec<F>],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.is_empty() || parent_values.is_empty() {
        return Err(SynthesisError::Unsatisfiable);
    }
    if !parent.is_empty() && parent.len() != parent_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for child in children {
        if child.len() != parent_values.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
    }
    let b = F::from_u64(base_b as u64);
    for idx in 0..parent_values.len() {
        let mut pow = F::ONE;
        let mut expected = F::ZERO;
        for child in children {
            expected += pow * child[idx];
            pow *= b;
        }
        if parent.is_empty() {
            if parent_values[idx] != expected {
                return Err(SynthesisError::Unsatisfiable);
            }
        } else {
            cs.enforce(
                || format!("{label}_{idx}"),
                |lc| lc + parent[idx].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (SpartanF::from_canonical_u64(expected.as_canonical_u64()), CS::one()),
            );
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn enforce_rlc_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[Mat<F>],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public(cs, parent, children, rho_mats, label)
}

#[allow(dead_code)]
pub fn enforce_rlc_public_non_commitment_with_rho_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_rlc_public_with_rho_vars(cs, parent, children, rho_mats, label)
}

#[allow(dead_code)]
pub fn enforce_dec_public_non_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_dec_public(cs, parent, children, base_b, label)
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
                    if coeff == F::ZERO {
                        continue;
                    }
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

fn enforce_y_zcol_rlc_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    rho_mats: &[Mat<F>],
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
                    if coeff == F::ZERO {
                        continue;
                    }
                    terms.push((
                        SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
                        child.y_zcol[src_row].c0,
                        child.y_zcol[src_row].c1,
                    ));
                }
            }
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{dst_row}"));
    }
    Ok(())
}

fn enforce_aux_rlc_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaimVar],
    rho_mats: &[Mat<F>],
    aux_idx: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    let mut terms = Vec::new();
    for (child, rho) in children.iter().zip(rho_mats.iter()) {
        let coeff = rho[(0, 0)];
        if coeff == F::ZERO {
            continue;
        }
        terms.push((
            SpartanF::from_canonical_u64(coeff.as_canonical_u64()),
            child.aux_openings[aux_idx].c0,
            child.aux_openings[aux_idx].c1,
        ));
    }
    enforce_k_eq_constant_f_linear_combination(cs, target, &terms, label);
    Ok(())
}

fn enforce_y_row_rlc_target_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    row_idx: usize,
    d_pad: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (dst_row, target) in target.iter().enumerate() {
        let mut linear_terms = Vec::new();
        let mut terms = Vec::new();
        if dst_row < D {
            for (child_idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
                for src_row in 0..D {
                    let coeff_var = rho.entry(dst_row, src_row)?;
                    let coeff_value = rho.entry_value(dst_row, src_row)?;
                    let value = child.y_ring_values[row_idx][src_row];
                    if coeff_value == F::ZERO || value == K::ZERO {
                        continue;
                    }
                    if child_idx < constant_child_prefix {
                        let coeffs = value.as_coeffs();
                        linear_terms.push((
                            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                            coeff_var.get_variable(),
                        ));
                        continue;
                    }
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
                    if coeff_c0 == F::ZERO && coeff_c1 == F::ZERO {
                        continue;
                    }
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

fn enforce_y_zcol_rlc_target_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    d_pad: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (dst_row, target) in target.iter().enumerate() {
        let mut linear_terms = Vec::new();
        let mut terms = Vec::new();
        if dst_row < D {
            for (child_idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
                for src_row in 0..D {
                    let coeff_var = rho.entry(dst_row, src_row)?;
                    let coeff_value = rho.entry_value(dst_row, src_row)?;
                    let value = child.y_zcol_values[src_row];
                    if coeff_value == F::ZERO || value == K::ZERO {
                        continue;
                    }
                    if child_idx < constant_child_prefix {
                        let coeffs = value.as_coeffs();
                        linear_terms.push((
                            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                            coeff_var.get_variable(),
                        ));
                        continue;
                    }
                    let term = scale_k_by_f_var(
                        cs.namespace(|| format!("{label}_term_{child_idx}_{src_row}_{dst_row}")),
                        &coeff_var,
                        coeff_value,
                        &child.y_zcol[src_row],
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

fn enforce_y_zcol_rlc_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
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
                    let (coeff_c0, coeff_c1) = basis_k_row_scale(dst_row, &child.y_zcol_values, coeff_idx);
                    if coeff_c0 == F::ZERO && coeff_c1 == F::ZERO {
                        continue;
                    }
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

fn enforce_aux_rlc_target_with_vars<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    aux_idx: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    let mut linear_terms = Vec::new();
    let mut terms = Vec::new();
    for (child_idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        let coeff_var = rho.entry(0, 0)?;
        let coeff_value = rho.entry_value(0, 0)?;
        let value = child.aux_openings_values[aux_idx];
        if coeff_value == F::ZERO || value == K::ZERO {
            continue;
        }
        if child_idx < constant_child_prefix {
            let coeffs = value.as_coeffs();
            linear_terms.push((
                SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
                SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
                coeff_var.get_variable(),
            ));
            continue;
        }
        let term = scale_k_by_f_var(
            cs.namespace(|| format!("{label}_term_{child_idx}")),
            &coeff_var,
            coeff_value,
            &child.aux_openings[aux_idx],
            value,
            &format!("{label}_term_{child_idx}"),
        )?;
        terms.push(term);
    }
    enforce_k_affine_sum_eq(cs, target, &linear_terms, &terms, label);
    Ok(())
}

fn enforce_aux_rlc_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaimVar],
    rhos: &[RotRhoVar],
    aux_idx: usize,
    label: &str,
) -> Result<(), SynthesisError> {
    if children.len() != rhos.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut linear_terms = Vec::new();
    for (child_idx, child) in children.iter().enumerate() {
        let value = child.aux_openings_values[aux_idx];
        if value == K::ZERO {
            continue;
        }
        let coeffs = value.as_coeffs();
        linear_terms.push((
            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
            rhos[child_idx].coeffs[0].get_variable(),
        ));
    }
    enforce_k_affine_sum_eq(cs, target, &linear_terms, &[], label);
    Ok(())
}

fn enforce_y_row_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaimVar],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut scalars = Vec::with_capacity(children.len());
    for _ in children {
        let coeff = pow.as_coeffs();
        if coeff[1] != F::ZERO {
            return Err(SynthesisError::Unsatisfiable);
        }
        scalars.push(SpartanF::from_canonical_u64(coeff[0].as_canonical_u64()));
        pow *= b;
    }
    for (idx, target) in target.iter().enumerate() {
        let mut terms = Vec::new();
        for (child, coeff) in children.iter().zip(scalars.iter()) {
            if *coeff == SpartanF::ZERO {
                continue;
            }
            terms.push((*coeff, child.y_ring[row_idx][idx].c0, child.y_ring[row_idx][idx].c1));
        }
        enforce_k_eq_constant_f_linear_combination(cs, target, &terms, &format!("{label}_{idx}"));
    }
    Ok(())
}

fn enforce_aux_dec_target<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaimVar],
    aux_idx: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut terms = Vec::with_capacity(children.len());
    for child in children {
        let coeff = pow.as_coeffs();
        if coeff[1] != F::ZERO {
            return Err(SynthesisError::Unsatisfiable);
        }
        let coeff = SpartanF::from_canonical_u64(coeff[0].as_canonical_u64());
        if coeff != SpartanF::ZERO {
            terms.push((coeff, child.aux_openings[aux_idx].c0, child.aux_openings[aux_idx].c1));
        }
        pow *= b;
    }
    enforce_k_eq_constant_f_linear_combination(cs, target, &terms, label);
    Ok(())
}

fn enforce_y_row_dec_target_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &[KNumVar],
    children: &[CeClaim<Commitment, F, K>],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if target.len() != d_pad {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    for (idx, target) in target.iter().enumerate() {
        let mut pow = K::ONE;
        let mut expected = K::ZERO;
        for child in children {
            expected += pow * child.y_ring[row_idx][idx];
            pow *= b;
        }
        enforce_k_eq_native(cs, target, expected, &format!("{label}_{idx}"));
    }
    Ok(())
}

fn enforce_y_row_rlc_eq_dec_target_with_rho_coeffs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    rlc_children: &[CeClaimVar],
    dec_children: &[CeClaim<Commitment, F, K>],
    rhos: &[RotRhoVar],
    row_idx: usize,
    d_pad: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    if rlc_children.len() != rhos.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let b = K::from(F::from_u64(base_b as u64));
    for dst_row in 0..d_pad {
        if dst_row >= D {
            continue;
        }
        let mut linear_terms = Vec::new();
        for (child_idx, child) in rlc_children.iter().enumerate() {
            for coeff_idx in 0..D {
                let (coeff_c0, coeff_c1) = basis_k_row_scale(dst_row, &child.y_ring_values[row_idx], coeff_idx);
                if coeff_c0 == F::ZERO && coeff_c1 == F::ZERO {
                    continue;
                }
                linear_terms.push((
                    SpartanF::from_canonical_u64(coeff_c0.as_canonical_u64()),
                    SpartanF::from_canonical_u64(coeff_c1.as_canonical_u64()),
                    rhos[child_idx].coeffs[coeff_idx].get_variable(),
                ));
            }
        }
        let mut pow = K::ONE;
        let mut expected = K::ZERO;
        for child in dec_children {
            expected += pow * child.y_ring[row_idx][dst_row];
            pow *= b;
        }
        enforce_k_affine_sum_eq_constant(cs, &linear_terms, &[], expected, &format!("{label}_{dst_row}"));
    }
    Ok(())
}

fn enforce_aux_dec_target_constant_children<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    target: &KNumVar,
    children: &[CeClaim<Commitment, F, K>],
    aux_idx: usize,
    base_b: u32,
    label: &str,
) -> Result<(), SynthesisError> {
    let b = K::from(F::from_u64(base_b as u64));
    let mut pow = K::ONE;
    let mut expected = K::ZERO;
    for child in children {
        expected += pow * child.aux_openings[aux_idx];
        pow *= b;
    }
    enforce_k_eq_native(cs, target, expected, label);
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
                if *coeff_c0 != SpartanF::ZERO {
                    acc = acc + (*coeff_c0, *variable);
                }
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
                if *coeff_c1 != SpartanF::ZERO {
                    acc = acc + (*coeff_c1, *variable);
                }
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

fn enforce_k_affine_sum_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    linear_terms: &[(SpartanF, SpartanF, bellpepper_core::Variable)],
    product_terms: &[KNumVar],
    expected: K,
    label: &str,
) {
    let coeffs = expected.as_coeffs();
    cs.enforce(
        || format!("{label}_c0_sum"),
        |lc| {
            let mut acc = lc;
            for (coeff_c0, _, variable) in linear_terms {
                if *coeff_c0 != SpartanF::ZERO {
                    acc = acc + (*coeff_c0, *variable);
                }
            }
            for term in product_terms {
                acc = acc + term.c0;
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()), CS::one()),
    );
    cs.enforce(
        || format!("{label}_c1_sum"),
        |lc| {
            let mut acc = lc;
            for (_, coeff_c1, variable) in linear_terms {
                if *coeff_c1 != SpartanF::ZERO {
                    acc = acc + (*coeff_c1, *variable);
                }
            }
            for term in product_terms {
                acc = acc + term.c1;
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc + (SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()), CS::one()),
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
        if value == F::ZERO {
            continue;
        }
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
        if coeffs[0] != F::ZERO {
            acc_c0 += basis_coeff * coeffs[0];
        }
        if coeffs[1] != F::ZERO {
            acc_c1 += basis_coeff * coeffs[1];
        }
    }
    (acc_c0, acc_c1)
}
