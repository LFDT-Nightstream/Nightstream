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

pub(crate) fn debug_locate_rlc_public_with_rho_vars_constant_prefix_stage(
    cs: &mut TestConstraintSystem<SpartanF>,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<(), String> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || constant_child_prefix > children.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err("preflight".into());
    }

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        let child_c_data_ok = if idx < constant_child_prefix {
            child.c_data.is_empty() || child.c_data.len() == parent.c_data.len()
        } else {
            child.c_data.len() == parent.c_data.len()
        };
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || !child_c_data_ok
            || rho.entry_value(0, 0).is_err()
        {
            return Err(format!("preflight_child_{idx}"));
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))
            .map_err(|err| format!("r_{idx}: {err}"))?;
        checkpoint(cs, &format!("shared_point_{idx}"))?;
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
    )
    .map_err(|err| format!("x: {err}"))?;
    checkpoint(cs, "x")?;

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
    )
    .map_err(|err| format!("c: {err}"))?;
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
        )
        .map_err(|err| format!("y_ring_{idx}: {err}"))?;
        checkpoint(cs, &format!("y_ring_{idx}"))?;
    }

    Ok(())
}

pub(crate) fn debug_measure_rlc_public_with_rho_vars_constant_prefix_stage_ranges(
    cs: &mut ShapeCS<Rv64imDeciderEngine>,
    parent: &CeClaimVar,
    children: &[CeClaimVar],
    rho_mats: &[RotRhoMatrixVar],
    constant_child_prefix: usize,
    zero_commit_suffix_len: usize,
    label: &str,
) -> Result<RlcPublicStageCheckpoints, SynthesisError> {
    if children.is_empty()
        || children.len() != rho_mats.len()
        || constant_child_prefix > children.len()
        || parent.x_rows != D
        || parent.x_cols != parent.m_in
        || parent.r.len() != parent.r_values.len()
        || parent.y_zcol.len() != parent.y_zcol_values.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    let stage_start = cs.num_constraints();
    let mut checkpoints = RlcPublicStageCheckpoints { stage_ends: Vec::new() };

    for (idx, (child, rho)) in children.iter().zip(rho_mats.iter()).enumerate() {
        let child_c_data_ok = if idx < constant_child_prefix {
            child.c_data.is_empty() || child.c_data.len() == parent.c_data.len()
        } else {
            child.c_data.len() == parent.c_data.len()
        };
        if child.m_in != parent.m_in
            || child.x_rows != D
            || child.x_cols != parent.m_in
            || child.r_values != parent.r_values
            || child.y_ring.len() != parent.y_ring.len()
            || child.y_zcol.len() != parent.y_zcol.len()
            || !child_c_data_ok
            || rho.entry_value(0, 0).is_err()
        {
            return Err(SynthesisError::Unsatisfiable);
        }
        enforce_equal_k_slice(cs, &parent.r, &child.r, &format!("{label}_r_{idx}"))?;
        checkpoints.push(format!("shared_point_{idx}"), cs.num_constraints() - stage_start);
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
    checkpoints.push("x".into(), cs.num_constraints() - stage_start);

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
