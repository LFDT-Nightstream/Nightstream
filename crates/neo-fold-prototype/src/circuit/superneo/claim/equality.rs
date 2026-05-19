use super::alloc::compact_x_values_from_native_claim;
use super::fields::{
    enforce_f_slice_eq, enforce_f_slice_eq_native, enforce_k_slice_eq, enforce_k_slice_eq_native,
    enforce_packed_bytes_eq_native,
};
use super::*;

pub fn enforce_claim_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "c_data"),
        &actual.commitment.data,
        &expected.c.data,
        &format!("{label}_c_data"),
    )?;
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "x"),
        &actual.public_input.x,
        expected.X.as_slice(),
        &format!("{label}_x"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "r"),
        &actual.openings.r,
        &expected.r,
        &format!("{label}_r"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "s_col"),
        &actual.norm_check.s_col,
        &expected.s_col,
        &format!("{label}_s_col"),
    )?;
    if actual.openings.y_ring.len() != expected.y_ring.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (row_idx, (actual_row, expected_row)) in actual
        .openings
        .y_ring
        .iter()
        .zip(expected.y_ring.iter())
        .enumerate()
    {
        enforce_k_slice_eq_native(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "ct"),
        &actual.openings.ct,
        &expected.ct,
        &format!("{label}_ct"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "aux_openings"),
        &actual.openings.aux_openings,
        &expected.aux_openings,
        &format!("{label}_aux_openings"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "y_zcol"),
        &actual.norm_check.y_zcol,
        &expected.y_zcol,
        &format!("{label}_y_zcol"),
    )?;
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "c_step_coords"),
        &actual.step_binding.c_step_coords,
        &expected.c_step_coords,
        &format!("{label}_c_step_coords"),
    )?;
    enforce_packed_bytes_eq_native(
        &mut cs.namespace(|| "fold_digest"),
        &actual.step_binding.fold_digest_encoding,
        &expected.fold_digest,
        &format!("{label}_fold_digest"),
    )?;
    if actual.public_input.m_in != expected.m_in
        || actual.step_binding.u_offset != expected.u_offset
        || actual.step_binding.u_len != expected.u_len
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub fn enforce_claim_projection_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CircuitCeClaim,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.commitment.data.len() != expected.commitment.data.len()
        || actual.public_input.m_in != expected.public_input.m_in
        || actual.openings.r.len() != expected.openings.r.len()
        || actual.openings.y_ring.len() != expected.openings.y_ring.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    enforce_f_slice_eq(
        &mut cs.namespace(|| "c_data"),
        &actual.commitment.data,
        &expected.commitment.data,
        &format!("{label}_c_data"),
    )?;
    enforce_claim_x_projection_eq(&mut cs.namespace(|| "x"), actual, expected, &format!("{label}_x"))?;
    enforce_k_slice_eq(
        &mut cs.namespace(|| "r"),
        &actual.openings.r,
        &expected.openings.r,
        &format!("{label}_r"),
    )?;
    for (row_idx, (actual_row, expected_row)) in actual
        .openings
        .y_ring
        .iter()
        .zip(expected.openings.y_ring.iter())
        .enumerate()
    {
        enforce_k_slice_eq(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    Ok(())
}

pub fn enforce_claim_projection_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.commitment.data.len() != expected.c.data.len()
        || actual.public_input.m_in != expected.m_in
        || actual.openings.r.len() != expected.r.len()
        || actual.openings.y_ring.len() != expected.y_ring.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let _ = compact_x_values_from_native_claim(expected)?;
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "c_data"),
        &actual.commitment.data,
        &expected.c.data,
        &format!("{label}_c_data"),
    )?;
    enforce_claim_x_projection_eq_native(&mut cs.namespace(|| "x"), actual, expected, &format!("{label}_x"))?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "r"),
        &actual.openings.r,
        &expected.r,
        &format!("{label}_r"),
    )?;
    for (row_idx, (actual_row, expected_row)) in actual
        .openings
        .y_ring
        .iter()
        .zip(expected.y_ring.iter())
        .enumerate()
    {
        enforce_k_slice_eq_native(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    Ok(())
}

pub fn enforce_claim_y_ring_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.openings.y_ring.len() < expected.y_ring.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (row_idx, (actual_row, expected_row)) in actual
        .openings
        .y_ring
        .iter()
        .zip(expected.y_ring.iter())
        .enumerate()
    {
        enforce_k_slice_eq_native(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    Ok(())
}

pub fn enforce_claim_y_zcol_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.norm_check.y_zcol.len() < expected.y_zcol.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    enforce_k_slice_eq_native(cs, &actual.norm_check.y_zcol, &expected.y_zcol, label)
}

pub fn enforce_claim_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CircuitCeClaim,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_f_slice_eq(
        &mut cs.namespace(|| "c_data"),
        &actual.commitment.data,
        &expected.commitment.data,
        &format!("{label}_c_data"),
    )?;
    enforce_f_slice_eq(
        &mut cs.namespace(|| "x"),
        &actual.public_input.x,
        &expected.public_input.x,
        &format!("{label}_x"),
    )?;
    enforce_k_slice_eq(
        &mut cs.namespace(|| "r"),
        &actual.openings.r,
        &expected.openings.r,
        &format!("{label}_r"),
    )?;
    enforce_k_slice_eq(
        &mut cs.namespace(|| "s_col"),
        &actual.norm_check.s_col,
        &expected.norm_check.s_col,
        &format!("{label}_s_col"),
    )?;
    if actual.openings.y_ring.len() != expected.openings.y_ring.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (row_idx, (actual_row, expected_row)) in actual
        .openings
        .y_ring
        .iter()
        .zip(expected.openings.y_ring.iter())
        .enumerate()
    {
        enforce_k_slice_eq(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    enforce_k_slice_eq(
        &mut cs.namespace(|| "ct"),
        &actual.openings.ct,
        &expected.openings.ct,
        &format!("{label}_ct"),
    )?;
    enforce_k_slice_eq(
        &mut cs.namespace(|| "aux_openings"),
        &actual.openings.aux_openings,
        &expected.openings.aux_openings,
        &format!("{label}_aux_openings"),
    )?;
    enforce_k_slice_eq(
        &mut cs.namespace(|| "y_zcol"),
        &actual.norm_check.y_zcol,
        &expected.norm_check.y_zcol,
        &format!("{label}_y_zcol"),
    )?;
    enforce_f_slice_eq(
        &mut cs.namespace(|| "c_step_coords"),
        &actual.step_binding.c_step_coords,
        &expected.step_binding.c_step_coords,
        &format!("{label}_c_step_coords"),
    )?;
    enforce_f_slice_eq(
        &mut cs.namespace(|| "fold_digest"),
        &actual.step_binding.fold_digest_encoding,
        &expected.step_binding.fold_digest_encoding,
        &format!("{label}_fold_digest"),
    )?;
    if actual.public_input.m_in != expected.public_input.m_in
        || actual.step_binding.u_offset != expected.step_binding.u_offset
        || actual.step_binding.u_len != expected.step_binding.u_len
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

fn enforce_claim_x_projection_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.public_input.m_in != expected.m_in {
        return Err(SynthesisError::Unsatisfiable);
    }
    let expected_compact_x = compact_x_values_from_native_claim(expected)?;
    if actual.public_input.x.len() == actual.public_input.m_in {
        for (col, expected_value) in expected_compact_x.iter().enumerate() {
            let expected_x = SpartanF::from_canonical_u64(expected_value.as_canonical_u64());
            cs.enforce(
                || format!("{label}_{col}"),
                |lc| lc + actual.public_input.x[col].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (expected_x, CS::one()),
            );
        }
        return Ok(());
    }
    if actual.public_input.rows != D
        || actual.public_input.cols != expected.m_in
        || actual.public_input.x.len()
            != actual
                .public_input
                .rows
                .saturating_mul(actual.public_input.cols)
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    for col in 0..actual.public_input.m_in {
        let row = col % D;
        let block = col / D;
        let idx = row
            .checked_mul(actual.public_input.cols)
            .and_then(|start| start.checked_add(block))
            .ok_or(SynthesisError::Unsatisfiable)?;
        let actual_x = actual
            .public_input
            .x
            .get(idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let expected_x = SpartanF::from_canonical_u64(expected_compact_x[col].as_canonical_u64());
        cs.enforce(
            || format!("{label}_{row}_{col}"),
            |lc| lc + actual_x.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (expected_x, CS::one()),
        );
    }
    Ok(())
}

fn enforce_claim_x_projection_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CircuitCeClaim,
    expected: &CircuitCeClaim,
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.public_input.m_in != expected.public_input.m_in {
        return Err(SynthesisError::Unsatisfiable);
    }
    for col in 0..actual.public_input.m_in {
        let actual_x = claim_projection_x_lane(actual, col)?;
        let expected_x = claim_projection_x_lane(expected, col)?;
        cs.enforce(
            || format!("{label}_{col}"),
            |lc| lc + actual_x.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected_x.get_variable(),
        );
    }
    Ok(())
}

fn claim_projection_x_lane(claim: &CircuitCeClaim, col: usize) -> Result<&AllocatedNum<SpartanF>, SynthesisError> {
    if claim.public_input.x.len() == claim.public_input.m_in {
        return claim
            .public_input
            .x
            .get(col)
            .ok_or(SynthesisError::Unsatisfiable);
    }
    if claim.public_input.rows != D
        || claim.public_input.cols != claim.public_input.m_in
        || claim.public_input.x.len()
            != claim
                .public_input
                .rows
                .saturating_mul(claim.public_input.cols)
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let row = col % D;
    let block = col / D;
    let idx = row
        .checked_mul(claim.public_input.cols)
        .and_then(|start| start.checked_add(block))
        .ok_or(SynthesisError::Unsatisfiable)?;
    claim
        .public_input
        .x
        .get(idx)
        .ok_or(SynthesisError::Unsatisfiable)
}
