use super::fields::{alloc_f_slice, alloc_k_slice, alloc_packed_bytes_as_fields, packed_bytes_field_values};
use super::*;

mod surfaces;
mod validate;

use surfaces::*;
use validate::*;

pub(super) fn compact_x_values_from_native_claim(claim: &CeClaim<Commitment, F, K>) -> Result<Vec<F>, SynthesisError> {
    if claim.X.rows() != D || claim.X.cols() != claim.m_in {
        return Err(SynthesisError::Unsatisfiable);
    }
    // CE `x` is the canonical embedded field-vector surface used by
    // Construction 2 and the recursive Π_RLC / Π_DEC checks. Falling back to a
    // full `X` matrix based on witness values makes the recursive circuit
    // shape depend on the concrete chunk, which violates HyperNova §6.3's
    // fixed-shape requirement.
    Ok((0..claim.m_in)
        .map(|col| claim.X[(col % D, col / D)])
        .collect())
}

pub fn alloc_ce_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_full_openings_surface(cs, claim, label)?,
        alloc_full_norm_check_surface(cs, claim, label)?,
        alloc_step_binding_surface(cs, claim, label)?,
    ))
}

pub fn alloc_ce_claim_without_fold_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_full_openings_surface(cs, claim, label)?,
        alloc_full_norm_check_surface(cs, claim, label)?,
        alloc_step_binding_without_digest_surface(cs, claim, label)?,
    ))
}

pub fn alloc_ce_claim_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_full_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        alloc_step_binding_surface(cs, claim, label)?,
    ))
}

pub fn alloc_ce_claim_with_shared_point_without_fold_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_full_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        alloc_step_binding_without_digest_surface(cs, claim, label)?,
    ))
}

pub fn alloc_ce_claim_public_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_public_openings_surface(cs, claim, label)?,
        alloc_full_norm_check_surface(cs, claim, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_public_surface_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_public_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data: &[AllocatedNum<SpartanF>],
    c_data_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;
    ensure_commitment_alias(claim, c_data, c_data_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::from_allocated(c_data.to_vec(), c_data_values.to_vec()),
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_public_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_public_surface_with_alias_c_data_and_shared_point_compact_x<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data: &[AllocatedNum<SpartanF>],
    c_data_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;
    ensure_commitment_alias(claim, c_data, c_data_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::from_allocated(c_data.to_vec(), c_data_values.to_vec()),
        alloc_compact_public_input_surface(cs, claim, label)?,
        alloc_public_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_projection_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_compact_public_input_surface(cs, claim, label)?,
        alloc_y_ring_openings_surface(cs, claim, label)?,
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_dec_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_y_ring_openings_surface(cs, claim, label)?,
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_dec_surface_with_shared_r<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_r(claim, shared_r_values)?;

    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_full_public_input_surface(cs, claim, label)?,
        alloc_y_ring_openings_with_shared_r_surface(cs, claim, shared_r, shared_r_values, label)?,
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_projection_surface_with_shared_r<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_r(claim, shared_r_values)?;

    Ok(CircuitCeClaim::from_parts(
        alloc_commitment_surface(cs, claim, label)?,
        alloc_compact_public_input_surface(cs, claim, label)?,
        alloc_y_ring_openings_with_shared_r_surface(cs, claim, shared_r, shared_r_values, label)?,
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_x_r_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(claim.c.data.clone()),
        alloc_compact_public_input_surface(cs, claim, label)?,
        alloc_point_openings_surface(cs, claim, label)?,
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_x_r_surface_with_shared_r<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_shared_r(claim, shared_r_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(claim.c.data.clone()),
        alloc_compact_public_input_surface(cs, claim, label)?,
        CircuitCeOpenings::point_only(claim, shared_r.to_vec(), shared_r_values.to_vec()),
        CircuitCeNormCheck::values_only(claim),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_without_f_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_value_surface(claim, c_data_values, x_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(c_data_values.to_vec()),
        CircuitCePublicInput::values_only(claim, x_values.to_vec()),
        alloc_public_openings_surface(cs, claim, label)?,
        alloc_full_norm_check_surface(cs, claim, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_without_f_surface_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_value_surface(claim, c_data_values, x_values)?;
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(c_data_values.to_vec()),
        CircuitCePublicInput::values_only(claim, x_values.to_vec()),
        alloc_public_openings_with_shared_point_surface(cs, claim, shared_r, shared_r_values, label)?,
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_y_zcol_surface_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_value_surface(claim, c_data_values, x_values)?;
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(c_data_values.to_vec()),
        CircuitCePublicInput::values_only(claim, x_values.to_vec()),
        CircuitCeOpenings::point_only(claim, shared_r.to_vec(), shared_r_values.to_vec()),
        alloc_norm_check_with_shared_s_col_surface(cs, claim, shared_s_col, shared_s_col_values, label)?,
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_x_surface_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_commitment_values(claim, c_data_values)?;
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;
    let x_values = claim.X.as_slice().to_vec();

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(c_data_values.to_vec()),
        alloc_public_input_from_values(cs, claim, x_values, label)?,
        CircuitCeOpenings::point_only(claim, shared_r.to_vec(), shared_r_values.to_vec()),
        CircuitCeNormCheck::from_claim_parts(claim, shared_s_col.to_vec(), shared_s_col_values.to_vec(), Vec::new()),
        CircuitCeStepBinding::values_only(claim),
    ))
}

pub fn alloc_ce_claim_point_only_with_shared_point(
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
) -> Result<CircuitCeClaim, SynthesisError> {
    ensure_value_surface(claim, c_data_values, x_values)?;
    ensure_shared_point(claim, shared_r_values, shared_s_col_values)?;

    Ok(CircuitCeClaim::from_parts(
        CircuitCeCommitment::values_only(c_data_values.to_vec()),
        CircuitCePublicInput::values_only(claim, x_values.to_vec()),
        CircuitCeOpenings::point_only(claim, shared_r.to_vec(), shared_r_values.to_vec()),
        CircuitCeNormCheck::from_claim_parts(claim, shared_s_col.to_vec(), shared_s_col_values.to_vec(), Vec::new()),
        CircuitCeStepBinding::values_only(claim),
    ))
}
