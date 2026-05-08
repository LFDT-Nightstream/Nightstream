use super::*;

pub(super) fn ensure_value_surface(
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
) -> Result<(), SynthesisError> {
    if c_data_values.len() != claim.c.data.len() || x_values.len() != claim.X.as_slice().len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn ensure_commitment_values(
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
) -> Result<(), SynthesisError> {
    if c_data_values.len() != claim.c.data.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn ensure_shared_r(claim: &CeClaim<Commitment, F, K>, shared_r_values: &[K]) -> Result<(), SynthesisError> {
    if claim.r.as_slice() != shared_r_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn ensure_shared_point(
    claim: &CeClaim<Commitment, F, K>,
    shared_r_values: &[K],
    shared_s_col_values: &[K],
) -> Result<(), SynthesisError> {
    if claim.r.as_slice() != shared_r_values || claim.s_col.as_slice() != shared_s_col_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub(super) fn ensure_commitment_alias(
    claim: &CeClaim<Commitment, F, K>,
    c_data: &[AllocatedNum<SpartanF>],
    c_data_values: &[F],
) -> Result<(), SynthesisError> {
    if claim.c.data.len() != c_data.len() || claim.c.data.as_slice() != c_data_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}
