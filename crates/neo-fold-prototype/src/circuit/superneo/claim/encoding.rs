use super::fields::packed_bytes_field_values;
use super::*;

pub(super) fn push_constant_lc_field(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    value: SpartanF,
) {
    field_terms.push(Vec::new());
    field_constants.push(value);
    field_values.push(value);
}

pub(super) fn push_variable_lc_field(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    variable: Variable,
    value: SpartanF,
) {
    field_terms.push(vec![(variable, SpartanF::ONE)]);
    field_constants.push(SpartanF::ZERO);
    field_values.push(value);
}

pub(super) fn extend_packed_bytes_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    bytes: &[u8],
) {
    for value in packed_bytes_field_values(bytes) {
        push_constant_lc_field(field_terms, field_constants, field_values, value);
    }
}

pub(super) fn extend_allocated_slice_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[AllocatedNum<SpartanF>],
    native_values: &[SpartanF],
) -> Result<(), SynthesisError> {
    if values.len() != native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (value, native_value) in values.iter().zip(native_values.iter()) {
        push_variable_lc_field(
            field_terms,
            field_constants,
            field_values,
            value.get_variable(),
            *native_value,
        );
    }
    Ok(())
}

pub(super) fn extend_f_slice_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[AllocatedNum<SpartanF>],
    native_values: &[F],
) -> Result<(), SynthesisError> {
    if values.len() != native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    push_constant_lc_field(
        field_terms,
        field_constants,
        field_values,
        SpartanF::from_canonical_u64(values.len() as u64),
    );
    for (value, native_value) in values.iter().zip(native_values.iter()) {
        push_variable_lc_field(
            field_terms,
            field_constants,
            field_values,
            value.get_variable(),
            SpartanF::from_canonical_u64(native_value.as_canonical_u64()),
        );
    }
    Ok(())
}

pub(super) fn extend_f_slice_prefix_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[AllocatedNum<SpartanF>],
    logical_values: &[F],
) -> Result<(), SynthesisError> {
    if values.len() < logical_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    extend_f_slice_as_lc_fields(
        field_terms,
        field_constants,
        field_values,
        &values[..logical_values.len()],
        logical_values,
    )
}

pub(super) fn extend_superneo_compact_x_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[AllocatedNum<SpartanF>],
    x_rows: usize,
    x_cols: usize,
    native_values: &[F],
) -> Result<(), SynthesisError> {
    if x_rows != D || native_values.len() != x_cols {
        return Err(SynthesisError::Unsatisfiable);
    }
    push_constant_lc_field(
        field_terms,
        field_constants,
        field_values,
        SpartanF::from_canonical_u64(x_cols as u64),
    );
    let use_compact_values = values.len() == x_cols;
    let use_full_values = values.len() == x_rows.saturating_mul(x_cols);
    if !use_compact_values && !use_full_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (col, native_value) in native_values.iter().enumerate() {
        let idx = if use_compact_values {
            col
        } else {
            (col % x_rows) * x_cols + (col / x_rows)
        };
        push_variable_lc_field(
            field_terms,
            field_constants,
            field_values,
            values[idx].get_variable(),
            SpartanF::from_canonical_u64(native_value.as_canonical_u64()),
        );
    }
    Ok(())
}

pub(super) fn superneo_compact_x_values(values: &[F], x_rows: usize, x_cols: usize) -> Result<Vec<F>, SynthesisError> {
    if values.len() == x_cols {
        return Ok(values.to_vec());
    }
    if x_rows != D || values.len() != x_rows.saturating_mul(x_cols) {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut compact = Vec::with_capacity(x_cols);
    for col in 0..x_cols {
        compact.push(values[(col % x_rows) * x_cols + (col / x_rows)]);
    }
    Ok(compact)
}

pub(super) fn extend_k_slice_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[KNumVar],
    native_values: &[K],
) -> Result<(), SynthesisError> {
    if values.len() != native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    push_constant_lc_field(
        field_terms,
        field_constants,
        field_values,
        SpartanF::from_canonical_u64(values.len() as u64),
    );
    let coeff_len = native_values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    push_constant_lc_field(
        field_terms,
        field_constants,
        field_values,
        SpartanF::from_canonical_u64(coeff_len as u64),
    );
    for (value, native_value) in values.iter().zip(native_values.iter()) {
        let coeffs = native_value.as_coeffs();
        push_variable_lc_field(
            field_terms,
            field_constants,
            field_values,
            value.c0,
            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
        );
        push_variable_lc_field(
            field_terms,
            field_constants,
            field_values,
            value.c1,
            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
        );
    }
    Ok(())
}

pub(super) fn extend_k_slice_prefix_as_lc_fields(
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    values: &[KNumVar],
    native_values: &[K],
) -> Result<(), SynthesisError> {
    if values.len() < native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    extend_k_slice_as_lc_fields(
        field_terms,
        field_constants,
        field_values,
        &values[..native_values.len()],
        native_values,
    )
}

pub(super) fn alloc_claim_scalar_as_lc_field<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    field_terms: &mut Vec<Vec<(Variable, SpartanF)>>,
    field_constants: &mut Vec<SpartanF>,
    field_values: &mut Vec<SpartanF>,
    value: u64,
    label: &str,
) -> Result<(), SynthesisError> {
    let scalar = SpartanF::from_canonical_u64(value);
    let allocated = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(scalar))?;
    push_variable_lc_field(
        field_terms,
        field_constants,
        field_values,
        allocated.get_variable(),
        scalar,
    );
    Ok(())
}
