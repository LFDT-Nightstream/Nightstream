use super::alloc::compact_x_values_from_native_claim;
use super::encoding::*;
use super::fields::{extend_f_slice_values, extend_k_slice_values, packed_bytes_field_values};
use super::*;

fn me_digest_field_capacity(
    c_data_len: usize,
    x_len: usize,
    r_len: usize,
    s_col_len: usize,
    y_zcol_len: usize,
    y_ring_lens: impl Iterator<Item = usize>,
    ct_len: usize,
    aux_openings_len: usize,
    c_step_coords_len: usize,
    fold_digest_encoding_len: usize,
) -> usize {
    let mut total = packed_bytes_field_values(b"neo/ccs/me_input_digest_poseidon/v2").len();
    total += 1 + c_data_len;
    total += 1 + x_len;
    total += 2 + (2 * r_len);
    total += 2 + (2 * s_col_len);
    total += 2 + (2 * y_zcol_len);
    total += 1;
    for row_len in y_ring_lens {
        total += 2 + (2 * row_len);
    }
    total += 2 + (2 * ct_len);
    total += 2 + (2 * aux_openings_len);
    total += 1 + c_step_coords_len;
    total += 3;
    total + fold_digest_encoding_len
}

fn me_input_projection_digest_field_capacity(
    c_data_len: usize,
    x_len: usize,
    r_len: usize,
    y_ring_lens: impl IntoIterator<Item = usize>,
) -> usize {
    let mut total = packed_bytes_field_values(b"neo/ccs/me_input_projection_digest_poseidon/v2").len();
    total += 1 + c_data_len;
    total += 1 + x_len;
    total += 2 + (2 * r_len);
    total += 1;
    for row_len in y_ring_lens {
        total += 2 + (2 * row_len);
    }
    total + 1
}

pub fn me_digest_poseidon<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CircuitCeClaim,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let field_capacity = me_digest_field_capacity(
        claim.commitment.data.len(),
        claim.public_input.x.len(),
        claim.openings.r.len(),
        claim.norm_check.s_col.len(),
        claim.norm_check.y_zcol.len(),
        claim.openings.y_ring.iter().map(|row| row.len()),
        claim.openings.ct.len(),
        claim.openings.aux_openings.len(),
        claim.step_binding.c_step_coords.len(),
        claim.step_binding.fold_digest_encoding.len(),
    );
    let mut field_terms = Vec::with_capacity(field_capacity);
    let mut field_constants = Vec::with_capacity(field_capacity);
    let mut field_values = Vec::with_capacity(field_capacity);
    extend_packed_bytes_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        b"neo/ccs/me_input_digest_poseidon/v2",
    );
    extend_f_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.commitment.data,
        &claim.commitment.data_values,
    )?;
    extend_f_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.public_input.x,
        &claim.public_input.x_values,
    )?;
    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.r,
        &claim.openings.r_values,
    )?;
    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.norm_check.s_col,
        &claim.norm_check.s_col_values,
    )?;
    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.norm_check.y_zcol,
        &claim.norm_check.y_zcol_values,
    )?;

    push_constant_lc_field(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        SpartanF::from_canonical_u64(claim.openings.y_ring.len() as u64),
    );
    for (row_idx, row) in claim.openings.y_ring.iter().enumerate() {
        let _ = row_idx;
        extend_k_slice_as_lc_fields(
            &mut field_terms,
            &mut field_constants,
            &mut field_values,
            row,
            &claim.openings.y_ring_values[row_idx],
        )?;
    }

    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.ct,
        &claim.openings.ct_values,
    )?;
    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.aux_openings,
        &claim.openings.aux_openings_values,
    )?;
    extend_f_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.step_binding.c_step_coords,
        &claim.step_binding.c_step_coords_values,
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_m_in")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        claim.public_input.m_in as u64,
        "m_in",
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_u_offset")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        claim.step_binding.u_offset as u64,
        "u_offset",
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_u_len")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        claim.step_binding.u_len as u64,
        "u_len",
    )?;
    extend_allocated_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.step_binding.fold_digest_encoding,
        &claim.step_binding.fold_digest_encoding_values,
    )?;

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub fn me_input_projection_digest_poseidon<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CircuitCeClaim,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let field_capacity = me_input_projection_digest_field_capacity(
        claim.commitment.data.len(),
        claim.public_input.m_in,
        claim.openings.r.len(),
        claim.openings.y_ring.iter().map(|row| row.len()),
    );
    let mut field_terms = Vec::with_capacity(field_capacity);
    let mut field_constants = Vec::with_capacity(field_capacity);
    let mut field_values = Vec::with_capacity(field_capacity);
    extend_packed_bytes_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        b"neo/ccs/me_input_projection_digest_poseidon/v2",
    );
    extend_f_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.commitment.data,
        &claim.commitment.data_values,
    )?;
    extend_superneo_compact_x_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.public_input.x,
        claim.public_input.rows,
        claim.public_input.m_in,
        &superneo_compact_x_values(
            &claim.public_input.x_values,
            claim.public_input.rows,
            claim.public_input.cols,
        )?,
    )?;
    extend_k_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.r,
        &claim.openings.r_values,
    )?;
    push_constant_lc_field(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        SpartanF::from_canonical_u64(claim.openings.y_ring.len() as u64),
    );
    for (row_idx, row) in claim.openings.y_ring.iter().enumerate() {
        extend_k_slice_as_lc_fields(
            &mut field_terms,
            &mut field_constants,
            &mut field_values,
            row,
            &claim.openings.y_ring_values[row_idx],
        )?;
    }
    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub fn me_digest_poseidon_with_native_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CircuitCeClaim,
    native_claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    if claim.commitment.data.len() < native_claim.c.data.len()
        || claim.public_input.x.len() < native_claim.X.as_slice().len()
        || claim.openings.r.len() < native_claim.r.len()
        || claim.norm_check.s_col.len() < native_claim.s_col.len()
        || claim.norm_check.y_zcol.len() < native_claim.y_zcol.len()
        || claim.openings.y_ring.len() < native_claim.y_ring.len()
        || claim.openings.ct.len() < native_claim.ct.len()
        || claim.openings.aux_openings.len() < native_claim.aux_openings.len()
        || claim.step_binding.c_step_coords.len() < native_claim.c_step_coords.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let field_capacity = me_digest_field_capacity(
        native_claim.c.data.len(),
        native_claim.X.as_slice().len(),
        native_claim.r.len(),
        native_claim.s_col.len(),
        native_claim.y_zcol.len(),
        native_claim.y_ring.iter().map(|row| row.len()),
        native_claim.ct.len(),
        native_claim.aux_openings.len(),
        native_claim.c_step_coords.len(),
        claim.step_binding.fold_digest_encoding.len(),
    );
    let mut field_terms = Vec::with_capacity(field_capacity);
    let mut field_constants = Vec::with_capacity(field_capacity);
    let mut field_values = Vec::with_capacity(field_capacity);
    extend_packed_bytes_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        b"neo/ccs/me_input_digest_poseidon/v2",
    );
    extend_f_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.commitment.data,
        &native_claim.c.data,
    )?;
    extend_f_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.public_input.x,
        native_claim.X.as_slice(),
    )?;
    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.r,
        &native_claim.r,
    )?;
    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.norm_check.s_col,
        &native_claim.s_col,
    )?;
    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.norm_check.y_zcol,
        &native_claim.y_zcol,
    )?;

    push_constant_lc_field(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        SpartanF::from_canonical_u64(native_claim.y_ring.len() as u64),
    );
    for (row_idx, native_row) in native_claim.y_ring.iter().enumerate() {
        let _ = row_idx;
        extend_k_slice_prefix_as_lc_fields(
            &mut field_terms,
            &mut field_constants,
            &mut field_values,
            &claim.openings.y_ring[row_idx],
            native_row,
        )?;
    }

    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.ct,
        &native_claim.ct,
    )?;
    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.aux_openings,
        &native_claim.aux_openings,
    )?;
    extend_f_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.step_binding.c_step_coords,
        &native_claim.c_step_coords,
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_m_in")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        native_claim.m_in as u64,
        "m_in",
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_u_offset")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        native_claim.u_offset as u64,
        "u_offset",
    )?;
    alloc_claim_scalar_as_lc_field(
        &mut cs.namespace(|| format!("{label}_u_len")),
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        native_claim.u_len as u64,
        "u_len",
    )?;
    extend_allocated_slice_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.step_binding.fold_digest_encoding,
        &claim.step_binding.fold_digest_encoding_values,
    )?;

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub fn me_input_projection_digest_poseidon_with_native_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CircuitCeClaim,
    native_claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    if claim.commitment.data.len() < native_claim.c.data.len()
        || claim.openings.r.len() < native_claim.r.len()
        || claim.openings.y_ring.len() < native_claim.y_ring.len()
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let field_capacity = me_input_projection_digest_field_capacity(
        native_claim.c.data.len(),
        native_claim.m_in,
        native_claim.r.len(),
        native_claim.y_ring.iter().map(|row| row.len()),
    );
    let mut field_terms = Vec::with_capacity(field_capacity);
    let mut field_constants = Vec::with_capacity(field_capacity);
    let mut field_values = Vec::with_capacity(field_capacity);
    extend_packed_bytes_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        b"neo/ccs/me_input_projection_digest_poseidon/v2",
    );
    extend_f_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.commitment.data,
        &native_claim.c.data,
    )?;
    let native_compact_x = compact_x_values_from_native_claim(native_claim)?;
    extend_superneo_compact_x_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.public_input.x,
        claim.public_input.rows,
        native_claim.m_in,
        &native_compact_x,
    )?;
    extend_k_slice_prefix_as_lc_fields(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        &claim.openings.r,
        &native_claim.r,
    )?;
    push_constant_lc_field(
        &mut field_terms,
        &mut field_constants,
        &mut field_values,
        SpartanF::from_canonical_u64(native_claim.y_ring.len() as u64),
    );
    for (row_idx, native_row) in native_claim.y_ring.iter().enumerate() {
        extend_k_slice_prefix_as_lc_fields(
            &mut field_terms,
            &mut field_constants,
            &mut field_values,
            &claim.openings.y_ring[row_idx],
            native_row,
        )?;
    }
    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

pub fn me_digest_poseidon_values(claim: &CircuitCeClaim) -> [SpartanF; 4] {
    let mut preimage = Vec::new();
    preimage.extend(packed_bytes_field_values(b"neo/ccs/me_input_digest_poseidon/v2"));
    extend_f_slice_values(&mut preimage, &claim.commitment.data_values);
    extend_f_slice_values(&mut preimage, &claim.public_input.x_values);
    extend_k_slice_values(&mut preimage, &claim.openings.r_values);
    extend_k_slice_values(&mut preimage, &claim.norm_check.s_col_values);
    extend_k_slice_values(&mut preimage, &claim.norm_check.y_zcol_values);

    preimage.push(SpartanF::from_canonical_u64(claim.openings.y_ring.len() as u64));
    for row in &claim.openings.y_ring_values {
        extend_k_slice_values(&mut preimage, row);
    }

    extend_k_slice_values(&mut preimage, &claim.openings.ct_values);
    extend_k_slice_values(&mut preimage, &claim.openings.aux_openings_values);
    extend_f_slice_values(&mut preimage, &claim.step_binding.c_step_coords_values);
    preimage.push(SpartanF::from_canonical_u64(claim.public_input.m_in as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.step_binding.u_offset as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.step_binding.u_len as u64));
    preimage.extend(
        claim
            .step_binding
            .fold_digest_encoding_values
            .iter()
            .copied(),
    );

    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(
        &preimage
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>(),
    )
    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}

pub fn me_input_projection_digest_poseidon_values(claim: &CircuitCeClaim) -> Result<[SpartanF; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(packed_bytes_field_values(
        b"neo/ccs/me_input_projection_digest_poseidon/v2",
    ));
    extend_f_slice_values(&mut preimage, &claim.commitment.data_values);
    preimage.push(SpartanF::from_canonical_u64(claim.public_input.m_in as u64));
    let compact_x_values = superneo_compact_x_values(
        &claim.public_input.x_values,
        claim.public_input.rows,
        claim.public_input.cols,
    )?;
    for value in compact_x_values {
        preimage.push(SpartanF::from_canonical_u64(value.as_canonical_u64()));
    }
    extend_k_slice_values(&mut preimage, &claim.openings.r_values);

    preimage.push(SpartanF::from_canonical_u64(claim.openings.y_ring.len() as u64));
    for row in &claim.openings.y_ring_values {
        extend_k_slice_values(&mut preimage, row);
    }

    Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(
        &preimage
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>(),
    )
    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())))
}

pub fn me_digest_poseidon_values_from_native_claim(claim: &CeClaim<Commitment, F, K>) -> [SpartanF; 4] {
    let mut preimage = Vec::new();
    preimage.extend(packed_bytes_field_values(b"neo/ccs/me_input_digest_poseidon/v2"));
    extend_f_slice_values(&mut preimage, &claim.c.data);
    extend_f_slice_values(&mut preimage, claim.X.as_slice());
    extend_k_slice_values(&mut preimage, &claim.r);
    extend_k_slice_values(&mut preimage, &claim.s_col);
    extend_k_slice_values(&mut preimage, &claim.y_zcol);

    preimage.push(SpartanF::from_canonical_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring {
        extend_k_slice_values(&mut preimage, row);
    }

    extend_k_slice_values(&mut preimage, &claim.ct);
    extend_k_slice_values(&mut preimage, &claim.aux_openings);
    extend_f_slice_values(&mut preimage, &claim.c_step_coords);
    preimage.push(SpartanF::from_canonical_u64(claim.m_in as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.u_offset as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.u_len as u64));
    preimage.extend(packed_bytes_field_values(&claim.fold_digest));

    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(
        &preimage
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>(),
    )
    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}

pub fn me_input_projection_digest_poseidon_values_from_native_claim(
    claim: &CeClaim<Commitment, F, K>,
) -> Result<[SpartanF; 4], SynthesisError> {
    let mut preimage = Vec::new();
    preimage.extend(packed_bytes_field_values(
        b"neo/ccs/me_input_projection_digest_poseidon/v2",
    ));
    extend_f_slice_values(&mut preimage, &claim.c.data);
    preimage.push(SpartanF::from_canonical_u64(claim.m_in as u64));
    for value in compact_x_values_from_native_claim(claim)? {
        preimage.push(SpartanF::from_canonical_u64(value.as_canonical_u64()));
    }
    extend_k_slice_values(&mut preimage, &claim.r);

    preimage.push(SpartanF::from_canonical_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring {
        extend_k_slice_values(&mut preimage, row);
    }

    Ok(neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(
        &preimage
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>(),
    )
    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())))
}
