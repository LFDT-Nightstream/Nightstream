use super::*;

pub(super) fn alloc_commitment_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeCommitment, SynthesisError> {
    let data = alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?;
    Ok(CircuitCeCommitment::from_allocated(data, claim.c.data.clone()))
}

pub(super) fn alloc_full_public_input_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCePublicInput, SynthesisError> {
    let x = alloc_f_slice(cs, claim.X.as_slice(), &format!("{label}_x"))?;
    Ok(CircuitCePublicInput::from_claim_parts(
        claim,
        x,
        claim.X.as_slice().to_vec(),
    ))
}

pub(super) fn alloc_compact_public_input_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCePublicInput, SynthesisError> {
    let compact_x_values = super::compact_x_values_from_native_claim(claim)?;
    let x = alloc_f_slice(cs, &compact_x_values, &format!("{label}_x"))?;
    Ok(CircuitCePublicInput::from_claim_parts(claim, x, compact_x_values))
}

pub(super) fn alloc_public_input_from_values<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    x_values: Vec<F>,
    label: &str,
) -> Result<CircuitCePublicInput, SynthesisError> {
    let x = alloc_f_slice(cs, &x_values, &format!("{label}_x"))?;
    Ok(CircuitCePublicInput::from_claim_parts(claim, x, x_values))
}

pub(super) fn alloc_full_openings_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let r = alloc_r(cs, claim, label)?;
    let y_ring = alloc_y_ring(cs, claim, label)?;
    let ct = alloc_ct(cs, claim, label)?;
    let aux_openings = alloc_aux_openings(cs, claim, label)?;
    Ok(CircuitCeOpenings::from_claim_parts(
        claim,
        r,
        claim.r.clone(),
        y_ring,
        ct,
        aux_openings,
    ))
}

pub(super) fn alloc_full_openings_with_shared_point_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let y_ring = alloc_y_ring(cs, claim, label)?;
    let ct = alias_ct_from_y_ring(&y_ring, &claim.y_ring, &claim.ct)?;
    let aux_openings = alloc_aux_openings(cs, claim, label)?;
    Ok(CircuitCeOpenings::from_claim_parts(
        claim,
        shared_r.to_vec(),
        shared_r_values.to_vec(),
        y_ring,
        ct,
        aux_openings,
    ))
}

pub(super) fn alloc_public_openings_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let r = alloc_r(cs, claim, label)?;
    let y_ring = alloc_y_ring(cs, claim, label)?;
    let ct = alloc_ct(cs, claim, label)?;
    Ok(CircuitCeOpenings::from_claim_parts(
        claim,
        r,
        claim.r.clone(),
        y_ring,
        ct,
        Vec::new(),
    ))
}

pub(super) fn alloc_public_openings_with_shared_point_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let y_ring = alloc_y_ring(cs, claim, label)?;
    let ct = alias_ct_from_y_ring(&y_ring, &claim.y_ring, &claim.ct)?;
    Ok(CircuitCeOpenings::from_claim_parts(
        claim,
        shared_r.to_vec(),
        shared_r_values.to_vec(),
        y_ring,
        ct,
        Vec::new(),
    ))
}

pub(super) fn alloc_y_ring_openings_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let r = alloc_r(cs, claim, label)?;
    let y_ring = alloc_y_ring(cs, claim, label)?;
    Ok(CircuitCeOpenings::y_ring_only(claim, r, claim.r.clone(), y_ring))
}

pub(super) fn alloc_y_ring_openings_with_shared_r_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let y_ring = alloc_y_ring(cs, claim, label)?;
    Ok(CircuitCeOpenings::y_ring_only(
        claim,
        shared_r.to_vec(),
        shared_r_values.to_vec(),
        y_ring,
    ))
}

pub(super) fn alloc_point_openings_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeOpenings, SynthesisError> {
    let r = alloc_r(cs, claim, label)?;
    Ok(CircuitCeOpenings::point_only(claim, r, claim.r.clone()))
}

pub(super) fn alloc_full_norm_check_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeNormCheck, SynthesisError> {
    let s_col = alloc_s_col(cs, claim, label)?;
    let y_zcol = alloc_y_zcol(cs, claim, label)?;
    Ok(CircuitCeNormCheck::from_claim_parts(
        claim,
        s_col,
        claim.s_col.clone(),
        y_zcol,
    ))
}

pub(super) fn alloc_norm_check_with_shared_s_col_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CircuitCeNormCheck, SynthesisError> {
    let y_zcol = alloc_y_zcol(cs, claim, label)?;
    Ok(CircuitCeNormCheck::from_claim_parts(
        claim,
        shared_s_col.to_vec(),
        shared_s_col_values.to_vec(),
        y_zcol,
    ))
}

fn alias_ct_from_y_ring(
    y_ring: &[Vec<KNumVar>],
    y_ring_values: &[Vec<K>],
    ct_values: &[K],
) -> Result<Vec<KNumVar>, SynthesisError> {
    if ct_values.len() > y_ring.len() || y_ring_values.len() != y_ring.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let mut ct = Vec::with_capacity(ct_values.len());
    for (idx, expected) in ct_values.iter().enumerate() {
        let row = y_ring.get(idx).ok_or(SynthesisError::Unsatisfiable)?;
        let row_values = y_ring_values
            .get(idx)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let first = row.first().ok_or(SynthesisError::Unsatisfiable)?;
        if row_values.first().copied() != Some(*expected) {
            return Err(SynthesisError::Unsatisfiable);
        }
        ct.push(first.clone());
    }
    Ok(ct)
}

pub(super) fn alloc_step_binding_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeStepBinding, SynthesisError> {
    let c_step_coords = alloc_c_step_coords(cs, claim, label)?;
    let fold_digest_encoding_values = packed_bytes_field_values(&claim.fold_digest);
    let fold_digest_encoding = alloc_packed_bytes_as_fields(cs, &claim.fold_digest, &format!("{label}_fold_digest"))?;
    Ok(CircuitCeStepBinding::from_claim_parts(
        claim,
        c_step_coords,
        fold_digest_encoding,
        fold_digest_encoding_values,
    ))
}

pub(super) fn alloc_step_binding_without_digest_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CircuitCeStepBinding, SynthesisError> {
    let c_step_coords = alloc_c_step_coords(cs, claim, label)?;
    Ok(CircuitCeStepBinding::without_fold_digest(claim, c_step_coords))
}

fn alloc_r<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    alloc_k_slice(cs, &claim.r, &format!("{label}_r"))
}

fn alloc_s_col<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    alloc_k_slice(cs, &claim.s_col, &format!("{label}_s_col"))
}

fn alloc_y_ring<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<Vec<KNumVar>>, SynthesisError> {
    claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect()
}

fn alloc_ct<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    alloc_k_slice(cs, &claim.ct, &format!("{label}_ct"))
}

fn alloc_aux_openings<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))
}

fn alloc_y_zcol<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))
}

fn alloc_c_step_coords<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    alloc_f_slice(cs, &claim.c_step_coords, &format!("{label}_c_step_coords"))
}
