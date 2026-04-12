//! Owns CE-claim allocation and Poseidon2 claim-digest gadgets for RV64IM main-relation circuits.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::CeClaim;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;
use p3_field::PrimeField64;
use spartan2::{bellpepper::poseidon2::hash_packed_goldilocks_fields, provider::goldi::F as SpartanF};

use super::k_field::{alloc_k, KNum, KNumVar};

#[derive(Clone)]
pub struct CeClaimVar {
    pub c_data: Vec<AllocatedNum<SpartanF>>,
    pub c_data_values: Vec<F>,
    pub x: Vec<AllocatedNum<SpartanF>>,
    pub x_values: Vec<F>,
    pub x_rows: usize,
    pub x_cols: usize,
    pub r: Vec<KNumVar>,
    pub r_values: Vec<K>,
    pub s_col: Vec<KNumVar>,
    pub s_col_values: Vec<K>,
    pub y_ring: Vec<Vec<KNumVar>>,
    pub y_ring_values: Vec<Vec<K>>,
    pub ct: Vec<KNumVar>,
    pub ct_values: Vec<K>,
    pub aux_openings: Vec<KNumVar>,
    pub aux_openings_values: Vec<K>,
    pub y_zcol: Vec<KNumVar>,
    pub y_zcol_values: Vec<K>,
    pub c_step_coords: Vec<AllocatedNum<SpartanF>>,
    pub c_step_coords_values: Vec<F>,
    pub fold_digest_encoding: Vec<AllocatedNum<SpartanF>>,
    pub fold_digest_encoding_values: Vec<SpartanF>,
    pub m_in: usize,
    pub u_offset: usize,
    pub u_len: usize,
}

pub fn alloc_ce_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CeClaimVar, SynthesisError> {
    let c_data = alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?;
    let x = alloc_f_slice(cs, claim.X.as_slice(), &format!("{label}_x"))?;
    let r = alloc_k_slice(cs, &claim.r, &format!("{label}_r"))?;
    let s_col = alloc_k_slice(cs, &claim.s_col, &format!("{label}_s_col"))?;
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alloc_k_slice(cs, &claim.ct, &format!("{label}_ct"))?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;
    let c_step_coords = alloc_f_slice(cs, &claim.c_step_coords, &format!("{label}_c_step_coords"))?;
    let fold_digest_encoding_values = packed_bytes_field_values(&claim.fold_digest);
    let fold_digest_encoding = alloc_packed_bytes_as_fields(cs, &claim.fold_digest, &format!("{label}_fold_digest"))?;

    Ok(CeClaimVar {
        c_data,
        c_data_values: claim.c.data.clone(),
        x,
        x_values: claim.X.as_slice().to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r,
        r_values: claim.r.clone(),
        s_col,
        s_col_values: claim.s_col.clone(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords,
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding,
        fold_digest_encoding_values,
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

pub fn alloc_ce_claim_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CeClaimVar, SynthesisError> {
    if claim.r.as_slice() != shared_r_values || claim.s_col.as_slice() != shared_s_col_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    let c_data = alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?;
    let x = alloc_f_slice(cs, claim.X.as_slice(), &format!("{label}_x"))?;
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alias_ct_from_y_ring(&y_ring, &claim.y_ring, &claim.ct)?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;
    let c_step_coords = alloc_f_slice(cs, &claim.c_step_coords, &format!("{label}_c_step_coords"))?;
    let fold_digest_encoding_values = packed_bytes_field_values(&claim.fold_digest);
    let fold_digest_encoding = alloc_packed_bytes_as_fields(cs, &claim.fold_digest, &format!("{label}_fold_digest"))?;

    Ok(CeClaimVar {
        c_data,
        c_data_values: claim.c.data.clone(),
        x,
        x_values: claim.X.as_slice().to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r: shared_r.to_vec(),
        r_values: shared_r_values.to_vec(),
        s_col: shared_s_col.to_vec(),
        s_col_values: shared_s_col_values.to_vec(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords,
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding,
        fold_digest_encoding_values,
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

pub fn alloc_ce_claim_public_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<CeClaimVar, SynthesisError> {
    let c_data = alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?;
    let x = alloc_f_slice(cs, claim.X.as_slice(), &format!("{label}_x"))?;
    let r = alloc_k_slice(cs, &claim.r, &format!("{label}_r"))?;
    let s_col = alloc_k_slice(cs, &claim.s_col, &format!("{label}_s_col"))?;
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alloc_k_slice(cs, &claim.ct, &format!("{label}_ct"))?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;

    Ok(CeClaimVar {
        c_data,
        c_data_values: claim.c.data.clone(),
        x,
        x_values: claim.X.as_slice().to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r,
        r_values: claim.r.clone(),
        s_col,
        s_col_values: claim.s_col.clone(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

pub fn alloc_ce_claim_public_surface_with_shared_point<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
    label: &str,
) -> Result<CeClaimVar, SynthesisError> {
    if claim.r.as_slice() != shared_r_values || claim.s_col.as_slice() != shared_s_col_values {
        return Err(SynthesisError::Unsatisfiable);
    }
    let c_data = alloc_f_slice(cs, &claim.c.data, &format!("{label}_c_data"))?;
    let x = alloc_f_slice(cs, claim.X.as_slice(), &format!("{label}_x"))?;
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alias_ct_from_y_ring(&y_ring, &claim.y_ring, &claim.ct)?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;

    Ok(CeClaimVar {
        c_data,
        c_data_values: claim.c.data.clone(),
        x,
        x_values: claim.X.as_slice().to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r: shared_r.to_vec(),
        r_values: shared_r_values.to_vec(),
        s_col: shared_s_col.to_vec(),
        s_col_values: shared_s_col_values.to_vec(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

pub fn alloc_ce_claim_without_f_surface<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    label: &str,
) -> Result<CeClaimVar, SynthesisError> {
    if c_data_values.len() != claim.c.data.len() || x_values.len() != claim.X.as_slice().len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    let r = alloc_k_slice(cs, &claim.r, &format!("{label}_r"))?;
    let s_col = alloc_k_slice(cs, &claim.s_col, &format!("{label}_s_col"))?;
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alloc_k_slice(cs, &claim.ct, &format!("{label}_ct"))?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;

    Ok(CeClaimVar {
        c_data: Vec::new(),
        c_data_values: c_data_values.to_vec(),
        x: Vec::new(),
        x_values: x_values.to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r,
        r_values: claim.r.clone(),
        s_col,
        s_col_values: claim.s_col.clone(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
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
) -> Result<CeClaimVar, SynthesisError> {
    if c_data_values.len() != claim.c.data.len()
        || x_values.len() != claim.X.as_slice().len()
        || claim.r.as_slice() != shared_r_values
        || claim.s_col.as_slice() != shared_s_col_values
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let y_ring = claim
        .y_ring
        .iter()
        .enumerate()
        .map(|(row_idx, row)| alloc_k_slice(cs, row, &format!("{label}_y_ring_{row_idx}")))
        .collect::<Result<Vec<_>, _>>()?;
    let ct = alias_ct_from_y_ring(&y_ring, &claim.y_ring, &claim.ct)?;
    let aux_openings = alloc_k_slice(cs, &claim.aux_openings, &format!("{label}_aux_openings"))?;
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;

    Ok(CeClaimVar {
        c_data: Vec::new(),
        c_data_values: c_data_values.to_vec(),
        x: Vec::new(),
        x_values: x_values.to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r: shared_r.to_vec(),
        r_values: shared_r_values.to_vec(),
        s_col: shared_s_col.to_vec(),
        s_col_values: shared_s_col_values.to_vec(),
        y_ring,
        y_ring_values: claim.y_ring.clone(),
        ct,
        ct_values: claim.ct.clone(),
        aux_openings,
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
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
) -> Result<CeClaimVar, SynthesisError> {
    if c_data_values.len() != claim.c.data.len()
        || x_values.len() != claim.X.as_slice().len()
        || claim.r.as_slice() != shared_r_values
        || claim.s_col.as_slice() != shared_s_col_values
    {
        return Err(SynthesisError::Unsatisfiable);
    }
    let y_zcol = alloc_k_slice(cs, &claim.y_zcol, &format!("{label}_y_zcol"))?;

    Ok(CeClaimVar {
        c_data: Vec::new(),
        c_data_values: c_data_values.to_vec(),
        x: Vec::new(),
        x_values: x_values.to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r: shared_r.to_vec(),
        r_values: shared_r_values.to_vec(),
        s_col: shared_s_col.to_vec(),
        s_col_values: shared_s_col_values.to_vec(),
        y_ring: Vec::new(),
        y_ring_values: claim.y_ring.clone(),
        ct: Vec::new(),
        ct_values: claim.ct.clone(),
        aux_openings: Vec::new(),
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol,
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
}

pub fn alloc_ce_claim_point_only_with_shared_point(
    claim: &CeClaim<Commitment, F, K>,
    c_data_values: &[F],
    x_values: &[F],
    shared_r: &[KNumVar],
    shared_r_values: &[K],
    shared_s_col: &[KNumVar],
    shared_s_col_values: &[K],
) -> Result<CeClaimVar, SynthesisError> {
    if c_data_values.len() != claim.c.data.len()
        || x_values.len() != claim.X.as_slice().len()
        || claim.r.as_slice() != shared_r_values
        || claim.s_col.as_slice() != shared_s_col_values
    {
        return Err(SynthesisError::Unsatisfiable);
    }

    Ok(CeClaimVar {
        c_data: Vec::new(),
        c_data_values: c_data_values.to_vec(),
        x: Vec::new(),
        x_values: x_values.to_vec(),
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        r: shared_r.to_vec(),
        r_values: shared_r_values.to_vec(),
        s_col: shared_s_col.to_vec(),
        s_col_values: shared_s_col_values.to_vec(),
        y_ring: Vec::new(),
        y_ring_values: claim.y_ring.clone(),
        ct: Vec::new(),
        ct_values: claim.ct.clone(),
        aux_openings: Vec::new(),
        aux_openings_values: claim.aux_openings.clone(),
        y_zcol: Vec::new(),
        y_zcol_values: claim.y_zcol.clone(),
        c_step_coords: Vec::new(),
        c_step_coords_values: claim.c_step_coords.clone(),
        fold_digest_encoding: Vec::new(),
        fold_digest_encoding_values: packed_bytes_field_values(&claim.fold_digest),
        m_in: claim.m_in,
        u_offset: claim.u_offset,
        u_len: claim.u_len,
    })
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

pub fn me_digest_poseidon<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &CeClaimVar,
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields(
        cs,
        &mut preimage,
        b"neo/ccs/me_input_digest_poseidon/v2",
        &format!("{label}_domain"),
    )?;
    extend_f_slice_as_fields(cs, &mut preimage, &claim.c_data, &format!("{label}_c_data"))?;
    extend_f_slice_as_fields(cs, &mut preimage, &claim.x, &format!("{label}_x"))?;
    extend_k_slice_as_fields(cs, &mut preimage, &claim.r, &claim.r_values, &format!("{label}_r"))?;
    extend_k_slice_as_fields(
        cs,
        &mut preimage,
        &claim.s_col,
        &claim.s_col_values,
        &format!("{label}_s_col"),
    )?;
    extend_k_slice_as_fields(
        cs,
        &mut preimage,
        &claim.y_zcol,
        &claim.y_zcol_values,
        &format!("{label}_y_zcol"),
    )?;

    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.y_ring.len() as u64),
        &format!("{label}_y_ring_len"),
    )?);
    for (row_idx, row) in claim.y_ring.iter().enumerate() {
        extend_k_slice_as_fields(
            cs,
            &mut preimage,
            row,
            &claim.y_ring_values[row_idx],
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }

    extend_k_slice_as_fields(cs, &mut preimage, &claim.ct, &claim.ct_values, &format!("{label}_ct"))?;
    extend_k_slice_as_fields(
        cs,
        &mut preimage,
        &claim.aux_openings,
        &claim.aux_openings_values,
        &format!("{label}_aux"),
    )?;
    extend_f_slice_as_fields(
        cs,
        &mut preimage,
        &claim.c_step_coords,
        &format!("{label}_c_step_coords"),
    )?;
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.m_in as u64),
        &format!("{label}_m_in"),
    )?);
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.u_offset as u64),
        &format!("{label}_u_offset"),
    )?);
    preimage.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(claim.u_len as u64),
        &format!("{label}_u_len"),
    )?);
    preimage.extend(claim.fold_digest_encoding.iter().cloned());

    hash_packed_goldilocks_fields(cs.namespace(|| format!("{label}_hash")), &preimage)
}

pub fn me_digest_poseidon_values(claim: &CeClaimVar) -> [SpartanF; 4] {
    let mut preimage = Vec::new();
    preimage.extend(packed_bytes_field_values(b"neo/ccs/me_input_digest_poseidon/v2"));
    extend_f_slice_values(&mut preimage, &claim.c_data_values);
    extend_f_slice_values(&mut preimage, &claim.x_values);
    extend_k_slice_values(&mut preimage, &claim.r_values);
    extend_k_slice_values(&mut preimage, &claim.s_col_values);
    extend_k_slice_values(&mut preimage, &claim.y_zcol_values);

    preimage.push(SpartanF::from_canonical_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring_values {
        extend_k_slice_values(&mut preimage, row);
    }

    extend_k_slice_values(&mut preimage, &claim.ct_values);
    extend_k_slice_values(&mut preimage, &claim.aux_openings_values);
    extend_f_slice_values(&mut preimage, &claim.c_step_coords_values);
    preimage.push(SpartanF::from_canonical_u64(claim.m_in as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.u_offset as u64));
    preimage.push(SpartanF::from_canonical_u64(claim.u_len as u64));
    preimage.extend(claim.fold_digest_encoding_values.iter().copied());

    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(
        &preimage
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>(),
    )
    .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}

pub fn enforce_claim_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &CeClaimVar,
    expected: &CeClaim<Commitment, F, K>,
    label: &str,
) -> Result<(), SynthesisError> {
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "c_data"),
        &actual.c_data,
        &expected.c.data,
        &format!("{label}_c_data"),
    )?;
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "x"),
        &actual.x,
        expected.X.as_slice(),
        &format!("{label}_x"),
    )?;
    enforce_k_slice_eq_native(&mut cs.namespace(|| "r"), &actual.r, &expected.r, &format!("{label}_r"))?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "s_col"),
        &actual.s_col,
        &expected.s_col,
        &format!("{label}_s_col"),
    )?;
    if actual.y_ring.len() != expected.y_ring.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (row_idx, (actual_row, expected_row)) in actual.y_ring.iter().zip(expected.y_ring.iter()).enumerate() {
        enforce_k_slice_eq_native(
            &mut cs.namespace(|| format!("y_ring_{row_idx}")),
            actual_row,
            expected_row,
            &format!("{label}_y_ring_{row_idx}"),
        )?;
    }
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "ct"),
        &actual.ct,
        &expected.ct,
        &format!("{label}_ct"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "aux_openings"),
        &actual.aux_openings,
        &expected.aux_openings,
        &format!("{label}_aux_openings"),
    )?;
    enforce_k_slice_eq_native(
        &mut cs.namespace(|| "y_zcol"),
        &actual.y_zcol,
        &expected.y_zcol,
        &format!("{label}_y_zcol"),
    )?;
    enforce_f_slice_eq_native(
        &mut cs.namespace(|| "c_step_coords"),
        &actual.c_step_coords,
        &expected.c_step_coords,
        &format!("{label}_c_step_coords"),
    )?;
    enforce_packed_bytes_eq_native(
        &mut cs.namespace(|| "fold_digest"),
        &actual.fold_digest_encoding,
        &expected.fold_digest,
        &format!("{label}_fold_digest"),
    )?;
    if actual.m_in != expected.m_in || actual.u_offset != expected.u_offset || actual.u_len != expected.u_len {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(())
}

pub fn packed_bytes_field_values(bytes: &[u8]) -> Vec<SpartanF> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_LIMB));
    out.push(SpartanF::from_canonical_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(SpartanF::from_canonical_u64(u64::from_le_bytes(limb)));
    }
    out
}

fn alloc_f_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[F],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || {
                Ok(SpartanF::from_canonical_u64(value.as_canonical_u64()))
            })
        })
        .collect()
}

fn alloc_k_slice<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    values: &[K],
    label: &str,
) -> Result<Vec<KNumVar>, SynthesisError> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| alloc_k(cs, Some(KNum::from_neo_k(*value)), &format!("{label}_{idx}")))
        .collect()
}

fn extend_packed_bytes_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    bytes: &[u8],
    label: &str,
) -> Result<(), SynthesisError> {
    const BYTES_PER_LIMB: usize = 7;
    dst.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(bytes.len() as u64),
        &format!("{label}_len"),
    )?);
    for (idx, chunk) in bytes.chunks(BYTES_PER_LIMB).enumerate() {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(alloc_constant(
            cs,
            SpartanF::from_canonical_u64(u64::from_le_bytes(limb)),
            &format!("{label}_limb_{idx}"),
        )?);
    }
    Ok(())
}

fn alloc_packed_bytes_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bytes: &[u8],
    label: &str,
) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
    let mut out = Vec::new();
    extend_packed_bytes_as_fields(cs, &mut out, bytes, label)?;
    Ok(out)
}

fn enforce_packed_bytes_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>],
    expected_bytes: &[u8],
    label: &str,
) -> Result<(), SynthesisError> {
    let expected = packed_bytes_field_values(expected_bytes);
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}_eq"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (*expected, CS::one()),
        );
    }
    Ok(())
}

fn extend_f_slice_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    values: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    dst.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(values.len() as u64),
        &format!("{label}_len"),
    )?);
    dst.extend(values.iter().cloned());
    Ok(())
}

fn extend_f_slice_values(dst: &mut Vec<SpartanF>, values: &[F]) {
    dst.push(SpartanF::from_canonical_u64(values.len() as u64));
    dst.extend(
        values
            .iter()
            .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    );
}

fn enforce_f_slice_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>],
    expected: &[F],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        cs.enforce(
            || format!("{label}_{idx}_eq"),
            |lc| lc + actual.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (SpartanF::from_canonical_u64(expected.as_canonical_u64()), CS::one()),
        );
    }
    Ok(())
}

fn extend_k_slice_as_fields<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    dst: &mut Vec<AllocatedNum<SpartanF>>,
    values: &[KNumVar],
    native_values: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    if values.len() != native_values.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    dst.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(values.len() as u64),
        &format!("{label}_len"),
    )?);
    let coeff_len = native_values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    dst.push(alloc_constant(
        cs,
        SpartanF::from_canonical_u64(coeff_len as u64),
        &format!("{label}_coeff_len"),
    )?);
    for (idx, (value, native_value)) in values.iter().zip(native_values.iter()).enumerate() {
        let coeffs = native_value.as_coeffs();
        dst.push(copy_allocated(
            cs,
            value.c0,
            SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
            &format!("{label}_{idx}_c0"),
        )?);
        dst.push(copy_allocated(
            cs,
            value.c1,
            SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
            &format!("{label}_{idx}_c1"),
        )?);
    }
    Ok(())
}

fn extend_k_slice_values(dst: &mut Vec<SpartanF>, values: &[K]) {
    dst.push(SpartanF::from_canonical_u64(values.len() as u64));
    let coeff_len = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    dst.push(SpartanF::from_canonical_u64(coeff_len as u64));
    for value in values {
        let coeffs = value.as_coeffs();
        dst.push(SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()));
        dst.push(SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()));
    }
}

fn enforce_k_slice_eq_native<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[KNumVar],
    expected: &[K],
    label: &str,
) -> Result<(), SynthesisError> {
    if actual.len() != expected.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let expected = KNum::from_neo_k(*expected);
        cs.enforce(
            || format!("{label}_{idx}_c0_eq"),
            |lc| lc + actual.c0,
            |lc| lc + CS::one(),
            |lc| lc + (expected.c0, CS::one()),
        );
        cs.enforce(
            || format!("{label}_{idx}_c1_eq"),
            |lc| lc + actual.c1,
            |lc| lc + CS::one(),
            |lc| lc + (expected.c1, CS::one()),
        );
    }
    Ok(())
}

fn alloc_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}

fn copy_allocated<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    variable: bellpepper_core::Variable,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_eq"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + variable,
    );
    Ok(out)
}
