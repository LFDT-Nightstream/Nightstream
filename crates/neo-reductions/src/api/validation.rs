//! Canonical shape checks for the public PiCCS, PiRLC, and PiDEC boundary.
//!
//! This module owns structural validation only. It does not own transcript
//! replay, algebraic acceptance, commitment binding, or protocol selection.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

use crate::error::PiCcsError;

#[inline]
pub(crate) fn ensure_superneo_width(s: &CcsStructure<F>) -> Result<(), PiCcsError> {
    if s.m == 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "SuperNeo-only mode requires CCS width m > 0 (got m={})",
            s.m
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ell_n_for_ccs(s: &CcsStructure<F>) -> usize {
    s.n.max(crate::common::superneo_carrier_width(s.m))
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize
}

pub(crate) fn validate_mcs_claims(
    label: &str,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
) -> Result<(), PiCcsError> {
    for (idx, inst) in mcs_list.iter().enumerate() {
        if inst.m_in > s.m {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}].m_in={} exceeds CCS width m={}",
                inst.m_in, s.m
            )));
        }
        if inst.m_in % D != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}].m_in={} is not a whole number of degree-{D} ring elements",
                inst.m_in
            )));
        }
        if inst.x.len() != inst.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}].x.len()={} does not match m_in={}",
                inst.x.len(),
                inst.m_in
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_mcs_witnesses(
    label: &str,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
) -> Result<(), PiCcsError> {
    for (idx, (inst, wit)) in mcs_list.iter().zip(mcs_witnesses.iter()).enumerate() {
        if wit.private_len(inst.m_in, s.m).is_none() {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: mcs_list[{idx}] private witness does not complete m_in={} to CCS width m={}",
                inst.m_in, s.m
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_ce_claim_shape(
    label: &str,
    s: &CcsStructure<F>,
    ce: &CeClaim<Cmt, F, K>,
) -> Result<(), PiCcsError> {
    if ce.m_in > s.m {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: m_in={} exceeds CCS width m={}",
            ce.m_in, s.m
        )));
    }
    if ce.m_in % D != 0 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: m_in={} is not a whole number of degree-{D} ring elements",
            ce.m_in
        )));
    }
    let x_cols = neo_ccs::superneo_public_x_cols(ce.m_in);
    if ce.X.rows() != D || ce.X.cols() != x_cols {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: X has shape {}x{}, expected {}x{}",
            ce.X.rows(),
            ce.X.cols(),
            D,
            x_cols
        )));
    }
    crate::engines::pi_ccs_protocol::validate_inactive_x_zero(label, ce)?;
    let ell_n = ell_n_for_ccs(s);
    if ce.r.len() != ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: r length mismatch (expected {ell_n}, got {})",
            ce.r.len()
        )));
    }
    let paper_count = s.t() + 1;
    if ce.y_ring.len() != paper_count {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: y_ring.len()={} must equal the identity-first paper count {}",
            ce.y_ring.len(),
            paper_count
        )));
    }
    if ce.ct.len() != ce.y_ring.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: ct.len()={} must equal y_ring.len()={}",
            ce.ct.len(),
            ce.y_ring.len()
        )));
    }
    let d_pad = D.next_power_of_two();
    for (j, row) in ce.y_ring.iter().enumerate() {
        if row.len() != d_pad {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: y_ring[{j}].len()={} must be the canonical padded length {d_pad}",
                row.len()
            )));
        }
        if row.iter().skip(D).any(|value| *value != K::ZERO) {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: y_ring[{j}] has nonzero padding"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_ce_claims_shape(
    label: &str,
    s: &CcsStructure<F>,
    claims: &[CeClaim<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    for (idx, claim) in claims.iter().enumerate() {
        validate_ce_claim_shape(&format!("{label}[{idx}]"), s, claim)?;
    }
    Ok(())
}

pub(crate) fn validate_pi_ccs_outputs(
    label: &str,
    s: &CcsStructure<F>,
    outputs: &[CeClaim<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    let d_pad = D.next_power_of_two();
    for (index, output) in outputs.iter().enumerate() {
        let owner = format!("{label}[{index}]");
        let matrix_count = s.t() + 1;
        if output.m_in > s.m || output.m_in % D != 0 {
            return Err(PiCcsError::InvalidInput(format!(
                "{owner}: m_in={} must not exceed m={} and must contain whole degree-{D} ring elements",
                output.m_in, s.m
            )));
        }
        let x_cols = neo_ccs::superneo_public_x_cols(output.m_in);
        if output.X.rows() != D || output.X.cols() != x_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "{owner}: X has shape {}x{}, expected {}x{}",
                output.X.rows(),
                output.X.cols(),
                D,
                x_cols
            )));
        }
        crate::engines::pi_ccs_protocol::validate_inactive_x_zero(&owner, output)?;
        let point_len = ell_n_for_ccs(s);
        if output.r.len() != point_len {
            return Err(PiCcsError::InvalidInput(format!(
                "{owner}: r length mismatch (expected {point_len}, got {})",
                output.r.len()
            )));
        }
        if output.y_ring.len() != matrix_count || output.ct.len() != matrix_count {
            return Err(PiCcsError::InvalidInput(format!(
                "{owner}: y_ring and ct must each have exactly {} entries",
                matrix_count
            )));
        }
        for (matrix_index, (row, constant_term)) in output.y_ring.iter().zip(&output.ct).enumerate() {
            if row.len() != d_pad {
                return Err(PiCcsError::InvalidInput(format!(
                    "{owner}: y_ring[{matrix_index}] must have canonical padded length {d_pad}, got {}",
                    row.len()
                )));
            }
            if row.first() != Some(constant_term) {
                return Err(PiCcsError::InvalidInput(format!(
                    "{owner}: ct[{matrix_index}] does not equal the y_ring constant term"
                )));
            }
            if row.iter().skip(D).any(|value| *value != K::ZERO) {
                return Err(PiCcsError::InvalidInput(format!(
                    "{owner}: y_ring[{matrix_index}] has nonzero padding"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_rlc_batch_compatibility(
    label: &str,
    params: &NeoParams,
    claims: &[CeClaim<Cmt, F, K>],
) -> Result<(), PiCcsError> {
    let Some(first) = claims.first() else {
        return Err(PiCcsError::InvalidInput(format!("{label}: empty inputs")));
    };
    for (index, claim) in claims.iter().enumerate() {
        if claim.m_in != first.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: m_in mismatch at input {index} (expected {}, got {})",
                first.m_in, claim.m_in
            )));
        }
        if claim.fold_digest != first.fold_digest {
            return Err(PiCcsError::InvalidInput(format!(
                "{label}: fold transcript mismatch at input {index}"
            )));
        }
    }
    let required = crate::common::min_k_rho_for_rlc_count(params, &crate::common::RotRing::goldilocks(), claims.len())?;
    if params.k_rho < required {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: {} inputs require k_rho >= {required}, got {}",
            claims.len(),
            params.k_rho
        )));
    }
    Ok(())
}

pub(crate) fn checked_superneo_d_pad(label: &str, ell_d: usize) -> Result<usize, PiCcsError> {
    let expected = D.next_power_of_two().trailing_zeros() as usize;
    if ell_d != expected {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: ell_d must be derived from D={D} (expected {expected}, got {ell_d})"
        )));
    }
    Ok(D.next_power_of_two())
}

pub(crate) fn validate_dec_boundary_inputs(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    z_split: &[Mat<F>],
    child_commitments: &[Cmt],
    ell_d: usize,
) -> Result<(), PiCcsError> {
    ensure_superneo_width(s)?;
    validate_ce_claim_shape("dec_parent", s, parent)?;
    if z_split.len() != child_commitments.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC child input mismatch: |Z_split|={} but |child_commitments|={}",
            z_split.len(),
            child_commitments.len()
        )));
    }
    if z_split.len() != params.k_rho as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC requires exactly k_rho={} child witnesses, got {}",
            params.k_rho,
            z_split.len()
        )));
    }
    checked_superneo_d_pad("DEC ell_d", ell_d)?;
    for (idx, z) in z_split.iter().enumerate() {
        crate::common::validate_packed_witness_nc_range(params, z, s.m, &format!("dec: Z_split[{idx}]"))?;
    }
    Ok(())
}

pub(crate) fn validate_dec_boundary_inputs_from_trusted_split(
    s: &CcsStructure<F>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, F, K>,
    z_split: &[Mat<F>],
    child_commitments: &[Cmt],
    ell_d: usize,
) -> Result<(), PiCcsError> {
    ensure_superneo_width(s)?;
    validate_ce_claim_shape("dec_parent", s, parent)?;
    if z_split.len() != child_commitments.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC child input mismatch: |Z_split|={} but |child_commitments|={}",
            z_split.len(),
            child_commitments.len()
        )));
    }
    if z_split.len() != params.k_rho as usize {
        return Err(PiCcsError::InvalidInput(format!(
            "DEC requires exactly k_rho={} child witnesses, got {}",
            params.k_rho,
            z_split.len()
        )));
    }
    checked_superneo_d_pad("DEC ell_d", ell_d)?;
    for (idx, z) in z_split.iter().enumerate() {
        crate::common::validate_superneo_witness_mat(z, s.m)
            .map_err(|e| PiCcsError::InvalidInput(format!("dec trusted split: Z_split[{idx}] shape failed: {e}")))?;
    }
    Ok(())
}
