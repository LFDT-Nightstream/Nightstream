//! Builds canonical zero accumulator carries for generic direct CCS programs.
//!
//! This mirrors the RV32IM zero-carry seed shape, but derives all sizes from
//! the direct CCS structure and public input layout instead of VM constants.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::build_dims_and_policy;
use p3_field::PrimeCharacteristicRing;

use super::DirectCcsFPrimeSnarkError;
use crate::proof::Carry;

pub(crate) fn build_direct_canonical_zero_carry(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    public_input_len: usize,
) -> Result<Carry, DirectCcsFPrimeSnarkError> {
    if params.k_rho == 0 {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct CCS canonical zero carry requires k_rho > 0".into(),
        ));
    }
    if public_input_len > structure.m {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct CCS canonical zero carry public input len {public_input_len} exceeds CCS column count {}",
            structure.m
        )));
    }
    let dims =
        build_dims_and_policy(params, structure).map_err(|err| DirectCcsFPrimeSnarkError::Input(err.to_string()))?;
    let claim = direct_zero_claim(
        params,
        structure,
        public_input_len,
        dims.ell_n,
        dims.ell_m,
        1usize << dims.ell_d,
    );
    let witness = Mat::zero(D, structure.m.div_ceil(D), F::ZERO);
    let claim_count = params.k_rho as usize;
    Ok(Carry {
        claims: vec![claim; claim_count],
        witnesses: vec![witness; claim_count],
    })
}

fn direct_zero_claim(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    public_input_len: usize,
    r_len: usize,
    s_col_len: usize,
    padded_ring_len: usize,
) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::zero(D, public_input_len, F::ZERO),
        r: vec![K::ZERO; r_len],
        s_col: vec![K::ZERO; s_col_len],
        y_ring: vec![vec![K::ZERO; padded_ring_len]; structure.matrices.len()],
        ct: vec![K::ZERO; structure.matrices.len()],
        aux_openings: Vec::new(),
        y_zcol: vec![K::ZERO; padded_ring_len],
        m_in: public_input_len,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}
