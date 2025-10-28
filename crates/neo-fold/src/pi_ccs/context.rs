//! Context module: Computed dimensions and security policy checks
//! 
//! This module provides helpers for computing the reduction parameters
//! from Section 4.3 of the Neo paper, including:
//! - ℓ_d = log(d) where d is the Ajtai dimension
//! - ℓ_n = log(n) where n is the CCS row count
//! - ℓ = log(dn) for the full hypercube in sum-check
//! - d_sc = sum-check degree bound
//! - Extension field security policy validation

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ccs::CcsStructure;
use neo_params::NeoParams;
use neo_math::{F, D};

/// Computed dimensions for the CCS reduction
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    /// log(d) - Ajtai dimension (row dimension of Z)
    pub ell_d: usize,
    /// log(n) - CCS row dimension
    pub ell_n: usize,
    /// log(dn) - full hypercube dimension for sum-check
    pub ell: usize,
    /// Sum-check degree bound: max(F+eq, NC+eq, Eval+eq)
    pub d_sc: usize,
}

/// Extension field security policy check result
#[derive(Debug, Clone)]
pub struct ExtensionPolicy {
    pub s_supported: u64,
    pub lambda: u32,
    pub slack_bits: i32,
}

/// Build dimensions and validate extension field security policy
/// 
/// # Paper Reference
/// Section 4.3: Parameters
/// - Sum-check over {0,1}^{log(dn)} hypercube
/// - Degree bound accounts for F, NC, and Eval polynomials gated by eq
pub fn build_dims_and_policy(
    params: &NeoParams,
    s: &CcsStructure<F>,
) -> Result<Dims, PiCcsError> {
    if s.n == 0 {
        return Err(PiCcsError::InvalidInput("n=0 not allowed".into()));
    }

    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;
    
    let ell = ell_d + ell_n;

    let d_sc = core::cmp::max(
        s.max_degree() as usize + 1,
        core::cmp::max(2, 2 * (params.b as usize)),
    );

    let ext = params.extension_check(ell as u32, d_sc as u32)
        .map_err(|e| PiCcsError::ExtensionPolicyFailed(e.to_string()))?;

    if ext.slack_bits < 0 {
        return Err(PiCcsError::ExtensionPolicyFailed(format!(
            "Insufficient security slack: {} bits (need ≥ 0 for {}-bit security)",
            ext.slack_bits, params.lambda
        )));
    }

    Ok(Dims { ell_d, ell_n, ell, d_sc })
}

