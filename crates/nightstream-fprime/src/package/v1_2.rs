//! Exact SuperNeo v1.2 PiCCS output boundary in the Lean-emitted layout.
//!
//! This module reads package-owned segments. It does not define PiCCS
//! semantics, choose offsets, or merge the Pad and CCS evaluation families.

/// The Nightstream Goldilocks decomposition profile: base `b`, digit count
/// `k_rho`, and norm bound `B = b^k_rho`.
pub(crate) const DECOMPOSITION_BASE: u64 = 2;
pub(crate) const DECOMPOSITION_K_RHO: u32 = 16;
pub(crate) const DECOMPOSITION_NORM_BOUND: u64 = DECOMPOSITION_BASE.pow(DECOMPOSITION_K_RHO);

pub(crate) const PI_CCS_V1_2_COEFFICIENT_COUNT: usize = 54;
pub(crate) const PI_CCS_V1_2_MATRIX_COUNT: usize = 4;
pub const PI_CCS_V1_2_ROUND_COUNT: usize = 28;
pub(crate) const PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT: usize = 9;
