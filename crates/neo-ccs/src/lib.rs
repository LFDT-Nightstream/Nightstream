#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! CCS frontend for Neo: structures, relations (MCS/ME), and row-wise checks.
//!
//! Implements the MUST and SHOULD in the Neo spec, matching the paper's §4.1 relations
//! (MCS & ME), the row-wise CCS check, and the consistency equalities used by Π_CCS/Π_RLC/Π_DEC.

// Audit-ready core modules
/// Error types for CCS operations.
pub mod error;
/// Matrix types and operations.
pub mod matrix;
/// Polynomial types and evaluation.
pub mod poly;
/// R1CS to CCS conversion utility (kept: used in tests).
pub mod r1cs;
/// Core CCS relations and consistency checks.
pub mod relations;
/// Traits for commitment scheme integration.
pub mod traits;
/// Utility functions for tensor products and matrix operations.
pub mod utils;

// Tests module
#[cfg(test)]
mod tests;

// Re-export core types
pub use error::{CcsError, DimMismatch, RelationError};
pub use matrix::{Mat, MatRef};
pub use poly::{SparsePoly, Term};
pub use r1cs::r1cs_to_ccs;
// Main CCS types and functions (audit-ready)
pub use relations::{
    CcsStructure, McsInstance, McsWitness, MeInstance, MeWitness,
    check_mcs_opening, check_me_consistency, check_ccs_rowwise_zero,
    check_ccs_rowwise_relaxed,
};
pub use traits::SModuleHomomorphism;
pub use utils::{tensor_point, mat_vec_mul_fk, mat_vec_mul_ff, validate_power_of_two};

// ===== DEPRECATED LEGACY BRIDGE TYPES =====
// These are kept temporarily for bridge compatibility but are NOT audit-ready.
// They will be removed in a future version once the bridge is modernized.

/// Legacy Matrix Evaluation instance - for bridge compatibility only
/// 
/// ⚠️ **DEPRECATED & NOT AUDIT-READY**: Use `relations::MeInstance<C, F, K>` instead.
/// This type exists only to keep the spartan-bridge compiling during modernization.
#[deprecated(since = "0.1.0", note = "Use relations::MeInstance<C, F, K> instead")]
#[derive(Clone, Debug)]
pub struct MEInstance {
    /// Ajtai commitment coordinates c ∈ F_q^{d×κ}
    pub c_coords: Vec<neo_math::F>, 
    /// ME outputs y_j = ⟨M_j^T r^b, Z⟩ for each matrix j
    pub y_outputs: Vec<neo_math::F>, 
    /// Public random point r^b from sum-check 
    pub r_point: Vec<neo_math::F>, 
    /// Base parameter for range constraints
    pub base_b: u64,
    /// Transcript header digest for binding to neo-fold
    pub header_digest: [u8; 32],
}

/// Legacy Matrix Evaluation witness - for bridge compatibility only
/// 
/// ⚠️ **DEPRECATED & NOT AUDIT-READY**: Use `relations::MeWitness<F>` instead.
/// This type exists only to keep the spartan-bridge compiling during modernization.
#[deprecated(since = "0.1.0", note = "Use relations::MeWitness<F> instead")]
#[derive(Clone, Debug)]
pub struct MEWitness {
    /// Witness digits Z in base b: |Z|_∞ < b
    pub z_digits: Vec<i64>, 
    /// Weight vectors v_j = M_j^T r^b for computing ⟨v_j, Z⟩ = y_j
    pub weight_vectors: Vec<Vec<neo_math::F>>, 
    /// Optional Ajtai linear map rows L for c = L(Z) verification
    pub ajtai_rows: Option<Vec<Vec<neo_math::F>>>, 
}

