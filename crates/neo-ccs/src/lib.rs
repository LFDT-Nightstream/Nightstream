#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! CCS frontend for Neo: structures, relations (CCS/CE), and row-wise checks.
//!
//! Implements the MUST and SHOULD in the Neo spec, matching the paper's §4.1 relations
//! (CCS & CE), the row-wise CCS check, and the consistency equalities used by Π_CCS/Π_RLC/Π_DEC.

// Audit-ready core modules
/// Production cryptographic primitives (Poseidon2 implementation).
pub mod crypto;
/// Error types for CCS operations.
pub mod error;
/// Cryptographic gadgets for CCS circuits.
pub mod gadgets;
/// Matrix types and operations.
pub mod matrix;
/// Polynomial types and evaluation.
pub mod poly;
/// R1CS to CCS conversion utility (kept: used in tests).
pub mod r1cs;
/// Core CCS relations and consistency checks.
pub mod relations;
/// Sparse matrix types (CSC) and caches.
pub mod sparse;
/// Traits for commitment scheme integration.
pub mod traits;
/// Utility functions for tensor products and matrix operations.
pub mod utils;

// Re-export core types
pub use error::{CcsError, DimMismatch, RelationError};
pub use matrix::{CsrMatrix, Mat, MatRef};
pub use poly::{SparsePoly, Term};
pub use r1cs::r1cs_to_ccs;
pub use sparse::{CcsMatrix, CscMat, SparseCache};

// Cancellation-resistant direct sum (recommended for production)
pub use utils::direct_sum_transcript_mixed;
// Main CCS types and functions (audit-ready)
pub use relations::{
    build_superneo_ring_forms, check_ccs_claim_opening, check_ccs_rowwise_relaxed, check_ccs_rowwise_zero,
    check_ce_consistency, CcsClaim, CcsStructure, CcsWitness, CeClaim, CeWitness,
};
pub use traits::SModuleHomomorphism;
pub use utils::{direct_sum, direct_sum_mixed, mat_vec_mul_ff, mat_vec_mul_fk, tensor_point, validate_power_of_two};
