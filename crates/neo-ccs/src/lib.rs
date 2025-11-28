#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! CCS frontend for Neo: structures, relations (MCS/ME), and row-wise checks.
//!
//! Implements the MUST and SHOULD in the Neo spec, matching the paper's §4.1 relations
//! (MCS & ME), the row-wise CCS check, and the consistency equalities used by Π_CCS/Π_RLC/Π_DEC.

// Audit-ready core modules
/// Error types for CCS operations.
pub mod error;
/// Production cryptographic primitives (Poseidon2 implementation).
pub mod crypto;
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
/// Traits for commitment scheme integration.
pub mod traits;
/// Utility functions for tensor products and matrix operations.
pub mod utils;

// Tests are now in tests/ccs_property_tests.rs as integration tests

// Re-export core types
pub use error::{CcsError, DimMismatch, RelationError};
pub use matrix::{Mat, MatRef, CsrMatrix};
pub use poly::{SparsePoly, Term};
pub use r1cs::r1cs_to_ccs;

// 🔒 SECURITY FIX: Use the cancellation-resistant implementation from utils
pub use utils::direct_sum_transcript_mixed;
// Main CCS types and functions (audit-ready)
pub use relations::{
    CcsStructure, McsInstance, McsWitness, MeInstance, MeWitness,
    check_mcs_opening, check_me_consistency, check_ccs_rowwise_zero,
    check_ccs_rowwise_relaxed,
};
pub use traits::SModuleHomomorphism;
pub use utils::{tensor_point, mat_vec_mul_fk, mat_vec_mul_ff, validate_power_of_two, direct_sum, direct_sum_mixed};
