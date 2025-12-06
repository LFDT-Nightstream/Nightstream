//! Neo-Reductions: CCS folding engine implementing Π_CCS protocol
//!
//! This crate contains both the optimized engine and paper-exact reference implementations
//! of the CCS reduction protocol described in the Neo paper.

#![allow(non_snake_case)]

// Public modules
pub mod api; // public API for Π_CCS folding and RLC/DEC operations
pub mod common; // shared utilities and helper functions
pub mod engines; // internal engine trait + wrappers (includes optimized_engine, paper_exact_engine, crosscheck_engine)
pub mod error;
pub mod sumcheck;
// Re-export RLC/DEC from engines for a stable path
pub use engines::pi_rlc_dec;

// Re-export engine modules for convenience
pub use engines::optimized_engine;
pub use engines::paper_exact_engine;

// Re-exports for convenience
pub use api as pi_ccs; // main public API
pub use engines::paper_exact_engine as pi_ccs_paper_exact;

// Re-export commonly used types
pub use engines::optimized_engine::{
    pi_ccs_prove, pi_ccs_prove_simple, pi_ccs_verify, Challenges, GenericCcsOracle, PiCcsProof,
};
pub use error::PiCcsError;

// Re-export common utilities
pub use common::{sample_rot_rhos, split_b_matrix_k, RotRing};
