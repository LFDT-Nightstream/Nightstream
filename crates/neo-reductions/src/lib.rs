//! SuperNeo reductions with one canonical rectangular-paper PiCCS protocol.
//!
//! The optimized engine and independent PaperExact engine share only neutral
//! protocol messages and transcript flow. Their polynomial evaluators are
//! separate so tests can require exact proof-byte equality.

#![allow(non_snake_case)]

// Public modules
pub mod api; // public API for Π_CCS folding and RLC/DEC operations
pub mod common; // shared utilities and helper functions
pub mod engines; // internal engine trait + wrappers (includes optimized_engine, paper_exact_engine, crosscheck_engine)
pub mod error;
pub mod sumcheck;
pub mod superneo_eval; // transformed-matrix evaluators for SuperNeo migration
                       // Re-export RLC/DEC from engines for a stable path
pub use engines::pi_rlc_dec;

// Re-export engine modules for convenience
pub use engines::optimized_engine;
#[cfg(feature = "paper-exact")]
pub use engines::paper_exact_engine;

// Re-exports for convenience
pub use api as pi_ccs; // main public API
#[cfg(feature = "paper-exact")]
pub use engines::paper_exact_engine as pi_ccs_paper_exact;

// Re-export commonly used types
pub use engines::optimized_engine::{pi_ccs_prove, pi_ccs_prove_simple, pi_ccs_verify, Challenges, PiCcsProof};
pub use engines::pi_ccs_execution_receipt::{
    verify_and_export_pi_ccs_receipt, PiCcsCanonicalStatement, PiCcsExecutionProof, PiCcsExecutionReceipt,
    PiCcsReceiptK,
};

pub use error::PiCcsError;

// Re-export common utilities
pub use common::{
    rot_rhos_from_mats, rot_rhos_to_mats, sample_rot_rhos_n, sample_rot_rhos_n_typed, split_b_matrix_k,
    split_b_matrix_k_with_nonzero_flags, RotRho, RotRing,
};
