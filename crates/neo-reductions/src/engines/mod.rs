//! Π_CCS, Π_RLC, and Π_DEC engines. Owns the optimized and paper-exact
//! backends, the cross-checking wrapper, and the shared protocol and receipt
//! modules.

#![allow(non_snake_case)]

// Shared utilities for all engines
pub mod utils;

// Engine implementation modules
pub mod optimized_engine;
pub mod pi_ccs;
pub mod pi_ccs_execution_receipt;
pub mod pi_ccs_joint;
pub mod pi_ccs_joint_protocol;
pub mod pi_ccs_protocol;
pub mod pi_rlc_dec;

#[cfg(feature = "paper-exact")]
pub mod paper_exact_engine;

#[cfg(feature = "paper-exact")]
pub mod crosscheck_engine;

// Re-export the trait and implementations from pi_ccs
pub use pi_ccs::{OptimizedEngine, PiCcsEngine, PiCcsProof};

#[cfg(feature = "paper-exact")]
pub use pi_ccs::PaperExactEngine;

#[cfg(feature = "paper-exact")]
pub use crosscheck_engine::CrossCheckEngine;
