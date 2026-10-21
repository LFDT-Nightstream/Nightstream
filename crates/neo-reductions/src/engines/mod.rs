//! Internal engine modules used by the public facade: the optimized and
//! paper-exact prover/verifier backends and the cross-checking wrapper.

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
