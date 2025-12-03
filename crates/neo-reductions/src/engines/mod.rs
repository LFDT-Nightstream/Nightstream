//! Internal engine modules used by the public facade.
//!
//! This keeps the external API stable while allowing multiple
//! prover/verification backends (optimized, paper-exact, and
//! an optional cross-checking wrapper used during development).

#![allow(non_snake_case)]

// Shared utilities for all engines
pub mod utils;

// Engine implementation modules
pub mod optimized_engine;
pub mod pi_rlc_dec;
pub mod pi_ccs;
pub mod mojo_gpu_engine;
pub use mojo_gpu_engine::engine::MojoGpuEngine;

#[cfg(feature = "paper-exact")]
pub mod paper_exact_engine;

#[cfg(feature = "paper-exact")]
pub mod crosscheck_engine;

// Re-export the trait and implementations from pi_ccs
pub use pi_ccs::{PiCcsEngine, OptimizedEngine, PiCcsProof};

#[cfg(feature = "paper-exact")]
pub use pi_ccs::PaperExactEngine;

#[cfg(feature = "paper-exact")]
pub use crosscheck_engine::{CrossCheckEngine, CrosscheckCfg};
