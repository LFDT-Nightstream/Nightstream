//! Neo-Fold: High-level API for CCS folding
//!
//! This crate provides a convenient API for the CCS reduction protocol,
//! re-exporting functionality from the neo-reductions crate.

#![allow(non_snake_case)]

// Re-export everything from neo-reductions
pub use neo_reductions::{
    error, optimized_engine, paper_exact_engine, pi_ccs, pi_ccs_paper_exact, pi_ccs_prove, pi_ccs_prove_simple,
    pi_ccs_verify, sumcheck, CcsOracle, Challenges, PiCcsError, PiCcsProof,
};

// Route A: Split CCS prover for batched sum-check
pub use neo_reductions::{finalize_ccs_after_batch, prepare_ccs_for_batch, CcsBatchContext};

// Ergonomic per-step session API layered on top of the coordinator
pub mod session;

// Memory sidecar helpers (Route A integration plumbing)
pub mod memory_sidecar;

// Finalization hooks for shard obligations
pub mod finalize;

// Shard-level folding (CPU + Memory Sidecar)
pub mod shard;

mod shard_proof_types;
