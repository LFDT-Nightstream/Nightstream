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

// Ergonomic per-step session API layered on top of the coordinator
pub mod session;

// Runner for the `TestExport` JSON schema (shared by tests and wasm)
pub mod test_export;

// Memory sidecar helpers (Route A integration plumbing)
pub mod memory_sidecar;

// Finalization hooks for shard obligations
pub mod finalize;

// Shard-level folding (CPU + Memory Sidecar)
pub mod shard;

pub mod riscv_trace_shard;
mod rv64_ram_bridge;
pub mod rv64_trace_shard;

// Output binding integration
pub mod output_binding;

// Time-opening Stage-8 helpers (manifest/reduction/joint lane)
pub mod time_opening;

mod shard_proof_types;
