//! Neo-Spartan-Bridge: Integration layer between Neo folding and Spartan2 SNARK
//!
//! This crate provides a Bellpepper R1CS circuit that verifies Neo shard folding (Π‑CCS + Π‑RLC/Π‑DEC),
//! with optional Route‑A memory semantics and output binding, and an API for producing/verifying a
//! single Spartan2 proof for it.
//!
//! ## High-level pieces
//! - `circuit/`: `FoldRunCircuit` that enforces Π-CCS + Π-RLC/Π-DEC checks and binds accumulator digests.
//! - `gadgets/`: Poseidon2 transcript, K-field arithmetic, and sumcheck helpers.
//! - `api`: pinned `(pk,vk)` setup plus `prove_fold_run`/`verify_fold_run`.

#![allow(non_snake_case)]

pub mod api;
pub mod circuit;
pub mod error;
pub mod gadgets;
pub mod statement;

// Re-export commonly used types
pub use error::SpartanBridgeError;
pub use statement::SpartanShardStatement;

/// The fixed circuit field for Spartan2 integration.
/// This is Spartan2's Goldilocks field, matching Neo's field modulus.
pub type CircuitF = spartan2::provider::goldi::F;

pub use api::{
    compute_accumulator_digest_v2, prove_fold_run, setup_fold_run, setup_fold_run_shape, verify_fold_run, FoldRunShape,
    SpartanEngine, SpartanProverKey, SpartanSnark, SpartanVerifierKey,
};
