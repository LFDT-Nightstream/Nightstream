//! Neo-Spartan-Bridge: Integration layer between Neo folding and Spartan2 SNARK
//!
//! This crate provides a modular architecture for proving Neo FoldRun executions
//! using Spartan2, with pluggable PCS backends.
//!
//! ## Architecture
//!
//! 1. **Engine trait**: Defines the backend (field, group, PCS, transcript, Z-layout)
//! 2. **K-field gadgets**: Represent K (degree-2 extension) as 2 limbs over F
//! 3. **FoldRun circuit**: Synthesizes R1CS constraints for:
//!    - Π-CCS terminal identity verification
//!    - Sumcheck round checks
//!    - RLC equalities
//!    - DEC equalities
//!    - Accumulator chaining
//! 4. **Prove/Verify API**: High-level interface to Spartan2

#![allow(non_snake_case)]

pub mod api;
pub mod circuit;
pub mod engine;
pub mod error;
pub mod gadgets;

// Re-export commonly used types
pub use error::SpartanBridgeError;

/// The fixed circuit field for Spartan2 integration.
/// This is Spartan2's Goldilocks field, matching Neo's field modulus.
pub type CircuitF = spartan2::provider::goldi::F;

// Engine is experimental and gated
#[cfg(feature = "experimental-engine")]
pub use engine::{BridgeEngine, HashMleEngine, ZPolyLayout};

pub use api::{prove_fold_run, verify_fold_run};
