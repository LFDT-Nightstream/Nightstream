//! Production-ready cryptographic primitives for Neo
//!
//! This module provides production implementations of cryptographic primitives
//! used throughout the Neo protocol, specifically:
//! - Poseidon2 hash function (off-circuit only for transcripts and digests)
//!
//! ## Production Architecture
//!
//! Neo uses a **public-ρ embedded verifier** approach for IVC:
//! - Fiat-Shamir challenges (ρ) are computed off-circuit using Poseidon2 transcripts
//! - The embedded verifier circuit only proves multiplication/linearity constraints
//! - The verifier recomputes the same ρ to verify proof soundness
//!
//! This approach is cryptographically sound, efficient, and avoids complex in-circuit hashing.

// Production Poseidon2 off-circuit hasher over Goldilocks (p3 0.3.0 API)
pub mod poseidon2_goldilocks;