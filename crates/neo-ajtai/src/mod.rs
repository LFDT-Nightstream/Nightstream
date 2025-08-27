//! Ajtai Matrix Commitment Scheme
//! 
//! This module implements Neo's core Ajtai-based lattice commitment scheme
//! with S-module homomorphism and pay-per-bit embedding as described in
//! Nguyen & Setty 2025 (ePrint 2025/294).
//!
//! The module is organized as:
//! - `commit`: Core commitment operations (Setup, Commit, Verify)
//! - `embedding`: Pay-per-bit decomposition and bit-sparse operations
//! - `rot`: Rotation-matrix ring S and fast rot(a)·v operations

pub mod commit;
pub mod embedding;
pub mod rot;

// Re-export main types and functions
pub use commit::{AjtaiCommitter, NeoParams};
pub use embedding::{decomp_b, split_b, pay_per_bit_cost};
pub use rot::{RotationRing, ChallengeSet};

// Re-export parameter presets aligned with Neo paper §6
pub use commit::{GOLDILOCKS_PARAMS, SECURE_PARAMS, TOY_PARAMS};
