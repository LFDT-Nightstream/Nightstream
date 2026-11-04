//! Poseidon2 Configuration Parameters (Single Source of Truth)
//!
//! This module defines ONLY the canonical Poseidon2 parameters used throughout Neo.
//! The actual implementation lives in other crates but MUST import these constants.
//!
//! ## References
//! - Poseidon2 paper: Grassi et al. 2023 (https://eprint.iacr.org/2023/323)

/// Plonky3 checks the 128-bit round numbers (8 full, 22 partial) for this width over Goldilocks.
pub const WIDTH: usize = 16;

pub const CAPACITY: usize = 4;

pub const RATE: usize = 12;

pub const DIGEST_LEN: usize = 4;

pub const SEED: [u8; 32] = *b"neo_poseidon2_goldilocks_seed___";
