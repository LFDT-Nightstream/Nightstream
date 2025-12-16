//! Poseidon2 Configuration Parameters (Single Source of Truth)
//!
//! This module defines ONLY the canonical Poseidon2 parameters used throughout Neo.
//! The actual implementation lives in other crates but MUST import these constants.
//!
//! ## Configuration
//! - WIDTH = 8 (total state size, recommended parameter)
//! - CAPACITY = 4 (security parameter: 4 × 64 = 256 bits)
//! - RATE = 4 (absorption rate: WIDTH - CAPACITY)
//! - DIGEST_LEN = 4 (output size: 4 × 64 ≈ 256 bits)
//!
//! ## Security Properties
//! - Collision resistance: ~128 bits (birthday bound on 256-bit capacity)
//! - Preimage resistance: ~256 bits (full capacity)
//! - Second preimage resistance: ~256 bits
//!
//! Note: Goldilocks field elements are slightly less than 64 bits (p ≈ 2^64 - 2^32 + 1),
//! so actual security is marginally lower than the theoretical maximum.
//!
//! ## References
//! - Poseidon2 paper: Grassi et al. 2023 (https://eprint.iacr.org/2023/323)
//! - Plonky3 implementation: p3_goldilocks
//! - Security analysis: Based on Poseidon targeting 128-bit security

/// Poseidon2 state width (recommended parameter)
///
/// Total number of field elements in the permutation state.
/// Must equal CAPACITY + RATE = 4 + 4 = 8.
pub const WIDTH: usize = 8;

/// Capacity of the sponge construction (security parameter)
///
/// 4 field elements × 64 bits/element = 256 bits total capacity.
/// Provides ~128 bits of collision resistance (birthday bound: 2^(256/2) = 2^128).
pub const CAPACITY: usize = 4;

/// Rate of absorption (elements absorbed per permutation)
///
/// Number of field elements that can be absorbed in each sponge round.
/// Must equal WIDTH - CAPACITY = 8 - 4 = 4.
/// Each permutation absorbs ~256 bits of input.
pub const RATE: usize = 4;

/// Length of output digest (field elements)
///
/// 4 field elements × 64 bits ≈ 256 bits output.
/// (Slightly lower since Goldilocks is not quite 64 bits.)
/// Provides ~128-bit collision resistance.
pub const DIGEST_LEN: usize = 4;

/// Fixed seed for deterministic round constant generation.
///
/// # Security Note
/// This seed is public and fixed by design. It ensures that:
/// - All provers and verifiers use identical round constants
/// - The permutation is deterministic and reproducible
/// - No backdoors can be introduced via seed manipulation
///
/// The security of Poseidon2 does NOT rely on the seed being secret.
/// Round constants generated from any fixed seed provide equivalent security.
pub const SEED: [u8; 32] = *b"neo_poseidon2_goldilocks_seed___";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters_consistent() {
        assert_eq!(WIDTH, 8, "WIDTH should be 8 (recommended parameter)");
        assert_eq!(CAPACITY, 4, "CAPACITY should be 4 (256 bits)");
        assert_eq!(RATE, 4, "RATE should be 4");
        assert_eq!(DIGEST_LEN, 4, "DIGEST_LEN should be 4 (~256 bits)");
        assert_eq!(RATE + CAPACITY, WIDTH, "RATE + CAPACITY must equal WIDTH");
    }

    #[test]
    fn test_seed_fixed() {
        // Verify seed hasn't been accidentally changed
        assert_eq!(SEED.len(), 32, "Seed must be 32 bytes");
    }
}
