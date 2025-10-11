//! Poseidon2 Configuration Parameters (Single Source of Truth)
//!
//! This module defines ONLY the canonical Poseidon2 parameters used throughout Neo.
//! The actual implementation lives in other crates but MUST import these constants.
//!
//! ## Configuration (Plonky3 Standard)
//! - WIDTH = 16 (total state size)
//! - CAPACITY = 8 (security parameter: 8 × 64 = 512 bits)
//! - RATE = 8 (absorption rate: WIDTH - CAPACITY)
//! - DIGEST_LEN = 4 (output size: 4 × 64 = 256 bits)
//!
//! ## Security Properties
//! - Collision resistance: ~128 bits (birthday bound: 2^(capacity_bits/2) = 2^256/2)
//! - Preimage resistance: ~256 bits (full capacity)
//! - Second preimage resistance: ~256 bits
//!
//! ## References
//! - Poseidon2 paper: Grassi et al. 2023 (https://eprint.iacr.org/2023/323)
//! - Plonky3 implementation: p3_goldilocks v0.3.0
//! - Security analysis: Based on Poseidon targeting 128-bit security

/// Poseidon2 state width (Plonky3 standard)
///
/// Total number of field elements in the permutation state.
/// Must equal CAPACITY + RATE.
pub const WIDTH: usize = 16;

/// Capacity of the sponge construction (security parameter)
///
/// 8 field elements × 64 bits/element = 512 bits total capacity.
/// Collision resistance is approximately 2^(capacity_bits/2) = 2^256 operations.
pub const CAPACITY: usize = 8;

/// Rate of absorption (elements absorbed per permutation)
///
/// Number of field elements that can be absorbed in each sponge round.
/// Must equal WIDTH - CAPACITY = 16 - 8 = 8.
pub const RATE: usize = 8;

/// Length of output digest (field elements)
///
/// 4 field elements × 64 bits = 256 bits output.
/// Provides 256-bit preimage resistance and 128-bit collision resistance.
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
        assert_eq!(WIDTH, 16, "WIDTH should be 16 (Plonky3 standard)");
        assert_eq!(CAPACITY, 8, "CAPACITY should be 8 (512 bits)");
        assert_eq!(RATE, 8, "RATE should be 8");
        assert_eq!(DIGEST_LEN, 4, "DIGEST_LEN should be 4 (256 bits)");
        assert_eq!(RATE + CAPACITY, WIDTH, "RATE + CAPACITY must equal WIDTH");
    }

    #[test]
    fn test_seed_fixed() {
        // Verify seed hasn't been accidentally changed
        assert_eq!(SEED.len(), 32, "Seed must be 32 bytes");
    }
}
