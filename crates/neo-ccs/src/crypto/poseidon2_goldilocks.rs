//! Production Poseidon2 Implementation over Goldilocks Field
//!
//! This module provides the Poseidon2 hash function implementation.
//! All parameters (WIDTH, CAPACITY, etc.) are imported from `neo-params` 
//! which serves as the single source of truth.
//!
//! ## Usage
//! 
//! ```ignore
//! use neo_ccs::crypto::poseidon2_goldilocks as p2;
//! 
//! // Hash field elements
//! let input = [Goldilocks::from_u64(42); 10];
//! let digest = p2::poseidon2_hash(&input);
//! 
//! // Hash bytes (packed efficiently)
//! let bytes = b"hello world";
//! let digest = p2::poseidon2_hash_packed_bytes(bytes);
//! ```

use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use once_cell::sync::Lazy;

// Import parameters from neo-params (single source of truth)
pub use neo_params::poseidon2_goldilocks::{WIDTH, CAPACITY, RATE, DIGEST_LEN, SEED};

/// Cached Poseidon2 permutation instance.
/// 
/// Computed once at first use and cached forever for performance.
/// Round constant generation is expensive (~milliseconds), so this avoids
/// recomputing them on every hash operation.
/// 
/// Uses parameters from `neo-params` for consistency across the codebase.
pub static PERM: Lazy<Poseidon2Goldilocks<{ WIDTH }>> = Lazy::new(|| {
    let mut rng = ChaCha8Rng::from_seed(SEED);
    Poseidon2Goldilocks::<{ WIDTH }>::new_from_rng_128(&mut rng)
});

/// Returns a reference to the cached Poseidon2 permutation.
/// 
/// This is the primary way to access the permutation. It ensures:
/// - Consistent parameters across the entire codebase
/// - Optimal performance (no recomputation of round constants)
/// - Thread-safe lazy initialization
pub fn permutation() -> &'static Poseidon2Goldilocks<{ WIDTH }> {
    &*PERM
}

/// Standard sponge construction with proper padding.
/// 
/// Implements the Poseidon2 sponge: absorb input → pad → squeeze output.
/// Padding: Add 1 to first state element, then permute.
/// 
/// # Security
/// - Input is absorbed in chunks of RATE (8 elements)
/// - Each chunk is XORed into state and permuted
/// - Final padding prevents length-extension attacks
/// - Output squeezed from first DIGEST_LEN elements
pub fn poseidon2_hash(input: &[Goldilocks]) -> [Goldilocks; DIGEST_LEN] {
    let perm = permutation();
    let mut state = [Goldilocks::ZERO; WIDTH];

    // Absorb phase: XOR input into state at rate, permute after each chunk
    for chunk in input.chunks(RATE) {
        for (i, &x) in chunk.iter().enumerate() {
            state[i] = state[i] + x;
        }
        state = perm.permute(state);
    }
    
    // Padding: Add 1 to first position and final permute
    state[0] = state[0] + Goldilocks::ONE;
    state = perm.permute(state);

    // Squeeze phase: Extract first DIGEST_LEN elements as output
    let mut out = [Goldilocks::ZERO; DIGEST_LEN];
    out.copy_from_slice(&state[..DIGEST_LEN]);
    out
}

/// Hash raw bytes by converting each byte to a field element.
/// 
/// WARNING: This is inefficient (1 field element per byte).
/// Use `poseidon2_hash_packed_bytes` for better performance.
/// 
/// Useful for compatibility with byte-oriented APIs.
pub fn poseidon2_hash_bytes(input: &[u8]) -> [Goldilocks; DIGEST_LEN] {
    let felts: Vec<Goldilocks> = input
        .iter()
        .map(|&b| Goldilocks::from_u64(b as u64))
        .collect();
    poseidon2_hash(&felts)
}

/// Hash bytes with efficient packing (8 bytes per field element).
/// 
/// Packs input bytes into u64 field elements (8 bytes per element).
/// Appends length as final element for unambiguous padding.
/// 
/// # Performance
/// - 8× more efficient than `poseidon2_hash_bytes`
/// - Preferred for hashing arbitrary byte strings
/// 
/// # Security
/// - Length encoding prevents length-extension attacks
/// - Little-endian packing is canonical and deterministic
pub fn poseidon2_hash_packed_bytes(input: &[u8]) -> [Goldilocks; DIGEST_LEN] {
    use core::mem::size_of;
    const LIMB: usize = size_of::<u64>();
    let mut felts = Vec::with_capacity((input.len() + LIMB - 1) / LIMB + 1);

    // Pack 8 bytes per field element (little-endian)
    for chunk in input.chunks(LIMB) {
        let mut buf = [0u8; LIMB];
        buf[..chunk.len()].copy_from_slice(chunk);
        felts.push(Goldilocks::from_u64(u64::from_le_bytes(buf)));
    }
    
    // Append length for unambiguous framing
    felts.push(Goldilocks::from_u64(input.len() as u64));
    
    poseidon2_hash(&felts)
}

/// Hash a single field element.
/// 
/// Convenience function for hashing a single Goldilocks element.
pub fn poseidon2_hash_single(x: Goldilocks) -> [Goldilocks; DIGEST_LEN] {
    poseidon2_hash(&[x])
}
