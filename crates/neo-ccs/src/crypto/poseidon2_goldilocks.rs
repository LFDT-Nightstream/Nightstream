//! Production Poseidon2 over Goldilocks Field (single source of truth)
//!
//! - WIDTH = 12, CAPACITY = 4, RATE = 8
//! - ~128-bit collision security (capacity bits ≈ 256)
//! - Used for transcripts, context digests, and any off-circuit hashing

use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Poseidon2 parameters
pub const WIDTH: usize = 12;
/// Capacity of the sponge construction (security parameter)
pub const CAPACITY: usize = 4;
/// Rate of absorption (WIDTH - CAPACITY = 8)
pub const RATE: usize = WIDTH - CAPACITY; // 8
/// Length of output digest (same as capacity, 4 field elements = 32 bytes)
pub const DIGEST_LEN: usize = CAPACITY;   // we squeeze 4 fe's (=32 bytes)

/// Fixed seed so all permutations are identical across crates/runs.
pub const SEED: [u8; 32] = *b"neo_poseidon2_goldilocks_seed___";

/// Expose the permutation so other crates can use the *same* Poseidon2 instance.
pub fn permutation() -> Poseidon2Goldilocks<{ WIDTH }> {
    let mut rng = ChaCha8Rng::from_seed(SEED);
    Poseidon2Goldilocks::<{ WIDTH }>::new_from_rng_128(&mut rng)
}

/// Standard sponge with add-absorb + finalize (padding = add 1 then permute)
pub fn poseidon2_hash(input: &[Goldilocks]) -> [Goldilocks; DIGEST_LEN] {
    let perm = permutation();
    let mut state = [Goldilocks::ZERO; WIDTH];

    for chunk in input.chunks(RATE) {
        for (i, &x) in chunk.iter().enumerate() {
            state[i] = state[i] + x;
        }
        state = perm.permute(state);
    }
    state[0] = state[0] + Goldilocks::ONE;
    state = perm.permute(state);

    let mut out = [Goldilocks::ZERO; DIGEST_LEN];
    out.copy_from_slice(&state[..DIGEST_LEN]);
    out
}

/// Hash raw bytes by converting each byte to a field element
pub fn poseidon2_hash_bytes(input: &[u8]) -> [Goldilocks; DIGEST_LEN] {
    let felts: Vec<Goldilocks> = input.iter().map(|&b| Goldilocks::from_u64(b as u64)).collect();
    poseidon2_hash(&felts)
}

/// Hash bytes with efficient packing (8 bytes per field element) plus length suffix
pub fn poseidon2_hash_packed_bytes(input: &[u8]) -> [Goldilocks; DIGEST_LEN] {
    use core::mem::size_of;
    const LIMB: usize = size_of::<u64>();
    let mut felts = Vec::with_capacity((input.len() + LIMB - 1) / LIMB + 1);

    for chunk in input.chunks(LIMB) {
        let mut buf = [0u8; LIMB];
        buf[..chunk.len()].copy_from_slice(chunk);
        felts.push(Goldilocks::from_u64(u64::from_le_bytes(buf)));
    }
    felts.push(Goldilocks::from_u64(input.len() as u64));
    poseidon2_hash(&felts)
}

/// Hash a single field element
pub fn poseidon2_hash_single(x: Goldilocks) -> [Goldilocks; DIGEST_LEN] {
    poseidon2_hash(&[x])
}