//! Direct behavior checks for the Poseidon2 hash helpers.

use neo_ccs::crypto::poseidon2_goldilocks::{
    poseidon2_hash, poseidon2_hash_packed_bytes, poseidon2_hash_single, Poseidon2Hasher, RATE,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

// ---------------------------------------------------------------------------
// 1. poseidon2_determinism
// ---------------------------------------------------------------------------
#[test]
fn poseidon2_determinism() {
    let input = [F::from_u64(1), F::from_u64(2), F::from_u64(3)];

    let h1 = poseidon2_hash(&input);
    let h2 = poseidon2_hash(&input);

    assert_eq!(h1, h2, "same input should produce same hash output");
}

// ---------------------------------------------------------------------------
// 2. poseidon2_domain_separation
// ---------------------------------------------------------------------------
#[test]
fn poseidon2_domain_separation() {
    let input_a = [F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    let input_b = [F::from_u64(1), F::from_u64(2), F::from_u64(4)]; // last element differs

    let h_a = poseidon2_hash(&input_a);
    let h_b = poseidon2_hash(&input_b);

    assert_ne!(h_a, h_b, "different inputs should produce different hashes");
}

// ---------------------------------------------------------------------------
// 3. poseidon2_hash_single_matches
// ---------------------------------------------------------------------------
#[test]
fn poseidon2_hash_single_matches() {
    let x = F::from_u64(42);

    let h_single = poseidon2_hash_single(x);
    let h_array = poseidon2_hash(&[x]);

    assert_eq!(h_single, h_array, "hash_single(x) should equal hash(&[x])");
}

// ---------------------------------------------------------------------------
// 4. poseidon2_packed_bytes_length_encoding
// ---------------------------------------------------------------------------
#[test]
fn poseidon2_packed_bytes_length_encoding() {
    // [1, 2] and [1, 2, 0] should hash differently because the length
    // is appended to the packed elements, disambiguating them.
    let input_a = &[1u8, 2];
    let input_b = &[1u8, 2, 0];

    let h_a = poseidon2_hash_packed_bytes(input_a);
    let h_b = poseidon2_hash_packed_bytes(input_b);

    assert_ne!(
        h_a, h_b,
        "packed_bytes should distinguish [1,2] from [1,2,0] via length encoding"
    );
}

#[test]
fn poseidon2_packed_bytes_is_injective_before_hashing() {
    let zero = [0u8; 8];
    let modulus = 0xffff_ffff_0000_0001u64.to_le_bytes();

    assert_ne!(
        poseidon2_hash_packed_bytes(&zero),
        poseidon2_hash_packed_bytes(&modulus),
        "byte packing must not reduce a full u64 modulo the Goldilocks modulus"
    );
}

#[test]
fn incremental_hash_matches_every_split_across_full_and_partial_blocks() {
    let input: Vec<_> = (0..2 * RATE + 1)
        .map(|index| F::from_u64(u64::MAX - index as u64))
        .collect();
    for length in 0..=input.len() {
        let expected = poseidon2_hash(&input[..length]);
        for first in 0..=length {
            for second in first..=length {
                let mut hasher = Poseidon2Hasher::default();
                hasher.update(&input[..first]);
                hasher.update(&[]);
                hasher.update(&input[first..second]);
                hasher.update(&input[second..length]);
                assert_eq!(hasher.finalize(), expected, "length={length}, updates={first},{second}");
            }
        }
    }
}
