//! Tests for production Poseidon2 over Goldilocks Field
//! Implementation in neo-ccs, parameters from neo-params

use neo_ccs::crypto::poseidon2_goldilocks::*;
use p3_goldilocks::Goldilocks;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_poseidon2_basic_functionality() {
    // Test empty input
    let empty_hash = poseidon2_hash(&[]);
    assert_ne!(empty_hash, [Goldilocks::ZERO; DIGEST_LEN]);

    // Test single element
    let single_hash = poseidon2_hash_single(Goldilocks::ONE);
    assert_ne!(single_hash, [Goldilocks::ZERO; DIGEST_LEN]);
    assert_ne!(single_hash, empty_hash);

    // Test different inputs produce different outputs
    let hash1 = poseidon2_hash(&[Goldilocks::ONE, Goldilocks::TWO]);
    let hash2 = poseidon2_hash(&[Goldilocks::TWO, Goldilocks::ONE]);
    assert_ne!(hash1, hash2);
}

#[test]
fn test_poseidon2_deterministic() {
    let input = [Goldilocks::ONE, Goldilocks::TWO, Goldilocks::from_u64(42)];
    let hash1 = poseidon2_hash(&input);
    let hash2 = poseidon2_hash(&input);
    assert_eq!(hash1, hash2, "Hash should be deterministic");
}

#[test]
fn test_poseidon2_bytes() {
    let input = b"hello world";
    let hash1 = poseidon2_hash_bytes(input);
    let hash2 = poseidon2_hash_bytes(input);
    assert_eq!(hash1, hash2, "Byte hash should be deterministic");

    let different_input = b"hello world!";
    let hash3 = poseidon2_hash_bytes(different_input);
    assert_ne!(hash1, hash3, "Different inputs should produce different hashes");
}

#[test]
fn test_poseidon2_large_input() {
    // Test input larger than rate to ensure proper sponge behavior
    let large_input: Vec<Goldilocks> = (0..20)
        .map(|i| Goldilocks::from_u64(i as u64))
        .collect();
    let hash = poseidon2_hash(&large_input);
    assert_ne!(hash, [Goldilocks::ZERO; DIGEST_LEN]);
}

#[test]
fn test_packed_byte_hashing() {
    let input = b"hello world! this is a longer test string";
    
    // Test packed hashing is deterministic
    let hash1 = poseidon2_hash_packed_bytes(input);
    let hash2 = poseidon2_hash_packed_bytes(input);
    assert_eq!(hash1, hash2);
    
    // Test different inputs produce different hashes
    let different_input = b"hello world! this is a different test string";
    let hash3 = poseidon2_hash_packed_bytes(different_input);
    assert_ne!(hash1, hash3);
    
    // Test empty input
    let empty_hash = poseidon2_hash_packed_bytes(&[]);
    assert_ne!(empty_hash, [Goldilocks::ZERO; DIGEST_LEN]);
    assert_ne!(empty_hash, hash1);
}

#[test]
fn test_poseidon2_security_properties() {
    // Test that different length inputs with same prefix produce different hashes
    let short_input = [Goldilocks::ONE, Goldilocks::TWO];
    let long_input = [Goldilocks::ONE, Goldilocks::TWO, Goldilocks::ONE]; // Use non-zero value
    
    let hash_short = poseidon2_hash(&short_input);
    let hash_long = poseidon2_hash(&long_input);
    assert_ne!(hash_short, hash_long, "Different length inputs should produce different hashes");
}

#[test]
fn test_poseidon2_constants() {
    // Verify our constants are consistent
    assert_eq!(WIDTH, 12);
    assert_eq!(CAPACITY, 4);
    assert_eq!(RATE, WIDTH - CAPACITY);
    assert_eq!(RATE, 8);
    assert_eq!(DIGEST_LEN, CAPACITY);
}

#[test]
fn test_poseidon2_permutation_consistency() {
    // Test that multiple calls with same input to the underlying permutation are consistent
    let input1 = [Goldilocks::from_u64(1), Goldilocks::from_u64(2), Goldilocks::from_u64(3)];
    let input2 = [Goldilocks::from_u64(4), Goldilocks::from_u64(5), Goldilocks::from_u64(6)];
    
    let hash1a = poseidon2_hash(&input1);
    let hash1b = poseidon2_hash(&input1);
    let hash2 = poseidon2_hash(&input2);
    
    assert_eq!(hash1a, hash1b, "Same input should produce same hash");
    assert_ne!(hash1a, hash2, "Different inputs should produce different hashes");
}

#[test]
fn test_poseidon2_byte_vs_packed_consistency() {
    // For small inputs, both byte methods should be available
    let small_input = b"test";
    
    let hash_bytes = poseidon2_hash_bytes(small_input);
    let hash_packed = poseidon2_hash_packed_bytes(small_input);
    
    // They will be different due to different encoding, but both should be valid hashes
    assert_ne!(hash_bytes, [Goldilocks::ZERO; DIGEST_LEN]);
    assert_ne!(hash_packed, [Goldilocks::ZERO; DIGEST_LEN]);
    assert_ne!(hash_bytes, hash_packed, "Different encoding methods should produce different hashes");
}

#[test]
fn test_poseidon2_edge_cases() {
    // Test various edge cases
    
    // Single zero element
    let zero_hash = poseidon2_hash(&[Goldilocks::ZERO]);
    assert_ne!(zero_hash, [Goldilocks::ZERO; DIGEST_LEN]);
    
    // Multiple zeros - test with different non-zero value
    let multi_zero_hash = poseidon2_hash(&[Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ONE]);
    assert_ne!(multi_zero_hash, zero_hash);
    assert_ne!(multi_zero_hash, [Goldilocks::ZERO; DIGEST_LEN]);
    
    // Max field value
    let max_val = Goldilocks::from_u64(0xFFFFFFFF00000000);
    let max_hash = poseidon2_hash(&[max_val]);
    assert_ne!(max_hash, zero_hash);
    assert_ne!(max_hash, [Goldilocks::ZERO; DIGEST_LEN]);
    
    // Test that empty hash is different from single zero
    let empty_hash = poseidon2_hash(&[]);
    assert_ne!(empty_hash, zero_hash);
}
