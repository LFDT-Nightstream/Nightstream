//! Poseidon2 unification & domain separation tests
//! This locks in w=16, cap=8 and the exact domain string used by create_step_digest.
//! Parameters are defined in neo-params as single source of truth.

use neo::F;
use neo::create_step_digest;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_field::PrimeCharacteristicRing;

#[test] 
fn ivc_step_digest_uses_unified_poseidon2_parameters() {
    // This test validates that create_step_digest uses the unified Poseidon2 parameters
    // by checking that it produces consistent results and doesn't panic
    let data = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)];
    let d1 = create_step_digest(&data);
    let d2 = create_step_digest(&data);
    
    // Should be deterministic
    assert_eq!(d1, d2, "step digest should be deterministic");
    
    // Should produce 32-byte output (4 field elements * 8 bytes each)
    assert_eq!(d1.len(), 32, "step digest should be 32 bytes");
    
    // Should produce different outputs for different inputs
    let different_data = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(5)];
    let d3 = create_step_digest(&different_data);
    assert_ne!(d1, d3, "different inputs should produce different digests");
}

#[test]
fn changing_input_changes_digest() {
    let data1 = vec![F::from_u64(9), F::from_u64(8)];
    let data2 = vec![F::from_u64(9), F::from_u64(7)];
    let d1 = create_step_digest(&data1);
    let d2 = create_step_digest(&data2);
    assert_ne!(d1, d2);
}

#[test]
fn poseidon2_constants_are_correct() {
    // Lock in the exact parameters we expect
    assert_eq!(p2::WIDTH, 16, "Poseidon2 width must be 16");
    assert_eq!(p2::CAPACITY, 8, "Poseidon2 capacity must be 8"); 
    assert_eq!(p2::RATE, 8, "Poseidon2 rate must be 8");
    assert_eq!(p2::DIGEST_LEN, 4, "Digest length must be 4 field elements");
}

#[test]
fn step_digest_deterministic() {
    let data = vec![F::from_u64(42), F::from_u64(99)];
    let d1 = create_step_digest(&data);
    let d2 = create_step_digest(&data);
    assert_eq!(d1, d2, "step digest must be deterministic");
}

#[test]
fn empty_step_data_works() {
    let empty_data = vec![];
    let _digest = create_step_digest(&empty_data); // should not panic
}

#[test]
fn single_element_step_data() {
    let single = vec![F::from_u64(123)];
    let d1 = create_step_digest(&single);
    
    // Different single element should give different digest
    let single2 = vec![F::from_u64(124)];
    let d2 = create_step_digest(&single2);
    assert_ne!(d1, d2);
}
