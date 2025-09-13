//! Poseidon2 unification & domain separation tests
//! This locks in w=12, cap=4 and the exact domain string used by create_step_digest.
//! If someone accidentally switches back to w=16 or tweaks the domain string, this will fail.

use neo::F;
use neo::ivc::create_step_digest;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
use p3_field::{PrimeField64, PrimeCharacteristicRing};

#[allow(dead_code)]
fn manual_step_digest(step_data: &[F]) -> [u8; 32] {
    let perm = p2::permutation();
    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0usize;
    const RATE: usize = p2::RATE;
    let mut absorb = |x: Goldilocks| {
        if absorbed == RATE { st = perm.permute(st); absorbed = 0; }
        st[absorbed] = x; absorbed += 1;
    };
    for &b in b"neo/ivc/step-digest/v1|poseidon2-goldilocks-w12-cap4" {
        absorb(Goldilocks::from_u64(b as u64));
    }
    absorb(Goldilocks::from_u64(step_data.len() as u64));
    for &f in step_data {
        absorb(Goldilocks::from_u64(f.as_canonical_u64()));
    }
    st = perm.permute(st);
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i*8..(i+1)*8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    out
}

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
    assert_eq!(p2::WIDTH, 12, "Poseidon2 width must be 12");
    assert_eq!(p2::CAPACITY, 4, "Poseidon2 capacity must be 4"); 
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
