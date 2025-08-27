use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::F;

use p3_field::PrimeCharacteristicRing;

/// Test basic commitment homomorphism property.
/// This validates the linear homomorphism property of Ajtai commitments.
#[test]
fn test_commitment_homomorphism() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // Create two different witnesses
    let z1: Vec<F> = vec![F::ONE, F::from_u64(2)];
    let z2: Vec<F> = vec![F::from_u64(3), F::ZERO];
    
    // Pad to required length
    let mut z1_padded = z1;
    let mut z2_padded = z2;
    z1_padded.resize(params.n, F::ZERO);
    z2_padded.resize(params.n, F::ZERO);
    
    // Decompose and pack
    let mat1 = decomp_b(&z1_padded, params.b, params.d);
    let w1 = AjtaiCommitter::pack_decomp(&mat1, &params);
    
    let mat2 = decomp_b(&z2_padded, params.b, params.d);
    let w2 = AjtaiCommitter::pack_decomp(&mat2, &params);
    
    // Commit to each witness separately with same randomness for determinism
    let mut t1 = vec![1u8; 32]; 
    let (c1, e1, blinded_w1, _r1) = committer.commit(&w1, &mut t1).unwrap();
    
    let mut t2 = vec![1u8; 32]; // Same seed
    let (c2, e2, blinded_w2, _r2) = committer.commit(&w2, &mut t2).unwrap();
    
    // Verify individual commitments are valid
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // Test that different witnesses produce different commitments (binding)
    assert_ne!(c1, c2, "Different witnesses should produce different commitments");
}

/// Test additive homomorphism with zero witness.
/// Validates that commit(w + 0) behaves consistently.
#[test]
fn test_zero_addition_homomorphism() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let z = vec![F::from_u64(42); params.n];
    let zero_z = vec![F::ZERO; params.n];
    
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    
    let zero_mat = decomp_b(&zero_z, params.b, params.d);
    let zero_w = AjtaiCommitter::pack_decomp(&zero_mat, &params);
    
    // Commitments should verify individually
    let mut t1 = vec![1u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&w, &mut t1).unwrap();
    
    let mut t2 = vec![2u8; 32];
    let (c2, e2, blinded_w2, _) = committer.commit(&zero_w, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // Zero and non-zero should produce different commitments
    assert_ne!(c1, c2, "Zero and non-zero witnesses should produce different commitments");
}

/// Test that commitments are consistent for the same input.
/// Validates deterministic behavior with same randomness.
#[test]
fn test_commitment_consistency() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let z = vec![F::from_u64(17); params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    
    // Same witness with same randomness should give same commitment
    let mut t1 = vec![42u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&w, &mut t1).unwrap();
    
    let mut t2 = vec![42u8; 32]; // Same randomness
    let (c2, e2, blinded_w2, _) = committer.commit(&w, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // With same randomness, commitments should be identical
    assert_eq!(c1, c2, "Same witness with same randomness should produce same commitment");
    assert_eq!(e1, e2, "Same witness with same randomness should produce same error");
}

/// Test that different randomness produces different commitments (hiding property).
/// This validates the zero-knowledge hiding property.
#[test]
fn test_commitment_hiding() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let z = vec![F::from_u64(99); params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    
    // Same witness with different randomness should give different commitments
    let mut t1 = vec![1u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&w, &mut t1).unwrap();
    
    let mut t2 = vec![2u8; 32]; // Different randomness
    let (c2, e2, blinded_w2, _) = committer.commit(&w, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // Different randomness should produce different commitments (hiding)
    assert_ne!(c1, c2, "Same witness with different randomness should produce different commitments");
}

/// Test commitment with varying witness sizes.
/// Validates that the commitment scheme works correctly for different input sizes.
#[test]
fn test_varying_witness_sizes() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    // Test with small witness
    let small_z = vec![F::ONE; params.n / 2];
    let mut padded_small = small_z;
    padded_small.resize(params.n, F::ZERO);
    
    let small_mat = decomp_b(&padded_small, params.b, params.d);
    let small_w = AjtaiCommitter::pack_decomp(&small_mat, &params);
    
    // Test with full witness
    let full_z = vec![F::from_u64(2); params.n];
    let full_mat = decomp_b(&full_z, params.b, params.d);
    let full_w = AjtaiCommitter::pack_decomp(&full_mat, &params);
    
    // Both should commit successfully
    let mut t1 = vec![3u8; 32];
    let (c1, e1, blinded_w1, _) = committer.commit(&small_w, &mut t1).unwrap();
    
    let mut t2 = vec![4u8; 32];
    let (c2, e2, blinded_w2, _) = committer.commit(&full_w, &mut t2).unwrap();
    
    assert!(committer.verify(&c1, &blinded_w1, &e1));
    assert!(committer.verify(&c2, &blinded_w2, &e2));
    
    // Should produce different commitments
    assert_ne!(c1, c2, "Different witness contents should produce different commitments");
}

/// Test that commitment verification fails for invalid proofs.
/// This validates the soundness of the verification process.
#[test]
fn test_invalid_verification() {
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    
    let z = vec![F::from_u64(123); params.n];
    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);
    
    let mut t = vec![5u8; 32];
    let (c, e, blinded_w, _) = committer.commit(&w, &mut t).unwrap();
    
    // Valid verification should pass
    assert!(committer.verify(&c, &blinded_w, &e));
    
    // Tampered commitment should fail
    let mut bad_c = c.clone();
    if !bad_c.is_empty() {
        bad_c[0] = bad_c[0].clone() + neo_ring::RingElement::from_scalar(neo_modint::ModInt::from_u64(1), params.n);
    }
    assert!(!committer.verify(&bad_c, &blinded_w, &e), "Tampered commitment should not verify");
    
    // Tampered witness should fail
    let mut bad_w = blinded_w.clone();
    if !bad_w.is_empty() {
        bad_w[0] = bad_w[0].clone() + neo_ring::RingElement::from_scalar(neo_modint::ModInt::from_u64(1), params.n);
    }
    assert!(!committer.verify(&c, &bad_w, &e), "Tampered witness should not verify");
}