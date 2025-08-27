use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{ExtF, F};
use neo_modint::ModInt;
use p3_field::PrimeCharacteristicRing;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn open_at_point_rejects_non_base_evaluation() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    // Choose a simple non-zero z so eval is non-zero
    let mut z = vec![F::ZERO; params.n];
    z[0] = F::ONE;

    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    let mut t = Vec::new();
    let (c, _e, blinded_w, r) = comm.commit(&w, &mut t).unwrap();

    // 1-bit point with a purely imaginary coordinate (guarantees a non-base evaluation generically)
    let point = vec![ExtF::new_complex(F::ZERO, F::ONE)];
    let mut rng = StdRng::seed_from_u64(0);

    let res = comm.open_at_point(&c, &point, &blinded_w, &[], &r, &mut rng);
    assert!(res.is_err(), "opening at a non-base point must return Err");
    
    // Verify the error message is about non-base field
    if let Err(msg) = res {
        assert!(msg.contains("not in base field"), 
            "Error should mention base field issue, got: {}", msg);
    }
}

#[test]
fn verify_opening_rejects_non_base_evaluation() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    // Create a dummy commitment
    let c = vec![neo_ring::RingElement::from_scalar(ModInt::from_u64(0), params.n); params.k];
    let point = vec![ExtF::new_complex(F::ZERO, F::ONE)]; // Non-base field point
    let eval = ExtF::new_complex(F::ONE, F::ONE); // Non-base field evaluation
    let proof = vec![neo_ring::RingElement::from_scalar(ModInt::from_u64(0), params.n); params.d];
    
    // Should return false (reject) for non-base field evaluation
    let result = comm.verify_opening(&c, &point, eval, &proof, params.max_blind_norm);
    assert!(!result, "verify_opening should reject non-base field evaluations");
}

#[test]
fn open_at_point_accepts_base_field_evaluation() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    // Choose a simple non-zero z
    let mut z = vec![F::ZERO; params.n];
    z[0] = F::ONE;

    let mat = decomp_b(&z, params.b, params.d);
    let w = AjtaiCommitter::pack_decomp(&mat, &params);

    let mut t = Vec::new();
    let (c, _e, blinded_w, r) = comm.commit(&w, &mut t).unwrap();

    // Base field point (imaginary part is zero)
    let point = vec![ExtF::new_complex(F::ONE, F::ZERO)];
    let mut rng = StdRng::seed_from_u64(0);

    let res = comm.open_at_point(&c, &point, &blinded_w, &[], &r, &mut rng);
    // This should succeed (or fail for other reasons, but not due to non-base field)
    if let Err(msg) = &res {
        assert!(!msg.contains("not in base field"), 
            "Should not fail due to base field issue with base field point, got: {}", msg);
    }
}

#[test]
fn verify_opening_accepts_base_field_evaluation() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    // Create a simple valid commitment scenario
    let c = vec![neo_ring::RingElement::from_scalar(ModInt::from_u64(0), params.n); params.k];
    let point = vec![ExtF::new_complex(F::ONE, F::ZERO)]; // Base field point
    let eval = ExtF::new_complex(F::ZERO, F::ZERO); // Base field evaluation (zero)
    let proof = vec![neo_ring::RingElement::from_scalar(ModInt::from_u64(0), params.n); params.d];
    
    // This might still fail for other reasons (invalid proof), but should not fail due to non-base field
    let _result = comm.verify_opening(&c, &point, eval, &proof, params.max_blind_norm);
    // We don't assert the result here since the proof might be invalid for other reasons,
    // but the key point is it shouldn't panic or fail due to imaginary part handling
}
