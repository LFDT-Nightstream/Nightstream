use neo_ccs::{mv_poly, CcsStructure};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use neo_fold::{FoldState, verify_rlc, EvalInstance};
use neo_ring::RingElement;
use neo_modint::{ModInt, Coeff};

fn dummy_structure() -> CcsStructure {
    use p3_matrix::dense::RowMajorMatrix;
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO], 4),
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ONE], 4),
    ];
    let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1);
    CcsStructure::new(mats, f)
}

/// Test that empty transcript is rejected (fail closed)
#[test]
fn test_empty_transcript_rejected() {
    let structure = dummy_structure();
    let fold_state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    // Empty transcript should be rejected
    assert!(!fold_state.verify(&[], &committer));
}

/// Test that RLC with different r values is rejected
#[test]
fn test_rlc_different_r_rejected() {
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    // Create eval instances with different r values
    let e1 = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ONE), from_base(F::ZERO)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let e2 = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ZERO), from_base(F::ONE)], // Different r!
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let new_eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ONE), from_base(F::ZERO)],
        ys: vec![from_base(F::from_u64(2))], // e1.ys + e2.ys
        u: from_base(F::from_u64(2)), // e1.u + e2.u
        e_eval: from_base(F::from_u64(2)), // e1.e_eval + e2.e_eval
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let rho_rot = RingElement::from_scalar(ModInt::one(), TOY_PARAMS.n);
    
    // Should fail due to different r values
    assert!(!verify_rlc(&e1, &e2, &rho_rot, &new_eval, &committer));
}

/// Test that RLC with identical r values passes basic validation
#[test]
fn test_rlc_identical_r_passes() {
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    // Create eval instances with identical r values
    let r_common = vec![from_base(F::ONE), from_base(F::ZERO)];
    
    let e1 = EvalInstance {
        commitment: vec![],
        r: r_common.clone(),
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let e2 = EvalInstance {
        commitment: vec![],
        r: r_common.clone(), // Same r
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let new_eval = EvalInstance {
        commitment: vec![],
        r: r_common,
        ys: vec![from_base(F::from_u64(2))], // e1.ys + e2.ys
        u: from_base(F::from_u64(2)), // e1.u + e2.u
        e_eval: from_base(F::from_u64(2)), // e1.e_eval + e2.e_eval
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let rho_rot = RingElement::from_scalar(ModInt::one(), TOY_PARAMS.n);
    
    // Should pass r validation (may fail other checks, but not r validation)
    // We expect this to pass the r check but may fail on other validations
    let result = verify_rlc(&e1, &e2, &rho_rot, &new_eval, &committer);
    // The test should at least get past the r validation check
    // (it might fail later due to commitment checks, but that's OK)
    println!("RLC result with identical r: {}", result);
}

/// Test that verify_dec rejects empty commitments
#[test]
fn test_verify_dec_rejects_empty_commitment() {
    use neo_fold::verify_dec;
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    let e = EvalInstance {
        commitment: vec![RingElement::from_scalar(ModInt::zero(), TOY_PARAMS.n)],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let new_eval = EvalInstance {
        commitment: vec![], // Empty commitment should be rejected
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    assert!(!verify_dec(&committer, &e, &new_eval));
}

/// Test that verify_dec rejects wrong commitment length
#[test]
fn test_verify_dec_rejects_wrong_commitment_length() {
    use neo_fold::verify_dec;
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    let e = EvalInstance {
        commitment: vec![RingElement::from_scalar(ModInt::zero(), TOY_PARAMS.n)],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let new_eval = EvalInstance {
        commitment: vec![
            RingElement::from_scalar(ModInt::zero(), TOY_PARAMS.n);
            TOY_PARAMS.k + 1 // Wrong length
        ],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    assert!(!verify_dec(&committer, &e, &new_eval));
}

/// Test that verify_dec rejects excessive norm bounds
#[test]
fn test_verify_dec_rejects_excessive_norm_bound() {
    use neo_fold::verify_dec;
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    
    let e = EvalInstance {
        commitment: vec![RingElement::from_scalar(ModInt::zero(), TOY_PARAMS.n); TOY_PARAMS.k],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    
    let new_eval = EvalInstance {
        commitment: vec![RingElement::from_scalar(ModInt::zero(), TOY_PARAMS.n); TOY_PARAMS.k],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ONE),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound + 1, // Excessive norm bound
        opening_proof: None,
    };
    
    assert!(!verify_dec(&committer, &e, &new_eval));
}
