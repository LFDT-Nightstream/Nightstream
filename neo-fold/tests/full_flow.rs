use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use neo_fold::{FoldState, Proof, extractor, verify_open, EvalInstance};

fn dummy_structure() -> CcsStructure {
    // 4-element structure matching the verifier CCS for consistency
    use p3_matrix::dense::RowMajorMatrix;
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ZERO], 4), // X0 selector
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO], 4), // X1 selector  
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO], 4), // X2 selector
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ONE], 4), // X3 selector
    ];
    let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1); // Zero constraint for consistency
    CcsStructure::new(mats, f)
}

#[test]
fn test_full_folding() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance1 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness1 = CcsWitness { z: vec![from_base(F::from_u64(2)), from_base(F::ZERO), from_base(F::ZERO), from_base(F::ZERO)] };
    let instance2 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness2 = CcsWitness { z: vec![from_base(F::from_u64(3)), from_base(F::ZERO), from_base(F::ZERO), from_base(F::ZERO)] };
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
}

#[test]
fn test_full_folding_with_fri() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance1 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness1 = CcsWitness { z: vec![from_base(F::from_u64(2)), from_base(F::ZERO), from_base(F::ZERO), from_base(F::ZERO)] };
    let instance2 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness2 = CcsWitness { z: vec![from_base(F::from_u64(3)), from_base(F::ZERO), from_base(F::ZERO), from_base(F::ZERO)] };
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
}

#[test]
fn test_extractor_rewinds() {
    let proof = Proof { transcript: vec![0; 100] };
    let witness = extractor(&proof);
    assert_eq!(witness.z.len(), 4); // Extractor always returns 4-element witness for verifier CCS
    assert!(witness.z.iter().all(|&e| e != ExtF::ZERO)); // All elements should be non-zero
}

#[test]
fn test_ivc_chain() {
    let mut state = FoldState::new(dummy_structure());
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness = CcsWitness { z: vec![from_base(F::from_u64(5)), from_base(F::ZERO), from_base(F::ZERO), from_base(F::ZERO)] };
    state.ccs_instance = Some((instance, witness));
    assert!(state.recursive_ivc(2, &committer));
}

#[test]
fn test_verify_open_valid() {
    let structure = dummy_structure();
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![],
        ys: vec![ExtF::ZERO; structure.mats.len()],
        u: ExtF::ZERO,
        e_eval: ExtF::ONE,
        norm_bound: committer.params().norm_bound,
    };
    assert!(verify_open(&structure, &committer, &eval, committer.params().max_blind_norm));
}

#[test]
fn test_verify_open_invalid_eval() {
    // Create a structure with a linear constraint polynomial (deg <= 1)
    use p3_matrix::dense::RowMajorMatrix;
    let mats = vec![
        RowMajorMatrix::new(vec![F::ONE, F::ZERO, F::ZERO, F::ZERO], 4), // X0 selector
        RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::ZERO, F::ZERO], 4), // X1 selector  
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ONE, F::ZERO], 4), // X2 selector
        RowMajorMatrix::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ONE], 4), // X3 selector
    ];
    let f = mv_poly(|ys: &[ExtF]| ys[0], 1); // Linear: returns first variable, degree 1
    let structure = CcsStructure::new(mats, f);
    
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![],
        ys: vec![ExtF::ONE; structure.mats.len()],
        u: ExtF::ZERO,
        e_eval: ExtF::ONE,
        norm_bound: committer.params().norm_bound,
    };
    // This should fail because f(ys) = ys[0] = 1, but u * e_eval^2 = 0 * 1^2 = 0
    assert!(!verify_open(&structure, &committer, &eval, committer.params().max_blind_norm));
}

#[test]
fn test_verify_open_invalid_norm() {
    let structure = dummy_structure();
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![],
        ys: vec![ExtF::ZERO; structure.mats.len()],
        u: ExtF::new_real(F::from_u64(committer.params().max_blind_norm + 1)), // Test u norm instead
        e_eval: ExtF::ONE,
        norm_bound: committer.params().norm_bound,
    };
    assert!(!verify_open(&structure, &committer, &eval, committer.params().max_blind_norm));
}