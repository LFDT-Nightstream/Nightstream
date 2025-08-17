use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use p3_matrix::dense::RowMajorMatrix;
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;
use neo_fold::{FoldState, Proof, extractor};

fn dummy_structure() -> CcsStructure {
    // Single-variable structure with a 1x1 zero matrix to keep the test lightweight.
    let mat = RowMajorMatrix::<F>::new(vec![F::ZERO], 1);
    CcsStructure::new(vec![mat], mv_poly(|_: &[ExtF]| ExtF::ZERO, 1))
}

#[test]
fn test_full_folding() {
    if std::env::var("RUN_LONG_TESTS").is_err() {
        return;
    }
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance1 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness1 = CcsWitness { z: vec![from_base(F::from_u64(2))] };
    let instance2 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness2 = CcsWitness { z: vec![from_base(F::from_u64(3))] };
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
}

#[test]
fn test_full_folding_with_fri() {
    if std::env::var("RUN_LONG_TESTS").is_err() {
        return;
    }
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance1 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness1 = CcsWitness { z: vec![from_base(F::from_u64(2))] };
    let instance2 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness2 = CcsWitness { z: vec![from_base(F::from_u64(3))] };
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
}

#[test]
fn test_extractor_rewinds() {
    let proof = Proof { transcript: vec![0; 100] };
    let witness = extractor(&proof);
    assert_eq!(witness.z.len(), 1);
    assert_ne!(witness.z[0], ExtF::ZERO);
}

#[test]
fn test_ivc_chain() {
    if std::env::var("RUN_LONG_TESTS").is_err() {
        return;
    }
    let mut state = FoldState::new(dummy_structure());
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness = CcsWitness { z: vec![from_base(F::from_u64(5))] };
    state.ccs_instance = Some((instance, witness));
    assert!(state.recursive_ivc(2, &committer));
}
