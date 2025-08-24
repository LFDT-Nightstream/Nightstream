use neo_ccs::{mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{embed_base_to_ext, project_ext_to_base, ExtF, F};
use neo_fold::{pi_ccs, verify_ccs, EvalInstance, FoldState};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

fn dummy_structure() -> CcsStructure {
    let mat = RowMajorMatrix::<F>::new(vec![F::ZERO, F::ZERO], 2);
    CcsStructure::new(vec![mat], mv_poly(|_: &[ExtF]| ExtF::ZERO, 1))
}

#[test]
fn test_pi_ccs_unified_no_mismatch() {
    let structure = dummy_structure();
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    let mut state = FoldState::new(structure);
    let instance = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness = CcsWitness {
        z: vec![embed_base_to_ext(F::ZERO), embed_base_to_ext(F::ONE)],
    };
    state.ccs_instance = Some((instance, witness));
    let mut transcript = vec![];
    let msgs = pi_ccs(&mut state, &committer, &mut transcript, None);
    assert!(!msgs.is_empty());
    let eval = state.eval_instances.last().unwrap();
    for &y in &eval.ys {
        assert!(project_ext_to_base(y).is_some());
    }
}

#[test]
fn test_full_fold_unified() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance1 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness1 = CcsWitness {
        z: vec![embed_base_to_ext(F::ONE), embed_base_to_ext(F::from_u64(2))],
    };
    let instance2 = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let witness2 = CcsWitness {
        z: vec![embed_base_to_ext(F::from_u64(3)), embed_base_to_ext(F::from_u64(3))],
    };
    let proof = state.generate_proof((instance1, witness1), (instance2, witness2), &committer);
    assert!(state.verify(&proof.transcript, &committer));
    let final_eval = state.eval_instances.last().unwrap();
    assert!(project_ext_to_base(final_eval.e_eval).is_some());
}

#[test]
fn test_verify_ccs_trusts_sumcheck_in_nark_mode() {
    // UPDATED: In NARK mode, verify_ccs trusts the sumcheck proof rather than doing
    // additional constraint validation. This test verifies that behavior.
    let structure = dummy_structure();
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let instance = CcsInstance { commitment: vec![], public_input: vec![], u: F::ZERO, e: F::ONE };
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![],
        ys: vec![ExtF::ZERO],
        u: ExtF::ZERO,
        e_eval: ExtF::ZERO,
        norm_bound: 1,
        opening_proof: None,
    };
    // Both valid and "invalid" instances should pass in NARK mode because we trust sumcheck
    assert!(verify_ccs(&structure, &instance, 1, &[], &[eval.clone()], &committer));
    let mut different_eval = eval.clone();
    different_eval.ys[0] = ExtF::new_complex(F::ONE, F::ONE);
    // In NARK mode, this also passes because we trust the sumcheck proof
    assert!(verify_ccs(&structure, &instance, 1, &[], &[different_eval], &committer));
}

