//! Red-team tests for the direct-CCS frontend.
//!
//! Confirms the frontend validation path rejects bad inputs *before* any
//! folding work happens — so users get clear errors at synthesis time
//! instead of cryptic engine failures later.
//!
//! These tests exercise:
//! - `R1cs::validate_shape` rejection for mismatched matrix shapes.
//! - `R1cs::is_satisfied_by` rejection at the offending row.
//! - `build_instance` rejection on non-satisfying or out-of-bound assignments.

use std::sync::Arc;

use neo_ajtai::{set_global_pp_seeded, setup_par, AjtaiSModule};
use neo_ccs::matrix::Mat as NeoMat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use neo_fold_clean::config;
use neo_fold_clean::frontends::direct_ccs::{self, FrontendError, R1cs};
use neo_fold_clean::lifecycle;
use neo_fold_clean::lifecycle::preprocess_with_test_log;
use neo_fold_clean::preprocess as lifecycle_preprocess;
use neo_fold_clean::{finish_uncompressed, prove, verify_uncompressed, FoldSchedule};

fn three_term_addition() -> R1cs {
    wide_three_term_addition(1)
}

fn wide_three_term_addition(cols: usize) -> R1cs {
    let m = D * cols;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

fn one_term_copy() -> R1cs {
    let m = D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

#[test]
fn shape_mismatch_at_validation_time() {
    let mut r1cs = three_term_addition();
    // Make B's column count diverge from A.
    r1cs.b = NeoMat::zero(1, D + 1, F::default());
    match r1cs.validate_shape() {
        Err(FrontendError::ShapeMismatch { .. }) => {}
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
fn public_input_split_too_large() {
    let mut r1cs = three_term_addition();
    r1cs.m_in = D + 1;
    match r1cs.validate_shape() {
        Err(FrontendError::PublicInputTooLarge { .. }) => {}
        other => panic!("expected PublicInputTooLarge, got {:?}", other),
    }
}

#[test]
fn assignment_length_mismatch_rejected() {
    let r1cs = three_term_addition();
    let z_short = vec![F::ONE, F::ONE, F::ZERO]; // length 3 but R1CS needs D
    match r1cs.is_satisfied_by(&z_short) {
        Err(FrontendError::AssignmentLength { got: 3, .. }) => {}
        other => panic!("expected AssignmentLength, got {:?}", other),
    }
}

#[test]
fn build_instance_rejects_preprocessing_public_input_mismatch() {
    let r1cs_for_prep = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs_for_prep, /* seed = */ 13).expect("preprocess");

    let mut r1cs_for_instance = three_term_addition();
    r1cs_for_instance.m_in = 2;

    let z = satisfying_three_term_assignment();
    match direct_ccs::build_instance(&prep, &r1cs_for_instance, &z) {
        Err(FrontendError::PreprocessingPublicInputMismatch {
            r1cs_m_in: 2,
            prep_m_in: Some(3),
        }) => {}
        other => panic!("expected PreprocessingPublicInputMismatch, got {:?}", other),
    }
}

#[test]
fn build_instance_rejects_preprocessing_structure_mismatch() {
    let r1cs_for_prep = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs_for_prep, /* seed = */ 14).expect("preprocess");

    let r1cs_for_instance = one_term_copy();
    let z = satisfying_three_term_assignment();

    match direct_ccs::build_instance(&prep, &r1cs_for_instance, &z) {
        Err(FrontendError::PreprocessingStructureMismatch) => {}
        other => panic!("expected PreprocessingStructureMismatch, got {:?}", other),
    }
}

#[test]
fn build_instance_rejects_preprocessing_polynomial_mismatch() {
    let r1cs_for_prep = three_term_addition();
    let mut bad_structure = r1cs_for_prep.to_structure();
    bad_structure.f = SparsePoly::new(
        3,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }],
    );
    let params = config::r1cs_params(bad_structure.n, bad_structure.m).expect("production-core R1CS params");
    let _ = direct_ccs::ajtai::setup_seeded(&params, &bad_structure, /* seed = */ 15);
    let prep = lifecycle_preprocess(
        params,
        bad_structure,
        direct_ccs::ajtai_rlc_mixer,
        direct_ccs::ajtai_dec_mixer,
        Some(r1cs_for_prep.m_in),
    )
    .expect("preprocess with deliberately bad polynomial");

    let z = satisfying_three_term_assignment();
    match direct_ccs::build_instance(&prep, &r1cs_for_prep, &z) {
        Err(FrontendError::PreprocessingStructureMismatch) => {}
        other => panic!("expected PreprocessingStructureMismatch, got {:?}", other),
    }
}

#[test]
fn preprocess_rejects_ajtai_dimension_mismatch() {
    let r1cs = three_term_addition();
    let structure = r1cs.to_structure();
    let params = config::r1cs_params(structure.n, structure.m).expect("production-core R1CS params");
    let expected_cols = structure.m.div_ceil(D);
    let wrong_cols = expected_cols + 41;
    let log = global_log(D, wrong_cols, params.kappa() as usize, 0xA170_0001);

    match preprocess_with_test_log(
        params,
        structure,
        log,
        direct_ccs::ajtai_rlc_mixer,
        direct_ccs::ajtai_dec_mixer,
        Some(r1cs.m_in),
    ) {
        Err(lifecycle::Error::AjtaiDimensionMismatch {
            expected_d: D,
            expected_cols: got_expected_cols,
            got_d: D,
            got_cols,
        }) if got_expected_cols == expected_cols && got_cols == wrong_cols => {}
        Ok(_) => panic!("expected AjtaiDimensionMismatch, got Ok"),
        Err(other) => panic!("expected AjtaiDimensionMismatch, got {other}"),
    }
}

#[test]
fn preprocess_rejects_ajtai_kappa_mismatch() {
    let r1cs = wide_three_term_addition(43);
    let structure = r1cs.to_structure();
    let params = config::r1cs_params(structure.n, structure.m).expect("production-core R1CS params");
    let cols = structure.m.div_ceil(D);
    let wrong_kappa = params.kappa() as usize + 1;
    let log = global_log(D, cols, wrong_kappa, 0xA170_0002);

    match preprocess_with_test_log(
        params.clone(),
        structure,
        log,
        direct_ccs::ajtai_rlc_mixer,
        direct_ccs::ajtai_dec_mixer,
        Some(r1cs.m_in),
    ) {
        Err(lifecycle::Error::AjtaiKappaMismatch { expected, got })
            if expected == params.kappa() as usize && got == wrong_kappa => {}
        Ok(_) => panic!("expected AjtaiKappaMismatch, got Ok"),
        Err(other) => panic!("expected AjtaiKappaMismatch, got {other}"),
    }
}

#[test]
fn verifier_rejects_proof_committed_with_different_same_shaped_ajtai_setup() {
    let r1cs = three_term_addition();
    let verifier_prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 0xA170_0100).expect("canonical prep");
    let structure = r1cs.to_structure();
    let params = verifier_prep.params.clone();
    let wrong_log = owned_log(D, structure.m.div_ceil(D), params.kappa() as usize, 0xA170_0200);
    let prover_prep = preprocess_with_test_log(
        params,
        structure,
        wrong_log,
        direct_ccs::ajtai_rlc_mixer,
        direct_ccs::ajtai_dec_mixer,
        Some(r1cs.m_in),
    )
    .expect("same-shaped non-canonical prover prep");

    let z1 = satisfying_three_term_assignment();
    let mut z2 = satisfying_three_term_assignment();
    z2[1] = F::ZERO;
    z2[2] = F::ONE;

    let i1 = direct_ccs::build_instance(&prover_prep, &r1cs, &z1).expect("instance 1");
    let i2 = direct_ccs::build_instance(&prover_prep, &r1cs, &z2).expect("instance 2");
    let batches = FoldSchedule::RowsPerStep(1)
        .partition(vec![i1, i2])
        .expect("partition");
    let proof = prove(&prover_prep, batches).expect("prove with non-canonical setup");
    let finished = finish_uncompressed(&prover_prep, proof).expect("finish with non-canonical setup");

    verify_uncompressed(&prover_prep, &finished).expect("self-check with matching non-canonical setup");
    assert!(
        verify_uncompressed(&verifier_prep, &finished).is_err(),
        "canonical verifier accepted a proof generated with a different same-shaped Ajtai setup"
    );
}

#[test]
fn unsatisfied_assignment_reports_row() {
    let r1cs = three_term_addition();
    // (a, b, c) = (1, 0, 0): a + b = 1 ≠ 0 = c. Constraint row 0 should fail.
    let mut z = vec![F::default(); D];
    z[0] = F::ONE; // constant
    z[1] = F::ONE; // a
    z[2] = F::ZERO; // b
    z[3] = F::ZERO; // c
    match r1cs.is_satisfied_by(&z) {
        Err(FrontendError::Unsatisfied { row: 0 }) => {}
        other => panic!("expected Unsatisfied row 0, got {:?}", other),
    }
}

#[test]
fn build_instance_rejects_non_satisfying_assignment() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 11).expect("preprocess");

    // (a, b, c) = (1, 1, 0): 1 + 1 = 2 ≠ 0. The R1CS check fires *before* any commit.
    let mut z = vec![F::default(); prep.structure().m];
    z[0] = F::ONE;
    z[1] = F::ONE;
    z[2] = F::ONE;
    z[3] = F::ZERO;

    match direct_ccs::build_instance(&prep, &r1cs, &z) {
        Err(FrontendError::Unsatisfied { row: 0 }) => {}
        other => panic!("expected Unsatisfied row 0, got {:?}", other),
    }
}

#[test]
fn build_instance_rejects_norm_violation() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 12).expect("preprocess");

    // (a, b, c) = (2, 0, 2): a + b = 2 = c (R1CS satisfies), but ‖z‖_∞ = 2 ≥ b = 2.
    // The norm check fires AFTER R1CS satisfaction, so this exercises the
    // RelationError path.
    let mut z = vec![F::default(); prep.structure().m];
    z[0] = F::ONE;
    z[1] = F::from_u64(2);
    z[2] = F::ZERO;
    z[3] = F::from_u64(2);

    match direct_ccs::build_instance(&prep, &r1cs, &z) {
        Err(FrontendError::Relations(_)) => {}
        other => panic!("expected RelationError (norm bound), got {:?}", other),
    }
}

fn satisfying_three_term_assignment() -> Vec<F> {
    let mut z = vec![F::default(); D];
    z[0] = F::ONE;
    z[1] = F::ONE;
    z[2] = F::ZERO;
    z[3] = F::ONE;
    z
}

fn global_log(d: usize, cols: usize, kappa: usize, seed: u64) -> AjtaiSModule {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    set_global_pp_seeded(d, kappa, cols, seed_bytes).expect("install test Ajtai setup");
    AjtaiSModule::from_global_for_dims(d, cols).expect("test Ajtai module")
}

fn owned_log(d: usize, cols: usize, kappa: usize, seed: u64) -> AjtaiSModule {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = ChaCha8Rng::from_seed(seed_bytes);
    let pp = setup_par(&mut rng, d, kappa, cols).expect("owned test Ajtai setup");
    AjtaiSModule::new(Arc::new(pp))
}
