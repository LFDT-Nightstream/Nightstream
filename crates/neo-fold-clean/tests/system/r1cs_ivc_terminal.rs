//! End-to-end terminal checks for the selected generic R1CS IVC path.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvc, R1csIvcPreprocessing};
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::digest::{digest32_as_fields, digest_fields_as_digest32};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor,
    minimal_ivc_test_params, one_product_r1cs,
};

#[test]
fn generic_ivc_verifies_one_authoritative_f_prime_step() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let prep = R1csIvcPreprocessing::new_seeded(minimal_ivc_test_params(), &app, plan, 0x1F15_C007)
        .expect("compile authoritative generic R1CS IVC relation");
    assert!(prep.prep.enforces_terminal_induction());

    let mut chain = R1csIvc::new(&prep);
    chain
        .extend(assignment_one_product(3, 7))
        .expect("append one satisfying application step");
    let proof = chain.finish().expect("finish compact HyperNova proof");

    assert!(proof.final_fold.is_none());
    let ProofState::Active { running, latest } = &proof.state.proof else {
        panic!("one-step IVC proof must be active");
    };
    assert_eq!(running.claims.len(), prep.prep.params.k_rho() as usize);
    assert_eq!(latest.instances.len(), 1);
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .expect("terminal verifier accepts the authoritative F-prime instance");

    let mut bad_latest = proof.clone();
    let ProofState::Active { latest, .. } = &mut bad_latest.state.proof else {
        unreachable!()
    };
    let instance = &mut latest.instances[0];
    let global_column = instance.claim.m_in;
    let packed_coordinate = (global_column % neo_math::D, global_column / neo_math::D);
    instance.witness.Z[packed_coordinate] = if instance.witness.Z[packed_coordinate] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    instance.claim.c = prep.prep.log.commit(&instance.witness.Z);
    neo_fold_clean::verify_uncompressed(&prep.prep, &bad_latest)
        .expect_err("a recommitted invalid latest witness must fail the relation");
}

#[test]
fn stateful_ivc_binds_the_initial_application_state() {
    let app = increment_r1cs();
    let initial = semantic_digest(1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        app.m(),
        app.m_in,
        vec![0, 1],
        vec![0, 2],
        Some(digest_fields_as_digest32(initial)),
    );
    let prep = R1csIvcPreprocessing::new_seeded(minimal_ivc_test_params(), &app, plan, 0x1F15_C008)
        .expect("compile authoritative stateful R1CS IVC relation");
    let mut chain = R1csIvc::new(&prep);
    chain
        .extend(increment_assignment(1))
        .expect("1 to 2 base step");
    chain
        .extend(increment_assignment(9))
        .expect_err("a disconnected application input must fail");
    let proof = chain.finish().expect("finish stateful HyperNova proof");
    assert_eq!(
        digest32_as_fields(proof.state.semantic_state_digest),
        semantic_digest(2)
    );
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof).expect("stateful authoritative F-prime instance verifies");
}

fn increment_r1cs() -> R1cs {
    let mut a = Mat::zero(1, neo_math::D, F::ZERO);
    let mut b = Mat::zero(1, neo_math::D, F::ZERO);
    let mut c = Mat::zero(1, neo_math::D, F::ZERO);
    a[(0, 0)] = F::ONE;
    a[(0, 1)] = F::ONE;
    b[(0, 0)] = F::ONE;
    c[(0, 2)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

fn increment_assignment(input: u64) -> Vec<F> {
    let mut assignment = vec![F::ZERO; neo_math::D];
    assignment[0] = F::ONE;
    assignment[1] = F::from_u64(input);
    assignment[2] = F::from_u64(input + 1);
    assignment
}

fn semantic_digest(value: u64) -> [F; 4] {
    encode_poseidon_trace(&build_semantic_state_preimage_fields(&[F::ONE, F::from_u64(value)])).digest_native
}
