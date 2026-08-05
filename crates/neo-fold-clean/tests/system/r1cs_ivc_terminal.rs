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
    assignment_one_product, make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs,
    tiny_params,
};

#[test]
#[ignore = "full recursive fixed-point preprocessing exceeds the five-minute test cap"]
fn generic_ivc_verifies_running_accumulator_and_latest_f_prime() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let prep = R1csIvcPreprocessing::new_seeded(tiny_params(), &app, plan, 0x1F15_C007)
        .expect("compile authoritative generic R1CS IVC relation");
    assert!(prep.prep.enforces_terminal_induction());

    let mut chain = R1csIvc::new(&prep);
    for (step, (a, b)) in [(3, 7), (4, 9), (5, 11)].into_iter().enumerate() {
        chain
            .extend(assignment_one_product(a, b))
            .unwrap_or_else(|error| panic!("append satisfying app step {}: {error}", step + 1));
    }
    let proof = chain.finish().expect("finish compact HyperNova proof");

    assert!(proof.final_fold.is_none());
    let ProofState::Active { running, latest } = &proof.state.proof else {
        panic!("three-step IVC proof must be active");
    };
    let running = running.materialize().expect("materialized running state");
    assert!(!running.claims.is_empty());
    assert_eq!(latest.instances.len(), 1);
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .expect("terminal verifier accepts running accumulator plus latest F' instance");

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

    let mut bad_history = proof.clone();
    let ProofState::Active { running, .. } = &mut bad_history.state.proof else {
        unreachable!()
    };
    running
        .as_materialized_mut()
        .expect("materialized running state")
        .claims[0]
        .c
        .data[0] += F::ONE;
    neo_fold_clean::verify_uncompressed(&prep.prep, &bad_history).expect_err("a changed accumulated claim must fail");
}

#[test]
#[ignore = "full recursive fixed-point preprocessing exceeds the five-minute test cap"]
fn stateful_ivc_threads_the_authoritative_application_state() {
    let app = increment_r1cs();
    let initial = semantic_digest(1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        app.m(),
        app.m_in,
        vec![1],
        vec![2],
        Some(digest_fields_as_digest32(initial)),
    );
    let prep = R1csIvcPreprocessing::new_seeded(tiny_params(), &app, plan, 0x1F15_C008)
        .expect("compile authoritative stateful R1CS IVC relation");
    let mut chain = R1csIvc::new(&prep);
    chain
        .extend(increment_assignment(1))
        .expect("1 to 2 base step");
    chain
        .extend(increment_assignment(9))
        .expect_err("a disconnected recursive app input must fail");
    chain
        .extend(increment_assignment(2))
        .expect("2 to 3 recursive step");
    let proof = chain.finish().expect("finish stateful HyperNova proof");
    assert_eq!(
        digest32_as_fields(proof.state.semantic_state_digest),
        semantic_digest(3)
    );
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .expect("stateful running accumulator plus latest F' verifies");
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
    encode_poseidon_trace(&build_semantic_state_preimage_fields(&[F::from_u64(value)])).digest_native
}
