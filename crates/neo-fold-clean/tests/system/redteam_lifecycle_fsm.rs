#[path = "../support/mod.rs"]
mod support;

use neo_ccs::matrix::Mat;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::nifs::OptimizedCpuNifsProver;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use support::r1cs_compiler_fixtures::{
    make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor, tiny_params,
};

fn fibonacci_transition_r1cs() -> R1cs {
    let mut a = Mat::zero(2, neo_math::D, F::ZERO);
    a[(0, 2)] = F::ONE;
    a[(1, 1)] = F::ONE;
    a[(1, 2)] = F::ONE;
    let mut b = Mat::zero(2, neo_math::D, F::ZERO);
    b[(0, 0)] = F::ONE;
    b[(1, 0)] = F::ONE;
    let mut c = Mat::zero(2, neo_math::D, F::ZERO);
    c[(0, 3)] = F::ONE;
    c[(1, 4)] = F::ONE;
    R1cs { a, b, c, m_in: 5 }
}

fn fibonacci_assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; neo_math::D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(b);
    z[4] = F::from_u64(a + b);
    z
}

fn semantic_digest(a: u64, b: u64) -> [u8; 32] {
    let preimage = neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields(&[
        F::from_u64(a),
        F::from_u64(b),
    ]);
    neo_fold_clean::paper::digest::digest_fields_as_digest32(encode_poseidon_trace(&preimage).digest_native)
}

#[test]
#[ignore = "the adapter-backed recursive production relation exceeds the five-minute test cap; run this audit alone with `cargo test --release -p neo-fold-clean --test system_redteam_lifecycle_fsm adapter_backed_builder_recovers_after_rejected_recursive_append -- --ignored --exact`"]
fn adapter_backed_builder_recovers_after_rejected_recursive_append() {
    let r1cs = fibonacci_transition_r1cs();
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        r1cs.m(),
        r1cs.m_in,
        vec![1, 2],
        vec![3, 4],
        Some(semantic_digest(1, 1)),
    );
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, tiny_params(), 0xF5_17_0095)
        .expect("preprocess lifecycle fixture");
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");
    let mut adapter = OptimizedCpuNifsProver;

    chain
        .append_assignment_with_nifs_adapter(fibonacci_assignment(1, 1), &mut adapter)
        .expect("base append");

    let mut unsatisfied = fibonacci_assignment(1, 2);
    unsatisfied[4] += F::ONE;
    assert!(
        chain
            .append_assignment_with_nifs_adapter(unsatisfied, &mut adapter)
            .is_err(),
        "fixture must exercise a recoverable recursive compile error"
    );

    chain
        .append_assignment_with_nifs_adapter(fibonacci_assignment(1, 2), &mut adapter)
        .expect("builder should remain usable after returning Err");
    let audit = chain
        .finish_with_audit_and_nifs_adapter(&mut adapter)
        .expect("recovered chain should finalize");

    let verified = neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit);
    assert!(
        verified.is_ok(),
        "a public builder that successfully recovers and finalizes must produce a verifier-accepted audit; got {verified:?}"
    );
}

#[test]
fn cpu_builder_rejected_oversized_base_chunk_is_transactional() {
    let mut r1cs = fibonacci_transition_r1cs();
    r1cs.m_in = 0;
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let params = tiny_params();
    let max_fresh = params.max_fresh_count();
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, params, 0xF5_17_0097)
        .expect("preprocess stateless lifecycle fixture");
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");

    let assignment = fibonacci_assignment(1, 1);
    let rejected = chain
        .append_assignments(vec![assignment.clone(); max_fresh + 1])
        .expect_err("fixture must exercise the lifecycle max-fresh rejection");
    assert!(matches!(
        rejected,
        r1cs_f_prime::Error::Lifecycle(neo_fold_clean::lifecycle::Error::BatchTooLarge { got, max })
            if got == max_fresh + 1 && max == max_fresh
    ));

    assert!(
        chain.audit().is_none(),
        "a rejected base chunk must not create an audit"
    );
    assert_eq!(
        chain.context().chain_state.chunk_count,
        0,
        "a rejected base chunk must not advance the chunk counter"
    );
    assert_eq!(
        chain.context().chain_state.step_count,
        0,
        "a rejected base chunk must not advance the step counter"
    );
}
