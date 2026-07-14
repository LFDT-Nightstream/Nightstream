#[path = "../support/mod.rs"]
mod support;

use neo_ccs::matrix::Mat;
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::r1cs_f_prime::{self, R1csChainBuilder};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_fold_clean::paper::nifs::CpuNifsProver;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use support::r1cs_compiler_fixtures::{
    make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor, tiny_params,
};

fn max_one_fresh_params() -> neo_fold_clean::paper::params::Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        4,
        1u64 << 16,
        2,
        2,
        1,
        2,
        60,
    )
    .expect("test parameters must admit exactly one fresh input");
    let params = neo_fold_clean::paper::params::Params::test_only_from_neo_params(inner);
    assert_eq!(params.max_fresh_count(), 1);
    params
}

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
    let mut adapter = CpuNifsProver;

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
fn cpu_builder_recovers_after_rejected_oversized_base_chunk() {
    let mut r1cs = fibonacci_transition_r1cs();
    r1cs.m_in = 0;
    let plan = make_tiny_lifecycle_plan(r1cs.m(), r1cs.m_in);
    let prep = r1cs_f_prime::preprocess_seeded_with_params(&r1cs, &plan, max_one_fresh_params(), 0xF5_17_0097)
        .expect("preprocess stateless lifecycle fixture");
    let mut chain = R1csChainBuilder::new(&prep).expect("start chain");

    let rejected = chain
        .append_assignments(vec![fibonacci_assignment(1, 1), fibonacci_assignment(1, 1)])
        .expect_err("fixture must exercise the lifecycle max-fresh rejection");
    assert!(matches!(
        rejected,
        r1cs_f_prime::Error::Lifecycle(neo_fold_clean::lifecycle::Error::BatchTooLarge { got: 2, max: 1 })
    ));

    chain
        .append_assignment(fibonacci_assignment(1, 1))
        .expect("a rejected base chunk must leave a fresh builder reusable");
    let audit = chain
        .finish_with_audit()
        .expect("recovered chain should finalize");
    let verified = neo_fold_clean::verify_uncompressed_audit(&prep.prep, &audit);
    assert!(
        verified.is_ok(),
        "a valid retry after a rejected base chunk must produce a verifier-accepted audit; got {verified:?}"
    );
}
