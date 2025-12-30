#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, ProveInput};
use neo_fold::shard::StepLinkingConfig;
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn boolean_ccs(n: usize) -> neo_ccs::CcsStructure<F> {
    // z_i^2 = z_i for i=0..n-1
    let mut A = Mat::zero(n, n, F::ZERO);
    let mut B = Mat::zero(n, n, F::ZERO);
    let mut C = Mat::zero(n, n, F::ZERO);
    for i in 0..n {
        A[(i, i)] = F::ONE;
        B[(i, i)] = F::ONE;
        C[(i, i)] = F::ONE;
    }
    r1cs_to_ccs(A, B, C)
}

#[test]
fn test_session_multi_step_verification_requires_step_linking_or_unsafe_opt_out() {
    let n = 3usize;
    let ccs = boolean_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n).expect("AjtaiSModule init");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);

    // Two individually valid steps with mismatched public x[1].
    for &x1 in &[F::ZERO, F::ONE] {
        let x = vec![F::ONE, x1];
        let w = vec![F::ZERO];
        let input = ProveInput {
            ccs: &ccs,
            public_input: &x,
            witness: &w,
            output_claims: &[],
        };
        session.add_step_from_io(&input).expect("add_step");
    }

    let run = session.fold_and_prove(&ccs).expect("fold_and_prove");
    let mcss_public = session.mcss_public();

    // Default behavior: multi-step verification must not silently skip chaining.
    assert!(
        session.verify(&ccs, &mcss_public, &run).is_err(),
        "expected missing step linking to be rejected"
    );

    // Explicit escape hatch: allow unlinked verification.
    session.unsafe_allow_unlinked_steps();
    assert!(
        session.verify(&ccs, &mcss_public, &run).expect("verify runs"),
        "expected verification to succeed with unsafe opt-out"
    );

    // When step linking is enabled, the mismatch must be detected.
    session.set_step_linking(StepLinkingConfig::new(vec![(1, 1)]));
    assert!(
        session.verify(&ccs, &mcss_public, &run).is_err(),
        "expected step linking mismatch to be rejected"
    );
}

#[test]
fn test_session_step_linking_allows_valid_chaining() {
    let n = 3usize;
    let ccs = boolean_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    setup_ajtai_for_dims(n);
    let l = AjtaiSModule::from_global_for_dims(D, n).expect("AjtaiSModule init");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);
    session.set_step_linking(StepLinkingConfig::new(vec![(1, 1)]));

    // Two valid steps with matching x[1].
    for _ in 0..2 {
        let x = vec![F::ONE, F::ZERO];
        let w = vec![F::ZERO];
        let input = ProveInput {
            ccs: &ccs,
            public_input: &x,
            witness: &w,
            output_claims: &[],
        };
        session.add_step_from_io(&input).expect("add_step");
    }

    let run = session.fold_and_prove(&ccs).expect("fold_and_prove");
    let mcss_public = session.mcss_public();
    assert!(
        session.verify(&ccs, &mcss_public, &run).expect("verify runs"),
        "expected verification to succeed with step linking enabled"
    );
}
