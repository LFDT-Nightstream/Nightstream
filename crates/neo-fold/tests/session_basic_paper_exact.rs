#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, ProveInput};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

#[test]
#[cfg(feature = "paper-exact")]
fn test_session_single_fold_exact_paper() {
    // Use the same R1CS structure as the nontrivial test that works
    let n_constraints = 5usize;
    let n_vars = 5usize;

    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);

    // Row 0: (x0 + x1) * (x2) = w0
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    B[(0, 2)] = F::ONE;
    C[(0, 3)] = F::ONE;

    // Row 1: (w0) * (x1) = w1
    A[(1, 3)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 4)] = F::ONE;

    // Row 2: x0 * x0 = x0 (bool)
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;

    // Row 3: x1 * x1 = x1 (bool)
    A[(3, 1)] = F::ONE;
    B[(3, 1)] = F::ONE;
    C[(3, 1)] = F::ONE;

    // Row 4: x2 * x2 = x2 (bool)
    A[(4, 2)] = F::ONE;
    B[(4, 2)] = F::ONE;
    C[(4, 2)] = F::ONE;

    let ccs = neo_ccs::r1cs_to_ccs(A, B, C);

    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(n_constraints).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    // Valid witness: x=[1,1,1], w=[2,2]
    let public_input = vec![F::ONE, F::ONE, F::ONE]; // x0,x1,x2
    let witness = vec![F::from_u64(2), F::from_u64(2)]; // w0,w1

    // Create session
    let mut session = FoldingSession::new(FoldingMode::PaperExact, params, l.clone());

    // Use ProveInput directly
    let input = ProveInput {
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    };

    session
        .add_step_from_io(&input)
        .expect("add_step should succeed");

    let run = session
        .fold_and_prove(&ccs)
        .expect("fold_and_prove should produce a FoldRun");

    // Verify
    let public_mcss = session.mcss_public();
    let ok = session
        .verify(&ccs, &public_mcss, &run)
        .expect("verify should run");
    assert!(ok, "session verification should pass");
}
