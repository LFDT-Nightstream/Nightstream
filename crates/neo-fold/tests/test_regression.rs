#![allow(non_snake_case)]

use neo_fold::session::{FoldingSession, ProveInput};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::Mat;
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use rand_chacha::rand_core::SeedableRng;
use neo_params::NeoParams;
use neo_math::{F, D};
use p3_field::PrimeCharacteristicRing;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

#[test]
fn test_regression_m0_non_identity() {
    // Regression test: Ensure optimized oracle works when M_0 is NOT identity.
    // We construct a CCS where M_0 (A) has off-diagonal elements.
    
    let n_constraints = 3usize;
    let n_vars = 3usize;
    
    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);
    
    // Row 0: (x0 + x1) * x2 = x0
    // M_0 (A) row 0 has 1s at col 0 and 1. This is NOT identity.
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    B[(0, 2)] = F::ONE;
    C[(0, 0)] = F::ONE;
    
    // Row 1: x1 * x1 = x1
    A[(1, 1)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 1)] = F::ONE;
    
    // Row 2: x2 * x2 = x2
    A[(2, 2)] = F::ONE;
    B[(2, 2)] = F::ONE;
    C[(2, 2)] = F::ONE;
    
    let ccs = neo_ccs::r1cs_to_ccs(A, B, C);
    
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n_constraints)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");
    
    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    // Valid witness: x=[1, 0, 1] (x0=1, x1=0, x2=1)
    // Row 0: (1+0)*1 = 1 (ok)
    // Row 1: 0*0 = 0 (ok)
    // Row 2: 1*1 = 1 (ok)
    let public_input = vec![F::ONE, F::ZERO, F::ONE]; 
    let witness = vec![]; // no private witness

    // Create session with optimized mode
    let mut session = FoldingSession::new(
        FoldingMode::Optimized,
        params,
        l.clone(),
    );

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
    assert!(ok, "session verification should pass with non-identity M_0");
}

#[test]
fn test_regression_with_witness() {
    // Regression test: Ensure optimized oracle works with non-empty witness.
    let n_constraints = 3usize;
    let n_vars = 3usize;
    
    let mut A = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut B = Mat::zero(n_constraints, n_vars, F::ZERO);
    let mut C = Mat::zero(n_constraints, n_vars, F::ZERO);
    
    // Row 0: x0 * x1 = w0
    A[(0, 0)] = F::ONE;
    B[(0, 1)] = F::ONE;
    C[(0, 2)] = F::ONE; // w0 is at index 2 (mapped to x2 for simplicity in this R1CS setup)
    
    // Row 1: w0 * w0 = w0 (bool check)
    A[(1, 2)] = F::ONE;
    B[(1, 2)] = F::ONE;
    C[(1, 2)] = F::ONE;
    
    // Row 2: x0 * x0 = x0 (bool check)
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;
    
    let ccs = neo_ccs::r1cs_to_ccs(A, B, C);
    
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n_constraints)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");
    
    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    // Valid witness: x=[1, 1], w=[1]
    // x0=1, x1=1 => w0=1
    // w0*w0 = 1*1 = 1 (ok)
    // x0*x0 = 1*1 = 1 (ok)
    // In R1CS->CCS, vars are [x0, x1, w0]
    let public_input = vec![F::ONE, F::ONE]; 
    let witness = vec![F::ONE];

    let mut session = FoldingSession::new(
        FoldingMode::Optimized,
        params,
        l.clone(),
    );

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

    let public_mcss = session.mcss_public();
    let ok = session
        .verify(&ccs, &public_mcss, &run)
        .expect("verify should run");
    assert!(ok, "session verification should pass with witness");
}
