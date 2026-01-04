#![allow(non_snake_case)]

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{me_from_z_balanced, Accumulator, FoldingSession};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    set_global_pp(pp).expect("set_global_pp");
}

/// Build a 5×5 R1CS with *non-trivial* linear combos:
/// Variables z = [x0, x1, x2, w0, w1]
/// Constraints:
///   (1) (x0 + x1) * (x2) = w0
///   (2) (w0)       * (x1) = w1
///   (3) x0 * x0 = x0   (booleanize x0)
///   (4) x1 * x1 = x1   (booleanize x1)
///   (5) x2 * x2 = x2   (booleanize x2)
///
/// Valid examples:
///   - x = [1,1,1], w = [2,2]
///   - x = [1,0,1], w = [1,0]
///   - x = [0,1,1], w = [1,1]
#[test]
#[cfg(feature = "paper-exact")]
fn test_session_multifold_k3_three_steps_r1cs_paper_exact_nontrivial() {
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

    // Row 2: x0 * x0 = x0
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;

    // Row 3: x1 * x1 = x1
    A[(3, 1)] = F::ONE;
    B[(3, 1)] = F::ONE;
    C[(3, 1)] = F::ONE;

    // Row 4: x2 * x2 = x2
    A[(4, 2)] = F::ONE;
    B[(4, 2)] = F::ONE;
    C[(4, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(A, B, C);

    let params =
        NeoParams::goldilocks_auto_r1cs_ccs(n_constraints).expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(n_vars);
    let l = AjtaiSModule::from_global_for_dims(D, n_vars).expect("AjtaiSModule init");

    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let ell_n = dims.ell_n;

    let r: Vec<K> = vec![K::from(F::from_u64(5)); ell_n];

    let m_in = 3;

    // Seed 1: x=[1,1,1], w=[2,2]
    let z_seed_1: Vec<F> = vec![F::ONE, F::ONE, F::ONE, F::from_u64(2), F::from_u64(2)];
    // Seed 2: x=[1,0,1], w=[1,0]
    let z_seed_2: Vec<F> = vec![F::ONE, F::ZERO, F::ONE, F::ONE, F::ZERO];

    let (me1, Z1) = me_from_z_balanced(&params, &ccs, &l, &z_seed_1, &r, m_in).expect("seed1 ME ok");
    let (me2, Z2) = me_from_z_balanced(&params, &ccs, &l, &z_seed_2, &r, m_in).expect("seed2 ME ok");

    let acc = Accumulator {
        me: vec![me1, me2],
        witnesses: vec![Z1, Z2],
    };

    let mut session = FoldingSession::new(FoldingMode::PaperExact, params, l.clone())
        .with_initial_accumulator(acc, &ccs)
        .expect("with_initial_accumulator");

    // Step 1: x = [1,1,1], w = [2,2]
    {
        let x: Vec<F> = vec![F::ONE, F::ONE, F::ONE];
        let w: Vec<F> = vec![F::from_u64(2), F::from_u64(2)];
        session.add_step_io(&ccs, &x, &w).expect("add_step 1");
    }

    // Step 2: x = [1,0,1], w = [1,0]
    {
        let x: Vec<F> = vec![F::ONE, F::ZERO, F::ONE];
        let w: Vec<F> = vec![F::ONE, F::ZERO];
        session.add_step_io(&ccs, &x, &w).expect("add_step 2");
    }

    // Step 3: x = [0,1,1], w = [1,1]
    {
        let x: Vec<F> = vec![F::ZERO, F::ONE, F::ONE];
        let w: Vec<F> = vec![F::ONE, F::ONE];
        session.add_step_io(&ccs, &x, &w).expect("add_step 3");
    }

    let run = session
        .fold_and_prove(&ccs)
        .expect("fold_and_prove should produce a FoldRun");

    assert_eq!(run.steps.len(), 3, "should have three fold steps");

    for (i, step) in run.steps.iter().enumerate() {
        assert_eq!(
            step.fold.dec_children.len(),
            params.k_rho as usize,
            "step {} should have k_rho={} DEC children",
            i,
            params.k_rho
        );
    }

    // Final outputs are the dec_children of the last step
    let final_outputs = &run.steps.last().unwrap().fold.dec_children;
    assert_eq!(
        final_outputs.len(),
        params.k_rho as usize,
        "final outputs should have k_rho={}",
        params.k_rho
    );

    let mcss_public = session.mcss_public();
    session.unsafe_allow_unlinked_steps();
    let ok = session
        .verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "paper-exact verification should pass");
}
