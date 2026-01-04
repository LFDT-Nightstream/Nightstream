#![allow(non_snake_case)]

use neo_ajtai::AjtaiSModule;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::FoldingSession;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
#[cfg(feature = "paper-exact")]
fn test_session_single_fold_with_crosscheck() {
    use neo_reductions::engines::CrosscheckCfg;

    // Use the same R1CS structure as the paper_exact test
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

    // Valid witness: x=[1,1,1], w=[2,2]
    let public_input = vec![F::ONE, F::ONE, F::ONE]; // x0,x1,x2
    let witness = vec![F::from_u64(2), F::from_u64(2)]; // w0,w1

    // Configure crosscheck with selective checks
    let crosscheck_cfg = CrosscheckCfg {
        fail_fast: true,   // Stop on first error
        initial_sum: true, // Check initial sum
        per_round: true,   // Per-round checks
        terminal: true,    // Terminal identity check
        outputs: true,     // Check output instances
    };

    // Create session with crosscheck mode
    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(
        FoldingMode::OptimizedWithCrosscheck(crosscheck_cfg),
        &ccs,
    )
    .expect("new_ajtai");
    session
        .add_step_io(&ccs, &public_input, &witness)
        .expect("add_step_io should succeed");

    let _run = session
        .prove_and_verify_collected(&ccs)
        .expect("prove_and_verify_collected should succeed");

    println!("Crosscheck verification passed!");
}
