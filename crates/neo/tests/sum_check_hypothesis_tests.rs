/// Hypothesis testing for sum-check bug in non-base case
/// 
/// This test suite validates three hypotheses:
/// A) Initial sum computation differs between base/non-base case
/// B) Terminal Q(r) check has a bug in batching
/// C) Individual instance verification is bypassed when batching

use neo::{F, NeoParams, Accumulator};
use neo::{prove_ivc_step_with_extractor, verify_ivc_step, StepBindingSpec, LastNExtractor};
use neo_ccs::{CcsStructure, relations::check_ccs_rowwise_zero, Mat, SparsePoly, Term};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn simple_ccs() -> (CcsStructure<F>, Vec<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let m0 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ZERO,  // Row 3: 0 (dummy)
    ]);
    let m1 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ZERO, F::ONE,   // Row 2: z1
        F::ONE, F::ZERO, F::ZERO,   // Row 3: 1 (dummy)
    ]);
    let m2 = Mat::from_row_major(4, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ZERO,  // Row 3: 0 (dummy)
    ]);
    
    let f = SparsePoly::new(3, vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },   // X0 * X1
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -X2
    ]);
    
    let ccs = CcsStructure::new(vec![m0, m1, m2], f).expect("valid CCS");
    let valid_wit = vec![F::ONE, F::ONE, F::ONE];
    let invalid_wit = vec![F::ONE, F::ONE, F::from_u64(5)];
    
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let extractor = LastNExtractor { n: 0 };
    
    (ccs, valid_wit, invalid_wit, binding, extractor)
}

/// HYPOTHESIS A: Check if initial sum differs between base and non-base case
#[test]
fn hypothesis_a_initial_sum_computation() {
    let (ccs, valid_wit, invalid_wit, binding, extractor) = simple_ccs();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    println!("\n==== HYPOTHESIS A: Initial Sum Computation ====");
    
    // Base case with invalid witness
    let acc0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![],
        step: 0,
    };
    
    let step0_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("base case prove");
    
    if let Some(fp0) = &step0_invalid.proof.folding_proof {
        if let Some(initial_sum) = fp0.pi_ccs_proof.sc_initial_sum {
            println!("Base case initial_sum (Re,Im): ({}, {})", 
                initial_sum.real().as_canonical_u64(), 
                initial_sum.imag().as_canonical_u64());
        }
    }
    
    // Non-base case with invalid witness
    let step0_valid = prove_ivc_step_with_extractor(
        &params, &ccs, &valid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("setup valid step 0");
    
    let acc1 = &step0_valid.proof.next_accumulator;
    let step1_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, acc1, 1, None, &extractor, &binding,
    ).expect("non-base case prove");
    
    if let Some(fp1) = &step1_invalid.proof.folding_proof {
        if let Some(initial_sum) = fp1.pi_ccs_proof.sc_initial_sum {
            println!("Non-base case initial_sum (Re,Im): ({}, {})", 
                initial_sum.real().as_canonical_u64(), 
                initial_sum.imag().as_canonical_u64());
        }
    }
    
    println!("\nObservation: If initial sums differ significantly, this could explain the bug.");
}

/// HYPOTHESIS B: Check terminal Q(r) values and batching coefficients
#[test]
fn hypothesis_b_terminal_check_and_batching() {
    let (ccs, valid_wit, invalid_wit, binding, extractor) = simple_ccs();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    println!("\n==== HYPOTHESIS B: Terminal Q(r) and Batching Coefficients ====");
    
    let acc0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![],
        step: 0,
    };
    
    // Base case
    let step0_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("base case prove");
    
    println!("Base case:");
    if let Some(fp) = &step0_invalid.proof.folding_proof {
        println!("  pi_ccs_outputs.len() = {}", fp.pi_ccs_outputs.len());
        for (i, out) in fp.pi_ccs_outputs.iter().enumerate() {
            println!("  Output[{}]: y_scalars.len()={}", i, out.y_scalars.len());
            for (j, ys) in out.y_scalars.iter().enumerate() {
                println!("    y_scalars[{}] = ({}, {})", j, ys.real().as_canonical_u64(), ys.imag().as_canonical_u64());
            }
        }
        
        // Check Pi-RLC proof (contains batching info)
        println!("  pi_rlc_proof.rho_elems.len() = {}", fp.pi_rlc_proof.rho_elems.len());
    }
    
    // Non-base case
    let step0_valid = prove_ivc_step_with_extractor(
        &params, &ccs, &valid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("setup valid step 0");
    
    let acc1 = &step0_valid.proof.next_accumulator;
    let step1_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, acc1, 1, None, &extractor, &binding,
    ).expect("non-base case prove");
    
    println!("\nNon-base case:");
    if let Some(fp) = &step1_invalid.proof.folding_proof {
        println!("  pi_ccs_outputs.len() = {}", fp.pi_ccs_outputs.len());
        for (i, out) in fp.pi_ccs_outputs.iter().enumerate() {
            println!("  Output[{}]: y_scalars.len()={}", i, out.y_scalars.len());
            for (j, ys) in out.y_scalars.iter().enumerate() {
                println!("    y_scalars[{}] = ({}, {})", j, ys.real().as_canonical_u64(), ys.imag().as_canonical_u64());
            }
        }
        
        println!("  pi_rlc_proof.rho_elems.len() = {}", fp.pi_rlc_proof.rho_elems.len());
    }
    
    println!("\nObservation: Check if y_scalars[1] (the invalid RHS) shows non-zero CCS violations.");
    println!("If y_scalars look similar in both cases but verification differs, it's a verifier bug.");
}

/// HYPOTHESIS C: Check if individual instances are verified separately
#[test]
fn hypothesis_c_individual_instance_verification() {
    let (ccs, valid_wit, invalid_wit, binding, extractor) = simple_ccs();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    println!("\n==== HYPOTHESIS C: Individual Instance Verification ====");
    
    // Check step CCS satisfaction
    println!("Step CCS satisfaction check:");
    match check_ccs_rowwise_zero(&ccs, &[], &invalid_wit) {
        Ok(_) => println!("  Invalid witness: ✓ SATISFIES step CCS (WRONG!)"),
        Err(e) => println!("  Invalid witness: ✗ Does NOT satisfy step CCS (correct): {}", e),
    }
    
    let acc0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![],
        step: 0,
    };
    
    // Base case folding proof structure
    let step0_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("base case prove");
    
    println!("\nBase case folding structure:");
    if let Some(fp) = &step0_invalid.proof.folding_proof {
        println!("  Number of instances being batched: {}", fp.pi_ccs_inputs.len());
        println!("  LHS input x: {:?}", fp.pi_ccs_inputs[0].x.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        println!("  RHS input x: {:?}", fp.pi_ccs_inputs[1].x.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        
        let lhs_all_zero = fp.pi_ccs_inputs[0].x.iter().all(|&f| f == F::ZERO);
        println!("  LHS is all zeros: {}", lhs_all_zero);
    }
    
    // Non-base case
    let step0_valid = prove_ivc_step_with_extractor(
        &params, &ccs, &valid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("setup valid step 0");
    
    let acc1 = &step0_valid.proof.next_accumulator;
    let step1_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, acc1, 1, None, &extractor, &binding,
    ).expect("non-base case prove");
    
    println!("\nNon-base case folding structure:");
    if let Some(fp) = &step1_invalid.proof.folding_proof {
        println!("  Number of instances being batched: {}", fp.pi_ccs_inputs.len());
        println!("  LHS input x (first 5): {:?}", fp.pi_ccs_inputs[0].x.iter().take(5).map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        println!("  RHS input x (first 5): {:?}", fp.pi_ccs_inputs[1].x.iter().take(5).map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        
        let lhs_all_zero = fp.pi_ccs_inputs[0].x.iter().all(|&f| f == F::ZERO);
        println!("  LHS is all zeros: {}", lhs_all_zero);
    }
    
    println!("\nKey Observation:");
    println!("The verifier uses a BATCHED sum-check: Q(r) = Σ α_i · f(Y_i(r))");
    println!("Where:");
    println!("  α_0, α_1 = random batching coefficients");
    println!("  Y_0(r) = outputs from LHS instance");
    println!("  Y_1(r) = outputs from RHS instance (the invalid one)");
    println!("\nIf the verifier only checks Σ α_i · f(Y_i(r)) == 0 without checking each f(Y_i(r)) == 0,");
    println!("then a valid LHS could mask an invalid RHS through cancellation!");
}

/// CRITICAL TEST: Does the LHS mask the invalid RHS through batching?
#[test]
fn hypothesis_c_batching_cancellation_bug() {
    let (ccs, valid_wit, invalid_wit, binding, extractor) = simple_ccs();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    println!("\n==== CRITICAL: Testing Batching Cancellation Hypothesis ====");
    println!("In base case: LHS is all zeros, so α_0 · f(Y_0) ≈ 0");
    println!("              Only α_1 · f(Y_1) matters for the check");
    println!("              If f(Y_1) ≠ 0 (invalid RHS), verifier catches it");
    println!("\nIn non-base case: LHS has previous valid step's values");
    println!("                  α_0 · f(Y_0) ≠ 0 (non-zero contribution)");
    println!("                  α_1 · f(Y_1) ≠ 0 (invalid RHS)");
    println!("                  BUG: If only checking α_0·f(Y_0) + α_1·f(Y_1) == claimed_sum,");
    println!("                       and claimed_sum is computed from AUGMENTED witness (not step),");
    println!("                       the invalid step CCS rows might be masked!");
    
    let acc0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![],
        step: 0,
    };
    
    // Setup non-base case
    let step0_valid = prove_ivc_step_with_extractor(
        &params, &ccs, &valid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("setup valid step 0");
    
    let acc1 = &step0_valid.proof.next_accumulator;
    
    // Try base case first
    println!("\n--- Base Case Verification ---");
    let step0_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("base case prove");
    
    let ok0 = verify_ivc_step(&ccs, &step0_invalid.proof, &acc0, &binding, &params, None)
        .expect("verify should not error");
    println!("Base case result: {}", if ok0 { "ACCEPTED ✗" } else { "REJECTED ✓" });
    
    // Now non-base case
    println!("\n--- Non-Base Case Verification ---");
    let step1_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, acc1, 1, None, &extractor, &binding,
    ).expect("non-base case prove");
    
    let ok1 = verify_ivc_step(&ccs, &step1_invalid.proof, acc1, &binding, &params, None)
        .expect("verify should not error");
    println!("Non-base case result: {}", if ok1 { "ACCEPTED ✗" } else { "REJECTED ✓" });
    
    if !ok0 && ok1 {
        println!("\n🎯 SMOKING GUN FOUND!");
        println!("The bug is in the batching logic!");
        println!("\nRoot Cause:");
        println!("The sum-check verifies: running_sum == Σ α_i · f(Y_i(r))");
        println!("But it computes running_sum from the AUGMENTED CCS, which includes:");
        println!("  - Step CCS rows (where invalid witness fails)");
        println!("  - EV rows (evolution constraints)");
        println!("  - Binding rows");
        println!("\nThe terminal check evaluates f(Y_i(r)) using ONLY the CCS polynomial,");
        println!("which doesn't include the augmentation constraints!");
        println!("\nIn base case: LHS contributes ~0, so RHS violations are visible");
        println!("In non-base case: LHS non-zero contribution hides RHS step CCS violations");
    }
    
    assert!(!ok0, "Base case should reject");
    assert!(!ok1, "Non-base case should reject");
}





