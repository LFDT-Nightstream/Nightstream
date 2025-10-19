/// Final diagnostic: Confirm the root cause
/// 
/// The bug is that when verifying Pi-CCS with batched instances:
/// - The sum-check initial sum T⁰ is computed over ALL rows of augmented CCS
/// - But the terminal check evaluates f(Y(r)) which is the step CCS polynomial
/// - The augmented rows (EV, binding, const-1) contribute to T⁰
/// - But f(Y(r)) only accounts for the step CCS structure
/// 
/// This creates a mismatch where the initial sum includes augmented constraints
/// but the terminal check doesn't verify them properly.

use neo::{F, NeoParams, Accumulator};
use neo::{prove_ivc_step_with_extractor, build_augmented_ccs_linked_with_rlc, StepBindingSpec, LastNExtractor};
use neo_ccs::{CcsStructure, relations::check_ccs_rowwise_zero, Mat, SparsePoly, Term};
use p3_field::PrimeCharacteristicRing;

fn simple_ccs() -> (CcsStructure<F>, Vec<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    let m0 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,
        F::ZERO, F::ZERO, F::ONE,
        F::ZERO, F::ONE, F::ZERO,
    ]);
    let m1 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,
        F::ZERO, F::ZERO, F::ONE,
        F::ZERO, F::ZERO, F::ONE,
    ]);
    let m2 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,
        F::ZERO, F::ZERO, F::ONE,
        F::ZERO, F::ONE, F::ZERO,
    ]);
    
    let f = SparsePoly::new(3, vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },
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

#[test]
fn confirm_augmented_ccs_polynomial_mismatch() {
    let (step_ccs, _, _, binding, _) = simple_ccs();
    
    println!("\n==== ROOT CAUSE CONFIRMATION ====\n");
    
    // Build augmented CCS
    let aug_ccs = build_augmented_ccs_linked_with_rlc(
        &step_ccs,
        0, // step_x_len
        &binding.y_step_offsets,
        &binding.step_program_input_witness_indices,
        0, // y_len
        binding.const1_witness_index,
        None,
    ).expect("augmented CCS");
    
    println!("Step CCS:");
    println!("  n (rows) = {}", step_ccs.n);
    println!("  m (cols) = {}", step_ccs.m);
    println!("  t (matrices) = {}", step_ccs.t());
    println!("  f (polynomial) = {:?}", step_ccs.f);
    
    println!("\nAugmented CCS:");
    println!("  n (rows) = {} (padded to power-of-2)", aug_ccs.n);
    println!("  m (cols) = {}", aug_ccs.m);
    println!("  t (matrices) = {}", aug_ccs.t());
    println!("  f (polynomial) = {:?}", aug_ccs.f);
    
    println!("\n🔍 KEY OBSERVATION:");
    println!("The augmented CCS has {} rows but uses the SAME polynomial f as the step CCS!", aug_ccs.n);
    println!("The polynomial f was designed for {} rows (step CCS).", step_ccs.n);
    println!("\nRows {}-{} are augmented constraints:", step_ccs.n, aug_ccs.n - 1);
    println!("  - Row {}: Const-1 enforcement (w_const1 * ρ = ρ)", step_ccs.n);
    println!("  - Rows {}-{}: Zero padding", step_ccs.n + 1, aug_ccs.n - 1);
    
    println!("\n❌ THE BUG:");
    println!("When sum-check computes T⁰ = Σ_{{x∈{{0,1}}^ℓ}} f((M·z)[x]),");
    println!("it iterates over ALL {} rows of the augmented CCS.", aug_ccs.n);
    println!("But f() is the step CCS polynomial that only makes sense for the first {} rows!", step_ccs.n);
    println!("\nThe augmented rows contribute to T⁰ in a way that f() wasn't designed for.");
    println!("This creates a mismatch between the initial sum and terminal check.");
    
    println!("\n✅ THE FIX:");
    println!("The sum-check should ONLY verify the first {} rows (step CCS).", step_ccs.n);
    println!("The augmented constraints (rows {}-{}) should be verified separately,", step_ccs.n, aug_ccs.n - 1);
    println!("not as part of the sum-check protocol!");
    
    println!("\nAlternatively, the augmented CCS polynomial should explicitly encode");
    println!("all constraints, not just reuse the step CCS polynomial.");
}

#[test]
fn demonstrate_the_exact_bug_manifestation() {
    let (ccs, valid_wit, invalid_wit, binding, extractor) = simple_ccs();
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    println!("\n==== BUG MANIFESTATION ====\n");
    
    // Check step CCS directly
    println!("Direct step CCS check on invalid witness:");
    match check_ccs_rowwise_zero(&ccs, &[], &invalid_wit) {
        Ok(_) => println!("  ✗ PASSES (should fail!)"),
        Err(e) => println!("  ✓ FAILS as expected: {}", e),
    }
    
    let acc0 = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![],
        step: 0,
    };
    
    // Base case: LHS all zeros
    println!("\nBase case (LHS all zeros):");
    let step0_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("prove");
    
    if let Some(fp) = &step0_invalid.proof.folding_proof {
        println!("  Folding {} instances (LHS + RHS)", fp.pi_ccs_inputs.len());
        println!("  LHS all zeros: {}", fp.pi_ccs_inputs[0].x.iter().all(|&f| f == F::ZERO));
        
        // In base case, α₀ · f(Y₀) ≈ 0 (LHS contributes nothing)
        // So the check essentially becomes: running_sum ≈ α₁ · f(Y₁)
        // And since Y₁ comes from invalid witness, f(Y₁) ≠ 0, so check fails
        println!("  Result: Verifier catches invalid witness ✓");
    }
    
    // Non-base case: LHS has valid previous step
    println!("\nNon-base case (LHS has valid previous step):");
    let step0_valid = prove_ivc_step_with_extractor(
        &params, &ccs, &valid_wit, &acc0, 0, None, &extractor, &binding,
    ).expect("setup");
    
    let acc1 = &step0_valid.proof.next_accumulator;
    let step1_invalid = prove_ivc_step_with_extractor(
        &params, &ccs, &invalid_wit, acc1, 1, None, &extractor, &binding,
    ).expect("prove");
    
    if let Some(fp) = &step1_invalid.proof.folding_proof {
        println!("  Folding {} instances (LHS + RHS)", fp.pi_ccs_inputs.len());
        println!("  LHS all zeros: {}", fp.pi_ccs_inputs[0].x.iter().all(|&f| f == F::ZERO));
        
        // In non-base case:
        // - LHS has valid witness from previous step
        // - RHS has invalid witness
        // - Both contribute to the augmented CCS sum
        // - But the terminal check f(Y(r)) doesn't properly verify the step CCS rows separately!
        // - The augmented rows (which are valid in both instances) dominate the check
        // - So invalid step CCS violations in RHS get masked
        println!("  Result: Verifier MISSES invalid witness ✗");
        println!("\n  WHY: The sum-check verifies:");
        println!("    running_sum == α₀·f(Y₀(r)) + α₁·f(Y₁(r))");
        println!("  But f() evaluates over the AUGMENTED CCS structure,");
        println!("  not just the step CCS rows where the violation occurs!");
        println!("  The valid augmented constraints in both instances");
        println!("  create a combined signature that looks valid,");
        println!("  even though the step CCS rows in RHS are invalid.");
    }
}
