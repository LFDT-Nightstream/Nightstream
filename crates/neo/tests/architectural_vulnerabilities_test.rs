//! Architectural Vulnerability Tests for Neo IVC
//!
//! These tests are designed to FAIL with the current implementation and expose
//! the specific architectural problems identified in the external review:
//!
//! 1. **Folding Chain Duplication**: Currently duplicating step instances instead of chaining
//! 2. **Final SNARK Public Input**: Using arbitrary [x] instead of augmented CCS layout
//! 3. **Missing Witness Binding**: x_witness_indices=[] removes public input security
//! 4. **Fixed Ajtai Seed**: Using [42u8; 32] instead of secure random generation
//!
//! Each test should FAIL, confirming the vulnerability exists, before we fix the implementation.

use anyhow::Result;
use neo::{NeoParams, F, ivc::{prove_ivc_step_with_extractor, prove_ivc_final_snark, Accumulator, StepBindingSpec, LastNExtractor}};
use neo_ccs::{r1cs_to_ccs, Mat, CcsStructure};
use p3_field::PrimeCharacteristicRing;

/// Build simple incrementer CCS: next_x = prev_x + delta
fn build_increment_ccs() -> CcsStructure<F> {
    let rows = 1;
    let cols = 4; // [const=1, prev_x, delta, next_x]

    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let c_trips = Vec::new();

    // Constraint: next_x - prev_x - delta = 0
    a_trips.push((0, 3, F::ONE));   // +next_x
    a_trips.push((0, 1, -F::ONE));  // -prev_x  
    a_trips.push((0, 2, -F::ONE));  // -delta
    b_trips.push((0, 0, F::ONE));   // × const 1

    let a_data = triplets_to_dense(rows, cols, a_trips);
    let b_data = triplets_to_dense(rows, cols, b_trips);
    let c_data = triplets_to_dense(rows, cols, c_trips);

    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a_data),
        Mat::from_row_major(rows, cols, b_data),
        Mat::from_row_major(rows, cols, c_data)
    )
}

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        dense[row * cols + col] = val;
    }
    dense
}

fn build_step_witness(prev_x: u64, delta: u64) -> Vec<F> {
    let next_x = prev_x + delta;
    vec![
        F::ONE,                    // const
        F::from_u64(prev_x),       // prev_x
        F::from_u64(delta),        // delta (public input)
        F::from_u64(next_x),       // next_x (y_step)
    ]
}

// ================================================================================================
// VULNERABILITY TEST 1: Folding Chain Duplication
// ================================================================================================

/// 🚨 VULNERABILITY TEST 1: Folding Chain Duplication
/// 
/// This test should FAIL because the current implementation duplicates the current step's
/// MCS instance to fill the "previous" slot instead of chaining the prior folded state.
/// 
/// Expected failure: The final proof should only attest to the last step, not the full chain.
#[test]
fn test_vulnerability_folding_chain_duplication() -> Result<()> {
    println!("🚨 VULNERABILITY TEST 1: Folding Chain Duplication");
    println!("Expected: This test should FAIL - folding duplicates instead of chaining");
    println!("========================================================================");

    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        
        // Bind step_x = [H(prev_acc)[4] || delta] → witness indices [0,1,2,3,2]
        x_witness_indices: vec![0, 1, 2, 3, 2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let mut accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    let mut ivc_proofs = Vec::new();
    let mut x = 0u64;
    let extractor = LastNExtractor { n: 1 };
    
    // Run 3 steps with distinct, easily trackable values
    let deltas = [100u64, 200u64, 300u64];
    let mut expected_chain_sum = 0u64;
    
    for (step_i, &delta) in deltas.iter().enumerate() {
        println!("   Step {}: prev_x={}, delta={}, expected_next_x={}", 
                step_i, x, delta, x + delta);
        
        let step_witness = build_step_witness(x, delta);
        let step_public_input = vec![F::from_u64(delta)];
        
        let step_result = prove_ivc_step_with_extractor(
            &params,
            &step_ccs,
            &step_witness,
            &accumulator,
            accumulator.step,
            Some(&step_public_input),
            &extractor,
            &binding_spec,
        ).map_err(|e| anyhow::anyhow!("IVC step failed: {}", e))?;

        accumulator = step_result.proof.next_accumulator.clone();
        ivc_proofs.push(step_result.proof);
        x += delta;
        expected_chain_sum += delta;
    }
    
    println!("   Expected chain sum (if folding works): {}", expected_chain_sum);
    println!("   Expected last step only (if folding broken): {}", deltas[2]);
    
    // Generate final SNARK with the arithmetic result
    let final_public_input = vec![F::from_u64(x)]; // x = 600 (full chain)
    let final_proof = prove_ivc_final_snark(&params, &ivc_proofs, &final_public_input)
        .map_err(|e| anyhow::anyhow!("Final SNARK failed: {}", e))?;
    
    let final_augmented_ccs = ivc_proofs.last().unwrap()
        .augmented_ccs.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Missing augmented CCS"))?;
    
    // Verify the proof with the correct full chain result
    let is_valid_full_chain = neo::verify(final_augmented_ccs, &final_public_input, &final_proof)
        .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
    
    if !is_valid_full_chain {
        println!("❌ VULNERABILITY CONFIRMED: Proof verification failed with full chain result!");
        return Err(anyhow::anyhow!("Folding chain duplication: proof doesn't verify with full chain"));
    }
    
    // 🚨 CRITICAL TEST: Try to verify with just the last step's result
    // If folding is broken (duplicating), this should also verify incorrectly
    let last_step_only_input = vec![F::from_u64(deltas[2])]; // Just 300, not 600
    let is_valid_last_step_only = neo::verify(final_augmented_ccs, &last_step_only_input, &final_proof)
        .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
    
    if is_valid_last_step_only {
        println!("❌ VULNERABILITY CONFIRMED: Proof verifies with last step only!");
        println!("   This proves folding is duplicating instances, not chaining state");
        return Err(anyhow::anyhow!(
            "Folding chain duplication vulnerability: proof verifies with last step ({}) instead of full chain ({})",
            deltas[2], expected_chain_sum
        ));
    }
    
    println!("✅ Folding chain integrity test PASSED (vulnerability fixed)");
    Ok(())
}

// ================================================================================================
// VULNERABILITY TEST 2: Final SNARK Public Input Format
// ================================================================================================

/// 🚨 VULNERABILITY TEST 2: Final SNARK Public Input Format Binding
/// 
/// This test should FAIL because the current implementation uses arbitrary [x] format
/// instead of the augmented CCS layout [step_x || ρ || y_prev || y_next].
/// 
/// Expected failure: The final SNARK should reject simple [x] format.
#[test]
fn test_vulnerability_final_snark_public_input_format() -> Result<()> {
    println!("🚨 VULNERABILITY TEST 2: Final SNARK Public Input Format Binding");
    println!("Expected: This test should FAIL - using wrong public input format");
    println!("================================================================");

    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    // 🔒 SECURITY FIX ACCOMMODATION: Provide proper witness binding to test other vulnerabilities
    // step_x = [H(prev_acc)[4 elements] || delta] = 5 elements total
    // We need to bind all 5 elements to witness positions
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        
        x_witness_indices: vec![0, 1, 2, 3, 2], // Bind step_x[0..4] to witness[0..3], step_x[4] (delta) to witness[2]
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    let mut ivc_proofs = Vec::new();
    let extractor = LastNExtractor { n: 1 };
    
    // Single step for simplicity
    let delta = 42u64;
    let step_witness = build_step_witness(0, delta);
    let step_public_input = vec![F::from_u64(delta)];
    
    let step_result = prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &step_witness,
        &accumulator,
        accumulator.step,
        Some(&step_public_input),
        &extractor,
        &binding_spec,
    ).map_err(|e| anyhow::anyhow!("IVC step failed: {}", e))?;

    ivc_proofs.push(step_result.proof.clone());
    let final_augmented_ccs = step_result.proof.augmented_ccs.as_ref().unwrap();
    
    // 🚨 WRONG FORMAT: Using simple [x] instead of augmented CCS layout
    let wrong_format_input = vec![F::from_u64(delta)]; // Simple arithmetic result
    
    println!("   Attempting to generate final SNARK with WRONG format: {:?}", wrong_format_input);
    
    // This should fail if the implementation correctly enforces augmented CCS layout
    match prove_ivc_final_snark(&params, &ivc_proofs, &wrong_format_input) {
        Ok(final_proof) => {
            println!("❌ VULNERABILITY CONFIRMED: Final SNARK generation succeeded with wrong format!");
            
            // Even worse - try to verify it
            let is_valid = neo::verify(final_augmented_ccs, &wrong_format_input, &final_proof)
                .map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
                
            if is_valid {
                println!("❌ DOUBLE VULNERABILITY: Wrong format also verifies!");
                return Err(anyhow::anyhow!(
                    "Final SNARK public input format vulnerability: wrong format [{}] accepted instead of augmented CCS layout",
                    delta
                ));
            } else {
                println!("⚠️  Partial vulnerability: Wrong format generates but doesn't verify");
                return Err(anyhow::anyhow!("Final SNARK should reject wrong format at generation time"));
            }
        }
        Err(e) => {
            println!("✅ Final SNARK public input format test PASSED: Wrong format correctly rejected ({})", e);
            return Ok(());
        }
    }
}

// ================================================================================================
// VULNERABILITY TEST 3: Missing Witness Binding
// ================================================================================================

/// 🚨 VULNERABILITY TEST 3: Missing Witness Binding (x_witness_indices=[])
/// 
/// This test should FAIL because x_witness_indices=[] removes the binding between
/// the step's declared public input and the actual value used in the witness.
/// 
/// Expected failure: Should be able to prove with mismatched public input vs witness.
#[test]
fn test_vulnerability_missing_witness_binding() -> Result<()> {
    println!("🚨 VULNERABILITY TEST 3: Missing Witness Binding (x_witness_indices=[])");
    println!("Expected: This test should FAIL - can prove with mismatched public input");
    println!("======================================================================");

    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_ccs();
    
    // 🚨 VULNERABLE: x_witness_indices=[] removes binding
    let vulnerable_binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        
        x_witness_indices: vec![],      // This is the vulnerability!
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    let extractor = LastNExtractor { n: 1 };
    
    // Create honest witness with delta=10
    let honest_delta = 10u64;
    let honest_witness = build_step_witness(0, honest_delta);
    
    // 🚨 ATTACK: Use malicious public input with delta=999
    let malicious_delta = 999u64;
    let malicious_public_input = vec![F::from_u64(malicious_delta)];
    
    println!("   Honest witness uses delta: {}", honest_delta);
    println!("   Malicious public input claims delta: {}", malicious_delta);
    println!("   Attempting to prove with mismatched values...");
    
    // This should fail if witness binding is properly enforced
    match prove_ivc_step_with_extractor(
        &params,
        &step_ccs,
        &honest_witness,
        &accumulator,
        accumulator.step,
        Some(&malicious_public_input),
        &extractor,
        &vulnerable_binding_spec,
    ) {
        Ok(_) => {
            println!("❌ VULNERABILITY CONFIRMED: Proof succeeded with mismatched public input!");
            println!("   x_witness_indices=[] allows public input manipulation");
            return Err(anyhow::anyhow!(
                "Missing witness binding vulnerability: honest witness {} + malicious public input {} succeeded",
                honest_delta, malicious_delta
            ));
        }
        Err(e) => {
            println!("✅ Witness binding test PASSED: Mismatched values correctly rejected ({})", e);
            return Ok(());
        }
    }
}

// ================================================================================================
// VULNERABILITY TEST 4: Fixed Ajtai Seed
// ================================================================================================

/// 🚨 VULNERABILITY TEST 4: Fixed Ajtai Seed ([42u8; 32])
/// 
/// This test should FAIL because the current implementation uses a fixed seed
/// instead of secure random generation, creating predictable SRS/PP.
/// 
/// Expected failure: Multiple runs should produce identical Ajtai parameters.
#[test]
fn test_vulnerability_fixed_ajtai_seed() -> Result<()> {
    println!("🚨 VULNERABILITY TEST 4: Fixed Ajtai Seed ([42u8; 32])");
    println!("Expected: This test should FAIL - Ajtai parameters are deterministic");
    println!("===================================================================");

    // Generate parameters twice and check if they're identical
    println!("   Generating first set of parameters...");
    let params1 = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    println!("   Generating second set of parameters...");
    let params2 = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Extract some identifiable components from the parameters
    // Note: This is a simplified check - in practice we'd need to access internal Ajtai matrices
    println!("   Comparing parameter structures...");
    
    // For now, we'll check if the parameters produce identical behavior
    // by running identical computations and seeing if we get identical results
    let step_ccs = build_increment_ccs();
    // 🔒 SECURITY FIX ACCOMMODATION: Provide proper witness binding to test other vulnerabilities
    // step_x = [H(prev_acc)[4 elements] || delta] = 5 elements total
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],        
        x_witness_indices: vec![0, 1, 2, 3, 2], // Bind step_x[0..4] to witness[0..3], step_x[4] (delta) to witness[2]
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    let extractor = LastNExtractor { n: 1 };
    let step_witness = build_step_witness(0, 42);
    let step_public_input = vec![F::from_u64(42)];
    
    // Run identical computation with both parameter sets
    let step_result1 = prove_ivc_step_with_extractor(
        &params1,
        &step_ccs,
        &step_witness,
        &accumulator,
        accumulator.step,
        Some(&step_public_input),
        &extractor,
        &binding_spec,
    ).map_err(|e| anyhow::anyhow!("IVC step 1 failed: {}", e))?;

    let step_result2 = prove_ivc_step_with_extractor(
        &params2,
        &step_ccs,
        &step_witness,
        &accumulator,
        accumulator.step,
        Some(&step_public_input),
        &extractor,
        &binding_spec,
    ).map_err(|e| anyhow::anyhow!("IVC step 2 failed: {}", e))?;
    
    // Check if the commitment coordinates are identical (indicating fixed seed)
    let coords1 = &step_result1.proof.next_accumulator.c_coords;
    let coords2 = &step_result2.proof.next_accumulator.c_coords;
    
    println!("   Commitment coords 1 length: {}", coords1.len());
    println!("   Commitment coords 2 length: {}", coords2.len());
    
    if coords1.len() == coords2.len() && coords1 == coords2 {
        println!("❌ VULNERABILITY CONFIRMED: Ajtai parameters are deterministic!");
        println!("   Identical inputs produce identical commitment coordinates");
        println!("   This indicates fixed seed [42u8; 32] is being used");
        return Err(anyhow::anyhow!(
            "Fixed Ajtai seed vulnerability: parameters are deterministic, not randomly generated"
        ));
    }
    
    // Also check if the commitment digests are identical
    let digest1 = step_result1.proof.next_accumulator.c_z_digest;
    let digest2 = step_result2.proof.next_accumulator.c_z_digest;
    
    if digest1 == digest2 {
        println!("❌ VULNERABILITY CONFIRMED: Commitment digests are identical!");
        println!("   Digest 1: {:?}", digest1);
        println!("   Digest 2: {:?}", digest2);
        return Err(anyhow::anyhow!(
            "Fixed Ajtai seed vulnerability: commitment digests are deterministic"
        ));
    }
    
    println!("✅ Ajtai seed randomness test PASSED (vulnerability fixed)");
    println!("   Parameters appear to use secure random generation");
    Ok(())
}

// ================================================================================================
// COMPREHENSIVE VULNERABILITY SUITE
// ================================================================================================

/// 🧪 COMPREHENSIVE VULNERABILITY TEST RUNNER
/// 
/// This test runs all vulnerability tests and expects them to FAIL,
/// confirming that the architectural issues exist before we fix them.
#[test]
fn test_comprehensive_vulnerability_suite() -> Result<()> {
    println!("🧪 COMPREHENSIVE VULNERABILITY TEST SUITE");
    println!("=========================================");
    println!("Running all architectural vulnerability tests...");
    println!("These tests should FAIL, confirming vulnerabilities exist.\n");

    let mut vulnerabilities_found = 0;
    let mut total_tests = 0;
    
    // Test 1: Folding Chain Duplication
    total_tests += 1;
    println!("🔍 Test 1/4: Folding Chain Duplication");
    match test_vulnerability_folding_chain_duplication() {
        Ok(_) => {
            println!("   ⚠️  Test PASSED - Vulnerability may be fixed or test needs refinement");
        }
        Err(e) => {
            println!("   ❌ Test FAILED as expected - Vulnerability confirmed: {}", e);
            vulnerabilities_found += 1;
        }
    }
    
    // Test 2: Final SNARK Public Input Format
    total_tests += 1;
    println!("\n🔍 Test 2/4: Final SNARK Public Input Format");
    match test_vulnerability_final_snark_public_input_format() {
        Ok(_) => {
            println!("   ⚠️  Test PASSED - Vulnerability may be fixed or test needs refinement");
        }
        Err(e) => {
            println!("   ❌ Test FAILED as expected - Vulnerability confirmed: {}", e);
            vulnerabilities_found += 1;
        }
    }
    
    // Test 3: Missing Witness Binding
    total_tests += 1;
    println!("\n🔍 Test 3/4: Missing Witness Binding");
    match test_vulnerability_missing_witness_binding() {
        Ok(_) => {
            println!("   ⚠️  Test PASSED - Vulnerability may be fixed or test needs refinement");
        }
        Err(e) => {
            println!("   ❌ Test FAILED as expected - Vulnerability confirmed: {}", e);
            vulnerabilities_found += 1;
        }
    }
    
    // Test 4: Fixed Ajtai Seed
    total_tests += 1;
    println!("\n🔍 Test 4/4: Fixed Ajtai Seed");
    match test_vulnerability_fixed_ajtai_seed() {
        Ok(_) => {
            println!("   ⚠️  Test PASSED - Vulnerability may be fixed or test needs refinement");
        }
        Err(e) => {
            println!("   ❌ Test FAILED as expected - Vulnerability confirmed: {}", e);
            vulnerabilities_found += 1;
        }
    }
    
    // Summary
    println!("\n🏁 VULNERABILITY TEST SUMMARY:");
    println!("==============================");
    println!("Vulnerabilities found: {}/{} tests", vulnerabilities_found, total_tests);
    
    if vulnerabilities_found == total_tests {
        println!("🚨 ALL VULNERABILITIES CONFIRMED - Implementation needs fixes!");
        println!("   This is the expected result before fixing the implementation.");
        return Err(anyhow::anyhow!("All {} architectural vulnerabilities confirmed", vulnerabilities_found));
    } else if vulnerabilities_found > 0 {
        println!("⚠️  PARTIAL VULNERABILITIES - Some issues may be fixed or tests need refinement");
        return Err(anyhow::anyhow!("{}/{} vulnerabilities found", vulnerabilities_found, total_tests));
    } else {
        println!("✅ NO VULNERABILITIES FOUND - Implementation appears secure!");
        println!("   This would be unexpected with the current implementation.");
        return Ok(());
    }
}
