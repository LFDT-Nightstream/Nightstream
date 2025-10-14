//! Test for subtle bit-flip attacks on ρ
//! 
//! This test checks if the system can detect even single-bit modifications to ρ,
//! which would be a very subtle attack vector.

use anyhow::Result;
use neo::{F, NeoParams, StepOutputExtractor, LastNExtractor};
use neo::{Accumulator, StepBindingSpec, IvcStepInput, prove_ivc_step, verify_ivc_step};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        dense[row * cols + col] = val;
    }
    dense
}

// Simple step circuit: incrementer x' = x + delta (copied from working tests)
fn build_incrementer_step_ccs() -> CcsStructure<F> {
    // Minimum of 3 rows required (gets padded to 4 for ℓ=2)
    let rows = 3;
    let cols = 4; // [const=1, prev_x, delta, next_x]

    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let c_trips = Vec::new();

    // Constraint 0: next_x - prev_x - delta = 0
    a_trips.push((0, 3, F::ONE));   // +next_x
    a_trips.push((0, 1, -F::ONE));  // -prev_x  
    a_trips.push((0, 2, -F::ONE));  // -delta
    b_trips.push((0, 0, F::ONE));   // × const 1

    // Dummy constraints (rows 1, 2): 0 * 1 = 0 (trivially satisfied)
    for row in 1..3 {
        a_trips.push((row, 0, F::ZERO));  // 0
        b_trips.push((row, 0, F::ONE));   // × 1
        // c is zero by default
    }

    let a_data = triplets_to_dense(rows, cols, a_trips);
    let b_data = triplets_to_dense(rows, cols, b_trips);
    let c_data = triplets_to_dense(rows, cols, c_trips);

    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a_data),
        Mat::from_row_major(rows, cols, b_data),
        Mat::from_row_major(rows, cols, c_data)
    )
}

#[test]
fn test_single_bit_flip_in_rho() -> Result<()> {
    println!("🚀 Testing single bit flip in ρ (subtle attack)");
    
    // Setup (same as other tests)
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // No public input binding needed for this test
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let initial_acc = Accumulator { 
        c_z_digest: [0;32], 
        c_coords: vec![], 
        y_compact: vec![F::ZERO], 
        step: 0 
    };

    // Use delta=7 for the incrementer
    let prev_x = initial_acc.y_compact[0];
    let delta = F::from_u64(7);
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];

    // Generate valid proof
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &initial_acc,
        step: 0,
        public_input: None, // No public input needed - testing ρ tampering, not input binding
        y_step: &y_step,
        binding_spec: &binding_spec,
        app_input_binding: neo::AppInputBinding::WitnessBound,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("Failed to prove IVC step");
    let mut proof_with_flipped_rho = step_result.proof;
    
    println!("🔍 Original ρ: {}", proof_with_flipped_rho.public_inputs.rho().as_canonical_u64());
    
    // Flip a single bit in ρ (flip the least significant bit)
    let original_rho_u64 = proof_with_flipped_rho.public_inputs.rho().as_canonical_u64();
    let flipped_rho_u64 = original_rho_u64 ^ 1; // XOR with 1 flips the LSB
    let flipped_rho = F::from_u64(flipped_rho_u64);
    
    proof_with_flipped_rho.public_inputs.__test_tamper_rho(flipped_rho);
    
    println!("🔍 Flipped ρ:  {} (flipped bit 0)", flipped_rho_u64);
    println!("🔍 Difference: {} (should be 1)", original_rho_u64 ^ flipped_rho_u64);
    
    // Test verification with single bit flip
    println!("\n🧪 Testing verification with single bit flip in ρ...");
    let result = match verify_ivc_step(&step_ccs, &proof_with_flipped_rho, &initial_acc, &binding_spec, &params, None) {
        Ok(result) => result,
        Err(e) => {
            println!("⚠️  Verification returned error: {}", e);
            false
        }
    };
    
    println!("   Single bit flip verification: {}", if result { "❌ ACCEPTED (UNSOUND!)" } else { "✅ REJECTED (SOUND)" });
    
    if result {
        println!("\n❌ CRITICAL: Single bit flip in ρ was accepted!");
        println!("   This suggests the system doesn't properly validate ρ consistency.");
        panic!("UNSOUND: Single bit flip in ρ was accepted!");
    } else {
        println!("\n✅ SECURITY VALIDATED: Single bit flip in ρ was correctly rejected");
        println!("   The system properly validates ρ consistency even for subtle changes.");
        println!("   🎉 TEST PASSED: Subtle attack failed, system is sound!");
    }
    
    Ok(())
}

#[test]
fn test_multiple_bit_flips_in_rho() -> Result<()> {
    println!("🚀 Testing multiple bit flips in ρ (various positions)");
    
    // Setup
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_incrementer_step_ccs();
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        step_program_input_witness_indices: vec![], // No public input binding needed for this test
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    
    let initial_acc = Accumulator { 
        c_z_digest: [0;32], 
        c_coords: vec![], 
        y_compact: vec![F::ZERO], 
        step: 0 
    };

    // Use delta=13 for the incrementer
    let prev_x = initial_acc.y_compact[0];
    let delta = F::from_u64(13);
    let next_x = prev_x + delta;
    let step_witness = vec![F::ONE, prev_x, delta, next_x];

    // Generate valid proof
    let extractor = LastNExtractor { n: 1 };
    let y_step = extractor.extract_y_step(&step_witness);
    
    let step_input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &initial_acc,
        step: 0,
        public_input: None, // No public input needed - testing ρ tampering, not input binding
        y_step: &y_step,
        binding_spec: &binding_spec,
        app_input_binding: neo::AppInputBinding::WitnessBound,
        prev_augmented_x: None,
    };
    
    let step_result = prove_ivc_step(step_input).expect("Failed to prove IVC step");
    let original_proof = step_result.proof;
    
    println!("🔍 Original ρ: {}", original_proof.public_inputs.rho().as_canonical_u64());
    
    // Test flipping different bit positions
    let bit_positions = [0, 1, 7, 15, 31, 32, 63]; // Various bit positions
    let mut all_rejected = true;
    
    for &bit_pos in &bit_positions {
        let mut proof_with_flipped_bit = original_proof.clone();
        let original_rho_u64 = proof_with_flipped_bit.public_inputs.rho().as_canonical_u64();
        let flipped_rho_u64 = original_rho_u64 ^ (1u64 << bit_pos); // Flip specific bit
        let flipped_rho = F::from_u64(flipped_rho_u64);
        
        proof_with_flipped_bit.public_inputs.__test_tamper_rho(flipped_rho);
        
        println!("\n🔍 Testing bit position {}: {} -> {}", 
                 bit_pos, original_rho_u64, flipped_rho_u64);
        
        let result = match verify_ivc_step(&step_ccs, &proof_with_flipped_bit, &initial_acc, &binding_spec, &params, None) {
            Ok(result) => result,
            Err(e) => {
                println!("   ⚠️  Verification returned error: {}", e);
                false
            }
        };
        
        if result {
            println!("   ❌ ACCEPTED (bit {} flip was not detected!)", bit_pos);
            all_rejected = false;
        } else {
            println!("   ✅ REJECTED (bit {} flip correctly detected)", bit_pos);
        }
    }
    
    if all_rejected {
        println!("\n✅ SECURITY VALIDATED: All bit flips were correctly rejected");
        println!("   The system detects even single-bit modifications to ρ.");
        println!("   🎉 TEST PASSED: All subtle attacks failed, system is sound!");
    } else {
        println!("\n❌ CRITICAL: Some bit flips were accepted!");
        panic!("UNSOUND: System accepted some bit flips in ρ!");
    }
    
    Ok(())
}
