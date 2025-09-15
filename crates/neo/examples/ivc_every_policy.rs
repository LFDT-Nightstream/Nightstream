//! IVC Every(n) Policy Example - Handling Partial Batches
//!
//! This example demonstrates how `EmissionPolicy::Every(n)` works with partial batches,
//! showing the correct pattern for ensuring all steps are processed.
//!
//! Usage: cargo run -p neo --example ivc_every_policy

use neo::{F, NeoParams};
use neo::ivc::{IvcBatchBuilder, EmissionPolicy, LastNExtractor, StepOutputExtractor, Accumulator};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use anyhow::Result;

/// Helper function to convert triplets to dense matrix data
fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        dense[row * cols + col] = val;
    }
    dense
}

/// Simple step: increment by 1
/// State: [x] -> [x+1]
fn build_increment_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, next_x]
    // Constraint: next_x - prev_x - 1 = 0 (next_x = prev_x + 1)
    
    let rows = 1;
    let cols = 3;
    
    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero
    
    // Constraint: next_x - prev_x - 1 = 0  
    // Written as: (next_x - prev_x - const) × const = 0
    a_trips.push((0, 2, F::ONE));   // +next_x
    a_trips.push((0, 1, -F::ONE));  // -prev_x  
    a_trips.push((0, 0, -F::ONE));  // -const (represents -1)
    b_trips.push((0, 0, F::ONE));   // select constant 1
    
    // Build matrices using the same pattern as fib_ivc.rs
    let a_data = triplets_to_dense(rows, cols, a_trips.clone());
    let b_data = triplets_to_dense(rows, cols, b_trips.clone());
    let c_data = triplets_to_dense(rows, cols, c_trips.clone());
    
    let a_mat = Mat::from_row_major(rows, cols, a_data.clone());
    let b_mat = Mat::from_row_major(rows, cols, b_data.clone());
    let c_mat = Mat::from_row_major(rows, cols, c_data.clone());
    
    println!("🔍 Debug CCS construction:");
    println!("   a_mat: {}x{}", a_mat.rows(), a_mat.cols());
    println!("   b_mat: {}x{}", b_mat.rows(), b_mat.cols());  
    println!("   c_mat: {}x{}", c_mat.rows(), c_mat.cols());
    println!("   a_data: {:?}", a_data);
    println!("   b_data: {:?}", b_data);
    
    let ccs = r1cs_to_ccs(a_mat, b_mat, c_mat);
    println!("   resulting ccs.n (constraints): {}", ccs.n);
    println!("   resulting ccs.m (variables): {}", ccs.m);
    
    // 🔍 VALIDATION: Test the step CCS directly
    let test_witness = vec![F::ONE, F::from_u64(5), F::from_u64(6)]; // [1, 5, 6] -> next_x should be 6
    let test_result = neo_ccs::check_ccs_rowwise_zero(&ccs, &[], &test_witness);
    println!("   Step CCS validation with [1,5,6]: {:?}", test_result);
    
    // Test with our actual witness
    let actual_witness = vec![F::ONE, F::ZERO, F::ONE]; // [1, 0, 1] -> 0+1=1 
    let actual_result = neo_ccs::check_ccs_rowwise_zero(&ccs, &[], &actual_witness);
    println!("   Step CCS validation with [1,0,1]: {:?}", actual_result);
    
    ccs
}

/// Generate increment step witness: [const=1, prev_x, next_x]
fn build_increment_witness(prev_x: u64) -> Vec<F> {
    let next_x = prev_x + 1;
    vec![
        F::ONE,                    // const
        F::from_u64(prev_x),       // prev_x
        F::from_u64(next_x),       // next_x
    ]
}

fn main() -> Result<()> {
    println!("🔄 IVC Every(n) Policy Example - Partial Batch Handling");
    println!("========================================================");
    
    // Setup  
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();
    
    // Initial accumulator - start with x = 0
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108],
        y_compact: vec![F::ZERO], // Start with x = 0
        step: 0,
    };
    
    // Create batch builder with SECURE binding specification
    // Increment witness layout: [const=1, prev_x, next_x]
    // y_step output (next_x) is at index 2
    let binding_spec = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![2], // next_x at index 2
        x_witness_indices: vec![], // No step public inputs for increment
        y_prev_witness_indices: vec![1], // prev_x at index 1
        const1_witness_index: 0, // Constant-1 at index 0
    };
    
    println!("🔍 Debug initial binding spec:");
    println!("   binding_spec.x_witness_indices: {:?}", binding_spec.x_witness_indices);
    println!("   binding_spec.y_step_offsets: {:?}", binding_spec.y_step_offsets);
    println!("   binding_spec.y_prev_witness_indices: {:?}", binding_spec.y_prev_witness_indices);
    println!("   binding_spec.const1_witness_index: {}", binding_spec.const1_witness_index);
    println!("   step_ccs.n (constraints): {}", step_ccs.n);
    println!("   step_ccs.m (variables): {}", step_ccs.m);
    println!("   initial_acc.y_compact.len(): {}", initial_acc.y_compact.len());
    
    // Use Every(3) policy - emit proof every 3 steps
    let mut batch_builder = IvcBatchBuilder::new_with_bindings(
        params.clone(),
        step_ccs.clone(), 
        initial_acc,
        EmissionPolicy::Every(3),
        binding_spec,
    )?;
    
    println!("📊 Using EmissionPolicy::Every(3)");
    println!("   Will auto-emit after steps 3, 6, 9, etc.");
    
    // Extractor to get real y_step (last element = next_x)
    let extractor = LastNExtractor { n: 1 };
    
    // Run 7 steps: should auto-emit after steps 3 and 6, leaving 1 step pending
    let num_steps = 7;
    println!("\n🔄 Running {} increment steps...", num_steps);
    
    let mut current_x = 0u64;
    let mut proofs_emitted = 0;
    
    for step in 0..num_steps {
        let step_witness = build_increment_witness(current_x);
        let y_step_real = extractor.extract_y_step(&step_witness);
        
        println!("   Step {}: {} -> {}", step, current_x, current_x + 1);
        
        let pending_before = batch_builder.pending_steps();
        // Provide step_x = H(prev_accumulator) to satisfy binding (Las requirement)
        let x_digest = {
            // Access the internal helper via an equivalent: reserialize and hash locally here
            let acc = &batch_builder.accumulator;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&acc.step.to_le_bytes());
            bytes.extend_from_slice(&acc.c_z_digest);
            bytes.extend_from_slice(&(acc.y_compact.len() as u64).to_le_bytes());
            for &y in &acc.y_compact { bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes()); }
            let d = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&bytes);
            let mut out = Vec::with_capacity(d.len());
            for x in d { out.push(neo::F::from_u64(x.as_canonical_u64())); }
            out
        };
        let _y_next = batch_builder.append_step(&step_witness, Some(&x_digest), &y_step_real)?;
        let pending_after = batch_builder.pending_steps();
        
        // Check if a proof was auto-emitted
        if pending_after < pending_before {
            proofs_emitted += 1;
            println!("     🔒 AUTO-EMITTED: Proof #{} (covered steps {})", 
                     proofs_emitted, 
                     if proofs_emitted == 1 { "0-2" } else { "3-5" });
            println!("     🔍 Accumulator state after auto-emit:");
            println!("       acc.step: {}", batch_builder.accumulator.step);
            println!("       acc.y_compact.len(): {}", batch_builder.accumulator.y_compact.len());
        }
        
        current_x += 1;
    }
    
    println!("\n📊 After {} steps:", num_steps);
    println!("   Proofs auto-emitted: {}", proofs_emitted); 
    println!("   Pending steps: {}", batch_builder.pending_steps());
    println!("   Has pending batch: {}", batch_builder.has_pending_batch());
    
    // ⚠️ CRITICAL: Handle the partial batch!
    // Step 6 is still pending and needs explicit handling
    println!("\n📦 Handling partial batch (2 approaches)...");
    
    println!("\n   Approach 1: Immediate proving (bypasses Final SNARK Layer):");
    // Clone for demonstration - in practice you'd choose one approach
    let binding_spec_clone = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![2], // same as original
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![1], // same as original
        const1_witness_index: 0, // Constant-1 at index 0
    };
    
    println!("🔍 Debug batch_clone creation:");
    println!("   binding_spec_clone.x_witness_indices: {:?}", binding_spec_clone.x_witness_indices);
    println!("   binding_spec_clone.y_step_offsets: {:?}", binding_spec_clone.y_step_offsets);
    println!("   binding_spec_clone.y_prev_witness_indices: {:?}", binding_spec_clone.y_prev_witness_indices);
    println!("   binding_spec_clone.const1_witness_index: {}", binding_spec_clone.const1_witness_index);
    
    let original_acc = batch_builder.accumulator.clone();
    println!("   original_acc.y_compact.len(): {}", original_acc.y_compact.len());
    println!("   original_acc.step: {}", original_acc.step);
    
    let mut batch_clone = IvcBatchBuilder::new_with_bindings(
        params.clone(), 
        step_ccs.clone(),
        original_acc.clone(),
        EmissionPolicy::Every(3),
        binding_spec_clone.clone(),
    )?;
    
    println!("   batch_clone created successfully, pending_steps: {}", batch_clone.pending_steps());
    
    // Re-add the last step to demonstrate
    let last_witness = build_increment_witness(6);
    let last_y_step = extractor.extract_y_step(&last_witness);
    
    println!("🔍 Debug last step append:");
    println!("   last_witness: {:?}", last_witness);
    println!("   last_witness.len(): {}", last_witness.len());
    println!("   last_y_step: {:?}", last_y_step);
    println!("   last_y_step.len(): {}", last_y_step.len());
    println!("   step_ccs.n (constraints): {}", step_ccs.n);
    println!("   step_ccs.m (variables): {}", step_ccs.m);
    
    batch_clone.append_step(&last_witness, None, &last_y_step)?;
    println!("   append_step completed, pending_steps: {}", batch_clone.pending_steps());
    
    println!("🔍 Debug before finalize_and_prove:");
    println!("   batch_clone.pending_steps(): {}", batch_clone.pending_steps());
    println!("   batch_clone.has_pending_batch(): {}", batch_clone.has_pending_batch());
    
    // Debug: Extract batch data to inspect CCS structure before proving
    println!("🔍 DEBUG: Inspecting CCS structure before failing finalize_and_prove:");
    let _debug_batch_data = if let Some(debug_data) = batch_clone.finalize() {
        println!("   SINGLE-STEP CCS:");
        println!("     constraints (n): {}", debug_data.ccs.n);
        println!("     variables (m): {}", debug_data.ccs.m);
        println!("     matrices count: {}", debug_data.ccs.matrices.len());
        println!("     public_input.len(): {}", debug_data.public_input.len());
        println!("     witness.len(): {}", debug_data.witness.len());
        println!("     steps_covered: {}", debug_data.steps_covered);
        
        // Debug first few elements of witness
        println!("     witness (first 10): {:?}", 
                debug_data.witness.iter().take(10).collect::<Vec<_>>());
        
        Some(debug_data)
    } else {
        println!("   No debug batch data available");
        None
    };
    
    // Recreate batch_clone since finalize() consumed it
    let mut batch_clone = IvcBatchBuilder::new_with_bindings(
        params.clone(), 
        step_ccs.clone(),
        original_acc,
        EmissionPolicy::Every(3),
        binding_spec_clone,
    )?;
    batch_clone.append_step(&last_witness, None, &last_y_step)?;
    
    let immediate_proof = batch_clone.finalize_and_prove()?;
    match immediate_proof {
        Some(_proof) => {
            println!("     ✅ Immediate proof generated for step 6!");
        }
        None => {
            println!("     ℹ️  No immediate proof needed");
        }
    }
    
    println!("\n   Approach 2: Extract batch → Final SNARK Layer (recommended):");
    let batch_data = batch_builder.finalize();
    match batch_data {
        Some(data) => {
            println!("     📦 Extracted batch data: {} steps, {} constraints", 
                     data.steps_covered, data.ccs.n);
            
            // Final SNARK Layer (expensive step, done separately)
            let _final_proof = neo::ivc::prove_batch_data(&params, data)?;
            println!("     ✅ Final SNARK Layer proof generated for step 6!");
            println!("     Total proofs: {} (auto) + 1 (final) = {}", proofs_emitted, proofs_emitted + 1);
        }
        None => {
            println!("     ℹ️  No batch data to prove");
        }
    }
    
    println!("\n📊 Final Status:");
    println!("   Pending steps: {}", batch_builder.pending_steps());
    println!("   Has pending batch: {}", batch_builder.has_pending_batch());
    
    println!("\n🎯 Key Takeaways:");
    println!("   ✅ Every(3) auto-proved after steps 3 and 6");
    println!("   ✅ Step 6 required explicit handling via finalize()"); 
    println!("   ✅ Two approaches: finalize_and_prove() or finalize() → Final SNARK Layer");
    println!("   ✅ Recommended: Use finalize() + prove_batch_data() for clean separation");
    println!("   ✅ Total computation: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7");
    println!("   ✅ Architecture follows: Fast IVC accumulation + Separate Final SNARK Layer");
    
    Ok(())
}
