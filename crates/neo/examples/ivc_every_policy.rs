//! IVC Every(n) Policy Example - Handling Partial Batches
//!
//! This example demonstrates how `EmissionPolicy::Every(n)` works with partial batches,
//! showing the correct pattern for ensuring all steps are processed.
//!
//! Usage: cargo run -p neo --example ivc_every_policy

use neo::{F, NeoParams};
use neo::ivc::{IvcBatchBuilder, EmissionPolicy, LastNExtractor, StepOutputExtractor, Accumulator};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;
use anyhow::Result;

/// Simple step: increment by 1
/// State: [x] -> [x+1]
fn build_increment_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, next_x]
    // Constraint: next_x - prev_x - 1 = 0 (next_x = prev_x + 1)
    
    let rows = 1;
    let cols = 3;
    
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols]; 
    let c_data = vec![F::ZERO; rows * cols];
    
    // next_x - prev_x - 1 = 0 (multiply by const=1)
    a_data[0 * cols + 2] = F::ONE;   // +next_x
    a_data[0 * cols + 1] = -F::ONE;  // -prev_x  
    a_data[0 * cols + 0] = -F::ONE;  // -const (represents -1)
    b_data[0 * cols + 0] = F::ONE;   // × const
    
    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
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
    
    // Use Every(3) policy - emit proof every 3 steps
    let mut batch_builder = IvcBatchBuilder::new(
        params.clone(),
        step_ccs.clone(), 
        initial_acc,
        EmissionPolicy::Every(3)
    );
    
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
        let _y_next = batch_builder.append_step(&step_witness, None, &y_step_real)?;
        let pending_after = batch_builder.pending_steps();
        
        // Check if a proof was auto-emitted
        if pending_after < pending_before {
            proofs_emitted += 1;
            println!("     🔒 AUTO-EMITTED: Proof #{} (covered steps {})", 
                     proofs_emitted, 
                     if proofs_emitted == 1 { "0-2" } else { "3-5" });
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
    let mut batch_clone = IvcBatchBuilder::new(
        params.clone(), 
        step_ccs.clone(),
        batch_builder.accumulator.clone(),
        EmissionPolicy::Every(3)
    );
    
    // Re-add the last step to demonstrate
    let last_witness = build_increment_witness(6);
    let last_y_step = extractor.extract_y_step(&last_witness);
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
