//! IVC Checkpoint Example - Efficient Batching with On-Demand SNARK Emission
//!
//! This example demonstrates the new `IvcBatchBuilder` which allows running
//! many IVC steps efficiently and only emitting a single SNARK proof when needed.
//! 
//! Key benefits:
//! - Fast IVC loop with embedded verifier validation
//! - Single SNARK proof for multiple steps (checkpoint-based compression)
//! - Configurable emission policies (Never, Every(n), OnDemand)
//! 
//! Usage: cargo run -p neo --example ivc_checkpoint

use neo::{F, NeoParams};
use neo::ivc::{IvcBatchBuilder, EmissionPolicy, LastNExtractor, StepOutputExtractor, Accumulator};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use anyhow::Result;

/// Simple step computation: Fibonacci sequence
/// State: [a, b] -> [b, a + b]
fn build_fibonacci_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_a, prev_b, next_a, next_b]
    // Constraints:
    //   1) next_a - prev_b = 0      (next_a = prev_b)
    //   2) next_b - prev_a - prev_b = 0  (next_b = prev_a + prev_b)
    
    let rows = 2;
    let cols = 5;
    
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];
    
    // Row 0: next_a - prev_b = 0 (multiply by const=1)
    a_data[0 * cols + 3] = F::ONE;   // +next_a
    a_data[0 * cols + 2] = -F::ONE;  // -prev_b
    b_data[0 * cols + 0] = F::ONE;   // × const
    
    // Row 1: next_b - prev_a - prev_b = 0 (multiply by const=1)  
    a_data[1 * cols + 4] = F::ONE;   // +next_b
    a_data[1 * cols + 1] = -F::ONE;  // -prev_a
    a_data[1 * cols + 2] = -F::ONE;  // -prev_b
    b_data[1 * cols + 0] = F::ONE;   // × const
    
    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data); 
    let c_mat = Mat::from_row_major(rows, cols, c_data);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Generate Fibonacci step witness: [const=1, prev_a, prev_b, next_a, next_b]
fn build_fibonacci_witness(prev_a: u64, prev_b: u64) -> Vec<F> {
    let next_a = prev_b;
    let next_b = prev_a + prev_b;
    
    vec![
        F::ONE,                    // const
        F::from_u64(prev_a),       // prev_a
        F::from_u64(prev_b),       // prev_b
        F::from_u64(next_a),       // next_a
        F::from_u64(next_b),       // next_b
    ]
}

fn main() -> Result<()> {
    println!("🚀 IVC Checkpoint Example - Efficient Step Batching");
    println!("==================================================");
    
    // Setup
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_fibonacci_step_ccs();
    
    println!("📊 Step CCS: {} constraints, {} variables", step_ccs.n, step_ccs.m);
    
    // Initial accumulator - start with Fibonacci seed [1, 1]  
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 108], // d*kappa = 54*2 for proper Ajtai dimensions
        y_compact: vec![F::ONE, F::ONE], // Fibonacci seed: [1, 1]
        step: 0,
    };
    
    // Create batch builder with SECURE binding specification
    // Fibonacci witness layout: [const=1, prev_a, prev_b, next_a, next_b]  
    // y_step outputs (next_a, next_b) are at indices 3, 4
    let binding_spec = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![3, 4], // next_a at index 3, next_b at index 4
        x_witness_indices: vec![], // No step public inputs for Fibonacci
        y_prev_witness_indices: vec![1, 2], // prev_a at index 1, prev_b at index 2
    };
    
    let mut batch_builder = IvcBatchBuilder::new_with_bindings(
        params, 
        step_ccs, 
        initial_acc, 
        EmissionPolicy::OnDemand,
        binding_spec,
    )?;
    
    println!("✅ IVC Batch Builder initialized");
    println!("   Initial state: {:?}", batch_builder.accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    
    // Extractor to get real y_step values (last 2 elements = [next_a, next_b])
    let extractor = LastNExtractor { n: 2 };
    
    // Run multiple Fibonacci steps without emitting SNARKs
    let num_steps = 10;
    println!("\n🔄 Running {} Fibonacci steps...", num_steps);
    
    let mut current_a = 1u64;
    let mut current_b = 1u64;
    
    for step in 0..num_steps {
        // Build witness for this step
        let step_witness = build_fibonacci_witness(current_a, current_b);
        
        // Extract real y_step values from the witness (the next Fibonacci numbers)
        let y_step_real = extractor.extract_y_step(&step_witness);
        
        // Append step to batch (no SNARK emission yet)
        let y_next = batch_builder.append_step(&step_witness, None, &y_step_real)?;
        
        // Update for next iteration
        let next_a = current_b;
        let next_b = current_a + current_b;
        current_a = next_a;
        current_b = next_b;
        
        println!("   Step {}: {:?} -> {:?}", 
                 step, 
                 batch_builder.accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                 y_next.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    }
    
    println!("\n📊 Batch Status:");
    println!("   Pending steps: {}", batch_builder.pending_steps());
    println!("   Has pending batch: {}", batch_builder.has_pending_batch());
    
    // Stage 4: Finalize IVC accumulation (fast, no SNARK yet)
    println!("\n📦 Finalizing IVC batch accumulation...");
    
    let batch_data = batch_builder.finalize();
    match batch_data {
        Some(data) => {
            println!("✅ IVC batch accumulated successfully!");
            println!("   Steps covered: {}", data.steps_covered);
            println!("   CCS size: {} constraints, {} variables", data.ccs.n, data.ccs.m);
            println!("   Public input size: {}", data.public_input.len());
            println!("   Witness size: {}", data.witness.len());
            
            // Final SNARK Layer: Convert accumulated batch to succinct proof (expensive)
            println!("\n🔒 Final SNARK Layer: Generating succinct proof...");
            let proof = neo::ivc::prove_batch_data(&params, data)?;
            
            println!("✅ Succinct proof generated!");
            println!("   Proof size: ~{} bytes", std::mem::size_of_val(&proof));
            println!("   Final accumulator state: {:?}", 
                     batch_builder.accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
            
            // The proof covers the computation: [1,1] -> [1,1] -> [1,2] -> [2,3] -> [3,5] -> ... -> [55, 89]
            // This single SNARK proves the correctness of all 10 Fibonacci steps!
        }
        None => {
            println!("ℹ️  No batch data to prove (batch was empty)");
        }
    }
    
    println!("\n📊 Final Status:");
    println!("   Pending steps after emission: {}", batch_builder.pending_steps());
    println!("   Has pending batch after emission: {}", batch_builder.has_pending_batch());
    
    println!("\n🎯 Key Benefits Demonstrated:");
    println!("   ✅ Fast IVC loop - each step validated with embedded verifier");
    println!("   ✅ Efficient batching - single SNARK for {} steps", num_steps);
    println!("   ✅ On-demand compression - proof only when needed");
    println!("   ✅ Sound ρ derivation - public challenge binding all steps");
    println!("   ✅ Real y_step extraction - no 'folding with itself' issues");
    
    Ok(())
}
