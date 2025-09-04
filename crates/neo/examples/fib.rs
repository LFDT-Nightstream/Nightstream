//! Fibonacci Neo SNARK Demo
//!
//! This demonstrates the complete Neo lattice-based SNARK pipeline with a Fibonacci
//! recurrence relation: z[i+2] = z[i+1] + z[i]
//!
//! Usage: cargo run -p neo --example fib

use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
use neo_ccs::{Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use anyhow::Result;
use std::time::Instant;

/// Build Fibonacci R1CS: z[i+2] = z[i+1] + z[i] for i = 0..n_steps-1
/// Variables: [1, z0, z1, ..., z_{n_steps+1}] (constant wire at index 0)
fn fibonacci_ccs(n_steps: usize) -> CcsStructure<F> {
    let m = n_steps + 2; // z0, z1, ..., z_{n_steps+1}
    let cols = m + 1;    // Add constant "1" wire at index 0
    let rows = n_steps;  // One constraint per recurrence step
    
    // Initialize matrices A, B, C for R1CS: (Az) ∘ (Bz) = Cz
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];
    
    for i in 0..n_steps {
        let row = i;
        let base_idx = row * cols;
        
        // A matrix: z[i+2] - z[i+1] - z[i]
        a_data[base_idx + (i + 3)] = F::ONE;        // +z[i+2] (offset by 1 for constant wire)
        a_data[base_idx + (i + 2)] = -F::ONE;       // -z[i+1]
        a_data[base_idx + (i + 1)] = -F::ONE;       // -z[i]
        
        // B matrix: select constant 1
        b_data[base_idx + 0] = F::ONE;              // constant wire
        
        // C matrix: 0 (constraint = 0)
        // Already zero-initialized
    }
    
    let a = Mat::from_row_major(rows, cols, a_data);
    let b = Mat::from_row_major(rows, cols, b_data);
    let c = Mat::from_row_major(rows, cols, c_data);
    
    // Convert R1CS to CCS
    r1cs_to_ccs(a, b, c)
}

/// Generate satisfying Fibonacci witness: [1, 0, 1, 1, 2, 3, 5, 8, ...]
fn generate_fibonacci_witness(n_terms: usize) -> Vec<F> {
    let mut z = vec![F::ONE]; // constant wire = 1
    
    if n_terms >= 1 { z.push(F::ZERO); }  // z0 = 0
    if n_terms >= 2 { z.push(F::ONE); }   // z1 = 1
    
    // Generate remaining Fibonacci numbers
    for i in 2..n_terms {
        let next = z[i] + z[i - 1]; // z[i-1] + z[i-2], offset by 1 for constant wire
        z.push(next);
    }
    
    z
}

fn main() -> Result<()> {
    println!("🔥 Neo Lattice Demo: Proving Fibonacci Series");
    println!("==============================================");
    
    // High-level input: Length of Fibonacci series to prove
    let fib_length = 8;
    
    // Step 1: Create Fibonacci CCS constraint system
    println!("\n📐 Step 1: Creating Fibonacci CCS...");
    let ccs = fibonacci_ccs(fib_length);
    println!("   CCS dimensions: {} constraints, {} variables", ccs.n, ccs.m);
    println!("   Enforcing recurrence: z[i+2] = z[i+1] + z[i] for i=0..{}", fib_length-1);
    
    // Step 2: Generate satisfying witness
    println!("\n🔢 Step 2: Generating Fibonacci witness...");
    let z = generate_fibonacci_witness(fib_length + 2);
    let z_values: Vec<u64> = z.iter().take(10).map(|x| x.as_canonical_u64()).collect();
    println!("   Witness (first 10): {:?}", z_values);
    
    // Step 3: Verify CCS satisfaction locally
    println!("\n✅ Step 3: Verifying CCS satisfaction...");
    let public_inputs = vec![]; // No public inputs for this example
    check_ccs_rowwise_zero(&ccs, &public_inputs, &z)
        .map_err(|e| anyhow::anyhow!("CCS check failed: {:?}", e))?;
    println!("   ✅ Local CCS verification passed!");
    
    // Step 4: Setup parameters
    println!("\n🔧 Step 4: Setting up Neo parameters...");
    println!("   Using auto-tuned parameters for ell=3, d_sc=2");
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    println!("   Lambda: {} bits (compatible with s=2)", params.lambda);
    println!("   Security: {} bits sum-check soundness", params.lambda);
    
    // Step 5: Generate Neo SNARK proof
    println!("\n🔀 Step 5: Generating Neo SNARK proof...");
    let prove_start = Instant::now();
    
    let proof = prove(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_inputs,
        witness: &z,
    })?;
    
    let prove_time = prove_start.elapsed();
    println!("   ✅ Neo SNARK proof generated successfully!");
    println!("   - Proof generation time: {:.2} ms", prove_time.as_secs_f64() * 1000.0);
    println!("   - Proof size: {} bytes", proof.size());
    
    // Step 6: Verify the proof
    println!("\n🔍 Step 6: Verifying Neo SNARK proof...");
    let verify_start = Instant::now();
    
    let is_valid = verify(&ccs, &public_inputs, &proof)?;
    let verify_time = verify_start.elapsed();
    
    if is_valid {
        println!("   ✅ Complete protocol verification PASSED!");
        println!("   - Verification time: {:.2} ms", verify_time.as_secs_f64() * 1000.0);
        
        // Final Performance Summary
        println!("\n==========================================");
        println!("🏁 FINAL PERFORMANCE SUMMARY");
        println!("==========================================");
        
        println!("Mode:                     {:>12}", "Neo SNARK");
        println!("Proof Generation Time:    {:>8.2} ms", prove_time.as_secs_f64() * 1000.0);
        println!("Proof Size:               {:>8} bytes", proof.size());
        println!("Verification Time:        {:>8.2} ms", verify_time.as_secs_f64() * 1000.0);
        println!("Total Time:               {:>8.2} ms", 
               (prove_time + verify_time).as_secs_f64() * 1000.0);
        println!("Verification Result:      {}", "✅ PASSED");
        println!("Fibonacci Length:         {:>8}", fib_length);
        println!("Post-Quantum Security:    {:>8}", "Yes");
        
        println!("==========================================");
        println!("\n🎉 Neo Protocol Flow Complete!");
        println!("   ✨ Fibonacci constraints successfully proven with Neo lattice-based SNARK");
    } else {
        println!("   ❌ Verification FAILED");
        return Err(anyhow::anyhow!("Proof verification failed"));
    }
    
    Ok(())
}
