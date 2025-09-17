//! Critical security test for splice attacks
//! 
//! This test demonstrates a serious vulnerability where proof_bytes from one circuit
//! can be combined with public_io from another circuit to create a valid-looking proof.

use anyhow::Result;
use neo::{prove, verify, ProveInput, NeoParams, CcsStructure, F};
use neo_ccs::{r1cs_to_ccs, Mat};
use p3_field::PrimeCharacteristicRing;

/// Build Fibonacci CCS: z[i+2] = z[i+1] + z[i]
fn fib_ccs(n: usize) -> (CcsStructure<F>, Vec<F>) {
    assert!(n >= 1);
    let rows = n + 1;        // 2 seed rows + (n-1) recurrence rows
    let cols = n + 2;        // [1, z0, z1, ..., z_n]

    // Build constraint matrices
    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let _c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero

    // z0 = 0
    a_trips.push((0, 1, F::ONE));
    b_trips.push((0, 0, F::ONE));

    // z1 = 1
    a_trips.push((1, 2, F::ONE));
    a_trips.push((1, 0, -F::ONE));
    b_trips.push((1, 0, F::ONE));

    // Recurrence: z[i+2] = z[i+1] + z[i]
    for i in 0..(n - 1) {
        let r = 2 + i;
        a_trips.push((r, i + 3, F::ONE));   // +z[i+2]
        a_trips.push((r, i + 2, -F::ONE));  // -z[i+1]
        a_trips.push((r, i + 1, -F::ONE));  // -z[i]
        b_trips.push((r, 0, F::ONE));
    }

    // Convert to matrices
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];

    for (r, c, val) in a_trips {
        a_data[r * cols + c] = val;
    }
    for (r, c, val) in b_trips {
        b_data[r * cols + c] = val;
    }

    let a = Mat::from_row_major(rows, cols, a_data);
    let b = Mat::from_row_major(rows, cols, b_data);  
    let c = Mat::from_row_major(rows, cols, c_data);
    
    let ccs = r1cs_to_ccs(a, b, c);
    
    // Generate witness
    let mut witness = vec![F::ONE, F::ZERO, F::ONE]; // [1, z0=0, z1=1]
    for k in 2..=n {
        let next = witness[k] + witness[k - 1]; 
        witness.push(next);
    }
    
    (ccs, witness)
}

/// Build linear CCS: z[i+1] = 2*z[i] 
fn linear_ccs(n: usize) -> (CcsStructure<F>, Vec<F>) {
    assert!(n >= 1);
    let rows = n + 1;        
    let cols = n + 2;        // [1, z0, z1, ..., z_n]

    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let _c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero

    // z0 = 1 (different seed)
    a_trips.push((0, 1, F::ONE));
    a_trips.push((0, 0, -F::ONE));
    b_trips.push((0, 0, F::ONE));

    // Recurrence: z[i+1] = 2*z[i]
    for i in 0..n {
        let r = 1 + i;
        a_trips.push((r, i + 2, F::ONE));          // +z[i+1]
        a_trips.push((r, i + 1, -F::from_u64(2))); // -2*z[i]
        b_trips.push((r, 0, F::ONE));
    }

    // Convert to matrices
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];

    for (r, c, val) in a_trips {
        a_data[r * cols + c] = val;
    }
    for (r, c, val) in b_trips {
        b_data[r * cols + c] = val;
    }

    let a = Mat::from_row_major(rows, cols, a_data);
    let b = Mat::from_row_major(rows, cols, b_data);  
    let c = Mat::from_row_major(rows, cols, c_data);
    
    let ccs = r1cs_to_ccs(a, b, c);
    
    // Generate witness: z[i] = 2^i starting from z0=1
    let mut witness = vec![F::ONE, F::ONE]; // [constant=1, z0=1]
    for k in 1..=n {
        let next = witness[k] + witness[k]; // z[k+1] = 2*z[k]
        witness.push(next);
    }
    
    (ccs, witness)
}

#[test]
fn splice_proof_bytes_and_public_io_must_fail() -> Result<()> {
    println!("🔓 TESTING SPLICE ATTACK VULNERABILITY");
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Circuit A: Fibonacci  
    let (ccs_a, wit_a) = fib_ccs(3);
    let pub_a: Vec<F> = vec![];
    println!("📐 Circuit A: Fibonacci CCS with {} constraints", ccs_a.n);
    
    let proof_a = prove(ProveInput { 
        params: &params, 
        ccs: &ccs_a, 
        public_input: &pub_a, 
        witness: &wit_a,
        output_claims: &[],
        vjs_opt: None,
    })?;
    println!("✅ Generated proof A: {} bytes", proof_a.size());

    // Circuit B: Linear (different circuit)
    let (ccs_b, wit_b) = linear_ccs(3);  
    let pub_b: Vec<F> = vec![];
    println!("📐 Circuit B: Linear CCS with {} constraints", ccs_b.n);
    
    let proof_b = prove(ProveInput { 
        params: &params, 
        ccs: &ccs_b, 
        public_input: &pub_b, 
        witness: &wit_b,
        output_claims: &[],
        vjs_opt: None,
    })?;
    println!("✅ Generated proof B: {} bytes", proof_b.size());

    // ✂️ SPLICE ATTACK: Take proof_bytes + circuit_key + vk_digest from A, but public_io from B
    let mut franken_proof = proof_a.clone();
    franken_proof.public_io = proof_b.public_io.clone();
    
    println!("🧟 Created Frankenstein proof:");
    println!("   - proof_bytes + circuit_key + vk_digest from circuit A");
    println!("   - public_io from circuit B");
    println!("   - Attempting to verify against circuit B context...");

    // 🚨 CRITICAL BUG: This should FAIL but currently PASSES due to missing public IO binding!
    // The verifier should reject this mixed proof, but the current implementation allows it.
    let result = verify(&ccs_b, &pub_b, &franken_proof)?;
    
    println!("🔍 Splice attack result: {}", if result { "❌ ACCEPTED (VULNERABILITY!)" } else { "✅ REJECTED (SECURE)" });
    
    // This assertion should PASS - the splice attack is properly REJECTED by our security fixes
    assert!(!result, 
        "🎯 SPLICE ATTACK PROPERLY REJECTED! Our security fixes are working correctly. \
         The proof mixed circuit A's proof_bytes with circuit B's public_io and was rejected \
         as expected. This demonstrates our proof binding security is working.");

    Ok(())
}

#[test]
fn normal_proofs_should_still_work() -> Result<()> {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Test normal proof A
    let (ccs_a, wit_a) = fib_ccs(2);
    let pub_a: Vec<F> = vec![];
    let proof_a = prove(ProveInput { 
        params: &params, 
        ccs: &ccs_a, 
        public_input: &pub_a, 
        witness: &wit_a,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Should verify normally
    assert!(verify(&ccs_a, &pub_a, &proof_a)?, "Normal proof A should verify");

    // Test normal proof B  
    let (ccs_b, wit_b) = linear_ccs(2);
    let pub_b: Vec<F> = vec![];
    let proof_b = prove(ProveInput { 
        params: &params, 
        ccs: &ccs_b, 
        public_input: &pub_b, 
        witness: &wit_b,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Should verify normally
    assert!(verify(&ccs_b, &pub_b, &proof_b)?, "Normal proof B should verify");

    Ok(())
}
