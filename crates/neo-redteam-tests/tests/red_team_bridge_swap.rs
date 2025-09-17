// neo-tests/tests/red_team_bridge_swap.rs
#![cfg(feature = "redteam")]

//! Red-team test: Cross-proof swap  
//! 
//! Intent: produce proofs for two different witnesses; try to verify each proof 
//! against the other's CCS/IO. This exercises the **single transcript** discipline 
//! across folding→SNARK: if public IO (which implicitly binds the commitments) 
//! doesn't match, verification fails.

use neo::{prove, verify, ProveInput, NeoParams, F};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn create_ccs_with_constant(constant: u64) -> neo_ccs::CcsStructure<F> {
    // CCS: (z0 - constant) * 1 = 0  → forces z0 = constant
    let a = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::from_u64(constant), F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn bridge_rejects_cross_proof_swap() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Create two different CCS structures and corresponding witnesses
    let ccs_a = create_ccs_with_constant(5); // z0 = 5
    let witness_a = vec![F::from_u64(5), F::from_u64(0)]; // satisfies z0 = 5
    let public_input_a = vec![];
    
    let ccs_b = create_ccs_with_constant(7); // z0 = 7
    let witness_b = vec![F::from_u64(7), F::from_u64(0)]; // satisfies z0 = 7
    let public_input_b = vec![];
    
    // Generate proofs for each CCS/witness pair
    let proof_a = prove(ProveInput {
        params: &params,
        ccs: &ccs_a,
        public_input: &public_input_a,
        witness: &witness_a,
        output_claims: &[],
        vjs_opt: None,
    }).expect("Proof A should succeed");
    
    let proof_b = prove(ProveInput {
        params: &params,
        ccs: &ccs_b,
        public_input: &public_input_b,
        witness: &witness_b,
        output_claims: &[],
        vjs_opt: None,
    }).expect("Proof B should succeed");
    
    // Verify that each proof works with its own CCS (sanity check)
    assert!(verify(&ccs_a, &public_input_a, &proof_a).unwrap(), 
        "Proof A should verify against CCS A");
    assert!(verify(&ccs_b, &public_input_b, &proof_b).unwrap(), 
        "Proof B should verify against CCS B");
        
    // Sanity check: verify the header digests differ (contexts are distinguishable)
    let tail_a = &proof_a.public_io[proof_a.public_io.len()-32..];
    let tail_b = &proof_b.public_io[proof_b.public_io.len()-32..];
    assert_ne!(tail_a, tail_b,
        "Setup bug: (ccs, public_input) contexts A and B are indistinguishable to the bridge");
    
    // Now try cross-verification (should fail)
    // Intentionally verify each proof against the *other* CCS (and its IO)
    let cross_verify_a = verify(&ccs_b, &public_input_b, &proof_a);  // proof_a (for ccs_a) vs ccs_b
    let cross_verify_b = verify(&ccs_a, &public_input_a, &proof_b);  // proof_b (for ccs_b) vs ccs_a
    
    // Handle both error and false cases
    match cross_verify_a {
        Ok(false) => {
            println!("✅ Bridge correctly rejects proof A against CCS B/IO B (returned false)");
        }
        Err(_) => {
            println!("✅ Bridge correctly rejects proof A against CCS B/IO B (returned error)");
        }
        Ok(true) => {
            panic!("Cross-verification A→B should not succeed");
        }
    }
    
    match cross_verify_b {
        Ok(false) => {
            println!("✅ Bridge correctly rejects proof B against CCS A/IO A (returned false)");
        }
        Err(_) => {
            println!("✅ Bridge correctly rejects proof B against CCS A/IO A (returned error)");
        }
        Ok(true) => {
            panic!("Cross-verification B→A should not succeed");
        }
    }
}

#[test]
fn bridge_rejects_different_public_inputs() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Use same CCS but different public inputs
    let ccs = create_ccs_with_constant(5);
    let witness = vec![F::from_u64(5), F::from_u64(0)];
    
    let public_input_empty = vec![];
    let public_input_extra = vec![F::from_u64(42)]; // Extra public input
    
    // Generate proof with empty public input
    let proof = prove(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input_empty,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    }).expect("Proof should succeed");
    
    // Verify against original public input (should work)
    assert!(verify(&ccs, &public_input_empty, &proof).unwrap(), 
        "Proof should verify against original public input");
    
    // Try to verify against different public input (should fail)
    let different_verify = verify(&ccs, &public_input_extra, &proof);
    
    match different_verify {
        Ok(false) => {
            println!("✅ Bridge correctly rejects proof with different public input (returned false)");
        }
        Err(_) => {
            println!("✅ Bridge correctly rejects proof with different public input (returned error)");
        }
        Ok(true) => {
            panic!("Verification with different public input should not succeed");
        }
    }
}

