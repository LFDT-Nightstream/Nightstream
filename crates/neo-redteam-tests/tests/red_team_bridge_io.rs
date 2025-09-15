// neo-tests/tests/red_team_bridge_io.rs
#![cfg(feature = "redteam")]

//! Red-team test: Public-IO binding / transcript split attempt
//! 
//! Intent: generate a valid proof, then **change public IO** before verification 
//! to simulate "re-binding `Z` without binding to `c`". Verification must fail 
//! because the bridge requires **binding identical public-IO bytes into the 
//! challenger** on both sides.

use neo::{prove, verify, ProveInput, NeoParams, F};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs() -> neo_ccs::CcsStructure<F> {
    // 1-row: (z0 - z1) * 1 = 0  → forces z0 = z1
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn bridge_rejects_public_io_tamper() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Create a valid CCS and witness
    let ccs = tiny_ccs();
    let witness = vec![F::from_u64(5), F::from_u64(5)]; // z0 = z1 = 5
    let mut public_input = vec![]; // no public inputs initially
    
    // Generate proof with original public inputs
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    };
    
    let proof = prove(prove_input).expect("Proof generation should succeed");
    
    // Tamper public IO bound into the Poseidon2 transcript
    public_input.push(F::from_u64(42)); // Add unexpected public input
    
    // Verification must fail under tampered IO
    let verification_result = verify(&ccs, &public_input, &proof);
    
    match verification_result {
        Ok(false) => {
            // Good: verification correctly rejected the tampered public IO
            println!("✅ Bridge correctly rejects proof with tampered public IO (returned false)");
        }
        Err(_) => {
            // Also good: tampered proof caused verification to fail with error
            println!("✅ Bridge correctly rejects proof with tampered public IO (returned error)");
        }
        Ok(true) => {
            panic!("Proof with tampered public IO was incorrectly accepted as valid");
        }
    }
}

#[test]
fn bridge_rejects_different_ccs() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Create original CCS and witness
    let ccs1 = tiny_ccs();
    let witness = vec![F::from_u64(5), F::from_u64(5)]; // z0 = z1 = 5
    let public_input = vec![];
    
    // Generate proof with original CCS
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs1,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    };
    
    let proof = prove(prove_input).expect("Proof generation should succeed");
    
    // Create a different CCS (different constraint matrix)
    let a2 = Mat::from_row_major(1, 2, vec![F::from_u64(2), -F::ONE]); // 2*z0 - z1 = 0
    let b2 = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c2 = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    let ccs2 = r1cs_to_ccs(a2, b2, c2);
    
    // Try to verify proof against different CCS
    let verification_result = verify(&ccs2, &public_input, &proof);
    
    match verification_result {
        Ok(false) => {
            println!("✅ Bridge correctly rejects proof verified against different CCS (returned false)");
        }
        Err(_) => {
            println!("✅ Bridge correctly rejects proof verified against different CCS (returned error)");
        }
        Ok(true) => {
            panic!("Proof verified against different CCS was incorrectly accepted as valid");
        }
    }
}

