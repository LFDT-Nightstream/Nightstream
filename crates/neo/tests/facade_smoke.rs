//! Smoke test for the Neo facade
//! 
//! This test ensures the facade's prove() and verify() functions work correctly
//! on a minimal circuit, validating the end-to-end API.

use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
use neo_ccs::{Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;
use anyhow::Result;

/// Create a minimal CCS: identity constraint z1 * 1 = z1 (always true)
fn minimal_identity_ccs() -> CcsStructure<F> {
    // R1CS matrices for: z1 * constant = z1 (tautology to avoid constraint issues)
    // Variables: [constant=1, z1] 
    let a = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);     // z1 coefficient  
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);     // constant 1
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ONE]);     // z1 result
    r1cs_to_ccs(a, b, c)
}

#[test]
fn facade_smoke_test() -> Result<()> {
    // Setup
    let ccs = minimal_identity_ccs();
    let witness = vec![F::ONE, F::from_u64(42)]; // [constant=1, z1=42] (satisfies z1 * 1 = z1)
    let public_input = vec![]; // No public inputs for this simple test
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2); // Small parameters for fast test
    
    // Prove
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs, 
        public_input: &public_input,
        witness: &witness,
    };
    
    let proof = prove(prove_input)?;
    
    // Basic proof structure validation for lean proofs (V2)
    assert_eq!(proof.v, 2, "Proof should have version 2 (lean proof)");
    assert_eq!(proof.circuit_key.len(), 32, "Circuit key should be 32 bytes");
    assert_eq!(proof.vk_digest.len(), 32, "VK digest should be 32 bytes");
    assert!(!proof.public_io.is_empty(), "Public IO should not be empty");
    assert!(!proof.proof_bytes.is_empty(), "Proof bytes should not be empty");
    assert!(proof.size() > 100, "Proof should have reasonable size");
    
    // Verify  
    let is_valid = verify(&ccs, &public_input, &proof)?;
    assert!(is_valid, "Proof should be valid");
    
    Ok(())
}

#[test]
fn facade_invalid_witness_should_fail() {
    let ccs = minimal_identity_ccs();
    let invalid_witness = vec![F::from_u64(2), F::from_u64(99)]; // [constant=2, z1=99], violates z1*2=z1
    let public_input = vec![];
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input, 
        witness: &invalid_witness,
    };
    
    // This should fail during proof generation due to unsatisfied constraints
    // The constraint z1*constant=z1 fails when constant≠1
    let result = prove(prove_input);
    assert!(result.is_err(), "Proof generation should fail for invalid witness");
}
