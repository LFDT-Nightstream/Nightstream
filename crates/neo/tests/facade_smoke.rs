//! Smoke test for the Neo facade
//! 
//! This test ensures the facade's prove() and verify() functions work correctly
//! on a minimal circuit, validating the end-to-end API.

use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
use neo_ccs::{Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;
use anyhow::Result;

/// Create a minimal CCS with 3 rows (ℓ=2 after padding): trivial constraints 0 * z0 = 0
fn minimal_identity_ccs() -> CcsStructure<F> {
    // R1CS with 3 rows to ensure ℓ=2 after padding to 4 rows
    // All constraints: 0 * z0 = 0 (trivially satisfied)
    // Variables: [z0, z1]
    let rows = 3;
    let cols = 2;
    
    let a = vec![F::ZERO; rows * cols];  // All zeros
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];  // All zeros
    
    // All rows: 0 * z0 = 0 (trivially satisfied)
    b[0 * cols + 0] = F::ONE;   // Row 0: multiply by z0
    b[1 * cols + 0] = F::ONE;   // Row 1: multiply by z0  
    b[2 * cols + 0] = F::ONE;   // Row 2: multiply by z0
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn facade_smoke_test() -> Result<()> {
    // Setup
    let ccs = minimal_identity_ccs();
    let witness = vec![F::ONE, F::from_u64(42)]; // [z0=1, z1=42] (satisfies 0*z0=0 for all rows)
    let public_input = vec![]; // No public inputs for this simple test
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2); // Small parameters for fast test
    
    // Prove
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs, 
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
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
    // Note: With trivial constraints (0*z0=0), the step CCS is always satisfied.
    // However, proof generation can still fail due to other checks (e.g., witness length mismatch)
    let ccs = minimal_identity_ccs();
    let invalid_witness = vec![F::from_u64(99)]; // Wrong length: expected 2 variables, got 1
    let public_input = vec![];
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input, 
        witness: &invalid_witness,
        output_claims: &[],
        vjs_opt: None,
    };
    
    // This should fail during proof generation due to witness length mismatch
    let result = prove(prove_input);
    assert!(result.is_err(), "Proof generation should fail for wrong-length witness");
}
