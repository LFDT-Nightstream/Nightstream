//! Security tests for Neo SNARK proof binding to prevent replay attacks
//!
//! These tests ensure that proofs are properly bound to their specific
//! (ccs, public_input) context and cannot be replayed for different contexts.

use anyhow::Result;
use serial_test::serial;
use neo::{prove_spartan2, verify_spartan2, ProveInput, NeoParams, CcsStructure, F};
use neo_ccs::{check_ccs_rowwise_zero, r1cs_to_ccs, Mat};
use p3_field::PrimeCharacteristicRing;

/// Create CCS for constraint: x * 1 = x (always true) with ℓ=2
fn one_row_ccs_x_eq_x() -> CcsStructure<F> {
    // R1CS with 3 rows (pads to 4, giving ℓ=2) and 2 columns [1, x]:
    // Row 0: x * 1 = x  (the actual constraint)
    // Row 1-3: 0 * 1 = 0  (trivial padding to reach ℓ=2)
    let rows = 4usize;  // Minimum 4 rows required
    let cols = 2usize; // [const, x]

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];

    // Row 0: x * 1 = x
    a[0 * cols + 1] = F::ONE; // A picks x
    b[0 * cols + 0] = F::ONE; // B picks 1
    c[0 * cols + 1] = F::ONE; // C expects x
    
    // Rows 1-3: 0 * 1 = 0 (padding)
    b[1 * cols + 0] = F::ONE;
    b[2 * cols + 0] = F::ONE;
    b[3 * cols + 0] = F::ONE;

    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

/// Create CCS for constraint: x * 1 = 0 (different statement) with ℓ=2
fn one_row_ccs_x_eq_0() -> CcsStructure<F> {
    // R1CS with 3 rows (pads to 4, giving ℓ=2):
    // Row 0: x * 1 = 0 (the actual constraint - different from x_eq_x)
    // Row 1-3: 0 * 1 = 0  (trivial padding to reach ℓ=2)
    let rows = 4usize;  // Minimum 4 rows required
    let cols = 2usize;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let     c = vec![F::ZERO; rows * cols]; // all zeros => RHS == 0

    // Row 0: x * 1 = 0
    a[0 * cols + 1] = F::ONE; // A picks x
    b[0 * cols + 0] = F::ONE; // B picks 1
    
    // Rows 1-3: 0 * 1 = 0 (padding)
    b[1 * cols + 0] = F::ONE;
    b[2 * cols + 0] = F::ONE;
    b[3 * cols + 0] = F::ONE;

    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

/// Helper to create a simple test fixture
fn create_simple_fibonacci_fixture() -> (CcsStructure<F>, Vec<F>, Vec<F>, NeoParams) {
    // Create minimal Fibonacci CCS for n=1: z2 = z1 + z0 with z0=0, z1=1
    let rows = 4;  // 2 seed constraints + 1 recurrence + 1 dummy (minimum 4 required)  
    let cols = 4;  // [1, z0, z1, z2]
    
    // A matrix (constraint coefficients)
    let mut a_data = vec![F::ZERO; rows * cols];
    a_data[0 * cols + 1] = F::ONE;           // z0 = 0
    a_data[1 * cols + 2] = F::ONE;           // z1 coefficient  
    a_data[1 * cols + 0] = -F::ONE;          // -1 constant
    a_data[2 * cols + 3] = F::ONE;           // z2 coefficient
    a_data[2 * cols + 2] = -F::ONE;          // -z1
    a_data[2 * cols + 1] = -F::ONE;          // -z0
    
    // B matrix (always selects constant 1)
    let mut b_data = vec![F::ZERO; rows * cols];
    for i in 0..rows {
        b_data[i * cols + 0] = F::ONE;       // constant wire
    }
    
    // C matrix (always zero for this encoding)
    let c_data = vec![F::ZERO; rows * cols];
    
    let a = Mat::from_row_major(rows, cols, a_data);
    let b = Mat::from_row_major(rows, cols, b_data);  
    let c = Mat::from_row_major(rows, cols, c_data);
    
    let ccs = r1cs_to_ccs(a, b, c);
    
    // Witness: [1, 0, 1, 1] = [constant, z0, z1, z2] 
    let witness = vec![F::ONE, F::ZERO, F::ONE, F::ONE];
    let public_input = vec![]; // No public inputs
    
    // Verify the fixture is correct
    check_ccs_rowwise_zero(&ccs, &public_input, &witness)
        .expect("Test fixture should satisfy CCS");
    
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    
    (ccs, public_input, witness, params)
}

#[test]
fn test_verify_rejects_proof_with_tampered_public_input() -> Result<()> {
    let (ccs, public_input, witness, params) = create_simple_fibonacci_fixture();
    
    // Generate valid proof for the original context
    let proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs, 
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Verification should succeed with original context (use explicit VK bytes to avoid registry races)
    let vk_bytes = neo_spartan_bridge::export_vk_bytes(&proof.circuit_key).expect("export vk");
    let valid = neo::verify_with_vk(&ccs, &public_input, &proof, &vk_bytes)?;
    assert!(valid, "Proof should verify with original context");
    
    // Create tampered public input (add a field element)
    let mut tampered_public_input = public_input.clone();
    tampered_public_input.push(F::from_u64(42));
    
    // Verification should fail with tampered public input
    let invalid = verify_spartan2(&ccs, &tampered_public_input, &proof)?;
    assert!(!invalid, "Proof must not verify with tampered public input - this prevents replay attacks");
    
    Ok(())
}

#[test]  
fn test_verify_rejects_proof_with_different_ccs() -> Result<()> {
    let (ccs, public_input, witness, params) = create_simple_fibonacci_fixture();
    
    // Generate valid proof for the original CCS
    let proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input, 
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Create a different CCS by modifying one matrix entry
    let mut different_ccs = ccs.clone();
    if !different_ccs.matrices.is_empty() && different_ccs.matrices[0].rows() > 0 && different_ccs.matrices[0].cols() > 0 {
        // Modify the first matrix slightly 
        different_ccs.matrices[0][(0, 0)] = different_ccs.matrices[0][(0, 0)] + F::ONE;
    }
    
    // Verification should fail with different CCS
    let invalid = verify_spartan2(&different_ccs, &public_input, &proof)?;
    assert!(!invalid, "Proof must not verify with different CCS - this prevents cross-circuit replay attacks");
    
    Ok(())
}

#[test]
fn test_verify_succeeds_with_correct_context() -> Result<()> {
    let (ccs, public_input, witness, params) = create_simple_fibonacci_fixture();
    
    // Generate proof  
    let proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None, 
    })?;
    
    // Verification should succeed with exact same context (use explicit VK bytes)
    let vk_bytes = neo_spartan_bridge::export_vk_bytes(&proof.circuit_key).expect("export vk");
    let valid = neo::verify_with_vk(&ccs, &public_input, &proof, &vk_bytes)?;
    assert!(valid, "Proof should verify with correct matching context");
    
    Ok(())
}

#[test]
fn test_unsupported_proof_version_rejected() -> Result<()> {
    let (ccs, public_input, witness, params) = create_simple_fibonacci_fixture();
    
    // Generate valid proof
    let mut proof = prove_spartan2(ProveInput {
        params: &params, 
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Tamper with version
    proof.v = 99; // Unsupported version
    
    // Should get an error for unsupported version  
    let result = verify_spartan2(&ccs, &public_input, &proof);
    assert!(result.is_err(), "Should reject unsupported proof version");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("unsupported proof version"), "Error should mention unsupported version");
    
    Ok(())
}

#[test]
fn test_malformed_proof_rejected() -> Result<()> {
    let (ccs, public_input, witness, params) = create_simple_fibonacci_fixture();
    
    // Generate valid proof
    let mut proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs, 
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    })?;
    
    // Make public_io too short to contain context digest
    proof.public_io = vec![0u8; 16]; // Less than 32 bytes required
    
    // Should get an error for malformed proof
    let result = verify_spartan2(&ccs, &public_input, &proof);
    assert!(result.is_err(), "Should reject malformed proof with short public_io");
    
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("malformed proof"), "Error should mention malformed proof");
    
    Ok(())
}

// ========================================================================
// CRITICAL SECURITY REGRESSION TESTS FROM AI REVIEW
// These tests MUST fail if statement binding is not implemented properly
// ========================================================================

#[test]
#[serial]
fn rejects_wrong_ccs_or_public_input() -> Result<()> {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);

    // Statement A
    let ccs_a = one_row_ccs_x_eq_x();
    let z_a = vec![F::ONE, F::from_u64(5)]; // [const=1, x=5]
    let pub_a: Vec<F> = vec![]; // keep empty for the test

    let proof_a = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs_a,
        public_input: &pub_a,
        witness: &z_a,
        output_claims: &[],
        vjs_opt: None,
    })?;

    // Baseline: verifies for the statement it was created for
    assert!(verify_spartan2(&ccs_a, &pub_a, &proof_a)?);

    // Different statement/context - try to verify the proof against different CCS
    let ccs_b = one_row_ccs_x_eq_0(); // Different circuit
    let pub_b: Vec<F> = vec![F::ONE]; // Different public input

    // ❗ Now try to verify proof A against circuit B - this must be REJECTED
    // The proof was generated for circuit A but we're trying to verify against circuit B
    let result = verify_spartan2(&ccs_b, &pub_b, &proof_a);
    match result {
        Ok(false) => {
            // This is what we want - verification correctly rejected the mismatched proof
        }
        Ok(true) => {
            panic!("verifier must bind proof to (ccs, public_input) and reject mismatched context");
        }
        Err(_) => {
            // Also acceptable - the verifier detected a mismatch and returned an error
            // This can happen due to context digest mismatch or VK not found
        }
    }

    Ok(())
}

#[test]
#[serial]
fn rejects_tampered_public_io() -> Result<()> {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = one_row_ccs_x_eq_x();
    let z = vec![F::ONE, F::from_u64(5)];
    let public_input: Vec<F> = vec![];

    // Ensure VK registry starts clean for deterministic behavior
    neo_spartan_bridge::clear_vk_registry();

    let proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &z,
        output_claims: &[],
        vjs_opt: None,
    })?;

    // flip one byte in the header/public-IO binding
    let mut forged = proof.clone();
    assert!(!forged.public_io.is_empty(), "public_io should not be empty");
    forged.public_io[0] ^= 1;

    // ❗ Should be false after you reintroduce statement-binding.
    // Current code will (wrongly) return true because public_io is not checked.
    assert!(
        !verify_spartan2(&ccs, &public_input, &forged)?,
        "tampering with public-IO must be detected"
    );
    Ok(())
}

#[test]
#[serial]
fn fails_when_vk_registry_is_missing() -> Result<()> {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let ccs = one_row_ccs_x_eq_x();
    let z = vec![F::ONE, F::from_u64(5)];
    let public_input: Vec<F> = vec![];

    let proof = prove_spartan2(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &z,
        output_claims: &[],
        vjs_opt: None,
    })?;

    // Blow away the ephemeral VK cache and ensure verify fails hard.
    neo_spartan_bridge::clear_vk_registry();
    assert!(
        verify_spartan2(&ccs, &public_input, &proof).is_err(),
        "verify should error when VK for circuit_key is not registered"
    );
    Ok(())
}
