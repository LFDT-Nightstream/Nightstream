// neo-tests/tests/integration_smoke.rs

//! Basic integration smoke tests for the Neo protocol
//! 
//! These tests verify that the core Neo API works correctly for normal use cases.
//! For security and vulnerability testing, see the neo-redteam-tests crate.

use neo::{prove_spartan2, ProveInput, NeoParams, F};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Create a tiny CCS for testing: constraint (z0 - z1) = 0
fn tiny_ccs() -> neo_ccs::CcsStructure<F> {
    // CCS with 3 rows (pads to 4, ℓ=2):
    // Row 0: (z0 - z1) * 1 = 0  → forces z0 = z1
    // Row 1-2: 0 * 0 = 0 (padding)
    let a = Mat::from_row_major(3, 2, vec![
        F::ONE, -F::ONE,   // Row 0: z0 - z1
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    let b = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,   // Row 0: * 1
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    let c = Mat::from_row_major(3, 2, vec![
        F::ZERO, F::ZERO,  // Row 0: = 0
        F::ZERO, F::ZERO,  // Row 1: padding
        F::ZERO, F::ZERO,  // Row 2: padding
    ]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn neo_basic_prove_verify_works() {
    let ccs = tiny_ccs();

    // witness: [5, 5], choose z0=z1=5 → satisfies (z0 - z1) = 0
    let witness = vec![F::from_u64(5), F::from_u64(5)];
    let public_input = vec![]; // no public inputs for this test

    // Use minimal parameters suitable for testing
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);

    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
        vjs_opt: None,
    };

    // Generate proof
    let proof = prove_spartan2(prove_input).expect("prove should succeed");

    // Verify proof
    let verification_result = neo::verify_spartan2(&ccs, &public_input, &proof).expect("verify should not error");
    assert!(verification_result, "valid proof should verify successfully");
    
    println!("✅ Basic Neo prove/verify cycle works correctly");
}

#[test]
fn neo_constraint_violation_fails() {
    let ccs = tiny_ccs();

    // Invalid witness: [1, 5], violates (z0 - z1) = 0 since 1 ≠ 5
    let invalid_witness = vec![F::from_u64(1), F::from_u64(5)];
    let public_input = vec![];

    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);

    let prove_input = ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &invalid_witness,
        output_claims: &[],
        vjs_opt: None,
    };

    // Proof generation should fail with constraint violation
    let result = prove_spartan2(prove_input);
    assert!(result.is_err(), "prove with invalid witness should fail");
    
    if let Err(e) = result {
        assert!(e.to_string().contains("constraint"), "error should mention constraint violation");
        println!("✅ Neo correctly rejects invalid witness: {}", e);
    }
}

#[test]
fn neo_params_validation_works() {
    // This should succeed with reasonable parameters
    let good_params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    assert!(good_params.lambda >= 64, "security parameter should be reasonable");
    
    println!("✅ Neo parameter validation works correctly");
}
