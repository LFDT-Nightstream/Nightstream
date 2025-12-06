//! Tests for M₀ = I_n validation required by Ajtai/NC pipeline.

#![allow(non_snake_case)]

use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Test that a valid square CCS with M₀ = I passes validation
#[test]
fn test_identity_validation_valid_square_ccs() {
    // Create a simple square R1CS (4x4)
    let n = 4;
    let m = 4;

    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    // Simple constraint: z[0] * z[1] = z[2]
    A[(0, 0)] = F::ONE;
    B[(0, 1)] = F::ONE;
    C[(0, 2)] = F::ONE;

    // r1cs_to_ccs should produce identity-first CCS for square input
    let ccs = r1cs_to_ccs(A, B, C);

    // Should pass validation
    assert!(ccs.assert_m0_is_identity_for_nc().is_ok());
}

/// Test that a non-square CCS fails validation with clear error
#[test]
fn test_identity_validation_fails_non_square() {
    // Create rectangular R1CS (1 constraint × 4 variables)
    let n = 1;
    let m = 4;

    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    A[(0, 1)] = F::ONE;
    A[(0, 2)] = F::ONE;
    B[(0, 0)] = F::ONE;
    C[(0, 3)] = F::ONE;

    // r1cs_to_ccs produces 3-matrix form for non-square (legacy)
    let ccs = r1cs_to_ccs(A, B, C);

    // Should fail validation
    let result = ccs.assert_m0_is_identity_for_nc();
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("square CCS"));
    assert!(err_msg.contains("1×4"));
}

/// Test that CCS with non-identity M₀ fails validation
#[test]
fn test_identity_validation_fails_non_identity_m0() {
    // Manually construct a square CCS where M₀ is NOT identity
    let n = 3;
    let m = 3;

    // M₀ is NOT identity (has 2s on diagonal)
    let mut m0 = Mat::zero(n, m, F::ZERO);
    m0[(0, 0)] = F::from_u64(2);
    m0[(1, 1)] = F::from_u64(2);
    m0[(2, 2)] = F::from_u64(2);

    let m1 = Mat::identity(n);
    let m2 = Mat::identity(n);

    // Create a simple polynomial f(X0, X1, X2) = X0
    let f = SparsePoly::new(
        3,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }],
    );

    let ccs = CcsStructure::new(vec![m0, m1, m2], f).expect("valid CCS structure");

    // Should fail validation
    let result = ccs.assert_m0_is_identity_for_nc();
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("M₀ = I_n"));
    assert!(err_msg.contains("identity matrix"));
}

/// Test that ensure_identity_first followed by validation works
#[test]
fn test_ensure_identity_first_then_validate() {
    // Create a rectangular R1CS
    let n = 2;
    let m = 3;

    let mut A = Mat::zero(n, m, F::ZERO);
    let mut B = Mat::zero(n, m, F::ZERO);
    let mut C = Mat::zero(n, m, F::ZERO);

    A[(0, 0)] = F::ONE;
    B[(0, 1)] = F::ONE;
    C[(0, 2)] = F::ONE;

    let ccs = r1cs_to_ccs(A, B, C);

    // Original rectangular CCS should fail
    assert!(ccs.assert_m0_is_identity_for_nc().is_err());

    // After ensure_identity_first (which leaves rectangular unchanged)
    let ccs_normalized = ccs.ensure_identity_first().expect("normalize");

    // Should still fail because it's still rectangular
    assert!(ccs_normalized.assert_m0_is_identity_for_nc().is_err());
}

/// Test the happy path: square R1CS → identity-first CCS → validation passes
#[test]
fn test_happy_path_square_r1cs_to_validated_ccs() {
    // Create a proper square R1CS (5x5)
    let n = 5;

    let mut A = Mat::zero(n, n, F::ZERO);
    let mut B = Mat::zero(n, n, F::ZERO);
    let mut C = Mat::zero(n, n, F::ZERO);

    // Row 0: (z0 + z1) * z2 = z3
    A[(0, 0)] = F::ONE;
    A[(0, 1)] = F::ONE;
    B[(0, 2)] = F::ONE;
    C[(0, 3)] = F::ONE;

    // Row 1: z3 * z1 = z4
    A[(1, 3)] = F::ONE;
    B[(1, 1)] = F::ONE;
    C[(1, 4)] = F::ONE;

    // Rows 2-4: boolean constraints (padding)
    A[(2, 0)] = F::ONE;
    B[(2, 0)] = F::ONE;
    C[(2, 0)] = F::ONE;

    A[(3, 1)] = F::ONE;
    B[(3, 1)] = F::ONE;
    C[(3, 1)] = F::ONE;

    A[(4, 2)] = F::ONE;
    B[(4, 2)] = F::ONE;
    C[(4, 2)] = F::ONE;

    // Convert to CCS (should be identity-first for square)
    let ccs = r1cs_to_ccs(A, B, C);

    // Verify it's square
    assert_eq!(ccs.n, ccs.m);
    assert_eq!(ccs.n, n);

    // Verify M₀ is identity
    assert!(ccs.matrices[0].is_identity());

    // Validation should pass
    assert!(ccs.assert_m0_is_identity_for_nc().is_ok());

    // ensure_identity_first should be a no-op
    let ccs_normalized = ccs.ensure_identity_first().expect("normalize");
    assert!(ccs_normalized.assert_m0_is_identity_for_nc().is_ok());
}

/// Test error message quality for debugging
#[test]
fn test_validation_error_messages_are_helpful() {
    // Non-square case
    {
        let n = 2;
        let m = 5;
        let mut A = Mat::zero(n, m, F::ZERO);
        let mut B = Mat::zero(n, m, F::ZERO);
        let mut C = Mat::zero(n, m, F::ZERO);

        A[(0, 0)] = F::ONE;
        B[(0, 0)] = F::ONE;
        C[(0, 0)] = F::ONE;

        let ccs = r1cs_to_ccs(A, B, C);
        let err = ccs.assert_m0_is_identity_for_nc().unwrap_err();
        let msg = err.to_string();

        // Check error message contains helpful info
        assert!(msg.contains("square"), "Error should mention 'square': {}", msg);
        assert!(msg.contains("2×5"), "Error should show actual dimensions: {}", msg);
        assert!(msg.contains("pad"), "Error should suggest padding: {}", msg);
    }

    // Non-identity M₀ case
    {
        let n = 3;
        let mut m0 = Mat::zero(n, n, F::ZERO);
        // Make it diagonal but not identity
        m0[(0, 0)] = F::from_u64(5);
        m0[(1, 1)] = F::from_u64(5);
        m0[(2, 2)] = F::from_u64(5);

        let m1 = Mat::identity(n);
        let f = SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            }],
        );

        let ccs = CcsStructure::new(vec![m0, m1], f).expect("valid structure");
        let err = ccs.assert_m0_is_identity_for_nc().unwrap_err();
        let msg = err.to_string();

        // Check error message is specific
        assert!(msg.contains("M₀"), "Error should mention M₀: {}", msg);
        assert!(msg.contains("identity"), "Error should mention identity: {}", msg);
        assert!(
            msg.contains("ensure_identity_first"),
            "Error should suggest fix: {}",
            msg
        );
    }
}
