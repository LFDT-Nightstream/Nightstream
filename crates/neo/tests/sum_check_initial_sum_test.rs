///! Test to verify sum-check correctness:
///! 1. We check initial_sum (T), not Q(r), for satisfiability
///! 2. Valid witnesses pass verification
///! 3. Invalid witnesses are caught by the verifier (for ℓ >= 2 CCS)
///!
///! Note: Both tests use 3-row CCS (ℓ=2 after padding to 4 rows) because single-row CCS (ℓ=1)
///! cannot have invalid witnesses detected at verification time due to augmented CCS glue.
///! We need at least 3 rows to get ℓ >= 2 after power-of-2 padding.

use neo::{F, NeoParams, Accumulator};
use neo::ivc::{prove_ivc_step_with_extractor, verify_ivc_step, StepBindingSpec, LastNExtractor};
use neo_ccs::{CcsStructure, relations::check_ccs_rowwise_zero, Mat, SparsePoly, Term};
use p3_field::PrimeCharacteristicRing;

/// Build a 3-row CCS directly with valid witness (need ℓ >= 2, which requires >= 3 rows)
/// CCS polynomial: f(M0, M1, M2) = M0·M1 - M2
/// Each row i: (M0·z)[i] * (M1·z)[i] - (M2·z)[i] = 0
/// Witness format: [const1, z0, z1]
/// Constraints:
///   Row 0: z0 * z0 = z0  (z0 is boolean)
///   Row 1: z1 * z1 = z1  (z1 is boolean)  
///   Row 2: z0 * z1 = z0  (if z0=1, then z1 must be 1)
/// Valid witness: [1, 1, 1]
fn simple_ccs_and_witness_valid() -> (CcsStructure<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // Create 3 matrices for 3 rows, 3 columns [const1, z0, z1]
    let m0 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
    ]);
    let m1 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ZERO, F::ONE,   // Row 2: z1
    ]);
    let m2 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
    ]);
    
    // CCS polynomial: f(X0, X1, X2) = X0·X1 - X2
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },   // X0 * X1
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -X2
    ];
    let f = SparsePoly::new(3, terms);
    
    let ccs = CcsStructure::new(vec![m0, m1, m2], f)
        .expect("valid CCS structure");

    // Valid witness: [const1=1, z0=1, z1=1]
    // The first element must be 1 for const-1 binding
    // Row 0: 1*1 - 1 = 0 ✓
    // Row 1: 1*1 - 1 = 0 ✓
    // Row 2: 1*1 - 1 = 0 ✓
    let witness = vec![F::ONE, F::ONE, F::ONE];  // [const1, z0, z1]

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,  // First element is the constant 1
    };
    let extractor = LastNExtractor { n: 0 };

    (ccs, witness, binding, extractor)
}

/// 3-row CCS with invalid witness to test verifier rejection (ℓ >= 2 required)
/// Same structure as valid, but with invalid witness
/// Witness format: [const1, z0, z1]
/// Constraints:
///   Row 0: z0 * z0 = z0  (z0 is boolean)
///   Row 1: z1 * z1 = z1  (z1 is boolean)  
///   Row 2: z0 * z1 = z0  (if z0=1, then z1 must be 1)
/// Invalid witness: [1, 1, 5] violates rows 1 and 2
fn simple_ccs_and_witness_invalid() -> (CcsStructure<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // Same structure as valid
    let m0 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
    ]);
    let m1 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ZERO, F::ONE,   // Row 2: z1
    ]);
    let m2 = Mat::from_row_major(3, 3, vec![
        F::ZERO, F::ONE, F::ZERO,   // Row 0: z0
        F::ZERO, F::ZERO, F::ONE,   // Row 1: z1
        F::ZERO, F::ONE, F::ZERO,   // Row 2: z0
    ]);
    
    // CCS polynomial: f(X0, X1, X2) = X0·X1 - X2
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },   // X0 * X1
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -X2
    ];
    let f = SparsePoly::new(3, terms);
    
    let ccs = CcsStructure::new(vec![m0, m1, m2], f)
        .expect("valid CCS structure");

    // Invalid witness: [const1=1, z0=1, z1=5]
    // Row 0: 1*1 - 1 = 0 ✓
    // Row 1: 5*5 - 5 = 25-5 = 20 ≠ 0 ❌
    // Row 2: 1*5 - 1 = 5-1 = 4 ≠ 0 ❌
    let witness = vec![F::ONE, F::ONE, F::from_u64(5)];  // [const1, z0, z1]

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,  // First element is the constant 1
    };
    let extractor = LastNExtractor { n: 0 };

    (ccs, witness, binding, extractor)
}

#[test]
fn test_valid_witness_passes_verification() {
    let (ccs, witness, binding, extractor) = simple_ccs_and_witness_valid();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    // Sanity: witness satisfies CCS rowwise
    check_ccs_rowwise_zero(&ccs, &[], &witness).expect("witness should satisfy CCS");

    // Prove
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &ccs,
        &witness,
        &acc,
        0,
        None,
        &extractor,
        &binding,
    ).expect("prove should succeed for valid witness");

    // Verify
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc,
        &binding,
        &params,
        None,
    ).expect("verify should not error");

    assert!(ok, "Valid witness should pass verification");
}

#[test]
fn test_invalid_witness_is_caught() {
    let (ccs, witness, binding, extractor) = simple_ccs_and_witness_invalid();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    // Sanity: witness does NOT satisfy CCS rowwise
    assert!(check_ccs_rowwise_zero(&ccs, &[], &witness).is_err(), "witness should NOT satisfy CCS");

    // The prover doesn't check rowwise satisfaction for IVC (by design),
    // so it can produce a proof for an invalid witness. The verifier should catch it.
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &ccs,
        &witness,
        &acc,
        0,
        None,
        &extractor,
        &binding,
    ).expect("prove succeeds (soundness check is in verify)");

    // Verify - should REJECT because witness doesn't satisfy CCS
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc,
        &binding,
        &params,
        None,
    ).expect("verify should not error");

    // With ℓ >= 2 (3-row CCS → padded to 4 → ℓ=2), the verifier can detect invalid witnesses
    // by checking initial_sum == 0 for base case (c_coords.is_empty())
    assert!(!ok, "Invalid witness should be rejected by verifier for ℓ >= 2 CCS");
}

/// Build a 9-row CCS to achieve ℓ=4 (9 rows → padded to 16 → ℓ = log₂(16) = 4)
/// CCS polynomial: f(M0, M1, M2) = M0·M1 - M2
/// Witness format: [const1, z0, z1, z2]
/// Constraints: All rows enforce z_i * z_i = z_i (boolean constraints)
/// Valid witness: [1, 1, 1, 1]
fn large_ccs_and_witness_valid() -> (CcsStructure<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // Create 9 rows, 4 columns [const1, z0, z1, z2]
    // Row 0-2: z0 * z0 = z0 (repeated 3 times)
    // Row 3-5: z1 * z1 = z1 (repeated 3 times)
    // Row 6-8: z2 * z2 = z2 (repeated 3 times)
    let m0 = Mat::from_row_major(9, 4, vec![
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 0: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 1: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 3: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 4: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 5: z1
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 6: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 7: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 8: z2
    ]);
    let m1 = Mat::from_row_major(9, 4, vec![
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 0: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 1: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 3: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 4: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 5: z1
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 6: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 7: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 8: z2
    ]);
    let m2 = Mat::from_row_major(9, 4, vec![
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 0: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 1: z0
        F::ZERO, F::ONE, F::ZERO, F::ZERO,   // Row 2: z0
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 3: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 4: z1
        F::ZERO, F::ZERO, F::ONE, F::ZERO,   // Row 5: z1
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 6: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 7: z2
        F::ZERO, F::ZERO, F::ZERO, F::ONE,   // Row 8: z2
    ]);
    
    // CCS polynomial: f(X0, X1, X2) = X0·X1 - X2
    let terms = vec![
        Term { coeff: F::ONE, exps: vec![1, 1, 0] },   // X0 * X1
        Term { coeff: -F::ONE, exps: vec![0, 0, 1] },  // -X2
    ];
    let f = SparsePoly::new(3, terms);
    
    let ccs = CcsStructure::new(vec![m0, m1, m2], f)
        .expect("valid CCS structure");

    // Valid witness: [const1=1, z0=1, z1=1, z2=1]
    // All rows: 1*1 - 1 = 0 ✓
    let witness = vec![F::ONE, F::ONE, F::ONE, F::ONE];

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let extractor = LastNExtractor { n: 0 };

    (ccs, witness, binding, extractor)
}

#[test]
fn test_large_ccs_ell_4() {
    let (ccs, witness, binding, extractor) = large_ccs_and_witness_valid();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };

    // Sanity: witness satisfies CCS rowwise
    check_ccs_rowwise_zero(&ccs, &[], &witness).expect("witness should satisfy CCS");

    // Prove
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &ccs,
        &witness,
        &acc,
        0,
        None,
        &extractor,
        &binding,
    ).expect("prove should succeed for valid witness");

    // Verify
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc,
        &binding,
        &params,
        None,
    ).expect("verify should not error");

    // With ℓ=4 (9-row CCS → padded to 16 → ℓ=4), valid witness should pass
    assert!(ok, "Valid witness should pass verification for ℓ=4 CCS");
}
