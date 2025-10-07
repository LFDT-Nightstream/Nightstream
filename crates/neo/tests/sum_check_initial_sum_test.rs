///! Test to verify sum-check correctness:
///! 1. We check initial_sum (T), not Q(r), for satisfiability
///! 2. Valid witnesses pass verification
///! 3. Invalid witnesses are caught by the verifier (for ℓ >= 2 CCS)
///!
///! Note: The invalid witness test uses a 3-row CCS (ℓ=2 after padding to 4 rows) because
///! single-row CCS (ℓ=1) cannot have invalid witnesses detected at verification time due to
///! augmented CCS glue. We need at least 3 rows to get ℓ >= 2 after power-of-2 padding.

use neo::{F, NeoParams, Accumulator};
use neo::ivc::{prove_ivc_step_with_extractor, verify_ivc_step, StepBindingSpec, LastNExtractor};
use neo_ccs::{CcsStructure, relations::check_ccs_rowwise_zero, Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a simple CCS: (z0 - z1) * z0 = 0
/// Valid witness: z0=1, z1=1
fn simple_ccs_and_witness_valid() -> (CcsStructure<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // One row, two vars: A = [1, -1], B = [1, 0], C = [0, 0]
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    let ccs = r1cs_to_ccs(a, b, c);

    // Valid witness: z0=1, z1=1 satisfies (1-1)*1=0
    let witness = vec![F::ONE, F::ONE];

    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let extractor = LastNExtractor { n: 0 };

    (ccs, witness, binding, extractor)
}

/// 3-row CCS with invalid witness to test verifier rejection (need ℓ >= 2, which requires >= 3 rows)
/// Constraints:
///   Row 0: (z0 - z1) * z0 = 0
///   Row 1: z1 * z1 = 0
///   Row 2: z0 * z0 = 0
/// Valid witness would be: z0=0, z1=0
/// Invalid witness: z0=1, z1=5 violates all rows
fn simple_ccs_and_witness_invalid() -> (CcsStructure<F>, Vec<F>, StepBindingSpec, LastNExtractor) {
    // Row 0: (z0 - z1) * z0 = 0
    // Row 1: z1 * z1 = 0
    // Row 2: z0 * z0 = 0
    let a = Mat::from_row_major(3, 2, vec![
        F::ONE, -F::ONE,  // Row 0: z0 - z1
        F::ZERO, F::ONE,  // Row 1: z1
        F::ONE, F::ZERO,  // Row 2: z0
    ]);
    let b = Mat::from_row_major(3, 2, vec![
        F::ONE, F::ZERO,  // Row 0: z0
        F::ZERO, F::ONE,  // Row 1: z1
        F::ONE, F::ZERO,  // Row 2: z0
    ]);
    let c = Mat::from_row_major(3, 2, vec![
        F::ZERO, F::ZERO, // Row 0: 0
        F::ZERO, F::ZERO, // Row 1: 0
        F::ZERO, F::ZERO, // Row 2: 0
    ]);
    let ccs = r1cs_to_ccs(a, b, c);

    // Invalid witness: z0=1, z1=5
    // Row 0: (1-5)*1 = -4 ≠ 0 ❌
    // Row 1: 5*5 = 25 ≠ 0 ❌
    // Row 2: 1*1 = 1 ≠ 0 ❌
    let witness = vec![F::ONE, F::from_u64(5)];

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

    // With ℓ >= 2 (2-row CCS), the verifier can detect invalid witnesses
    // by checking initial_sum == 0 for base case (c_coords.is_empty())
    assert!(!ok, "Invalid witness should be rejected by verifier for ℓ >= 2 CCS");
}

