//! Regression test: Π-CCS verification with minimal R1CS (ℓ=2) and y_len=0.
//!
//! This test encodes a 3-row R1CS (pads to 4, giving ℓ=2) with a satisfying witness.
//! Verifies that the system works correctly with minimal constraint count.

use neo::{F, NeoParams};
use neo::{Accumulator, StepBindingSpec, LastNExtractor, prove_ivc_step_with_extractor, verify_ivc_step};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a 3-row R1CS (ℓ=2 after padding) with 3 variables
/// Constraints: All trivially satisfied (0 * 1 = 0 for all rows)
fn minimal_r1cs_ell2() -> neo_ccs::CcsStructure<F> {
    let rows = 4;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 3;
    
    let a = vec![F::ZERO; rows * cols];  // All zeros
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];  // All zeros
    
    // All rows: 0 * z0 = 0 (trivially satisfied for any z0)
    b[0 * cols + 0] = F::ONE;   // Row 0: multiply by z0
    b[1 * cols + 0] = F::ONE;   // Row 1: multiply by z0  
    b[2 * cols + 0] = F::ONE;   // Row 2: multiply by z0
    b[3 * cols + 0] = F::ONE;   // Row 3: multiply by z0
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn pi_ccs_single_row_deg_shape() {
    // 1) Step CCS and satisfying witness
    let ccs = minimal_r1cs_ell2();
    let witness = vec![F::ONE, F::ZERO, F::ZERO]; // All constraints 0*z0=0 trivially satisfied

    // Sanity: row-wise check must pass
    neo_ccs::relations::check_ccs_rowwise_zero(&ccs, &[], &witness)
        .expect("witness should satisfy CCS row-wise");

    // 2) IVC setup with y_len=0 and no app inputs
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let acc0 = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let extractor = LastNExtractor { n: 0 };

    // 3) Prove single step (should succeed)
    let step_res = prove_ivc_step_with_extractor(
        &params,
        &ccs,
        &witness,
        &acc0,
        0,
        None,
        &extractor,
        &binding,
    ).expect("prove_ivc_step_with_extractor should succeed");

    // 4) Verify: Should succeed with ℓ=2
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc0,
        &binding,
        &params,
        None,
    ).expect("verify_ivc_step should not error");

    assert!(ok, "Verifier should accept valid witness for ℓ=2 CCS");
}
