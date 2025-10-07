//! Regression test: Π-CCS verification for single-row R1CS with y_len=0.
//!
//! This test encodes a 1-row R1CS constraint (z0 - z1) * z0 = 0 with a satisfying witness [1, 1].
//! Row-wise CCS check passes.
//!
//! FIXED: The verifier now correctly accepts valid witnesses for ℓ=1 cases by skipping the
//! initial_sum == 0 check, since the augmented CCS carries a constant offset from const-1
//! binding and other glue, making initial_sum non-zero even for valid witnesses.

use neo::{F, NeoParams};
use neo::ivc::{Accumulator, StepBindingSpec, LastNExtractor, prove_ivc_step_with_extractor, verify_ivc_step};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a 1-row, 2-var R1CS CCS for (z0 - z1) * z0 = 0
fn tiny_r1cs_single_row() -> neo_ccs::CcsStructure<F> {
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[test]
fn pi_ccs_single_row_deg_shape() {
    // 1) Step CCS and satisfying witness
    let ccs = tiny_r1cs_single_row();
    let witness = vec![F::ONE, F::ONE]; // (1 - 1) * 1 = 0

    // Sanity: row-wise check must pass
    neo_ccs::relations::check_ccs_rowwise_zero(&ccs, &[], &witness)
        .expect("witness should satisfy CCS row-wise");

    // 2) IVC setup with y_len=0 and no app inputs
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let acc0 = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let binding = StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
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

    // 4) Verify: With the ℓ=1 fix, verifier now correctly ACCEPTS valid witnesses
    //    even though initial_sum is non-zero due to augmented CCS glue
    let ok = verify_ivc_step(
        &ccs,
        &step_res.proof,
        &acc0,
        &binding,
        &params,
        None,
    ).expect("verify_ivc_step should not error");

    assert!(ok, "Verifier should accept valid witness for ℓ=1 CCS");
}
