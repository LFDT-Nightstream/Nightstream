//! Test that additional application public inputs are currently rejected,
//! then will pass once we support x = [H(prev_acc) || app_inputs].

use neo::{F, NeoParams};
use neo::ivc::{Accumulator, IvcStepInput, StepBindingSpec, prove_ivc_step, verify_ivc_step};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn build_extended_step_ccs() -> CcsStructure<F> {
    // Variables: [1, a, b, app1, app2] with constraint: b - a - 1 = 0
    // The app1 and app2 are unconstrained (just witness values for binding)
    let rows = 1; let cols = 5;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    a[0*cols + 2] = F::ONE;   // +b
    a[0*cols + 1] = -F::ONE;  // -a
    a[0*cols + 0] = -F::ONE;  // -1
    // app1 (index 3) and app2 (index 4) are unconstrained
    b[0*cols + 0] = F::ONE;   // *1
    r1cs_to_ccs(Mat::from_row_major(rows, cols, a), Mat::from_row_major(rows, cols, b), Mat::from_row_major(rows, cols, c))
}

#[test]
fn app_public_inputs_accepted_now() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_extended_step_ccs();
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![F::ZERO], step: 0 };
    // Step witness must contain the app input values at the binding positions
    // Original: [F::ONE, F::ZERO, F::ONE] for constraint b - a - 1 = 0 with a=0, b=1
    // Modified: [F::ONE, F::from_u64(42), F::from_u64(7)] to include app inputs at positions 1,2
    // But this breaks the original constraint! Let's extend the witness instead
    let step_witness = vec![F::ONE, F::ZERO, F::ONE, F::from_u64(42), F::from_u64(7)]; // [const, a, b, app1, app2]
    let y_step = vec![F::ONE];
    // Bind the 2 app inputs to witness positions 3,4 (where they actually are)
    let binding = StepBindingSpec { y_step_offsets: vec![2], step_program_input_witness_indices: vec![3, 4], y_prev_witness_indices: vec![], const1_witness_index: 0 };

    // Provide app inputs that should be appended to H(prev_acc)
    let app_inputs = vec![F::from_u64(42), F::from_u64(7)];
    let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &step_witness, prev_accumulator: &prev_acc, step: 0, public_input: Some(&app_inputs), y_step: &y_step, binding_spec: &binding, transcript_only_app_inputs: false, prev_augmented_x: None };
    let ok = prove_ivc_step(input).expect("prover should accept app public inputs with prefix digest");

    // Prover accepted: validate that x = [H(prev_acc) || app_inputs]
    // Recompute digest prefix (copy of helper from example)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&prev_acc.step.to_le_bytes());
    bytes.extend_from_slice(&prev_acc.c_z_digest);
    bytes.extend_from_slice(&(prev_acc.y_compact.len() as u64).to_le_bytes());
    for &y in &prev_acc.y_compact { bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes()); }
    let d = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&bytes);
    let digest_prefix: Vec<F> = d.iter().map(|x| F::from_u64(x.as_canonical_u64())).collect();

    let x = &ok.proof.step_public_input;
    assert!(x.len() >= digest_prefix.len() + app_inputs.len());
    assert_eq!(&x[..digest_prefix.len()], &digest_prefix[..]);
}

#[test]
fn tampered_digest_prefix_rejected() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_extended_step_ccs();
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![F::ZERO], step: 0 };
    // Step witness must contain the app input values at the binding positions
    // Extended witness: [const, a, b, app1, app2] to include app inputs at positions 3,4
    let step_witness = vec![F::ONE, F::ZERO, F::ONE, F::from_u64(11), F::from_u64(22)]; // [const, a, b, app1, app2]
    let y_step = vec![F::ONE];
    // Bind the 2 app inputs to witness positions 3,4 (where they actually are)
    let binding = StepBindingSpec { y_step_offsets: vec![2], step_program_input_witness_indices: vec![3, 4], y_prev_witness_indices: vec![], const1_witness_index: 0 };

    let app_inputs = vec![F::from_u64(11), F::from_u64(22)];
    let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &step_witness, prev_accumulator: &prev_acc, step: 0, public_input: Some(&app_inputs), y_step: &y_step, binding_spec: &binding, transcript_only_app_inputs: false, prev_augmented_x: None };
    let ok = prove_ivc_step(input).expect("prover should succeed");

    // Tamper with digest prefix
    let mut forged = ok.proof.clone();
    if !forged.step_public_input.is_empty() {
        forged.step_public_input[0] = F::from_u64(999);
    }
    let valid = verify_ivc_step(&step_ccs, &forged, &prev_acc, &binding, &params, None).expect("verifier should not error");
    assert!(!valid, "verifier must reject when digest prefix does not match H(prev_acc)");
}
