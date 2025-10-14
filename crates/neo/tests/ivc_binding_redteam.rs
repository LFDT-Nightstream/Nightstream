//! Red-team tests for IVC step_x binding: x must equal H(prev_accumulator)

use neo::{F, NeoParams};
use neo::{Accumulator, IvcStepInput, StepBindingSpec, prove_ivc_step, verify_ivc_step, AppInputBinding};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn build_increment_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, next_x]
    // Constraint: next_x - prev_x - 1 = 0  => (next_x - prev_x - const) * 1 = 0
    let rows = 4;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 3;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // Row 0: (next_x - prev_x - 1) * 1 = 0
    a[0 * cols + 2] = F::ONE;     // + next_x
    a[0 * cols + 1] = -F::ONE;    // - prev_x
    a[0 * cols + 0] = -F::ONE;    // - const (represents -1)
    b[0 * cols + 0] = F::ONE;     // * 1

    // Rows 1-3: dummy constraints (0 * 1 = 0)
    for row in 1..4 {
        a[row * cols + 0] = F::ZERO;
        b[row * cols + 0] = F::ONE;
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

fn compute_x_digest(acc: &Accumulator) -> Vec<F> {
    // Reproduce the library’s accumulator serialization + Poseidon2 hash
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&acc.step.to_le_bytes());
    bytes.extend_from_slice(&acc.c_z_digest);
    bytes.extend_from_slice(&(acc.y_compact.len() as u64).to_le_bytes());
    for &y in &acc.y_compact {
        bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    let digest = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&bytes);
    digest.iter().map(|x| F::from_u64(x.as_canonical_u64())).collect()
}

#[test]
fn prover_ignores_malicious_step_x_and_uses_digest_prefix() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();

    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    // Witness: [1, prev_x=0, next_x=1]
    let step_witness = vec![F::ONE, F::ZERO, F::ONE];
    let y_step = vec![F::ONE];

    let binding = StepBindingSpec {
        y_step_offsets: vec![2],
        // No app input binding needed for this test - we're just testing digest prefix
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Don't provide public input - the prover will use H(prev_acc) as the prefix
    // (In the NIVC wrapper, lane metadata is added automatically)
    let input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc,
        step: 0,
        public_input: None,
        y_step: &y_step,
        binding_spec: &binding,
        app_input_binding: AppInputBinding::WitnessBound,
        prev_augmented_x: None,
    };

    let result = prove_ivc_step(input).expect("prover should accept and prepend digest prefix");
    // Verify the accumulator digest equals H(prev_acc)
    let expected_prefix = compute_x_digest(&prev_acc);
    assert_eq!(result.proof.public_inputs.acc_digest(), &expected_prefix[..]);
}

#[test]
fn verifier_rejects_tampered_step_x() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();

    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    // Correct proof with step_x = H(prev_acc)
    let step_witness = vec![F::ONE, F::ZERO, F::ONE];
    let y_step = vec![F::ONE];
    let binding = StepBindingSpec {
        y_step_offsets: vec![2],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    let input_ok = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc,
        step: 0,
        public_input: None, // library fills with H(prev_acc)
        y_step: &y_step,
        binding_spec: &binding,
        app_input_binding: AppInputBinding::WitnessBound,
        prev_augmented_x: None,
    };
    let ok = prove_ivc_step(input_ok).expect("proving should succeed");

    // Tamper with step_x after proving
    let mut forged = ok.proof.clone();
    let correct_x = compute_x_digest(&prev_acc);
    assert_eq!(forged.public_inputs.acc_digest(), &correct_x[..]);
    // Tamper with the accumulator digest (first 4 elements)
    let buf = forged.public_inputs.__test_tamper_buffer();
    buf[0] = F::from_u64(999);
    buf[1] = F::from_u64(888);
    buf[2] = F::from_u64(777);
    buf[3] = F::from_u64(666);

    // Verifier must reject tampered x (should return Err, not Ok(false))
    let result = verify_ivc_step(&step_ccs, &forged, &prev_acc, &binding, &params, None);
    assert!(result.is_err(), "verifier must error when digest prefix is tampered");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Las binding check failed") || err_msg.contains("step_x prefix does not match"), 
            "error should indicate Las binding check failure, got: {}", err_msg);
}
