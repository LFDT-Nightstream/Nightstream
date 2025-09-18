//! Red-team tests for IVC step_x binding: x must equal H(prev_accumulator)

use neo::{F, NeoParams};
use neo::ivc::{Accumulator, IvcStepInput, StepBindingSpec, prove_ivc_step, verify_ivc_step};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn build_increment_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, next_x]
    // Constraint: next_x - prev_x - 1 = 0  => (next_x - prev_x - const) * 1 = 0
    let rows = 1;
    let cols = 3;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // Row 0: (next_x - prev_x - 1) * 1 = 0
    a[0 * cols + 2] = F::ONE;     // + next_x
    a[0 * cols + 1] = -F::ONE;    // - prev_x
    a[0 * cols + 0] = -F::ONE;    // - const (represents -1)
    b[0 * cols + 0] = F::ONE;     // * 1

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
    // Extend the circuit with 4 unconstrained app slots so we can bind them
    let step_ccs = build_increment_step_ccs_with_app_slots(4);

    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    // Witness: [1, prev_x=0, next_x=1, app0..app3]
    let step_witness = vec![
        F::ONE, F::ZERO, F::ONE,
        F::from_u64(123456), F::from_u64(789), F::from_u64(101112), F::from_u64(131415)
    ];
    let y_step = vec![F::ONE];

    let binding = StepBindingSpec {
        y_step_offsets: vec![2],
        // Bind only the *app* tail of step_x (digest prefix remains unbound by design).
        x_witness_indices: vec![3, 4, 5, 6],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Attempt to provide a fake digest as public input; this is treated as app inputs now
    let malicious_app_inputs = vec![F::from_u64(123456), F::from_u64(789), F::from_u64(101112), F::from_u64(131415)];
    let input = IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &step_witness,
        prev_accumulator: &prev_acc,
        step: 0,
        public_input: Some(&malicious_app_inputs),
        y_step: &y_step,
        binding_spec: &binding,
    };

    let result = prove_ivc_step(input).expect("prover should accept and prepend digest prefix");
    // Verify the prefix equals H(prev_acc)
    let expected_prefix = compute_x_digest(&prev_acc);
    assert!(result.proof.step_public_input.len() >= expected_prefix.len());
    assert_eq!(&result.proof.step_public_input[..expected_prefix.len()], &expected_prefix[..]);
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
        x_witness_indices: vec![],
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
    };
    let ok = prove_ivc_step(input_ok).expect("proving should succeed");

    // Tamper with step_x after proving
    let mut forged = ok.proof.clone();
    let correct_x = compute_x_digest(&prev_acc);
    assert_eq!(forged.step_public_input, correct_x);
    // Tamper with wrong size (should be 4 elements like correct_x)
    forged.step_public_input = vec![F::from_u64(999), F::from_u64(888), F::from_u64(777), F::from_u64(666)];

    // Verifier must reject tampered x
    let is_valid = verify_ivc_step(&step_ccs, &forged, &prev_acc, &binding).expect("verify must not error");
    assert!(!is_valid, "verifier should reject proof with tampered step_x");
}

// Helper used above: variables = [1, prev_x, next_x, app_0..app_{k-1}],
// with the single constraint next_x - prev_x - 1 = 0 and all app_* unconstrained.
fn build_increment_step_ccs_with_app_slots(k: usize) -> CcsStructure<F> {
    let rows = 1;
    let cols = 3 + k;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    // next_x - prev_x - 1 = 0
    a[0 * cols + 2] = F::ONE;
    a[0 * cols + 1] = -F::ONE;
    a[0 * cols + 0] = -F::ONE;
    b[0 * cols + 0] = F::ONE; // × 1
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}


