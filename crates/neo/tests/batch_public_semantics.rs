// crates/neo/tests/batch_public_semantics.rs
//
// Tests to validate that the IvcBatchBuilder correctly separates public inputs from witness,
// and that the public section truly affects constraint satisfaction.
//
// These tests would have caught the "public columns stuffed into witness" bug.

use neo::{F, NeoParams};
use neo::ivc::{
    IvcBatchBuilder, EmissionPolicy, StepBindingSpec, Accumulator,
};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::PrimeCharacteristicRing;

/// Simple step: x -> x + step_number
/// Witness: [const=1, prev_x, step_number, next_x]
fn build_increment_step_ccs() -> CcsStructure<F> {
    let rows = 1;
    let cols = 4;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // Constraint: next_x - prev_x - step_number = 0  (× const)
    // row 0: A: +next_x -prev_x -step_number,  B: × const(=1)
    a[0 * cols + 3] = F::ONE;  // +next_x
    a[0 * cols + 1] = -F::ONE; // -prev_x
    a[0 * cols + 2] = -F::ONE; // -step_number
    b[0 * cols + 0] = F::ONE;  // × const

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);

    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness: [1, prev_x, step_number, next_x]
fn build_increment_witness(prev_x: u64, step_number: u64) -> Vec<F> {
    let next_x = prev_x + step_number;
    vec![
        F::ONE,
        F::from_u64(prev_x),
        F::from_u64(step_number),
        F::from_u64(next_x),
    ]
}

/// 1) PUBLIC/WITNESS SEPARATION VERIFICATION
/// This verifies that the batch builder correctly separates public inputs from witness,
/// as required by the CCS constraint checking and prove/verify pipeline.
#[test]
fn batch_builder_properly_separates_public_witness() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();

    // Accumulator state has y_len = 1 (just x)
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],                // commitment folding disabled in this test
        y_compact: vec![F::from_u64(100)],
        step: 0,
    };

    // Binding for increment circuit:
    // next_x is witness index 3; prev_x is witness index 1; const-1 is witness index 0.
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let mut batch = IvcBatchBuilder::new_with_bindings(
        params,
        step_ccs,
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    ).expect("IvcBatchBuilder");

    // Build a small batch of 2 steps: 100->101->102
    let steps = 2usize;
    let mut prev = 100u64;
    for _ in 0..steps {
        let step_number = 1u64;
        let w = build_increment_witness(prev, step_number);
        let y_step_real = vec![F::from_u64(prev + step_number)]; // next_x
        batch.append_step(&w, None, &y_step_real).expect("append_step");
        prev += step_number;
    }

    let data = batch.finalize().expect("expected non-empty batch");

    let x_len = 0usize;
    let y_len = 1usize;
    let pub_len_per_step = x_len + 1 + 2 * y_len; // [step_x||ρ||y_prev||y_next] = 3
    let expected_public_len = steps * pub_len_per_step;
    
    // ✅ This test verifies the CORRECTED behavior: public inputs properly separated
    assert_eq!(
        data.public_input.len(),
        expected_public_len,
        "BatchData.public_input must contain all per-step public data; \
         got {}, expected {} (steps={}, pub_per_step={})",
        data.public_input.len(),
        expected_public_len,
        steps,
        pub_len_per_step
    );

    let wit_len_per_step = 4 + y_len; // [1, prev_x, step_num, next_x] + u = 5 
    let expected_witness_len = steps * wit_len_per_step;

    assert_eq!(
        data.witness.len(),
        expected_witness_len,
        "BatchData.witness must contain only witness data; \
         got {}, expected {} (steps={}, wit_per_step={})",
        data.witness.len(),
        expected_witness_len, 
        steps,
        wit_len_per_step
    );
}

/// 2) RED‑TEAM: tamper ρ in the public inputs and the CCS must reject.
/// We flip the ρ within the public inputs and expect `check_ccs_rowwise_zero` to fail.
/// This proves the public inputs are actually used by the constraint system.
#[test]
fn batch_public_tamper_rho_fails() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();

    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::from_u64(7)],
        step: 0,
    };

    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![3], // next_x
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![1], // prev_x
        const1_witness_index: 0,
    };

    let mut batch = IvcBatchBuilder::new_with_bindings(
        params,
        step_ccs,
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    ).expect("IvcBatchBuilder");

    // Two steps: 7->8->9
    let mut prev = 7u64;
    for step_number in [1u64, 1u64] {
        let w = build_increment_witness(prev, step_number);
        let y_step_real = vec![F::from_u64(prev + step_number)]; // next_x
        batch.append_step(&w, None, &y_step_real).expect("append_step");
        prev += step_number;
    }

    let data = batch.finalize().expect("expected non-empty batch");

    // Confirm public inputs are properly populated (not empty)
    assert!(
        !data.public_input.is_empty(),
        "public_input must be populated for this test to be meaningful"
    );

    // Public input layout: [step1_public || step2_public] 
    // where each step_public = [step_x || ρ || y_prev || y_next]
    let _steps = 2;
    let x_len = 0;
    let y_len = 1;
    let pub_len_per_step = x_len + 1 + 2 * y_len; // [x || rho || y_prev || y_next] = 3
    let rho_step0 = 0 * pub_len_per_step + x_len;  // = 0 (first ρ)
    let _rho_step1 = 1 * pub_len_per_step + x_len;  // = 3 (second ρ)
    
    let mut tampered_public = data.public_input.clone();
    
    // Flip ρ for the first step: tamper the actual public input
    tampered_public[rho_step0] = tampered_public[rho_step0] + F::ONE;

    // With properly separated public/witness, tampering ρ must break constraints.
    let err = check_ccs_rowwise_zero(&data.ccs, &tampered_public, &data.witness)
        .expect_err("Tampering ρ in public inputs should make CCS constraints fail");
    eprintln!("Expected failure after ρ tamper in public inputs: {:?}", err);
}
