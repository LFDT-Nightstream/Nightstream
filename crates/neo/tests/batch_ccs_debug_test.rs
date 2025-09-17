//! Minimal test to reproduce and debug the batch CCS construction issue
//!
//! This test isolates the problem where individual IVC steps work perfectly,
//! but combining multiple steps into a batch CCS creates constraint violations.

use neo::{NeoParams, F, ivc::*};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::PrimeCharacteristicRing;

/// Simple increment step: next_x = prev_x + 1
/// Constraint: next_x - prev_x - 1 = 0
/// Variables: [const=1, prev_x, next_x]
fn build_simple_increment_ccs() -> CcsStructure<F> {
    let rows = 1;  // 1 constraint
    let cols = 3;  // 3 variables: [const, prev_x, next_x]
    
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols]; 
    let c_data = vec![F::ZERO; rows * cols];
    
    // Constraint: next_x - prev_x - const = 0
    // Written as: (next_x - prev_x - const) × 1 = 0
    a_data[0 * cols + 0] = -F::ONE;  // -const
    a_data[0 * cols + 1] = -F::ONE;  // -prev_x
    a_data[0 * cols + 2] = F::ONE;   // +next_x
    b_data[0 * cols + 0] = F::ONE;   // × 1 (using const as selector)
    
    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Create witness for increment: prev_x -> next_x
fn build_increment_witness(prev_x: u64) -> Vec<F> {
    vec![
        F::ONE,                           // const = 1
        F::from_u64(prev_x),             // prev_x 
        F::from_u64(prev_x + 1),         // next_x = prev_x + 1
    ]
}

/// Simple extractor: y_step = [next_x] (last element)
struct LastElementExtractor;

impl StepOutputExtractor for LastElementExtractor {
    fn extract_y_step(&self, witness: &[F]) -> Vec<F> {
        vec![witness[witness.len() - 1]] // Return [next_x]
    }
}

#[test]
fn test_individual_steps_work() {
    let step_ccs = build_simple_increment_ccs();

    let witness_0 = build_increment_witness(0);
    assert!(
        check_ccs_rowwise_zero(&step_ccs, &[], &witness_0).is_ok(),
        "Step 0 constraints must be satisfied"
    );

    let witness_1 = build_increment_witness(1);
    assert!(
        check_ccs_rowwise_zero(&step_ccs, &[], &witness_1).is_ok(),
        "Step 1 constraints must be satisfied"
    );
}

#[test]  
fn test_batch_constraints_satisfied() {
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_simple_increment_ccs();

    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: vec![F::ZERO],
        step: 0,
    };

    let binding_spec = StepBindingSpec {
        x_witness_indices: vec![],
        y_step_offsets: vec![2],
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    let mut batch_builder = IvcBatchBuilder::new_with_bindings(
        params.clone(),
        step_ccs.clone(),
        initial_acc,
        EmissionPolicy::Never,
        binding_spec,
    ).unwrap();

    let extractor = LastElementExtractor;

    let w0 = build_increment_witness(0);
    let y0 = extractor.extract_y_step(&w0);
    batch_builder.append_step(&w0, None, &y0).unwrap();

    let w1 = build_increment_witness(1);
    let y1 = extractor.extract_y_step(&w1);
    batch_builder.append_step(&w1, None, &y1).unwrap();

    let batch_data = batch_builder.finalize().unwrap().unwrap();

    // Under current implementation we expect packed mode (all-private)
    assert!(batch_data.public_input.is_empty(), "Batch public_input must be empty in packed mode");
    assert_eq!(batch_data.ccs.m, batch_data.witness.len(), "m must equal |w| in packed mode");
    assert_eq!(batch_data.steps_covered, 2, "must cover 2 steps");

    assert!(
        check_ccs_rowwise_zero(&batch_data.ccs, &[], &batch_data.witness).is_ok(),
        "Batch CCS constraints must be satisfied in packed mode"
    );
}
