//! End-to-end test to verify proving works with m_in > 0 (public inputs)
//!
//! This asserts that the pipeline succeeds when CCS.m matches |x|+|w| and
//! public inputs are handled via m_in = |x|.

use neo::{NeoParams, F};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// CCS with m = 4 and two constraints:
/// rows:
///   (x0 - w0) * 1 = 0
///   (x1 - w1) * 1 = 0
/// variables: [x0, x1, w0, w1]
fn create_public_equals_witness_ccs() -> CcsStructure<F> {
    let rows = 2;
    let cols = 4;

    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols];

    // Row 0: x0 - w0
    a_data[0 * cols + 0] = F::ONE;   // +x0
    a_data[0 * cols + 2] = -F::ONE;  // -w0
    b_data[0 * cols + 0] = F::ONE;   // ×1 via x0 column

    // Row 1: x1 - w1
    a_data[1 * cols + 1] = F::ONE;   // +x1
    a_data[1 * cols + 3] = -F::ONE;  // -w1
    b_data[1 * cols + 1] = F::ONE;   // ×1 via x1 column

    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);

    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]  
fn test_prove_with_public_inputs_after_fix() {
    let ccs = create_public_equals_witness_ccs();
    
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    
    // Public inputs (x0,x1) and witness (w0,w1) that satisfy constraints
    let public_input = vec![F::from_u64(7), F::from_u64(11)];
    let witness      = vec![F::from_u64(7), F::from_u64(11)];

    let result = neo::prove(neo::ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
        output_claims: &[],
    });

    assert!(result.is_ok(), "Proving must succeed when CCS.m matches |x|+|w| and m_in=|x|");
}
