//! Stage 6: Finalize NIVC micro (1 step) — with EV

use anyhow::Result;
use neo::{NeoParams, F, NivcProgram, NivcState, NivcStepSpec, NivcFinalizeOptions};
use neo_ccs::{r1cs_to_ccs, Mat};
use p3_field::PrimeCharacteristicRing;

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets { dense[row * cols + col] = val; }
    dense
}

fn fibonacci_step_ccs() -> neo_ccs::CcsStructure<F> {
    let rows = 2; let cols = 5; // [1, a_prev, b_prev, a_next, b_next]
    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let c_trips = Vec::new();
    a_trips.push((0, 3, F::ONE)); a_trips.push((0, 2, -F::ONE)); b_trips.push((0, 0, F::ONE));
    a_trips.push((1, 4, F::ONE)); a_trips.push((1, 1, -F::ONE)); a_trips.push((1, 2, -F::ONE)); b_trips.push((1, 0, F::ONE));
    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));
    r1cs_to_ccs(a, b, c)
}

#[test]
fn finalize_one_step_with_ev() -> Result<()> {
    let params = NeoParams::goldilocks_small_circuits();
    let step_ccs = fibonacci_step_ccs();
    let binding = neo::ivc::StepBindingSpec { y_step_offsets: vec![4], step_program_input_witness_indices: vec![], y_prev_witness_indices: vec![], const1_witness_index: 0 };
    let program = NivcProgram::new(vec![NivcStepSpec { ccs: step_ccs.clone(), binding }]);
    let y0 = vec![F::ONE];
    let mut st = NivcState::new(params.clone(), program.clone(), y0.clone())?;
    let wit = vec![F::ONE, F::from_u64(0), F::from_u64(1), F::from_u64(1), F::from_u64(1)];
    st.step(0, &[], &wit)?;
    let chain = st.into_proof();
    let (proof, final_ccs, final_public_input) =
        neo::finalize_nivc_chain_with_options(&program, &params, chain, NivcFinalizeOptions { embed_ivc_ev: true })?
        .ok_or_else(|| anyhow::anyhow!("No steps to finalize"))?;
    let ok = neo::verify(&final_ccs, &final_public_input, &proof)?;
    assert!(ok);
    Ok(())
}
