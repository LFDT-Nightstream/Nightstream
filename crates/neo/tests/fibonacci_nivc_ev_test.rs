//! NIVC Fibonacci EV Embed Test (small number of steps)
//!
//! This test mirrors the fib_folding_nivc example but runs only a few steps
//! and finalizes with the EV-in-circuit embedding enabled. It is intended to
//! quickly catch regressions in the NIVC finalize + Spartan flow.

use anyhow::Result;
use neo::{NeoParams, F, NivcProgram, NivcState, NivcStepSpec, NivcFinalizeOptions};
use neo_ccs::{r1cs_to_ccs, Mat, CcsStructure};
use p3_field::PrimeCharacteristicRing;

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets { dense[row * cols + col] = val; }
    dense
}

/// Build CCS for single Fibonacci step: (a, b) -> (b, a+b)
/// Variables: [1, a_prev, b_prev, a_next, b_next]
fn fibonacci_step_ccs() -> CcsStructure<F> {
    let rows = 4;  // Minimum 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 5;  // [1, a_prev, b_prev, a_next, b_next]

    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let c_trips = Vec::new(); // Always zero

    // Constraint 0: a_next - b_prev = 0
    a_trips.push((0, 3, F::ONE));   // +a_next
    a_trips.push((0, 2, -F::ONE));  // -b_prev
    b_trips.push((0, 0, F::ONE));   // × 1

    // Constraint 1: b_next - a_prev - b_prev = 0
    a_trips.push((1, 4, F::ONE));   // +b_next
    a_trips.push((1, 1, -F::ONE));  // -a_prev
    a_trips.push((1, 2, -F::ONE));  // -b_prev
    b_trips.push((1, 0, F::ONE));   // × 1

    // Dummy constraints for rows 2-3 (0 * 1 = 0)
    b_trips.push((2, 0, F::ONE));
    b_trips.push((3, 0, F::ONE));

    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));
    r1cs_to_ccs(a, b, c)
}

/// Build witness for a single Fibonacci step: (a, b) -> (b, a+b)
/// Witness layout: [1, a_prev, b_prev, a_next, b_next]
fn build_fibonacci_step_witness(a: u64, b: u64) -> Vec<F> {
    const P128: u128 = 18446744069414584321u128; // Goldilocks prime
    let add_mod_p = |x: u64, y: u64| -> u64 {
        let s = (x as u128) + (y as u128);
        let s = if s >= P128 { s - P128 } else { s };
        s as u64
    };
    let a_next = b;
    let b_next = add_mod_p(a, b);
    vec![
        F::ONE,
        F::from_u64(a),
        F::from_u64(b),
        F::from_u64(a_next),
        F::from_u64(b_next),
    ]
}

#[test]
fn test_fibonacci_nivc_ev_embed_small() -> Result<()> {
    // Ensure a clean VK registry so circuit key shape changes never hit stale VKs
    neo_spartan_bridge::clear_vk_registry();
    // Determinism for reproducible CI runs
    std::env::set_var("NEO_DETERMINISTIC", "1");
    // Enforce strict public IO parity for EV embedding to fail fast on mismatches
    std::env::set_var("NEO_STRICT_IO_PARITY", "1");
    // Keep full pipeline enabled (Pi-CCS on) to test integrated path
    // Note: Do NOT force RLC identity when testing EV embedding; it breaks commit-evo parity.

    // Params and CCS
    let params = NeoParams::goldilocks_small_circuits();
    let step_ccs = fibonacci_step_ccs();

    // Binding: output y_step is b_next (index 4), const1 at 0. No app inputs.
    let binding = neo::StepBindingSpec {
        y_step_offsets: vec![4],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Program with a single step type (lane 0)
    let program = NivcProgram::new(vec![NivcStepSpec { ccs: step_ccs.clone(), binding }]);

    // Initial accumulator compact output: start from F(1) = 1
    let y0 = vec![F::ONE];
    let mut st = NivcState::new(params.clone(), program.clone(), y0.clone())?;

    // Run a few Fibonacci steps (small to keep CI fast)
    let num_steps = 3u64;
    let mut a = 0u64; // F(0)
    let mut b = 1u64; // F(1)
    for _ in 0..num_steps {
        let wit = build_fibonacci_step_witness(a, b);
        st.step(0, &[], &wit)?;
        // Update state
        let next_a = b;
        const P128: u128 = 18446744069414584321u128;
        let s = (a as u128) + (b as u128);
        let next_b = if s >= P128 { (s - P128) as u64 } else { s as u64 };
        a = next_a; b = next_b;
    }

    // Finalize with EV embedding (Track B default)
    let chain = st.into_proof();
    let (proof, final_ccs, final_public_input) =
        neo::finalize_nivc_chain_with_options(&program, &params, chain, NivcFinalizeOptions { embed_ivc_ev: true })?
        .ok_or_else(|| anyhow::anyhow!("No steps to finalize"))?;

    // Verify lean proof
    let ok = neo::verify_spartan2(&final_ccs, &final_public_input, &proof)?;
    assert!(ok, "Final SNARK verification must succeed for small NIVC EV run");

    // Sanity: public IO is non-empty and decodes
    let ys = neo::decode_public_io_y(&proof.public_io)?;
    assert!(!ys.is_empty(), "public IO should contain at least one y limb");

    Ok(())
}
