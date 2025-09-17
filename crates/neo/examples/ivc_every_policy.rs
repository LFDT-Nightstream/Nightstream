//! Simple step/verify API over Neo IVC
//!
//! This example shows a minimal API similar to Nova-style step recursion:
//!   step(state, io, witness) -> state
//!   verify(state, io) -> bool
//!
//! We implement a trivial incrementer: state x -> x+1 each step.
//!
//! Usage: cargo run -p neo --example ivc_every_policy

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use neo::{F, NeoParams};
use neo::ivc_chain as ivc;
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;
use anyhow::Result;
use std::time::Instant;

/// Helper function to convert triplets to dense matrix data
fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        dense[row * cols + col] = val;
    }
    dense
}

/// Simple step: add a public delta
/// State: [x] -> [x + delta]
fn build_increment_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_x, delta, next_x]
    // Constraint: next_x - prev_x - delta = 0
    let rows = 1;
    let cols = 4;

    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero

    // Constraint: next_x - prev_x - delta = 0
    // Written as: (next_x - prev_x - delta) × const = 0
    a_trips.push((0, 3, F::ONE));   // +next_x
    a_trips.push((0, 1, -F::ONE));  // -prev_x
    a_trips.push((0, 2, -F::ONE));  // -delta
    b_trips.push((0, 0, F::ONE));   // × const 1

    // Build matrices
    let a_data = triplets_to_dense(rows, cols, a_trips.clone());
    let b_data = triplets_to_dense(rows, cols, b_trips.clone());
    let c_data = triplets_to_dense(rows, cols, c_trips.clone());

    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);

    let ccs = r1cs_to_ccs(a_mat, b_mat, c_mat);

    // Quick validation on a couple of witnesses
    // witness = [1, prev_x, delta, next_x]
    let test_witness = vec![F::ONE, F::from_u64(5), F::from_u64(7), F::from_u64(12)];
    let _ = neo_ccs::check_ccs_rowwise_zero(&ccs, &[], &test_witness);
    let actual_witness = vec![F::ONE, F::ZERO, F::ONE, F::ONE];
    let _ = neo_ccs::check_ccs_rowwise_zero(&ccs, &[], &actual_witness);

    ccs
}

/// Generate step witness: [const=1, prev_x, delta, next_x]
fn build_increment_witness(prev_x: u64, delta: u64) -> Vec<F> {
    let next_x = prev_x + delta;
    vec![
        F::ONE,                    // const
        F::from_u64(prev_x),       // prev_x
        F::from_u64(delta),        // delta (also bound as step public input)
        F::from_u64(next_x),       // next_x
    ]
}

fn main() -> Result<()> {
    // Configure Rayon (optional)
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .ok();

    println!("🔄 Neo IVC simple API example");
    println!("=================================");

    let total_start = Instant::now();

    // Setup
    let params_start = Instant::now();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let step_ccs = build_increment_step_ccs();
    let params_time = params_start.elapsed();

    // Binding spec for witness layout: [1, prev_x, delta, next_x]
    let binding_spec = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![3],                // next_x at index 3
        x_witness_indices: vec![2],             // bind step public input X[0] to witness[2] (delta)
        y_prev_witness_indices: vec![1],
        const1_witness_index: 0,
    };

    // Initial state x = 0
    let state0 = ivc::State::new(
        params.clone(),
        step_ccs.clone(),
        vec![F::ZERO],
        binding_spec,
    )?;

    // Run steps: x -> x+1
    let num_steps = 7u64;
    println!("Running {num_steps} increment steps...");
    let steps_start = Instant::now();
    let mut state = state0;
    let mut x = 0u64;
    for step_i in 0..num_steps {
        // choose a public delta per step (vary 1..=3 for demo)
        let delta = 1u64 + (step_i % 3) as u64; // 1,2,3,1,2,3,...
        let wit = build_increment_witness(x, delta);
        // supply delta as per-step public input X
        let io = [F::from_u64(delta)];
        state = ivc::step(state, &io, &wit);
        x += delta;
    }
    let steps_time = steps_start.elapsed();

    // Finalize and prove (generate the cryptographic proof)
    let prove_start = Instant::now();
    let proof_result = ivc::finalize_and_prove(state)?;
    let prove_time = prove_start.elapsed();

    match proof_result {
        Some((proof, batch_ccs, batch_public_input)) => {
            println!("\n✅ Proof generation completed successfully");
            println!("Proof size: {} bytes", proof.size());
            
            // Now verify the proof using the correct batch CCS
            let verify_start = Instant::now();
            let is_valid = neo::verify(&batch_ccs, &batch_public_input, &proof)?;
            let verify_time = verify_start.elapsed();
            
            println!("Proof verification: {}", if is_valid { "✅ VALID" } else { "❌ INVALID" });
            
            println!(
                "Batch CCS: {} constraints, {} variables, {} matrices",
                batch_ccs.n, batch_ccs.m, batch_ccs.matrices.len()
            );
            println!(
                "Step CCS: {} constraints, {} variables, {} matrices", 
                step_ccs.n, step_ccs.m, step_ccs.matrices.len()
            );
            println!(
                "Timings: setup={:.2}ms, steps={:.2}ms, prove={:.2}ms, verify={:.2}ms, total={:.2}ms",
                params_time.as_secs_f64() * 1000.0,
                steps_time.as_secs_f64() * 1000.0,
                prove_time.as_secs_f64() * 1000.0,
                verify_time.as_secs_f64() * 1000.0,
                total_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        None => {
            println!("\nNo steps to prove (empty batch)");
            println!(
                "Timings: setup={:.2}ms, steps={:.2}ms, total={:.2}ms",
                params_time.as_secs_f64() * 1000.0,
                steps_time.as_secs_f64() * 1000.0,
                total_start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    Ok(())
}
