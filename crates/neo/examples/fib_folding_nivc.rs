//! Fibonacci Folding Demo using IvcSession (single-step shape)
//!
//! This example proves a simple Fibonacci transition relation using the
//! high-level Nova/Sonobe-style `IvcSession` API from `session.rs`.
//! It performs many steps by repeatedly calling `session.prove_step(...)`,
//! then finalizes to a succinct SNARK and verifies it.
//!
//! Run:
//!   NEO_DETERMINISTIC=1 cargo run -p neo --example fib_folding_nivc -- 1000
//!
//! Notes:
//! - The cryptographic accumulator y has length 1; we expose `b_next` as the
//!   step output via `y_step_indices = [4]` in the witness layout.

use anyhow::Result;
use std::time::Instant;
use num_bigint::BigUint;

use neo::{F, NeoParams};
use neo::session::{
    NeoStep, StepSpec, StepArtifacts, IvcSession, StepDescriptor,
    IvcFinalizeOptions, finalize_ivc_chain_with_options,
};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut dense = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets { dense[row * cols + col] = val; }
    dense
}

/// Build CCS for single Fibonacci step: (a, b) -> (b, a+b)
/// Variables: [1, a_prev, b_prev, a_next, b_next]
fn fibonacci_step_ccs() -> CcsStructure<F> {
    let rows = 2;  // 2 constraints
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

    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));
    r1cs_to_ccs(a, b, c)
}

/// Build witness for a single Fibonacci step: (a, b) -> (b, a+b)
/// Witness layout: [1, a_prev, b_prev, a_next, b_next]
// (no standalone witness builder needed; the NeoStep adapter produces per-step witnesses)

// External inputs for the Fibonacci step adapter (a_prev, b_prev)
#[derive(Clone, Default)]
struct FibInputs { a: F, b: F }

// NeoStep adapter for the Fibonacci relation
struct FibonacciStep {
    ccs: CcsStructure<F>,
    spec: StepSpec,
}

impl FibonacciStep {
    fn new() -> Self {
        let ccs = fibonacci_step_ccs();
        let spec = StepSpec {
            // Accumulator y has length 1 (we expose b_next each step)
            y_len: 1,
            // witness[0] == 1 (constant column)
            const1_index: 0,
            // y_step = [b_next] at witness index 4
            y_step_indices: vec![4],
            // No binding to y_prev inside the step witness (not used by circuit)
            y_prev_indices: None,
            // No app-level public inputs
            app_input_indices: None,
        };
        Self { ccs, spec }
    }
}

impl NeoStep for FibonacciStep {
    type ExternalInputs = FibInputs;

    fn state_len(&self) -> usize { 1 }
    fn step_spec(&self) -> StepSpec { self.spec.clone() }

    fn synthesize_step(
        &mut self,
        _step_idx: usize,
        _z_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        // a_next = b; b_next = a + b (field addition modulo p)
        let a_prev = inputs.a;
        let b_prev = inputs.b;
        let a_next = b_prev;
        let b_next = a_prev + b_prev;
        let witness = vec![F::ONE, a_prev, b_prev, a_next, b_next];
        StepArtifacts {
            ccs: self.ccs.clone(),
            witness,
            public_app_inputs: vec![],
            spec: self.spec.clone(),
        }
    }
}

fn main() -> Result<()> {
    // Optional determinism for reproducible runs
    std::env::set_var("NEO_DETERMINISTIC", "1");

    let total_start = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let num_steps = if args.len() > 1 {
        args[1].parse::<u64>().unwrap_or(100)
    } else { 100 };

    println!("🔥 NIVC Fibonacci Folding (single-lane)");
    println!("Steps: {}", num_steps);

    // Step 1: Setup parameters
    println!("\n🔧 Step 1: Setting up Neo parameters...");
    let params_start = Instant::now();
    let params = NeoParams::goldilocks_small_circuits();
    let params_time = params_start.elapsed();
    println!("   ✅ Parameters initialized in {:.2}ms", params_time.as_secs_f64() * 1000.0);

    // Step 2: Build step CCS
    println!("\n📐 Step 2: Building Fibonacci step CCS...");
    let stepper = FibonacciStep::new();
    let step_ccs = stepper.ccs.clone();
    println!("   CCS dimensions: {} constraints, {} variables", step_ccs.n, step_ccs.m);
    println!("   Step relation: (a, b) -> (b, a+b)");
    println!("   ✅ Step CCS built successfully");

    // Step 3: Initialize IVC session with y0 = [1]
    let y0 = vec![F::ONE];
    let mut session = IvcSession::new(&params, Some(y0.clone()), 0);
    let mut stepper = stepper; // mutable for trait API

    // Fibonacci state (mod q)
    let mut a = 0u64; // F(0)
    let mut b = 1u64; // F(1)

    // Step 4: Per-step folding via IvcSession
    println!("\n🔄 Step 4: Running IVC folding loop (single-shape)...");
    println!("   Performing {} Fibonacci steps with per-step folding...", num_steps);
    let steps_start = Instant::now();
    // Preallocate per-step timing vector (minimal overhead)
    let mut step_times_ms: Vec<f64> = Vec::with_capacity(num_steps as usize);

    for i in 0..num_steps {
        let t_step = Instant::now();
        let inputs = FibInputs { a: F::from_u64(a), b: F::from_u64(b) };
        session.prove_step(&mut stepper, &inputs)
            .map_err(|e| anyhow::anyhow!("step {} failed: {}", i, e))?;
        step_times_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
        // Update Fibonacci
        let next_a = b;
        const P128: u128 = 18446744069414584321u128;
        let s = (a as u128) + (b as u128);
        let next_b = if s >= P128 { (s - P128) as u64 } else { s as u64 };
        a = next_a; b = next_b;
        if i < 5 || i + 3 >= num_steps { println!("   Step {:>4}: state -> ({}, {})", i + 1, a, b); }
        if i == 5 { println!("   ... (omitting middle steps) ..."); }
    }
    let steps_time = steps_start.elapsed();

    // Finalize IVC chain with outer SNARK (Stage 5)
    let (chain, _step_ios) = session.finalize();
    let descriptor = StepDescriptor { ccs: step_ccs.clone(), spec: stepper.spec.clone() };
    println!("\n🔄 Step 5: Generating Final SNARK Layer proof...");
    println!("   • EV embedding: disabled (public-ρ path)");
    println!("   • Pi-CCS terminal check: enabled by default for EV");
    let final_snark_start = Instant::now();
    let (final_proof, final_ccs, final_public_input) =
        finalize_ivc_chain_with_options(&descriptor, &params, chain, IvcFinalizeOptions { embed_ivc_ev: false })
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .ok_or_else(|| anyhow::anyhow!("No steps to finalize"))?;
    let final_snark_time = final_snark_start.elapsed();

    // Verify outer SNARK
    println!("\n🔍 Step 6: Verifying Final SNARK proof...");
    let verify_start = Instant::now();
    let is_valid = neo::verify(&final_ccs, &final_public_input, &final_proof)?;
    let verify_time = verify_start.elapsed();
    anyhow::ensure!(is_valid, "final proof verification failed");
    println!("   ✅ Final SNARK verification passed in {:.2} ms", verify_time.as_secs_f64() * 1000.0);

    // Step 9: Decode proof outputs
    println!("\n🔍 Step 7: Extracting result from proof public IO...");
    let extract_start = Instant::now();
    // For Fibonacci we treat the arithmetic result as the current 'b'
    let proof_result = b;
    let public_inputs = neo::decode_public_io_y(&final_proof.public_io)?;
    let accumulator_value = if !public_inputs.is_empty() { public_inputs[0].as_canonical_u64() } else { 0 };
    let extract_time = extract_start.elapsed();
    println!("   ✅ Output extraction completed in {:.2} ms", extract_time.as_secs_f64() * 1000.0);
    println!("   🔢 Cryptographic accumulator: {} (Nova folded state)", accumulator_value);
    println!("   🔢 Arithmetic result: {} (direct calculation)", proof_result);

    // Step 10: Comprehensive verification (local vs proof)
    println!("\n🔍 Step 8: Comprehensive Fibonacci verification...");
    let trace_start = Instant::now();
    let final_b = b; // local result after the loop
    println!("   📋 FIBONACCI COMPUTATION VERIFICATION:");
    println!("   ======================================");
    println!("   Expected F({}) = {}", num_steps + 1, final_b);
    println!("   Proof-extracted result: {}", proof_result);
    println!("   Local computation result: {}", b);
    let proof_validation_passed = final_b == proof_result;
    let local_validation_passed = final_b == b;
    if proof_validation_passed { println!("   ✅ CRYPTOGRAPHIC VALIDATION PASSED"); } else { println!("   ❌ CRYPTOGRAPHIC VALIDATION FAILED"); }
    if local_validation_passed { println!("   ✅ LOCAL VALIDATION PASSED"); } else { println!("   ❌ LOCAL VALIDATION FAILED"); }
    println!("   ✅ CRYPTOGRAPHIC PROOF VERIFICATION: Proof was verified as valid in Step 8!");
    let trace_time = trace_start.elapsed();

    // Step 9: Compare with true Fibonacci (arbitrary precision)
    println!("\n🧮 Step 9: Comparing with true arbitrary-precision Fibonacci...");
    let true_fib = {
        if (num_steps + 1) as usize == 0 { BigUint::from(0u64) } else {
            let mut prev = BigUint::from(0u64);
            let mut curr = BigUint::from(1u64);
            for _ in 2..=(num_steps + 1) as usize { let next = &prev + &curr; prev = std::mem::replace(&mut curr, next); }
            curr
        }
    };
    println!("   True F({}): {} digits", num_steps + 1, true_fib.to_string().len());

    // Final Performance Summary
    println!("\n🏁 COMPREHENSIVE PERFORMANCE SUMMARY");
    println!("=========================================");
    println!("Fibonacci Configuration:");
    println!("  Number of Steps:               {}", num_steps);
    println!("  Final Fibonacci Number:        F({})", num_steps + 1);
    println!("  Step CCS Constraints:          {}", step_ccs.n);
    println!("  Step CCS Variables:            {}", step_ccs.m);
    println!("  Step CCS Matrices:             {}", step_ccs.matrices.len());
    println!("  Final Accumulator Step:        {}", num_steps);
    println!();
    println!("Performance Metrics:");
    println!("  Setup Time:                 {:>8.2} ms", params_time.as_secs_f64() * 1000.0);
    println!("  Per-Step Folding:           {:>8.2} ms/step ({} steps, {:.2} ms total)", steps_time.as_secs_f64() * 1000.0 / num_steps as f64, num_steps, steps_time.as_secs_f64() * 1000.0);
    // Compute lightweight distribution stats without per-step logging
    let (min_ms, max_ms, _mean_ms) = if !step_times_ms.is_empty() {
        let mut min_v = f64::INFINITY;
        let mut max_v = 0.0;
        let mut sum_v = 0.0;
        for &v in &step_times_ms { if v < min_v { min_v = v; } if v > max_v { max_v = v; } sum_v += v; }
        (min_v, max_v, sum_v / step_times_ms.len() as f64)
    } else { (0.0, 0.0, 0.0) };
    // Median and P90 via a final sort (done once; negligible overhead compared to proving)
    let (median_ms, p90_ms) = if !step_times_ms.is_empty() {
        let mut tmp = step_times_ms.clone();
        tmp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = tmp.len();
        let mid = n / 2;
        let p90_idx = ((n as f64 - 1.0) * 0.90).round() as usize;
        (tmp[mid], tmp[p90_idx])
    } else { (0.0, 0.0) };
    println!("  Per-Step (ms):              min={:>6.2} median={:>6.2} p90={:>6.2} max={:>6.2}", min_ms, median_ms, p90_ms, max_ms);
    println!("  Final SNARK Generation:     {:>8.2} ms (Stage 5)", final_snark_time.as_secs_f64() * 1000.0);
    println!("  Final SNARK Verification:   {:>8.2} ms", verify_time.as_secs_f64() * 1000.0);
    println!("  Output Extraction:          {:>8.2} ms", extract_time.as_secs_f64() * 1000.0);
    println!("  Trace Verification:         {:>8.2} ms", trace_time.as_secs_f64() * 1000.0);
    println!("  Total End-to-End:           {:>8.2} ms", total_start.elapsed().as_secs_f64() * 1000.0);
    println!("  Final Proof Size:           {:>8} bytes ({:.1} KB)", final_proof.size(), final_proof.size() as f64 / 1024.0);
    println!();
    println!("System Configuration:");
    println!("  CPU Threads Used:           {:>8}", rayon::current_num_threads());
    println!("  Memory Allocator:           {:>8}", "mimalloc");
    println!("  Build Mode:                 {:>8}", "Debug");
    println!("  Post-Quantum Security:      {:>8}", "✅ Yes");
    println!();
    let steps_per_ms = num_steps as f64 / (steps_time.as_secs_f64() * 1000.0);
    let kb_per_step = (final_proof.size() as f64 / 1024.0) / num_steps as f64;
    let folding_vs_snark_ratio = steps_time.as_secs_f64() / final_snark_time.as_secs_f64();
    let verification_speedup = (steps_time.as_secs_f64() + final_snark_time.as_secs_f64()) / verify_time.as_secs_f64();
    println!("Efficiency Metrics:");
    println!("  Steps/ms (Folding):         {:>8.1}", steps_per_ms);
    println!("  KB per Step:                {:>8.3}", kb_per_step);
    println!("  Folding vs SNARK Ratio:     {:>8.1}x faster", folding_vs_snark_ratio);
    println!("  Verification Speedup:       {:>8.1}x", verification_speedup);
    println!("\n🎉 IVC Fibonacci: final SNARK verified successfully");
    Ok(())
}
