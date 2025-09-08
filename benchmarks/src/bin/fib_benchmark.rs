//! Fibonacci Neo SNARK Benchmark
//!
//! This builds a CCS equivalent to the SP1 loop with seeds z0=0, z1=1, and
//! constraints z[i+2] = z[i+1] + z[i] for i=0..n-2. It then proves for several n
//! and prints a CSV: `n,prover time (ms),proof size (bytes)`.
//!
//! Usage: cargo run --release --bin fib_benchmark
//!        # or customize ns: N="100,1000,10000,50000" cargo run --release --bin fib_benchmark

use anyhow::Result;
use neo::{prove, verify, NeoParams, ProveInput, F};
use neo_ccs::{CcsStructure, r1cs_to_ccs, Mat, check_ccs_rowwise_zero};
use p3_field::PrimeCharacteristicRing;
use std::env;
use std::time::Instant;
use benchmarks::util::{pad_ccs_rows_to_pow2, triplets_to_dense, proof_size_bytes};

/// Build sparse R1CS matrices encoding:
///   seeds:  z0 = 0, z1 = 1
///   step:   z[i+2] - z[i+1] - z[i] = 0 for i = 0..n-2
///
/// Variables (columns): [1, z0, z1, ..., z_n]
/// Rows (constraints): 2 seed rows + (n-1) step rows => total rows = n + 1
/// Returns properly padded CCS for evaluation over Boolean hypercube.
fn fibonacci_ccs_equivalent_to_sp1(n: usize) -> CcsStructure<F> {
    assert!(n >= 1, "n must be >= 1");

    let rows = n + 1;        // 2 seed rows + (n-1) recurrence rows
    let cols = n + 2;        // [1, z0, z1, ..., z_n]

    // Triplets (row, col, val) for sparse A, B, C
    let mut a_trips: Vec<(usize, usize, F)> = Vec::with_capacity(3 * (n - 1) + 2);
    let mut b_trips: Vec<(usize, usize, F)> = Vec::with_capacity(rows);
    let     c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero

    // --- Seed constraints ---
    // Row 0: z0 = 0  => A: +1*z0
    a_trips.push((0, 1, F::ONE));                 // col 1 = z0
    b_trips.push((0, 0, F::ONE));                 // select constant 1

    // Row 1: z1 - 1 = 0  => A: +1*z1 + (-1)*1
    a_trips.push((1, 2, F::ONE));                 // col 2 = z1
    a_trips.push((1, 0, -F::ONE));                // col 0 = constant * (-1)
    b_trips.push((1, 0, F::ONE));                 // select constant 1

    // --- Recurrence rows ---
    for i in 0..(n - 1) {
        let r = 2 + i;
        a_trips.push((r, (i + 3),  F::ONE));  // +z[i+2]
        a_trips.push((r, (i + 2), -F::ONE));  // -z[i+1]
        a_trips.push((r, (i + 1), -F::ONE));  // -z[i]
        b_trips.push((r, 0, F::ONE));         // B selects constant 1
    }

    // Build matrices from sparse triplets
    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));

    // Convert to CCS and pad to power-of-2
    let ccs = r1cs_to_ccs(a, b, c);
    pad_ccs_rows_to_pow2(ccs)
}

/// Produce the witness vector [1, z0, z1, ..., z_n] with z0=0, z1=1 (mod F)
#[inline]
fn fibonacci_witness(n: usize) -> Vec<F> {
    assert!(n >= 1);
    let mut z: Vec<F> = Vec::with_capacity(n + 2);
    z.push(F::ONE);  // constant 1
    z.push(F::ZERO); // z0
    z.push(F::ONE);  // z1
    for k in 2..=n {
        let next = z[k] + z[k - 1]; // note: z index is shifted by +1 due to constant at 0
        z.push(next);
    }
    z
}

/// Prove once for a given n, returning (prove_time_ms, proof_size_bytes).
fn prove_once(n: usize, params: &NeoParams) -> Result<(f64, usize)> {
    // Build CCS + witness
    let ccs = fibonacci_ccs_equivalent_to_sp1(n);
    let wit = fibonacci_witness(n);

    // Local sanity check (no public inputs)
    check_ccs_rowwise_zero(&ccs, &[], &wit)
        .map_err(|e| anyhow::anyhow!("CCS witness check failed: {:?}", e))?;

    // Prove
    let t0 = Instant::now();
    let proof = prove(ProveInput { params, ccs: &ccs, public_input: &[], witness: &wit })?;
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Verify (not included in timing)
    let _ok = verify(&ccs, &[], &proof)?;

    Ok((prove_ms, proof_size_bytes(&proof)))
}

fn main() -> Result<()> {
    println!("Neo Fibonacci Benchmark");
    println!("=======================");
    
    // Auto-tuned params for Goldilocks
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Allow overriding the benchmark list via env var N="10,20,50,100"
    let ns: Vec<usize> = match env::var("N") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![100, 1000],
    };

    // Warm-up (small n) to stabilize allocations, not timed in the table
    let _ = prove_once(1, &params)?;

    // Store results instead of printing immediately
    let mut results = Vec::new();

    // Run the benchmark set
    for &n in &ns {
        println!("Running benchmark for n = {}...", n);
        let (ms, bytes) = prove_once(n, &params)?;
        results.push((n, ms, bytes));
    }

    // Print all results at the end with header
    println!("\nResults:");
    println!("n,prover time (ms),proof size (bytes)");
    for (n, ms, bytes) in results {
        println!("{},{:.0},{}", n, ms, bytes);
    }

    Ok(())
}
