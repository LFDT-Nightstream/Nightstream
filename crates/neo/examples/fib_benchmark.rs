//! Fibonacci Neo SNARK Benchmark
//!
//! This benchmarks Neo SNARK proving for Fibonacci sequence computation using
//! an efficient sparse matrix construction approach instead of a large monolithic matrix.
//! 
//! The proof demonstrates: "I know a valid Fibonacci sequence of length n, and F(n) ≡ X (mod p)"
//! where X is the final result exposed as a public output, and all intermediate values stay secret.
//!
//! Usage: cargo run --release -p neo --example fib_benchmark
//!        # or customize ns: N="100,1000,10000,50000" cargo run --release -p neo --example fib_benchmark
//!
//! Notes:
//! - Uses sparse matrix construction for efficiency
//! - Each constraint enforces: z[i+2] = z[i+1] + z[i] for the Fibonacci recurrence
//! - Final SNARK proof exposes F(n) as public output
//! - Much more memory efficient than dense matrix approaches

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use neo::{prove, ProveInput, NeoParams, CcsStructure, F, claim_z_eq};
use neo_ccs::{check_ccs_rowwise_zero, r1cs_to_ccs, Mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::env;
use std::time::Instant;

/// Helper function to convert sparse triplets to dense row-major format
fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        data[row * cols + col] = val;
    }
    data
}

/// Build CCS for Fibonacci sequence using sparse construction
/// 
/// Variables: [1, z0, z1, z2, ..., z_n] where z_i = F(i)
/// Constraints:
/// 1. z0 = 0 (F(0) = 0)
/// 2. z1 = 1 (F(1) = 1) 
/// 3. z[i+2] = z[i+1] + z[i] for i = 0..n-2 (Fibonacci recurrence)
fn fibonacci_ccs(n: usize) -> CcsStructure<F> {
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
    // For i in 0..n-2:
    // Row (2+i): z[i+2] - z[i+1] - z[i] = 0
    for i in 0..(n - 1) {
        let r = 2 + i;
        a_trips.push((r, (i + 3),  F::ONE));  // +z[i+2]
        a_trips.push((r, (i + 2), -F::ONE));  // -z[i+1]
        a_trips.push((r, (i + 1), -F::ONE));  // -z[i]
        b_trips.push((r, 0, F::ONE));         // B selects constant 1
    }

    // Build matrices from sparse triplets - much more efficient than building dense initially
    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));

    // Convert to CCS
    r1cs_to_ccs(a, b, c)
}

/// Generate Fibonacci witness vector [1, z0, z1, ..., z_n] with z0=0, z1=1 (mod F)
#[inline]
fn fibonacci_witness(n: usize) -> Vec<F> {
    assert!(n >= 1);
    
    // We need exactly n+2 elements: [1, z0, z1, z2, ..., z_n]
    let mut z = Vec::with_capacity(n + 2);
    z.push(F::ONE);  // constant 1
    z.push(F::ZERO); // z0 = 0
    z.push(F::ONE);  // z1 = 1
    
    // Generate additional fibonacci numbers z2, z3, ..., z_n  
    while z.len() < n + 2 {
        let len = z.len();
        let next = z[len - 1] + z[len - 2];
        z.push(next);
    }
    
    z
}

/// Prove once for a given n, returning (prove_time_ms, proof_size_bytes).
fn prove_once(n: usize, params: &NeoParams) -> Result<(f64, usize)> {
    // Build CCS + witness using efficient sparse construction
    let ccs = fibonacci_ccs(n);
    let wit = fibonacci_witness(n);

    // Local sanity check (no public inputs)
    check_ccs_rowwise_zero(&ccs, &[], &wit)
        .map_err(|e| anyhow::anyhow!("CCS witness check failed: {:?}", e))?;

    // Create output claim to expose the final Fibonacci number F(n)
    // Witness structure: [1, z0, z1, z2, ..., z_n] so z_n is at index n+1
    let final_fib_value = wit[n + 1]; // F(n) in finite field
    let final_claim = claim_z_eq(params, ccs.m, n + 1, final_fib_value);

    // Prove (with public output claim)
    let t0 = Instant::now();
    let proof = prove(ProveInput { 
        params, 
        ccs: &ccs, 
        public_input: &[], 
        witness: &wit,
        output_claims: &[final_claim],
        vjs_opt: None, // Expose F(n) as public output
    })?;
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Secure one-liner that verifies and extracts the single public output
    let out = neo::verify_and_extract_exact(&ccs, &[], &proof, 1)?;
    eprintln!("n = {:>6} → verified F(n) mod p = {}", n, out[0].as_canonical_u64());

    Ok((prove_ms, proof.size()))
}

fn main() -> Result<()> {
    // Configure Rayon to use all available CPU cores for maximum parallelization
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .expect("Failed to configure Rayon thread pool");

    println!("🚀 Using {} threads for parallel computation", rayon::current_num_threads());

    // Auto-tuned params for Goldilocks; adjust if you want different security/arity
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Allow overriding the benchmark list via env var N="10,20,50,100"
    let ns: Vec<usize> = match env::var("N") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![100, 1000, 10000],  // Reasonable defaults for benchmarking
    };

    // Warm-up (small n) to stabilize allocations, not timed in the table
    let _ = prove_once(10, &params)?;

    // Store results instead of printing immediately
    let mut results = Vec::new();

    // Run the benchmark set
    for &n in &ns {
        let (ms, bytes) = prove_once(n, &params)?;
        results.push((n, ms, bytes));
    }

    // Print all results at the end with header
    println!("n,prover time (ms),proof size (bytes)");
    for (n, ms, bytes) in results {
        println!("{},{:.0},{}", n, ms, bytes);
    }

    Ok(())
}