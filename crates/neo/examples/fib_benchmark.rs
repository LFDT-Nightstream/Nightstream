//! Fibonacci Neo SNARK Demo (SP1-equivalent CCS + benchmark)
//!
//! This builds a CCS equivalent to the SP1 loop with seeds z0=0, z1=1, and
//! constraints z[i+2] = z[i+1] + z[i] for i=0..n-2. It then proves for several n
//! and prints a CSV: `n,prover time (ms),proof size (bytes)`.
//!
//! The proof demonstrates: "I know a valid Fibonacci sequence of length n, and F(n) ≡ X (mod p)"
//! where X is the final result exposed as a public output, and all intermediate values stay secret.
//!
//! Usage: cargo run --release -p neo --example fib_benchmark
//!        # or customize ns: N="100,1000,10000,50000" cargo run --release -p neo --example fib_benchmark
//!
//! Notes:
//! - The R1CS is constructed sparsely: only the few non-zeros per row are stored.
//! - We verify after proving, but we *only* time the prove() call for the CSV.
//! - Each proof includes an OutputClaim that exposes F(n) mod p as public output.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use neo::{prove, CcsStructure, NeoParams, ProveInput, F, claim_z_eq};
use neo_ccs::{r1cs_to_ccs, Mat, check_ccs_rowwise_zero};
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

/// Pad all CCS matrices with zero-rows so that ccs.n is power-of-two.
fn pad_ccs_rows_to_pow2(ccs: CcsStructure<F>) -> CcsStructure<F> {
    let n = ccs.n;
    let n_pad = n.next_power_of_two();
    if n_pad == n { return ccs; }

    let m = ccs.m;
    let mut padded = Vec::with_capacity(ccs.matrices.len());
    for mat in &ccs.matrices {
        let mut out = Mat::zero(n_pad, m, F::ZERO);
        // copy existing rows
        for r in 0..n {
            for c in 0..m {
                out[(r, c)] = mat[(r, c)];
            }
        }
        padded.push(out);
    }

    // Rebuild CCS with identical polynomial f, just with padded matrices
    CcsStructure::new(padded, ccs.f.clone())
        .expect("valid CCS after row padding")
}

/// Build sparse R1CS matrices encoding:
///   seeds:  z0 = 0, z1 = 1
///   step:   z[i+2] - z[i+1] - z[i] = 0 for i = 0..n-2
///
/// Variables (columns): [1, z0, z1, ..., z_n]
/// Rows (constraints): 2 seed rows + (n-1) step rows => total rows = n + 1
///
/// Encoding as R1CS (Az) ∘ (Bz) = Cz where each row has:
///   - B selects the constant wire (so Bz = 1)
///   - C is 0
///   - A row is the linear form we want to equal 0
///
/// IMPORTANT: we construct the matrices *sparsely*.
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
    // For i in 0..n-2:
    // Row (2+i): z[i+2] - z[i+1] - z[i] = 0
    // Columns: z_k sits at col (k+1). So:
    //   z[i]   -> col (i+1)
    //   z[i+1] -> col (i+2)
    //   z[i+2] -> col (i+3)
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

    // Convert to CCS and pad to power-of-2 for Π_CCS compatibility
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
        Err(_) => vec![10000],  // Test non-power-of-two values! 100, 1000, 10000, 50000
    };

    // Warm-up (small n) to stabilize allocations, not timed in the table
    let _ = prove_once(1, &params)?;

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
