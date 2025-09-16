#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use neo::{prove, ProveInput, NeoParams, CcsStructure, F, claim_z_eq};
use neo_ccs::{r1cs_to_ccs, Mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "neo-cli", version, about = "Neo CLI for Fibonacci proofs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a Fibonacci proof and write it to a file
    Gen {
        /// Fibonacci index n (proves F(n) modulo Goldilocks)
        #[arg(short = 'n', long = "n")]
        n: usize,
        /// Output file path
        #[arg(short = 'o', long = "out", default_value = "fib_proof.bin")]
        out: PathBuf,
    },
    /// Verify a previously generated proof file
    Verify {
        /// Input proof file path
        #[arg(short = 'f', long = "file")]
        file: PathBuf,
        /// Optional: assert the Fibonacci index `n` used to build the circuit
        /// If provided and mismatched, verification will fail fast
        #[arg(short = 'n', long = "n")]
        n: Option<usize>,
        /// Optional: expected value of F(n) modulo Goldilocks prime (as u64)
        /// If provided, the verified output must equal this value
        #[arg(short = 'e', long = "expect")]
        expect: Option<u64>,
    },
}

// ---------- Fibonacci CCS and witness builders (adapted from examples/fib.rs) ----------

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        data[row * cols + col] = val;
    }
    data
}

fn fibonacci_ccs(n: usize) -> CcsStructure<F> {
    assert!(n >= 1, "n must be >= 1");

    let rows = n + 1;        // 2 seed rows + (n-1) recurrence rows
    let cols = n + 2;        // [1, z0, z1, ..., z_n]

    let mut a_trips: Vec<(usize, usize, F)> = Vec::with_capacity(3 * (n - 1) + 2);
    let mut b_trips: Vec<(usize, usize, F)> = Vec::with_capacity(rows);
    let     c_trips: Vec<(usize, usize, F)> = Vec::new(); // always zero

    // Seed constraints
    a_trips.push((0, 1, F::ONE));                 // z0 = 0
    b_trips.push((0, 0, F::ONE));                 // select constant 1

    a_trips.push((1, 2, F::ONE));                 // z1 - 1 = 0
    a_trips.push((1, 0, -F::ONE));                // -1*1
    b_trips.push((1, 0, F::ONE));

    // Recurrence rows: z[i+2] - z[i+1] - z[i] = 0
    for i in 0..(n - 1) {
        let r = 2 + i;
        a_trips.push((r, (i + 3),  F::ONE));  // +z[i+2]
        a_trips.push((r, (i + 2), -F::ONE));  // -z[i+1]
        a_trips.push((r, (i + 1), -F::ONE));  // -z[i]
        b_trips.push((r, 0, F::ONE));         // select constant 1
    }

    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));

    r1cs_to_ccs(a, b, c)
}

fn generate_fibonacci_witness(n: usize) -> Vec<F> {
    assert!(n >= 1);
    let mut z = Vec::with_capacity(n + 2);
    z.push(F::ONE);  // constant 1
    z.push(F::ZERO); // z0 = 0
    z.push(F::ONE);  // z1 = 1
    while z.len() < n + 2 {
        let len = z.len();
        let next = z[len - 1] + z[len - 2];
        z.push(next);
    }
    z
}

// ---------- File format ----------

#[derive(serde::Serialize, serde::Deserialize)]
struct FibProofFile {
    /// Fibonacci length n used to derive the CCS
    n: usize,
    /// Lean proof
    proof: neo::Proof,
    /// Verifier key bytes (stable bincode encoding)
    vk_bytes: Vec<u8>,
}

fn stable_bincode_options() -> impl bincode::Options + Copy {
    use bincode::{DefaultOptions, Options};
    DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
}

fn main() -> Result<()> {
    // Use all CPUs (helpful during proving)
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global();

    let cli = Cli::parse();
    match cli.command {
        Commands::Gen { n, out } => cmd_gen(n, out),
        Commands::Verify { file, n, expect } => cmd_verify(file, n, expect),
    }
}

fn cmd_gen(n: usize, out: PathBuf) -> Result<()> {
    println!("Generating Fibonacci proof for n = {}", n);

    // Build circuit and witness
    let ccs = fibonacci_ccs(n);
    let z = generate_fibonacci_witness(n);
    let public_inputs: Vec<F> = vec![];

    // Params (same as example)
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);

    // Expose F(n) as public result
    let final_fib = z[n + 1];
    let final_claim = claim_z_eq(&params, ccs.m, n + 1, final_fib);

    let proof = prove(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_inputs,
        witness: &z,
        output_claims: &[final_claim],
    })?;

    println!("Proof generated: {} bytes", proof.size());

    // Retrieve VK from registry and serialize with stable options
    let vk_arc = neo_spartan_bridge::lookup_vk(&proof.circuit_key)
        .ok_or_else(|| anyhow::anyhow!("VK not found in registry after proving"))?;
    let vk_bytes = {
        use bincode::Options;
        stable_bincode_options().serialize(&*vk_arc)?
    };

    let pkg = FibProofFile { n, proof, vk_bytes };

    // Save using bincode (compact)
    let bytes = {
        use bincode::Options;
        stable_bincode_options().serialize(&pkg)?
    };
    fs::write(&out, bytes)?;
    println!("Wrote proof package to {}", out.display());
    Ok(())
}

fn cmd_verify(file: PathBuf, n_override: Option<usize>, expect: Option<u64>) -> Result<()> {
    println!("Verifying proof from {}", file.display());

    // Load file
    let data = fs::read(&file)?;
    let pkg: FibProofFile = {
        use bincode::Options;
        stable_bincode_options().deserialize(&data)?
    };

    // Choose circuit length: prefer explicit --n if provided
    let chosen_n = n_override.unwrap_or(pkg.n);
    if let Some(n_arg) = n_override {
        if n_arg != pkg.n {
            println!(
                "Warning: file declares n = {}, but --n given as {}. Using --n for verification.",
                pkg.n, n_arg
            );
        }
    }

    // Re-derive CCS from chosen n
    let ccs = fibonacci_ccs(chosen_n);
    let public_inputs: Vec<F> = vec![];

    // Verify using provided VK bytes (cross-process safe)
    let is_valid = neo::verify_with_vk(&ccs, &public_inputs, &pkg.proof, &pkg.vk_bytes)?;
    if !is_valid {
        anyhow::bail!("Proof verification failed");
    }

    // Extract verified public result from cryptographic public_io and optionally check expectation
    let verified_outputs = neo::verify_and_extract_exact(&ccs, &public_inputs, &pkg.proof, 1)?;
    let y_verified = verified_outputs[0].as_canonical_u64();

    if let Some(exp) = expect {
        if y_verified != exp {
            anyhow::bail!(
                "Verified output mismatch: F({}) ≡ {} (mod p), expected {}",
                chosen_n, y_verified, exp
            );
        }
        println!(
            "Valid proof for F({}) with expected value {} (mod Goldilocks)",
            chosen_n, y_verified
        );
    } else {
        println!(
            "Valid proof for F({}) ≡ {} (mod Goldilocks)",
            chosen_n, y_verified
        );
    }
    Ok(())
}
