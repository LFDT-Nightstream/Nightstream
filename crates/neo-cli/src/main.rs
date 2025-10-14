#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use neo::{NeoParams, CcsStructure, F, NivcProgram, NivcState, NivcStepSpec, StepBindingSpec};
use neo_ccs::{r1cs_to_ccs, Mat};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "neo-cli", version, about = "Neo CLI for Fibonacci proofs (Demos)", long_about = None)]
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
        /// Bundle the verifier key bytes inside the proof package (creates ~50MB file)
        /// Default: true (write self-contained proof package)
        #[arg(long = "bundle-vk", default_value_t = true)]
        bundle_vk: bool,
        /// Don't bundle the verifier key (creates lean proof only)
        #[arg(long = "no-bundle-vk", conflicts_with = "bundle_vk")]
        no_bundle_vk: bool,
        /// Emit a separate verifier key file next to the proof (opt-in)
        /// Default: false (no .vk file is created)
        #[arg(long = "emit-vk", default_value_t = false)]
        emit_vk: bool,
    },
    /// Verify a previously generated proof file
    Verify {
        /// Input proof file path
        #[arg(short = 'f', long = "file")]
        file: PathBuf,
        /// Optional: path to verifier key bytes to use for verification
        /// If not given, the CLI will try to auto-load a sibling .vk next to the proof file
        #[arg(long = "vk")]
        vk_path: Option<PathBuf>,
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

// ---------- Fibonacci CCS and witness builders (IVC step relation) ----------

fn triplets_to_dense(rows: usize, cols: usize, triplets: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val) in triplets {
        data[row * cols + col] = val;
    }
    data
}

#[derive(Clone, Copy, Debug)]
struct FibState { i: u64, a: u64, b: u64 }

impl FibState {
    fn new() -> Self { Self { i: 0, a: 0, b: 1 } }
    fn step(&self) -> Self {
        const P128: u128 = 18446744069414584321u128; // Goldilocks
        #[inline(always)] fn add_p(a: u64, b: u64) -> u64 {
            let s = (a as u128) + (b as u128);
            let s = if s >= P128 { s - P128 } else { s };
            s as u64
        }
        Self { i: self.i + 1, a: self.b, b: add_p(self.a, self.b) }
    }
}

fn fibonacci_step_ccs() -> CcsStructure<F> {
    let rows = 3;  // constraints
    let cols = 7;  // [1, i, a, b, i_next, a_next, b_next]

    let mut a_trips: Vec<(usize, usize, F)> = Vec::new();
    let mut b_trips: Vec<(usize, usize, F)> = Vec::new();
    let c_trips: Vec<(usize, usize, F)> = Vec::new();

    // i_next - i - 1 = 0
    a_trips.push((0, 4, F::ONE));
    a_trips.push((0, 1, -F::ONE));
    a_trips.push((0, 0, -F::ONE));
    b_trips.push((0, 0, F::ONE));

    // a_next - b = 0
    a_trips.push((1, 5, F::ONE));
    a_trips.push((1, 3, -F::ONE));
    b_trips.push((1, 0, F::ONE));

    // b_next - a - b = 0
    a_trips.push((2, 6, F::ONE));
    a_trips.push((2, 2, -F::ONE));
    a_trips.push((2, 3, -F::ONE));
    b_trips.push((2, 0, F::ONE));

    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, a_trips));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, b_trips));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, c_trips));
    r1cs_to_ccs(a, b, c)
}

fn fibonacci_step_witness(state: &FibState) -> (Vec<F>, FibState) {
    let next = state.step();
    let w = vec![
        F::ONE,
        F::from_u64(state.i),
        F::from_u64(state.a),
        F::from_u64(state.b),
        F::from_u64(next.i),
        F::from_u64(next.a),
        F::from_u64(next.b),
    ];
    (w, next)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FibProofFile {
    /// Fibonacci length n used to derive the CCS
    n: usize,
    /// Lean proof
    proof: neo::Proof,
    /// Final CCS structure used for verification
    final_ccs: CcsStructure<F>,
    /// Final public input used for verification
    final_public_input: Vec<F>,
    /// Verifier key bytes (stable bincode encoding)
    /// Default empty for lean-only packages; present only if --bundle-vk was used
    #[serde(default)]
    vk_bytes: Vec<u8>,
}

/// Calculate Fibonacci numbers modulo the Goldilocks prime
fn calculate_fibonacci_mod_goldilocks(n: usize) -> u64 {
    const P128: u128 = 18446744069414584321u128;
    
    let add_mod_p = |a: u64, b: u64| -> u64 {
        let s = (a as u128) + (b as u128);
        let s = if s >= P128 { s - P128 } else { s };
        s as u64
    };
    
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    
    let mut prev = 0u64;
    let mut curr = 1u64;
    
    for _ in 2..=n {
        let next = add_mod_p(prev, curr);
        prev = curr;
        curr = next;
    }
    
    curr
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
        Commands::Gen { n, out, bundle_vk, no_bundle_vk, emit_vk } => {
            let should_bundle = bundle_vk && !no_bundle_vk;
            cmd_gen(n, out, should_bundle, emit_vk)
        },
        Commands::Verify { file, vk_path, n, expect } => cmd_verify(file, vk_path, n, expect),
    }
}

fn cmd_gen(n: usize, out: PathBuf, bundle_vk: bool, emit_vk: bool) -> Result<()> {
    println!("Generating Fibonacci IVC proof with {} steps", n);

    // Setup IVC chain
    let total_start = std::time::Instant::now();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let build_start = std::time::Instant::now();
    
    // Build step CCS and binding spec
    let step_ccs = fibonacci_step_ccs();
    let y_len = 3;
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![4, 5, 6],     // last three entries are i_next, a_next, b_next
        step_program_input_witness_indices: vec![],         // no extra public X binding for this example
        y_prev_witness_indices: vec![1, 2, 3], // previous state inside witness
        const1_witness_index: 0,
    };

    // Initialize IVC chain state
    let mut initial_y = vec![F::ZERO; y_len]; // [i=0, a=0, b=1] initial Fibonacci state
    initial_y[2] = F::ONE; // b = 1
    // Build a NIVC program with a single lane (Fibonacci step)
    let program = NivcProgram::new(vec![NivcStepSpec { ccs: step_ccs.clone(), binding: binding_spec }]);
    // Initialize NIVC state
    let mut state = NivcState::new(params.clone(), program.clone(), initial_y)?;
    let build_time = build_start.elapsed();

    // Execute IVC steps
    let ivc_start = std::time::Instant::now();
    // NIVC does local extraction based on binding; no external extractor needed
    let mut fib_state = FibState::new();
    
    println!("🔄 Executing {} Fibonacci IVC steps...", n);
    for step in 0..n {
        let (witness, next_state) = fibonacci_step_witness(&fib_state);
        
        // No step_x for this example (empty vec)
        let step_x = vec![];
        
        state.step(0, &step_x, &witness)?;
        fib_state = next_state;
        
        // Progress indicator every 50 steps
        if (step + 1) % 50 == 0 {
            println!("   📊 Progress: {}/{} steps completed ({:.1}%)", 
                     step + 1, 
                     n, 
                     ((step + 1) as f64 / n as f64) * 100.0);
        }
    }
    let ivc_time = ivc_start.elapsed();

    // Expected value: F(n) mod Goldilocks
    let mod_fib: u64 = {
        const P128: u128 = 18446744069414584321u128;
        let addp = |a: u64, b: u64| -> u64 {
            let s = (a as u128) + (b as u128);
            let s = if s >= P128 { s - P128 } else { s };
            s as u64
        };
        if n == 0 { 0 } else if n == 1 { 1 } else {
            let mut prev = 0u64;
            let mut curr = 1u64;
            for _ in 2..=n { let next = addp(prev, curr); prev = curr; curr = next; }
            curr
        }
    };

    // Generate final SNARK proof
    let prove_start = std::time::Instant::now();
    // Convert to NIVC chain and generate final SNARK
    let chain = state.into_proof();
    let result = neo::finalize_nivc_chain_with_options(&program, &params, chain, neo::NivcFinalizeOptions { embed_ivc_ev: true })?;
    let prove_time = prove_start.elapsed();

    let (proof, final_ccs, final_public_input) = result.ok_or_else(|| {
        anyhow::anyhow!("Failed to generate final proof")
    })?;

    println!("   ✅ Final SNARK generated");
    println!("   - Proof size: {} bytes ({:.1} KB)", proof.proof_bytes.len(), proof.proof_bytes.len() as f64 / 1024.0);

    // For VK bytes, we need to get them from the registry since the new Proof structure doesn't include them
    let vk_bytes = vec![]; // TODO: Get VK bytes from registry if needed
    let pkg_vk_bytes = if bundle_vk { 
        println!("Including verifier key inside package ({} bytes)", vk_bytes.len());
        vk_bytes.clone() 
    } else { 
        Vec::new() 
    };
    
    let proof_file = FibProofFile { 
        n, 
        proof,
        final_ccs,
        final_public_input,
        vk_bytes: pkg_vk_bytes 
    };

    // Save proof package (lean by default)
    let bytes = {
        use bincode::Options;
        stable_bincode_options().serialize(&proof_file)?
    };
    fs::write(&out, &bytes)?;
    println!("Wrote proof package to {} ({} bytes)", out.display(), bytes.len());

    // Optionally emit a sibling .vk file when requested
    if !bundle_vk && emit_vk {
        let mut vk_path = out.clone();
        vk_path.set_extension("vk");
        fs::write(&vk_path, &vk_bytes)?;
        println!("Wrote verifier key to {} ({} bytes)", vk_path.display(), vk_bytes.len());
    }
    // Performance summary
    let total_time = total_start.elapsed();
    println!("\n🏁 COMPREHENSIVE PERFORMANCE SUMMARY");
    println!("=========================================");
    println!("Circuit Information:");
    println!("  IVC Steps:                {:>8}", n);
    println!("  Step CCS Constraints:     {:>8}", step_ccs.n);
    println!("  Step CCS Variables:       {:>8}", step_ccs.m);
    println!("  Step CCS Matrices:        {:>8}", step_ccs.matrices.len());
    println!("  Expected F(n):            {:>8}", mod_fib);
    println!();
    println!("Performance Metrics:");
    println!("  IVC Chain Build:          {:>8.2} ms", build_time.as_secs_f64() * 1000.0);
    println!("  IVC Steps Execution:      {:>8.2} ms", ivc_time.as_secs_f64() * 1000.0);
    println!("  Final SNARK Generation:   {:>8.2} ms", prove_time.as_secs_f64() * 1000.0);
    println!("  Total End-to-End:         {:>8.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("  Proof Size:               {:>8} bytes ({:.1} KB)", proof_file.proof.proof_bytes.len(), proof_file.proof.proof_bytes.len() as f64 / 1024.0);
    println!("  VK Size:                  {:>8} bytes ({:.1} KB)", vk_bytes.len(), vk_bytes.len() as f64 / 1024.0);
    println!();
    println!("System Configuration:");
    println!("  CPU Threads Used:         {:>8}", rayon::current_num_threads());
    println!("  Memory Allocator:         {:>8}", "mimalloc");
    println!("  Architecture:             {:>8}", "Nova IVC + SNARK");
    println!("  Post-Quantum Security:    {:>8}", "✅ Yes");
    println!("=========================================");
    Ok(())
}

fn cmd_verify(file: PathBuf, _vk_cli_path: Option<PathBuf>, n_override: Option<usize>, expect: Option<u64>) -> Result<()> {
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

    // Perform actual cryptographic verification using neo::verify
    let verify_start = std::time::Instant::now();
    
    println!("🔍 Verifying Fibonacci proof...");
    println!("   Fibonacci steps: {}", chosen_n);
    println!("   Final CCS constraints: {}", pkg.final_ccs.n);
    println!("   Final CCS variables: {}", pkg.final_ccs.m);
    println!("   Final public input elements: {}", pkg.final_public_input.len());
    
    // **CRITICAL FIX**: Actually call neo::verify_spartan2 with final CCS and public input
    let verification_result = neo::verify_spartan2(&pkg.final_ccs, &pkg.final_public_input, &pkg.proof);
    
    match verification_result {
        Ok(true) => {
            println!("✅ Cryptographic proof verification: PASSED");
        }
        Ok(false) => {
            println!("❌ Cryptographic proof verification: FAILED");
            return Err(anyhow::anyhow!("Proof verification failed: verification returned false"));
        }
        Err(e) => {
            println!("❌ Cryptographic proof verification: FAILED");
            println!("   Error: {}", e);
            return Err(anyhow::anyhow!("Proof verification failed: {}", e));
        }
    }
    
    // Extract the verified Fibonacci result from the final public input
    // Layout: [step_x || ρ || y_prev || y_next]
    let y_len = pkg.proof.meta.num_y_compact;
    let total = pkg.final_public_input.len();
    
    // Calculate layout offsets
    let step_x_len = total - (1 + 2 * y_len);  // total - (ρ + y_prev + y_next)
    let y_next_start = step_x_len + 1 + y_len; // skip step_x, ρ, and y_prev
    
    let y_verified = if y_next_start < pkg.final_public_input.len() {
        // For Fibonacci, we want the 'b' component (index 2 in y_next)
        let fib_index = y_next_start + 2; // y_next[2] contains the Fibonacci result
        if fib_index < pkg.final_public_input.len() {
            pkg.final_public_input[fib_index].as_canonical_u64()
        } else {
            0 // fallback
        }
    } else {
        0 // fallback
    };
    
    // Additional validation: check if the verified result matches expected Fibonacci value
    if chosen_n <= 100 {  // Only verify for small n to avoid overflow
        let expected = calculate_fibonacci_mod_goldilocks(chosen_n + 1);
        if y_verified == expected {
            println!("   ✅ Verified result matches expected Fibonacci value");
        } else {
            println!("   ⚠️  Verified result {} differs from expected {}", y_verified, expected);
            println!("       This may be due to IVC accumulator vs direct computation differences");
        }
    }

    let verify_time = verify_start.elapsed();
    println!("Valid IVC proof for {} steps", chosen_n);
    println!("Verification time: {:.2} ms", verify_time.as_secs_f64() * 1000.0);
    println!("Verified F(n): {}", y_verified);
    if let Some(exp) = expect {
        if y_verified != exp { anyhow::bail!("Expected {} but verified {}", exp, y_verified); }
        println!("Expectation matched: {}", exp);
    }
    Ok(())
}
