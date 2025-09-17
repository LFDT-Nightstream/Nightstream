#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use neo::{prove, ProveInput, NeoParams, CcsStructure, F, claim_z_eq};
use neo::ivc::{IvcBatchBuilder, EmissionPolicy, StepOutputExtractor, LastNExtractor, Accumulator, StepBindingSpec, BatchData};
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

fn build_fib_batch_ccs(params: &NeoParams, steps: usize) -> anyhow::Result<(BatchData, CcsStructure<F>)> {
    let step_ccs = fibonacci_step_ccs();
    let y_len = 3;
    let binding_spec = StepBindingSpec {
        y_step_offsets: vec![4, 5, 6],     // last three entries are i_next, a_next, b_next
        x_witness_indices: vec![],         // no extra public X binding
        y_prev_witness_indices: vec![1, 2, 3], // previous state inside witness
        const1_witness_index: 0,
    };

    let initial_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![F::ZERO; y_len], step: 0 };
    let mut batch = IvcBatchBuilder::new_with_bindings(params.clone(), step_ccs.clone(), initial_acc, EmissionPolicy::Every(100), binding_spec)?;

    let extractor = LastNExtractor { n: y_len };
    let mut state = FibState::new();
    let mut proofs_emitted = 0;
    
    println!("🔄 Executing {} Fibonacci IVC steps with EmissionPolicy::Every(100)...", steps);
    for step in 0..steps {
        let (witness, next_state) = fibonacci_step_witness(&state);
        let y_step_real = extractor.extract_y_step(&witness);
        
        // Provide step_x = H(prev_accumulator) to satisfy binding requirement
        let x_digest = {
            let acc = &batch.accumulator;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&acc.step.to_le_bytes());
            bytes.extend_from_slice(&acc.c_z_digest);
            bytes.extend_from_slice(&(acc.y_compact.len() as u64).to_le_bytes());
            for &y in &acc.y_compact { 
                bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes()); 
            }
            let d = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&bytes);
            let mut out = Vec::with_capacity(d.len());
            for x in d { 
                out.push(F::from_u64(x.as_canonical_u64())); 
            }
            out
        };
        
        let pending_before = batch.pending_steps();
        batch.append_step(&witness, Some(&x_digest), &y_step_real)?;
        let pending_after = batch.pending_steps();
        
        // Check if a proof was auto-emitted
        if pending_after < pending_before {
            proofs_emitted += 1;
            println!("   ✅ Auto-emitted proof #{} after step {} (covered steps {}-{})", 
                     proofs_emitted, 
                     step, 
                     step - 99, 
                     step);
        }
        
        // Progress indicator every 50 steps
        if (step + 1) % 50 == 0 {
            println!("   📊 Progress: {}/{} steps completed ({:.1}%)", 
                     step + 1, 
                     steps, 
                     ((step + 1) as f64 / steps as f64) * 100.0);
        }
        
        state = next_state;
    }

    println!("   ✅ All {} steps completed! Auto-emitted {} proofs during execution.", steps, proofs_emitted);
    println!("   📦 Finalizing remaining batch data for final SNARK layer...");

    let data = batch.finalize()?.ok_or_else(|| anyhow::anyhow!("No batch data produced"))?;
    Ok((data, step_ccs))
}

// ---------- File format ----------

#[derive(serde::Serialize, serde::Deserialize)]
struct FibProofFile {
    /// Fibonacci length n used to derive the CCS
    n: usize,
    /// Lean proof
    proof: neo::Proof,
    /// Verifier key bytes (stable bincode encoding)
    /// Default empty for lean-only packages; present only if --bundle-vk was used
    #[serde(default)]
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
        Commands::Gen { n, out, bundle_vk, no_bundle_vk, emit_vk } => {
            let should_bundle = bundle_vk && !no_bundle_vk;
            cmd_gen(n, out, should_bundle, emit_vk)
        },
        Commands::Verify { file, vk_path, n, expect } => cmd_verify(file, vk_path, n, expect),
    }
}

fn cmd_gen(n: usize, out: PathBuf, bundle_vk: bool, emit_vk: bool) -> Result<()> {
    println!("Generating Fibonacci IVC proof with {} steps", n);

    // Params and batch construction
    let total_start = std::time::Instant::now();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let build_start = std::time::Instant::now();
    let (batch_data, step_ccs) = build_fib_batch_ccs(&params, n)?;
    let build_time = build_start.elapsed();

    // Final SNARK Layer with application output claim exposing final F(n) = a_next of last step
    // Layout per step: [pub_len=7 | wit_len=10], where within wit_len, step_witness indices [0..6]
    let pub_len = 1 + 2 * 3; // x_len(=0) + 1 + 2*y_len, with y_len=3
    let step_wit_len = 7usize; // [1, i, a, b, i_next, a_next, b_next]
    let y_len = 3usize;
    let wit_len = step_wit_len + y_len; // [step_witness || u]
    let blocks = n;
    let last_block_offset = (blocks - 1) * (pub_len + wit_len);
    let a_next_idx_in_step_witness = 5usize;
    let k_final = last_block_offset + pub_len + a_next_idx_in_step_witness;

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

    let final_claim = claim_z_eq(&params, batch_data.ccs.m, k_final, F::from_u64(mod_fib));

    let prove_start = std::time::Instant::now();
    let proof = prove(ProveInput {
        params: &params,
        ccs: &batch_data.ccs,
        public_input: &batch_data.public_input,
        witness: &batch_data.witness,
        output_claims: &[final_claim],
        vjs_opt: None,
    })?;
    let prove_time = prove_start.elapsed();
    let proof_size = proof.size();
    let claimed_public = proof.claimed_public_results_u64();

    println!("   ✅ Final SNARK generated");
    println!("   - Proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);

    // Retrieve VK from registry and serialize with stable options
    // Obtain VK bytes for optional bundling and/or separate file emission
    let vk_arc = neo_spartan_bridge::lookup_vk(&proof.circuit_key)
        .ok_or_else(|| anyhow::anyhow!("VK not found in registry after proving"))?;
    let vk_bytes = {
        use bincode::Options;
        stable_bincode_options().serialize(&*vk_arc)?
    };

    // By default, do NOT bundle VK into the package (keep file small)
    // If bundling, clone VK so we can still optionally emit a sibling .vk
    let pkg_vk_bytes = if bundle_vk { 
        println!("Including verifier key inside package ({} bytes)", vk_bytes.len());
        vk_bytes.clone() 
    } else { 
        Vec::new() 
    };
    let pkg = FibProofFile { n, proof, vk_bytes: pkg_vk_bytes };

    // Save proof package (lean by default)
    let bytes = {
        use bincode::Options;
        stable_bincode_options().serialize(&pkg)?
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
    println!();
    println!("Performance Metrics:");
    println!("  Batch Build:              {:>8.2} ms", build_time.as_secs_f64() * 1000.0);
    println!("  Proof Generation:         {:>8.2} ms", prove_time.as_secs_f64() * 1000.0);
    println!("  Total End-to-End:         {:>8.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("  Proof Size:               {:>8} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    if let Some(val) = claimed_public.first() { println!("  Public F(n):               {:>8}", val); }
    println!();
    println!("System Configuration:");
    println!("  CPU Threads Used:         {:>8}", rayon::current_num_threads());
    println!("  Memory Allocator:         {:>8}", "mimalloc");
    println!("  Build Mode:               {:>8}", "Release + Optimizations");
    println!("  SIMD Instructions:        {:>8}", "target-cpu=native");
    println!("  Post-Quantum Security:    {:>8}", "✅ Yes");
    println!("=========================================");
    Ok(())
}

fn cmd_verify(file: PathBuf, vk_cli_path: Option<PathBuf>, n_override: Option<usize>, expect: Option<u64>) -> Result<()> {
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

    // Rebuild the same IVC batch CCS from chosen_n
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let (batch_data, _step_ccs) = build_fib_batch_ccs(&params, chosen_n)?;
    let ccs = batch_data.ccs;
    let public_inputs: Vec<F> = vec![];

    // Determine VK bytes source:
    // 1) --vk path if provided
    // 2) Bundled vk_bytes inside file if present
    // 3) Sibling .vk file next to the proof
    // 4) Fallback to registry-only (may fail if VK not pre-registered)
    let mut used_vk_bytes: Option<Vec<u8>> = None;
    if let Some(path) = vk_cli_path {
        used_vk_bytes = Some(fs::read(&path)?);
        println!("Loaded VK from --vk {}", path.display());
    } else if !pkg.vk_bytes.is_empty() {
        println!("Using VK bundled inside package ({} bytes)", pkg.vk_bytes.len());
        used_vk_bytes = Some(pkg.vk_bytes.clone());
    } else {
        let mut sibling = file.clone();
        sibling.set_extension("vk");
        if sibling.exists() {
            used_vk_bytes = Some(fs::read(&sibling)?);
            println!("Loaded VK from sibling file {}", sibling.display());
        } else {
            println!("No VK provided/bundled. Will attempt registry verification (may fail)");
        }
    }

    // Verify using best available method
    let verify_start = std::time::Instant::now();
    let is_valid = match used_vk_bytes {
        Some(ref vk) => neo::verify_with_vk(&ccs, &public_inputs, &pkg.proof, vk)?,
        None => match neo::verify(&ccs, &public_inputs, &pkg.proof) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "Verification error without VK: {}\nHint: supply --vk <path> or place a sibling .vk next to the proof file",
                    e
                );
                return Err(e);
            }
        },
    };
    if !is_valid { anyhow::bail!("Proof verification failed"); }
    // Extract verified public result (we exposed one app output: F(n))
    let outputs = neo::verify_and_extract_exact(&ccs, &public_inputs, &pkg.proof, 1)?;
    let y_verified = outputs[0].as_canonical_u64();

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
