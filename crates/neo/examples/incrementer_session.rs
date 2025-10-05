//! Incrementer using the high-level IvcSession API (Nova/Sonobe-style)
//!
//! Proves x -> x + delta over multiple steps. The step relation is:
//!   witness = [const=1, prev_x, delta, next_x]
//!   constraint: next_x - prev_x - delta = 0
//! y_step = [next_x] is exposed via `y_step_indices = [3]`.
//!
//! Run: cargo run -p neo --example incrementer_session --release

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use neo::{F, NeoParams};
use neo::session::{
    NeoStep, StepSpec, StepArtifacts, IvcSession, StepDescriptor,
    verify_chain_with_descriptor, IvcFinalizeOptions, finalize_ivc_chain_with_options,
};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use std::time::Instant;

/// Build the step CCS for: next_x - prev_x - delta = 0
fn build_increment_step_ccs() -> CcsStructure<F> {
    // Columns (witness): [const=1, prev_x, delta, next_x]
    // Single constraint row
    let rows = 1usize;
    let cols = 4usize;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // A: next_x - prev_x - delta
    a[0 * cols + 3] = F::ONE;     // + next_x
    a[0 * cols + 1] = -F::ONE;    // - prev_x
    a[0 * cols + 2] = -F::ONE;    // - delta
    // B: multiply by const 1
    b[0 * cols + 0] = F::ONE;     // × const

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    let ccs = r1cs_to_ccs(a_mat, b_mat, c_mat);

    // quick sanity
    let test = vec![F::ONE, F::from_u64(5), F::from_u64(7), F::from_u64(12)];
    let _ = neo_ccs::check_ccs_rowwise_zero(&ccs, &[], &test);
    ccs
}

/// Build a witness from previous state and external delta
fn build_witness(prev_x: F, delta: F) -> Vec<F> {
    let next_x = prev_x + delta;
    vec![F::ONE, prev_x, delta, next_x]
}

/// External inputs (app-level) for the step
#[derive(Clone, Default)]
struct ExtInputs { delta: F }

/// Adapter implementing NeoStep
struct IncrementerStep {
    ccs: CcsStructure<F>,
    spec: StepSpec,
}

impl IncrementerStep {
    fn new() -> Self {
        let ccs = build_increment_step_ccs();
        let spec = StepSpec {
            y_len: 1,
            const1_index: 0,           // witness[0] == 1
            y_step_indices: vec![3],    // next_x at index 3
            // Bind that the circuit reads the previous state from witness[1]
            y_prev_indices: Some(vec![1]),
            // Bind app input (delta) to witness[2]
            app_input_indices: Some(vec![2]),
        };
        Self { ccs, spec }
    }
}

impl NeoStep for IncrementerStep {
    type ExternalInputs = ExtInputs;

    fn state_len(&self) -> usize { 1 }
    fn step_spec(&self) -> StepSpec { self.spec.clone() }

    fn synthesize_step(
        &mut self,
        _step_idx: usize,
        z_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        let prev_x = z_prev[0];
        let delta = inputs.delta;
        let witness = build_witness(prev_x, delta);
        StepArtifacts {
            ccs: self.ccs.clone(),
            witness,
            // Public app input tail in step_x
            public_app_inputs: vec![delta],
            spec: self.spec.clone(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rayon optional
    rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get()).build_global().ok();

    let total_start = Instant::now();
    let params = NeoParams::goldilocks_small_circuits();

    // Initial state x=0
    let mut session = IvcSession::new(&params, Some(vec![F::ZERO]), 0);
    let mut stepper = IncrementerStep::new();

    // Run N steps with deltas 1,2,3 repeating
    let n: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut x_local: u64 = 0;
    for i in 0..n {
        let delta_u = 1 + (i % 3) as u64; // 1,2,3 pattern
        let inputs = ExtInputs { delta: F::from_u64(delta_u) };
        let _proof = session.prove_step(&mut stepper, &inputs)?;
        x_local += delta_u;
    }

    // Finalize and verify chain
    let chain = session.finalize();
    let descriptor = StepDescriptor { ccs: stepper.ccs.clone(), spec: stepper.spec.clone() };
    let ok = verify_chain_with_descriptor(&descriptor, &chain, &[F::ZERO], &params)?;

    println!("✅ Chain verify: {}", ok);
    println!("Final state (acc): {:?}", chain.final_accumulator.y_compact);
    println!("Local result: {}", x_local);
    println!("Steps: {}", chain.chain_length);
    println!("Total time: {:.2}ms", total_start.elapsed().as_secs_f64() * 1000.0);

    // Stage 5: Final succinct proof from IVC chain (single-step shape)
    if let Some((final_proof, final_ccs, final_public_input)) = finalize_ivc_chain_with_options(
        &descriptor, &params, chain, IvcFinalizeOptions { embed_ivc_ev: false }
    )? {
        println!("\n🔒 Verifying final SNARK...");
        let valid = neo::verify(&final_ccs, &final_public_input, &final_proof)?;
        println!("Final SNARK valid: {}", valid);
    } else {
        println!("\n(no steps, nothing to finalize)");
    }

    Ok(())
}
