#![allow(non_snake_case)]

use std::time::Instant;
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_fold::pi_ccs::FoldingMode;
use neo_ccs::{Mat, r1cs_to_ccs};
use neo_ajtai::{setup as ajtai_setup, set_global_pp, AjtaiSModule};
use neo_params::NeoParams;
use neo_math::{F, D};
use rand::SeedableRng;
use p3_field::PrimeCharacteristicRing;

// Constants
const N_STEPS: usize = 100;

struct FibonacciStep {
    n_vars: usize,
    n_constraints: usize,
}

impl FibonacciStep {
    fn new() -> Self {
        // z structure: [1, F_i, F_{i+1}, F_{i+2}]
        // Constraints: (F_i + F_{i+1}) * 1 = F_{i+2}
        // n=1, m=4
        Self {
            n_vars: 4,
            n_constraints: 1,
        }
    }
}

impl NeoStep for FibonacciStep {
    type ExternalInputs = ();

    fn state_len(&self) -> usize {
        2 // [F_i, F_{i+1}]
    }

    fn step_spec(&self) -> StepSpec {
        StepSpec {
            y_len: 2,
            const1_index: 0,
            y_step_indices: vec![2, 3], // F_{i+1}, F_{i+2} become next state
            app_input_indices: None,
            m_in: 3, // Public inputs: 1, F_i, F_{i+1}
        }
    }

    fn synthesize_step(
        &mut self,
        step_idx: usize,
        y_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        // Initial state for step 0
        let (f0, f1) = if step_idx == 0 {
            (F::ZERO, F::ONE)
        } else {
            (y_prev[0], y_prev[1])
        };

        let f2 = f0 + f1;

        // Construct z: [1, f0, f1, f2]
        let z = vec![F::ONE, f0, f1, f2];

        // Construct R1CS
        // A: 1*f0 + 1*f1
        // B: 1*1
        // C: 1*f2
        let mut A = Mat::zero(self.n_constraints, self.n_vars, F::ZERO);
        let mut B = Mat::zero(self.n_constraints, self.n_vars, F::ZERO);
        let mut C = Mat::zero(self.n_constraints, self.n_vars, F::ZERO);

        // Row 0
        // A: z[1] + z[2] => f0 + f1
        A[(0, 1)] = F::ONE;
        A[(0, 2)] = F::ONE;

        // B: z[0] => 1
        B[(0, 0)] = F::ONE;

        // C: z[3] => f2
        C[(0, 3)] = F::ONE;

        let ccs = r1cs_to_ccs(A, B, C);

        StepArtifacts {
            ccs,
            witness: z,
            public_app_inputs: vec![],
            spec: self.step_spec(),
        }
    }
}

#[test]
fn test_perf_fibonacci_10k_optimized() {
    // Setup
    let mut stepper = FibonacciStep::new();
    
    // Params
    // Ensure we have enough rows/cols. n=1, m=4.
    let params = NeoParams::goldilocks_auto_r1cs_ccs(stepper.n_constraints.max(stepper.n_vars))
        .expect("params");
    
    // Setup Ajtai
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, stepper.n_vars).expect("Ajtai setup");
    let _ = set_global_pp(pp);
    let l = AjtaiSModule::from_global_for_dims(D, stepper.n_vars).expect("AjtaiSModule");

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);

    println!("Starting {} steps of Fibonacci...", N_STEPS);
    let start = Instant::now();

    for _ in 0..N_STEPS {
        session.prove_step(&mut stepper, &()).expect("prove_step failed");
    }

    let folding_time = start.elapsed();
    println!("Folding {} steps took {:?}", N_STEPS, folding_time);

    // Finalize
    println!("Finalizing...");
    let start_fin = Instant::now();
    
    // Use a dummy step to get the CCS structure for finalize
    let dummy = stepper.synthesize_step(0, &[F::ZERO, F::ONE], &());
    let run = session.finalize(&dummy.ccs).expect("finalize failed");
    
    let finalize_time = start_fin.elapsed();
    println!("Finalize took {:?}", finalize_time);
    println!("Total time: {:?}", folding_time + finalize_time);
    
    // Verify
    let public_mcss = session.mcss_public();
    let ok = session.verify(&dummy.ccs, &public_mcss, &run).expect("verify failed");
    assert!(ok, "Verification failed");
    println!("Verification passed!");
}

