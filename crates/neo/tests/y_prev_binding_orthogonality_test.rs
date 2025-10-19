//! Test that y_prev binding is correctly disabled in the Session API
//!
//! **ARCHITECTURAL NOTE**: y_prev binding is incompatible with ρ-folding because
//! the accumulator's `y_compact` is a cryptographic commitment, not raw application state.
//!
//! These tests verify that:
//! - The Session API never enables y_prev binding (always empty vec)
//! - Application circuits work correctly without binding to the accumulator
//! - State continuity is the prover's responsibility (self-contained witnesses)
//!
//! For verified state continuity, applications should use `app_input_indices` to
//! pass previous outputs as public inputs, NOT bind to the accumulator.

use neo::*;
use neo::session::{NeoStep, StepSpec, StepArtifacts, FoldingSession, StepDescriptor, verify_chain_with_descriptor};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

/// Build a simple incrementer CCS: next_x = prev_x + delta
/// Witness layout: [const=1, prev_x, delta, next_x]
fn build_incrementer_ccs() -> CcsStructure<F> {
    let rows = 4; // minimum for sumcheck security
    let cols = 4;
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];
    
    // Row 0: next_x - prev_x - delta = 0
    a[0 * cols + 3] = F::ONE;     // + next_x
    a[0 * cols + 1] = -F::ONE;    // - prev_x
    a[0 * cols + 2] = -F::ONE;    // - delta
    b[0 * cols + 0] = F::ONE;     // × const
    
    // Rows 1-3: dummy constraints (0 × 1 = 0)
    for row in 1..4 {
        b[row * cols + 0] = F::ONE;
    }
    
    r1cs_to_ccs(
        Mat::from_row_major(rows, cols, a),
        Mat::from_row_major(rows, cols, b),
        Mat::from_row_major(rows, cols, c),
    )
}

fn build_witness(prev_x: F, delta: F) -> Vec<F> {
    vec![F::ONE, prev_x, delta, prev_x + delta]
}

#[derive(Clone, Default)]
struct ExtInputs { delta: F }

struct IncrementerStep {
    ccs: CcsStructure<F>,
    spec: StepSpec,
}

impl IncrementerStep {
    fn new() -> Self {
        let ccs = build_incrementer_ccs();
        let spec = StepSpec {
            y_len: 1,
            const1_index: 0,
            y_step_indices: vec![3],    // next_x at witness[3]
            app_input_indices: Some(vec![2]), // delta at witness[2]
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
        StepArtifacts {
            ccs: self.ccs.clone(),
            witness: build_witness(z_prev[0], inputs.delta),
            public_app_inputs: vec![inputs.delta],
            spec: self.spec.clone(),
        }
    }
}

/// Test that y_prev binding field has been removed from StepBindingSpec
#[test]
fn test_y_prev_binding_removed() {
    let spec = StepSpec {
        y_len: 1,
        const1_index: 0,
        y_step_indices: vec![3],
        app_input_indices: Some(vec![2]),
    };
    
    // Just verify that binding_spec works without y_prev binding
    let witness_bound_spec = spec.binding_spec(AppInputBinding::WitnessBound);
    assert_eq!(witness_bound_spec.y_step_offsets, vec![3usize]);
    assert_eq!(witness_bound_spec.step_program_input_witness_indices, vec![2usize]);
    
    let transcript_only_spec = spec.binding_spec(AppInputBinding::TranscriptOnly);
    assert_eq!(transcript_only_spec.y_step_offsets, vec![3usize]);
    assert_eq!(transcript_only_spec.step_program_input_witness_indices, Vec::<usize>::new());
}

/// Test that circuits work correctly without y_prev binding (WitnessBound mode)
#[test]
fn test_self_contained_witnesses_witness_bound() {
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        Some(&params),
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::WitnessBound,
    );
    let mut stepper = IncrementerStep::new();
    
    // Prove 3 steps
    for i in 1..=3 {
        let delta = F::from_u64(i);
        session.prove_step(&mut stepper, &ExtInputs { delta }).expect("prove_step failed");
    }
    
    // Verify chain
    let (chain, step_ios) = session.finalize();
    let descriptor = StepDescriptor {
        ccs: stepper.ccs.clone(),
        spec: stepper.spec.clone(),
    };
    
    let valid = verify_chain_with_descriptor(
        &descriptor,
        &chain,
        &[F::ZERO],  // initial_state
        &params,
        &step_ios,
        AppInputBinding::WitnessBound,
    ).expect("Verification failed");
    
    assert!(valid, "Chain verification should succeed without y_prev binding");
}

/// Test that circuits work correctly without y_prev binding (TranscriptOnly mode)
#[test]
fn test_self_contained_witnesses_transcript_only() {
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        Some(&params),
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::TranscriptOnly,
    );
    let mut stepper = IncrementerStep::new();
    
    // Prove 3 steps
    for i in 1..=3 {
        let delta = F::from_u64(i);
        session.prove_step(&mut stepper, &ExtInputs { delta }).expect("prove_step failed");
    }
    
    // Verify chain
    let (chain, step_ios) = session.finalize();
    let descriptor = StepDescriptor {
        ccs: stepper.ccs.clone(),
        spec: stepper.spec.clone(),
    };
    
    let valid = verify_chain_with_descriptor(
        &descriptor,
        &chain,
        &[F::ZERO],  // initial_state
        &params,
        &step_ios,
        AppInputBinding::TranscriptOnly,
    ).expect("Verification failed");
    
    assert!(valid, "Chain verification should succeed without y_prev binding");
}
