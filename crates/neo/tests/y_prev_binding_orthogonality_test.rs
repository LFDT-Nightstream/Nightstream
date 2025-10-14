//! Test that y_prev binding is orthogonal to AppInputBinding mode
//!
//! This test verifies the security property that y_prev equality constraints
//! are enforced in BOTH WitnessBound and TranscriptOnly modes, ensuring that
//! the circuit uses the same y_prev as the accumulator regardless of how
//! app inputs are bound.
//!
//! ## Background
//!
//! The AppInputBinding mode controls how application public inputs are bound:
//! - WitnessBound: app inputs get in-circuit equality constraints (belt-and-suspenders)
//! - TranscriptOnly: app inputs only influence Fiat-Shamir transcript (NIVC mode)
//!
//! However, y_prev binding is about STATE CONSISTENCY, not app input handling.
//! Without y_prev binding, a malicious circuit could use y_prev' != y_prev
//! internally while still producing valid proofs against the accumulator's y_prev.
//!
//! This test ensures that y_prev_witness_indices are respected in BOTH modes.

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
            y_prev_indices: Some(vec![1]), // prev_x at witness[1] - THIS IS THE KEY
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

/// Test that y_prev binding works in WitnessBound mode
#[test]
fn test_y_prev_binding_in_witness_bound_mode() {
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        &params,
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::WitnessBound, // <-- WitnessBound mode
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
        &[F::ZERO],
        &params,
        &step_ios,
        AppInputBinding::WitnessBound,
    ).expect("verification failed");
    
    assert!(valid, "Chain should be valid in WitnessBound mode with y_prev binding");
}

/// Test that y_prev binding ALSO works in TranscriptOnly mode
/// This is the key test - y_prev binding should be orthogonal to app input mode
#[test]
fn test_y_prev_binding_in_transcript_only_mode() {
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        &params,
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::TranscriptOnly, // <-- TranscriptOnly mode
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
        &[F::ZERO],
        &params,
        &step_ios,
        AppInputBinding::TranscriptOnly,
    ).expect("verification failed");
    
    assert!(valid, "Chain should be valid in TranscriptOnly mode with y_prev binding");
}

/// Test that y_prev binding is enforced in both modes by checking the binding spec
#[test]
fn test_binding_spec_includes_y_prev_in_both_modes() {
    let spec = StepSpec {
        y_len: 1,
        const1_index: 0,
        y_step_indices: vec![3],
        y_prev_indices: Some(vec![1]),
        app_input_indices: Some(vec![2]),
    };
    
    // WitnessBound mode should include y_prev binding
    let witness_bound_spec = spec.binding_spec(AppInputBinding::WitnessBound);
    assert_eq!(
        witness_bound_spec.y_prev_witness_indices,
        vec![1],
        "WitnessBound mode should include y_prev_witness_indices"
    );
    assert_eq!(
        witness_bound_spec.step_program_input_witness_indices,
        vec![2],
        "WitnessBound mode should include app input indices"
    );
    
    // TranscriptOnly mode should ALSO include y_prev binding (orthogonal)
    let transcript_only_spec = spec.binding_spec(AppInputBinding::TranscriptOnly);
    assert_eq!(
        transcript_only_spec.y_prev_witness_indices,
        vec![1],
        "TranscriptOnly mode should ALSO include y_prev_witness_indices (orthogonal to app input mode)"
    );
    assert_eq!(
        transcript_only_spec.step_program_input_witness_indices,
        Vec::<usize>::new(),
        "TranscriptOnly mode should NOT include app input indices (transcript-only binding)"
    );
}

/// Test that omitting y_prev_indices results in no y_prev binding in either mode
#[test]
fn test_omitting_y_prev_indices_results_in_no_binding() {
    let spec = StepSpec {
        y_len: 1,
        const1_index: 0,
        y_step_indices: vec![3],
        y_prev_indices: None, // <-- Explicitly None
        app_input_indices: Some(vec![2]),
    };
    
    let witness_bound_spec = spec.binding_spec(AppInputBinding::WitnessBound);
    assert!(
        witness_bound_spec.y_prev_witness_indices.is_empty(),
        "No y_prev binding when y_prev_indices is None (WitnessBound)"
    );
    
    let transcript_only_spec = spec.binding_spec(AppInputBinding::TranscriptOnly);
    assert!(
        transcript_only_spec.y_prev_witness_indices.is_empty(),
        "No y_prev binding when y_prev_indices is None (TranscriptOnly)"
    );
}

/// Test that WitnessBound finalization preserves app input binding constraints
/// This is critical: finalization must use the SAME binding mode as proving
#[test]
fn test_witness_bound_finalization_preserves_app_input_binding() {
    use neo::session::{finalize_ivc_chain_with_options, IvcFinalizeOptions};
    
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        &params,
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::WitnessBound, // <-- WitnessBound mode
    );
    let mut stepper = IncrementerStep::new();
    
    // Prove 2 steps
    for i in 1..=2 {
        let delta = F::from_u64(i);
        session.prove_step(&mut stepper, &ExtInputs { delta }).expect("prove_step failed");
    }
    
    // Finalize chain
    let (chain, _step_ios) = session.finalize();
    let descriptor = StepDescriptor {
        ccs: stepper.ccs.clone(),
        spec: stepper.spec.clone(),
    };
    
    // Test WitnessBound finalization (should succeed - binding matches proving)
    let result_witness_bound = finalize_ivc_chain_with_options(
        &descriptor,
        &params,
        chain.clone(),
        AppInputBinding::WitnessBound, // <-- MUST match proving mode
        IvcFinalizeOptions { embed_ivc_ev: true }
    );
    
    assert!(
        result_witness_bound.is_ok(),
        "Finalization with WitnessBound should succeed when proving used WitnessBound"
    );
    
    // Verify the final SNARK
    if let Ok(Some((final_proof, final_ccs, final_public_input))) = result_witness_bound {
        let valid = neo::verify_spartan2(&final_ccs, &final_public_input, &final_proof)
            .expect("verify_spartan2 failed");
        assert!(valid, "Final SNARK should be valid when binding modes match");
    }
}

/// Test that TranscriptOnly finalization works for TranscriptOnly sessions
#[test]
fn test_transcript_only_finalization() {
    use neo::session::{finalize_ivc_chain_with_options, IvcFinalizeOptions};
    
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(
        &params,
        Some(vec![F::ZERO]),
        0,
        AppInputBinding::TranscriptOnly, // <-- TranscriptOnly mode
    );
    let mut stepper = IncrementerStep::new();
    
    // Prove 2 steps
    for i in 1..=2 {
        let delta = F::from_u64(i);
        session.prove_step(&mut stepper, &ExtInputs { delta }).expect("prove_step failed");
    }
    
    // Finalize chain
    let (chain, _step_ios) = session.finalize();
    let descriptor = StepDescriptor {
        ccs: stepper.ccs.clone(),
        spec: stepper.spec.clone(),
    };
    
    // Test TranscriptOnly finalization
    let result_transcript_only = finalize_ivc_chain_with_options(
        &descriptor,
        &params,
        chain,
        AppInputBinding::TranscriptOnly, // <-- MUST match proving mode
        IvcFinalizeOptions { embed_ivc_ev: true }
    );
    
    assert!(
        result_transcript_only.is_ok(),
        "Finalization with TranscriptOnly should succeed when proving used TranscriptOnly"
    );
    
    // Verify the final SNARK
    if let Ok(Some((final_proof, final_ccs, final_public_input))) = result_transcript_only {
        let valid = neo::verify_spartan2(&final_ccs, &final_public_input, &final_proof)
            .expect("verify_spartan2 failed");
        assert!(valid, "Final SNARK should be valid for TranscriptOnly mode");
    }
}

