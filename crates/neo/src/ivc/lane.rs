//! IVC API as single-lane NIVC (Option A implementation)
//!
//! This module implements IVC as a thin wrapper over NIVC with exactly one lane.
//! Running NIVC with lanes=1 gives IVC semantics with no forked code paths.
//!
//! ## Design
//! - IVC is implemented as NIVC with a single-lane program (LaneId(0))
//! - All IVC functions delegate to NIVC with lane 0
//! - No code duplication: NIVC handles all the core logic
//! - Small overhead (Vec<LaneRunningState> of length 1) is negligible
//!
//! ## API Mapping
//! - `prove_ivc_step` → NIVC with single-step program, lane 0
//! - `verify_ivc_step` → NIVC verifier with lane 0
//! - `prove_ivc_chain` → NIVC chain with all steps on lane 0
//! - `verify_ivc_chain` → NIVC chain verifier

use crate::F;
use crate::shared::types::*;
use crate::nivc::{NivcProgram, NivcStepSpec, NivcState, NivcStepProof, NivcChainProof, NivcAccumulators, LaneRunningState};
use neo_ccs::CcsStructure;

/// Prove a single IVC step using single-lane NIVC
///
/// This is the main IVC proving function, implemented as a thin wrapper over NIVC
/// with exactly one lane (lane 0). All IVC semantics are preserved.
///
/// # Security
/// - Base case: enforces canonical zero vector for LHS when prev_accumulator is empty
/// - Transcript: lane 0 is absorbed (constant domain separation)
/// - Public ρ: same EV path as legacy IVC
///
/// # Note
/// This produces proofs with NIVC-style public inputs (lane selector + lanes_root).
/// For backward compatibility with pre-Option-A proofs, use the internal pipeline directly.
pub fn prove_ivc_step(input: IvcStepInput) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // Build single-lane NIVC program
    let program = NivcProgram::new(vec![NivcStepSpec {
        ccs: input.step_ccs.clone(),
        binding: input.binding_spec.clone(),
    }]);

    // Initialize NIVC state with single lane
    // Use prev_accumulator.y_compact as initial state y0
    let mut state = NivcState::new(
        *input.params,
        program,
        input.prev_accumulator.y_compact.clone(),
    )?;
    
    // Set the starting step counter to match the previous accumulator
    state.acc.step = input.prev_accumulator.step;
    
    // Populate lane 0 with the previous accumulator state if it exists
    // This ensures proper chaining when c_coords is non-empty
    if !input.prev_accumulator.c_coords.is_empty() {
        state.acc.lanes[0].c_coords = input.prev_accumulator.c_coords.clone();
        state.acc.lanes[0].c_digest = input.prev_accumulator.c_z_digest;
    }
    
    // Set prev_augmented_x if provided (for linkage)
    if let Some(prev_aug_x) = input.prev_augmented_x {
        state.prev_aug_x_by_lane[0] = Some(prev_aug_x.to_vec());
    }

    // Build step_io from public_input (if any)
    // In IVC, step_io is just the step's public input (no lane selector needed internally)
    let step_io = input.public_input.unwrap_or(&[]);

    // Prove step on lane 0
    // NIVC will internally call prove_ivc_step_chained with the appropriate parameters
    let nivc_proof = crate::nivc::pipeline::prover::step(
        &mut state,
        0, // Lane 0 (single lane)
        step_io,
        input.step_witness,
    )?;

    // Extract IVC result from NIVC proof
    let ivc_proof = nivc_proof.inner;
    let next_state = state.acc.global_y.clone();

    Ok(IvcStepResult {
        proof: ivc_proof,
        next_state,
    })
}

/// Verify an IVC step using single-lane NIVC verifier
///
/// This verifies a single IVC step by checking it against NIVC semantics with lane 0.
///
/// # Security
/// - Base case: enforces canonical zero vector for LHS augmented X
/// - Binding: uses trusted binding_spec (NOT from proof)
/// - Transcript: recomputes ρ and validates proof consistency
pub fn verify_ivc_step(
    step_ccs: &CcsStructure<F>,
    ivc_proof: &IvcProof,
    prev_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
    params: &crate::NeoParams,
    _prev_augmented_x: Option<&[F]>,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Build single-lane NIVC program
    let program = NivcProgram::new(vec![NivcStepSpec {
        ccs: step_ccs.clone(),
        binding: binding_spec.clone(),
    }]);

    // Build initial NIVC accumulator state from prev_accumulator
    let mut acc = NivcAccumulators::new(1, prev_accumulator.y_compact.clone());
    acc.step = prev_accumulator.step;
    acc.lanes[0].c_coords = prev_accumulator.c_coords.clone();
    acc.lanes[0].c_digest = prev_accumulator.c_z_digest;

    // Wrap IVC proof as NIVC step proof
    // In single-lane IVC, there's no lane selector in step_io (it's implicit: always lane 0)
    // Extract step_io from proof's step_public_input by removing the accumulator digest prefix
    let acc_digest_fields = crate::shared::digest::compute_accumulator_digest_fields(prev_accumulator)?;
    let step_public_input = ivc_proof.public_inputs.wrapper_public_input_x();
    let step_io = if step_public_input.len() > acc_digest_fields.len() {
        step_public_input[acc_digest_fields.len()..].to_vec()
    } else {
        vec![]
    };

    let nivc_step = NivcStepProof {
        lane_idx: 0, // Lane 0
        step_io,
        inner: ivc_proof.clone(),
    };

    // Build NIVC chain with single step
    let nivc_chain = NivcChainProof {
        steps: vec![nivc_step],
        final_acc: NivcAccumulators {
            lanes: vec![LaneRunningState {
                me: None,
                wit: None,
                c_coords: ivc_proof.next_accumulator.c_coords.clone(),
                c_digest: ivc_proof.next_accumulator.c_z_digest,
                lhs_mcs: None,
                lhs_mcs_wit: None,
            }],
            global_y: ivc_proof.next_accumulator.y_compact.clone(),
            step: ivc_proof.next_accumulator.step,
        },
    };

    // Store prev_augmented_x for NIVC verifier if provided
    // We need to pass initial state y0 to verify_nivc_chain
    let y0 = prev_accumulator.y_compact.clone();

    // Use NIVC verifier with single-lane program
    crate::nivc::pipeline::verifier::verify_chain(&program, params, &nivc_chain, &y0)
        .map_err(|e| e.into())
}

/// Prove an entire IVC chain using single-lane NIVC
///
/// This proves a sequence of IVC steps by running them all on lane 0 of a NIVC program.
pub fn prove_ivc_chain(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_inputs: &[IvcChainStepInput],
    initial_accumulator: Accumulator,
    binding_spec: &StepBindingSpec,
) -> Result<IvcChainProof, Box<dyn std::error::Error>> {
    // Build single-lane NIVC program
    let program = NivcProgram::new(vec![NivcStepSpec {
        ccs: step_ccs.clone(),
        binding: binding_spec.clone(),
    }]);

    // Initialize NIVC state
    let mut state = NivcState::new(
        *params,
        program,
        initial_accumulator.y_compact.clone(),
    )?;
    state.acc.step = initial_accumulator.step;
    
    // Populate initial lane state if accumulator has commitment
    if !initial_accumulator.c_coords.is_empty() {
        state.acc.lanes[0].c_coords = initial_accumulator.c_coords.clone();
        state.acc.lanes[0].c_digest = initial_accumulator.c_z_digest;
    }

    // Execute all steps on lane 0
    for step_input in step_inputs {
        let step_io = step_input.public_input.as_ref().map(|v| v.as_slice()).unwrap_or(&[]);
        
        crate::nivc::pipeline::prover::step(
            &mut state,
            0, // Lane 0
            step_io,
            &step_input.witness,
        )?;
    }

    // Convert NIVC chain proof to IVC chain proof
    let nivc_chain = state.into_proof();
    let ivc_steps: Vec<IvcProof> = nivc_chain.steps.into_iter().map(|sp| sp.inner).collect();
    
    Ok(IvcChainProof {
        steps: ivc_steps,
        final_accumulator: Accumulator {
            c_z_digest: nivc_chain.final_acc.lanes[0].c_digest,
            c_coords: nivc_chain.final_acc.lanes[0].c_coords.clone(),
            y_compact: nivc_chain.final_acc.global_y,
            step: nivc_chain.final_acc.step,
        },
        chain_length: nivc_chain.final_acc.step,
    })
}

/// Verify an entire IVC chain using single-lane NIVC verifier
pub fn verify_ivc_chain(
    step_ccs: &CcsStructure<F>,
    chain: &IvcChainProof,
    initial_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
    params: &crate::NeoParams,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Build single-lane NIVC program
    let program = NivcProgram::new(vec![NivcStepSpec {
        ccs: step_ccs.clone(),
        binding: binding_spec.clone(),
    }]);

    // Convert IVC chain to NIVC chain
    let nivc_steps: Vec<NivcStepProof> = chain.steps.iter().enumerate().map(|(idx, ivc_proof)| {
        // Extract step_io from step_public_input (remove accumulator digest prefix)
        // The proof was generated through the lane wrapper, so app_inputs = [lane_idx || step_io || lanes_root]
        let acc_digest_len = if idx == 0 {
            // For first step, compute digest length from initial accumulator
            crate::shared::digest::compute_accumulator_digest_fields(initial_accumulator)
                .map(|d| d.len())
                .unwrap_or(4)
        } else {
            // For subsequent steps, we can use the previous proof's accumulator
            4 // DIGEST_LEN is always 4
        };
        
        let step_public_input = ivc_proof.public_inputs.wrapper_public_input_x();
        let step_io = if step_public_input.len() > acc_digest_len {
            let app_inputs = &step_public_input[acc_digest_len..];
            // app_inputs format: [lane_idx(1) || step_io || lanes_root(4)]
            if app_inputs.len() > 5 {
                app_inputs[1..app_inputs.len()-4].to_vec()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        
        NivcStepProof {
            lane_idx: 0,
            step_io,
            inner: ivc_proof.clone(),
        }
    }).collect();

    let nivc_chain = NivcChainProof {
        steps: nivc_steps,
        final_acc: NivcAccumulators {
            lanes: vec![LaneRunningState {
                me: None,
                wit: None,
                c_coords: chain.final_accumulator.c_coords.clone(),
                c_digest: chain.final_accumulator.c_z_digest,
                lhs_mcs: None,
                lhs_mcs_wit: None,
            }],
            global_y: chain.final_accumulator.y_compact.clone(),
            step: chain.final_accumulator.step,
        },
    };

    // Use NIVC verifier with initial_y from initial_accumulator
    let initial_y = &initial_accumulator.y_compact;
    crate::nivc::pipeline::verifier::verify_chain(&program, params, &nivc_chain, initial_y)
        .map_err(|e| e.into())
}
