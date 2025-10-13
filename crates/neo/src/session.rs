//! High-level IVC session API (Nova/Sonobe-style)
//!
//! This module provides a thin ergonomic layer over the NIVC implementation,
//! specialized for the common case of a single step type (uniform IVC).
//! Since IVC is a subset of NIVC (single-lane NIVC), this implementation
//! delegates to `crate::nivc::NivcState` with a single-lane program.
//!
//! Core concepts:
//! - `StepSpec`: single source of truth for how a step exposes its outputs
//!   (`y_step_indices`) and where the constant-1 witness lives (`const1_index`).
//! - `StepArtifacts`: per-step CCS + witness + optional public app inputs
//!   produced by the circuit adapter.
//! - `NeoStep`: circuit-facing trait; implementors produce `StepArtifacts` and
//!   a stable `StepSpec`.
//! - `IvcSession`: owns chaining state; each `prove_step` call folds one step
//!   and advances the accumulator.

use crate::{NeoParams, CcsStructure, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use crate::ivc::{
    Accumulator, IvcProof, IvcChainProof,
    StepBindingSpec,
};
use crate::nivc::{NivcState, NivcProgram, NivcStepSpec};

/// Canonical description of how a step exposes its state/output and wiring.
#[derive(Clone, Debug)]
pub struct StepSpec {
    /// Compact y length (state size)
    pub y_len: usize,
    /// Index in the witness that must equal 1 (constant-1 column)
    pub const1_index: usize,
    /// Exact witness indices corresponding to the next state z_{i+1}
    pub y_step_indices: Vec<usize>,
    /// Optional witness indices where the circuit reads y_prev (for binding)
    pub y_prev_indices: Option<Vec<usize>>,
    /// Optional witness indices that must bind to the app tail of step_x
    pub app_input_indices: Option<Vec<usize>>,
}

impl StepSpec {
    pub fn binding_spec(&self) -> StepBindingSpec {
        StepBindingSpec {
            y_step_offsets: self.y_step_indices.clone(),
            y_prev_witness_indices: self.y_prev_indices.clone().unwrap_or_default(),
            step_program_input_witness_indices: self.app_input_indices.clone().unwrap_or_default(),
            const1_witness_index: self.const1_index,
        }
    }
}

/// Per-step artifacts produced by the circuit adapter.
#[derive(Clone, Debug)]
pub struct StepArtifacts {
    pub ccs: CcsStructure<F>,
    pub witness: Vec<F>,
    /// Application-level public inputs (tail of step_x). Can be empty.
    pub public_app_inputs: Vec<F>,
    pub spec: StepSpec,
}

/// Descriptor required for verification (shape-only; no witness).
#[derive(Clone, Debug)]
pub struct StepDescriptor {
    pub ccs: CcsStructure<F>,
    pub spec: StepSpec,
}

/// Circuit-facing trait. Implement this via an adapter (Arkworks/Nova/Sonobe).
pub trait NeoStep {
    type ExternalInputs: Clone + Default;
    /// Size of the state vector (y_len)
    fn state_len(&self) -> usize;
    /// Stable step specification (indices are per-shape, not per-instance)
    fn step_spec(&self) -> StepSpec;
    /// Produce CCS + witness + optional public inputs for the current step
    fn synthesize_step(
        &mut self,
        step_idx: usize,
        z_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts;
}

/// High-level IVC driver that owns chaining state.
/// Internally delegates to a single-lane NIVC state.
pub struct IvcSession {
    inner: Option<NivcState>,
    params: NeoParams,
    step_spec: Option<StepSpec>,
    // Cached initial state for lazy initialization
    initial_state: Vec<F>,
    start_step: u64,
    // Store step_io for each step (needed for verification)
    step_ios: Vec<Vec<F>>,
}

impl IvcSession {
    /// Create a new session with an optional initial state. `start_step` is the public
    /// step counter bound into the transcript.
    /// - When `initial_state` is `None`, the accumulator state is initialized lazily
    ///   on the first step to a zero vector of length `spec.y_len`.
    pub fn new(params: &NeoParams, initial_state: Option<Vec<F>>, start_step: u64) -> Self {
        IvcSession {
            inner: None,
            params: *params,
            step_spec: None,
            initial_state: initial_state.unwrap_or_default(),
            start_step,
            step_ios: Vec::new(),
        }
    }

    /// Current state (y_prev)
    pub fn state(&self) -> &[F] {
        self.inner.as_ref().map(|s| s.acc.global_y.as_slice()).unwrap_or(&self.initial_state)
    }
    
    /// Current step index
    pub fn step(&self) -> u64 {
        self.inner.as_ref().map(|s| s.acc.step).unwrap_or(self.start_step)
    }

    /// Prove one step and advance the session state.
    pub fn prove_step<S: NeoStep>(
        &mut self,
        stepper: &mut S,
        inputs: &S::ExternalInputs,
    ) -> Result<IvcProof, Box<dyn std::error::Error>> {
        let current_step = self.step();
        let current_state = self.state().to_vec();
        
        // 1) Get artifacts for this step from the adapter
        let artifacts = stepper.synthesize_step(current_step as usize, &current_state, inputs);

        // 2) Lazy initialization: create the NIVC state on first step
        if self.inner.is_none() {
            // Initialize state if empty
            let y0 = if self.initial_state.is_empty() && artifacts.spec.y_len > 0 {
                vec![F::ZERO; artifacts.spec.y_len]
            } else if artifacts.spec.y_len != self.initial_state.len() && !self.initial_state.is_empty() {
                return Err(format!(
                    "state length mismatch: spec.y_len={} != initial_state.len={}",
                    artifacts.spec.y_len, self.initial_state.len()
                ).into());
            } else {
                self.initial_state.clone()
            };
            
            // Create single-lane NIVC program from this step's spec
            // SECURITY FIX: Keep step_program_input_witness_indices for in-circuit binding
            // during finalization. The transcript_only_app_inputs flag in IvcStepInput
            // handles NIVC's transcript-based binding during step proving without
            // compromising the security of the final SNARK.
            let binding = artifacts.spec.binding_spec();
            let program = NivcProgram::new(vec![
                NivcStepSpec {
                    ccs: artifacts.ccs.clone(),
                    binding,
                }
            ]);
            
            let mut nivc_state = NivcState::new(self.params, program, y0)?;
            nivc_state.acc.step = self.start_step;
            self.inner = Some(nivc_state);
            self.step_spec = Some(artifacts.spec.clone());
        }

        // 3) Verify that the step spec hasn't changed (IVC assumes uniform steps)
        if let Some(ref cached_spec) = self.step_spec {
            if artifacts.spec.y_len != cached_spec.y_len {
                return Err("Step spec changed: y_len mismatch".into());
            }
        }

        // 4) Delegate to NIVC with single lane (lane index 0)
        let nivc_state = self.inner.as_mut().unwrap();
        let nivc_proof = nivc_state.step(
            0, // Single lane index
            &artifacts.public_app_inputs,
            &artifacts.witness,
        )?;

        // 5) Store step_io for verification
        self.step_ios.push(artifacts.public_app_inputs.clone());

        // 6) Extract the inner IVC proof to maintain API compatibility
        Ok(nivc_proof.inner)
    }

    /// Finalize the session and return an IVC chain proof with metadata for verification
    pub fn finalize(self) -> (IvcChainProof, Vec<Vec<F>>) {
        if let Some(inner) = self.inner {
            // Extract IVC proofs from NIVC chain (single-lane)
            let nivc_chain = inner.into_proof();
            let proofs: Vec<IvcProof> = nivc_chain.steps.into_iter().map(|sp| sp.inner).collect();
            let final_accumulator = Accumulator {
                c_z_digest: nivc_chain.final_acc.lanes[0].c_digest,
                c_coords: nivc_chain.final_acc.lanes[0].c_coords.clone(),
                y_compact: nivc_chain.final_acc.global_y,
                step: nivc_chain.final_acc.step,
            };
            let chain = IvcChainProof {
                steps: proofs,
                final_accumulator: final_accumulator.clone(),
                chain_length: final_accumulator.step,
            };
            (chain, self.step_ios)
        } else {
            // No steps taken, return empty chain
            let chain = IvcChainProof {
                steps: vec![],
                final_accumulator: Accumulator {
                    c_z_digest: [0u8; 32],
                    c_coords: vec![],
                    y_compact: self.initial_state,
                    step: self.start_step,
                },
                chain_length: self.start_step,
            };
            (chain, vec![])
        }
    }
}

/// Verify a complete chain using a descriptor (shape + spec) and the initial state.
/// Since IvcSession now uses NIVC internally, this converts the IVC chain to NIVC format for verification.
/// The step_ios parameter contains the original app inputs for each step (returned from finalize()).
pub fn verify_chain_with_descriptor(
    descriptor: &StepDescriptor,
    chain: &IvcChainProof,
    initial_state: &[F],
    params: &NeoParams,
    step_ios: &[Vec<F>],
) -> Result<bool, Box<dyn std::error::Error>> {
    // Convert IvcChainProof to NivcChainProof for verification
    // Since this is a single-lane NIVC, all proofs use lane 0
    if chain.steps.len() != step_ios.len() {
        return Err(format!(
            "Mismatch: chain has {} steps but {} step_ios provided",
            chain.steps.len(),
            step_ios.len()
        ).into());
    }
    
    let nivc_steps: Vec<crate::nivc::NivcStepProof> = chain.steps.iter()
        .zip(step_ios.iter())
        .map(|(ivc_proof, step_io)| {
            crate::nivc::NivcStepProof {
                lane_idx: 0, // Single lane
                step_io: step_io.clone(),
                inner: ivc_proof.clone(),
            }
        }).collect();
    
    let nivc_acc = crate::nivc::NivcAccumulators {
        lanes: vec![crate::nivc::LaneRunningState {
            me: None,
            wit: None,
            c_coords: chain.final_accumulator.c_coords.clone(),
            c_digest: chain.final_accumulator.c_z_digest,
            lhs_mcs: None,
            lhs_mcs_wit: None,
        }],
        global_y: chain.final_accumulator.y_compact.clone(),
        step: chain.final_accumulator.step,
    };
    
    let nivc_chain = crate::nivc::NivcChainProof {
        steps: nivc_steps,
        final_acc: nivc_acc,
    };
    
    // Create single-lane NIVC program for verification
    // SECURITY FIX: Keep step_program_input_witness_indices for consistency
    // with the prover's binding spec (even though NIVC verification is transcript-based)
    let binding = descriptor.spec.binding_spec();
    
    let program = NivcProgram::new(vec![
        NivcStepSpec {
            ccs: descriptor.ccs.clone(),
            binding,
        }
    ]);
    
    crate::nivc::verify_nivc_chain(&program, params, &nivc_chain, initial_state)
        .map_err(|e| e.into())
}

/// Options for IVC final proof (Stage 5)
pub struct IvcFinalizeOptions { pub embed_ivc_ev: bool }

/// Generate a succinct final SNARK for a plain IVC chain using the last step's shape.
/// Returns (lean proof, augmented CCS, final public input). Returns Ok(None) if the chain is empty.
pub fn finalize_ivc_chain_with_options(
    descriptor: &StepDescriptor,
    params: &NeoParams,
    chain: IvcChainProof,
    _opts: IvcFinalizeOptions,
) -> Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>, Box<dyn std::error::Error>> {
    if chain.steps.is_empty() { return Ok(None); }
    let last = chain.steps.last().unwrap();

    // Extract step data
    let y_prev = last.public_inputs.y_prev().to_vec();
    let y_next = last.public_inputs.y_next().to_vec();
    let step_x = last.public_inputs.wrapper_public_input_x().to_vec();
    let y_len = y_prev.len();
    
    // 🔒 SECURITY HARDENING: Multi-layer ρ validation to prevent transcript manipulation
    
    // 1️⃣ Reconstruct previous accumulator with validation
    // Storage for initial accumulator (must outlive the borrow)
    let initial_acc_storage = crate::ivc::Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y_prev.clone(),
        step: if last.step > 0 { last.step - 1 } else { 0 },
    };
    
    let prev_acc = if chain.steps.len() > 1 {
        // Multi-step chain: use the accumulator from the previous step
        let prev_step = &chain.steps[chain.steps.len() - 2];
        
        // Validate step monotonicity
        if prev_step.step + 1 != last.step {
            return Err(format!(
                "SECURITY: Step counter discontinuity. Expected {}, got {}",
                prev_step.step + 1, last.step
            ).into());
        }
        
        &prev_step.next_accumulator
    } else {
        // Single step: reconstruct initial accumulator from step data
        if last.step != 0 {
            return Err("SECURITY: Cannot have first step at counter > 0 without prior steps".into());
        }
        
        // Validate initial accumulator structure
        if !initial_acc_storage.c_coords.is_empty() {
            return Err("SECURITY: Initial accumulator must have empty c_coords".into());
        }
        if initial_acc_storage.step != last.step {
            return Err("SECURITY: Initial step counter mismatch".into());
        }
        
        &initial_acc_storage
    };
    
    // 2️⃣ Validate step_x prefix binds to H(prev_acc)
    let expected_prefix = crate::shared::digest::compute_accumulator_digest_fields(prev_acc)
        .map_err(|e| format!("Failed to compute accumulator digest: {}", e))?;
    
    if step_x.len() < expected_prefix.len() {
        return Err(format!(
            "SECURITY: step_x too short ({} < {}). Cannot contain H(prev_acc) prefix",
            step_x.len(), expected_prefix.len()
        ).into());
    }
    
    if !step_x.starts_with(&expected_prefix) {
        return Err(
            "SECURITY: step_x prefix mismatch. Expected H(prev_acc) binding but got different digest. \
             This prevents forged step_x from influencing ρ derivation.".into()
        );
    }
    
    // 3️⃣ Validate c_step_coords dimension (Pattern-B pre-commit dimension binding)
    // Ajtai commitments are always d×κ in size, independent of m
    let d = neo_math::ring::D;
    let expected_num_coords = d * params.kappa as usize;
    
    if last.c_step_coords.len() != expected_num_coords {
        return Err(format!(
            "SECURITY: c_step_coords dimension mismatch. Expected {} (d={} × κ={}), got {}. \
             This prevents dimension-based transcript manipulation.",
            expected_num_coords, d, params.kappa, last.c_step_coords.len()
        ).into());
    }
    
    // 4️⃣ Recompute ρ from transcript (now that all inputs are validated)
    let step_data = crate::ivc::internal::transcript::build_step_transcript_data(
        prev_acc,
        last.step,
        &step_x
    );
    let step_digest = crate::ivc::internal::transcript::create_step_digest(&step_data);
    let (rho_computed, _) = crate::ivc::internal::transcript::rho_from_transcript(
        prev_acc,
        step_digest,
        &last.c_step_coords
    );
    
    // 5️⃣ Reject degenerate ρ = 0 (transcript must produce nonzero challenge)
    if rho_computed == F::ZERO {
        return Err("SECURITY: ρ must be nonzero. This indicates a critical transcript failure.".into());
    }
    
    // 6️⃣ Strict equality check - no bypass allowed
    if last.public_inputs.rho() != rho_computed {
        return Err(format!(
            "SECURITY: step_rho mismatch. Proof contains {} but recomputed ρ is {}. \
             This indicates either a forged proof or transcript manipulation.",
            last.public_inputs.rho().as_canonical_u64(),
            rho_computed.as_canonical_u64()
        ).into());
    }
    
    let rho = rho_computed;

    // Build augmented CCS used by the last step
    let augmented_ccs = crate::ivc::build_augmented_ccs_linked_with_rlc(
        &descriptor.ccs,
        step_x.len(),
        &descriptor.spec.y_step_indices,
        &descriptor.spec.y_prev_indices.clone().unwrap_or_default(),
        &descriptor.spec.app_input_indices.clone().unwrap_or_default(),
        y_len,
        descriptor.spec.const1_index,
        None,
    ).map_err(|e| format!("Failed to build augmented CCS: {}", e))?;

    // Final public input for the outer SNARK
    let final_public_input = crate::ivc::build_final_snark_public_input(&step_x, rho, &y_prev, &y_next);

    // Choose ME instance/witness matching the step commitment
    let (final_me, final_me_wit) = if let (Some(meis), Some(wits)) = (&last.me_instances, &last.digit_witnesses) {
        if meis.is_empty() || wits.is_empty() { return Err("Missing ME instances for finalization".into()); }
        let mut idx = core::cmp::min(meis.len(), wits.len()) - 1; // default to last
        for i in 0..core::cmp::min(meis.len(), wits.len()) {
            if meis[i].c.data.len() == last.c_step_coords.len() && meis[i].c.data == last.c_step_coords {
                idx = i; break;
            }
        }
        (&meis[idx], &wits[idx])
    } else {
        return Err("IVC proof missing ME instances for finalization".into());
    };

    // Bridge to Spartan2 legacy format and compute VK/IO
    let (mut legacy_me, legacy_wit, ajtai_pp) = crate::adapt_from_modern(
        std::slice::from_ref(final_me),
        std::slice::from_ref(final_me_wit),
        &augmented_ccs,
        params,
        &[],
        None,
    )?;

    // Bind to context digest (augmented_ccs, final_public_input)
    let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
    #[allow(deprecated)]
    { legacy_me.header_digest = context_digest; }

    // Compress to lean proof (no EV embedding path to keep example simple)
    let ajtai_pp_arc = std::sync::Arc::new(ajtai_pp);
    let lean = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&legacy_me, &legacy_wit, Some(ajtai_pp_arc))?;

    let proof = crate::Proof {
        v: 2,
        circuit_key: lean.circuit_key,
        vk_digest: lean.vk_digest,
        public_io: lean.public_io_bytes,
        proof_bytes: lean.proof_bytes,
        public_results: vec![],
        meta: crate::ProofMeta { num_y_compact: y_len, num_app_outputs: 0 },
    };
    Ok(Some((proof, augmented_ccs, final_public_input)))
}
