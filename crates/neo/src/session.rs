//! High-level IVC session API (Nova/Sonobe-style)
//!
//! This module provides a thin ergonomic layer over the low-level IVC API in
//! `crate::ivc`, removing the need for integrators to construct binding specs,
//! manage prev_augmented_x, or thread folding linkage manually.
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
use p3_field::PrimeCharacteristicRing;
use crate::ivc::{
    Accumulator, IvcProof, IvcChainProof, IvcStepInput,
    StepBindingSpec, prove_ivc_step_chained, verify_ivc_chain,
};

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
pub struct IvcSession {
    params: NeoParams,
    acc: Accumulator,
    step: u64,
    proofs: Vec<IvcProof>,
    // Carry folding linkage across steps (strict chaining)
    prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    prev_me_wit: Option<neo_ccs::MeWitness<F>>,
    prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
}

impl IvcSession {
    /// Create a new session with an initial state. `start_step` is the public
    /// step counter bound into the transcript.
    pub fn new(params: &NeoParams, initial_state: Vec<F>, start_step: u64) -> Self {
        IvcSession {
            params: *params,
            acc: Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: initial_state, step: start_step },
            step: start_step,
            proofs: Vec::new(),
            prev_me: None,
            prev_me_wit: None,
            prev_lhs_mcs: None,
        }
    }

    /// Current state (y_prev)
    pub fn state(&self) -> &[F] { &self.acc.y_compact }
    /// Current step index
    pub fn step(&self) -> u64 { self.step }

    /// Prove one step and advance the session state.
    pub fn prove_step<S: NeoStep>(
        &mut self,
        stepper: &mut S,
        inputs: &S::ExternalInputs,
    ) -> Result<IvcProof, Box<dyn std::error::Error>> {
        // 1) Get artifacts for this step from the adapter
        let artifacts = stepper.synthesize_step(self.step as usize, &self.acc.y_compact, inputs);

        // Sanity: state length consistency
        if artifacts.spec.y_len != self.acc.y_compact.len() {
            return Err(format!(
                "state length mismatch: spec.y_len={} != acc.y_len={}",
                artifacts.spec.y_len, self.acc.y_compact.len()
            ).into());
        }

        // 2) Build binding spec from the single source of truth (StepSpec)
        let binding = artifacts.spec.binding_spec();

        // 3) Extract y_step from witness (exact slots)
        let mut y_step = Vec::with_capacity(artifacts.spec.y_step_indices.len());
        for &idx in &artifacts.spec.y_step_indices {
            let val = *artifacts.witness.get(idx).ok_or_else(|| format!(
                "y_step index {} out of bounds for witness (len={})",
                idx, artifacts.witness.len()
            ))?;
            y_step.push(val);
        }

        // 4) Prepare IVC input and fold with strict linkage
        let input = IvcStepInput {
            params: &self.params,
            step_ccs: &artifacts.ccs,
            step_witness: &artifacts.witness,
            prev_accumulator: &self.acc,
            step: self.step,
            public_input: if artifacts.public_app_inputs.is_empty() { None } else { Some(&artifacts.public_app_inputs) },
            y_step: &y_step,
            binding_spec: &binding,
            // If no app_input_indices provided, allow transcript-only app inputs
            transcript_only_app_inputs: artifacts.spec.app_input_indices.as_ref().map_or(true, |v| v.is_empty()),
            prev_augmented_x: self.proofs.last().map(|p| p.step_augmented_public_input.as_slice()),
        };

        let (step_result, me_out, me_wit_out, lhs_next) =
            prove_ivc_step_chained(input, self.prev_me.clone(), self.prev_me_wit.clone(), self.prev_lhs_mcs.clone())?;

        let proof = step_result.proof;

        // 5) Advance chaining state
        self.prev_me = Some(me_out);
        self.prev_me_wit = Some(me_wit_out);
        self.prev_lhs_mcs = Some(lhs_next);
        self.acc = proof.next_accumulator.clone();
        self.step = proof.step + 1;
        self.proofs.push(proof.clone());

        Ok(proof)
    }

    /// Finalize the session and return an IVC chain proof
    pub fn finalize(self) -> IvcChainProof {
        IvcChainProof { steps: self.proofs, final_accumulator: self.acc, chain_length: self.step }
    }
}

/// Verify a complete chain using a descriptor (shape + spec) and the initial state.
pub fn verify_chain_with_descriptor(
    descriptor: &StepDescriptor,
    chain: &IvcChainProof,
    initial_state: &[F],
    params: &NeoParams,
) -> Result<bool, Box<dyn std::error::Error>> {
    let binding = descriptor.spec.binding_spec();
    let initial_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: initial_state.to_vec(), step: 0 };
    verify_ivc_chain(&descriptor.ccs, chain, &initial_acc, &binding, params)
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
    let rho = if last.step_rho != F::ZERO { last.step_rho } else { F::ONE }; // fallback for older proofs
    let y_prev = last.step_y_prev.clone();
    let y_next = last.step_y_next.clone();
    let step_x = last.step_public_input.clone();
    let y_len = y_prev.len();

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
