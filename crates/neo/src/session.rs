//! High-level folding session API (Nova/Sonobe-style)
//!
//! This module provides an ergonomic session-based API for incremental folding computation,
//! specialized for the common case of a single step type (uniform IVC).
//!
//! ## Dual Backend Architecture
//!
//! `FoldingSession` supports two backend implementations, selected via `AppInputBinding` mode:
//!
//! ### 1. StandaloneIvc Backend (`AppInputBinding::WitnessBound`)
//! - Uses the lower-level IVC API directly
//! - Enforces **in-circuit witness binding constraints** 
//! - Guarantees belt-and-suspenders security: app inputs in the transcript match witness values
//! - **Use when:** Your circuit consumes app inputs from witness columns (e.g., `witness[2] = delta`)
//!
//! ### 2. NivcSingleLane Backend (`AppInputBinding::TranscriptOnly`)  
//! - Delegates to `crate::nivc::NivcState` with a single-lane program
//! - Uses **transcript-only binding** (Fiat-Shamir + verifier checks)
//! - No in-circuit witness binding constraints
//! - **Use when:** 
//!   - Your circuit reads app inputs from public `x` (not witness columns), OR
//!   - Your circuit doesn't consume external inputs, OR
//!   - App inputs are metadata-only (routing, scheduling hints)
//!
//! ## Core Concepts
//!
//! - `StepSpec`: single source of truth for how a step exposes its outputs
//!   (`y_step_indices`) and where the constant-1 witness lives (`const1_index`).
//! - `StepArtifacts`: per-step CCS + witness + optional public app inputs
//!   produced by the circuit adapter.
//! - `NeoStep`: circuit-facing trait; implementors produce `StepArtifacts` and
//!   a stable `StepSpec`.
//! - `FoldingSession`: owns chaining state; each `prove_step` call folds one step
//!   and advances the accumulator.

use crate::{NeoParams, CcsStructure, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use crate::ivc::{
    Accumulator, IvcProof, IvcChainProof,
    StepBindingSpec,
};
use crate::nivc::{NivcState, NivcProgram, NivcStepSpec};
use crate::shared::types::AppInputBinding;

/// Canonical description of how a step exposes its state/output and wiring.
/// 
/// **IMPORTANT**: This API does NOT support `y_prev` binding, which is incompatible with
/// ρ-folding. The accumulator's `y_compact` is a cryptographic commitment, not raw application
/// state. For application state continuity, either:
/// - Keep state in the witness (prover's responsibility), OR
/// - Pass previous outputs as `app_input_indices` (verified continuity)
#[derive(Clone, Debug)]
pub struct StepSpec {
    /// Compact y length (state size)
    pub y_len: usize,
    /// Index in the witness that must equal 1 (constant-1 column)
    pub const1_index: usize,
    /// Exact witness indices corresponding to the next state z_{i+1}
    pub y_step_indices: Vec<usize>,
    /// Optional witness indices for app inputs. Binding behavior depends on mode:
    /// - WitnessBound: enforces in-circuit equality (belt-and-suspenders)
    /// - TranscriptOnly: binding via Fiat-Shamir transcript only (NIVC mode)
    pub app_input_indices: Option<Vec<usize>>,
}

impl StepSpec {
    /// Convert to StepBindingSpec based on the chosen binding mode.
    /// 
    /// **App Input Binding (mode-dependent):**
    /// - **WitnessBound**: Enforces in-circuit equality between app inputs and witness.
    ///   Use when the circuit consumes app inputs from witness columns.
    /// - **TranscriptOnly**: App inputs only influence Fiat-Shamir transcript.
    ///   Use when circuit reads from public `x` directly or inputs are metadata.
    /// 
    /// **Note on y_prev binding:**
    /// - Always uses empty `y_prev_witness_indices` (incompatible with ρ-folding)
    /// - The accumulator's `y_compact` is a cryptographic commitment, not raw app state
    pub fn binding_spec(&self, mode: AppInputBinding) -> StepBindingSpec {
        match mode {
            AppInputBinding::WitnessBound => {
                // Uniform IVC: enforce that witness values match the app inputs
                // that seed the transcript (belt-and-suspenders)
                StepBindingSpec {
                    y_step_offsets: self.y_step_indices.clone(),
                    step_program_input_witness_indices: self.app_input_indices.clone().unwrap_or_default(),
                    const1_witness_index: self.const1_index,
                }
            }
        AppInputBinding::TranscriptOnly => {
            // NIVC or when circuit reads from public `x` directly:
            // no in-circuit witness binding for APP INPUTS, FS transcript + verifier checks only
            StepBindingSpec {
                y_step_offsets: self.y_step_indices.clone(),
                step_program_input_witness_indices: vec![],
                const1_witness_index: self.const1_index,
            }
        }
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
    
    /// Optional: Return security parameters for the circuit.
    /// 
    /// Returns `(ell, d_sc)` where:
    /// - `ell`: log₂(max_padded_circuit_size) across all steps
    /// - `d_sc`: maximum polynomial degree in CCS constraints
    /// 
    /// If `None` (default), parameters will be auto-detected from the first circuit.
    /// Override this when you know the circuit size ahead of time to avoid synthesis overhead.
    /// 
    /// # Example
    /// 
    /// ```ignore
    /// fn security_params(&self) -> Option<(u32, u32)> {
    ///     // Circuit has max 1024 constraints, degree 3
    ///     Some((10, 3))  // log2(1024) = 10
    /// }
    /// ```
    fn security_params(&self) -> Option<(u32, u32)> {
        None  // Default: auto-detect from first synthesis
    }
}

/// Backend implementation strategy for FoldingSession
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IvcBackend {
    /// Use standalone IVC with WitnessBound mode (direct IVC API)
    StandaloneIvc,
    /// Use NIVC with TranscriptOnly mode (single-lane NIVC)
    NivcSingleLane,
}

/// High-level folding session that owns chaining state.
/// 
/// Can operate in two modes:
/// - **WitnessBound**: Uses standalone IVC with in-circuit witness binding
/// - **TranscriptOnly**: Uses NIVC single-lane mode with transcript-only binding
pub struct FoldingSession {
    // NIVC backend (used when backend == NivcSingleLane)
    nivc_inner: Option<NivcState>,
    // Standalone IVC backend state (used when backend == StandaloneIvc)
    ivc_accumulator: Option<Accumulator>,
    #[allow(dead_code)]  // Reserved for future standalone IVC backend
    ivc_prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    #[allow(dead_code)]  // Reserved for future standalone IVC backend
    ivc_prev_me_wit: Option<neo_ccs::MeWitness<F>>,
    #[allow(dead_code)]  // Reserved for future standalone IVC backend
    ivc_prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
    #[allow(dead_code)]  // Reserved for future standalone IVC backend
    ivc_proofs: Vec<IvcProof>,
    
    /// Security parameters (lazy-initialized on first prove_step if None)
    params: Option<NeoParams>,
    step_spec: Option<StepSpec>,
    binding_mode: Option<AppInputBinding>,
    backend: IvcBackend,
    // Cached initial state for lazy initialization
    initial_state: Vec<F>,
    start_step: u64,
    // Store step_io for each step (needed for verification)
    step_ios: Vec<Vec<F>>,
    
    // Enable automatic diagnostic capture on constraint failures (via NEO_DIAGNOSTICS env var)
    #[cfg(feature = "prove-diagnostics")]
    #[allow(dead_code)]  // Will be used when we integrate CCS checking with diagnostics
    enable_diagnostics: bool,
}

impl FoldingSession {
    /// Create a new folding session.
    /// 
    /// # Parameters
    /// - `params`: Optional security parameters. If `None`:
    ///   - First tries `stepper.security_params()` (if implemented)
    ///   - Otherwise auto-detects from first circuit synthesis
    /// - `initial_state`: Optional initial state. If `None`, initializes to zero vector
    /// - `start_step`: Initial step counter
    /// - `binding_mode`: How to bind app inputs:
    ///   - `WitnessBound`: Use standalone IVC with in-circuit witness binding constraints
    ///   - `TranscriptOnly`: Use NIVC single-lane mode with transcript-only binding
    /// 
    /// # When to use WitnessBound vs TranscriptOnly
    /// 
    /// Use **WitnessBound** when:
    /// - Your circuit consumes app inputs from witness columns (e.g., `witness[2] = delta`)
    /// - You want belt-and-suspenders security: ensure the value that seeds ρ matches the witness
    /// 
    /// Use **TranscriptOnly** when:
    /// - Your circuit reads app inputs from public `x` directly (not from witness columns)
    /// - Your circuit doesn't consume external inputs
    /// - App inputs are metadata-only (routing, scheduling hints)
    /// 
    /// # Examples
    /// 
    /// ```ignore
    /// // Auto-detect parameters (recommended)
    /// let session = FoldingSession::new(
    ///     None,  // Auto-detect from first circuit
    ///     None,
    ///     0,
    ///     AppInputBinding::TranscriptOnly,
    /// );
    /// 
    /// // Explicit parameters (advanced)
    /// let params = NeoParams::goldilocks_for_circuit(10, 3, 2);
    /// let session = FoldingSession::new(
    ///     Some(&params),
    ///     None,
    ///     0,
    ///     AppInputBinding::TranscriptOnly,
    /// );
    /// ```
    pub fn new(
        params: Option<&NeoParams>, 
        initial_state: Option<Vec<F>>, 
        start_step: u64,
        binding_mode: AppInputBinding,
    ) -> Self {
        let backend = match binding_mode {
            AppInputBinding::WitnessBound => IvcBackend::StandaloneIvc,
            AppInputBinding::TranscriptOnly => IvcBackend::NivcSingleLane,
        };
        
        FoldingSession {
            nivc_inner: None,
            ivc_accumulator: None,
            ivc_prev_me: None,
            ivc_prev_me_wit: None,
            ivc_prev_lhs_mcs: None,
            ivc_proofs: Vec::new(),
            params: params.copied(),  // None means lazy-initialize on first prove_step
            step_spec: None,
            binding_mode: Some(binding_mode),
            backend,
            initial_state: initial_state.unwrap_or_default(),
            start_step,
            step_ios: Vec::new(),
            #[cfg(feature = "prove-diagnostics")]
            enable_diagnostics: std::env::var("NEO_DIAGNOSTICS").is_ok(),
        }
    }

    /// Current state (y_prev)
    pub fn state(&self) -> &[F] {
        match self.backend {
            IvcBackend::NivcSingleLane => {
                self.nivc_inner.as_ref().map(|s| s.acc.global_y.as_slice()).unwrap_or(&self.initial_state)
            }
            IvcBackend::StandaloneIvc => {
                self.ivc_accumulator.as_ref().map(|a| a.y_compact.as_slice()).unwrap_or(&self.initial_state)
            }
        }
    }
    
    /// Current step index
    pub fn step(&self) -> u64 {
        match self.backend {
            IvcBackend::NivcSingleLane => {
                self.nivc_inner.as_ref().map(|s| s.acc.step).unwrap_or(self.start_step)
            }
            IvcBackend::StandaloneIvc => {
                self.ivc_accumulator.as_ref().map(|a| a.step).unwrap_or(self.start_step)
            }
        }
    }

    /// Get parameters (returns None if not yet initialized)
    /// 
    /// Parameters are auto-initialized on first `prove_step()` call.
    pub fn params(&self) -> Option<&NeoParams> {
        self.params.as_ref()
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

        // 1.5) Lazy-initialize params if not set
        if self.params.is_none() {
            let params = if let Some((ell, d_sc)) = stepper.security_params() {
                // Use explicit security params from the circuit
                NeoParams::goldilocks_for_circuit(ell, d_sc, 2)
            } else {
                // Auto-detect from the synthesized circuit
                let (ell, d_sc) = NeoParams::compute_circuit_params(
                    artifacts.ccs.n,
                    artifacts.ccs.max_degree() as u32
                );
                NeoParams::goldilocks_for_circuit(ell, d_sc, 2)
            };
            self.params = Some(params);
        }

        // 1.6) Validate CCS structure requirements
        // SECURITY: ℓ = ceil(log2(n)) must be ≥ 2 for the sumcheck protocol
        // n is padded to next power of 2 (max 2), so n=3 → 4 → ℓ=2 is acceptable
        let n = artifacts.ccs.n;
        if n < 3 {
            return Err(format!(
                "CCS validation failed: n={} is too small (minimum n=3 required). \
                The sumcheck challenge length ℓ=ceil(log2(n_padded)) must be ≥ 2 for protocol security. \
                n is padded to next power-of-2 (minimum 2), so n=3→4→ℓ=2, n=2→2→ℓ=1 (too small). \
                Please ensure your circuit has at least 3 constraint rows.",
                n
            ).into());
        }

        // 1.7) Log step configuration for debugging
        #[cfg(feature = "neo-logs")]
        {
            let binding_mode = self.binding_mode.unwrap_or(AppInputBinding::WitnessBound);
            let app_inputs_len = artifacts.public_app_inputs.len();
            let digest_len = 4; // DIGEST_LEN from neo-params
            eprintln!("📋 StepSpec Configuration:");
            eprintln!("  binding_mode: {:?}", binding_mode);
            eprintln!("  y_len: {}", artifacts.spec.y_len);
            eprintln!("  app_input_indices: {:?}", artifacts.spec.app_input_indices);
            eprintln!("  public_app_inputs.len(): {}", app_inputs_len);
            eprintln!();
            eprintln!("  ⚙️  Neo will construct:");
            eprintln!("  step_x = [H(prev_acc)[0..{}] || app_inputs[0..{}]]", digest_len, app_inputs_len);
            eprintln!("  step_x_len = {} + {} = {}", digest_len, app_inputs_len, digest_len + app_inputs_len);
            eprintln!("           │    │   │");
            eprintln!("           │    │   └─ your app inputs");
            eprintln!("           │    └───── accumulator digest (mandatory)");
            eprintln!("           └────────── total");
            eprintln!();
        }

        // 2) Dispatch to appropriate backend
        match self.backend {
            IvcBackend::NivcSingleLane => self.prove_step_nivc(artifacts),
            IvcBackend::StandaloneIvc => self.prove_step_standalone_ivc(artifacts),
        }
    }

    /// Prove step using NIVC backend (TranscriptOnly mode)
    fn prove_step_nivc(
        &mut self,
        artifacts: StepArtifacts,
    ) -> Result<IvcProof, Box<dyn std::error::Error>> {
        // Lazy initialization: create the NIVC state on first step
        if self.nivc_inner.is_none() {
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
            
            let mode = AppInputBinding::TranscriptOnly;
            
            // Warn if app_input_indices are declared (they won't be enforced in TranscriptOnly mode)
            // NOTE: y_prev_indices ARE enforced even in TranscriptOnly mode (orthogonal to app input mode)
            if artifacts.spec.app_input_indices.as_ref().is_some_and(|v| !v.is_empty()) {
                eprintln!("⚠️  WARNING: FoldingSession uses TranscriptOnly mode.");
                eprintln!("    app_input_indices will NOT be enforced via in-circuit equality constraints.");
                eprintln!("    App inputs are cryptographically bound via Fiat-Shamir transcript only.");
                eprintln!("    This is safe IF your circuit reads app inputs from public `x` (not witness).");
                eprintln!("    NOTE: y_prev_indices ARE still enforced for state consistency (orthogonal).");
            }
            
            // Create single-lane NIVC program from this step's spec
            let binding = artifacts.spec.binding_spec(mode);
            let program = NivcProgram::new(vec![
                NivcStepSpec {
                    ccs: artifacts.ccs.clone(),
                    binding,
                }
            ]);
            
            let mut nivc_state = NivcState::new(self.params.unwrap(), program, y0)?;
            nivc_state.acc.step = self.start_step;
            self.nivc_inner = Some(nivc_state);
            self.step_spec = Some(artifacts.spec.clone());
        }

        // Verify that the step spec hasn't changed (IVC assumes uniform steps)
        if let Some(ref cached_spec) = self.step_spec {
            if artifacts.spec.y_len != cached_spec.y_len {
                return Err("Step spec changed: y_len mismatch".into());
            }
        }

        // Delegate to NIVC with single lane (lane index 0)
        let nivc_state = self.nivc_inner.as_mut().unwrap();
        let nivc_proof = nivc_state.step(
            0, // Single lane index
            &artifacts.public_app_inputs,
            &artifacts.witness,
        )?;

        // Store step_io for verification
        self.step_ios.push(artifacts.public_app_inputs.clone());

        // Extract the inner IVC proof to maintain API compatibility
        Ok(nivc_proof.inner)
    }

    /// Prove step using standalone IVC backend (WitnessBound mode)
    fn prove_step_standalone_ivc(
        &mut self,
        artifacts: StepArtifacts,
    ) -> Result<IvcProof, Box<dyn std::error::Error>> {
        use crate::ivc::pipeline::prover::prove_ivc_step_chained;
        use crate::shared::binding::{IndexExtractor, StepOutputExtractor};
        use crate::shared::types::IvcStepInput;

        // Lazy initialization: create initial accumulator on first step
        if self.ivc_accumulator.is_none() {
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
            
            self.ivc_accumulator = Some(Accumulator {
                c_z_digest: [0u8; 32],
                c_coords: vec![],
                y_compact: y0,
                step: self.start_step,
            });
            self.step_spec = Some(artifacts.spec.clone());
        }

        // Verify that the step spec hasn't changed (IVC assumes uniform steps)
        if let Some(ref cached_spec) = self.step_spec {
            if artifacts.spec.y_len != cached_spec.y_len {
                return Err("Step spec changed: y_len mismatch".into());
            }
        }

        let prev_acc = self.ivc_accumulator.as_ref().unwrap();
        let binding_mode = self.binding_mode.unwrap_or(AppInputBinding::WitnessBound);
        let binding = artifacts.spec.binding_spec(binding_mode);

        // Extract y_step from witness
        let extractor = IndexExtractor { indices: binding.y_step_offsets.clone() };
        let y_step = extractor.extract_y_step(&artifacts.witness);

        // Build IvcStepInput
        let step_input = IvcStepInput {
            params: &self.params.unwrap(),
            step_ccs: &artifacts.ccs,
            step_witness: &artifacts.witness,
            prev_accumulator: prev_acc,
            step: prev_acc.step,
            public_input: if artifacts.public_app_inputs.is_empty() {
                None
            } else {
                Some(&artifacts.public_app_inputs[..])
            },
            y_step: &y_step,
            binding_spec: &binding,
            app_input_binding: binding_mode,
            prev_augmented_x: self.ivc_proofs.last().map(|p| p.public_inputs.step_augmented_public_input()),
        };

        // Call lower-level IVC prover
        let (step_result, me_out, me_wit_out, lhs_next) = prove_ivc_step_chained(
            step_input,
            self.ivc_prev_me.clone(),
            self.ivc_prev_me_wit.clone(),
            self.ivc_prev_lhs_mcs.clone(),
        )?;

        // Update state for next step
        self.ivc_accumulator = Some(step_result.proof.next_accumulator.clone());
        self.ivc_prev_me = Some(me_out);
        self.ivc_prev_me_wit = Some(me_wit_out);
        self.ivc_prev_lhs_mcs = Some(lhs_next);
        
        // Store proof and step_io
        self.step_ios.push(artifacts.public_app_inputs);
        let proof = step_result.proof.clone();
        self.ivc_proofs.push(proof.clone());

        Ok(proof)
    }

    /// Finalize the session and return an IVC chain proof with metadata for verification
    pub fn finalize(self) -> (IvcChainProof, Vec<Vec<F>>) {
        match self.backend {
            IvcBackend::NivcSingleLane => {
                if let Some(inner) = self.nivc_inner {
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
                        chain_length: final_accumulator.step,  // Use final step counter for consistency with verifier
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
                        chain_length: 0,  // Zero steps taken
                    };
                    (chain, vec![])
                }
            }
            IvcBackend::StandaloneIvc => {
                if let Some(final_acc) = self.ivc_accumulator {
                    let chain = IvcChainProof {
                        steps: self.ivc_proofs,
                        final_accumulator: final_acc.clone(),
                        chain_length: final_acc.step,  // Use final step counter for consistency with verifier
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
                        chain_length: 0,
                    };
                    (chain, vec![])
                }
            }
        }
    }
}

/// Verify a complete chain using a descriptor (shape + spec) and the initial state.
/// 
/// This function automatically dispatches to the appropriate verification method based on
/// the binding mode:
/// - **TranscriptOnly**: Verifies via NIVC single-lane (transcript-based binding)
/// - **WitnessBound**: Verifies via standalone IVC (in-circuit witness binding)
/// 
/// The step_ios parameter contains the original app inputs for each step (returned from finalize()).
pub fn verify_chain_with_descriptor(
    descriptor: &StepDescriptor,
    chain: &IvcChainProof,
    initial_state: &[F],
    params: &NeoParams,
    step_ios: &[Vec<F>],
    binding_mode: AppInputBinding,
) -> Result<bool, Box<dyn std::error::Error>> {
    match binding_mode {
        AppInputBinding::TranscriptOnly => {
            verify_chain_nivc_backend(descriptor, chain, initial_state, params, step_ios)
        }
        AppInputBinding::WitnessBound => {
            verify_chain_ivc_backend(descriptor, chain, initial_state, params, step_ios)
        }
    }
}

/// Verify chain using NIVC backend (TranscriptOnly mode)
fn verify_chain_nivc_backend(
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
    let mode = AppInputBinding::TranscriptOnly;
    let binding = descriptor.spec.binding_spec(mode);
    
    let program = NivcProgram::new(vec![
        NivcStepSpec {
            ccs: descriptor.ccs.clone(),
            binding,
        }
    ]);
    
    crate::nivc::verify_nivc_chain(&program, params, &nivc_chain, initial_state)
        .map_err(|e| e.into())
}

/// Verify chain using standalone IVC backend (WitnessBound mode)
fn verify_chain_ivc_backend(
    descriptor: &StepDescriptor,
    chain: &IvcChainProof,
    _initial_state: &[F],
    params: &NeoParams,
    _step_ios: &[Vec<F>],
) -> Result<bool, Box<dyn std::error::Error>> {
    use crate::ivc::pipeline::verifier::verify_ivc_step;

    if chain.steps.is_empty() {
        return Ok(true); // Empty chain is trivially valid
    }

    let binding_mode = AppInputBinding::WitnessBound;
    let binding = descriptor.spec.binding_spec(binding_mode);

    // Verify each step in sequence
    let mut prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: chain.steps[0].public_inputs.y_prev().to_vec(),
        step: if chain.steps[0].step > 0 { chain.steps[0].step - 1 } else { 0 },
    };

    for (idx, proof) in chain.steps.iter().enumerate() {
        let prev_augmented_x = if idx > 0 {
            Some(chain.steps[idx - 1].public_inputs.step_augmented_public_input())
        } else {
            None
        };

        let valid = verify_ivc_step(
            &descriptor.ccs,
            proof,
            &prev_acc,
            &binding,
            params,
            prev_augmented_x,
        )?;

        if !valid {
            return Err(format!("IVC step {} verification failed", idx).into());
        }

        // Update prev_acc for next iteration
        prev_acc = proof.next_accumulator.clone();
    }

    // Verify final accumulator matches
    if chain.final_accumulator.c_z_digest != prev_acc.c_z_digest 
        || chain.final_accumulator.y_compact != prev_acc.y_compact 
        || chain.final_accumulator.step != prev_acc.step
    {
        return Err("Final accumulator mismatch".into());
    }

    Ok(true)
}

/// Options for IVC final proof (Stage 5)
pub struct IvcFinalizeOptions { pub embed_ivc_ev: bool }

/// Generate a succinct final SNARK for a plain IVC chain using the last step's shape.
/// Returns (lean proof, augmented CCS, final public input). Returns Ok(None) if the chain is empty.
pub fn finalize_ivc_chain_with_options(
    descriptor: &StepDescriptor,
    params: &NeoParams,
    chain: IvcChainProof,
    binding_mode: AppInputBinding,
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
    // CRITICAL: Use the same binding mode that was used during proving
    // app_input binding depends on the mode used during the session
    let binding_spec = descriptor.spec.binding_spec(binding_mode);
    
    let augmented_ccs = crate::ivc::build_augmented_ccs_linked_with_rlc(
        &descriptor.ccs,
        step_x.len(),
        &descriptor.spec.y_step_indices,
        &binding_spec.step_program_input_witness_indices,
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
