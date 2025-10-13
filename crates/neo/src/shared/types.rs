//! Shared types used by both IVC and NIVC
//!
//! These types form the contract between IVC and NIVC, ensuring they can work together
//! without duplication.

use crate::F;
use neo_ccs::CcsStructure;
use std::ops::Range;

/// IVC Accumulator - the running state that gets folded at each step
#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Digest of the running commitment coordinates (binding for ρ derivation).
    pub c_z_digest: [u8; 32],
    /// **NEW**: Full commitment coordinates (public in the IVC step CCS).
    /// These are the actual Ajtai commitment coordinates that get folded.
    pub c_coords: Vec<F>,
    /// Compact y-outputs (the "protocol-internal" y's exposed by folding).
    /// These are the Y_j(r) scalars produced by the folding pipeline.
    pub y_compact: Vec<F>,
    /// Step counter bound into the transcript (prevents replay/mixing).
    pub step: u64,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            c_z_digest: [0u8; 32],
            c_coords: vec![],
            y_compact: vec![],
            step: 0,
        }
    }
}

/// Commitment structure for full commitment binding (replaces digest-only binding)
#[derive(Clone, Debug)]
pub struct Commitment {
    /// The exact serialized commitment bytes
    pub bytes: Vec<u8>,
    /// Domain for separation (e.g., "CCS.witness", "RLC.fold")  
    pub domain: &'static str,
}

impl Commitment {
    pub fn new(bytes: Vec<u8>, domain: &'static str) -> Self {
        Self { bytes, domain }
    }
    
    /// Create from digest (compatibility with existing code)
    pub fn from_digest(digest: [u8; 32], domain: &'static str) -> Self {
        Self { bytes: digest.to_vec(), domain }
    }
}

/// Trusted specification for step circuit binding.
/// **CRITICAL**: These offsets must come from a trusted source (circuit specification),
/// NOT from the prover/proof. Trusting prover-supplied offsets defeats security.
#[derive(Clone, Debug)]
pub struct StepBindingSpec {
    /// Positions of y_step values in the step witness (for linked witness binding)
    pub y_step_offsets: Vec<usize>,
    /// Positions of step program-supplied public inputs in the step witness (binds the tail of step_x)
    pub step_program_input_witness_indices: Vec<usize>,
    /// Positions of y_prev (state input) in the step witness - must be length y_len
    pub y_prev_witness_indices: Vec<usize>,
    /// Index of the constant-1 column in the step witness (for stitching constraints)
    pub const1_witness_index: usize,
}

/// Structured public input segments with zero duplication.
///
/// This stores all public input data in a single contiguous buffer and provides
/// typed access to different segments. The layout is verified at construction
/// and ranges are guaranteed to stay valid because buffer mutations are controlled.
///
/// Layout: `[acc_digest || app_x || transport || ρ || y_prev || y_next]`
///
/// Where:
/// - `acc_digest`: H(prev_accumulator) - 4 field elements
/// - `app_x`: Optional application public inputs
/// - `transport`: NIVC metadata (lane_idx + lanes_root) - 5 elements for single-lane
/// - `ρ`: Random challenge from transcript - 1 element
/// - `y_prev`: Previous state - variable length
/// - `y_next`: Next state - same length as y_prev
#[derive(Clone, Debug)]
pub struct PublicInputSegments<F> {
    /// Contiguous buffer containing all public input data
    buffer: Vec<F>,
    
    /// Range for accumulator digest H(prev_acc)
    acc_digest_range: Range<usize>,
    
    /// Range for application public inputs (may be empty)
    app_x_range: Range<usize>,
    
    /// Range for transport metadata (NIVC envelope: lane_idx + lanes_root)
    transport_range: Range<usize>,
    
    /// Index of the ρ element
    rho_idx: usize,
    
    /// Range for y_prev state
    y_prev_range: Range<usize>,
    
    /// Range for y_next state
    y_next_range: Range<usize>,
}

impl<F: Clone + Copy> PublicInputSegments<F> {
    /// Construct a new PublicInputSegments from individual components.
    ///
    /// This is the only way to create a PublicInputSegments, ensuring that
    /// all ranges are valid and consistent with the buffer layout.
    pub fn new(
        acc_digest: Vec<F>,
        app_x: Vec<F>,
        transport: Vec<F>,
        rho: F,
        y_prev: Vec<F>,
        y_next: Vec<F>,
    ) -> Self {
        assert_eq!(y_prev.len(), y_next.len(), 
                   "y_prev and y_next must have the same length");
        
        let mut buffer = Vec::with_capacity(
            acc_digest.len() + app_x.len() + transport.len() + 1 + 2 * y_prev.len()
        );
        
        // Helper to push a segment and return its range
        let push_range = |buf: &mut Vec<F>, data: Vec<F>| -> Range<usize> {
            let start = buf.len();
            buf.extend(data);
            start..buf.len()
        };
        
        let acc_digest_range = push_range(&mut buffer, acc_digest);
        let app_x_range = push_range(&mut buffer, app_x);
        let transport_range = push_range(&mut buffer, transport);
        let rho_idx = buffer.len();
        buffer.push(rho);
        let y_prev_range = push_range(&mut buffer, y_prev);
        let y_next_range = push_range(&mut buffer, y_next);
        
        Self {
            buffer,
            acc_digest_range,
            app_x_range,
            transport_range,
            rho_idx,
            y_prev_range,
            y_next_range,
        }
    }
    
    /// Get the accumulator digest H(prev_acc)
    pub fn acc_digest(&self) -> &[F] {
        &self.buffer[self.acc_digest_range.clone()]
    }
    
    /// Get the application public inputs (may be empty)
    pub fn app_public_input_x(&self) -> &[F] {
        &self.buffer[self.app_x_range.clone()]
    }
    
    /// Get the transport metadata (NIVC envelope)
    pub fn transport(&self) -> &[F] {
        &self.buffer[self.transport_range.clone()]
    }
    
    /// Get ρ (random challenge)
    pub fn rho(&self) -> F {
        self.buffer[self.rho_idx].clone()
    }
    
    /// Get y_prev (previous state)
    pub fn y_prev(&self) -> &[F] {
        &self.buffer[self.y_prev_range.clone()]
    }
    
    /// Get y_next (next state)
    pub fn y_next(&self) -> &[F] {
        &self.buffer[self.y_next_range.clone()]
    }
    
    /// Get the wrapper public input: [acc_digest || app_x || transport]
    ///
    /// This is what goes into the step CCS as public input before augmentation.
    pub fn wrapper_public_input_x(&self) -> &[F] {
        &self.buffer[self.acc_digest_range.start..self.transport_range.end]
    }
    
    /// Get the full augmented public input: [wrapper || ρ || y_prev || y_next]
    ///
    /// This is what gets used for folding in the augmented CCS.
    pub fn step_augmented_public_input(&self) -> &[F] {
        &self.buffer // entire buffer
    }
    
    /// Get the step public input (wrapper) for legacy compatibility.
    ///
    /// This is the same as `wrapper_public_input_x()`.
    #[deprecated(note = "Use wrapper_public_input_x() for clarity")]
    pub fn step_public_input(&self) -> &[F] {
        self.wrapper_public_input_x()
    }
    
    // Test-only mutation helpers
    
    /// Replace a portion of the accumulator digest (for tampering tests).
    ///
    /// # Panics
    /// Panics if `new_values` is longer than the digest.
    #[cfg(test)]
    pub fn tamper_acc_digest(&mut self, new_values: &[F]) {
        assert!(new_values.len() <= self.acc_digest_range.len(),
                "Cannot tamper with more elements than exist in acc_digest");
        let start = self.acc_digest_range.start;
        self.buffer[start..start + new_values.len()].copy_from_slice(new_values);
    }
    
    /// Replace the ρ value (for tampering tests).
    #[cfg(test)]
    pub fn tamper_rho(&mut self, new_rho: F) {
        self.buffer[self.rho_idx] = new_rho;
    }
    
    /// Replace the entire buffer (for advanced tampering tests).
    ///
    /// # Panics
    /// Panics if the new buffer doesn't match the expected length.
    #[cfg(test)]
    pub fn replace_buffer(&mut self, new_buffer: Vec<F>) {
        assert_eq!(new_buffer.len(), self.buffer.len(),
                   "Buffer replacement must preserve length to keep ranges valid");
        self.buffer = new_buffer;
    }
    
    /// Get mutable access to a slice of the buffer (for tampering tests).
    #[cfg(test)]
    pub fn buffer_mut(&mut self) -> &mut [F] {
        &mut self.buffer
    }
    
    /// **TEST HELPER**: Get mutable access to the buffer for tampering tests.
    /// This is intentionally not cfg(test) so integration tests can use it.
    #[doc(hidden)]
    pub fn __test_tamper_buffer(&mut self) -> &mut [F] {
        &mut self.buffer
    }
    
    /// **TEST HELPER**: Replace the ρ value for tampering tests.
    /// This is intentionally not cfg(test) so integration tests can use it.
    #[doc(hidden)]
    pub fn __test_tamper_rho(&mut self, new_rho: F) {
        self.buffer[self.rho_idx] = new_rho;
    }
    
    /// **TEST HELPER**: Replace a portion of the accumulator digest for tampering tests.
    /// This is intentionally not cfg(test) so integration tests can use it.
    #[doc(hidden)]
    pub fn __test_tamper_acc_digest(&mut self, new_values: &[F]) {
        assert!(new_values.len() <= self.acc_digest_range.len(),
                "Cannot tamper with more elements than exist in acc_digest");
        let start = self.acc_digest_range.start;
        self.buffer[start..start + new_values.len()].copy_from_slice(new_values);
    }
}

/// IVC-specific proof for a single step
#[derive(Clone)]
pub struct IvcProof {
    /// The cryptographic proof for this step
    pub step_proof: crate::Proof,
    /// The accumulator after this step
    pub next_accumulator: Accumulator,
    /// Step number in the IVC chain
    pub step: u64,
    /// Optional step-specific metadata
    pub metadata: Option<Vec<u8>>,
    
    /// **NEW**: Structured public input segments (zero-duplication storage)
    ///
    /// This replaces the old separate fields:
    /// - `step_public_input` → use `public_inputs.wrapper_public_input_x()`
    /// - `step_augmented_public_input` → use `public_inputs.step_augmented_public_input()`
    /// - `step_rho` → use `public_inputs.rho()`
    /// - `step_y_prev` → use `public_inputs.y_prev()`
    /// - `step_y_next` → use `public_inputs.y_next()`
    pub public_inputs: PublicInputSegments<F>,
    
    /// Augmented input for the previous accumulator (LHS instance) used during folding
    pub prev_step_augmented_public_input: Vec<F>,
    
    /// **NEW**: The per-step commitment coordinates used in opening/lincomb
    pub c_step_coords: Vec<F>,
    /// **ARCHITECTURE**: ME instances from folding (for Stage 5 Final SNARK Layer)
    pub me_instances: Option<Vec<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>>,
    /// **ARCHITECTURE**: Digit witnesses from folding (for Stage 5 Final SNARK Layer)  
    pub digit_witnesses: Option<Vec<neo_ccs::MeWitness<F>>>,
    /// **ARCHITECTURE**: Folding proof data (for Stage 5 Final SNARK Layer)
    pub folding_proof: Option<neo_fold::FoldingProof>,
    // 🔒 REMOVED: Binding metadata no longer in proof (security vulnerability!)
    // Verifier must get these from a trusted StepBindingSpec instead
}

impl IvcProof {
    /// Legacy accessor: get step_public_input (wrapper envelope)
    #[deprecated(note = "Use public_inputs.wrapper_public_input_x() instead")]
    pub fn step_public_input(&self) -> &[F] {
        self.public_inputs.wrapper_public_input_x()
    }
    
    /// Legacy accessor: get step_augmented_public_input
    #[deprecated(note = "Use public_inputs.step_augmented_public_input() instead")]
    pub fn step_augmented_public_input(&self) -> &[F] {
        self.public_inputs.step_augmented_public_input()
    }
    
    /// Legacy accessor: get step_rho
    #[deprecated(note = "Use public_inputs.rho() instead")]
    pub fn step_rho(&self) -> F {
        self.public_inputs.rho()
    }
    
    /// Legacy accessor: get step_y_prev
    #[deprecated(note = "Use public_inputs.y_prev() instead")]
    pub fn step_y_prev(&self) -> &[F] {
        self.public_inputs.y_prev()
    }
    
    /// Legacy accessor: get step_y_next
    #[deprecated(note = "Use public_inputs.y_next() instead")]
    pub fn step_y_next(&self) -> &[F] {
        self.public_inputs.y_next()
    }
}

/// Input for a single IVC step
#[derive(Clone, Debug)]
pub struct IvcStepInput<'a> {
    /// Neo parameters for proving
    pub params: &'a crate::NeoParams,
    /// Base step CCS (the computation to be proven)
    pub step_ccs: &'a CcsStructure<F>,
    /// Witness for the step computation
    pub step_witness: &'a [F],
    /// Previous accumulator state
    pub prev_accumulator: &'a Accumulator,
    /// Current step number
    pub step: u64,
    /// Optional public input for the step
    pub public_input: Option<&'a [F]>,
    /// **REAL per-step contribution used by Nova EV**: y_next = y_prev + ρ * y_step
    /// This is the actual step output that gets folded (NOT a placeholder)
    pub y_step: &'a [F],
    /// **SECURITY**: Trusted binding specification (NOT from prover!)
    pub binding_spec: &'a StepBindingSpec,
    /// If true, app inputs in step_x are transcript-only and are NOT read from the witness.
    /// In this mode, `step_program_input_witness_indices` may be empty even when step_x has app inputs (NIVC).
    pub transcript_only_app_inputs: bool,
    /// Optional: the previous step's augmented public input [step_x || ρ || y_prev || y_next].
    /// If provided, it is propagated into the proof's `prev_step_augmented_public_input` to
    /// ensure exact chaining linkage.
    pub prev_augmented_x: Option<&'a [F]>,
}

/// Result of executing an IVC step
#[derive(Clone)]
pub struct IvcStepResult {
    /// The proof for this step
    pub proof: IvcProof,
    /// Updated computation state (for continuing the chain)
    pub next_state: Vec<F>,
}

/// IVC chain proof containing multiple steps
#[derive(Clone)]
pub struct IvcChainProof {
    /// Individual step proofs
    pub steps: Vec<IvcProof>,
    /// Final accumulator state
    pub final_accumulator: Accumulator,
    /// Total number of steps in the chain
    pub chain_length: u64,
}

/// Optional metadata for structural commitment binding
#[derive(Clone, Default)]
pub struct BindingMetadata<'a> {
    pub kv_pairs: &'a [(&'a str, u128)],
}

/// Input for a single step in an IVC chain
#[derive(Clone, Debug)]
pub struct IvcChainStepInput {
    pub witness: Vec<F>,
    pub public_input: Option<Vec<F>>,
    pub step: u64,
}

/// Configuration for building the complete Nova augmentation CCS
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    /// Length of hash inputs for in-circuit ρ derivation
    pub hash_input_len: usize,
    /// Length of compact y vector (accumulator state)
    pub y_len: usize,
    /// Ajtai public parameters (kappa, m, d)
    pub ajtai_pp: (usize, usize, usize),
    /// Number of commitment limbs/elements (typically d * kappa)
    pub commit_len: usize,
}

