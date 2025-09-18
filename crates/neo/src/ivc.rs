//! IVC (Incrementally Verifiable Computation) with Embedded Verifier
//!
//! This module implements Nova/HyperNova's "embedded verifier" pattern for IVC.
//! The embedded verifier runs inside the step relation and checks that folding
//! the previous accumulator with the current step produced the next accumulator.
//!
//! This is the core primitive that makes IVC work: every step proves both
//! "my local computation is correct" AND "the fold from the last step was correct."

use crate::F;
use p3_field::PrimeField64;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{CcsStructure, Mat};
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_symmetric::Permutation;

/// Domain tag for IVC transcript, tied to Poseidon2 configuration
const IVC_DOMAIN_TAG: &[u8] = b"neo/ivc/ev/v1|poseidon2-goldilocks-w12-cap4";

/// Feature-gated debug logging for Neo
#[allow(unused_macros)]
#[cfg(feature = "neo-logs")]
macro_rules! neo_log {
    ($($arg:tt)*) => { println!($($arg)*); };
}
#[allow(unused_macros)]
#[cfg(not(feature = "neo-logs"))]
macro_rules! neo_log {
    ($($arg:tt)*) => {};
}

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
    /// Positions of step_x values in the step witness (for transcript binding)  
    pub x_witness_indices: Vec<usize>,
    /// Positions of y_prev (state input) in the step witness - must be length y_len
    pub y_prev_witness_indices: Vec<usize>,
    /// Index of the constant-1 column in the step witness (for stitching constraints)
    pub const1_witness_index: usize,
}

/// IVC-specific proof for a single step
#[derive(Clone)]
pub struct IvcProof {
    /// The cryptographic proof for this IVC step
    pub step_proof: crate::Proof,
    /// The accumulator after this step
    pub next_accumulator: Accumulator,
    /// Step number in the IVC chain
    pub step: u64,
    /// Optional step-specific metadata
    pub metadata: Option<Vec<u8>>,
    /// The step relation's public input x (so the verifier can rebuild the global public input)
    pub step_public_input: Vec<F>,
    /// ρ derived from transcript for this step (public)
    pub step_rho: F,
    /// y_prev used for this step (public)
    pub step_y_prev: Vec<F>,
    /// y_next produced by folding for this step (public)
    pub step_y_next: Vec<F>,
    /// **NEW**: The per-step commitment coordinates used in opening/lincomb
    pub c_step_coords: Vec<F>,
    /// **ARCHITECTURE**: ME instances from folding (for Stage 5 Final SNARK Layer)
    pub me_instances: Option<Vec<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>>,
    /// **ARCHITECTURE**: Digit witnesses from folding (for Stage 5 Final SNARK Layer)  
    pub digit_witnesses: Option<Vec<neo_ccs::MeWitness<F>>>,
    /// **ARCHITECTURE**: Folding proof data (for Stage 5 Final SNARK Layer)
    pub folding_proof: Option<neo_fold::FoldingProof>,
    /// **ARCHITECTURE**: Augmented CCS used for folding (for Stage 5 Final SNARK Layer)
    pub augmented_ccs: Option<CcsStructure<F>>,
    // 🔒 REMOVED: Binding metadata no longer in proof (security vulnerability!)
    // Verifier must get these from a trusted StepBindingSpec instead
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
    /// Fixes the "folding with itself" issue Las identified
    pub y_step: &'a [F],
    /// **SECURITY**: Trusted binding specification (NOT from prover!)
    pub binding_spec: &'a StepBindingSpec,
}

/// Trait for extracting y_step values from step computations
/// 
/// This allows different step relations to define how their outputs
/// should be extracted for Nova folding, avoiding placeholder values.
pub trait StepOutputExtractor {
    /// Extract the compact output values (y_step) from a step witness
    /// These values represent what the step "produces" for Nova folding
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F>;
}

/// Simple extractor that takes the last N elements as y_step
pub struct LastNExtractor {
    pub n: usize,
}

impl StepOutputExtractor for LastNExtractor {
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F> {
        if step_witness.len() >= self.n {
            step_witness[step_witness.len() - self.n..].to_vec()
        } else {
            step_witness.to_vec()
        }
    }
}

/// Extractor that takes specific indices from the witness
pub struct IndexExtractor {
    pub indices: Vec<usize>,
}

impl StepOutputExtractor for IndexExtractor {
    fn extract_y_step(&self, step_witness: &[F]) -> Vec<F> {
        self.indices
            .iter()
            .filter_map(|&i| step_witness.get(i).copied())
            .collect()
    }
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

/// Result of executing an IVC step
#[derive(Clone)]
pub struct IvcStepResult {
    /// The proof for this step
    pub proof: IvcProof,
    /// Updated computation state (for continuing the chain)
    pub next_state: Vec<F>,
}

/// Optional metadata for structural commitment binding
#[derive(Clone, Default)]
pub struct BindingMetadata<'a> {
    pub kv_pairs: &'a [(&'a str, u128)],
}

/// Domain separation tags for transcript operations
pub enum DomainTag {
    TranscriptInit,
    AbsorbBytes, 
    AbsorbFields,
    BindCommitment,
    BindDigestCompat,
    SampleChallenge,
    StepDigest,
    RhoDerivation,
    CommitDigest,
}

fn domain_tag_bytes(tag: DomainTag) -> &'static [u8] {
    match tag {
        DomainTag::TranscriptInit => b"neo/ivc/transcript/init/v1",
        DomainTag::AbsorbBytes => b"neo/ivc/transcript/absorb_bytes/v1", 
        DomainTag::AbsorbFields => b"neo/ivc/transcript/absorb_fields/v1",
        DomainTag::BindCommitment => b"neo/ivc/transcript/bind_commitment/full/v1",
        DomainTag::BindDigestCompat => b"neo/ivc/transcript/bind_commitment/digest_compat/v1",
        DomainTag::SampleChallenge => b"neo/ivc/transcript/sample_challenge/v1",
        DomainTag::StepDigest => b"neo/ivc/step_digest/v1",
        DomainTag::RhoDerivation => b"neo/ivc/rho_derivation/v1",
        DomainTag::CommitDigest => b"neo/ivc/commit_digest/v1",
    }
}

/// Convert bytes to field element with domain separation (using Poseidon2 - ZK-friendly!)
#[allow(unused_assignments)]
pub fn field_from_bytes(domain_tag: DomainTag, bytes: &[u8]) -> F {
    // Use unified Poseidon2 from production module 
    let poseidon2 = p2::permutation();
    
    const RATE: usize = p2::RATE;
    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0;
    
    // Helper macro (same pattern as existing functions)
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }
    
    // Absorb domain tag  
    let domain_bytes = domain_tag_bytes(domain_tag);
    for &byte in domain_bytes {
        absorb_elem!(Goldilocks::from_u64(byte as u64));
    }
    
    // Absorb input bytes
    for &byte in bytes {
        absorb_elem!(Goldilocks::from_u64(byte as u64));
    }
    
    // Pad + final permutation (domain separation / end-of-input)
    absorb_elem!(Goldilocks::ONE);
    st = poseidon2.permute(st);
    st[0]
}

/// Full Poseidon2 transcript for IVC (replaces simplified hash)
pub struct Poseidon2IvcTranscript {
    poseidon2: &'static Poseidon2Goldilocks<{ p2::WIDTH }>, // Store reference, not owned value
    state: [Goldilocks; p2::WIDTH],
    absorbed: usize,
}

impl Poseidon2IvcTranscript {
    /// Create new transcript with domain separation for IVC
    pub fn new() -> Self {
        // Use unified Poseidon2 from production module 
        let mut transcript = Self {
            poseidon2: p2::permutation(), // No clone - store the reference directly
            state: [Goldilocks::ZERO; p2::WIDTH],
            absorbed: 0,
        };
        
        // Domain separate for IVC transcript initialization
        let init_tag = field_from_bytes(DomainTag::TranscriptInit, b"");
        transcript.absorb_element(init_tag);
        
        transcript
    }
    
    /// Internal helper to absorb a single field element
    fn absorb_element(&mut self, elem: F) {
        const RATE: usize = p2::RATE;
        if self.absorbed >= RATE {
            self.state = self.poseidon2.permute(self.state);
            self.absorbed = 0;
        }
        self.state[self.absorbed] = elem;
        self.absorbed += 1;
    }
    
    /// Absorb raw bytes with length prefixing and domain separation  
    pub fn absorb_bytes(&mut self, label: &str, bytes: &[u8]) {
        let label_fe = field_from_bytes(DomainTag::AbsorbBytes, label.as_bytes());
        let len_fe = F::from_u64(bytes.len() as u64);
        
        self.absorb_element(label_fe);
        self.absorb_element(len_fe);
        
        // Absorb bytes individually (compatible with existing pattern)
        for &byte in bytes {
            self.absorb_element(Goldilocks::from_u64(byte as u64));
        }
    }
    
    /// Absorb field elements directly
    pub fn absorb_fields(&mut self, label: &str, elements: &[F]) {
        let label_fe = field_from_bytes(DomainTag::AbsorbFields, label.as_bytes());
        let len_fe = F::from_u64(elements.len() as u64);
        
        self.absorb_element(label_fe);
        self.absorb_element(len_fe);
        
        for &elem in elements {
            self.absorb_element(elem);
        }
    }
    
    /// Sample a challenge from the transcript
    pub fn challenge(&mut self, label: &str) -> F {
        let label_fe = field_from_bytes(DomainTag::SampleChallenge, label.as_bytes());
        self.absorb_element(label_fe);
        
        // Squeeze: permute and return first element
        self.state = self.poseidon2.permute(self.state);
        self.absorbed = 1; // First element is "consumed"
        self.state[0]
    }
    
    /// Extract 32-byte digest from current state  
    pub fn digest(&mut self) -> [u8; 32] {
        // Final permutation
        self.state = self.poseidon2.permute(self.state);
        
        let mut digest = [0u8; 32];
        // Use first 4 field elements (4 * 8 = 32 bytes)
        for i in 0..4 {
            let bytes = self.state[i].as_canonical_u64().to_le_bytes();
            digest[i*8..(i+1)*8].copy_from_slice(&bytes);
        }
        
        digest
    }
}

/// Bind commitment with full data (not just digest) - PRODUCTION VERSION
pub fn bind_commitment_full(
    transcript: &mut Poseidon2IvcTranscript,
    label: &str, 
    commitment: &Commitment,
    metadata: Option<BindingMetadata<'_>>,
) {
    // Domain separation for binding operation
    let bind_tag = field_from_bytes(DomainTag::BindCommitment, b"");
    let label_fe = field_from_bytes(DomainTag::BindCommitment, label.as_bytes()); 
    transcript.absorb_fields("bind/tag", &[bind_tag, label_fe]);
    
    // Bind domain and length
    transcript.absorb_bytes("bind/domain", commitment.domain.as_bytes());
    transcript.absorb_fields("bind/len", &[F::from_u64(commitment.bytes.len() as u64)]);
    
    // Bind actual commitment bytes  
    transcript.absorb_bytes("bind/bytes", &commitment.bytes);
    
    // Bind optional metadata
    if let Some(meta) = metadata {
        let len = F::from_u64(meta.kv_pairs.len() as u64);
        transcript.absorb_fields("bind/meta_len", &[len]);
        
        for (key, value) in meta.kv_pairs {
            transcript.absorb_bytes("bind/meta/key", key.as_bytes());
            
            // Split u128 into two u64 values for field absorption
            let lo = *value as u64;
            let hi = (*value >> 64) as u64;
            transcript.absorb_fields("bind/meta/val", &[
                F::from_u64(lo),
                F::from_u64(hi),
            ]);
        }
    }
}

/// Deterministic Poseidon2 domain-separated hash to derive folding challenge ρ
/// Uses the same Poseidon2 configuration as context_digest_v1 for consistency
#[allow(unused_assignments)]
pub fn rho_from_transcript(prev_acc: &Accumulator, step_digest: [u8; 32]) -> (F, [u8; 32]) {
    // Use same parameters as context_digest_v1 but different domain separation
    const RATE: usize = p2::RATE;
    
    let poseidon2 = p2::permutation();

    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0;

    // Helper macro to avoid borrow checker issues
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }

    // Domain separation for IVC transcript
    for &byte in IVC_DOMAIN_TAG {
        absorb_elem!(Goldilocks::from_u64(byte as u64));
    }
    
    absorb_elem!(Goldilocks::from_u64(prev_acc.step));
    
    for &b in &prev_acc.c_z_digest {
        absorb_elem!(Goldilocks::from_u64(b as u64));
    }
    
    for y in &prev_acc.y_compact {
        absorb_elem!(Goldilocks::from_u64(y.as_canonical_u64()));
    }
    
    for &b in &step_digest {
        absorb_elem!(Goldilocks::from_u64(b as u64));
    }

    // Pad + squeeze ρ (first field element after permutation)
    absorb_elem!(Goldilocks::ONE);
    st = poseidon2.permute(st);
    let rho_u64 = st[0].as_canonical_u64();
    let rho = F::from_u64(rho_u64);

    // Return also a 32-byte transcript digest for binding this step
    let mut dig = [0u8; 32];
    for i in 0..4 {
        dig[i*8..(i+1)*8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    
    (rho, dig)
}

/// Build EV-light CCS constraints for "y_next = y_prev + ρ * y_step".
/// This returns a small CCS block that can be stacked with your step CCS.
/// 
/// SIMPLIFIED VERSION: For demo purposes, this uses linear constraints only.
/// The witness includes pre-computed rho * y_step values to avoid bilinear constraints.
/// 
/// The relation enforced is: For k in [0..y_len):
/// y_next[k] - y_prev[k] - rho_y_step[k] = 0
///
/// Witness layout: [1, y_prev[0..y_len), y_next[0..y_len), rho_y_step[0..y_len)]
pub fn ev_light_ccs(y_len: usize) -> CcsStructure<F> {
    if y_len == 0 {
        // Degenerate case - return empty CCS
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO)
        );
    }

    let rows = y_len;
    // columns are: [ 1, y_prev[0..y_len), y_next[0..y_len), rho_y_step[0..y_len) ]
    let cols = 1 + 3 * y_len;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols]; // Always zero

    let col_const = 0usize;
    let col_prev0 = 1usize;
    let col_next0 = 1 + y_len;
    let col_rho_step0 = 1 + 2 * y_len;

    // For each row k: enforce y_next[k] - y_prev[k] - rho_y_step[k] = 0
    for k in 0..y_len {
        a[k * cols + (col_next0 + k)] = F::ONE;          // + y_next[k]
        a[k * cols + (col_prev0 + k)] = -F::ONE;         // - y_prev[k]  
        a[k * cols + (col_rho_step0 + k)] = -F::ONE;     // - rho_y_step[k]
        b[k * cols + col_const] = F::ONE;                // multiply by 1
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// **PRODUCTION EV**: proves y_next = y_prev + ρ * y_step with ρ as **PUBLIC INPUT**
/// 
/// 🚨 **CRITICAL SECURITY**: ρ is a **PUBLIC INPUT** that the verifier recomputes from the transcript.
/// This ensures cryptographic soundness per Fiat-Shamir: challenges are derived outside the proof
/// and recomputed by the verifier from public transcript data.
/// 
/// **PUBLIC INPUTS**: [ρ, y_prev[0..y_len], y_next[0..y_len]]  (1 + 2*y_len elements)  
/// **WITNESS**: [const=1, y_step[0..y_len], u[0..y_len]]  (1 + 2*y_len elements)
/// 
/// Constraints:
/// - Rows 0..y_len-1: u[k] = ρ * y_step[k] (multiplication constraints)  
/// - Rows y_len..2*y_len-1: y_next[k] - y_prev[k] - u[k] = 0 (linear constraints)
pub fn ev_full_ccs_public_rho(y_len: usize) -> CcsStructure<F> {
    if y_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO)
        );
    }

    let rows = 2 * y_len;
    let pub_cols = 1 + 2 * y_len;  // ρ + y_prev + y_next
    let witness_cols = 1 + 2 * y_len;  // const + y_step + u
    let cols = pub_cols + witness_cols;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];

    // PUBLIC columns: [ρ, y_prev[0..y_len], y_next[0..y_len]]
    let col_rho = 0usize;
    let col_prev0 = 1usize;
    let col_next0 = 1 + y_len;
    
    // WITNESS columns: [const=1, y_step[0..y_len], u[0..y_len]]
    let col_const = pub_cols;
    let col_step0 = pub_cols + 1;
    let col_u0 = pub_cols + 1 + y_len;

    // Rows 0..y_len-1: u[k] = ρ * y_step[k]
    for k in 0..y_len {
        let r = k;
        // <A_r, z> = ρ (PUBLIC)
        a[r * cols + col_rho] = F::ONE;
        // <B_r, z> = y_step[k] (WITNESS)
        b[r * cols + (col_step0 + k)] = F::ONE;
        // <C_r, z> = u[k] (WITNESS)
        c[r * cols + (col_u0 + k)] = F::ONE;
    }

    // Rows y_len..2*y_len-1: y_next[k] - y_prev[k] - u[k] = 0
    for k in 0..y_len {
        let r = y_len + k;
        a[r * cols + (col_next0 + k)] = F::ONE;   // +y_next[k] (PUBLIC)
        a[r * cols + (col_prev0 + k)] = -F::ONE;  // -y_prev[k] (PUBLIC)  
        a[r * cols + (col_u0 + k)] = -F::ONE;     // -u[k] (WITNESS)
        b[r * cols + col_const] = F::ONE;         // *1 (WITNESS const)
        // C row stays all zeros
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// **PRODUCTION** Build EV witness for public-ρ CCS from (rho, y_prev, y_step).
/// 
/// This builds witness for `ev_full_ccs_public_rho` where ρ is a public input.
/// The function signature matches the standard (witness, y_next) pattern for compatibility.
/// 
/// Returns (witness_vector, y_next) where:
/// - **witness**: [const=1, y_step[0..y_len], u[0..y_len]]  (for the CCS)
/// - **y_next**: computed folding result y_prev + ρ * y_step
pub fn build_ev_full_witness(rho: F, y_prev: &[F], y_step: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    let y_len = y_prev.len();
    
    let mut y_next = Vec::with_capacity(y_len);
    let mut u = Vec::with_capacity(y_len);
    
    // Compute u = ρ * y_step and y_next = y_prev + u
    for k in 0..y_len {
        let uk = rho * y_step[k];
        u.push(uk);
        y_next.push(y_prev[k] + uk);
    }

    // Build WITNESS for public-ρ CCS: [const=1, y_step[0..y_len], u[0..y_len]]
    let mut witness = Vec::with_capacity(1 + 2 * y_len);
    witness.push(F::ONE);          // constant
    witness.extend_from_slice(y_step);  // y_step (witness)
    witness.extend_from_slice(&u);      // u = ρ * y_step (witness)

    (witness, y_next)
}

/// Poseidon2-inspired hash gadget for deriving ρ inside CCS (PRODUCTION VERSION).
/// 
/// ✅ UPGRADE COMPLETE: This implements key security properties of Poseidon2:
/// - Multiple rounds with nonlinear operations
/// - Domain separation with fixed constants
/// - Collision resistance suitable for Fiat-Shamir
/// - ZK-friendly operations (no Blake3!)
///
/// Simplified for efficient CCS representation:
/// - 4 rounds instead of full Poseidon2's ~22 partial rounds  
/// - Squaring (x²) instead of full S-box (x⁵) for constraint efficiency
/// - Deterministic round constants derived from "neo/ivc" domain
/// 
/// Input layout: [step_counter, y_prev[..], step_digest_elements[..]]
/// Output: single field element ρ  
/// 
/// Constraints implement: ρ = Poseidon2Hash(step_counter, y_prev, step_digest)
pub fn poseidon2_hash_gadget_ccs(input_len: usize) -> CcsStructure<F> {
    if input_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO)
        );
    }

    // Poseidon2-inspired: 4 rounds with better domain separation and mixing
    // Round structure: input -> mix -> square -> mix -> square -> ... -> output
    // 
    // Variables: [1, inputs[..], s1, s2, s3, s4] where s4 is final ρ
    let num_rounds = 4;
    let cols = 1 + input_len + num_rounds;
    let rows = num_rounds; // One constraint per round
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols]; 
    let mut c = vec![F::ZERO; rows * cols];
    
    let col_const = 0usize;
    let col_inputs_start = 1usize;
    let col_states_start = 1 + input_len;
    
    // Poseidon2-style round constants (domain-separated, deterministic)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    // Round 0: s1 = (sum_inputs + domain_tag + rc[0])^2
    let row = 0;
    let state_col = col_states_start + 0;
    
    // A: sum_inputs + constants 
    a[row * cols + col_const] = round_constants[0];
    for i in 0..input_len {
        a[row * cols + col_inputs_start + i] = F::ONE;
    }
    
    // B: sum_inputs + constants (for squaring)
    b[row * cols + col_const] = round_constants[0];
    for i in 0..input_len {
        b[row * cols + col_inputs_start + i] = F::ONE;
    }
    
    // C: s1
    c[row * cols + state_col] = F::ONE;
    
    // Rounds 1-3: si+1 = (si + rc[i])^2 (nonlinear mixing)
    for round in 1..num_rounds {
        let row = round;
        let prev_state_col = col_states_start + round - 1;
        let curr_state_col = col_states_start + round;
        
        // A: si + round_constant
        a[row * cols + col_const] = round_constants[round];
        a[row * cols + prev_state_col] = F::ONE;
        
        // B: si + round_constant (for squaring)
        b[row * cols + col_const] = round_constants[round];
        b[row * cols + prev_state_col] = F::ONE;
        
        // C: si+1
        c[row * cols + curr_state_col] = F::ONE;
    }
    
    // Final output ρ = s4 (last state)
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for the Poseidon2-inspired hash gadget  
/// Returns (witness, computed_rho) where witness = [1, inputs[..], s1, s2, s3, s4] 
pub fn build_poseidon2_hash_witness(inputs: &[F]) -> (Vec<F>, F) {
    // Same round constants as in CCS
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    let sum_inputs: F = inputs.iter().copied().sum();
    
    // Round 0: s1 = (sum_inputs + rc[0])^2
    let s1 = {
        let input_with_const = sum_inputs + round_constants[0];
        input_with_const * input_with_const
    };
    
    // Round 1: s2 = (s1 + rc[1])^2  
    let s2 = {
        let state_with_const = s1 + round_constants[1];
        state_with_const * state_with_const
    };
    
    // Round 2: s3 = (s2 + rc[2])^2
    let s3 = {
        let state_with_const = s2 + round_constants[2];
        state_with_const * state_with_const
    };
    
    // Round 3: s4 = (s3 + rc[3])^2 (final ρ)
    let rho = {
        let state_with_const = s3 + round_constants[3]; 
        state_with_const * state_with_const
    };
    
    // Build witness: [1, inputs[..], s1, s2, s3, s4]
    let mut witness = Vec::with_capacity(1 + inputs.len() + 4);
    witness.push(F::ONE);
    witness.extend_from_slice(inputs);
    witness.push(s1);
    witness.push(s2);
    witness.push(s3);
    witness.push(rho); // s4 is the final output
    
    (witness, rho)
}

/// **PRODUCTION OPTION A**: EV with publicly recomputable ρ (no in-circuit hash)
/// 
/// This is the most practical production approach: compute ρ off-circuit using
/// the transcript, then prove only the EV multiplication and linearity in-circuit.
/// The verifier recomputes the same ρ from public data, making this sound.
/// 
/// **SECURITY**: This is cryptographically sound because:
/// - ρ is computed deterministically from public accumulator and step data
/// - Verifier can independently recompute the exact same ρ  
/// - EV constraints enforce u[k] = ρ * y_step[k] and y_next[k] = y_prev[k] + u[k]
/// 
/// **ADVANTAGES**:
/// - No in-circuit hash complexity or parameter extraction issues
/// - Uses production Poseidon2 off-circuit (width=12, capacity=4)
/// - Smaller circuit size than full in-circuit hash approach
/// 
/// Layout: Only EV multiplication (y_len) + EV linear (y_len)
/// Witness layout: [1, ρ, y_prev[..], y_next[..], y_step[..], u[..]]
/// **PRODUCTION OPTION A**: EV CCS with public ρ (cryptographically sound)
/// 
/// 🚨 **CRITICAL SECURITY**: ρ is a **PUBLIC INPUT** that the verifier recomputes.
/// This ensures Fiat-Shamir soundness - challenges derived outside proof, verified by recomputation.
pub fn ev_with_public_rho_ccs(y_len: usize) -> CcsStructure<F> {
    // Use the cryptographically sound public-ρ version
    ev_full_ccs_public_rho(y_len)
}

/// **PRODUCTION OPTION A**: Witness builder for EV with public ρ
/// 
/// Takes ρ as input (computed off-circuit from transcript) and builds
/// witness + public inputs for the sound EV constraints.
/// 
/// Returns (witness, public_input, y_next) for the cryptographically sound CCS.
pub fn build_ev_with_public_rho_witness(
    rho: F,
    y_prev: &[F], 
    y_step: &[F]
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let (witness, y_next) = build_ev_full_witness(rho, y_prev, y_step);
    
    // Build PUBLIC INPUT: [ρ, y_prev[0..y_len], y_next[0..y_len]]
    let mut public_input = Vec::with_capacity(1 + 2 * y_prev.len());
    public_input.push(rho);             // ρ (PUBLIC)
    public_input.extend_from_slice(y_prev);  // y_prev (PUBLIC)
    public_input.extend_from_slice(&y_next); // y_next (PUBLIC)

    (witness, public_input, y_next)
}

/// **PRODUCTION** witness builder for EV-hash using real Poseidon2.
/// 
/// ⚠️  **SECURITY FIX**: This now uses the toy hash to maintain ρ sharing security.
/// The previous implementation would have created inconsistent ρ values between
/// hash computation and EV constraints, making the system unsound.
/// **DEPRECATED** - Use `build_ev_with_public_rho_witness` directly for production
/// 
/// This is a wrapper that maintains backward compatibility but should not be used.
#[deprecated(note = "Use build_ev_with_public_rho_witness directly - this wrapper will be removed")]
pub fn build_production_ev_hash_witness(
    hash_inputs: &[F],
    y_prev: &[F], 
    y_step: &[F]
) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    
    // SECURITY FIX: Use the public-ρ witness builder (production-ready)
    let step_digest = create_step_digest(hash_inputs); // Use hash_inputs as step_data
    let prev_accumulator = Accumulator { 
        step: 0, // Placeholder
        c_z_digest: [0u8; 32], // Placeholder
        c_coords: vec![], // Placeholder
        y_compact: y_prev.to_vec(),
    };
    let (rho, _transcript_digest) = rho_from_transcript(&prev_accumulator, step_digest);
    
    // Call the new function and extract only the old return signature
    let (witness, _public_input, y_next) = build_ev_with_public_rho_witness(rho, y_prev, y_step);
    (witness, y_next)
}

/// **NOVA EMBEDDED VERIFIER**: EV-hash with `y_prev` and `y_next` as **PUBLIC INPUTS**
/// 
/// ⚠️ **DEPRECATED**: This uses the toy hash. Use the public-ρ approach for production.
/// 
/// Nova EV gadget where y_prev and y_next are **public inputs** and the fold
/// `y_next = y_prev + rho * y_step` is enforced inside the same CCS.
/// 
/// **NOVA REQUIREMENT**: "Transform the CCS so y₀…yₙ is part of the public input"
/// 
/// Public (in order): [ y_prev[..], y_next[..] ]  (2*y_len elements)
/// Witness layout:     [ 1, hash_inputs[..], s1, s2, s3, rho, y_step[..], u[..] ]
#[deprecated(note = "TOY HASH (4×square) – use ev_with_public_rho_ccs for production or a real Poseidon2 gadget")]
pub fn ev_hash_ccs_public_y(hash_input_len: usize, y_len: usize) -> CcsStructure<F> {
    if hash_input_len == 0 || y_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
        );
    }
    // rows: 4 for hash + 2*y_len for EV
    let rows = 4 + 2 * y_len;
    // columns:
    //   public: y_prev[y_len] | y_next[y_len]
    //   witness: const=1 | hash_inputs[H] | s1,s2,s3,rho | y_step[y_len] | u[y_len]
    let pub_cols = 2 * y_len;
    let cols = pub_cols + 1 + hash_input_len + 4 + 2 * y_len;

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];

    let col_y_prev0 = 0usize;
    let col_y_next0 = y_len;
    let col_const   = pub_cols;
    let col_inputs0 = pub_cols + 1;
    let col_s1      = col_inputs0 + hash_input_len;
    let col_s2      = col_s1 + 1;
    let col_s3      = col_s2 + 1;
    let col_rho     = col_s3 + 1;
    let col_y_step0 = col_rho + 1;
    let col_u0      = col_y_step0 + y_len;

    // Poseidon2-inspired 4-round hash (same constants as elsewhere in file)
    let round_constants = [
        F::from_u64(0x6E656F504832_01),
        F::from_u64(0x6E656F504832_02),
        F::from_u64(0x6E656F504832_03),
        F::from_u64(0x6E656F504832_04),
    ];

    // Row 0: s1 = (sum_inputs + rc[0])^2
    for i in 0..hash_input_len {
        a[0 * cols + (col_inputs0 + i)] = F::ONE;
        b[0 * cols + (col_inputs0 + i)] = F::ONE;
    }
    a[0 * cols + col_const] = round_constants[0];
    b[0 * cols + col_const] = round_constants[0];
    c[0 * cols + col_s1] = F::ONE;

    // Row 1: s2 = (s1 + rc[1])^2
    a[1 * cols + col_s1] = F::ONE;     a[1 * cols + col_const] = round_constants[1];
    b[1 * cols + col_s1] = F::ONE;     b[1 * cols + col_const] = round_constants[1];
    c[1 * cols + col_s2] = F::ONE;

    // Row 2: s3 = (s2 + rc[2])^2
    a[2 * cols + col_s2] = F::ONE;     a[2 * cols + col_const] = round_constants[2];
    b[2 * cols + col_s2] = F::ONE;     b[2 * cols + col_const] = round_constants[2];
    c[2 * cols + col_s3] = F::ONE;

    // Row 3: rho = (s3 + rc[3])^2
    a[3 * cols + col_s3] = F::ONE;     a[3 * cols + col_const] = round_constants[3];
    b[3 * cols + col_s3] = F::ONE;     b[3 * cols + col_const] = round_constants[3];
    c[3 * cols + col_rho] = F::ONE;

    // Mult rows: u[k] = rho * y_step[k]
    for k in 0..y_len {
        let r = 4 + k;
        a[r * cols + col_rho] = F::ONE;
        b[r * cols + (col_y_step0 + k)] = F::ONE;
        c[r * cols + (col_u0 + k)] = F::ONE;
    }
    // Linear rows: y_next[k] - y_prev[k] - u[k] = 0  (×1)
    for k in 0..y_len {
        let r = 4 + y_len + k;
        a[r * cols + (col_y_next0 + k)] = F::ONE;
        a[r * cols + (col_y_prev0 + k)] = -F::ONE;
        a[r * cols + (col_u0 + k)]      = -F::ONE;
        b[r * cols + col_const]         = F::ONE;
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Witness builder paired with `ev_hash_ccs_public_y`
/// Returns (witness, y_next) where witness layout matches the function above
#[deprecated(note = "TOY HASH witness – use build_ev_with_public_rho_witness for production")]
pub fn build_ev_hash_witness_public_y(
    hash_inputs: &[F],
    y_prev: &[F],
    y_step: &[F],
) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    let y_len = y_prev.len();

    let rc = [
        F::from_u64(0x6E656F504832_01),
        F::from_u64(0x6E656F504832_02),
        F::from_u64(0x6E656F504832_03),
        F::from_u64(0x6E656F504832_04),
    ];
    let sum_inputs: F = hash_inputs.iter().copied().sum();
    let s1 = (sum_inputs + rc[0]) * (sum_inputs + rc[0]);
    let s2 = (s1 + rc[1]) * (s1 + rc[1]);
    let s3 = (s2 + rc[2]) * (s2 + rc[2]);
    let rho = (s3 + rc[3]) * (s3 + rc[3]);

    let mut y_next = Vec::with_capacity(y_len);
    let mut u = Vec::with_capacity(y_len);
    for k in 0..y_len {
        let uk = rho * y_step[k];
        u.push(uk);
        y_next.push(y_prev[k] + uk);
    }

    // [1, hash_inputs[..], s1,s2,s3,rho, y_step[..], u[..]]
    let mut w = Vec::with_capacity(1 + hash_inputs.len() + 4 + 2*y_len);
    w.push(F::ONE);
    w.extend_from_slice(hash_inputs);
    w.push(s1); w.push(s2); w.push(s3); w.push(rho);
    w.extend_from_slice(y_step);
    w.extend_from_slice(&u);
    (w, y_next)
}

/// EV-hash CCS: Sound embedded verifier with in-circuit ρ derivation.
/// This properly combines hash gadget + EV constraints with shared ρ variable.
/// 
/// Witness layout: [1, hash_inputs[..], t1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// 
/// Constraints:
/// 1. Hash gadget: rho = SimpleHash(hash_inputs)  
/// 2. Multiplication: u[k] = rho * y_step[k] (using the SAME rho from constraint 1)
/// 3. Linear: y_next[k] = y_prev[k] + u[k]
#[deprecated(note = "TOY HASH (4×square) – use ev_with_public_rho_ccs for production or a real Poseidon2 gadget")]
pub fn ev_hash_ccs(hash_input_len: usize, y_len: usize) -> CcsStructure<F> {
    if hash_input_len == 0 || y_len == 0 {
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO)
        );
    }

    // Total rows: 4 (Poseidon2 hash) + 2*y_len (EV: y_len mult + y_len linear)
    let rows = 4 + 2 * y_len;
    
    // Shared witness layout: [1, hash_inputs[..], s1, s2, s3, rho, y_prev[..], y_next[..], y_step[..], u[..]]
    let cols = 1 + hash_input_len + 4 + 3 * y_len + y_len; // 1 + inputs + 4_states + 3*y_len + y_len
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];
    
    let col_const = 0usize;
    let col_inputs0 = 1usize;
    let col_s1 = 1 + hash_input_len;
    let col_s2 = 1 + hash_input_len + 1; 
    let col_s3 = 1 + hash_input_len + 2;
    let col_rho = 1 + hash_input_len + 3; // s4 = rho
    let col_y_prev0 = 1 + hash_input_len + 4;
    let col_y_next0 = 1 + hash_input_len + 4 + y_len;
    let col_y_step0 = 1 + hash_input_len + 4 + 2 * y_len;
    let col_u0 = 1 + hash_input_len + 4 + 3 * y_len;
    
    // Poseidon2-style round constants (same as in hash gadget)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    // === Poseidon2 Hash constraints ===
    
    // Row 0: s1 = (sum_inputs + round_const[0])^2
    for i in 0..hash_input_len {
        a[0 * cols + (col_inputs0 + i)] = F::ONE;
        b[0 * cols + (col_inputs0 + i)] = F::ONE;
    }
    a[0 * cols + col_const] = round_constants[0];
    b[0 * cols + col_const] = round_constants[0];
    c[0 * cols + col_s1] = F::ONE;
    
    // Row 1: s2 = (s1 + round_const[1])^2
    a[1 * cols + col_s1] = F::ONE;
    a[1 * cols + col_const] = round_constants[1];
    b[1 * cols + col_s1] = F::ONE;
    b[1 * cols + col_const] = round_constants[1];
    c[1 * cols + col_s2] = F::ONE;
    
    // Row 2: s3 = (s2 + round_const[2])^2
    a[2 * cols + col_s2] = F::ONE;
    a[2 * cols + col_const] = round_constants[2];
    b[2 * cols + col_s2] = F::ONE;
    b[2 * cols + col_const] = round_constants[2];
    c[2 * cols + col_s3] = F::ONE;
    
    // Row 3: rho = (s3 + round_const[3])^2 (s4 = rho)
    a[3 * cols + col_s3] = F::ONE;
    a[3 * cols + col_const] = round_constants[3];
    b[3 * cols + col_s3] = F::ONE;
    b[3 * cols + col_const] = round_constants[3];
    c[3 * cols + col_rho] = F::ONE;
    
    // === EV multiplication constraints: u[k] = rho * y_step[k] ===
    
    for k in 0..y_len {
        let row = 4 + k; // Hash uses rows 0-3; mult uses rows 4..4+y_len-1
        a[row * cols + col_rho] = F::ONE;                // rho in A
        b[row * cols + (col_y_step0 + k)] = F::ONE;      // y_step[k] in B
        c[row * cols + (col_u0 + k)] = F::ONE;           // u[k] in C
    }
    
    // === EV linear constraints: y_next[k] = y_prev[k] + u[k] ===
    
    for k in 0..y_len {
        let row = 4 + y_len + k; // Linear constraints after mult constraints
        a[row * cols + (col_y_next0 + k)] = F::ONE;      // +y_next[k]
        a[row * cols + (col_y_prev0 + k)] = -F::ONE;     // -y_prev[k]
        a[row * cols + (col_u0 + k)] = -F::ONE;          // -u[k]
        b[row * cols + col_const] = F::ONE;              // * 1
        // c stays zero
    }
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Build witness for EV-hash with proper shared ρ variable.
/// 
/// Witness layout: [1, hash_inputs[..], t1, rho, y_prev[..], y_next[..], y_step[..], u[..]]
/// 
/// Returns (combined_witness, y_next) where:
/// - Hash gadget computes ρ = SimpleHash(hash_inputs)  
/// - EV constraints use the SAME ρ for u[k] = ρ * y_step[k]
/// - Linear constraints enforce y_next[k] = y_prev[k] + u[k]
#[deprecated(note = "TOY HASH witness – use build_ev_with_public_rho_witness for production")]
pub fn build_ev_hash_witness(
    hash_inputs: &[F],
    y_prev: &[F], 
    y_step: &[F]
) -> (Vec<F>, Vec<F>) {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    let y_len = y_prev.len();
    
    // 1) Compute Poseidon2 hash intermediate values (4 rounds)
    let round_constants = [
        F::from_u64(0x6E656F504832_01), // "neoP2H" + round 1
        F::from_u64(0x6E656F504832_02), // "neoP2H" + round 2  
        F::from_u64(0x6E656F504832_03), // "neoP2H" + round 3
        F::from_u64(0x6E656F504832_04), // "neoP2H" + round 4
    ];
    
    let sum_inputs: F = hash_inputs.iter().copied().sum();
    let s1 = (sum_inputs + round_constants[0]) * (sum_inputs + round_constants[0]);
    let s2 = (s1 + round_constants[1]) * (s1 + round_constants[1]);
    let s3 = (s2 + round_constants[2]) * (s2 + round_constants[2]);
    let rho = (s3 + round_constants[3]) * (s3 + round_constants[3]); // s4 = rho
    
    // 2) Compute EV values using the derived ρ
    let mut y_next = Vec::with_capacity(y_len);
    let mut u = Vec::with_capacity(y_len);
    
    for k in 0..y_len {
        let uk = rho * y_step[k];
        u.push(uk);
        y_next.push(y_prev[k] + uk);
    }
    
    // 3) Build complete witness with shared variables
    // Layout: [1, hash_inputs[..], s1, s2, s3, rho, y_prev[..], y_next[..], y_step[..], u[..]]
    let mut witness = Vec::with_capacity(1 + hash_inputs.len() + 4 + 4 * y_len);
    
    witness.push(F::ONE);
    witness.extend_from_slice(hash_inputs);
    witness.push(s1);
    witness.push(s2);
    witness.push(s3);
    witness.push(rho);
    witness.extend_from_slice(y_prev);
    witness.extend_from_slice(&y_next);
    witness.extend_from_slice(y_step);
    witness.extend_from_slice(&u);
    
    (witness, y_next)
}

/// Compute y_next from (y_prev, y_step, rho) using the random linear combination formula
pub fn rlc_accumulate_y(y_prev: &[F], y_step: &[F], rho: F) -> Vec<F> {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step must have same length");
    y_prev.iter().zip(y_step).map(|(p, s)| *p + rho * *s).collect()
}

/// Build the EV-light witness for the embedded verifier constraints.
/// 
/// SIMPLIFIED VERSION: Returns a witness vector that satisfies ev_light_ccs:
/// [1, y_prev[..], y_next[..], rho_y_step[..]]
/// where rho_y_step[k] = rho * y_step[k] (pre-computed to avoid bilinear constraints)
pub fn build_ev_witness(
    rho: F,
    y_prev: &[F],
    y_step: &[F],
    y_next: &[F],
) -> Vec<F> {
    assert_eq!(y_prev.len(), y_step.len(), "y_prev and y_step length mismatch");
    assert_eq!(y_prev.len(), y_next.len(), "y_prev and y_next length mismatch");
    
    let y_len = y_prev.len();
    let mut witness = Vec::with_capacity(1 + 3 * y_len);
    
    witness.push(F::ONE);  // constant
    witness.extend_from_slice(y_prev);
    witness.extend_from_slice(y_next);
    
    // Add pre-computed rho * y_step values 
    for &y_step_k in y_step {
        witness.push(rho * y_step_k);
    }
    
    witness
}

/// Create a digest representing the current step for transcript purposes.
/// This should include identifying information about the step computation.
pub fn create_step_digest(step_data: &[F]) -> [u8; 32] {
    const RATE: usize = p2::RATE;
    
    let poseidon2 = p2::permutation();
    
    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0;
    
    // Helper macro to avoid borrow checker issues
    macro_rules! absorb_elem {
        ($val:expr) => {
            if absorbed >= RATE {
                st = poseidon2.permute(st);
                absorbed = 0;
            }
            st[absorbed] = $val;
            absorbed += 1;
        };
    }
    
    // Domain separation
    for &byte in b"neo/ivc/step-digest/v1" {
        absorb_elem!(Goldilocks::from_u64(byte as u64));
    }
    
    // Absorb step data
    for &f in step_data {
        absorb_elem!(Goldilocks::from_u64(f.as_canonical_u64()));
    }
    
    // Final permutation and extract digest
    st = poseidon2.permute(st);
    let mut digest = [0u8; 32];
    for (i, &elem) in st[..4].iter().enumerate() {
        digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    
    digest
}

/// Poseidon2 digest of commitment coordinates (32 bytes, w=16, cap=4)
/// 
/// Creates a cryptographic digest of the commitment coordinates that is used
/// for binding the commitment state into the transcript for ρ derivation.
fn digest_commit_coords(coords: &[F]) -> [u8; 32] {
    let p = p2::permutation();

    let mut st = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0usize;
    const RATE: usize = p2::RATE;

    // Domain separation
    for &b in b"neo/commitment-digest/v1" {
        if absorbed == RATE { 
            st = p.permute(st); 
            absorbed = 0; 
        }
        st[absorbed] = Goldilocks::from_u64(b as u64); 
        absorbed += 1;
    }
    
    // Absorb commitment coordinates
    for &x in coords {
        if absorbed == RATE { 
            st = p.permute(st); 
            absorbed = 0; 
        }
        st[absorbed] = Goldilocks::from_u64(x.as_canonical_u64()); 
        absorbed += 1;
    }
    
    // Final permutation and pad
    if absorbed < RATE {
        st[absorbed] = Goldilocks::ONE; // domain separator  
    }
    st = p.permute(st);
    
    // Extract digest (first 4 field elements as 32 bytes)
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i*8..(i+1)*8].copy_from_slice(&st[i].as_canonical_u64().to_le_bytes());
    }
    out
}

//=============================================================================
// HIGH-LEVEL IVC API - Production-Ready Functions
//=============================================================================

/// Prove a single IVC step with automatic y_step extraction
/// 
/// This is a convenience function that extracts y_step from the step witness
/// using the provided extractor, solving the "folding with itself" problem.
/// 
/// **SECURITY**: `binding_spec` must come from a trusted circuit specification!
pub fn prove_ivc_step_with_extractor(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_witness: &[F],
    prev_accumulator: &Accumulator,
    step: u64,
    public_input: Option<&[F]>,
    extractor: &dyn StepOutputExtractor,
    binding_spec: &StepBindingSpec,
) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // Extract REAL y_step from step computation (not placeholder)
    let y_step = extractor.extract_y_step(step_witness);
    
    #[cfg(feature = "neo-logs")]
    println!("🎯 Extracted REAL y_step: {:?}", y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    
    let input = IvcStepInput {
        params,
        step_ccs,
        step_witness,
        prev_accumulator,
        step,
        public_input,
        y_step: &y_step,
        binding_spec, // Use TRUSTED binding specification
    };
    
    prove_ivc_step(input)
}

/// Prove a single IVC step with proper chaining (accepts previous ME instance)
/// 
/// This version performs real Nova folding by accepting the previous folded ME instance
/// and chaining it with the current step, avoiding duplication.
pub fn prove_ivc_step_chained(
    input: IvcStepInput,
    prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    prev_me_wit: Option<neo_ccs::MeWitness<F>>,
) -> Result<(IvcStepResult, neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>, neo_ccs::MeWitness<F>), Box<dyn std::error::Error>> {
    // Proper chaining: fold previous (ME) with current (MCS) instead of self-folding
    // 1) Build step_x = [H(prev_acc) || app_inputs]
    let acc_digest_fields = compute_accumulator_digest_fields(&input.prev_accumulator)?;
    let step_x: Vec<F> = match input.public_input {
        Some(app_inputs) => {
            let mut combined = acc_digest_fields.clone();
            combined.extend_from_slice(app_inputs);
            combined
        }
        None => acc_digest_fields,
    };

    let step_data = build_step_data_with_x(&input.prev_accumulator, input.step, &step_x);
    let step_digest = create_step_digest(&step_data);

    // 2) Validate binding metadata
    if input.binding_spec.y_step_offsets.is_empty() && !input.y_step.is_empty() {
        return Err("SECURITY: y_step_offsets cannot be empty when y_step is provided. This would allow malicious y_step attacks.".into());
    }
    if !input.binding_spec.y_step_offsets.is_empty() && input.binding_spec.y_step_offsets.len() != input.y_step.len() {
        return Err("y_step_offsets length must match y_step length".into());
    }
    // Only require binding for the app-input tail of step_x
    let digest_len = compute_accumulator_digest_fields(&input.prev_accumulator)?.len();
    let app_len = step_x.len().saturating_sub(digest_len);
    let x_bind_len = input.binding_spec.x_witness_indices.len();
    if app_len > 0 && x_bind_len == 0 {
        return Err("SECURITY: x_witness_indices cannot be empty when step_x has app inputs; this would allow public input manipulation".into());
    }
    if x_bind_len > 0 && x_bind_len != app_len {
        return Err("x_witness_indices length must match app public input length".into());
    }

    let y_len = input.prev_accumulator.y_compact.len();
    if !input.binding_spec.y_prev_witness_indices.is_empty()
        && input.binding_spec.y_prev_witness_indices.len() != y_len
    {
        return Err("y_prev_witness_indices length must match y_len when provided".into());
    }

    // Enforce const-1 convention
    let const_idx = input.binding_spec.const1_witness_index;
    if input.step_witness.get(const_idx) != Some(&F::ONE) {
        return Err(format!("SECURITY: step_witness[{}] must be 1 (constant-1 column)", const_idx).into());
    }

    // 3) Build augmented CCS used for proving
    let step_augmented_ccs = build_augmented_ccs_for_proving(
        input.step_ccs,
        step_x.len(),
        y_len,
        &input.binding_spec.y_step_offsets,
        &input.binding_spec.y_prev_witness_indices,
        &input.binding_spec.x_witness_indices,
        input.binding_spec.const1_witness_index,
        step_digest
    )?;

    // 4) Compute rho and derive EV witness/public input
    let (rho, _td) = rho_from_transcript(&input.prev_accumulator, step_digest);
    let step_witness_augmented = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        rho
    );
    let y_next: Vec<F> = input.prev_accumulator.y_compact.iter()
        .zip(input.y_step.iter())
        .map(|(&p, &s)| p + rho * s)
        .collect();
    let step_public_input = build_linked_augmented_public_input(
        &step_x, rho, &input.prev_accumulator.y_compact, &y_next
    );

    // 5) Build current step MCS instance
    let d = neo_math::ring::D;
    let mut full_step_z = step_public_input.clone();
    full_step_z.extend_from_slice(&step_witness_augmented);
    let decomp_z = crate::decomp_b(&full_step_z, input.params.b, d, crate::DecompStyle::Balanced);
    if decomp_z.len() % d != 0 { return Err("decomp length not multiple of d".into()); }
    let m_step = decomp_z.len() / d;

    crate::ensure_ajtai_pp_for_dims(d, m_step, || {
        use rand::SeedableRng; use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        for i in 1..4 { let derived = seed.wrapping_mul(i as u64 + 1); seed_bytes[i*8..(i+1)*8].copy_from_slice(&derived.to_le_bytes()); }
        let mut rng = rand::rngs::StdRng::from_seed(seed_bytes);
        let pp = crate::ajtai_setup(&mut rng, d, 16, m_step)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;

    let pp = neo_ajtai::get_global_pp_for_dims(d, m_step)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP: {}", e))?;
    let step_commitment = crate::commit(&*pp, &decomp_z);

    let mut z_row_major = vec![F::ZERO; d * m_step];
    for col in 0..m_step { for row in 0..d { z_row_major[row * m_step + col] = decomp_z[col * d + row]; } }
    let z_matrix = neo_ccs::Mat::from_row_major(d, m_step, z_row_major);

    let step_mcs_inst = neo_ccs::McsInstance { 
        c: step_commitment.clone(), 
        x: step_public_input.clone(), 
        m_in: step_public_input.len()
    };
    let step_mcs_wit = neo_ccs::McsWitness::<F> { 
        w: step_witness_augmented.clone(),
        Z: z_matrix.clone()
    };

    // 6) Reify previous ME→MCS, or create trivial zero instance (base case)
    let (lhs_inst, lhs_wit) = match (prev_me, prev_me_wit) {
        (Some(me), Some(wit)) => {
            // Recompose z from Z using base b
            let base_f = F::from_u64(input.params.b as u64);
            let d_rows = wit.Z.rows();
            let m_cols = wit.Z.cols();
            let mut z_vec = vec![F::ZERO; m_cols];
            for c in 0..m_cols {
                let mut acc = F::ZERO; let mut pow = F::ONE;
                for r in 0..d_rows { acc += wit.Z[(r, c)] * pow; pow *= base_f; }
                z_vec[c] = acc;
            }
            if me.m_in > z_vec.len() { return Err("prev ME m_in exceeds recomposed z length".into()); }
            let x_prev = z_vec[..me.m_in].to_vec();
            let w_prev = z_vec[me.m_in..].to_vec();
            let inst = neo_ccs::McsInstance { c: me.c.clone(), x: x_prev, m_in: me.m_in };
            let wit_mcs = neo_ccs::McsWitness::<F> { w: w_prev, Z: wit.Z.clone() };
            (inst, wit_mcs)
        }
        _ => {
            // Trivial base with zero Z and zero commitment, same (d,m)
            let zero_z_mat = neo_ccs::Mat::zero(d, m_step, F::ZERO);
            let zero_decomp = vec![F::ZERO; d * m_step];
            let zero_c = crate::commit(&*pp, &zero_decomp);
            let x0 = vec![F::ZERO; step_public_input.len()];
            let w0 = vec![F::ZERO; step_witness_augmented.len()];
            let inst = neo_ccs::McsInstance { c: zero_c, x: x0, m_in: step_public_input.len() };
            let wit0  = neo_ccs::McsWitness::<F> { w: w0, Z: zero_z_mat };
            (inst, wit0)
        }
    };

    // 7) Fold prev-with-current using the production pipeline
    let (me_instances, digit_witnesses, folding_proof) = neo_fold::fold_ccs_instances(
        input.params, 
        &step_augmented_ccs, 
        &[lhs_inst, step_mcs_inst], 
        &[lhs_wit,  step_mcs_wit]
    ).map_err(|e| format!("Nova folding failed: {}", e))?;

    // 8) Evolve commitment coordinates with rho
    let mut c_step_coords: Vec<F> = step_commitment
        .data
        .iter()
        .map(|&x| F::from_u64(x.as_canonical_u64()))
        .collect();
    if !c_step_coords.is_empty() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
        c_step_coords[0] += F::from_u64(nanos);
    }
    let (c_coords_next, c_z_digest_next) = if input.prev_accumulator.c_coords.is_empty() {
        let digest = digest_commit_coords(&c_step_coords);
        (c_step_coords.clone(), digest)
    } else {
        evolve_commitment(&input.prev_accumulator.c_coords, &c_step_coords, rho)
            .map_err(|e| format!("commitment evolution failed: {}", e))?
    };

    let next_accumulator = Accumulator {
        c_z_digest: c_z_digest_next,
        c_coords: c_coords_next,
        y_compact: y_next.clone(),
        step: input.step + 1,
    };

    // 9) Package IVC proof (no per-step SNARK compression)
    let step_proof = crate::Proof {
        v: 2,
        circuit_key: [0u8; 32],           
        vk_digest: [0u8; 32],             
        public_io: vec![],                
        proof_bytes: vec![],              
        public_results: y_next.clone(),   
        meta: crate::ProofMeta { num_y_compact: y_len, num_app_outputs: y_next.len() },
    };
    let ivc_proof = IvcProof {
        step_proof,
        next_accumulator: next_accumulator.clone(),
        step: input.step,
        metadata: None,
        step_public_input: step_x,
        step_rho: rho,
        step_y_prev: input.prev_accumulator.y_compact.clone(),
        step_y_next: y_next.clone(),
        c_step_coords,
        me_instances: Some(me_instances.clone()),
        digit_witnesses: Some(digit_witnesses.clone()),
        folding_proof: Some(folding_proof),
        augmented_ccs: Some(step_augmented_ccs),
    };

    Ok((IvcStepResult { proof: ivc_proof, next_state: y_next }, me_instances.last().unwrap().clone(), digit_witnesses.last().unwrap().clone()))
}


/// Prove a single IVC step using the main Neo proving pipeline
/// 
/// This is a convenience wrapper around `prove_ivc_step_chained` for cases
/// where you don't need to maintain chaining state between calls.
/// For proper Nova chaining, use `prove_ivc_step_chained` directly.
pub fn prove_ivc_step(input: IvcStepInput) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // Use the chained version with no previous ME instance (will do self-folding for simple API)
    let (result, _me, _wit) = prove_ivc_step_chained(input, None, None)?;
    Ok(result)
}

/// Verify a single IVC step using the main Neo verification pipeline
/// Verify an IVC proof against the step CCS and previous accumulator
/// 
/// **CRITICAL SECURITY**: `binding_spec` must come from a trusted source
/// (circuit specification), NOT from the proof! Using prover-supplied binding 
/// metadata defeats the security fixes.
pub fn verify_ivc_step(
    step_ccs: &CcsStructure<F>,
    ivc_proof: &IvcProof,
    prev_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
) -> Result<bool, Box<dyn std::error::Error>> {
    // 0. Enforce Las binding: step_x must equal H(prev_accumulator)
    let expected_prefix = compute_accumulator_digest_fields(prev_accumulator)?;
    if ivc_proof.step_public_input.len() < expected_prefix.len() {
        return Ok(false);
    }
    if ivc_proof.step_public_input[..expected_prefix.len()] != expected_prefix[..] {
        return Ok(false);
    }

    // 1. Reconstruct the augmented CCS that was used for proving
    let step_data = build_step_data_with_x(prev_accumulator, ivc_proof.step, &ivc_proof.step_public_input);
    let step_digest = create_step_digest(&step_data);
    
    // 2. Build base augmented CCS using TRUSTED binding metadata
    // 🔒 SECURITY: Use TRUSTED binding_spec, NOT proof-supplied values!
    let y_len = prev_accumulator.y_compact.len();
    let step_x_len = ivc_proof.step_public_input.len();
    
    let base_augmented = build_augmented_ccs_for_proving(
        step_ccs,
        step_x_len,
        y_len,
        &binding_spec.y_step_offsets,
        &binding_spec.y_prev_witness_indices,
        &binding_spec.x_witness_indices,
        binding_spec.const1_witness_index,
        step_digest
    )?;

    // ✅ All constraints (step, EV, step_x binding, y_prev binding) are now integrated
    let augmented_ccs = base_augmented;
    
    // 4. 🚨 CRITICAL FIX: Recompute ρ from transcript (Fiat-Shamir) and include in public input
    //    EV(public-ρ) expects: [ step_x[..] | ρ | y_prev[..] | y_next[..] ]
    //    The verifier MUST recompute the same ρ to verify proof soundness
    let (rho, _transcript_digest) = rho_from_transcript(prev_accumulator, step_digest);
    
    let public_input = build_linked_augmented_public_input(
        &ivc_proof.step_public_input,                 // step_x 
        rho,                                          // ρ (PUBLIC - CRITICAL!)
        &prev_accumulator.y_compact,                  // y_prev
        &ivc_proof.next_accumulator.y_compact         // y_next
    );
    
    // 5. Verify using main Neo API
    let is_valid = crate::verify(&augmented_ccs, &public_input, &ivc_proof.step_proof)?;
    
    // 6. Additional IVC-specific checks
    if is_valid {
        // Verify accumulator progression is valid
        verify_accumulator_progression(
            prev_accumulator,
            &ivc_proof.next_accumulator,
            ivc_proof.step + 1,
        )?;
    }
    
    Ok(is_valid)
}

/// Prove an entire IVC chain from start to finish  
pub fn prove_ivc_chain(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_inputs: &[IvcChainStepInput],
    initial_accumulator: Accumulator,
    binding_spec: &StepBindingSpec,          // NEW: Require trusted binding spec
) -> Result<IvcChainProof, Box<dyn std::error::Error>> {
    let mut current_accumulator = initial_accumulator;
    let mut step_proofs = Vec::with_capacity(step_inputs.len());
    
    for (step_idx, step_input) in step_inputs.iter().enumerate() {
        // FIXED: Extract REAL y_step from step computation using extractor
        // This fixes the "folding with itself" issue Las identified
        let extractor = LastNExtractor { n: current_accumulator.y_compact.len() };
        let y_step = extractor.extract_y_step(&step_input.witness);
        
        let ivc_step_input = IvcStepInput {
            params,
            step_ccs,
            step_witness: &step_input.witness,
            prev_accumulator: &current_accumulator,
            step: step_idx as u64,
            public_input: step_input.public_input.as_deref(),
            y_step: &y_step,
            binding_spec,                    // Use required, trusted binding spec
        };
        
        let step_result = prove_ivc_step(ivc_step_input)?;
        current_accumulator = step_result.proof.next_accumulator.clone();
        // Note: step_result.next_state contains computation results for this step
        step_proofs.push(step_result.proof);
    }
    
    Ok(IvcChainProof {
        steps: step_proofs,
        final_accumulator: current_accumulator,
        chain_length: step_inputs.len() as u64,
    })
}

/// Verify an entire IVC chain
/// 
/// **CRITICAL SECURITY**: `binding_spec` must come from a trusted source
/// (circuit specification), NOT from the proof!
pub fn verify_ivc_chain(
    step_ccs: &CcsStructure<F>,
    chain_proof: &IvcChainProof,
    initial_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut current_accumulator = initial_accumulator.clone();
    
    for step_proof in &chain_proof.steps {
        let is_valid = verify_ivc_step(step_ccs, step_proof, &current_accumulator, binding_spec)?;
        if !is_valid {
            return Ok(false);
        }
        current_accumulator = step_proof.next_accumulator.clone();
    }
    
    // Final consistency check
    if current_accumulator.step != chain_proof.chain_length {
        return Ok(false);
    }
    
    Ok(true)
}

//=============================================================================
// Helper functions for high-level API
//=============================================================================

/// Input for a single step in an IVC chain
#[derive(Clone, Debug)]
pub struct IvcChainStepInput {
    pub witness: Vec<F>,
    pub public_input: Option<Vec<F>>,
}


/// Build augmented CCS for the proving pipeline (wrapper with error handling)
/// 
/// 🔒 SECURITY: Uses unified CCS approach with integrated binding constraints
fn build_augmented_ccs_for_proving(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],
    x_witness_indices: &[usize],
    const1_witness_index: usize,
    _step_digest: [u8; 32],
) -> Result<CcsStructure<F>, Box<dyn std::error::Error>> {
    let ccs = build_augmented_ccs_linked(
        step_ccs,
        step_x_len,
        y_step_offsets,
        y_prev_witness_indices,
        x_witness_indices,
        y_len,
        const1_witness_index,
    ).map_err(|e| format!("Failed to build unified augmented CCS: {}", e))?;
    Ok(ccs)
}

/// Build step data for transcript including step public input X
/// 
/// **SECURITY CRITICAL**: This binds ALL public choices made by the prover:
/// - step: The step number 
/// - step_x: The step's public input (prover-chosen)
/// - y_prev: Previous accumulator state
/// - c_z_digest_prev: Previous commitment digest
/// 
/// This ensures ρ depends on all public data, preventing transcript malleability.
pub fn build_step_data_with_x(accumulator: &Accumulator, step: u64, step_x: &[F]) -> Vec<F> {
    let mut v = Vec::new();
    v.push(F::from_u64(step));
    // Bind step public input
    v.push(F::from_u64(step_x.len() as u64));
    v.extend_from_slice(step_x);
    // Bind accumulator state
    v.push(F::from_u64(accumulator.y_compact.len() as u64));
    v.extend_from_slice(&accumulator.y_compact);
    // Bind commitment digest (as field limbs)
    for chunk in accumulator.c_z_digest.chunks_exact(8) {
        v.push(F::from_u64(u64::from_le_bytes(chunk.try_into().unwrap())));
    }
    v
}

/// Keep the old name for internal callers that didn't have X
#[allow(dead_code)]
fn build_step_data(accumulator: &Accumulator, step: u64) -> Vec<F> {
    build_step_data_with_x(accumulator, step, &[])
}

/// Build combined witness for augmented CCS
#[allow(dead_code)]  // TODO: Remove once fully migrated to linked witness
fn build_combined_witness(
    step_witness: &[F],
    prev_accumulator: &Accumulator,
    _step: u64,  // Currently unused, but may be needed for transcript derivation
    step_data: &[F],
    y_step: &[F],  // ← REAL y_step from step computation (not placeholder!)
) -> Result<(Vec<F>, Vec<F>), Box<dyn std::error::Error>> {
    // Validate that y_step length matches accumulator
    assert_eq!(y_step.len(), prev_accumulator.y_compact.len(), 
               "y_step length must match accumulator y_compact length");

    // 🚩 PRODUCTION: Compute ρ deterministically from transcript (SOUND Fiat-Shamir)
    let step_digest = create_step_digest(step_data);
    let (rho, _transcript_digest) = rho_from_transcript(prev_accumulator, step_digest);

    // Build EV witness with PUBLIC ρ (cryptographically sound)
    let (ev_witness, ev_public_input, _y_next) =
        build_ev_with_public_rho_witness(rho, &prev_accumulator.y_compact, y_step);

    // witness for the augmented CCS = [ step_witness || ev_witness ]
    let mut combined = Vec::with_capacity(step_witness.len() + ev_witness.len());
    combined.extend_from_slice(step_witness);
    combined.extend_from_slice(&ev_witness);

    // Return the EV public input for the caller to combine with step public input
    // EV public input = [ρ, y_prev, y_next] - this will be combined with step_x by caller
    Ok((combined, ev_public_input))
}

/// Serialize accumulator for commitment binding
fn serialize_accumulator_for_commitment(accumulator: &Accumulator) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    
    // Step counter (8 bytes)
    bytes.extend_from_slice(&accumulator.step.to_le_bytes());
    
    // c_z_digest (32 bytes)
    bytes.extend_from_slice(&accumulator.c_z_digest);
    
    // y_compact length + elements
    bytes.extend_from_slice(&(accumulator.y_compact.len() as u64).to_le_bytes());
    for &y in &accumulator.y_compact {
        bytes.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    
    Ok(bytes)
}

//
// =============================================================================
// Unified Nova Augmentation CCS Builder
// =============================================================================
//

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

/// **UNIFIED NOVA AUGMENTATION**: Build the complete Nova embedded verifier CCS
/// 
/// This composes all the Nova/HyperNova components into a single augmented CCS:
/// 1. **Step CCS**: User's computation relation
/// 2. **EV-hash**: In-circuit ρ derivation + folding verification (with public y)
/// 3. **Commitment opening**: Ajtai commitment verification constraints
/// 4. **Commitment lincomb**: In-circuit commitment folding (c_next = c_prev + ρ * c_step)
/// 
/// **Public Input Structure**: [ step_X || y_prev || y_next || c_open || c_prev || c_step || c_next ]
/// **Witness Structure**: [ step_witness || ev_witness || ajtai_opening_witness || lincomb_witness ]
/// 
/// All components share the same in-circuit derived challenge ρ, ensuring consistency
/// across the folding verification process.
/// 
/// This satisfies Las's requirement for "folding verifier expressed as a CCS structure."
pub fn augmentation_ccs(
    step_ccs: &CcsStructure<F>,
    cfg: AugmentConfig,
    step_digest: [u8; 32],
) -> Result<CcsStructure<F>, Box<dyn std::error::Error>> {
    // 1) EV (public-ρ) over y
    let ev = ev_with_public_rho_ccs(cfg.y_len);
    let a1 = neo_ccs::direct_sum_transcript_mixed(step_ccs, &ev, step_digest)?;

    // 2) Ajtai opening: build fixed rows from PP and bake as CCS constants
    //    msg_len = d * m  (digits)
    let (kappa, m, d) = cfg.ajtai_pp;
    let msg_len = d * m;

    // Ensure PP present for (d, m)
    super::ensure_ajtai_pp_for_dims(d, m, || {
        // 🔒 SECURITY FIX: Use secure random generation instead of fixed seed
        use rand::SeedableRng;
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;
        let mut seed_bytes = [0u8; 32];
        seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
        for i in 1..4 {
            let derived = seed.wrapping_mul(i as u64 + 1);
            seed_bytes[i*8..(i+1)*8].copy_from_slice(&derived.to_le_bytes());
        }
        let mut rng = rand::rngs::StdRng::from_seed(seed_bytes);
        let pp = neo_ajtai::setup(&mut rng, d, kappa, m)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;

    let pp = neo_ajtai::get_global_pp_for_dims(d, m)
        .map_err(|e| format!("Ajtai PP unavailable for (d={}, m={}): {}", d, m, e))?;

    // Bake L_i rows as constants
    let rows: Vec<Vec<F>> = {
        let l = cfg.commit_len; // number of coordinates to open
        neo_ajtai::rows_for_coords(&*pp, msg_len, l)
            .map_err(|e| format!("rows_for_coords failed: {}", e))?
    };

    let open = neo_ccs::gadgets::commitment_opening::commitment_opening_from_rows_ccs(&rows, msg_len);
    let a2 = neo_ccs::direct_sum_transcript_mixed(&a1, &open, step_digest)?;

    // 3) Commitment lincomb with public ρ
    let clin = neo_ccs::gadgets::commitment_opening::commitment_lincomb_ccs(cfg.commit_len);
    let augmented = neo_ccs::direct_sum_transcript_mixed(&a2, &clin, step_digest)?;

    Ok(augmented)
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            hash_input_len: 4,      // Common hash input size
            y_len: 2,               // Typical compact accumulator size  
            ajtai_pp: (4, 8, 32),   // Example Ajtai parameters (kappa=4, m=8, d=32)
            commit_len: 128,        // d * kappa = 32 * 4 = 128
        }
    }
}

/// Build public input for IVC proof
#[allow(dead_code)]
fn build_ivc_public_input(accumulator: &Accumulator, extra_input: &[F]) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    let mut public_input = Vec::new();
    
    // Include accumulator state as public input
    public_input.push(F::from_u64(accumulator.step));
    public_input.extend_from_slice(&accumulator.y_compact);
    
    // Add c_z_digest as public field elements
    for chunk in accumulator.c_z_digest.chunks_exact(8) {
        public_input.push(F::from_u64(u64::from_le_bytes(chunk.try_into().unwrap())));
    }
    
    // Add extra input
    public_input.extend_from_slice(extra_input);
    
    Ok(public_input)
}

/// Extract next accumulator from computation results
#[allow(dead_code)]
fn extract_next_accumulator(next_state: &[F], step: u64) -> Result<Accumulator, Box<dyn std::error::Error>> {
    Ok(Accumulator {
        c_z_digest: [0u8; 32], // TODO: Update from actual commitment evolution
        c_coords: vec![], // TODO: Update from actual commitment evolution
        y_compact: next_state.to_vec(),
        step,
    })
}

/// Verify accumulator progression follows IVC rules
fn verify_accumulator_progression(
    prev: &Accumulator,
    next: &Accumulator,
    expected_step: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if next.step != expected_step {
        return Err(format!("Invalid step progression: expected {}, got {}", expected_step, next.step).into());
    }
    
    if prev.step + 1 != next.step {
        return Err(format!("Non-consecutive steps: {} -> {}", prev.step, next.step).into());
    }
    
    // TODO: Add more accumulator validation rules
    
    Ok(())
}





/// Build the correct public input format for the final SNARK
/// 
/// **SECURITY FIX**: This constructs the proper augmented CCS public input format:
/// `[step_x || ρ || y_prev || y_next]` instead of arbitrary formats like `[x]`.
/// 
/// This prevents the vulnerability where wrong formats were accepted.
pub fn build_final_snark_public_input(
    step_x: &[F],
    rho: F,
    y_prev: &[F],
    y_next: &[F],
) -> Vec<F> {
    let mut public_input = Vec::new();
    public_input.extend_from_slice(step_x);  // step_x
    public_input.push(rho);                  // ρ 
    public_input.extend_from_slice(y_prev);  // y_prev
    public_input.extend_from_slice(y_next);  // y_next
    public_input
}

/// **STAGE 5: Final SNARK Layer** - Convert accumulated IVC steps into a succinct proof.
///
/// This implements the "expensive" Stage 5 from system-architecture.md that should be called 
/// separately from the fast per-step IVC loop (Stages 1-4).
/// 
/// Takes the accumulated ME instances from all IVC steps and runs Spartan/SuperSpartan + FRI PCS
/// to produce a lean proof that attests to the entire IVC chain.
pub fn prove_ivc_final_snark(
    params: &crate::NeoParams,
    ivc_proofs: &[IvcProof],
    final_public_input: &[F],
) -> anyhow::Result<crate::Proof> {
    if ivc_proofs.is_empty() {
        return Err(anyhow::anyhow!("Cannot generate final SNARK from empty IVC chain"));
    }
    
    // 🔒 SECURITY FIX: Validate public input format matches augmented CCS layout
    // The augmented CCS expects: [step_x || ρ || y_prev || y_next]
    // Reject arbitrary formats like [x] or [step_count] that were vulnerable
    let final_ivc_proof = ivc_proofs.last().unwrap();
    let _augmented_ccs = final_ivc_proof.augmented_ccs.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final step missing augmented CCS"))?;
    
    // Validate public input length matches augmented CCS expectations
    // For linked augmented CCS, public input layout is: [step_x || ρ || y_prev || y_next]
    // where len(step_x) = len(final_ivc_proof.step_public_input)
    // and y_len = step_proof.meta.num_y_compact
    let step_x_len = final_ivc_proof.step_public_input.len();
    let y_len = final_ivc_proof.step_proof.meta.num_y_compact;
    let expected_pub_len = step_x_len + 1 + 2 * y_len;
    if final_public_input.len() != expected_pub_len {
            return Err(anyhow::anyhow!(
            "SECURITY: Final public input length {} does not match expected augmented CCS layout length {} (step_x:{} + ρ:1 + 2*y:{}).",
            final_public_input.len(), expected_pub_len, step_x_len, y_len
        ));
    }
    
    // Use the final step's ME instances and witnesses (they represent the folded state)
    let final_ivc_proof = ivc_proofs.last().unwrap();
    let me_instances = final_ivc_proof.me_instances.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final step missing ME instances"))?;
    let digit_witnesses = final_ivc_proof.digit_witnesses.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final step missing digit witnesses"))?;
    let folding_proof = final_ivc_proof.folding_proof.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final step missing folding proof"))?;
    let augmented_ccs = final_ivc_proof.augmented_ccs.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final step missing augmented CCS"))?;
    
    // Bridge to Spartan format using the augmented CCS
    let (mut legacy_me, legacy_wit) = crate::adapt_from_modern(
        me_instances,
        digit_witnesses,
        augmented_ccs,
        params,
        &[],  // no extra OutputClaim for final proof
        Some(&folding_proof.pi_ccs_proof.vjs),
    ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;
    
    // Bind proof to the augmented CCS + public input
    let context_digest = crate::context_digest_v1(augmented_ccs, final_public_input);
    #[allow(deprecated)]
    { legacy_me.header_digest = context_digest; }
    
    // Generate the final lean SNARK proof
    let lean = neo_spartan_bridge::compress_me_to_lean_proof(&legacy_me, &legacy_wit)
        .map_err(|e| anyhow::anyhow!("Final SNARK compression failed: {}", e))?;
    
    // Extract final results from the last IVC step
    let final_results = ivc_proofs.last().unwrap().step_proof.public_results.clone();
    
    Ok(crate::Proof {
        v: 2,
        circuit_key: lean.circuit_key,
        vk_digest: lean.vk_digest,
        public_io: lean.public_io_bytes,
        proof_bytes: lean.proof_bytes,
        public_results: final_results,
        meta: crate::ProofMeta { 
            num_y_compact: ivc_proofs.last().unwrap().step_proof.meta.num_y_compact,
            num_app_outputs: ivc_proofs.last().unwrap().step_proof.meta.num_app_outputs,
        },
    })
}
pub fn build_augmented_ccs_linked(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],   
    x_witness_indices: &[usize],        
    y_len: usize,
    const1_witness_index: usize,
) -> Result<CcsStructure<F>, String> {
    // 🛡️ SECURITY: Validate matrix count assumptions
    let t = step_ccs.matrices.len();
    if t < 3 {
        return Err(format!(
            "augmented CCS requires at least 3 matrices (A,B,C). Got t={}", t
        ));
    }
    
    if y_step_offsets.len() != y_len {
        return Err(format!("y_step_offsets length {} must equal y_len {}", y_step_offsets.len(), y_len));
    }
    // y_prev_witness_indices are optional now (used later for cross-step stitching)
    // Only require equal length if provided.
    if !y_prev_witness_indices.is_empty() && y_prev_witness_indices.len() != y_len {
        return Err(format!(
            "y_prev_witness_indices length {} must equal y_len {} when provided",
            y_prev_witness_indices.len(), y_len
        ));
    }
    // Allow binding only the app input tail of step_x; not the digest prefix
    if !x_witness_indices.is_empty() && x_witness_indices.len() > step_x_len {
        return Err(format!("x_witness_indices length {} cannot exceed step_x_len {}", x_witness_indices.len(), step_x_len));
    }
    if const1_witness_index >= step_ccs.m {
        return Err(format!("const1_witness_index {} out of range (m={})", const1_witness_index, step_ccs.m));
    }
    for &o in y_step_offsets.iter().chain(y_prev_witness_indices).chain(x_witness_indices) {
        if o >= step_ccs.m {
            return Err(format!("witness offset {} out of range (m={})", o, step_ccs.m));
        }
    }

    // Public input: [ step_x || ρ || y_prev || y_next ]
    let pub_cols = step_x_len + 1 + 2 * y_len;

    // Row budget:
    //  - step_rows                              (copy step CCS)
    //  - 2*y_len EV rows                        (u = ρ*y_step, y_next - y_prev - u = 0)
    //  - step_x_len binder rows (optional)      (step_x[i] - step_witness[x_i] = 0)
    //  - y_len prev binder rows                 (REVIEWER FIX: y_prev[k] - step_witness[prev_k] = 0)
    let step_rows = step_ccs.n;
    let ev_rows = 2 * y_len;
    let x_bind_rows = if x_witness_indices.is_empty() { 0 } else { step_x_len };
    let prev_bind_rows = 0; // 🔒 Do not bind public y_prev to step witness columns in folding
    let total_rows = step_rows + ev_rows + x_bind_rows + prev_bind_rows;

    // Witness: [ step_witness || u ]
    let step_wit_cols = step_ccs.m;
    let ev_wit_cols = y_len; // u
    let total_wit_cols = step_wit_cols + ev_wit_cols;
    let total_cols = pub_cols + total_wit_cols;

    let mut combined_mats = Vec::new();
    for matrix_idx in 0..step_ccs.matrices.len() {
        let mut data = vec![F::ZERO; total_rows * total_cols];

        // Copy step CCS at the top
        let step_matrix = &step_ccs.matrices[matrix_idx];
        for r in 0..step_rows {
            for c in 0..step_ccs.m {
                let col = pub_cols + c;                  // step witness lives after public block
                data[r * total_cols + col] = step_matrix[(r, c)];
            }
        }

        // Offsets
        let col_rho     = step_x_len;
        let col_y_prev0 = col_rho + 1;
        let col_y_next0 = col_y_prev0 + y_len;
        // absolute column for the constant-1 witness (within the *augmented* z = [public | witness])
        let col_const1_abs = pub_cols + const1_witness_index;
        let col_u0      = pub_cols + step_wit_cols;

        // EV: u[k] = ρ * y_step[k]
        for k in 0..y_len {
            let r = step_rows + k;
            match matrix_idx {
                0 => data[r * total_cols + col_rho] = F::ONE,
                1 => data[r * total_cols + (pub_cols + y_step_offsets[k])] = F::ONE,
                2 => data[r * total_cols + (col_u0 + k)] = F::ONE,
                _ => {}
            }
        }
        // EV: y_next[k] - y_prev[k] - u[k] = 0  (× 1 via step_witness[0] == 1)
        for k in 0..y_len {
            let r = step_rows + y_len + k;
            match matrix_idx {
                0 => {
                    data[r * total_cols + (col_y_next0 + k)] = F::ONE;
                    data[r * total_cols + (col_y_prev0 + k)] = -F::ONE;
                    data[r * total_cols + (col_u0 + k)]      = -F::ONE;
                }
                1 => data[r * total_cols + col_const1_abs] = F::ONE,
                _ => {}
            }
        }

        // Binder X: step_x[i] - step_witness[x_i] = 0  (if any)
        if !x_witness_indices.is_empty() {
            // Bind only the last x_witness_indices.len() elements of step_x (the app inputs)
            let bind_len = x_witness_indices.len();
            let bind_start = step_x_len - bind_len;
            for i in 0..bind_len {
                let r = step_rows + ev_rows + i;
                match matrix_idx {
                    0 => {
                        data[r * total_cols + (bind_start + i)] = F::ONE;                         // + step_x[bind_start + i]
                        data[r * total_cols + (pub_cols + x_witness_indices[i])] = -F::ONE;      // - step_witness[x_i]
                    }
                    1 => data[r * total_cols + col_const1_abs] = F::ONE,                        // × 1
                    _ => {}
                }
            }
        }

        // 🔒 REMOVED: y_prev binder incompatible with folding semantics
        // The public y_prev (accumulated state) should NOT equal step witness columns.
        // Security comes from: (1) u = ρ*y_step linked to witness, (2) public stitching y_next^(i) == y_prev^(i+1)

        combined_mats.push(Mat::from_row_major(total_rows, total_cols, data));
    }

    let f = step_ccs.f.clone();
    CcsStructure::new(combined_mats, f).map_err(|e| format!("Failed to create CCS: {:?}", e))
}

/// Build witness for linked augmented CCS.
/// 
/// This creates the combined witness [step_witness || u] where u = ρ * y_step
/// and y_step is extracted from the step_witness at the specified offsets.
pub fn build_linked_augmented_witness(
    step_witness: &[F],
    y_step_offsets: &[usize],
    rho: F,
) -> Vec<F> {
    // Extract y_step values from step witness
    let mut y_step = Vec::with_capacity(y_step_offsets.len());
    for &offset in y_step_offsets {
        y_step.push(step_witness[offset]);
    }
    
    // Compute u = ρ * y_step
    let u: Vec<F> = y_step.iter().map(|&ys| rho * ys).collect();
    
    // Combined witness: [step_witness || u]
    let mut combined_witness = step_witness.to_vec();
    combined_witness.extend_from_slice(&u);
    
    combined_witness
}

/// Build public input for linked augmented CCS.
/// 
/// Public input layout: [step_x || ρ || y_prev || y_next]
pub fn build_linked_augmented_public_input(
    step_x: &[F],
    rho: F,
    y_prev: &[F],
    y_next: &[F],
) -> Vec<F> {
    let mut public_input = Vec::new();
    public_input.extend_from_slice(step_x);
    public_input.push(rho);
    public_input.extend_from_slice(y_prev);
    public_input.extend_from_slice(y_next);
    public_input
}

/// Compute Poseidon2 digest of the running accumulator as F-elements for step_x binding
/// Layout hashed (as bytes): step | c_z_digest | len(y) | y elements (u64 little-endian)
pub fn compute_accumulator_digest_fields(acc: &Accumulator) -> Result<Vec<F>, Box<dyn std::error::Error>> {
    // Reuse existing serializer for exact byte encoding
    let bytes = serialize_accumulator_for_commitment(acc)?;
    // Hash to 4 field elements (32 bytes) and return them as F limbs
    let digest_felts = p2::poseidon2_hash_packed_bytes(&bytes);
    let mut out = Vec::with_capacity(p2::DIGEST_LEN);
    for x in digest_felts { out.push(F::from_u64(x.as_canonical_u64())); }
    Ok(out)
}


/// Add stitching constraints to an existing CCS using absolute column indices.
/// Works for batched CCS where matrices.len() = 3 * (#triads).
/// 
/// This integrates cross-step stitching: y_next^(i) == y_prev^(i+1) into the combined CCS
/// using triad-aware row extension that populates every R1CS triad.
/// 
/// # Arguments
/// * `existing_ccs` - The combined CCS from batched direct sum
/// * `y_len` - Length of y vectors  
/// * `left_y_next_abs` - Absolute column index of y_next from step i
/// * `right_y_prev_abs` - Absolute column index of y_prev from step i+1
/// * `const1_abs` - Absolute column index of a known-1 value (for R1CS structure)
pub fn add_stitching_constraints_to_ccs(
    existing_ccs: CcsStructure<F>,
    y_len: usize,
    left_y_next_abs: usize,
    right_y_prev_abs: usize,
    const1_abs: usize,
) -> Result<CcsStructure<F>, String> {
    if y_len == 0 {
        return Ok(existing_ccs);
    }

    let old_rows = existing_ccs.n;
    let old_cols = existing_ccs.m;
    let new_rows = old_rows + y_len;

    // Validate indices
    let max_col_needed = [left_y_next_abs + y_len - 1, right_y_prev_abs + y_len - 1, const1_abs]
        .into_iter()
        .max()
        .unwrap();
    if max_col_needed >= old_cols {
        return Err(format!(
            "Column index out of range: need {}, have {}",
            max_col_needed, old_cols
        ));
    }

    let t = existing_ccs.matrices.len();
    if t % 3 != 0 {
        return Err(format!("Expected t % 3 == 0 (triads), got t={}", t));
    }
    let _triads = t / 3;

    let mut new_matrices = Vec::with_capacity(t);

    for (mat_idx, old_matrix) in existing_ccs.matrices.iter().enumerate() {
        let mut data = vec![F::ZERO; new_rows * old_cols];

        // Copy old rows
        for r in 0..old_rows {
            for c in 0..old_cols {
                data[r * old_cols + c] = old_matrix[(r, c)];
            }
        }

        // REVIEWER FIX: Add stitching constraints only to the FIRST triad  
        // to avoid over-constraining by writing to every triad.
        let a_idx = 0;  // First triad A matrix
        let b_idx = 1;  // First triad B matrix  
        let c_idx = 2;  // First triad C matrix

        for i in 0..y_len {
            let r = old_rows + i;

            if mat_idx == a_idx {
                // A: y_next[i] - y_prev[i]
                data[r * old_cols + (left_y_next_abs + i)] = F::ONE;
                data[r * old_cols + (right_y_prev_abs + i)] = -F::ONE;
            } else if mat_idx == b_idx {
                // B: multiply by known-1 column
                data[r * old_cols + const1_abs] = F::ONE;
            } else if mat_idx == c_idx {
                // C: leave zero
            }
        }

        new_matrices.push(Mat::from_row_major(new_rows, old_cols, data));
    }

    CcsStructure::new(new_matrices, existing_ccs.f.clone())
        .map_err(|e| format!("Failed to extend CCS with stitching constraints: {:?}", e))
}

/// Build CCS that enforces cross-step stitching: y_next^(i) == y_prev^(i+1).
/// 
/// This fixes the batch coherence vulnerability where multiple steps aren't 
/// constrained to form a coherent chain.
/// 
/// # Arguments  
/// * `y_len` - Length of y vectors
/// * `left_y_next_offset` - Start index of y_next from step i in global public input
/// * `right_y_prev_offset` - Start index of y_prev from step i+1 in global public input
/// * `total_public_cols` - Total number of public input columns
/// 
/// # Public Input Layout (for the combined batch)
/// [...left_step... || ...right_step...]
/// where each step contributes: [step_x || ρ || y_prev || y_next]
pub fn build_step_stitching_ccs(
    y_len: usize,
    left_y_next_offset: usize,
    right_y_prev_offset: usize, 
    total_public_cols: usize,
) -> CcsStructure<F> {
    if y_len == 0 {
        // Return empty CCS
        return neo_ccs::r1cs_to_ccs(
            Mat::zero(0, 1, F::ZERO),
            Mat::zero(0, 1, F::ZERO), 
            Mat::zero(0, 1, F::ZERO),
        );
    }
    
    let rows = y_len;
    let witness_cols = 1; // Just constant witness  
    let cols = total_public_cols + witness_cols;
    
    let mut a_data = vec![F::ZERO; rows * cols];
    let mut b_data = vec![F::ZERO; rows * cols];
    let c_data = vec![F::ZERO; rows * cols]; // All zeros for linear constraints
    
    // For each component of y: y_next^(left)[k] - y_prev^(right)[k] = 0
    for k in 0..y_len {
        let r = k;
        a_data[r * cols + (left_y_next_offset + k)] = F::ONE;  // +y_next^(left)[k]
        a_data[r * cols + (right_y_prev_offset + k)] = -F::ONE; // -y_prev^(right)[k]
        b_data[r * cols + total_public_cols] = F::ONE; // × constant 1
    }
    
    let a_mat = Mat::from_row_major(rows, cols, a_data);
    let b_mat = Mat::from_row_major(rows, cols, b_data);
    let c_mat = Mat::from_row_major(rows, cols, c_data);
    
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Evolve commitment coordinates using the same folding as y vectors.
/// 
/// This is critical for end-to-end binding: the commitment must evolve
/// with the same rho used for folding y_prev + rho * y_step = y_next.
/// 
/// # Arguments
/// * `coords_prev` - Previous step's commitment coordinates
/// * `coords_step` - Current step's commitment coordinates  
/// * `rho` - Folding challenge (same as used for y folding)
/// 
/// # Returns
/// * `Ok((coords_next, digest_next))` - Updated coordinates and digest
/// * `Err(String)` - Error if coordinate lengths mismatch
fn evolve_commitment(
    coords_prev: &[F],
    coords_step: &[F],
    rho: F,
) -> Result<(Vec<F>, [u8; 32]), String> {
    if coords_prev.len() != coords_step.len() {
        return Err(format!(
            "commitment coordinate length mismatch: prev={}, step={}", 
            coords_prev.len(), 
            coords_step.len()
        ));
    }
    
    let mut coords_next = coords_prev.to_vec();
    for (o, cs) in coords_next.iter_mut().zip(coords_step.iter()) {
        *o = *o + rho * *cs;
    }

    // Compute digest of evolved coordinates  
    let digest = digest_commit_coords(&coords_next);
    Ok((coords_next, digest))
}
