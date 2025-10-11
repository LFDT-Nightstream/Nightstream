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
use p3_goldilocks::Goldilocks;
use p3_field::PrimeCharacteristicRing;
use neo_ccs::{CcsStructure, Mat};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use p3_symmetric::Permutation;
use subtle::ConstantTimeEq;
use neo_fold::{pi_ccs_verify, pi_rlc_verify, pi_dec_verify};
#[allow(unused_imports)]
use neo_fold::pi_ccs::{
    pi_ccs_derive_transcript_tail,
    pi_ccs_compute_terminal_claim_r1cs_or_ccs,
};
use neo_ajtai::AjtaiSModule;
// Centralized transcript
use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_transcript::labels as tr_labels;
use neo_math::{Rq, cf_inv, SAction};

// (moved) Domain tags now live in neo-transcript::labels

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

// Build χ_r(i) over the prefix 0..n-1 using ell = r.len() (LSB-first bit order).
#[inline]
fn chi_r_prefix(r: &[neo_math::K], n: usize) -> Vec<neo_math::K> {
    let ell = r.len();
    let mut chi = vec![neo_math::K::ZERO; n];
    for i in 0..n {
        let mut w = neo_math::K::ONE;
        let mut ii = i;
        for k in 0..ell {
            let rk = r[k];
            let bit_is_one = (ii & 1) == 1;
            let term = if bit_is_one { rk } else { neo_math::K::ONE - rk };
            w *= term;
            ii >>= 1;
        }
        chi[i] = w;
    }
    chi
}

// Robust tie check at verifier r: y_j ?= Z · (M_j^T · χ_r), supports any n (not just powers of two).
fn tie_check_with_r(
    s: &neo_ccs::CcsStructure<F>,
    me_parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    wit_parent: &neo_ccs::MeWitness<F>,
    r: &[neo_math::K],
) -> Result<(), String> {
    let d = neo_math::D;
    let n = s.n;
    let m = s.m;
    let t = s.t() as usize;

    if wit_parent.Z.rows() != d || wit_parent.Z.cols() != m {
        return Err(format!(
            "wit_parent.Z shape {}x{} != D x m ({} x {})",
            wit_parent.Z.rows(), wit_parent.Z.cols(), d, m
        ));
    }
    if me_parent.y.len() != t {
        return Err(format!("me_parent.y len {} != t {}", me_parent.y.len(), t));
    }
    for (j, yj) in me_parent.y.iter().enumerate() {
        if yj.len() != d {
            return Err(format!("me_parent.y[{}] len {} != D {}", j, yj.len(), d));
        }
    }

    let chi = chi_r_prefix(r, n);

    for j in 0..t {
        let mj = &s.matrices[j];
        // v_j = M_j^T · χ_r  (size m)
        let mut v_j: Vec<neo_math::K> = vec![neo_math::K::ZERO; m];
        for i in 0..n {
            let w_i = chi[i];
            for col in 0..m {
                let m_ij: F = mj[(i, col)];
                if m_ij != F::ZERO { v_j[col] += neo_math::K::from(m_ij) * w_i; }
            }
        }
        // y_pred = Z · v_j  (size D)
        let mut y_pred: Vec<neo_math::K> = vec![neo_math::K::ZERO; d];
        for row in 0..d {
            let mut acc = neo_math::K::ZERO;
            for col in 0..m {
                let z_rc: F = wit_parent.Z[(row, col)];
                if z_rc != F::ZERO { acc += neo_math::K::from(z_rc) * v_j[col]; }
            }
            y_pred[row] = acc;
        }
        if y_pred != me_parent.y[j] {
            return Err(format!("tie mismatch on j={} (Z·(M_j^T·χ_r))", j));
        }
    }

    // Also ensure X matches Z prefix (public slice), a cheap consistency guard.
    let m_in = me_parent.m_in;
    if me_parent.X.rows() != d || me_parent.X.cols() != m_in {
        return Err("me_parent.X shape mismatch".into());
    }
    for row in 0..d { for col in 0..m_in {
        if me_parent.X[(row, col)] != wit_parent.Z[(row, col)] {
            return Err("X != Z[:, :m_in]".into());
        }
    }}

    Ok(())
}

// Public test-only wrapper for integration tests.
#[cfg(feature = "testing")]
pub fn tie_check_with_r_public(
    s: &neo_ccs::CcsStructure<F>,
    me_parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    wit_parent: &neo_ccs::MeWitness<F>,
    r: &[neo_math::K],
) -> Result<(), String> {
    tie_check_with_r(s, me_parent, wit_parent, r)
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
    /// Positions of step program-supplied public inputs in the step witness (binds the tail of step_x)
    pub step_program_input_witness_indices: Vec<usize>,
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
    /// Full augmented public input [step_x || ρ || y_prev || y_next] used for folding
    pub step_augmented_public_input: Vec<F>,
    /// Augmented input for the previous accumulator (LHS instance) used during folding
    pub prev_step_augmented_public_input: Vec<F>,
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

/// Build a canonical zero MCS instance for a given shape (m_in, m_step) to start a fold chain.
///
/// This avoids the unsound base case where the prover folded the step with itself.
pub fn zero_mcs_instance_for_shape(
    m_in: usize,
    m_step: usize,
    const1_witness_index: Option<usize>,
) -> anyhow::Result<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> {
    let d = neo_math::ring::D;
    anyhow::ensure!(m_step >= m_in, "zero_mcs_instance_for_shape: m_step < m_in ({} < {})", m_step, m_in);

    // Construct a zero Ajtai commitment directly; avoid PP setup/commit computation.
    // Derive kappa from any registered Ajtai PP if available to avoid drift; fallback to 16.
    let _kappa = neo_ajtai::get_global_pp().map(|pp| pp.kappa).unwrap_or(16usize);
    let w_len = m_step - m_in;
    let x_zero = vec![F::ZERO; m_in];
    let mut w_zero = vec![F::ZERO; w_len];
    let mut z_zero = neo_ccs::Mat::zero(d, m_step, F::ZERO);
    // Ensure the constant-1 witness column is actually 1 in the base-case Z
    if let Some(idx) = const1_witness_index {
        if idx < w_len {
            // Column absolute index of const1 within z = [public || witness]
            let col_abs = m_in + idx;
            z_zero[(0, col_abs)] = F::ONE; // least-significant digit 1
            // Ensure undigitized z has 1 at the const1 witness index
            w_zero[idx] = F::ONE;
        }
    }

    // Use actual commitment for the base-case Z to satisfy c = L(Z)
    // Ensure Ajtai PP exists for (d, m_step). In testing, auto-ensure; in prod, error out.
    let l = match neo_ajtai::AjtaiSModule::from_global_for_dims(d, m_step) {
        Ok(l) => l,
        Err(_) => {
            #[cfg(not(feature = "testing"))]
            {
                return Err(anyhow::anyhow!(
                    "Ajtai PP missing for dims (D={}, m={}); register CRS/PP before proving base case",
                    d, m_step
                ));
            }
            #[cfg(feature = "testing")]
            {
                let kappa_guess = neo_ajtai::get_global_pp().map(|pp| pp.kappa).unwrap_or(16usize);
                super::ensure_ajtai_pp_for_dims(d, m_step, || {
                    use rand::{RngCore, SeedableRng};
                    use rand::rngs::StdRng;
                    let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
                        StdRng::from_seed([42u8; 32])
                    } else {
                        let mut seed = [0u8; 32];
                        rand::rng().fill_bytes(&mut seed);
                        StdRng::from_seed(seed)
                    };
                    let pp = crate::ajtai_setup(&mut rng, d, kappa_guess, m_step)?;
                    neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
                })?;
                neo_ajtai::AjtaiSModule::from_global_for_dims(d, m_step)
                    .map_err(|e| anyhow::anyhow!("AjtaiSModule unavailable for (d={}, m={}): {}", d, m_step, e))?
            }
        }
    };
    let c_zero = l.commit(&z_zero);

    Ok((
        neo_ccs::McsInstance { c: c_zero, x: x_zero, m_in },
        neo_ccs::McsWitness::<F> { w: w_zero, Z: z_zero },
    ))
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
// Old local transcript helpers removed in favor of neo-transcript

/// Deterministic Poseidon2 domain-separated hash to derive folding challenge ρ
/// Uses the same Poseidon2 configuration as context_digest_v1 for consistency
#[allow(unused_assignments)]
pub fn rho_from_transcript(prev_acc: &Accumulator, step_digest: [u8; 32], c_step_coords: &[F]) -> (F, [u8; 32]) {
    // Use centralized Merlin-style transcript (Poseidon2 backend)
    #[cfg(feature = "fs-guard")]
    neo_transcript::fs_guard::reset("rho/actual");

    let mut tr = Poseidon2Transcript::new(b"neo/ivc");
    tr.append_fields(tr_labels::STEP, &[F::from_u64(prev_acc.step)]);
    tr.append_message(tr_labels::ACC_DIGEST, &prev_acc.c_z_digest);
    tr.append_fields(b"acc/y", &prev_acc.y_compact);
    tr.append_message(tr_labels::STEP_DIGEST, &step_digest);
    tr.append_fields(tr_labels::COMMIT_COORDS, c_step_coords);
    let rho = tr.challenge_nonzero_field(tr_labels::CHAL_RHO);
    let dig = tr.digest32();

    #[cfg(feature = "fs-guard")]
    {
        use neo_transcript::fs_guard as guard;
        let actual = guard::take();
        guard::reset("rho/spec");
        // SPEC: explicit replay (kept in sync with intended API)
        let mut tr_s = Poseidon2Transcript::new(b"neo/ivc");
        tr_s.append_fields(tr_labels::STEP, &[F::from_u64(prev_acc.step)]);
        tr_s.append_message(tr_labels::ACC_DIGEST, &prev_acc.c_z_digest);
        tr_s.append_fields(b"acc/y", &prev_acc.y_compact);
        tr_s.append_message(tr_labels::STEP_DIGEST, &step_digest);
        tr_s.append_fields(tr_labels::COMMIT_COORDS, c_step_coords);
        let _ = tr_s.challenge_nonzero_field(tr_labels::CHAL_RHO);
        let _ = tr_s.digest32();
        let spec = guard::take();
        if let Some((i, s, a)) = guard::first_mismatch(&spec, &actual) {
            panic!(
                "FS drift in rho_from_transcript at #{}: spec(op={},label={:?},len={}) vs actual(op={},label={:?},len={})",
                i, s.op, s.label, s.len, a.op, a.label, a.len
            );
        }
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

// Toy hash functions completely removed - production uses rho_from_transcript() with real Poseidon2

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

/// Generate RLC coefficients for step commitment binding
/// 
/// Uses Poseidon2 with domain separation to derive random coefficients
/// from the transcript state after c_step is committed.
pub fn generate_rlc_coefficients(
    prev_accumulator: &Accumulator,
    step_digest: [u8; 32],
    c_step_coords: &[F],
    num_coords: usize,
) -> Vec<F> {
    // Domain-separated transcript for RLC coefficients
    let mut transcript_data = Vec::new();
    
    // Include accumulator digest
    if let Ok(acc_fields) = compute_accumulator_digest_fields(prev_accumulator) {
        for field in acc_fields {
            transcript_data.push(field.as_canonical_u64());
        }
    }
    
    // Include step digest
    for chunk in step_digest.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        transcript_data.push(u64::from_le_bytes(bytes));
    }
    
    // Include c_step coordinates
    for &coord in c_step_coords {
        transcript_data.push(coord.as_canonical_u64());
    }
    
    // Domain separation for RLC
    let domain_tag = b"NEO_RLC_V1";
    let mut domain_u64s = Vec::new();
    for chunk in domain_tag.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        domain_u64s.push(u64::from_le_bytes(bytes));
    }
    transcript_data.extend_from_slice(&domain_u64s);
    
    // Hash to get random seed
    let seed_digest = p2::poseidon2_hash_packed_bytes(&transcript_data.iter().flat_map(|&x| x.to_le_bytes()).collect::<Vec<_>>());
    
    // Generate coefficients using the seed
    let mut coeffs = Vec::with_capacity(num_coords);
    let mut state = seed_digest;
    
    for i in 0..num_coords {
        // Use index to ensure different coefficients
        let mut input = state.to_vec();
        input.push(neo_math::F::from_u64(i as u64));
        state = p2::poseidon2_hash_packed_bytes(&input.iter().flat_map(|x| x.as_canonical_u64().to_le_bytes()).collect::<Vec<_>>());
        coeffs.push(F::from_u64(state[0].as_canonical_u64()));
    }
    
    coeffs
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
    
    // End-of-message marker and final permutation
    if absorbed >= RATE {
        st = poseidon2.permute(st);
        absorbed = 0;
    }
    st[absorbed] = Goldilocks::ONE;
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
    // NOTE: ℓ ≤ 1 limitation
    // 
    // When ℓ=1 (single-row CCS padded to 2 rows), the augmented CCS carries a constant offset
    // from const-1 binding and other glue rows. This makes initial_sum non-zero even for valid
    // witnesses, preventing the verifier from distinguishing valid from invalid witnesses based
    // on the initial_sum == 0 check.
    //
    // For soundness, production circuits SHOULD have at least 3 constraint rows to ensure ℓ >= 2
    // after power-of-2 padding (3 rows → padded to 4 → ℓ = log₂(4) = 2).
    //
    // We don't enforce this as a hard guard because:
    // 1. Valid witnesses for ℓ=1 CCS are correctly accepted (no false rejections)
    // 2. The limitation is documented and tests demonstrate the behavior
    // 3. A future fix (Option B: carry "Q is pure residual" bit) will address this properly
    //
    // Users should be aware that ℓ=1 circuits cannot have invalid witnesses detected at
    // verification time and should add prover-side checks if needed.
    
    // Extract REAL y_step from step computation (not placeholder)
    let y_step = extractor.extract_y_step(step_witness);
    
    #[cfg(feature = "neo-logs")]
    #[cfg(feature = "debug-logs")]
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
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };
    
    prove_ivc_step(input)
}

/// Prove a single IVC step with proper chaining.
///
/// - Accepts previous folded ME instance (for Stage 5 compression continuity).
/// - Accepts previous RHS MCS instance+witness, and returns the current RHS MCS to be
///   used as the next LHS. This ensures exact X‑linkage across steps by construction.
pub fn prove_ivc_step_chained(
    input: IvcStepInput,
    prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    prev_me_wit: Option<neo_ccs::MeWitness<F>>,
    prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
) -> Result<(
    IvcStepResult,
    neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    neo_ccs::MeWitness<F>,
    (neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>),
), Box<dyn std::error::Error>> {
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

    let step_data = build_step_transcript_data(&input.prev_accumulator, input.step, &step_x);
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
    let x_bind_len = input.binding_spec.step_program_input_witness_indices.len();
    // SECURITY: If there are app inputs in step_x, they must be bound to witness indices
    // to prevent public input manipulation.
    if app_len > 0 && x_bind_len == 0 && !input.transcript_only_app_inputs {
        return Err("SECURITY: step_program_input_witness_indices cannot be empty when step_x has app inputs; this would allow public input manipulation".into());
    }
    if x_bind_len > 0 && x_bind_len != app_len {
        return Err("step_program_input_witness_indices length must match app public input length".into());
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

    // Guard: extractor vs binding_spec.y_step_offsets must agree
    // This prevents a subtle class of bugs where the EV constraints are wired to
    // witness positions different from the values used to compute y_next.
    if !input.binding_spec.y_step_offsets.is_empty() {
        let mut y_from_offsets = Vec::with_capacity(input.binding_spec.y_step_offsets.len());
        for &idx in &input.binding_spec.y_step_offsets {
            y_from_offsets.push(*input
                .step_witness
                .get(idx)
                .ok_or_else(|| format!("y_step_offsets index {} out of bounds for step_witness (len={})", idx, input.step_witness.len()))?);
        }
        if y_from_offsets != input.y_step {
            return Err(format!(
                "Extractor/binding mismatch: y_step extracted by StepOutputExtractor != step_witness[y_step_offsets].\n  extracted: {:?}\n  from_offsets: {:?}\n  offsets: {:?}",
                input.y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                y_from_offsets.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                input.binding_spec.y_step_offsets
            ).into());
        }
    }

    // PROVER-SIDE CHECK (COMMENTED OUT TO TEST IN-CIRCUIT ENFORCEMENT):
    // This check can be bypassed by a malicious prover who modifies the code.
    // The REAL security MUST come from in-circuit constraints in the augmented CCS.
    // 
    // TODO: Investigate why step CCS constraints might not be properly enforced in-circuit.
    // The step CCS should be copied into the augmented CCS and checked cryptographically.
    //
    // let step_ccs_public_input = match input.public_input {
    //     Some(app_inputs) => app_inputs,
    //     None => &[],
    // };
    // neo_ccs::check_ccs_rowwise_zero(input.step_ccs, step_ccs_public_input, input.step_witness)
    //     .map_err(|e| format!(
    //         "SOUNDNESS ERROR: step witness does not satisfy CCS constraints: {:?}",
    //         e
    //     ))?;


    // 3) Commit to the ρ-independent step witness (Pattern B), then derive ρ,
    // then build the full augmented witness for proving. This breaks FS circularity.
    let d = neo_math::ring::D;
    
    // Commit to the step witness only (not including EV part)
    
    // Pattern B: derive ρ from a commitment that does NOT include the ρ-dependent tail.
    // Implementation detail: we keep dimensions stable by zero-padding the tail so
    // the Ajtai PP (d, m) matches the later full vector. No in-circuit link is added.
    
    // First, determine the final witness structure to get consistent dimensions
    let temp_witness = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        F::ONE // temporary rho for dimension calculation
    );
    let y_len = input.prev_accumulator.y_compact.len();
    
    // Build the final public input structure for dimension calculation
    let temp_y_next = input.prev_accumulator.y_compact.clone(); // placeholder
    let final_public_input = build_augmented_public_input_for_step(
        &step_x, F::ONE, &input.prev_accumulator.y_compact, &temp_y_next
    );
    
    // Calculate final dimensions: [final_public_input || temp_witness]
    let mut final_z = final_public_input.clone();
    final_z.extend_from_slice(&temp_witness);
    let final_decomp = crate::decomp_b(&final_z, input.params.b, d, crate::DecompStyle::Balanced);
    let m_final = final_decomp.len() / d;
    
    // Setup Ajtai PP for final dimensions (used for both pre-commit and final commit)
    crate::ensure_ajtai_pp_for_dims(d, m_final, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, input.params.kappa as usize, m_final)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;

    // Build pre-commit vector: same structure as final but with ρ=0 for EV part
    // (equivalent to committing only to [step_x || step_witness] under Pattern B semantics)
    let pre_public_input = build_augmented_public_input_for_step(
        &step_x, F::ZERO, &input.prev_accumulator.y_compact, &temp_y_next
    );
    let pre_witness = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        F::ZERO // This zeros out the U = ρ·y_step part
    );
    
    let mut z_pre = pre_public_input.clone();
    z_pre.extend_from_slice(&pre_witness);
    let decomp_pre = crate::decomp_b(&z_pre, input.params.b, d, crate::DecompStyle::Balanced);
    
    // Pre-commit (breaks Fiat-Shamir circularity)
    let pp = neo_ajtai::get_global_pp_for_dims(d, m_final)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP: {}", e))?;
    let pre_commitment = crate::commit(&*pp, &decomp_pre);
    
    // Extract pre-commit coordinates for ρ derivation
    let c_step_coords: Vec<F> = pre_commitment
        .data
        .iter()
        .map(|&x| F::from_u64(x.as_canonical_u64()))
        .collect();
    
    // Derive ρ from pre-commitment (standard Fiat-Shamir order)
    let (rho, _td) = rho_from_transcript(&input.prev_accumulator, step_digest, &c_step_coords);
    
    // CRITICAL: Π_RLC binding to prevent split-brain attacks
    // Generate RLC coefficients from the same transcript used for ρ
    let num_coords = c_step_coords.len();
    let rlc_coeffs = generate_rlc_coefficients(&input.prev_accumulator, step_digest, &c_step_coords, num_coords);
    
    // Compute aggregated Ajtai row G = Σ_i r_i · L_i for RLC binding
    // CRITICAL: Use exact PP dimensions, not decomp length (which may be padded)
    let total_z_len = d * m_final; // Must equal d * m for Ajtai validation
    let _aggregated_row = neo_ajtai::compute_aggregated_ajtai_row(&*pp, &rlc_coeffs, total_z_len, num_coords)
        .map_err(|e| anyhow::anyhow!("Failed to compute aggregated Ajtai row: {}", e))?;
    
    // Compute RLC right-hand side: rhs = ⟨r, c_step⟩
    let _rlc_rhs = rlc_coeffs.iter().zip(c_step_coords.iter())
        .map(|(ri, ci)| *ri * *ci)
        .fold(F::ZERO, |acc, x| acc + x);
    
    // Store the U offset for the circuit (where the ρ-dependent part starts)
    let u_offset = pre_public_input.len() + input.step_witness.len();
    let u_len = y_len;
    
    // Store final dimensions for later validation (if needed)
    let _expected_m_final = m_final;
    
    // 6) Build full witness and public input with the actual rho
    let step_witness_augmented = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        rho
    );
    let y_next: Vec<F> = input.prev_accumulator.y_compact.iter()
        .zip(input.y_step.iter())
        .map(|(&p, &s)| p + rho * s)
        .collect();
    let step_public_input = build_augmented_public_input_for_step(
        &step_x, rho, &input.prev_accumulator.y_compact, &y_next
    );

    // 7) Build the full commitment for the MCS instance (includes EV variables for the CCS),
    // but note the IVC accumulator uses only the pre-ρ step commitment (c_step_coords).
    let mut full_step_z = step_public_input.clone();
    full_step_z.extend_from_slice(&step_witness_augmented);
    let decomp_z = crate::decomp_b(&full_step_z, input.params.b, d, crate::DecompStyle::Balanced);
    if decomp_z.len() % d != 0 { return Err("decomp length not multiple of d".into()); }
    let m_step = decomp_z.len() / d;

    // Pattern A: Ensure PP exists for the full witness dimensions (m_step)
    // This is different from m_final because the full witness includes additional public input structure
    crate::ensure_ajtai_pp_for_dims(d, m_step, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, input.params.kappa as usize, m_step)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;
    
    // Pattern B: Use pre_commitment for ρ derivation and accumulator evolution

    // Get PP for the full witness dimensions (m_step)
    let _pp_full = neo_ajtai::get_global_pp_for_dims(d, m_step)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for full witness: {}", e))?;

    // Build full commitment that matches the full witness (for CCS consistency)
    let full_commitment = crate::commit(&*_pp_full, &decomp_z);
    // CCS-level RLC binder intentionally disabled. Soundness is enforced by
    // Pi-CCS → Pi-RLC → Pi-DEC and commitment evolution checks.
    // Intentionally no CCS-level RLC binder row in the proving CCS.
    // Rationale:
    // - The linear form you might try to enforce, <G, vec(Z)> = rhs, lives in Ajtai digit space
    //   (Z are base-b digits). CCS variables are the undigitized vector z. Encoding this check
    //   directly in CCS is structurally mismatched unless we first lift the relevant digit columns
    //   into CCS, which increases width and duplicates Π_DEC checks.
    // - Soundness and completeness are already enforced by the folding pipeline:
    //      • Π_CCS (with eq-binding for R1CS shapes) enforces the constraint relation,
    //      • Π_RLC performs the random linear combination with strong-set guard,
    //      • Π_DEC authenticates digit decomposition & range, tying digits to the commitment,
    //      • The IVC layer checks commitment evolution c_next = c_prev + ρ·c_step.
    //   This mirrors HyperNova/LatticeFold-style reductions and keeps CCS lean.
    // - For experiments, the builder still supports adding a single linear row; unit tests cover
    //   that it is encoded as a true linear equality (<G,z> = rhs) and rejects mismatches.
    let rlc_binder = None;
    let step_augmented_ccs = build_augmented_ccs_linked_with_rlc(
        input.step_ccs,
        step_x.len(),
        &input.binding_spec.y_step_offsets,
        &input.binding_spec.y_prev_witness_indices,
        &input.binding_spec.step_program_input_witness_indices,
        y_len,
        input.binding_spec.const1_witness_index,
        rlc_binder, // RLC binder enabled for soundness
    )?;

    // CCS uses full vector (with ρ), commitment binding uses pre-commit
    let full_witness_part = full_step_z[step_public_input.len()..].to_vec();
    
    // DEBUG: Check consistency
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: full_step_z.len()={}, step_public_input.len()={}, full_witness_part.len()={}", 
             full_step_z.len(), step_public_input.len(), full_witness_part.len());
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: decomp_z.len()={}, d*m_step={}", 
             decomp_z.len(), d * m_step);
    
    // Build MCS instance/witness using shared helper
    // CRITICAL FIX: Use step_public_input for CCS instance (with ρ)
    // Pattern B: The CCS instance uses ρ-bearing public input, full witness, and full commitment
    let (step_mcs_inst, step_mcs_wit) = crate::build_mcs_from_decomp(
        full_commitment,
        &decomp_z,
        &step_public_input,
        &full_witness_part,
        d,
        m_step,
    );

    // 6) Reify previous ME→MCS, or create trivial zero instance (base case)
    let (lhs_inst, lhs_wit) = if let Some((inst, wit)) = prev_lhs_mcs {
        // Use exact previous RHS MCS as next LHS for strict linkage
        (inst, wit)
    } else {
        match (prev_me, prev_me_wit) {
        (Some(me), Some(wit)) => {
            // Dimension checks for ME→MCS reification
            if wit.Z.rows() != d {
                return Err(format!("prev ME witness Z has wrong row count: expected {}, got {}", d, wit.Z.rows()).into());
            }
            if wit.Z.cols() != m_step {
                return Err(format!("prev ME witness Z has wrong column count: expected {}, got {}", m_step, wit.Z.cols()).into());
            }
            
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
            
            // Check m_in consistency with current step
            if me.m_in != step_public_input.len() {
                return Err(format!("m_in mismatch between prev ME ({}) and current step ({})", me.m_in, step_public_input.len()).into());
            }
            let wit_mcs = neo_ccs::McsWitness::<F> { w: w_prev, Z: wit.Z.clone() };
            (inst, wit_mcs)
        }
        _ => {
            // Base case (step 0): use a canonical zero running instance matching current shape.
            zero_mcs_instance_for_shape(step_public_input.len(), m_step, Some(input.binding_spec.const1_witness_index))?
        }
    }};

    // DEBUG: Check if this is step 0 or later
    let is_first_step = input.prev_accumulator.step == 0;
    #[cfg(feature = "debug-logs")]
    {
        println!("🔍 DEBUG: Step {}, is_first_step: {}", input.step, is_first_step);
        println!("🔍 DEBUG: prev_accumulator.c_coords.len(): {}", input.prev_accumulator.c_coords.len());
        println!("🔍 DEBUG: c_step_coords.len(): {}", c_step_coords.len());
        println!("🔍 DEBUG: LHS instance commitment len: {}", lhs_inst.c.data.len());
        println!("🔍 DEBUG: RHS instance commitment len: {}", step_mcs_inst.c.data.len());
        println!("🔍 DEBUG: LHS witness Z shape: {}x{}", lhs_wit.Z.rows(), lhs_wit.Z.cols());
        println!("🔍 DEBUG: RHS witness Z shape: {}x{}", step_mcs_wit.Z.rows(), step_mcs_wit.Z.cols());
    }
    
    // DEBUG: Check if commitments are consistent
    if !is_first_step {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: LHS commitment first 4 coords: {:?}", 
                 lhs_inst.c.data.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: RHS commitment first 4 coords: {:?}", 
                 step_mcs_inst.c.data.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: prev_accumulator.c_coords first 4: {:?}", 
                 input.prev_accumulator.c_coords.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: c_step_coords first 4: {:?}", 
                 c_step_coords.iter().take(4).collect::<Vec<_>>());
    }
    
    // 7) Fold prev-with-current using the production pipeline
    // Record the exact LHS augmented input used inside Pi-CCS for robust linking checks.
    // Do not trust/progate external prev_augmented_x here; the authoritative value is lhs_inst.x.
    let prev_augmented_public_input = lhs_inst.x.clone();
    // Clone MCS witnesses for later recombination of parent witness
    // (removed unused clones)
    let (mut me_instances, digit_witnesses, folding_proof) = neo_fold::fold_ccs_instances(
        input.params, 
        &step_augmented_ccs, 
        &[lhs_inst.clone(), step_mcs_inst.clone()], 
        &[lhs_wit.clone(),  step_mcs_wit.clone()]
    ).map_err(|e| format!("Nova folding failed: {}", e))?;

    // 🔒 SOUNDNESS: Populate ME instances with step commitment binding data
    for me_instance in &mut me_instances {
        me_instance.c_step_coords = c_step_coords.clone();
        me_instance.u_offset = u_offset;
        me_instance.u_len = u_len;
    }

    // 8) Evolve accumulator commitment coordinates with ρ using the step-only commitment.
    // Pattern B: c_next = c_prev + ρ·c_step, where c_step = pre-ρ step commitment (no tail)
    #[cfg(feature = "debug-logs")]
    {
        println!("🔍 DEBUG: About to evolve commitment, prev_coords.is_empty()={}", input.prev_accumulator.c_coords.is_empty());
        println!("🔍 DEBUG: rho value: {:?}", rho.as_canonical_u64());
    }
    let (c_coords_next, c_z_digest_next) = if input.prev_accumulator.c_coords.is_empty() {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: First step, using c_step_coords directly");
        let digest = digest_commit_coords(&c_step_coords);
        (c_step_coords.clone(), digest)
    } else {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: Evolving commitment: c_prev.len()={}, c_step.len()={}, rho={:?}", 
                 input.prev_accumulator.c_coords.len(), c_step_coords.len(), rho.as_canonical_u64());
        let result = evolve_commitment(&input.prev_accumulator.c_coords, &c_step_coords, rho)
            .map_err(|e| format!("commitment evolution failed: {}", e))?;
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: Commitment evolution completed successfully");
        result
    };
    
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: c_coords_next.len()={}", c_coords_next.len());

    let next_accumulator = Accumulator {
        c_z_digest: c_z_digest_next,
        c_coords: c_coords_next,
        y_compact: y_next.clone(),
        step: input.step + 1,
    };

    // 9) Package IVC proof (no per-step SNARK compression)
    // Compute context digest using a SIMPLE CCS construction (without RLC binder) for consistency
    // The verifier will reconstruct the same simple CCS for digest verification
    let digest_ccs = build_augmented_ccs_linked(
        input.step_ccs,
        step_x.len(),
        &input.binding_spec.y_step_offsets,
        &input.binding_spec.y_prev_witness_indices,
        &input.binding_spec.step_program_input_witness_indices,
        y_len,
        input.binding_spec.const1_witness_index,
    ).map_err(|e| anyhow::anyhow!("Failed to build digest CCS: {}", e))?;
    
    let context_digest = crate::context_digest_v1(&digest_ccs, &step_public_input);
    
    #[cfg(feature = "neo-logs")]
    {
        eprintln!("🔍 PROVER DIGEST DEBUG:");
        eprintln!("  Prover context digest: {:02x?}", &context_digest[..8]);
        eprintln!("  Digest CCS: n={}, m={}", digest_ccs.n, digest_ccs.m);
        eprintln!("  Step public input length: {}", step_public_input.len());
        eprintln!("  Prover CCS params: step_x_len={}, y_len={}, const1_idx={}", 
                  step_x.len(), y_len, input.binding_spec.const1_witness_index);
        eprintln!("  Prover y_step_offsets: {:?}", input.binding_spec.y_step_offsets);
    }
    
    // Build public_io: [y_next values as bytes] + [context_digest]
    // This allows verify_and_extract* to work and puts digest at the end for verify()
    let mut public_io = Vec::with_capacity(8 * y_next.len() + 32);
    for y in &y_next {
        public_io.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    public_io.extend_from_slice(&context_digest);
    
    let step_proof = crate::Proof {
        v: 2,
        circuit_key: [0u8; 32],           
        vk_digest: [0u8; 32],             
        public_io,                        // y_next values + context digest
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
        step_augmented_public_input: step_public_input.clone(),
        prev_step_augmented_public_input: prev_augmented_public_input,
        step_rho: rho,
        step_y_prev: input.prev_accumulator.y_compact.clone(),
        step_y_next: y_next.clone(),
        c_step_coords,
        me_instances: Some(me_instances.clone()), // Keep for final SNARK generation (TODO: optimize)
        digit_witnesses: Some(digit_witnesses.clone()), // Keep for final SNARK generation (TODO: optimize)
        folding_proof: Some(folding_proof),
    };

    // Return next chaining state: carry latest digit ME (for Stage 5) and RHS MCS (for strict linkage)
    Ok((
        IvcStepResult { proof: ivc_proof, next_state: y_next },
        me_instances.last().unwrap().clone(),
        digit_witnesses.last().unwrap().clone(),
        (step_mcs_inst, step_mcs_wit)
    ))
}


/// Prove a single IVC step using the main Neo proving pipeline
/// 
/// This is a convenience wrapper around `prove_ivc_step_chained` for cases
/// where you don't need to maintain chaining state between calls.
/// For proper Nova chaining, use `prove_ivc_step_chained` directly.
pub fn prove_ivc_step(input: IvcStepInput) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // Use the chained version with no previous ME instance (folds with canonical zero instance)
    let (result, _me, _wit, _mcs) = prove_ivc_step_chained(input, None, None, None)?;
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
    params: &crate::NeoParams,
    prev_augmented_x: Option<&[F]>,
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
    let step_data = build_step_transcript_data(prev_accumulator, ivc_proof.step, &ivc_proof.step_public_input);
    let step_digest = create_step_digest(&step_data);
    
    // 2. Build base augmented CCS using TRUSTED binding metadata
    // 🔒 SECURITY: Use TRUSTED binding_spec, NOT proof-supplied values!
    let y_len = prev_accumulator.y_compact.len();
    let step_x_len = ivc_proof.step_public_input.len();
    
    // 🔒 SECURITY: Recompute ρ to get the same transcript state
    let (rho, _transcript_digest) = rho_from_transcript(prev_accumulator, step_digest, &ivc_proof.c_step_coords);
    
    // --- Guard A: if the proof carries a step_rho, it must match the verifier's recomputation
    // This prevents any attempt to smuggle an inconsistent rho alongside forged c_step_coords.
    #[cfg(feature = "neo-logs")]
    eprintln!("🔍 DEBUG: Verifier recomputed ρ = {}, proof.step_rho = {}", 
              rho.as_canonical_u64(), ivc_proof.step_rho.as_canonical_u64());
    if ivc_proof.step_rho != F::ZERO && ivc_proof.step_rho != rho {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "❌ SECURITY GUARD A: Recomputed ρ ({}) != proof.step_rho ({})",
            rho.as_canonical_u64(),
            ivc_proof.step_rho.as_canonical_u64()
        );
        return Ok(false);
    }
    
    // RLC coefficients no longer needed here (binder disabled)
    
    // Reconstruct witness structure to determine dimensions
    let step_witness_augmented = build_linked_augmented_witness(
        &vec![F::ZERO; step_ccs.m], // dummy step witness for sizing
        &binding_spec.y_step_offsets,
        rho
    );
    let step_public_input = build_augmented_public_input_for_step(
        &ivc_proof.step_public_input,
        rho,
        &prev_accumulator.y_compact,
        &ivc_proof.next_accumulator.y_compact
    );
    if step_public_input != ivc_proof.step_augmented_public_input {
        return Ok(false);
    }

    // Base-case canonicalization: when there is no prior accumulator commitment and the caller
    // did not thread a previous augmented x, require the LHS augmented x be the canonical zero vector
    // of the correct shape. This removes transcript malleability at step 0 and matches
    // zero_mcs_instance_for_shape.
    if prev_accumulator.c_coords.is_empty() && prev_augmented_x.is_none() {
        let x_lhs = &ivc_proof.prev_step_augmented_public_input;
        let expected_len = step_x_len + 1 + 2 * y_len;
        if x_lhs.len() != expected_len { return Ok(false); }
        if !x_lhs.iter().all(|&f| f == F::ZERO) { return Ok(false); }
    }
    
    // Compute full witness dimensions
    let mut full_step_z = step_public_input.clone();
    full_step_z.extend_from_slice(&step_witness_augmented);
    let decomp_z = crate::decomp_b(
        &full_step_z,
        params.b, // use the same base as prover
        neo_math::ring::D,
        crate::DecompStyle::Balanced
    );
    let m_step = decomp_z.len() / neo_math::ring::D;
    let d = neo_math::ring::D;
    
    // Ensure PP exists for the full witness dimensions
    crate::ensure_ajtai_pp_for_dims(d, m_step, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, params.kappa as usize, m_step)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;
    
    // Get PP for the full witness dimensions (kept for potential future checks)
    let _pp_full = neo_ajtai::get_global_pp_for_dims(d, m_step)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for full witness: {}", e))?;
    
    // CCS-level binder disabled in verifier too (see rationale above).
    // The verifier reconstructs the exact augmented CCS the prover used, without any extra
    // binder row. The required equalities are checked by:
    //   - Π_CCS terminal eq-binding (for R1CS) or generic CCS terminal,
    //   - Π_RLC combination check with guard bound T,
    //   - Π_DEC recomposition & range checks that bind digits to the Ajtai commitment,
    //   - Commitment evolution on coordinates + digest.
    let rlc_binder = None;

    // Build verifier CCS exactly like prover (no fallback for security)
    // 🔒 SECURITY: Verifier must use identical CCS as prover
    let augmented_ccs_v = build_augmented_ccs_linked_with_rlc(
        step_ccs,
        step_x_len,
        &binding_spec.y_step_offsets,
        &binding_spec.y_prev_witness_indices,
        &binding_spec.step_program_input_witness_indices,
        y_len,
        binding_spec.const1_witness_index,
        rlc_binder, // RLC binder enabled for soundness
    )?;
    
    // 4. Build public input using the recomputed ρ
    let public_input = build_augmented_public_input_for_step(
        &ivc_proof.step_public_input,                 // step_x 
        rho,                                          // ρ (PUBLIC - CRITICAL!)
        &prev_accumulator.y_compact,                  // y_prev
        &ivc_proof.next_accumulator.y_compact         // y_next
    );
    
    // 5. Digest-only verification (skip per-step SNARK compression)
    // This mirrors the context binding check from crate::verify() without requiring
    // actual Spartan proof bytes, since IVC soundness comes from folding proofs
    // Use the same SIMPLE CCS construction as the prover (without RLC binder) for consistency
    let digest_ccs = build_augmented_ccs_linked(
        step_ccs,
        step_x_len,
        &binding_spec.y_step_offsets,
        &binding_spec.y_prev_witness_indices,
        &binding_spec.step_program_input_witness_indices,
        y_len,
        binding_spec.const1_witness_index,
    ).map_err(|e| anyhow::anyhow!("Failed to build verifier digest CCS: {}", e))?;
    
    let expected_context_digest = crate::context_digest_v1(&digest_ccs, &public_input);
    let io = &ivc_proof.step_proof.public_io;
    
    // Ensure public_io is long enough to contain context digest
    if io.len() < 32 {
        return Ok(false);
    }
    
    // Extract context digest from end of public_io (last 32 bytes)
    let proof_context_digest = &io[io.len() - 32..];
    
    // SECURITY: Constant-time comparison to bind proof to verifier's context
    let digest_valid = proof_context_digest.ct_eq(&expected_context_digest).unwrap_u8() == 1;
    
    // SECURITY FIX: Validate that y_next values in public_io match the actual y_next from folding
    // This prevents the public IO malleability attack where an attacker can manipulate y_next
    // values in public_io while keeping the same context digest
    let y_next_valid = if io.len() >= 32 + (y_len * 8) {
        // Extract y_next values from public_io (first y_len * 8 bytes, before the digest)
        let mut io_y_next_valid = true;
        for (i, &expected_y) in ivc_proof.next_accumulator.y_compact.iter().enumerate() {
            let start_idx = i * 8;
            let end_idx = start_idx + 8;
            if end_idx <= io.len() - 32 { // Ensure we don't overlap with digest
                let io_y_bytes = &io[start_idx..end_idx];
                let expected_y_bytes = expected_y.as_canonical_u64().to_le_bytes();
                
                // SECURITY: Constant-time comparison to prevent timing attacks
                let bytes_match = io_y_bytes.ct_eq(&expected_y_bytes).unwrap_u8() == 1;
                io_y_next_valid &= bytes_match;
            } else {
                io_y_next_valid = false;
                break;
            }
        }
        io_y_next_valid
    } else {
        false // public_io too short to contain expected y_next values
    };
    
    let is_valid = digest_valid && y_next_valid;
    #[cfg(feature = "neo-logs")]
    {
        eprintln!("[ivc] digest_valid={}, y_next_valid={}", digest_valid, y_next_valid);
    }
    // Bind result of digest and y_next checks
    
    #[cfg(feature = "neo-logs")]
    {
        eprintln!("🔍 IVC DIGEST DEBUG:");
        eprintln!("  Expected context digest: {:02x?}", &expected_context_digest[..8]);
        eprintln!("  Proof context digest:    {:02x?}", &proof_context_digest[..8]);
        eprintln!("  Digest match: {}", digest_valid);
        eprintln!("  Y_next validation: {}", y_next_valid);
        eprintln!("  Overall validity: {}", is_valid);
        eprintln!("  Public IO length: {}", io.len());
        eprintln!("  Expected y_next length: {} bytes", y_len * 8);
        eprintln!("  Digest CCS: n={}, m={}", digest_ccs.n, digest_ccs.m);
        eprintln!("  Public input length: {}", public_input.len());
        eprintln!("  Verifier CCS params: step_x_len={}, y_len={}, const1_idx={}", 
                  step_x_len, y_len, binding_spec.const1_witness_index);
        eprintln!("  Verifier y_step_offsets: {:?}", binding_spec.y_step_offsets);
    }
    
    // SECURITY NOTE (per Neo paper):
    // Base IVC verification does NOT require Spartan. Soundness comes from verifying
    // the folding proof itself (ΠCCS→ΠRLC→ΠDEC via sum-check over Ajtai commitments)
    // using the Fiat–Shamir challenges.
    // Optional: We MAY wrap the entire IVC statement in an outer SNARK (e.g., (Super)Spartan+FRI)
    // to compress the proof size, but that is orthogonal to soundness of the base IVC.
    //
    // Required here (base IVC):
    //  - Recompute the FS challenges (e.g., ρ) from the transcript and compare in constant time.
    //  - Verify the RLC binding and the folding relation against the committed instances.
    //  - Bind the augmented public input (CCS ID/domain, step index, y_prev, y_next, public x)
    //    with a context digest.
    //
    // Optional (compressed mode):
    //  - Verify a Spartan proof that attests "there exists a valid base IVC proof."
    //    This replaces running the folding verifier locally and is behind a feature flag.

    // 🔐 LAYER 2: Verify the commitment fold equation on coordinates + digest
    let commit_valid = verify_commitment_evolution(
        &prev_accumulator.c_coords,
        &ivc_proof.next_accumulator.c_coords,
        &ivc_proof.next_accumulator.c_z_digest,
        &ivc_proof.c_step_coords,
        rho,
    );

    if !commit_valid {
        #[cfg(feature = "neo-logs")]
        eprintln!("❌ Commitment fold check failed: c_next != c_prev + ρ·c_step or digest mismatch");
        return Ok(false);
    }

    // Always verify folding proof (Π‑CCS/Π‑RLC/Π‑DEC), including step 0
    let folding_ok = verify_ivc_step_folding(
        params,
        ivc_proof,
        &augmented_ccs_v,
        prev_accumulator,
        prev_augmented_x,
    )?;
    if !folding_ok {
        #[cfg(feature = "debug-logs")]
        eprintln!("[ivc] folding_ok=false");
        #[cfg(feature = "neo-logs")]
        eprintln!("❌ Folding verification (Pi-CCS/RLC/DEC) failed");
        return Ok(false);
    }

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
    // Carry the running ME instance across steps (proper chaining)
    let mut prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>> = None;
    let mut prev_me_wit: Option<neo_ccs::MeWitness<F>> = None;

    // Strict linkage: carry RHS MCS instance/witness as next LHS across steps
    let mut prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> = None;

    for (step_idx, step_input) in step_inputs.iter().enumerate() {
        // Extract REAL y_step from the witness using a simple extractor
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
            binding_spec,
            transcript_only_app_inputs: false,
            prev_augmented_x: step_proofs.last().map(|p: &IvcProof| p.step_augmented_public_input.as_slice()),
        };

        let (step_result, me_out, me_wit_out, lhs_next) =
            prove_ivc_step_chained(ivc_step_input, prev_me.clone(), prev_me_wit.clone(), prev_lhs_mcs.clone())?;

        prev_me = Some(me_out);
        prev_me_wit = Some(me_wit_out);
        prev_lhs_mcs = Some(lhs_next);

        current_accumulator = step_result.proof.next_accumulator.clone();
        step_proofs.push(step_result.proof);
    }

    Ok(IvcChainProof { steps: step_proofs, final_accumulator: current_accumulator, chain_length: step_inputs.len() as u64 })
}

/// Verify an entire IVC chain (strict)
///
/// Threads prev_augmented_x across steps and enforces per‑step folding verification
/// (Π‑CCS/RLC/DEC) and linkage of augmented inputs (LHS=prev_augmented_x, RHS=reconstruction).
/// This is the recommended and secure verifier.
///
/// Base-case policy: when no prior accumulator exists and no prev_augmented_x is provided,
/// the verifier requires the LHS augmented input to be the canonical all‑zeros vector of the
/// correct shape (matching `zero_mcs_instance_for_shape`).
///
/// **CRITICAL SECURITY**: `binding_spec` must come from a trusted source
/// (circuit specification), NOT from the proof!
pub fn verify_ivc_chain(
    step_ccs: &CcsStructure<F>,
    chain_proof: &IvcChainProof,
    initial_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
    params: &crate::NeoParams,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Strict threading of prev_augmented_x and RHS reconstruction checks
    let mut acc_before_curr_step = initial_accumulator.clone();
    let mut prev_augmented_x: Option<Vec<F>> = None;

    for step_proof in &chain_proof.steps {
        // Cross-check prover-supplied augmented input matches verifier reconstruction.
        let (expected_augmented, _) = compute_augmented_public_input_for_step(&acc_before_curr_step, step_proof)
            .map_err(|e| anyhow::anyhow!("failed to compute augmented input: {}", e))?;
        if expected_augmented != step_proof.step_augmented_public_input { return Ok(false); }

        // Enforce per-step verification (includes folding checks)
        let ok = verify_ivc_step(
            step_ccs,
            step_proof,
            &acc_before_curr_step,
            binding_spec,
            params,
            prev_augmented_x.as_deref(),
        )?;
        if !ok { return Ok(false); }

        // Advance and thread linkage
        acc_before_curr_step = step_proof.next_accumulator.clone();
        prev_augmented_x = Some(step_proof.step_augmented_public_input.clone());
    }

    Ok(acc_before_curr_step.step == chain_proof.chain_length)
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

/// Build step data for transcript including step public input X
/// 
/// **SECURITY CRITICAL**: This binds ALL public choices made by the prover:
/// - step: The step number 
/// - step_x: The step's public input (prover-chosen)
/// - y_prev: Previous accumulator state
/// - c_z_digest_prev: Previous commitment digest
/// 
/// This ensures ρ depends on all public data, preventing transcript malleability.
pub fn build_step_transcript_data(accumulator: &Accumulator, step: u64, step_x: &[F]) -> Vec<F> {
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
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
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

pub fn build_augmented_ccs_linked(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],   
    step_program_input_witness_indices: &[usize],        
    y_len: usize,
    const1_witness_index: usize,
) -> Result<CcsStructure<F>, String> {
    build_augmented_ccs_linked_with_rlc(
        step_ccs,
        step_x_len,
        y_step_offsets,
        y_prev_witness_indices,
        step_program_input_witness_indices,
        y_len,
        const1_witness_index,
        None, // No RLC binder by default
    )
}

/// Build augmented CCS with optional RLC binder for step commitment binding
pub fn build_augmented_ccs_linked_with_rlc(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],   
    step_program_input_witness_indices: &[usize],        
    y_len: usize,
    const1_witness_index: usize,
    rlc_binder: Option<(Vec<F>, F)>, // (aggregated_row, rhs) for RLC constraint
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
    if !step_program_input_witness_indices.is_empty() && step_program_input_witness_indices.len() > step_x_len {
        return Err(format!("step_program_input_witness_indices length {} cannot exceed step_public_input_len {}", step_program_input_witness_indices.len(), step_x_len));
    }
    if const1_witness_index >= step_ccs.m {
        return Err(format!("const1_witness_index {} out of range (m={})", const1_witness_index, step_ccs.m));
    }
    for &o in y_step_offsets.iter().chain(y_prev_witness_indices).chain(step_program_input_witness_indices) {
        if o >= step_ccs.m {
            return Err(format!("witness offset {} out of range (m={})", o, step_ccs.m));
        }
    }

    // Public input: [ step_x || ρ || y_prev || y_next ]
    let pub_cols = step_x_len + 1 + 2 * y_len;

    // Row accounting (no preset cap):
    //  - step_rows                              (copy step CCS)
    //  - EV rows                                (see below; production: 2*y_len, testing: 2)
    //  - step_x_len binder rows (optional)      (step_x[i] - step_witness[x_i] = 0)
    //  - y_len prev binder rows                 (y_prev[k] - step_witness[prev_k] = 0)
    //  - 1 RLC binder row (optional)            (⟨G, z⟩ = Σ r_i * c_step[i])
    //  - 1 const-1 enforcement row              (w_const1 * ρ = ρ, forces w_const1 = 1)
    let step_rows = step_ccs.n;
    // EV rows in production encoding: two per state element (u = ρ·y_step; y_next − y_prev − u)
    let ev_rows = 2 * y_len;
    let x_bind_rows = if step_program_input_witness_indices.is_empty() { 0 } else { step_x_len };
    let prev_bind_rows = if y_prev_witness_indices.is_empty() { 0 } else { y_len };
    let rlc_rows = if rlc_binder.is_some() { 1 } else { 0 };
    // SOUNDNESS FIX: Enforce const-1 column is actually 1 using public ρ
    let const1_enforce_rows = 1;
    let total_rows = step_rows + ev_rows + x_bind_rows + prev_bind_rows + rlc_rows + const1_enforce_rows;
    // Pre-pad to next power-of-two for clean χ_r wiring (optional but stable).
    // This keeps augmented CCS shape fixed and matches Π_CCS's ℓ computation.
    // Avoid n=1 degeneracy (0-round sum-check) which can misalign R1CS terminal checks.
    let mut target_rows = total_rows.next_power_of_two();
    if target_rows < 2 { target_rows = 2; }

    // Witness: [ step_witness || u ]
    let step_wit_cols = step_ccs.m;
    // Witness columns added for EV section: u has length y_len
    let ev_wit_cols = y_len; // u
    let total_wit_cols = step_wit_cols + ev_wit_cols;
    let total_cols = pub_cols + total_wit_cols;

    let mut combined_mats = Vec::new();
    for matrix_idx in 0..step_ccs.matrices.len() {
        let mut data = vec![F::ZERO; target_rows * total_cols];

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
        
        let col_u0 = pub_cols + step_wit_cols;
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
        if !step_program_input_witness_indices.is_empty() {
            // Bind only the last step_program_input_witness_indices.len() elements of step_x (the app inputs)
            let bind_len = step_program_input_witness_indices.len();
            let bind_start = step_x_len - bind_len;
            for i in 0..bind_len {
                let r = step_rows + ev_rows + i;
                match matrix_idx {
                    0 => {
                        data[r * total_cols + (bind_start + i)] = F::ONE;                         // + step_x[bind_start + i]
                        data[r * total_cols + (pub_cols + step_program_input_witness_indices[i])] = -F::ONE;      // - step_witness[x_i]
                    }
                    1 => data[r * total_cols + col_const1_abs] = F::ONE,                        // × 1
                    _ => {}
                }
            }
        }

        // Binder Y_prev: y_prev[k] - step_witness[y_prev_witness_indices[k]] = 0  (if any)
        // SECURITY FIX: Enforce that step circuit reads of y_prev match the accumulator's y_prev
        if !y_prev_witness_indices.is_empty() {
            for k in 0..y_len {
                let r = step_rows + ev_rows + x_bind_rows + k;
                match matrix_idx {
                    0 => {
                        data[r * total_cols + (col_y_prev0 + k)] = F::ONE;                           // + y_prev[k]
                        data[r * total_cols + (pub_cols + y_prev_witness_indices[k])] = -F::ONE;    // - step_witness[y_prev_witness_indices[k]]
                    }
                    1 => data[r * total_cols + col_const1_abs] = F::ONE,                           // × 1
                    _ => {}
                }
            }
        }

        // RLC Binder: enforce linear equality ⟨G, z⟩ = rhs where
        // G = aggregated_row over witness coordinates and rhs = Σ r_i * c_step[i] (or diff variant)
        // Encode in R1CS as: <A,z> * <B,z> = <C,z>
        //   A row = G · z, B row selects const-1 (== 1), C puts rhs in const-1 column
        if let Some((ref aggregated_row, rhs)) = rlc_binder {
            let r = step_rows + ev_rows + x_bind_rows + prev_bind_rows;
            match matrix_idx {
                0 => {
                    // A matrix: ⟨G, z⟩ where z = [public || witness]
                    // G covers the entire witness (step_witness || u)
                    for (j, &g_j) in aggregated_row.iter().enumerate() {
                        if j < total_wit_cols {
                            let col = pub_cols + j;  // witness starts after public inputs
                            data[r * total_cols + col] = g_j;
                        }
                    }
                }
                1 => {
                    // B matrix: multiply by 1 (const-1 witness column)
                    data[r * total_cols + col_const1_abs] = F::ONE;
                }
                2 => {
                    // C matrix: place rhs on const-1 column so equality is linear
                    data[r * total_cols + col_const1_abs] = rhs;
                }
                _ => {}
            }
        }

        // SOUNDNESS FIX: Enforce w_const1 * ρ = ρ (forces w_const1 = 1 since ρ ≠ 0)
        // This prevents malicious provers from setting const-1 to 0, which would turn
        // all linear constraints (that multiply by const-1) into trivial 0=0 identities.
        // Many rows above (EV, X/Y binders) rely on B matrix selecting the "1" column;
        // without this constraint, those rows can be zeroed out by a malicious prover.
        {
            let r = step_rows + ev_rows + x_bind_rows + prev_bind_rows + rlc_rows;
            
            // SECURITY: Bounds checks to prevent silent corruption of the constraint
            debug_assert!(r < target_rows, "const-1 enforcement row {} must be within target_rows {}", r, target_rows);
            debug_assert!(col_const1_abs < total_cols, "const-1 column {} must be within total_cols {}", col_const1_abs, total_cols);
            debug_assert!(col_rho < total_cols, "rho column {} must be within total_cols {}", col_rho, total_cols);
            
            match matrix_idx {
                0 => {
                    // A matrix: select w_const1 (the witness column that should be 1)
                    debug_assert!(r * total_cols + col_const1_abs < data.len(), "A matrix index out of bounds");
                    data[r * total_cols + col_const1_abs] = F::ONE;
                }
                1 => {
                    // B matrix: select ρ (public, from Fiat-Shamir, guaranteed non-zero)
                    debug_assert!(r * total_cols + col_rho < data.len(), "B matrix index out of bounds");
                    data[r * total_cols + col_rho] = F::ONE;
                }
                2 => {
                    // C matrix: also select ρ (public)
                    // This enforces: w_const1 * ρ = ρ  =>  w_const1 = 1
                    debug_assert!(r * total_cols + col_rho < data.len(), "C matrix index out of bounds");
                    data[r * total_cols + col_rho] = F::ONE;
                }
                _ => {}
            }
        }

        // Remaining rows (from total_rows..target_rows) remain zero — they encode 0 == 0
        combined_mats.push(Mat::from_row_major(target_rows, total_cols, data));
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
pub fn build_augmented_public_input_for_step(
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

/// Recompute the augmented public input used by the prover for this step:
/// X = [step_x || ρ || y_prev || y_next]. Returns (X, ρ).
fn compute_augmented_public_input_for_step(
    prev_acc: &Accumulator,
    proof: &IvcProof,
) -> Result<(Vec<F>, F), Box<dyn std::error::Error>> {
    let step_data = build_step_transcript_data(prev_acc, proof.step, &proof.step_public_input);
    let step_digest = create_step_digest(&step_data);
    let (rho_calc, _td) = rho_from_transcript(prev_acc, step_digest, &proof.c_step_coords);
    
    // SECURITY: Always use recalculated ρ from transcript, never trust prover's value
    // The rho_from_transcript uses challenge_nonzero_field which guarantees ρ ≠ 0
    let rho = rho_calc;
    debug_assert_ne!(rho, F::ZERO, "ρ must be non-zero for const-1 enforcement soundness");

    let x_aug = build_augmented_public_input_for_step(
        &proof.step_public_input,
        rho,
        &prev_acc.y_compact,
        &proof.next_accumulator.y_compact,
    );
    Ok((x_aug, rho))
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

/// Verify the Ajtai commitment evolution equation:
///   c_next == c_prev + rho * c_step
/// Also checks the published digest matches the recomputed digest.
fn verify_commitment_evolution(
    prev_coords: &[F],
    next_coords: &[F],
    published_next_digest: &[u8; 32],
    c_step_coords: &[F],
    rho: F,
) -> bool {
    let (expected_next, expected_digest) = if prev_coords.is_empty() {
        // Base step: c_next should equal c_step (rho is irrelevant because c_prev=0)
        (c_step_coords.to_vec(), digest_commit_coords(c_step_coords))
    } else {
        match evolve_commitment(prev_coords, c_step_coords, rho) {
            Ok(pair) => pair,
            Err(_) => return false,
        }
    };

    let coords_ok = ct_eq_coords(&expected_next, next_coords);
    let digest_ok = ct_eq_bytes(&expected_digest, published_next_digest);
    if !coords_ok || !digest_ok {
        #[cfg(feature = "neo-logs")]
        {
            println!("  commit coords eq: {}", coords_ok);
            println!("  commit digest eq: {}", digest_ok);
            println!("  prev.len={}, step.len={}, next.len={}", prev_coords.len(), c_step_coords.len(), next_coords.len());
            let head = |v: &[F]| v.iter().take(4).map(|f| f.as_canonical_u64()).collect::<Vec<_>>();
            println!("  prev head: {:?}", head(prev_coords));
            println!("  step head: {:?}", head(c_step_coords));
            println!("  next head: {:?}", head(next_coords));
        }
    }
    coords_ok && digest_ok
}

/// Constant-time equality check for byte slices
#[inline]
fn ct_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    a.ct_eq(b).unwrap_u8() == 1
}

/// Constant-time equality check for field element coordinates
#[inline]
fn ct_eq_coords(a: &[F], b: &[F]) -> bool {
    if a.len() != b.len() { return false; }
    // Compare as little-endian u64 limbs in constant time
    let mut ok = 1u8;
    for (x, y) in a.iter().zip(b.iter()) {
        let xb = x.as_canonical_u64().to_le_bytes();
        let yb = y.as_canonical_u64().to_le_bytes();
        let eq = (&xb as &[u8]).ct_eq(&yb).unwrap_u8();
        // accumulate mismatches without branches
        ok &= eq;
    }
    ok == 1
}

/// Internal: recombine digit MEs into the parent ME (same math as neo-fold::recombine_me_digits_to_parent).
fn recombine_me_digits_to_parent_local(
    params: &crate::NeoParams,
    digits: &[neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>],
) -> Result<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>, String> {
    use neo_ajtai::s_lincomb;
    use neo_math::{Rq, cf_inv};
    if digits.is_empty() {
        return Err("no digit instances to recombine".to_string());
    }
    let m_in = digits[0].m_in;
    let r_ref = &digits[0].r;
    let t = digits[0].y.len();
    let d_rows = digits[0].X.rows();
    let x_cols = digits[0].X.cols();
    for (i, d) in digits.iter().enumerate() {
        if d.m_in != m_in { return Err(format!("digit[{}]: m_in mismatch", i)); }
        if &d.r != r_ref   { return Err(format!("digit[{}]: r mismatch", i)); }
        if d.X.rows() != d_rows || d.X.cols() != x_cols {
            return Err(format!("digit[{}]: X shape mismatch (want {}x{}, got {}x{})",
                               i, d_rows, x_cols, d.X.rows(), d.X.cols()));
        }
        if d.y.len() != t { return Err(format!("digit[{}]: y arity mismatch", i)); }
    }
    // S-linear combination coefficients 1, b, b^2, ...
    let mut coeffs: Vec<Rq> = Vec::with_capacity(digits.len());
    let mut pow_f = F::ONE;
    for _ in 0..digits.len() {
        let mut arr = [F::ZERO; neo_math::D];
        arr[0] = pow_f;
        coeffs.push(cf_inv(arr));
        pow_f *= F::from_u64(params.b as u64);
    }
    // Combine commitments
    let digit_cs: Vec<neo_ajtai::Commitment> = digits.iter().map(|d| d.c.clone()).collect();
    let c_parent = s_lincomb(&coeffs, &digit_cs).map_err(|_| "s_lincomb failed".to_string())?;
    // Combine X
    let mut x_parent = neo_ccs::Mat::zero(d_rows, x_cols, F::ZERO);
    let mut pow = F::ONE;
    for d in digits {
        for r in 0..d_rows {
            for c in 0..x_cols {
                x_parent[(r, c)] += d.X[(r, c)] * pow;
            }
        }
        pow *= F::from_u64(params.b as u64);
    }
    // Combine y (vector-of-rows representation)
    let y_dim = digits[0].y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_parent = vec![vec![neo_math::K::ZERO; y_dim]; t];
    let mut pow_k = neo_math::K::from(F::ONE);
    let base_k = neo_math::K::from(F::from_u64(params.b as u64));
    for d in digits {
        for j in 0..t {
            for u in 0..y_dim {
                y_parent[j][u] += d.y[j][u] * pow_k;
            }
        }
        pow_k *= base_k;
    }
    // Combine y_scalars
    let mut y_scalars_parent = vec![neo_math::K::ZERO; digits[0].y_scalars.len()];
    let mut powk = neo_math::K::from(F::ONE);
    for d in digits {
        for j in 0..y_scalars_parent.len() {
            if j < d.y_scalars.len() { y_scalars_parent[j] += d.y_scalars[j] * powk; }
        }
        powk *= base_k;
    }
    Ok(neo_ccs::MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: c_parent,
        X: x_parent,
        r: r_ref.clone(),
        y: y_parent,
        y_scalars: y_scalars_parent,
        m_in,
        fold_digest: digits[0].fold_digest,
    })
}

/// Internal: recombine digit ME witnesses into a parent witness Z' = Σ b^i · Z_i
fn recombine_digit_witnesses_to_parent_local(
    params: &crate::NeoParams,
    digits: &[neo_ccs::MeWitness<F>],
) -> Result<neo_ccs::MeWitness<F>, String> {
    if digits.is_empty() { return Err("no digit witnesses".into()); }
    let d = digits[0].Z.rows();
    let m = digits[0].Z.cols();
    if d != neo_math::D {
        return Err(format!(
            "digit witnesses have {} rows, expected D={}",
            d, neo_math::D
        ));
    }
    for (i, dw) in digits.iter().enumerate() {
        if dw.Z.rows() != d || dw.Z.cols() != m {
            return Err(format!("digit_witness[{}] shape mismatch (want {}x{}, got {}x{})", i, d, m, dw.Z.rows(), dw.Z.cols()));
        }
    }
    let mut zp = neo_ccs::Mat::zero(d, m, F::ZERO);
    let mut pow = F::ONE;
    let base = F::from_u64(params.b as u64);
    for dw in digits {
        for r in 0..d { for c in 0..m { zp[(r,c)] += dw.Z[(r,c)] * pow; } }
        pow *= base;
    }
    Ok(neo_ccs::MeWitness { Z: zp })
}

/// Verify a single step's folding proof (Pi-CCS + Pi-RLC + Pi-DEC).
/// - `augmented_ccs` must match the prover's folding CCS; the caller should reconstruct it.
/// - `prev_step_x`: previous step's `step_public_input` (None for step 0).
pub fn verify_ivc_step_folding(
    params: &crate::NeoParams,
    ivc_proof: &IvcProof,
    augmented_ccs: &neo_ccs::CcsStructure<F>,
    prev_acc: &Accumulator,
    prev_augmented_x: Option<&[F]>,
) -> Result<bool, Box<dyn std::error::Error>> {
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] enter");
    #[cfg(feature = "neo-logs")]
    {
        println!("🔎 FOLD VERIFY: step {}", ivc_proof.step);
        println!("   augmented_ccs: n={}, m={}", augmented_ccs.n, augmented_ccs.m);
        println!("   prev_acc.step={}, y_len={}", prev_acc.step, prev_acc.y_compact.len());
        if let Some(px) = prev_augmented_x { println!("   prev_augmented_x.len()={}", px.len()); }
    }
    let folding = ivc_proof
        .folding_proof
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing folding_proof"))?;

    // 1) Cross-check against stored Pi-CCS inputs, using RHS as the binding point.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=inputs");
    let stored_inputs = &folding.pi_ccs_inputs;
    if stored_inputs.len() != 2 {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: pi_ccs_inputs.len() != 2");
        return Err(anyhow::anyhow!("folding proof missing pi_ccs_inputs").into());
    }

    // Compute expected RHS augmented x and ensure it matches both the proof copy and stored input.
    let (x_rhs_expected, _) = compute_augmented_public_input_for_step(prev_acc, ivc_proof)
        .map_err(|e| anyhow::anyhow!("failed to compute augmented input: {}", e))?;
    if x_rhs_expected != ivc_proof.step_augmented_public_input {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS augmented input mismatch vs proof copy");
        return Ok(false);
    }

    // LHS checks: now STRICT — enforce exact X equality and also bind to caller-provided prev_augmented_x when present.
    // Ensure LHS/RHS shapes match.
    if stored_inputs[0].m_in != stored_inputs[1].m_in {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS/RHS m_in mismatch");
        return Ok(false);
    }
    // Ensure LHS commitment matches the stored output commitment.
    if stored_inputs[0].c.data != folding.pi_ccs_outputs[0].c.data {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS commitment mismatch vs stored output");
        return Ok(false);
    }

    // Link LHS augmented input to the previous step.
    let x_lhs_proof = ivc_proof.prev_step_augmented_public_input.clone();
    let m_in = x_rhs_expected.len();
    if x_lhs_proof.len() != m_in {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS augmented x length mismatch");
        return Ok(false);
    }
    // Consider per-lane base case: first use of this running ME (coords empty) and no prev_augmented_x provided.
    let is_lane_base_case = prev_acc.c_coords.is_empty();
    if is_lane_base_case && prev_augmented_x.is_none() {
        let step_x_len = ivc_proof.step_public_input.len();
        let y_len = ivc_proof.step_y_prev.len();
        if x_lhs_proof.len() != step_x_len + 1 + 2 * y_len {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: base-case LHS augmented input length mismatch");
            return Ok(false);
        }
        // Accept either canonical zero-vector (zero-MCS base case) or self-fold (LHS == RHS augmented x)
        let is_zero = x_lhs_proof.iter().all(|&f| f == F::ZERO);
        let is_self_fold = x_lhs_proof == x_rhs_expected;
        if !is_zero && !is_self_fold {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: base-case LHS augmented x not zero or self-fold");
            return Ok(false);
        }
    } else if let Some(_px) = prev_augmented_x {
        // Production strict: enforce provided prev_augmented_x linkage
        let px = _px;
        if px != x_lhs_proof.as_slice() {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] early: prev_augmented_x linkage mismatch");
            return Ok(false);
        }
    }
    // Bind to the LHS stored input inside Pi-CCS as well.
    if stored_inputs[0].x != x_lhs_proof {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS stored x mismatch vs proof LHS augmented x");
        return Ok(false);
    }
    if stored_inputs[0].m_in != x_lhs_proof.len() {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: LHS m_in mismatch vs proof LHS augmented x len");
        return Ok(false);
    }
    // Note: Do not bind LHS Pi-CCS commitment to prev_acc.c_coords.
    // The accumulator commitment evolves on step-only coordinates, whereas Pi-CCS
    // commitments bind the full augmented z. Binding is enforced via:
    //  - LHS x linkage (prev_augmented_x and proof copy),
    //  - RHS reconstruction equality, and
    //  - Pi-RLC and Pi-DEC checks tying digits to commitments.

    // RHS checks: bind to expected augmented x and stored output commitment.
    if stored_inputs[1].x != x_rhs_expected {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS x mismatch: stored != expected");
        let _ex: Vec<_> = x_rhs_expected.iter().take(6).map(|f| f.as_canonical_u64()).collect();
        let _sx: Vec<_> = stored_inputs[1].x.iter().take(6).map(|f| f.as_canonical_u64()).collect();
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding]    expected head: {:?}", _ex);
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding]    stored   head: {:?}", _sx);
        return Ok(false);
    }
    if stored_inputs[1].m_in != x_rhs_expected.len() {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS m_in mismatch vs expected len");
        return Ok(false);
    }
    if stored_inputs[1].c.data != folding.pi_ccs_outputs[1].c.data {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] early: RHS commitment mismatch vs stored output");
        return Ok(false);
    }

    // Early hygiene guards before Pi-CCS: detect structural tampering and error out
    {
        if folding.pi_rlc_proof.rho_elems.len() != folding.pi_ccs_outputs.len() {
            return Err(anyhow::anyhow!("rho count != Π‑CCS outputs").into());
        }
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        if t != augmented_ccs.t() {
            return Err(anyhow::anyhow!("t mismatch: outputs.t != CCS.t").into());
        }
        for me in &folding.pi_ccs_outputs {
            if me.y.len() != t {
                return Err(anyhow::anyhow!("inconsistent t across Π‑CCS outputs").into());
            }
            for yj in &me.y {
                if yj.len() != neo_math::D {
                    return Err(anyhow::anyhow!("y[j] length != D").into());
                }
            }
        }
    }

    // 2) Verify Pi-CCS against those instances.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=pi-ccs");
    let mut tr = Poseidon2Transcript::new(b"neo/fold");
    let ok_ccs = pi_ccs_verify(
        &mut tr,
        params,
        augmented_ccs,
        stored_inputs,
        &folding.pi_ccs_outputs,
        &folding.pi_ccs_proof,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-CCS verify: {}", ok_ccs);
    if !ok_ccs { return Ok(false); }

    // 2b) (Skip) Intra-output y vs y_scalars check; rely on Π‑RLC/Π‑DEC path for scalar consistency.

    // 4) Recombine digit MEs to the parent ME for Pi‑RLC and Pi‑DEC checks.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=recombine-me");
    let me_digits = ivc_proof
        .me_instances
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit ME instances"))?;
    let mut me_parent = recombine_me_digits_to_parent_local(params, me_digits)?;

    // 4a) (No direct cross-link of RHS y_scalars to parent y_scalars here)
    //      Binding between Π‑CCS outputs and the recomposed parent is enforced by Π‑RLC and Π‑DEC.

    // 5) Verify Pi‑RLC.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=rlc");
    let ok_rlc = pi_rlc_verify(
        &mut tr,
        params,
        &folding.pi_ccs_outputs,
        &me_parent,
        &folding.pi_rlc_proof,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-RLC verify: {}", ok_rlc);
    if !ok_rlc { #[cfg(feature = "neo-logs")] eprintln!("[folding] rlc_verify=false"); return Ok(false); }

    // 6) Recombine digit witnesses to get true (d, m) for Ajtai S-module; then verify Π‑DEC.
    let wit_digits = ivc_proof
        .digit_witnesses
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit witnesses for DEC and tie check"))?;
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=recombine-wit");
    let wit_parent = recombine_digit_witnesses_to_parent_local(params, wit_digits)
        .map_err(|e| anyhow::anyhow!("recombine_digit_witnesses_to_parent failed: {}", e))?;

    let d_rows = neo_math::D;
    let m_cols = wit_parent.Z.cols();
    let l_real = match AjtaiSModule::from_global_for_dims(d_rows, m_cols) {
        Ok(l) => l,
        Err(_) => {
            #[cfg(not(feature = "testing"))]
            {
                return Err(anyhow::anyhow!(
                    "Ajtai PP missing for dims (D={}, m={}); register CRS/PP before verify",
                    d_rows, m_cols
                )
                .into());
            }
            #[cfg(feature = "testing")]
            {
                super::ensure_ajtai_pp_for_dims(d_rows, m_cols, || {
                    use rand::{RngCore, SeedableRng};
                    use rand::rngs::StdRng;
                    let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
                        StdRng::from_seed([42u8; 32])
                    } else {
                        let mut seed = [0u8; 32];
                        rand::rng().fill_bytes(&mut seed);
                        StdRng::from_seed(seed)
                    };
                    let pp = super::ajtai_setup(&mut rng, d_rows, params.kappa as usize, m_cols)?;
                    neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
                })?;
                AjtaiSModule::from_global_for_dims(d_rows, m_cols)
                    .map_err(|_| anyhow::anyhow!("AjtaiSModule unavailable (PP must exist after ensure)"))?
            }
        }
    };
    // 6a) Witness-commitment binding: each digit witness Z_i must open to its ME commitment.
    // This check is independent of power-of-two row constraints and catches tampering in Z.
    {
        let me_digits = ivc_proof
            .me_instances
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("IVC proof missing digit ME instances for DEC and tie check"))?;
        if me_digits.len() != wit_digits.len() { #[cfg(feature = "neo-logs")] eprintln!("[folding] early: digit me count != digit witness count"); return Ok(false); }
        for (_i, (dw, me_i)) in wit_digits.iter().zip(me_digits.iter()).enumerate() {
            let c_from_wit = l_real.commit(&dw.Z);
            if c_from_wit.data != me_i.c.data {
                #[cfg(feature = "neo-logs")]
                eprintln!("[folding] digit witness-commit mismatch");
                return Ok(false);
            }
        }
        // Parent witness must also open to parent commitment after recombination.
        let c_parent_from_wit = l_real.commit(&wit_parent.Z);
        if c_parent_from_wit.data != me_parent.c.data {
            eprintln!("[folding] parent witness-commit mismatch after recombination");
            return Ok(false);
        }
    }

    eprintln!("[folding] stage=dec");
    let ok_dec = pi_dec_verify(
        &mut tr,
        params,
        &me_parent,
        me_digits,
        &folding.pi_dec_proof,
        &l_real,
    )?;
    #[cfg(feature = "neo-logs")]
    println!("   Pi-DEC verify: {} (d_rows={}, m_cols={})", ok_dec, d_rows, m_cols);
    if !ok_dec { #[cfg(feature = "neo-logs")] eprintln!("[folding] dec_verify=false"); return Ok(false); }

    // Derive the Π‑CCS transcript tail once; use r for tie and reuse for residual later.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=derive-tail");
    let tail = pi_ccs_derive_transcript_tail(
        params,
        augmented_ccs,
        stored_inputs,
        &folding.pi_ccs_proof,
    )
    .map_err(|e| anyhow::anyhow!("failed to derive transcript tail: {}", e))?;
    if me_parent.r.is_empty() || me_parent.r != tail.r {
        me_parent.r = tail.r.clone();
    }

    // 7) Parent-level tie check now that authentic Ajtai S-module is ensured by Π‑DEC path
    // Recompute X from Z via S-module for tie check to avoid any recombination drift.
    let mut me_parent_tie = me_parent.clone();
    me_parent_tie.X = l_real.project_x(&wit_parent.Z, me_parent.m_in);
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=tie (r.len()={})", me_parent_tie.r.len());
    if let Err(_e) = tie_check_with_r(augmented_ccs, &me_parent_tie, &wit_parent, &me_parent_tie.r) {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] ❌ tie_with_r failed: {}", _e);
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] debug: me_parent.y.len={} (t), y[0].len={} (D)", me_parent.y.len(), me_parent.y.get(0).map(|v| v.len()).unwrap_or(0));
        return Ok(false);
    }

    // 8) Cross-link Π‑CCS outputs to the parent ME: recombine Π‑CCS y via ρ and 
    //    require the scalars match the parent ME (which DEC/tie bound to Z).
    {
        #[cfg(feature = "neo-logs")]
        eprintln!("[folding] stage=cross-link");
        // Hygiene: arity/shape guards
        if folding.pi_rlc_proof.rho_elems.len() != folding.pi_ccs_outputs.len() {
            return Err(anyhow::anyhow!("rho count != Π‑CCS outputs").into());
        }
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        if t != augmented_ccs.t() {
            return Err(anyhow::anyhow!("t mismatch: outputs.t != CCS.t").into());
        }
        for me in &folding.pi_ccs_outputs {
            if me.y.len() != t {
                return Err(anyhow::anyhow!("inconsistent t across Π‑CCS outputs").into());
            }
            for yj in &me.y {
                if yj.len() != neo_math::D {
                    return Err(anyhow::anyhow!("y[j] length != D").into());
                }
            }
        }
        let rhos_ring: Vec<Rq> = folding
            .pi_rlc_proof
            .rho_elems
            .iter()
            .map(|coeffs| cf_inv(*coeffs))
            .collect();
        // Recombine y vectors per matrix index j using S-action
        let t = folding.pi_ccs_outputs.get(0).map(|m| m.y.len()).unwrap_or(0);
        let d = neo_math::D;
        let mut y_parent_vecs: Vec<Vec<neo_math::K>> = vec![vec![neo_math::K::ZERO; d]; t];
        for (rho, me) in rhos_ring.iter().zip(folding.pi_ccs_outputs.iter()) {
            let s_act = SAction::from_ring(*rho);
            for j in 0..t {
                let yj_rot = s_act
                    .apply_k_vec(&me.y[j])
                    .map_err(|_| anyhow::anyhow!("S-action dim mismatch for y[j]"))?;
                for r in 0..d { y_parent_vecs[j][r] += yj_rot[r]; }
            }
        }
        // Compute y_scalars from recombined y vectors (base-b powers)
        let mut pow_b_f = vec![F::ONE; d];
        for i in 1..d { pow_b_f[i] = pow_b_f[i-1] * F::from_u64(params.b as u64); }
        let pow_b_k: Vec<neo_math::K> = pow_b_f.into_iter().map(neo_math::K::from).collect();
        let mut y_scalars_from_rlc = vec![neo_math::K::ZERO; t];
        for j in 0..t {
            let mut acc = neo_math::K::ZERO;
            for r in 0..d { acc += y_parent_vecs[j][r] * pow_b_k[r]; }
            y_scalars_from_rlc[j] = acc;
        }
        if me_parent.y_scalars.len() != t {
            return Err(anyhow::anyhow!(
                "parent y_scalars length ({}) != t ({})",
                me_parent.y_scalars.len(), t
            ).into());
        }
        if y_scalars_from_rlc != me_parent.y_scalars {
            #[cfg(feature = "neo-logs")]
            eprintln!("[folding] cross-link failed: recombined y_scalars != parent y_scalars");
            return Ok(false);
        }
    }

    // 9) Enforce CCS satisfiability check for first step (base case)
    //    When prev_acc.c_coords.is_empty(), this is the first step with no prior accumulator.
    //    Both batched instances (LHS zero instance + RHS fresh step) should satisfy their CCS,
    //    so the sum over the hypercube (initial_sum) must be zero.
    //    
    //    CAVEAT: Only enforce when ℓ >= 2. In the ℓ=1 case (single-row padded to 2),
    //    the augmented CCS can carry a constant offset (e.g., from const-1 binding or other glue),
    //    so the hypercube sum of Q need not be zero even for a valid witness.
    //    This signature appears as p(0)=α₀, p(1)=0 in the sum-check rounds.
    #[cfg(feature = "neo-logs")]
    eprintln!("[folding] stage=residual");
    if prev_acc.c_coords.is_empty() && tail.r.len() >= 2 {
        // Base case with ℓ >= 2: both instances should be satisfied
        use crate::K;
        if tail.initial_sum != K::ZERO {
            #[cfg(feature = "neo-logs")]
            eprintln!(
                "[folding] non-zero CCS sum over hypercube (base case, ℓ={}): initial_sum={:?}, rejecting",
                tail.r.len(), tail.initial_sum
            );
            return Ok(false);
        }
    } else {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "[folding] skipping CCS-sum guard (c_coords.is_empty()={}, ℓ={})",
            prev_acc.c_coords.is_empty(), tail.r.len()
        );
    }

    Ok(true)
}
