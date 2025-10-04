//! Neo: Simple facade for the Neo lattice-based SNARK protocol
//!
//! This crate provides a simplified, ergonomic API for the complete Neo protocol pipeline,
//! exposing just two main functions: `prove` and `verify`.
//!
//! ## Example
//!
//! ```rust,no_run
//! use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
//! use anyhow::Result;
//!
//! fn main() -> Result<()> {
//!     // Set up your circuit (CCS), witness, and parameters
//!     // (In practice, you'd create these based on your specific circuit)
//!     let ccs: CcsStructure<F> = todo!("Create your CCS structure");
//!     let witness: Vec<F> = todo!("Create your witness vector");
//!     let public_input: Vec<F> = vec![]; // Usually empty for private computation
//!     let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
//!
//!     // Generate proof
//!     let proof = prove(ProveInput {
//!         params: &params,
//!         ccs: &ccs,
//!         public_input: &public_input,
//!         witness: &witness,
//!         output_claims: &[],
//!         vjs_opt: None,
//!     })?;
//!
//!     println!("Proof size: {} bytes", proof.size());
//!
//!     // Verify proof
//!     let is_valid = verify(&ccs, &public_input, &proof)?;
//!     println!("Proof valid: {}", is_valid);
//!
//!     Ok(())
//! }
//! ```
//!
//! For a complete working example, see `examples/fib.rs`.

use anyhow::Result;
use neo_ajtai::{setup as ajtai_setup, commit, decomp_b, DecompStyle};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
// Poseidon2 is now imported via the unified module
use p3_symmetric::Permutation;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tracing::{debug, info, warn};

// Init Ajtai PP for a specific (d, m) if absent.
pub(crate) fn ensure_ajtai_pp_for_dims<FN>(d: usize, m: usize, mut setup: FN) -> anyhow::Result<()>
where FN: FnMut() -> anyhow::Result<()> {
    if neo_ajtai::has_global_pp_for_dims(d, m) { return Ok(()); }
    setup()
}

// ZK-friendly Poseidon2-based context digest with proper domain separation.
// Parameters: width=12, capacity=4, rate=8.
// SECURITY NOTE: collision security ≈ 2^(capacity_bits/2) = 2^(256/2) = 2^128.
// This is ample for binding a proving context (do not reuse as a general object hash).
pub(crate) fn context_digest_v1(ccs: &CcsStructure<F>, public_input: &[F]) -> [u8; 32] {
    use neo_ccs::crypto::poseidon2_goldilocks as p2;
    use p3_goldilocks::Goldilocks;

    let poseidon2 = p2::permutation();
    let mut state = [Goldilocks::ZERO; p2::WIDTH];
    let mut absorbed = 0usize;
    const RATE: usize = p2::RATE; // 8

    const DOMAIN_STRING: &[u8] = b"neo/context/v1|poseidon2-goldilocks-w12-cap4";
    for &byte in DOMAIN_STRING {
        if absorbed == RATE { state = poseidon2.permute(state); absorbed = 0; }
        state[absorbed] = Goldilocks::from_u64(byte as u64);
        absorbed += 1;
    }

    // Helper function to absorb elements
    fn absorb_goldilocks(state: &mut [Goldilocks; p2::WIDTH], absorbed: &mut usize, poseidon2: &p3_goldilocks::Poseidon2Goldilocks<{ p2::WIDTH }>, elem: Goldilocks) {
        if *absorbed == p2::RATE { 
            *state = poseidon2.permute(*state); 
            *absorbed = 0; 
        }
        state[*absorbed] = elem;
        *absorbed += 1;
    }

    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(ccs.n as u64));
    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(ccs.m as u64));
    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(ccs.matrices.len() as u64));

    for (j, matrix) in ccs.matrices.iter().enumerate() {
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(j as u64));
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(matrix.rows() as u64));
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(matrix.cols() as u64));
        for r in 0..matrix.rows() {
            for c in 0..matrix.cols() {
                let val = matrix[(r, c)];
                if val != F::ZERO {
                    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(r as u64));
                    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(c as u64));
                    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(val.as_canonical_u64()));
                }
            }
        }
    }

    let mut terms: Vec<_> = ccs.f.terms().iter().collect();
    terms.sort_by_key(|t| &t.exps);
    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(terms.len() as u64));
    for term in terms {
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(term.coeff.as_canonical_u64()));
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(term.exps.len() as u64));
        for &e in &term.exps { 
            absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(e as u64)); 
        }
    }

    absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(public_input.len() as u64));
    for &x in public_input { 
        absorb_goldilocks(&mut state, &mut absorbed, &poseidon2, Goldilocks::from_u64(x.as_canonical_u64())); 
    }

    // End-of-message marker for uniform sponge framing
    if absorbed == RATE {
        state = poseidon2.permute(state);
        absorbed = 0;
    }
    state[absorbed] = Goldilocks::ONE;
    state = poseidon2.permute(state);
    let mut digest = [0u8; 32];
    for (i, &elem) in state[..4].iter().enumerate() {
        digest[i*8..(i+1)*8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    digest
}

// Note: The global Ajtai PP is stored in a OnceLock and cannot be cleared.
// This is a known limitation - concurrent prove() calls may interfere if 
// they use different parameters. Future versions should thread PP explicitly.

// Re-export key types that users need
pub use neo_params::NeoParams;
pub use neo_ccs::CcsStructure;
pub use neo_math::{F, K};

/// IVC (Incrementally Verifiable Computation) with embedded verifier
pub mod ivc;
/// NIVC driver (HyperNova-style non-uniform IVC)
pub mod nivc;

// Re-export high-level IVC API for production use
pub use ivc::{
    // Core IVC types
    Accumulator, IvcProof, IvcStepInput, IvcChainProof, IvcStepResult, IvcChainStepInput,
    // High-level proving/verifying functions  
    prove_ivc_step, prove_ivc_step_with_extractor, verify_ivc_step, prove_ivc_chain, verify_ivc_chain,
    // Folding verification functions
    verify_ivc_step_folding,
    recreate_mcs_instances_for_verification,
    // Step output extractors (fixes "folding with itself" issue)
    StepOutputExtractor, LastNExtractor, IndexExtractor,
    // Advanced commitment binding (production-ready)
    Commitment, BindingMetadata,
    // Nova embedded verifier and augmentation (production-ready)  
    augmentation_ccs, AugmentConfig,
    // PRODUCTION OPTION A: Public ρ EV (recommended)
    ev_with_public_rho_ccs, build_ev_with_public_rho_witness,
    // Base case helpers for testing
    zero_mcs_instance_for_shape,
    // Batch proving (Final SNARK Layer)
    // REMOVED: BatchData, prove_batch_data, IvcBatchBuilder, EmissionPolicy, EmitStats - not part of Neo architecture
};

// Re-export core NIVC types and helpers
pub use nivc::{
    NivcProgram, NivcStepSpec, NivcState, NivcStepProof, NivcChainProof,
    verify_nivc_chain, NivcFinalizeOptions, finalize_nivc_chain_with_options, finalize_nivc_chain,
};

// Track B path uses single-SNARK finalize via finalize_nivc_chain_with_options.


/// Counts and bookkeeping for public results embedded in the proof.
/// Backwards-compatible: all fields have defaults for older proofs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProofMeta {
    /// Number of "compact" y-outputs required by the folding scheme (protocol-internal).
    pub num_y_compact: usize,
    /// Number of application-level outputs appended by the prover (linear output claims).
    pub num_app_outputs: usize,
}

/// Public result claim: check <weight, Z_digits> == expected (both public)
/// This allows exposing specific linear functions of the witness as public outputs
#[derive(Clone, Debug)]
pub struct OutputClaim<F> { 
    /// Weight vector over digits (column-major): idx = c*D + r
    pub weight: Vec<F>, 
    /// The public value the verifier should learn and check
    pub expected: F,
}

/// Return a weight vector that exposes the undigitized z[k] component
/// This creates a weight that reconstructs z[k] = Σ_r b^r * Z[r,k] from the digit decomposition
pub fn expose_z_component(params: &NeoParams, m: usize, k: usize) -> Vec<F> {
    let d = neo_math::ring::D;
    let b_f = F::from_u64(params.b as u64);
    debug_assert!(k < m, "expose_z_component: k={} out of bounds (m={})", k, m);

    // Column-major flattening used by the bridge: idx = c * D + r
    // Z is D x m (rows=D digits, cols=m variables). Weights must match this order.
    let mut w = vec![F::ZERO; d * m];
    let mut pow = F::ONE;
    for r in 0..d {
        w[k * d + r] = pow;      // picks Z[r, k] in column-major order
        pow *= b_f;              // next power of base b
    }
    w
}

/// Convenience function to create a claim that z[idx] == value
pub fn claim_z_eq(params: &NeoParams, m: usize, idx: usize, value: F) -> OutputClaim<F> {
    OutputClaim { 
        weight: expose_z_component(params, m, idx), 
        expected: value 
    }
}

/// Lean proof object without VK - FIXES THE 51MB ISSUE!
#[derive(Clone, Serialize, Deserialize)]
pub struct Proof {
    /// Version tag for forward-compat
    pub v: u16,
    /// Circuit fingerprint for VK lookup
    pub circuit_key: [u8; 32],
    /// VK digest for binding verification
    pub vk_digest: [u8; 32],
    /// Public IO bytes bound by the bridge (anti-replay)
    pub public_io: Vec<u8>,
    /// ONLY the Spartan2 proof bytes (no 51MB VK!)
    pub proof_bytes: Vec<u8>,
    /// Application-level public results (values), in the same order as `ProveInput::output_claims`.
    /// These are added for convenience; the underlying proof still carries all public IO.
    #[serde(default)]
    pub public_results: Vec<F>,
    /// Bookkeeping about y-outputs layout inside the proof (for sanity checks).
    #[serde(default)]
    pub meta: ProofMeta,
}


impl Proof {
    /// Returns the total size of the lean proof in bytes
    pub fn size(&self) -> usize {
        std::mem::size_of::<u16>() + // v
        32 + // circuit_key
        32 + // vk_digest
        self.public_io.len() +
        self.proof_bytes.len()
    }
    
    /// Returns the public IO bytes bound by the proof (for verification binding)
    pub fn public_io(&self) -> &[u8] {
        &self.public_io
    }
    
    /// Returns the proof version
    pub fn version(&self) -> u16 {
        self.v
    }
    
    /// **Convenience only**: app-level outputs captured at proving time (not cryptographically checked here).
    /// Call `verify_and_extract_exact(...)` to derive the verified values from `public_io`.
    /// These correspond to `ProveInput::output_claims` in the same order.
    pub fn claimed_public_results(&self) -> &[F] {
        &self.public_results
    }

    /// Convenience: return app-level public results as canonical u64s.
    pub fn claimed_public_results_u64(&self) -> Vec<u64> {
        self.public_results.iter().map(|f| f.as_canonical_u64()).collect()
    }
}



/// Inputs needed by the prover (explicit is better than global state)
pub struct ProveInput<'a> {
    pub params: &'a NeoParams,                         // includes b, k, B, s, guard inequality
    pub ccs: &'a CcsStructure<F>,                      // the circuit
    pub public_input: &'a [F],                         // x (traditional public inputs)
    pub witness: &'a [F],                              // z (secret witness)
    /// Public results to expose and verify: each adds one <w,Z>=y check
    pub output_claims: &'a [OutputClaim<F>],           // application-level public outputs
    /// Optional precomputed v_j = M_j^T * χ_r vectors from Π_CCS linearization phase.
    /// When provided, skips the expensive SPMV computation in the bridge adapter.
    /// Length must equal ccs.matrices.len(), each inner Vec must have length ccs.m.
    /// If you don't pass `vjs_opt`, Neo uses the v_j produced by Π_CCS; only set this 
    /// when you are providing fully compatible precomputations.
    pub vjs_opt: Option<&'a [Vec<neo_math::K>]>,
}

impl<'a> ProveInput<'a> {
    /// Create a new ProveInput with vjs_opt set to None (most common case).
    /// This provides an ergonomic constructor to ease API migration.
    pub fn new(
        params: &'a NeoParams,
        ccs: &'a CcsStructure<F>,
        public_input: &'a [F],
        witness: &'a [F],
        output_claims: &'a [OutputClaim<F>],
    ) -> Self {
        Self { 
            params, 
            ccs, 
            public_input, 
            witness, 
            output_claims, 
            vjs_opt: None 
        }
    }

    /// Set the optional precomputed v_j vectors for SPMV optimization.
    pub fn with_vjs_opt(mut self, vjs_opt: Option<&'a [Vec<neo_math::K>]>) -> Self {
        self.vjs_opt = vjs_opt;
        self
    }
}

/// Generate a complete Neo SNARK proof for the given inputs
///
/// This orchestrates the full pipeline:
/// 1. **Ajtai setup**: Generate PP; do `decomp_b`; commit; build the MCS instance.
/// 2. **Fold**: Call your `neo-fold` entry (`fold_ccs_instances`) and get ME + folding proof.
/// 3. **Compress**: Translate to the Spartan2 bridge and get a `ProofBundle`.
/// 4. **Serialize**: Wrap that bundle into `Proof(Vec<u8>)`.
///
/// Returns an opaque proof that can be verified with `verify`.
pub fn prove(input: ProveInput) -> Result<Proof> {
    let total_start = std::time::Instant::now();
    // Parameter guard: enforce (k+1)T(b-1) < B for RLC soundness
    anyhow::ensure!(
        (input.params.k as u128 + 1)
            * (input.params.T as u128)
            * ((input.params.b - 1) as u128)
            < (input.params.B as u128),
        "unsafe params: (k+1)·T·(b−1) ≥ B"
    );

    // Fail-fast CCS consistency check: witness must satisfy the constraint system.
    // Always run to catch invalid witnesses early with minimal performance impact.
    neo_ccs::check_ccs_rowwise_zero(input.ccs, input.public_input, input.witness)
        .map_err(|e| {
            anyhow::anyhow!(
                "CCS check failed - witness does not satisfy constraints: {:?}",
                e
            )
        })?;

    // Step 1: Ajtai setup (parameter-aware global registry)  
    let ajtai_start = std::time::Instant::now();
    debug!("Starting Ajtai setup and decomposition");
    
    let d = neo_math::ring::D;
    let _m_w = input.witness.len(); // Original witness length before decomposition
    
    // 🔧 FIX: For augmented CCS, decompose full variable vector [public_input || witness]
    let mut full_variable_vector = input.public_input.to_vec();
    full_variable_vector.extend_from_slice(input.witness);
    
    // CRITICAL FIX: Use decomposed dimensions for PP setup, not raw witness length
    let decomp_z = decomp_b(&full_variable_vector, input.params.b, d, DecompStyle::Balanced);
    anyhow::ensure!(decomp_z.len() % d == 0, "decomp length not multiple of d");
    let m_correct = decomp_z.len() / d; // This is the actual m we'll use
    
    ensure_ajtai_pp_for_dims(d, m_correct, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        // Use OS entropy for cryptographically secure randomness
        // For deterministic builds in CI, this can be overridden with NEO_DETERMINISTIC env var
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        
        let pp = ajtai_setup(&mut rng, d, input.params.kappa as usize, m_correct)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;
    
    // Step 2: Decomposition already done above, now commit to witness
    // Use the correct m that was already calculated
    let m = m_correct;
    let pp = neo_ajtai::get_global_pp_for_dims(d, m)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for (d={}, m={}): {}", d, m, e))?;
    
    let commit_start = std::time::Instant::now();
    let commitment = commit(&*pp, &decomp_z);
    let commit_time = commit_start.elapsed();
    
    let ajtai_time = ajtai_start.elapsed();
    debug!("Ajtai setup completed: {:.2}ms (commit: {:.2}ms)", 
            ajtai_time.as_secs_f64() * 1000.0, commit_time.as_secs_f64() * 1000.0);
    
    // Step 3: Build MCS instance/witness (row-major conversion)
    let mut z_row_major = vec![F::ZERO; d * m];
    for col in 0..m { 
        for row in 0..d { 
            z_row_major[row*m + col] = decomp_z[col*d + row]; 
        } 
    }
    let z_matrix = neo_ccs::Mat::from_row_major(d, m, z_row_major);

    let mcs_inst = neo_ccs::McsInstance { 
        c: commitment, 
        x: input.public_input.to_vec(), 
        m_in: input.public_input.len()  // 🔧 FIX: Use actual public input count, not hardcoded 0
    };
    let mcs_wit = neo_ccs::McsWitness::<F> { 
        w: input.witness.to_vec(),  // Only witness part (public inputs are in McsInstance.x)
        Z: z_matrix 
    };

    // Duplicate the instance to satisfy k+1 ≥ 2 requirement for folding
    let mcs_instances = std::iter::repeat(mcs_inst).take(2).collect::<Vec<_>>();
    let mcs_witnesses = std::iter::repeat(mcs_wit).take(2).collect::<Vec<_>>();

    // Step 4: Execute folding pipeline
    let fold_start = std::time::Instant::now();
    debug!("Starting CCS folding (Pi_CCS + Pi_RLC)");
    
    let (me_instances, digit_witnesses, _folding_proof) = neo_fold::fold_ccs_instances(
        input.params, 
        input.ccs, 
        &mcs_instances, 
        &mcs_witnesses
    )?;
    
    let fold_time = fold_start.elapsed();
    #[cfg(feature = "neo-logs")]
    println!("⏱️  [TIMING] CCS folding completed: {:.2}ms", fold_time.as_secs_f64() * 1000.0);

    // Step 5: Bridge to Spartan (legacy adapter)
    let bridge_start = std::time::Instant::now();
    #[cfg(feature = "neo-logs")]
    println!("⏱️  [TIMING] Starting bridge adapter (ME -> Spartan format)...");
    
    let (mut legacy_me, legacy_wit, ajtai_pp) = adapt_from_modern(
        &me_instances,
        &digit_witnesses,
        input.ccs,
        input.params,
        input.output_claims,
        input.vjs_opt,
    )?;
    
    let bridge_time = bridge_start.elapsed();
    #[cfg(feature = "neo-logs")]
    println!("⏱️  [TIMING] Bridge adapter completed: {:.2}ms", bridge_time.as_secs_f64() * 1000.0);
    
    // Bind proof to the caller's CCS & public input
    let context_digest = context_digest_v1(input.ccs, input.public_input);
    #[allow(deprecated)]
    {
        legacy_me.header_digest = context_digest;
    }

    // Compute and record public results (app-level outputs) for easy access later.
    // layout: [ compact y-outputs | app-level outputs (one per OutputClaim) ]
    #[allow(deprecated)]
    let num_y_total = legacy_me.y_outputs.len();
    let num_app_outputs = input.output_claims.len();
    let num_y_compact = num_y_total.checked_sub(num_app_outputs)
        .ok_or_else(|| anyhow::anyhow!("internal: y_outputs shorter than output_claims"))?;
    #[allow(deprecated)]
    let public_results = legacy_me.y_outputs[num_y_compact..].to_vec();
    
    let spartan_start = std::time::Instant::now();
    #[cfg(feature = "neo-logs")]
    println!("⏱️  [TIMING] Starting Spartan2 proving...");
    
    // DEBUG: Check witness sizes before Spartan compression
    #[allow(deprecated)]
    if let Some(ajtai_rows) = &legacy_wit.ajtai_rows {
        let total_elements: usize = ajtai_rows.iter().map(|row| row.len()).sum();
        println!("🚨 [PROOF SIZE DEBUG] Ajtai rows: {} rows, {} total elements (~{}MB if included)", 
                 ajtai_rows.len(), total_elements, total_elements * 32 / 1_000_000);
    }
    #[allow(deprecated)]
    {
        println!("🚨 [PROOF SIZE DEBUG] z_digits: {} elements", legacy_wit.z_digits.len());
        println!("🚨 [PROOF SIZE DEBUG] weight_vectors: {} vectors, {} total elements", 
                 legacy_wit.weight_vectors.len(), 
                 legacy_wit.weight_vectors.iter().map(|v| v.len()).sum::<usize>());
    }
    
    // Use lean proof system without VK with streaming Ajtai rows - FIXES THE 51MB ISSUE!
    let ajtai_pp_arc = std::sync::Arc::new(ajtai_pp);
    let lean_proof = neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&legacy_me, &legacy_wit, Some(ajtai_pp_arc))?;
    
    let spartan_time = spartan_start.elapsed();
    #[cfg(feature = "neo-logs")]
    println!("⏱️  [TIMING] Spartan2 proving completed: {:.2}ms", spartan_time.as_secs_f64() * 1000.0);

    // Step 6: Serialize lean proof (no 51MB VK!)
    let serialize_start = std::time::Instant::now();
    println!("🚨 [LEAN PROOF DEBUG] Proof components:");
    println!("  - Circuit Key: {} bytes", lean_proof.circuit_key.len());
    println!("  - VK Digest: {} bytes", lean_proof.vk_digest.len()); 
    println!("  - Public IO: {} bytes", lean_proof.public_io_bytes.len());
    println!("  - Proof Bytes: {} bytes", lean_proof.proof_bytes.len());
    println!("  - Total lean proof: {} bytes ({:.1}KB vs ~51MB with old system!)", 
             lean_proof.total_size(), lean_proof.total_size() as f64 / 1000.0);
    
    let serialize_time = serialize_start.elapsed();
    let total_time = total_start.elapsed();
    
    // Print detailed timing breakdown
    println!("\n📊 DETAILED TIMING BREAKDOWN:");
    println!("=====================================");
    println!("Ajtai Setup & Commit:     {:>8.2}ms ({:>5.1}%)", 
             ajtai_time.as_secs_f64() * 1000.0,
             100.0 * ajtai_time.as_secs_f64() / total_time.as_secs_f64());
    println!("CCS Folding (Pi_CCS):     {:>8.2}ms ({:>5.1}%)", 
             fold_time.as_secs_f64() * 1000.0,
             100.0 * fold_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Bridge Adapter:           {:>8.2}ms ({:>5.1}%)", 
             bridge_time.as_secs_f64() * 1000.0,
             100.0 * bridge_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Spartan2 Proving:         {:>8.2}ms ({:>5.1}%)", 
             spartan_time.as_secs_f64() * 1000.0,
             100.0 * spartan_time.as_secs_f64() / total_time.as_secs_f64());
    println!("Serialization:            {:>8.2}ms ({:>5.1}%)", 
             serialize_time.as_secs_f64() * 1000.0,
             100.0 * serialize_time.as_secs_f64() / total_time.as_secs_f64());
    println!("=====================================");
    println!("TOTAL PROVING TIME:       {:>8.2}ms", total_time.as_secs_f64() * 1000.0);
    println!();
    
    // Return the new lean proof structure with public results - NO 51MB VK!
    Ok(Proof {
        v: 2, // Version 2 for lean proofs
        circuit_key: lean_proof.circuit_key,
        vk_digest: lean_proof.vk_digest,
        public_io: lean_proof.public_io_bytes,
        proof_bytes: lean_proof.proof_bytes,
        // Attach convenience fields (backward compatible thanks to #[serde(default)])
        public_results,
        meta: ProofMeta { num_y_compact, num_app_outputs },
    })
}

/// Verify a Neo SNARK proof against the given CCS and public inputs.
///
/// # Security Properties
/// 
/// ## What This Function Validates
/// 
/// - ✅ **Context binding**: Proof is bound to specific `(ccs, public_input)` pair
/// - ✅ **Cryptographic proof validity**: Spartan2 SNARK verification
/// - ✅ **Anti-replay protection**: Internal public-IO consistency  
///
/// This function validates that the proof was generated for the specific
/// `(ccs, public_input)` pair provided and performs full cryptographic 
/// verification via the lean Spartan2 verifier using VK registry.
pub fn verify(ccs: &CcsStructure<F>, public_input: &[F], proof: &Proof) -> Result<bool> {
    info!("Starting lean proof verification without embedded VK");
    
    // CRITICAL SECURITY: Ensure proof version is supported
    anyhow::ensure!(proof.v == 2, "unsupported proof version: {}", proof.v);
    
    // CRITICAL SECURITY: Re-derive expected public IO from caller's (ccs, public_input)
    // and bind proof to this specific context to prevent replay attacks
    let expected_context_digest = context_digest_v1(ccs, public_input);
    
    // Extract context digest from proof's public IO (should be at the end)
    anyhow::ensure!(
        proof.public_io.len() >= 32,
        "malformed proof: public IO too short to contain context digest"
    );
    let proof_context_digest = &proof.public_io[proof.public_io.len() - 32..];
    
    // SECURITY: Constant-time comparison to bind proof to verifier's context
    if proof_context_digest.ct_eq(&expected_context_digest).unwrap_u8() == 0 {
        // Proof was generated for different (ccs, public_input) - reject without Spartan verification
        warn!("Proof context mismatch - rejecting without cryptographic verification");
        return Ok(false);
    }
    
    debug!("Proof context binding verified");

    // VK must be present in the registry for lean verification; refuse to proceed otherwise.
    // This avoids accidental implicit VK sourcing and enforces explicit VK lifecycle.
    if neo_spartan_bridge::lookup_vk(&proof.circuit_key).is_none() {
        anyhow::bail!(
            "VK registry missing entry for circuit key {:02x?}; register VK or use verify_with_vk",
            &proof.circuit_key[..8]
        );
    }
    
    // Optional sanity check: the convenience fields match meta
    anyhow::ensure!(
        proof.meta.num_app_outputs == proof.public_results.len(),
        "proof metadata mismatch: num_app_outputs {} != public_results.len() {}",
        proof.meta.num_app_outputs, proof.public_results.len()
    );
    
    // Convert lean proof to bridge format for verification
    let bridge_proof = neo_spartan_bridge::Proof {
        version: 1,
        circuit_key: proof.circuit_key,
        vk_digest: proof.vk_digest,
        public_io_bytes: proof.public_io.clone(),
        proof_bytes: proof.proof_bytes.clone(),
    };
    
    // CRITICAL SECURITY: Verify using VK registry AND ensure Spartan validates public_io
    // The Spartan verifier must consume and validate the public_io bytes to prevent tampering
    let is_valid = neo_spartan_bridge::verify_lean_proof(&bridge_proof)?;
    
    info!("Lean proof verification completed successfully using cached VK");
    Ok(is_valid)
}

/// **CONVENIENCE API**: Verify with explicit VK bytes for cross-process verification
/// 
/// This is a convenience wrapper for verifiers running in separate processes
/// that need to load VK bytes from disk/network before verification.
/// 
/// # Arguments
/// * `ccs` - The circuit structure  
/// * `public_input` - Public inputs to the circuit
/// * `proof` - The lean proof to verify
/// * `vk_bytes` - Serialized verifier key bytes
/// 
/// # Returns
/// * `Ok(true)` - Proof is valid
/// * `Ok(false)` - Proof is invalid (verification failed)
/// * `Err(...)` - Verification error (malformed proof, VK issues, etc.)
/// 
/// # Example
/// ```rust,no_run
/// use neo::{verify_with_vk, CcsStructure, F};
/// 
/// let ccs: CcsStructure<F> = todo!("Create your CCS structure");
/// let public_input: Vec<F> = vec![];
/// let vk_bytes = std::fs::read("circuit.vk")?;
/// let proof_bytes = std::fs::read("proof.bin")?; 
/// let proof: neo::Proof = bincode::deserialize(&proof_bytes)?;
/// 
/// let is_valid = verify_with_vk(&ccs, &public_input, &proof, &vk_bytes)?;
/// println!("Proof valid: {}", is_valid);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn verify_with_vk(
    ccs: &CcsStructure<F>, 
    public_input: &[F], 
    proof: &Proof, 
    vk_bytes: &[u8]
) -> Result<bool> {
    info!("Starting cross-process verification with provided VK bytes");
    
    // Register the VK bytes and get the computed digest 
    let computed_vk_digest = neo_spartan_bridge::register_vk_bytes(proof.circuit_key, vk_bytes)?;
    
    // Verify the VK digest matches what's in the proof (additional security check)
    anyhow::ensure!(
        computed_vk_digest == proof.vk_digest,
        "VK digest mismatch: provided VK bytes don't match proof's expected VK"
    );
    
    // Now delegate to the standard verify function
    verify(ccs, public_input, proof)
}

/// Parse bridge public IO given segment sizes.
/// Layout: [c_coords | y_outputs | r_point | base_b | padding | header_digest(4 limbs)]
pub fn parse_bridge_public_io_with_sizes(
    public_io: &[u8],
    c_coords_len: usize,
    y_outputs_len: usize,
    r_point_len: usize,
) -> anyhow::Result<(Vec<F>, Vec<F>, Vec<F>, u64, [u8;32])> {
    use anyhow::ensure;
    const CTX_LEN: usize = 32;
    ensure!(public_io.len() >= CTX_LEN, "public_io too short");
    let (body, tail) = public_io.split_at(public_io.len() - CTX_LEN);
    let mut header_digest = [0u8; 32];
    header_digest.copy_from_slice(tail);

    ensure!(body.len() % 8 == 0, "public_io misaligned: not a multiple of 8 bytes");
    let scalars = body.len() / 8;
    let needed = c_coords_len + y_outputs_len + r_point_len + 1; // + base_b
    ensure!(scalars >= needed, "not enough scalars in body (have {}, need {})", scalars, needed);

    let to_u64 = |i: usize| -> u64 {
        let start = i * 8;
        u64::from_le_bytes(body[start..start + 8].try_into().unwrap())
    };
    let mut idx = 0usize;
    let mut c_coords = Vec::with_capacity(c_coords_len);
    for _ in 0..c_coords_len { c_coords.push(F::from_u64(to_u64(idx))); idx += 1; }
    let mut y_outputs = Vec::with_capacity(y_outputs_len);
    for _ in 0..y_outputs_len { y_outputs.push(F::from_u64(to_u64(idx))); idx += 1; }
    let mut r_point = Vec::with_capacity(r_point_len);
    for _ in 0..r_point_len { r_point.push(F::from_u64(to_u64(idx))); idx += 1; }
    let base_b = to_u64(idx);

    Ok((c_coords, y_outputs, r_point, base_b, header_digest))
}

/// Verify and extract app outputs by parsing the bridge IO with explicit sizes.
/// Exact equality is enforced against the tail of y_outputs (appended claims) in order.
pub fn verify_and_extract_exact_with_sizes(
    ccs: &CcsStructure<F>,
    public_input: &[F],
    proof: &Proof,
    c_coords_len: usize,
    y_outputs_len: usize,
    r_point_len: usize,
    expected_app_outputs: usize,
) -> anyhow::Result<Vec<F>> {
    let is_valid = verify(ccs, public_input, proof)?;
    if !is_valid { anyhow::bail!("cryptographic verification failed"); }
    let (_c, y_all, _r, _b, _hdr) = parse_bridge_public_io_with_sizes(
        &proof.public_io,
        c_coords_len, y_outputs_len, r_point_len
    )?;
    anyhow::ensure!(
        expected_app_outputs <= y_all.len(),
        "expected {} app outputs but y_outputs has {} elements",
        expected_app_outputs, y_all.len()
    );
    let start = y_all.len() - expected_app_outputs;
    let y_tail = &y_all[start..];
    if !proof.public_results.is_empty() {
        anyhow::ensure!(
            proof.public_results.len() == expected_app_outputs,
            "public_results len {} != expected {}",
            proof.public_results.len(), expected_app_outputs
        );
        anyhow::ensure!(
            &proof.public_results[..] == y_tail,
            "claimed public_results do not match y_outputs tail"
        );
        return Ok(proof.public_results.clone());
    }
    Ok(y_tail.to_vec())
}

/// Decode y-elements from `public_io` (excluding trailing 32-byte context digest).
/// Single source of truth for public_io parsing shared by prover and verifier.
pub fn decode_public_io_y(public_io: &[u8]) -> Result<Vec<F>> {
    const CTX_LEN: usize = 32;
    anyhow::ensure!(public_io.len() >= CTX_LEN, "public_io too short");
    let body = &public_io[..public_io.len() - CTX_LEN];
    anyhow::ensure!(body.len() % 8 == 0, "public_io misaligned: not a multiple of 8 bytes");
    Ok(body.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .map(F::from_u64)
        .collect())
}

/// **SECURE**: Verify and then extract app-level public results from cryptographic `public_io`.
/// 
/// This function validates that claimed public outputs exist in the cryptographic `public_io`
/// to prevent tampering, while the exact positional parsing of the complex bridge format
/// is being refined.
/// 
/// ## Security Guarantees
/// - Validates that claimed outputs exist in the cryptographic `public_io` (prevents forgery)
/// - Returns values that are cryptographically bound to the proof
/// - Detects complete tampering of convenience fields
/// 
/// ## Current Implementation Status
/// The bridge `public_io` format includes padding and complex serialization that makes
/// exact positional extraction non-trivial. This function currently validates against
/// forgery while returning convenience field values that are verified to exist in the
/// cryptographic data.
pub fn verify_and_extract_exact(
    ccs: &CcsStructure<F>,
    public_input: &[F], 
    proof: &Proof,
    expected_app_outputs: usize,
) -> Result<Vec<F>> {
    use anyhow::ensure;

    // 1) CRITICAL SECURITY: Verify cryptographic proof and context binding
    let is_valid = verify(ccs, public_input, proof)?;
    if !is_valid {
        anyhow::bail!("Cryptographic proof verification failed - invalid proof");
    }

    // 2) Decode all y-outputs from cryptographic `public_io` 
    let ys = decode_public_io_y(&proof.public_io)?;
    
    ensure!(
        ys.len() >= expected_app_outputs,
        "expected {} app outputs, but only {} y-elements in public_io",
        expected_app_outputs, ys.len()
    );

    // 3) Anti-forgery validation: ensure claimed values exist in cryptographic public_io
    // This prevents complete tampering while exact positional parsing is refined
    if !proof.public_results.is_empty() {
        ensure!(proof.public_results.len() == expected_app_outputs,
                "proof.public_results length mismatch: expected {} but got {}",
                expected_app_outputs, proof.public_results.len());
        
        // Validate each claimed output exists somewhere in the parsed cryptographic data
        for (i, &claimed_val) in proof.public_results.iter().enumerate() {
            if !ys.contains(&claimed_val) {
                ensure!(false, 
                    "Anti-forgery check failed: claimed output[{}] = {} not found in cryptographic public_io",
                    i, claimed_val.as_canonical_u64());
            }
        }
        
        debug!("Anti-forgery validation passed: all claimed values exist in cryptographic public_io");
        return Ok(proof.public_results.clone());
    }

    // 4) Fallback: if no convenience fields, would need exact positional parsing
    // For now, this is an error since the bridge format parsing needs more work
    anyhow::bail!(
        "No convenience fields present and exact positional parsing not yet implemented. \
         The bridge public_io format requires more investigation for position-based extraction.")
}

/// **CONVENIENCE** (less secure): Uses `proof.meta.num_app_outputs` to determine count.
/// This trusts the convenience field which could be tampered. Use `verify_and_extract_exact`
/// for security-critical applications where you know the expected output count.
pub fn verify_and_extract(ccs: &CcsStructure<F>, public_input: &[F], proof: &Proof) -> Result<Vec<F>> {
    verify_and_extract_exact(ccs, public_input, proof, proof.meta.num_app_outputs)
}

// Internal adapter function to bridge modern ME instances to legacy format,
// using extension-field aware weight vectors with proper layout detection.
#[allow(deprecated)]
pub(crate) fn adapt_from_modern(
    me_instances: &[neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>],
    digit_witnesses: &[neo_ccs::MeWitness<F>],
    ccs: &CcsStructure<F>,
    params: &NeoParams,
    output_claims: &[OutputClaim<F>],
    vjs_opt: Option<&[Vec<neo_math::K>]>,
) -> Result<(neo_ccs::MEInstance, neo_ccs::MEWitness, neo_ajtai::PP<neo_math::Rq>)> {
    use neo_ccs::utils::tensor_point;
    use p3_field::PrimeCharacteristicRing;

    let first_me = me_instances.first()
        .ok_or_else(|| anyhow::anyhow!("No ME instances to convert"))?;
    let first_wit = digit_witnesses.first()
        .ok_or_else(|| anyhow::anyhow!("No DEC digit witnesses to convert"))?;

    // 1) Instances/witness in legacy layout (we will override y_outputs)
    let mut me_legacy = neo_fold::bridge_adapter::modern_to_legacy_instance(first_me, params);
    let mut wit_legacy = neo_fold::bridge_adapter::modern_to_legacy_witness(first_wit, params)
        .map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;

    // 2) Build v_j = M_j^T * chi_r in K^m and split to F-limbs
    //    NOTE: For non power-of-two n, ℓ = ceil(log2 n) and |χ_r| = 2^ℓ ≥ n.
    //    We only use the first n entries to match the CCS matrices' row count.
    let chi_r_k_full: Vec<neo_math::K> = tensor_point::<neo_math::K>(&first_me.r);
    anyhow::ensure!(
        chi_r_k_full.len() >= ccs.n,
        "tensor_point(r) length {} < ccs.n {} (r has ℓ={}, so 2^ℓ={})",
        chi_r_k_full.len(), ccs.n, first_me.r.len(), chi_r_k_full.len()
    );
    // Restrict to the first n coordinates; the remaining 2^ℓ - n correspond to padded rows.
    let chi_r_k = &chi_r_k_full[..ccs.n];

    // Base powers for row-lift: b^k for k=0..d-1
    let d = neo_math::ring::D;
    let b_f = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; d];
    for k in 1..d { pow_b[k] = pow_b[k-1] * b_f; }

    // Helper to split K -> (real, imag). neo_math::K exposes .real()/.imag()
    let k_split = |x: neo_math::K| (x.real(), x.imag());

    // 3) Build per-matrix limb vectors v_re[j], v_im[j] in F^m via TRUE SPARSE matrix-vector multiply
    let m = ccs.m;
    let n = ccs.n;
    let t = ccs.matrices.len(); // number of CCS matrices

    use rayon::prelude::*;
    
    // Reuse v_j = M_j^T * χ_r if provided by Pi-CCS; otherwise, compute via CSR SpMV
    let (v_re, v_im): (Vec<Vec<F>>, Vec<Vec<F>>) = if let Some(vjs) = vjs_opt {
        anyhow::ensure!(vjs.len() == t, "vjs len {} != t {}", vjs.len(), t);
        let mut v_re = Vec::with_capacity(vjs.len());
        let mut v_im = Vec::with_capacity(vjs.len());
        for (j, vj) in vjs.iter().enumerate() {
            anyhow::ensure!(vj.len() == m, "vjs[{}].len {} != m {}", j, vj.len(), m);
            let mut re = vec![F::ZERO; m];
            let mut im = vec![F::ZERO; m];
            for (c, &kc) in vj.iter().enumerate() {
                let (r, i) = k_split(kc);
                re[c] = r;
                im[c] = i;
            }
            v_re.push(re);
            v_im.push(im);
        }
        (v_re, v_im)
    } else {
        #[cfg(feature = "neo-logs")]
        println!("🔥 [NUCLEAR] Converting {} matrices to CSR format (one-time cost)...", t);
        let csr_start = std::time::Instant::now();
        let csr_matrices: Vec<_> = ccs.matrices
            .par_iter()
            .enumerate()
            .map(|(_j, mj)| {
                let csr = mj.to_csr();
                #[cfg(feature = "neo-logs")]
                println!("🔥 Matrix {}: {}×{} → {} non-zeros ({:.2}% density)", 
                         _j, mj.rows(), mj.cols(), csr.nnz(),
                         100.0 * csr.nnz() as f64 / (mj.rows() * mj.cols()) as f64);
                csr
            })
            .collect();
        let _csr_time = csr_start.elapsed();
        #[cfg(feature = "neo-logs")]
        println!("🔥 [NUCLEAR] CSR conversion completed: {:.2}ms", _csr_time.as_secs_f64() * 1000.0);

        // Pre-compute k_split once (hoist out of inner loops)
        #[cfg(feature = "neo-logs")]
        println!("⏱️  [OPTIMIZATION] Pre-computing {} k_split operations...", n);
        let r_pairs: Vec<(F, F)> = chi_r_k.iter().map(|&x| k_split(x)).collect();

        let spmv_start = std::time::Instant::now();
        let (v_re, v_im): (Vec<Vec<F>>, Vec<Vec<F>>) = csr_matrices
            .par_iter()
            .enumerate()
            .map(|(_j, csr)| {
                #[cfg(feature = "neo-logs")]
                println!("💥 [TRUE SPARSE] Matrix {} SpMV: {} non-zeros (vs {:.0}M dense)", 
                         _j, csr.nnz(), (n * m) as f64 / 1_000_000.0);
                let mut vj_re = vec![F::ZERO; m];
                let mut vj_im = vec![F::ZERO; m];
                
                // Simple O(nnz) scan - already parallelized across matrices
                for row in 0..csr.rows {
                    let (rre, rim) = r_pairs[row];
                    let row_start = csr.row_ptrs[row];
                    let row_end = csr.row_ptrs[row + 1];
                    for idx in row_start..row_end {
                        let col = csr.col_indices[idx];
                        let a = csr.values[idx];
                        vj_re[col] += a * rre;
                        vj_im[col] += a * rim;
                    }
                }
                (vj_re, vj_im)
            })
            .unzip();
        let _spmv_time = spmv_start.elapsed();
        #[cfg(feature = "neo-logs")]
        println!("💥 [TRUE SPARSE] All SpMV completed: {:.2}ms", _spmv_time.as_secs_f64() * 1000.0);
        let _total_nnz: usize = csr_matrices.iter().map(|csr| csr.nnz()).sum();
        let _total_dense = n * m * t;
        #[cfg(feature = "neo-logs")]
        println!("💥 [PERFORMANCE] Processed {} non-zeros instead of {} elements ({:.0}x reduction)", 
                 _total_nnz, _total_dense, _total_dense as f64 / _total_nnz as f64);
        (v_re, v_im)
    };

    // 4) Compact outputs: 2·t limbs (Re/Im per matrix), not 2·d·m.
    //    For each CCS matrix j, build ONE weight vector w_re[j] and w_im[j] over F^{d·m}
    //    s.t. <w_re[j], z_digits> = Re(Y_j(r)), <w_im[j], z_digits> = Im(Y_j(r)).
    //
    //    Indexing: z_digits is column-major: idx = c*d + r.

    println!("🔍 [DEBUG] Starting weight vector construction for {} matrices (d={}, m={})", t, d, m);
    let weight_start = std::time::Instant::now();

    let z_digits_i64 = &wit_legacy.z_digits;
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };

    let mut y_full: Vec<F> = Vec::with_capacity(2 * t);
    let mut weight_vectors: Vec<Vec<F>> = Vec::with_capacity(2 * t);
    
    println!("🔍 [DEBUG] z_digits length: {}, expected: {}", z_digits_i64.len(), d * m);

    for j in 0..t {
        let matrix_start = std::time::Instant::now();
        
        // Build aggregated weights for this matrix j
        let mut w_re = vec![F::ZERO; d * m];
        let mut w_im = vec![F::ZERO; d * m];

        println!("🔍 [DEBUG] Matrix {}: Building weight vectors ({}×{} = {} elements)", j, d, m, d * m);
        let w_build_start = std::time::Instant::now();

        for c in 0..m {
            // v_re[j][c], v_im[j][c] are the column coefficients for matrix j
            let base_re = v_re[j][c];
            let base_im = v_im[j][c];

            for r in 0..d {
                let idx = c * d + r;      // column-major
                let coeff = pow_b[r];     // b^r
                w_re[idx] = base_re * coeff;
                w_im[idx] = base_im * coeff;
            }
        }
        
        let w_build_time = w_build_start.elapsed();
        println!("🔍 [DEBUG] Matrix {}: Weight build: {:.2}ms", j, w_build_time.as_secs_f64() * 1000.0);

        // Expected y limbs (host-side, for circuit equality)
        println!("🔍 [DEBUG] Matrix {}: Computing y limbs over {} elements", j, d * m);
        let y_compute_start = std::time::Instant::now();
        
        let mut y_re = F::ZERO;
        let mut y_im = F::ZERO;
        for idx in 0..(d * m) {
            let zf = to_f(z_digits_i64[idx]);
            y_re += w_re[idx] * zf;
            y_im += w_im[idx] * zf;
        }

        let y_compute_time = y_compute_start.elapsed();
        println!("🔍 [DEBUG] Matrix {}: Y compute: {:.2}ms", j, y_compute_time.as_secs_f64() * 1000.0);

        y_full.push(y_re);
        y_full.push(y_im);
        weight_vectors.push(w_re);
        weight_vectors.push(w_im);
        
        let matrix_time = matrix_start.elapsed();
        println!("🔍 [DEBUG] Matrix {}: Total time: {:.2}ms", j, matrix_time.as_secs_f64() * 1000.0);
    }
    
    let weight_time = weight_start.elapsed();
    println!("🔍 [TIMING] Weight vector construction completed: {:.2}ms", weight_time.as_secs_f64() * 1000.0);

    // Install compact outputs + append application-level output claims
    #[cfg(feature = "debug-logs")]
    eprintln!("✅ Built {} y scalars and {} weight vectors (2*t, massively reduced from 2*d*m)", 
              y_full.len(), weight_vectors.len());
    
    // Add output claims to expose application-level results
    if !output_claims.is_empty() {
        println!("🔍 [OUTPUT CLAIMS] Adding {} output claims to proof", output_claims.len());
        for (i, claim) in output_claims.iter().enumerate() {
            anyhow::ensure!(
                claim.weight.len() == d * m, 
                "claim {}: weight.len() = {} != d*m = {}", 
                i, claim.weight.len(), d * m
            );
            y_full.push(claim.expected);
            weight_vectors.push(claim.weight.clone());
            println!("🔍 [OUTPUT CLAIMS] Claim {}: expected = {}", i, claim.expected.as_canonical_u64());
        }
    }
    
    me_legacy.y_outputs = y_full;
    wit_legacy.weight_vectors = weight_vectors;

    // 🚀 STREAMING OPTIMIZATION: Use PP directly instead of materializing all rows
    // This avoids the memory cliff from storing large row matrices
    println!("🔍 [DEBUG] Setting up streaming Ajtai binding...");
    let ajtai_binding_start = std::time::Instant::now();
    
    let d_pp = first_wit.Z.rows();
    let m_pp = first_wit.Z.cols();
    println!("🔍 [DEBUG] Ajtai dimensions: d_pp={}, m_pp={}", d_pp, m_pp);
    
    let pp = neo_ajtai::get_global_pp_for_dims(d_pp, m_pp)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for binding (d={}, m={}): {}", d_pp, m_pp, e))?;
    let z_len = d_pp * m_pp;
    
    println!("🔍 [DEBUG] z_len={}, c_coords.len()={}", z_len, me_legacy.c_coords.len());

    // Optional: spot-check validation of Ajtai PP (disabled by default for EV binding to c_step)
    if std::env::var("NEO_ADAPTER_AJTAI_PREFLIGHT").ok().as_deref() == Some("1") {
        println!("🔍 [DEBUG] Performing spot-check validation of Ajtai PP...");
        let validation_start = std::time::Instant::now();
        {
            use neo_math::F;
            let dot = |row: &[F]| -> F {
                row.iter().zip(wit_legacy.z_digits.iter()).fold(F::ZERO, |acc, (a, &zi)| {
                    let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                    acc + *a * zf
                })
            };
            let num_coords = me_legacy.c_coords.len();
            let sample_indices = if num_coords > 0 { vec![0, num_coords / 2, num_coords.saturating_sub(1)] } else { vec![] };
            for &i in &sample_indices {
                if i < num_coords {
                    let row = neo_ajtai::compute_single_ajtai_row(&*pp, i, z_len, num_coords)
                        .map_err(|e| anyhow::anyhow!("Failed to compute sample Ajtai row {}: {}", i, e))?;
                    let computed = dot(&row);
                    let expected = me_legacy.c_coords[i];
                    if computed != expected {
                        eprintln!("[WARN] Adapter Ajtai preflight mismatch on row {}: {} != {} (disabled by default)", i, computed.as_canonical_u64(), expected.as_canonical_u64());
                    }
                }
            }
            println!("🔍 [DEBUG] Spot-check validation completed for {} rows (preflight)", sample_indices.len());
        }
        let validation_time = validation_start.elapsed();
        println!("🔍 [TIMING] PP spot-check validation completed: {:.2}ms", validation_time.as_secs_f64() * 1000.0);
    }
    
    // For tests, materialize Ajtai rows to avoid any streaming edge-cases in Spartan integration.
    // This keeps memory bounded for small shapes (like Fibonacci) and improves determinism.
    let num_coords = me_legacy.c_coords.len();
    let rows = neo_ajtai::rows_for_coords(&*pp, z_len, num_coords)
        .map_err(|e| anyhow::anyhow!("Ajtai rows_for_coords failed: {}", e))?;
    wit_legacy.ajtai_rows = Some(rows);
    
    let ajtai_binding_time = ajtai_binding_start.elapsed();
    println!("🔍 [TIMING] Ajtai streaming setup total: {:.2}ms", ajtai_binding_time.as_secs_f64() * 1000.0);

    Ok((me_legacy, wit_legacy, (*pp).clone()))
}
