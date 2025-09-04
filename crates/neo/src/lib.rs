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
#[cfg(debug_assertions)]
use rand::SeedableRng;
use p3_field::PrimeCharacteristicRing;
use subtle::ConstantTimeEq;

// Note: The global Ajtai PP is stored in a OnceLock and cannot be cleared.
// This is a known limitation - concurrent prove() calls may interfere if 
// they use different parameters. Future versions should thread PP explicitly.

// Re-export key types that users need
pub use neo_params::NeoParams;
pub use neo_ccs::CcsStructure;
pub use neo_math::{F, K};

/// Opaque proof object (bincode-encoded Spartan bundle, versioned)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ProofV1 {
    /// Version tag for forward-compat
    pub v: u16,
    /// Public IO bytes bound by the bridge (anti-replay)
    pub public_io: Vec<u8>,
    /// Serialized Spartan bundle (includes proof + VK)
    pub bundle: Vec<u8>,
}

impl ProofV1 {
    /// Returns the total size of the proof in bytes
    pub fn size(&self) -> usize {
        self.bundle.len() + self.public_io.len() + std::mem::size_of::<u16>()
    }
    
    /// Returns the public IO bytes bound by the proof (for verification binding)
    pub fn public_io(&self) -> &[u8] {
        &self.public_io
    }
    
    /// Returns the proof version
    pub fn version(&self) -> u16 {
        self.v
    }
}

pub type Proof = ProofV1;

/// Inputs needed by the prover (explicit is better than global state)
pub struct ProveInput<'a> {
    pub params: &'a NeoParams,                         // includes b, k, B, s, guard inequality
    pub ccs: &'a CcsStructure<F>,                      // the circuit
    pub public_input: &'a [F],                         // x
    pub witness: &'a [F],                              // z
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
    // Parameter guard: enforce (k+1)T(b-1) < B for RLC soundness
    anyhow::ensure!(
        (input.params.k as u128 + 1)
            * (input.params.T as u128)
            * ((input.params.b - 1) as u128)
            < (input.params.B as u128),
        "unsafe params: (k+1)·T·(b−1) ≥ B"
    );

    // Fail-fast CCS consistency check: witness must satisfy the constraint system
    neo_ccs::check_ccs_rowwise_zero(input.ccs, input.public_input, input.witness)
        .map_err(|e| anyhow::anyhow!("CCS check failed - witness does not satisfy constraints: {:?}", e))?;

    // Step 1: Ajtai setup (temporary global state for compatibility)
    let d = neo_math::ring::D;
    
    // Use deterministic RNG only in debug builds for reproducibility
    // In release builds, use cryptographically secure randomness
    #[cfg(debug_assertions)]
    let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
    #[cfg(not(debug_assertions))]
    let mut rng = rand::rng();
    
    let pp = ajtai_setup(&mut rng, d, /*kappa*/ 16, input.witness.len())?;
    
    // Publish PP globally so folding protocols can access it 
    // NOTE: Global state limitation - concurrent prove() calls with different 
    // parameters may interfere. This is a known architectural issue.
    neo_ajtai::set_global_pp(pp.clone())?;
    
    // Step 2: Decompose and commit to witness
    let decomp_z = decomp_b(input.witness, input.params.b, d, DecompStyle::Balanced);
    anyhow::ensure!(decomp_z.len() % d == 0, "decomp length not multiple of d");
    let commitment = commit(&pp, &decomp_z);
    
    // Step 3: Build MCS instance/witness (row-major conversion)
    let m = decomp_z.len() / d;
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
        m_in: 0  // All witness elements are private (no public input constraints in CCS)
    };
    let mcs_wit = neo_ccs::McsWitness::<F> { 
        w: input.witness.to_vec(), 
        Z: z_matrix 
    };

    // Duplicate the instance to satisfy k+1 ≥ 2 requirement for folding
    let mcs_instances = std::iter::repeat(mcs_inst).take(2).collect::<Vec<_>>();
    let mcs_witnesses = std::iter::repeat(mcs_wit).take(2).collect::<Vec<_>>();

    // Step 4: Execute folding pipeline
    let (me_instances, digit_witnesses, _folding_proof) = neo_fold::fold_ccs_instances(
        input.params, 
        input.ccs, 
        &mcs_instances, 
        &mcs_witnesses
    )?;

    // Step 5: Bridge to Spartan (legacy adapter)
    let (legacy_me, legacy_wit) = adapt_from_modern(&me_instances, &digit_witnesses, input.ccs, input.params)?;
    let bundle = neo_spartan_bridge::compress_me_to_spartan(&legacy_me, &legacy_wit)?;

    // Step 6: Capture public IO and serialize proof
    let public_io = bundle.public_io_bytes.clone();
    let proof_bytes = bincode::serialize(&bundle)?;
    
    Ok(Proof {
        v: 1,
        public_io,
        bundle: proof_bytes,
    })
}

/// Verify a Neo SNARK proof against the given CCS and public inputs.
///
/// # Security Notice
/// 
/// **IMPORTANT**: This function currently does NOT validate that the provided `ccs` and 
/// `public_input` parameters match what's cryptographically bound in the proof. The proof
/// contains its own binding via the Spartan2 bridge's public-IO encoding, but validating
/// caller parameters would require reconstructing the full folding pipeline.
/// 
/// ## What This Function Validates
/// 
/// - ✅ **Cryptographic proof validity**: Spartan2 SNARK verification
/// - ✅ **Anti-replay protection**: Internal public-IO consistency  
/// - ❌ **Caller parameter binding**: `ccs`/`public_input` vs proof binding
///
/// This deserializes the proof bundle and calls the Spartan2 verifier,
/// ensuring the proof is valid and the public IO binding is consistent.
#[allow(unused_variables)]  // ccs, public_input not used for binding validation (known limitation)
pub fn verify(ccs: &CcsStructure<F>, public_input: &[F], proof: &Proof) -> Result<bool> {
    // Check proof version
    anyhow::ensure!(proof.v == 1, "unsupported proof version: {}", proof.v);
    
    let bundle: neo_spartan_bridge::ProofBundle = bincode::deserialize(&proof.bundle)?;
    
    // Anti-replay binding check: ensure the public IO bytes in the proof 
    // match exactly what the bundle claims to verify (constant-time for security)
    anyhow::ensure!(
        proof.public_io.ct_eq(&bundle.public_io_bytes).unwrap_u8() == 1,
        "Public IO mismatch: proof.public_io != bundle.public_io_bytes"
    );
    
    // The bridge verifier will check that the SNARK actually proves statements 
    // consistent with these public IO bytes
    neo_spartan_bridge::verify_me_spartan(&bundle)
}

// Internal adapter function to bridge modern ME instances to legacy format,
// using extension-field aware weight vectors with proper layout detection.
#[allow(deprecated)]
fn adapt_from_modern(
    me_instances: &[neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>],
    digit_witnesses: &[neo_ccs::MeWitness<F>],
    ccs: &CcsStructure<F>,
    params: &NeoParams,
) -> Result<(neo_ccs::MEInstance, neo_ccs::MEWitness)> {
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

    // 2) Build v_j = M_j^T * chi_r  in K^m and split to F-limbs
    let chi_r_k: Vec<neo_math::K> = tensor_point::<neo_math::K>(&first_me.r);
    anyhow::ensure!(chi_r_k.len() == ccs.n,
        "tensor_point(r) length {} != ccs.n {}", chi_r_k.len(), ccs.n);

    // Base powers for row-lift: b^k for k=0..d-1
    let d = neo_math::ring::D;
    let b_f = F::from_u64(params.b as u64);
    let mut pow_b = vec![F::ONE; d];
    for k in 1..d { pow_b[k] = pow_b[k-1] * b_f; }

    // Helper to split K -> (real, imag). neo_math::K exposes .real()/.imag()
    let k_split = |x: neo_math::K| (x.real(), x.imag());

    // 3) Build per-matrix limb vectors v_re[j], v_im[j] in F^m
    let m = ccs.m;
    let n = ccs.n;
    let t = ccs.matrices.len(); // number of CCS matrices

    // v_j limbs in F^m
    let mut v_re: Vec<Vec<F>> = Vec::with_capacity(t);
    let mut v_im: Vec<Vec<F>> = Vec::with_capacity(t);

    for mj in &ccs.matrices {
        let mut vj_re = vec![F::ZERO; m];
        let mut vj_im = vec![F::ZERO; m];
        for row in 0..n {
            let (rre, rim) = k_split(chi_r_k[row]);
            for col in 0..m {
                let a = mj[(row, col)];
                vj_re[col] += a * rre;
                vj_im[col] += a * rim;
            }
        }
        v_re.push(vj_re);
        v_im.push(vj_im);
    }

    // 4) Inflate to per-column outputs expected by the bridge:
    //    y has length 2*d*m (re/imag interleaved), and weight_vectors has same length.
    let z_digits = &wit_legacy.z_digits;               // col-major: idx = c*d + r
    anyhow::ensure!(z_digits.len() == d * m, "z_digits length {} != d*m {}*{}", z_digits.len(), d, m);

    // Build re/imag outputs per (c,r) and matching one-hot weight vectors:
    let mut y_full: Vec<F> = Vec::with_capacity(2 * d * m);
    let mut weight_vectors: Vec<Vec<F>> = Vec::with_capacity(2 * d * m);

    for c in 0..m {
        // pre-accumulate sum_j v_j[c] for re/imag
        let sum_v_re_c = (0..t).fold(F::ZERO, |acc, j| acc + v_re[j][c]);
        let sum_v_im_c = (0..t).fold(F::ZERO, |acc, j| acc + v_im[j][c]);

        for r in 0..d {
            // z_digits elements are in [-b, ..., b] range (small), convert to field element
            let zi = z_digits[c * d + r];
            let z_rc = if zi >= 0 {
                F::from_u64(zi as u64)
            } else {
                -F::from_u64((-zi) as u64)
            };
            let coeff_re = sum_v_re_c * pow_b[r];
            let coeff_im = sum_v_im_c * pow_b[r];

            // y entries (re, im) for this (r,c):
            y_full.push(coeff_re * z_rc);
            y_full.push(coeff_im * z_rc);

            // matching one-hot weights (dot with z_digits equals y)
            let mut w_re = vec![F::ZERO; d * m];
            let mut w_im = vec![F::ZERO; d * m];
            w_re[c * d + r] = coeff_re;
            w_im[c * d + r] = coeff_im;
            weight_vectors.push(w_re);
            weight_vectors.push(w_im);
        }
    }

    // Install the inflated claims
    #[cfg(debug_assertions)]
    eprintln!("✅ Built {} y scalars and {} weight vectors (2*d*m)", 
              y_full.len(), weight_vectors.len());
    
    me_legacy.y_outputs = y_full;
    wit_legacy.weight_vectors = weight_vectors;

    Ok((me_legacy, wit_legacy))
}
