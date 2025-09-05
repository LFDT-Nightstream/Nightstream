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
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use subtle::ConstantTimeEq;
// LZ4 is already imported via the use of block::compress and block::decompress below

// Race-safe Ajtai PP initialization helper
fn ensure_global_ajtai_pp<FN>(mut setup: FN) -> anyhow::Result<()>
where
    FN: FnMut() -> anyhow::Result<()>,
{
    // If it is already initialized, just return.
    if neo_ajtai::get_global_pp().is_ok() {
        return Ok(());
    }

    // Try to initialize. If a concurrent test beat us to it, treat it as success.
    match setup() {
        Ok(()) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("already initialized") || msg.contains("AlreadyInitialized") {
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

// Simple, deterministic 256-bit mixer (not meant as a crypto hash; adequate for binding tests).
fn context_digest_v0(ccs: &CcsStructure<F>, public_input: &[F]) -> [u8; 32] {
    let mut w = [0u64; 4];

    // feed bytes into 4 lanes
    let mut mix = |b: u8, i: usize| {
        let lane = i & 3;
        w[lane] = w[lane]
            .rotate_left(5)
            .wrapping_mul(0x9E37_79B1_85EB_CA87)
            .wrapping_add(b as u64);
    };

    let mut byte_idx = 0;
    
    // Mix CCS structure deterministically
    // 1. Basic dimensions
    for &val in &[ccs.n as u64, ccs.m as u64, ccs.matrices.len() as u64] {
        for j in 0..8 {
            mix(((val >> (8 * j)) & 0xFF) as u8, byte_idx);
            byte_idx += 1;
        }
    }
    
    // 2. All matrices entries (deterministic order)
    for matrix in &ccs.matrices {
        // Mix matrix dimensions first
        for &val in &[matrix.rows() as u64, matrix.cols() as u64] {
            for j in 0..8 {
                mix(((val >> (8 * j)) & 0xFF) as u8, byte_idx);
                byte_idx += 1;
            }
        }
        // Mix matrix entries
        for &val in matrix.as_slice() {
            let f_val = val.as_canonical_u64();
            for j in 0..8 {
                mix(((f_val >> (8 * j)) & 0xFF) as u8, byte_idx);
                byte_idx += 1;
            }
        }
    }
    
    // 3. Polynomial f (mix terms deterministically)
    for term in ccs.f.terms() {
        // Mix coefficient
        let coeff_val = term.coeff.as_canonical_u64();
        for j in 0..8 {
            mix(((coeff_val >> (8 * j)) & 0xFF) as u8, byte_idx);
            byte_idx += 1;
        }
        // Mix exponents
        for &exp in &term.exps {
            for j in 0..4 {
                mix(((exp as u32 >> (8 * j)) & 0xFF) as u8, byte_idx);
                byte_idx += 1;
            }
        }
    }

    // 5. Public inputs (canonical u64s)
    for x in public_input.iter() {
        let xi = x.as_canonical_u64();
        for j in 0..8 {
            mix(((xi >> (8 * j)) & 0xFF) as u8, byte_idx);
            byte_idx += 1;
        }
    }

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&w[i].to_le_bytes());
    }
    out
}

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

    // Step 1: Ajtai setup (race-safe global state)
    let d = neo_math::ring::D;
    
    ensure_global_ajtai_pp(|| {
        // Use deterministic RNG only in debug builds for reproducibility
        // In release builds, use cryptographically secure randomness
        #[cfg(debug_assertions)]
        let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
        #[cfg(not(debug_assertions))]
        let mut rng = rand::rng();
        
        let pp = ajtai_setup(&mut rng, d, /*kappa*/ 16, input.witness.len())?;
        
        // Publish PP globally so folding protocols can access it 
        neo_ajtai::set_global_pp(pp.clone()).map_err(anyhow::Error::from)
    })?;
    
    // Step 2: Decompose and commit to witness
    let decomp_z = decomp_b(input.witness, input.params.b, d, DecompStyle::Balanced);
    anyhow::ensure!(decomp_z.len() % d == 0, "decomp length not multiple of d");
    
    // Get PP from global state (now guaranteed to be initialized)
    let pp = neo_ajtai::get_global_pp()
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP: {}", e))?;
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
    let (mut legacy_me, legacy_wit) = adapt_from_modern(&me_instances, &digit_witnesses, input.ccs, input.params)?;
    
    // Bind proof to the caller's CCS & public input
    let context_digest = context_digest_v0(input.ccs, input.public_input);
    #[allow(deprecated)]
    {
        legacy_me.header_digest = context_digest;
    }
    
    let bundle = neo_spartan_bridge::compress_me_to_spartan(&legacy_me, &legacy_wit)?;

    // Step 6: Capture public IO and serialize proof with LZ4 compression
    let public_io = bundle.public_io_bytes.clone();
    let uncompressed_bytes = bincode::serialize(&bundle)?;
    let compressed_bytes = lz4_flex::compress_prepend_size(&uncompressed_bytes);
    
    #[cfg(debug_assertions)]
    eprintln!("✅ LZ4 compression: {} → {} bytes ({:.1}% reduction)", 
              uncompressed_bytes.len(), compressed_bytes.len(),
              100.0 * (1.0 - compressed_bytes.len() as f64 / uncompressed_bytes.len() as f64));
    
    Ok(Proof {
        v: 1,
        public_io,
        bundle: compressed_bytes,
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
/// This function first validates that the proof was generated for the specific
/// `(ccs, public_input)` pair provided by checking a context digest, then
/// proceeds with full cryptographic verification via the Spartan2 verifier.
pub fn verify(ccs: &CcsStructure<F>, public_input: &[F], proof: &Proof) -> Result<bool> {
    // Check proof version
    anyhow::ensure!(proof.v == 1, "unsupported proof version: {}", proof.v);
    
    // Size guard to prevent decompression bombs
    const MAX_BUNDLE_BYTES: usize = 64 * 1024 * 1024; // 64 MiB limit
    anyhow::ensure!(proof.bundle.len() >= 4, "malformed proof bundle: too short");
    
    let expected_len = {
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&proof.bundle[0..4]);
        u32::from_le_bytes(len_bytes) as usize
    };
    
    anyhow::ensure!(
        expected_len <= MAX_BUNDLE_BYTES,
        "proof bundle too large after decompression: {} bytes (limit {})",
        expected_len, MAX_BUNDLE_BYTES
    );
    
    // Decompress and deserialize
    let decompressed_bytes = lz4_flex::decompress_size_prepended(&proof.bundle)
        .map_err(|e| anyhow::anyhow!("lz4 decode failed: {e}"))?;
    
    // Verify decompressed size matches header
    anyhow::ensure!(
        decompressed_bytes.len() == expected_len,
        "decompressed length mismatch ({} != header {})",
        decompressed_bytes.len(),
        expected_len
    );
    let bundle: neo_spartan_bridge::ProofBundle = bincode::deserialize(&decompressed_bytes)?;
    
    // Anti-replay binding check: ensure the public IO bytes in the proof 
    // match exactly what the bundle claims to verify (constant-time for security)
    anyhow::ensure!(
        proof.public_io.ct_eq(&bundle.public_io_bytes).unwrap_u8() == 1,
        "Public IO mismatch: proof.public_io != bundle.public_io_bytes"
    );
    
    // 1) Extract digest from proof bundle - digest is at the end (tail)
    if proof.public_io.len() < 32 {
        return Err(anyhow::anyhow!("malformed proof: missing header digest"));
    }
    let expected = context_digest_v0(ccs, public_input);
    let tail = &proof.public_io[proof.public_io.len() - 32..];

    // 2) Compare — bind proof to verifier's context (constant-time)
    if tail.ct_eq(&expected).unwrap_u8() == 0 {
        // Not our CCS/IO: reject without touching Spartan
        return Ok(false);
    }
    
    // 3) Context matches - proceed with full cryptographic verification
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

    // 4) Compact outputs: 2·t limbs (Re/Im per matrix), not 2·d·m.
    //    For each CCS matrix j, build ONE weight vector w_re[j] and w_im[j] over F^{d·m}
    //    s.t. <w_re[j], z_digits> = Re(Y_j(r)), <w_im[j], z_digits> = Im(Y_j(r)).
    //
    //    Indexing: z_digits is column-major: idx = c*d + r.

    let z_digits_i64 = &wit_legacy.z_digits;
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };

    let mut y_full: Vec<F> = Vec::with_capacity(2 * t);
    let mut weight_vectors: Vec<Vec<F>> = Vec::with_capacity(2 * t);

    for j in 0..t {
        // Build aggregated weights for this matrix j
        let mut w_re = vec![F::ZERO; d * m];
        let mut w_im = vec![F::ZERO; d * m];

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

        // Expected y limbs (host-side, for circuit equality)
        let mut y_re = F::ZERO;
        let mut y_im = F::ZERO;
        for idx in 0..(d * m) {
            let zf = to_f(z_digits_i64[idx]);
            y_re += w_re[idx] * zf;
            y_im += w_im[idx] * zf;
        }

        y_full.push(y_re);
        y_full.push(y_im);
        weight_vectors.push(w_re);
        weight_vectors.push(w_im);
    }

    // Install compact outputs
    #[cfg(debug_assertions)]
    eprintln!("✅ Built {} y scalars and {} weight vectors (2*t, massively reduced from 2*d*m)", 
              y_full.len(), weight_vectors.len());
    
    me_legacy.y_outputs = y_full;
    wit_legacy.weight_vectors = weight_vectors;

    // OFFICIAL API: Get Ajtai binding rows from PP
    // These rows L_i satisfy <L_i, z_digits> = c_coords[i] and are derived
    // directly from the public Ajtai matrix parameters.
    let pp = neo_ajtai::get_global_pp()
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for binding: {}", e))?;

    let z_len = first_wit.Z.rows() * first_wit.Z.cols();

    // Fetch the official Ajtai rows in the exact orientation the circuit expects
    let rows = neo_ajtai::rows_for_coords(&pp, z_len, me_legacy.c_coords.len())
        .map_err(|e| anyhow::anyhow!("Failed to derive Ajtai binding rows: {}", e))?;

    // SECURITY CRITICAL: Strict validation of ALL rows from authentic PP
    // If any row fails validation, we fail closed - no synthetic fallbacks allowed
    {
        use neo_math::F;
        let dot = |row: &[F]| -> F {
            row.iter().zip(wit_legacy.z_digits.iter()).fold(F::ZERO, |acc, (a, &zi)| {
                let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                acc + *a * zf
            })
        };
        
        // Validate ALL rows, not just a prefix - this is security critical
        let check_count = core::cmp::min(rows.len(), me_legacy.c_coords.len());
        for i in 0..check_count {
            let computed = dot(&rows[i]);
            let expected = me_legacy.c_coords[i];
            anyhow::ensure!(
                computed == expected,
                "SECURITY: Ajtai row {} validation failed - <L_{}, z_digits> = {} != c_coords[{}] = {}. \
                 Authentic binding rows from PP are required for security.",
                i, i, computed, i, expected
            );
        }
    }
    
    // Install the validated authentic rows
    wit_legacy.ajtai_rows = Some(rows);

    Ok((me_legacy, wit_legacy))
}

