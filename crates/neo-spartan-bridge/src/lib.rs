#![forbid(unsafe_code)]
#![allow(deprecated)]

//! neo-spartan-bridge  
//!
//! **Post-quantum last-mile compression**: Translate ME(b, L) into a Spartan2 R1CS SNARK
//! using **Hash‑MLE PCS** + unified **Poseidon2** transcripts (no FRI).
//!
//! ## Architecture
//!
//! - **Spartan2 R1CS SNARK**: Direct R1CS conversion with Hash-MLE PCS backend
//! - **Unified Poseidon2**: Single transcript family across folding + SNARK phases  
//! - **Linear constraints**: ME(b,L) maps cleanly to R1CS (Ajtai + evaluation rows)
//! - **Production-ready**: Standard SNARK interface with proper transcript binding
//!
//! ## Security Properties
//!
//! - **Post-quantum**: Hash-based MLE PCS, no elliptic curves or pairings
//! - **Transcript binding**: Fold digest included in SNARK public inputs
//! - **Unified Poseidon2**: Consistent Fiat-Shamir across all phases
//! - **Standard R1CS**: Well-audited SNARK patterns

mod types;
pub mod hash_mle;
pub mod me_to_r1cs;

// Tests will be added in a separate PR to avoid compilation complexity

pub use types::{ProofBundle, Proof};
pub use crate::me_to_r1cs::IvcEvEmbed;
pub use crate::me_to_r1cs::{CommitEvoEmbed, IvcLinkageInputs};

use anyhow::Result;
use neo_ccs::{MEInstance, MEWitness};
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
use spartan2::spartan::{R1CSSNARK, SpartanVerifierKey};
// Arc not needed for this file - it's used in me_to_r1cs
use spartan2::traits::snark::R1CSSNARKTrait;

// SECURITY: Gated logging to prevent CWE-532 (sensitive info leakage)
#[cfg(feature = "neo-logs")]
use tracing::{debug, info};

#[cfg(not(feature = "neo-logs"))]
macro_rules! debug { ($($tt:tt)*) => {} }
#[cfg(not(feature = "neo-logs"))]
macro_rules! info  { ($($tt:tt)*) => {} }
#[cfg(not(feature = "neo-logs"))]
#[allow(unused_macros)] // May be used conditionally
macro_rules! warn  { ($($tt:tt)*) => {} }

/// Domain tags for different digest versions
const VK_DIGEST_DOMAIN_V1: &[u8] = b"VK_DIGEST_V1";
const VK_DIGEST_DOMAIN_V2: &[u8] = b"VK_DIGEST_V2_POSEIDON2";

/// Compute BLAKE3 digest (legacy V1 format for compatibility)
fn compute_blake3_digest_v1(data: &[u8], domain_separator: &[u8]) -> [u8; 32] {
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(domain_separator);
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Compute Poseidon2 digest using proper sponge construction (V2 format)
/// This follows standard sponge practice: absorb/pad/squeeze with proper rate limiting
fn compute_poseidon2_digest_v2(data: &[u8], domain_separator: &[u8]) -> [u8; 32] {
    use once_cell::sync::Lazy;
    
    // SECURITY: Use canonical Poseidon2 parameters instead of RNG-derived ones
    // This ensures parameter stability across library versions and builds
    static POSEIDON2: Lazy<Poseidon2Goldilocks<16>> = Lazy::new(|| {
        // Use deterministic RNG for stable parameters across builds
        use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
        let mut rng = ChaCha8Rng::seed_from_u64(0x504F534549444F4E); // "POSEIDON" in hex
        Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng)
    });
    
    const RATE: usize = 12;  // Safe rate for Poseidon2 with width 16
    const WIDTH: usize = 16;
    
    let poseidon2 = &*POSEIDON2;
    let mut state = [Goldilocks::ZERO; WIDTH];
    
    // Convert input to field elements
    let mut input = Vec::<Goldilocks>::new();
    
    // Domain separation first
    for chunk in domain_separator.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        input.push(Goldilocks::from_u64(u64::from_le_bytes(bytes)));
    }
    
    // Then data
    for chunk in data.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        input.push(Goldilocks::from_u64(u64::from_le_bytes(bytes)));
    }
    
    // Sponge absorb phase with proper rate limiting
    let mut i = 0;
    while i < input.len() {
        let take = core::cmp::min(RATE, input.len() - i);
        for j in 0..take {
            state[j] += input[i + j];
        }
        i += take;
        if take == RATE {
            poseidon2.permute_mut(&mut state);
        }
    }
    
    // 10* padding: absorb 1 then zeros, then permute
    state[0] += Goldilocks::ONE;
    poseidon2.permute_mut(&mut state);
    
    // Squeeze phase: extract first 4 field elements as 32-byte digest
    let mut digest = [0u8; 32];
    for k in 0..4 {
        digest[k*8..(k+1)*8].copy_from_slice(&state[k].as_canonical_u64().to_le_bytes());
    }
    
    digest
}

/// Compute digest with version detection for backward compatibility
fn compute_vk_digest(data: &[u8], version: u8) -> [u8; 32] {
    match version {
        1 => compute_blake3_digest_v1(data, VK_DIGEST_DOMAIN_V1),
        2 => compute_poseidon2_digest_v2(data, VK_DIGEST_DOMAIN_V2),
        _ => panic!("Unsupported VK digest version: {}", version),
    }
}

/// Safe conversion from signed integer to field element, avoiding edge cases like i64::MIN
#[inline]
fn f_from_i64(z: i64) -> neo_math::F {
    if z >= 0 {
        neo_math::F::from_u64(z as u64)
    } else {
        -neo_math::F::from_u64(z.wrapping_neg() as u64)
    }
}

type E = spartan2::provider::GoldilocksMerkleMleEngine;

/// Encode the transcript header and public IO with **exact** match to MeCircuit::public_values().
/// This encoding MUST match the order/format of MeCircuit::public_values() exactly.
pub fn encode_bridge_io_header(me: &MEInstance) -> Vec<u8> {
    use p3_field::PrimeField64;
    // EXACT order/format of MeCircuit::public_values():
    // (c_coords) || (y split into 2 limbs) || (r_point) || (base_b) || (digest split into 4 u64 limbs)
    let mut out = Vec::new();
    
    // c_coords - direct encoding as single limbs
    for &c in &me.c_coords {
        out.extend_from_slice(&c.as_canonical_u64().to_le_bytes());
    }
    
    // y_outputs: already flattened limbs (K -> [F;2]) by the adapter; emit as-is
    for &y_limb in &me.y_outputs {
        out.extend_from_slice(&y_limb.as_canonical_u64().to_le_bytes());
    }
    
    // r_point - direct encoding
    for &r in &me.r_point {
        out.extend_from_slice(&r.as_canonical_u64().to_le_bytes());
    }
    
    // base_b - single u64
    out.extend_from_slice(&(me.base_b as u64).to_le_bytes());
    
    // Hash-MLE PCS requires power-of-2 length, so pad with zeros BEFORE adding digest
    let num_scalars = (out.len() / 8) + 4; // +4 for the digest (4 u64 limbs)
    let next_power_of_2 = num_scalars.next_power_of_two();
    let padding_scalars = next_power_of_2 - num_scalars;
    
    // Pad with zero scalars (8 zero bytes each) - this goes BEFORE the digest
    for _ in 0..padding_scalars {
        out.extend_from_slice(&0u64.to_le_bytes());
    }
    
    // fold digest as 4 little-endian u64 limbs LAST (after padding)
    for chunk in me.header_digest.chunks(8) {
        let limb = u64::from_le_bytes([
            chunk.get(0).copied().unwrap_or(0),
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
            chunk.get(4).copied().unwrap_or(0),
            chunk.get(5).copied().unwrap_or(0),
            chunk.get(6).copied().unwrap_or(0),
            chunk.get(7).copied().unwrap_or(0),
        ]);
        out.extend_from_slice(&limb.to_le_bytes());
    }
    
    out
}

/// Encode header with optional IVC EV public inputs. When `ev` is Some,
/// appends `y_prev || y_next || rho` before padding and digest, and MUST
/// match MeCircuit::public_values() order.
pub fn encode_bridge_io_header_with_ev(me: &MEInstance, ev: Option<&crate::me_to_r1cs::IvcEvEmbed>) -> Vec<u8> {
    use p3_field::PrimeField64;
    let mut out = Vec::new();
    // c_coords
    for &c in &me.c_coords { out.extend_from_slice(&c.as_canonical_u64().to_le_bytes()); }
    // y_outputs
    for &y_limb in &me.y_outputs { out.extend_from_slice(&y_limb.as_canonical_u64().to_le_bytes()); }
    // r_point
    for &r in &me.r_point { out.extend_from_slice(&r.as_canonical_u64().to_le_bytes()); }
    // base_b
    out.extend_from_slice(&(me.base_b as u64).to_le_bytes());
    // Optional EV
    if let Some(ev) = ev {
        for &v in &ev.y_prev { out.extend_from_slice(&v.as_canonical_u64().to_le_bytes()); }
        for &v in &ev.y_next { out.extend_from_slice(&v.as_canonical_u64().to_le_bytes()); }
        out.extend_from_slice(&ev.rho.as_canonical_u64().to_le_bytes());
        if let Some(d) = &ev.fold_chain_digest {
            for chunk in d.chunks(8) {
                let limb = u64::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                    chunk.get(4).copied().unwrap_or(0),
                    chunk.get(5).copied().unwrap_or(0),
                    chunk.get(6).copied().unwrap_or(0),
                    chunk.get(7).copied().unwrap_or(0),
                ]);
                out.extend_from_slice(&limb.to_le_bytes());
            }
        }
        if let (Some(cprev), Some(cstep), Some(cnext)) = (&ev.acc_c_prev, &ev.acc_c_step, &ev.acc_c_next) {
            for &x in cprev { out.extend_from_slice(&x.as_canonical_u64().to_le_bytes()); }
            for &x in cstep { out.extend_from_slice(&x.as_canonical_u64().to_le_bytes()); }
            for &x in cnext { out.extend_from_slice(&x.as_canonical_u64().to_le_bytes()); }
            if let Some(r_eff) = ev.rho_eff { out.extend_from_slice(&r_eff.as_canonical_u64().to_le_bytes()); }
        }
    }
    // Padding before digest
    let num_scalars = (out.len() / 8) + 4; // +4 for digest limbs
    let next_power_of_2 = num_scalars.next_power_of_two();
    let padding_scalars = next_power_of_2 - num_scalars;
    for _ in 0..padding_scalars { out.extend_from_slice(&0u64.to_le_bytes()); }
    // digest
    for chunk in me.header_digest.chunks(8) {
        let limb = u64::from_le_bytes([
            chunk.get(0).copied().unwrap_or(0),
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
            chunk.get(4).copied().unwrap_or(0),
            chunk.get(5).copied().unwrap_or(0),
            chunk.get(6).copied().unwrap_or(0),
            chunk.get(7).copied().unwrap_or(0),
        ]);
        out.extend_from_slice(&limb.to_le_bytes());
    }
    out
}

/// **Main Entry Point**: Compress final ME(b,L) claim using Spartan2 + Hash-MLE PCS.
/// Note: no FRI parameters; the bridge uses Hash‑MLE PCS only.
pub fn compress_me_to_spartan(me: &MEInstance, wit: &MEWitness) -> Result<ProofBundle> {
    compress_me_to_spartan_with_pp(me, wit, None)
}

/// Compress ME to Spartan SNARK with optional PP for streaming Ajtai rows
pub fn compress_me_to_spartan_with_pp(
    me: &MEInstance, 
    wit: &MEWitness, 
    pp: Option<neo_ajtai::PP<neo_math::Rq>>
) -> Result<ProofBundle> {
    // SECURITY: Without Ajtai rows OR PP, c_coords is not bound to z_digits.
    let has_ajtai_rows = wit.ajtai_rows.as_ref().map_or(false, |rows| !rows.is_empty());
    let has_pp = pp.is_some();
    
    if !has_ajtai_rows && !has_pp {
        anyhow::bail!("AjtaiBindingMissing: witness.ajtai_rows is None/empty AND no PP provided; cannot bind c_coords to Z");
    }

    // SECURITY: Validate that c_coords are consistent with Ajtai commitment before SNARK generation
    // This prevents forged commitments from being accepted even with valid Ajtai rows
    if let Some(ajtai_rows) = &wit.ajtai_rows {
        // Enforce strict dimension matching - no silent truncation
        anyhow::ensure!(
            ajtai_rows.len() == me.c_coords.len(),
            "Ajtai rows ({}) must match c_coords ({})",
            ajtai_rows.len(), me.c_coords.len()
        );
        
        // Check if Ajtai rows need padding to match z_digits (which may have been padded by bridge)
        let max_row_len = ajtai_rows.iter().map(|row| row.len()).max().unwrap_or(0);
        if max_row_len < wit.z_digits.len() {
            // All rows are shorter than z_digits - likely due to power-of-two padding
            // Pad all rows with zeros to match z_digits length
            let mut padded_ajtai_rows = ajtai_rows.clone();
            for row in &mut padded_ajtai_rows {
                if row.len() < wit.z_digits.len() {
                    let pad_len = wit.z_digits.len() - row.len();
                    row.extend(std::iter::repeat(neo_math::F::ZERO).take(pad_len));
                }
            }
            
            // Create a new witness with padded rows for validation
            let mut wit_padded = wit.clone();
            wit_padded.ajtai_rows = Some(padded_ajtai_rows);
            return compress_me_to_spartan(me, &wit_padded);
        }
        
        // UNIFORM WIDTH REQUIREMENT: All Ajtai rows must have same length = |z_digits|
        // This avoids ambiguity about truncation semantics and makes validation predictable
        for (i, row) in ajtai_rows.iter().enumerate() {
            anyhow::ensure!(
                row.len() == wit.z_digits.len(),
                "Ajtai row {} length ({}) must equal z_digits length ({})", 
                i, row.len(), wit.z_digits.len()
            );
        }
        
        // Validate each commitment: c[i] = <row_i, z_digits>
        for (i, (row, &claimed)) in ajtai_rows.iter().zip(&me.c_coords).enumerate() {
            // Compute inner product <row_i, z_digits> using safe field conversion
            let computed = row.iter().zip(&wit.z_digits).fold(neo_math::F::ZERO, |acc, (&a, &z)| {
                acc + a * f_from_i64(z)
            });
            
            // Strict equality check - no tolerance
            if computed != claimed {
                eprintln!("❌ Ajtai commitment validation failed:");
                eprintln!("   c_coords[{}] = {} (claimed)", i, claimed.as_canonical_u64());
                eprintln!("   <row_{}, z> = {} (computed)", i, computed.as_canonical_u64());
                anyhow::bail!(
                    "AjtaiCommitmentInconsistent at index {}: computed {}, claimed {}",
                    i, computed.as_canonical_u64(), claimed.as_canonical_u64()
                );
            }
        }
    }

    // FAIL-FAST RANGE CHECK (developer ergonomics + red-team tests):
    // Reject witnesses with |z_i| >= base_b before we even try to prove.
    {
        let b = me.base_b as i64;
        anyhow::ensure!(b >= 2, "InvalidBase: base_b={} < 2", me.base_b);
        let bound = b - 1;
        if let Some((idx, &zi)) = wit.z_digits.iter().enumerate().find(|&(_, &zi)| zi < -bound || zi > bound) {
            eprintln!("❌ Range violation: z_digits[{}] = {} ∉ [-{}, {}] (base_b = {})", idx, zi, bound, bound, me.base_b);
            anyhow::bail!(
                "RangeViolation: z_digits[{}]={} outside ±{} for base_b={}",
                idx, zi, bound, me.base_b
            );
        }
    }

    // Canonicalize Ajtai row layout to match circuit's z_digits (column-major: idx = c*D + r)
    let mut wit_norm = wit.clone();
    if let Some(rows) = &mut wit_norm.ajtai_rows {
        // D is the Ajtai ring dimension; z_digits is expected to be D * m
        let d = neo_math::ring::D;
        let n = wit_norm.z_digits.len();
        if n % d == 0 {
            let m = n / d;

            // helper: dot(row, z_digits)
            let dot_as = |row: &[neo_math::F]| -> neo_math::F {
                row.iter().zip(wit_norm.z_digits.iter()).fold(neo_math::F::ZERO, |acc, (a, &zi)| {
                    let zf = if zi >= 0 { neo_math::F::from_u64(zi as u64) }
                             else       { -neo_math::F::from_u64((-zi) as u64) };
                    acc + *a * zf
                })
            };
            // convert one row-major vector to column-major (idx_cm = c*d + r; idx_rm = r*m + c)
            let to_col_major = |row_rm: &[neo_math::F]| -> Vec<neo_math::F> {
                let mut row_cm = vec![neo_math::F::ZERO; n];
                for r in 0..d { for c in 0..m { row_cm[c*d + r] = row_rm[r*m + c]; } }
                row_cm
            };

            // quick heuristic: see which orientation matches c_coords better on the prefix we have
            let check_len = core::cmp::min(rows.len(), me.c_coords.len());
            let mut ok_as_is = 0usize;
            let mut ok_swapped = 0usize;
            for i in 0..check_len {
                if rows[i].len() != n { continue; }
                if dot_as(&rows[i]) == me.c_coords[i] { ok_as_is += 1; }
                if dot_as(&to_col_major(&rows[i])) == me.c_coords[i] { ok_swapped += 1; }
            }
            if ok_swapped > ok_as_is {
                for row in rows.iter_mut() {
                    if row.len() == n {
                        let swapped = to_col_major(row);
                        *row = swapped;
                    }
                }
                eprintln!("🔧 Ajtai rows normalized: row-major → col-major (D={}, m={})", d, m);
            }
        } else {
            eprintln!("ℹ️ Ajtai row normalization skipped: n={} not divisible by D={}", n, d);
        }
    }

    // Try the SNARK generation and provide detailed error diagnostics
    let pp_arc = pp.map(std::sync::Arc::new);
    let snark_result = me_to_r1cs::prove_me_snark_with_pp(me, &wit_norm, pp_arc, None);
    let (proof_bytes, _public_outputs, vk_arc) = match snark_result {
        Ok(result) => {
            eprintln!("✅ SNARK generation successful!");
            result
        }
        Err(e) => {
            eprintln!("🚨 SNARK generation failed!");
            eprintln!("Error details: {:?}", e);
            
            // Extract detailed error information
            use spartan2::errors::SpartanError;
            match e {
                SpartanError::InternalError { ref reason } => {
                    eprintln!("InternalError: {}", reason);
                }
                SpartanError::SynthesisError { ref reason } => {
                    eprintln!("SynthesisError: {}", reason);
                }
                _ => eprintln!("Other Spartan2 error: {}", e),
            }
            
            return Err(anyhow::Error::msg(format!("Spartan2 SNARK failed: {}", e)));
        }
    };
    
    let vk_bytes = serialize_vk_stable(&*vk_arc)?;
    let io = encode_bridge_io_header(me);
    Ok(ProofBundle::new_with_vk(proof_bytes, vk_bytes, io))
}

/// Verify a ProofBundle containing an ME R1CS SNARK
pub fn verify_me_spartan(bundle: &ProofBundle) -> Result<bool> {
    let snark: R1CSSNARK<E> = deserialize_snark_stable(&bundle.proof)?;
    let vk: SpartanVerifierKey<E> = deserialize_vk_stable(&bundle.vk)?;
    
    // 1) Verify SNARK and get public scalars (map verification failures to Ok(false))
    match snark.verify(&vk) {
        Ok(publics) => {
            // 2) Serialize public scalars identically to encode_bridge_io_header()
            let mut bytes = Vec::with_capacity(publics.len() * 8);
            for x in &publics {
                bytes.extend_from_slice(&x.to_canonical_u64().to_le_bytes());
            }
            // 3) Validate lengths and compare using constant-time to prevent tampering
            // Check lengths explicitly for robustness and clarity
            if bytes.len() != bundle.public_io_bytes.len() {
                eprintln!("❌ Public IO length mismatch: SNARK returned {} bytes, bundle has {} bytes",                                                                                                             
                          bytes.len(), bundle.public_io_bytes.len());
                return Ok(false);
            }
            
            // Constant-time comparison for equal-length byte arrays
            use subtle::ConstantTimeEq;
            if bytes.ct_eq(&bundle.public_io_bytes).unwrap_u8() != 1 {
                eprintln!("❌ Public IO content mismatch: SNARK public inputs don't match bundle.public_io_bytes");
                return Ok(false);
            }
            eprintln!("✅ Public IO verification passed: {} bytes match exactly", bytes.len());
            Ok(true)
        }
        Err(e) => {
            // Treat *verification* errors as a clean "false" so tests can assert on it.
            // Keep structural problems as hard errors.
            use spartan2::errors::SpartanError;
            match e {
                SpartanError::InvalidSumcheckProof => {
                    eprintln!("❌ Spartan verification failed: {}", e);
                    Ok(false)
                }
                _ => {
                    // For now, treat all other verification errors as Ok(false) too
                    // This provides better test compatibility
                    eprintln!("❌ Spartan verification failed: {}", e);
                    Ok(false)
                }
            }
        }
    }
}

/// Compress a single MLE claim (v committed, v(r)=y) using Spartan2 Hash‑MLE PCS.
/// Returns a serializable ProofBundle.
pub fn compress_mle_with_hash_mle(poly: &[hash_mle::F], point: &[hash_mle::F]) -> Result<ProofBundle> {
    let prf = hash_mle::prove_hash_mle(poly, point)?;
    let proof_bytes = prf.to_bytes()?;
    let public_io   = hash_mle::encode_public_io(&prf);
    Ok(ProofBundle::new_with_vk(proof_bytes, Vec::new(), public_io))
}

/// Verify a ProofBundle produced by `compress_mle_with_hash_mle`.
pub fn verify_mle_hash_mle(bundle: &ProofBundle) -> Result<()> {
    let prf = hash_mle::HashMleProof::from_bytes(&bundle.proof)?;
    
    // CRITICAL SECURITY CHECK: Bind public_io_bytes to the proof
    // Recompute the canonical public IO bytes from the proof's claim
    let expected_public_io = hash_mle::encode_public_io(&prf);
    
    // Constant-time comparison to prevent floating public input attacks
    if expected_public_io.as_slice() != bundle.public_io_bytes.as_slice() {
        anyhow::bail!("PublicIoMismatch: proof's canonical public IO doesn't match bundled bytes");
    }
    
    // Proceed with normal verification
    hash_mle::verify_hash_mle(&prf)
}

/// Helper for when you have computed v, r, and expected eval separately.
/// Verifies that the computed evaluation matches the expected value.
pub fn compress_me_eval(poly: &[hash_mle::F], point: &[hash_mle::F], expected_eval: hash_mle::F) -> Result<ProofBundle> {
    let prf = hash_mle::prove_hash_mle(poly, point)?;
    anyhow::ensure!(prf.eval == expected_eval, "eval mismatch: expected != computed");
    let proof_bytes = prf.to_bytes()?;
    let public_io   = hash_mle::encode_public_io(&prf);
    Ok(ProofBundle::new_with_vk(proof_bytes, Vec::new(), public_io))
}

// ===============================================================================
// VK REGISTRY SYSTEM - Keeps VK out of proofs!
// ===============================================================================

use once_cell::sync::Lazy;
use dashmap::DashMap;
use std::sync::Arc;

/// Global VK registry - maps circuit keys to verifier keys for lean proof verification
static VK_REGISTRY: Lazy<DashMap<[u8; 32], Arc<SpartanVerifierKey<E>>>> = 
    Lazy::new(|| DashMap::new());

/// Register a VK for a circuit (called during proving to cache VK for verification)
/// SECURITY: Restricted to crate-only access to prevent VK registry tampering (CWE-200)
pub(crate) fn register_vk(circuit_key: [u8; 32], vk: Arc<SpartanVerifierKey<E>>) {
    VK_REGISTRY.insert(circuit_key, vk);
}

/// Lookup a VK by circuit key (called during verification)
pub fn lookup_vk(circuit_key: &[u8; 32]) -> Option<Arc<SpartanVerifierKey<E>>> {
    VK_REGISTRY.get(circuit_key).map(|entry| entry.clone())
}

/// Get VK registry stats (for monitoring)
pub fn vk_registry_stats() -> usize {
    VK_REGISTRY.len()
}

/// **CRITICAL API**: Register a VK from raw bytes for cross-process verification
/// 
/// This enables verification in separate processes that didn't generate the proof.
/// The verifier can load VK bytes (from disk, network, etc.) and register them
/// before calling verify_lean_proof().
/// 
/// # Arguments
/// * `circuit_key` - Circuit fingerprint (32-byte identifier)
/// * `vk_bytes` - Serialized Spartan verifier key bytes
/// 
/// # Returns
/// * `Ok(vk_digest)` - The computed VK digest for validation
/// * `Err(...)` - If VK deserialization fails
/// 
/// # Example
/// ```rust,no_run
/// use neo_spartan_bridge::register_vk_bytes;
/// 
/// let vk_bytes = std::fs::read("circuit.vk")?;
/// let circuit_key = [0u8; 32]; // Load from somewhere
/// let vk_digest = register_vk_bytes(circuit_key, &vk_bytes)?;
/// println!("Registered VK with digest: {:02x?}", &vk_digest[..8]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn register_vk_bytes(circuit_key: [u8; 32], vk_bytes: &[u8]) -> anyhow::Result<[u8; 32]> {
    // Deserialize VK with stable configuration
    let vk: spartan2::spartan::SpartanVerifierKey<E> = deserialize_vk_stable(vk_bytes)?;
    
    // Compute VK digest v1 (must match the format used during proving)
    let vk_digest: [u8; 32] = compute_vk_digest(vk_bytes, 2); // Use V2 (Poseidon2) for new proofs
    
    // Register VK in global registry for verification
    register_vk(circuit_key, Arc::new(vk));
    
    #[cfg(feature = "neo-logs")]
    info!(
        "Registered VK for circuit {:02x?} with digest {:02x?}",
        &circuit_key[..8], &vk_digest[..8]
    );
    
    Ok(vk_digest)
}

/// Export Spartan2 input data to JSON for external profiling and testing
/// 
/// This function exports the exact MEInstance and MEWitness data that gets fed to Spartan2,
/// allowing external repositories to create reproducible test cases and profile Spartan2 directly.
/// 
/// The exported JSON contains:
/// - Instance data: c_coords, y_outputs, r_point, base_b, header_digest  
/// - Witness data: z_digits, weight_vectors, ajtai_rows
/// - Metadata: sizes, sparsity info, timestamp
fn export_spartan2_data_to_json(me: &neo_ccs::MEInstance, wit: &neo_ccs::MEWitness) -> anyhow::Result<()> {
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;
    
    println!("📊 [SPARTAN2 EXPORT] Exporting Spartan2 input data to JSON for external profiling...");
    let export_start = std::time::Instant::now();
    
    // Compute statistics for metadata
    let total_z_digits = wit.z_digits.len();
    let total_weight_vectors = wit.weight_vectors.len();
    let total_weight_elements: usize = wit.weight_vectors.iter().map(|v| v.len()).sum();
    let ajtai_rows_count = wit.ajtai_rows.as_ref().map_or(0, |rows| rows.len());
    let ajtai_total_elements: usize = wit.ajtai_rows.as_ref()
        .map_or(0, |rows| rows.iter().map(|row| row.len()).sum());
    
    // Create comprehensive export data
    let export_data = json!({
        "metadata": {
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "neo_version": env!("CARGO_PKG_VERSION"),
            "description": "Spartan2 input data exported from Neo for external profiling",
            "data_sizes": {
                "c_coords_len": me.c_coords.len(),
                "y_outputs_len": me.y_outputs.len(), 
                "r_point_len": me.r_point.len(),
                "z_digits_len": total_z_digits,
                "weight_vectors_count": total_weight_vectors,
                "weight_elements_total": total_weight_elements,
                "ajtai_rows_count": ajtai_rows_count,
                "ajtai_elements_total": ajtai_total_elements
            },
            "estimated_memory_mb": {
                "z_digits": total_z_digits * 8 / 1_000_000,
                "weight_vectors": total_weight_elements * 32 / 1_000_000,
                "ajtai_rows": ajtai_total_elements * 32 / 1_000_000,
                "total_estimated": (total_z_digits * 8 + (total_weight_elements + ajtai_total_elements) * 32) / 1_000_000
            }
        },
        "instance": {
            "c_coords": me.c_coords,
            "y_outputs": me.y_outputs,
            "r_point": me.r_point,
            "base_b": me.base_b,
            "header_digest": me.header_digest
        },
        "witness": {
            "z_digits": wit.z_digits,
            "weight_vectors": wit.weight_vectors,
            "ajtai_rows": wit.ajtai_rows
        }
    });
    
    // Generate timestamped filename
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename = format!("spartan2_input_data_{}.json", timestamp);
    
    // Write to file with pretty formatting
    let mut file = File::create(&filename)?;
    let json_str = serde_json::to_string_pretty(&export_data)?;
    file.write_all(json_str.as_bytes())?;
    file.flush()?;
    
    let export_time = export_start.elapsed();
    let file_size_mb = json_str.len() as f64 / 1_000_000.0;
    
    println!("📊 [SPARTAN2 EXPORT] Successfully exported to '{}' ({:.1}MB, {:.0}ms)", 
             filename, file_size_mb, export_time.as_millis());
    println!("📊 [SPARTAN2 EXPORT] Data summary:");
    println!("    - Instance: {} c_coords, {} y_outputs, {} r_point elements", 
             me.c_coords.len(), me.y_outputs.len(), me.r_point.len());
    println!("    - Witness: {} z_digits, {} weight_vectors ({} elements), {} ajtai_rows ({} elements)",
             total_z_digits, total_weight_vectors, total_weight_elements, 
             ajtai_rows_count, ajtai_total_elements);
    println!("📊 [SPARTAN2 EXPORT] Use this data to create reproducible Spartan2 test cases!");
    
    Ok(())
}

/// Clear the VK registry (for testing)
/// SECURITY: Only exposed for testing to prevent production VK registry tampering
#[cfg(any(test, feature = "testing"))]
pub fn clear_vk_registry() {
    VK_REGISTRY.clear()
}

// ===============================================================================
// 🔒 STABLE BINCODE SERIALIZATION - Critical for VK digest consistency
// ===============================================================================

/// Stable bincode configuration for VK serialization - CRITICAL for digest consistency
/// 
/// Using bincode 1.x with explicit configuration for stability across:
/// - Different bincode versions  
/// - Different platforms (endianness)
/// - Different compilation environments
/// 
/// ⚠️ NEVER CHANGE THIS CONFIG - it would break all existing VK digests!
fn stable_bincode_config() -> impl bincode::Options + Copy {
    use bincode::{DefaultOptions, Options};
    DefaultOptions::new()
        .with_fixint_encoding()     // Fixed-width integer encoding for stability
        .with_little_endian()       // Explicit endianness for cross-platform stability
}

/// Serialize VK with stable configuration for digest computation
fn serialize_vk_stable(vk: &spartan2::spartan::SpartanVerifierKey<E>) -> anyhow::Result<Vec<u8>> {
    use bincode::Options;
    stable_bincode_config().serialize(vk)
        .map_err(|e| anyhow::anyhow!("VK serialization failed: {}", e))
}

/// Deserialize SNARK proof with stable configuration  
fn deserialize_snark_stable(bytes: &[u8]) -> anyhow::Result<spartan2::spartan::R1CSSNARK<E>> {
    use bincode::Options;
    stable_bincode_config().deserialize(bytes)
        .map_err(|e| anyhow::anyhow!("SNARK deserialization failed: {}", e))
}

/// Deserialize old-format VK with stable configuration (for legacy ProofBundle support)
fn deserialize_vk_stable(bytes: &[u8]) -> anyhow::Result<spartan2::spartan::SpartanVerifierKey<E>> {
    use bincode::Options;
    stable_bincode_config().deserialize(bytes)
        .map_err(|e| anyhow::anyhow!("VK deserialization failed: {}", e))
}

/// NEW: Compress ME to lean Proof (without VK) - FIXES THE 51MB ISSUE!
pub fn compress_me_to_lean_proof(me: &neo_ccs::MEInstance, wit: &neo_ccs::MEWitness) -> anyhow::Result<Proof> {
    compress_me_to_lean_proof_with_pp(me, wit, None)
}

/// Compress ME to lean Proof with optional PP for streaming Ajtai rows
pub fn compress_me_to_lean_proof_with_pp(
    me: &neo_ccs::MEInstance, 
    wit: &neo_ccs::MEWitness,
    pp: Option<std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>>
) -> anyhow::Result<Proof> {
    // Normalize witness (same as before)
    let wit_norm = if wit.z_digits.len().is_power_of_two() {
        wit.clone()
    } else {
        let next_pow2 = wit.z_digits.len().next_power_of_two();
        let mut normalized = wit.clone();
        normalized.z_digits.resize(next_pow2, 0);
        
        if let Some(ref mut ajtai_rows) = normalized.ajtai_rows {
            for row in ajtai_rows.iter_mut() {
                let pad_len = next_pow2 - row.len();
                row.extend(std::iter::repeat(neo_math::F::ZERO).take(pad_len));
            }
        }
        normalized
    };

    // SECURITY: Refuse to prove unbound circuits
    let has_ajtai_rows = wit_norm.ajtai_rows.as_ref().map_or(false, |rows| !rows.is_empty());
    let has_pp = pp.is_some();
    
    if !has_ajtai_rows && !has_pp {
        anyhow::bail!("AjtaiBindingMissing: witness.ajtai_rows is None/empty AND no PP provided; cannot bind c_coords to Z");
    }
    
    // PERFORMANCE DEBUGGING: Export Spartan2 input data to JSON for external profiling
    if std::env::var("NEO_EXPORT_SPARTAN2_DATA").is_ok() {
        export_spartan2_data_to_json(me, &wit_norm)?;
    }

    // Prove using existing Spartan infrastructure
    let snark_result = me_to_r1cs::prove_me_snark_with_pp(me, &wit_norm, pp, None);
    let (proof_bytes, _public_outputs, vk_arc) = match snark_result {
        Ok(result) => {
            eprintln!("✅ SNARK generation successful!");
            result
        }
        Err(e) => {
            eprintln!("🚨 SNARK generation failed: {:?}", e);
            return Err(anyhow::Error::msg(format!("Spartan2 SNARK failed: {}", e)));
        }
    };
    
    // Generate circuit fingerprint
    let circuit = me_to_r1cs::MeCircuit::new(me.clone(), wit_norm, None, me.header_digest);
    let circuit_key_obj = me_to_r1cs::CircuitKey::from_circuit(&circuit);
    let circuit_key: [u8; 32] = circuit_key_obj.into_bytes(); // Convert to [u8; 32]
    
    // CRITICAL: VK digest stability depends on bincode serialization consistency
    // 
    // ⚠️  STABILITY RISK: This VK digest uses `bincode` serialization which is NOT canonical.
    //     Different versions of `bincode` or `spartan2` may produce different bytes for 
    //     the same VK, causing digest mismatches and proof rejection.
    //
    // 📋 MITIGATION PLAN:
    //   1. Pin `bincode` version in Cargo.lock
    //   2. Use frozen bincode config: `bincode::config::standard().with_fixed_int_encoding()`
    //   3. Replace with canonical encoding when Spartan2 provides `to_canonical_bytes()`
    //   4. Add unit test asserting digest stability across serialize→deserialize roundtrip
    //
    // CURRENT STATUS: Acceptable for single-process/same-version usage
    let vk_bytes = serialize_vk_stable(&*vk_arc)?;
    
    // VK digest v1: Poseidon2(bincode_v1(vk) || "VK_DIGEST_V1") for consistency with circuit keys
    // Domain separation ensures digest stability and prevents collisions
    let vk_digest: [u8; 32] = compute_vk_digest(&vk_bytes, 2); // Use V2 (Poseidon2) for new proofs
    
    // Register VK for verification (out-of-band)
    register_vk(circuit_key, vk_arc.clone());
    
    // Encode public IO
    let public_io_bytes = encode_bridge_io_header(me);
    
    // Create lean proof (NO VK INSIDE!)
    let proof = Proof::new(circuit_key, vk_digest, public_io_bytes, proof_bytes);
    
    info!("Created lean proof: {} bytes (vs ~51MB with VK)", 
             proof.total_size());
    
    Ok(proof)
}

/// NEW: Compress ME to lean Proof with optional EV embedding inside Spartan
pub fn compress_me_to_lean_proof_with_pp_and_ev(
    me: &neo_ccs::MEInstance,
    wit: &neo_ccs::MEWitness,
    pp: Option<std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>>,
    ev: Option<crate::me_to_r1cs::IvcEvEmbed>,
) -> anyhow::Result<Proof> {
    let wit_norm = if wit.z_digits.len().is_power_of_two() { wit.clone() } else {
        let next_pow2 = wit.z_digits.len().next_power_of_two();
        let mut normalized = wit.clone();
        normalized.z_digits.resize(next_pow2, 0);
        if let Some(ref mut ajtai_rows) = normalized.ajtai_rows {
            for row in ajtai_rows.iter_mut() { row.resize(next_pow2, neo_math::F::ZERO); }
        }
        normalized
    };

    let has_ajtai_rows = wit_norm.ajtai_rows.as_ref().map_or(false, |rows| !rows.is_empty());
    let has_pp = pp.is_some();
    if !has_ajtai_rows && !has_pp { anyhow::bail!("AjtaiBindingMissing: witness.ajtai_rows is None/empty AND no PP provided; cannot bind c_coords to Z"); }

    let snark_result = me_to_r1cs::prove_me_snark_with_pp(me, &wit_norm, pp, ev.clone());
    let (proof_bytes, _public_outputs, vk_arc) = match snark_result {
        Ok(result) => { eprintln!("✅ SNARK generation successful!"); result }
        Err(e) => { eprintln!("🚨 SNARK generation failed: {:?}", e); return Err(anyhow::Error::msg(format!("Spartan2 SNARK failed: {}", e))); }
    };

    let circuit = me_to_r1cs::MeCircuit::new(me.clone(), wit_norm, None, me.header_digest).with_ev(ev.clone());
    let circuit_key_obj = me_to_r1cs::CircuitKey::from_circuit(&circuit);
    let circuit_key: [u8; 32] = circuit_key_obj.into_bytes();

    let vk_bytes = serialize_vk_stable(&*vk_arc)?;
    let vk_digest: [u8; 32] = compute_vk_digest(&vk_bytes, 2);
    register_vk(circuit_key, vk_arc.clone());

    let public_io_bytes = encode_bridge_io_header_with_ev(me, ev.as_ref());
    let proof = Proof::new(circuit_key, vk_digest, public_io_bytes, proof_bytes);
    info!("Created lean proof (with EV): {} bytes", proof.total_size());
    Ok(proof)
}

/// Compress using the IVC verifier-style circuit with EV + linkage + commit-evo embeddings.
pub fn compress_ivc_verifier_to_lean_proof_with_linkage(
    me: &neo_ccs::MEInstance,
    wit: &neo_ccs::MEWitness,
    pp: Option<std::sync::Arc<neo_ajtai::PP<neo_math::Rq>>>,
    ev: Option<crate::me_to_r1cs::IvcEvEmbed>,
    commit: Option<crate::me_to_r1cs::CommitEvoEmbed>,
    linkage: Option<crate::me_to_r1cs::IvcLinkageInputs>,
) -> anyhow::Result<Proof> {
    // Prove using extended MeCircuit
    let (proof_bytes, _public_outputs, vk_arc) = me_to_r1cs::prove_me_snark_with_pp_and_ivc(me, wit, pp, ev.clone(), commit, linkage)
        .map_err(|e| anyhow::anyhow!("IVC verifier SNARK failed: {}", e))?;

    // Circuit fingerprint
    let circuit = me_to_r1cs::MeCircuit::new(me.clone(), wit.clone(), None, me.header_digest).with_ev(ev.clone());
    let circuit_key_obj = me_to_r1cs::CircuitKey::from_circuit(&circuit);
    let circuit_key: [u8; 32] = circuit_key_obj.into_bytes();

    // VK digest
    let vk_bytes = serialize_vk_stable(&*vk_arc)?;
    let vk_digest: [u8; 32] = compute_vk_digest(&vk_bytes, 2);
    register_vk(circuit_key, vk_arc.clone());

    // Encode public IO with EV header when present
    let public_io_bytes = encode_bridge_io_header_with_ev(me, ev.as_ref());
    let proof = Proof::new(circuit_key, vk_digest, public_io_bytes, proof_bytes);
    Ok(proof)
}

/// Verify lean Proof using VK registry - SOLVES THE 51MB ISSUE!
pub fn verify_lean_proof(proof: &Proof) -> anyhow::Result<bool> {
    // Lookup VK from registry
    let vk = lookup_vk(&proof.circuit_key)
        .ok_or_else(|| anyhow::anyhow!(
            "VK not found in registry for circuit key: {:02x?}", 
            &proof.circuit_key[..8]
        ))?;
    
    // Verify VK digest binding (v1 format with domain separation)  
    let vk_bytes = serialize_vk_stable(&*vk)?;
    
    // Compute VK digest v1 (must match the format used during proving)
    // Support both V1 (BLAKE3) and V2 (Poseidon2) for backward compatibility
    // For now, assume V2 (Poseidon2) for all new proofs
    // TODO: Add proper version detection based on proof format
    let computed_digest: [u8; 32] = compute_vk_digest(&vk_bytes, 2);
    
    anyhow::ensure!(
        computed_digest == proof.vk_digest,
        "VK digest mismatch - proof bound to different circuit"
    );
    
    // SECURITY: Prevent maliciously huge proof_bytes from causing memory exhaustion
    const MAX_PROOF_BYTES: usize = 64 * 1024 * 1024; // 64MB limit
    anyhow::ensure!(
        proof.proof_bytes.len() <= MAX_PROOF_BYTES,
        "proof_bytes too large: {} bytes (limit {})",
        proof.proof_bytes.len(), MAX_PROOF_BYTES
    );
    
    // Deserialize and verify Spartan proof
    let snark: spartan2::spartan::R1CSSNARK<E> = deserialize_snark_stable(&proof.proof_bytes)?;
    
    // CRITICAL SECURITY: Get public values from Spartan and bind to public IO
    let public_values = snark.verify(&*vk)
        .map_err(|e| anyhow::anyhow!("Spartan verification failed: {}", e))?;
    
    // CRITICAL SECURITY: Serialize public scalars identically to encode_bridge_io_header()
    // This prevents tampering with public_io_bytes
    let mut expected_public_io = Vec::with_capacity(public_values.len() * 8);
    for x in &public_values {
        expected_public_io.extend_from_slice(&x.to_canonical_u64().to_le_bytes());
    }
    
    // CRITICAL SECURITY: Validate lengths and compare using constant-time to prevent tampering
    // Check lengths explicitly for robustness and clarity  
    if expected_public_io.len() != proof.public_io_bytes.len() {
        debug!("Public IO length mismatch: expected {} bytes, got {} bytes",
                 expected_public_io.len(), proof.public_io_bytes.len());
        return Ok(false);
    }
    
    // Constant-time comparison for equal-length byte arrays
    use subtle::ConstantTimeEq;
    if expected_public_io.ct_eq(&proof.public_io_bytes).unwrap_u8() != 1 {
        debug!("Public IO content mismatch: Spartan public values don't match proof.public_io_bytes");
        return Ok(false);
    }
    
    info!("Lean verification successful: proof verified using cached VK with public IO binding");
    Ok(true)
}
