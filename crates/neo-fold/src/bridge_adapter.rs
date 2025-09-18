//! Bridge Adapter (v2): convert modern ME types to the legacy bridge format
//! and drive Spartan2 compression under the same Poseidon2 family.
//!
//! This lives in `neo-fold` so it can use the *same* transcript parameters
//! to derive a binding header digest (no ad‑hoc hashing).

#![allow(deprecated)] // we intentionally talk to legacy bridge types

use bincode;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{MeInstance, MeWitness};
use neo_math::{F, K, KExtensions};
use p3_field::PrimeField64;

use crate::transcript::FoldTranscript;

/// Split K=F_{q^2} into base-field limbs (re, im).
#[inline]
fn k_to_coeffs(x: &K) -> [F; 2] {
    // neo-math exposes a stable split for K; if your API differs, adjust here.
    // The convention is lexicographic (real, imag).
    x.as_coeffs()
}

/// Convert modern ME instance into the legacy bridge instance.
pub fn modern_to_legacy_instance(
    modern: &MeInstance<Cmt, F, K>,
    params: &neo_params::NeoParams,
) -> neo_ccs::MEInstance {
    // Commitment coordinates (F^d×κ flattened)
    let c_coords: Vec<F> = modern.c.data.clone();

    // y outputs: each K limb → [F;2]
    let mut y_outputs = Vec::new();
    for yj in &modern.y {
        for &yk in yj {
            let [re, im] = k_to_coeffs(&yk);
            y_outputs.push(re);
            y_outputs.push(im);
        }
    }

    // r point in K^ell → base limb array
    let mut r_point = Vec::new();
    for r in &modern.r {
        let [re, im] = k_to_coeffs(r);
        r_point.push(re);
        r_point.push(im);
    }

    // Header digest: bind to the SAME Poseidon2 family via FoldTranscript
    let mut tr = FoldTranscript::new(b"neo/bridge/v2");
    tr.absorb_u64(&[
        params.q,               // modulus identifier
        params.lambda as u64,   // target bits
        params.s as u64,        // extension degree (v1: 2)
        params.b as u64,        // base
        params.B as u64,        // big base
    ]);
    tr.absorb_f(&c_coords);
    tr.absorb_u64(&[modern.m_in as u64]);
    
    // CRITICAL SECURITY: Absorb X matrix for additional binding
    // While X = L_x(Z) is deterministically derivable from Z (via commitment c),
    // absorbing X provides defense-in-depth against potential edge cases
    tr.absorb_f(modern.X.as_slice());
    
    // CRITICAL SECURITY: Absorb actual values, not just shapes
    tr.absorb_u64(&[modern.y.len() as u64, modern.r.len() as u64]);
    
    // Bind to actual y outputs (evaluation results)
    for yj in &modern.y {
        for &yk in yj {
            let [re, im] = k_to_coeffs(&yk);
            tr.absorb_f(&[re, im]);
        }
    }
    
    // CRITICAL FIX: Bind to actual r values (evaluation point)
    for &ri in &modern.r {
        let [re, im] = k_to_coeffs(&ri);
        tr.absorb_f(&[re, im]);
    }
    
    let header_digest = tr.state_digest();

    neo_ccs::MEInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, // TODO: Pattern A - populate with actual pre-commit coords
        c_coords,
        y_outputs,
        r_point,
        base_b: params.b as u64,
        header_digest,
    }
}

/// Convert field element to balanced integer representation.
/// Maps F element to [-⌊(q-1)/2⌋, ⌊(q-1)/2⌋] to preserve sign for balanced digits.
#[inline]
fn f_to_balanced_i64(a: F) -> i64 {
    let u = a.as_canonical_u64();
    let q = F::ORDER_U64;
    let half = (q - 1) / 2;
    if u <= half { 
        u as i64 
    } else { 
        (u as i128 - q as i128) as i64 
    }
}

/// Convert modern witness to legacy witness (digits in base b).
pub fn modern_to_legacy_witness(
    modern: &MeWitness<F>,
    params: &neo_params::NeoParams,
) -> Result<neo_ccs::MEWitness, String> {
    let d = modern.Z.rows();
    let m = modern.Z.cols();
    let mut z_digits = Vec::with_capacity(d * m);
    let b = params.b as i64;

    // CRITICAL FIX: Use column-major order to match Ajtai commitment system
    // Ajtai uses Z[col * d + row] indexing, so we must flatten in the same order
    for c in 0..m {
        for r in 0..d {
            let limb = f_to_balanced_i64(modern.Z[(r, c)]);
            // CRITICAL: Witness must already be in range from decomposition
            // Runtime check for soundness (not just debug assert)
            if limb.abs() >= b {
                return Err(format!(
                    "digit out of range at (row {}, col {}): {}, base b={}",
                    r, c, limb, b
                ));
            }
            z_digits.push(limb);
        }
    }

    // CRITICAL FIX: Pad z_digits to next power of two to match synthesize() allocation
    // This ensures InvalidWitnessLength doesn't occur in Spartan2 proving
    let original_len = z_digits.len();
    let target_len = if original_len <= 1 { 1 } else { original_len.next_power_of_two() };
    if original_len < target_len {
        z_digits.resize(target_len, 0i64); // Pad with zeros
        eprintln!("🔍 modern_to_legacy_witness(): z_digits padded from {} to {} (power-of-two)", 
                  original_len, target_len);
    }

    Ok(neo_ccs::MEWitness {
        z_digits,
        weight_vectors: Vec::new(), // optional, bridge works without them
        ajtai_rows: None,
    })
}

/// Produce a Spartan2 proof bundle and serialize it (bincode).
pub fn compress_via_bridge(
    modern_instance: &MeInstance<Cmt, F, K>,
    modern_witness: &MeWitness<F>,
    params: &neo_params::NeoParams,
) -> Result<Vec<u8>, String> {
    let legacy_inst = modern_to_legacy_instance(modern_instance, params);
    let legacy_wit  = modern_to_legacy_witness(modern_witness, params)?;

    let bundle = neo_spartan_bridge::compress_me_to_spartan(&legacy_inst, &legacy_wit)
        .map_err(|e| format!("bridge compression error: {e}"))?;

    bincode::serialize(&bundle).map_err(|e| format!("bundle serialize error: {e}"))
}

/// Verify a serialized Spartan2 bundle.
pub fn verify_via_bridge(serialized: &[u8]) -> Result<bool, String> {
    let bundle: neo_spartan_bridge::ProofBundle =
        bincode::deserialize(serialized).map_err(|e| format!("bundle decode error: {e}"))?;
    neo_spartan_bridge::verify_me_spartan(&bundle)
        .map_err(|e| format!("bridge verify error: {e}"))
}

/// Compatibility wrapper for the old verify_via_bridge signature.
/// 
/// The public_inputs parameter is ignored in v2 as the bridge binds to
/// the instance data via the header digest.
#[deprecated(note = "Use verify_via_bridge(serialized) directly - public_inputs are bound via header digest")]
pub fn verify_via_bridge_with_io(_public_inputs: &[F], serialized: &[u8]) -> Result<bool, String> {
    // Return error if public_inputs is non-empty (as suggested in the review)
    if !_public_inputs.is_empty() {
        return Err("public_inputs are now bound via header digest - pass empty slice".to_string());
    }
    verify_via_bridge(serialized)
}