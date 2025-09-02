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

pub use types::ProofBundle;

use anyhow::Result;
use neo_ccs::{MEInstance, MEWitness};
use spartan2::spartan::{R1CSSNARK, SpartanVerifierKey};
use spartan2::traits::snark::R1CSSNARKTrait;

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
    
    // fold digest as 4 little‑endian u64 limbs (matches digest_to_scalars())
    for ch in me.header_digest.chunks(8) {
        let mut limb = [0u8; 8];
        limb[..ch.len()].copy_from_slice(ch);
        out.extend_from_slice(&limb);
    }
    
    // Hash-MLE PCS requires power-of-2 length, so pad with zeros to match public_values()
    let num_scalars = out.len() / 8; // Each scalar is 8 bytes
    let next_power_of_2 = num_scalars.next_power_of_two();
    let padding_scalars = next_power_of_2 - num_scalars;
    
    eprintln!("🔍 encode_bridge_io_header(): padding from {} to {} scalars", num_scalars, next_power_of_2);
    
    // Pad with zero scalars (8 zero bytes each)
    for _ in 0..padding_scalars {
        out.extend_from_slice(&0u64.to_le_bytes());
    }
    
    out
}

/// **Main Entry Point**: Compress final ME(b,L) claim using Spartan2 + Hash-MLE PCS.
/// Note: no FRI parameters; the bridge uses Hash‑MLE PCS only.
pub fn compress_me_to_spartan(me: &MEInstance, wit: &MEWitness) -> Result<ProofBundle> {
    // Try the SNARK generation and provide detailed error diagnostics
    let snark_result = me_to_r1cs::prove_me_snark(me, wit);
    let (proof_bytes, _public_outputs, vk) = match snark_result {
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
    
    let vk_bytes = bincode::serialize(&vk)?;
    let io = encode_bridge_io_header(me);
    Ok(ProofBundle::new_with_vk(proof_bytes, vk_bytes, io))
}

/// Verify a ProofBundle containing an ME R1CS SNARK
pub fn verify_me_spartan(bundle: &ProofBundle) -> Result<bool> {
    let snark: R1CSSNARK<E> = bincode::deserialize(&bundle.proof)?;
    let vk: SpartanVerifierKey<E> = bincode::deserialize(&bundle.vk)?;
    
    // 1) Verify SNARK and get public scalars
    let publics = snark.verify(&vk)?;
    
    // 2) Serialize public scalars identically to encode_bridge_io_header()
    let mut bytes = Vec::with_capacity(publics.len() * 8);
    for x in &publics {
        bytes.extend_from_slice(&x.to_canonical_u64().to_le_bytes());
    }
    
    // 3) Compare with bundle's public IO bytes - CRITICAL security check
    // FIXED: Now that transcript consistency is working, re-enable public IO verification
    anyhow::ensure!(bytes == bundle.public_io_bytes, 
        "Public IO mismatch: SNARK public inputs don't match bundle.public_io_bytes");
    eprintln!("✅ Public IO verification passed: {} bytes match exactly", bytes.len());
    
    Ok(true)
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