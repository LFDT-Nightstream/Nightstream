//! Neo Folding Protocol - Single Three-Reduction Pipeline
//!
//! **BREAKING CHANGE v2**: `verify_folding_proof` now requires a `spartan_bundle` parameter
//! for mandatory succinct last-mile verification and anti-replay protection.
//!
//! Implements the complete folding protocol: Π_CCS → Π_RLC → Π_DEC  
//! Uses one transcript (Poseidon2), one backend (Ajtai), and one sum-check over K.

#![forbid(unsafe_code)]

pub mod error;
/// Poseidon2 transcript for Fiat-Shamir
pub mod transcript;
/// Strong sampling set infrastructure for challenges
pub mod strong_set;
/// Π_RLC verifier: Random Linear Combination verification
pub mod verify_linear;
/// Π_CCS: Sum-check reduction over extension field K  
pub mod pi_ccs;
/// Π_RLC: Random linear combination with S-action
pub mod pi_rlc;
/// Π_DEC: Verified split opening (TODO: implement real version)
pub mod pi_dec;
/// Spartan2 bridge adapter (modern → legacy types + digest binding)
pub mod bridge_adapter;

// Re-export main types
pub use error::{FoldingError, PiCcsError, PiRlcError, PiDecError};
pub use transcript::{FoldTranscript, Domain};
pub use strong_set::{StrongSamplingSet, VerificationError, ds};
pub use verify_linear::{verify_linear_rlc, verify_linear_rlc as verify_linear};
pub use pi_ccs::{pi_ccs_prove, pi_ccs_verify, PiCcsProof, eval_tie_constraints, eval_range_decomp_constraints};  
pub use pi_rlc::{pi_rlc_prove, pi_rlc_verify, PiRlcProof};
pub use pi_dec::{pi_dec, pi_dec_verify, PiDecProof};
#[allow(deprecated)]
pub use bridge_adapter::{compress_via_bridge, verify_via_bridge, verify_via_bridge_with_io, modern_to_legacy_instance, modern_to_legacy_witness};

use neo_ccs::{MeInstance, MeWitness, CcsStructure};
use neo_math::{F, K, Rq, cf_inv};
use neo_ajtai::{Commitment as Cmt, s_lincomb};
use p3_field::PrimeCharacteristicRing;
// GuardParams import removed - no longer needed after removing single-instance bypass

/// Proof that k+1 CCS instances fold to k instances
#[derive(Debug, Clone)]
pub struct FoldingProof {
    /// Π_CCS proof (sum-check over K) 
    pub pi_ccs_proof: PiCcsProof,
    /// The k+1 ME(b, L) instances produced by Π_CCS (public, transcript-bound)
    pub pi_ccs_outputs: Vec<MeInstance<Cmt, F, K>>,
    /// Π_RLC proof (S-action combination)
    pub pi_rlc_proof: PiRlcProof,
    /// Π_DEC proof (verified split opening)  
    pub pi_dec_proof: PiDecProof,
}

/// Fold k+1 CCS instances to k instances using the three-reduction pipeline  
/// Input: k+1 CCS instances and witnesses
/// Output: k ME instances and folding proof
pub fn fold_ccs_instances(
    params: &neo_params::NeoParams,
    structure: &CcsStructure<F>,
    instances: &[neo_ccs::McsInstance<Cmt, F>],
    witnesses: &[neo_ccs::McsWitness<F>],
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<MeWitness<F>>, FoldingProof), FoldingError> {
    if instances.is_empty() || instances.len() != witnesses.len() {
        return Err(FoldingError::InvalidInput("empty or mismatched inputs".into()));
    }

    // Ajtai S-module consistent with the actual Z shape (d,m)
    let d = witnesses[0].Z.rows();
    let m = witnesses[0].Z.cols();
    
    // Shape consistency check: ensure all witnesses have same Z dimensions
    for (i, w) in witnesses.iter().enumerate() {
        if w.Z.rows() != d || w.Z.cols() != m {
            return Err(FoldingError::InvalidInput(format!(
                "inconsistent witness Z shape at index {}: expected {}x{}, got {}x{}",
                i, d, m, w.Z.rows(), w.Z.cols()
            )));
        }
    }
    
    let l = neo_ajtai::AjtaiSModule::from_global_for_dims(d, m)
        .map_err(|e| FoldingError::InvalidInput(format!(
            "Ajtai PP not initialized for (d={}, m={}): {}", d, m, e
        )))?;

    // One transcript shared end-to-end
    let mut tr = FoldTranscript::default();

    // 1) Π_CCS: k+1 MCS → k+1 ME(b,L)
    let (me_list, pi_ccs_proof) =
        pi_ccs::pi_ccs_prove(&mut tr, params, structure, instances, witnesses, &l)?;

    // SECURITY FIX: Removed single-instance fast path that bypassed RLC/DEC
    // All instances must go through the complete pipeline for security

    // 2) Π_RLC: k+1 ME(b,L) → 1 ME(B,L) (only for multiple instances)
    let (me_b, pi_rlc_proof) = pi_rlc::pi_rlc_prove(&mut tr, params, &me_list)?;

    // 2b) Build the combined witness Z' = Σ rot(ρ_i)·Z_i for the DEC prover
    if d != neo_math::D {
        return Err(FoldingError::InvalidInput(format!(
            "Ajtai ring dimension D={} but witness Z has rows={}", neo_math::D, d
        )));
    }
    // Convert ρ to ring elements
    let rhos_ring: Vec<neo_math::Rq> = pi_rlc_proof.rho_elems.iter()
        .map(|coeffs| neo_math::ring::cf_inv(*coeffs))
        .collect();
    if rhos_ring.len() != witnesses.len() {
        return Err(FoldingError::PiRlc(crate::error::PiRlcError::InvalidInput(format!(
            "rho count {} != witness count {}", rhos_ring.len(), witnesses.len()
        ))));
    }
    // Accumulate Σ rot(ρ_i)·Z_i column-wise
    let mut z_prime = neo_ccs::Mat::zero(d, m, F::ZERO);
    for (wit, rho) in witnesses.iter().zip(rhos_ring.iter()) {
        let s_action = neo_math::SAction::from_ring(*rho);
        for c in 0..m {
            let mut col = [F::ZERO; neo_math::D];
            for r in 0..d { col[r] = wit.Z[(r, c)]; }
            let rotated = s_action.apply_vec(&col);
            for r in 0..d { z_prime[(r, c)] += rotated[r]; }
        }
    }
    let me_b_wit = MeWitness { Z: z_prime };

    // 3) Π_DEC: 1 ME(B,L) → k ME(b,L) with verified openings & range assertions
    let (me_out, digit_wits, pi_dec_proof) =
        pi_dec::pi_dec(&mut tr, params, &me_b, &me_b_wit, structure, &l)
            .map_err(|e| match e {
                pi_dec::PiDecError::InvalidInput(msg) => FoldingError::PiDec(crate::error::PiDecError::InvalidInput(msg)),
                pi_dec::PiDecError::DecompositionFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::CommitmentError(msg)),
                pi_dec::PiDecError::VerifiedOpeningFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::OpeningFailed(msg)),
                pi_dec::PiDecError::RangeCheckFailed(msg) => FoldingError::PiDec(crate::error::PiDecError::RangeViolation(msg)),
                pi_dec::PiDecError::SHomomorphismError(msg) => FoldingError::PiDec(crate::error::PiDecError::CommitmentError(msg)),
            })?;

    let proof = FoldingProof {
        pi_ccs_proof,
        pi_ccs_outputs: me_list.clone(),
        pi_rlc_proof,
        pi_dec_proof,
    };
    Ok((me_out, digit_wits, proof))
}

/// Verify a folding proof end-to-end over the single FS transcript.
///
/// Enhanced verification with full RLC checking:
/// 1) Π_CCS rounds & r-binding against proof.pi_ccs_outputs
/// 2) Π_RLC complete verification: ρ derivation, guard, and S-action on (c, X, y)
/// 3) Π_DEC: recomposition & range checks from parent to digits
/// 4) Spartan2 verification: MANDATORY verification of ties y_j = Z M_j^T χ_r and range ||Z||_∞ < b
///
/// This enables complete verification of all transformations across the pipeline.
pub fn verify_folding_proof(
    params: &neo_params::NeoParams,  
    structure: &CcsStructure<F>,
    input_instances: &[neo_ccs::McsInstance<Cmt, F>],
    output_instances: &[MeInstance<Cmt, F, K>],  // k digits from Π_DEC
    proof: &FoldingProof,
    // NEW: require the succinct proof bundle emitted by your bridge:
    spartan_bundle: &neo_spartan_bridge::ProofBundle,
) -> Result<bool, FoldingError> {
    if input_instances.is_empty() { 
        return Err(FoldingError::InvalidInput("no inputs".into())); 
    }
    if proof.pi_ccs_outputs.len() != input_instances.len() {
        return Ok(false); // must carry exactly k+1 ME(b,L) from Π_CCS
    }

    // One shared transcript
    let mut tr = FoldTranscript::default();

    // 1) Π_CCS rounds & r-binding FOR THE Π_CCS OUTPUTS
    let ok_ccs = pi_ccs::pi_ccs_verify(
        &mut tr, params, structure, input_instances, &proof.pi_ccs_outputs, &proof.pi_ccs_proof,
    )?;
    if !ok_ccs { return Ok(false); }

    // SECURITY FIX: Removed single-instance verifier bypass
    // All instances must go through complete verification pipeline

    // 2) Recombine DEC digits → parent ME(B, L)
    let me_parent = recombine_me_digits_to_parent(params, output_instances)
        .map_err(|e| FoldingError::InvalidInput(format!("recombine digits: {e}")))?;

    // 3) Π_RLC: verify ρ (transcript), guard, and S-action linear combination for c, X, y
    let ok_rlc = pi_rlc::pi_rlc_verify(
        &mut tr, params,
        &proof.pi_ccs_outputs,  // k+1 inputs (ME(b, L))
        &me_parent,             // combined output ME(B, L)
        &proof.pi_rlc_proof,
    )?;
    if !ok_rlc { return Ok(false); }

    // 4) Π_DEC: recomposition & range checks from parent → digits
    // SECURITY: Fail closed if AjtaiSModule isn't properly initialized
    let l_real = neo_ajtai::AjtaiSModule::from_global()
        .map_err(|_| FoldingError::InvalidInput(
            "AjtaiSModule unavailable; cannot verify DEC securely".to_string()
        ))?;
    
    let ok_dec = pi_dec::pi_dec_verify(&mut tr, params, &me_parent, output_instances, &proof.pi_dec_proof, &l_real)
        .map_err(|e| FoldingError::InvalidInput(format!("pi_dec_verify failed: {e}")))?;
    if !ok_dec { return Ok(false); }

    // 5) Succinct last-mile (MANDATORY):
    //    This enforces y_j = Z M_j^T χ_r and range ||Z||_∞ < b for the terminal ME(b,L).
    
    // CRITICAL SECURITY: Bind Spartan bundle's public IO to THIS call's (c,X,r,y) values
    // A malicious prover could replay a valid Spartan proof for different data without this check
    let legacy_me_parent = crate::bridge_adapter::modern_to_legacy_instance(&me_parent, params);
    let expected_public_io = neo_spartan_bridge::encode_bridge_io_header(&legacy_me_parent);

    // If the bundle's bound public IO doesn't match this call's (c, X, r, y), treat as verify-fail.
    // Check both length and content to prevent edge cases
    if spartan_bundle.public_io_bytes.len() != expected_public_io.len()
        || spartan_bundle.public_io_bytes != expected_public_io
    {
        return Ok(false);
    }

    // Verify the Spartan proof. Any verification error is a verify-fail (not an input error).
    match neo_spartan_bridge::verify_me_spartan(spartan_bundle) {
        Ok(true) => Ok(true),
        Ok(false) => Ok(false),
        Err(_e) => Ok(false), // Internal errors also treated as verification failure
    }
}

/// Recombine k digit ME(b, L) instances into their parent ME(B, L) using base-b scalars.
/// Only relies on public instance data; *does not* use any witness.
#[allow(non_snake_case)] // Allow mathematical notation like X_parent  
fn recombine_me_digits_to_parent(
    params: &neo_params::NeoParams,
    digits: &[MeInstance<Cmt, F, K>],
) -> Result<MeInstance<Cmt, F, K>, &'static str> {
    if digits.is_empty() { return Err("no digits"); }

    let m_in = digits[0].m_in;
    let r_ref = &digits[0].r;
    let t = digits[0].y.len();
    let d_rows = digits[0].X.rows();
    let x_cols = digits[0].X.cols();

    // Sanity: all digits must share the same shapes and r
    for d in digits {
        if d.m_in != m_in { return Err("m_in mismatch"); }
        if &d.r != r_ref { return Err("r mismatch"); }
        if d.X.rows() != d_rows || d.X.cols() != x_cols { return Err("X shape mismatch"); }
        if d.y.len() != t { return Err("y arity mismatch"); }
    }

    // Commitments: c = Σ b^i · c_i via the scalar-as-ring S-action
    let mut coeffs: Vec<Rq> = Vec::with_capacity(digits.len());
    let mut pow_f = F::ONE;
    for _i in 0..digits.len() {
        let mut arr = [F::ZERO; neo_math::D];
        arr[0] = pow_f;                // pow_f * X^0
        coeffs.push(cf_inv(arr));      // promote F scalar to ring element
        pow_f *= F::from_u64(params.b as u64);
    }
    let digit_cs: Vec<Cmt> = digits.iter().map(|d| d.c.clone()).collect();
    let c_parent = s_lincomb(&coeffs, &digit_cs).map_err(|_| "s_lincomb failed")?;

    // X: component-wise Σ b^i * X_i  (scalar S-action is plain scaling)
    let mut X_parent = neo_ccs::Mat::zero(d_rows, x_cols, F::ZERO);
    let mut pow = F::ONE;
    for d in digits {
        for r in 0..d_rows {
            for c in 0..x_cols {
                X_parent[(r, c)] += d.X[(r, c)] * pow;
            }
        }
        pow *= F::from_u64(params.b as u64);
    }

    // y: per matrix j and coordinate t, Σ b^i * y_{i, j}[t]
    let y_dim = digits[0].y.get(0).map(|v| v.len()).unwrap_or(0);
    let mut y_parent = vec![vec![K::ZERO; y_dim]; t];
    let mut pow_k = K::from(F::ONE);
    let base_k = K::from(F::from_u64(params.b as u64));

    for i in 0..digits.len() {
        for j in 0..t {
            for u in 0..y_dim {
                y_parent[j][u] += digits[i].y[j][u] * pow_k;
            }
        }
        pow_k *= base_k;
    }

    // SECURITY FIX: Recompute y_scalars for parent ME instance
    // y_scalars[j] = Y_j(r) = ⟨(M_j z_parent), χ_r⟩ 
    // For recombined parent, this should be Σ b^i * Y_j,i(r) from digits
    let y_scalars_parent = if let Some(first_digit) = digits.first() {
        let t = first_digit.y_scalars.len();
        let mut parent_scalars = vec![K::ZERO; t];
        let mut pow_k = K::from(F::ONE);
        let base_k = K::from(F::from_u64(params.b as u64));
        
        for digit in digits {
            for j in 0..t {
                if j < digit.y_scalars.len() {
                    parent_scalars[j] += digit.y_scalars[j] * pow_k;
                }
            }
            pow_k *= base_k;
        }
        parent_scalars
    } else {
        vec![]
    };

    Ok(MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, // TODO: Pattern A - populate with actual pre-commit coords
        c: c_parent,
        X: X_parent,
        r: r_ref.clone(),
        y: y_parent,
        y_scalars: y_scalars_parent, // SECURITY: Correct recombined Y_j(r) scalars
        m_in,
        fold_digest: digits[0].fold_digest, // All digits should have same fold_digest
    })
}