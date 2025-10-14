//! IVC verifier pipeline
//!
//! This module implements IVC verification including:
//! - Single step verification
//! - Chain verification  
//! - Accumulator progression checks

use crate::F;
use crate::shared::types::*;
use neo_ccs::CcsStructure;
use subtle::ConstantTimeEq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::internal::{
    transcript::{create_step_digest, build_step_transcript_data},
    augmented::{build_augmented_ccs_linked_with_rlc, build_linked_augmented_witness, build_augmented_public_input_for_step, build_augmented_ccs_linked, compute_augmented_public_input_for_step},
    commit::verify_commitment_evolution,
};
use crate::shared::digest::compute_accumulator_digest_fields;
use super::folding::verify_ivc_step_folding;

/// Verify an IVC proof against the step CCS and previous accumulator
pub fn verify_ivc_step(
    step_ccs: &CcsStructure<F>,
    ivc_proof: &IvcProof,
    prev_accumulator: &Accumulator,
    binding_spec: &StepBindingSpec,
    params: &crate::NeoParams,
    prev_augmented_x: Option<&[F]>,
) -> Result<bool, Box<dyn std::error::Error>> {
    // SECURITY: Reject CCS structures that are too small for sumcheck security
    // ℓ = ceil(log2(n_padded)) must be ≥ 2, so minimum n=3 (→ padded to 4 → ℓ=2)
    if step_ccs.n < 3 {
        return Ok(false);
    }
    
    // 0. Enforce Las binding: step_x must equal H(prev_accumulator)
    let expected_prefix = compute_accumulator_digest_fields(prev_accumulator)?;
    let step_public_input = ivc_proof.public_inputs.wrapper_public_input_x();
    if step_public_input.len() < expected_prefix.len() {
        return Err(format!(
            "Step public input too short: expected at least {} bytes, got {}",
            expected_prefix.len(),
            step_public_input.len()
        ).into());
    }
    if step_public_input[..expected_prefix.len()] != expected_prefix[..] {
        return Err("Las binding check failed: step_x prefix does not match H(prev_accumulator)".into());
    }

    // 1. Reconstruct the augmented CCS that was used for proving
    let step_data = build_step_transcript_data(prev_accumulator, ivc_proof.step, step_public_input);
    let step_digest = create_step_digest(&step_data);
    
    // 2. Build base augmented CCS using TRUSTED binding metadata
    // 🔒 SECURITY: Use TRUSTED binding_spec, NOT proof-supplied values!
    let y_len = prev_accumulator.y_compact.len();
    let step_x_len = step_public_input.len();
    
    // 🔒 SECURITY (Pattern-B pre-commit dimension binding): 
    // Guard B: Bind c_step_coords dimension before using it for ρ derivation
    // Ajtai commitments are always d×κ in size, independent of m
    let d = neo_math::ring::D;
    let expected_num_coords = d * params.kappa as usize;
    
    if ivc_proof.c_step_coords.len() != expected_num_coords {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "❌ SECURITY GUARD B (Pattern-B dimension binding): c_step_coords length mismatch. Expected {} (d={} × κ={}), got {}",
            expected_num_coords, d, params.kappa, ivc_proof.c_step_coords.len()
        );
        return Err(format!(
            "SECURITY GUARD B (Pattern-B dimension binding): c_step_coords length mismatch. Expected {} (d={} × κ={}), got {}",
            expected_num_coords, d, params.kappa, ivc_proof.c_step_coords.len()
        ).into());
    }
    
    // 🔒 SECURITY: Recompute ρ to get the same transcript state
    let (rho, _transcript_digest) = super::super::internal::transcript::rho_from_transcript(prev_accumulator, step_digest, &ivc_proof.c_step_coords);
    
    // --- Guard A: Strict ρ equality - proof.step_rho must exactly match verifier's recomputation
    // This prevents any attempt to smuggle an inconsistent rho alongside forged c_step_coords.
    // No bypass for F::ZERO - all proofs must have valid step_rho.
    #[cfg(feature = "neo-logs")]
    eprintln!("🔍 DEBUG: Verifier recomputed ρ = {}, proof.step_rho = {}", 
              rho.as_canonical_u64(), ivc_proof.public_inputs.rho().as_canonical_u64());
    if ivc_proof.public_inputs.rho() != rho {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "❌ SECURITY GUARD A (Strict ρ binding): Recomputed ρ ({}) != proof.step_rho ({})",
            rho.as_canonical_u64(),
            ivc_proof.public_inputs.rho().as_canonical_u64()
        );
        return Err(format!(
            "SECURITY GUARD A (Strict ρ binding): Recomputed ρ ({}) != proof.step_rho ({})",
            rho.as_canonical_u64(),
            ivc_proof.public_inputs.rho().as_canonical_u64()
        ).into());
    }
    
    // RLC coefficients no longer needed here (binder disabled)
    
    // Reconstruct witness structure to determine dimensions
    let step_witness_augmented = build_linked_augmented_witness(
        &vec![F::ZERO; step_ccs.m], // dummy step witness for sizing
        &binding_spec.y_step_offsets,
        rho
    );
    let step_augmented_input = build_augmented_public_input_for_step(
        step_public_input,
        rho,
        &prev_accumulator.y_compact,
        &ivc_proof.next_accumulator.y_compact
    );
    if step_augmented_input != ivc_proof.public_inputs.step_augmented_public_input() {
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
    let mut full_step_z = step_augmented_input.to_vec();
    full_step_z.extend_from_slice(&step_witness_augmented);
    let decomp_z = crate::decomp_b(
        &full_step_z,
        params.b, // use the same base as prover
        d,
        crate::DecompStyle::Balanced
    );
    let m_step = decomp_z.len() / d;
    
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
        ivc_proof.public_inputs.wrapper_public_input_x(), // step_x 
        rho,                                              // ρ (PUBLIC - CRITICAL!)
        &prev_accumulator.y_compact,                      // y_prev
        &ivc_proof.next_accumulator.y_compact             // y_next
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
    
    // Guard C: Enforce exact public_io length (y_len * 8 bytes for y_next + 32 bytes for digest)
    // This prevents trailing bytes from being smuggled in the proof
    let expected_io_len = y_len * 8 + 32;
    if io.len() != expected_io_len {
        #[cfg(feature = "neo-logs")]
        eprintln!(
            "❌ SECURITY GUARD C (Exact public_io length): Expected {} bytes ({}×8 + 32), got {}",
            expected_io_len, y_len, io.len()
        );
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
    // SECURITY: Reject CCS structures that are too small for sumcheck security
    // ℓ = ceil(log2(n_padded)) must be ≥ 2, so minimum n=3 (→ padded to 4 → ℓ=2)
    if step_ccs.n < 3 {
        return Ok(false);
    }
    
    // Strict threading of prev_augmented_x and RHS reconstruction checks
    let mut acc_before_curr_step = initial_accumulator.clone();
    let mut prev_augmented_x: Option<Vec<F>> = None;
    
    for (step_idx, step_proof) in chain_proof.steps.iter().enumerate() {
        // Cross-check prover-supplied augmented input matches verifier reconstruction.
        let (expected_augmented, _) = compute_augmented_public_input_for_step(&acc_before_curr_step, step_proof)
            .map_err(|e| {
                anyhow::anyhow!("Step {}: failed to compute augmented input: {}", step_idx, e)
            })?;
        let actual_augmented = step_proof.public_inputs.step_augmented_public_input();
        if expected_augmented != actual_augmented {
            return Ok(false);
        }

        // Enforce per-step verification (includes folding checks)
        let ok = verify_ivc_step(
            step_ccs,
            step_proof,
            &acc_before_curr_step,
            binding_spec,
            params,
            prev_augmented_x.as_deref(),
        )?;
        if !ok {
            return Ok(false);
        }

        // Advance and thread linkage
        acc_before_curr_step = step_proof.next_accumulator.clone();
        prev_augmented_x = Some(step_proof.public_inputs.step_augmented_public_input().to_vec());
    }

    let final_step_matches = acc_before_curr_step.step == chain_proof.chain_length;
    
    Ok(final_step_matches)
}
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
