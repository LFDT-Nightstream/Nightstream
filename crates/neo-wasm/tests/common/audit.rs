//! Test-only replay harness for exercising the bare WASM relation.
//!
//! Production callers use `neo_wasm::{preprocess, prove, verify}`, whose
//! proof is checked by terminal induction. These helpers intentionally keep
//! full-history replay local to relation and batching regression tests.

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::lifecycle::verify_uncompressed_audit;
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::UncompressedAudit;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use neo_prover_metal::MetalNifsProver;
use neo_wasm::preprocess::{canonical_wasm_f_prime_shape_batched_with_initial_state_digest, semantic_state_digest};
use neo_wasm::RANGE_CHECKED_WITNESS_WIDTH;
use neo_wasm::{batch, WasmStepState, WasmVmStep};

pub struct AuditProof {
    run: UncompressedAudit,
}

#[derive(Debug)]
pub enum AuditProveError {
    Bridge(String),
    FinalStateMismatch,
    TranscriptMismatch,
}

impl core::fmt::Display for AuditProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bridge(msg) => write!(f, "bridge failed: {msg}"),
            Self::FinalStateMismatch => write!(
                f,
                "claimed final VM state does not match the replayed semantic-state digest"
            ),
            Self::TranscriptMismatch => write!(
                f,
                "the verified final commitment chain does not equal the claimed transcript's fold"
            ),
        }
    }
}

impl std::error::Error for AuditProveError {}

pub fn prove(prep: &R1csFPrimePreprocessing, trace: &[WasmVmStep]) -> Result<AuditProof, AuditProveError> {
    prove_batched(prep, trace, 1)
}

pub fn prove_batched(
    prep: &R1csFPrimePreprocessing,
    trace: &[WasmVmStep],
    batch_size: usize,
) -> Result<AuditProof, AuditProveError> {
    assert!(batch_size >= 1, "batch_size must be at least 1");
    assert!(!trace.is_empty(), "cannot prove an empty trace");

    let n_batches = batch::batch_count(trace.len(), batch_size);
    let mut chain =
        R1csChainBuilder::new(prep).map_err(|err| AuditProveError::Bridge(format!("R1csChainBuilder::new: {err}")))?;

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    let mut adapter = {
        let mut metal =
            MetalNifsProver::new().map_err(|err| AuditProveError::Bridge(format!("MetalNifsProver::new: {err}")))?;
        metal
            .prepare_static(
                &prep.prep.log,
                prep.prep.structure(),
                prep.prep.optimized_cache(),
                prep.prep.nebula().map(|config| &config.scheme),
            )
            .map_err(|err| AuditProveError::Bridge(format!("MetalNifsProver::prepare_static: {err}")))?;
        metal
    };

    for batch_idx in 0..n_batches {
        let assignment = batch::build_batched_witness(trace, batch_size, batch_idx);

        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        chain
            .append_assignment_with_nifs_adapter(assignment, &mut adapter)
            .map_err(|err| AuditProveError::Bridge(format!("append_assignment: {err}")))?;

        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        chain
            .append_assignment(assignment)
            .map_err(|err| AuditProveError::Bridge(format!("append_assignment: {err}")))?;
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    let run = chain
        .finish_with_audit_and_nifs_adapter(&mut adapter)
        .map_err(|err| AuditProveError::Bridge(format!("finish: {err}")))?;

    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    let run = chain
        .finish_with_audit()
        .map_err(|err| AuditProveError::Bridge(format!("finish: {err}")))?;
    Ok(AuditProof { run })
}

pub fn verify(
    prep: &R1csFPrimePreprocessing,
    proof: &AuditProof,
    claimed_final_state: WasmStepState,
) -> Result<(), AuditProveError> {
    validate_preprocessing(prep)?;

    verify_uncompressed_audit(&prep.prep, &proof.run)
        .map_err(|err| AuditProveError::Bridge(format!("verify_uncompressed_audit: {err}")))?;
    if proof.run.proof.state.semantic_state_digest != semantic_state_digest(claimed_final_state) {
        return Err(AuditProveError::FinalStateMismatch);
    }
    Ok(())
}

/// [`verify`] plus transcript binding for the standalone (no interleaving
/// proof) path: after the replayed chain verifies, the claimed final
/// state's `comm_chain` must equal the native fold of `transcript` — the
/// claimed event blocks in emission order.
pub fn verify_with_transcript(
    prep: &R1csFPrimePreprocessing,
    proof: &AuditProof,
    claimed_final_state: WasmStepState,
    initial_comm_chain: neo_wasm::CommChainState,
    transcript: &[[p3_goldilocks::Goldilocks; neo_wasm::comm_chain::COMM_CHAIN_BLOCK_WORDS]],
) -> Result<(), AuditProveError> {
    verify(prep, proof, claimed_final_state)?;
    let fold = neo_wasm::comm_chain::fold_event_blocks(initial_comm_chain, transcript);
    if claimed_final_state.comm_chain != fold.canonical_u64() {
        return Err(AuditProveError::TranscriptMismatch);
    }
    Ok(())
}

fn validate_preprocessing(prep: &R1csFPrimePreprocessing) -> Result<(), AuditProveError> {
    let prep_widths = &prep.plan().app_private_var_widths;
    let single_width = RANGE_CHECKED_WITNESS_WIDTH;
    if prep_widths.len() % single_width != 0 || prep_widths.is_empty() {
        return Err(AuditProveError::Bridge(format!(
            "preprocessing width-vector length {} is not a positive multiple of the single-step WASM witness width {single_width}",
            prep_widths.len()
        )));
    }
    let batch_size = prep_widths.len() / single_width;
    let initial_semantic_state_digest = prep
        .plan()
        .state_x_out
        .as_ref()
        .map(|_| prep.prep.initial_semantic_state_digest())
        .ok_or_else(|| AuditProveError::Bridge("WASM preprocessing is missing semantic-state carrying".into()))?;
    let canonical =
        canonical_wasm_f_prime_shape_batched_with_initial_state_digest(batch_size, initial_semantic_state_digest)
            .map_err(|err| AuditProveError::Bridge(format!("canonical WASM F' shape: {err}")))?;
    if prep.plan().app_private_var_widths != canonical.plan.app_private_var_widths {
        return Err(AuditProveError::Bridge(
            "preprocessing widths do not match the canonical WASM column widths".into(),
        ));
    }
    let expected = structure_digest(&canonical.structure.ccs);
    if *prep.prep.structure_digest() != expected {
        return Err(AuditProveError::Bridge(
            "preprocessing F' structure does not match the canonical WASM VM".into(),
        ));
    }
    if structure_digest(&prep.structure().ccs) != expected {
        return Err(AuditProveError::Bridge(
            "cached compiler F' structure does not match the canonical WASM VM".into(),
        ));
    }
    if prep.r1cs().m_in() != canonical.sparse_r1cs.m_in {
        return Err(AuditProveError::Bridge(format!(
            "preprocessing m_in {} does not match WASM m_in {}",
            prep.r1cs().m_in(),
            canonical.sparse_r1cs.m_in
        )));
    }
    Ok(())
}
