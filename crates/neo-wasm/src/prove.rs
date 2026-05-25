//! Top-level wasm prove / verify entrypoints.
//!
//! [`prove`] builds the per-step R1CS assignments and folds them through
//! `neo-fold-clean`'s r1cs_f_prime chain; [`verify`] verifies the chain
//! against the same preprocessing.
//!
//! The proof object is the IVC artifact only. The fold binds per-row R1CS
//! constraints (the wasm CCS), but does *not* bind the ROM/memory/lookup
//! semantics — those rely on debug checkers ([`sanity_check_lookup_row`],
//! [`sanity_check_memory_rows`]) today and will move under a lookup
//! argument when the shout/twist layer lands.
//!
//! [`sanity_check_lookup_row`]: crate::sanity_check_lookup_row
//! [`sanity_check_memory_rows`]: crate::sanity_check_memory_rows

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::lifecycle::verify_uncompressed as clean_verify_uncompressed;
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::Uncompressed;

use crate::batch;
use crate::ccs::WasmVmSpec;
use crate::ir::WasmStepTrace;
use crate::layout::WITNESS_WIDTH;
use crate::preprocess::canonical_wasm_f_prime_shape_batched;

pub struct WasmProof {
    pub main_run: Uncompressed,
}

#[derive(Debug)]
pub enum WasmProveError {
    Bridge(String),
}

impl core::fmt::Display for WasmProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bridge(msg) => write!(f, "bridge failed: {msg}"),
        }
    }
}

impl std::error::Error for WasmProveError {}

/// Single-step prove. Thin wrapper over [`prove_batched`] at `batch_size = 1`.
pub fn prove(prep: &R1csFPrimePreprocessing, trace: &[WasmStepTrace]) -> Result<WasmProof, WasmProveError> {
    prove_batched(prep, trace, 1)
}

/// Fold `trace` through `prep` in batches of `batch_size` wasm steps per
/// F'-shell fold.
///
/// `prep` must have been built with the same `batch_size` (see
/// [`crate::preprocess::preprocess_seeded_batched`]) — its column count
/// encodes the batch size and a mismatch surfaces as an assignment-length
/// error from the F' chain builder.
///
/// Short final batches are padded with synthetic state-preserving
/// padding rows (see [`crate::batch::padding_step_after`]); any trace
/// length divides cleanly into the same fixed-shape circuit.
///
/// The prover trusts that `prep` is canonical; it does not re-validate
/// (that's a verifier-side responsibility, currently elided).
pub fn prove_batched(
    prep: &R1csFPrimePreprocessing,
    trace: &[WasmStepTrace],
    batch_size: usize,
) -> Result<WasmProof, WasmProveError> {
    assert!(batch_size >= 1, "batch_size must be at least 1");
    assert!(!trace.is_empty(), "cannot prove an empty trace");

    let n_batches = batch::batch_count(trace.len(), batch_size);
    let mut chain =
        R1csChainBuilder::new(prep).map_err(|err| WasmProveError::Bridge(format!("R1csChainBuilder::new: {err}")))?;
    for batch_idx in 0..n_batches {
        let assignment = batch::build_batched_witness(trace, batch_size, batch_idx);
        chain
            .append_assignment(assignment)
            .map_err(|err| WasmProveError::Bridge(format!("append_assignment: {err}")))?;
    }
    let main_run = chain
        .finish()
        .map_err(|err| WasmProveError::Bridge(format!("finish: {err}")))?;

    Ok(WasmProof { main_run })
}

/// Verify the IVC chain against `prep`.
///
/// **Does not** bind the chain to a specific trace — that binding is the
/// shout/twist layer's job (program ROM is a public input, the shout proof
/// indexes into it via the (pc, opcode) columns). Until that lands, callers
/// must treat this as "the chain is *some* valid wasm execution under this
/// preprocessing" rather than "the chain is the specific wasm execution for
/// some intended trace".
///
/// **Also does not enforce inter-batch state continuity.** Cross-step
/// links pin `pc / sp / memory_pages / locals_fbp / param_init` equality
/// *within* each batch (block `i`'s `_after` equals block `i+1`'s
/// `_before`), but the boundary between batch `i`'s last block and
/// batch `i+1`'s first block is not linked. A prover could submit a
/// chain where batch `i+1` starts in arbitrary state, unrelated to
/// batch `i`'s end. Until cross-batch linking lands, treat a
/// multi-batch verified chain as "each batch is internally consistent"
/// rather than "the batches compose into one continuous execution".
pub fn verify(prep: &R1csFPrimePreprocessing, proof: &WasmProof) -> Result<(), WasmProveError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;
    clean_verify_uncompressed(&prep.prep, &proof.main_run)
        .map_err(|err| WasmProveError::Bridge(format!("verify_uncompressed: {err}")))?;
    Ok(())
}

/// Reject a `prep` whose full F' structure does not match the canonical
/// wasm VM. The width map is load-bearing: Boolean/Byte/U32 annotations are
/// enforced by the R1CS-F' bit frame, not by redundant rows in the wasm CCS.
///
/// Batch-size aware: infers the prep's `batch_size` from its plan width-vector
/// length (= `batch_size * single_step_witness_width`) and compares
/// against `canonical_wasm_f_prime_shape_batched` at that size.
fn validate_wasm_preprocessing(prep: &R1csFPrimePreprocessing, vm: &WasmVmSpec) -> Result<(), WasmProveError> {
    let prep_widths = &prep.plan().app_private_var_widths;
    if prep_widths.len() % WITNESS_WIDTH != 0 || prep_widths.is_empty() {
        return Err(WasmProveError::Bridge(format!(
            "preprocessing width-vector length {} is not a positive multiple of \
             the single-step wasm witness width {WITNESS_WIDTH}",
            prep_widths.len()
        )));
    }
    let batch_size = prep_widths.len() / WITNESS_WIDTH;
    let canonical = canonical_wasm_f_prime_shape_batched(vm, batch_size)
        .map_err(|err| WasmProveError::Bridge(format!("canonical wasm F' shape: {err}")))?;
    if prep.plan().app_private_var_widths != canonical.plan.app_private_var_widths {
        return Err(WasmProveError::Bridge(
            "preprocessing widths do not match the canonical wasm column widths".into(),
        ));
    }
    let expected = structure_digest(&canonical.structure.ccs);
    if *prep.prep.structure_digest() != expected {
        return Err(WasmProveError::Bridge(
            "preprocessing F' structure does not match the canonical wasm VM".into(),
        ));
    }
    if structure_digest(&prep.structure().ccs) != expected {
        return Err(WasmProveError::Bridge(
            "cached compiler F' structure does not match the canonical wasm VM".into(),
        ));
    }
    if prep.r1cs().m_in() != canonical.sparse_r1cs.m_in {
        return Err(WasmProveError::Bridge(format!(
            "preprocessing m_in {} does not match wasm m_in {}",
            prep.r1cs().m_in(),
            canonical.sparse_r1cs.m_in
        )));
    }
    Ok(())
}
