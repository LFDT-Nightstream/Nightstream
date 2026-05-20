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

use crate::builder::build_steps;
use crate::ccs::WasmVmSpec;
use crate::ir::WasmStepTrace;
use crate::preprocess::canonical_wasm_f_prime_shape;

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

/// Build per-step R1CS assignments from `trace` and fold them through
/// `neo-fold-clean`'s r1cs_f_prime chain. `prep` must describe the canonical
/// wasm CCS shape; mismatches are rejected up front.
pub fn prove(prep: &R1csFPrimePreprocessing, trace: &[WasmStepTrace]) -> Result<WasmProof, WasmProveError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;

    let prepared = build_steps(trace);
    let mut chain =
        R1csChainBuilder::new(prep).map_err(|err| WasmProveError::Bridge(format!("R1csChainBuilder::new: {err}")))?;
    for step in &prepared {
        chain
            .append_assignment(step.assignment.clone())
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
fn validate_wasm_preprocessing(prep: &R1csFPrimePreprocessing, vm: &WasmVmSpec) -> Result<(), WasmProveError> {
    let canonical = canonical_wasm_f_prime_shape(vm)
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
