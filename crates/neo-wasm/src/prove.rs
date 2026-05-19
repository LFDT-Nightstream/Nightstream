//! Top-level wasm prove / verify entrypoints.
//!
//! [`prove`] builds the per-step R1CS assignments and folds them through
//! `neo-fold-clean`'s r1cs_f_prime chain; [`verify`] verifies the chain
//! against the same preprocessing.
//!
//! The proof object is the IVC artifact only. Witness/trace soundness
//! is enforced by the CCS constraints under the fold, and exercised in
//! tests through [`sanity_check_lookup_row`] and [`sanity_check_memory_rows`].
//!
//! [`sanity_check_lookup_row`]: crate::sanity_check_lookup_row
//! [`sanity_check_memory_rows`]: crate::sanity_check_memory_rows

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::lifecycle::verify_uncompressed as clean_verify_uncompressed;
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::Uncompressed;

use crate::builder::WasmTraceBuilder;
use crate::ccs::WasmVmSpec;
use crate::ir::WasmStepTrace;

pub struct WasmProof {
    pub main_run: Uncompressed,
}

#[derive(Debug)]
pub enum WasmProveError {
    InvalidWitness(String),
    Bridge(String),
}

impl core::fmt::Display for WasmProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidWitness(msg) => write!(f, "invalid witness: {msg}"),
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

    let prepared = build_prepared(trace)?;
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

/// Reject a `prep` whose underlying R1CS shape or public-input split does
/// not match the canonical wasm VM. Compared digests are over the *app*
/// R1CS-to-CCS embedding, not the F'-augmented `prep.prep.structure_digest()`
/// — the latter wraps the wasm R1CS in F' bit-decomposition + recursive-plan
/// rows, so it never equals the bare wasm CCS digest.
fn validate_wasm_preprocessing(prep: &R1csFPrimePreprocessing, vm: &WasmVmSpec) -> Result<(), WasmProveError> {
    let core = vm.core_ccs_spec();
    let expected = structure_digest(&core.structure);
    let actual = structure_digest(&prep.r1cs.to_structure());
    if actual != expected {
        return Err(WasmProveError::Bridge(
            "preprocessing R1CS shape does not match the canonical wasm VM".into(),
        ));
    }
    if prep.r1cs.m_in() != core.m_in {
        return Err(WasmProveError::Bridge(format!(
            "preprocessing m_in {} does not match wasm m_in {}",
            prep.r1cs.m_in(),
            core.m_in
        )));
    }
    Ok(())
}

fn build_prepared(trace: &[WasmStepTrace]) -> Result<Vec<crate::step_build::WasmStepBuild>, WasmProveError> {
    let vm = WasmVmSpec::default();
    let builder = WasmTraceBuilder::new();
    builder
        .build_steps(&vm, trace)
        .map_err(|err| WasmProveError::InvalidWitness(err.to_string()))
}
