//! Top-level wasm prove / verify entrypoints.
//!
//! Two paths:
//! - [`prove`] / [`verify`] — full proof: wasm relation + IVC fold through
//!   `neo-fold-clean`'s r1cs_f_prime chain. Returns a [`WasmProof`].
//! - [`prove_relation`] / [`verify_relation`] — relation only, no fold.
//!   Same inputs, no preprocessing required. Useful for fast development
//!   loops; the fold cost dominates the full path.

use neo_fold_clean::frontends::r1cs_f_prime::{R1csChainBuilder, R1csFPrimePreprocessing};
use neo_fold_clean::lifecycle::verify_uncompressed as clean_verify_uncompressed;
use neo_fold_clean::paper::digest::structure_digest;
use neo_fold_clean::Uncompressed;

use crate::builder::WasmTraceBuilder;
use crate::ccs::WasmVmSpec;
use crate::ir::WasmStepTrace;
use crate::relation::{prove_wasm_relation, verify_wasm_relation, WasmRelationProof};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmPublicInput {
    pub transcript_seed: Vec<u8>,
    /// Initial values of all locals at function entry, indexed by local index.
    /// Params carry the call argument values; pure locals are zero (or absent).
    pub initial_locals: Vec<u32>,
}

pub struct WasmProverInput<'a> {
    pub public: WasmPublicInput,
    pub trace: &'a [WasmStepTrace],
    /// Next-PC ROM: sorted `(pc_before, control_choice, pc_after)` entries
    /// derived from the WASM binary. Committed to the transcript before
    /// Stage 1 challenges.
    pub pc_rom: Vec<(u64, u64, u64)>,
    /// PC-edge-kind ROM: sorted (pc_before, edge_kind) pairs derived from
    /// the WASM binary.
    pub pc_edge_kinds: Vec<(u64, u64)>,
    /// Function-entry ROM: sorted (function_ref, entry_pc) pairs derived
    /// from the WASM binary.
    pub function_entries: Vec<(u64, u64)>,
}

pub struct WasmVerifierInput<'a> {
    pub public: WasmPublicInput,
    pub trace: &'a [WasmStepTrace],
    pub pc_rom: Vec<(u64, u64, u64)>,
    pub pc_edge_kinds: Vec<(u64, u64)>,
    pub function_entries: Vec<(u64, u64)>,
}

pub struct WasmProof {
    pub relation: WasmRelationProof,
    pub main_run: Uncompressed,
}

#[derive(Debug)]
pub enum WasmProveError {
    InvalidWitness(String),
    Relation(String),
    Bridge(String),
}

impl core::fmt::Display for WasmProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidWitness(msg) => write!(f, "invalid witness: {msg}"),
            Self::Relation(msg) => write!(f, "relation failed: {msg}"),
            Self::Bridge(msg) => write!(f, "bridge failed: {msg}"),
        }
    }
}

impl std::error::Error for WasmProveError {}

/// Prove the wasm relation only (no IVC fold). Fast — useful for testing
/// constraint correctness without paying for the chain.
pub fn prove_relation(input: &WasmProverInput<'_>) -> Result<WasmRelationProof, WasmProveError> {
    prove_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
    )
    .map_err(WasmProveError::Relation)
}

/// Verify a relation-only proof against the trace.
pub fn verify_relation(input: &WasmVerifierInput<'_>, relation: &WasmRelationProof) -> Result<(), WasmProveError> {
    verify_wasm_relation(
        input.trace,
        &input.public.initial_locals,
        &input.pc_rom,
        &input.pc_edge_kinds,
        &input.function_entries,
        &input.public.transcript_seed,
        relation,
    )
    .map_err(WasmProveError::Relation)
}

/// Prove the full wasm execution: relation proof + IVC fold of the
/// per-step R1CS assignments through `neo-fold-clean`'s r1cs_f_prime chain.
///
/// `prep` must describe the canonical wasm CCS shape; mismatches are
/// rejected up front via [`validate_wasm_preprocessing`].
pub fn prove(prep: &R1csFPrimePreprocessing, input: &WasmProverInput<'_>) -> Result<WasmProof, WasmProveError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;

    let relation = prove_relation(input)?;
    let prepared = build_prepared(input.trace)?;
    if relation.boundary_rows.len() != prepared.len() {
        return Err(WasmProveError::Bridge(format!(
            "wasm relation exported {} boundary rows for {} prepared steps",
            relation.boundary_rows.len(),
            prepared.len()
        )));
    }

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

    Ok(WasmProof { relation, main_run })
}

/// Verify a full wasm proof.
///
/// Checks the relation against the trace and the IVC chain against
/// `prep`. **Does not** bind the chain to the specific trace — that
/// binding is the shout/twist layer's job (program ROM is a public
/// input, the shout proof indexes into it via the (pc, opcode) columns).
/// Until that lands, callers must treat this as "the chain is *some*
/// valid wasm execution under this preprocessing" rather than "the
/// chain is the specific wasm execution for this trace".
pub fn verify(
    prep: &R1csFPrimePreprocessing,
    input: &WasmVerifierInput<'_>,
    proof: &WasmProof,
) -> Result<(), WasmProveError> {
    let vm = WasmVmSpec::default();
    validate_wasm_preprocessing(prep, &vm)?;
    verify_relation(input, &proof.relation)?;
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
