//! Owns the WASM semantic kernel proof and IO surface types.

use crate::ir::WasmStepTrace;
use crate::relation::WasmRelationProof;
use crate::step_build::WasmStepBuild;
use neo_fold_clean::Uncompressed;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmKernelPublicInput {
    pub transcript_seed: Vec<u8>,
    /// Initial values of all locals at function entry, indexed by local index.
    /// Params carry the call argument values; pure locals are zero (or absent).
    pub initial_locals: Vec<u32>,
}

pub struct WasmKernelProverInput<'a> {
    pub public: WasmKernelPublicInput,
    pub trace: &'a [WasmStepTrace],
    /// Next-PC ROM: sorted `(pc_before, control_choice, pc_after)` entries derived from the WASM binary.
    /// Built via `build_pc_rom_from_binary`; committed to the transcript before Stage 1 challenges.
    pub pc_rom: Vec<(u64, u64, u64)>,
    /// PC-edge-kind ROM: sorted (pc_before, edge_kind) pairs derived from the WASM binary.
    pub pc_edge_kinds: Vec<(u64, u64)>,
    /// Function-entry ROM: sorted (function_ref, entry_pc) pairs derived from the WASM binary.
    pub function_entries: Vec<(u64, u64)>,
}

pub struct WasmKernelVerifierInput<'a> {
    pub public: WasmKernelPublicInput,
    pub trace: &'a [WasmStepTrace],
    /// Next-PC ROM: same ROM the prover used, derived from the same WASM binary.
    pub pc_rom: Vec<(u64, u64, u64)>,
    /// PC-edge-kind ROM: same ROM the prover used, derived from the same WASM binary.
    pub pc_edge_kinds: Vec<(u64, u64)>,
    /// Function-entry ROM: same ROM the prover used, derived from the same WASM binary.
    pub function_entries: Vec<(u64, u64)>,
}

pub struct WasmKernelOutput {
    pub prepared_steps: Vec<WasmStepBuild>,
}

pub struct WasmKernelProof {
    pub relation: WasmRelationProof,
}

pub struct WasmKernelRunProof {
    pub kernel: WasmKernelProof,
    pub main_run: Uncompressed,
}

#[derive(Debug)]
pub enum WasmKernelError {
    InvalidWitness(String),
    Relation(String),
    Bridge(String),
}

impl core::fmt::Display for WasmKernelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidWitness(msg) => write!(f, "invalid witness: {msg}"),
            Self::Relation(msg) => write!(f, "relation failed: {msg}"),
            Self::Bridge(msg) => write!(f, "bridge failed: {msg}"),
        }
    }
}

impl std::error::Error for WasmKernelError {}
