#![recursion_limit = "512"]

pub mod adapters;
mod application;
mod application_proof;
pub mod batch;
pub mod ccs;
pub mod comm_chain;
pub mod event_grammar;
mod gadgets;
pub mod witness_builder;
pub use gadgets::push_zero_test_gadget;
pub mod ir;
pub mod isa;
mod ivc_state;
pub mod layout;
mod lookup_circuit;
pub mod lookup_semantics;
pub mod memory_semantics;
pub mod nebula;
#[doc(hidden)]
pub mod preprocess;
mod prover;
mod r1cs_builder;
pub mod range_check;
pub mod relation_layout;
pub mod tables;
pub mod tagged_r1cs_builder;
mod witness_layout;

pub use adapters::wasmtime::{
    build_debug_function_id_map, build_pc_rom_from_binary, collect_wasmtime_component_run,
    collect_wasmtime_component_run_calls, collect_wasmtime_component_run_with_linker,
    collect_wasmtime_component_run_with_linker_and_args, collect_wasmtime_steps,
    extract_first_component_core_program_artifacts, extract_wasm_program_artifacts, traces_from_wasmtime_component,
    traces_from_wasmtime_component_with_linker, traces_from_wasmtime_steps, traces_from_wasmtime_steps_with_grammar,
    traces_from_wasmtime_wasm_bytes, WasmProgramArtifacts, WasmProgramDecodeEntry, WasmProgramTables, WasmTraceSink,
    WasmtimeTraceHandler, WasmtimeTraceMemoryAccess, WasmtimeTraceRun, WasmtimeTraceState, WasmtimeTraceStep,
};
pub use application::{WasmApplicationManifestError, WasmApplicationModule};
pub use application_proof::{
    WasmApplicationProof, WasmApplicationProofError, WasmApplicationProofStats, WasmApplicationProofSystem,
};
pub use ccs::WasmVmSpec;
pub use comm_chain::CommChainState;
pub use ir::{
    boundary_states, LinearMemoryAccess, LinearMemoryWordLane, StackValueAccess, WasmAuxOpcode, WasmBoundaryState,
    WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmGrammarRomEntry, WasmGrammarState, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
pub use isa::{
    opcode_code, opcode_info_from_code, WasmMemoryAccessInfo, WasmMemoryAccessKind, WasmMemoryExtension, WasmOpTable,
    WasmOpcode, WasmOpcodeClass, WasmOpcodeInfo,
};
pub use ivc_state::{WasmCrossStepColumnPair, WasmCrossStepLinkSpec};
pub use layout::{Column, ColumnWidth, WasmColumnSpec, COLUMN_SPECS};
#[doc(hidden)]
pub use lookup_circuit::{audit_compact_lookup_auxiliary_load_bearing, audit_compact_lookup_witness};
pub use lookup_semantics::{sanity_check_lookup_row, LookupBuiltin, LookupExpr, LookupPredicate, LookupSemantics};
pub use memory_semantics::{preload_from_program_artifacts, sanity_check_memory_rows, WasmMemoryPreload};
pub use nebula::{
    preprocess, prove, verify, WasmNebulaError, WasmNebulaLimits, WasmNebulaPreprocessing, WasmNebulaProfile,
    WasmNebulaProof,
};
pub use preprocess::{
    grammar_top_level_initial_state, grammar_top_level_initial_state_digest, preprocess_seeded_batched,
    semantic_state_digest, top_level_initial_state, top_level_initial_state_digest,
};
pub use prover::{WasmProver, WasmProverBackend};
pub use range_check::write_range_check_bits;
pub use relation_layout::{
    build_wasm_relation_layout, LinearMemoryColumns, SignExtensionColumns, WasmAuxiliaryRelations,
    WasmLookupBindingSpec, WasmLookupFamilyKind, WasmLookupFamilySpec, WasmMemoryActivation, WasmMemoryColumnKind,
    WasmMemoryColumnSpec, WasmMemorySpec, WasmRelationLayout,
};
pub use tables::WasmLookupArity;
pub use tagged_r1cs_builder::{WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag};
pub use witness_builder::build_witness_vector;
pub use witness_layout::RANGE_CHECKED_WITNESS_WIDTH;
