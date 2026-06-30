#![recursion_limit = "512"]

pub mod adapters;
pub mod batch;
pub mod ccs;
mod gadgets;
pub mod witness_builder;
pub use gadgets::push_zero_test_gadget;
pub mod ir;
pub mod isa;
mod ivc_state;
pub mod layout;
pub mod lookup_semantics;
pub mod memory_semantics;
pub mod preprocess;
pub mod prove;
mod r1cs_builder;
pub mod relation_layout;
pub mod step_build;
pub mod tables;
pub mod tagged_r1cs_builder;

pub use adapters::wasmtime::{
    build_pc_rom_from_binary, build_store_debug_function_id_map, collect_wasmtime_component_run,
    collect_wasmtime_component_run_with_linker, collect_wasmtime_steps, extract_first_component_core_program_artifacts,
    extract_wasm_program_artifacts, traces_from_wasmtime_component, traces_from_wasmtime_component_with_linker,
    traces_from_wasmtime_steps, traces_from_wasmtime_wasm_bytes, HasWasmTraceState, WasmProgramArtifacts,
    WasmProgramDecodeEntry, WasmProgramTables, WasmtimeTraceHandler, WasmtimeTraceMemoryAccess, WasmtimeTraceRun,
    WasmtimeTraceState, WasmtimeTraceStep,
};
pub use ccs::WasmVmSpec;
pub use ir::{
    boundary_states, LinearMemoryAccess, LinearMemoryWordLane, StackValueAccess, WasmAuxOpcode, WasmBoundaryState,
    WasmBuildError, WasmOutputState, WasmParamInitState, WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmStepTrace,
};
pub use isa::{
    opcode_code, opcode_info_from_code, WasmMemoryAccessInfo, WasmMemoryAccessKind, WasmMemoryExtension, WasmOpTable,
    WasmOpcode, WasmOpcodeClass, WasmOpcodeInfo,
};
pub use ivc_state::{WasmCrossStepColumnPair, WasmCrossStepLinkSpec};
pub use layout::{Column, ColumnWidth, WasmColumnSpec, COLUMN_SPECS};
pub use lookup_semantics::{sanity_check_lookup_row, LookupBuiltin, LookupExpr, LookupPredicate, LookupSemantics};
pub use memory_semantics::{preload_from_program_artifacts, sanity_check_memory_rows, WasmMemoryPreload};
pub use preprocess::{
    preprocess_seeded_batched, semantic_state_digest, top_level_initial_state, top_level_initial_state_digest,
};
pub use prove::{prove, prove_batched, verify, WasmProof, WasmProveError};
pub use relation_layout::{
    build_wasm_relation_layout, LinearMemoryColumns, SignExtensionColumns, WasmAuxiliaryRelations,
    WasmLookupBindingSpec, WasmLookupFamilyKind, WasmLookupFamilySpec, WasmMemoryActivation, WasmMemoryColumnKind,
    WasmMemoryColumnSpec, WasmMemorySpec, WasmRelationLayout,
};
pub use tables::WasmLookupArity;
pub use tagged_r1cs_builder::{WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag};
pub use witness_builder::{build_steps, build_witness_vector};
