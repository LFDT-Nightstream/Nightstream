#![recursion_limit = "512"]

pub mod adapters;
pub mod batch;
pub mod ccs;
mod gadgets;
pub mod witness_builder;
pub use gadgets::push_zero_test_gadget;
pub mod ir;
pub mod isa;
pub mod layout;
pub mod lookup_binding_builder;
pub mod lookup_semantics;
pub mod lower;
pub mod memory_semantics;
pub mod preprocess;
pub mod prove;
mod r1cs_builder;
pub mod step_build;
pub mod tables;
pub mod tagged_r1cs_builder;

pub use adapters::rwasm::{traces_from_rwasm_instr_states, traces_from_rwasm_tracer};
pub use adapters::wasmtime::{
    build_pc_rom_from_binary, collect_wasmtime_component_run, collect_wasmtime_component_run_with_linker,
    collect_wasmtime_steps, extract_first_component_core_program_artifacts, extract_wasm_program_artifacts,
    traces_from_wasmtime_component, traces_from_wasmtime_component_with_linker, traces_from_wasmtime_steps,
    traces_from_wasmtime_wasm_bytes, WasmProgramArtifacts, WasmProgramDecodeEntry, WasmProgramTables,
    WasmtimeTraceMemoryAccess, WasmtimeTraceRun, WasmtimeTraceState, WasmtimeTraceStep,
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
pub use layout::{ColumnWidth, WasmColumnSpec, COLUMN_SPECS};
pub use lookup_binding_builder::{
    build_wasm_lookup_binding_layout, CallColumns, Column, ControlColumns, FrameColumns, GlobalsColumns,
    LinearMemoryColumns, LocalsColumns, OpTableColumns, OperandStackColumns, OutputColumns, ParamInitColumns,
    SignExtensionColumns, StateColumns, WasmCrossStepColumnPair, WasmCrossStepLinkSpec, WasmLookupBindingLayout,
    WasmLookupBindingSpec, WasmLookupFamilyKind, WasmLookupFamilySpec, WasmMemoryActivation, WasmMemoryColumnKind,
    WasmMemoryColumnSpec, WasmMemorySpec,
};
pub use lookup_semantics::{sanity_check_lookup_row, LookupBuiltin, LookupExpr, LookupPredicate, LookupSemantics};
pub use lower::{build_row_traces, normalize_source, normalize_tracer, WasmExecutionStep, WasmTraceSource};
pub use memory_semantics::{preload_from_program_artifacts, sanity_check_memory_rows, WasmMemoryPreload};
pub use preprocess::{
    initial_semantic_state_digest, preprocess_seeded_batched, top_level_initial_state, top_level_initial_state_digest,
};
pub use prove::{prove, prove_batched, verify, WasmProof, WasmProveError};
pub use tables::{lookup_payload, WasmLookupArity, WasmLookupPayload};
pub use tagged_r1cs_builder::{WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag};
pub use witness_builder::{build_steps, build_witness_vector};
