//! Temporary compatibility barrel for the WASM machine-layer frontend.

pub use super::adapters::rwasm::{traces_from_rwasm_instr_states, traces_from_rwasm_tracer};
pub use super::builder::WasmTraceBuilder;
pub use super::ccs::WasmVmSpec;
pub use super::ir::{
    LinearMemoryAccess, LinearMemoryWordLane, StackLaneAccess, WasmBoundaryState, WasmBuildError, WasmStepTrace,
};
pub use super::isa::{opcode_info_from_code, WasmOpcode, WasmOpcodeClass, WasmOpcodeInfo, WasmShoutOpcode};
pub use super::layout::{WasmColumnSpec, COLUMN_SPECS};
pub use super::lookup_binding_builder::{
    build_wasm_lookup_binding_layout, CallColumns, Column, ControlColumns, FrameColumns, GlobalsColumns,
    LinearMemoryColumns, LocalsColumns, OperandStackColumns, ParamInitColumns, ShoutColumns, StateColumns,
    WasmCrossStepLinkSpec, WasmLookupBindingLayout, WasmLookupBindingSpec, WasmLookupFamilyKind, WasmLookupFamilySpec,
    WasmMemoryColumnKind, WasmMemoryColumnSpec, WasmMemorySpec,
};
pub use super::lower::{build_row_traces, normalize_source, normalize_tracer, WasmExecutionStep, WasmTraceSource};
pub use super::prove::{
    prove_relation, verify_relation, WasmProveError, WasmProverInput, WasmPublicInput, WasmVerifierInput,
};
pub use super::relation::{
    prove_wasm_relation, verify_wasm_relation, WasmBoundaryRow, WasmLookupRow, WasmMemoryEvent, WasmMemoryEventKind,
    WasmMemoryKind, WasmRelationProof,
};
pub use super::tables::{lookup_payload, WasmLookupArity, WasmLookupPayload};
