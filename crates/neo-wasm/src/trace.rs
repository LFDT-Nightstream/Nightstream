//! Compatibility barrel for the WASM frontend.

pub use super::adapters::rwasm::{traces_from_rwasm_instr_states, traces_from_rwasm_tracer};
pub use super::builder::{build_witness_vector, WasmTraceBuilder};
pub use super::ir::{boundary_states, StackLaneAccess, WasmBoundaryState, WasmBuildError, WasmStepTrace};
pub use super::lower::{build_row_traces, normalize_source, normalize_tracer, WasmExecutionStep, WasmTraceSource};
