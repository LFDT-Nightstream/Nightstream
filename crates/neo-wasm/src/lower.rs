use super::adapters::rwasm::traces_from_rwasm_tracer;
use super::adapters::wasmtime::WasmtimeTraceStep;
use super::ir::{WasmBuildError, WasmStepTrace};
use rwasm::Tracer;

pub type WasmExecutionStep = WasmStepTrace;

pub trait WasmTraceSource {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmExecutionStep>, WasmBuildError>;
}

impl WasmTraceSource for Tracer {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
        traces_from_rwasm_tracer(self)
    }
}

impl WasmTraceSource for [WasmExecutionStep] {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
        Ok(self.to_vec())
    }
}

impl WasmTraceSource for Vec<WasmExecutionStep> {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
        Ok(self.clone())
    }
}

impl WasmTraceSource for WasmtimeTraceStep {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
        super::adapters::wasmtime::traces_from_wasmtime_steps(core::slice::from_ref(self))
    }
}

pub fn normalize_source(source: &impl WasmTraceSource) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
    source.lower_to_wasm_ir()
}

pub fn normalize_tracer(tracer: &Tracer) -> Result<Vec<WasmExecutionStep>, WasmBuildError> {
    normalize_source(tracer)
}

pub fn build_row_traces(steps: &[WasmExecutionStep]) -> Vec<WasmExecutionStep> {
    steps.to_vec()
}
