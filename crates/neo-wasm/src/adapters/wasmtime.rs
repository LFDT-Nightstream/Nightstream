//! Owns direct Wasmtime tracing and normalization into the generic WASM IR.

use super::super::ir::{WasmBuildError, WasmPcEdgeKind, WasmStepTrace};
use super::super::isa::WasmOpcode;
use futures::executor::block_on;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use wasmtime::component::{Type as ComponentType, Val as ComponentVal};
use wasmtime::{
    component::{Component as WasmtimeComponent, Linker as WasmtimeComponentLinker},
    Config, DebugEvent, DebugHandler, Engine, FrameHandle, Func, Linker, Module, Store, StoreContextMut, Trap, Val,
};

mod decode;
mod normalize;
mod parse;
mod runtime_read;
use decode::DecodedOpcode;
use normalize::capture_frame;
use parse::{parse_first_component_core_module_artifacts, parse_wasm_artifacts, ParsedFunctionMeta};
use runtime_read::{build_debug_function_id_map, build_store_debug_function_id_map, val_to_string};
// Public path `adapters::wasmtime::traces_from_wasmtime_steps` is preserved via this re-export
// (also brings the name into scope for the component wrappers below).
pub use normalize::traces_from_wasmtime_steps;
pub use parse::{WasmProgramArtifacts, WasmProgramDecodeEntry, WasmProgramTables};

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct WasmtimeTraceStep {
    /// The cycle for memory and op-table lookups
    pub step: u64,
    pub frame_depth: usize,
    pub function: String,
    pub function_index: Option<u32>,
    pub pc: Option<u32>,
    /// Human-readable opcode label from wasmparser's Debug format, for display only.
    pub opcode: Option<String>,
    /// Structurally decoded opcode, populated at trace-collection time from
    /// `wasmparser::Operator`. Not serialized (runtime-only).
    pub opcode_decoded: Option<WasmOpcode>,
    /// Immediate value for `i32.const`, decoded at collection time.
    pub immediate_i32: Option<u32>,
    /// Normalized control-edge selector. `0` means default/fallthrough.
    pub control_choice: Option<u32>,
    /// Static classification of how this row's next pc is determined.
    pub pc_edge_kind: Option<WasmPcEdgeKind>,
    pub global_index: Option<u32>,
    pub global_value_before: Option<u32>,
    pub global_value_before_hi: Option<u32>,
    pub global_value_after: Option<u32>,
    pub global_value_after_hi: Option<u32>,
    pub table_id: Option<u32>,
    pub table_index: Option<u32>,
    pub table_value: Option<u32>,
    pub table_size: Option<u32>,
    pub function_ref: Option<u32>,
    /// Normalized function reference for the currently executing frame.
    pub current_function_ref: Option<u32>,
    pub target_function_is_guest: bool,
    pub function_type_id: Option<u32>,
    pub call_indirect_type_index: Option<u32>,
    pub expected_type_id: Option<u32>,
    pub call_param_count: Option<u8>,
    pub call_result_count: Option<u8>,
    pub memory_pages_before: Option<u32>,
    pub memory_pages_after: Option<u32>,
    pub memory: Option<WasmtimeTraceMemoryAccess>,
    pub locals: Vec<String>,
    pub locals_words_hi: Vec<u32>,
    pub operand_stack: Vec<String>,
    pub operand_stack_words: Vec<u32>,
    pub operand_stack_words_hi: Vec<u32>,
    /// Total number of locals (params + declared) in this frame.
    pub num_locals: u32,
    /// For `call` instructions: the binary offset of the instruction immediately after
    /// the call (= the return address). Populated at map-build time.
    pub call_return_pc: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmtimeTraceMemoryWordLane {
    pub word_addr: u64,
    pub value_before: u32,
    pub value_after: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmtimeTraceMemoryAccess {
    pub kind: String,
    pub memory_index: u32,
    pub width_bytes: u8,
    pub offset: u64,
    pub base_address: u64,
    pub effective_address: u64,
    pub byte_offset: u8,
    pub lane0: WasmtimeTraceMemoryWordLane,
    pub lane1: Option<WasmtimeTraceMemoryWordLane>,
    pub lane2: Option<WasmtimeTraceMemoryWordLane>,
    pub value_before_i32: Option<i32>,
    pub value_after_i32: Option<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmtimeTraceRun {
    /// Normalized string form of the export results, as produced by the
    /// reference wasmtime interpreter (`func.call_async`).
    ///
    /// This is the *reference* output and is NOT bound to the proof — the
    /// witness/CCS pipeline is built from `steps`, independently of this
    /// field, so it is for fixture/intent checks only. The proof-bound
    /// output is the `output` carried in the final semantic state, checked
    /// by `verify` against the prover-disclosed final `WasmStepState`.
    pub results: Vec<String>,
    pub steps: Vec<WasmtimeTraceStep>,
    /// Values of all locals (params + pure locals) at function entry, indexed by local index.
    /// Params have the argument values; pure locals are zero. Populated from the first frame step.
    pub initial_locals: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct WasmtimeDebugHandler;

#[derive(Debug, Default)]
/// Store data used by Wasmtime guest-debug tracing. This is public so callers can
/// configure component linkers for `*_component_*_with` helpers, which require
/// `Linker<WasmtimeTraceState>` in their callback signature.
pub struct WasmtimeTraceState {
    next_step: u64,
    steps: Vec<WasmtimeTraceStep>,
    opcode_map: Arc<BTreeMap<(u32, u32), DecodedOpcode>>,
    func_ref_ids: Arc<BTreeMap<usize, u32>>,
    function_metas: Arc<BTreeMap<u32, ParsedFunctionMeta>>,
    imported_function_count: u32,
}

/// Whether a wasmtime trap has a modeled terminal state, so the collected
/// steps (faulting row included) can stand in for a clean run with no results.
///
/// Most accepted traps are raised by a unique opcode. `TableOutOfBounds` is
/// the exception: wasmtime raises it for `table.get` / `table.set` /
/// `table.init` OOB too, but only `call_indirect` OOB is modeled, so it is
/// accepted only when the faulting (last collected) step is a `call_indirect`.
fn is_modeled_terminal_trap(trap: Option<&Trap>, last_step: Option<&WasmtimeTraceStep>) -> bool {
    match trap {
        Some(
            Trap::UnreachableCodeReached
            | Trap::IntegerDivisionByZero
            | Trap::IntegerOverflow
            | Trap::IndirectCallToNull
            | Trap::BadSignature,
        ) => true,
        Some(Trap::TableOutOfBounds) => {
            matches!(
                last_step.and_then(|step| step.opcode_decoded),
                Some(WasmOpcode::CallIndirect)
            )
        }
        _ => false,
    }
}

pub fn collect_wasmtime_steps(
    wasm_bytes: &[u8],
    export: &str,
    params: &[i32],
) -> Result<WasmtimeTraceRun, WasmBuildError> {
    let parsed = parse_wasm_artifacts(wasm_bytes)?;
    let imported_function_count = parsed.trace.imported_function_count;
    let opcode_map = Arc::new(parsed.trace.opcode_map);
    let function_metas = Arc::new(parsed.trace.function_metas);

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);

    let engine = Engine::new(&config)
        .map_err(|err| WasmBuildError::Trace(format!("failed to create Wasmtime engine: {err}")))?;
    let module = Module::from_binary(&engine, wasm_bytes)
        .map_err(|err| WasmBuildError::Trace(format!("failed to compile wasm bytes: {err}")))?;

    let mut store = Store::new(
        &engine,
        WasmtimeTraceState {
            next_step: 0,
            steps: Vec::new(),
            opcode_map,
            func_ref_ids: Arc::new(BTreeMap::new()),
            function_metas,
            imported_function_count,
        },
    );
    store.set_debug_handler(WasmtimeDebugHandler);

    {
        let mut edit = store
            .edit_breakpoints()
            .ok_or_else(|| WasmBuildError::Trace("guest debug not enabled".to_string()))?;
        edit.single_step(true)
            .map_err(|err| WasmBuildError::Trace(format!("failed to enable Wasmtime single-step mode: {err}")))?;
    }

    let linker = Linker::new(&engine);
    let instance = block_on(linker.instantiate_async(&mut store, &module))
        .map_err(|err| WasmBuildError::Trace(format!("failed to instantiate Wasmtime module: {err}")))?;
    let func_ref_ids = build_debug_function_id_map(&instance, &mut store)?;
    store.data_mut().func_ref_ids = Arc::new(func_ref_ids);
    let func: Func = instance
        .get_func(&mut store, export)
        .ok_or_else(|| WasmBuildError::Trace(format!("export '{export}' not found")))?;
    let param_vals: Vec<Val> = params.iter().map(|&v| Val::I32(v)).collect();
    let mut results = vec![Val::I32(0)];
    let results = match block_on(func.call_async(&mut store, &param_vals, &mut results)) {
        Ok(()) => results.iter().map(|&v| val_to_string(v)).collect(),
        // Traps with a modeled terminal state: keep the collected steps
        // (the faulting row included) and report no results. Other trap
        // causes (e.g. OOB access) are not provable yet and stay hard
        // errors.
        Err(err) if is_modeled_terminal_trap(err.downcast_ref::<Trap>(), store.data().steps.last()) => Vec::new(),
        Err(err) => {
            return Err(WasmBuildError::Trace(format!(
                "failed to execute Wasmtime export '{export}': {err}"
            )));
        }
    };

    let steps = store.data().steps.clone();
    let initial_locals = steps
        .iter()
        .find(|s| s.frame_depth == 0 && s.pc.is_some())
        .map(|s| {
            s.locals
                .iter()
                .map(|v| v.parse::<i128>().map(|n| (n as i32) as u32).unwrap_or(0))
                .collect()
        })
        .unwrap_or_default();

    Ok(WasmtimeTraceRun {
        results,
        steps,
        initial_locals,
    })
}

pub fn traces_from_wasmtime_wasm_bytes(wasm_bytes: &[u8], export: &str) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
    let run = collect_wasmtime_steps(wasm_bytes, export, &[])?;
    traces_from_wasmtime_steps(&run.steps)
}

pub fn collect_wasmtime_component_run(
    component_bytes: &[u8],
    export: &str,
) -> Result<WasmtimeTraceRun, WasmBuildError> {
    collect_wasmtime_component_run_with_linker(component_bytes, export, |_linker| Ok(()))
}

pub fn collect_wasmtime_component_run_with_linker<F>(
    component_bytes: &[u8],
    export: &str,
    configure_linker: F,
) -> Result<WasmtimeTraceRun, WasmBuildError>
where
    F: FnOnce(&mut WasmtimeComponentLinker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
{
    let parsed = parse_first_component_core_module_artifacts(component_bytes)?;
    let imported_function_count = parsed.trace.imported_function_count;
    let opcode_map = Arc::new(parsed.trace.opcode_map);
    let function_metas = Arc::new(parsed.trace.function_metas);

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    config.wasm_component_model(true);

    let engine = Engine::new(&config)
        .map_err(|err| WasmBuildError::Trace(format!("failed to create Wasmtime engine: {err}")))?;
    let component = WasmtimeComponent::new(&engine, component_bytes)
        .map_err(|err| WasmBuildError::Trace(format!("failed to compile component bytes: {err}")))?;

    let mut store = Store::new(
        &engine,
        WasmtimeTraceState {
            next_step: 0,
            steps: Vec::new(),
            opcode_map,
            func_ref_ids: Arc::new(BTreeMap::new()),
            function_metas,
            imported_function_count,
        },
    );
    store.set_debug_handler(WasmtimeDebugHandler);

    {
        let mut edit = store
            .edit_breakpoints()
            .ok_or_else(|| WasmBuildError::Trace("guest debug not enabled".to_string()))?;
        edit.single_step(true)
            .map_err(|err| WasmBuildError::Trace(format!("failed to enable Wasmtime single-step mode: {err}")))?;
    }

    let mut linker = WasmtimeComponentLinker::new(&engine);
    configure_linker(&mut linker)?;
    let instance = block_on(linker.instantiate_async(&mut store, &component))
        .map_err(|err| WasmBuildError::Trace(format!("failed to instantiate Wasmtime component: {err}")))?;
    let func_ref_ids = build_store_debug_function_id_map(&mut store)?;
    store.data_mut().func_ref_ids = Arc::new(func_ref_ids);
    let func = instance
        .get_func(&mut store, export)
        .ok_or_else(|| WasmBuildError::Trace(format!("component export '{export}' not found")))?;
    let mut results: Vec<ComponentVal> = func
        .ty(&store)
        .results()
        .map(default_component_result_value)
        .collect::<Result<_, _>>()?;
    block_on(func.call_async(&mut store, &[], &mut results))
        .map_err(|err| WasmBuildError::Trace(format!("failed to execute component export '{export}': {err}")))?;

    let steps = store.data().steps.clone();
    let initial_locals = steps
        .iter()
        .find(|s| s.frame_depth == 0 && s.pc.is_some())
        .map(|s| {
            s.locals
                .iter()
                .map(|v| v.parse::<i128>().map(|n| (n as i32) as u32).unwrap_or(0))
                .collect()
        })
        .unwrap_or_default();

    Ok(WasmtimeTraceRun {
        results: results
            .iter()
            .map(component_val_to_string)
            .collect::<Result<_, _>>()?,
        steps,
        initial_locals,
    })
}

pub fn traces_from_wasmtime_component(
    component_bytes: &[u8],
    export: &str,
) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
    let run = collect_wasmtime_component_run(component_bytes, export)?;
    traces_from_wasmtime_steps(&run.steps)
}

pub fn traces_from_wasmtime_component_with_linker<F>(
    component_bytes: &[u8],
    export: &str,
    configure_linker: F,
) -> Result<Vec<WasmStepTrace>, WasmBuildError>
where
    F: FnOnce(&mut WasmtimeComponentLinker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
{
    let run = collect_wasmtime_component_run_with_linker(component_bytes, export, configure_linker)?;
    traces_from_wasmtime_steps(&run.steps)
}

fn default_component_result_value(ty: ComponentType) -> Result<ComponentVal, WasmBuildError> {
    match ty {
        ComponentType::Bool => Ok(ComponentVal::Bool(false)),
        ComponentType::S8 => Ok(ComponentVal::S8(0)),
        ComponentType::U8 => Ok(ComponentVal::U8(0)),
        ComponentType::S16 => Ok(ComponentVal::S16(0)),
        ComponentType::U16 => Ok(ComponentVal::U16(0)),
        ComponentType::S32 => Ok(ComponentVal::S32(0)),
        ComponentType::U32 => Ok(ComponentVal::U32(0)),
        ComponentType::S64 => Ok(ComponentVal::S64(0)),
        ComponentType::U64 => Ok(ComponentVal::U64(0)),
        ComponentType::Float32 => Ok(ComponentVal::Float32(0.0f32)),
        ComponentType::Float64 => Ok(ComponentVal::Float64(0.0f64)),
        ComponentType::Char => Ok(ComponentVal::Char('\0')),
        ComponentType::String => Ok(ComponentVal::String(String::new())),
        other => Err(WasmBuildError::Unsupported(format!(
            "component trace helper does not yet support dynamic result placeholder for {other:?}"
        ))),
    }
}

fn component_val_to_string(val: &ComponentVal) -> Result<String, WasmBuildError> {
    Ok(match val {
        ComponentVal::Bool(v) => v.to_string(),
        ComponentVal::S8(v) => v.to_string(),
        ComponentVal::U8(v) => v.to_string(),
        ComponentVal::S16(v) => v.to_string(),
        ComponentVal::U16(v) => v.to_string(),
        ComponentVal::S32(v) => v.to_string(),
        ComponentVal::U32(v) => v.to_string(),
        ComponentVal::S64(v) => v.to_string(),
        ComponentVal::U64(v) => v.to_string(),
        ComponentVal::Float32(v) => v.to_string(),
        ComponentVal::Float64(v) => v.to_string(),
        ComponentVal::Char(v) => v.to_string(),
        ComponentVal::String(v) => v.clone(),
        other => {
            return Err(WasmBuildError::Unsupported(format!(
                "component trace helper does not yet support result formatting for {other:?}"
            )))
        }
    })
}

impl DebugHandler for WasmtimeDebugHandler {
    type Data = WasmtimeTraceState;

    fn handle(
        &self,
        mut store: StoreContextMut<'_, Self::Data>,
        event: DebugEvent<'_>,
    ) -> impl Future<Output = ()> + Send {
        async move {
            if !matches!(event, DebugEvent::Breakpoint) {
                return;
            }

            let frames = store.debug_exit_frames().collect::<Vec<FrameHandle>>();
            let step = store.data().next_step;
            let mut rows = Vec::with_capacity(frames.len());
            for (frame_depth, frame) in frames.iter().enumerate() {
                match capture_frame(step, frame_depth, frame, &mut store) {
                    Ok(row) => rows.push(row),
                    Err(error) => rows.push(WasmtimeTraceStep {
                        step,
                        frame_depth,
                        function: "<frame-inspection-error>".to_string(),
                        locals: vec![error.to_string()],
                        ..Default::default()
                    }),
                }
            }

            let state = store.data_mut();
            state.next_step += 1;
            state.steps.extend(rows);
        }
    }
}

/// Parse a wasm binary into the static program artifacts used to seed
/// verifier/prover memory tables.
pub fn extract_wasm_program_artifacts(wasm_bytes: &[u8]) -> Result<WasmProgramArtifacts, WasmBuildError> {
    parse_wasm_artifacts(wasm_bytes)
}

pub fn extract_first_component_core_program_artifacts(
    component_bytes: &[u8],
) -> Result<WasmProgramArtifacts, WasmBuildError> {
    parse_first_component_core_module_artifacts(component_bytes)
}

/// Build the next-PC ROM from validated WASM bytecode using statically resolved control edges.
///
/// Conditional control rows (`if`, `br_if`) emit both possible successors. Structural rows
/// (`block`, `loop`, `else`, inner `end`) are kept as explicit ROM entries so the trace can
/// carry them as regular witness rows.
pub fn build_pc_rom_from_binary(wasm_bytes: &[u8]) -> Result<Vec<(u64, u64, u64)>, WasmBuildError> {
    Ok(extract_wasm_program_artifacts(wasm_bytes)?.tables.pc_rom)
}
