//! Owns direct Wasmtime tracing and normalization into the generic WASM IR.

use super::super::ir::{WasmBuildError, WasmPcEdgeKind, WasmVmStep};
use super::super::isa::WasmOpcode;
use futures::executor::block_on;
use std::collections::BTreeMap;
use std::future::Future;
use std::marker::PhantomData;
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
pub use runtime_read::build_debug_function_id_map;
use runtime_read::{build_single_trace_store_debug_function_id_map, val_to_string};
// Public path `adapters::wasmtime::traces_from_wasmtime_steps` is preserved via this re-export
// (also brings the name into scope for the component wrappers below).
pub use normalize::{traces_from_wasmtime_steps, traces_from_wasmtime_steps_with_host_events};
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
    /// Declared max page count (capped at the wasm32 limit), constant for the
    /// execution. Threaded into the carried `max_memory_pages` boundary.
    pub memory_max_pages: Option<u32>,
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
    /// Byte offset immediately after this instruction's encoding. For `call`
    /// this is the return PC; for branches it is the linear successor, not
    /// necessarily the runtime next PC.
    pub pc_after_instruction: Option<u64>,
    /// Per-call host-event input words recorded by the embedder's host
    /// function while servicing this host-call row (see
    /// [`WasmtimeTraceState::record_call_inputs`]). Consumed by event-bound
    /// normalization.
    pub host_call_inputs: Vec<u64>,
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

#[derive(Clone, Debug)]
pub struct WasmtimeTraceRun {
    /// Verifier-owned static program tables used by host-event-aware
    /// normalization and memory preloading.
    pub program_tables: WasmProgramTables,
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
}

/// Single-step tracing hook for store data that exposes [`WasmtimeTraceState`].
pub struct WasmtimeTraceHandler<T>(PhantomData<fn() -> T>);

impl<T> WasmtimeTraceHandler<T> {
    pub fn new() -> Self {
        WasmtimeTraceHandler(PhantomData)
    }
}

// Avoid derive-imposed bounds on `T`; the handler stores only PhantomData.
impl<T> Default for WasmtimeTraceHandler<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for WasmtimeTraceHandler<T> {
    fn clone(&self) -> Self {
        WasmtimeTraceHandler(PhantomData)
    }
}

impl<T> Copy for WasmtimeTraceHandler<T> {}

impl<T> std::fmt::Debug for WasmtimeTraceHandler<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WasmtimeTraceHandler")
    }
}

#[derive(Debug, Default)]
/// Store data used by Wasmtime guest-debug tracing. This is public so callers can
/// configure component linkers for `*_component_*_with` helpers, which require
/// `Linker<WasmtimeTraceState>` in their callback signature.
pub struct WasmtimeTraceState {
    next_step: u64,
    steps: Vec<WasmtimeTraceStep>,
    /// Per-module lowering tables, behind a single `Arc` so the breakpoint hook
    /// can cheaply clone a handle out (one refcount bump) and read them while the
    /// live frame is read through `&mut store`.
    tables: Arc<LoweringTables>,
}

/// Per-instance lowering tables: the static decode/metadata derived from the
/// module, plus the post-instantiation funcref-id map.
#[derive(Clone, Debug, Default)]
pub(crate) struct LoweringTables {
    pub(crate) opcode_map: BTreeMap<(u32, u32), DecodedOpcode>,
    /// Raw-funcref-pointer to module-local id, filled post-instantiation via
    /// [`WasmtimeTraceState::set_func_ref_ids`] (empty until then).
    pub(crate) func_ref_ids: BTreeMap<usize, u32>,
    pub(crate) function_metas: BTreeMap<u32, ParsedFunctionMeta>,
    pub(crate) imported_function_count: u32,
    /// Declared max pages for memory 0 (a module constant), seeded from the parse
    /// artifacts at construction. `None` when the module has no default memory.
    pub(crate) memory_max_pages: Option<u32>,
}

/// Routes captured wasm steps to trace state keyed by `Instance::debug_index_in_store()`.
pub trait WasmTraceSink {
    fn wasm_trace_state(&self, instance_index: u32) -> Option<&WasmtimeTraceState>;
    fn wasm_trace_state_mut(&mut self, instance_index: u32) -> Option<&mut WasmtimeTraceState>;
    /// The debug hook cannot return errors, so missing trace registrations must
    /// be surfaced through the sink.
    fn record_untraced_instance(&mut self, instance_index: u32);
}

impl WasmTraceSink for WasmtimeTraceState {
    fn wasm_trace_state(&self, _instance_index: u32) -> Option<&WasmtimeTraceState> {
        Some(self)
    }
    fn wasm_trace_state_mut(&mut self, _instance_index: u32) -> Option<&mut WasmtimeTraceState> {
        Some(self)
    }
    fn record_untraced_instance(&mut self, _instance_index: u32) {}
}

impl WasmtimeTraceState {
    /// Build trace state from parsed program artifacts.
    ///
    /// Funcref normalization also requires a post-instantiation
    /// [`WasmtimeTraceState::set_func_ref_ids`] call.
    pub fn from_program_artifacts(artifacts: &WasmProgramArtifacts) -> Self {
        WasmtimeTraceState {
            next_step: 0,
            steps: Vec::new(),
            tables: Arc::new(LoweringTables {
                opcode_map: artifacts.trace.opcode_map.clone(),
                func_ref_ids: BTreeMap::new(),
                function_metas: artifacts.trace.function_metas.clone(),
                imported_function_count: artifacts.trace.imported_function_count,
                memory_max_pages: artifacts.tables.max_memory_pages,
            }),
        }
    }

    /// The trace rows collected so far, in capture order.
    pub fn steps(&self) -> &[WasmtimeTraceStep] {
        &self.steps
    }

    /// Take ownership of the collected trace rows, leaving the state empty so it
    /// can be reused for a subsequent run.
    pub fn take_steps(&mut self) -> Vec<WasmtimeTraceStep> {
        std::mem::take(&mut self.steps)
    }

    /// Install the post-instantiation raw-funcref to module-local id map.
    ///
    /// Install this before tracing rows that may contain funcrefs.
    pub fn set_func_ref_ids(&mut self, func_ref_ids: BTreeMap<usize, u32>) {
        Arc::make_mut(&mut self.tables).func_ref_ids = func_ref_ids;
    }

    /// Record per-call host-event input words for the in-flight host call
    /// (for example, ref ids or caller identities). Call from
    /// inside a host-function implementation (`store.data_mut()`): the debug
    /// hook captures each instruction before it executes, so the latest
    /// captured step is the host-call row being serviced and the batch
    /// attaches to it — no call-order bookkeeping. Repeated calls append.
    pub fn record_call_inputs(&mut self, words: &[u64]) -> Result<(), WasmBuildError> {
        let row = self.steps.last_mut().ok_or_else(|| {
            WasmBuildError::Trace("record_call_inputs: no captured step; not inside a traced host call".to_string())
        })?;
        let is_host_call = matches!(row.opcode_decoded, Some(WasmOpcode::Call | WasmOpcode::CallIndirect))
            && !row.target_function_is_guest;
        if !is_host_call {
            return Err(WasmBuildError::Trace(format!(
                "record_call_inputs: latest captured step (cycle {}, opcode {:?}) is not a host-call row",
                row.step, row.opcode
            )));
        }
        row.host_call_inputs.extend_from_slice(words);
        Ok(())
    }
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
        // OOB linear-memory access is modeled only for load/store opcodes.
        Some(Trap::MemoryOutOfBounds) => last_step
            .and_then(|step| step.opcode_decoded)
            .is_some_and(|op| op.memory_access_info().is_some()),
        _ => false,
    }
}

pub fn collect_wasmtime_steps(
    wasm_bytes: &[u8],
    export: &str,
    params: &[i32],
) -> Result<WasmtimeTraceRun, WasmBuildError> {
    let parsed = parse_wasm_artifacts(wasm_bytes)?;

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);

    let engine = Engine::new(&config)
        .map_err(|err| WasmBuildError::Trace(format!("failed to create Wasmtime engine: {err}")))?;
    let module = Module::from_binary(&engine, wasm_bytes)
        .map_err(|err| WasmBuildError::Trace(format!("failed to compile wasm bytes: {err}")))?;

    let mut store = Store::new(&engine, WasmtimeTraceState::from_program_artifacts(&parsed));
    store.set_debug_handler(WasmtimeTraceHandler::<WasmtimeTraceState>::new());

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
    store.data_mut().set_func_ref_ids(func_ref_ids);
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

    Ok(WasmtimeTraceRun {
        program_tables: parsed.tables,
        results,
        steps,
    })
}

pub fn traces_from_wasmtime_wasm_bytes(wasm_bytes: &[u8], export: &str) -> Result<Vec<WasmVmStep>, WasmBuildError> {
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
    collect_wasmtime_component_run_with_linker_and_args(component_bytes, export, &[], configure_linker)
}

/// [`collect_wasmtime_component_run_with_linker`] for exports with
/// parameters: `args` are passed to the component-level call (canonical ABI
/// lowering lands them in the export's locals).
pub fn collect_wasmtime_component_run_with_linker_and_args<F>(
    component_bytes: &[u8],
    export: &str,
    args: &[ComponentVal],
    configure_linker: F,
) -> Result<WasmtimeTraceRun, WasmBuildError>
where
    F: FnOnce(&mut WasmtimeComponentLinker<WasmtimeTraceState>) -> Result<(), WasmBuildError>,
{
    let parsed = parse_first_component_core_module_artifacts(component_bytes)?;

    let mut config = Config::new();
    config.guest_debug(true);
    config.wasm_reference_types(true);
    config.wasm_function_references(true);
    config.wasm_component_model(true);

    let engine = Engine::new(&config)
        .map_err(|err| WasmBuildError::Trace(format!("failed to create Wasmtime engine: {err}")))?;
    let component = WasmtimeComponent::new(&engine, component_bytes)
        .map_err(|err| WasmBuildError::Trace(format!("failed to compile component bytes: {err}")))?;

    let mut store = Store::new(&engine, WasmtimeTraceState::from_program_artifacts(&parsed));
    store.set_debug_handler(WasmtimeTraceHandler::<WasmtimeTraceState>::new());

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
    let func_ref_ids = build_single_trace_store_debug_function_id_map(&mut store)?;
    store.data_mut().set_func_ref_ids(func_ref_ids);
    let func = instance
        .get_func(&mut store, export)
        .ok_or_else(|| WasmBuildError::Trace(format!("component export '{export}' not found")))?;
    let mut results: Vec<ComponentVal> = func
        .ty(&store)
        .results()
        .map(default_component_result_value)
        .collect::<Result<_, _>>()?;
    block_on(func.call_async(&mut store, args, &mut results))
        .map_err(|err| WasmBuildError::Trace(format!("failed to execute component export '{export}': {err}")))?;

    let steps = store.data().steps.clone();

    Ok(WasmtimeTraceRun {
        program_tables: parsed.tables,
        results: results
            .iter()
            .map(component_val_to_string)
            .collect::<Result<_, _>>()?,
        steps,
    })
}

pub fn traces_from_wasmtime_component(component_bytes: &[u8], export: &str) -> Result<Vec<WasmVmStep>, WasmBuildError> {
    let run = collect_wasmtime_component_run(component_bytes, export)?;
    traces_from_wasmtime_steps(&run.steps)
}

pub fn traces_from_wasmtime_component_with_linker<F>(
    component_bytes: &[u8],
    export: &str,
    configure_linker: F,
) -> Result<Vec<WasmVmStep>, WasmBuildError>
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

impl<T: WasmTraceSink + Send + 'static> DebugHandler for WasmtimeTraceHandler<T> {
    type Data = T;

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
            // Only the innermost frame is executing this step. Outer frames can
            // belong to a different core instance in cross-instance calls.
            let Some(frame) = frames.first() else {
                return;
            };
            let instance_index = match frame.instance(&mut store) {
                Ok(instance) => instance.debug_index_in_store(),
                // TODO: we may actually want to record errors into the store,
                // so that they can't be surfaced later (even if we don't
                // short-circuit, as the handle function can't error)
                Err(_) => return,
            };
            let (tables, step) = match store.data().wasm_trace_state(instance_index) {
                Some(state) => (state.tables.clone(), state.next_step),
                None => {
                    store.data_mut().record_untraced_instance(instance_index);
                    return;
                }
            };
            let row = match capture_frame(step, 0, frame, &mut store, &tables) {
                Ok(row) => row,
                Err(error) => WasmtimeTraceStep {
                    step,
                    frame_depth: 0,
                    function: "<frame-inspection-error>".to_string(),
                    locals: vec![error.to_string()],
                    ..Default::default()
                },
            };

            if let Some(state) = store.data_mut().wasm_trace_state_mut(instance_index) {
                state.next_step += 1;
                state.steps.push(row);
            }
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
