//! Owns direct Wasmtime tracing and normalization into the generic WASM IR.

use super::super::ir::{
    LinearMemoryAccess, LinearMemoryWordLane, StackLaneAccess, WasmBuildError, WasmParamInitState, WasmPcEdgeKind,
    WasmStepTrace,
};
use super::super::isa::{opcode_code, opcode_info_from_code, WasmOpcode};
use super::super::lower::WasmTraceSource;
use futures::executor::block_on;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use wasmparser::{Parser, Payload};
use wasmtime::component::{Type as ComponentType, Val as ComponentVal};
use wasmtime::{
    component::{Component as WasmtimeComponent, Linker as WasmtimeComponentLinker},
    Config, DebugEvent, DebugHandler, Engine, FrameHandle, Func, Linker, Module, Store, StoreContextMut, Val,
};

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct WasmtimeTraceStep {
    /// The cycle for twist and shout
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
    /// Normalized string form of the export results for this helper path.
    pub results: Vec<String>,
    pub steps: Vec<WasmtimeTraceStep>,
    /// Static `(pc_before, control_choice, pc_after)` rows for the witness-facing pc ROM.
    pub pc_rom: Vec<(u64, u64, u64)>,
    pub pc_edge_kinds: Vec<(u64, u64)>,
    /// Static `(pc_before, function_ref)` rows binding each code PC to its containing function.
    pub pc_function_refs: Vec<(u64, u64)>,
    /// Static `(function_ref, entry_pc)` pairs for defined functions in the embedded core module.
    pub function_entries: Vec<(u64, u64)>,
    /// Static `(function_ref, type_id)` pairs for call-indirect type checks.
    pub function_types: Vec<(u64, u64)>,
    /// Static `(function_ref, param_count)` pairs for dynamic call stack arity.
    pub function_param_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, result_count)` pairs for dynamic call stack arity.
    pub function_result_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, params + declared locals)` pairs for frame-base transitions.
    pub function_local_counts: Vec<(u64, u64)>,
    /// Static `(function_ref, is_guest)` pairs for call-frame entry.
    pub function_guest_flags: Vec<(u64, u64)>,
    /// Static `(pc_before, function_ref)` pairs for direct call targets.
    pub call_targets: Vec<(u64, u64)>,
    /// Static `(raw_type_index, expected_type_id)` pairs for call-indirect type checks.
    pub module_types: Vec<(u64, u64)>,
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

#[derive(Clone, Debug)]
struct DecodedOpcode {
    text: String,
    memory: Option<DecodedMemoryOpcode>,
    /// Structurally decoded from `wasmparser::Operator` at map-build time.
    decoded: Option<(WasmOpcode, Option<u32>)>,
    control: Option<DecodedControlOpcode>,
    pc_edge_kind: WasmPcEdgeKind,
    call_indirect_type_index: Option<u32>,
    expected_type_id: Option<u32>,
    /// For `call` instructions: binary offset of the instruction after the call = return address.
    call_return_pc: Option<u64>,
}

struct ParsedWasmArtifacts {
    // Static decode table keyed by Wasmtime's `(defined_function_index, pc)` pair so guest-debug
    // frames can recover opcode/immediate metadata without reparsing at trace time.
    opcode_map: BTreeMap<(u32, u32), DecodedOpcode>,
    pc_rom: Vec<(u64, u64, u64)>,
    pc_edge_kinds: Vec<(u64, u64)>,
    pc_function_refs: Vec<(u64, u64)>,
    // Static `(function_ref, entry_pc)` pairs keyed by the normalized funcref id space used in
    // tables and direct/indirect calls.
    function_entries: Vec<(u64, u64)>,
    function_types: Vec<(u64, u64)>,
    function_param_counts: Vec<(u64, u64)>,
    function_result_counts: Vec<(u64, u64)>,
    function_local_counts: Vec<(u64, u64)>,
    function_guest_flags: Vec<(u64, u64)>,
    call_targets: Vec<(u64, u64)>,
    module_types: Vec<(u64, u64)>,
    imported_function_count: u32,
    function_metas: BTreeMap<u32, ParsedFunctionMeta>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParsedFunctionMeta {
    type_id: u32,
    param_count: u8,
    result_count: u8,
    num_locals: u32,
    entry_pc: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct DecodedMemoryOpcode {
    kind: DecodedMemoryAccessKind,
    memory_index: u32,
    offset: u64,
}

#[derive(Clone, Copy, Debug)]
enum DecodedMemoryAccessKind {
    I32Load,
    I32Load8S,
    I32Load8U,
    I32Load16S,
    I32Load16U,
    I64Load,
    I32Store,
    I32Store8,
    I32Store16,
    I64Store,
}

#[derive(Clone, Copy, Debug)]
enum DecodedControlOpcode {
    BrTable { len: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlFrameKind {
    Block,
    Loop,
    If,
}

#[derive(Clone, Debug)]
struct ControlFrame {
    kind: ControlFrameKind,
    branch_target: Option<u64>,
    pending_to_end: Vec<u64>,
    pending_if_false: Option<u64>,
}

impl ControlFrame {
    fn new(kind: ControlFrameKind, branch_target: Option<u64>) -> Self {
        Self {
            kind,
            branch_target,
            pending_to_end: Vec::new(),
            pending_if_false: None,
        }
    }
}

impl DecodedMemoryAccessKind {
    fn width_bytes(self) -> u8 {
        match self {
            Self::I32Load | Self::I32Store => 4,
            Self::I64Load | Self::I64Store => 8,
            Self::I32Load8S | Self::I32Load8U | Self::I32Store8 => 1,
            Self::I32Load16S | Self::I32Load16U | Self::I32Store16 => 2,
        }
    }

    fn is_store(self) -> bool {
        matches!(
            self,
            Self::I32Store | Self::I32Store8 | Self::I32Store16 | Self::I64Store
        )
    }
}

impl WasmTraceSource for [WasmtimeTraceStep] {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
        traces_from_wasmtime_steps(self)
    }
}

impl WasmTraceSource for Vec<WasmtimeTraceStep> {
    fn lower_to_wasm_ir(&self) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
        traces_from_wasmtime_steps(self)
    }
}

pub fn collect_wasmtime_steps(
    wasm_bytes: &[u8],
    export: &str,
    params: &[i32],
) -> Result<WasmtimeTraceRun, WasmBuildError> {
    let parsed = parse_wasm_artifacts(wasm_bytes)?;
    let imported_function_count = parsed.imported_function_count;
    let opcode_map = Arc::new(parsed.opcode_map);
    let function_metas = Arc::new(parsed.function_metas);

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
    block_on(func.call_async(&mut store, &param_vals, &mut results))
        .map_err(|err| WasmBuildError::Trace(format!("failed to execute Wasmtime export '{export}': {err}")))?;
    let results = results.iter().map(|&v| val_to_string(v)).collect();

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
        pc_rom: parsed.pc_rom,
        pc_edge_kinds: parsed.pc_edge_kinds,
        pc_function_refs: parsed.pc_function_refs,
        function_entries: parsed.function_entries,
        function_types: parsed.function_types,
        function_param_counts: parsed.function_param_counts,
        function_result_counts: parsed.function_result_counts,
        function_local_counts: parsed.function_local_counts,
        function_guest_flags: parsed.function_guest_flags,
        call_targets: parsed.call_targets,
        module_types: parsed.module_types,
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
    let imported_function_count = parsed.imported_function_count;
    let pc_rom = parsed.pc_rom.clone();
    let function_entries = parsed.function_entries.clone();
    let function_types = parsed.function_types.clone();
    let function_param_counts = parsed.function_param_counts.clone();
    let function_result_counts = parsed.function_result_counts.clone();
    let function_local_counts = parsed.function_local_counts.clone();
    let function_guest_flags = parsed.function_guest_flags.clone();
    let call_targets = parsed.call_targets.clone();
    let module_types = parsed.module_types.clone();
    let pc_edge_kinds = parsed.pc_edge_kinds.clone();
    let pc_function_refs = parsed.pc_function_refs.clone();
    let opcode_map = Arc::new(parsed.opcode_map);
    let function_metas = Arc::new(parsed.function_metas);

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
        pc_rom,
        pc_edge_kinds,
        pc_function_refs,
        function_entries,
        function_types,
        function_param_counts,
        function_result_counts,
        function_local_counts,
        function_guest_flags,
        call_targets,
        module_types,
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

pub fn traces_from_wasmtime_steps(rows: &[WasmtimeTraceStep]) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
    let mut supported = Vec::new();
    for row in rows {
        if let Some(normalized) = normalize_supported_row(row)? {
            supported.push(normalized);
        }
    }

    let mut out = Vec::with_capacity(supported.len());
    // Runtime call stack: (return_pc, caller_fbp). Grows on Call, shrinks on non-final Return.
    let mut call_stack: Vec<(u64, u64)> = Vec::new();
    // Frame base pointer: absolute offset in the flat locals array where current function's
    // locals start. FBP_callee = FBP_caller + num_locals_caller.
    let mut fbp: u64 = 0;
    let mut param_init_state = WasmParamInitState::ZERO;

    for (idx, current) in supported.iter().enumerate() {
        let next = supported.get(idx + 1);
        let pc_before = u64::from(current.pc);
        let pc_after = next
            .map(|row| u64::from(row.pc))
            .unwrap_or_else(|| pc_before.saturating_add(1));
        let stack_reads = current
            .stack_reads_override
            .unwrap_or(current.info.stack_reads);
        let stack_writes = current
            .stack_writes_override
            .unwrap_or(current.info.stack_writes);
        let sp_before = current.operand_stack.len() as u64;
        let expected_sp_after = sp_before
            .saturating_sub(u64::from(stack_reads))
            .saturating_add(u64::from(stack_writes));
        let sp_after = next
            .map(|row| row.operand_stack.len() as u64)
            .unwrap_or(expected_sp_after);
        let stack_read0 = read_lane(&current.operand_stack, sp_before, stack_reads, 0);
        let stack_read1 = read_lane(&current.operand_stack, sp_before, stack_reads, 1);
        let stack_read2 = read_lane(&current.operand_stack, sp_before, stack_reads, 2);
        let stack_write0 = write_lane(current, next, sp_after)?;
        // Only the very last step of the whole trace is halted.
        let halted = next.is_none();

        // local_read_value: the local's value before this step (local.get: pushed onto stack).
        // local_write_value: the value being stored into the local (local.set / local.tee:
        //   the top of operand stack at this step, captured before execution).
        let local_read_value = if matches!(current.opcode, WasmOpcode::LocalGet) {
            current.local_value
        } else {
            None
        };
        let local_write_value = if matches!(current.opcode, WasmOpcode::LocalSet | WasmOpcode::LocalTee) {
            current.operand_stack.last().copied()
        } else {
            None
        };
        let global_read_value = if matches!(current.opcode, WasmOpcode::GlobalGet) {
            current.global_value_before
        } else {
            None
        };
        let global_write_value = if matches!(current.opcode, WasmOpcode::GlobalSet) {
            current.global_value_after
        } else {
            None
        };

        // FBP tracking and call/return handling.
        let current_fbp = fbp;
        let call_stack_push;
        let call_stack_pop;
        let callee_initial_params;
        let guest_callee_fbp;

        match current.opcode {
            WasmOpcode::Call | WasmOpcode::CallIndirect => {
                let return_pc = current.call_return_pc.unwrap_or(pc_after);
                if !current.target_function_is_guest {
                    // Imported/host callees do not produce guest core rows. In that case the next
                    // guest row is already the post-call continuation, so there is no guest
                    // call-stack boundary to model in this trace.
                    call_stack_push = None;
                    call_stack_pop = None;
                    callee_initial_params = vec![];
                    guest_callee_fbp = None;
                } else {
                    let param_count = current.call_param_count.ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing call parameter count for guest call at cycle {}",
                            current.cycle
                        ))
                    })?;
                    let expected_stack_reads = if matches!(current.opcode, WasmOpcode::CallIndirect) {
                        param_count.checked_add(1).ok_or_else(|| {
                            WasmBuildError::Trace(format!(
                                "call_indirect parameter count overflow at cycle {}",
                                current.cycle
                            ))
                        })?
                    } else {
                        param_count
                    };
                    if stack_reads != expected_stack_reads {
                        return Err(WasmBuildError::Trace(format!(
                            "call stack read count {} does not match expected count {} at cycle {}",
                            stack_reads, expected_stack_reads, current.cycle
                        )));
                    }
                    let callee_fbp = current_fbp
                        .checked_add(u64::from(current.num_locals))
                        .ok_or_else(|| {
                            WasmBuildError::Trace(format!("callee frame base overflow at cycle {}", current.cycle))
                        })?;
                    call_stack_push = Some((return_pc, current_fbp));
                    call_stack_pop = None;
                    callee_initial_params = collect_callee_initial_params(next, callee_fbp, param_count);
                    guest_callee_fbp = Some(callee_fbp);
                    call_stack.push((return_pc, current_fbp));
                    fbp = callee_fbp;
                }
            }
            WasmOpcode::Return | WasmOpcode::End if !call_stack.is_empty() => {
                // Non-final return: restore caller's FBP from the call stack.
                let (ret_pc, caller_fbp) = call_stack.pop().unwrap();
                call_stack_push = None;
                call_stack_pop = Some((ret_pc, caller_fbp));
                callee_initial_params = vec![];
                guest_callee_fbp = None;
                fbp = caller_fbp;
            }
            _ => {
                call_stack_push = None;
                call_stack_pop = None;
                callee_initial_params = vec![];
                guest_callee_fbp = None;
            }
        }

        let program_cycle = out.len() as u64;
        let param_init_before = param_init_state;
        let mut param_init_after = WasmParamInitState::ZERO;
        if call_stack_push.is_some() {
            param_init_after = WasmParamInitState {
                active: !callee_initial_params.is_empty(),
                remaining: u32::try_from(callee_initial_params.len()).map_err(|_| {
                    WasmBuildError::Trace(format!(
                        "call parameter count does not fit u32 at cycle {}",
                        current.cycle
                    ))
                })?,
            };
        }

        out.push(WasmStepTrace {
            // Sequential index within the normalized trace. Structural-only opcodes
            // (loop, block, inner End) are filtered before this loop, so this is
            // always consecutive — matching Stage 3's cycle_delta == 1 invariant.
            cycle: program_cycle,
            row_kind: super::super::ir::WasmRowKind::Program,
            pc_before,
            pc_after,
            control_choice: current.control_choice,
            pc_edge_kind: current.pc_edge_kind,
            param_init_before,
            param_init_after,
            wide_values_enabled: current.wide_values_enabled,
            opcode_code: current.info.code,
            opcode: current.opcode,
            info: current.info,
            stack_reads_override: current.stack_reads_override,
            stack_writes_override: current.stack_writes_override,
            sp_before,
            sp_after,
            current_function_ref: current.current_function_ref.unwrap_or(0),
            current_function_num_locals: current.num_locals,
            stack_read0,
            stack_read0_hi: read_lane_hi(&current.operand_stack_hi, stack_reads, 0),
            stack_read1,
            stack_read1_hi: read_lane_hi(&current.operand_stack_hi, stack_reads, 1),
            stack_read2,
            stack_read2_hi: read_lane_hi(&current.operand_stack_hi, stack_reads, 2),
            stack_write0,
            stack_write0_hi: write_lane_hi(current, next)?,
            linear_memory: current.linear_memory,
            linear_memory_offset: current.linear_memory_offset,
            memory_pages_before: current.memory_pages_before,
            memory_pages_after: current.memory_pages_after,
            halted,
            locals_fbp: current_fbp,
            locals_fbp_after: fbp,
            local_index: current.local_index,
            local_read_value,
            local_read_value_hi: if matches!(current.opcode, WasmOpcode::LocalGet) {
                current.local_value_hi
            } else {
                None
            },
            local_write_value,
            local_write_value_hi: if matches!(current.opcode, WasmOpcode::LocalSet | WasmOpcode::LocalTee) {
                current.operand_stack_hi.last().copied()
            } else {
                None
            },
            global_index: current.global_index,
            global_read_value,
            global_read_value_hi: if matches!(current.opcode, WasmOpcode::GlobalGet) {
                current.global_value_before_hi
            } else {
                None
            },
            global_write_value,
            global_write_value_hi: if matches!(current.opcode, WasmOpcode::GlobalSet) {
                current.global_value_after_hi
            } else {
                None
            },
            table_id: current.table_id,
            table_index: current.table_index,
            table_value: current.table_value,
            function_ref: current.function_ref,
            target_function_is_guest: current.target_function_is_guest,
            function_type_id: current.function_type_id,
            expected_type_id: current.expected_type_id,
            call_indirect_type_index: current.call_indirect_type_index,
            table_size: current.table_size,
            call_param_count: current.call_param_count,
            call_result_count: current.call_result_count,
            call_stack_push,
            call_stack_pop,
        });
        param_init_state = param_init_after;
        if matches!(current.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect) && !callee_initial_params.is_empty() {
            let param_count = callee_initial_params.len();
            let callee_function_ref = current.function_ref.ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing callee function ref for guest call at cycle {}",
                    current.cycle
                ))
            })?;
            let callee_fbp = guest_callee_fbp.ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing callee frame base for guest call at cycle {}",
                    current.cycle
                ))
            })?;
            for (param_index, (dst_addr, value)) in callee_initial_params.into_iter().enumerate() {
                let expected_dst_addr = callee_fbp.checked_add(param_index as u64).ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "callee local address overflow at call cycle {} param {}",
                        current.cycle, param_index
                    ))
                })?;
                if dst_addr != expected_dst_addr {
                    return Err(WasmBuildError::Trace(format!(
                        "callee local address mismatch at call cycle {} param {}: expected {}, got {}",
                        current.cycle, param_index, expected_dst_addr, dst_addr
                    )));
                }
                let param_index_u32 = u32::try_from(param_index).map_err(|_| {
                    WasmBuildError::Trace(format!("call parameter index {param_index} does not fit u32"))
                })?;
                let Some(src) = read_lane(&current.operand_stack, sp_before, stack_reads, param_index) else {
                    return Err(WasmBuildError::Trace(format!(
                        "missing call argument lane {param_index} at cycle {}",
                        current.cycle
                    )));
                };
                let remaining_before = param_count - param_index;
                let remaining_after = remaining_before - 1;
                let remaining_after_u32 = u32::try_from(remaining_after).map_err(|_| {
                    WasmBuildError::Trace(format!(
                        "remaining call parameter count does not fit u32 at cycle {}",
                        current.cycle
                    ))
                })?;
                let aux_param_init_before = param_init_state;
                let aux_param_init_after = WasmParamInitState {
                    active: remaining_after != 0,
                    remaining: remaining_after_u32,
                };
                out.push(WasmStepTrace {
                    cycle: out.len() as u64,
                    row_kind: super::super::ir::WasmRowKind::Aux(super::super::ir::WasmAuxOpcode::CallParamInit),
                    pc_before: pc_after,
                    pc_after,
                    control_choice: 0,
                    pc_edge_kind: WasmPcEdgeKind::Static,
                    param_init_before: aux_param_init_before,
                    param_init_after: aux_param_init_after,
                    wide_values_enabled: read_lane_hi(&current.operand_stack_hi, stack_reads, param_index)
                        .is_some_and(|hi| hi != 0),
                    opcode_code: 0,
                    opcode: WasmOpcode::Nop,
                    info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
                    stack_reads_override: Some(0),
                    stack_writes_override: Some(0),
                    sp_before: sp_after,
                    sp_after,
                    current_function_ref: callee_function_ref,
                    current_function_num_locals: current.num_locals,
                    stack_read0: Some(src),
                    stack_read0_hi: read_lane_hi(&current.operand_stack_hi, stack_reads, param_index),
                    stack_read1: None,
                    stack_read1_hi: None,
                    stack_read2: None,
                    stack_read2_hi: None,
                    stack_write0: None,
                    stack_write0_hi: None,
                    linear_memory: None,
                    linear_memory_offset: 0,
                    memory_pages_before: current.memory_pages_after,
                    memory_pages_after: current.memory_pages_after,
                    halted: false,
                    locals_fbp: callee_fbp,
                    locals_fbp_after: callee_fbp,
                    local_index: Some(param_index_u32),
                    local_read_value: None,
                    local_read_value_hi: None,
                    local_write_value: Some(value),
                    local_write_value_hi: read_lane_hi(&current.operand_stack_hi, stack_reads, param_index),
                    global_index: None,
                    global_read_value: None,
                    global_read_value_hi: None,
                    global_write_value: None,
                    global_write_value_hi: None,
                    table_id: None,
                    table_index: None,
                    table_value: None,
                    function_ref: None,
                    target_function_is_guest: false,
                    function_type_id: None,
                    call_indirect_type_index: None,
                    expected_type_id: None,
                    table_size: None,
                    call_param_count: current.call_param_count,
                    call_result_count: None,
                    call_stack_push: None,
                    call_stack_pop: None,
                });
                debug_assert_eq!(
                    aux_param_init_before.remaining,
                    u32::try_from(remaining_before).expect("remaining fits u32")
                );
                param_init_state = aux_param_init_after;
            }
        }
    }

    if out.is_empty() {
        return Err(WasmBuildError::Unsupported(
            "wasmtime trace did not contain any currently supported wasm rows".to_string(),
        ));
    }

    Ok(out)
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

/// Extracts initial parameter locals from the callee's first step's locals snapshot,
/// converting local indices to absolute addresses using the callee's FBP.
fn collect_callee_initial_params(next: Option<&SupportedRow>, callee_fbp: u64, param_count: u8) -> Vec<(u64, u32)> {
    let Some(next) = next else {
        return vec![];
    };
    next.locals_snapshot
        .iter()
        .take(usize::from(param_count))
        .enumerate()
        .map(|(i, &v)| (callee_fbp + i as u64, v))
        .collect()
}

#[derive(Clone, Debug)]
struct SupportedRow {
    cycle: u64,
    pc: u32,
    control_choice: u32,
    pc_edge_kind: WasmPcEdgeKind,
    opcode: WasmOpcode,
    info: super::super::isa::WasmOpcodeInfo,
    wide_values_enabled: bool,
    stack_reads_override: Option<u8>,
    stack_writes_override: Option<u8>,
    operand_stack: Vec<u32>,
    operand_stack_hi: Vec<u32>,
    immediate_i32: Option<u32>,
    /// For local.get / local.set / local.tee: the 0-based local index.
    local_index: Option<u32>,
    /// For local.get: the value of local[local_index] before this step executes
    /// (captured from the wasmtime frame's locals snapshot).
    local_value: Option<u32>,
    local_value_hi: Option<u32>,
    /// For global.get / global.set: the 0-based global index.
    global_index: Option<u32>,
    /// For global.get / global.set: value of the global before this step.
    global_value_before: Option<u32>,
    global_value_before_hi: Option<u32>,
    /// For global.set: value written into the global this step.
    global_value_after: Option<u32>,
    global_value_after_hi: Option<u32>,
    table_id: Option<u32>,
    table_index: Option<u32>,
    table_value: Option<u32>,
    table_size: Option<u32>,
    function_ref: Option<u32>,
    current_function_ref: Option<u32>,
    target_function_is_guest: bool,
    function_type_id: Option<u32>,
    call_param_count: Option<u8>,
    call_result_count: Option<u8>,
    call_indirect_type_index: Option<u32>,
    expected_type_id: Option<u32>,
    memory_pages_before: Option<u32>,
    memory_pages_after: Option<u32>,
    /// For `call` instructions: binary offset of the instruction after the call (= return address).
    call_return_pc: Option<u64>,
    /// Total number of locals (params + declared) in this frame at this step.
    num_locals: u32,
    /// Parsed local values at this step (before execution). Used to build aux param-init rows at
    /// call boundaries.
    locals_snapshot: Vec<u32>,
    linear_memory: Option<LinearMemoryAccess>,
    linear_memory_offset: u64,
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

fn normalize_supported_row(row: &WasmtimeTraceStep) -> Result<Option<SupportedRow>, WasmBuildError> {
    if row.frame_depth != 0 {
        return Ok(None);
    }
    let Some(pc) = row.pc else {
        return Ok(None);
    };

    let (opcode, immediate_i32) = match row.opcode_decoded {
        Some(op) => (op, row.immediate_i32),
        None => return Ok(None),
    };

    if matches!(opcode, WasmOpcode::Trap | WasmOpcode::Unsupported) {
        return Ok(None);
    }

    let operand_stack = row.operand_stack_words.clone();
    let operand_stack_hi = row.operand_stack_words_hi.clone();
    let code = opcode_code(opcode);
    let (stack_reads_override, stack_writes_override) = match opcode {
        WasmOpcode::Call => row.call_param_count.map(|params| (Some(params), Some(0))),
        WasmOpcode::CallIndirect => row
            .call_param_count
            .map(|params| (Some(params.saturating_add(1)), Some(0))),
        _ => Some((None, None)),
    }
    .unwrap_or((None, None));

    // For local.get / local.set / local.tee the immediate holds the local index.
    // The frame's locals snapshot (captured before execution) gives the pre-step value.
    let local_index = match opcode {
        WasmOpcode::LocalGet | WasmOpcode::LocalSet | WasmOpcode::LocalTee => immediate_i32,
        _ => None,
    };
    let local_value = local_index.and_then(|idx| {
        row.locals
            .get(idx as usize)
            .and_then(|v| parse_stack_word(v).ok())
    });
    let local_value_hi = local_index.and_then(|idx| row.locals_words_hi.get(idx as usize).copied());
    let global_index = match opcode {
        WasmOpcode::GlobalGet | WasmOpcode::GlobalSet => row.global_index,
        _ => None,
    };
    let table_id = match opcode {
        WasmOpcode::TableSize | WasmOpcode::TableGet | WasmOpcode::TableSet | WasmOpcode::CallIndirect => row.table_id,
        _ => None,
    };
    let memory_pages_before = row.memory_pages_before;
    let memory_pages_after = row.memory_pages_after;
    let control_choice = row.control_choice.unwrap_or(0);
    let pc_edge_kind = row.pc_edge_kind.ok_or_else(|| {
        WasmBuildError::Trace(format!(
            "supported Wasmtime row at step {} is missing pc_edge_kind",
            row.step
        ))
    })?;
    let linear_memory = match opcode {
        WasmOpcode::I32Load
        | WasmOpcode::I64Load
        | WasmOpcode::I32Load8S
        | WasmOpcode::I32Load8U
        | WasmOpcode::I32Load16S
        | WasmOpcode::I32Load16U
        | WasmOpcode::I32Store
        | WasmOpcode::I64Store
        | WasmOpcode::I32Store8
        | WasmOpcode::I32Store16 => {
            let memory = row.memory.as_ref().ok_or_else(|| {
                WasmBuildError::Trace(format!("missing wasmtime memory access for opcode {}", opcode.name()))
            })?;
            if memory.memory_index != 0 {
                return Err(WasmBuildError::Unsupported(format!(
                    "multiple memories are not supported yet: memory_index={}",
                    memory.memory_index
                )));
            }
            let lane0 = LinearMemoryWordLane {
                word_addr: memory.lane0.word_addr,
                value_before: memory.lane0.value_before,
                value_after: memory.lane0.value_after,
            };

            let lane1 = memory.lane1.map(|lane| LinearMemoryWordLane {
                word_addr: lane.word_addr,
                value_before: lane.value_before,
                value_after: lane.value_after,
            });
            let lane2 = memory.lane2.map(|lane| LinearMemoryWordLane {
                word_addr: lane.word_addr,
                value_before: lane.value_before,
                value_after: lane.value_after,
            });
            let access = LinearMemoryAccess {
                width_bytes: memory.width_bytes,
                byte_offset: memory.byte_offset,
                lane0,
                lane1,
                lane2,
            };
            Some(access)
        }
        _ => None,
    };
    let locals_snapshot: Vec<u32> = row
        .locals
        .iter()
        .map(|v| v.parse::<i128>().map(|n| (n as i32) as u32).unwrap_or(0))
        .collect();

    Ok(Some(SupportedRow {
        cycle: row.step,
        pc,
        control_choice,
        pc_edge_kind,
        opcode,
        info: opcode_info_from_code(code),
        wide_values_enabled: matches!(
            opcode,
            WasmOpcode::I64Const
                | WasmOpcode::I64Add
                | WasmOpcode::I64Sub
                | WasmOpcode::I64Eqz
                | WasmOpcode::I64Eq
                | WasmOpcode::I64Ne
                | WasmOpcode::I64And
                | WasmOpcode::I64Or
                | WasmOpcode::I64Xor
                | WasmOpcode::I64Mul
                | WasmOpcode::I64Load
                | WasmOpcode::I64Store
        ),
        stack_reads_override,
        stack_writes_override,
        operand_stack,
        operand_stack_hi,
        immediate_i32,
        local_index,
        local_value,
        local_value_hi,
        global_index,
        global_value_before: row.global_value_before,
        global_value_before_hi: row.global_value_before_hi,
        global_value_after: row.global_value_after,
        global_value_after_hi: row.global_value_after_hi,
        table_id,
        table_index: row.table_index,
        table_value: row.table_value,
        table_size: row.table_size,
        function_ref: row.function_ref,
        current_function_ref: row.current_function_ref,
        target_function_is_guest: row.target_function_is_guest,
        function_type_id: row.function_type_id,
        call_param_count: row.call_param_count,
        call_result_count: row.call_result_count,
        call_indirect_type_index: row.call_indirect_type_index,
        expected_type_id: row.expected_type_id,
        memory_pages_before,
        memory_pages_after,
        call_return_pc: row.call_return_pc,
        num_locals: row.num_locals,
        locals_snapshot,
        linear_memory,
        linear_memory_offset: row.memory.as_ref().map(|memory| memory.offset).unwrap_or(0),
    }))
}

fn capture_frame(
    step: u64,
    frame_depth: usize,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<WasmtimeTraceStep, WasmBuildError> {
    let (function, function_index, pc) = match frame
        .wasm_function_index_and_pc(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame function/pc: {err}")))?
    {
        Some((func_index, pc)) => {
            let function_index = func_index.as_u32();
            (format!("{func_index:?}"), Some(function_index), Some(pc))
        }
        None => ("<host-or-unknown>".to_string(), None, None),
    };
    let decoded_opcode = function_index
        .zip(pc)
        .and_then(|key| store.data().opcode_map.get(&key).cloned());
    let current_function_ref = function_index.and_then(|index| {
        store
            .data()
            .imported_function_count
            .checked_add(index)
            .and_then(|function_ref| function_ref.checked_add(1))
    });
    let opcode = decoded_opcode.as_ref().map(|decoded| decoded.text.clone());
    let (opcode_decoded, immediate_i32) = decoded_opcode
        .as_ref()
        .and_then(|d| d.decoded)
        .map_or((None, None), |(op, imm)| (Some(op), imm));

    let num_locals = frame
        .num_locals(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime locals length: {err}")))?;
    let func_ref_ids = store.data().func_ref_ids.clone();
    let mut locals = Vec::with_capacity(num_locals as usize);
    let mut locals_words_hi = Vec::with_capacity(num_locals as usize);
    for index in 0..num_locals {
        let value = frame
            .local(&mut *store, index)
            .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime local {index}: {err}")))?;
        locals.push(val_to_string(value));
        let (_, hi) = normalize_value_lanes(value, func_ref_ids.as_ref(), &mut *store)?;
        locals_words_hi.push(hi);
    }

    let num_stacks = frame
        .num_stacks(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime operand stack length: {err}")))?;
    let mut operand_stack = Vec::with_capacity(num_stacks as usize);
    let mut operand_stack_words = Vec::with_capacity(num_stacks as usize);
    let mut operand_stack_words_hi = Vec::with_capacity(num_stacks as usize);
    for index in 0..num_stacks {
        let value = frame.stack(&mut *store, index).map_err(|err| {
            WasmBuildError::Trace(format!("failed to inspect Wasmtime operand stack value {index}: {err}"))
        })?;
        operand_stack.push(val_to_string(value));
        let (lo, hi) = normalize_value_lanes(value, func_ref_ids.as_ref(), &mut *store)?;
        operand_stack_words.push(lo);
        operand_stack_words_hi.push(hi);
    }
    let global_index = match opcode_decoded {
        Some(WasmOpcode::GlobalGet | WasmOpcode::GlobalSet) => immediate_i32,
        _ => None,
    };
    let table_id = match opcode_decoded {
        Some(WasmOpcode::TableSize | WasmOpcode::TableGet | WasmOpcode::TableSet | WasmOpcode::CallIndirect) => {
            immediate_i32
        }
        _ => None,
    };
    let memory_pages_now = read_memory_pages_if_present(0, frame, store)?;
    let (global_value_before, global_value_before_hi) = match global_index {
        Some(index) => {
            let (lo, hi) = read_global_lanes(index, frame, store)?;
            (Some(lo), Some(hi))
        }
        None => (None, None),
    };
    let (global_value_after, global_value_after_hi) = match opcode_decoded {
        Some(WasmOpcode::GlobalSet) => (
            operand_stack_words.last().copied(),
            operand_stack_words_hi.last().copied(),
        ),
        _ => (None, None),
    };
    let table_size = match table_id {
        Some(table_id) => {
            let size = read_table_size(table_id, frame, store)?;
            Some(size)
        }
        None => None,
    };
    let table_index = match opcode_decoded {
        Some(WasmOpcode::TableGet) => operand_stack_words.last().copied(),
        Some(WasmOpcode::TableSet) => operand_stack_words
            .get(operand_stack_words.len().saturating_sub(2))
            .copied(),
        Some(WasmOpcode::CallIndirect) => operand_stack_words.last().copied(),
        _ => None,
    };
    let table_value = match opcode_decoded {
        Some(WasmOpcode::TableGet) => match (table_id, table_index) {
            (Some(table_id), Some(table_index)) => Some(read_table_funcref_u32(table_id, table_index, frame, store)?),
            _ => None,
        },
        Some(WasmOpcode::TableSet) => operand_stack_words.last().copied(),
        Some(WasmOpcode::CallIndirect) => match (table_id, table_index) {
            (Some(table_id), Some(table_index)) => Some(read_table_funcref_u32(table_id, table_index, frame, store)?),
            _ => None,
        },
        _ => None,
    };
    let function_type_id = match opcode_decoded {
        Some(WasmOpcode::RefFunc) => {
            immediate_i32.and_then(|function_ref| function_type_id_from_ref(function_ref, store))
        }
        Some(WasmOpcode::TableGet | WasmOpcode::TableSet | WasmOpcode::CallIndirect) => {
            table_value.and_then(|function_ref| function_type_id_from_ref(function_ref, store))
        }
        Some(WasmOpcode::Call) => immediate_i32
            .and_then(|function_index| function_index.checked_add(1))
            .and_then(|function_ref| function_type_id_from_ref(function_ref, store)),
        _ => None,
    };
    let function_ref = match opcode_decoded {
        Some(WasmOpcode::Call) => immediate_i32.and_then(|function_index| function_index.checked_add(1)),
        Some(WasmOpcode::CallIndirect) => table_value,
        Some(WasmOpcode::RefFunc) => immediate_i32,
        Some(WasmOpcode::TableGet | WasmOpcode::TableSet) => table_value,
        _ => None,
    };
    let (call_param_count, call_result_count) = match opcode_decoded {
        Some(WasmOpcode::Call) => immediate_i32
            .and_then(|function_index| function_index.checked_add(1))
            .and_then(|function_ref| function_arity_from_ref(function_ref, store))
            .map_or((None, None), |(params, results)| (Some(params), Some(results))),
        Some(WasmOpcode::CallIndirect) => table_value
            .and_then(|function_ref| function_arity_from_ref(function_ref, store))
            .map_or((None, None), |(params, results)| (Some(params), Some(results))),
        _ => (None, None),
    };
    let call_indirect_type_index = match opcode_decoded {
        Some(WasmOpcode::CallIndirect) => decoded_opcode
            .as_ref()
            .and_then(|d| d.call_indirect_type_index),
        _ => None,
    };
    let (memory_pages_before, memory_pages_after) = match opcode_decoded {
        Some(WasmOpcode::MemorySize) => {
            let pages = memory_pages_now.ok_or_else(|| {
                WasmBuildError::Trace("memory.size observed without memory 0 in current frame".to_string())
            })?;
            (Some(pages), Some(pages))
        }
        Some(WasmOpcode::MemoryGrow) => {
            let pages_before = memory_pages_now.ok_or_else(|| {
                WasmBuildError::Trace("memory.grow observed without memory 0 in current frame".to_string())
            })?;
            let delta = operand_stack_words.last().copied().unwrap_or(0);
            let pages_after = delta.checked_add(pages_before).unwrap_or(pages_before);
            (Some(pages_before), Some(pages_after))
        }
        _ => (memory_pages_now, memory_pages_now),
    };
    let control_choice =
        match (opcode_decoded, decoded_opcode.as_ref().and_then(|d| d.control)) {
            (Some(WasmOpcode::If), _) | (Some(WasmOpcode::BrIf), _) => operand_stack_words
                .last()
                .copied()
                .map(|cond| if cond == 0 { 0 } else { 1 }),
            (Some(WasmOpcode::BrTable), Some(DecodedControlOpcode::BrTable { len })) => operand_stack_words
                .last()
                .copied()
                .map(|index| if index < len { index + 1 } else { 0 }),
            _ => None,
        };
    let memory = capture_memory_access(
        decoded_opcode.as_ref(),
        frame,
        store,
        &operand_stack_words,
        &operand_stack_words_hi,
    )?;

    Ok(WasmtimeTraceStep {
        step,
        frame_depth,
        function,
        function_index,
        pc,
        opcode,
        opcode_decoded,
        immediate_i32,
        control_choice,
        pc_edge_kind: decoded_opcode.as_ref().map(|d| d.pc_edge_kind),
        global_index,
        global_value_before,
        global_value_before_hi,
        global_value_after,
        global_value_after_hi,
        table_id,
        table_index,
        table_value,
        table_size,
        function_ref,
        current_function_ref,
        target_function_is_guest: function_ref
            .is_some_and(|function_ref| function_ref > store.data().imported_function_count),
        function_type_id,
        call_indirect_type_index,
        expected_type_id: decoded_opcode.as_ref().and_then(|d| d.expected_type_id),
        call_param_count,
        call_result_count,
        memory_pages_before,
        memory_pages_after,
        memory,
        locals,
        locals_words_hi,
        operand_stack,
        operand_stack_words,
        operand_stack_words_hi,
        num_locals: num_locals as u32,
        call_return_pc: decoded_opcode.as_ref().and_then(|d| d.call_return_pc),
    })
}

/// Build the next-PC ROM from validated WASM bytecode using statically resolved control edges.
///
/// Conditional control rows (`if`, `br_if`) emit both possible successors. Structural rows
/// (`block`, `loop`, `else`, inner `end`) are kept as explicit ROM entries so the trace can
/// carry them as regular witness rows.
pub fn build_pc_rom_from_binary(wasm_bytes: &[u8]) -> Result<Vec<(u64, u64, u64)>, WasmBuildError> {
    Ok(parse_wasm_artifacts(wasm_bytes)?.pc_rom)
}

fn parse_wasm_artifacts(wasm_bytes: &[u8]) -> Result<ParsedWasmArtifacts, WasmBuildError> {
    let mut builder = ParsedWasmArtifactsBuilder::default();
    for payload in Parser::new(0).parse_all(wasm_bytes) {
        let payload = payload.map_err(|err| WasmBuildError::Trace(format!("failed to parse wasm payload: {err}")))?;
        builder.consume_payload(payload)?;
    }
    builder.finish()
}

fn parse_first_component_core_module_artifacts(component_bytes: &[u8]) -> Result<ParsedWasmArtifacts, WasmBuildError> {
    let mut builder = ParsedWasmArtifactsBuilder::default();
    let mut inside_first_module = false;

    for payload in Parser::new(0).parse_all(component_bytes) {
        let payload =
            payload.map_err(|err| WasmBuildError::Trace(format!("failed to parse component payload: {err}")))?;
        match payload {
            // Current component tests execute the first embedded core module directly. Multi-module
            // components need an explicit resolution step for the actually executed core module.
            Payload::ModuleSection { .. } if !inside_first_module => inside_first_module = true,
            Payload::End(_) if inside_first_module => return builder.finish(),
            _ if inside_first_module => builder.consume_payload(payload)?,
            _ => {}
        }
    }

    Err(WasmBuildError::Unsupported(
        "component did not contain any embedded core module sections".to_string(),
    ))
}

struct ParsedWasmArtifactsBuilder {
    opcode_map: BTreeMap<(u32, u32), DecodedOpcode>,
    pc_rom: Vec<(u64, u64, u64)>,
    pc_edge_kinds: Vec<(u64, u64)>,
    pc_function_refs: Vec<(u64, u64)>,
    unresolved_call_edges: Vec<(u64, u32)>,
    function_entries: Vec<(u64, u64)>,
    call_targets: Vec<(u64, u64)>,
    function_metas: BTreeMap<u32, ParsedFunctionMeta>,
    defined_function_index: u32,
    imported_function_count: u32,
    raw_type_id_by_index: BTreeMap<u32, u32>,
    raw_type_shape_by_index: BTreeMap<u32, (u8, u8)>,
    signature_ids: BTreeMap<String, u32>,
    next_type_id: u32,
    defined_function_type_indices: Vec<u32>,
}

impl Default for ParsedWasmArtifactsBuilder {
    fn default() -> Self {
        Self {
            opcode_map: BTreeMap::new(),
            pc_rom: Vec::new(),
            pc_edge_kinds: Vec::new(),
            pc_function_refs: Vec::new(),
            unresolved_call_edges: Vec::new(),
            function_entries: Vec::new(),
            call_targets: Vec::new(),
            function_metas: BTreeMap::new(),
            defined_function_index: 0,
            imported_function_count: 0,
            raw_type_id_by_index: BTreeMap::new(),
            raw_type_shape_by_index: BTreeMap::new(),
            signature_ids: BTreeMap::new(),
            next_type_id: 1,
            defined_function_type_indices: Vec::new(),
        }
    }
}

impl ParsedWasmArtifactsBuilder {
    fn push_pc_rom_edge(&mut self, pc_before: u64, control_choice: u64, pc_after: u64) {
        self.pc_rom.push((pc_before, control_choice, pc_after));
    }

    fn finish(mut self) -> Result<ParsedWasmArtifacts, WasmBuildError> {
        let unresolved_call_edges = std::mem::take(&mut self.unresolved_call_edges);
        for (pc_before, function_ref) in unresolved_call_edges {
            let entry_pc = self
                .function_metas
                .get(&function_ref)
                .and_then(|meta| meta.entry_pc)
                .ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing entry pc for direct call target function ref {function_ref}"
                    ))
                })?;
            self.push_pc_rom_edge(pc_before, 0, entry_pc);
        }
        self.pc_rom.sort_unstable();
        self.pc_rom.dedup();
        self.pc_edge_kinds.sort_unstable();
        self.pc_edge_kinds.dedup();
        self.pc_function_refs.sort_unstable();
        self.pc_function_refs.dedup();
        self.function_entries.sort_unstable();
        self.function_entries.dedup();
        self.call_targets.sort_unstable();
        self.call_targets.dedup();
        let mut function_types = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.type_id)))
            .collect::<Vec<_>>();
        function_types.sort_unstable();
        function_types.dedup();
        let mut function_param_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.param_count)))
            .collect::<Vec<_>>();
        function_param_counts.sort_unstable();
        function_param_counts.dedup();
        let mut function_result_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.result_count)))
            .collect::<Vec<_>>();
        function_result_counts.sort_unstable();
        function_result_counts.dedup();
        let mut function_local_counts = self
            .function_metas
            .iter()
            .map(|(&function_ref, meta)| (u64::from(function_ref), u64::from(meta.num_locals)))
            .collect::<Vec<_>>();
        function_local_counts.sort_unstable();
        function_local_counts.dedup();
        let mut function_guest_flags = self
            .function_metas
            .keys()
            .map(|&function_ref| {
                let is_guest = u64::from(function_ref > self.imported_function_count);
                (u64::from(function_ref), is_guest)
            })
            .collect::<Vec<_>>();
        function_guest_flags.sort_unstable();
        function_guest_flags.dedup();
        let mut module_types = self
            .raw_type_id_by_index
            .iter()
            .map(|(&raw_type_index, &type_id)| (u64::from(raw_type_index), u64::from(type_id)))
            .collect::<Vec<_>>();
        module_types.sort_unstable();
        module_types.dedup();
        Ok(ParsedWasmArtifacts {
            opcode_map: self.opcode_map,
            pc_rom: self.pc_rom,
            pc_edge_kinds: self.pc_edge_kinds,
            pc_function_refs: self.pc_function_refs,
            function_entries: self.function_entries,
            function_types,
            function_param_counts,
            function_result_counts,
            function_local_counts,
            function_guest_flags,
            call_targets: self.call_targets,
            module_types,
            imported_function_count: self.imported_function_count,
            function_metas: self.function_metas,
        })
    }

    fn consume_payload(&mut self, payload: Payload<'_>) -> Result<(), WasmBuildError> {
        match payload {
            Payload::TypeSection(reader) => {
                for (raw_type_index, func_type_result) in reader.into_iter_err_on_gc_types().enumerate() {
                    let func_type = func_type_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm func type: {err}")))?;
                    let signature = canonical_wasmparser_func_signature(&func_type);
                    let type_id = *self.signature_ids.entry(signature).or_insert_with(|| {
                        let assigned = self.next_type_id;
                        self.next_type_id = self.next_type_id.saturating_add(1);
                        assigned
                    });
                    self.raw_type_id_by_index
                        .insert(raw_type_index as u32, type_id);
                    self.raw_type_shape_by_index.insert(
                        raw_type_index as u32,
                        (func_type.params().len() as u8, func_type.results().len() as u8),
                    );
                }
            }
            Payload::ImportSection(reader) => {
                for import_result in reader {
                    let import = import_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm import: {err}")))?;
                    if let wasmparser::TypeRef::Func(raw_type_index) = import.ty {
                        let function_ref = self.imported_function_count.saturating_add(1);
                        self.imported_function_count = self.imported_function_count.saturating_add(1);
                        let type_id = *self
                            .raw_type_id_by_index
                            .get(&raw_type_index)
                            .ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing normalized type id for imported function type {raw_type_index}"
                                ))
                            })?;
                        let (param_count, result_count) = *self
                            .raw_type_shape_by_index
                            .get(&raw_type_index)
                            .ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing raw type shape for imported function type {raw_type_index}"
                                ))
                            })?;
                        self.function_metas.insert(
                            function_ref,
                            ParsedFunctionMeta {
                                type_id,
                                param_count,
                                result_count,
                                num_locals: u32::from(param_count),
                                entry_pc: None,
                            },
                        );
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for raw_type_index_result in reader {
                    let raw_type_index = raw_type_index_result.map_err(|err| {
                        WasmBuildError::Trace(format!("failed to decode wasm function type index: {err}"))
                    })?;
                    self.defined_function_type_indices.push(raw_type_index);
                }
            }
            Payload::CodeSectionEntry(body) => {
                let function_ref = self
                    .imported_function_count
                    .checked_add(self.defined_function_index)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| WasmBuildError::Trace("function ref id overflow".to_string()))?;
                let raw_type_index = *self
                    .defined_function_type_indices
                    .get(self.defined_function_index as usize)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing type index for defined function {}",
                            self.defined_function_index
                        ))
                    })?;
                let (param_count, result_count) = *self
                    .raw_type_shape_by_index
                    .get(&raw_type_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing raw type shape for defined function type {raw_type_index}"
                        ))
                    })?;
                let mut declared_local_count = 0_u32;
                let locals_reader = body
                    .get_locals_reader()
                    .map_err(|err| WasmBuildError::Trace(format!("failed to read wasm locals: {err}")))?;
                for local_result in locals_reader {
                    let (count, _) = local_result
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm local decl: {err}")))?;
                    declared_local_count = declared_local_count
                        .checked_add(count)
                        .ok_or_else(|| WasmBuildError::Trace("declared local count overflow".to_string()))?;
                }
                let num_locals = u32::from(param_count)
                    .checked_add(declared_local_count)
                    .ok_or_else(|| WasmBuildError::Trace("total local count overflow".to_string()))?;
                let mut reader = body
                    .get_operators_reader()
                    .map_err(|err| WasmBuildError::Trace(format!("failed to read wasm operators: {err}")))?;
                let mut curr_depth = 0_usize;
                let mut control_stack: Vec<ControlFrame> = Vec::new();
                let mut entry_pc = None;
                while !reader.eof() {
                    let offset = reader.original_position() as u32;
                    if entry_pc.is_none() {
                        entry_pc = Some(u64::from(offset));
                    }
                    let pc_before = u64::from(offset);
                    let operator = reader
                        .read()
                        .map_err(|err| WasmBuildError::Trace(format!("failed to decode wasm operator: {err}")))?;
                    let pc_after = reader.original_position() as u64;
                    let is_function_end = matches!(&operator, wasmparser::Operator::End) && curr_depth == 0;
                    let decoded = match &operator {
                        wasmparser::Operator::Loop { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::Loop, None))
                        }
                        wasmparser::Operator::Block { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::Block, None))
                        }
                        wasmparser::Operator::If { .. } => {
                            curr_depth += 1;
                            Some((WasmOpcode::If, None))
                        }
                        wasmparser::Operator::Else => Some((WasmOpcode::Else, None)),
                        wasmparser::Operator::Unreachable => Some((WasmOpcode::Unreachable, None)),
                        wasmparser::Operator::Return => Some((WasmOpcode::Return, None)),
                        wasmparser::Operator::End => {
                            if curr_depth == 0 {
                                Some((WasmOpcode::End, None))
                            } else {
                                curr_depth -= 1;
                                Some((WasmOpcode::End, None))
                            }
                        }
                        _ => decode_opcode(&operator),
                    };
                    let call_return_pc = match &operator {
                        wasmparser::Operator::Call { .. } | wasmparser::Operator::CallIndirect { .. } => {
                            Some(reader.original_position() as u64)
                        }
                        _ => None,
                    };
                    let (call_indirect_type_index, expected_type_id) = match &operator {
                        wasmparser::Operator::CallIndirect { type_index, .. } => {
                            let expected_type_id = *self.raw_type_id_by_index.get(type_index).ok_or_else(|| {
                                WasmBuildError::Trace(format!(
                                    "missing normalized type id for call_indirect type {type_index}"
                                ))
                            })?;
                            (Some(*type_index), Some(expected_type_id))
                        }
                        _ => (None, None),
                    };
                    let pc_edge_kind = match &operator {
                        wasmparser::Operator::Return => WasmPcEdgeKind::ReturnLike,
                        wasmparser::Operator::End if is_function_end => WasmPcEdgeKind::ReturnLike,
                        wasmparser::Operator::CallIndirect { .. } => WasmPcEdgeKind::DynamicCallIndirect,
                        wasmparser::Operator::Unreachable => WasmPcEdgeKind::Terminal,
                        _ => WasmPcEdgeKind::Static,
                    };
                    self.pc_edge_kinds
                        .push((pc_before, u64::from(pc_edge_kind.as_u32())));
                    self.pc_function_refs
                        .push((pc_before, u64::from(function_ref)));
                    self.opcode_map.insert(
                        (self.defined_function_index, offset),
                        DecodedOpcode {
                            text: format!("{operator:?}"),
                            memory: decode_memory_opcode(&operator),
                            decoded,
                            control: decode_control_opcode(&operator),
                            pc_edge_kind,
                            call_indirect_type_index,
                            expected_type_id,
                            call_return_pc,
                        },
                    );

                    match &operator {
                        wasmparser::Operator::Loop { .. } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            control_stack.push(ControlFrame::new(ControlFrameKind::Loop, Some(pc_after)));
                        }
                        wasmparser::Operator::Block { .. } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            control_stack.push(ControlFrame::new(ControlFrameKind::Block, None));
                        }
                        wasmparser::Operator::If { .. } => {
                            self.push_pc_rom_edge(pc_before, 1, pc_after);
                            let mut frame = ControlFrame::new(ControlFrameKind::If, None);
                            frame.pending_if_false = Some(pc_before);
                            control_stack.push(frame);
                        }
                        wasmparser::Operator::Else => {
                            let frame = control_stack.last_mut().ok_or_else(|| {
                                WasmBuildError::Trace("encountered else without an open control frame".to_string())
                            })?;
                            if frame.kind != ControlFrameKind::If {
                                return Err(WasmBuildError::Trace(
                                    "encountered else outside an if frame".to_string(),
                                ));
                            }
                            if let Some(if_pc) = frame.pending_if_false.take() {
                                self.push_pc_rom_edge(if_pc, 0, pc_after);
                            }
                            frame.pending_to_end.push(pc_before);
                        }
                        wasmparser::Operator::End => {
                            if let Some(mut frame) = control_stack.pop() {
                                if let Some(if_pc) = frame.pending_if_false.take() {
                                    self.push_pc_rom_edge(if_pc, 0, pc_after);
                                }
                                for edge_pc in frame.pending_to_end.drain(..) {
                                    self.push_pc_rom_edge(edge_pc, 0, pc_after);
                                }
                                self.push_pc_rom_edge(pc_before, 0, pc_after);
                            }
                        }
                        wasmparser::Operator::BrIf { relative_depth } => {
                            self.push_pc_rom_edge(pc_before, 0, pc_after);
                            let depth = *relative_depth as usize;
                            let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br_if depth {depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 1, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::Br { relative_depth } => {
                            let depth = *relative_depth as usize;
                            let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br depth {depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 0, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::BrTable { targets } => {
                            for (choice_index, depth_result) in targets.targets().enumerate() {
                                let depth = depth_result.map_err(|err| {
                                    WasmBuildError::Trace(format!("failed to decode br_table target: {err}"))
                                })? as usize;
                                let Some(target_index) = control_stack.len().checked_sub(depth + 1) else {
                                    return Err(WasmBuildError::Trace(format!(
                                        "br_table depth {depth} exceeded control nesting"
                                    )));
                                };
                                let target_frame = &mut control_stack[target_index];
                                if let Some(target) = target_frame.branch_target {
                                    self.push_pc_rom_edge(pc_before, choice_index as u64 + 1, target);
                                } else {
                                    target_frame.pending_to_end.push(pc_before);
                                }
                            }
                            let default_depth = targets.default() as usize;
                            let Some(target_index) = control_stack.len().checked_sub(default_depth + 1) else {
                                return Err(WasmBuildError::Trace(format!(
                                    "br_table default depth {default_depth} exceeded control nesting"
                                )));
                            };
                            let target_frame = &mut control_stack[target_index];
                            if let Some(target) = target_frame.branch_target {
                                self.push_pc_rom_edge(pc_before, 0, target);
                            } else {
                                target_frame.pending_to_end.push(pc_before);
                            }
                        }
                        wasmparser::Operator::Call { function_index } => {
                            let function_ref = function_index.saturating_add(1);
                            self.call_targets.push((pc_before, u64::from(function_ref)));
                            if function_ref <= self.imported_function_count {
                                self.push_pc_rom_edge(pc_before, 0, pc_after);
                            } else {
                                match self
                                    .function_metas
                                    .get(&function_ref)
                                    .and_then(|meta| meta.entry_pc)
                                {
                                    Some(entry_pc) => self.push_pc_rom_edge(pc_before, 0, entry_pc),
                                    None => self.unresolved_call_edges.push((pc_before, function_ref)),
                                }
                            }
                        }
                        wasmparser::Operator::Unreachable | wasmparser::Operator::Return => {}
                        _ => self.push_pc_rom_edge(pc_before, 0, pc_after),
                    }
                }
                let type_id = *self
                    .raw_type_id_by_index
                    .get(&raw_type_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing normalized type id for defined function type {raw_type_index}"
                        ))
                    })?;
                self.function_metas.insert(
                    function_ref,
                    ParsedFunctionMeta {
                        type_id,
                        param_count,
                        result_count,
                        num_locals,
                        entry_pc,
                    },
                );
                if let Some(entry_pc) = entry_pc {
                    self.function_entries
                        .push((u64::from(function_ref), entry_pc));
                }
                self.defined_function_index += 1;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Decode a `wasmparser::Operator` into a `(WasmOpcode, immediate)` pair.
/// Returns `None` for unsupported operators. Called once per instruction at
/// map-build time so the normalization path never parses strings.
fn decode_opcode(operator: &wasmparser::Operator<'_>) -> Option<(WasmOpcode, Option<u32>)> {
    match operator {
        wasmparser::Operator::Nop => Some((WasmOpcode::Nop, None)),
        wasmparser::Operator::I32Const { value } => Some((WasmOpcode::I32Const, Some(*value as u32))),
        wasmparser::Operator::I64Const { .. } => Some((WasmOpcode::I64Const, None)),
        wasmparser::Operator::RefFunc { function_index } => {
            Some((WasmOpcode::RefFunc, Some(function_index.saturating_add(1))))
        }
        wasmparser::Operator::I32Add => Some((WasmOpcode::I32Add, None)),
        wasmparser::Operator::I64Add => Some((WasmOpcode::I64Add, None)),
        wasmparser::Operator::I32Sub => Some((WasmOpcode::I32Sub, None)),
        wasmparser::Operator::I64Sub => Some((WasmOpcode::I64Sub, None)),
        wasmparser::Operator::I64Load { .. } => Some((WasmOpcode::I64Load, None)),
        wasmparser::Operator::I64Store { .. } => Some((WasmOpcode::I64Store, None)),
        wasmparser::Operator::I64And => Some((WasmOpcode::I64And, None)),
        wasmparser::Operator::I64Or => Some((WasmOpcode::I64Or, None)),
        wasmparser::Operator::I64Xor => Some((WasmOpcode::I64Xor, None)),
        wasmparser::Operator::I64Mul => Some((WasmOpcode::I64Mul, None)),
        wasmparser::Operator::I32Load { .. } => Some((WasmOpcode::I32Load, None)),
        wasmparser::Operator::I32Load8S { .. } => Some((WasmOpcode::I32Load8S, None)),
        wasmparser::Operator::I32Load8U { .. } => Some((WasmOpcode::I32Load8U, None)),
        wasmparser::Operator::I32Load16S { .. } => Some((WasmOpcode::I32Load16S, None)),
        wasmparser::Operator::I32Load16U { .. } => Some((WasmOpcode::I32Load16U, None)),
        wasmparser::Operator::I32Store { .. } => Some((WasmOpcode::I32Store, None)),
        wasmparser::Operator::I32Store8 { .. } => Some((WasmOpcode::I32Store8, None)),
        wasmparser::Operator::I32Store16 { .. } => Some((WasmOpcode::I32Store16, None)),
        wasmparser::Operator::MemorySize { .. } => Some((WasmOpcode::MemorySize, None)),
        wasmparser::Operator::MemoryGrow { .. } => Some((WasmOpcode::MemoryGrow, None)),
        wasmparser::Operator::TableSize { table } => Some((WasmOpcode::TableSize, Some(*table))),
        wasmparser::Operator::TableGet { table } => Some((WasmOpcode::TableGet, Some(*table))),
        wasmparser::Operator::TableSet { table } => Some((WasmOpcode::TableSet, Some(*table))),
        wasmparser::Operator::Drop => Some((WasmOpcode::Drop, None)),
        wasmparser::Operator::Br { .. } => Some((WasmOpcode::Br, None)),
        wasmparser::Operator::Block { .. } => Some((WasmOpcode::Block, None)),
        wasmparser::Operator::Loop { .. } => Some((WasmOpcode::Loop, None)),
        wasmparser::Operator::If { .. } => Some((WasmOpcode::If, None)),
        wasmparser::Operator::Else => Some((WasmOpcode::Else, None)),
        wasmparser::Operator::Unreachable => Some((WasmOpcode::Unreachable, None)),
        wasmparser::Operator::I32Clz => Some((WasmOpcode::I32Clz, None)),
        wasmparser::Operator::I32Ctz => Some((WasmOpcode::I32Ctz, None)),
        wasmparser::Operator::I32Popcnt => Some((WasmOpcode::I32Popcnt, None)),
        wasmparser::Operator::I32Eqz => Some((WasmOpcode::I32Eqz, None)),
        wasmparser::Operator::I64Eqz => Some((WasmOpcode::I64Eqz, None)),
        wasmparser::Operator::I32Eq => Some((WasmOpcode::I32Eq, None)),
        wasmparser::Operator::I32Ne => Some((WasmOpcode::I32Ne, None)),
        wasmparser::Operator::I64Eq => Some((WasmOpcode::I64Eq, None)),
        wasmparser::Operator::I64Ne => Some((WasmOpcode::I64Ne, None)),
        wasmparser::Operator::I32LtS => Some((WasmOpcode::I32LtS, None)),
        wasmparser::Operator::I32LtU => Some((WasmOpcode::I32LtU, None)),
        wasmparser::Operator::I32GtS => Some((WasmOpcode::I32GtS, None)),
        wasmparser::Operator::I32GtU => Some((WasmOpcode::I32GtU, None)),
        wasmparser::Operator::I32LeS => Some((WasmOpcode::I32LeS, None)),
        wasmparser::Operator::I32LeU => Some((WasmOpcode::I32LeU, None)),
        wasmparser::Operator::I32GeS => Some((WasmOpcode::I32GeS, None)),
        wasmparser::Operator::I32GeU => Some((WasmOpcode::I32GeU, None)),
        wasmparser::Operator::I32And => Some((WasmOpcode::I32And, None)),
        wasmparser::Operator::I32Or => Some((WasmOpcode::I32Or, None)),
        wasmparser::Operator::I32Xor => Some((WasmOpcode::I32Xor, None)),
        wasmparser::Operator::I32Mul => Some((WasmOpcode::I32Mul, None)),
        wasmparser::Operator::I32Shl => Some((WasmOpcode::I32Shl, None)),
        wasmparser::Operator::I32ShrU => Some((WasmOpcode::I32ShrU, None)),
        wasmparser::Operator::I32ShrS => Some((WasmOpcode::I32ShrS, None)),
        wasmparser::Operator::I32Rotl => Some((WasmOpcode::I32Rotl, None)),
        wasmparser::Operator::I32Rotr => Some((WasmOpcode::I32Rotr, None)),
        wasmparser::Operator::I32DivU => Some((WasmOpcode::I32DivU, None)),
        wasmparser::Operator::I32DivS => Some((WasmOpcode::I32DivS, None)),
        wasmparser::Operator::I32RemU => Some((WasmOpcode::I32RemU, None)),
        wasmparser::Operator::I32RemS => Some((WasmOpcode::I32RemS, None)),
        wasmparser::Operator::Select => Some((WasmOpcode::Select, None)),
        wasmparser::Operator::TypedSelect { .. } => Some((WasmOpcode::Select, None)),
        wasmparser::Operator::BrIf { .. } => Some((WasmOpcode::BrIf, None)),
        wasmparser::Operator::BrTable { .. } => Some((WasmOpcode::BrTable, None)),
        wasmparser::Operator::Return => Some((WasmOpcode::Return, None)),
        // Local index stored as immediate; locals value captured at runtime from frame snapshot.
        wasmparser::Operator::LocalGet { local_index } => Some((WasmOpcode::LocalGet, Some(*local_index))),
        wasmparser::Operator::LocalSet { local_index } => Some((WasmOpcode::LocalSet, Some(*local_index))),
        wasmparser::Operator::LocalTee { local_index } => Some((WasmOpcode::LocalTee, Some(*local_index))),
        wasmparser::Operator::GlobalGet { global_index } => Some((WasmOpcode::GlobalGet, Some(*global_index))),
        wasmparser::Operator::GlobalSet { global_index } => Some((WasmOpcode::GlobalSet, Some(*global_index))),
        wasmparser::Operator::Call { function_index } => Some((WasmOpcode::Call, Some(*function_index))),
        wasmparser::Operator::CallIndirect { table_index, .. } => Some((WasmOpcode::CallIndirect, Some(*table_index))),
        _ => None,
    }
}

fn decode_control_opcode(operator: &wasmparser::Operator<'_>) -> Option<DecodedControlOpcode> {
    match operator {
        wasmparser::Operator::BrTable { targets } => Some(DecodedControlOpcode::BrTable { len: targets.len() }),
        _ => None,
    }
}

fn decode_memory_opcode(operator: &wasmparser::Operator<'_>) -> Option<DecodedMemoryOpcode> {
    match operator {
        wasmparser::Operator::I32Load { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load8U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load8U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load8S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load8S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load16S { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load16S,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Load16U { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Load16U,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Load { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Load,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store8 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store8,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I32Store16 { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I32Store16,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        wasmparser::Operator::I64Store { memarg } => Some(DecodedMemoryOpcode {
            kind: DecodedMemoryAccessKind::I64Store,
            memory_index: memarg.memory,
            offset: memarg.offset,
        }),
        _ => None,
    }
}

fn capture_memory_access(
    decoded_opcode: Option<&DecodedOpcode>,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
    operand_stack: &[u32],
    operand_stack_hi: &[u32],
) -> Result<Option<WasmtimeTraceMemoryAccess>, WasmBuildError> {
    let Some(memory_opcode) = decoded_opcode.and_then(|opcode| opcode.memory) else {
        return Ok(None);
    };

    let base_address = match memory_opcode.kind {
        DecodedMemoryAccessKind::I32Load
        | DecodedMemoryAccessKind::I64Load
        | DecodedMemoryAccessKind::I32Load8S
        | DecodedMemoryAccessKind::I32Load8U
        | DecodedMemoryAccessKind::I32Load16S
        | DecodedMemoryAccessKind::I32Load16U => operand_stack.last().copied().map(u64::from),
        DecodedMemoryAccessKind::I32Store
        | DecodedMemoryAccessKind::I64Store
        | DecodedMemoryAccessKind::I32Store8
        | DecodedMemoryAccessKind::I32Store16 => operand_stack
            .get(operand_stack.len().saturating_sub(2))
            .copied()
            .map(u64::from),
    };
    let Some(base_address) = base_address else {
        return Ok(None);
    };
    let Some(effective_address) = base_address.checked_add(memory_opcode.offset) else {
        return Err(WasmBuildError::Trace("wasmtime effective address overflow".to_string()));
    };

    let width_bytes = memory_opcode.kind.width_bytes();
    let loaded_value_i32 = match memory_opcode.kind {
        DecodedMemoryAccessKind::I32Load | DecodedMemoryAccessKind::I32Store => {
            read_word(memory_opcode.memory_index, effective_address, frame, store)? as i32
        }
        DecodedMemoryAccessKind::I32Load8S => {
            i32::from(read_byte(memory_opcode.memory_index, effective_address, frame, store)? as i8)
        }
        DecodedMemoryAccessKind::I32Load8U | DecodedMemoryAccessKind::I32Store8 => {
            i32::from(read_byte(memory_opcode.memory_index, effective_address, frame, store)?)
        }
        DecodedMemoryAccessKind::I32Load16S => {
            i32::from(read_halfword(memory_opcode.memory_index, effective_address, frame, store)? as i16)
        }
        DecodedMemoryAccessKind::I32Load16U | DecodedMemoryAccessKind::I32Store16 => i32::from(read_halfword(
            memory_opcode.memory_index,
            effective_address,
            frame,
            store,
        )?),
        DecodedMemoryAccessKind::I64Load | DecodedMemoryAccessKind::I64Store => 0,
    };
    let value_after_i32 = match memory_opcode.kind {
        DecodedMemoryAccessKind::I32Load
        | DecodedMemoryAccessKind::I32Load8S
        | DecodedMemoryAccessKind::I32Load8U
        | DecodedMemoryAccessKind::I32Load16S
        | DecodedMemoryAccessKind::I32Load16U => Some(loaded_value_i32),
        DecodedMemoryAccessKind::I32Store
        | DecodedMemoryAccessKind::I32Store8
        | DecodedMemoryAccessKind::I32Store16 => operand_stack
            .last()
            .copied()
            .map(|value| value as i32)
            .or(Some(loaded_value_i32)),
        DecodedMemoryAccessKind::I64Load | DecodedMemoryAccessKind::I64Store => None,
    };

    let base_word_addr = effective_address / 4;
    let byte_offset = (effective_address & 0b11) as u8;
    let lane0_before = read_word(memory_opcode.memory_index, base_word_addr * 4, frame, store)?;
    let lane1_before = read_word(memory_opcode.memory_index, (base_word_addr + 1) * 4, frame, store)?;
    let uses_lane2 = matches!(
        memory_opcode.kind,
        DecodedMemoryAccessKind::I64Load | DecodedMemoryAccessKind::I64Store
    ) && byte_offset + width_bytes > 8;
    let lane2_before = if uses_lane2 {
        Some(read_word(
            memory_opcode.memory_index,
            (base_word_addr + 2) * 4,
            frame,
            store,
        )?)
    } else {
        None
    };

    let mut lane0 = WasmtimeTraceMemoryWordLane {
        word_addr: base_word_addr,
        value_before: lane0_before,
        value_after: lane0_before,
    };

    let mut lane1 = WasmtimeTraceMemoryWordLane {
        word_addr: base_word_addr + 1,
        value_before: lane1_before,
        value_after: lane1_before,
    };
    let mut lane2 = lane2_before.map(|value_before| WasmtimeTraceMemoryWordLane {
        word_addr: base_word_addr + 2,
        value_before,
        value_after: value_before,
    });

    if memory_opcode.kind.is_store() {
        let write_bytes = match memory_opcode.kind {
            DecodedMemoryAccessKind::I64Store => {
                let write_lo = operand_stack.last().copied().unwrap_or(lane0.value_before);
                let write_hi = operand_stack_hi
                    .last()
                    .copied()
                    .unwrap_or(lane1.value_before);
                let mut out = [0u8; 8];
                out[..4].copy_from_slice(&write_lo.to_le_bytes());
                out[4..].copy_from_slice(&write_hi.to_le_bytes());
                out
            }
            _ => {
                let write_value = value_after_i32.unwrap_or(loaded_value_i32) as u32;
                let mut out = [0u8; 8];
                out[..4].copy_from_slice(&write_value.to_le_bytes());
                out
            }
        };

        let byte_offset = usize::from(byte_offset);
        let width_bytes = usize::from(width_bytes);
        if matches!(memory_opcode.kind, DecodedMemoryAccessKind::I64Store) {
            let mut bytes = [0u8; 12];
            bytes[..4].copy_from_slice(&lane0.value_before.to_le_bytes());
            bytes[4..8].copy_from_slice(&lane1.value_before.to_le_bytes());
            if let Some(lane2_ref) = lane2.as_ref() {
                bytes[8..12].copy_from_slice(&lane2_ref.value_before.to_le_bytes());
            }
            bytes[byte_offset..byte_offset + width_bytes].copy_from_slice(&write_bytes[..width_bytes]);
            lane0.value_after = u32::from_le_bytes(bytes[..4].try_into().expect("lane0 bytes"));
            lane1.value_after = u32::from_le_bytes(bytes[4..8].try_into().expect("lane1 bytes"));
            if let Some(lane2_ref) = lane2.as_mut() {
                lane2_ref.value_after = u32::from_le_bytes(bytes[8..12].try_into().expect("lane2 bytes"));
            }
        } else {
            let mut lane0_bytes = lane0.value_before.to_le_bytes();
            let split = width_bytes.min(4usize.saturating_sub(byte_offset));
            lane0_bytes[byte_offset..byte_offset + split].copy_from_slice(&write_bytes[..split]);
            lane0.value_after = u32::from_le_bytes(lane0_bytes);

            if byte_offset + width_bytes > 4 {
                let mut lane1_bytes = lane1.value_before.to_le_bytes();
                let lane1_count = byte_offset + width_bytes - 4;
                lane1_bytes[..lane1_count].copy_from_slice(&write_bytes[split..split + lane1_count]);
                lane1.value_after = u32::from_le_bytes(lane1_bytes);
            }
        }
    }

    Ok(Some(WasmtimeTraceMemoryAccess {
        kind: match memory_opcode.kind {
            DecodedMemoryAccessKind::I32Load => "i32.load".to_string(),
            DecodedMemoryAccessKind::I32Load8S => "i32.load8_s".to_string(),
            DecodedMemoryAccessKind::I32Load8U => "i32.load8_u".to_string(),
            DecodedMemoryAccessKind::I32Load16S => "i32.load16_s".to_string(),
            DecodedMemoryAccessKind::I32Load16U => "i32.load16_u".to_string(),
            DecodedMemoryAccessKind::I64Load => "i64.load".to_string(),
            DecodedMemoryAccessKind::I32Store => "i32.store".to_string(),
            DecodedMemoryAccessKind::I32Store8 => "i32.store8".to_string(),
            DecodedMemoryAccessKind::I32Store16 => "i32.store16".to_string(),
            DecodedMemoryAccessKind::I64Store => "i64.store".to_string(),
        },
        memory_index: memory_opcode.memory_index,
        width_bytes,
        offset: memory_opcode.offset,
        base_address,
        effective_address,
        byte_offset,
        lane0,
        lane1: (byte_offset + width_bytes > 4).then_some(lane1),
        lane2,
        value_before_i32: matches!(
            memory_opcode.kind,
            DecodedMemoryAccessKind::I32Load
                | DecodedMemoryAccessKind::I32Load8S
                | DecodedMemoryAccessKind::I32Load8U
                | DecodedMemoryAccessKind::I32Load16S
                | DecodedMemoryAccessKind::I32Load16U
                | DecodedMemoryAccessKind::I32Store
                | DecodedMemoryAccessKind::I32Store8
                | DecodedMemoryAccessKind::I32Store16
        )
        .then_some(loaded_value_i32),
        value_after_i32,
    }))
}

fn build_debug_function_id_map(
    instance: &wasmtime::Instance,
    mut store: impl wasmtime::AsContextMut<Data = WasmtimeTraceState>,
) -> Result<BTreeMap<usize, u32>, WasmBuildError> {
    let mut out = BTreeMap::new();
    let mut function_index = 0_u32;
    while let Some(func) = instance.debug_function(store.as_context_mut(), function_index) {
        let raw = func.to_raw(store.as_context_mut()) as usize;
        out.insert(raw, function_index.saturating_add(1));
        function_index = function_index.saturating_add(1);
    }
    Ok(out)
}

fn build_store_debug_function_id_map(
    store: &mut Store<WasmtimeTraceState>,
) -> Result<BTreeMap<usize, u32>, WasmBuildError> {
    let mut out = BTreeMap::new();
    for instance in store.debug_all_instances() {
        for (raw, function_id) in build_debug_function_id_map(&instance, &mut *store)? {
            out.insert(raw, function_id);
        }
    }
    Ok(out)
}

fn canonical_wasmparser_func_signature(ty: &wasmparser::FuncType) -> String {
    let params = ty
        .params()
        .iter()
        .map(|ty| canonical_wasmparser_val_type(*ty))
        .collect::<Vec<_>>()
        .join(",");
    let results = ty
        .results()
        .iter()
        .map(|ty| canonical_wasmparser_val_type(*ty))
        .collect::<Vec<_>>()
        .join(",");
    format!("{params}->{results}")
}

fn canonical_wasmparser_val_type(ty: wasmparser::ValType) -> String {
    format!("{ty:?}")
}

fn function_type_id_from_ref(function_ref: u32, store: &StoreContextMut<'_, WasmtimeTraceState>) -> Option<u32> {
    if function_ref == 0 {
        return Some(0);
    }
    store
        .data()
        .function_metas
        .get(&function_ref)
        .map(|meta| meta.type_id)
}

fn function_arity_from_ref(function_ref: u32, store: &StoreContextMut<'_, WasmtimeTraceState>) -> Option<(u8, u8)> {
    store
        .data()
        .function_metas
        .get(&function_ref)
        .map(|meta| (meta.param_count, meta.result_count))
}

fn normalize_value_lanes(
    val: Val,
    func_ref_ids: &BTreeMap<usize, u32>,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<(u32, u32), WasmBuildError> {
    match val {
        Val::I32(v) => Ok((v as u32, 0)),
        Val::I64(v) => {
            let bits = v as u64;
            Ok((bits as u32, (bits >> 32) as u32))
        }
        Val::FuncRef(None) => Ok((0, 0)),
        Val::FuncRef(Some(func)) => {
            let raw = func.to_raw(&mut *store) as usize;
            func_ref_ids
                .get(&raw)
                .copied()
                .map(|id| (id, 0))
                .ok_or_else(|| WasmBuildError::Trace(format!("missing normalized function id for funcref 0x{raw:x}")))
        }
        other => Err(WasmBuildError::Unsupported(format!(
            "unsupported Wasmtime operand stack value for current WASM row surface: {other:?}"
        ))),
    }
}

fn read_global_lanes(
    global_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<(u32, u32), WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let global = instance
        .debug_global(&mut *store, global_index)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime global {} at current frame", global_index)))?;
    let func_ref_ids = store.data().func_ref_ids.clone();
    normalize_value_lanes(global.get(&mut *store), func_ref_ids.as_ref(), store)
}

fn read_table_funcref_u32(
    table_id: u32,
    table_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let table = instance
        .debug_table(&mut *store, table_id)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime table {} at current frame", table_id)))?;
    match table.get(&mut *store, u64::from(table_index)) {
        Some(wasmtime::Ref::Func(None)) => Ok(0),
        Some(wasmtime::Ref::Func(Some(func))) => {
            let raw = func.to_raw(&mut *store) as usize;
            store.data().func_ref_ids.get(&raw).copied().ok_or_else(|| {
                WasmBuildError::Trace(format!("missing normalized function id for table funcref 0x{raw:x}"))
            })
        }
        Some(other) => Err(WasmBuildError::Unsupported(format!(
            "only funcref tables are supported right now, found value {other:?}"
        ))),
        None => Err(WasmBuildError::Trace(format!(
            "table.get out of bounds for table {} index {}",
            table_id, table_index
        ))),
    }
}

fn read_table_size(
    table_id: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let table = instance
        .debug_table(&mut *store, table_id)
        .ok_or_else(|| WasmBuildError::Trace(format!("missing Wasmtime table {} at current frame", table_id)))?;
    u32::try_from(table.size(&mut *store))
        .map_err(|_| WasmBuildError::Trace(format!("table {table_id} size exceeded u32")))
}

fn read_memory_pages_if_present(
    memory_index: u32,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<Option<u32>, WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;
    let Some(memory) = instance.debug_memory(&mut *store, memory_index) else {
        return Ok(None);
    };
    let pages = u32::try_from(memory.size(&mut *store))
        .map_err(|_| WasmBuildError::Trace(format!("memory {memory_index} page count exceeded u32")))?;
    Ok(Some(pages))
}

fn read_memory_bytes<const N: usize>(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<[u8; N], WasmBuildError> {
    let instance = frame
        .instance(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame instance: {err}")))?;

    let Some(memory) = instance.debug_memory(&mut *store, memory_index) else {
        return Err(WasmBuildError::Trace(format!(
            "missing Wasmtime memory {memory_index} at address {effective_address}"
        )));
    };

    let mut bytes = [0_u8; N];

    memory
        .read(&mut *store, effective_address as usize, &mut bytes)
        .map_err(|err| {
            WasmBuildError::Trace(format!(
                "failed to read Wasmtime memory at address {effective_address}: {err}"
            ))
        })?;

    Ok(bytes)
}

fn read_word(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u32, WasmBuildError> {
    Ok(u32::from_le_bytes(read_memory_bytes::<4>(
        memory_index,
        effective_address,
        frame,
        store,
    )?))
}

fn read_byte(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u8, WasmBuildError> {
    Ok(read_memory_bytes::<1>(memory_index, effective_address, frame, store)?[0])
}

fn read_halfword(
    memory_index: u32,
    effective_address: u64,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, WasmtimeTraceState>,
) -> Result<u16, WasmBuildError> {
    Ok(u16::from_le_bytes(read_memory_bytes::<2>(
        memory_index,
        effective_address,
        frame,
        store,
    )?))
}

fn parse_stack_word(value: &str) -> Result<u32, WasmBuildError> {
    parse_signed_u32(value)
        .map_err(|err| WasmBuildError::Trace(format!("failed to parse Wasmtime operand stack value '{value}': {err}")))
}

fn parse_signed_u32(value: &str) -> Result<u32, WasmBuildError> {
    let parsed = value.parse::<i128>().map_err(|err| {
        WasmBuildError::Trace(format!("failed to parse signed i32-compatible value '{value}': {err}"))
    })?;
    Ok((parsed as i32) as u32)
}

fn read_lane(stack: &[u32], sp_before: u64, reads: u8, lane: usize) -> Option<StackLaneAccess> {
    let reads = reads as usize;
    if reads == 0 || lane >= reads {
        return None;
    }
    let stack_index = stack.len().checked_sub(reads)?.checked_add(lane)?;
    let addr = sp_before
        .checked_sub(reads as u64)?
        .checked_add(lane as u64)?;
    stack
        .get(stack_index)
        .copied()
        .map(|value| StackLaneAccess { addr, value })
}

fn read_lane_hi(stack_hi: &[u32], reads: u8, lane: usize) -> Option<u32> {
    let reads = reads as usize;
    if reads == 0 || lane >= reads {
        return None;
    }
    let stack_index = stack_hi.len().checked_sub(reads)?.checked_add(lane)?;
    stack_hi.get(stack_index).copied()
}

fn write_lane(
    current: &SupportedRow,
    next: Option<&SupportedRow>,
    sp_after: u64,
) -> Result<Option<StackLaneAccess>, WasmBuildError> {
    if current.info.stack_writes == 0 {
        return Ok(None);
    }

    let value = match current.opcode {
        WasmOpcode::I32Const => current.immediate_i32.ok_or_else(|| {
            WasmBuildError::Trace(format!(
                "missing Wasmtime immediate for i32.const at cycle {}",
                current.cycle
            ))
        })?,
        WasmOpcode::RefFunc => current.immediate_i32.ok_or_else(|| {
            WasmBuildError::Trace(format!(
                "missing normalized funcref immediate at cycle {}",
                current.cycle
            ))
        })?,
        // local.get pushes the local's pre-execution value; no post-step stack needed.
        WasmOpcode::LocalGet => current.local_value.ok_or_else(|| {
            WasmBuildError::Trace(format!("missing local value for local.get at cycle {}", current.cycle))
        })?,
        WasmOpcode::GlobalGet => current.global_value_before.ok_or_else(|| {
            WasmBuildError::Trace(format!(
                "missing global value for global.get at cycle {}",
                current.cycle
            ))
        })?,
        WasmOpcode::TableGet => current.table_value.ok_or_else(|| {
            WasmBuildError::Trace(format!("missing table value for table.get at cycle {}", current.cycle))
        })?,
        // local.tee leaves the current top of stack unchanged on the stack.
        WasmOpcode::LocalTee => current.operand_stack.last().copied().ok_or_else(|| {
            WasmBuildError::Trace(format!("missing stack top for local.tee at cycle {}", current.cycle))
        })?,
        _ => next
            .and_then(|row| row.operand_stack.last().copied())
            .ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing Wasmtime post-state stack value for {} at cycle {}",
                    current.info.name, current.cycle
                ))
            })?,
    };

    Ok(Some(StackLaneAccess {
        addr: sp_after.saturating_sub(1),
        value,
    }))
}

fn write_lane_hi(current: &SupportedRow, next: Option<&SupportedRow>) -> Result<Option<u32>, WasmBuildError> {
    if current.info.stack_writes == 0 {
        return Ok(None);
    }

    let write_value_hi = match current.opcode {
        WasmOpcode::I64Const => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing Wasmtime post-state high limb for {} at cycle {}",
                    current.info.name, current.cycle
                ))
            })?,
        WasmOpcode::I64Add
        | WasmOpcode::I64Sub
        | WasmOpcode::I64Load
        | WasmOpcode::I64And
        | WasmOpcode::I64Or
        | WasmOpcode::I64Xor
        | WasmOpcode::I64Mul => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing Wasmtime post-state high limb for {} at cycle {}",
                    current.info.name, current.cycle
                ))
            })?,
        WasmOpcode::LocalGet => current.local_value_hi.unwrap_or(0),
        WasmOpcode::GlobalGet => current.global_value_before_hi.unwrap_or(0),
        _ => 0,
    };

    Ok(Some(write_value_hi))
}

fn val_to_string(val: Val) -> String {
    match val {
        Val::I32(x) => x.to_string(),
        Val::I64(x) => x.to_string(),
        Val::F32(x) => f32::from_bits(x).to_string(),
        Val::F64(x) => f64::from_bits(x).to_string(),
        Val::FuncRef(None) => "nullfuncref".to_string(),
        Val::FuncRef(Some(_)) => "funcref".to_string(),
        other => format!("{other:?}"),
    }
}
