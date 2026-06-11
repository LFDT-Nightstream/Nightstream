//! Normalization of recorded Wasmtime steps into the generic WASM IR.
//!
//! Two phases: `capture_frame` runs inside the guest-debug breakpoint handler
//! and snapshots one frame into a `WasmtimeTraceStep`; `traces_from_wasmtime_steps`
//! later folds those steps into `SupportedRow`s and emits `WasmStepTrace` rows
//! (program + aux). `SupportedRow` is the private intermediate that carries the
//! decoded per-step facts between the two phases.
//!
//! Owns the step→row→IR lowering. Reads raw values through `runtime_read` and
//! opcode/control metadata through `decode`; it does not run the engine or
//! parse binaries.

use super::decode::{DecodedControlOpcode, DecodedMemoryAccessKind, DecodedOpcode};
use super::runtime_read::{
    function_arity_from_ref, function_type_id_from_ref, normalize_value_lanes, parse_stack_word, read_byte,
    read_global_lanes, read_halfword, read_lane, read_lane_hi, read_memory_pages_if_present, read_table_funcref_u32,
    read_table_size, read_word, val_to_string,
};
use super::{WasmtimeTraceMemoryAccess, WasmtimeTraceMemoryWordLane, WasmtimeTraceState, WasmtimeTraceStep};
use crate::ir::{
    LinearMemoryAccess, LinearMemoryWordLane, StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmOutputState,
    WasmParamInitState, WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmStepTrace,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode, WasmOpcodeInfo};
use wasmtime::{FrameHandle, StoreContextMut};

#[derive(Clone, Debug)]
struct SupportedRow {
    cycle: u64,
    pc: u32,
    control_choice: u32,
    pc_edge_kind: WasmPcEdgeKind,
    opcode: WasmOpcode,
    info: WasmOpcodeInfo,
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
        return Err(WasmBuildError::Unsupported(format!(
            "row decodes to {} at step {}: this execution cannot be proven",
            opcode.name(),
            row.step
        )));
    }

    let operand_stack = row.operand_stack_words.clone();
    let operand_stack_hi = row.operand_stack_words_hi.clone();
    let code = opcode_code(opcode);
    let (stack_reads_override, stack_writes_override) = match opcode {
        WasmOpcode::Call => row.call_param_count.map(|params| {
            let writes = if row.target_function_is_guest {
                0
            } else {
                row.call_result_count.unwrap_or(0)
            };
            (Some(params), Some(writes))
        }),
        WasmOpcode::CallIndirect => row.call_param_count.map(|params| {
            let writes = if row.target_function_is_guest {
                0
            } else {
                row.call_result_count.unwrap_or(0)
            };
            (Some(params.saturating_add(1)), Some(writes))
        }),
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
        | WasmOpcode::I32Store16
        | WasmOpcode::I64Store8
        | WasmOpcode::I64Store16
        | WasmOpcode::I64Store32
        | WasmOpcode::I64Load8U
        | WasmOpcode::I64Load16U
        | WasmOpcode::I64Load32U
        | WasmOpcode::I64Load8S
        | WasmOpcode::I64Load16S
        | WasmOpcode::I64Load32S => {
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
        wide_values_enabled: opcode.uses_wide_values(),
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

pub(crate) fn capture_frame(
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
        | DecodedMemoryAccessKind::I32Load16U
        | DecodedMemoryAccessKind::I64Load8U
        | DecodedMemoryAccessKind::I64Load16U
        | DecodedMemoryAccessKind::I64Load32U
        | DecodedMemoryAccessKind::I64Load8S
        | DecodedMemoryAccessKind::I64Load16S
        | DecodedMemoryAccessKind::I64Load32S => operand_stack.last().copied().map(u64::from),
        DecodedMemoryAccessKind::I32Store
        | DecodedMemoryAccessKind::I64Store
        | DecodedMemoryAccessKind::I32Store8
        | DecodedMemoryAccessKind::I32Store16
        | DecodedMemoryAccessKind::I64Store8
        | DecodedMemoryAccessKind::I64Store16
        | DecodedMemoryAccessKind::I64Store32 => operand_stack
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
        DecodedMemoryAccessKind::I32Load
        | DecodedMemoryAccessKind::I32Store
        | DecodedMemoryAccessKind::I64Store32
        | DecodedMemoryAccessKind::I64Load32U
        | DecodedMemoryAccessKind::I64Load32S => {
            read_word(memory_opcode.memory_index, effective_address, frame, store)? as i32
        }
        DecodedMemoryAccessKind::I32Load8S | DecodedMemoryAccessKind::I64Load8S => {
            i32::from(read_byte(memory_opcode.memory_index, effective_address, frame, store)? as i8)
        }
        DecodedMemoryAccessKind::I32Load8U
        | DecodedMemoryAccessKind::I32Store8
        | DecodedMemoryAccessKind::I64Store8
        | DecodedMemoryAccessKind::I64Load8U => {
            i32::from(read_byte(memory_opcode.memory_index, effective_address, frame, store)?)
        }
        DecodedMemoryAccessKind::I32Load16S | DecodedMemoryAccessKind::I64Load16S => {
            i32::from(read_halfword(memory_opcode.memory_index, effective_address, frame, store)? as i16)
        }
        DecodedMemoryAccessKind::I32Load16U
        | DecodedMemoryAccessKind::I32Store16
        | DecodedMemoryAccessKind::I64Store16
        | DecodedMemoryAccessKind::I64Load16U => i32::from(read_halfword(
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
        | DecodedMemoryAccessKind::I32Load16U
        | DecodedMemoryAccessKind::I64Load8U
        | DecodedMemoryAccessKind::I64Load16U
        | DecodedMemoryAccessKind::I64Load32U
        | DecodedMemoryAccessKind::I64Load8S
        | DecodedMemoryAccessKind::I64Load16S
        | DecodedMemoryAccessKind::I64Load32S => Some(loaded_value_i32),
        DecodedMemoryAccessKind::I32Store
        | DecodedMemoryAccessKind::I32Store8
        | DecodedMemoryAccessKind::I32Store16
        | DecodedMemoryAccessKind::I64Store8
        | DecodedMemoryAccessKind::I64Store16
        | DecodedMemoryAccessKind::I64Store32 => operand_stack
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
            DecodedMemoryAccessKind::I64Store8 => "i64.store8".to_string(),
            DecodedMemoryAccessKind::I64Store16 => "i64.store16".to_string(),
            DecodedMemoryAccessKind::I64Store32 => "i64.store32".to_string(),
            DecodedMemoryAccessKind::I64Load8U => "i64.load8_u".to_string(),
            DecodedMemoryAccessKind::I64Load16U => "i64.load16_u".to_string(),
            DecodedMemoryAccessKind::I64Load32U => "i64.load32_u".to_string(),
            DecodedMemoryAccessKind::I64Load8S => "i64.load8_s".to_string(),
            DecodedMemoryAccessKind::I64Load16S => "i64.load16_s".to_string(),
            DecodedMemoryAccessKind::I64Load32S => "i64.load32_s".to_string(),
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
                | DecodedMemoryAccessKind::I64Store8
                | DecodedMemoryAccessKind::I64Store16
                | DecodedMemoryAccessKind::I64Store32
                | DecodedMemoryAccessKind::I64Load8U
                | DecodedMemoryAccessKind::I64Load16U
                | DecodedMemoryAccessKind::I64Load32U
                | DecodedMemoryAccessKind::I64Load8S
                | DecodedMemoryAccessKind::I64Load16S
                | DecodedMemoryAccessKind::I64Load32S
        )
        .then_some(loaded_value_i32),
        value_after_i32,
    }))
}

fn write_lane(
    current: &SupportedRow,
    next: Option<&SupportedRow>,
    sp_after: u64,
    stack_writes: u8,
) -> Result<Option<StackValueAccess>, WasmBuildError> {
    if stack_writes == 0 {
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
        WasmOpcode::Call | WasmOpcode::CallIndirect => next
            .and_then(|row| row.operand_stack.last().copied())
            .ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing Wasmtime post-call stack result for {} at cycle {}",
                    current.info.name, current.cycle
                ))
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

    Ok(Some(StackValueAccess::new(
        sp_after.saturating_sub(1).saturating_mul(2),
        value,
    )))
}

fn write_lane_hi(
    current: &SupportedRow,
    next: Option<&SupportedRow>,
    stack_writes: u8,
) -> Result<Option<u32>, WasmBuildError> {
    if stack_writes == 0 || !current.wide_values_enabled {
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
        | WasmOpcode::I64Mul
        | WasmOpcode::I64Shl
        | WasmOpcode::I64ShrS
        | WasmOpcode::I64ShrU
        | WasmOpcode::I64Rotl
        | WasmOpcode::I64Rotr
        | WasmOpcode::I64DivS
        | WasmOpcode::I64DivU
        | WasmOpcode::I64RemS
        | WasmOpcode::I64RemU
        | WasmOpcode::I64Clz
        | WasmOpcode::I64Ctz
        | WasmOpcode::I64Popcnt
        // Signed subword/word loads sign-extend into the hi limb; wasmtime's
        // post-state operand_stack_hi already holds the replicated sign bits.
        | WasmOpcode::I64Load8S
        | WasmOpcode::I64Load16S
        | WasmOpcode::I64Load32S
        | WasmOpcode::I64ExtendI32S
        | WasmOpcode::I64Extend8S
        | WasmOpcode::I64Extend16S
        | WasmOpcode::I64Extend32S => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing Wasmtime post-state high limb for {} at cycle {}",
                    current.info.name, current.cycle
                ))
            })?,
        WasmOpcode::LocalGet => current.local_value_hi.unwrap_or(0),
        WasmOpcode::GlobalGet => current.global_value_before_hi.unwrap_or(0),
        WasmOpcode::Call | WasmOpcode::CallIndirect => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .unwrap_or(0),
        WasmOpcode::Select => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .unwrap_or(0),
        _ => 0,
    };

    Ok(Some(write_value_hi))
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
    let mut call_stack_depth: u64 = 0;
    // Frame base pointer: absolute offset in the flat locals array where current function's
    // locals start. FBP_callee = FBP_caller + num_locals_caller.
    let mut fbp: u64 = 0;
    let mut param_init_state = WasmParamInitState::ZERO;
    let mut output_enabled = false;
    let mut output_value_lo = 0;
    let mut output_value_hi = 0;

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
        let stack_read_hi = |lane| {
            current
                .wide_values_enabled
                .then(|| read_lane_hi(&current.operand_stack_hi, stack_reads, lane))
                .flatten()
        };
        let stack_read0 = read_lane(&current.operand_stack, sp_before, stack_reads, 0)
            .map(|read| read.with_optional_hi(stack_read_hi(0)));
        let stack_read1 = read_lane(&current.operand_stack, sp_before, stack_reads, 1)
            .map(|read| read.with_optional_hi(stack_read_hi(1)));
        let stack_read2 = read_lane(&current.operand_stack, sp_before, stack_reads, 2)
            .map(|read| read.with_optional_hi(stack_read_hi(2)));
        let div_zero_trap = current.opcode.traps_on_zero_divisor()
            && stack_read1.is_some_and(|lane| lane.value_lo == 0 && lane.value_hi.unwrap_or(0) == 0);
        let stack_write0 = if div_zero_trap {
            // No post-state result exists; CCS pins this synthetic write to zero.
            let addr = sp_after.saturating_sub(1).saturating_mul(2);
            let hi = current.wide_values_enabled.then_some(0);
            Some(StackValueAccess::new(addr, 0).with_optional_hi(hi))
        } else {
            let write_value_hi = write_lane_hi(current, next, stack_writes)?;
            write_lane(current, next, sp_after, stack_writes)?.map(|write| write.with_optional_hi(write_value_hi))
        };
        // Only the very last step of the whole trace is halted.
        let halted = next.is_none();
        // A trap is terminal: wasmtime stops stepping at the faulting
        // instruction, so a trapping opcode can only be the last row.
        let trapped = matches!(current.opcode, WasmOpcode::Unreachable) || div_zero_trap;
        if trapped && !halted {
            return Err(WasmBuildError::Trace(format!(
                "trapping {} row at cycle {} is not the final step",
                current.info.name, current.cycle
            )));
        }
        // Div/rem traps keep the parse-derived Static edge; `trapped` marks termination.
        let output_enabled_before = output_enabled;
        let output_value_lo_before = output_value_lo;
        let output_value_hi_before = output_value_hi;
        let mut output_captured = false;
        // A trapped execution has no output: capture is for clean halts only
        // (the CCS enforces the same exclusion).
        if halted && !trapped && !output_enabled {
            if let Some(value) = current.operand_stack.last().copied() {
                output_enabled = true;
                output_value_lo = value;
                output_value_hi = current.operand_stack_hi.last().copied().unwrap_or(0);
                output_captured = true;
            }
        }
        let output_enabled_after = output_enabled;
        let output_value_lo_after = output_value_lo;
        let output_value_hi_after = output_value_hi;

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
        let call_stack_depth_before = call_stack_depth;
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
                    call_stack_depth = call_stack_depth.checked_add(1).ok_or_else(|| {
                        WasmBuildError::Trace(format!("call stack depth overflow at cycle {}", current.cycle))
                    })?;
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
                call_stack_depth = call_stack_depth.checked_sub(1).ok_or_else(|| {
                    WasmBuildError::Trace(format!("call stack depth underflow at cycle {}", current.cycle))
                })?;
                fbp = caller_fbp;
            }
            _ => {
                call_stack_push = None;
                call_stack_pop = None;
                callee_initial_params = vec![];
                guest_callee_fbp = None;
            }
        }
        let call_stack_depth_after = call_stack_depth;

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
            row_kind: WasmRowKind::Program,
            state_before: WasmStepState {
                pc: pc_before,
                sp: sp_before,
                output: WasmOutputState {
                    enabled: output_enabled_before,
                    value_lo: output_value_lo_before,
                    value_hi: output_value_hi_before,
                },
                call_stack_depth: call_stack_depth_before,
                memory_pages: current.memory_pages_before,
                locals_fbp: current_fbp,
                halted: false,
                trapped: false,
                param_init: param_init_before,
            },
            state_after: WasmStepState {
                pc: pc_after,
                sp: sp_after,
                output: WasmOutputState {
                    enabled: output_enabled_after,
                    value_lo: output_value_lo_after,
                    value_hi: output_value_hi_after,
                },
                call_stack_depth: call_stack_depth_after,
                memory_pages: current.memory_pages_after,
                locals_fbp: fbp,
                halted,
                trapped,
                param_init: param_init_after,
            },
            control_choice: current.control_choice,
            pc_edge_kind: current.pc_edge_kind,
            wide_values_enabled: current.wide_values_enabled,
            opcode: current.opcode,
            info: current.info,
            stack_reads_override: current.stack_reads_override,
            stack_writes_override: current.stack_writes_override,
            output_captured,
            current_function_ref: current.current_function_ref.unwrap_or(0),
            current_function_num_locals: current.num_locals,
            stack_read0,
            stack_read1,
            stack_read2,
            stack_write0,
            linear_memory: current.linear_memory,
            linear_memory_offset: current.linear_memory_offset,
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
                    row_kind: WasmRowKind::Aux(WasmAuxOpcode::CallParamInit),
                    state_before: WasmStepState {
                        pc: pc_after,
                        sp: sp_after,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        locals_fbp: callee_fbp,
                        halted: false,
                        trapped: false,
                        param_init: aux_param_init_before,
                    },
                    state_after: WasmStepState {
                        pc: pc_after,
                        sp: sp_after,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        locals_fbp: callee_fbp,
                        halted: false,
                        trapped: false,
                        param_init: aux_param_init_after,
                    },
                    control_choice: 0,
                    pc_edge_kind: WasmPcEdgeKind::Static,
                    wide_values_enabled: read_lane_hi(&current.operand_stack_hi, stack_reads, param_index)
                        .is_some_and(|hi| hi != 0),
                    opcode: WasmOpcode::Nop,
                    info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
                    stack_reads_override: Some(0),
                    stack_writes_override: Some(0),
                    output_captured: false,
                    current_function_ref: callee_function_ref,
                    current_function_num_locals: current.num_locals,
                    stack_read0: Some(
                        src.with_optional_hi(
                            current
                                .wide_values_enabled
                                .then(|| read_lane_hi(&current.operand_stack_hi, stack_reads, param_index))
                                .flatten(),
                        ),
                    ),
                    stack_read1: None,
                    stack_read2: None,
                    stack_write0: None,
                    linear_memory: None,
                    linear_memory_offset: 0,
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
