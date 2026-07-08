//! Per-step normalization of recorded Wasmtime steps.
//!
//! Owns two per-row phases: `capture_frame` runs inside the guest-debug
//! breakpoint handler and snapshots one frame into a `WasmtimeTraceStep`;
//! `normalize_step` decodes one step into a `NormalizedStep`, the
//! private intermediate carrying the per-step facts. Neither phase looks at
//! neighboring steps. The cross-row trace assembly (state machine, aux-row
//! synthesis) lives in the `trace_build` child module. Reads raw values
//! through `runtime_read` and opcode/control metadata through `decode`; it
//! does not run the engine or parse binaries.

mod trace_build;

pub use trace_build::{traces_from_wasmtime_steps, traces_from_wasmtime_steps_with_grammar};

use super::decode::{DecodedControlOpcode, DecodedMemoryAccessKind, DecodedOpcode};
use super::runtime_read::{
    function_arity_from_ref, function_type_id_from_ref, normalize_value_lanes, parse_stack_word, read_byte,
    read_global_lanes, read_halfword, read_memory_pages_if_present, read_table_funcref_u32, read_table_size, read_word,
    val_to_string,
};
use super::{LoweringTables, WasmtimeTraceMemoryAccess, WasmtimeTraceMemoryWordLane, WasmtimeTraceStep};
use crate::ir::{LinearMemoryAccess, LinearMemoryWordLane, WasmBuildError, WasmPcEdgeKind};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode, WasmOpcodeInfo};
use wasmtime::{FrameHandle, StoreContextMut};

#[derive(Clone, Debug)]
struct NormalizedStep {
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
    /// Declared max page count (carried constant), capped at the wasm32 limit.
    max_memory_pages: Option<u32>,
    /// For `call` instructions: binary offset of the instruction after the call (= return address).
    call_return_pc: Option<u64>,
    /// Byte offset immediately after this instruction's encoding.
    pc_after_instruction: Option<u64>,
    /// Total number of locals (params + declared) in this frame at this step.
    num_locals: u32,
    /// Parsed local values at this step (before execution). Used to build aux param-init rows at
    /// call boundaries.
    locals_snapshot: Vec<u32>,
    linear_memory: Option<LinearMemoryAccess>,
    linear_memory_offset: u64,
}

fn normalize_step(row: &WasmtimeTraceStep) -> Result<Option<NormalizedStep>, WasmBuildError> {
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
    // Call rows pop only the indirect table index, if any. Args are popped by
    // aux rows (param-init for guest callees, host-arg for host callees) and
    // host results are pushed by a host-result aux row.
    let (stack_reads_override, stack_writes_override) = match opcode {
        WasmOpcode::Call => (Some(0), Some(0)),
        WasmOpcode::CallIndirect => (Some(1), Some(0)),
        _ => (None, None),
    };

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

    Ok(Some(NormalizedStep {
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
        max_memory_pages: row.memory_max_pages,
        call_return_pc: row.call_return_pc,
        pc_after_instruction: row.pc_after_instruction,
        num_locals: row.num_locals,
        locals_snapshot,
        linear_memory,
        linear_memory_offset: row.memory.as_ref().map(|memory| memory.offset).unwrap_or(0),
    }))
}

pub(crate) fn capture_frame<T>(
    step: u64,
    frame_depth: usize,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
    tables: &LoweringTables,
) -> Result<WasmtimeTraceStep, WasmBuildError> {
    let (function, function_index, pc) = match frame
        .wasm_function_index_and_pc(&mut *store)
        .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime frame function/pc: {err}")))?
    {
        Some((func_index, pc)) => {
            let function_index = func_index.as_u32();
            // The opcode map is keyed by raw module byte offset.
            (format!("{func_index:?}"), Some(function_index), Some(pc.raw()))
        }
        None => ("<host-or-unknown>".to_string(), None, None),
    };
    let decoded_opcode = function_index
        .zip(pc)
        .and_then(|key| tables.opcode_map.get(&key).cloned());
    let current_function_ref = function_index.and_then(|index| {
        tables
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
    let func_ref_ids = &tables.func_ref_ids;
    let mut locals = Vec::with_capacity(num_locals as usize);
    let mut locals_words_hi = Vec::with_capacity(num_locals as usize);
    for index in 0..num_locals {
        let value = frame
            .local(&mut *store, index)
            .map_err(|err| WasmBuildError::Trace(format!("failed to inspect Wasmtime local {index}: {err}")))?;
        locals.push(val_to_string(value));
        let (_, hi) = normalize_value_lanes(value, func_ref_ids, &mut *store)?;
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
        let (lo, hi) = normalize_value_lanes(value, func_ref_ids, &mut *store)?;
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
    // Module constant seeded from parse artifacts.
    let memory_max_now = tables.memory_max_pages;
    let (global_value_before, global_value_before_hi) = match global_index {
        Some(index) => {
            let (lo, hi) = read_global_lanes(index, frame, store, func_ref_ids)?;
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
            (Some(table_id), Some(table_index)) => Some(read_table_funcref_u32(
                table_id,
                table_index,
                frame,
                store,
                func_ref_ids,
            )?),
            _ => None,
        },
        Some(WasmOpcode::TableSet) => operand_stack_words.last().copied(),
        // Skip the funcref read on an OOB index: there is no entry, and the
        // trap is derived from the index/size comparison instead.
        Some(WasmOpcode::CallIndirect) => match (table_id, table_index, table_size) {
            (Some(table_id), Some(table_index), Some(table_size)) if table_index < table_size => Some(
                read_table_funcref_u32(table_id, table_index, frame, store, func_ref_ids)?,
            ),
            _ => None,
        },
        _ => None,
    };
    let function_type_id = match opcode_decoded {
        Some(WasmOpcode::RefFunc) => {
            immediate_i32.and_then(|function_ref| function_type_id_from_ref(function_ref, &tables.function_metas))
        }
        Some(WasmOpcode::TableGet | WasmOpcode::TableSet | WasmOpcode::CallIndirect) => {
            table_value.and_then(|function_ref| function_type_id_from_ref(function_ref, &tables.function_metas))
        }
        Some(WasmOpcode::Call) => immediate_i32
            .and_then(|function_index| function_index.checked_add(1))
            .and_then(|function_ref| function_type_id_from_ref(function_ref, &tables.function_metas)),
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
            .and_then(|function_ref| function_arity_from_ref(function_ref, &tables.function_metas))
            .map_or((None, None), |(params, results)| (Some(params), Some(results))),
        Some(WasmOpcode::CallIndirect) => table_value
            .and_then(|function_ref| function_arity_from_ref(function_ref, &tables.function_metas))
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
            // memory.grow takes a delta; failure leaves the size unchanged.
            let max = memory_max_now.unwrap_or(u32::MAX);
            let grown = pages_before
                .checked_add(delta)
                .filter(|after| *after <= max);
            (Some(pages_before), Some(grown.unwrap_or(pages_before)))
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
            .is_some_and(|function_ref| function_ref > tables.imported_function_count),
        function_type_id,
        call_indirect_type_index,
        expected_type_id: decoded_opcode.as_ref().and_then(|d| d.expected_type_id),
        call_param_count,
        call_result_count,
        memory_pages_before,
        memory_pages_after,
        memory_max_pages: memory_max_now,
        memory,
        locals,
        locals_words_hi,
        operand_stack,
        operand_stack_words,
        operand_stack_words_hi,
        num_locals: num_locals as u32,
        call_return_pc: decoded_opcode.as_ref().and_then(|d| d.call_return_pc),
        pc_after_instruction: decoded_opcode.as_ref().map(|d| d.pc_after_instruction),
    })
}

fn capture_memory_access<T>(
    decoded_opcode: Option<&DecodedOpcode>,
    frame: &FrameHandle,
    store: &mut StoreContextMut<'_, T>,
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

    // On OOB, record the faulting row but skip memory reads; CCS derives the trap.
    let memory_bytes = read_memory_pages_if_present(memory_opcode.memory_index, frame, store)?
        .map_or(0, |pages| u64::from(pages) * 65536);
    let is_oob = effective_address + u64::from(width_bytes) > memory_bytes;

    let loaded_value_i32 = if is_oob {
        0
    } else {
        match memory_opcode.kind {
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
        }
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
    let lane0_before = if is_oob {
        0
    } else {
        read_word(memory_opcode.memory_index, base_word_addr * 4, frame, store)?
    };
    let lane1_before = if is_oob {
        0
    } else {
        read_word(memory_opcode.memory_index, (base_word_addr + 1) * 4, frame, store)?
    };
    let uses_lane2 = matches!(
        memory_opcode.kind,
        DecodedMemoryAccessKind::I64Load | DecodedMemoryAccessKind::I64Store
    ) && byte_offset + width_bytes > 8;
    let lane2_before = if uses_lane2 {
        Some(if is_oob {
            0
        } else {
            read_word(memory_opcode.memory_index, (base_word_addr + 2) * 4, frame, store)?
        })
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
