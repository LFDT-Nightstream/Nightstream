//! Owns the current `wasm` adapter that produces the normalized WASM IR.

use rwasm::mem::MemoryRecordEnum;
use rwasm::{Tracer, TracerInstrState};

use super::super::ir::{StackLaneAccess, WasmBuildError, WasmPcEdgeKind, WasmStepTrace};
use super::super::isa::{opcode_info_from_concrete, WasmOpcode};

pub fn traces_from_rwasm_tracer(tracer: &Tracer) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
    traces_from_rwasm_instr_states(&tracer.logs, 0)
}

pub fn traces_from_rwasm_instr_states(
    rows: &[TracerInstrState],
    initial_stack_pointer: u64,
) -> Result<Vec<WasmStepTrace>, WasmBuildError> {
    let mut out = Vec::with_capacity(rows.len());
    let mut sp = initial_stack_pointer;

    for (idx, row) in rows.iter().enumerate() {
        let info = opcode_info_from_concrete(row.opcode);
        if matches!(info.opcode, WasmOpcode::Unsupported) {
            return Err(WasmBuildError::Unsupported(format!(
                "unsupported WASM opcode at row {idx}: code={}",
                row.opcode.code()
            )));
        }
        if matches!(
            info.opcode,
            WasmOpcode::LocalGet
                | WasmOpcode::LocalSet
                | WasmOpcode::LocalTee
                | WasmOpcode::I32Load
                | WasmOpcode::I32Store
        ) {
            return Err(WasmBuildError::Unsupported(format!(
                "locals or linear-memory opcode at row {idx}: use the wasmtime adapter for programs with locals or memory"
            )));
        }

        let pc_before = u64::from(row.program_counter);
        let pc_after = rows
            .get(idx + 1)
            .map(|next| u64::from(next.program_counter))
            .unwrap_or_else(|| pc_before.saturating_add(1));
        let sp_before = sp;
        let sp_after = sp_before
            .saturating_sub(u64::from(info.stack_reads))
            .saturating_add(u64::from(info.stack_writes));
        let stack_read0 = read_lane(row.memory_access.a, read_addr0(sp_before, info.stack_reads));
        let stack_read1 = read_lane(row.memory_access.b, read_addr1(sp_before, info.stack_reads));
        let stack_read2 = read_lane(row.memory_access.c, read_addr2(sp_before, info.stack_reads));
        let stack_write0 = write_lane(
            row.memory_access.c,
            if info.stack_writes > 0 {
                Some(sp_after.saturating_sub(1))
            } else {
                None
            },
            matches!(info.opcode, WasmOpcode::I32Const),
            row.value as u32,
        );
        let opcode_code = u16::try_from(row.opcode.code())
            .map_err(|_| WasmBuildError::Trace(format!("opcode code does not fit u16 at row {idx}")))?;
        let halted = matches!(info.opcode, WasmOpcode::Return | WasmOpcode::Trap);

        out.push(WasmStepTrace {
            cycle: idx as u64,
            row_kind: super::super::ir::WasmRowKind::Program,
            pc_before,
            pc_after,
            control_choice: 0,
            pc_edge_kind: WasmPcEdgeKind::Static,
            param_init_before: super::super::ir::WasmParamInitState::ZERO,
            param_init_after: super::super::ir::WasmParamInitState::ZERO,
            wide_values_enabled: false,
            opcode_code,
            opcode: info.opcode,
            info,
            stack_reads_override: None,
            stack_writes_override: None,
            sp_before,
            sp_after,
            current_function_ref: 0,
            current_function_num_locals: 0,
            stack_read0,
            stack_read0_hi: None,
            stack_read1,
            stack_read1_hi: None,
            stack_read2,
            stack_read2_hi: None,
            stack_write0,
            stack_write0_hi: None,
            linear_memory: None,
            linear_memory_offset: 0,
            memory_pages_before: None,
            memory_pages_after: None,
            halted,
            locals_fbp: 0,
            locals_fbp_after: 0,
            local_index: None,
            local_read_value: None,
            local_read_value_hi: None,
            local_write_value: None,
            local_write_value_hi: None,
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
            call_param_count: None,
            call_result_count: None,
            call_stack_push: None,
            call_stack_pop: None,
        });
        sp = sp_after;
    }

    Ok(out)
}

fn read_lane(slot: Option<MemoryRecordEnum>, addr: Option<u64>) -> Option<StackLaneAccess> {
    match (slot, addr) {
        (Some(MemoryRecordEnum::Read(read)), Some(addr)) => Some(StackLaneAccess {
            addr,
            value: read.value,
        }),
        _ => None,
    }
}

fn write_lane(
    slot: Option<MemoryRecordEnum>,
    addr: Option<u64>,
    fallback_enabled: bool,
    fallback_value: u32,
) -> Option<StackLaneAccess> {
    match (slot, addr) {
        (Some(MemoryRecordEnum::Write(write)), Some(addr)) => Some(StackLaneAccess {
            addr,
            value: write.value,
        }),
        (None, Some(addr)) if fallback_enabled => Some(StackLaneAccess {
            addr,
            value: fallback_value,
        }),
        _ => None,
    }
}

fn read_addr0(sp_before: u64, reads: u8) -> Option<u64> {
    match reads {
        0 => None,
        1 => Some(sp_before.saturating_sub(1)),
        2 => Some(sp_before.saturating_sub(2)),
        _ => Some(sp_before.saturating_sub(3)),
    }
}

fn read_addr1(sp_before: u64, reads: u8) -> Option<u64> {
    match reads {
        0 | 1 => None,
        2 => Some(sp_before.saturating_sub(1)),
        _ => Some(sp_before.saturating_sub(2)),
    }
}

fn read_addr2(sp_before: u64, reads: u8) -> Option<u64> {
    if reads >= 3 {
        Some(sp_before.saturating_sub(1))
    } else {
        None
    }
}
