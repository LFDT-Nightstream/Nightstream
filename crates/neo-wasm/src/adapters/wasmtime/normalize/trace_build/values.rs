use super::super::NormalizedStep;
use crate::ir::{StackValueAccess, WasmBuildError};
use crate::isa::WasmOpcode;

pub(super) fn collect_callee_initial_params(
    next: Option<&NormalizedStep>,
    callee_fbp: u64,
    param_count: u8,
) -> Vec<(u64, u32)> {
    let Some(next) = next else {
        return vec![];
    };
    next.locals_snapshot
        .iter()
        .take(usize::from(param_count))
        .enumerate()
        .map(|(i, &(lo, _))| (callee_fbp + i as u64, lo))
        .collect()
}

pub(super) fn call_indirect_traps(
    table_value: Option<u32>,
    expected_type_id: Option<u32>,
    function_type_id: Option<u32>,
) -> bool {
    table_value == Some(0) || (expected_type_id.is_some() && expected_type_id != function_type_id)
}

pub(super) fn call_indirect_oob(table_index: Option<u32>, table_size: Option<u32>) -> bool {
    matches!((table_index, table_size), (Some(index), Some(size)) if index >= size)
}

pub(super) fn write_lane(
    current: &NormalizedStep,
    next: Option<&NormalizedStep>,
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
        WasmOpcode::LocalGet => current.local_value_lo.ok_or_else(|| {
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

pub(super) fn write_lane_hi(
    current: &NormalizedStep,
    next: Option<&NormalizedStep>,
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
        WasmOpcode::Call
        | WasmOpcode::CallIndirect
        | WasmOpcode::ReturnCall
        | WasmOpcode::ReturnCallIndirect
        | WasmOpcode::Select => next
            .and_then(|row| row.operand_stack_hi.last().copied())
            .unwrap_or(0),
        _ => 0,
    };

    Ok(Some(write_value_hi))
}
