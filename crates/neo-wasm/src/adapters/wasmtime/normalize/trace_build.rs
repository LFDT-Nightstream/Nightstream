//! Assembly of normalized `SupportedRow`s into proof-facing `WasmStepTrace`
//! rows.
//!
//! Owns the cross-row work: the running trace state machine (sp, fbp, call
//! stack, param-init and host-call mode trackers), consistency checks between
//! adjacent rows, and the synthesis of aux rows that do not exist in
//! wasmtime's step stream — guest `CallParamInit` and host
//! `HostCallArg`/`HostCallResult` rows. Per-row decoding lives in the parent
//! `normalize` module; this module reads it only through `SupportedRow`.

use super::super::runtime_read::{read_lane, read_lane_hi};
use super::super::WasmtimeTraceStep;
use super::{normalize_supported_row, SupportedRow};
use crate::ir::{
    StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmCountdownState, WasmOutputState, WasmPcEdgeKind, WasmRowKind,
    WasmStepState, WasmStepTrace,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode};

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

/// `call_indirect` traps before calling when the table entry is a null
/// funcref or the callee's normalized type id differs from the
/// instruction's expected type id. Mirrors the CCS trap gates in
/// `ccs/call.rs`.
fn call_indirect_traps(table_value: Option<u32>, expected_type_id: Option<u32>, function_type_id: Option<u32>) -> bool {
    table_value == Some(0) || (expected_type_id.is_some() && expected_type_id != function_type_id)
}

/// `call_indirect` also traps when the table index is out of bounds. This is
/// the most upstream cause: there is no entry to read, so the funcref read is
/// skipped (`table_value` stays `None`) and the null/type-mismatch causes do
/// not apply. Mirrors `COL_CI_OOB` in `ccs/trap.rs`.
fn call_indirect_oob(table_index: Option<u32>, table_size: Option<u32>) -> bool {
    matches!((table_index, table_size), (Some(index), Some(size)) if index >= size)
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
    // Runtime call stack: (return_pc, caller_fbp, caller_stack_base) per live
    // guest frame. Grows on Call, shrinks on non-final Return.
    let mut call_stack: Vec<(u64, u64, u64)> = Vec::new();
    let mut call_stack_depth: u64 = 0;
    // Wasmtime snapshots the operand stack per frame, but the VM's `stack`
    // memory is one global address space. Each frame's slots live above the
    // caller's residual operands: global sp = stack_base + frame-local len.
    let mut stack_base: u64 = 0;
    // Frame base pointer: absolute offset in the flat locals array where current function's
    // locals start. FBP_callee = FBP_caller + num_locals_caller.
    let mut fbp: u64 = 0;
    let mut param_init_state = WasmCountdownState::ZERO;
    let mut output_enabled = false;
    let mut output_value_lo = 0;
    let mut output_value_hi = 0;

    for (idx, current) in supported.iter().enumerate() {
        let next = supported.get(idx + 1);
        let pc_before = u64::from(current.pc);
        // Terminal traps have no next row; static-edge traps still bind the PC
        // ROM to the PC after the faulting instruction.
        let pc_after = next.map(|row| u64::from(row.pc)).unwrap_or_else(|| {
            current
                .pc_after_instruction
                .unwrap_or_else(|| pc_before.saturating_add(1))
        });
        let stack_reads = current
            .stack_reads_override
            .unwrap_or(current.info.stack_reads);
        let stack_writes = current
            .stack_writes_override
            .unwrap_or(current.info.stack_writes);
        let sp_before = stack_base + current.operand_stack.len() as u64;
        let expected_sp_after = sp_before
            .saturating_sub(u64::from(stack_reads))
            .saturating_add(u64::from(stack_writes));
        // Guest call args are popped by aux rows, so derive call-row sp_after
        // from this row's own arity instead of the next Wasmtime frame.
        let is_call_row = matches!(current.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect);
        // A non-final return continues in the caller frame, whose base is on
        // the call stack (popped in the opcode match below).
        let is_return_row = matches!(current.opcode, WasmOpcode::Return | WasmOpcode::End) && !call_stack.is_empty();
        let next_row_base = if is_return_row {
            call_stack.last().map(|&(_, _, base)| base).unwrap_or(0)
        } else {
            stack_base
        };
        let sp_after = if is_call_row {
            expected_sp_after
        } else {
            next.map(|row| next_row_base + row.operand_stack.len() as u64)
                .unwrap_or(expected_sp_after)
        };
        let stack_read_hi = |lane| {
            current
                .wide_values_enabled
                .then(|| read_lane_hi(&current.operand_stack_hi, stack_reads, lane))
                .flatten()
        };
        let lane_read = |lane| {
            read_lane(&current.operand_stack, sp_before, stack_reads, lane)
                .map(|read: StackValueAccess| read.with_optional_hi(stack_read_hi(lane)))
        };
        let (stack_read0, stack_read1, stack_read2) = (lane_read(0), lane_read(1), lane_read(2));
        let div_zero_trap = current.opcode.traps_on_zero_divisor()
            && stack_read1.is_some_and(|lane| lane.value_lo == 0 && lane.value_hi.unwrap_or(0) == 0);
        // Signed division overflow: MIN / -1 for the op's width. The wide
        // flag doubles as the width discriminant for the two div_s opcodes.
        let (min_lo, min_hi) = if current.wide_values_enabled {
            (0, 0x8000_0000)
        } else {
            (0x8000_0000, 0)
        };
        let neg1_hi = if current.wide_values_enabled { u32::MAX } else { 0 };
        let div_overflow_trap = current.opcode.traps_on_signed_overflow()
            && stack_read0.is_some_and(|lane| lane.value_lo == min_lo && lane.value_hi.unwrap_or(0) == min_hi)
            && stack_read1.is_some_and(|lane| lane.value_lo == u32::MAX && lane.value_hi.unwrap_or(0) == neg1_hi);
        let div_trap = div_zero_trap || div_overflow_trap;
        // Mirrors the CCS `mem_oob` comparison.
        let mem_oob = current.linear_memory.as_ref().is_some_and(|access| {
            let last_lane =
                access.lane0.word_addr + u64::from(access.lane1.is_some()) + u64::from(access.lane2.is_some());
            last_lane >= u64::from(current.memory_pages_before.unwrap_or(0)) * 16384
        });
        let stack_write0 = if div_trap || mem_oob {
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
        let ci_trap = matches!(current.opcode, WasmOpcode::CallIndirect)
            && (call_indirect_oob(current.table_index, current.table_size)
                || call_indirect_traps(current.table_value, current.expected_type_id, current.function_type_id));
        // A trap is terminal: wasmtime stops stepping at the faulting
        // instruction, so a trapping opcode can only be the last row.
        let trapped = matches!(current.opcode, WasmOpcode::Unreachable) || div_trap || ci_trap || mem_oob;
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
                if !current.target_function_is_guest || ci_trap {
                    // Imported/host callees do not produce guest core rows. In that case the next
                    // guest row is already the post-call continuation, so there is no guest
                    // call-stack boundary to model in this trace. A trapping call_indirect never
                    // enters the callee either, regardless of the target's guest flag.
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
                    // Guest call rows pop only the table index (indirect) or
                    // nothing (direct); args are popped by the aux rows.
                    let expected_stack_reads = u8::from(matches!(current.opcode, WasmOpcode::CallIndirect));
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
                    call_stack.push((return_pc, current_fbp, stack_base));
                    call_stack_depth = call_stack_depth.checked_add(1).ok_or_else(|| {
                        WasmBuildError::Trace(format!("call stack depth overflow at cycle {}", current.cycle))
                    })?;
                    fbp = callee_fbp;
                    // Callee slots start above the caller's residual operands,
                    // i.e. at global sp after the index pop and the aux-row
                    // arg pops.
                    stack_base = sp_after
                        .checked_sub(u64::from(param_count))
                        .ok_or_else(|| {
                            WasmBuildError::Trace(format!(
                                "operand stack underflow popping call args at cycle {}",
                                current.cycle
                            ))
                        })?;
                }
            }
            WasmOpcode::Return | WasmOpcode::End if !call_stack.is_empty() => {
                // Non-final return: restore the caller's FBP and operand-stack
                // base from the call stack.
                let (ret_pc, caller_fbp, caller_base) = call_stack.pop().unwrap();
                call_stack_push = None;
                call_stack_pop = Some((ret_pc, caller_fbp));
                callee_initial_params = vec![];
                guest_callee_fbp = None;
                call_stack_depth = call_stack_depth.checked_sub(1).ok_or_else(|| {
                    WasmBuildError::Trace(format!("call stack depth underflow at cycle {}", current.cycle))
                })?;
                fbp = caller_fbp;
                stack_base = caller_base;
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
        let mut param_init_after = WasmCountdownState::ZERO;
        if call_stack_push.is_some() {
            param_init_after = WasmCountdownState {
                active: !callee_initial_params.is_empty(),
                remaining: u32::try_from(callee_initial_params.len()).map_err(|_| {
                    WasmBuildError::Trace(format!(
                        "call parameter count does not fit u32 at cycle {}",
                        current.cycle
                    ))
                })?,
            };
        }

        // Host calls enter host-arg mode and may owe a result push; the aux
        // rows emitted below walk both back to zero before the next program
        // row.
        let is_host_call = matches!(current.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && !current.target_function_is_guest
            && !ci_trap;
        let mut host_args_after = WasmCountdownState::ZERO;
        let mut host_result_pending_after = false;
        let mut host_call_arity = None;
        if is_host_call {
            let param_count = current.call_param_count.ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing call parameter count for host call at cycle {}",
                    current.cycle
                ))
            })?;
            let result_count = current.call_result_count.ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing call result count for host call at cycle {}",
                    current.cycle
                ))
            })?;
            if result_count > 1 {
                return Err(WasmBuildError::Unsupported(format!(
                    "host call with {result_count} results at cycle {} is unsupported: the canonical ABI \
                     caps flat results at 1 (wider results go through a return pointer)",
                    current.cycle
                )));
            }
            if next.is_none() && (param_count > 0 || result_count > 0) {
                return Err(WasmBuildError::Trace(format!(
                    "host call at cycle {} is the final trace row; its argument/result aux rows have no continuation",
                    current.cycle
                )));
            }
            host_args_after = WasmCountdownState {
                active: param_count > 0,
                remaining: u32::from(param_count),
            };
            host_result_pending_after = result_count == 1;
            host_call_arity = Some((param_count, result_count));
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
                max_memory_pages: current.max_memory_pages,
                locals_fbp: current_fbp,
                halted: false,
                trapped: false,
                param_init: param_init_before,
                // Host aux sequences complete within one normalizer
                // iteration, so a program row always starts with the
                // host-call state fully unwound.
                host_args: WasmCountdownState::ZERO,
                host_result_pending: false,
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
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                halted,
                trapped,
                param_init: param_init_after,
                host_args: host_args_after,
                host_result_pending: host_result_pending_after,
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
            // A trapping call_indirect never calls: its callee-metadata ROM
            // reads are de-gated in the CCS, and the freed columns must
            // satisfy the host-shaped arity rows (is_guest = 0, counts = 0).
            target_function_is_guest: current.target_function_is_guest && !ci_trap,
            function_type_id: current.function_type_id,
            expected_type_id: current.expected_type_id,
            call_indirect_type_index: current.call_indirect_type_index,
            table_size: current.table_size,
            call_param_count: current.call_param_count.filter(|_| !ci_trap),
            call_result_count: current.call_result_count.filter(|_| !ci_trap),
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
            // Aux rows pop stack args top-down, so locals initialize in
            // reverse parameter order.
            let index_pops = usize::from(matches!(current.opcode, WasmOpcode::CallIndirect));
            for (pop_index, &(dst_addr, value)) in callee_initial_params.iter().rev().enumerate() {
                let param_index = param_count - 1 - pop_index;
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
                // The arg sits below the already-popped indirect table index.
                let aux_sp_before = sp_after - pop_index as u64;
                let aux_sp_after = aux_sp_before - 1;
                let stack_pos = current
                    .operand_stack
                    .len()
                    .checked_sub(index_pops + 1 + pop_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing call argument for param {param_index} at cycle {}",
                            current.cycle
                        ))
                    })?;
                let src_value = current.operand_stack[stack_pos];
                let src_value_hi = current.operand_stack_hi.get(stack_pos).copied();
                let src = StackValueAccess::new((aux_sp_before - 1).saturating_mul(2), src_value);
                let remaining_before = param_count - pop_index;
                let remaining_after = remaining_before - 1;
                let remaining_after_u32 = u32::try_from(remaining_after).map_err(|_| {
                    WasmBuildError::Trace(format!(
                        "remaining call parameter count does not fit u32 at cycle {}",
                        current.cycle
                    ))
                })?;
                let aux_param_init_before = param_init_state;
                let aux_param_init_after = WasmCountdownState {
                    active: remaining_after != 0,
                    remaining: remaining_after_u32,
                };
                out.push(WasmStepTrace {
                    cycle: out.len() as u64,
                    row_kind: WasmRowKind::Aux(WasmAuxOpcode::CallParamInit),
                    state_before: WasmStepState {
                        pc: pc_after,
                        sp: aux_sp_before,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        max_memory_pages: current.max_memory_pages,
                        locals_fbp: callee_fbp,
                        halted: false,
                        trapped: false,
                        param_init: aux_param_init_before,
                        host_args: WasmCountdownState::ZERO,
                        host_result_pending: false,
                    },
                    state_after: WasmStepState {
                        pc: pc_after,
                        sp: aux_sp_after,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        max_memory_pages: current.max_memory_pages,
                        locals_fbp: callee_fbp,
                        halted: false,
                        trapped: false,
                        param_init: aux_param_init_after,
                        host_args: WasmCountdownState::ZERO,
                        host_result_pending: false,
                    },
                    control_choice: 0,
                    pc_edge_kind: WasmPcEdgeKind::Static,
                    wide_values_enabled: src_value_hi.is_some_and(|hi| hi != 0),
                    opcode: WasmOpcode::Nop,
                    info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
                    stack_reads_override: Some(1),
                    stack_writes_override: Some(0),
                    output_captured: false,
                    current_function_ref: callee_function_ref,
                    current_function_num_locals: current.num_locals,
                    stack_read0: Some(
                        src.with_optional_hi(current.wide_values_enabled.then(|| src_value_hi).flatten()),
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
                    local_write_value_hi: src_value_hi,
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
        if let Some((param_count, result_count)) = host_call_arity {
            let host_aux_state = |sp: u64, host_args: WasmCountdownState, host_result_pending: bool| WasmStepState {
                pc: pc_after,
                sp,
                output: WasmOutputState {
                    enabled: output_enabled,
                    value_lo: output_value_lo,
                    value_hi: output_value_hi,
                },
                call_stack_depth: call_stack_depth_after,
                memory_pages: current.memory_pages_after,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                halted: false,
                trapped: false,
                param_init: WasmCountdownState::ZERO,
                host_args,
                host_result_pending,
            };
            let host_aux_row = |cycle: u64,
                                aux_opcode: WasmAuxOpcode,
                                state_before: WasmStepState,
                                state_after: WasmStepState| WasmStepTrace {
                cycle,
                row_kind: WasmRowKind::Aux(aux_opcode),
                state_before,
                state_after,
                control_choice: 0,
                pc_edge_kind: WasmPcEdgeKind::Static,
                wide_values_enabled: false,
                opcode: WasmOpcode::Nop,
                info: opcode_info_from_code(opcode_code(WasmOpcode::Nop)),
                stack_reads_override: Some(0),
                stack_writes_override: Some(0),
                output_captured: false,
                current_function_ref: current.current_function_ref.unwrap_or(0),
                current_function_num_locals: current.num_locals,
                stack_read0: None,
                stack_read1: None,
                stack_read2: None,
                stack_write0: None,
                linear_memory: None,
                linear_memory_offset: 0,
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
            };

            // Args pop top-down: the first aux row pops the last argument.
            // The indirect table index was already popped on the call row.
            let index_pops = usize::from(matches!(current.opcode, WasmOpcode::CallIndirect));
            let owes_result = result_count == 1;
            let mut host_args_state = host_args_after;
            for pop_index in 0..usize::from(param_count) {
                let aux_sp_before = sp_after - pop_index as u64;
                let aux_sp_after = aux_sp_before - 1;
                let stack_pos = current
                    .operand_stack
                    .len()
                    .checked_sub(index_pops + 1 + pop_index)
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing host call argument for param {} at cycle {}",
                            usize::from(param_count) - 1 - pop_index,
                            current.cycle
                        ))
                    })?;
                let src_value = current.operand_stack[stack_pos];
                let src_value_hi = current.operand_stack_hi.get(stack_pos).copied();
                let src = StackValueAccess::new((aux_sp_before - 1).saturating_mul(2), src_value)
                    .with_optional_hi(src_value_hi);
                let host_args_before = host_args_state;
                host_args_state = WasmCountdownState {
                    active: host_args_before.remaining > 1,
                    remaining: host_args_before.remaining - 1,
                };
                out.push(WasmStepTrace {
                    wide_values_enabled: src_value_hi.is_some_and(|hi| hi != 0),
                    stack_reads_override: Some(1),
                    stack_read0: Some(src),
                    ..host_aux_row(
                        out.len() as u64,
                        WasmAuxOpcode::HostCallArg,
                        host_aux_state(aux_sp_before, host_args_before, owes_result),
                        host_aux_state(aux_sp_after, host_args_state, owes_result),
                    )
                });
            }
            if owes_result {
                let next_row = next.ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing Wasmtime post-call stack result for host call at cycle {}",
                        current.cycle
                    ))
                })?;
                let result_value = next_row.operand_stack.last().copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing Wasmtime post-call stack result for host call at cycle {}",
                        current.cycle
                    ))
                })?;
                let result_value_hi = next_row.operand_stack_hi.last().copied();
                let aux_sp_before = sp_after - u64::from(param_count);
                let write = StackValueAccess::new(aux_sp_before.saturating_mul(2), result_value)
                    .with_optional_hi(result_value_hi);
                out.push(WasmStepTrace {
                    wide_values_enabled: result_value_hi.is_some_and(|hi| hi != 0),
                    stack_writes_override: Some(1),
                    stack_write0: Some(write),
                    ..host_aux_row(
                        out.len() as u64,
                        WasmAuxOpcode::HostCallResult,
                        host_aux_state(aux_sp_before, WasmCountdownState::ZERO, true),
                        host_aux_state(aux_sp_before + 1, WasmCountdownState::ZERO, false),
                    )
                });
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
