//! Assembly of normalized `NormalizedStep`s into proof-facing `WasmVmStep`
//! rows.
//!
//! Owns the cross-row work: the running trace state machine (sp, fbp, call
//! stack and local-init trackers), consistency checks between
//! adjacent rows, and the synthesis of aux rows that do not exist in
//! wasmtime's step stream — guest `LocalZero`/`FrameMove` and host-event gather
//! rows. Per-row decoding lives in the parent
//! `normalize` module; this module reads it only through `NormalizedStep`.

mod frame_move;
mod local_zero;
mod reentry;
mod turn;
mod values;

use self::frame_move::{emit_frame_move_rows, frame_slide};
use self::local_zero::emit_local_zero_rows;
use self::turn::{plan_turn_exit, setup_turn};
use self::values::{call_indirect_oob, call_indirect_traps, guest_callee_locals, write_lane, write_lane_hi};
use super::super::import_inputs::recover_import_inputs;
use super::super::runtime_read::{read_lane, read_lane_hi};
use super::super::WasmtimeTraceStep;
use super::host_event_emit::{
    emit_block_plan, plan_import_phase, EventBlockPlan, HostCallEventPlan, HostEventRowContext,
};
use super::memory::LinearMemoryImage;
use super::normalize_step;
use crate::comm_chain::CommChainState;
use crate::host_event_bindings::HostEventBindings;
use crate::ir::{
    StackValueAccess, WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmFrameMoveState, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
use crate::isa::WasmOpcode;

pub(super) fn build_trace(
    rows: &[WasmtimeTraceStep],
    host_events: Option<(&HostEventBindings, &super::super::WasmProgramTables)>,
    initial_comm_chain: CommChainState,
    linear_memory: Option<LinearMemoryImage>,
) -> Result<Vec<WasmVmStep>, WasmBuildError> {
    let mut supported = Vec::new();
    for row in rows {
        if let Some(normalized) = normalize_step(row)? {
            supported.push(normalized);
        }
    }
    // Import-free callers get the canonical single-shot bindings: an empty
    // boundary template for the invoked export, so nothing is absorbed and
    // host imports have no template to prove against.
    let import_free_bindings;
    let (bindings, program) = match host_events {
        Some((bindings, program)) => (bindings, Some(program)),
        None => {
            let export_fref = supported
                .first()
                .and_then(|first| first.current_function_ref)
                .unwrap_or(0);
            import_free_bindings = HostEventBindings::import_free(export_fref);
            (&import_free_bindings, None)
        }
    };
    let tracks_linear_memory = linear_memory.is_some();
    let mut linear_memory = linear_memory.unwrap_or_default();
    let mut out: Vec<WasmVmStep> = Vec::with_capacity(supported.len());
    // Shared return-context stack. Guest calls push/pop normally; split
    // imports reserve one entry until resume, below any reentrant guest frames.
    let mut call_stack: Vec<(u64, u64, u64)> = Vec::new();
    let mut call_stack_depth: u64 = 0;
    let mut suspended: Vec<reentry::Suspended<'_>> = vec![];
    // Wasmtime snapshots the operand stack per frame, but the VM's `stack`
    // memory is one global address space. Each frame's slots live above the
    // caller's residual operands: global sp = stack_base + frame-local len.
    let mut stack_base: u64 = 0;
    // Frame base pointer: absolute offset in the flat locals array where current function's
    // locals start. FBP_callee = FBP_caller + num_locals_caller.
    let mut fbp: u64 = 0;
    let mut frame_move_state = WasmFrameMoveState::ZERO;
    let mut local_zero_state = WasmCountdownState::ZERO;
    // Callee attribution carry: set on host-call rows, preserved everywhere
    // else (no clearing — see `WasmStepState::host_callee_fref`).
    let mut host_callee_fref: u32 = 0;
    // Host-event commitment chain; each absorbed block folds it forward.
    let mut comm_chain = initial_comm_chain.canonical_u64();
    // Host-event absorb machinery (block buffer and perm group
    // state); carried across rows, mutated only by host-call events.
    let mut event_absorb = WasmEventAbsorbState::ZERO;
    // Host-event gather state (schedule, args base, cursor).
    let mut host_event_state = crate::ir::WasmHostEventState::ZERO;
    // The verifier mirrors the first turn's entry schedule in
    // `host_event_top_level_initial_state`.
    let mut export_boundary = if let Some(first) = supported.first() {
        let setup = setup_turn(bindings, first, program, false, &mut linear_memory)?;
        host_event_state = crate::ir::WasmHostEventState {
            turn_export_fref: setup.fref,
            host_frame_depth: 0,
            awaiting_host: false,
            suspend_pending: false,
            events_remaining: setup.entry_plans.len() as u32,
            event_index: 0,
            args_base: 0,
            slot_cursor: 0,
        };
        host_callee_fref = setup.fref;
        // The first export's locals are already zero in the initial stack RAM.
        // Reserve the whole frame; subsequent entries still zero reused cells.
        stack_base = u64::from(first.num_locals);
        if first.num_locals < u32::from(setup.param_count) {
            return Err(WasmBuildError::Trace("export has more parameters than locals".into()));
        }
        Some(setup)
    } else {
        None
    };
    let mut entry_emitted = false;
    // Activation completion latch; export entry or import resume clears it.
    let mut turn_done = false;
    let mut output_enabled = false;
    let mut output_value_lo = 0;
    let mut output_value_hi = 0;

    for (idx, current) in supported.iter().enumerate() {
        let next = supported.get(idx + 1);
        let pc_before = u64::from(current.pc);
        // A return-like edge with no caller ends the current export invocation.
        let turn_terminal =
            current.pc_edge_kind == WasmPcEdgeKind::ReturnLike && call_stack_depth == host_event_state.host_frame_depth;
        let activation_finished = next.is_none() || turn_terminal;
        // Rows finishing an activation keep their one-past pc (terminal traps have no next
        // row; a turn boundary bridges to the next turn's entry pc).
        let pc_after = if matches!(current.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && !current.target_function_is_guest
        {
            current
                .pc_after_instruction
                .ok_or_else(|| WasmBuildError::Trace("import lacks continuation PC".into()))?
        } else if activation_finished {
            current
                .pc_after_instruction
                .unwrap_or_else(|| pc_before.saturating_add(1))
        } else {
            u64::from(next.expect("unfinished activation has a successor").pc)
        };
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
        let is_call_row = matches!(
            current.opcode,
            WasmOpcode::Call | WasmOpcode::CallIndirect | WasmOpcode::ReturnCall | WasmOpcode::ReturnCallIndirect
        );
        // A non-final return continues in the caller frame, whose base is on
        // the call stack (popped in the opcode match below). Only a function-
        // ending `end` is return-like; structured block/loop `end` rows stay
        // in the current frame.
        let is_return_row = matches!(current.opcode, WasmOpcode::Return | WasmOpcode::End)
            && current.pc_edge_kind == WasmPcEdgeKind::ReturnLike
            && call_stack_depth > host_event_state.host_frame_depth;
        let next_row_base = if is_return_row {
            call_stack.last().map(|&(_, _, base)| base).unwrap_or(0)
        } else {
            stack_base
        };
        let sp_after = if is_call_row || activation_finished {
            // A later turn has a fresh stack, so terminal rows use their own arity.
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
        let ci_trap = matches!(
            current.opcode,
            WasmOpcode::CallIndirect | WasmOpcode::ReturnCallIndirect
        ) && (call_indirect_oob(current.table_index, current.table_size)
            || call_indirect_traps(current.table_value, current.expected_type_id, current.function_type_id));
        // A trap is terminal: wasmtime stops stepping at the faulting
        // instruction, so a trapping opcode can only be the last row.
        let trapped = matches!(current.opcode, WasmOpcode::Unreachable) || div_trap || ci_trap || mem_oob;
        if trapped && !activation_finished {
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
        // A trapped execution has no output: capture is for clean activation returns only
        // (the CCS enforces the same exclusion).
        // A return's results are its frame above fbp less its static slide distance.
        let has_result = match current.frame_slide {
            Some((_, distance)) if current.pc_edge_kind == WasmPcEdgeKind::ReturnLike => {
                let result_count = sp_before
                    .checked_sub(fbp)
                    .and_then(|frame_cells| frame_cells.checked_sub(u64::from(distance)))
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "return result count underflow at cycle {}: sp {sp_before}, frame base {fbp}, slide distance {distance}",
                            current.cycle
                        ))
                    })?;
                result_count != 0
            }
            _ => !current.operand_stack.is_empty(),
        };
        if activation_finished && !trapped && !output_enabled && has_result {
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
        // Model the host consuming a captured result. (Mutable: a host-event
        // host call overrides this below to pop its args on the call row.)
        let mut sp_after = sp_after - u64::from(output_captured);
        // A clean return also drops the finished activation's frame.
        if activation_finished && !trapped {
            sp_after = fbp;
        }

        // local_read_value: the local's value before this step (local.get: pushed onto stack).
        // local_write_value: the value being stored into the local (local.set / local.tee:
        //   the top of operand stack at this step, captured before execution).
        let local_read_value = if matches!(current.opcode, WasmOpcode::LocalGet) {
            current.local_value_lo
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

        // Frame tracking and call/return handling. A frame is
        // `[params][declared locals][operands]` at fbp; `stack_base` is its
        // operand base.
        let current_fbp = fbp;
        let current_stack_base = stack_base;
        let call_stack_depth_before = call_stack_depth;
        let call_stack_push;
        let call_stack_pop;
        // Cells a return or tail call keeps, bottom-up, and the frame move
        // that copies them down.
        let mut kept_cells: Vec<(u32, u32)> = vec![];
        let mut frame_move_after = WasmFrameMoveState::ZERO;
        let mut locals_to_zero = 0;
        let cell = |pos: usize| {
            (
                current.operand_stack[pos],
                current.operand_stack_hi.get(pos).copied().unwrap_or(0),
            )
        };

        match current.opcode {
            WasmOpcode::Call | WasmOpcode::CallIndirect => {
                let return_pc = current.pc_after_instruction.unwrap_or(pc_after);
                if !current.target_function_is_guest || ci_trap {
                    // Imported/host callees do not produce guest core rows. In that case the next
                    // guest row is already the post-call continuation, so there is no guest
                    // call-stack boundary to model in this trace. A trapping call_indirect never
                    // enters the callee either, regardless of the target's guest flag.
                    call_stack_push = None;
                    call_stack_pop = None;
                } else {
                    let param_count = current.call_param_count.ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "missing call parameter count for guest call at cycle {}",
                            current.cycle
                        ))
                    })?;
                    // Guest call rows pop only the table index (indirect) or
                    // nothing (direct); the args become the callee's params.
                    let expected_stack_reads = u8::from(matches!(current.opcode, WasmOpcode::CallIndirect));
                    if stack_reads != expected_stack_reads {
                        return Err(WasmBuildError::Trace(format!(
                            "call stack read count {} does not match expected count {} at cycle {}",
                            stack_reads, expected_stack_reads, current.cycle
                        )));
                    }
                    let callee_fbp = sp_after
                        .checked_sub(u64::from(param_count))
                        .ok_or_else(|| {
                            WasmBuildError::Trace(format!(
                                "operand stack underflow popping call args at cycle {}",
                                current.cycle
                            ))
                        })?;
                    let locals = guest_callee_locals(current, next, param_count)?;
                    call_stack_push = Some((return_pc, current_fbp, current_stack_base));
                    call_stack_pop = None;
                    call_stack.push((return_pc, current_fbp, current_stack_base));
                    call_stack_depth = call_stack_depth.checked_add(1).ok_or_else(|| {
                        WasmBuildError::Trace(format!("call stack depth overflow at cycle {}", current.cycle))
                    })?;
                    fbp = callee_fbp;
                    stack_base = callee_fbp + u64::from(locals);
                    locals_to_zero = locals - u32::from(param_count);
                }
            }
            WasmOpcode::ReturnCall | WasmOpcode::ReturnCallIndirect if !ci_trap => {
                if !current.target_function_is_guest {
                    return Err(WasmBuildError::Unsupported(format!(
                        "{} to a host import at cycle {} is not supported yet",
                        current.info.name, current.cycle
                    )));
                }
                let param_count = current.call_param_count.ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing call parameter count for tail call at cycle {}",
                        current.cycle
                    ))
                })?;
                let expected_stack_reads = u8::from(matches!(current.opcode, WasmOpcode::ReturnCallIndirect));
                if stack_reads != expected_stack_reads {
                    return Err(WasmBuildError::Trace(format!(
                        "tail-call stack read count {} does not match expected count {} at cycle {}",
                        stack_reads, expected_stack_reads, current.cycle
                    )));
                }
                let locals = guest_callee_locals(current, next, param_count)?;
                // The arguments replace the current frame at its base.
                let args_end = current.operand_stack.len() - usize::from(stack_reads);
                let args_start = args_end
                    .checked_sub(usize::from(param_count))
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!(
                            "operand stack underflow collecting tail-call args at cycle {}",
                            current.cycle
                        ))
                    })?;
                kept_cells = (args_start..args_end).map(cell).collect();
                sp_after = current_fbp + u64::from(param_count);
                frame_move_after = WasmFrameMoveState {
                    active: param_count != 0,
                    remaining: u32::from(param_count),
                    distance: sp_before - u64::from(stack_reads) - sp_after,
                };
                call_stack_push = None;
                call_stack_pop = None;
                stack_base = current_fbp + u64::from(locals);
                locals_to_zero = locals - u32::from(param_count);
            }
            WasmOpcode::Return | WasmOpcode::End if is_return_row => {
                // Non-final return: the results move down to the callee's
                // frame base, and the caller's frame comes back from the
                // call stack.
                let (ret_pc, caller_fbp, caller_base) = call_stack.pop().unwrap();
                call_stack_push = None;
                call_stack_pop = Some((ret_pc, caller_fbp, caller_base));
                call_stack_depth = call_stack_depth.checked_sub(1).ok_or_else(|| {
                    WasmBuildError::Trace(format!("call stack depth underflow at cycle {}", current.cycle))
                })?;
                (kept_cells, frame_move_after) = frame_slide(current, sp_before, sp_after)?;
                fbp = caller_fbp;
                stack_base = caller_base;
            }
            WasmOpcode::Br | WasmOpcode::BrIf | WasmOpcode::BrTable => {
                call_stack_push = None;
                call_stack_pop = None;
                if activation_finished {
                    let distance = current.frame_slide.map_or(0, |(_, distance)| distance);
                    sp_after -= u64::from(distance);
                }
                (kept_cells, frame_move_after) = frame_slide(current, sp_before, sp_after)?;
            }
            _ => {
                call_stack_push = None;
                call_stack_pop = None;
            }
        }
        let mut call_stack_depth_after = call_stack_depth;

        let program_cycle = out.len() as u64;
        let frame_move_before = frame_move_state;
        let zero_after = WasmCountdownState {
            active: locals_to_zero != 0 && !frame_move_after.active,
            remaining: locals_to_zero,
        };

        // Host calls enter host-arg mode and may owe a result push; the aux
        // rows emitted below walk both back to zero before the next program
        // row.
        // Emit entry events before the turn's first program row.
        if !entry_emitted {
            entry_emitted = true;
            if let Some(setup) = &export_boundary {
                let mut ctx = HostEventRowContext {
                    pc: pc_before,
                    sp: sp_before,
                    stack_frame_base: current_stack_base,
                    output: WasmOutputState {
                        enabled: output_enabled_before,
                        value_lo: output_value_lo_before,
                        value_hi: output_value_hi_before,
                    },
                    call_stack_depth: call_stack_depth_before,
                    memory_pages: current.memory_pages_before,
                    max_memory_pages: current.max_memory_pages,
                    locals_fbp: current_fbp,
                    host_callee_fref,
                    current_function_ref: current.current_function_ref.unwrap_or(0),
                    activation_finished: turn_done,
                };
                if local_zero_state.active {
                    let state = out
                        .last()
                        .ok_or_else(|| {
                            WasmBuildError::Trace("local zeroing requires a preceding re-entry boundary".into())
                        })?
                        .state_after;
                    let after = emit_local_zero_rows(&mut out, &ctx, state);
                    local_zero_state = after.local_zero;
                }
                for plan in &setup.entry_plans {
                    emit_block_plan(
                        &mut out,
                        &mut ctx,
                        &mut comm_chain,
                        &mut event_absorb,
                        &mut host_event_state,
                        plan,
                    );
                }
            }
        }

        if tracks_linear_memory && !mem_oob {
            linear_memory.apply_program_access(current)?;
        }

        let is_host_call = matches!(current.opcode, WasmOpcode::Call | WasmOpcode::CallIndirect)
            && !current.target_function_is_guest
            && !ci_trap;
        let mut host_event_plan: Option<HostCallEventPlan> = None;
        let mut host_context = None;
        let mut return_count = None;
        let host_event_state_before_row = host_event_state;
        let host_callee_fref_before = host_callee_fref;
        let comm_chain_before_row = comm_chain;
        let event_absorb_before_row = event_absorb;
        if is_host_call {
            host_callee_fref = current.function_ref.ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "missing callee function ref for host call at cycle {}",
                    current.cycle
                ))
            })?;
            let template = bindings.imports.get(&host_callee_fref).ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "no host-event template for host import fref {host_callee_fref} at cycle {}",
                    current.cycle
                ))
            })?;
            if !template.is_split() && next.is_some_and(|row| row.activation_depth > current.activation_depth) {
                return Err(WasmBuildError::Unsupported(
                    "same-instance reentry under an atomic import".into(),
                ));
            }
            if !template.is_split()
                && template.input_count > 0
                && next.is_none_or(|row| Some(u64::from(row.pc)) != current.pc_after_instruction)
            {
                return Err(WasmBuildError::Trace(format!(
                    "import memory recovery at cycle {} requires the caller continuation; \
                     same-instance reentry is not supported by normalization",
                    current.cycle
                )));
            }
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
            if next.is_none() && result_count > 0 {
                return Err(WasmBuildError::Trace(format!(
                    "host call at cycle {} is the final trace row; its result has no continuation",
                    current.cycle
                )));
            }
            let index_pops = usize::from(matches!(current.opcode, WasmOpcode::CallIndirect));
            let args_start = current
                .operand_stack
                .len()
                .checked_sub(index_pops + usize::from(param_count))
                .ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "operand stack underflow collecting host call args at cycle {}",
                        current.cycle
                    ))
                })?;
            let arg_limbs: Vec<(u32, u32)> = (0..usize::from(param_count))
                .map(|k| {
                    let pos = args_start + k;
                    (
                        current.operand_stack[pos],
                        current.operand_stack_hi.get(pos).copied().unwrap_or(0),
                    )
                })
                .collect();
            let result_limbs = if result_count == 1 && !template.is_split() {
                let next_row = next.ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing Wasmtime post-call stack result for host call at cycle {}",
                        current.cycle
                    ))
                })?;
                let lo = next_row.operand_stack.last().copied().ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "missing Wasmtime post-call stack result for host call at cycle {}",
                        current.cycle
                    ))
                })?;
                Some((lo, next_row.operand_stack_hi.last().copied().unwrap_or(0)))
            } else {
                None
            };
            let inputs = if template.is_split() {
                vec![]
            } else {
                recover_import_inputs(
                    template,
                    &arg_limbs,
                    current.memory_pages_before,
                    current.host_call_memory.as_ref(),
                )?
            };
            let args_base = sp_before - index_pops as u64 - u64::from(param_count);
            host_event_plan = Some(
                plan_import_phase(
                    &template.events,
                    args_base,
                    &arg_limbs,
                    result_limbs,
                    &inputs,
                    current.memory_pages_before,
                    &mut linear_memory,
                )
                .map_err(|err| {
                    WasmBuildError::Trace(format!(
                        "host call to fref {host_callee_fref} at cycle {}: {err}",
                        current.cycle
                    ))
                })?,
            );
            // Atomic calls release arguments now; split calls retain them
            // below the unchanged SP until resume releases their cells.
            return_count = Some(template.return_schedule_count());
            if template.is_split() {
                sp_after = sp_before - index_pops as u64;
                let context = crate::ir::WasmHostContext {
                    continuation_pc: pc_after,
                    locals_fbp: current_fbp,
                    stack_frame_base: current_stack_base,
                    export_fref: host_event_state.turn_export_fref,
                    import_fref: host_callee_fref,
                    args_base,
                    parent_host_frame_depth: host_event_state.host_frame_depth,
                    retained_sp: sp_after,
                };
                host_event_state.suspend_pending = true;
                host_context = Some(context);
                suspended.push(reentry::Suspended {
                    context,
                    call_index: idx,
                    activation_depth: current.activation_depth,
                    export: export_boundary.as_ref().expect("active export").clone(),
                });
            } else {
                sp_after = args_base;
            }
        }
        // The call row latches the event schedule and argument-region base.
        if let Some(plan) = &host_event_plan {
            host_event_state = crate::ir::WasmHostEventState {
                turn_export_fref: host_event_state.turn_export_fref,
                host_frame_depth: host_event_state.host_frame_depth,
                awaiting_host: host_event_state.awaiting_host,
                suspend_pending: host_event_state.suspend_pending,
                events_remaining: plan.blocks.len() as u32,
                event_index: 0,
                args_base: plan.args_base,
                slot_cursor: 0,
            };
        }
        // A clean export return loads the exit schedule and attribution. Result
        // publication is optional: constant-only exit events also apply to
        // resultless exports.
        let mut exit_plans: Option<Vec<EventBlockPlan>> = None;
        let mut exit_counts: Option<(u32, u32)> = None;
        if activation_finished && !trapped {
            if let Some(setup) = &export_boundary {
                let plans = plan_turn_exit(
                    setup.template,
                    current,
                    output_captured.then_some((output_value_lo_after, output_value_hi_after)),
                    &linear_memory,
                )?;
                host_callee_fref = setup.fref;
                host_event_state = crate::ir::WasmHostEventState {
                    turn_export_fref: host_event_state.turn_export_fref,
                    host_frame_depth: host_event_state.host_frame_depth,
                    awaiting_host: host_event_state.host_frame_depth != 0,
                    suspend_pending: false,
                    events_remaining: plans.len() as u32,
                    event_index: setup.template.entry.len() as u32,
                    args_base: host_event_state.args_base,
                    slot_cursor: 0,
                };
                exit_counts = Some((setup.template.entry.len() as u32, setup.template.exit.len() as u32));
                exit_plans = Some(plans);
            }
        }

        let zero_before = local_zero_state;
        out.push(WasmVmStep {
            // Sequential index within the normalized trace. Structural-only opcodes
            // (loop, block, inner End) are filtered before this loop, so this is
            // always consecutive — matching Stage 3's cycle_delta == 1 invariant.
            cycle: program_cycle,
            row_kind: WasmRowKind::Program,
            state_before: WasmStepState {
                pc: pc_before,
                sp: sp_before,
                stack_frame_base: current_stack_base,
                output: WasmOutputState {
                    enabled: output_enabled_before,
                    value_lo: output_value_lo_before,
                    value_hi: output_value_hi_before,
                },
                call_stack_depth: call_stack_depth_before,
                memory_pages: current.memory_pages_before,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: current_fbp,
                activation_finished: turn_done,
                trapped: false,
                frame_move: frame_move_before,
                local_zero: zero_before,
                host_callee_fref: host_callee_fref_before,
                comm_chain: comm_chain_before_row,
                event_absorb: event_absorb_before_row,
                host_events: host_event_state_before_row,
            },
            state_after: WasmStepState {
                pc: pc_after,
                sp: sp_after,
                stack_frame_base: stack_base,
                output: WasmOutputState {
                    enabled: output_enabled_after,
                    value_lo: output_value_lo_after,
                    value_hi: output_value_hi_after,
                },
                call_stack_depth: call_stack_depth_after,
                memory_pages: current.memory_pages_after,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                activation_finished: turn_done || activation_finished,
                trapped,
                frame_move: frame_move_after,
                local_zero: zero_after,
                host_callee_fref,
                // The chain only moves on permutation rows.
                comm_chain,
                event_absorb,
                host_events: host_event_state,
            },
            control_choice: current.control_choice,
            br_table_len: if current.opcode == WasmOpcode::BrTable {
                current.immediate_i32.expect("decoded br_table arm count")
            } else {
                0
            },
            pc_edge_kind: current.pc_edge_kind,
            wide_values_enabled: current.wide_values_enabled,
            opcode: current.opcode,
            info: current.info,
            stack_reads_override: current.stack_reads_override,
            stack_writes_override: current.stack_writes_override,
            output_captured,
            current_function_ref: current.current_function_ref.unwrap_or(0),
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
            host_context: None,
            host_event_return_schedule_count: return_count,
            host_event_rom_slot: None,
            // Count cells carry the presence bias (count + 1).
            host_event_initial_schedule_count: host_event_plan
                .as_ref()
                .map(|plan| plan.blocks.len() as u32 + 1)
                .or(exit_counts.map(|(pre, _)| pre + 1)),
            host_event_exit_schedule_count: exit_counts.map(|(_, exit)| exit),
        });
        if let Some(context) = host_context {
            let after = reentry::suspend(&mut out, current, context)?;
            call_stack.push((pc_after, current_fbp, current_stack_base));
            call_stack_depth = after.call_stack_depth;
            call_stack_depth_after = call_stack_depth;
            host_event_state = after.host_events;
        }
        turn_done = turn_done || activation_finished;
        frame_move_state = frame_move_after;
        local_zero_state = zero_after;
        if frame_move_state.active || local_zero_state.active {
            // Moved cells and fresh locals belong to the frame this row enters.
            let entered = next.expect("frame transition has a successor");
            let state = out
                .last()
                .expect("frame transition precedes its aux rows")
                .state_after;
            let ctx = HostEventRowContext {
                pc: state.pc,
                sp: state.sp,
                stack_frame_base: state.stack_frame_base,
                output: state.output,
                call_stack_depth: state.call_stack_depth,
                memory_pages: state.memory_pages,
                max_memory_pages: state.max_memory_pages,
                locals_fbp: state.locals_fbp,
                host_callee_fref: state.host_callee_fref,
                current_function_ref: entered.current_function_ref.unwrap_or(0),
                activation_finished: state.activation_finished,
            };
            let state = emit_frame_move_rows(&mut out, &ctx, state, &kept_cells);
            let state = emit_local_zero_rows(&mut out, &ctx, state);
            frame_move_state = state.frame_move;
            local_zero_state = state.local_zero;
        }
        if let Some(plan) = &host_event_plan {
            let mut ctx = HostEventRowContext {
                pc: pc_after,
                sp: sp_after,
                stack_frame_base: stack_base,
                output: WasmOutputState {
                    enabled: output_enabled,
                    value_lo: output_value_lo,
                    value_hi: output_value_hi,
                },
                call_stack_depth: call_stack_depth_after,
                memory_pages: current.memory_pages_after,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                host_callee_fref,
                current_function_ref: current.current_function_ref.unwrap_or(0),
                activation_finished: turn_done,
            };
            for block_plan in &plan.blocks {
                emit_block_plan(
                    &mut out,
                    &mut ctx,
                    &mut comm_chain,
                    &mut event_absorb,
                    &mut host_event_state,
                    block_plan,
                );
            }
            let gather_pushes = if plan.blocks.is_empty() || host_context.is_some() {
                0
            } else {
                u64::from(current.call_result_count.unwrap_or(0))
            };
            debug_assert_eq!(ctx.sp, sp_after + gather_pushes);
        }

        // Export boundary: the exit template's blocks absorb after the
        // capture row (its own after-state carries the exit latch).
        if let Some(plans) = exit_plans {
            let mut ctx = HostEventRowContext {
                pc: pc_after,
                sp: sp_after,
                stack_frame_base: stack_base,
                output: WasmOutputState {
                    enabled: output_enabled_after,
                    value_lo: output_value_lo_after,
                    value_hi: output_value_hi_after,
                },
                call_stack_depth: call_stack_depth_after,
                memory_pages: current.memory_pages_after,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                host_callee_fref,
                current_function_ref: current.current_function_ref.unwrap_or(0),
                activation_finished: turn_done,
            };
            for plan in &plans {
                emit_block_plan(
                    &mut out,
                    &mut ctx,
                    &mut comm_chain,
                    &mut event_absorb,
                    &mut host_event_state,
                    plan,
                );
            }
        }

        // A suspended import may receive another export activation, or resume.
        // Export exits return to the host; they never supply the import result.
        if host_event_state.awaiting_host {
            let next_row = next.ok_or_else(|| WasmBuildError::Trace("trace ends with a suspended import".into()))?;
            let saved = suspended.last().expect("pending import has saved context");
            if next_row.activation_depth > saved.activation_depth {
                export_boundary = Some(reentry::enter(
                    &mut out,
                    current,
                    next_row,
                    bindings,
                    program,
                    &mut linear_memory,
                    Some(saved),
                )?);
                entry_emitted = false;
            } else {
                let saved = suspended.pop().expect("checked");
                reentry::resume(
                    &mut out,
                    &saved,
                    &supported[saved.call_index],
                    next_row,
                    bindings,
                    &mut linear_memory,
                )?;
                call_stack.pop().expect("host context stack entry");
                export_boundary = Some(saved.export);
                entry_emitted = true;
            }
        } else if activation_finished && next.is_some() {
            export_boundary = Some(reentry::enter(
                &mut out,
                current,
                next.expect("checked"),
                bindings,
                program,
                &mut linear_memory,
                None,
            )?);
            entry_emitted = false;
        }
        if let Some(last) = out.last() {
            let state = last.state_after;
            fbp = state.locals_fbp;
            stack_base = state.stack_frame_base;
            call_stack_depth = state.call_stack_depth;
            host_event_state = state.host_events;
            host_callee_fref = state.host_callee_fref;
            comm_chain = state.comm_chain;
            event_absorb = state.event_absorb;
            turn_done = state.activation_finished;
            output_enabled = state.output.enabled;
            output_value_lo = state.output.value_lo;
            output_value_hi = state.output.value_hi;
            local_zero_state = state.local_zero;
            frame_move_state = state.frame_move;
        }
    }

    if !suspended.is_empty() {
        return Err(WasmBuildError::Trace(
            "trace ends inside a suspended host activation".into(),
        ));
    }
    if out.is_empty() {
        return Err(WasmBuildError::Unsupported(
            "wasmtime trace did not contain any currently supported wasm rows".to_string(),
        ));
    }

    Ok(out)
}
