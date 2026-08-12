//! Assembly of normalized `NormalizedStep`s into proof-facing `WasmVmStep`
//! rows.
//!
//! Owns the cross-row work: the running trace state machine (sp, fbp, call
//! stack and param-init trackers), consistency checks between
//! adjacent rows, and the synthesis of aux rows that do not exist in
//! wasmtime's step stream — guest `CallParamInit` and host-event gather
//! rows. Per-row decoding lives in the parent
//! `normalize` module; this module reads it only through `NormalizedStep`.

mod tail_call;
mod turn;
mod values;

use self::turn::{plan_turn_exit, setup_turn};
use self::values::{call_indirect_oob, call_indirect_traps, collect_callee_initial_params, write_lane, write_lane_hi};
use super::super::runtime_read::{read_lane, read_lane_hi};
use super::super::WasmtimeTraceStep;
use super::grammar_emit::{emit_block_plan, plan_import_call, GrammarAuxCtx, GrammarBlockPlan, GrammarCallPlan};
use super::memory::LinearMemoryImage;
use super::normalize_step;
use crate::comm_chain::CommChainState;
use crate::event_grammar::HostEventGrammar;
use crate::ir::{
    StackValueAccess, WasmAuxOpcode, WasmBuildError, WasmCountdownState, WasmEventAbsorbState, WasmOutputState,
    WasmPcEdgeKind, WasmRowKind, WasmStepState, WasmVmStep,
};
use crate::isa::{opcode_code, opcode_info_from_code, WasmOpcode};

pub(super) fn build_trace(
    rows: &[WasmtimeTraceStep],
    grammar: Option<(&HostEventGrammar, &[crate::event_grammar::TurnClaims])>,
    initial_comm_chain: CommChainState,
    linear_memory: Option<LinearMemoryImage>,
) -> Result<Vec<WasmVmStep>, WasmBuildError> {
    let mut supported = Vec::new();
    for row in rows {
        if let Some(normalized) = normalize_step(row)? {
            supported.push(normalized);
        }
    }
    // Import-free callers get the canonical single-shot grammar: an empty
    // boundary template for the invoked export, so nothing is absorbed and
    // host imports have no template to prove against.
    let import_free_grammar;
    let import_free_turns;
    let (grammar_tables, turns) = match grammar {
        Some((tables, turns)) => (tables, turns),
        None => {
            let export_fref = supported
                .first()
                .and_then(|first| first.current_function_ref)
                .unwrap_or(0);
            import_free_grammar = HostEventGrammar::import_free(export_fref);
            import_free_turns = [crate::event_grammar::TurnClaims::default()];
            (&import_free_grammar, &import_free_turns[..])
        }
    };
    let tracks_linear_memory = linear_memory.is_some();
    let mut linear_memory = linear_memory.unwrap_or_default();
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
    let mut tail_call_pending = false;
    // Callee attribution carry: set on host-call rows, preserved everywhere
    // else (no clearing — see `WasmStepState::host_callee_fref`).
    let mut host_callee_fref: u32 = 0;
    // Host-event commitment chain; each absorbed block folds it forward.
    let mut comm_chain = initial_comm_chain.canonical_u64();
    // Host-event absorb machinery (block buffer and perm group
    // state); carried across rows, mutated only by host-call events.
    let mut event_absorb = WasmEventAbsorbState::ZERO;
    // Grammar gather machinery state (schedule, args base, cursor).
    let mut grammar_state = crate::ir::WasmGrammarState::ZERO;
    // The verifier mirrors the first turn's entry schedule in
    // `grammar_top_level_initial_state`.
    let mut turn_index = 0usize;
    let mut export_boundary = if let Some(first) = supported.first() {
        let claims = turns.first().ok_or_else(|| {
            WasmBuildError::Trace("event grammar requires claim words for at least the first turn".to_string())
        })?;
        let setup = setup_turn(grammar_tables, first, claims, false, &mut linear_memory)?;
        grammar_state = crate::ir::WasmGrammarState {
            turn_export_fref: setup.fref,
            events_remaining: setup.entry_plans.len() as u32,
            event_index: 0,
            args_base: 0,
            slot_cursor: 0,
        };
        host_callee_fref = setup.fref;
        Some(setup)
    } else {
        None
    };
    let mut entry_emitted = false;
    // Carried halt latch; a turn boundary clears it for re-entry.
    let mut turn_done = false;
    let mut output_enabled = false;
    let mut output_value_lo = 0;
    let mut output_value_hi = 0;

    for (idx, current) in supported.iter().enumerate() {
        let next = supported.get(idx + 1);
        let pc_before = u64::from(current.pc);
        // A return-like edge with no caller ends the current export invocation.
        let turn_terminal = current.pc_edge_kind == WasmPcEdgeKind::ReturnLike && call_stack.is_empty();
        let halted = next.is_none() || turn_terminal;
        // Halting rows keep their one-past pc (terminal traps have no next
        // row; a turn boundary bridges to the next turn's entry pc).
        let pc_after = if halted {
            current
                .pc_after_instruction
                .unwrap_or_else(|| pc_before.saturating_add(1))
        } else {
            next.map(|row| u64::from(row.pc)).unwrap_or_else(|| {
                current
                    .pc_after_instruction
                    .unwrap_or_else(|| pc_before.saturating_add(1))
            })
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
            && !call_stack.is_empty();
        let next_row_base = if is_return_row {
            call_stack.last().map(|&(_, _, base)| base).unwrap_or(0)
        } else {
            stack_base
        };
        let sp_after = if is_call_row || halted {
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
        // Model the host consuming a captured result. (Mutable: a grammar
        // host call overrides this below to pop its args on the call row.)
        let mut sp_after = sp_after - u64::from(output_captured);

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
        let current_stack_base = stack_base;
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
                    call_stack_push = Some((return_pc, current_fbp, current_stack_base));
                    call_stack_pop = None;
                    callee_initial_params = collect_callee_initial_params(next, callee_fbp, param_count);
                    guest_callee_fbp = Some(callee_fbp);
                    call_stack.push((return_pc, current_fbp, current_stack_base));
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
                let callee_fbp = current_fbp
                    .checked_add(u64::from(current.num_locals))
                    .ok_or_else(|| {
                        WasmBuildError::Trace(format!("callee frame base overflow at cycle {}", current.cycle))
                    })?;
                call_stack_push = None;
                call_stack_pop = None;
                callee_initial_params = collect_callee_initial_params(next, callee_fbp, param_count);
                guest_callee_fbp = Some(callee_fbp);
                fbp = callee_fbp;
            }
            WasmOpcode::Return | WasmOpcode::End if is_return_row => {
                // Non-final return: restore the caller's FBP and operand-stack
                // base from the call stack.
                let (ret_pc, caller_fbp, caller_base) = call_stack.pop().unwrap();
                call_stack_push = None;
                call_stack_pop = Some((ret_pc, caller_fbp, caller_base));
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
        if guest_callee_fbp.is_some() {
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
        let tail_call_pending_before = tail_call_pending;
        let tail_call_pending_after =
            matches!(current.opcode, WasmOpcode::ReturnCall | WasmOpcode::ReturnCallIndirect) && !ci_trap;

        // Host calls enter host-arg mode and may owe a result push; the aux
        // rows emitted below walk both back to zero before the next program
        // row.
        // Emit entry events before the turn's first program row.
        if !entry_emitted {
            entry_emitted = true;
            if let Some(setup) = &export_boundary {
                let mut ctx = GrammarAuxCtx {
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
                    current_function_num_locals: current.num_locals,
                    halted: turn_done,
                };
                for plan in &setup.entry_plans {
                    emit_block_plan(
                        &mut out,
                        &mut ctx,
                        &mut comm_chain,
                        &mut event_absorb,
                        &mut grammar_state,
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
        let mut grammar_plan: Option<GrammarCallPlan> = None;
        let grammar_state_before_row = grammar_state;
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
            let result_limbs = if result_count == 1 {
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
            let template = grammar_tables
                .imports
                .get(&host_callee_fref)
                .ok_or_else(|| {
                    WasmBuildError::Trace(format!(
                        "no grammar template for host import fref {host_callee_fref} at cycle {}",
                        current.cycle
                    ))
                })?;
            template.validate(param_count, result_count)?;
            let args_base = sp_before - index_pops as u64 - u64::from(param_count);
            grammar_plan = Some(
                plan_import_call(
                    template,
                    args_base,
                    &arg_limbs,
                    result_limbs,
                    &current.host_call_claims,
                    &mut linear_memory,
                )
                .map_err(|err| {
                    WasmBuildError::Trace(format!(
                        "host call to fref {host_callee_fref} at cycle {}: {err}",
                        current.cycle
                    ))
                })?,
            );
            // The call row pops all args; a result gather row pushes the
            // result later in the schedule.
            sp_after = args_base;
        }
        // The call row latches the event schedule and argument-region base.
        if let Some(plan) = &grammar_plan {
            grammar_state = crate::ir::WasmGrammarState {
                turn_export_fref: grammar_state.turn_export_fref,
                events_remaining: plan.blocks.len() as u32,
                event_index: 0,
                args_base: plan.args_base,
                slot_cursor: 0,
            };
        }
        // A clean export halt loads the exit schedule and attribution. Result
        // publication is optional: constant-only exit events also apply to
        // resultless exports.
        let mut exit_plans: Option<Vec<GrammarBlockPlan>> = None;
        let mut exit_counts: Option<(u32, u32)> = None;
        if halted && !trapped {
            if let Some(setup) = &export_boundary {
                let plans = plan_turn_exit(
                    setup.template,
                    current,
                    &turns[turn_index],
                    output_captured.then_some((output_value_lo_after, output_value_hi_after)),
                    &linear_memory,
                )?;
                host_callee_fref = setup.fref;
                grammar_state = crate::ir::WasmGrammarState {
                    turn_export_fref: grammar_state.turn_export_fref,
                    events_remaining: plans.len() as u32,
                    event_index: setup.template.entry.len() as u32,
                    args_base: grammar_state.args_base,
                    slot_cursor: 0,
                };
                exit_counts = Some((setup.template.entry.len() as u32, setup.template.exit.len() as u32));
                exit_plans = Some(plans);
            }
        }

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
                halted: turn_done,
                trapped: false,
                param_init: param_init_before,
                tail_call_pending: tail_call_pending_before,
                host_callee_fref: host_callee_fref_before,
                comm_chain: comm_chain_before_row,
                event_absorb: event_absorb_before_row,
                grammar: grammar_state_before_row,
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
                halted: turn_done || halted,
                trapped,
                param_init: param_init_after,
                tail_call_pending: tail_call_pending_after,
                host_callee_fref,
                // The chain only moves on permutation rows.
                comm_chain,
                event_absorb,
                grammar: grammar_state,
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
            grammar_rom_slot: None,
            // Count cells carry the presence bias (count + 1).
            grammar_pre_count: grammar_plan
                .as_ref()
                .map(|plan| plan.blocks.len() as u32 + 1)
                .or(exit_counts.map(|(pre, _)| pre + 1)),
            grammar_post_count: exit_counts.map(|(_, post)| post),
        });
        turn_done = turn_done || halted;
        param_init_state = param_init_after;
        tail_call_pending = tail_call_pending_after;
        if is_call_row && !callee_initial_params.is_empty() {
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
            let index_pops = usize::from(matches!(
                current.opcode,
                WasmOpcode::CallIndirect | WasmOpcode::ReturnCallIndirect
            ));
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
                out.push(WasmVmStep {
                    cycle: out.len() as u64,
                    row_kind: WasmRowKind::Aux(WasmAuxOpcode::CallParamInit),
                    state_before: WasmStepState {
                        pc: pc_after,
                        sp: aux_sp_before,
                        stack_frame_base: stack_base,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        max_memory_pages: current.max_memory_pages,
                        locals_fbp: callee_fbp,
                        halted: turn_done,
                        trapped: false,
                        param_init: aux_param_init_before,
                        tail_call_pending,
                        host_callee_fref,
                        comm_chain,
                        event_absorb,
                        grammar: grammar_state,
                    },
                    state_after: WasmStepState {
                        pc: pc_after,
                        sp: aux_sp_after,
                        stack_frame_base: stack_base,
                        output: WasmOutputState {
                            enabled: output_enabled,
                            value_lo: output_value_lo,
                            value_hi: output_value_hi,
                        },
                        call_stack_depth: call_stack_depth_after,
                        memory_pages: current.memory_pages_after,
                        max_memory_pages: current.max_memory_pages,
                        locals_fbp: callee_fbp,
                        halted: turn_done,
                        trapped: false,
                        param_init: aux_param_init_after,
                        tail_call_pending,
                        host_callee_fref,
                        comm_chain,
                        event_absorb,
                        grammar: grammar_state,
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
                    grammar_rom_slot: None,
                    grammar_pre_count: None,
                    grammar_post_count: None,
                });
                debug_assert_eq!(
                    aux_param_init_before.remaining,
                    u32::try_from(remaining_before).expect("remaining fits u32")
                );
                param_init_state = aux_param_init_after;
            }
        }
        if tail_call_pending {
            let param_count = u64::from(current.call_param_count.unwrap_or(0));
            let tail_sp_before = sp_after.checked_sub(param_count).ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "operand stack underflow after tail-call parameter initialization at cycle {}",
                    current.cycle
                ))
            })?;
            if tail_sp_before < stack_base {
                return Err(WasmBuildError::Trace(format!(
                    "tail-call frame base {} exceeds stack pointer {} at cycle {}",
                    stack_base, tail_sp_before, current.cycle
                )));
            }
            let state_before = WasmStepState {
                pc: pc_after,
                sp: tail_sp_before,
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
                halted: turn_done,
                trapped: false,
                param_init: WasmCountdownState::ZERO,
                tail_call_pending: true,
                host_callee_fref,
                comm_chain,
                event_absorb,
                grammar: grammar_state,
            };
            let mut state_after = state_before;
            state_after.sp = stack_base;
            state_after.tail_call_pending = false;
            out.push(tail_call::tail_enter_row(
                out.len() as u64,
                state_before,
                state_after,
                current.function_ref.unwrap_or(0),
            ));
            tail_call_pending = false;
        }
        if let Some(plan) = &grammar_plan {
            let mut ctx = GrammarAuxCtx {
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
                current_function_num_locals: current.num_locals,
                halted: turn_done,
            };
            for block_plan in &plan.blocks {
                emit_block_plan(
                    &mut out,
                    &mut ctx,
                    &mut comm_chain,
                    &mut event_absorb,
                    &mut grammar_state,
                    block_plan,
                );
            }
            let gather_pushes = if plan.blocks.is_empty() {
                0
            } else {
                u64::from(current.call_result_count.unwrap_or(0))
            };
            debug_assert_eq!(ctx.sp, sp_after + gather_pushes);
        }

        // Export boundary: the exit template's blocks absorb after the
        // capture row (its own after-state carries the exit latch).
        if let Some(plans) = exit_plans {
            let mut ctx = GrammarAuxCtx {
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
                current_function_num_locals: current.num_locals,
                halted: turn_done,
            };
            for plan in &plans {
                emit_block_plan(
                    &mut out,
                    &mut ctx,
                    &mut comm_chain,
                    &mut event_absorb,
                    &mut grammar_state,
                    plan,
                );
            }
        }

        // Bridge to the next export and load its entry attribution and schedule.
        if halted && next.is_some() {
            let next_row = next.expect("checked");
            turn_index += 1;
            let claims = turns.get(turn_index).ok_or_else(|| {
                WasmBuildError::Trace(format!(
                    "trace re-enters turn {} but only {} turn claim sets were supplied",
                    turn_index + 1,
                    turns.len()
                ))
            })?;
            let setup = setup_turn(grammar_tables, next_row, claims, true, &mut linear_memory)?;
            let entry_count = setup.entry_plans.len() as u32;
            let boundary_state = |pc: u64,
                                  sp: u64,
                                  stack_frame_base: u64,
                                  output: WasmOutputState,
                                  host_fref: u32,
                                  gstate: crate::ir::WasmGrammarState,
                                  done: bool| {
                WasmStepState {
                    pc,
                    sp,
                    stack_frame_base,
                    output,
                    call_stack_depth: 0,
                    memory_pages: current.memory_pages_after,
                    max_memory_pages: current.max_memory_pages,
                    locals_fbp: fbp,
                    halted: done,
                    trapped: false,
                    param_init: WasmCountdownState::ZERO,
                    tail_call_pending: false,
                    host_callee_fref: host_fref,
                    comm_chain,
                    event_absorb,
                    grammar: gstate,
                }
            };
            let state_before = boundary_state(
                pc_after,
                sp_after,
                stack_base,
                WasmOutputState {
                    enabled: output_enabled_after,
                    value_lo: output_value_lo_after,
                    value_hi: output_value_hi_after,
                },
                host_callee_fref,
                grammar_state,
                true,
            );
            host_callee_fref = setup.fref;
            grammar_state = crate::ir::WasmGrammarState {
                turn_export_fref: setup.fref,
                events_remaining: entry_count,
                event_index: 0,
                args_base: grammar_state.args_base,
                slot_cursor: 0,
            };
            let state_after = boundary_state(
                u64::from(next_row.pc),
                0,
                0,
                WasmOutputState::ZERO,
                host_callee_fref,
                grammar_state,
                false,
            );
            let helper_ctx = GrammarAuxCtx {
                pc: pc_after,
                sp: sp_after,
                stack_frame_base: stack_base,
                output: WasmOutputState::ZERO,
                call_stack_depth: 0,
                memory_pages: current.memory_pages_after,
                max_memory_pages: current.max_memory_pages,
                locals_fbp: fbp,
                host_callee_fref,
                current_function_ref: current.current_function_ref.unwrap_or(0),
                current_function_num_locals: current.num_locals,
                halted: true,
            };
            out.push(WasmVmStep {
                // Export entry-count cell carries the presence bias.
                grammar_pre_count: Some(entry_count + 1),
                ..helper_ctx.row(out.len() as u64, WasmAuxOpcode::TurnBoundary, state_before, state_after)
            });
            turn_done = false;
            output_enabled = false;
            output_value_lo = 0;
            output_value_hi = 0;
            stack_base = 0;
            export_boundary = Some(setup);
            entry_emitted = false;
        }
    }

    if out.is_empty() {
        return Err(WasmBuildError::Unsupported(
            "wasmtime trace did not contain any currently supported wasm rows".to_string(),
        ));
    }
    if turn_index + 1 != turns.len() {
        return Err(WasmBuildError::Trace(format!(
            "trace ran {} turn(s) but {} turn claim sets were supplied",
            turn_index + 1,
            turns.len()
        )));
    }

    Ok(out)
}
