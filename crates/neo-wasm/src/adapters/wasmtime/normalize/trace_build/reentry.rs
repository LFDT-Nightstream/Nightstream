//! Normalizer-owned suspended activations and boundary rows. Saved values are
//! separately authenticated by the relation's context RAM.

use super::super::host_event_emit::{emit_block_plan, plan_import_phase, HostEventRowContext};
use super::super::{memory::LinearMemoryImage, NormalizedStep};
use super::turn::{setup_turn, TurnSetup};
use crate::host_event_bindings::HostEventBindings;
use crate::ir::{WasmAuxOpcode, WasmBuildError, WasmCountdownState, WasmOutputState, WasmStepState, WasmVmStep};

pub(super) struct Suspended<'a> {
    pub context: crate::ir::WasmHostContext,
    pub call_index: usize,
    pub activation_depth: usize,
    pub export: TurnSetup<'a>,
}

fn row_context(state: WasmStepState, current: &NormalizedStep) -> HostEventRowContext {
    HostEventRowContext {
        pc: state.pc,
        sp: state.sp,
        stack_frame_base: state.stack_frame_base,
        output: state.output,
        call_stack_depth: state.call_stack_depth,
        memory_pages: state.memory_pages,
        max_memory_pages: state.max_memory_pages,
        locals_fbp: state.locals_fbp,
        host_callee_fref: state.host_callee_fref,
        current_function_ref: current.current_function_ref.unwrap_or(0),
        current_function_num_locals: current.num_locals,
        activation_finished: state.activation_finished,
    }
}

pub(super) fn suspend(
    out: &mut Vec<WasmVmStep>,
    call: &NormalizedStep,
    context: crate::ir::WasmHostContext,
) -> Result<WasmStepState, WasmBuildError> {
    let before = out.last().expect("suspend follows split call").state_after;
    let mut after = before;
    after.call_stack_depth = before
        .call_stack_depth
        .checked_add(1)
        .ok_or_else(|| WasmBuildError::Trace("host frame depth overflows".into()))?;
    after.host_events.host_frame_depth = after.call_stack_depth;
    after.host_events.awaiting_host = true;
    after.host_events.suspend_pending = false;
    out.push(WasmVmStep {
        host_context: Some(context),
        ..row_context(before, call).row(out.len() as u64, WasmAuxOpcode::HostSuspend, before, after)
    });
    Ok(after)
}

pub(super) fn enter<'a>(
    out: &mut Vec<WasmVmStep>,
    current: &NormalizedStep,
    next: &NormalizedStep,
    bindings: &'a HostEventBindings,
    program: Option<&crate::WasmProgramTables>,
    memory: &mut LinearMemoryImage,
    suspended: Option<&Suspended<'a>>,
) -> Result<TurnSetup<'a>, WasmBuildError> {
    let setup = setup_turn(bindings, next, program, true, memory)?;
    let before = out
        .last()
        .expect("entry follows a call or exit")
        .state_after;
    let mut after = before;
    after.pc = u64::from(next.pc);
    after.output = WasmOutputState::ZERO;
    after.activation_finished = false;
    after.host_callee_fref = setup.fref;
    after.host_events.turn_export_fref = setup.fref;
    after.host_events.awaiting_host = false;
    after.host_events.events_remaining = setup.entry_plans.len() as u32;
    after.host_events.event_index = 0;
    after.host_events.slot_cursor = 0;
    if let Some(suspended) = suspended {
        after.locals_fbp = suspended.context.locals_end;
        after.stack_frame_base = suspended.context.retained_sp;
        if before.sp != suspended.context.retained_sp {
            return Err(WasmBuildError::Trace(
                "reentrant guest invocation did not preserve suspended operand storage".into(),
            ));
        }
    } else {
        after.sp = 0;
        after.stack_frame_base = 0;
    }
    let remaining = next
        .num_locals
        .checked_sub(u32::from(setup.param_count))
        .ok_or_else(|| WasmBuildError::Trace("export has more parameters than locals".into()))?;
    after.local_zero = WasmCountdownState {
        active: remaining != 0,
        remaining,
    };
    out.push(WasmVmStep {
        call_param_count: Some(setup.param_count),
        call_result_count: Some(setup.result_count),
        target_function_is_guest: true,
        host_context: suspended.map(|saved| saved.context),
        host_event_initial_schedule_count: Some(setup.entry_plans.len() as u32 + 1),
        host_event_exit_schedule_count: Some(setup.template.exit.len() as u32),
        ..row_context(before, current).row(out.len() as u64, WasmAuxOpcode::TurnBoundary, before, after)
    });
    Ok(setup)
}

pub(super) fn resume(
    out: &mut Vec<WasmVmStep>,
    saved: &Suspended<'_>,
    call: &NormalizedStep,
    next: &NormalizedStep,
    bindings: &HostEventBindings,
    memory: &mut LinearMemoryImage,
) -> Result<(), WasmBuildError> {
    if next.activation_depth != saved.activation_depth
        || u64::from(next.pc) != saved.context.continuation_pc
        || next.current_function_ref != call.current_function_ref
    {
        return Err(WasmBuildError::Trace(
            "split import did not resume at its caller continuation".into(),
        ));
    }
    let before = out
        .last()
        .expect("resume follows captured execution")
        .state_after;
    if before.memory_pages != next.memory_pages_before {
        return Err(WasmBuildError::Unsupported(
            "host memory growth across a split import is unsupported".into(),
        ));
    }
    let template = &bindings.imports[&saved.context.import_fref];
    let count = usize::from(call.call_param_count.unwrap_or(0));
    let end = call.operand_stack.len() - usize::from(call.opcode == crate::WasmOpcode::CallIndirect);
    let args: Vec<_> = (end - count..end)
        .map(|i| (call.operand_stack[i], call.operand_stack_hi[i]))
        .collect();
    let inputs = super::super::super::import_inputs::recover_import_inputs(
        template,
        &args,
        next.memory_pages_before,
        call.host_call_memory.as_ref(),
    )?;
    let result = if call.call_result_count == Some(1) {
        Some((
            *next
                .operand_stack
                .last()
                .ok_or_else(|| WasmBuildError::Trace("missing import result".into()))?,
            next.operand_stack_hi.last().copied().unwrap_or(0),
        ))
    } else {
        None
    };
    let plan = plan_import_phase(
        template.return_events.as_deref().expect("split import"),
        saved.context.args_base,
        &args,
        result,
        &inputs,
        next.memory_pages_before,
        memory,
    )?;
    let mut after = before;
    after.pc = saved.context.continuation_pc;
    after.locals_fbp = saved.context.locals_fbp;
    after.stack_frame_base = saved.context.stack_frame_base;
    after.host_events.turn_export_fref = saved.context.export_fref;
    after.host_callee_fref = saved.context.import_fref;
    after.host_events.args_base = saved.context.args_base;
    after.host_events.host_frame_depth = saved.context.parent_host_frame_depth;
    after.host_events.awaiting_host = false;
    after.sp = saved.context.args_base;
    after.call_stack_depth -= 1;
    after.activation_finished = false;
    after.output = WasmOutputState::ZERO;
    after.host_events.event_index = template.events.len() as u32;
    after.host_events.events_remaining = plan.blocks.len() as u32;
    after.host_events.slot_cursor = 0;
    out.push(WasmVmStep {
        host_context: Some(saved.context),
        host_event_initial_schedule_count: Some(template.events.len() as u32 + 1),
        host_event_return_schedule_count: Some(template.return_schedule_count()),
        ..row_context(before, call).row(out.len() as u64, WasmAuxOpcode::HostResume, before, after)
    });
    let mut ctx = row_context(after, call);
    for block in &plan.blocks {
        emit_block_plan(
            out,
            &mut ctx,
            &mut after.comm_chain,
            &mut after.event_absorb,
            &mut after.host_events,
            block,
        );
    }
    Ok(())
}
