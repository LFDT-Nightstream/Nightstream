//! Suspended import frames: authoritative context RAM, activation boundaries,
//! and resumption. Capture depth is not an input to these constraints.

use crate::gadgets::push_gated_linear_zero;
use crate::host_event_bindings::IMPORT_SCHEDULE_COUNT_FACTOR;
use crate::host_event_layout::HOST_CONTEXT_HALF_FACTOR;
use crate::ir::WasmVmStep;
use crate::layout::*;
use crate::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use neo_application::ZeroTest;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const HOST_DEPTH_ZERO: ZeroTest = ZeroTest::column(
    COL_HOST_FRAME_DEPTH_BEFORE,
    COL_HOST_FRAME_DEPTH_INV,
    COL_HOST_FRAME_DEPTH_IS_ZERO,
);
const RETURN_COUNT_ZERO: ZeroTest = ZeroTest::column(
    COL_HOST_RETURN_COUNT,
    COL_HOST_RETURN_COUNT_INV,
    COL_HOST_RETURN_COUNT_IS_ZERO,
);

pub(super) fn push_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(super::always("host activation lifecycle"), |b| {
        // A split call owes exactly the next row; no gather, guest instruction,
        // boundary, or padding may bypass the context save.
        b.push_linear_zero([(COL_HOST_SUSPEND, F::ONE), (COL_HOST_SUSPEND_PENDING_BEFORE, -F::ONE)]);
        b.push_linear_zero([(COL_HOST_SUSPEND_PENDING_AFTER, F::ONE), (COL_HOST_SPLIT_CALL, -F::ONE)]);
        // Two bounded 16-bit halves uniquely decode a u32 ROM word in F.
        b.push_linear_zero([
            (COL_HOST_IMPORT_SCHEDULE_COUNTS, F::ONE),
            (COL_HOST_INITIAL_COUNT, -F::ONE),
            (COL_HOST_RETURN_COUNT, -F::from_u64(IMPORT_SCHEDULE_COUNT_FACTOR.into())),
        ]);
        // At most one of these gates is 1 on a row, so each sum is a gate.
        b.push_row(
            [
                (COL_HOST_CALL_ACTIVE, F::ONE),
                (COL_HOST_RESUME, F::ONE),
                (COL_TURN_BOUNDARY, F::ONE),
                (COL_HOST_EVENT_EXIT_LATCH, F::ONE),
            ],
            [
                (COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT, F::ONE),
                (COL_HOST_INITIAL_COUNT, -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [
                (crate::layout::COL_HOST_EVENT_EXIT_SCHEDULE_COUNT, F::ONE),
                (COL_HOST_RETURN_COUNT, -F::ONE),
            ],
            [],
        );
        HOST_DEPTH_ZERO.push_constraints(b);
        RETURN_COUNT_ZERO.push_constraints(b);
        b.push_row(
            [(COL_HOST_CALL_ACTIVE, F::ONE)],
            [(COL_ONE, F::ONE), (COL_HOST_RETURN_COUNT_IS_ZERO, -F::ONE)],
            [(COL_HOST_SPLIT_CALL, F::ONE)],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE)],
            [(COL_HOST_AWAITING_BEFORE, F::ONE)],
            [(COL_HOST_NESTED_ENTRY, F::ONE)],
        );
        b.push_row(
            [(COL_HOST_EVENT_EXIT_LATCH, F::ONE)],
            [(COL_ONE, F::ONE), (COL_HOST_FRAME_DEPTH_IS_ZERO, -F::ONE)],
            [(COL_HOST_NESTED_EXIT, F::ONE)],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_RESUME, F::ONE)],
            [(COL_ACTIVATION_FINISHED_BEFORE, F::ONE)],
            [(COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE)],
        );
        b.push_linear_zero([
            (COL_HOST_CONTEXT_READ, F::ONE),
            (COL_HOST_NESTED_ENTRY, -F::ONE),
            (COL_HOST_RESUME, -F::ONE),
        ]);
        for (depth, floor, guest) in [
            (
                COL_CALL_STACK_DEPTH_BEFORE,
                COL_HOST_FRAME_DEPTH_BEFORE,
                COL_GUEST_DEPTH_BEFORE,
            ),
            (
                COL_CALL_STACK_DEPTH_AFTER,
                COL_HOST_FRAME_DEPTH_AFTER,
                COL_GUEST_DEPTH_AFTER,
            ),
        ] {
            // All three are u32; this also prevents popping below the host frame.
            b.push_linear_zero([(depth, F::ONE), (floor, -F::ONE), (guest, -F::ONE)]);
        }
        b.push_linear_zero([
            (COL_HOST_AWAITING_AFTER, F::ONE),
            (COL_HOST_AWAITING_BEFORE, -F::ONE),
            (COL_HOST_SUSPEND, -F::ONE),
            (COL_HOST_NESTED_EXIT, -F::ONE),
            (COL_HOST_NESTED_ENTRY, F::ONE),
            (COL_HOST_RESUME, F::ONE),
        ]);
        push_gated_linear_zero(b, COL_IS_PROGRAM_ROW, [(COL_HOST_AWAITING_BEFORE, F::ONE)]);
        push_gated_linear_zero(
            b,
            COL_HOST_CONTEXT_READ,
            [(COL_HOST_AWAITING_BEFORE, F::ONE), (COL_ONE, -F::ONE)],
        );
        push_gated_linear_zero(b, COL_HOST_CONTEXT_READ, [(COL_GUEST_DEPTH_BEFORE, F::ONE)]);
        push_gated_linear_zero(b, COL_HOST_CONTEXT_READ, [(COL_HOST_FRAME_DEPTH_IS_ZERO, F::ONE)]);
        push_gated_linear_zero(b, COL_HOST_CONTEXT_READ, [(COL_HOST_EVENTS_REMAINING_BEFORE, F::ONE)]);
        push_gated_linear_zero(b, COL_HOST_CONTEXT_READ, [(COL_OBJECT_ACTIVE_BEFORE, F::ONE)]);
        push_gated_linear_zero(
            b,
            COL_HOST_SUSPEND,
            [
                (COL_HOST_FRAME_DEPTH_AFTER, F::ONE),
                (COL_CALL_STACK_DEPTH_AFTER, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            COL_HOST_RESUME,
            [
                (COL_HOST_FRAME_DEPTH_AFTER, F::ONE),
                (COL_HOST_CONTEXT_PARENT_DEPTH, -F::ONE),
            ],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (COL_HOST_SUSPEND, -F::ONE),
                (COL_HOST_RESUME, -F::ONE),
            ],
            [
                (COL_HOST_FRAME_DEPTH_AFTER, F::ONE),
                (COL_HOST_FRAME_DEPTH_BEFORE, -F::ONE),
            ],
            [],
        );
        // Root entry is only possible with no suspended work.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_HOST_FRAME_DEPTH_BEFORE, F::ONE)],
            [],
        );

        push_gated_linear_zero(
            b,
            COL_HOST_SUSPEND,
            [(COL_HOST_CONTEXT_ADDR, F::ONE), (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            COL_HOST_CONTEXT_READ,
            [
                (COL_HOST_CONTEXT_ADDR, F::ONE),
                (COL_HOST_FRAME_DEPTH_BEFORE, -F::ONE),
                (COL_ONE, F::ONE),
            ],
        );
        // Rocq `ProgramBinding.full_host_context_refs` and `full_host_context_args_depth`:
        // each RAM word binds two bounded fields.
        b.push_linear_zero([
            (COL_HOST_CONTEXT_REFS, F::ONE),
            (COL_HOST_CONTEXT_EXPORT, -F::ONE),
            (COL_HOST_CONTEXT_IMPORT, -F::from_u32(HOST_CONTEXT_HALF_FACTOR)),
        ]);
        b.push_linear_zero([
            (COL_HOST_CONTEXT_ARGS_DEPTH, F::ONE),
            (COL_HOST_CONTEXT_ARGS_BASE, -F::ONE),
            (COL_HOST_CONTEXT_PARENT_DEPTH, -F::from_u32(HOST_CONTEXT_HALF_FACTOR)),
        ]);
        for (field, source) in [
            (COL_HOST_CONTEXT_FRAME[0], COL_PC_AFTER),
            (COL_HOST_CONTEXT_FRAME[1], COL_LOCALS_FBP_BEFORE),
            (COL_HOST_CONTEXT_FRAME[2], COL_STACK_FRAME_BASE_BEFORE),
            (COL_HOST_CONTEXT_EXPORT, COL_TURN_EXPORT_FREF_BEFORE),
            (COL_HOST_CONTEXT_IMPORT, COL_HOST_CALLEE_FREF_BEFORE),
            (COL_HOST_CONTEXT_ARGS_BASE, COL_HOST_EVENT_ARGS_BASE_AFTER),
            (COL_HOST_CONTEXT_PARENT_DEPTH, COL_HOST_FRAME_DEPTH_BEFORE),
            (COL_HOST_CONTEXT_SP, COL_SP_AFTER),
        ] {
            push_gated_linear_zero(b, COL_HOST_SUSPEND, [(field, F::ONE), (source, -F::ONE)]);
        }
        // Entry starts its frame above every suspended cell, including args.
        for (target, source) in [
            (COL_LOCALS_FBP_AFTER, COL_HOST_CONTEXT_SP),
            (COL_SP_BEFORE, COL_HOST_CONTEXT_SP),
        ] {
            push_gated_linear_zero(b, COL_HOST_NESTED_ENTRY, [(target, F::ONE), (source, -F::ONE)]);
        }
        for (target, source) in [
            (COL_PC_AFTER, COL_HOST_CONTEXT_FRAME[0]),
            (COL_LOCALS_FBP_AFTER, COL_HOST_CONTEXT_FRAME[1]),
            (COL_STACK_FRAME_BASE_AFTER, COL_HOST_CONTEXT_FRAME[2]),
            (COL_TURN_EXPORT_FREF_AFTER, COL_HOST_CONTEXT_EXPORT),
            (COL_HOST_CALLEE_FREF_AFTER, COL_HOST_CONTEXT_IMPORT),
            (COL_HOST_EVENT_ARGS_BASE_AFTER, COL_HOST_CONTEXT_ARGS_BASE),
            (COL_SP_AFTER, COL_HOST_CONTEXT_ARGS_BASE),
            (COL_SP_BEFORE, COL_HOST_CONTEXT_SP),
        ] {
            push_gated_linear_zero(b, COL_HOST_RESUME, [(target, F::ONE), (source, -F::ONE)]);
        }
        push_gated_linear_zero(b, COL_HOST_RESUME, [(COL_HOST_RETURN_COUNT_IS_ZERO, F::ONE)]);
        push_gated_linear_zero(
            b,
            COL_HOST_RESUME,
            [
                (COL_HOST_EVENTS_REMAINING_AFTER, F::ONE),
                (COL_HOST_RETURN_COUNT, -F::ONE),
                (COL_ONE, F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            COL_HOST_RESUME,
            [
                (COL_HOST_EVENT_INDEX_AFTER, F::ONE),
                (crate::layout::COL_HOST_EVENT_BASE, -F::ONE),
                (COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT, -F::ONE),
                (COL_ONE, F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            COL_HOST_RESUME,
            [
                (COL_HOST_RESUME_POP, F::ONE),
                (COL_SP_BEFORE, -F::ONE),
                (COL_SP_AFTER, F::ONE),
            ],
        );
        push_gated_linear_zero(b, COL_HOST_RESUME, [(COL_ACTIVATION_FINISHED_AFTER, F::ONE)]);
        // COL_HOST_RESUME_POP is part (unconditionally) of the stack balance
        // equation so it MUST be 0 for other opcodes
        b.push_row(
            [(COL_ONE, F::ONE), (COL_HOST_RESUME, -F::ONE)],
            [(COL_HOST_RESUME_POP, F::ONE)],
            [],
        );
        for col in [
            COL_OUTPUT_ENABLED_AFTER,
            COL_OUTPUT_VALUE_LO_AFTER,
            COL_OUTPUT_VALUE_HI_AFTER,
        ] {
            push_gated_linear_zero(b, COL_HOST_RESUME, [(col, F::ONE)]);
        }
    });
}

pub(crate) fn fill_witness(wit: &mut [F], trace: &WasmVmStep) {
    let flag = |v: bool| if v { F::ONE } else { F::ZERO };
    let before = trace.state_before;
    let after = trace.state_after;
    wit[COL_HOST_FRAME_DEPTH_BEFORE] = F::from_u64(before.host_events.host_frame_depth);
    wit[COL_HOST_FRAME_DEPTH_AFTER] = F::from_u64(after.host_events.host_frame_depth);
    wit[COL_HOST_AWAITING_BEFORE] = flag(before.host_events.awaiting_host);
    wit[COL_HOST_AWAITING_AFTER] = flag(after.host_events.awaiting_host);
    wit[COL_HOST_SUSPEND_PENDING_BEFORE] = flag(before.host_events.suspend_pending);
    wit[COL_HOST_SUSPEND_PENDING_AFTER] = flag(after.host_events.suspend_pending);
    wit[COL_HOST_SUSPEND] = flag(trace.row_kind.is_host_suspend());
    // Subtraction in the field permits negative-test witnesses; range checks reject them.
    wit[COL_GUEST_DEPTH_BEFORE] = F::from_u64(before.call_stack_depth) - wit[COL_HOST_FRAME_DEPTH_BEFORE];
    wit[COL_GUEST_DEPTH_AFTER] = F::from_u64(after.call_stack_depth) - wit[COL_HOST_FRAME_DEPTH_AFTER];
    wit[COL_HOST_RETURN_COUNT] = F::from_u64(
        trace
            .host_event_return_schedule_count
            .or(trace.host_event_exit_schedule_count)
            .unwrap_or(0)
            .into(),
    );
    if wit[COL_HOST_CALL_ACTIVE] == F::ONE
        || trace.row_kind.is_host_resume()
        || wit[COL_TURN_BOUNDARY] == F::ONE
        || wit[COL_HOST_EVENT_EXIT_LATCH] == F::ONE
    {
        wit[COL_HOST_INITIAL_COUNT] = wit[COL_HOST_EVENT_INITIAL_SCHEDULE_COUNT];
    }
    wit[COL_HOST_IMPORT_SCHEDULE_COUNTS] =
        wit[COL_HOST_INITIAL_COUNT] + F::from_u64(IMPORT_SCHEDULE_COUNT_FACTOR.into()) * wit[COL_HOST_RETURN_COUNT];
    HOST_DEPTH_ZERO.assign(wit);
    RETURN_COUNT_ZERO.assign(wit);
    wit[COL_HOST_SPLIT_CALL] = wit[COL_HOST_CALL_ACTIVE] * (F::ONE - wit[COL_HOST_RETURN_COUNT_IS_ZERO]);
    wit[COL_HOST_RESUME] = flag(trace.row_kind.is_host_resume());
    wit[COL_HOST_NESTED_ENTRY] = flag(trace.row_kind.is_turn_boundary() && before.host_events.awaiting_host);
    wit[COL_HOST_NESTED_EXIT] = wit[COL_HOST_EVENT_EXIT_LATCH] * (F::ONE - wit[COL_HOST_FRAME_DEPTH_IS_ZERO]);
    wit[COL_HOST_CONTEXT_READ] = wit[COL_HOST_RESUME] + wit[COL_HOST_NESTED_ENTRY];
    wit[COL_HOST_CLEAR_ACTIVATION_FINISHED] =
        (wit[COL_TURN_BOUNDARY] + wit[COL_HOST_RESUME]) * flag(before.activation_finished);
    if let Some(context) = trace.host_context {
        wit[COL_HOST_CONTEXT_EXPORT] = F::from_u32(context.export_fref);
        wit[COL_HOST_CONTEXT_IMPORT] = F::from_u32(context.import_fref);
        wit[COL_HOST_CONTEXT_FRAME[0]] = F::from_u64(context.continuation_pc);
        wit[COL_HOST_CONTEXT_FRAME[1]] = F::from_u64(context.locals_fbp);
        wit[COL_HOST_CONTEXT_FRAME[2]] = F::from_u64(context.stack_frame_base);
        wit[COL_HOST_CONTEXT_REFS] =
            wit[COL_HOST_CONTEXT_EXPORT] + F::from_u32(HOST_CONTEXT_HALF_FACTOR) * wit[COL_HOST_CONTEXT_IMPORT];
        wit[COL_HOST_CONTEXT_ARGS_BASE] = F::from_u64(context.args_base);
        wit[COL_HOST_CONTEXT_PARENT_DEPTH] = F::from_u64(context.parent_host_frame_depth);
        wit[COL_HOST_CONTEXT_ARGS_DEPTH] = wit[COL_HOST_CONTEXT_ARGS_BASE]
            + F::from_u32(HOST_CONTEXT_HALF_FACTOR) * wit[COL_HOST_CONTEXT_PARENT_DEPTH];
        wit[COL_HOST_CONTEXT_SP] = F::from_u64(context.retained_sp);
        wit[COL_HOST_CONTEXT_ADDR] = if wit[COL_HOST_SUSPEND] == F::ONE {
            F::from_u64(before.call_stack_depth)
        } else {
            wit[COL_HOST_FRAME_DEPTH_BEFORE] - F::ONE
        };
    }
    if trace.row_kind.is_host_resume() {
        wit[COL_HOST_RESUME_POP] = F::from_u64(before.sp) - F::from_u64(after.sp);
    }
}
