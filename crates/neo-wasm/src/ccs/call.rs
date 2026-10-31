//! Owns call entry, return-context RAM, frame bases, local initialization,
//! frame moves for returns and tail calls, and host-import attribution.

use super::super::gadgets::push_gated_linear_zero;
use super::super::ir::{FUNCTION_CALL_METADATA_GUEST_FACTOR, FUNCTION_CALL_METADATA_RESULT_FACTOR};
use super::super::isa::WasmOpcode;
use super::super::layout::{
    selector_col, COL_ACTIVATION_FINISHED_AFTER, COL_ACTIVATION_FINISHED_BEFORE, COL_CALL_INDIRECT_IS_NOT_TRAP,
    COL_CALL_INDIRECT_IS_TRAP, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR,
    COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_CALLER_SP_BASE_VALUE, COL_CALL_STACK_DEPTH_AFTER,
    COL_CALL_STACK_DEPTH_BEFORE, COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT,
    COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA, COL_CI_HOST_CALL, COL_ENTERED_FUNCTION_NUM_LOCALS,
    COL_EXIT_SLIDE_DISTANCE, COL_FRAME_MOVE_ACTIVE_AFTER, COL_FRAME_MOVE_ACTIVE_BEFORE, COL_FRAME_MOVE_DISTANCE_AFTER,
    COL_FRAME_MOVE_DISTANCE_BEFORE, COL_FRAME_MOVE_REMAINING_AFTER, COL_FRAME_MOVE_REMAINING_AFTER_INV,
    COL_FRAME_MOVE_REMAINING_AFTER_IS_ZERO, COL_FRAME_MOVE_REMAINING_BEFORE, COL_FRAME_SLIDE_ACTIVE, COL_FRAME_SP_DROP,
    COL_FUNCTION_REF, COL_GATHER_ACTIVE, COL_GUEST_ENTRY_ACTIVE, COL_HOST_CALLEE_FREF_AFTER,
    COL_HOST_CALLEE_FREF_BEFORE, COL_HOST_EVENT_EXIT_LATCH, COL_IS_PROGRAM_ROW, COL_LOCALS_FBP_AFTER,
    COL_LOCALS_FBP_BEFORE, COL_LOCAL_ZERO_ACTIVE_AFTER, COL_LOCAL_ZERO_ACTIVE_BEFORE, COL_LOCAL_ZERO_REMAINING_AFTER,
    COL_LOCAL_ZERO_REMAINING_AFTER_INV, COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO, COL_LOCAL_ZERO_REMAINING_BEFORE,
    COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_OUTPUT_CAPTURED, COL_OUTPUT_ENABLED_AFTER,
    COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_AFTER,
    COL_OUTPUT_VALUE_LO_BEFORE, COL_PADDING_ACTIVE, COL_PC_AFTER, COL_PC_BEFORE, COL_PC_ROM_CALL_RETURN_CHOICE,
    COL_PERM_PENDING_BEFORE, COL_PERM_ROUND_BEFORE_IS_ZERO, COL_SP_AFTER, COL_SP_BEFORE, COL_STACK_FRAME_BASE_AFTER,
    COL_STACK_FRAME_BASE_BEFORE, COL_STACK_READS, COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI,
    COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
    COL_STACK_WRITES, COL_TABLE_INDEX, COL_TABLE_VALUE, COL_TARGET_FUNCTION_IS_GUEST, COL_TRAPPED_AFTER,
    COL_TRAPPED_BEFORE, COL_TURN_BOUNDARY, PC_ROM_CALL_RETURN_CHOICE,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::always;
use crate::layout::{
    COL_HOST_CLEAR_ACTIVATION_FINISHED, COL_HOST_FRAME_DEPTH_BEFORE, COL_HOST_NESTED_ENTRY, COL_HOST_RESUME,
    COL_HOST_SUSPEND,
};
use neo_application::ZeroTest;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub(crate) const LOCAL_ZERO_REMAINING_AFTER_ZERO_TEST: ZeroTest = ZeroTest::column(
    COL_LOCAL_ZERO_REMAINING_AFTER,
    COL_LOCAL_ZERO_REMAINING_AFTER_INV,
    COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO,
);

pub(crate) const FRAME_MOVE_REMAINING_AFTER_ZERO_TEST: ZeroTest = ZeroTest::column(
    COL_FRAME_MOVE_REMAINING_AFTER,
    COL_FRAME_MOVE_REMAINING_AFTER_INV,
    COL_FRAME_MOVE_REMAINING_AFTER_IS_ZERO,
);

/// Emit every call/frame row the wasm VM needs in a single place. Ordering
/// inside follows the natural lifecycle of a call: row-kind classification →
/// aux-row shape → local zeroing and frame moves → return-pc restoration →
/// frame transitions → dynamic call-arity lookups.
pub(super) fn push_call_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(always("function call metadata"), |b| {
        b.push_linear_zero([
            (COL_CALL_TARGET_METADATA, F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
            (
                COL_CALL_RESULT_COUNT,
                -F::from_u64(FUNCTION_CALL_METADATA_RESULT_FACTOR),
            ),
            (
                COL_TARGET_FUNCTION_IS_GUEST,
                -F::from_u64(FUNCTION_CALL_METADATA_GUEST_FACTOR),
            ),
        ]);
    });

    b.with_tag(always("row kind one hot"), |b| {
        // The host-event perm row kind is the derived flag
        // `perm_pending_before + (perm_round_before != 0)`; writing the sum
        // as `... + pending - round_is_zero = 0` folds its `+1` into the
        // one-hot's `-1`. `pending = 1 ∧ round != 0` would double-count, but
        // is unreachable: every row that raises `pending` provably lands
        // `round_after = 0` (see `ccs/poseidon.rs`).
        b.push_linear_zero([
            (COL_IS_PROGRAM_ROW, F::ONE),
            (COL_FRAME_MOVE_ACTIVE_BEFORE, F::ONE),
            (COL_LOCAL_ZERO_ACTIVE_BEFORE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
            (COL_TURN_BOUNDARY, F::ONE),
            (COL_HOST_RESUME, F::ONE),
            (COL_HOST_SUSPEND, F::ONE),
            (COL_PERM_PENDING_BEFORE, F::ONE),
            (COL_PERM_ROUND_BEFORE_IS_ZERO, -F::ONE),
        ]);
    });

    b.with_tag(always("padding row state preservation"), |b| {
        // Padding rows are synthetic state-preserving placeholders used
        // to round a trace up to a multiple of the F'-shell batch size.
        // Force every state column the cross-step links care about to
        // satisfy `_after == _before`, so a padding row is a true fixed
        // point for the chain. pc/sp are already pinned via the shared
        // "non-program row shape" tag below.
        let padding_gate = COL_PADDING_ACTIVE;
        push_gated_linear_zero(
            b,
            padding_gate,
            [(COL_MEMORY_PAGES_AFTER, F::ONE), (COL_MEMORY_PAGES_BEFORE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [(COL_LOCALS_FBP_AFTER, F::ONE), (COL_LOCALS_FBP_BEFORE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_STACK_FRAME_BASE_AFTER, F::ONE),
                (COL_STACK_FRAME_BASE_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_CALL_STACK_DEPTH_AFTER, F::ONE),
                (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_FRAME_MOVE_ACTIVE_AFTER, F::ONE),
                (COL_FRAME_MOVE_ACTIVE_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_FRAME_MOVE_REMAINING_AFTER, F::ONE),
                (COL_FRAME_MOVE_REMAINING_BEFORE, -F::ONE),
            ],
        );
    });

    push_simple_output_constraints(b);

    b.with_tag(always("non-program row shape"), |b| {
        // Aux rows keep pc fixed. Padding rows read and write nothing;
        // frame-move rows copy one cell each. Host-event perm rows read
        // and write nothing too; their rows live in `ccs/host_event_chain.rs`. The
        // pc-pin gate folds the perm-row flag `pending + 1 - round_is_zero`
        // in directly.
        let aux_row_gate_with_perm = [
            (COL_HOST_SUSPEND, F::ONE),
            (COL_FRAME_MOVE_ACTIVE_BEFORE, F::ONE),
            (COL_LOCAL_ZERO_ACTIVE_BEFORE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
            (COL_PERM_PENDING_BEFORE, F::ONE),
            (COL_ONE, F::ONE),
            (COL_PERM_ROUND_BEFORE_IS_ZERO, -F::ONE),
        ];

        b.push_row(
            aux_row_gate_with_perm,
            [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE)],
            [],
        );
    });

    b.with_tag(always("guest call flag"), |b| {
        push_guest_call_flag_constraints(b);
    });

    b.with_tag(always("local zero initialization"), |b| {
        push_local_zero_constraints(b);
    });

    b.with_tag(always("frame move"), |b| {
        push_frame_move_constraints(b);
    });

    b.with_tag(always("host call attribution"), |b| {
        push_host_call_attribution_constraints(b);
    });

    b.with_tag(always("activation completion"), |b| {
        // Entry and resume clear activation completion. Other rows preserve
        // the latch; program rows cannot execute while it is set.
        push_gated_linear_zero(b, COL_TURN_BOUNDARY, [(COL_ACTIVATION_FINISHED_AFTER, F::ONE)]);
        // Root entry requires a finished turn; nested entry requires suspension.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_ONE, F::ONE), (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE)],
            [],
        );
    });

    b.with_tag(always("turn boundary row"), |b| {
        // Root entry requires empty operand and call stacks and starts a
        // fresh frame at the stack bottom. Other state-machine constraints
        // require spent event schedules and an idle permutation.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_SP_BEFORE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_CALL_STACK_DEPTH_BEFORE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_LOCALS_FBP_AFTER, F::ONE)],
            [],
        );
    });

    b.with_tag(always("return pc restoration"), |b| {
        b.push_row(
            [(COL_CALL_STACK_POP_PRESENT, F::ONE)],
            [(COL_PC_AFTER, F::ONE), (COL_CALL_STACK_RETURN_PC_VALUE, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_CALL_STACK_POP_PRESENT, F::ONE)],
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::Return).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::End).unwrap(), -F::ONE),
            ],
            [],
        );
    });

    b.with_tag(always("call stack transition"), |b| {
        push_call_stack_transition_constraints(b);
    });

    b.with_tag(always("stack frame base transition"), |b| {
        push_stack_frame_base_transition_constraints(b);
    });

    b.with_tag(always("locals fbp transition"), |b| {
        push_locals_fbp_transition_constraints(b);
    });

    b.with_tag(always("dynamic call stack arity"), |b| {
        push_dynamic_call_stack_arity_constraints(b);
    });
}

fn push_simple_output_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.with_tag(always("simple output carry"), |b| {
        for (after, before) in [
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE),
            (COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE),
            (COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE),
        ] {
            // Activation completion and boundary rows own output transitions.
            b.push_row(
                [
                    (COL_ONE, F::ONE),
                    (COL_ACTIVATION_FINISHED_AFTER, -F::ONE),
                    (COL_TURN_BOUNDARY, -F::ONE),
                    (COL_HOST_RESUME, -F::ONE),
                ],
                [(after, F::ONE), (before, -F::ONE)],
                [],
            );
            // Carry captured output until a boundary. A resultless boundary
            // preserves the already-zero state.
            b.push_row(
                [
                    (COL_OUTPUT_ENABLED_BEFORE, F::ONE),
                    (COL_TURN_BOUNDARY, -F::ONE),
                    (COL_HOST_RESUME, -F::ONE),
                ],
                [(after, F::ONE), (before, -F::ONE)],
                [],
            );
        }

        // Capture raises the flag; a boundary clears it only when set.
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_RESUME, F::ONE)],
            [(COL_OUTPUT_ENABLED_BEFORE, F::ONE)],
            [
                (COL_OUTPUT_ENABLED_BEFORE, F::ONE),
                (COL_OUTPUT_CAPTURED, F::ONE),
                (COL_OUTPUT_ENABLED_AFTER, -F::ONE),
            ],
        );
        // The re-armed output is zeroed.
        push_gated_linear_zero(b, COL_TURN_BOUNDARY, [(COL_OUTPUT_VALUE_LO_AFTER, F::ONE)]);
        push_gated_linear_zero(b, COL_TURN_BOUNDARY, [(COL_OUTPUT_VALUE_HI_AFTER, F::ONE)]);
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_ONE, F::ONE), (COL_ACTIVATION_FINISHED_AFTER, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_OUTPUT_ENABLED_BEFORE, F::ONE)],
            [],
        );
        // A clean activation return captures exactly its function's results:
        // the frame above fbp less the return's static slide distance. It drops
        // its whole frame. The boundary term cancels the activation-finished
        // latch reset between turns.
        b.push_row(
            clean_activation_return_terms(),
            [
                (COL_SP_BEFORE, F::ONE),
                (COL_LOCALS_FBP_BEFORE, -F::ONE),
                (COL_EXIT_SLIDE_DISTANCE, -F::ONE),
                (COL_OUTPUT_CAPTURED, -F::ONE),
            ],
            [],
        );
        b.push_row(
            clean_activation_return_terms(),
            [(COL_SP_AFTER, F::ONE), (COL_LOCALS_FBP_BEFORE, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [
                (COL_STACK_READ_ADDR_LO[0], F::ONE),
                (COL_SP_BEFORE, -F::from_u64(2)),
                (COL_ONE, F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [
                (COL_OUTPUT_VALUE_LO_AFTER, F::ONE),
                (COL_STACK_READ_VALUE_LO[0], -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [
                (COL_OUTPUT_VALUE_HI_AFTER, F::ONE),
                (COL_STACK_READ_VALUE_HI[0], -F::ONE),
            ],
            [],
        );
    });
}

/// One exactly on a row that cleanly finishes the current activation.
fn clean_activation_return_terms() -> [(usize, F); 5] {
    [
        (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
        (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
        (COL_TRAPPED_AFTER, -F::ONE),
        (COL_TRAPPED_BEFORE, F::ONE),
        (COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
    ]
}

fn push_dynamic_call_stack_arity_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();
    let return_call = selector_col(WasmOpcode::ReturnCall).unwrap();
    let return_call_indirect = selector_col(WasmOpcode::ReturnCallIndirect).unwrap();

    // Call rows read nothing for direct calls, or only the table index for
    // indirect calls (trapping ones included). Guest args stay in place as
    // the callee's params; host args are read by gather rows and accounted
    // for in the global sp identity.
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (return_call, F::ONE),
            (return_call_indirect, F::ONE),
        ],
        [
            (COL_STACK_READS, F::ONE),
            (call_indirect, -F::ONE),
            (return_call_indirect, -F::ONE),
        ],
        [],
    );
    push_gated_linear_zero(
        b,
        COL_CALL_INDIRECT_IS_NOT_TRAP,
        [(COL_FUNCTION_REF, F::ONE), (COL_TABLE_VALUE, -F::ONE)],
    );
    // Bind the table read to the index popped from the stack top.
    b.push_row(
        [(call_indirect, F::ONE), (return_call_indirect, F::ONE)],
        [(COL_TABLE_INDEX, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
        [],
    );
    b.push_row(
        [(call_indirect, F::ONE), (return_call_indirect, F::ONE)],
        [
            (COL_STACK_READ_ADDR_LO[0], F::ONE),
            (COL_SP_BEFORE, -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
        [],
    );

    // Call rows never write: guest results land on the matching Return/End,
    // and host results land on gather rows.
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (return_call, F::ONE),
            (return_call_indirect, F::ONE),
        ],
        [(COL_STACK_WRITES, F::ONE)],
        [],
    );
}

/// Successful call-like rows minus guest entries. Supported tail calls are
/// guest-only, so they cancel and only ordinary host calls remain.
pub(super) fn host_call_gate_terms() -> [(usize, F); 4] {
    [
        (selector_col(WasmOpcode::Call).unwrap(), F::ONE),
        (selector_col(WasmOpcode::ReturnCall).unwrap(), F::ONE),
        (COL_CALL_INDIRECT_IS_NOT_TRAP, F::ONE),
        (COL_GUEST_ENTRY_ACTIVE, -F::ONE),
    ]
}

fn push_host_call_attribution_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    // Indirect host calls fall through to the instruction after the call.
    // Direct calls get pc_after from the static pc ROM edge, and guest
    // indirect calls from the `function_entries` binding. Indirect host calls
    // use DynamicCallIndirect, so pc_after is bound through the call site's
    // return-pc slot, gated on `ci_not_trap * (1 - is_guest)`.
    b.push_row(
        [(COL_CALL_INDIRECT_IS_NOT_TRAP, F::ONE)],
        [(COL_ONE, F::ONE), (COL_TARGET_FUNCTION_IS_GUEST, -F::ONE)],
        [(COL_CI_HOST_CALL, F::ONE)],
    );
    push_gated_linear_zero(
        b,
        COL_CI_HOST_CALL,
        [
            // force the control-choice of the next pc lookup to equal
            // PC_ROM_CALL_RETURN_CHOICE
            //
            // this is sound because the synthetic aux opcodes don't allow
            // changing the pc, so we can set it in advance before entering the
            // host call mode
            (COL_PC_ROM_CALL_RETURN_CHOICE, F::ONE),
            (COL_ONE, -F::from_u64(PC_ROM_CALL_RETURN_CHOICE)),
        ],
    );

    // Callee attribution carry: a host call latches the (ROM/table-bound)
    // callee fref; every other row — program, aux, padding — preserves it.
    // Consumers (the event absorb) read it only on rows of the event that
    // set it, so the stale value between events is inert.
    b.push_row(
        host_call_gate_terms(),
        [(COL_HOST_CALLEE_FREF_AFTER, F::ONE), (COL_FUNCTION_REF, -F::ONE)],
        [],
    );
    b.push_row(
        [
            (COL_ONE, F::ONE),
            (selector_col(WasmOpcode::Call).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::ReturnCall).unwrap(), -F::ONE),
            (COL_CALL_INDIRECT_IS_NOT_TRAP, -F::ONE),
            (COL_GUEST_ENTRY_ACTIVE, F::ONE),
            // ...and host-event boundaries repoint attribution to the current
            // or next turn's export.
            (COL_HOST_EVENT_EXIT_LATCH, -F::ONE),
            (COL_TURN_BOUNDARY, -F::ONE),
            (COL_HOST_RESUME, -F::ONE),
        ],
        [
            (COL_HOST_CALLEE_FREF_AFTER, F::ONE),
            (COL_HOST_CALLEE_FREF_BEFORE, -F::ONE),
        ],
        [],
    );
}

fn push_guest_call_flag_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();
    let return_call = selector_col(WasmOpcode::ReturnCall).unwrap();
    let return_call_indirect = selector_col(WasmOpcode::ReturnCallIndirect).unwrap();

    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (return_call, F::ONE),
            (return_call_indirect, F::ONE),
        ],
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE)],
        [(COL_GUEST_ENTRY_ACTIVE, F::ONE)],
    );
    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE)],
        [(COL_CALL_STACK_PUSH_PRESENT, F::ONE)],
    );
    push_gated_linear_zero(
        b,
        return_call,
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE), (COL_ONE, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        return_call_indirect,
        [
            (COL_TARGET_FUNCTION_IS_GUEST, F::ONE),
            (COL_ONE, -F::ONE),
            (COL_CALL_INDIRECT_IS_TRAP, F::ONE),
        ],
    );
}

fn push_local_zero_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let zero = COL_LOCAL_ZERO_ACTIVE_BEFORE;
    LOCAL_ZERO_REMAINING_AFTER_ZERO_TEST.push_constraints(b);
    // Zeroing waits for a pending frame move: tail-call arguments must land
    // before the zero pushes can overwrite their source cells.
    b.push_row(
        [(COL_ONE, F::ONE), (COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO, -F::ONE)],
        [(COL_ONE, F::ONE), (COL_FRAME_MOVE_ACTIVE_AFTER, -F::ONE)],
        [(COL_LOCAL_ZERO_ACTIVE_AFTER, F::ONE)],
    );
    push_gated_linear_zero(
        b,
        zero,
        [
            (COL_LOCAL_ZERO_REMAINING_BEFORE, F::ONE),
            (COL_LOCAL_ZERO_REMAINING_AFTER, -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    // Each row pushes the next declared local as a zero.
    push_gated_linear_zero(
        b,
        zero,
        [(COL_STACK_WRITE0_ADDR_LO, F::ONE), (COL_SP_BEFORE, -F::from_u64(2))],
    );
    push_gated_linear_zero(b, zero, [(COL_STACK_WRITE0_VALUE_LO, F::ONE)]);
    push_gated_linear_zero(b, zero, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);
    // Only a guest call or turn boundary may seed the verifier-bound count.
    b.push_row(
        [
            (COL_ONE, F::ONE),
            (COL_GUEST_ENTRY_ACTIVE, -F::ONE),
            (COL_TURN_BOUNDARY, -F::ONE),
            (zero, -F::ONE),
        ],
        [
            (COL_LOCAL_ZERO_REMAINING_AFTER, F::ONE),
            (COL_LOCAL_ZERO_REMAINING_BEFORE, -F::ONE),
        ],
        [],
    );
    b.push_row(
        [(COL_GUEST_ENTRY_ACTIVE, F::ONE), (COL_TURN_BOUNDARY, F::ONE)],
        [
            (COL_LOCAL_ZERO_REMAINING_AFTER, F::ONE),
            (COL_CALL_PARAM_COUNT, F::ONE),
            (COL_ENTERED_FUNCTION_NUM_LOCALS, -F::ONE),
        ],
        [],
    );
}

/// A return moves the callee's results down to its frame base, and a tail
/// call moves its arguments down to the replaced frame's base. The seeding
/// row lowers sp to the moved region's end; each move row then copies cell
/// `sp - remaining + distance` to `sp - remaining`. Copies run bottom-up, so
/// overlapping regions are safe.
fn push_frame_move_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let slide = COL_FRAME_SLIDE_ACTIVE;
    let moving = COL_FRAME_MOVE_ACTIVE_BEFORE;
    // Guest entries that save no return context are tail calls.
    let tail = [(COL_GUEST_ENTRY_ACTIVE, F::ONE), (COL_CALL_STACK_PUSH_PRESENT, -F::ONE)];
    let not_seeding = [
        (COL_ONE, F::ONE),
        (slide, -F::ONE),
        (COL_GUEST_ENTRY_ACTIVE, -F::ONE),
        (COL_CALL_STACK_PUSH_PRESENT, F::ONE),
    ];

    FRAME_MOVE_REMAINING_AFTER_ZERO_TEST.push_constraints(b);
    b.push_linear_zero([
        (COL_FRAME_MOVE_ACTIVE_AFTER, F::ONE),
        (COL_FRAME_MOVE_REMAINING_AFTER_IS_ZERO, F::ONE),
        (COL_ONE, -F::ONE),
    ]);

    // Branches and popping returns slide: the taken edge's static countdown and
    // distance are ROM reads into the move registers, and sp drops by the distance.
    b.push_linear_zero([
        (slide, F::ONE),
        (selector_col(WasmOpcode::Br).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::BrIf).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::BrTable).unwrap(), -F::ONE),
        (COL_CALL_STACK_POP_PRESENT, -F::ONE),
    ]);
    push_gated_linear_zero(
        b,
        slide,
        [(COL_FRAME_SP_DROP, F::ONE), (COL_FRAME_MOVE_DISTANCE_AFTER, -F::ONE)],
    );
    // A tail call keeps its arguments and drops the replaced frame.
    b.push_row(
        tail,
        [
            (COL_FRAME_MOVE_REMAINING_AFTER, F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
        ],
        [],
    );
    b.push_row(
        tail,
        [
            (COL_SP_AFTER, F::ONE),
            (COL_LOCALS_FBP_BEFORE, -F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
        ],
        [],
    );
    b.push_row(
        tail,
        [
            (COL_FRAME_MOVE_DISTANCE_AFTER, F::ONE),
            (COL_SP_BEFORE, -F::ONE),
            (COL_STACK_READS, F::ONE),
            (COL_SP_AFTER, F::ONE),
        ],
        [],
    );
    // Only frame transitions drop cells outside the stack balance; their
    // rows pin sp_after directly.
    b.push_row(
        not_seeding
            .into_iter()
            .chain([(COL_TURN_BOUNDARY, -F::ONE)])
            .chain(
                clean_activation_return_terms()
                    .into_iter()
                    .map(|(column, coefficient)| (column, -coefficient)),
            ),
        [(COL_FRAME_SP_DROP, F::ONE)],
        [],
    );

    push_gated_linear_zero(
        b,
        moving,
        [
            (COL_FRAME_MOVE_REMAINING_BEFORE, F::ONE),
            (COL_FRAME_MOVE_REMAINING_AFTER, -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        moving,
        [
            (COL_STACK_WRITE0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -F::from_u64(2)),
            (COL_FRAME_MOVE_REMAINING_BEFORE, F::from_u64(2)),
        ],
    );
    push_gated_linear_zero(
        b,
        moving,
        [
            (COL_STACK_READ_ADDR_LO[0], F::ONE),
            (COL_STACK_WRITE0_ADDR_LO, -F::ONE),
            (COL_FRAME_MOVE_DISTANCE_BEFORE, -F::from_u64(2)),
        ],
    );
    push_gated_linear_zero(
        b,
        moving,
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_STACK_READ_VALUE_LO[0], -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        moving,
        [
            (COL_STACK_WRITE0_VALUE_HI, F::ONE),
            (COL_STACK_READ_VALUE_HI[0], -F::ONE),
        ],
    );
    // Move rows and state-preserving padding keep the distance.
    b.push_row(
        [(moving, F::ONE), (COL_PADDING_ACTIVE, F::ONE)],
        [
            (COL_FRAME_MOVE_DISTANCE_AFTER, F::ONE),
            (COL_FRAME_MOVE_DISTANCE_BEFORE, -F::ONE),
        ],
        [],
    );
    // Every other row preserves the countdown and clears the distance.
    b.push_row(
        not_seeding.into_iter().chain([(moving, -F::ONE)]),
        [
            (COL_FRAME_MOVE_REMAINING_AFTER, F::ONE),
            (COL_FRAME_MOVE_REMAINING_BEFORE, -F::ONE),
        ],
        [],
    );
    b.push_row(
        not_seeding
            .into_iter()
            .chain([(moving, -F::ONE), (COL_PADDING_ACTIVE, -F::ONE)]),
        [(COL_FRAME_MOVE_DISTANCE_AFTER, F::ONE)],
        [],
    );
}

fn push_call_stack_transition_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let push = COL_CALL_STACK_PUSH_PRESENT;
    let pop = COL_CALL_STACK_POP_PRESENT;

    // Push increments the return-context stack, pop decrements it, and every
    // other row preserves it. Range-checking on the depth columns rules out
    // underflow in the bounded witness model.
    b.push_linear_zero([
        (COL_CALL_STACK_DEPTH_AFTER, F::ONE),
        (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE),
        (push, -F::ONE),
        (pop, F::ONE),
        (COL_HOST_SUSPEND, -F::ONE),
        (COL_HOST_RESUME, F::ONE),
    ]);

    push_gated_linear_zero(
        b,
        push,
        [(COL_CALL_STACK_ADDR, F::ONE), (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (COL_PC_ROM_CALL_RETURN_CHOICE, F::ONE),
            (COL_ONE, -F::from_u64(PC_ROM_CALL_RETURN_CHOICE)),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [(COL_CALL_STACK_ADDR, F::ONE), (COL_CALL_STACK_DEPTH_AFTER, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (COL_CALL_STACK_CALLER_FBP_VALUE, F::ONE),
            (COL_LOCALS_FBP_BEFORE, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (COL_CALL_STACK_CALLER_SP_BASE_VALUE, F::ONE),
            (COL_STACK_FRAME_BASE_BEFORE, -F::ONE),
        ],
    );
    // A clean return finishes the activation at its outermost frame, but a trap terminates
    // immediately and may leave abandoned caller frames on the call stack.
    b.push_row(
        [(COL_ACTIVATION_FINISHED_AFTER, F::ONE), (COL_TRAPPED_AFTER, -F::ONE)],
        [
            (COL_CALL_STACK_DEPTH_BEFORE, F::ONE),
            (COL_HOST_FRAME_DEPTH_BEFORE, -F::ONE),
        ],
        [],
    );
}

fn push_stack_frame_base_transition_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let entry = [(COL_GUEST_ENTRY_ACTIVE, F::ONE), (COL_TURN_BOUNDARY, F::ONE)];
    let pop = COL_CALL_STACK_POP_PRESENT;

    // A new frame's operands start above its locals.
    b.push_row(
        entry,
        [
            (COL_STACK_FRAME_BASE_AFTER, F::ONE),
            (COL_LOCALS_FBP_AFTER, -F::ONE),
            (COL_ENTERED_FUNCTION_NUM_LOCALS, -F::ONE),
        ],
        [],
    );
    push_gated_linear_zero(
        b,
        pop,
        [
            (COL_STACK_FRAME_BASE_AFTER, F::ONE),
            (COL_CALL_STACK_CALLER_SP_BASE_VALUE, -F::ONE),
        ],
    );
    b.push_row(
        [
            (COL_ONE, F::ONE),
            (COL_GUEST_ENTRY_ACTIVE, -F::ONE),
            (COL_TURN_BOUNDARY, -F::ONE),
            (pop, -F::ONE),
            (COL_HOST_RESUME, -F::ONE),
        ],
        [
            (COL_STACK_FRAME_BASE_AFTER, F::ONE),
            (COL_STACK_FRAME_BASE_BEFORE, -F::ONE),
        ],
        [],
    );
}

fn push_locals_fbp_transition_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let push = COL_CALL_STACK_PUSH_PRESENT;
    let pop = COL_CALL_STACK_POP_PRESENT;

    // A called frame starts at its arguments; a tail call reuses its frame.
    push_gated_linear_zero(
        b,
        push,
        [
            (COL_LOCALS_FBP_AFTER, F::ONE),
            (COL_SP_AFTER, -F::ONE),
            (COL_CALL_PARAM_COUNT, F::ONE),
        ],
    );
    // An entered frame reserves its parameters for the entry inputs.
    push_gated_linear_zero(
        b,
        COL_TURN_BOUNDARY,
        [
            (COL_SP_AFTER, F::ONE),
            (COL_LOCALS_FBP_AFTER, -F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [
            (COL_LOCALS_FBP_AFTER, F::ONE),
            (COL_CALL_STACK_CALLER_FBP_VALUE, -F::ONE),
        ],
    );
    b.push_row(
        [
            (COL_ONE, F::ONE),
            (push, -F::ONE),
            (pop, -F::ONE),
            (COL_HOST_RESUME, -F::ONE),
            (COL_TURN_BOUNDARY, -F::ONE),
        ],
        [(COL_LOCALS_FBP_AFTER, F::ONE), (COL_LOCALS_FBP_BEFORE, -F::ONE)],
        [],
    );
}
