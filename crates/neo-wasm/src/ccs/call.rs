//! Owns the per-step call / frame / parameter-initialization CCS rows.
//!
//! Wasm guest calls split into three coupled mechanisms:
//!
//! 1. **Call-stack frame plumbing**: `frame.locals_fbp_*` advances on
//!    push (enter callee) and rewinds on pop (return to caller);
//!    `state.pc_after` is restored from `call.call_stack_pop_return_pc`
//!    on returns.
//! 2. **Parameter initialization**: a guest call pushes
//!    `function_types.param_count` parameters onto the stack, which the
//!    callee must copy into its first `param_count` locals before
//!    executing its body. We model that as a sequence of synthetic
//!    "aux" rows immediately after the call row, each writing one
//!    parameter and decrementing `param_init.param_init_remaining_*`
//!    until it hits zero. `param_init_active_*` flags whether the
//!    current row is one of these aux rows.
//! 3. **Call arity**: `control.stack_{reads,writes}` is fixed for most
//!    opcodes by the static one-hot decode, but `Call` and
//!    `CallIndirect` are dynamic — their arities come from the callee's
//!    declared type via the `function_types` lookup family.

use super::super::gadgets::{push_gated_linear_zero, push_zero_test_gadget};
use super::super::isa::WasmOpcode;
use super::super::ivc_state::StateColumns;
use super::super::layout::{
    selector_col, CALL_RETURN_PC_CHOICE, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_FUNCTION_REF,
    COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE,
    COL_OUTPUT_CAPTURED, COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER,
    COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE, COL_STACK_READ0_ADDR_LO,
    COL_STACK_READ0_VALUE_HI, COL_STACK_READ0_VALUE_LO, COL_TABLE_INDEX, COL_TABLE_VALUE, COL_TARGET_FUNCTION_IS_GUEST,
};
use super::super::lookup_binding_builder::{
    CallColumns, ControlColumns, FrameColumns, ParamInitColumns, WasmLookupBindingLayout,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::{always, idx};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

/// Emit every call/frame/param-init row the wasm VM needs in a single
/// place. Ordering inside follows the natural lifecycle of a call:
/// row-kind classification → aux-row shape → enter/exit param init →
/// per-aux-row witness shape → return-pc restoration → frame fbp
/// transition → dynamic call-arity lookups.
pub(super) fn push_call_constraints(b: &mut R1csBuilder, layout: &WasmLookupBindingLayout) {
    let control = layout.control;
    let state = layout.state;
    let param_init = layout.param_init;
    let call = layout.call;
    let frame = layout.frame;

    b.with_tag(always("row kind one hot"), |b| {
        b.push_linear_zero([
            (idx(control.is_program_row), F::ONE),
            (idx(param_init.param_init_active_before), F::ONE),
            (idx(control.padding_active), F::ONE),
            (COL_ONE, -F::ONE),
        ]);
    });

    b.with_tag(always("padding row state preservation"), |b| {
        // Padding rows are synthetic state-preserving placeholders used
        // to round a trace up to a multiple of the F'-shell batch size.
        // Force every state column the cross-step links care about to
        // satisfy `_after == _before`, so a padding row is a true fixed
        // point for the chain. pc/sp are already pinned via the shared
        // "non-program row shape" tag below.
        let padding_gate = idx(control.padding_active);
        push_gated_linear_zero(
            b,
            padding_gate,
            [(COL_MEMORY_PAGES_AFTER, F::ONE), (COL_MEMORY_PAGES_BEFORE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (idx(frame.locals_fbp_after), F::ONE),
                (idx(frame.locals_fbp_before), -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (idx(call.call_stack_depth_after), F::ONE),
                (idx(call.call_stack_depth_before), -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (idx(param_init.param_init_active_after), F::ONE),
                (idx(param_init.param_init_active_before), -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (idx(param_init.param_init_remaining_after), F::ONE),
                (idx(param_init.param_init_remaining_before), -F::ONE),
            ],
        );
    });

    push_simple_output_constraints(b, &control, &state);

    b.with_tag(always("non-program row shape"), |b| {
        // Aux rows keep pc fixed and write no stack values. Padding rows read
        // nothing; param-init rows pop one arg slot.
        let aux_row_gate = [
            (idx(param_init.param_init_active_before), F::ONE),
            (idx(control.padding_active), F::ONE),
        ];

        b.push_row(
            aux_row_gate,
            [(idx(state.pc_after), F::ONE), (idx(state.pc_before), -F::ONE)],
            [],
        );
        b.push_row(
            [(idx(control.padding_active), F::ONE)],
            [(idx(control.stack_reads), F::ONE)],
            [],
        );
        push_gated_linear_zero(
            b,
            idx(param_init.param_init_active_before),
            [(idx(control.stack_reads), F::ONE), (COL_ONE, -F::ONE)],
        );
        b.push_row(aux_row_gate, [(idx(control.stack_writes), F::ONE)], []);

        let param_init_row_gate = idx(param_init.param_init_active_before);

        // Param-init writes the popped stack value into the callee local.
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(COL_STACK_READ0_VALUE_LO, F::ONE), (COL_LOCAL_VALUE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(COL_STACK_READ0_VALUE_HI, F::ONE), (COL_LOCAL_VALUE_HI, -F::ONE)],
        );
    });

    b.with_tag(always("guest call flag"), |b| {
        push_guest_call_flag_constraints(b, &call);
    });

    b.with_tag(always("call param init enter mode"), |b| {
        push_call_param_init_enter_mode_constraints(b, &control, &param_init, &call);
    });

    b.with_tag(always("call param init exit mode"), |b| {
        push_call_param_init_exit_mode_constraints(b, &param_init);
    });

    b.with_tag(always("call param init aux row"), |b| {
        push_call_param_init_aux_row_constraints(b, &state, &param_init);
    });

    b.with_tag(always("return pc restoration"), |b| {
        b.push_row(
            [(idx(call.call_stack_pop_present), F::ONE)],
            [
                (idx(state.pc_after), F::ONE),
                (idx(call.call_stack_access_return_pc), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(idx(call.call_stack_pop_present), F::ONE)],
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::Return).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::End).unwrap(), -F::ONE),
            ],
            [],
        );
    });

    b.with_tag(always("call stack transition"), |b| {
        push_call_stack_transition_constraints(b, &control, &call, &frame);
    });

    b.with_tag(always("locals fbp transition"), |b| {
        push_locals_fbp_transition_constraints(b, &call, &frame);
    });

    b.with_tag(always("dynamic call stack arity"), |b| {
        push_dynamic_call_stack_arity_constraints(b, &control, &state, &call);
    });
}

fn push_simple_output_constraints(b: &mut R1csBuilder, control: &ControlColumns, state: &StateColumns) {
    let enabled_delta = [(COL_OUTPUT_ENABLED_AFTER, F::ONE), (COL_OUTPUT_ENABLED_BEFORE, -F::ONE)];
    b.with_tag(always("simple output carry"), |b| {
        for (after, before) in [
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE),
            (COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE),
            (COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE),
        ] {
            b.push_row(
                [(COL_ONE, F::ONE), (idx(control.halted), -F::ONE)],
                [(after, F::ONE), (before, -F::ONE)],
                [],
            );
            push_gated_linear_zero(b, COL_OUTPUT_ENABLED_BEFORE, [(after, F::ONE), (before, -F::ONE)]);
        }

        b.push_linear_zero(
            enabled_delta
                .into_iter()
                .chain([(COL_OUTPUT_CAPTURED, -F::ONE)]),
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_ONE, F::ONE), (idx(control.halted), -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_OUTPUT_ENABLED_BEFORE, F::ONE)],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [
                (COL_STACK_READ0_ADDR_LO, F::ONE),
                (idx(state.sp_before), -F::from_u64(2)),
                (COL_ONE, F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_OUTPUT_VALUE_LO_AFTER, F::ONE), (COL_STACK_READ0_VALUE_LO, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_OUTPUT_CAPTURED, F::ONE)],
            [(COL_OUTPUT_VALUE_HI_AFTER, F::ONE), (COL_STACK_READ0_VALUE_HI, -F::ONE)],
            [],
        );
    });
}

fn push_dynamic_call_stack_arity_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    state: &StateColumns,
    call: &CallColumns,
) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    // Guest call rows read nothing for direct calls, or only the table index
    // for indirect calls; args are popped by param-init aux rows.
    push_gated_linear_zero(
        b,
        idx(call.call_stack_push_present),
        [(idx(control.stack_reads), F::ONE), (call_indirect, -F::ONE)],
    );
    // Host calls still pop args on-row; see README for the remaining arity cap.
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (idx(call.call_stack_push_present), -F::ONE),
        ],
        [
            (idx(control.stack_reads), F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
            (call_indirect, -F::ONE),
        ],
        [],
    );
    push_gated_linear_zero(
        b,
        call_indirect,
        [(COL_FUNCTION_REF, F::ONE), (COL_TABLE_VALUE, -F::ONE)],
    );
    // Bind the table read to the index popped from the stack top.
    push_gated_linear_zero(
        b,
        call_indirect,
        [(COL_TABLE_INDEX, F::ONE), (COL_STACK_READ0_VALUE_LO, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        call_indirect,
        [
            (COL_STACK_READ0_ADDR_LO, F::ONE),
            (idx(state.sp_before), -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
    );

    // stack_writes on call rows splits on guest vs host:
    // - guest call (call_stack_push_present == 1): results land later on
    //   the matching Return/End, so the call row itself writes 0.
    // - host call (Call/CallIndirect selector == 1, push_present == 0):
    //   the host's results land on this row, so writes == result_count.
    // This pins the host call's stack footprint to its declared type
    // signature; the host cannot push more results than result_count.
    b.push_row(
        [(idx(call.call_stack_push_present), F::ONE)],
        [(idx(control.stack_writes), F::ONE)],
        [],
    );
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (idx(call.call_stack_push_present), -F::ONE),
        ],
        [(idx(control.stack_writes), F::ONE), (COL_CALL_RESULT_COUNT, -F::ONE)],
        [],
    );
}

fn push_guest_call_flag_constraints(b: &mut R1csBuilder, call: &CallColumns) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE)],
        [(idx(call.call_stack_push_present), F::ONE)],
    );
}

fn push_call_param_init_enter_mode_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    param_init: &ParamInitColumns,
    call: &CallColumns,
) {
    let guest_call = idx(call.call_stack_push_present);

    b.push_row(
        [
            (idx(control.is_program_row), F::ONE),
            (idx(call.call_stack_push_present), -F::ONE),
        ],
        // Only guest calls may enter param-init mode from a program row.
        // Aux rows are excluded by `is_program_row = 0`, so multi-param init
        // can continue until the global remaining-after zero test turns it off.
        [(idx(param_init.param_init_active_after), F::ONE)],
        [],
    );

    // guest_call => param_init_remaining' == param_count
    push_gated_linear_zero(
        b,
        guest_call,
        [
            (idx(param_init.param_init_remaining_after), F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
        ],
    );
}

fn push_call_param_init_exit_mode_constraints(b: &mut R1csBuilder, param_init: &ParamInitColumns) {
    b.push_linear_zero([
        (idx(param_init.param_init_active_after), F::ONE),
        (idx(param_init.param_init_remaining_after_is_zero), F::ONE),
        (COL_ONE, -F::ONE),
    ]);

    // if we reached the end of the local initialization sequence
    push_zero_test_gadget(
        b,
        idx(param_init.param_init_remaining_after),
        idx(param_init.param_init_remaining_after_inv),
        idx(param_init.param_init_remaining_after_is_zero),
    );
}

fn push_call_param_init_aux_row_constraints(b: &mut R1csBuilder, state: &StateColumns, param_init: &ParamInitColumns) {
    let selector = idx(param_init.param_init_active_before);

    // in_param_init_mode => param_init_remaining' = param_init_remaining - 1
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(param_init.param_init_remaining_before), F::ONE),
            (idx(param_init.param_init_remaining_after), -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );

    // Param-init reads the current stack top.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ0_ADDR_LO, F::ONE),
            (idx(state.sp_before), -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
    );

    // Pops run top-down, so this row initializes local `remaining_before - 1`.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_LOCAL_INDEX, F::ONE),
            (idx(param_init.param_init_remaining_before), -F::ONE),
            (COL_ONE, F::ONE),
        ],
    );

    // the pc is not constrained for the aux opcode (since it's not a real
    // opcode, it's not in the next pc table)
    //
    // we assert here that it doesn't change
    push_gated_linear_zero(
        b,
        selector,
        [(idx(state.pc_after), F::ONE), (idx(state.pc_before), -F::ONE)],
    );
}

fn push_call_stack_transition_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    call: &CallColumns,
    frame: &FrameColumns,
) {
    let push = idx(call.call_stack_push_present);
    let pop = idx(call.call_stack_pop_present);

    // Push increments the return-context stack, pop decrements it, and every
    // other row preserves it. Range-checking on the depth columns rules out
    // underflow in the bounded witness model.
    b.push_linear_zero([
        (idx(call.call_stack_depth_after), F::ONE),
        (idx(call.call_stack_depth_before), -F::ONE),
        (push, -F::ONE),
        (pop, F::ONE),
    ]);

    push_gated_linear_zero(
        b,
        push,
        [
            (idx(call.call_stack_addr), F::ONE),
            (idx(call.call_stack_depth_before), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (idx(call.call_stack_return_pc_choice), F::ONE),
            (COL_ONE, -F::from_u64(CALL_RETURN_PC_CHOICE)),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [
            (idx(call.call_stack_addr), F::ONE),
            (idx(call.call_stack_depth_after), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (idx(call.call_stack_access_caller_fbp), F::ONE),
            (idx(frame.locals_fbp_before), -F::ONE),
        ],
    );
    push_gated_linear_zero(b, idx(control.halted), [(idx(call.call_stack_depth_before), F::ONE)]);
}

fn push_locals_fbp_transition_constraints(b: &mut R1csBuilder, call: &CallColumns, frame: &FrameColumns) {
    let guest_call = idx(call.call_stack_push_present);
    let pop = idx(call.call_stack_pop_present);

    push_gated_linear_zero(
        b,
        guest_call,
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(frame.locals_fbp_before), -F::ONE),
            (idx(frame.current_function_num_locals), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(call.call_stack_access_caller_fbp), -F::ONE),
        ],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (guest_call, -F::ONE), (pop, -F::ONE)],
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(frame.locals_fbp_before), -F::ONE),
        ],
        [],
    );
}
