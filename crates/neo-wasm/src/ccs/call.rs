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
use super::super::layout::{selector_col, CALL_RETURN_PC_CHOICE, COL_ONE};
use super::super::lookup_binding_builder::{
    CallColumns, ControlColumns, FrameColumns, FunctionTypeColumns, LocalsColumns, ModuleTypeColumns,
    OperandStackColumns, OutputColumns, ParamInitColumns, StateColumns, TableColumns, WasmLookupBindingLayout,
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
    let stack = layout.stack;
    let output = layout.output;
    let locals = layout.locals;
    let function_types = layout.function_types;
    let module_types = layout.module_types;
    let table = layout.table;

    b.with_tag(always("row kind one hot"), |b| {
        b.push_linear_zero([
            (idx(control.is_program_row), F::ONE),
            (idx(param_init.param_init_active_before), F::ONE),
            (idx(control.padding_active), F::ONE),
            (COL_ONE, -F::ONE),
        ]);
    });

    let memory_pages = layout.memory_pages;
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
            [(idx(memory_pages.after), F::ONE), (idx(memory_pages.before), -F::ONE)],
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

    push_simple_output_constraints(b, &control, &state, &stack, &output);

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
            [(idx(stack.read0_value_lo), F::ONE), (idx(locals.value_lo), -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(idx(stack.read0_value_hi), F::ONE), (idx(locals.value_hi), -F::ONE)],
        );
    });

    b.with_tag(always("guest call flag"), |b| {
        push_guest_call_flag_constraints(b, &call, &function_types);
    });

    b.with_tag(always("call param init enter mode"), |b| {
        push_call_param_init_enter_mode_constraints(b, &control, &param_init, &call, &function_types);
    });

    b.with_tag(always("call param init exit mode"), |b| {
        push_call_param_init_exit_mode_constraints(b, &param_init);
    });

    b.with_tag(always("call param init aux row"), |b| {
        push_call_param_init_aux_row_constraints(b, &state, &param_init, &stack, &locals);
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

    b.with_tag(
        super::opcode_tag("call_indirect type constraints", WasmOpcode::CallIndirect),
        |b| {
            push_call_indirect_type_constraints(b, &function_types, &module_types);
        },
    );

    b.with_tag(always("dynamic call stack arity"), |b| {
        push_dynamic_call_stack_arity_constraints(b, &control, &state, &stack, &call, &function_types, &table);
    });
}

fn push_simple_output_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    state: &StateColumns,
    stack: &OperandStackColumns,
    output: &OutputColumns,
) {
    let enabled_delta = [
        (idx(output.enabled_after), F::ONE),
        (idx(output.enabled_before), -F::ONE),
    ];
    b.with_tag(always("simple output carry"), |b| {
        for (after, before) in [
            (output.enabled_after, output.enabled_before),
            (output.value_lo_after, output.value_lo_before),
            (output.value_hi_after, output.value_hi_before),
        ] {
            b.push_row(
                [(COL_ONE, F::ONE), (idx(control.halted), -F::ONE)],
                [(idx(after), F::ONE), (idx(before), -F::ONE)],
                [],
            );
            push_gated_linear_zero(
                b,
                idx(output.enabled_before),
                [(idx(after), F::ONE), (idx(before), -F::ONE)],
            );
        }

        b.push_linear_zero(
            enabled_delta
                .into_iter()
                .chain([(idx(output.captured), -F::ONE)]),
        );
        b.push_row(
            [(idx(output.captured), F::ONE)],
            [(COL_ONE, F::ONE), (idx(control.halted), -F::ONE)],
            [],
        );
        b.push_row(
            [(idx(output.captured), F::ONE)],
            [(idx(output.enabled_before), F::ONE)],
            [],
        );
        b.push_row(
            [(idx(output.captured), F::ONE)],
            [
                (idx(stack.read0_addr_lo), F::ONE),
                (idx(state.sp_before), -F::from_u64(2)),
                (COL_ONE, F::from_u64(2)),
            ],
            [],
        );
        b.push_row(
            [(idx(output.captured), F::ONE)],
            [
                (idx(output.value_lo_after), F::ONE),
                (idx(stack.read0_value_lo), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(idx(output.captured), F::ONE)],
            [
                (idx(output.value_hi_after), F::ONE),
                (idx(stack.read0_value_hi), -F::ONE),
            ],
            [],
        );
    });
}

fn push_call_indirect_type_constraints(
    b: &mut R1csBuilder,
    function_types: &FunctionTypeColumns,
    module_types: &ModuleTypeColumns,
) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [
            (idx(function_types.type_id), F::ONE),
            (idx(module_types.expected_type_id), -F::ONE),
        ],
    );
}

fn push_dynamic_call_stack_arity_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    state: &StateColumns,
    stack: &OperandStackColumns,
    call: &CallColumns,
    function_types: &FunctionTypeColumns,
    table: &TableColumns,
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
            (idx(function_types.param_count), -F::ONE),
            (call_indirect, -F::ONE),
        ],
        [],
    );
    push_gated_linear_zero(
        b,
        call_indirect,
        [(idx(function_types.function_ref), F::ONE), (idx(table.value), -F::ONE)],
    );
    // Bind the table read to the index popped from the stack top.
    push_gated_linear_zero(
        b,
        call_indirect,
        [(idx(table.index), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        call_indirect,
        [
            (idx(stack.read0_addr_lo), F::ONE),
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
        [
            (idx(control.stack_writes), F::ONE),
            (idx(function_types.result_count), -F::ONE),
        ],
        [],
    );
}

fn push_guest_call_flag_constraints(b: &mut R1csBuilder, call: &CallColumns, function_types: &FunctionTypeColumns) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(idx(function_types.is_guest), F::ONE)],
        [(idx(call.call_stack_push_present), F::ONE)],
    );
}

fn push_call_param_init_enter_mode_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    param_init: &ParamInitColumns,
    call: &CallColumns,
    function_types: &FunctionTypeColumns,
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
            (idx(function_types.param_count), -F::ONE),
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

fn push_call_param_init_aux_row_constraints(
    b: &mut R1csBuilder,
    state: &StateColumns,
    param_init: &ParamInitColumns,
    stack: &OperandStackColumns,
    locals: &LocalsColumns,
) {
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
            (idx(stack.read0_addr_lo), F::ONE),
            (idx(state.sp_before), -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
    );

    // Pops run top-down, so this row initializes local `remaining_before - 1`.
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(locals.index), F::ONE),
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
