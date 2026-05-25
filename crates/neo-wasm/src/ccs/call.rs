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
use super::super::layout::{selector_col, COL_ONE};
use super::super::lookup_binding_builder::{
    CallColumns, ControlColumns, FrameColumns, FunctionTypeColumns, LocalsColumns, ModuleTypeColumns,
    OperandStackColumns, ParamInitColumns, StateColumns, TableColumns, WasmLookupBindingLayout,
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

    b.with_tag(always("non-program row shape"), |b| {
        // pc_after == pc_before; stack_reads == stack_writes == 0
        // (which together with the global sp linear constraint imply
        // sp_after == sp_before).
        //
        // Shared by param-init aux rows (which advance state machine
        // bookkeeping but don't change pc/sp) and padding rows (which
        // preserve everything). The two gate columns are mutually
        // exclusive booleans by the row-kind one-hot, so their sum is
        // still in {0, 1}.
        let aux_row_gate = [
            (idx(param_init.param_init_active_before), F::ONE),
            (idx(control.padding_active), F::ONE),
        ];

        b.push_row(
            aux_row_gate,
            [(idx(state.pc_after), F::ONE), (idx(state.pc_before), -F::ONE)],
            [],
        );
        b.push_row(aux_row_gate, [(idx(control.stack_reads), F::ONE)], []);
        b.push_row(aux_row_gate, [(idx(control.stack_writes), F::ONE)], []);

        let param_init_row_gate = idx(param_init.param_init_active_before);

        // write to the locals memory the value read from the stack
        //
        // remember that there is only one lane for locals access
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(idx(stack.read0_value), F::ONE), (idx(locals.value_lo), -F::ONE)],
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
        push_call_param_init_aux_row_constraints(b, &state, &param_init, &function_types, &stack, &locals);
    });

    b.with_tag(always("return pc restoration"), |b| {
        b.push_row(
            [(idx(call.call_stack_pop_present), F::ONE)],
            [
                (idx(state.pc_after), F::ONE),
                (idx(call.call_stack_pop_return_pc), -F::ONE),
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
        push_dynamic_call_stack_arity_constraints(b, &control, &function_types, &table);
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
    function_types: &FunctionTypeColumns,
    table: &TableColumns,
) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::Call).unwrap(),
        [
            (idx(control.stack_reads), F::ONE),
            (idx(function_types.param_count), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::Call).unwrap(),
        [(idx(control.stack_writes), F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [(idx(function_types.function_ref), F::ONE), (idx(table.value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [
            (idx(control.stack_reads), F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [(idx(control.stack_writes), F::ONE)],
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
    function_types: &FunctionTypeColumns,
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

    push_gated_linear_zero(
        b,
        selector,
        [
            // stack_addr + remaining = sp_before + param_count
            //
            // remaining goes down, so stack_addr may go up (the rhs is constant while selector is on)
            (idx(stack.read0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (idx(param_init.param_init_remaining_before), F::ONE),
        ],
    );

    push_gated_linear_zero(
        b,
        selector,
        // The aux row writes callee local `param_count - remaining_before`.
        [
            (idx(locals.index), F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (idx(param_init.param_init_remaining_before), F::ONE),
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
            (idx(call.call_stack_pop_caller_fbp), -F::ONE),
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
