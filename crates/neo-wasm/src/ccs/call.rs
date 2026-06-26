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
use super::super::layout::{
    selector_col, CALL_RETURN_PC_CHOICE, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR,
    COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE, COL_CALL_STACK_POP_CALLER_FBP, COL_CALL_STACK_POP_PRESENT,
    COL_CALL_STACK_POP_RETURN_PC, COL_CALL_STACK_PUSH_PRESENT, COL_CALL_STACK_RETURN_PC_CHOICE,
    COL_CURRENT_FUNCTION_NUM_LOCALS, COL_FUNCTION_REF, COL_HALTED, COL_IS_PROGRAM_ROW, COL_LOCALS_FBP_AFTER,
    COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_MEMORY_PAGES_AFTER,
    COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_OUTPUT_CAPTURED, COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE,
    COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE, COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE,
    COL_PADDING_ACTIVE, COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER,
    COL_PARAM_INIT_REMAINING_AFTER_INV, COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PARAM_INIT_REMAINING_BEFORE,
    COL_PC_AFTER, COL_PC_BEFORE, COL_SP_BEFORE, COL_STACK_READ0_ADDR_LO, COL_STACK_READ0_VALUE_HI,
    COL_STACK_READ0_VALUE_LO, COL_STACK_READS, COL_STACK_WRITES, COL_TABLE_INDEX, COL_TABLE_VALUE,
    COL_TARGET_FUNCTION_IS_GUEST,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::always;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

/// Emit every call/frame/param-init row the wasm VM needs in a single
/// place. Ordering inside follows the natural lifecycle of a call:
/// row-kind classification → aux-row shape → enter/exit param init →
/// per-aux-row witness shape → return-pc restoration → frame fbp
/// transition → dynamic call-arity lookups.
pub(super) fn push_call_constraints(b: &mut R1csBuilder) {
    b.with_tag(always("row kind one hot"), |b| {
        b.push_linear_zero([
            (COL_IS_PROGRAM_ROW, F::ONE),
            (COL_PARAM_INIT_ACTIVE_BEFORE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
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
                (COL_CALL_STACK_DEPTH_AFTER, F::ONE),
                (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_PARAM_INIT_ACTIVE_AFTER, F::ONE),
                (COL_PARAM_INIT_ACTIVE_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_PARAM_INIT_REMAINING_AFTER, F::ONE),
                (COL_PARAM_INIT_REMAINING_BEFORE, -F::ONE),
            ],
        );
    });

    push_simple_output_constraints(b);

    b.with_tag(always("non-program row shape"), |b| {
        // Aux rows keep pc fixed and write no stack values. Padding rows read
        // nothing; param-init rows pop one arg slot.
        let aux_row_gate = [(COL_PARAM_INIT_ACTIVE_BEFORE, F::ONE), (COL_PADDING_ACTIVE, F::ONE)];

        b.push_row(aux_row_gate, [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE)], []);
        b.push_row([(COL_PADDING_ACTIVE, F::ONE)], [(COL_STACK_READS, F::ONE)], []);
        push_gated_linear_zero(
            b,
            COL_PARAM_INIT_ACTIVE_BEFORE,
            [(COL_STACK_READS, F::ONE), (COL_ONE, -F::ONE)],
        );
        b.push_row(aux_row_gate, [(COL_STACK_WRITES, F::ONE)], []);

        let param_init_row_gate = COL_PARAM_INIT_ACTIVE_BEFORE;

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
        push_guest_call_flag_constraints(b);
    });

    b.with_tag(always("call param init enter mode"), |b| {
        push_call_param_init_enter_mode_constraints(b);
    });

    b.with_tag(always("call param init exit mode"), |b| {
        push_call_param_init_exit_mode_constraints(b);
    });

    b.with_tag(always("call param init aux row"), |b| {
        push_call_param_init_aux_row_constraints(b);
    });

    b.with_tag(always("return pc restoration"), |b| {
        b.push_row(
            [(COL_CALL_STACK_POP_PRESENT, F::ONE)],
            [(COL_PC_AFTER, F::ONE), (COL_CALL_STACK_POP_RETURN_PC, -F::ONE)],
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

    b.with_tag(always("locals fbp transition"), |b| {
        push_locals_fbp_transition_constraints(b);
    });

    b.with_tag(always("dynamic call stack arity"), |b| {
        push_dynamic_call_stack_arity_constraints(b);
    });
}

fn push_simple_output_constraints(b: &mut R1csBuilder) {
    let enabled_delta = [(COL_OUTPUT_ENABLED_AFTER, F::ONE), (COL_OUTPUT_ENABLED_BEFORE, -F::ONE)];
    b.with_tag(always("simple output carry"), |b| {
        for (after, before) in [
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE),
            (COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE),
            (COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE),
        ] {
            b.push_row(
                [(COL_ONE, F::ONE), (COL_HALTED, -F::ONE)],
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
            [(COL_ONE, F::ONE), (COL_HALTED, -F::ONE)],
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
                (COL_SP_BEFORE, -F::from_u64(2)),
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

fn push_dynamic_call_stack_arity_constraints(b: &mut R1csBuilder) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    // Guest call rows read nothing for direct calls, or only the table index
    // for indirect calls; args are popped by param-init aux rows.
    push_gated_linear_zero(
        b,
        COL_CALL_STACK_PUSH_PRESENT,
        [(COL_STACK_READS, F::ONE), (call_indirect, -F::ONE)],
    );
    // Host calls still pop args on-row; see README for the remaining arity cap.
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (COL_CALL_STACK_PUSH_PRESENT, -F::ONE),
        ],
        [
            (COL_STACK_READS, F::ONE),
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
            (COL_SP_BEFORE, -F::from_u64(2)),
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
        [(COL_CALL_STACK_PUSH_PRESENT, F::ONE)],
        [(COL_STACK_WRITES, F::ONE)],
        [],
    );
    b.push_row(
        [
            (call_selector, F::ONE),
            (call_indirect, F::ONE),
            (COL_CALL_STACK_PUSH_PRESENT, -F::ONE),
        ],
        [(COL_STACK_WRITES, F::ONE), (COL_CALL_RESULT_COUNT, -F::ONE)],
        [],
    );
}

fn push_guest_call_flag_constraints(b: &mut R1csBuilder) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE)],
        [(COL_CALL_STACK_PUSH_PRESENT, F::ONE)],
    );
}

fn push_call_param_init_enter_mode_constraints(b: &mut R1csBuilder) {
    let guest_call = COL_CALL_STACK_PUSH_PRESENT;

    b.push_row(
        [(COL_IS_PROGRAM_ROW, F::ONE), (COL_CALL_STACK_PUSH_PRESENT, -F::ONE)],
        // Only guest calls may enter param-init mode from a program row.
        // Aux rows are excluded by `is_program_row = 0`, so multi-param init
        // can continue until the global remaining-after zero test turns it off.
        [(COL_PARAM_INIT_ACTIVE_AFTER, F::ONE)],
        [],
    );

    // guest_call => param_init_remaining' == param_count
    push_gated_linear_zero(
        b,
        guest_call,
        [
            (COL_PARAM_INIT_REMAINING_AFTER, F::ONE),
            (COL_CALL_PARAM_COUNT, -F::ONE),
        ],
    );
}

fn push_call_param_init_exit_mode_constraints(b: &mut R1csBuilder) {
    b.push_linear_zero([
        (COL_PARAM_INIT_ACTIVE_AFTER, F::ONE),
        (COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, F::ONE),
        (COL_ONE, -F::ONE),
    ]);

    // if we reached the end of the local initialization sequence
    push_zero_test_gadget(
        b,
        COL_PARAM_INIT_REMAINING_AFTER,
        COL_PARAM_INIT_REMAINING_AFTER_INV,
        COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO,
    );
}

fn push_call_param_init_aux_row_constraints(b: &mut R1csBuilder) {
    let selector = COL_PARAM_INIT_ACTIVE_BEFORE;

    // in_param_init_mode => param_init_remaining' = param_init_remaining - 1
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_PARAM_INIT_REMAINING_BEFORE, F::ONE),
            (COL_PARAM_INIT_REMAINING_AFTER, -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );

    // Param-init reads the current stack top.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
    );

    // Pops run top-down, so this row initializes local `remaining_before - 1`.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_LOCAL_INDEX, F::ONE),
            (COL_PARAM_INIT_REMAINING_BEFORE, -F::ONE),
            (COL_ONE, F::ONE),
        ],
    );

    // the pc is not constrained for the aux opcode (since it's not a real
    // opcode, it's not in the next pc table)
    //
    // we assert here that it doesn't change
    push_gated_linear_zero(b, selector, [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE)]);
}

fn push_call_stack_transition_constraints(b: &mut R1csBuilder) {
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
            (COL_CALL_STACK_RETURN_PC_CHOICE, F::ONE),
            (COL_ONE, -F::from_u64(CALL_RETURN_PC_CHOICE)),
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
            (COL_CALL_STACK_POP_CALLER_FBP, F::ONE),
            (COL_LOCALS_FBP_BEFORE, -F::ONE),
        ],
    );
    push_gated_linear_zero(b, COL_HALTED, [(COL_CALL_STACK_DEPTH_BEFORE, F::ONE)]);
}

fn push_locals_fbp_transition_constraints(b: &mut R1csBuilder) {
    let guest_call = COL_CALL_STACK_PUSH_PRESENT;
    let pop = COL_CALL_STACK_POP_PRESENT;

    push_gated_linear_zero(
        b,
        guest_call,
        [
            (COL_LOCALS_FBP_AFTER, F::ONE),
            (COL_LOCALS_FBP_BEFORE, -F::ONE),
            (COL_CURRENT_FUNCTION_NUM_LOCALS, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [(COL_LOCALS_FBP_AFTER, F::ONE), (COL_CALL_STACK_POP_CALLER_FBP, -F::ONE)],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (guest_call, -F::ONE), (pop, -F::ONE)],
        [(COL_LOCALS_FBP_AFTER, F::ONE), (COL_LOCALS_FBP_BEFORE, -F::ONE)],
        [],
    );
}
