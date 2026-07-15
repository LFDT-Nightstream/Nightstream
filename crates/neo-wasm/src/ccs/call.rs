//! Owns the per-step call / frame / parameter-initialization CCS rows.
//!
//! Wasm guest calls split into three coupled mechanisms:
//!
//! 1. **Call-stack frame plumbing**: `frame.locals_fbp_*` advances on
//!    push (enter callee) and rewinds on pop (return to caller);
//!    `state.pc_after` is restored from the call-stack return-pc cell
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
//!
//! Host (imported) callees produce no guest rows. Their call row enters
//! host-arg mode with `host_args.remaining = param_count`; each
//! `HostCallArg` row pops one argument, and `HostCallResult` pushes the
//! single flat result when `result_count = 1`.

use super::super::gadgets::{push_gated_linear_zero, push_zero_test_gadget};
use super::super::isa::WasmOpcode;
use super::super::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_PARAM_COUNT, COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR,
    COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE,
    COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_RETURN_PC_VALUE, COL_CI_HOST_CALL, COL_CURRENT_FUNCTION_NUM_LOCALS,
    COL_FUNCTION_REF, COL_GUEST_CALL_ACTIVE, COL_HALTED, COL_HOST_ARGS_ACTIVE_AFTER, COL_HOST_ARGS_ACTIVE_BEFORE,
    COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_AFTER_INV, COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO,
    COL_HOST_ARGS_REMAINING_BEFORE, COL_HOST_RESULT_ACTIVE, COL_HOST_RESULT_PENDING_AFTER,
    COL_HOST_RESULT_PENDING_BEFORE, COL_IS_PROGRAM_ROW, COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE, COL_LOCAL_INDEX,
    COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI, COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_OUTPUT_CAPTURED,
    COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE,
    COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE, COL_PADDING_ACTIVE, COL_PARAM_INIT_ACTIVE_AFTER,
    COL_PARAM_INIT_ACTIVE_BEFORE, COL_PARAM_INIT_REMAINING_AFTER, COL_PARAM_INIT_REMAINING_AFTER_INV,
    COL_PARAM_INIT_REMAINING_AFTER_IS_ZERO, COL_PARAM_INIT_REMAINING_BEFORE, COL_PC_AFTER, COL_PC_BEFORE,
    COL_PC_ROM_CALL_RETURN_CHOICE, COL_SP_BEFORE, COL_STACK_READ0_ADDR_LO, COL_STACK_READ0_VALUE_HI,
    COL_STACK_READ0_VALUE_LO, COL_STACK_READS, COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITES, COL_TABLE_INDEX,
    COL_TABLE_VALUE, COL_TARGET_FUNCTION_IS_GUEST, COL_TRAPPED_AFTER, PC_ROM_CALL_RETURN_CHOICE,
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
            (COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE),
            (COL_HOST_RESULT_ACTIVE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
            (COL_ONE, -F::ONE),
        ]);
        // host_result_active = host_result_pending_before · ¬host_args_active_before:
        // the result row is the first row after the arg pops while a push is
        // still owed. Feeding the one-hot above, this also forces the pending
        // flag to be consumed before the next program row.
        b.push_row(
            [(COL_HOST_RESULT_PENDING_BEFORE, F::ONE)],
            [(COL_ONE, F::ONE), (COL_HOST_ARGS_ACTIVE_BEFORE, -F::ONE)],
            [(COL_HOST_RESULT_ACTIVE, F::ONE)],
        );
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
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_HOST_ARGS_ACTIVE_AFTER, F::ONE),
                (COL_HOST_ARGS_ACTIVE_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_HOST_ARGS_REMAINING_AFTER, F::ONE),
                (COL_HOST_ARGS_REMAINING_BEFORE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            padding_gate,
            [
                (COL_HOST_RESULT_PENDING_AFTER, F::ONE),
                (COL_HOST_RESULT_PENDING_BEFORE, -F::ONE),
            ],
        );
    });

    push_simple_output_constraints(b);

    b.with_tag(always("non-program row shape"), |b| {
        // Aux rows keep pc fixed. Padding rows read and write nothing;
        // param-init and host-arg rows pop one arg slot each; host-result
        // rows push the single host result.
        let aux_row_gate = [
            (COL_PARAM_INIT_ACTIVE_BEFORE, F::ONE),
            (COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE),
            (COL_HOST_RESULT_ACTIVE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
        ];

        b.push_row(aux_row_gate, [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE)], []);
        b.push_row(
            [(COL_PADDING_ACTIVE, F::ONE), (COL_HOST_RESULT_ACTIVE, F::ONE)],
            [(COL_STACK_READS, F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_PARAM_INIT_ACTIVE_BEFORE, F::ONE),
                (COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE),
            ],
            [(COL_STACK_READS, F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_PARAM_INIT_ACTIVE_BEFORE, F::ONE),
                (COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE),
                (COL_PADDING_ACTIVE, F::ONE),
            ],
            [(COL_STACK_WRITES, F::ONE)],
            [],
        );
        push_gated_linear_zero(
            b,
            COL_HOST_RESULT_ACTIVE,
            [(COL_STACK_WRITES, F::ONE), (COL_ONE, -F::ONE)],
        );

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

    b.with_tag(always("host call enter mode"), |b| {
        push_host_call_enter_mode_constraints(b);
    });

    b.with_tag(always("host call exit mode"), |b| {
        push_host_call_exit_mode_constraints(b);
    });

    b.with_tag(always("host call arg aux row"), |b| {
        push_host_call_arg_aux_row_constraints(b);
    });

    b.with_tag(always("host call result aux row"), |b| {
        push_host_call_result_aux_row_constraints(b);
    });

    b.with_tag(always("host call state preservation"), |b| {
        push_host_call_state_preservation_constraints(b);
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

    // Call rows read nothing for direct calls, or only the table index for
    // indirect calls (trapping ones included). Guest args are popped by
    // param-init aux rows, host args by host-arg aux rows.
    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_STACK_READS, F::ONE), (call_indirect, -F::ONE)],
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

    // Call rows never write: guest results land on the matching Return/End,
    // and host results land on the trailing host-result aux row.
    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_STACK_WRITES, F::ONE)],
        [],
    );
}

/// Linear gate terms selecting host-call program rows: a `call` or
/// non-trapping `call_indirect` whose callee is not a traced guest function.
/// `guest_call_active = (call + ci)·is_guest` and `ci_not_trap = ci·¬trap`, so the
/// sum is exactly 1 on host-call rows and 0 everywhere else (guest calls,
/// ci-trap rows, non-call rows, aux rows).
fn host_call_gate_terms() -> [(usize, F); 3] {
    [
        (selector_col(WasmOpcode::Call).unwrap(), F::ONE),
        (COL_CALL_INDIRECT_IS_NOT_TRAP, F::ONE),
        (COL_GUEST_CALL_ACTIVE, -F::ONE),
    ]
}

fn push_host_call_enter_mode_constraints(b: &mut R1csBuilder) {
    // Program rows that are not host calls leave host-arg mode off and owe
    // no result push. Aux rows zero this gate; they are pinned by the
    // decrement/consumption/preservation rows instead. Pinning
    // `host_args_active_after` also pins `host_args_remaining_after` to zero
    // transitively: the exit-mode identity forces `remaining_after_is_zero`,
    // and the zero-test gadget forces the counter itself.
    let non_host_program = [
        (COL_IS_PROGRAM_ROW, F::ONE),
        // TODO: (minor) negation of host_call_gate_terms, may be worth
        // generalizing in the future?
        (selector_col(WasmOpcode::Call).unwrap(), -F::ONE),
        (COL_CALL_INDIRECT_IS_NOT_TRAP, -F::ONE),
        (COL_GUEST_CALL_ACTIVE, F::ONE),
    ];
    b.push_row(non_host_program, [(COL_HOST_ARGS_ACTIVE_AFTER, F::ONE)], []);
    b.push_row(non_host_program, [(COL_HOST_RESULT_PENDING_AFTER, F::ONE)], []);

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

    // host call => remaining' == param_count and pending' == result_count,
    // both ROM-bound to the callee's declared type. `pending` is a Boolean
    // column, so a host signature with more than one result is unsatisfiable
    // (the canonical ABI caps flat results at 1).
    b.push_row(
        host_call_gate_terms(),
        [(COL_HOST_ARGS_REMAINING_AFTER, F::ONE), (COL_CALL_PARAM_COUNT, -F::ONE)],
        [],
    );
    b.push_row(
        host_call_gate_terms(),
        [
            (COL_HOST_RESULT_PENDING_AFTER, F::ONE),
            (COL_CALL_RESULT_COUNT, -F::ONE),
        ],
        [],
    );
}

fn push_host_call_exit_mode_constraints(b: &mut R1csBuilder) {
    b.push_linear_zero([
        (COL_HOST_ARGS_ACTIVE_AFTER, F::ONE),
        (COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO, F::ONE),
        (COL_ONE, -F::ONE),
    ]);

    push_zero_test_gadget(
        b,
        COL_HOST_ARGS_REMAINING_AFTER,
        COL_HOST_ARGS_REMAINING_AFTER_INV,
        COL_HOST_ARGS_REMAINING_AFTER_IS_ZERO,
    );
}

fn push_host_call_arg_aux_row_constraints(b: &mut R1csBuilder) {
    let selector = COL_HOST_ARGS_ACTIVE_BEFORE;

    // in host-arg mode => remaining' = remaining - 1
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_HOST_ARGS_REMAINING_BEFORE, F::ONE),
            (COL_HOST_ARGS_REMAINING_AFTER, -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );

    // Each arg row pops the current stack top.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -F::from_u64(2)),
            (COL_ONE, F::from_u64(2)),
        ],
    );

    // The owed result push carries through the arg pops.
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_HOST_RESULT_PENDING_AFTER, F::ONE),
            (COL_HOST_RESULT_PENDING_BEFORE, -F::ONE),
        ],
    );
}

fn push_host_call_result_aux_row_constraints(b: &mut R1csBuilder) {
    let selector = COL_HOST_RESULT_ACTIVE;

    // The result lands on the post-pop stack top. The global sp identity gives
    // sp' = sp + 1, so the write address is 2 * sp_before. The value is a
    // host-oracle input; this row only pins its stack placement.
    push_gated_linear_zero(
        b,
        selector,
        [(COL_STACK_WRITE0_ADDR_LO, F::ONE), (COL_SP_BEFORE, -F::from_u64(2))],
    );

    // The push consumes the owed-result flag and never re-enters arg mode
    // (zeroing `host_args_active_after` also zeroes the remaining counter via
    // the exit-mode identity + zero-test gadget).
    push_gated_linear_zero(b, selector, [(COL_HOST_RESULT_PENDING_AFTER, F::ONE)]);
    push_gated_linear_zero(b, selector, [(COL_HOST_ARGS_ACTIVE_AFTER, F::ONE)]);
}

fn push_host_call_state_preservation_constraints(b: &mut R1csBuilder) {
    // Param-init rows carry the host-call state through unchanged (provably
    // zero there: a guest call cannot enter the host modes), and host aux
    // rows carry the param-init state. Without these rows a malicious prover
    // could flip the other mode's `_after` state on an aux row and inject
    // arbitrary pop/local-write sequences after a call.
    for (after, before) in [
        (COL_HOST_ARGS_ACTIVE_AFTER, COL_HOST_ARGS_ACTIVE_BEFORE),
        (COL_HOST_ARGS_REMAINING_AFTER, COL_HOST_ARGS_REMAINING_BEFORE),
        (COL_HOST_RESULT_PENDING_AFTER, COL_HOST_RESULT_PENDING_BEFORE),
    ] {
        push_gated_linear_zero(b, COL_PARAM_INIT_ACTIVE_BEFORE, [(after, F::ONE), (before, -F::ONE)]);
    }
    for (after, before) in [
        (COL_PARAM_INIT_ACTIVE_AFTER, COL_PARAM_INIT_ACTIVE_BEFORE),
        (COL_PARAM_INIT_REMAINING_AFTER, COL_PARAM_INIT_REMAINING_BEFORE),
    ] {
        b.push_row(
            [(COL_HOST_ARGS_ACTIVE_BEFORE, F::ONE), (COL_HOST_RESULT_ACTIVE, F::ONE)],
            [(after, F::ONE), (before, -F::ONE)],
            [],
        );
    }
}

fn push_guest_call_flag_constraints(b: &mut R1csBuilder) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE)],
        [(COL_GUEST_CALL_ACTIVE, F::ONE)],
    );
}

fn push_call_param_init_enter_mode_constraints(b: &mut R1csBuilder) {
    let guest_call = COL_GUEST_CALL_ACTIVE;

    b.push_row(
        [(COL_IS_PROGRAM_ROW, F::ONE), (COL_GUEST_CALL_ACTIVE, -F::ONE)],
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
    let push = COL_GUEST_CALL_ACTIVE;
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
    // A clean halt returns from the top-level frame, but a trap terminates
    // immediately and may leave abandoned caller frames on the call stack.
    b.push_row(
        [(COL_HALTED, F::ONE), (COL_TRAPPED_AFTER, -F::ONE)],
        [(COL_CALL_STACK_DEPTH_BEFORE, F::ONE)],
        [],
    );
}

fn push_locals_fbp_transition_constraints(b: &mut R1csBuilder) {
    let guest_call = COL_GUEST_CALL_ACTIVE;
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
        [
            (COL_LOCALS_FBP_AFTER, F::ONE),
            (COL_CALL_STACK_CALLER_FBP_VALUE, -F::ONE),
        ],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (guest_call, -F::ONE), (pop, -F::ONE)],
        [(COL_LOCALS_FBP_AFTER, F::ONE), (COL_LOCALS_FBP_BEFORE, -F::ONE)],
        [],
    );
}
