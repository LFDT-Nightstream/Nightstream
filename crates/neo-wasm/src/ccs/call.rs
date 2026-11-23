use super::super::gadgets::push_gated_linear_zero;
use super::super::ir::{FUNCTION_CALL_METADATA_GUEST_FACTOR, FUNCTION_CALL_METADATA_RESULT_FACTOR};
use super::super::isa::WasmOpcode;
use super::super::layout::{
    selector_col, COL_ACTIVATION_FINISHED_AFTER, COL_ACTIVATION_FINISHED_BEFORE, COL_CALL_PARAM_COUNT,
    COL_CALL_RESULT_COUNT, COL_CALL_STACK_ADDR, COL_CALL_STACK_CALLER_FBP_VALUE, COL_CALL_STACK_CALLER_SP_BASE_VALUE,
    COL_CALL_STACK_DEPTH_AFTER, COL_CALL_STACK_DEPTH_BEFORE, COL_CALL_STACK_POP_PRESENT, COL_CALL_STACK_PUSH_PRESENT,
    COL_CALL_STACK_RETURN_PC_VALUE, COL_CALL_TARGET_METADATA, COL_CI_HOST_CALL, COL_ENTERED_FUNCTION_NUM_LOCALS,
    COL_EXIT_SLIDE_DISTANCE, COL_FRAME_MOVE_ACTIVE_AFTER, COL_FRAME_MOVE_ACTIVE_BEFORE, COL_FRAME_MOVE_DISTANCE_AFTER,
    COL_FRAME_MOVE_DISTANCE_BEFORE, COL_FRAME_MOVE_REMAINING_AFTER, COL_FRAME_MOVE_REMAINING_AFTER_INV,
    COL_FRAME_MOVE_REMAINING_AFTER_IS_ZERO, COL_FRAME_MOVE_REMAINING_BEFORE, COL_FRAME_SLIDE_ACTIVE, COL_FRAME_SP_DROP,
    COL_FUNCTION_REF, COL_GATHER_ACTIVE, COL_GUEST_ENTRY_ACTIVE, COL_HOST_CALLEE_FREF_AFTER,
    COL_HOST_CALLEE_FREF_BEFORE, COL_HOST_CALL_ACTIVE, COL_HOST_EVENT_EXIT_LATCH, COL_IS_PROGRAM_ROW,
    COL_LOCALS_FBP_AFTER, COL_LOCALS_FBP_BEFORE, COL_LOCAL_ZERO_ACTIVE_AFTER, COL_LOCAL_ZERO_ACTIVE_BEFORE,
    COL_LOCAL_ZERO_REMAINING_AFTER, COL_LOCAL_ZERO_REMAINING_AFTER_INV, COL_LOCAL_ZERO_REMAINING_AFTER_IS_ZERO,
    COL_LOCAL_ZERO_REMAINING_BEFORE, COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_OUTPUT_CAPTURED,
    COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE,
    COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE, COL_PADDING_ACTIVE, COL_PC_AFTER, COL_PC_BEFORE,
    COL_SP_AFTER, COL_SP_BEFORE, COL_STACK_FRAME_BASE_AFTER, COL_STACK_FRAME_BASE_BEFORE, COL_STACK_READS,
    COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ADDR_LO,
    COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITES, COL_TARGET_FUNCTION_IS_GUEST,
    COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE, COL_TRAP_UNWIND, COL_TURN_BOUNDARY,
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
        b.push_linear_zero([
            (COL_IS_PROGRAM_ROW, F::ONE),
            (COL_FRAME_MOVE_ACTIVE_BEFORE, F::ONE),
            (COL_LOCAL_ZERO_ACTIVE_BEFORE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
            (COL_TURN_BOUNDARY, F::ONE),
            (COL_HOST_RESUME, F::ONE),
            (COL_HOST_SUSPEND, F::ONE),
            (COL_ONE, -F::ONE),
        ]);
    });

    b.with_tag(always("padding row state preservation"), |b| {
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
        let aux_row_gate = [
            (COL_HOST_SUSPEND, F::ONE),
            (COL_FRAME_MOVE_ACTIVE_BEFORE, F::ONE),
            (COL_LOCAL_ZERO_ACTIVE_BEFORE, F::ONE),
            (COL_PADDING_ACTIVE, F::ONE),
            (COL_GATHER_ACTIVE, F::ONE),
        ];

        b.push_row(aux_row_gate, [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE)], []);
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
        push_gated_linear_zero(b, COL_TURN_BOUNDARY, [(COL_ACTIVATION_FINISHED_AFTER, F::ONE)]);
        b.push_row(
            [
                (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
                (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
                (COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
            ],
            [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_NESTED_ENTRY, -F::ONE)],
            [(COL_ONE, F::ONE), (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE)],
            [],
        );
    });

    b.with_tag(always("turn boundary row"), |b| {
        b.push_row(
            [
                (COL_TURN_BOUNDARY, F::ONE),
                (COL_HOST_NESTED_ENTRY, -F::ONE),
                (COL_TRAP_UNWIND, -F::ONE),
            ],
            [(COL_SP_BEFORE, F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_TURN_BOUNDARY, F::ONE),
                (COL_HOST_NESTED_ENTRY, -F::ONE),
                (COL_TRAP_UNWIND, -F::ONE),
            ],
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
        for (enabled, value) in [
            (COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_LO_BEFORE),
            (COL_OUTPUT_ENABLED_BEFORE, COL_OUTPUT_VALUE_HI_BEFORE),
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_VALUE_LO_AFTER),
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_VALUE_HI_AFTER),
        ] {
            b.push_row([(COL_ONE, F::ONE), (enabled, -F::ONE)], [(value, F::ONE)], []);
        }
        for (after, before) in [
            (COL_OUTPUT_ENABLED_AFTER, COL_OUTPUT_ENABLED_BEFORE),
            (COL_OUTPUT_VALUE_LO_AFTER, COL_OUTPUT_VALUE_LO_BEFORE),
            (COL_OUTPUT_VALUE_HI_AFTER, COL_OUTPUT_VALUE_HI_BEFORE),
        ] {
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

        b.push_row(
            [(COL_TURN_BOUNDARY, F::ONE), (COL_HOST_RESUME, F::ONE)],
            [(COL_OUTPUT_ENABLED_BEFORE, F::ONE)],
            [
                (COL_OUTPUT_ENABLED_BEFORE, F::ONE),
                (COL_OUTPUT_CAPTURED, F::ONE),
                (COL_OUTPUT_ENABLED_AFTER, -F::ONE),
            ],
        );
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

fn clean_activation_return_terms() -> [(usize, F); 6] {
    [
        (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
        (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
        (COL_TRAPPED_AFTER, -F::ONE),
        (COL_TRAPPED_BEFORE, F::ONE),
        (COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
        (COL_TRAP_UNWIND, -F::ONE),
    ]
}

fn push_dynamic_call_stack_arity_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
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
        [
            (COL_STACK_READS, F::ONE),
            (call_indirect, -F::ONE),
            (return_call_indirect, -F::ONE),
        ],
        [],
    );
    b.push_row(
        [(call_indirect, F::ONE), (return_call_indirect, F::ONE)],
        [(COL_FUNCTION_REF, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
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

pub(super) fn host_call_gate_terms() -> [(usize, F); 5] {
    [
        (selector_col(WasmOpcode::Call).unwrap(), F::ONE),
        (selector_col(WasmOpcode::ReturnCall).unwrap(), F::ONE),
        (selector_col(WasmOpcode::CallIndirect).unwrap(), F::ONE),
        (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), F::ONE),
        (COL_GUEST_ENTRY_ACTIVE, -F::ONE),
    ]
}

fn push_host_call_attribution_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.push_row(
        [
            (selector_col(WasmOpcode::CallIndirect).unwrap(), F::ONE),
            (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), F::ONE),
        ],
        [(COL_ONE, F::ONE), (COL_TARGET_FUNCTION_IS_GUEST, -F::ONE)],
        [(COL_CI_HOST_CALL, F::ONE)],
    );
    push_gated_linear_zero(
        b,
        COL_HOST_CALL_ACTIVE,
        [(COL_PC_AFTER, F::ONE), (COL_PC_BEFORE, -F::ONE), (COL_ONE, -F::ONE)],
    );

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
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), -F::ONE),
            (COL_GUEST_ENTRY_ACTIVE, F::ONE),
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
        [(COL_TARGET_FUNCTION_IS_GUEST, F::ONE), (COL_ONE, -F::ONE)],
    );
}

fn push_local_zero_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let zero = COL_LOCAL_ZERO_ACTIVE_BEFORE;
    LOCAL_ZERO_REMAINING_AFTER_ZERO_TEST.push_constraints(b);
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
    push_gated_linear_zero(
        b,
        zero,
        [(COL_STACK_WRITE0_ADDR_LO, F::ONE), (COL_SP_BEFORE, -F::from_u64(2))],
    );
    push_gated_linear_zero(b, zero, [(COL_STACK_WRITE0_VALUE_LO, F::ONE)]);
    push_gated_linear_zero(b, zero, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);
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

fn push_frame_move_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let slide = COL_FRAME_SLIDE_ACTIVE;
    let moving = COL_FRAME_MOVE_ACTIVE_BEFORE;
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
    b.push_row(
        [(moving, F::ONE), (COL_PADDING_ACTIVE, F::ONE)],
        [
            (COL_FRAME_MOVE_DISTANCE_AFTER, F::ONE),
            (COL_FRAME_MOVE_DISTANCE_BEFORE, -F::ONE),
        ],
        [],
    );
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

    b.push_row(
        [(COL_TRAP_UNWIND, F::ONE)],
        [(COL_CALL_STACK_DEPTH_BEFORE, F::ONE)],
        [
            (COL_CALL_STACK_DEPTH_BEFORE, F::ONE),
            (COL_CALL_STACK_DEPTH_AFTER, -F::ONE),
            (push, F::ONE),
            (pop, -F::ONE),
            (COL_HOST_SUSPEND, F::ONE),
            (COL_HOST_RESUME, -F::ONE),
        ],
    );

    push_gated_linear_zero(
        b,
        push,
        [(COL_CALL_STACK_ADDR, F::ONE), (COL_CALL_STACK_DEPTH_BEFORE, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        push,
        [
            (COL_CALL_STACK_RETURN_PC_VALUE, F::ONE),
            (COL_PC_BEFORE, -F::ONE),
            (COL_ONE, -F::ONE),
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

    push_gated_linear_zero(
        b,
        push,
        [
            (COL_LOCALS_FBP_AFTER, F::ONE),
            (COL_SP_AFTER, -F::ONE),
            (COL_CALL_PARAM_COUNT, F::ONE),
        ],
    );
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
