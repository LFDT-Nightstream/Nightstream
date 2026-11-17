use super::super::gadgets::{push_gated_linear_zero, push_unsigned_ge_of_u32_operands};
use super::super::isa::{WasmMemoryAccessKind, WasmOpcode};
use super::super::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_IS_TRAP, COL_CI_ENTRY_IS_NULL,
    COL_CI_ENTRY_NULL_INV, COL_CI_OOB, COL_CI_TYPE_EQ, COL_CI_TYPE_EQ_INV, COL_CMP_GE, COL_CMP_LOW,
    COL_DIV_DIVIDEND_IS_MIN, COL_DIV_DIVIDEND_MIN_INV, COL_DIV_DIVISOR_INV, COL_DIV_DIVISOR_IS_NEG1,
    COL_DIV_DIVISOR_IS_ZERO, COL_DIV_DIVISOR_NEG1_INV, COL_DIV_OVERFLOW, COL_DIV_OVERFLOW_COND, COL_DIV_TRAP,
    COL_EXPECTED_TYPE_ID, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_REF, COL_FUNCTION_TYPE_ID,
    COL_IS_PROGRAM_ROW, COL_MEMORY_PAGES_BEFORE, COL_MEM_LOAD_LIVE, COL_MEM_OOB, COL_MEM_STORE_LIVE, COL_ONE,
    COL_OUTPUT_CAPTURED, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO, COL_TABLE_INDEX, COL_TABLE_OOB, COL_TABLE_SET_LIVE, COL_TABLE_SIZE, COL_TABLE_VALUE,
    COL_TRAPPED_AFTER, COL_TRAPPED_BEFORE,
};
use super::super::relation_layout::{LinearMemoryColumns, WasmRelationLayout};
use super::{always, idx, shared, WasmTaggedR1csBuilder};
use neo_application::ZeroTest;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub(crate) const CALL_INDIRECT_ENTRY_ZERO_TEST: ZeroTest =
    ZeroTest::column(COL_TABLE_VALUE, COL_CI_ENTRY_NULL_INV, COL_CI_ENTRY_IS_NULL);

pub(crate) const DIVISOR_ZERO_TEST: ZeroTest<2> = ZeroTest {
    expression: [
        (COL_STACK_READ_VALUE_LO[1], F::ONE),
        (COL_STACK_READ_VALUE_HI[1], F::ONE),
    ],
    inverse: COL_DIV_DIVISOR_INV,
    is_zero: COL_DIV_DIVISOR_IS_ZERO,
};

pub(crate) const CALL_INDIRECT_TYPE_ZERO_TEST: ZeroTest<2> = ZeroTest {
    expression: [(COL_FUNCTION_TYPE_ID, F::ONE), (COL_EXPECTED_TYPE_ID, F::NEG_ONE)],
    inverse: COL_CI_TYPE_EQ_INV,
    is_zero: COL_CI_TYPE_EQ,
};

/// The dividend test packs U32 limbs as lo + 2^32 * hi and subtracts the
/// selector-chosen MIN.
///
/// For i64, the result of the packing may overflow the field element, so
/// this test is not a general equality check.
///
/// However, in this case we are specifically comparing to 2^63, and for
/// limbs compared to 32-bits, there is only one combination that equals
/// to it.
///
/// The reason for this is that the result is in [0, 2^64) and the
/// modulus is q = 2^64 - 59
///
/// This means only the last 2^64 - q = 59 values overflow/are aliased.
///
/// So for values in [0, 59) there are two preimages, but 2^63 >
/// 58 so it has a unique combination of limbs.
///
/// For i32 the combination is just straight-forwardly injective in the
/// whole range. And the high limb is pinned to zero in that case by the
/// `narrow high limbs zero` constraint (wide_values_enabled = 0 on
/// signed i32 div/rem rows).
pub(crate) fn dividend_min_zero_test() -> ZeroTest<6> {
    let i32_div_s = selector_col(WasmOpcode::I32DivS).expect("i32.div_s selector");
    let i32_rem_s = selector_col(WasmOpcode::I32RemS).expect("i32.rem_s selector");
    let i64_div_s = selector_col(WasmOpcode::I64DivS).expect("i64.div_s selector");
    let i64_rem_s = selector_col(WasmOpcode::I64RemS).expect("i64.rem_s selector");
    ZeroTest {
        expression: [
            (COL_STACK_READ_VALUE_LO[0], F::ONE),
            (COL_STACK_READ_VALUE_HI[0], F::from_u64(1 << 32)),
            (i32_div_s, -F::from_u64(1 << 31)),
            (i32_rem_s, -F::from_u64(1 << 31)),
            (i64_div_s, -F::from_u64(1 << 63)),
            (i64_rem_s, -F::from_u64(1 << 63)),
        ],
        inverse: COL_DIV_DIVIDEND_MIN_INV,
        is_zero: COL_DIV_DIVIDEND_IS_MIN,
    }
}

pub(crate) fn divisor_neg1_zero_test() -> ZeroTest<6> {
    let i32_div_s = selector_col(WasmOpcode::I32DivS).expect("i32.div_s selector");
    let i32_rem_s = selector_col(WasmOpcode::I32RemS).expect("i32.rem_s selector");
    let i64_div_s = selector_col(WasmOpcode::I64DivS).expect("i64.div_s selector");
    let i64_rem_s = selector_col(WasmOpcode::I64RemS).expect("i64.rem_s selector");
    ZeroTest {
        expression: [
            (COL_STACK_READ_VALUE_LO[1], F::ONE),
            (COL_STACK_READ_VALUE_HI[1], F::ONE),
            (i32_div_s, -F::from_u64(u32::MAX as u64)),
            (i32_rem_s, -F::from_u64(u32::MAX as u64)),
            (i64_div_s, -F::from_u64(u32::MAX as u64 + u32::MAX as u64)),
            (i64_rem_s, -F::from_u64(u32::MAX as u64 + u32::MAX as u64)),
        ],
        inverse: COL_DIV_DIVISOR_NEG1_INV,
        is_zero: COL_DIV_DIVISOR_IS_NEG1,
    }
}

pub(super) fn push_trap_constraints(b: &mut WasmTaggedR1csBuilder<'_>, layout: &WasmRelationLayout) {
    b.with_tag(always("div trap"), |b| {
        push_div_trap_constraints(b);
    });
    b.with_tag(
        shared(
            "table bounds and indirect call trap",
            &[
                WasmOpcode::TableGet,
                WasmOpcode::TableSet,
                WasmOpcode::CallIndirectResolve,
            ],
        ),
        |b| {
            push_table_bounds_and_call_indirect_trap_constraints(b);
        },
    );
    b.with_tag(shared("linear memory oob trap", &linear_memory_ops()), |b| {
        push_linear_memory_oob_trap_constraints(b, &layout.linear_memory);
    });
    b.with_tag(always("trap transition"), |b| {
        push_trapped_state_transition_constraints(b);
    });
}

fn linear_memory_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.memory_access_info().is_some())
        .collect()
}

fn memory_ops_of_kind(kind: WasmMemoryAccessKind) -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| {
            op.memory_access_info()
                .is_some_and(|access| access.kind == kind)
        })
        .collect()
}

fn push_linear_memory_oob_trap_constraints(b: &mut WasmTaggedR1csBuilder<'_>, linear_memory: &LinearMemoryColumns) {
    let mut mem_selectors: Vec<usize> = linear_memory_ops()
        .into_iter()
        .map(|op| selector_col(op).expect("linear memory selector"))
        .collect();
    mem_selectors.extend([
        super::host_event_chain::gather_memory_read_kind_col(),
        super::host_event_chain::gather_memory_write_kind_col(),
    ]);
    push_unsigned_ge_of_u32_operands(
        b,
        mem_selectors.iter().copied(),
        [
            (idx(linear_memory.lane0_addr), F::ONE),
            (idx(linear_memory.use_lane1), F::ONE),
            (idx(linear_memory.use_lane2), F::ONE),
        ],
        [(COL_MEMORY_PAGES_BEFORE, F::from_u64(crate::preprocess::WASM_PAGE_LANES))],
        COL_CMP_LOW,
        COL_CMP_GE,
    );
    b.push_row(
        mem_selectors.iter().map(|&s| (s, F::ONE)),
        [(COL_CMP_GE, F::ONE)],
        [(COL_MEM_OOB, F::ONE)],
    );
    b.push_row(
        memory_ops_of_kind(WasmMemoryAccessKind::Load)
            .into_iter()
            .map(|op| (selector_col(op).expect("load selector"), F::ONE)),
        [(COL_ONE, F::ONE), (COL_MEM_OOB, -F::ONE)],
        [(COL_MEM_LOAD_LIVE, F::ONE)],
    );
    b.push_row(
        memory_ops_of_kind(WasmMemoryAccessKind::Store)
            .into_iter()
            .map(|op| (selector_col(op).expect("store selector"), F::ONE)),
        [(COL_ONE, F::ONE), (COL_MEM_OOB, -F::ONE)],
        [(COL_MEM_STORE_LIVE, F::ONE)],
    );
}

fn div_rem_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.traps_on_zero_divisor())
        .collect()
}

fn signed_div_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.traps_on_signed_overflow())
        .collect()
}

fn push_div_trap_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    DIVISOR_ZERO_TEST.push_constraints(b);

    dividend_min_zero_test().push_constraints(b);
    divisor_neg1_zero_test().push_constraints(b);
    b.push_row(
        [(COL_DIV_DIVIDEND_IS_MIN, F::ONE)],
        [(COL_DIV_DIVISOR_IS_NEG1, F::ONE)],
        [(COL_DIV_OVERFLOW_COND, F::ONE)],
    );
    b.push_row(
        signed_div_ops()
            .into_iter()
            .map(|op| (selector_col(op).expect("signed div selector"), F::ONE)),
        [(COL_DIV_OVERFLOW_COND, F::ONE)],
        [(COL_DIV_OVERFLOW, F::ONE)],
    );
    b.push_row(
        div_rem_ops()
            .into_iter()
            .map(|op| (selector_col(op).expect("div/rem selector"), F::ONE)),
        [(COL_DIV_DIVISOR_IS_ZERO, F::ONE)],
        [(COL_DIV_TRAP, F::ONE), (COL_DIV_OVERFLOW, -F::ONE)],
    );
    push_gated_linear_zero(b, COL_DIV_TRAP, [(COL_STACK_WRITE0_VALUE_LO, F::ONE)]);
    push_gated_linear_zero(b, COL_DIV_TRAP, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);
}

/// SPEC.md section 7.8.
fn push_table_bounds_and_call_indirect_trap_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let table_get = selector_col(WasmOpcode::TableGet).unwrap();
    let table_set = selector_col(WasmOpcode::TableSet).unwrap();
    let resolve = selector_col(WasmOpcode::CallIndirectResolve).unwrap();
    let indirect_selectors = [(resolve, F::ONE)];

    push_unsigned_ge_of_u32_operands(
        b,
        [table_get, table_set, resolve],
        [(COL_TABLE_INDEX, F::ONE)],
        [(COL_TABLE_SIZE, F::ONE)],
        COL_CMP_LOW,
        COL_CMP_GE,
    );

    b.push_row(indirect_selectors, [(COL_CMP_GE, F::ONE)], [(COL_CI_OOB, F::ONE)]);
    b.push_row(
        [(table_get, F::ONE), (table_set, F::ONE)],
        [(COL_CMP_GE, F::ONE)],
        [(COL_TABLE_OOB, F::ONE)],
    );
    b.push_row(
        [(table_set, F::ONE)],
        [(COL_ONE, F::ONE), (COL_CMP_GE, -F::ONE)],
        [(COL_TABLE_SET_LIVE, F::ONE)],
    );
    push_gated_linear_zero(b, COL_TABLE_OOB, [(COL_STACK_WRITE0_VALUE_LO, F::ONE)]);
    push_gated_linear_zero(b, COL_TABLE_OOB, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);

    CALL_INDIRECT_ENTRY_ZERO_TEST.push_constraints(b);

    CALL_INDIRECT_TYPE_ZERO_TEST.push_constraints(b);
    // Rocq `IndirectBindings.full_resolve_checks`.
    b.push_row(
        indirect_selectors
            .into_iter()
            .chain([(COL_CI_OOB, -F::ONE)]),
        [(COL_ONE, F::ONE), (COL_CI_ENTRY_IS_NULL, -F::ONE)],
        [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, F::ONE)],
    );

    b.push_row(
        [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, F::ONE)],
        [(COL_CI_TYPE_EQ, F::ONE)],
        indirect_selectors
            .into_iter()
            .chain([(COL_CALL_INDIRECT_IS_TRAP, -F::ONE)]),
    );

    b.push_row(
        indirect_selectors,
        [(COL_ONE, F::ONE), (COL_CALL_INDIRECT_IS_TRAP, -F::ONE)],
        [(COL_CALL_INDIRECT_IS_NOT_TRAP, F::ONE)],
    );
    push_gated_linear_zero(
        b,
        COL_FUNCTION_CALL_TYPE_LOOKUP_GATE,
        [(COL_FUNCTION_REF, F::ONE), (COL_TABLE_VALUE, -F::ONE)],
    );
}

fn push_trapped_state_transition_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.push_linear_zero([
        (COL_TRAPPED_AFTER, F::ONE),
        (COL_TRAPPED_BEFORE, -F::ONE),
        (selector_col(WasmOpcode::Unreachable).unwrap(), -F::ONE),
        (COL_DIV_TRAP, -F::ONE),
        (COL_CALL_INDIRECT_IS_TRAP, -F::ONE),
        (COL_MEM_OOB, -F::ONE),
        (COL_TABLE_OOB, -F::ONE),
    ]);
    b.push_row([(COL_IS_PROGRAM_ROW, F::ONE)], [(COL_TRAPPED_BEFORE, F::ONE)], []);
    b.push_row([(COL_OUTPUT_CAPTURED, F::ONE)], [(COL_TRAPPED_AFTER, F::ONE)], []);
}
