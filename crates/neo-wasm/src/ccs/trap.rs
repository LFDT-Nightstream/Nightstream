use super::super::gadgets::{
    push_gated_linear_zero, push_unsigned_ge_gadget, push_zero_test_expr_gadget, push_zero_test_gadget,
};
use super::super::isa::{WasmMemoryAccessKind, WasmOpcode};
use super::super::layout::{
    selector_col, COL_CALL_INDIRECT_IS_NOT_TRAP, COL_CALL_INDIRECT_IS_TRAP, COL_CI_ENTRY_IS_NULL,
    COL_CI_ENTRY_NULL_INV, COL_CI_OOB, COL_CI_TYPE_EQ, COL_CI_TYPE_EQ_INV, COL_CMP_GE, COL_CMP_LOW,
    COL_DIV_DIVIDEND_IS_MIN, COL_DIV_DIVIDEND_MIN_INV, COL_DIV_DIVISOR_INV, COL_DIV_DIVISOR_IS_NEG1,
    COL_DIV_DIVISOR_IS_ZERO, COL_DIV_DIVISOR_NEG1_INV, COL_DIV_OVERFLOW, COL_DIV_OVERFLOW_COND, COL_DIV_TRAP,
    COL_EXPECTED_TYPE_ID, COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, COL_FUNCTION_TYPE_ID, COL_MEMORY_PAGES_BEFORE,
    COL_MEM_LOAD_LIVE, COL_MEM_OOB, COL_MEM_STORE_LIVE, COL_ONE, COL_OUTPUT_CAPTURED, COL_STACK_READ0_VALUE_HI,
    COL_STACK_READ0_VALUE_LO, COL_STACK_READ1_VALUE_HI, COL_STACK_READ1_VALUE_LO, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO, COL_TABLE_INDEX, COL_TABLE_SIZE, COL_TABLE_VALUE,
};
use super::super::lookup_binding_builder::{
    CallColumns, ControlColumns, LinearMemoryColumns, StateColumns, WasmLookupBindingLayout,
};
use super::{always, idx, opcode_tag, shared, R1csBuilder};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Emit trap-cause flags and the state transition that carries `trapped`.
pub(super) fn push_trap_constraints(b: &mut R1csBuilder, layout: &WasmLookupBindingLayout) {
    b.with_tag(always("div trap"), |b| {
        push_div_trap_constraints(b);
    });
    b.with_tag(opcode_tag("call_indirect trap", WasmOpcode::CallIndirect), |b| {
        push_call_indirect_trap_constraints(b, &layout.call);
    });
    b.with_tag(shared("linear memory oob trap", &linear_memory_ops()), |b| {
        push_linear_memory_oob_trap_constraints(b, &layout.linear_memory);
    });
    b.with_tag(always("trap transition"), |b| {
        push_trapped_state_transition_constraints(b, &layout.control, &layout.state);
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

/// Derives the linear-memory OOB trap bit.
///
/// The bound is word-lane based:
/// `lane0_addr + use_lane1 + use_lane2 >= 16384 * memory_pages_before`.
/// This is exact because wasm pages are aligned to 4-byte lanes.
///
/// OOB rows de-gate memory tuples; the trap transition consumes `COL_MEM_OOB`.
fn push_linear_memory_oob_trap_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    let mem_selectors: Vec<usize> = linear_memory_ops()
        .into_iter()
        .map(|op| selector_col(op).expect("linear memory selector"))
        .collect();
    // ge = highest touched lane >= memory size in lanes.
    push_unsigned_ge_gadget(
        b,
        mem_selectors.iter().copied(),
        [
            (idx(linear_memory.lane0_addr), F::ONE),
            (idx(linear_memory.use_lane1), F::ONE),
            (idx(linear_memory.use_lane2), F::ONE),
        ],
        [(COL_MEMORY_PAGES_BEFORE, F::from_u64(16384))],
        COL_CMP_LOW,
        COL_CMP_GE,
    );
    // mem_oob = (Σ load/store selectors) · ge.
    b.push_row(
        mem_selectors.iter().map(|&s| (s, F::ONE)),
        [(COL_CMP_GE, F::ONE)],
        [(COL_MEM_OOB, F::ONE)],
    );
    // OOB rows assert no real memory access.
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

/// Div/rem opcodes that trap on a zero divisor. Spec-derived from
/// [`WasmOpcode::traps_on_zero_divisor`].
fn div_rem_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.traps_on_zero_divisor())
        .collect()
}

/// Signed div opcodes that trap on overflow (MIN / -1). Spec-derived from
/// [`WasmOpcode::traps_on_signed_overflow`].
fn signed_div_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.traps_on_signed_overflow())
        .collect()
}

fn push_div_trap_constraints(b: &mut R1csBuilder) {
    // Div/rem by zero is terminal. Since both divisor limbs are U32,
    // read1_lo + read1_hi is below the field modulus, so the sum is zero
    // exactly when both limbs are zero.
    let divisor = [(COL_STACK_READ1_VALUE_LO, F::ONE), (COL_STACK_READ1_VALUE_HI, F::ONE)];
    push_zero_test_expr_gadget(b, divisor, COL_DIV_DIVISOR_INV, COL_DIV_DIVISOR_IS_ZERO);

    // Signed division overflow (MIN / -1) is a trap because the result
    // is not representable in 2's complement.
    //
    // The dividend test packs U32 limbs as lo + 2^32 * hi and subtracts the
    // selector-chosen MIN.
    //
    // For i64, the result of the packing may overflow the field element, so
    // this test is not a general equality check.
    //
    // However, in this case we are specifically comparing to 2^63, and for
    // limbs compared to 32-bits, there is only one combination that equals
    // to it.
    //
    // The reason for this is that the result is in [0, 2^64) and the
    // Goldilocks modulus is q = 2^64 - 2^32 + 1
    //
    // This means only the last 2^64 - q = 2^32 - 1 values overflow/are aliased.
    //
    // So for values in [0, 2^32 - 1) there are two preimages, but 2^63 >
    // 2^32 - 2 so it has a unique combination of limbs.
    //
    // For i32 the combination is just straight-forwardly injective in the
    // whole range. And the high limb is pinned to zero in that case by the
    // `narrow high limbs zero` constraint (wide_values_enabled = 0 on
    // i32.div_s rows).
    let i32_div_s = selector_col(WasmOpcode::I32DivS).expect("i32.div_s selector");
    let i64_div_s = selector_col(WasmOpcode::I64DivS).expect("i64.div_s selector");
    let dividend_min = [
        (COL_STACK_READ0_VALUE_LO, F::ONE),
        (COL_STACK_READ0_VALUE_HI, F::from_u64(1 << 32)),
        (i32_div_s, -F::from_u64(1 << 31)),
        (i64_div_s, -F::from_u64(1 << 63)),
    ];
    push_zero_test_expr_gadget(b, dividend_min, COL_DIV_DIVIDEND_MIN_INV, COL_DIV_DIVIDEND_IS_MIN);

    // NOTE: this is not a limb composition, just a limb sum.
    //
    // Composition is not needed because we just need both limbs to equal
    // specific values.
    //
    // Since the limbs are 32-bit constrained, there is only one way to
    // add to u32::MAX * 2 in the 64-bit case, and in the other case the
    // high limb is pinned to zero (same `narrow high limbs zero`
    // constraint as above), so it degrades to a simple equality check.
    let divisor_neg1 = [
        (COL_STACK_READ1_VALUE_LO, F::ONE),
        (COL_STACK_READ1_VALUE_HI, F::ONE),
        // limb sum of -1i32: u32::MAX (the high limb is 0)
        (i32_div_s, -F::from_u64(u32::MAX as u64)),
        // limb sum of -1i64: both limbs are u32::MAX
        (i64_div_s, -F::from_u64(u32::MAX as u64 + u32::MAX as u64)),
    ];
    push_zero_test_expr_gadget(b, divisor_neg1, COL_DIV_DIVISOR_NEG1_INV, COL_DIV_DIVISOR_IS_NEG1);
    b.push_row(
        [(COL_DIV_DIVIDEND_IS_MIN, F::ONE)],
        [(COL_DIV_DIVISOR_IS_NEG1, F::ONE)],
        [(COL_DIV_OVERFLOW_COND, F::ONE)],
    );
    // The scratch flags above are only meaningful on div_s rows; this
    // selector-gated product makes them harmless everywhere else.
    b.push_row(
        signed_div_ops()
            .into_iter()
            .map(|op| (selector_col(op).expect("signed div selector"), F::ONE)),
        [(COL_DIV_OVERFLOW_COND, F::ONE)],
        [(COL_DIV_OVERFLOW, F::ONE)],
    );
    // Zero and -1 divisors are mutually exclusive, so combining the
    // zero-divisor and signed-overflow trap causes still leaves div_trap
    // boolean.
    b.push_row(
        div_rem_ops()
            .into_iter()
            .map(|op| (selector_col(op).expect("div/rem selector"), F::ONE)),
        [(COL_DIV_DIVISOR_IS_ZERO, F::ONE)],
        [(COL_DIV_TRAP, F::ONE), (COL_DIV_OVERFLOW, -F::ONE)],
    );
    // The faulting row has normal stack arity, but no real result: pin
    // the synthetic write to zero and de-gate the op-table lookup below.
    push_gated_linear_zero(b, COL_DIV_TRAP, [(COL_STACK_WRITE0_VALUE_LO, F::ONE)]);
    push_gated_linear_zero(b, COL_DIV_TRAP, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);
}

fn push_call_indirect_trap_constraints(b: &mut R1csBuilder, call: &CallColumns) {
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    // ge = selector * ( table_index >= table_size )
    push_unsigned_ge_gadget(
        b,
        [call_indirect],
        [(COL_TABLE_INDEX, F::ONE)],
        [(COL_TABLE_SIZE, F::ONE)],
        COL_CMP_LOW,
        COL_CMP_GE,
    );

    // COL_CI_OOB = selector * ge.
    b.push_row(
        [(call_indirect, F::ONE)],
        [(COL_CMP_GE, F::ONE)],
        [(COL_CI_OOB, F::ONE)],
    );

    // 0 encodes the null funcref
    push_zero_test_gadget(b, COL_TABLE_VALUE, COL_CI_ENTRY_NULL_INV, COL_CI_ENTRY_IS_NULL);

    // typecheck
    push_zero_test_expr_gadget(
        b,
        [(COL_FUNCTION_TYPE_ID, F::ONE), (COL_EXPECTED_TYPE_ID, -F::ONE)],
        COL_CI_TYPE_EQ_INV,
        COL_CI_TYPE_EQ,
    );
    //
    // we have 3 possible trap cases, characterized by the following two equations:
    //
    // 1. type_lookup = call_indirect * !oob * !null
    // -> (r1cs trick, works because oob is already gated by call_indirect)
    // 1. type_lookup = (call_indirect - oob) * (1 - null)
    //
    // 2. type_lookup * type_eq = call_indirect - is_trap
    //
    // if type_lookup = 1,
    //
    // type_lookup => call_indirect (because of 1)
    //
    // so
    //
    // type_eq = 1 - is_trap
    // ->
    // type_eq = !is_trap
    //
    // if type_lookup = 0, then
    //
    // either call_indirect = 0, so 0 = 0 - 0 and is_trap = 0
    //
    // or call_indirect = 1, and is_trap = 1
    //
    // which forces either oob or null
    //
    b.push_row(
        [(call_indirect, F::ONE), (COL_CI_OOB, -F::ONE)],
        [(COL_ONE, F::ONE), (COL_CI_ENTRY_IS_NULL, -F::ONE)],
        [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, F::ONE)],
    );

    b.push_row(
        [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, F::ONE)],
        [(COL_CI_TYPE_EQ, F::ONE)],
        [(call_indirect, F::ONE), (COL_CALL_INDIRECT_IS_TRAP, -F::ONE)],
    );

    b.push_row(
        [(call_indirect, F::ONE)],
        [(COL_ONE, F::ONE), (COL_CALL_INDIRECT_IS_TRAP, -F::ONE)],
        [(COL_CALL_INDIRECT_IS_NOT_TRAP, F::ONE)],
    );
    // A trapping row never enters the callee: no call-stack push (and via
    // the existing enter-mode gating, no param-init mode).
    b.push_row(
        [(COL_CALL_INDIRECT_IS_TRAP, F::ONE)],
        [(idx(call.call_stack_push_present), F::ONE)],
        [],
    );
}

fn push_trapped_state_transition_constraints(b: &mut R1csBuilder, control: &ControlColumns, state: &StateColumns) {
    b.push_linear_zero([
        (idx(state.trapped_after), F::ONE),
        (idx(state.trapped_before), -F::ONE),
        (selector_col(WasmOpcode::Unreachable).unwrap(), -F::ONE),
        (COL_DIV_TRAP, -F::ONE),
        (COL_CALL_INDIRECT_IS_TRAP, -F::ONE),
        (COL_MEM_OOB, -F::ONE),
    ]);
    b.push_row(
        [(idx(control.is_program_row), F::ONE)],
        [(idx(state.trapped_before), F::ONE)],
        [],
    );
    b.push_row(
        [(COL_OUTPUT_CAPTURED, F::ONE)],
        [(idx(state.trapped_after), F::ONE)],
        [],
    );
}
