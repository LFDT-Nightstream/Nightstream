//! Owns trap-cause derivation and the carried trapped-state transition.

use super::super::gadgets::{push_gated_linear_zero, push_zero_test_expr_gadget, push_zero_test_gadget};
use super::super::isa::WasmOpcode;
use super::super::layout::{
    selector_col, COL_CI_ENTRY_IS_NULL, COL_CI_ENTRY_LIVE, COL_CI_ENTRY_NULL_INV, COL_CI_LIVE, COL_CI_TRAP,
    COL_CI_TYPE_EQ, COL_CI_TYPE_EQ_INV, COL_CI_TYPE_MISMATCH, COL_DIV_DIVIDEND_IS_MIN, COL_DIV_DIVIDEND_MIN_INV,
    COL_DIV_DIVISOR_INV, COL_DIV_DIVISOR_IS_NEG1, COL_DIV_DIVISOR_IS_ZERO, COL_DIV_DIVISOR_NEG1_INV, COL_DIV_OVERFLOW,
    COL_DIV_OVERFLOW_COND, COL_DIV_TRAP, COL_ONE,
};
use super::super::lookup_binding_builder::{
    CallColumns, Column, ControlColumns, FunctionTypeColumns, ModuleTypeColumns, OperandStackColumns, StateColumns,
    TableColumns, WasmLookupBindingLayout,
};
use super::{always, idx, opcode_tag, R1csBuilder};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// Emit trap-cause flags and the state transition that carries `trapped`.
pub(super) fn push_trap_constraints(b: &mut R1csBuilder, layout: &WasmLookupBindingLayout) {
    b.with_tag(always("div trap"), |b| {
        push_div_trap_constraints(b, &layout.stack);
    });
    b.with_tag(opcode_tag("call_indirect trap", WasmOpcode::CallIndirect), |b| {
        push_call_indirect_trap_constraints(
            b,
            &layout.call,
            &layout.table,
            &layout.function_types,
            &layout.module_types,
        );
    });
    b.with_tag(always("trap transition"), |b| {
        push_trapped_state_transition_constraints(b, &layout.control, &layout.state, layout.output.captured);
    });
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

fn push_div_trap_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    // Div/rem by zero is terminal. Since both divisor limbs are U32,
    // read1_lo + read1_hi is below the field modulus, so the sum is zero
    // exactly when both limbs are zero.
    let divisor = [(idx(stack.read1_value_lo), F::ONE), (idx(stack.read1_value_hi), F::ONE)];
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
        (idx(stack.read0_value_lo), F::ONE),
        (idx(stack.read0_value_hi), F::from_u64(1 << 32)),
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
        (idx(stack.read1_value_lo), F::ONE),
        (idx(stack.read1_value_hi), F::ONE),
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
    push_gated_linear_zero(b, COL_DIV_TRAP, [(idx(stack.write0_value_lo), F::ONE)]);
    push_gated_linear_zero(b, COL_DIV_TRAP, [(idx(stack.write0_value_hi), F::ONE)]);
}

/// `call_indirect` trap causes: a null table entry or a callee whose
/// normalized type id differs from the instruction's expected type id.
///
/// The evidence is bound on both sides, so the prover can neither fabricate
/// nor hide a trap:
/// - the table entry is bound by the `tables` memory read (gated on
///   `table.read_enabled`, which stays on for trapping rows),
/// - the callee type id is bound by the `function_types` ROM read gated on
///   `COL_CI_ENTRY_LIVE` (active exactly when the entry is non-null),
/// - the expected type id is bound by the per-pc program decode ROM.
///
/// A trapping row's `COL_CI_TRAP` feeds the trapped-flag transition and the
/// pc-edge-kind equation in ccs.rs (mirroring `COL_DIV_TRAP`), de-gates the
/// callee metadata / entry-pc reads via `COL_CI_LIVE`, and forbids a
/// call-stack push.
fn push_call_indirect_trap_constraints(
    b: &mut R1csBuilder,
    call: &CallColumns,
    table: &TableColumns,
    function_types: &FunctionTypeColumns,
    module_types: &ModuleTypeColumns,
) {
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    push_zero_test_gadget(b, idx(table.value), COL_CI_ENTRY_NULL_INV, COL_CI_ENTRY_IS_NULL);
    push_zero_test_expr_gadget(
        b,
        [
            (idx(function_types.type_id), F::ONE),
            (idx(module_types.expected_type_id), -F::ONE),
        ],
        COL_CI_TYPE_EQ_INV,
        COL_CI_TYPE_EQ,
    );
    // mismatch = (1 - is_null) · (1 - type_eq). Scratch on non-call_indirect
    // rows, made harmless by the selector-gated trap product below.
    b.push_row(
        [(COL_ONE, F::ONE), (COL_CI_ENTRY_IS_NULL, -F::ONE)],
        [(COL_ONE, F::ONE), (COL_CI_TYPE_EQ, -F::ONE)],
        [(COL_CI_TYPE_MISMATCH, F::ONE)],
    );
    // trap = ind · (is_null + mismatch); the two causes are mutually
    // exclusive (mismatch carries a 1 - is_null factor), so the sum is
    // boolean. On a healthy call_indirect row this forces is_null = 0 and
    // type_eq = 1, subsuming the old standalone type-equality constraint.
    b.push_row(
        [(call_indirect, F::ONE)],
        [(COL_CI_ENTRY_IS_NULL, F::ONE), (COL_CI_TYPE_MISMATCH, F::ONE)],
        [(COL_CI_TRAP, F::ONE)],
    );
    b.push_row(
        [(call_indirect, F::ONE)],
        [(COL_ONE, F::ONE), (COL_CI_ENTRY_IS_NULL, -F::ONE)],
        [(COL_CI_ENTRY_LIVE, F::ONE)],
    );
    b.push_row(
        [(call_indirect, F::ONE)],
        [(COL_ONE, F::ONE), (COL_CI_TRAP, -F::ONE)],
        [(COL_CI_LIVE, F::ONE)],
    );
    // A trapping row never enters the callee: no call-stack push (and via
    // the existing enter-mode gating, no param-init mode).
    b.push_row(
        [(COL_CI_TRAP, F::ONE)],
        [(idx(call.call_stack_push_present), F::ONE)],
        [],
    );
}

fn push_trapped_state_transition_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    state: &StateColumns,
    output_captured: Column,
) {
    b.push_linear_zero([
        (idx(state.trapped_after), F::ONE),
        (idx(state.trapped_before), -F::ONE),
        (selector_col(WasmOpcode::Unreachable).unwrap(), -F::ONE),
        (COL_DIV_TRAP, -F::ONE),
        (COL_CI_TRAP, -F::ONE),
    ]);
    b.push_row(
        [(idx(control.is_program_row), F::ONE)],
        [(idx(state.trapped_before), F::ONE)],
        [],
    );
    b.push_row(
        [(idx(output_captured), F::ONE)],
        [(idx(state.trapped_after), F::ONE)],
        [],
    );
}
