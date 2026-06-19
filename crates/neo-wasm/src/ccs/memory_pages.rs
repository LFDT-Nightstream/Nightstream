//! Owns the linear-memory page-count state machine.
//!
//! `memory.size` reads the current page count onto the stack; `memory.grow`
//! updates it, bounded by the module's declared maximum; and the count is
//! carried across rows (mutable — only `memory.grow` changes it) alongside the
//! maximum (a verifier-authoritative constant). The `max` is what makes
//! `memory.grow` and the linear-memory OOB bound check sound: it cannot be
//! forged, so the growth bound and the threaded current size are anchored.
//!
//! This is deliberately separate from [`super::stack_io`]: those are uniform
//! word-level stack↔family identifications, whereas grow is a multi-constraint
//! state transition (a `>=` comparison plus a success/failure branch).

use super::super::gadgets::{push_gated_linear_zero, push_unsigned_ge_gadget};
use super::super::isa::WasmOpcode;
use super::super::layout::{selector_col, COL_CMP_GE, COL_CMP_LOW, COL_GROW_SUCCESS, COL_ONE};
use super::super::lookup_binding_builder::{MemoryPagesColumns, OperandStackColumns, WasmLookupBindingLayout};
use super::{always, idx, shared, R1csBuilder};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const MEMORY_PAGE_OPS: &[WasmOpcode] = &[WasmOpcode::MemorySize, WasmOpcode::MemoryGrow];

pub(super) fn push_memory_pages_constraints(b: &mut R1csBuilder, layout: &WasmLookupBindingLayout) {
    let stack = layout.stack;
    let memory_pages = layout.memory_pages;

    b.with_tag(shared("memory page constraints", MEMORY_PAGE_OPS), |b| {
        push_size_and_grow_constraints(b, &stack, &memory_pages);
    });
    b.with_tag(always("memory pages carry"), |b| {
        // Only memory.grow may change the threaded page count.
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::MemoryGrow).unwrap(), -F::ONE),
            ],
            [(idx(memory_pages.after), F::ONE), (idx(memory_pages.before), -F::ONE)],
            [],
        );
        // Max pages are verifier-carried and constant.
        b.push_linear_zero([
            (idx(memory_pages.max_after), F::ONE),
            (idx(memory_pages.max_before), -F::ONE),
        ]);
    });
}

fn push_size_and_grow_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, memory_pages: &MemoryPagesColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::MemorySize).unwrap(),
        [
            (idx(memory_pages.before), F::ONE),
            (idx(stack.write0_value_lo), -F::ONE),
        ],
    );
    // memory.grow takes a page delta. The initial state has `before <= max`
    // from the validated memory type, and this transition preserves it.
    // Success iff `delta <= max - before`; result is old size or -1.
    let grow = selector_col(WasmOpcode::MemoryGrow).unwrap();
    // success = grow · ( (max - before) >= delta )
    push_unsigned_ge_gadget(
        b,
        [grow],
        [
            (idx(memory_pages.max_before), F::ONE),
            (idx(memory_pages.before), -F::ONE),
        ],
        [(idx(stack.read0_value_lo), F::ONE)],
        COL_CMP_LOW,
        COL_CMP_GE,
    );
    b.push_row([(grow, F::ONE)], [(COL_CMP_GE, F::ONE)], [(COL_GROW_SUCCESS, F::ONE)]);
    // after = before + success * delta.
    b.push_row(
        [(COL_GROW_SUCCESS, F::ONE)],
        [(idx(stack.read0_value_lo), F::ONE)],
        [(idx(memory_pages.after), F::ONE), (idx(memory_pages.before), -F::ONE)],
    );
    // result = success ? before : 0xFFFFFFFF.
    push_gated_linear_zero(
        b,
        COL_GROW_SUCCESS,
        [
            (idx(stack.write0_value_lo), F::ONE),
            (idx(memory_pages.before), -F::ONE),
        ],
    );
    b.push_row(
        [(grow, F::ONE), (COL_GROW_SUCCESS, -F::ONE)],
        [
            (idx(stack.write0_value_lo), F::ONE),
            (COL_ONE, -F::from_u64(u32::MAX as u64)),
        ],
        [],
    );
}
