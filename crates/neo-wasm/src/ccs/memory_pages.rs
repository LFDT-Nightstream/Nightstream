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
use super::super::layout::{
    selector_col, COL_CMP_GE, COL_CMP_LOW, COL_GROW_SUCCESS, COL_MAX_MEMORY_PAGES_AFTER, COL_MAX_MEMORY_PAGES_BEFORE,
    COL_MEMORY_PAGES_AFTER, COL_MEMORY_PAGES_BEFORE, COL_ONE, COL_STACK_READ0_VALUE_LO, COL_STACK_WRITE0_VALUE_LO,
};
use super::{always, shared, R1csBuilder};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const MEMORY_PAGE_OPS: &[WasmOpcode] = &[WasmOpcode::MemorySize, WasmOpcode::MemoryGrow];

pub(super) fn push_memory_pages_constraints(b: &mut R1csBuilder) {
    b.with_tag(shared("memory page constraints", MEMORY_PAGE_OPS), |b| {
        push_size_and_grow_constraints(b);
    });
    b.with_tag(always("memory pages carry"), |b| {
        // Only memory.grow may change the threaded page count.
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::MemoryGrow).unwrap(), -F::ONE),
            ],
            [(COL_MEMORY_PAGES_AFTER, F::ONE), (COL_MEMORY_PAGES_BEFORE, -F::ONE)],
            [],
        );
        // Max pages are verifier-carried and constant.
        b.push_linear_zero([
            (COL_MAX_MEMORY_PAGES_AFTER, F::ONE),
            (COL_MAX_MEMORY_PAGES_BEFORE, -F::ONE),
        ]);
    });
}

fn push_size_and_grow_constraints(b: &mut R1csBuilder) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::MemorySize).unwrap(),
        [(COL_MEMORY_PAGES_BEFORE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
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
            (COL_MAX_MEMORY_PAGES_BEFORE, F::ONE),
            (COL_MEMORY_PAGES_BEFORE, -F::ONE),
        ],
        [(COL_STACK_READ0_VALUE_LO, F::ONE)],
        COL_CMP_LOW,
        COL_CMP_GE,
    );
    b.push_row([(grow, F::ONE)], [(COL_CMP_GE, F::ONE)], [(COL_GROW_SUCCESS, F::ONE)]);
    // after = before + success * delta.
    b.push_row(
        [(COL_GROW_SUCCESS, F::ONE)],
        [(COL_STACK_READ0_VALUE_LO, F::ONE)],
        [(COL_MEMORY_PAGES_AFTER, F::ONE), (COL_MEMORY_PAGES_BEFORE, -F::ONE)],
    );
    // result = success ? before : 0xFFFFFFFF.
    push_gated_linear_zero(
        b,
        COL_GROW_SUCCESS,
        [(COL_STACK_WRITE0_VALUE_LO, F::ONE), (COL_MEMORY_PAGES_BEFORE, -F::ONE)],
    );
    b.push_row(
        [(grow, F::ONE), (COL_GROW_SUCCESS, -F::ONE)],
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_ONE, -F::from_u64(u32::MAX as u64)),
        ],
        [],
    );
}
