//! Owns the per-step "operand stack ↔ memory family" bindings.
//!
//! Every constraint in this module enforces the same shape: when a
//! wasm opcode reads from or writes to a host-state family (locals,
//! globals, tables, table sizes, or linear-memory page counts), the
//! relevant family-value column must equal the corresponding stack
//! column on that row. The actual lookup that proves the family value
//! comes from the right memory is the lookup-binding layer's job;
//! these constraints are just the in-CCS link to the operand stack
//! plus the gate columns the lookup layer reads.
//!
//! Linear-memory data lives elsewhere (see [`super::linear_memory`])
//! because it needs byte-level shuffling for unaligned access; the
//! constraints here are uniformly word-level identifications, so they
//! cluster naturally together.

use super::super::gadgets::push_gated_linear_zero;
use super::super::isa::WasmOpcode;
use super::super::layout::{
    selector_col, COL_CI_OOB, COL_GLOBAL_VALUE, COL_GLOBAL_VALUE_HI, COL_LOCAL_VALUE, COL_LOCAL_VALUE_HI,
    COL_LOCAL_WRITE_ENABLED, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO, COL_TABLE_INDEX, COL_TABLE_READ_ENABLED, COL_TABLE_SIZE, COL_TABLE_SIZE_READ_ENABLED,
    COL_TABLE_VALUE,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::{opcode_tag, shared};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

const LOCAL_WRITE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const TABLE_READ_OPS: &[WasmOpcode] = &[
    WasmOpcode::TableGet,
    WasmOpcode::CallIndirect,
    WasmOpcode::ReturnCallIndirect,
];
const TABLE_SIZE_READ_OPS: &[WasmOpcode] = &[
    WasmOpcode::TableSize,
    WasmOpcode::CallIndirect,
    WasmOpcode::ReturnCallIndirect,
];
const LOCAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalGet, WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const GLOBAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::GlobalGet, WasmOpcode::GlobalSet];
const TABLE_VALUE_OPS: &[WasmOpcode] = &[
    WasmOpcode::TableGet,
    WasmOpcode::TableSet,
    WasmOpcode::CallIndirect,
    WasmOpcode::ReturnCallIndirect,
];

/// Emit every operand-stack ↔ memory-family binding the wasm VM
/// needs. First the gate-column declarations the lookup layer reads
/// (`locals.write_enabled`, `table.read_enabled`), then the per-family
/// value bindings.
pub(super) fn push_stack_io_constraints(b: &mut R1csBuilder) {
    b.with_tag(shared("locals write gate", LOCAL_WRITE_OPS), |b| {
        b.push_linear_zero([
            (COL_LOCAL_WRITE_ENABLED, F::ONE),
            (selector_col(WasmOpcode::LocalSet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::LocalTee).unwrap(), -F::ONE),
        ]);
    });

    b.with_tag(shared("table read gate", TABLE_READ_OPS), |b| {
        // The table-entry read is off on an OOB call_indirect row: there is no
        // valid entry to read, and the OOB trap must not depend on it.
        b.push_linear_zero([
            (COL_TABLE_READ_ENABLED, F::ONE),
            (selector_col(WasmOpcode::TableGet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), -F::ONE),
            (COL_CI_OOB, F::ONE),
        ]);
    });

    b.with_tag(shared("table size read gate", TABLE_SIZE_READ_OPS), |b| {
        b.push_linear_zero([
            (COL_TABLE_SIZE_READ_ENABLED, F::ONE),
            (selector_col(WasmOpcode::TableSize).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), -F::ONE),
        ]);
    });

    b.with_tag(shared("locals value constraints", LOCAL_VALUE_OPS), |b| {
        push_local_value_constraints(b);
    });
    b.with_tag(shared("globals value constraints", GLOBAL_VALUE_OPS), |b| {
        push_global_value_constraints(b);
    });
    b.with_tag(shared("table value constraints", TABLE_VALUE_OPS), |b| {
        push_table_value_constraints(b);
    });
    b.with_tag(opcode_tag("table size constraints", WasmOpcode::TableSize), |b| {
        push_table_size_constraints(b);
    });
}

fn push_local_value_constraints(b: &mut R1csBuilder) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalGet).unwrap(),
        [(COL_LOCAL_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        [(COL_LOCAL_VALUE, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(COL_LOCAL_VALUE, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(COL_LOCAL_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
    );
    // Mirror the low-limb local bindings for i64 high limbs.
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalGet).unwrap(),
        [(COL_LOCAL_VALUE_HI, F::ONE), (COL_STACK_WRITE0_VALUE_HI, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        [(COL_LOCAL_VALUE_HI, F::ONE), (COL_STACK_READ_VALUE_HI[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(COL_LOCAL_VALUE_HI, F::ONE), (COL_STACK_READ_VALUE_HI[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(COL_LOCAL_VALUE_HI, F::ONE), (COL_STACK_WRITE0_VALUE_HI, -F::ONE)],
    );
}

fn push_global_value_constraints(b: &mut R1csBuilder) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        [(COL_GLOBAL_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        [(COL_GLOBAL_VALUE, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        [(COL_GLOBAL_VALUE_HI, F::ONE), (COL_STACK_WRITE0_VALUE_HI, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        [(COL_GLOBAL_VALUE_HI, F::ONE), (COL_STACK_READ_VALUE_HI[0], -F::ONE)],
    );
}

fn push_table_value_constraints(b: &mut R1csBuilder) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(COL_TABLE_VALUE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(COL_TABLE_VALUE, F::ONE), (COL_STACK_READ_VALUE_LO[1], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(COL_TABLE_INDEX, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(COL_TABLE_INDEX, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
}

fn push_table_size_constraints(b: &mut R1csBuilder) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSize).unwrap(),
        [(COL_TABLE_SIZE, F::ONE), (COL_STACK_WRITE0_VALUE_LO, -F::ONE)],
    );
}
