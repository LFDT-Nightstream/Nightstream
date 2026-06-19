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
use super::super::layout::{selector_col, COL_CI_OOB};
use super::super::lookup_binding_builder::{
    GlobalsColumns, LocalsColumns, OperandStackColumns, TableColumns, TableSizeColumns, WasmLookupBindingLayout,
};
use super::super::tagged_r1cs_builder::WasmTaggedR1csBuilder;
use super::{idx, opcode_tag, shared};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

const LOCAL_WRITE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const TABLE_READ_OPS: &[WasmOpcode] = &[WasmOpcode::TableGet, WasmOpcode::CallIndirect];
const TABLE_SIZE_READ_OPS: &[WasmOpcode] = &[WasmOpcode::TableSize, WasmOpcode::CallIndirect];
const LOCAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalGet, WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const GLOBAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::GlobalGet, WasmOpcode::GlobalSet];
const TABLE_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::TableGet, WasmOpcode::TableSet, WasmOpcode::CallIndirect];

/// Emit every operand-stack ↔ memory-family binding the wasm VM
/// needs. First the gate-column declarations the lookup layer reads
/// (`locals.write_enabled`, `table.read_enabled`), then the per-family
/// value bindings.
pub(super) fn push_stack_io_constraints(b: &mut R1csBuilder, layout: &WasmLookupBindingLayout) {
    let stack = layout.stack;
    let locals = layout.locals;
    let globals = layout.globals;
    let table = layout.table;
    let table_sizes = layout.table_sizes;

    b.with_tag(shared("locals write gate", LOCAL_WRITE_OPS), |b| {
        b.push_linear_zero([
            (idx(locals.write_enabled), F::ONE),
            (selector_col(WasmOpcode::LocalSet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::LocalTee).unwrap(), -F::ONE),
        ]);
    });

    b.with_tag(shared("table read gate", TABLE_READ_OPS), |b| {
        // The table-entry read is off on an OOB call_indirect row: there is no
        // valid entry to read, and the OOB trap must not depend on it.
        b.push_linear_zero([
            (idx(table.read_enabled), F::ONE),
            (selector_col(WasmOpcode::TableGet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
            (COL_CI_OOB, F::ONE),
        ]);
    });

    b.with_tag(shared("table size read gate", TABLE_SIZE_READ_OPS), |b| {
        b.push_linear_zero([
            (idx(table_sizes.read_enabled), F::ONE),
            (selector_col(WasmOpcode::TableSize).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
        ]);
    });

    b.with_tag(shared("locals value constraints", LOCAL_VALUE_OPS), |b| {
        push_local_value_constraints(b, &stack, &locals);
    });
    b.with_tag(shared("globals value constraints", GLOBAL_VALUE_OPS), |b| {
        push_global_value_constraints(b, &stack, &globals);
    });
    b.with_tag(shared("table value constraints", TABLE_VALUE_OPS), |b| {
        push_table_value_constraints(b, &stack, &table);
    });
    b.with_tag(opcode_tag("table size constraints", WasmOpcode::TableSize), |b| {
        push_table_size_constraints(b, &stack, &table_sizes);
    });
}

fn push_local_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, locals: &LocalsColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalGet).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
    );
    // Mirror the low-limb local bindings for i64 high limbs.
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalGet).unwrap(),
        [(idx(locals.value_hi), F::ONE), (idx(stack.write0_value_hi), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        [(idx(locals.value_hi), F::ONE), (idx(stack.read0_value_hi), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_hi), F::ONE), (idx(stack.read0_value_hi), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_hi), F::ONE), (idx(stack.write0_value_hi), -F::ONE)],
    );
}

fn push_global_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, globals: &GlobalsColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        [(idx(globals.value), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        [(idx(globals.value), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        [(idx(globals.value_hi), F::ONE), (idx(stack.write0_value_hi), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        [(idx(globals.value_hi), F::ONE), (idx(stack.read0_value_hi), -F::ONE)],
    );
}

fn push_table_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, table: &TableColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(idx(table.value), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(idx(table.value), F::ONE), (idx(stack.read1_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(idx(table.index), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(idx(table.index), F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
}

fn push_table_size_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, table_sizes: &TableSizeColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSize).unwrap(),
        [(idx(table_sizes.value), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
    );
}
