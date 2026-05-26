//! Owns the WASM VM contract and phase-1 core CCS.
//!
//! Constraint families with substantial volume live in child modules
//! under `ccs/`; see [`linear_memory`] for the linear-memory load/store
//! row family. This file owns the top-level builder, the constraint
//! tag helpers (`always`, `shared`, `opcode_tag`), and the small shared
//! utilities (`idx`, `f_u64`, …) that those submodules consume via
//! `use super::*`.

mod call;
mod linear_memory;
mod stack_io;

use super::gadgets::{
    add_conditional_select_gadget, push_gated_linear_zero, push_zero_test_gadget, ConditionalSelectCols,
};
use super::isa::{opcode_code, opcode_info_from_code, WasmOpcode, WasmShoutOpcode};
use super::layout::{
    selector_col, COL_ONE, COL_PC_EDGE_KIND, COL_SELECT_OUT_DELTA, COL_WIDE_AUX0, COL_WIDE_AUX1, SELECTOR_COLS,
    WITNESS_WIDTH,
};
use super::lookup_binding_builder::{
    build_wasm_lookup_binding_layout, Column, ControlColumns, OperandStackColumns, ShoutColumns, StateColumns,
};
use super::tagged_r1cs_builder::{
    WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag, WasmTaggedR1csBuilder,
};
use crate::layout::{
    COL_CMP_AND, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO, COL_CMP_LO_DIFF, COL_CMP_LO_INV,
    COL_CMP_LO_IS_ZERO, COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV,
};
use neo_ccs::CcsStructure;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

type R1csBuilder = WasmTaggedR1csBuilder;

/// Static CCS shape of the wasm VM: the fixed-point constraint structure
/// each step's witness must satisfy, plus the witness-vector layout pins
/// (public-input length, total width, `1` column).
#[derive(Clone, Debug)]
pub struct WasmCoreCcs {
    pub structure: CcsStructure<F>,
    pub m_in: usize,
    pub witness_width: usize,
    pub const_one_col: usize,
}

#[derive(Clone, Debug)]
pub struct WasmVmSpec {
    core: WasmCoreCcs,
    constraint_catalog: WasmConstraintCatalog,
}

impl Default for WasmVmSpec {
    fn default() -> Self {
        Self::new().expect("valid WASM core CCS")
    }
}

impl WasmVmSpec {
    pub fn new() -> Result<Self, String> {
        let (core, constraint_catalog) = build_core_ccs_spec()?;
        Ok(Self {
            core,
            constraint_catalog,
        })
    }

    pub fn constraint_catalog(&self) -> &WasmConstraintCatalog {
        &self.constraint_catalog
    }

    pub fn core_ccs_spec(&self) -> &WasmCoreCcs {
        &self.core
    }

    pub fn name(&self) -> &'static str {
        "wasm"
    }
}

const I64_OPS: &[WasmOpcode] = &[
    WasmOpcode::I64Const,
    WasmOpcode::I64Add,
    WasmOpcode::I64Sub,
    WasmOpcode::I64Load,
    WasmOpcode::I64Store,
    WasmOpcode::I64Eqz,
    WasmOpcode::I64And,
    WasmOpcode::I64Or,
    WasmOpcode::I64Xor,
    WasmOpcode::I64Mul,
];

pub(super) const LINEAR_MEMORY_OPS: &[WasmOpcode] = &[
    WasmOpcode::I32Load,
    WasmOpcode::I32Load8S,
    WasmOpcode::I32Load8U,
    WasmOpcode::I32Load16S,
    WasmOpcode::I32Load16U,
    WasmOpcode::I64Load,
    WasmOpcode::I32Store,
    WasmOpcode::I32Store8,
    WasmOpcode::I32Store16,
    WasmOpcode::I64Store,
];

fn always(label: &'static str) -> WasmConstraintTag {
    WasmConstraintTag {
        label,
        scope: WasmConstraintScope::Always,
    }
}

pub(super) fn opcode_tag(label: &'static str, opcode: WasmOpcode) -> WasmConstraintTag {
    WasmConstraintTag {
        label,
        scope: WasmConstraintScope::Opcode(opcode),
    }
}

pub(super) fn shared(label: &'static str, opcodes: &[WasmOpcode]) -> WasmConstraintTag {
    WasmConstraintTag {
        label,
        scope: WasmConstraintScope::Opcodes(opcodes.to_vec().into_boxed_slice()),
    }
}

fn opcodes_with_stack_reads(reads: u8) -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|&op| opcode_info_from_code(opcode_code(op)).stack_reads == reads)
        .collect()
}

fn opcodes_with_stack_signature(reads: u8, writes: u8) -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|&op| {
            let info = opcode_info_from_code(opcode_code(op));
            info.stack_reads == reads && info.stack_writes == writes
        })
        .collect()
}

fn fixed_stack_reads_terms(control: &ControlColumns) -> Vec<(usize, F)> {
    let mut terms = vec![(idx(control.stack_reads), F::ONE)];
    for op in WasmOpcode::supported()
        .into_iter()
        .filter(|op| !matches!(op, WasmOpcode::Call | WasmOpcode::CallIndirect))
    {
        let reads = opcode_info_from_code(opcode_code(op)).stack_reads;
        if reads != 0 {
            terms.push((
                selector_col(op).expect("supported opcode selector"),
                -F::from_u64(u64::from(reads)),
            ));
        }
    }
    terms
}

fn fixed_stack_writes_terms(control: &ControlColumns) -> Vec<(usize, F)> {
    let mut terms = vec![(idx(control.stack_writes), F::ONE)];
    for op in WasmOpcode::supported()
        .into_iter()
        .filter(|op| !matches!(op, WasmOpcode::Call | WasmOpcode::CallIndirect))
    {
        let writes = opcode_info_from_code(opcode_code(op)).stack_writes;
        if writes != 0 {
            terms.push((
                selector_col(op).expect("supported opcode selector"),
                -F::from_u64(u64::from(writes)),
            ));
        }
    }
    terms
}

fn fixed_stack_arity_gate_terms(control: &ControlColumns) -> [(usize, F); 3] {
    [
        (idx(control.is_program_row), F::ONE),
        (selector_col(WasmOpcode::Call).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
    ]
}

fn build_core_ccs_spec() -> Result<(WasmCoreCcs, WasmConstraintCatalog), String> {
    let layout = build_wasm_lookup_binding_layout();
    let control = layout.control;
    let state = layout.state;
    let call = layout.call;
    let stack = layout.stack;
    let locals = layout.locals;
    let globals = layout.globals;
    let linear_memory = layout.linear_memory;
    let shout = layout.shout;
    let mut b = WasmTaggedR1csBuilder::new(WITNESS_WIDTH, COL_ONE)?;

    b.with_tag(shared("wide value gating", I64_OPS), |b| {
        b.push_row(
            [(idx(control.is_program_row), F::ONE)],
            [
                (idx(control.wide_values_enabled), F::ONE),
                (selector_col(WasmOpcode::I64Const).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Add).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Sub).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Load).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Store).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Eqz).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64And).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Or).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Xor).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::I64Mul).unwrap(), -F::ONE),
            ],
            [],
        );
    });

    b.with_tag(shared("linear memory lane0 gate", LINEAR_MEMORY_OPS), |b| {
        // lane0 usage only depends on the opcode
        //
        // but the other lanes depends on the type of access/alignment
        b.push_linear_zero([
            (idx(linear_memory.use_lane0), F::ONE),
            (selector_col(WasmOpcode::I32Load).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Load8S).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Load8U).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Load16S).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Load16U).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I64Load).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Store).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Store8).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I32Store16).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::I64Store).unwrap(), -F::ONE),
        ]);
    });

    stack_io::push_stack_io_constraints(&mut b, layout);

    b.with_tag(always("narrow high limbs zero"), |b| {
        for column in [
            stack.read0_value_hi,
            stack.read1_value_hi,
            stack.read2_value_hi,
            stack.write0_value_hi,
            locals.value_hi,
            globals.value_hi,
        ] {
            b.push_row(
                [(idx(column), F::ONE)],
                [(COL_ONE, F::ONE), (idx(control.wide_values_enabled), -F::ONE)],
                [],
            );
        }
    });

    call::push_call_constraints(&mut b, layout);

    b.with_tag(always("opcode selector one hot"), |b| {
        b.push_linear_zero(
            SELECTOR_COLS
                .into_iter()
                .map(|col| (col, F::ONE))
                .chain([(idx(control.is_program_row), -F::ONE)]),
        );
    });

    b.with_tag(always("opcode decode"), |b| {
        b.push_linear_zero(
            [
                (idx(control.opcode_code), F::ONE),
                (
                    selector_col(WasmOpcode::Nop).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Nop)),
                ),
                (
                    selector_col(WasmOpcode::I32Const).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Const)),
                ),
                (
                    selector_col(WasmOpcode::I64Const).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Const)),
                ),
                (
                    selector_col(WasmOpcode::RefFunc).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::RefFunc)),
                ),
                (
                    selector_col(WasmOpcode::I32Add).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Add)),
                ),
                (
                    selector_col(WasmOpcode::I64Add).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Add)),
                ),
                (
                    selector_col(WasmOpcode::I32Sub).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Sub)),
                ),
                (
                    selector_col(WasmOpcode::I64Sub).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Sub)),
                ),
                (
                    selector_col(WasmOpcode::I32Load).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Load)),
                ),
                (
                    selector_col(WasmOpcode::I32Load8S).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Load8S)),
                ),
                (
                    selector_col(WasmOpcode::I32Load8U).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Load8U)),
                ),
                (
                    selector_col(WasmOpcode::I32Load16S).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Load16S)),
                ),
                (
                    selector_col(WasmOpcode::I32Load16U).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Load16U)),
                ),
                (
                    selector_col(WasmOpcode::I64Load).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Load)),
                ),
                (
                    selector_col(WasmOpcode::I32Store).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Store)),
                ),
                (
                    selector_col(WasmOpcode::I32Store8).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Store8)),
                ),
                (
                    selector_col(WasmOpcode::I32Store16).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Store16)),
                ),
                (
                    selector_col(WasmOpcode::I64Store).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Store)),
                ),
                (
                    selector_col(WasmOpcode::MemorySize).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::MemorySize)),
                ),
                (
                    selector_col(WasmOpcode::MemoryGrow).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::MemoryGrow)),
                ),
                (
                    selector_col(WasmOpcode::TableSize).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::TableSize)),
                ),
                (
                    selector_col(WasmOpcode::TableGet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::TableGet)),
                ),
                (
                    selector_col(WasmOpcode::TableSet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::TableSet)),
                ),
                (
                    selector_col(WasmOpcode::Drop).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Drop)),
                ),
                (
                    selector_col(WasmOpcode::Br).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Br)),
                ),
                (
                    selector_col(WasmOpcode::Block).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Block)),
                ),
                (
                    selector_col(WasmOpcode::Loop).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Loop)),
                ),
                (
                    selector_col(WasmOpcode::If).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::If)),
                ),
                (
                    selector_col(WasmOpcode::Else).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Else)),
                ),
                (
                    selector_col(WasmOpcode::End).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::End)),
                ),
                (
                    selector_col(WasmOpcode::Unreachable).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Unreachable)),
                ),
                (
                    selector_col(WasmOpcode::I32Clz).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Clz)),
                ),
                (
                    selector_col(WasmOpcode::I32Ctz).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Ctz)),
                ),
                (
                    selector_col(WasmOpcode::I32Popcnt).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Popcnt)),
                ),
                (
                    selector_col(WasmOpcode::I32Eqz).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Eqz)),
                ),
                (
                    selector_col(WasmOpcode::I64Eqz).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Eqz)),
                ),
                (
                    selector_col(WasmOpcode::I32Eq).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Eq)),
                ),
                (
                    selector_col(WasmOpcode::I32Ne).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Ne)),
                ),
                (
                    selector_col(WasmOpcode::I32LtS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32LtS)),
                ),
                (
                    selector_col(WasmOpcode::I32LtU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32LtU)),
                ),
                (
                    selector_col(WasmOpcode::I32GtS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32GtS)),
                ),
                (
                    selector_col(WasmOpcode::I32GtU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32GtU)),
                ),
                (
                    selector_col(WasmOpcode::I32LeS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32LeS)),
                ),
                (
                    selector_col(WasmOpcode::I32LeU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32LeU)),
                ),
                (
                    selector_col(WasmOpcode::I32GeS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32GeS)),
                ),
                (
                    selector_col(WasmOpcode::I32GeU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32GeU)),
                ),
                (
                    selector_col(WasmOpcode::I32And).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32And)),
                ),
                (
                    selector_col(WasmOpcode::I32Or).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Or)),
                ),
                (
                    selector_col(WasmOpcode::I32Xor).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Xor)),
                ),
                (
                    selector_col(WasmOpcode::I32Mul).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Mul)),
                ),
                (
                    selector_col(WasmOpcode::I64And).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64And)),
                ),
                (
                    selector_col(WasmOpcode::I64Or).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Or)),
                ),
                (
                    selector_col(WasmOpcode::I64Xor).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Xor)),
                ),
                (
                    selector_col(WasmOpcode::I64Mul).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I64Mul)),
                ),
                (
                    selector_col(WasmOpcode::I32Shl).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Shl)),
                ),
                (
                    selector_col(WasmOpcode::I32ShrU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32ShrU)),
                ),
                (
                    selector_col(WasmOpcode::I32ShrS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32ShrS)),
                ),
                (
                    selector_col(WasmOpcode::I32Rotl).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Rotl)),
                ),
                (
                    selector_col(WasmOpcode::I32Rotr).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32Rotr)),
                ),
                (
                    selector_col(WasmOpcode::I32DivU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32DivU)),
                ),
                (
                    selector_col(WasmOpcode::I32DivS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32DivS)),
                ),
                (
                    selector_col(WasmOpcode::I32RemU).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32RemU)),
                ),
                (
                    selector_col(WasmOpcode::I32RemS).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::I32RemS)),
                ),
                (
                    selector_col(WasmOpcode::Select).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Select)),
                ),
                (
                    selector_col(WasmOpcode::BrIf).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::BrIf)),
                ),
                (
                    selector_col(WasmOpcode::BrTable).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::BrTable)),
                ),
                (
                    selector_col(WasmOpcode::Call).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Call)),
                ),
                (
                    selector_col(WasmOpcode::CallIndirect).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::CallIndirect)),
                ),
                (
                    selector_col(WasmOpcode::Return).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::Return)),
                ),
                (
                    selector_col(WasmOpcode::LocalGet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::LocalGet)),
                ),
                (
                    selector_col(WasmOpcode::LocalSet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::LocalSet)),
                ),
                (
                    selector_col(WasmOpcode::LocalTee).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::LocalTee)),
                ),
                (
                    selector_col(WasmOpcode::GlobalGet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::GlobalGet)),
                ),
                (
                    selector_col(WasmOpcode::GlobalSet).unwrap(),
                    -f_u16(opcode_code(WasmOpcode::GlobalSet)),
                ),
            ]
            .into_iter(),
        );
    });

    // sp after + stack reads = sp before + stack writes
    b.push_linear_zero([
        (idx(state.sp_after), F::ONE),
        (idx(state.sp_before), -F::ONE),
        (idx(control.stack_reads), F::ONE),
        (idx(control.stack_writes), -F::ONE),
    ]);
    b.with_tag(always("fixed stack arity"), |b| {
        b.push_row(
            fixed_stack_arity_gate_terms(&control),
            fixed_stack_reads_terms(&control),
            [],
        );
        b.push_row(
            fixed_stack_arity_gate_terms(&control),
            fixed_stack_writes_terms(&control),
            [],
        );
    });
    b.push_linear_zero([
        (idx(control.stack_read0_active), F::ONE),
        (idx(control.stack_read1_active), F::ONE),
        (idx(control.stack_read2_active), F::ONE),
        (idx(control.stack_reads), -F::ONE),
    ]);
    b.push_linear_zero([
        (idx(control.stack_write0_active), F::ONE),
        (idx(control.stack_writes), -F::ONE),
    ]);
    b.push_row(
        [(idx(control.stack_read1_active), F::ONE)],
        [(COL_ONE, F::ONE), (idx(control.stack_read0_active), -F::ONE)],
        [],
    );
    b.push_row(
        [(idx(control.stack_read2_active), F::ONE)],
        [(COL_ONE, F::ONE), (idx(control.stack_read1_active), -F::ONE)],
        [],
    );

    b.with_tag(always("pc edge kind constraints"), |b| {
        // Edge kind encodes the next-PC source:
        // 0 = static pc ROM, 1 = return-like, 2 = call_indirect target,
        // 3 = terminal unreachable. Return-like rows are identified by
        // `halted` for the final frame and by `call_stack_pop_present` for
        // non-final returns; the latter intentionally covers both explicit
        // `return` and a callee's function-ending `end`.
        b.push_linear_zero(
            [
                (idx(control.halted), F::ONE),
                (idx(call.call_stack_pop_present), F::ONE),
                (COL_PC_EDGE_KIND, -F::ONE),
                (selector_col(WasmOpcode::CallIndirect).unwrap(), F::from_u64(2)),
                (selector_col(WasmOpcode::Unreachable).unwrap(), F::from_u64(2)),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            selector_col(WasmOpcode::Return).unwrap(),
            [(COL_PC_EDGE_KIND, F::ONE), (COL_ONE, -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            selector_col(WasmOpcode::CallIndirect).unwrap(),
            [(COL_PC_EDGE_KIND, F::ONE), (COL_ONE, -F::from_u64(2))],
        );
        push_gated_linear_zero(
            b,
            selector_col(WasmOpcode::Unreachable).unwrap(),
            [(COL_PC_EDGE_KIND, F::ONE), (COL_ONE, -F::from_u64(3))],
        );
    });
    b.with_tag(always("pc rom active gate"), |b| {
        push_zero_test_gadget(
            b,
            idx(control.pc_edge_kind),
            idx(control.pc_edge_kind_inv),
            idx(control.pc_edge_kind_is_static),
        );
        b.push_row(
            [(idx(control.is_program_row), F::ONE)],
            [(idx(control.pc_edge_kind_is_static), F::ONE)],
            [(idx(control.pc_rom_active), F::ONE)],
        );
    });

    b.push_linear_zero(
        [
            (idx(shout.enabled), F::ONE),
            (selector_for_lookup(WasmOpcode::I32Clz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Ctz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32LtS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32LtU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32GtS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32GtU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32LeS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32LeU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32GeS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32GeU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32And), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Or), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Xor), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Mul), -F::ONE),
            (selector_for_lookup(WasmOpcode::I64And), -F::ONE),
            (selector_for_lookup(WasmOpcode::I64Or), -F::ONE),
            (selector_for_lookup(WasmOpcode::I64Xor), -F::ONE),
            (selector_for_lookup(WasmOpcode::I64Mul), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Shl), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32ShrU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32ShrS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Rotl), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Rotr), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32DivU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32DivS), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32RemU), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32RemS), -F::ONE),
        ]
        .into_iter(),
    );

    let stack_write0_at_sp_before_ops = opcodes_with_stack_signature(0, 1);
    let stack_read_at_sp_minus1_ops = opcodes_with_stack_reads(1);
    let stack_write0_at_sp_minus1_ops = opcodes_with_stack_signature(1, 1);
    let stack_read_at_sp_minus2_ops = opcodes_with_stack_reads(2);
    let stack_write0_at_sp_minus2_ops = opcodes_with_stack_signature(2, 1);

    b.with_tag(
        shared("stack write0 addr = sp_before", &stack_write0_at_sp_before_ops),
        |b| {
            push_stack_write0_addr_sp_before(b, &stack_write0_at_sp_before_ops, &state, &stack);
        },
    );
    b.with_tag(
        shared("stack read0 addr = sp_before - 1", &stack_read_at_sp_minus1_ops),
        |b| {
            push_stack_read0_addr_sp_minus_1(b, &stack_read_at_sp_minus1_ops, &state, &stack);
        },
    );
    b.with_tag(
        shared("stack write0 addr = sp_before - 1", &stack_write0_at_sp_minus1_ops),
        |b| {
            push_stack_write0_addr_sp_minus_1(b, &stack_write0_at_sp_minus1_ops, &state, &stack);
        },
    );

    b.with_tag(
        shared("stack read0 addr = sp_before - 2", &stack_read_at_sp_minus2_ops),
        |b| {
            push_stack_read0_addr_sp_minus_2(b, &stack_read_at_sp_minus2_ops, &state, &stack);
        },
    );
    b.with_tag(
        shared("stack read1 addr = sp_before - 1", &stack_read_at_sp_minus2_ops),
        |b| {
            push_stack_read1_addr_sp_minus_1(b, &stack_read_at_sp_minus2_ops, &state, &stack);
        },
    );
    b.with_tag(
        shared("stack write0 addr = sp_before - 2", &stack_write0_at_sp_minus2_ops),
        |b| {
            push_stack_write0_addr_sp_minus_2(b, &stack_write0_at_sp_minus2_ops, &state, &stack);
        },
    );

    let sel_select = selector_col(WasmOpcode::Select).unwrap();
    b.with_tag(opcode_tag("select stack addrs", WasmOpcode::Select), |b| {
        push_select_stack_addrs(b, &state, &stack);
    });

    b.with_tag(opcode_tag("i32.add relation", WasmOpcode::I32Add), |b| {
        push_add_relation(b, &stack)
    });
    b.with_tag(opcode_tag("i32.sub relation", WasmOpcode::I32Sub), |b| {
        push_sub_relation(b, &stack)
    });
    b.with_tag(opcode_tag("i64.add relation", WasmOpcode::I64Add), |b| {
        push_i64_add_relation(b, &stack)
    });
    b.with_tag(opcode_tag("i64.sub relation", WasmOpcode::I64Sub), |b| {
        push_i64_sub_relation(b, &stack)
    });
    b.with_tag(opcode_tag("i64.eqz high limb zero", WasmOpcode::I64Eqz), |b| {
        push_i64_eqz_high_zero(b, &stack);
    });
    b.with_tag(shared("comparator zero-test", COMPARATOR_OPS), |b| {
        push_comparator_constraints(b, &stack);
    });
    linear_memory::push_linear_memory_constraints(&mut b, &stack, &linear_memory);
    b.with_tag(opcode_tag("select conditional gadget", WasmOpcode::Select), |b| {
        add_conditional_select_gadget(
            b,
            ConditionalSelectCols {
                selector: sel_select,
                cond: idx(stack.read2_value),
                lhs: idx(stack.read0_value),
                rhs: idx(stack.read1_value),
                out: idx(stack.write0_value),
                scratch_out_delta: COL_SELECT_OUT_DELTA,
                scratch_inverse: COL_SELECT_SCRATCH_INV,
                cond_is_zero: COL_SELECT_COND_IS_ZERO,
            },
        )
    });

    b.with_tag(always("shout constraints"), |b| {
        push_shout_constraints(b, &stack, &shout);
    });
    let (structure, constraint_catalog) = b.build()?;

    Ok((
        WasmCoreCcs {
            structure,
            m_in: 1,
            witness_width: WITNESS_WIDTH,
            const_one_col: COL_ONE,
        },
        constraint_catalog,
    ))
}

fn f_u16(v: u16) -> F {
    F::from_u64(u64::from(v))
}

pub(super) fn f_u64(v: u64) -> F {
    F::from_u64(v)
}

fn selector_for_lookup(op: WasmOpcode) -> usize {
    selector_col(op).expect("lookup opcode selector column")
}

fn push_stack_write0_addr_sp_before(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp selector"), F::ONE)),
        [(idx(stack.write0_addr), F::ONE), (idx(state.sp_before), -F::ONE)],
        [],
    );
}

fn push_stack_read0_addr_sp_minus_1(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read0 sp-1 selector"), F::ONE)),
        [
            (idx(stack.read0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, F::ONE),
        ],
        [],
    );
}

fn push_stack_write0_addr_sp_minus_1(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp-1 selector"), F::ONE)),
        [
            (idx(stack.write0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, F::ONE),
        ],
        [],
    );
}

fn push_select_stack_addrs(b: &mut R1csBuilder, state: &StateColumns, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::Select).unwrap();
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.read0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, f_u64(3)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.read1_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, f_u64(2)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.read2_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.write0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, f_u64(3)),
        ],
    );
}

fn push_stack_read0_addr_sp_minus_2(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read0 sp-2 selector"), F::ONE)),
        [
            (idx(stack.read0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, f_u64(2)),
        ],
        [],
    );
}

fn push_stack_read1_addr_sp_minus_1(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read1 sp-1 selector"), F::ONE)),
        [
            (idx(stack.read1_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, F::ONE),
        ],
        [],
    );
}

fn push_stack_write0_addr_sp_minus_2(
    b: &mut R1csBuilder,
    ops: &[WasmOpcode],
    state: &StateColumns,
    stack: &OperandStackColumns,
) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp-2 selector"), F::ONE)),
        [
            (idx(stack.write0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (COL_ONE, f_u64(2)),
        ],
        [],
    );
}

fn push_add_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    // write0 = (read0 + read1) mod 2^32. COL_WIDE_AUX0 holds the carry bit.
    // Soundness relies on COL_WIDE_AUX0's `ColumnWidth::Boolean` tag: without
    // that, a cheating prover could pick any field element for the carry and
    // solve for a matching write0. Given the boolean range, the U32 tags on
    // read0, read1, write0 ensure the equation has a unique solution.
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Add).unwrap(),
        [
            (idx(stack.read0_value), F::ONE),
            (idx(stack.read1_value), F::ONE),
            (idx(stack.write0_value), -F::ONE),
            (COL_WIDE_AUX0, -f_u64(1_u64 << 32)),
        ],
    );
}

fn push_sub_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    // write0 = (read0 - read1) mod 2^32. COL_WIDE_AUX0 holds the borrow bit
    // (1 iff read0 < read1); same soundness argument as [`push_add_relation`]:
    // the Boolean width tag on COL_WIDE_AUX0 is what pins the borrow to {0, 1},
    // and the U32 tags on read0/read1/write0 make the solution unique.
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Sub).unwrap(),
        [
            (idx(stack.read0_value), F::ONE),
            (idx(stack.read1_value), -F::ONE),
            (idx(stack.write0_value), -F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
        ],
    );
}

fn push_i64_add_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::I64Add).unwrap();
    b.push_row(
        [
            (idx(stack.read0_value), F::ONE),
            (idx(stack.read1_value), F::ONE),
            (idx(stack.write0_value), -F::ONE),
            (COL_WIDE_AUX0, -f_u64(1_u64 << 32)),
        ],
        [(selector, F::ONE)],
        [],
    );
    b.push_row(
        [
            (idx(stack.read0_value_hi), F::ONE),
            (idx(stack.read1_value_hi), F::ONE),
            (idx(stack.write0_value_hi), -F::ONE),
            (COL_WIDE_AUX0, F::ONE),
            (COL_WIDE_AUX1, -f_u64(1_u64 << 32)),
        ],
        [(selector, F::ONE)],
        [],
    );
}

fn push_i64_sub_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::I64Sub).unwrap();
    b.push_row(
        [
            (idx(stack.read0_value), F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
            (idx(stack.write0_value), -F::ONE),
            (idx(stack.read1_value), -F::ONE),
        ],
        [(selector, F::ONE)],
        [],
    );
    b.push_row(
        [
            (idx(stack.read0_value_hi), F::ONE),
            (COL_WIDE_AUX1, f_u64(1_u64 << 32)),
            (idx(stack.write0_value_hi), -F::ONE),
            (idx(stack.read1_value_hi), -F::ONE),
            (COL_WIDE_AUX0, -F::ONE),
        ],
        [(selector, F::ONE)],
        [],
    );
}

fn push_i64_eqz_high_zero(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I64Eqz).unwrap(),
        [(idx(stack.write0_value_hi), F::ONE)],
    );
}

const COMPARATOR_OPS: &[WasmOpcode] = &[
    WasmOpcode::I32Eqz,
    WasmOpcode::I64Eqz,
    WasmOpcode::I32Eq,
    WasmOpcode::I32Ne,
];

/// CCS-native gates for i32.eqz / i64.eqz / i32.eq / i32.ne.
///
/// The opcode's selector pins `COL_CMP_LO_DIFF` to the right zero-test input
/// (`read0`, `read0_lo`, or `read0 - read1`); the shared zero-test gadget
/// forces `COL_CMP_LO_IS_ZERO = (cmp_lo_diff == 0)`.
///
/// **i64.eqz needs a split limb-by-limb zero-test.** The Goldilocks
/// modulus `q = 2^64 - 2^32 + 1` does not have an injective u64 →
/// field-element embedding for the obvious `lo + hi*2^32` map: the value
/// `(lo=1, hi=0xffffffff)` is exactly `q ≡ 0`, so a single field
/// zero-test would wrongly accept it. We pin `COL_CMP_LO_DIFF = read0_lo`
/// and `COL_CMP_HI_DIFF = read0_hi`, zero-test each limb, and AND the
/// flags into `COL_CMP_AND`. The i64.eqz write-back uses `cmp_and`; the
/// i32 write-backs use `cmp_lo_is_zero` directly.
///
/// On non-comparator rows all four selectors are 0, the diff-pinning
/// gates degenerate, and the witness sets `cmp_lo_diff = cmp_hi_diff = 0` →
/// both flags = 1, `cmp_and = 1`. None of those values are observed
/// elsewhere on non-comparator rows.
fn push_comparator_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let sel_eqz_i32 = selector_col(WasmOpcode::I32Eqz).unwrap();
    let sel_eqz_i64 = selector_col(WasmOpcode::I64Eqz).unwrap();
    let sel_eq = selector_col(WasmOpcode::I32Eq).unwrap();
    let sel_ne = selector_col(WasmOpcode::I32Ne).unwrap();

    // cmp_lo_diff = read0_value on i32.eqz rows.
    push_gated_linear_zero(
        b,
        sel_eqz_i32,
        [(COL_CMP_LO_DIFF, F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
    // cmp_lo_diff = read0_value (lo limb only) on i64.eqz rows.
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_CMP_LO_DIFF, F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
    // cmp_lo_diff = read0_value - read1_value on i32.eq / i32.ne rows.
    b.push_row(
        [(sel_eq, F::ONE), (sel_ne, F::ONE)],
        [
            (COL_CMP_LO_DIFF, F::ONE),
            (idx(stack.read0_value), -F::ONE),
            (idx(stack.read1_value), F::ONE),
        ],
        [],
    );

    push_zero_test_gadget(b, COL_CMP_LO_DIFF, COL_CMP_LO_INV, COL_CMP_LO_IS_ZERO);

    // cmp_hi_diff = read0_value_hi on i64.eqz rows; unconstrained otherwise
    // (witness sets it to 0 so the hi zero-test flag is 1).
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_CMP_HI_DIFF, F::ONE), (idx(stack.read0_value_hi), -F::ONE)],
    );

    push_zero_test_gadget(b, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO);

    // cmp_and = cmp_lo_is_zero * cmp_hi_is_zero (unconditional).
    b.push_row(
        [(COL_CMP_LO_IS_ZERO, F::ONE)],
        [(COL_CMP_HI_IS_ZERO, F::ONE)],
        [(COL_CMP_AND, F::ONE)],
    );

    // write0_value = cmp_lo_is_zero on i32.eqz / i32.eq rows.
    b.push_row(
        [(sel_eqz_i32, F::ONE), (sel_eq, F::ONE)],
        [(idx(stack.write0_value), F::ONE), (COL_CMP_LO_IS_ZERO, -F::ONE)],
        [],
    );
    // write0_value = cmp_and on i64.eqz rows.
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(idx(stack.write0_value), F::ONE), (COL_CMP_AND, -F::ONE)],
    );
    // write0_value = 1 - cmp_lo_is_zero on ne rows.
    push_gated_linear_zero(
        b,
        sel_ne,
        [
            (idx(stack.write0_value), F::ONE),
            (COL_CMP_LO_IS_ZERO, F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
}

fn push_shout_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, shout: &ShoutColumns) {
    b.push_row(
        WasmShoutOpcode::all()
            .into_iter()
            .map(|op| (selector_for_shout(op), F::ONE)),
        [(idx(shout.value), F::ONE), (idx(stack.write0_value), -F::ONE)],
        [],
    );
    b.push_row(
        WasmShoutOpcode::all()
            .into_iter()
            .map(|op| (selector_for_shout(op), F::from_u64(u64::from(op.to_shout_id())))),
        [(COL_ONE, F::ONE)],
        [(idx(shout.id), F::ONE)],
    );
}

pub(super) fn idx(column: Column) -> usize {
    column.0
}

fn selector_for_shout(op: WasmShoutOpcode) -> usize {
    selector_for_lookup(match op {
        WasmShoutOpcode::I32Clz => WasmOpcode::I32Clz,
        WasmShoutOpcode::I32Ctz => WasmOpcode::I32Ctz,
        WasmShoutOpcode::I32LtS => WasmOpcode::I32LtS,
        WasmShoutOpcode::I32LtU => WasmOpcode::I32LtU,
        WasmShoutOpcode::I32GtS => WasmOpcode::I32GtS,
        WasmShoutOpcode::I32GtU => WasmOpcode::I32GtU,
        WasmShoutOpcode::I32LeS => WasmOpcode::I32LeS,
        WasmShoutOpcode::I32LeU => WasmOpcode::I32LeU,
        WasmShoutOpcode::I32GeS => WasmOpcode::I32GeS,
        WasmShoutOpcode::I32GeU => WasmOpcode::I32GeU,
        WasmShoutOpcode::I32And => WasmOpcode::I32And,
        WasmShoutOpcode::I32Or => WasmOpcode::I32Or,
        WasmShoutOpcode::I32Xor => WasmOpcode::I32Xor,
        WasmShoutOpcode::I32Mul => WasmOpcode::I32Mul,
        WasmShoutOpcode::I64And => WasmOpcode::I64And,
        WasmShoutOpcode::I64Or => WasmOpcode::I64Or,
        WasmShoutOpcode::I64Xor => WasmOpcode::I64Xor,
        WasmShoutOpcode::I64Mul => WasmOpcode::I64Mul,
        WasmShoutOpcode::I32Shl => WasmOpcode::I32Shl,
        WasmShoutOpcode::I32ShrU => WasmOpcode::I32ShrU,
        WasmShoutOpcode::I32ShrS => WasmOpcode::I32ShrS,
        WasmShoutOpcode::I32Rotl => WasmOpcode::I32Rotl,
        WasmShoutOpcode::I32Rotr => WasmOpcode::I32Rotr,
        WasmShoutOpcode::I32DivU => WasmOpcode::I32DivU,
        WasmShoutOpcode::I32DivS => WasmOpcode::I32DivS,
        WasmShoutOpcode::I32RemU => WasmOpcode::I32RemU,
        WasmShoutOpcode::I32RemS => WasmOpcode::I32RemS,
    })
}
