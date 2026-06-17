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
mod trap;

use super::gadgets::{push_gated_linear_zero, push_u32_le_bytes_decomp, push_zero_test_gadget};
use super::isa::{opcode_code, opcode_info_from_code, WasmOpTable, WasmOpcode};
use super::layout::{
    selector_col, COL_ONE, COL_PC_EDGE_KIND, COL_SELECT_OUT_DELTA_HI, COL_SELECT_OUT_DELTA_LO, COL_WIDE_AUX0,
    COL_WIDE_AUX1, PUBLIC_INPUTS, SELECTOR_COLS, WITNESS_WIDTH,
};
use super::lookup_binding_builder::{
    build_wasm_lookup_binding_layout, Column, ControlColumns, OpTableColumns, OperandStackColumns,
    SignExtensionColumns, StateColumns,
};
use super::tagged_r1cs_builder::{
    WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag, WasmTaggedR1csBuilder,
};
use crate::layout::{
    COL_CALL_INDIRECT_IS_TRAP, COL_CMP_AND, COL_CMP_HI_DIFF, COL_CMP_HI_INV, COL_CMP_HI_IS_ZERO, COL_CMP_LO_DIFF,
    COL_CMP_LO_INV, COL_CMP_LO_IS_ZERO, COL_DIV_TRAP, COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV,
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

/// Opcodes whose rows participate in the wide-value gating constraint.
/// Spec-derived from [`WasmOpcode::uses_wide_values`] so this set cannot
/// drift from the witness builder or the test-row helpers — see the doc
/// on the method.
pub(super) fn wide_value_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.uses_wide_values())
        .collect()
}

/// Opcodes whose CCS gates fire on linear-memory rows. Spec-derived from
/// [`WasmOpcode::uses_linear_memory`] so the list cannot drift from the
/// opcode declaration.
pub(super) fn linear_memory_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.uses_linear_memory())
        .collect()
}

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
    let op_table = layout.op_table;
    let mut b = WasmTaggedR1csBuilder::new(WITNESS_WIDTH, COL_ONE)?;

    b.with_tag(shared("wide value gating", &wide_value_ops()), |b| {
        // is_program_row · (wide_values_enabled − Σ wide-value-op selectors) = 0,
        // so on a program row wide_values_enabled = 1 iff a wide-value op is
        // active. Spec-derived from `WasmOpcode::uses_wide_values` so the gate
        // cannot drift from the witness builder or the test-row helpers.
        b.push_row(
            [(idx(control.is_program_row), F::ONE)],
            std::iter::once((idx(control.wide_values_enabled), F::ONE)).chain(
                wide_value_ops()
                    .into_iter()
                    .map(|op| (selector_col(op).expect("wide value op selector"), -F::ONE)),
            ),
            [],
        );
    });

    b.with_tag(shared("linear memory lane0 gate", &linear_memory_ops()), |b| {
        // Every linear-memory op touches lane0 (the lowest word); the other
        // lanes depend on access width / alignment. Derived from
        // `linear_memory_ops` so the gate can't drift from the opcode set.
        b.push_linear_zero(
            std::iter::once((idx(linear_memory.use_lane0), F::ONE)).chain(
                linear_memory_ops()
                    .into_iter()
                    .map(|op| (selector_col(op).expect("memory op selector"), -F::ONE)),
            ),
        );
    });

    stack_io::push_stack_io_constraints(&mut b, layout);

    b.with_tag(always("narrow high limbs zero"), |b| {
        b.push_row(
            [(idx(stack.read0_value_hi), F::ONE)],
            [
                (COL_ONE, F::ONE),
                (idx(control.wide_values_enabled), -F::ONE),
                (idx(layout.output.captured), -F::ONE),
            ],
            [],
        );
        for column in [
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
        // opcode_code = Σ_op selector(op) · opcode_code(op). Selectors are
        // one-hot per program row, so this pins opcode_code to the active
        // opcode's byte. Derived from `WasmOpcode::supported()` so it can't
        // drift from the opcode set.
        b.push_linear_zero(std::iter::once((idx(control.opcode_code), F::ONE)).chain(
            WasmOpcode::supported().into_iter().map(|op| {
                (
                    selector_col(op).expect("supported opcode selector"),
                    -f_u16(opcode_code(op)),
                )
            }),
        ));
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
        // `return` and a callee's function-ending `end`. A div trap row
        // halts but keeps its Static edge kind, and a call_indirect trap
        // row halts but keeps its DynamicCallIndirect edge kind (the per-pc
        // edge-kind ROM binds both); the -trap terms absorb their `halted`
        // contributions.
        b.push_linear_zero(
            [
                (idx(control.halted), F::ONE),
                (idx(call.call_stack_pop_present), F::ONE),
                (COL_PC_EDGE_KIND, -F::ONE),
                (selector_col(WasmOpcode::CallIndirect).unwrap(), F::from_u64(2)),
                (selector_col(WasmOpcode::Unreachable).unwrap(), F::from_u64(2)),
                (COL_DIV_TRAP, -F::ONE),
                (COL_CALL_INDIRECT_IS_TRAP, -F::ONE),
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

    trap::push_trap_constraints(&mut b, layout);

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

    // A div trap de-gates the op-table relation; div/rem lookup tables only
    // model completed, non-trapping arithmetic.
    b.push_linear_zero(
        core::iter::once((idx(op_table.enabled), F::ONE))
            .chain(
                WasmOpTable::all()
                    .into_iter()
                    .map(|op| (selector_for_op_table(op), -F::ONE)),
            )
            .chain(core::iter::once((COL_DIV_TRAP, F::ONE))),
    );

    let stack_write0_at_sp_before_ops = opcodes_with_stack_signature(0, 1);
    let stack_read_at_sp_minus1_ops = opcodes_with_stack_reads(1);
    let stack_write0_at_sp_minus1_ops = opcodes_with_stack_signature(1, 1);
    let stack_read_at_sp_minus2_ops = opcodes_with_stack_reads(2);
    let stack_write0_at_sp_minus2_ops = opcodes_with_stack_signature(2, 1);

    b.with_tag(always("stack high limb addresses"), |b| {
        for (addr_hi, addr_lo) in [
            (stack.read0_addr_hi, stack.read0_addr_lo),
            (stack.read1_addr_hi, stack.read1_addr_lo),
            (stack.read2_addr_hi, stack.read2_addr_lo),
            (stack.write0_addr_hi, stack.write0_addr_lo),
        ] {
            b.push_linear_zero([(idx(addr_hi), F::ONE), (idx(addr_lo), -F::ONE), (COL_ONE, -F::ONE)]);
        }
    });

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
    b.with_tag(opcode_tag("i32.wrap_i64 relation", WasmOpcode::I32WrapI64), |b| {
        push_i32_wrap_i64_relation(b, &stack)
    });
    b.with_tag(
        opcode_tag("i64.extend_i32_u low relation", WasmOpcode::I64ExtendI32U),
        |b| push_i64_extend_i32_u_low_relation(b, &stack),
    );
    b.with_tag(
        opcode_tag("i64.extend_i32_u high zero", WasmOpcode::I64ExtendI32U),
        |b| push_i64_extend_i32_u_high_relation(b, &stack),
    );
    for (opcode, width_bytes, writes_i64) in [
        (WasmOpcode::I64ExtendI32S, 4, true),
        (WasmOpcode::I32Extend8S, 1, false),
        (WasmOpcode::I32Extend16S, 2, false),
        (WasmOpcode::I64Extend8S, 1, true),
        (WasmOpcode::I64Extend16S, 2, true),
        (WasmOpcode::I64Extend32S, 4, true),
    ] {
        b.with_tag(opcode_tag("integer sign-extension relation", opcode), |b| {
            push_integer_sign_extend_relation(b, &stack, &layout.sign_extension, opcode, width_bytes, writes_i64);
        });
    }
    b.with_tag(
        shared(
            "i64 comparator high limb zero",
            &[WasmOpcode::I64Eqz, WasmOpcode::I64Eq, WasmOpcode::I64Ne],
        ),
        |b| {
            push_i64_comparator_high_zero(b, &stack);
        },
    );
    b.with_tag(shared("comparator zero-test", COMPARATOR_OPS), |b| {
        push_comparator_constraints(b, &stack);
    });
    linear_memory::push_linear_memory_constraints(&mut b, &stack, &linear_memory, &layout.sign_extension);
    b.with_tag(opcode_tag("select conditional mux", WasmOpcode::Select), |b| {
        push_select_constraints(b, &stack);
    });

    b.with_tag(always("op_table constraints"), |b| {
        push_shout_constraints(b, &stack, &op_table);
    });
    let (structure, constraint_catalog) = b.build()?;

    Ok((
        WasmCoreCcs {
            structure,
            m_in: PUBLIC_INPUTS,
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
        [(idx(stack.write0_addr_lo), F::ONE), (idx(state.sp_before), -f_u64(2))],
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
            (idx(stack.read0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(2)),
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
            (idx(stack.write0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(2)),
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
            (idx(stack.read0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(6)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.read1_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(4)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.read2_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(2)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.write0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(6)),
        ],
    );
}

/// `selector · (out − (cond != 0 ? lhs : rhs)) = 0` for both value limbs,
/// where cond is the i32 at stack read2 and lhs/rhs are reads 0/1.
///
/// The zero-test and delta rows are intentionally global: the witness builder
/// populates `COL_SELECT_COND_IS_ZERO`, `COL_SELECT_SCRATCH_INV`, and both
/// delta columns on every row.
fn push_select_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::Select).unwrap();
    push_zero_test_gadget(
        b,
        idx(stack.read2_value_lo),
        COL_SELECT_SCRATCH_INV,
        COL_SELECT_COND_IS_ZERO,
    );
    push_select_mux_limb(
        b,
        selector,
        idx(stack.read0_value_lo),
        idx(stack.read1_value_lo),
        idx(stack.write0_value_lo),
        COL_SELECT_OUT_DELTA_LO,
    );
    push_select_mux_limb(
        b,
        selector,
        idx(stack.read0_value_hi),
        idx(stack.read1_value_hi),
        idx(stack.write0_value_hi),
        COL_SELECT_OUT_DELTA_HI,
    );
}

fn push_select_mux_limb(b: &mut R1csBuilder, selector: usize, lhs: usize, rhs: usize, out: usize, delta: usize) {
    // delta = (cond != 0) · (lhs − rhs)
    b.push_row(
        [(COL_ONE, F::ONE), (COL_SELECT_COND_IS_ZERO, -F::ONE)],
        [(lhs, F::ONE), (rhs, -F::ONE)],
        [(delta, F::ONE)],
    );
    // selector · ((out − rhs) − delta) = 0
    push_gated_linear_zero(b, selector, [(out, F::ONE), (rhs, -F::ONE), (delta, -F::ONE)]);
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
            (idx(stack.read0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(4)),
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
            (idx(stack.read1_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(2)),
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
            (idx(stack.write0_addr_lo), F::ONE),
            (idx(state.sp_before), -f_u64(2)),
            (COL_ONE, f_u64(4)),
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
            (idx(stack.read0_value_lo), F::ONE),
            (idx(stack.read1_value_lo), F::ONE),
            (idx(stack.write0_value_lo), -F::ONE),
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
            (idx(stack.read0_value_lo), F::ONE),
            (idx(stack.read1_value_lo), -F::ONE),
            (idx(stack.write0_value_lo), -F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
        ],
    );
}

fn push_i64_add_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::I64Add).unwrap();
    b.push_row(
        [
            (idx(stack.read0_value_lo), F::ONE),
            (idx(stack.read1_value_lo), F::ONE),
            (idx(stack.write0_value_lo), -F::ONE),
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
            (idx(stack.read0_value_lo), F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
            (idx(stack.write0_value_lo), -F::ONE),
            (idx(stack.read1_value_lo), -F::ONE),
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

fn push_i32_wrap_i64_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let selector = selector_col(WasmOpcode::I32WrapI64).unwrap();
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(stack.write0_value_lo), F::ONE),
            (idx(stack.read0_value_lo), -F::ONE),
        ],
    );
    push_gated_linear_zero(b, selector, [(idx(stack.write0_value_hi), F::ONE)]);
}

fn push_i64_extend_i32_u_low_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    b.push_row(
        [(selector_col(WasmOpcode::I64ExtendI32U).unwrap(), F::ONE)],
        [
            (idx(stack.write0_value_lo), F::ONE),
            (idx(stack.read0_value_lo), -F::ONE),
        ],
        [],
    );
}

fn push_i64_extend_i32_u_high_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I64ExtendI32U).unwrap(),
        [(idx(stack.write0_value_hi), F::ONE)],
    );
}

fn push_integer_sign_extend_relation(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    sign_extension: &SignExtensionColumns,
    opcode: WasmOpcode,
    width_bytes: usize,
    writes_i64: bool,
) {
    debug_assert!((1..=4).contains(&width_bytes));
    let selector = selector_col(opcode).unwrap();
    push_u32_le_bytes_decomp(b, [selector], idx(stack.read0_value_lo), sign_extension.bytes.map(idx));

    let sign_source = sign_extension.bytes[width_bytes - 1];
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(sign_source), F::ONE),
            (idx(sign_extension.low7), -F::ONE),
            (idx(sign_extension.bit), -f_u64(128)),
        ],
    );

    let retained_mask = if width_bytes == 4 {
        u32::MAX
    } else {
        (1u32 << (width_bytes * 8)) - 1
    };
    let sign_fill = u32::MAX ^ retained_mask;
    let retained_bytes = sign_extension.bytes[..width_bytes]
        .iter()
        .enumerate()
        .map(|(byte_index, &byte)| (idx(byte), -f_u64(1u64 << (byte_index * 8))));
    b.push_row(
        [(selector, F::ONE)],
        std::iter::once((idx(stack.write0_value_lo), F::ONE))
            .chain(retained_bytes)
            .chain(std::iter::once((idx(sign_extension.bit), -f_u64(u64::from(sign_fill))))),
        [],
    );

    if writes_i64 {
        b.push_row(
            [(selector, F::ONE)],
            [
                (idx(stack.write0_value_hi), F::ONE),
                (idx(sign_extension.bit), -f_u64(0xffff_ffff)),
            ],
            [],
        );
    }
}

/// Force `write0_value_hi = 0` on every comparator row that produces a
/// u32 result. i64.eqz / i64.eq / i64.ne have `wide_values_enabled = 1`
/// for their inputs, which disables the "narrow high limbs zero" rule for
/// `write0_value_hi`; this constraint pins the output hi limb back to 0.
fn push_i64_comparator_high_zero(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    b.push_row(
        [
            (selector_col(WasmOpcode::I64Eqz).unwrap(), F::ONE),
            (selector_col(WasmOpcode::I64Eq).unwrap(), F::ONE),
            (selector_col(WasmOpcode::I64Ne).unwrap(), F::ONE),
        ],
        [(idx(stack.write0_value_hi), F::ONE)],
        [],
    );
}

const COMPARATOR_OPS: &[WasmOpcode] = &[
    WasmOpcode::I32Eqz,
    WasmOpcode::I64Eqz,
    WasmOpcode::I32Eq,
    WasmOpcode::I32Ne,
    WasmOpcode::I64Eq,
    WasmOpcode::I64Ne,
];

/// CCS-native gates for i32.eqz / i64.eqz / i32.eq / i32.ne / i64.eq / i64.ne.
///
/// The opcode's selector pins `COL_CMP_LO_DIFF` to the zero-test input for
/// the lo limb (`read0`, `read0_lo`, `read0 - read1`, or `read0_lo -
/// read1_lo`); the zero-test gadget forces
/// `COL_CMP_LO_IS_ZERO = (cmp_lo_diff == 0)`.
///
/// **The i64 comparators need a split limb-by-limb zero-test.** The
/// Goldilocks modulus `q = 2^64 - 2^32 + 1` does not have an injective
/// u64 → field-element embedding for the obvious `lo + hi*2^32` map: the
/// value `(lo=1, hi=0xffffffff)` is exactly `q ≡ 0`, so a single field
/// zero-test would wrongly accept it. For i64.eqz / i64.eq / i64.ne we
/// pin `COL_CMP_HI_DIFF` to the hi-limb diff, zero-test it independently,
/// and AND the two flags into `COL_CMP_AND`. The i64 write-backs use
/// `cmp_and`; the i32 write-backs use `cmp_lo_is_zero` directly (their
/// inputs fit safely below q, so the hi limb is unused).
///
/// On non-comparator rows all six selectors are 0, the diff-pinning
/// gates degenerate, and the witness sets `cmp_lo_diff = cmp_hi_diff = 0`
/// → both flags = 1, `cmp_and = 1`. None of those values are observed
/// elsewhere on non-comparator rows.
fn push_comparator_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    let sel_eqz_i32 = selector_col(WasmOpcode::I32Eqz).unwrap();
    let sel_eqz_i64 = selector_col(WasmOpcode::I64Eqz).unwrap();
    let sel_eq = selector_col(WasmOpcode::I32Eq).unwrap();
    let sel_ne = selector_col(WasmOpcode::I32Ne).unwrap();
    let sel_i64_eq = selector_col(WasmOpcode::I64Eq).unwrap();
    let sel_i64_ne = selector_col(WasmOpcode::I64Ne).unwrap();

    // cmp_lo_diff = read0_value on i32.eqz rows.
    push_gated_linear_zero(
        b,
        sel_eqz_i32,
        [(COL_CMP_LO_DIFF, F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    // cmp_lo_diff = read0_value (lo limb only) on i64.eqz rows.
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_CMP_LO_DIFF, F::ONE), (idx(stack.read0_value_lo), -F::ONE)],
    );
    // cmp_lo_diff = read0_value - read1_value on i32.eq/ne and i64.eq/ne rows.
    // (Same lo-limb diff expression for all four; hi-limb diff is pinned
    // separately below for the i64 pair.)
    b.push_row(
        [
            (sel_eq, F::ONE),
            (sel_ne, F::ONE),
            (sel_i64_eq, F::ONE),
            (sel_i64_ne, F::ONE),
        ],
        [
            (COL_CMP_LO_DIFF, F::ONE),
            (idx(stack.read0_value_lo), -F::ONE),
            (idx(stack.read1_value_lo), F::ONE),
        ],
        [],
    );

    push_zero_test_gadget(b, COL_CMP_LO_DIFF, COL_CMP_LO_INV, COL_CMP_LO_IS_ZERO);

    // cmp_hi_diff bindings for the i64 comparators (i64.eqz: read0_hi;
    // i64.eq/ne: read0_hi - read1_hi). Unconstrained on every other row
    // — the witness sets it to 0 so the hi zero-test flag becomes 1.
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_CMP_HI_DIFF, F::ONE), (idx(stack.read0_value_hi), -F::ONE)],
    );
    b.push_row(
        [(sel_i64_eq, F::ONE), (sel_i64_ne, F::ONE)],
        [
            (COL_CMP_HI_DIFF, F::ONE),
            (idx(stack.read0_value_hi), -F::ONE),
            (idx(stack.read1_value_hi), F::ONE),
        ],
        [],
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
        [(idx(stack.write0_value_lo), F::ONE), (COL_CMP_LO_IS_ZERO, -F::ONE)],
        [],
    );
    // write0_value = cmp_and on i64.eqz / i64.eq rows.
    b.push_row(
        [(sel_eqz_i64, F::ONE), (sel_i64_eq, F::ONE)],
        [(idx(stack.write0_value_lo), F::ONE), (COL_CMP_AND, -F::ONE)],
        [],
    );
    // write0_value = 1 - cmp_lo_is_zero on i32.ne rows.
    push_gated_linear_zero(
        b,
        sel_ne,
        [
            (idx(stack.write0_value_lo), F::ONE),
            (COL_CMP_LO_IS_ZERO, F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    // write0_value = 1 - cmp_and on i64.ne rows.
    push_gated_linear_zero(
        b,
        sel_i64_ne,
        [
            (idx(stack.write0_value_lo), F::ONE),
            (COL_CMP_AND, F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
}

fn push_shout_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, op_table: &OpTableColumns) {
    b.push_row(
        WasmOpTable::all()
            .into_iter()
            .map(|op| (selector_for_op_table(op), F::ONE)),
        [(idx(op_table.value), F::ONE), (idx(stack.write0_value_lo), -F::ONE)],
        [],
    );
    b.push_row(
        WasmOpTable::all()
            .into_iter()
            .map(|op| (selector_for_op_table(op), F::from_u64(u64::from(op.op_table_id())))),
        [(COL_ONE, F::ONE)],
        [(idx(op_table.id), F::ONE)],
    );
}

pub(super) fn idx(column: Column) -> usize {
    column.0
}

fn selector_for_op_table(op: WasmOpTable) -> usize {
    selector_for_lookup(match op {
        WasmOpTable::I32Clz => WasmOpcode::I32Clz,
        WasmOpTable::I32Ctz => WasmOpcode::I32Ctz,
        WasmOpTable::I32LtS => WasmOpcode::I32LtS,
        WasmOpTable::I32LtU => WasmOpcode::I32LtU,
        WasmOpTable::I32GtS => WasmOpcode::I32GtS,
        WasmOpTable::I32GtU => WasmOpcode::I32GtU,
        WasmOpTable::I32LeS => WasmOpcode::I32LeS,
        WasmOpTable::I32LeU => WasmOpcode::I32LeU,
        WasmOpTable::I32GeS => WasmOpcode::I32GeS,
        WasmOpTable::I32GeU => WasmOpcode::I32GeU,
        WasmOpTable::I32And => WasmOpcode::I32And,
        WasmOpTable::I32Or => WasmOpcode::I32Or,
        WasmOpTable::I32Xor => WasmOpcode::I32Xor,
        WasmOpTable::I32Mul => WasmOpcode::I32Mul,
        WasmOpTable::I64And => WasmOpcode::I64And,
        WasmOpTable::I64Or => WasmOpcode::I64Or,
        WasmOpTable::I64Xor => WasmOpcode::I64Xor,
        WasmOpTable::I64Mul => WasmOpcode::I64Mul,
        WasmOpTable::I32Shl => WasmOpcode::I32Shl,
        WasmOpTable::I32ShrU => WasmOpcode::I32ShrU,
        WasmOpTable::I32ShrS => WasmOpcode::I32ShrS,
        WasmOpTable::I32Rotl => WasmOpcode::I32Rotl,
        WasmOpTable::I32Rotr => WasmOpcode::I32Rotr,
        WasmOpTable::I32DivU => WasmOpcode::I32DivU,
        WasmOpTable::I32DivS => WasmOpcode::I32DivS,
        WasmOpTable::I32RemU => WasmOpcode::I32RemU,
        WasmOpTable::I32RemS => WasmOpcode::I32RemS,
        WasmOpTable::I32Popcnt => WasmOpcode::I32Popcnt,
        WasmOpTable::I64LtS => WasmOpcode::I64LtS,
        WasmOpTable::I64LtU => WasmOpcode::I64LtU,
        WasmOpTable::I64GtS => WasmOpcode::I64GtS,
        WasmOpTable::I64GtU => WasmOpcode::I64GtU,
        WasmOpTable::I64LeS => WasmOpcode::I64LeS,
        WasmOpTable::I64LeU => WasmOpcode::I64LeU,
        WasmOpTable::I64GeS => WasmOpcode::I64GeS,
        WasmOpTable::I64GeU => WasmOpcode::I64GeU,
        WasmOpTable::I64Shl => WasmOpcode::I64Shl,
        WasmOpTable::I64ShrS => WasmOpcode::I64ShrS,
        WasmOpTable::I64ShrU => WasmOpcode::I64ShrU,
        WasmOpTable::I64Rotl => WasmOpcode::I64Rotl,
        WasmOpTable::I64Rotr => WasmOpcode::I64Rotr,
        WasmOpTable::I64DivS => WasmOpcode::I64DivS,
        WasmOpTable::I64DivU => WasmOpcode::I64DivU,
        WasmOpTable::I64RemS => WasmOpcode::I64RemS,
        WasmOpTable::I64RemU => WasmOpcode::I64RemU,
        WasmOpTable::I64Clz => WasmOpcode::I64Clz,
        WasmOpTable::I64Ctz => WasmOpcode::I64Ctz,
        WasmOpTable::I64Popcnt => WasmOpcode::I64Popcnt,
    })
}
