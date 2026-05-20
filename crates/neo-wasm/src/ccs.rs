//! Owns the WASM VM contract and phase-1 core CCS.

use crate::layout::{COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV};

use super::gadgets::{
    add_conditional_select_gadget, push_gated_linear_zero, push_u32_le_bytes, push_zero_test_gadget,
    ConditionalSelectCols,
};
use super::isa::{opcode_code, opcode_info_from_code, WasmOpcode, WasmShoutOpcode};
use super::layout::{
    selector_col, ColumnWidth, COLUMN_SPECS, COL_ONE, COL_PC_EDGE_KIND, COL_SELECT_OUT_DELTA, COL_WIDE_AUX0,
    COL_WIDE_AUX1, SELECTOR_COLS, WITNESS_WIDTH,
};
use super::lookup_binding_builder::{
    build_wasm_lookup_binding_layout, CallColumns, Column, ControlColumns, FrameColumns, FunctionTypeColumns,
    GlobalsColumns, LinearMemoryColumns, LocalsColumns, MemoryPagesColumns, OperandStackColumns, ParamInitColumns,
    ShoutColumns, StateColumns, TableColumns, TableSizeColumns, ValueLimbByteColumns,
};
use super::tagged_r1cs_builder::{
    WasmConstraintCatalog, WasmConstraintScope, WasmConstraintTag, WasmTaggedR1csBuilder,
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

const LINEAR_MEMORY_OPS: &[WasmOpcode] = &[
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

const LOCAL_WRITE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const TABLE_READ_OPS: &[WasmOpcode] = &[WasmOpcode::TableGet, WasmOpcode::CallIndirect];
const LOCAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::LocalGet, WasmOpcode::LocalSet, WasmOpcode::LocalTee];
const GLOBAL_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::GlobalGet, WasmOpcode::GlobalSet];
const MEMORY_PAGE_OPS: &[WasmOpcode] = &[WasmOpcode::MemorySize, WasmOpcode::MemoryGrow];
const TABLE_VALUE_OPS: &[WasmOpcode] = &[WasmOpcode::TableGet, WasmOpcode::TableSet, WasmOpcode::CallIndirect];
fn always(label: &'static str) -> WasmConstraintTag {
    WasmConstraintTag {
        label,
        scope: WasmConstraintScope::Always,
    }
}

fn opcode_tag(label: &'static str, opcode: WasmOpcode) -> WasmConstraintTag {
    WasmConstraintTag {
        label,
        scope: WasmConstraintScope::Opcode(opcode),
    }
}

fn shared(label: &'static str, opcodes: &[WasmOpcode]) -> WasmConstraintTag {
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
    let param_init = layout.param_init;
    let call = layout.call;
    let frame = layout.frame;
    let stack = layout.stack;
    let locals = layout.locals;
    let globals = layout.globals;
    let memory_pages = layout.memory_pages;
    let table = layout.table;
    let table_sizes = layout.table_sizes;
    let function_types = layout.function_types;
    let module_types = layout.module_types;
    let stack_read0_bytes = layout.stack_read0_bytes;
    let stack_read1_bytes = layout.stack_read1_bytes;
    let stack_read2_bytes = layout.stack_read2_bytes;
    let stack_write0_bytes = layout.stack_write0_bytes;
    let local_value_bytes = layout.local_value_bytes;
    let global_value_bytes = layout.global_value_bytes;
    let linear_memory = layout.linear_memory;
    let shout = layout.shout;
    let mut b = WasmTaggedR1csBuilder::new(WITNESS_WIDTH, COL_ONE)?;

    b.with_tag(always("boolean columns"), |b| {
        for spec in COLUMN_SPECS
            .iter()
            .filter(|s| s.width == ColumnWidth::Boolean)
        {
            b.push_boolean(spec.index);
        }
    });

    b.with_tag(always("value byte decomposition"), |b| {
        for (word, bytes) in [
            (stack.read0_value, stack_read0_bytes),
            (stack.read1_value, stack_read1_bytes),
            (stack.read2_value, stack_read2_bytes),
            (stack.write0_value, stack_write0_bytes),
            (locals.value_lo, local_value_bytes),
            (globals.value, global_value_bytes),
        ] {
            push_value_limb_bytes_constraints(b, word, bytes);
        }
        for (word, bytes) in [
            (stack.read0_value_hi, stack_read0_bytes.hi),
            (stack.read1_value_hi, stack_read1_bytes.hi),
            (stack.read2_value_hi, stack_read2_bytes.hi),
            (stack.write0_value_hi, stack_write0_bytes.hi),
            (locals.value_hi, local_value_bytes.hi),
            (globals.value_hi, global_value_bytes.hi),
        ] {
            push_u32_le_bytes(b, COL_ONE, idx(word), bytes.map(idx));
        }
    });

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

    b.with_tag(shared("locals write gate", LOCAL_WRITE_OPS), |b| {
        b.push_linear_zero([
            (idx(locals.write_enabled), F::ONE),
            (selector_col(WasmOpcode::LocalSet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::LocalTee).unwrap(), -F::ONE),
        ]);
    });

    b.with_tag(shared("table read gate", TABLE_READ_OPS), |b| {
        b.push_linear_zero([
            (idx(table.read_enabled), F::ONE),
            (selector_col(WasmOpcode::TableGet).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
        ]);
    });

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

    b.with_tag(always("row kind one hot"), |b| {
        b.push_linear_zero([
            (idx(control.is_program_row), F::ONE),
            (idx(param_init.param_init_active_before), F::ONE),
            (COL_ONE, -F::ONE),
        ]);
    });

    b.with_tag(always("aux call param init shape"), |b| {
        let param_init_row_gate = idx(param_init.param_init_active_before);

        // pc_after == pc_before
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(idx(state.pc_after), F::ONE), (idx(state.pc_before), -F::ONE)],
        );

        // init local outputs don't modify the stack
        // these two also imply sp_after == sp_before
        push_gated_linear_zero(b, param_init_row_gate, [(idx(control.stack_reads), F::ONE)]);
        push_gated_linear_zero(b, param_init_row_gate, [(idx(control.stack_writes), F::ONE)]);

        // write to the locals memory the value read from the stack
        //
        // remember that there is only one lane for locals access
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(idx(stack.read0_value), F::ONE), (idx(locals.value_lo), -F::ONE)],
        );
        push_gated_linear_zero(
            b,
            param_init_row_gate,
            [(idx(stack.read0_value_hi), F::ONE), (idx(locals.value_hi), -F::ONE)],
        );
    });

    b.with_tag(always("guest call flag"), |b| {
        push_guest_call_flag_constraints(b, &call, &function_types);
    });

    b.with_tag(always("call param init enter mode"), |b| {
        push_call_param_init_enter_mode_constraints(b, &control, &param_init, &call, &function_types);
    });

    b.with_tag(always("call param init exit mode"), |b| {
        push_call_param_init_exit_mode_constraints(b, &param_init);
    });

    b.with_tag(always("call param init aux row"), |b| {
        push_call_param_init_aux_row_constraints(b, &state, &param_init, &function_types, &stack, &locals);
    });

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

    b.push_linear_zero(
        [
            (idx(control.halted), F::ONE),
            (idx(call.call_stack_pop_present), F::ONE),
            (selector_col(WasmOpcode::Return).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::End).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::Unreachable).unwrap(), -F::ONE),
        ]
        .into_iter(),
    );
    b.push_linear_zero(
        [
            (COL_PC_EDGE_KIND, F::ONE),
            (selector_col(WasmOpcode::Return).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::End).unwrap(), -F::ONE),
            (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::from_u64(2)),
            (selector_col(WasmOpcode::Unreachable).unwrap(), -F::from_u64(3)),
        ]
        .into_iter(),
    );
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

    b.with_tag(always("return pc restoration"), |b| {
        b.push_row(
            [(idx(call.call_stack_pop_present), F::ONE)],
            [
                (idx(state.pc_after), F::ONE),
                (idx(call.call_stack_pop_return_pc), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(idx(call.call_stack_pop_present), F::ONE)],
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::Return).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::End).unwrap(), -F::ONE),
            ],
            [],
        );
    });
    b.with_tag(always("locals fbp transition"), |b| {
        push_locals_fbp_transition_constraints(b, &call, &frame);
    });

    b.push_linear_zero(
        [
            (idx(shout.enabled), F::ONE),
            (selector_for_lookup(WasmOpcode::I32Clz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Ctz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Eqz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I64Eqz), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Eq), -F::ONE),
            (selector_for_lookup(WasmOpcode::I32Ne), -F::ONE),
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
    push_linear_memory_constraints(&mut b, &stack, stack_read1_bytes, stack_write0_bytes, &linear_memory);
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
    b.with_tag(shared("locals value constraints", LOCAL_VALUE_OPS), |b| {
        push_local_value_constraints(b, &stack, &locals);
    });
    b.with_tag(shared("globals value constraints", GLOBAL_VALUE_OPS), |b| {
        push_global_value_constraints(b, &stack, &globals);
    });
    b.with_tag(shared("memory page constraints", MEMORY_PAGE_OPS), |b| {
        push_memory_pages_constraints(b, &stack, &memory_pages);
    });
    b.with_tag(shared("table value constraints", TABLE_VALUE_OPS), |b| {
        push_table_value_constraints(b, &stack, &table);
    });
    b.with_tag(opcode_tag("table size constraints", WasmOpcode::TableSize), |b| {
        push_table_size_constraints(b, &stack, &table_sizes);
    });
    b.with_tag(
        opcode_tag("call_indirect type constraints", WasmOpcode::CallIndirect),
        |b| {
            push_call_indirect_type_constraints(b, &function_types, &module_types);
        },
    );
    b.with_tag(always("dynamic call stack arity"), |b| {
        push_dynamic_call_stack_arity_constraints(b, &control, &function_types, &table);
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

fn f_u64(v: u64) -> F {
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
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Add).unwrap(),
        [
            (idx(stack.read0_value), F::ONE),
            (idx(stack.read1_value), F::ONE),
            (idx(stack.write0_value), -F::ONE),
        ],
    );
}

fn push_sub_relation(b: &mut R1csBuilder, stack: &OperandStackColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Sub).unwrap(),
        [
            (idx(stack.read0_value), F::ONE),
            (idx(stack.read1_value), -F::ONE),
            (idx(stack.write0_value), -F::ONE),
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

fn push_local_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, locals: &LocalsColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalGet).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalSet).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::LocalTee).unwrap(),
        [(idx(locals.value_lo), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
}

fn push_global_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, globals: &GlobalsColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalGet).unwrap(),
        [(idx(globals.value), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::GlobalSet).unwrap(),
        [(idx(globals.value), F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
}

fn push_table_value_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, table: &TableColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(idx(table.value), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(idx(table.value), F::ONE), (idx(stack.read1_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableGet).unwrap(),
        [(idx(table.index), F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSet).unwrap(),
        [(idx(table.index), F::ONE), (idx(stack.read0_value), -F::ONE)],
    );
}

fn push_memory_pages_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, memory_pages: &MemoryPagesColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::MemorySize).unwrap(),
        [(idx(memory_pages.before), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::MemorySize).unwrap(),
        [(idx(memory_pages.after), F::ONE), (idx(memory_pages.before), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::MemoryGrow).unwrap(),
        [(idx(memory_pages.before), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
}

fn push_table_size_constraints(b: &mut R1csBuilder, stack: &OperandStackColumns, table_sizes: &TableSizeColumns) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::TableSize).unwrap(),
        [(idx(table_sizes.value), F::ONE), (idx(stack.write0_value), -F::ONE)],
    );
}

fn push_call_indirect_type_constraints(
    b: &mut R1csBuilder,
    function_types: &FunctionTypeColumns,
    module_types: &super::lookup_binding_builder::ModuleTypeColumns,
) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [
            (idx(function_types.type_id), F::ONE),
            (idx(module_types.expected_type_id), -F::ONE),
        ],
    );
}

fn push_dynamic_call_stack_arity_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    function_types: &FunctionTypeColumns,
    table: &TableColumns,
) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::Call).unwrap(),
        [
            (idx(control.stack_reads), F::ONE),
            (idx(function_types.param_count), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::Call).unwrap(),
        [(idx(control.stack_writes), F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [(idx(function_types.function_ref), F::ONE), (idx(table.value), -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [
            (idx(control.stack_reads), F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::CallIndirect).unwrap(),
        [(idx(control.stack_writes), F::ONE)],
    );
}

fn push_guest_call_flag_constraints(b: &mut R1csBuilder, call: &CallColumns, function_types: &FunctionTypeColumns) {
    let call_selector = selector_col(WasmOpcode::Call).unwrap();
    let call_indirect = selector_col(WasmOpcode::CallIndirect).unwrap();

    b.push_row(
        [(call_selector, F::ONE), (call_indirect, F::ONE)],
        [(idx(function_types.is_guest), F::ONE)],
        [(idx(call.call_stack_push_present), F::ONE)],
    );
}

fn push_call_param_init_enter_mode_constraints(
    b: &mut R1csBuilder,
    control: &ControlColumns,
    param_init: &ParamInitColumns,
    call: &CallColumns,
    function_types: &FunctionTypeColumns,
) {
    let guest_call = idx(call.call_stack_push_present);

    b.push_row(
        [
            (idx(control.is_program_row), F::ONE),
            (idx(call.call_stack_push_present), -F::ONE),
        ],
        // Only guest calls may enter param-init mode from a program row.
        // Aux rows are excluded by `is_program_row = 0`, so multi-param init
        // can continue until the global remaining-after zero test turns it off.
        [(idx(param_init.param_init_active_after), F::ONE)],
        [],
    );

    // guest_call => param_init_remaining' == param_count
    push_gated_linear_zero(
        b,
        guest_call,
        [
            (idx(param_init.param_init_remaining_after), F::ONE),
            (idx(function_types.param_count), -F::ONE),
        ],
    );
}

fn push_call_param_init_exit_mode_constraints(b: &mut R1csBuilder, param_init: &ParamInitColumns) {
    b.push_linear_zero([
        (idx(param_init.param_init_active_after), F::ONE),
        (idx(param_init.param_init_remaining_after_is_zero), F::ONE),
        (COL_ONE, -F::ONE),
    ]);

    // if we reached the end of the local initialization sequence
    push_zero_test_gadget(
        b,
        idx(param_init.param_init_remaining_after),
        idx(param_init.param_init_remaining_after_inv),
        idx(param_init.param_init_remaining_after_is_zero),
    );
}

fn push_call_param_init_aux_row_constraints(
    b: &mut R1csBuilder,
    state: &StateColumns,
    param_init: &ParamInitColumns,
    function_types: &FunctionTypeColumns,
    stack: &OperandStackColumns,
    locals: &LocalsColumns,
) {
    let selector = idx(param_init.param_init_active_before);

    // in_param_init_mode => param_init_remaining' = param_init_remaining - 1
    push_gated_linear_zero(
        b,
        selector,
        [
            (idx(param_init.param_init_remaining_before), F::ONE),
            (idx(param_init.param_init_remaining_after), -F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );

    push_gated_linear_zero(
        b,
        selector,
        [
            // stack_addr + remaining = sp_before + param_count
            //
            // remaining goes down, so stack_addr may go up (the rhs is constant while selector is on)
            (idx(stack.read0_addr), F::ONE),
            (idx(state.sp_before), -F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (idx(param_init.param_init_remaining_before), F::ONE),
        ],
    );

    push_gated_linear_zero(
        b,
        selector,
        // The aux row writes callee local `param_count - remaining_before`.
        [
            (idx(locals.index), F::ONE),
            (idx(function_types.param_count), -F::ONE),
            (idx(param_init.param_init_remaining_before), F::ONE),
        ],
    );

    // the pc is not constrained for the aux opcode (since it's not a real
    // opcode, it's not in the next pc table)
    //
    // we assert here that it doesn't change
    push_gated_linear_zero(
        b,
        selector,
        [(idx(state.pc_after), F::ONE), (idx(state.pc_before), -F::ONE)],
    );
}

fn push_locals_fbp_transition_constraints(b: &mut R1csBuilder, call: &CallColumns, frame: &FrameColumns) {
    let guest_call = idx(call.call_stack_push_present);
    let pop = idx(call.call_stack_pop_present);

    push_gated_linear_zero(
        b,
        guest_call,
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(frame.locals_fbp_before), -F::ONE),
            (idx(frame.current_function_num_locals), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        pop,
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(call.call_stack_pop_caller_fbp), -F::ONE),
        ],
    );
    b.push_row(
        [(COL_ONE, F::ONE), (guest_call, -F::ONE), (pop, -F::ONE)],
        [
            (idx(frame.locals_fbp_after), F::ONE),
            (idx(frame.locals_fbp_before), -F::ONE),
        ],
        [],
    );
}

fn push_linear_memory_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    stack_read1_bytes: ValueLimbByteColumns,
    stack_write0_bytes: ValueLimbByteColumns,
    linear_memory: &LinearMemoryColumns,
) {
    let load_selector = selector_col(WasmOpcode::I32Load).unwrap();
    let i64_load_selector = selector_col(WasmOpcode::I64Load).unwrap();
    let load8s_selector = selector_col(WasmOpcode::I32Load8S).unwrap();
    let load8_selector = selector_col(WasmOpcode::I32Load8U).unwrap();
    let load16s_selector = selector_col(WasmOpcode::I32Load16S).unwrap();
    let load16_selector = selector_col(WasmOpcode::I32Load16U).unwrap();
    let store_selector = selector_col(WasmOpcode::I32Store).unwrap();
    let i64_store_selector = selector_col(WasmOpcode::I64Store).unwrap();
    let store8_selector = selector_col(WasmOpcode::I32Store8).unwrap();
    let store16_selector = selector_col(WasmOpcode::I32Store16).unwrap();
    let linear_memory_selectors = [
        load_selector,
        i64_load_selector,
        load8s_selector,
        load8_selector,
        load16s_selector,
        load16_selector,
        store_selector,
        i64_store_selector,
        store8_selector,
        store16_selector,
    ];
    b.with_tag(shared("linear memory address normalization", LINEAR_MEMORY_OPS), |b| {
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.lane0_addr), f_u64(4)),
                (idx(linear_memory.byte_offset), F::ONE),
                (idx(stack.read0_value), -F::ONE),
                (idx(linear_memory.imm_offset), -F::ONE),
            ],
            [],
        );
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.offset_is[0]), F::ONE),
                (idx(linear_memory.offset_is[1]), F::ONE),
                (idx(linear_memory.offset_is[2]), F::ONE),
                (idx(linear_memory.offset_is[3]), F::ONE),
                (COL_ONE, -F::ONE),
            ],
            [],
        );
        b.push_row(
            linear_memory_selectors
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.byte_offset), F::ONE),
                (idx(linear_memory.offset_is[1]), -F::ONE),
                (idx(linear_memory.offset_is[2]), -f_u64(2)),
                (idx(linear_memory.offset_is[3]), -f_u64(3)),
            ],
            [],
        );
        for selector in linear_memory_selectors {
            push_u32_le_bytes(
                b,
                selector,
                idx(linear_memory.lane0_value),
                linear_memory.lane0_bytes.map(idx),
            );
            push_u32_le_bytes(
                b,
                selector,
                idx(linear_memory.lane1_value),
                linear_memory.lane1_bytes.map(idx),
            );
        }
        for selector in [i64_load_selector, i64_store_selector] {
            push_u32_le_bytes(
                b,
                selector,
                idx(linear_memory.lane2_value),
                linear_memory.lane2_bytes.map(idx),
            );
        }
    });

    b.with_tag(shared("linear memory width selectors", LINEAR_MEMORY_OPS), |b| {
        // Under each gate, both offset families are one-hot, so the weighted 1..4
        // fingerprint is injective and can replace four per-case equalities.
        b.push_linear_zero(
            [
                (idx(linear_memory.byte_width_offset_is[0]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[1]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[2]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_byte_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_byte_width),
            [
                (idx(linear_memory.byte_width_offset_is[0]), F::ONE),
                (idx(linear_memory.byte_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.byte_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.byte_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );

        b.push_linear_zero(
            [
                (idx(linear_memory.half_width_offset_is[0]), F::ONE),
                (idx(linear_memory.half_width_offset_is[1]), F::ONE),
                (idx(linear_memory.half_width_offset_is[2]), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_half_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_half_width),
            [
                (idx(linear_memory.half_width_offset_is[0]), F::ONE),
                (idx(linear_memory.half_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.half_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.half_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );

        b.push_linear_zero(
            [
                (idx(linear_memory.full_width_offset_is[0]), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), F::ONE),
                (idx(linear_memory.full_width_offset_is[2]), F::ONE),
                (idx(linear_memory.full_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_full_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_full_width),
            [
                (idx(linear_memory.full_width_offset_is[0]), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.full_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.full_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.double_width_offset_is[0]), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), F::ONE),
                (idx(linear_memory.double_width_offset_is[2]), F::ONE),
                (idx(linear_memory.double_width_offset_is[3]), F::ONE),
                (idx(linear_memory.is_double_width), -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.is_double_width),
            [
                (idx(linear_memory.double_width_offset_is[0]), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), f_u64(4)),
                (idx(linear_memory.offset_is[0]), -F::ONE),
                (idx(linear_memory.offset_is[1]), -f_u64(2)),
                (idx(linear_memory.offset_is[2]), -f_u64(3)),
                (idx(linear_memory.offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.i64_load_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[1]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[2]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[3]), F::ONE),
                (i64_load_selector, -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            i64_load_selector,
            [
                (idx(linear_memory.i64_load_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_load_offset_is[1]), f_u64(2)),
                (idx(linear_memory.i64_load_offset_is[2]), f_u64(3)),
                (idx(linear_memory.i64_load_offset_is[3]), f_u64(4)),
                (idx(linear_memory.double_width_offset_is[0]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), -f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), -f_u64(4)),
            ],
        );
        b.push_linear_zero(
            [
                (idx(linear_memory.i64_store_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[1]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[2]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[3]), F::ONE),
                (i64_store_selector, -F::ONE),
            ]
            .into_iter(),
        );
        push_gated_linear_zero(
            b,
            i64_store_selector,
            [
                (idx(linear_memory.i64_store_offset_is[0]), F::ONE),
                (idx(linear_memory.i64_store_offset_is[1]), f_u64(2)),
                (idx(linear_memory.i64_store_offset_is[2]), f_u64(3)),
                (idx(linear_memory.i64_store_offset_is[3]), f_u64(4)),
                (idx(linear_memory.double_width_offset_is[0]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -f_u64(2)),
                (idx(linear_memory.double_width_offset_is[2]), -f_u64(3)),
                (idx(linear_memory.double_width_offset_is[3]), -f_u64(4)),
            ],
        );
    });

    b.with_tag(shared("linear memory lane usage", LINEAR_MEMORY_OPS), |b| {
        b.push_row(
            [load16_selector, store16_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [load_selector, store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.full_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.full_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [(idx(linear_memory.is_double_width), F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [(idx(linear_memory.use_lane1), F::ONE), (COL_ONE, -F::ONE)],
            [],
        );
        b.push_row(
            [i64_load_selector, i64_store_selector]
                .into_iter()
                .map(|selector| (selector, F::ONE)),
            [
                (idx(linear_memory.use_lane2), F::ONE),
                (idx(linear_memory.double_width_offset_is[1]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[2]), -F::ONE),
                (idx(linear_memory.double_width_offset_is[3]), -F::ONE),
            ],
            [],
        );
    });

    b.with_tag(shared("linear memory lane adjacency", LINEAR_MEMORY_OPS), |b| {
        push_gated_linear_zero(
            b,
            idx(linear_memory.use_lane1),
            [
                (idx(linear_memory.lane1_addr), F::ONE),
                (idx(linear_memory.lane0_addr), -F::ONE),
                (COL_ONE, -F::ONE),
            ],
        );
        push_gated_linear_zero(
            b,
            idx(linear_memory.use_lane2),
            [
                (idx(linear_memory.lane2_addr), F::ONE),
                (idx(linear_memory.lane1_addr), -F::ONE),
                (COL_ONE, -F::ONE),
            ],
        );
    });

    let linear_memory_load_access_byte_ops = [
        WasmOpcode::I32Load,
        WasmOpcode::I32Load8S,
        WasmOpcode::I32Load8U,
        WasmOpcode::I32Load16S,
        WasmOpcode::I32Load16U,
    ];
    b.with_tag(
        shared(
            "linear memory access bytes (loads)",
            &linear_memory_load_access_byte_ops,
        ),
        |b| {
            b.push_row(
                linear_memory_load_access_byte_ops.into_iter().map(|op| {
                    (
                        selector_col(op).expect("linear memory load access bytes selector"),
                        F::ONE,
                    )
                }),
                [
                    (idx(stack.write0_value), F::ONE),
                    (idx(linear_memory.access_bytes[0]), -F::ONE),
                    (idx(linear_memory.access_bytes[1]), -f_u64(1_u64 << 8)),
                    (idx(linear_memory.access_bytes[2]), -f_u64(1_u64 << 16)),
                    (idx(linear_memory.access_bytes[3]), -f_u64(1_u64 << 24)),
                ],
                [],
            );
        },
    );
    let linear_memory_store_access_byte_ops = [WasmOpcode::I32Store, WasmOpcode::I32Store8, WasmOpcode::I32Store16];
    b.with_tag(
        shared(
            "linear memory access bytes (stores)",
            &linear_memory_store_access_byte_ops,
        ),
        |b| {
            b.push_row(
                linear_memory_store_access_byte_ops.into_iter().map(|op| {
                    (
                        selector_col(op).expect("linear memory store access bytes selector"),
                        F::ONE,
                    )
                }),
                [
                    (idx(stack.read1_value), F::ONE),
                    (idx(linear_memory.access_bytes[0]), -F::ONE),
                    (idx(linear_memory.access_bytes[1]), -f_u64(1_u64 << 8)),
                    (idx(linear_memory.access_bytes[2]), -f_u64(1_u64 << 16)),
                    (idx(linear_memory.access_bytes[3]), -f_u64(1_u64 << 24)),
                ],
                [],
            );
        },
    );

    b.with_tag(
        opcode_tag("linear memory load32 byte routing", WasmOpcode::I32Load),
        |b| {
            push_linear_memory_load32_byte_selection(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_s routing", WasmOpcode::I32Load8S),
        |b| {
            push_linear_memory_load8_s_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load8_u routing", WasmOpcode::I32Load8U),
        |b| {
            push_linear_memory_load8_u_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_s routing", WasmOpcode::I32Load16S),
        |b| {
            push_linear_memory_load16_s_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory load16_u routing", WasmOpcode::I32Load16U),
        |b| {
            push_linear_memory_load16_u_constraints(b, linear_memory);
        },
    );
    b.with_tag(
        opcode_tag("linear memory store32 byte routing", WasmOpcode::I32Store),
        |b| {
            push_linear_memory_store32_byte_selection(b, linear_memory);
        },
    );
    b.with_tag(opcode_tag("linear memory store8 routing", WasmOpcode::I32Store8), |b| {
        push_linear_memory_store8_constraints(b, linear_memory);
    });
    b.with_tag(
        opcode_tag("linear memory store16 routing", WasmOpcode::I32Store16),
        |b| {
            push_linear_memory_store16_constraints(b, linear_memory);
        },
    );
    b.with_tag(opcode_tag("linear memory load64 routing", WasmOpcode::I64Load), |b| {
        push_linear_memory_load64_constraints(b, stack, stack_write0_bytes, linear_memory);
    });
    b.with_tag(opcode_tag("linear memory store64 routing", WasmOpcode::I64Store), |b| {
        push_linear_memory_store64_constraints(b, stack, stack_read1_bytes, linear_memory);
    });
}

fn push_linear_memory_load32_byte_selection(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    for (selector, lane_bytes) in [
        (idx(linear_memory.full_width_offset_is[0]), linear_memory.lane0_bytes),
        (
            idx(linear_memory.full_width_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        push_matching_byte_constraints(b, selector, linear_memory.access_bytes, lane_bytes);
    }
}

fn push_linear_memory_store32_byte_selection(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    for (selector, access_bytes, lane_bytes) in [
        (
            idx(linear_memory.full_width_offset_is[0]),
            linear_memory.access_bytes,
            linear_memory.lane0_bytes,
        ),
        (
            idx(linear_memory.full_width_offset_is[1]),
            linear_memory.access_bytes,
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[2]),
            linear_memory.access_bytes,
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            idx(linear_memory.full_width_offset_is[3]),
            linear_memory.access_bytes,
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        push_matching_byte_constraints(b, selector, access_bytes, lane_bytes);
    }
}

fn push_linear_memory_load8_u_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_subword_constraints(b, selector_col(WasmOpcode::I32Load8U).unwrap(), 1, linear_memory);
}

fn push_linear_memory_load8_s_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load8S).unwrap(),
        1,
        idx(linear_memory.access_bytes[0]),
        linear_memory,
    );
}

fn push_linear_memory_store8_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_store_subword_constraints(b, selector_col(WasmOpcode::I32Store8).unwrap(), 1, linear_memory);
}

fn push_linear_memory_load16_s_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_signed_subword_constraints(
        b,
        selector_col(WasmOpcode::I32Load16S).unwrap(),
        2,
        idx(linear_memory.access_bytes[1]),
        linear_memory,
    );
}

fn push_linear_memory_load16_u_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_load_subword_constraints(b, selector_col(WasmOpcode::I32Load16U).unwrap(), 2, linear_memory);
}

fn push_linear_memory_load64_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    stack_write0_bytes: ValueLimbByteColumns,
    linear_memory: &LinearMemoryColumns,
) {
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_load_offset_is[0]),
        [
            (idx(stack.write0_value), F::ONE),
            (idx(linear_memory.lane0_value), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_load_offset_is[0]),
        [
            (idx(stack.write0_value_hi), F::ONE),
            (idx(linear_memory.lane1_value), -F::ONE),
        ],
    );
    for (case_selector, low_bytes, high_bytes) in [
        (
            idx(linear_memory.i64_load_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
            [
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
            ],
        ),
        (
            idx(linear_memory.i64_load_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
            [
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
            ],
        ),
        (
            idx(linear_memory.i64_load_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
            [
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
                linear_memory.lane2_bytes[2],
            ],
        ),
    ] {
        for (byte, lane_byte) in stack_write0_bytes.lo.into_iter().zip(low_bytes) {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(lane_byte), -F::ONE)]);
        }
        for (byte, lane_byte) in stack_write0_bytes.hi.into_iter().zip(high_bytes) {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(lane_byte), -F::ONE)]);
        }
    }
}

fn push_linear_memory_store64_constraints(
    b: &mut R1csBuilder,
    stack: &OperandStackColumns,
    stack_read1_bytes: ValueLimbByteColumns,
    linear_memory: &LinearMemoryColumns,
) {
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_store_offset_is[0]),
        [
            (idx(stack.read1_value), F::ONE),
            (idx(linear_memory.lane0_value), -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        idx(linear_memory.i64_store_offset_is[0]),
        [
            (idx(stack.read1_value_hi), F::ONE),
            (idx(linear_memory.lane1_value), -F::ONE),
        ],
    );
    for (case_selector, target_bytes) in [
        (
            idx(linear_memory.i64_store_offset_is[1]),
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
            ],
        ),
        (
            idx(linear_memory.i64_store_offset_is[2]),
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
            ],
        ),
        (
            idx(linear_memory.i64_store_offset_is[3]),
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
                linear_memory.lane1_bytes[3],
                linear_memory.lane2_bytes[0],
                linear_memory.lane2_bytes[1],
                linear_memory.lane2_bytes[2],
            ],
        ),
    ] {
        for (byte, target_byte) in stack_read1_bytes
            .lo
            .into_iter()
            .chain(stack_read1_bytes.hi)
            .zip(target_bytes)
        {
            push_gated_linear_zero(b, case_selector, [(idx(byte), F::ONE), (idx(target_byte), -F::ONE)]);
        }
    }
}

fn push_linear_memory_store16_constraints(b: &mut R1csBuilder, linear_memory: &LinearMemoryColumns) {
    push_linear_memory_store_subword_constraints(b, selector_col(WasmOpcode::I32Store16).unwrap(), 2, linear_memory);
}

fn push_linear_memory_load_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
) {
    for byte in &linear_memory.access_bytes[width_bytes..] {
        push_gated_linear_zero(b, selector, [(idx(*byte), F::ONE)]);
    }
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
}

fn push_linear_memory_load_signed_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    sign_source_byte: usize,
    linear_memory: &LinearMemoryColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
    push_gated_linear_zero(
        b,
        selector,
        [
            (sign_source_byte, F::ONE),
            (idx(linear_memory.sign_ext_low7), -F::ONE),
            (idx(linear_memory.sign_ext_bit), -f_u64(128)),
        ],
    );
    for byte in &linear_memory.access_bytes[width_bytes..] {
        push_gated_linear_zero(
            b,
            selector,
            [(idx(*byte), F::ONE), (idx(linear_memory.sign_ext_bit), -f_u64(255))],
        );
    }
}

fn push_linear_memory_store_subword_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
) {
    push_linear_memory_subword_byte_constraints(b, selector, width_bytes, linear_memory);
}

fn push_linear_memory_subword_byte_constraints(
    b: &mut R1csBuilder,
    selector: usize,
    width_bytes: usize,
    linear_memory: &LinearMemoryColumns,
) {
    if width_bytes == 1 {
        push_gated_linear_zero(b, selector, [(idx(linear_memory.use_lane1), F::ONE)]);
    } else {
        push_gated_linear_zero(
            b,
            selector,
            [
                (idx(linear_memory.use_lane1), F::ONE),
                (idx(linear_memory.half_width_offset_is[3]), -F::ONE),
            ],
        );
    }
    for (case_selector, lane_bytes) in [
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[0])
            } else {
                idx(linear_memory.half_width_offset_is[0])
            },
            [
                linear_memory.lane0_bytes[0],
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[1])
            } else {
                idx(linear_memory.half_width_offset_is[1])
            },
            [
                linear_memory.lane0_bytes[1],
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[2])
            } else {
                idx(linear_memory.half_width_offset_is[2])
            },
            [
                linear_memory.lane0_bytes[2],
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
            ],
        ),
        (
            if width_bytes == 1 {
                idx(linear_memory.byte_width_offset_is[3])
            } else {
                idx(linear_memory.half_width_offset_is[3])
            },
            [
                linear_memory.lane0_bytes[3],
                linear_memory.lane1_bytes[0],
                linear_memory.lane1_bytes[1],
                linear_memory.lane1_bytes[2],
            ],
        ),
    ] {
        for (access_byte, lane_byte) in linear_memory.access_bytes[..width_bytes]
            .iter()
            .zip(lane_bytes[..width_bytes].iter())
        {
            push_gated_linear_zero(
                b,
                case_selector,
                [(idx(*access_byte), F::ONE), (idx(*lane_byte), -F::ONE)],
            );
        }
    }
}

fn push_matching_byte_constraints(b: &mut R1csBuilder, selector: usize, lhs: [Column; 4], rhs: [Column; 4]) {
    for (lhs_col, rhs_col) in lhs.into_iter().zip(rhs) {
        push_gated_linear_zero(b, selector, [(idx(lhs_col), F::ONE), (idx(rhs_col), -F::ONE)]);
    }
}

fn idx(column: Column) -> usize {
    column.0
}

fn push_value_limb_bytes_constraints(b: &mut R1csBuilder, word_lo: Column, bytes: ValueLimbByteColumns) {
    push_u32_le_bytes(b, COL_ONE, idx(word_lo), bytes.lo.map(idx));
}

fn selector_for_shout(op: WasmShoutOpcode) -> usize {
    selector_for_lookup(match op {
        WasmShoutOpcode::I32Clz => WasmOpcode::I32Clz,
        WasmShoutOpcode::I32Ctz => WasmOpcode::I32Ctz,
        WasmShoutOpcode::I32Eqz => WasmOpcode::I32Eqz,
        WasmShoutOpcode::I64Eqz => WasmOpcode::I64Eqz,
        WasmShoutOpcode::I32Eq => WasmOpcode::I32Eq,
        WasmShoutOpcode::I32Ne => WasmOpcode::I32Ne,
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
