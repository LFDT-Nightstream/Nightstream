//! Owns the WASM VM contract and phase-1 core CCS.
//!
//! Constraint families with substantial volume live in child modules
//! under `ccs/`; see [`linear_memory`] for the linear-memory load/store
//! row family. This file owns the top-level builder, the constraint
//! tag helpers (`always`, `host_event`, `shared`, `opcode_tag`), and the small shared
//! utilities (`idx`, `f_u64`, …) that those submodules consume via
//! `use super::*`.

mod call;
pub mod host_event_chain;
mod linear_memory;
mod memory_pages;
pub(crate) mod reentry;
mod stack_io;
mod trap;

pub(crate) use call::{FRAME_MOVE_REMAINING_AFTER_ZERO_TEST, LOCAL_ZERO_REMAINING_AFTER_ZERO_TEST};
pub(crate) use host_event_chain::HOST_EVENTS_REMAINING_ZERO_TEST;
pub(crate) use trap::{
    dividend_min_zero_test, divisor_neg1_zero_test, CALL_INDIRECT_ENTRY_ZERO_TEST, CALL_INDIRECT_TYPE_ZERO_TEST,
    DIVISOR_ZERO_TEST,
};

use super::gadgets::{push_gated_linear_zero, push_u32_le_bytes_decomp};
use super::isa::{opcode_code, opcode_info_from_code, WasmOpcode};
use super::layout::{
    selector_col, Column, COL_LOCALS_FBP_BEFORE, COL_LOCAL_ADDR_HI, COL_LOCAL_ADDR_LO, COL_LOCAL_INDEX,
    COL_LOCAL_ZERO_ACTIVE_BEFORE, COL_ONE, COL_PC_EDGE_KIND, COL_SELECT_OUT_DELTA_HI, COL_SELECT_OUT_DELTA_LO,
    COL_WIDE_AUX0, COL_WIDE_AUX1, PUBLIC_INPUTS, SELECTOR_COLS,
};
use super::relation_layout::{build_wasm_relation_layout, SignExtensionColumns};
use super::tagged_r1cs_builder::{WasmConstraintScope, WasmConstraintTag, WasmR1csBuilder, WasmTaggedR1csBuilder};
use crate::layout::{
    COL_ACTIVATION_FINISHED_AFTER, COL_ACTIVATION_FINISHED_BEFORE, COL_CALL_INDIRECT_IS_TRAP,
    COL_CALL_STACK_POP_PRESENT, COL_CMP_AND, COL_DIV_TRAP, COL_FRAME_MOVE_ACTIVE_BEFORE, COL_GATHER_ACTIVE,
    COL_GLOBAL_VALUE_HI, COL_IS_PROGRAM_ROW, COL_LOCAL_VALUE_HI, COL_MEM_OOB, COL_OPCODE_CODE, COL_OUTPUT_CAPTURED,
    COL_PC_EDGE_KIND_INV, COL_PC_EDGE_KIND_IS_STATIC, COL_PC_ROM_ACTIVE, COL_PROGRAM_GLOBAL_INDEX_ACTIVE,
    COL_PROGRAM_LOCAL_INDEX_ACTIVE, COL_PROGRAM_TABLE_ID_ACTIVE, COL_SELECT_COND_IS_ZERO, COL_SELECT_SCRATCH_INV,
    COL_SEL_SELECT, COL_SP_AFTER, COL_SP_BEFORE, COL_STACK_READS, COL_STACK_READ_ACTIVE, COL_STACK_READ_ADDR_HI,
    COL_STACK_READ_ADDR_LO, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_ACTIVE,
    COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO, COL_STACK_WRITE0_VALUE_HI, COL_STACK_WRITE0_VALUE_LO,
    COL_STACK_WRITES, COL_TABLE_OOB, COL_WIDE_VALUES_ENABLED, COL_ZERO_TEST_HI_INPUT, COL_ZERO_TEST_HI_INV,
    COL_ZERO_TEST_HI_IS_ZERO, COL_ZERO_TEST_LO_INPUT, COL_ZERO_TEST_LO_INV, COL_ZERO_TEST_LO_IS_ZERO,
};
use neo_application::{ApplicationRelation, ConditionalSelect, ZeroTest};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

pub(crate) const PC_EDGE_KIND_ZERO_TEST: ZeroTest =
    ZeroTest::column(COL_PC_EDGE_KIND, COL_PC_EDGE_KIND_INV, COL_PC_EDGE_KIND_IS_STATIC);

pub(crate) const SELECT_COND_ZERO_TEST: ZeroTest = ZeroTest::column(
    COL_STACK_READ_VALUE_LO[2],
    COL_SELECT_SCRATCH_INV,
    COL_SELECT_COND_IS_ZERO,
);

pub(crate) const LO_ZERO_TEST: ZeroTest =
    ZeroTest::column(COL_ZERO_TEST_LO_INPUT, COL_ZERO_TEST_LO_INV, COL_ZERO_TEST_LO_IS_ZERO);

pub(crate) const HI_ZERO_TEST: ZeroTest =
    ZeroTest::column(COL_ZERO_TEST_HI_INPUT, COL_ZERO_TEST_HI_INV, COL_ZERO_TEST_HI_IS_ZERO);

pub(crate) const SELECT_LO_MUX: ConditionalSelect<2> = ConditionalSelect {
    activation: COL_SEL_SELECT,
    condition: [(COL_ONE, F::ONE), (COL_SELECT_COND_IS_ZERO, F::NEG_ONE)],
    lhs: COL_STACK_READ_VALUE_LO[0],
    rhs: COL_STACK_READ_VALUE_LO[1],
    output: COL_STACK_WRITE0_VALUE_LO,
    delta: COL_SELECT_OUT_DELTA_LO,
};

pub(crate) const SELECT_HI_MUX: ConditionalSelect<2> = ConditionalSelect {
    activation: COL_SEL_SELECT,
    condition: [(COL_ONE, F::ONE), (COL_SELECT_COND_IS_ZERO, F::NEG_ONE)],
    lhs: COL_STACK_READ_VALUE_HI[0],
    rhs: COL_STACK_READ_VALUE_HI[1],
    output: COL_STACK_WRITE0_VALUE_HI,
    delta: COL_SELECT_OUT_DELTA_HI,
};

pub(super) fn wide_value_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.uses_wide_values())
        .collect()
}

pub(super) fn linear_memory_ops() -> Vec<WasmOpcode> {
    WasmOpcode::supported()
        .into_iter()
        .filter(|op| op.uses_linear_memory())
        .collect()
}

fn always(label: &'static str) -> WasmConstraintTag {
    WasmConstraintTag::new(label, WasmConstraintScope::Always)
}

pub(super) fn host_event(label: &'static str) -> WasmConstraintTag {
    WasmConstraintTag::new(label, WasmConstraintScope::HostEvent)
}

pub(super) fn opcode_tag(label: &'static str, opcode: WasmOpcode) -> WasmConstraintTag {
    WasmConstraintTag::new(label, WasmConstraintScope::Opcode(opcode))
}

pub(super) fn shared(label: &'static str, opcodes: &[WasmOpcode]) -> WasmConstraintTag {
    WasmConstraintTag::new(label, WasmConstraintScope::Opcodes(opcodes.to_vec().into_boxed_slice()))
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

fn fixed_stack_reads_terms() -> Vec<(usize, F)> {
    let mut terms = vec![(COL_STACK_READS, F::ONE), (COL_FRAME_MOVE_ACTIVE_BEFORE, -F::ONE)];
    for op in WasmOpcode::supported().into_iter().filter(|op| {
        !matches!(
            op,
            WasmOpcode::Call | WasmOpcode::CallIndirect | WasmOpcode::ReturnCall | WasmOpcode::ReturnCallIndirect
        )
    }) {
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

fn fixed_stack_writes_terms() -> Vec<(usize, F)> {
    let mut terms = vec![
        (COL_STACK_WRITES, F::ONE),
        (COL_FRAME_MOVE_ACTIVE_BEFORE, -F::ONE),
        (COL_LOCAL_ZERO_ACTIVE_BEFORE, -F::ONE),
    ];
    for op in WasmOpcode::supported().into_iter().filter(|op| {
        !matches!(
            op,
            WasmOpcode::Call | WasmOpcode::CallIndirect | WasmOpcode::ReturnCall | WasmOpcode::ReturnCallIndirect
        )
    }) {
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

fn fixed_stack_arity_gate_terms() -> [(usize, F); 6] {
    [
        (COL_ONE, F::ONE),
        (selector_col(WasmOpcode::Call).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::CallIndirect).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::ReturnCall).unwrap(), -F::ONE),
        (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), -F::ONE),
        (COL_GATHER_ACTIVE, -F::ONE),
    ]
}

pub fn build_wasm_relation() -> Result<ApplicationRelation<WasmConstraintScope>, String> {
    let layout = build_wasm_relation_layout();
    let linear_memory = layout.linear_memory;
    let mut builder = WasmR1csBuilder::new(PUBLIC_INPUTS, COL_ONE).map_err(|error| error.to_string())?;
    let mut b = builder.tagged(always("unlabeled"));

    b.with_tag(shared("wide value gating", &wide_value_ops()), |b| {
        b.push_row(
            [(COL_IS_PROGRAM_ROW, F::ONE)],
            std::iter::once((COL_WIDE_VALUES_ENABLED, F::ONE)).chain(
                wide_value_ops()
                    .into_iter()
                    .map(|op| (selector_col(op).expect("wide value op selector"), -F::ONE)),
            ),
            [],
        );
    });

    b.with_tag(shared("linear memory lane0 gate", &linear_memory_ops()), |b| {
        b.push_linear_zero(
            std::iter::once((idx(linear_memory.use_lane0), F::ONE)).chain(
                linear_memory_ops()
                    .into_iter()
                    .map(|op| (selector_col(op).expect("memory op selector"), -F::ONE)),
            ),
        );
    });

    stack_io::push_stack_io_constraints(&mut b);
    memory_pages::push_memory_pages_constraints(&mut b);

    b.with_tag(always("narrow high limbs zero"), |b| {
        b.push_row(
            [(COL_STACK_READ_VALUE_HI[0], F::ONE)],
            [
                (COL_ONE, F::ONE),
                (COL_WIDE_VALUES_ENABLED, -F::ONE),
                (COL_OUTPUT_CAPTURED, -F::ONE),
            ],
            [],
        );
        for column in [
            COL_STACK_READ_VALUE_HI[1],
            COL_STACK_READ_VALUE_HI[2],
            COL_STACK_WRITE0_VALUE_HI,
            COL_LOCAL_VALUE_HI,
            COL_GLOBAL_VALUE_HI,
        ] {
            b.push_row(
                [(column, F::ONE)],
                [(COL_ONE, F::ONE), (COL_WIDE_VALUES_ENABLED, -F::ONE)],
                [],
            );
        }
    });

    call::push_call_constraints(&mut b);
    reentry::push_constraints(&mut b);

    b.with_tag(always("activation_finished state transition"), |b| {
        b.push_row(
            [(COL_IS_PROGRAM_ROW, F::ONE)],
            [(COL_ACTIVATION_FINISHED_BEFORE, F::ONE)],
            [],
        );
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (COL_IS_PROGRAM_ROW, -F::ONE),
                (super::layout::COL_TURN_BOUNDARY, -F::ONE),
                (super::layout::COL_HOST_RESUME, -F::ONE),
            ],
            [
                (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
                (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
            ],
            [],
        );
    });
    host_event_chain::push_constraints(&mut b);

    b.with_tag(always("opcode selector one hot"), |b| {
        b.push_linear_zero(
            SELECTOR_COLS
                .into_iter()
                .map(|col| (col, F::ONE))
                .chain([(COL_IS_PROGRAM_ROW, -F::ONE)]),
        );
    });

    b.with_tag(always("table descriptor packing"), |b| {
        b.push_linear_zero([
            (crate::layout::COL_TABLE_DESCRIPTOR, F::ONE),
            (crate::layout::COL_TABLE_ID, -F::ONE),
            (
                crate::layout::COL_EXPECTED_TYPE_ID,
                -F::from_u32(crate::table_descriptor::TABLE_TYPE_FACTOR),
            ),
        ]);
    });

    b.with_tag(always("program immediate gates"), |b| {
        let gates: [(usize, fn(WasmOpcode) -> bool); 3] = [
            (COL_PROGRAM_LOCAL_INDEX_ACTIVE, WasmOpcode::uses_local_index_immediate),
            (COL_PROGRAM_GLOBAL_INDEX_ACTIVE, WasmOpcode::uses_global_index_immediate),
            (COL_PROGRAM_TABLE_ID_ACTIVE, WasmOpcode::uses_table_id_immediate),
        ];
        for (gate, uses_immediate) in gates {
            b.push_linear_zero(
                std::iter::once((gate, F::ONE)).chain(
                    WasmOpcode::supported()
                        .into_iter()
                        .filter(|opcode| uses_immediate(*opcode))
                        .map(|opcode| (selector_col(opcode).expect("immediate consumer selector"), -F::ONE)),
                ),
            );
        }
    });

    b.with_tag(always("memory activation support"), |b| {
        for support in crate::memory_activation::derived_activation_supports() {
            b.push_row(
                [(support.gate, F::ONE)],
                std::iter::once((COL_ONE, F::ONE)).chain(support.atoms.into_iter().map(|atom| (atom, -F::ONE))),
                [],
            );
        }
    });

    b.with_tag(always("opcode decode"), |b| {
        b.push_linear_zero(
            std::iter::once((COL_OPCODE_CODE, F::ONE)).chain(WasmOpcode::supported().into_iter().map(|op| {
                (
                    selector_col(op).expect("supported opcode selector"),
                    -f_u16(opcode_code(op)),
                )
            })),
        );
    });

    b.with_tag(always("program header packing"), |b| {
        b.push_linear_zero([
            (crate::layout::COL_PROGRAM_HEADER, F::ONE),
            (COL_OPCODE_CODE, -F::ONE),
            (COL_PC_EDGE_KIND, -F::from_u64(65536)),
        ]);
    });
    b.with_tag(always("host slot descriptor packing"), |b| {
        use crate::layout::*;
        b.push_linear_zero([
            (COL_HOST_EVENT_SLOT_DESCRIPTOR, F::ONE),
            (COL_HOST_EVENT_SLOT_KIND, -F::ONE),
            (COL_HOST_EVENT_SLOT_ARG, -F::from_u64(32)),
            (COL_HOST_EVENT_SLOT_VARIANT, -F::from_u64(8192)),
        ]);
    });
    b.with_tag(always("pc fallthrough flag"), |b| {
        b.push_linear_zero(
            std::iter::once((crate::layout::COL_PC_FALLTHROUGH, F::ONE)).chain(
                WasmOpcode::supported()
                    .into_iter()
                    .filter(|op| op.falls_through())
                    .map(|op| (selector_col(op).unwrap(), -F::ONE)),
            ),
        );
        push_gated_linear_zero(
            b,
            crate::layout::COL_PC_FALLTHROUGH,
            [
                (crate::layout::COL_PC_AFTER, F::ONE),
                (crate::layout::COL_PC_BEFORE, -F::ONE),
                (COL_ONE, -F::ONE),
            ],
        );
    });

    b.push_linear_zero([
        (COL_SP_AFTER, F::ONE),
        (COL_SP_BEFORE, -F::ONE),
        (COL_STACK_READS, F::ONE),
        (COL_STACK_WRITES, -F::ONE),
        (super::layout::COL_FRAME_SP_DROP, F::ONE),
        (host_event_chain::gather_arg_read_kind_col(), -F::ONE),
        (host_event_chain::gather_memory_read_kind_col(), -F::ONE),
        (host_event_chain::gather_memory_write_kind_col(), -F::ONE),
        (host_event_chain::gather_memory_local_base_col(), F::ONE),
        (host_event_chain::gather_memory_output_base_col(), F::ONE),
        (super::layout::COL_OUTPUT_CAPTURED, F::ONE),
        (host_event_chain::host_call_params_col(), F::ONE),
        (super::layout::COL_HOST_RESUME_POP, F::ONE),
    ]);
    b.with_tag(always("fixed stack arity"), |b| {
        b.push_row(fixed_stack_arity_gate_terms(), fixed_stack_reads_terms(), []);
        b.push_row(fixed_stack_arity_gate_terms(), fixed_stack_writes_terms(), []);
    });
    b.push_linear_zero([
        (COL_STACK_READ_ACTIVE[0], F::ONE),
        (COL_STACK_READ_ACTIVE[1], F::ONE),
        (COL_STACK_READ_ACTIVE[2], F::ONE),
        (COL_STACK_READS, -F::ONE),
    ]);
    b.push_linear_zero([(COL_STACK_WRITE0_ACTIVE, F::ONE), (COL_STACK_WRITES, -F::ONE)]);
    b.push_row(
        [(COL_STACK_READ_ACTIVE[1], F::ONE)],
        [(COL_ONE, F::ONE), (COL_STACK_READ_ACTIVE[0], -F::ONE)],
        [],
    );
    b.push_row(
        [(COL_STACK_READ_ACTIVE[2], F::ONE)],
        [(COL_ONE, F::ONE), (COL_STACK_READ_ACTIVE[1], -F::ONE)],
        [],
    );

    b.with_tag(always("pc edge kind constraints"), |b| {
        b.push_linear_zero(
            [
                (COL_ACTIVATION_FINISHED_AFTER, F::ONE),
                (COL_ACTIVATION_FINISHED_BEFORE, -F::ONE),
                (super::layout::COL_HOST_CLEAR_ACTIVATION_FINISHED, F::ONE),
                (COL_CALL_STACK_POP_PRESENT, F::ONE),
                (COL_PC_EDGE_KIND, -F::ONE),
                (selector_col(WasmOpcode::CallIndirect).unwrap(), F::from_u64(2)),
                (selector_col(WasmOpcode::ReturnCallIndirect).unwrap(), F::from_u64(2)),
                (selector_col(WasmOpcode::Unreachable).unwrap(), F::from_u64(2)),
                (COL_DIV_TRAP, -F::ONE),
                (COL_CALL_INDIRECT_IS_TRAP, -F::ONE),
                (COL_MEM_OOB, -F::ONE),
                (COL_TABLE_OOB, -F::ONE),
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
            selector_col(WasmOpcode::ReturnCallIndirect).unwrap(),
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
        PC_EDGE_KIND_ZERO_TEST.push_constraints(b);
        b.push_row(
            [
                (COL_IS_PROGRAM_ROW, F::ONE),
                (selector_col(WasmOpcode::Call).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::ReturnCall).unwrap(), -F::ONE),
            ],
            [(COL_PC_EDGE_KIND_IS_STATIC, F::ONE)],
            [(COL_PC_ROM_ACTIVE, F::ONE), (crate::layout::COL_PC_FALLTHROUGH, F::ONE)],
        );
    });

    let stack_write0_at_sp_before_ops = opcodes_with_stack_signature(0, 1);
    let stack_read_at_sp_minus1_ops = opcodes_with_stack_reads(1);
    let stack_write0_at_sp_minus1_ops = opcodes_with_stack_signature(1, 1);
    let stack_read_at_sp_minus2_ops = opcodes_with_stack_reads(2);
    let stack_write0_at_sp_minus2_ops = opcodes_with_stack_signature(2, 1);

    b.with_tag(always("local addresses"), |b| {
        b.push_linear_zero([
            (COL_LOCAL_ADDR_LO, F::ONE),
            (COL_LOCALS_FBP_BEFORE, -f_u64(2)),
            (COL_LOCAL_INDEX, -f_u64(2)),
        ]);
        b.push_linear_zero([
            (COL_LOCAL_ADDR_HI, F::ONE),
            (COL_LOCAL_ADDR_LO, -F::ONE),
            (COL_ONE, -F::ONE),
        ]);
    });

    b.with_tag(always("stack high limb addresses"), |b| {
        for (addr_hi, addr_lo) in [
            (COL_STACK_READ_ADDR_HI[0], COL_STACK_READ_ADDR_LO[0]),
            (COL_STACK_READ_ADDR_HI[1], COL_STACK_READ_ADDR_LO[1]),
            (COL_STACK_READ_ADDR_HI[2], COL_STACK_READ_ADDR_LO[2]),
            (COL_STACK_WRITE0_ADDR_HI, COL_STACK_WRITE0_ADDR_LO),
        ] {
            b.push_linear_zero([(addr_hi, F::ONE), (addr_lo, -F::ONE), (COL_ONE, -F::ONE)]);
        }
    });

    b.with_tag(
        shared("stack write0 addr = sp_before", &stack_write0_at_sp_before_ops),
        |b| {
            push_stack_write0_addr_sp_before(b, &stack_write0_at_sp_before_ops);
        },
    );
    b.with_tag(
        shared("stack read0 addr = sp_before - 1", &stack_read_at_sp_minus1_ops),
        |b| {
            push_stack_read0_addr_sp_minus_1(b, &stack_read_at_sp_minus1_ops);
        },
    );
    b.with_tag(
        shared("stack write0 addr = sp_before - 1", &stack_write0_at_sp_minus1_ops),
        |b| {
            push_stack_write0_addr_sp_minus_1(b, &stack_write0_at_sp_minus1_ops);
        },
    );

    b.with_tag(
        shared("stack read0 addr = sp_before - 2", &stack_read_at_sp_minus2_ops),
        |b| {
            push_stack_read0_addr_sp_minus_2(b, &stack_read_at_sp_minus2_ops);
        },
    );
    b.with_tag(
        shared("stack read1 addr = sp_before - 1", &stack_read_at_sp_minus2_ops),
        |b| {
            push_stack_read1_addr_sp_minus_1(b, &stack_read_at_sp_minus2_ops);
        },
    );
    b.with_tag(
        shared("stack write0 addr = sp_before - 2", &stack_write0_at_sp_minus2_ops),
        |b| {
            push_stack_write0_addr_sp_minus_2(b, &stack_write0_at_sp_minus2_ops);
        },
    );

    b.with_tag(opcode_tag("select stack addrs", WasmOpcode::Select), |b| {
        push_select_stack_addrs(b);
    });

    b.with_tag(opcode_tag("i32.add relation", WasmOpcode::I32Add), |b| {
        push_add_relation(b)
    });
    b.with_tag(opcode_tag("i32.sub relation", WasmOpcode::I32Sub), |b| {
        push_sub_relation(b)
    });
    b.with_tag(opcode_tag("i64.add relation", WasmOpcode::I64Add), |b| {
        push_i64_add_relation(b)
    });
    b.with_tag(opcode_tag("i64.sub relation", WasmOpcode::I64Sub), |b| {
        push_i64_sub_relation(b)
    });
    b.with_tag(opcode_tag("i32.wrap_i64 relation", WasmOpcode::I32WrapI64), |b| {
        push_i32_wrap_i64_relation(b)
    });
    b.with_tag(
        opcode_tag("i64.extend_i32_u low relation", WasmOpcode::I64ExtendI32U),
        |b| push_i64_extend_i32_u_low_relation(b),
    );
    b.with_tag(
        opcode_tag("i64.extend_i32_u high zero", WasmOpcode::I64ExtendI32U),
        |b| push_i64_extend_i32_u_high_relation(b),
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
            push_integer_sign_extend_relation(b, &layout.sign_extension, opcode, width_bytes, writes_i64);
        });
    }
    b.with_tag(
        shared(
            "i64 comparator high limb zero",
            &[WasmOpcode::I64Eqz, WasmOpcode::I64Eq, WasmOpcode::I64Ne],
        ),
        |b| {
            push_i64_comparator_high_zero(b);
        },
    );
    b.with_tag(shared("comparator zero-test", COMPARATOR_OPS), |b| {
        push_comparator_constraints(b);
    });
    linear_memory::push_linear_memory_constraints(&mut b, &linear_memory, &layout.sign_extension);
    b.with_tag(opcode_tag("select conditional mux", WasmOpcode::Select), |b| {
        push_select_constraints(b);
    });

    crate::range_check::push_range_check_rows(&mut b);
    b.with_tag(
        shared("conditional control choice", &[WasmOpcode::BrIf, WasmOpcode::If]),
        |b| {
            let selectors = [
                (selector_col(WasmOpcode::BrIf).unwrap(), F::ONE),
                (selector_col(WasmOpcode::If).unwrap(), F::ONE),
            ];
            b.push_row(
                selectors,
                [(COL_ZERO_TEST_LO_INPUT, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
                [],
            );
            b.push_row(
                selectors,
                [
                    (super::layout::COL_CONTROL_CHOICE, F::ONE),
                    (COL_ZERO_TEST_LO_IS_ZERO, F::ONE),
                    (COL_ONE, -F::ONE),
                ],
                [],
            );
        },
    );
    b.with_tag(opcode_tag("br_table control choice", WasmOpcode::BrTable), |b| {
        use super::layout::{COL_BR_TABLE_IN_BOUNDS, COL_BR_TABLE_LEN, COL_CMP_GE, COL_CMP_LOW, COL_CONTROL_CHOICE};
        let selector = selector_col(WasmOpcode::BrTable).unwrap();
        super::gadgets::push_unsigned_ge_gadget(
            b,
            [selector],
            [(COL_STACK_READ_VALUE_LO[0], F::ONE)],
            [(COL_BR_TABLE_LEN, F::ONE)],
            COL_CMP_LOW,
            COL_CMP_GE,
        );
        b.push_row(
            [(selector, F::ONE)],
            [(COL_ONE, F::ONE), (COL_CMP_GE, -F::ONE)],
            [(COL_BR_TABLE_IN_BOUNDS, F::ONE)],
        );
        b.push_row(
            [(COL_BR_TABLE_IN_BOUNDS, F::ONE)],
            [
                (COL_CONTROL_CHOICE, F::ONE),
                (COL_STACK_READ_VALUE_LO[0], -F::ONE),
                (COL_ONE, -F::ONE),
            ],
            [],
        );
        b.push_row(
            [(selector, F::ONE), (COL_BR_TABLE_IN_BOUNDS, -F::ONE)],
            [(COL_CONTROL_CHOICE, F::ONE)],
            [],
        );
    });
    b.with_tag(always("control choice outside branches"), |b| {
        b.push_row(
            [
                (COL_ONE, F::ONE),
                (selector_col(WasmOpcode::BrIf).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::If).unwrap(), -F::ONE),
                (selector_col(WasmOpcode::BrTable).unwrap(), -F::ONE),
            ],
            [(super::layout::COL_CONTROL_CHOICE, F::ONE)],
            [],
        );
    });
    b.with_tag(always("edge key"), |b| {
        b.push_linear_zero([
            (super::layout::COL_EDGE_ADDR, F::ONE),
            (super::layout::COL_PC_BEFORE, -F::ONE),
            (super::layout::COL_CONTROL_CHOICE, -F::ONE),
        ]);
    });
    b.with_tag(always("stack read-modify-write gates"), |b| {
        let rmw = super::layout::COL_STACK_RMW;
        b.push_linear_zero(
            std::iter::once((rmw, F::ONE)).chain(
                WasmOpcode::supported()
                    .into_iter()
                    .filter(|op| op.uses_stack_rmw())
                    .map(|op| (selector_col(op).unwrap(), -F::ONE)),
            ),
        );
        for (separate, original) in super::layout::COL_STACK_SEPARATE.into_iter().zip([
            COL_STACK_READ_ACTIVE[0],
            COL_STACK_WRITE0_ACTIVE,
            crate::layout::COL_STACK_WRITE0_HI_ACTIVE,
        ]) {
            b.push_linear_zero([(separate, F::ONE), (original, -F::ONE), (rmw, F::ONE)]);
        }
        push_gated_linear_zero(
            b,
            rmw,
            [(COL_STACK_READ_ADDR_LO[0], F::ONE), (COL_STACK_WRITE0_ADDR_LO, -F::ONE)],
        );
    });
    b.with_tag(always("stack read0 high gate"), |b| {
        b.push_linear_zero(
            [
                (super::layout::COL_STACK_READ0_HI_ACTIVE, F::ONE),
                (super::layout::COL_STACK_SEPARATE[0], -F::ONE),
            ]
            .into_iter()
            .chain(
                WasmOpcode::supported()
                    .into_iter()
                    .filter(|op| op.omits_stack_read0_hi())
                    .map(|op| (selector_col(op).unwrap(), F::ONE)),
            ),
        );
    });
    let columns = crate::witness_layout::range_checked_column_registry();
    let r1cs = builder.build(&columns).map_err(|error| error.to_string())?;
    ApplicationRelation::new(r1cs, columns).map_err(|error| error.to_string())
}

fn f_u16(v: u16) -> F {
    F::from_u64(u64::from(v))
}

pub(super) fn f_u64(v: u64) -> F {
    F::from_u64(v)
}

fn push_stack_write0_addr_sp_before(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp selector"), F::ONE)),
        [(COL_STACK_WRITE0_ADDR_LO, F::ONE), (COL_SP_BEFORE, -f_u64(2))],
        [],
    );
}

fn push_stack_read0_addr_sp_minus_1(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read0 sp-1 selector"), F::ONE)),
        [
            (COL_STACK_READ_ADDR_LO[0], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(2)),
        ],
        [],
    );
}

fn push_stack_write0_addr_sp_minus_1(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp-1 selector"), F::ONE)),
        [
            (COL_STACK_WRITE0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(2)),
        ],
        [],
    );
}

fn push_select_stack_addrs(b: &mut WasmTaggedR1csBuilder<'_>) {
    let selector = selector_col(WasmOpcode::Select).unwrap();
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ_ADDR_LO[0], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(6)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ_ADDR_LO[1], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(4)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_READ_ADDR_LO[2], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(2)),
        ],
    );
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_WRITE0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(6)),
        ],
    );
}

fn push_select_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    SELECT_COND_ZERO_TEST.push_constraints(b);
    SELECT_LO_MUX.push_constraints_assuming_boolean_inputs(b);
    SELECT_HI_MUX.push_constraints_assuming_boolean_inputs(b);
}

fn push_stack_read0_addr_sp_minus_2(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read0 sp-2 selector"), F::ONE)),
        [
            (COL_STACK_READ_ADDR_LO[0], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(4)),
        ],
        [],
    );
}

fn push_stack_read1_addr_sp_minus_1(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack read1 sp-1 selector"), F::ONE)),
        [
            (COL_STACK_READ_ADDR_LO[1], F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(2)),
        ],
        [],
    );
}

fn push_stack_write0_addr_sp_minus_2(b: &mut WasmTaggedR1csBuilder<'_>, ops: &[WasmOpcode]) {
    b.push_row(
        ops.iter()
            .map(|&op| (selector_col(op).expect("stack write0 sp-2 selector"), F::ONE)),
        [
            (COL_STACK_WRITE0_ADDR_LO, F::ONE),
            (COL_SP_BEFORE, -f_u64(2)),
            (COL_ONE, f_u64(4)),
        ],
        [],
    );
}

fn push_add_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Add).unwrap(),
        [
            (COL_STACK_READ_VALUE_LO[0], F::ONE),
            (COL_STACK_READ_VALUE_LO[1], F::ONE),
            (COL_STACK_WRITE0_VALUE_LO, -F::ONE),
            (COL_WIDE_AUX0, -f_u64(1_u64 << 32)),
        ],
    );
}

fn push_sub_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I32Sub).unwrap(),
        [
            (COL_STACK_READ_VALUE_LO[0], F::ONE),
            (COL_STACK_READ_VALUE_LO[1], -F::ONE),
            (COL_STACK_WRITE0_VALUE_LO, -F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
        ],
    );
}

fn push_i64_add_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    let selector = selector_col(WasmOpcode::I64Add).unwrap();
    b.push_row(
        [
            (COL_STACK_READ_VALUE_LO[0], F::ONE),
            (COL_STACK_READ_VALUE_LO[1], F::ONE),
            (COL_STACK_WRITE0_VALUE_LO, -F::ONE),
            (COL_WIDE_AUX0, -f_u64(1_u64 << 32)),
        ],
        [(selector, F::ONE)],
        [],
    );
    b.push_row(
        [
            (COL_STACK_READ_VALUE_HI[0], F::ONE),
            (COL_STACK_READ_VALUE_HI[1], F::ONE),
            (COL_STACK_WRITE0_VALUE_HI, -F::ONE),
            (COL_WIDE_AUX0, F::ONE),
            (COL_WIDE_AUX1, -f_u64(1_u64 << 32)),
        ],
        [(selector, F::ONE)],
        [],
    );
}

fn push_i64_sub_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    let selector = selector_col(WasmOpcode::I64Sub).unwrap();
    b.push_row(
        [
            (COL_STACK_READ_VALUE_LO[0], F::ONE),
            (COL_WIDE_AUX0, f_u64(1_u64 << 32)),
            (COL_STACK_WRITE0_VALUE_LO, -F::ONE),
            (COL_STACK_READ_VALUE_LO[1], -F::ONE),
        ],
        [(selector, F::ONE)],
        [],
    );
    b.push_row(
        [
            (COL_STACK_READ_VALUE_HI[0], F::ONE),
            (COL_WIDE_AUX1, f_u64(1_u64 << 32)),
            (COL_STACK_WRITE0_VALUE_HI, -F::ONE),
            (COL_STACK_READ_VALUE_HI[1], -F::ONE),
            (COL_WIDE_AUX0, -F::ONE),
        ],
        [(selector, F::ONE)],
        [],
    );
}

fn push_i32_wrap_i64_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    let selector = selector_col(WasmOpcode::I32WrapI64).unwrap();
    push_gated_linear_zero(
        b,
        selector,
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_STACK_READ_VALUE_LO[0], -F::ONE),
        ],
    );
    push_gated_linear_zero(b, selector, [(COL_STACK_WRITE0_VALUE_HI, F::ONE)]);
}

fn push_i64_extend_i32_u_low_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.push_row(
        [(selector_col(WasmOpcode::I64ExtendI32U).unwrap(), F::ONE)],
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_STACK_READ_VALUE_LO[0], -F::ONE),
        ],
        [],
    );
}

fn push_i64_extend_i32_u_high_relation(b: &mut WasmTaggedR1csBuilder<'_>) {
    push_gated_linear_zero(
        b,
        selector_col(WasmOpcode::I64ExtendI32U).unwrap(),
        [(COL_STACK_WRITE0_VALUE_HI, F::ONE)],
    );
}

fn push_integer_sign_extend_relation(
    b: &mut WasmTaggedR1csBuilder<'_>,
    sign_extension: &SignExtensionColumns,
    opcode: WasmOpcode,
    width_bytes: usize,
    writes_i64: bool,
) {
    debug_assert!((1..=4).contains(&width_bytes));
    let selector = selector_col(opcode).unwrap();
    push_u32_le_bytes_decomp(b, [selector], COL_STACK_READ_VALUE_LO[0], sign_extension.bytes.map(idx));

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
        std::iter::once((COL_STACK_WRITE0_VALUE_LO, F::ONE))
            .chain(retained_bytes)
            .chain(std::iter::once((idx(sign_extension.bit), -f_u64(u64::from(sign_fill))))),
        [],
    );

    if writes_i64 {
        b.push_row(
            [(selector, F::ONE)],
            [
                (COL_STACK_WRITE0_VALUE_HI, F::ONE),
                (idx(sign_extension.bit), -f_u64(0xffff_ffff)),
            ],
            [],
        );
    }
}

fn push_i64_comparator_high_zero(b: &mut WasmTaggedR1csBuilder<'_>) {
    b.push_row(
        [
            (selector_col(WasmOpcode::I64Eqz).unwrap(), F::ONE),
            (selector_col(WasmOpcode::I64Eq).unwrap(), F::ONE),
            (selector_col(WasmOpcode::I64Ne).unwrap(), F::ONE),
        ],
        [(COL_STACK_WRITE0_VALUE_HI, F::ONE)],
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

/// The modulus `q = 2^64 - 59` does not have an injective u64 → field-element
/// embedding for `lo + hi*2^32`: `(lo=0xffffffc5, hi=0xffffffff)` is `q ≡ 0`.
fn push_comparator_constraints(b: &mut WasmTaggedR1csBuilder<'_>) {
    let sel_eqz_i32 = selector_col(WasmOpcode::I32Eqz).unwrap();
    let sel_eqz_i64 = selector_col(WasmOpcode::I64Eqz).unwrap();
    let sel_eq = selector_col(WasmOpcode::I32Eq).unwrap();
    let sel_ne = selector_col(WasmOpcode::I32Ne).unwrap();
    let sel_i64_eq = selector_col(WasmOpcode::I64Eq).unwrap();
    let sel_i64_ne = selector_col(WasmOpcode::I64Ne).unwrap();

    push_gated_linear_zero(
        b,
        sel_eqz_i32,
        [(COL_ZERO_TEST_LO_INPUT, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_ZERO_TEST_LO_INPUT, F::ONE), (COL_STACK_READ_VALUE_LO[0], -F::ONE)],
    );
    b.push_row(
        [
            (sel_eq, F::ONE),
            (sel_ne, F::ONE),
            (sel_i64_eq, F::ONE),
            (sel_i64_ne, F::ONE),
        ],
        [
            (COL_ZERO_TEST_LO_INPUT, F::ONE),
            (COL_STACK_READ_VALUE_LO[0], -F::ONE),
            (COL_STACK_READ_VALUE_LO[1], F::ONE),
        ],
        [],
    );

    LO_ZERO_TEST.push_constraints(b);

    push_gated_linear_zero(
        b,
        sel_eqz_i64,
        [(COL_ZERO_TEST_HI_INPUT, F::ONE), (COL_STACK_READ_VALUE_HI[0], -F::ONE)],
    );
    b.push_row(
        [(sel_i64_eq, F::ONE), (sel_i64_ne, F::ONE)],
        [
            (COL_ZERO_TEST_HI_INPUT, F::ONE),
            (COL_STACK_READ_VALUE_HI[0], -F::ONE),
            (COL_STACK_READ_VALUE_HI[1], F::ONE),
        ],
        [],
    );

    HI_ZERO_TEST.push_constraints(b);

    b.push_row(
        [(COL_ZERO_TEST_LO_IS_ZERO, F::ONE)],
        [(COL_ZERO_TEST_HI_IS_ZERO, F::ONE)],
        [(COL_CMP_AND, F::ONE)],
    );

    b.push_row(
        [(sel_eqz_i32, F::ONE), (sel_eq, F::ONE)],
        [(COL_STACK_WRITE0_VALUE_LO, F::ONE), (COL_ZERO_TEST_LO_IS_ZERO, -F::ONE)],
        [],
    );
    b.push_row(
        [(sel_eqz_i64, F::ONE), (sel_i64_eq, F::ONE)],
        [(COL_STACK_WRITE0_VALUE_LO, F::ONE), (COL_CMP_AND, -F::ONE)],
        [],
    );
    push_gated_linear_zero(
        b,
        sel_ne,
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_ZERO_TEST_LO_IS_ZERO, F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
    push_gated_linear_zero(
        b,
        sel_i64_ne,
        [
            (COL_STACK_WRITE0_VALUE_LO, F::ONE),
            (COL_CMP_AND, F::ONE),
            (COL_ONE, -F::ONE),
        ],
    );
}

pub(super) fn idx(column: Column) -> usize {
    column.0
}
