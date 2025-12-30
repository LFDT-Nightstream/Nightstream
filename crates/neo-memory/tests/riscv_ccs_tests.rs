//! Tests for the RV32 B1 shared-bus step CCS.

use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_ccs::traits::SModuleHomomorphism;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{build_rv32_b1_step_ccs, rv32_b1_chunk_to_witness, rv32_b1_shared_cpu_bus_config};
use neo_memory::riscv::lookups::{
    decode_instruction, encode_program, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::rom_init::prog_init_words;
use neo_memory::witness::LutTableSpec;
use neo_memory::{CpuArithmetization, R1csCpu};
use neo_params::NeoParams;
use neo_vm_trace::{trace_program, StepTrace, TwistEvent, TwistOpKind, VmTrace};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[derive(Clone, Copy, Default)]
struct NoopCommit;

impl SModuleHomomorphism<F, ()> for NoopCommit {
    fn commit(&self, _z: &Mat<F>) -> () {}

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn pow2_ceil_k(min_k: usize) -> (usize, usize) {
    let k = min_k.next_power_of_two().max(2);
    let d = k.trailing_zeros() as usize;
    (k, d)
}

const RV32I_SHOUT_TABLE_IDS: [u32; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

fn rv32i_table_specs(xlen: usize) -> HashMap<u32, LutTableSpec> {
    HashMap::from([
        (
            0u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::And,
                xlen,
            },
        ),
        (
            1u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Xor,
                xlen,
            },
        ),
        (
            2u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Or,
                xlen,
            },
        ),
        (
            3u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            4u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sub,
                xlen,
            },
        ),
        (
            5u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Slt,
                xlen,
            },
        ),
        (
            6u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sltu,
                xlen,
            },
        ),
        (
            7u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sll,
                xlen,
            },
        ),
        (
            8u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Srl,
                xlen,
            },
        ),
        (
            9u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Sra,
                xlen,
            },
        ),
        (
            10u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Eq,
                xlen,
            },
        ),
        (
            11u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Neq,
                xlen,
            },
        ),
    ])
}

#[test]
fn rv32_b1_ccs_happy_path_small_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 1 }, // x1 = 0x1000
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        }, // x1 = 0x1005
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // x2 = 7
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 0x100c
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0x100,
        }, // mem[0x100] = x3
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 4,
            rs1: 0,
            imm: 0x100,
        }, // x4 = mem[0x100]
        RiscvInstruction::Auipc { rd: 5, imm: 0 }, // x5 = pc
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    // mem_layouts: keep k small to reduce bus tail width.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200); // covers addresses up to 0x1ff
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32m_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -6,
        }, // x1 = -6
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x40);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    // Minimal table set for this program: ADD + RV32M (MUL/DIV/REM families).
    let shout_table_ids: [u32; 9] = [3, 12, 13, 14, 15, 16, 17, 18, 19];
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = HashMap::from([
        (
            3u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Add,
                xlen,
            },
        ),
        (
            12u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Mul,
                xlen,
            },
        ),
        (
            13u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Mulh,
                xlen,
            },
        ),
        (
            14u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Mulhu,
                xlen,
            },
        ),
        (
            15u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Mulhsu,
                xlen,
            },
        ),
        (
            16u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Div,
                xlen,
            },
        ),
        (
            17u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Divu,
                xlen,
            },
        ),
        (
            18u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Rem,
                xlen,
            },
        ),
        (
            19u32,
            LutTableSpec::RiscvOpcode {
                opcode: RiscvOpcode::Remu,
                xlen,
            },
        ),
    ]);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_happy_path_rv32a_amoaddw_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // x1 = 0x100
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        }, // x2 = 10
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 10
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        }, // x3 = 5
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        }, // x4 = old, mem = old + 5
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 5,
            rs1: 1,
            imm: 0,
        }, // x5 = new
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rejects_tampered_ram_write_value_for_amoaddw() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 10,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let amo_step_idx = 4usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(amo_step_idx);

    let ram_wv_w_idx = layout
        .ram_wv
        .checked_sub(layout.m_in)
        .expect("ram_wv must be in private witness");
    mcs_wit.w[ram_wv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered RAM write value should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_happy_path_rv32a_word_amos_program() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        }, // x1 = 0x100
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0b1111,
        }, // x2 = 15
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 1,
            rs2: 2,
            imm: 0,
        }, // mem[0x100] = 15
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 0b1010,
        }, // x3 = 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoandW,
            rd: 4,
            rs1: 1,
            rs2: 3,
        }, // mem &= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoorW,
            rd: 5,
            rs1: 1,
            rs2: 3,
        }, // mem |= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoxorW,
            rd: 6,
            rs1: 1,
            rs2: 3,
        }, // mem ^= 10
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoswapW,
            rd: 7,
            rs1: 1,
            rs2: 2,
        }, // mem = 15
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 8,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_chunk_size_2_padding_carries_state() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    for (mcs_inst, mcs_wit) in chunks {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_branches_and_jal() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        }, // x2 = 1
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 99,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 5,
        }, // x3 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 77,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 7,
        }, // x4 = 7
        RiscvInstruction::Jal { rd: 5, imm: 8 }, // x5=pc+4, jump over next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 123,
        }, // skipped
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);

    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rv32i_alu_ops() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        }, // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 - x2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::And,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 5,
            rs1: 1,
            imm: 0x0f,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 6,
            rs1: 1,
            imm: 0x0f,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sll,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 11,
            rs1: 2,
            rs2: 1,
        }, // (3 < 5) => 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 12,
            rs1: 2,
            imm: 4,
        }, // (3 < 4) => 1
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_branches_blt_bge_bltu_bgeu() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::Branch {
            cond: BranchCondition::Lt,
            rs1: 1,
            rs2: 2,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 111,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 0,
            imm: 3,
        }, // x3 = 3
        RiscvInstruction::Branch {
            cond: BranchCondition::Ge,
            rs1: 2,
            rs2: 1,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 222,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 4,
        }, // x4 = 4
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: -1,
        }, // x5 = 0xffff_ffff
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 1,
        }, // x6 = 1
        RiscvInstruction::Branch {
            cond: BranchCondition::Ltu,
            rs1: 5,
            rs2: 6,
            imm: 8,
        }, // not taken -> execute next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 7,
            rs1: 0,
            imm: 7,
        }, // x7 = 7
        RiscvInstruction::Branch {
            cond: BranchCondition::Geu,
            rs1: 5,
            rs2: 6,
            imm: 8,
        }, // taken -> skip next instruction
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 8,
            rs1: 0,
            imm: 888,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 8,
            rs1: 0,
            imm: 8,
        }, // x8 = 8
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 128).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_jalr_masks_lsb() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 13,
        }, // x1 = 13
        RiscvInstruction::Jalr { rd: 0, rs1: 1, imm: 0 }, // pc = (13 + 0) & !1 = 12
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // skipped
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 9,
        }, // executed
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}

#[test]
fn rv32_b1_ccs_rejects_step_after_halt_within_chunk() {
    let xlen = 32usize;
    let program = vec![RiscvInstruction::Halt, RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let w0 = u32::from_le_bytes(program_bytes[0..4].try_into().expect("word0"));
    let w1 = u32::from_le_bytes(program_bytes[4..8].try_into().expect("word1"));

    let regs = vec![0u64; 32];
    let steps: Vec<StepTrace<u64, u64>> = vec![
        StepTrace {
            cycle: 0,
            pc_before: 0,
            pc_after: 4,
            opcode: w0,
            regs_before: regs.clone(),
            regs_after: regs.clone(),
            twist_events: vec![TwistEvent {
                twist_id: PROG_ID,
                kind: TwistOpKind::Read,
                addr: 0,
                value: w0 as u64,
            }],
            shout_events: Vec::new(),
            halted: true,
        },
        StepTrace {
            cycle: 1,
            pc_before: 4,
            pc_after: 8,
            opcode: w1,
            regs_before: regs.clone(),
            regs_after: regs.clone(),
            twist_events: vec![TwistEvent {
                twist_id: PROG_ID,
                kind: TwistOpKind::Read,
                addr: 4,
                value: w1 as u64,
            }],
            shout_events: Vec::new(),
            halted: true,
        },
    ];
    let trace = VmTrace { steps };

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("chunks");
    assert_eq!(chunks.len(), 1, "expected single chunk");
    let (mcs_inst, mcs_wit) = chunks.pop().expect("chunk");
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "step after HALT should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_pc_out() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let pc_out_w_idx = layout
        .pc_out
        .checked_sub(layout.m_in)
        .expect("pc_out must be in private witness");
    mcs_wit.w[pc_out_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered witness should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_non_boolean_prog_read_addr_bit() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx];
    let bit_col_id = prog_cols.ra_bits.start + 2; // keep alignment bits [0,1] untouched
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("prog bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    let new_bit = F::from_u64(2);
    mcs_wit.w[bit_w_idx] = new_bit;

    let delta = new_bit - old_bit;
    let pc_in_w_idx = layout
        .pc_in
        .checked_sub(layout.m_in)
        .expect("pc_in must be in private witness");
    let pc_out_w_idx = layout
        .pc_out
        .checked_sub(layout.m_in)
        .expect("pc_out must be in private witness");
    mcs_wit.w[pc_in_w_idx] += delta * F::from_u64(1 << 2);
    mcs_wit.w[pc_out_w_idx] += delta * F::from_u64(1 << 2);

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "non-boolean prog addr bit should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_non_boolean_shout_addr_bit() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (Shout ADD active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let instr = decode_instruction(trace.steps[add_step_idx].opcode).expect("decode");
    let (rs1_idx, rs2_idx) = match instr {
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rs1,
            rs2,
            ..
        } => (rs1 as usize, rs2 as usize),
        other => panic!("expected ADD at step {add_step_idx}, got {other:?}"),
    };

    // Flip one ADD shout key bit to a non-boolean value, and adjust CPU columns so that
    // all *linear* bindings still hold. Bitness constraints should still reject.
    let add_shout_idx = layout.shout_idx(3).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx];
    let bit_col_id = shout_cols.addr_bits.start + 0; // key bit 0 (rs1 bit 0)
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    let new_bit = F::from_u64(2);
    mcs_wit.w[bit_w_idx] = new_bit;
    let delta = new_bit - old_bit;

    // Update packed key (lookup_key) to match the mutated bits.
    let lookup_key_w_idx = layout
        .lookup_key(0)
        .checked_sub(layout.m_in)
        .expect("lookup_key must be in private witness");
    mcs_wit.w[lookup_key_w_idx] += delta;

    // Update rs1_val and corresponding register snapshots to match the mutated even-bit packing.
    let rs1_val_w_idx = layout
        .rs1_val(0)
        .checked_sub(layout.m_in)
        .expect("rs1_val must be in private witness");
    mcs_wit.w[rs1_val_w_idx] += delta;

    let reg1_in_w_idx = layout
        .reg_in(rs1_idx, 0)
        .checked_sub(layout.m_in)
        .expect("reg_in must be in private witness");
    let reg1_out_w_idx = layout
        .reg_out(rs1_idx, 0)
        .checked_sub(layout.m_in)
        .expect("reg_out must be in private witness");
    mcs_wit.w[reg1_in_w_idx] += delta;
    mcs_wit.w[reg1_out_w_idx] += delta;

    // Keep rs2 snapshots consistent as well (defensive sanity); no bit change expected.
    let reg2_in_w_idx = layout
        .reg_in(rs2_idx, 0)
        .checked_sub(layout.m_in)
        .expect("reg_in must be in private witness");
    let reg2_out_w_idx = layout
        .reg_out(rs2_idx, 0)
        .checked_sub(layout.m_in)
        .expect("reg_out must be in private witness");
    mcs_wit.w[reg2_out_w_idx] = mcs_wit.w[reg2_in_w_idx];

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "non-boolean shout addr bit should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_rom_value_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx];
    let rv_z = layout.bus.bus_cell(prog_cols.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("rv in witness");
    mcs_wit.w[rv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "rom value mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_regfile() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    // Tamper with a non-rd register output (x2) without updating reg_in.
    let r = 2usize;
    let reg_out_w_idx = layout
        .reg_out(r, 0)
        .checked_sub(layout.m_in)
        .expect("reg_out in witness");
    mcs_wit.w[reg_out_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered regfile should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_tampered_x0() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let x0_out_w_idx = layout
        .reg_out(0, 0)
        .checked_sub(layout.m_in)
        .expect("x0 out in witness");
    mcs_wit.w[x0_out_w_idx] = F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "tampered x0 should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_binds_public_initial_and_final_state() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        }, // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 7,
        }, // x2 = 7
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 8usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    assert_eq!(chunks.len(), 1, "chunk_size>N should create one chunk");
    let (mcs_inst, mcs_wit) = chunks.remove(0);

    check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");

    let first = trace.steps.first().expect("trace non-empty");
    assert_eq!(mcs_inst.x[layout.pc0], F::from_u64(first.pc_before));
    for r in 0..32 {
        assert_eq!(
            mcs_inst.x[layout.regs0_start + r],
            F::from_u64(first.regs_before[r])
        );
    }
    assert_eq!(mcs_inst.x[layout.regs0_start], F::ZERO);

    let last = trace.steps.last().expect("trace non-empty");
    assert_eq!(mcs_inst.x[layout.pc_final], F::from_u64(last.pc_after));
    for r in 0..32 {
        assert_eq!(
            mcs_inst.x[layout.regs_final_start + r],
            F::from_u64(last.regs_after[r])
        );
    }
    assert_eq!(mcs_inst.x[layout.regs_final_start], F::ZERO);

    let mut x_bad = mcs_inst.x.clone();
    x_bad[layout.pc0] += F::ONE;
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &x_bad, &mcs_wit.w).is_err(),
        "tampered pc0 should not satisfy CCS"
    );

    let mut x_bad = mcs_inst.x.clone();
    x_bad[layout.pc_final] += F::ONE;
    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &x_bad, &mcs_wit.w).is_err(),
        "tampered pc_final should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_rom_addr_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let prog_cols = &layout.bus.twist_cols[layout.prog_twist_idx];
    let bit_col_id = prog_cols.ra_bits.start + 2; // keep alignment bits [0,1] untouched
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("prog bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "rom address mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_decode_bit_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 8).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let (mcs_inst, mut mcs_wit) = steps.remove(0);

    let bit_z = layout.instr_bit(0, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("instr_bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "decode bit mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_shout_key_bit_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (Shout ADD active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let add_shout_idx = layout.shout_idx(3).expect("ADD shout idx");
    let shout_cols = &layout.bus.shout_cols[add_shout_idx];
    let bit_col_id = shout_cols.addr_bits.start + 0; // rs1 bit 0
    let bit_z = layout.bus.bus_cell(bit_col_id, 0);
    let bit_w_idx = bit_z.checked_sub(layout.m_in).expect("bit in witness");
    let old_bit = mcs_wit.w[bit_w_idx];
    mcs_wit.w[bit_w_idx] = F::ONE - old_bit;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "shout key mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_wrong_shout_table_activation() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        }, // x1 = 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        }, // x2 = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2 (ADD table active)
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let add_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(add_step_idx);

    let eq_shout_idx = layout.shout_idx(10).expect("EQ shout idx");
    let eq_cols = &layout.bus.shout_cols[eq_shout_idx];
    let has_lookup_z = layout.bus.bus_cell(eq_cols.has_lookup, 0);
    let has_lookup_w_idx = has_lookup_z
        .checked_sub(layout.m_in)
        .expect("has_lookup in witness");
    mcs_wit.w[has_lookup_w_idx] = F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "wrong shout table activation should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_ram_read_value_mismatch() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 9,
        }, // x1 = 9
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0x100,
        }, // mem[0x100] = x1
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0x100,
        }, // x2 = mem[0x100]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 64).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, 1).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        1,
    )
    .expect("shared bus");

    let mut steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    let lw_step_idx = 2usize;
    let (mcs_inst, mut mcs_wit) = steps.remove(lw_step_idx);

    let ram_cols = &layout.bus.twist_cols[layout.ram_twist_idx];
    let rv_z = layout.bus.bus_cell(ram_cols.rv, 0);
    let rv_w_idx = rv_z.checked_sub(layout.m_in).expect("rv in witness");
    mcs_wit.w[rv_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "ram read value mismatch should not satisfy CCS"
    );
}

#[test]
fn rv32_b1_ccs_rejects_chunk_size_2_continuity_break() {
    let xlen = 32usize;
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut cpu_vm = RiscvCpu::new(xlen);
    cpu_vm.load_program(0, program.clone());
    let memory = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);
    let trace = trace_program(cpu_vm, memory, shout, 16).expect("trace");
    assert!(trace.did_halt(), "expected Halt");

    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x80);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(PROG_ID, 0, &program_bytes);

    let shout_table_ids = RV32I_SHOUT_TABLE_IDS;
    let chunk_size = 2usize;
    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts, &shout_table_ids, chunk_size).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let table_specs = rv32i_table_specs(xlen);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_chunk_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(
        rv32_b1_shared_cpu_bus_config(&layout, &shout_table_ids, mem_layouts, initial_mem).expect("cfg"),
        chunk_size,
    )
    .expect("shared bus");

    let mut chunks = CpuArithmetization::build_ccs_chunks(&cpu, &trace, chunk_size).expect("build chunks");
    let (mcs_inst, mut mcs_wit) = chunks.remove(0);

    let pc_in_w_idx = layout
        .pc_in(1)
        .checked_sub(layout.m_in)
        .expect("pc_in lane 1 in witness");
    mcs_wit.w[pc_in_w_idx] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).is_err(),
        "continuity break should not satisfy CCS"
    );
}
