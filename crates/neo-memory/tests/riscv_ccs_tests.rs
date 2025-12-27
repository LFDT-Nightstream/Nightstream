//! Tests for the RV32 B1 shared-bus step CCS.

use std::collections::HashMap;

use neo_ccs::matrix::Mat;
use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_ccs::traits::SModuleHomomorphism;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{build_rv32_b1_step_ccs, rv32_b1_shared_cpu_bus_config, rv32_b1_step_to_witness};
use neo_memory::riscv::lookups::{encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID};
use neo_memory::witness::LutTableSpec;
use neo_memory::{CpuArithmetization, R1csCpu};
use neo_params::NeoParams;
use neo_vm_trace::trace_program;
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

fn prog_init_words(base: u64, program_bytes: &[u8]) -> HashMap<(u32, u64), F> {
    assert_eq!(program_bytes.len() % 4, 0, "program must be 4-byte aligned");
    let mut out = HashMap::new();
    for (i, chunk) in program_bytes.chunks_exact(4).enumerate() {
        let addr = base + (i as u64) * 4;
        let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64;
        if w != 0 {
            out.insert((PROG_ID.0, addr), F::from_u64(w));
        }
    }
    out
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

    let initial_mem = prog_init_words(0, &program_bytes);

    let (ccs, layout) = build_rv32_b1_step_ccs(&mem_layouts).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");

    let table_specs = HashMap::from([(
        3u32,
        LutTableSpec::RiscvOpcode {
            opcode: RiscvOpcode::Add,
            xlen,
        },
    )]);

    let cpu = R1csCpu::new(
        ccs,
        params,
        NoopCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_step_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(rv32_b1_shared_cpu_bus_config(&layout, mem_layouts, initial_mem))
    .expect("shared bus");

    let steps = CpuArithmetization::build_ccs_steps(&cpu, &trace).expect("build steps");
    for (mcs_inst, mcs_wit) in steps {
        check_ccs_rowwise_zero(&cpu.ccs, &mcs_inst.x, &mcs_wit.w).expect("CCS satisfied");
    }
}
