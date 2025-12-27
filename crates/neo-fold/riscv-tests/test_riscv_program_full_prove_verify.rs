//! End-to-end prove+verify for a small RV32 program under the B1 shared-bus step circuit.
//!
//! This exercises:
//! - B1 instruction fetch via `PROG_ID` Twist reads
//! - shared CPU bus tail wiring (Twist + Shout)
//! - implicit Shout table spec (`LutTableSpec::RiscvOpcode`)
//! - the RV32 B1 step CCS glue constraints

#![allow(non_snake_case)]

use std::collections::HashMap;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::matrix::Mat;
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, CommitMixers};
use neo_math::{F, K};
use neo_memory::builder::build_shard_witness_shared_cpu_bus;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{build_rv32_b1_step_ccs, rv32_b1_shared_cpu_bus_config, rv32_b1_step_to_witness};
use neo_memory::riscv::lookups::{
    encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::witness::{LutTableSpec, StepInstanceBundle};
use neo_memory::R1csCpu;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

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

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(neo_math::D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(neo_math::D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
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
fn test_riscv_program_full_prove_verify() {
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
    let mut vm = RiscvCpu::new(xlen);
    vm.load_program(0, program);
    let twist = RiscvMemory::with_program_in_twist(xlen, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(xlen);

    // Keep k small to reduce bus tail width and proof work.
    let (k_prog, d_prog) = pow2_ceil_k(program_bytes.len());
    let (k_ram, d_ram) = pow2_ceil_k(0x200);
    let mem_layouts = HashMap::from([
        (0u32, PlainMemLayout { k: k_ram, d: d_ram, n_side: 2 }),
        (1u32, PlainMemLayout { k: k_prog, d: d_prog, n_side: 2 }),
    ]);
    let initial_mem = prog_init_words(0, &program_bytes);

    // Build CCS + shared-bus CPU arithmetization.
    let (ccs_base, layout) = build_rv32_b1_step_ccs(&mem_layouts).expect("ccs");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs_base.n).expect("params");

    let table_specs = HashMap::from([(
        3u32,
        LutTableSpec::RiscvOpcode {
            opcode: RiscvOpcode::Add,
            xlen,
        },
    )]);

    let cpu = R1csCpu::new(
        ccs_base,
        params.clone(),
        DummyCommit::default(),
        layout.m_in,
        &HashMap::new(),
        &table_specs,
        rv32_b1_step_to_witness(layout.clone()),
    )
    .with_shared_cpu_bus(rv32_b1_shared_cpu_bus_config(&layout, mem_layouts.clone(), initial_mem.clone()))
    .expect("shared bus inject");

    // Build shared-bus step bundles (includes CPU MCS + metadata-only mem/lut instances).
    let lut_tables = HashMap::new();
    let steps = build_shard_witness_shared_cpu_bus::<_, Cmt, K, _, _, _>(
        vm,
        twist,
        shout,
        /*max_steps=*/ 64,
        /*chunk_size=*/ 1,
        &mem_layouts,
        &lut_tables,
        &table_specs,
        &initial_mem,
        &cpu,
    )
    .expect("build shard witness");

    let steps_public: Vec<StepInstanceBundle<Cmt, F, K>> = steps.iter().map(StepInstanceBundle::from).collect();

    let mixers = default_mixers();
    let mut tr_prove = Poseidon2Transcript::new(b"riscv-b1-full");
    // PaperExact is intentionally slow (brute-force oracle) and can make this end-to-end
    // test take minutes. Use the optimized engine here and keep PaperExact covered by
    // smaller unit tests.
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &cpu.ccs,
        &steps,
        &[],
        &[],
        &DummyCommit::default(),
        mixers,
    )
    .expect("prove");

    let mut tr_verify = Poseidon2Transcript::new(b"riscv-b1-full");
    let _ = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &cpu.ccs,
        &steps_public,
        &[],
        &proof,
        mixers,
    )
    .expect("verify");
}
