//! End-to-end proving + verification for a small RISC-V program.
//!
//! This test exercises:
//! - `neo_vm_trace::trace_program` to execute a RISC-V program
//! - `neo_memory::riscv::ccs` to build a CCS + per-step witnesses
//! - `neo_fold::session::FoldingSession` to prove and verify the execution trace

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, ProveInput};
use neo_memory::riscv::ccs::{
    build_riscv_alu_step_ccs, check_ccs_satisfaction, witness_from_trace_step, RiscvWitnessLayout,
};
use neo_memory::riscv::lookups::{RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables};
use neo_params::NeoParams;
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

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

#[test]
fn test_riscv_program_full_prove_verify() {
    // Program: compute 5 + 7 = 12 into x3
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
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = x1 + x2
        RiscvInstruction::Halt,
    ];

    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, program);
    let memory = RiscvMemory::new(32);
    let shout_tables = RiscvShoutTables::new(32);

    let trace = trace_program(cpu, memory, shout_tables, 100).expect("trace_program should succeed");
    assert!(trace.did_halt(), "program should halt");

    let last_step = trace.steps.last().expect("non-empty trace");
    let output = last_step.regs_after[3];
    assert_eq!(output, 12, "expected x3 = 12");

    // Build CCS + parameters for proving
    let layout = RiscvWitnessLayout::new();
    let ccs = build_riscv_alu_step_ccs(&layout);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.n).expect("params");
    let l = DummyCommit::default();

    // Prove every executed step
    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l);
    for step_idx in 0..trace.steps.len() {
        let mut witness = witness_from_trace_step(&trace, step_idx, &layout).expect("witness");
        witness.set_output(output);

        assert!(
            check_ccs_satisfaction(&ccs, witness.as_slice()),
            "step {step_idx} witness should satisfy CCS"
        );

        let input = ProveInput {
            ccs: &ccs,
            public_input: witness.public_inputs(),
            witness: witness.private_witness(),
            output_claims: &[],
        };
        session
            .add_step_from_io(&input)
            .expect("add_step_from_io should succeed");
    }

    let run = session
        .fold_and_prove(&ccs)
        .expect("fold_and_prove should succeed");
    let mcss_public = session.mcss_public();
    let ok = session
        .verify(&ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "verification should succeed");
}
