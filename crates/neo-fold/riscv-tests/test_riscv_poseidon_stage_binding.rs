#![cfg(feature = "poseidon-precompile")]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn poseidon_program_proves_and_verifies_with_committed_local_lane() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 11,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 },
        RiscvInstruction::Poseidon2Finalize,
        RiscvInstruction::Poseidon2SqueezeWord { rd: 12, idx: 0 },
        RiscvInstruction::Halt,
    ];
    let rom = encode_program(&program);

    let run = Rv32TraceWiring::from_rom(0, &rom)
        .xlen(32)
        .min_trace_len(8)
        .chunk_rows(128)
        .max_steps(256)
        .shout_auto_minimal()
        .prove()
        .expect("poseidon prove");

    assert!(run.requires_poseidon_stage());
    let step = &run.proof().steps[0];
    assert!(!step.mem.poseidon_cycle_me_claims.is_empty());
    assert!(!step.mem.poseidon_local_me_claims.is_empty());
    assert!(!step.poseidon_cycle_fold.is_empty());
    assert!(step.poseidon_local_time.is_some());
    assert!(!step.poseidon_local_fold.is_empty());

    run.verify_proof(run.proof()).expect("poseidon verify");
}

#[test]
fn non_poseidon_program_still_proves_and_verifies() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Halt,
    ];
    let rom = encode_program(&program);

    let run = Rv32TraceWiring::from_rom(0, &rom)
        .xlen(32)
        .min_trace_len(8)
        .chunk_rows(128)
        .max_steps(256)
        .shout_auto_minimal()
        .prove()
        .expect("non-poseidon prove");

    assert!(!run.requires_poseidon_stage());
    let step = &run.proof().steps[0];
    assert!(step.mem.poseidon_cycle_me_claims.is_empty());
    assert!(step.mem.poseidon_local_me_claims.is_empty());
    assert!(step.poseidon_cycle_fold.is_empty());
    assert!(step.poseidon_local_time.is_none());
    assert!(step.poseidon_local_fold.is_empty());

    run.verify_proof(run.proof()).expect("non-poseidon verify");
}

#[test]
fn poseidon_tampered_lane_commitment_fails_verification() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 11,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Poseidon2AbsorbElem { rs1: 10, rs2: 11 },
        RiscvInstruction::Poseidon2Finalize,
        RiscvInstruction::Poseidon2SqueezeWord { rd: 12, idx: 0 },
        RiscvInstruction::Halt,
    ];
    let rom = encode_program(&program);

    let run = Rv32TraceWiring::from_rom(0, &rom)
        .xlen(32)
        .min_trace_len(8)
        .chunk_rows(128)
        .max_steps(256)
        .shout_auto_minimal()
        .prove()
        .expect("poseidon prove");

    let mut tampered = run.proof().clone();
    let step = tampered
        .steps
        .get_mut(0)
        .expect("at least one step in poseidon proof");
    assert!(!step.mem.poseidon_cycle_me_claims.is_empty());
    assert!(!step.mem.poseidon_local_me_claims.is_empty());
    let cycle_c = step.mem.poseidon_cycle_me_claims[0].c.clone();
    step.mem.poseidon_cycle_me_claims[0].c = step.mem.poseidon_local_me_claims[0].c.clone();
    step.mem.poseidon_local_me_claims[0].c = cycle_c;

    let err = run
        .verify_proof(&tampered)
        .expect_err("tampered poseidon commitments must fail verification");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("fold") || msg.contains("commit") || msg.contains("poseidon") || msg.contains("sumcheck"),
        "unexpected error for tampered proof: {msg}"
    );
}
