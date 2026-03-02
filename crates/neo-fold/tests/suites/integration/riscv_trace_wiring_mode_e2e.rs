#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn trace_mode_program_bytes() -> Vec<u8> {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
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
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    encode_program(&program)
}

#[test]
fn rv32_trace_wiring_mode_prove_verify() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_init_u32(/*reg=*/ 3, /*value=*/ 9)
        .ram_init_u32(/*addr=*/ 16, /*value=*/ 7)
        .min_trace_len(8)
        .prove()
        .expect("trace wiring prove");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.fold_count(), 1, "trace-wiring mode should produce one folding step");
    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        8,
        "trace_min_len should set padded trace length"
    );
    assert_eq!(run.layout().t, 8, "layout t should match padded trace length");
}

#[test]
fn rv32_trace_wiring_mode_keeps_exec_unpadded_layout() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        3,
        "trace execution rows remain unpadded; power-of-two padding is a proving/layout concern"
    );
    assert!(
        run.layout().t >= run.exec_table().rows.len(),
        "layout.t should cover exec rows (layout.t={}, exec_rows={})",
        run.layout().t,
        run.exec_table().rows.len()
    );
    assert_eq!(run.layout().t, 3, "layout.t should match unpadded trace length");
}

#[test]
fn rv32_trace_wiring_mode_ram_output_binding_prove_verify() {
    // Program: ADDI x1, x0, 7; SW x1, 16(x0); HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Store {
            op: neo_memory::riscv::lookups::RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 16,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .output_claim(/*addr=*/ 16, /*value=*/ neo_math::F::from_u64(7))
        .prove()
        .expect("trace wiring prove with RAM output binding");

    run.verify()
        .expect("trace wiring verify with RAM output binding");
}

#[test]
fn rv32_trace_wiring_mode_reg_output_binding_prove_verify() {
    // Program: ADDI x2, x0, 3; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*value=*/ neo_math::F::from_u64(3))
        .prove()
        .expect("trace wiring prove with REG output binding");

    run.verify()
        .expect("trace wiring verify with REG output binding");
}

#[test]
fn rv32_trace_wiring_mode_wrong_reg_output_claim_fails_verify() {
    // Program: ADDI x2, x0, 3; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*value=*/ neo_math::F::from_u64(4))
        .prove()
        .expect("trace wiring prove with wrong REG claim still produces a proof");

    let err = run
        .verify()
        .expect_err("trace wiring verify should fail for wrong REG output claim");
    let msg = format!("{err}");
    assert!(msg.contains("output sumcheck failed"), "unexpected verify error: {msg}");
}

#[test]
fn rv32_trace_wiring_mode_mismatched_output_binding_cfg_fails_verify_proof() {
    // Program: ADDI x1, x0, 7; SW x1, 16(x0); HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Store {
            op: neo_memory::riscv::lookups::RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 16,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    // Prove with the canonical output claim.
    let run_ok = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .output_claim(/*addr=*/ 16, /*value=*/ neo_math::F::from_u64(7))
        .prove()
        .expect("trace wiring prove with canonical RAM output claim");

    // Build a verifier context with a mismatched output-claim template.
    let run_mismatch = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .output_claim(/*addr=*/ 16, /*value=*/ neo_math::F::from_u64(8))
        .prove()
        .expect("trace wiring prove with mismatched RAM output claim");

    let err = run_mismatch
        .verify_proof(run_ok.proof())
        .expect_err("verification should fail when proof is checked under mismatched output binding config");
    let msg = err.to_string();
    assert!(
        msg.contains("challenges_public mismatch")
            || msg.contains("Π_CCS challenges_public mismatch")
            || msg.contains("output sumcheck failed")
            || msg.contains("output binding final check failed")
            || msg.contains("output sparse final check failed"),
        "unexpected verify error: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_mode_allows_without_insecure_ack() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace-wiring mode should not require insecure benchmark-only ack");
    run.verify()
        .expect("trace-wiring proof should verify without insecure benchmark-only ack");
}

#[test]
fn rv32_trace_wiring_mode_chunked_ivc() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");

    run.verify().expect("trace wiring verify with chunked ivc");
    assert_eq!(run.fold_count(), 2, "expected two fold steps with trace_chunk_rows=2");
}

#[test]
fn rv32_trace_shout_override_must_superset_inferred_set() {
    let program_bytes = trace_mode_program_bytes();
    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shout_ops([RiscvOpcode::Xor])
        .prove()
    {
        Ok(_) => panic!("shout override that misses required tables must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("superset") && msg.contains("Add"),
        "unexpected error message: {msg}"
    );
}
