#![cfg(not(feature = "poseidon-precompile"))]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn trace_prove_rejects_poseidon_program_when_feature_disabled() {
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

    let res = Rv32TraceWiring::from_rom(0, &rom)
        .xlen(32)
        .min_trace_len(8)
        .chunk_rows(128)
        .max_steps(256)
        .shout_auto_minimal()
        .prove();
    let err = match res {
        Ok(_) => panic!("prove must fail when poseidon-precompile feature is disabled"),
        Err(err) => err,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("poseidon-precompile feature is disabled")
            || msg.contains("program uses Poseidon2 precompile instructions"),
        "unexpected error: {msg}"
    );
}
