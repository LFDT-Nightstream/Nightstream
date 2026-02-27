#![cfg(not(feature = "poseidon-precompile"))]

use neo_memory::riscv::lookups::{
    decode_instruction, decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvShoutTables,
    PROG_ID,
};
use neo_vm_trace::trace_program;

#[test]
fn poseidon_decode_rejected_when_feature_disabled() {
    let program = vec![RiscvInstruction::Poseidon2Finalize, RiscvInstruction::Halt];
    let bytes = encode_program(&program);
    let words = decode_program(&bytes).expect_err("decode must fail when poseidon-precompile is disabled");
    assert!(
        words.contains("poseidon-precompile feature is disabled"),
        "unexpected error: {words}"
    );

    let word = u32::from_le_bytes(bytes[0..4].try_into().expect("word bytes"));
    let err = decode_instruction(word).expect_err("single-instruction decode must fail");
    assert!(
        err.contains("poseidon-precompile feature is disabled"),
        "unexpected error: {err}"
    );
}

#[test]
fn poseidon_vm_execution_rejected_when_feature_disabled() {
    let program = vec![RiscvInstruction::Poseidon2Finalize, RiscvInstruction::Halt];
    let bytes = encode_program(&program);
    let decoded = decode_program(&bytes).expect_err("decode should already fail");
    assert!(
        decoded.contains("poseidon-precompile feature is disabled"),
        "unexpected decode error: {decoded}"
    );

    // Build a manually decoded fallback to exercise VM execution guard as well.
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, vec![RiscvInstruction::Poseidon2Finalize, RiscvInstruction::Halt]);
    let twist = RiscvMemory::with_program_in_twist(32, PROG_ID, 0, &bytes);
    let shout = RiscvShoutTables::new(32);
    let err = trace_program(cpu, twist, shout, 32).expect_err("execution must fail");
    assert!(
        err.contains("poseidon-precompile feature is disabled")
            || err.contains("poseidon2 precompile instruction executed but feature `poseidon-precompile` is disabled"),
        "unexpected execution error: {err}"
    );
}
