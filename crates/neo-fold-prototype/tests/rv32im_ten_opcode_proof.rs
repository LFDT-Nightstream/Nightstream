//! End-to-end proof test for a fresh 10-instruction RV32IM program.

use neo_fold_prototype::rv32im::audit::{audit_rv32im_public_proof, audit_rv32im_public_proof_against_input};
use neo_fold_prototype::rv32im::layout::{
    RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID, RV32_REGISTER_COUNT,
};
use neo_fold_prototype::rv32im::tables::Rv32FamilyTag;
use neo_fold_prototype::rv32im::{
    build_parity_case_from_source, build_rv32im_audit_witness_bundle, encode_add, encode_addi, encode_and,
    encode_ecall, encode_lui, encode_lw, encode_mul, encode_ori, encode_sw, prove_rv32im_public_proof, MemoryWord,
    Rv32imParityCaseManifest, Rv32imParitySourceCase, Rv32imProofInput,
};

const START_PC: u32 = 0x1000;
const DATA_ADDR: u32 = 0x1000;
const EXPECTED_STORED_VALUE: u32 = 115;

fn ten_opcode_source_case() -> Rv32imParitySourceCase {
    let program_words = vec![
        encode_lui(1, 0x0000_1000),
        encode_addi(2, 0, 7),
        encode_addi(3, 0, 9),
        encode_add(4, 2, 3),
        encode_mul(5, 4, 2),
        encode_ori(6, 5, 3),
        encode_and(7, 6, 4),
        encode_sw(6, 1, 0),
        encode_lw(8, 1, 0),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: Rv32imParityCaseManifest {
            name: "ten_opcode_randomish_mix_ecall".into(),
            fixture_id: "ten_opcode_randomish_mix_ecall_v1".into(),
            protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
            lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
            family_tags: vec![
                Rv32FamilyTag::NativeAlu,
                Rv32FamilyTag::Multiply,
                Rv32FamilyTag::AlignedMemory,
                Rv32FamilyTag::ControlFlow,
            ],
        },
        start_pc: START_PC,
        program_words,
        initial_registers: [0; RV32_REGISTER_COUNT],
        initial_memory: vec![MemoryWord {
            addr: DATA_ADDR,
            value: 0,
        }],
        transcript_seed: b"rv32im-ten-op-randomish-v1".to_vec(),
    }
}

#[test]
fn rv32im_proves_and_verifies_fresh_ten_opcode_program() {
    let source = ten_opcode_source_case();
    let max_steps = source.program_words.len();
    let (_, derived) = build_parity_case_from_source(source.clone(), max_steps).expect("build derived parity case");
    let input = Rv32imProofInput { source, max_steps };

    let witness = build_rv32im_audit_witness_bundle(&input).expect("build rv32im audit witness bundle");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    audit_rv32im_public_proof(&proof).expect("audit rv32im public proof");
    audit_rv32im_public_proof_against_input(&input, &proof).expect("proof matches public input");
    let verified = build_rv32im_audit_witness_bundle(&input).expect("rebuild rv32im audit witness bundle");

    assert_eq!(verified.digest, witness.digest);
    assert_eq!(verified.trace.digest, witness.trace.digest);
    assert_eq!(verified.kernel_claims.digest, witness.kernel_claims.digest);

    assert_eq!(derived.kernel.final_pc, START_PC + 10 * 4);
    assert!(derived.kernel.halted);
    assert_eq!(derived.kernel.final_registers[1], DATA_ADDR);
    assert_eq!(derived.kernel.final_registers[4], 16);
    assert_eq!(derived.kernel.final_registers[5], 112);
    assert_eq!(derived.kernel.final_registers[6], EXPECTED_STORED_VALUE);
    assert_eq!(derived.kernel.final_registers[7], 16);
    assert_eq!(derived.kernel.final_registers[8], EXPECTED_STORED_VALUE);
    assert_eq!(
        derived.kernel.final_memory,
        vec![MemoryWord {
            addr: DATA_ADDR,
            value: EXPECTED_STORED_VALUE,
        }]
    );

    assert_eq!(proof.statement.initial_pc, u64::from(START_PC));
    assert_eq!(proof.statement.final_pc, u64::from(derived.kernel.final_pc));
    assert!(proof.statement.halted);
    assert_eq!(proof.statement.final_state_digest, derived.kernel.final_state_digest);
    assert_eq!(
        proof.statement.transcript_final_digest,
        derived.kernel.transcript_final_digest
    );
    assert_eq!(
        proof.claim.accepted.terminal.final_state_digest,
        derived.kernel.final_state_digest
    );
    assert_eq!(proof.claim.root0.terminal.root0_digest, derived.kernel.root0_digest);
    assert_eq!(witness.trace.execution_digest, derived.kernel.execution_digest);
    assert_eq!(witness.kernel_claims.root0_digest(), derived.kernel.root0_digest);
    assert_eq!(
        witness.kernel_claims.final_state_digest(),
        derived.kernel.final_state_digest
    );
    assert_eq!(
        witness.kernel_claims.transcript_final_digest(),
        derived.kernel.transcript_final_digest
    );
}
