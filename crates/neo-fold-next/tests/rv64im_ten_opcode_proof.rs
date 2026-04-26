//! End-to-end proof test for a fresh 10-instruction RV64IM program.

use neo_fold_next::rv64im::audit::{audit_rv64im_public_proof, audit_rv64im_public_proof_against_input};
use neo_fold_next::rv64im::layout::{
    RV64IM_PARITY_LOWERING_VERSION_ID, RV64IM_PARITY_PROTOCOL_VERSION_ID, RV64_REGISTER_COUNT,
};
use neo_fold_next::rv64im::tables::Rv64FamilyTag;
use neo_fold_next::rv64im::{
    build_parity_case_from_source, build_rv64im_audit_witness_bundle, encode_add, encode_addi, encode_and,
    encode_ecall, encode_ld, encode_lui, encode_mul, encode_ori, encode_sd, prove_rv64im_public_proof, MemoryWord,
    Rv64imParityCaseManifest, Rv64imParitySourceCase, Rv64imProofInput,
};

const START_PC: u64 = 0x1000;
const DATA_ADDR: u64 = 0x1000;
const EXPECTED_STORED_VALUE: u64 = 115;

fn ten_opcode_source_case() -> Rv64imParitySourceCase {
    let program_words = vec![
        encode_lui(1, 0x0000_1000),
        encode_addi(2, 0, 7),
        encode_addi(3, 0, 9),
        encode_add(4, 2, 3),
        encode_mul(5, 4, 2),
        encode_ori(6, 5, 3),
        encode_and(7, 6, 4),
        encode_sd(6, 1, 0),
        encode_ld(8, 1, 0),
        encode_ecall(),
    ];

    Rv64imParitySourceCase {
        manifest: Rv64imParityCaseManifest {
            name: "ten_opcode_randomish_mix_ecall".into(),
            fixture_id: "ten_opcode_randomish_mix_ecall_v1".into(),
            protocol_version_id: RV64IM_PARITY_PROTOCOL_VERSION_ID,
            lowering_version_id: RV64IM_PARITY_LOWERING_VERSION_ID,
            family_tags: vec![
                Rv64FamilyTag::NativeAlu,
                Rv64FamilyTag::Multiply,
                Rv64FamilyTag::AlignedMemory,
                Rv64FamilyTag::ControlFlow,
            ],
        },
        start_pc: START_PC,
        program_words,
        initial_registers: [0; RV64_REGISTER_COUNT],
        initial_memory: vec![MemoryWord {
            addr: DATA_ADDR,
            value: 0,
        }],
        transcript_seed: b"rv64im-ten-op-randomish-v1".to_vec(),
    }
}

#[test]
fn rv64im_proves_and_verifies_fresh_ten_opcode_program() {
    let source = ten_opcode_source_case();
    let max_steps = source.program_words.len();
    let (_, derived) = build_parity_case_from_source(source.clone(), max_steps).expect("build derived parity case");
    let input = Rv64imProofInput { source, max_steps };

    let witness = build_rv64im_audit_witness_bundle(&input).expect("build rv64im audit witness bundle");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    audit_rv64im_public_proof(&proof).expect("audit rv64im public proof");
    audit_rv64im_public_proof_against_input(&input, &proof).expect("proof matches public input");
    let verified = build_rv64im_audit_witness_bundle(&input).expect("rebuild rv64im audit witness bundle");

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

    assert_eq!(proof.statement.initial_pc, START_PC);
    assert_eq!(proof.statement.final_pc, derived.kernel.final_pc);
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
