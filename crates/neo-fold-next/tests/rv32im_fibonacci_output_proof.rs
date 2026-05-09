//! End-to-end RV32IM proof test for a looped Fibonacci program with fixed public input/output.

use neo_fold_next::core::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{audit_rv32im_public_proof, audit_rv32im_public_proof_against_input};
use neo_fold_next::rv32im::layout::{
    RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID, RV32_REGISTER_COUNT,
};
use neo_fold_next::rv32im::tables::Rv32FamilyTag;
use neo_fold_next::rv32im::{
    build_parity_case_from_source, encode_add, encode_addi, encode_beq, encode_ecall, encode_jal,
    prove_rv32im_public_proof_with_options, MemoryWord, Rv32imParityCaseManifest, Rv32imParitySourceCase,
    Rv32imProofInput, Rv32imPublicProofOptions,
};

const START_PC: u32 = 0x1000;
const INPUT_REGISTER: usize = 10;
const OUTPUT_REGISTER: usize = 11;
const FIB_INPUT: u32 = 7;
const EXPECTED_FIB_OUTPUT: u32 = 13;
const MAX_STEPS: usize = 64;

fn fibonacci_source_case() -> Rv32imParitySourceCase {
    let mut initial_registers = [0; RV32_REGISTER_COUNT];
    initial_registers[INPUT_REGISTER] = FIB_INPUT;

    let program_words = vec![
        encode_addi(1, 0, 0),
        encode_addi(2, 0, 1),
        encode_addi(3, 0, 0),
        encode_add(4, INPUT_REGISTER as u8, 0),
        encode_beq(3, 4, 24),
        encode_add(5, 1, 2),
        encode_add(1, 2, 0),
        encode_add(2, 5, 0),
        encode_addi(3, 3, 1),
        encode_jal(0, -20),
        encode_add(OUTPUT_REGISTER as u8, 1, 0),
        encode_ecall(),
    ];

    Rv32imParitySourceCase {
        manifest: Rv32imParityCaseManifest {
            name: "fibonacci_loop_input_x10_output_x11_ecall".into(),
            fixture_id: "fibonacci_loop_input_x10_output_x11_ecall_v1".into(),
            protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
            lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
            family_tags: vec![Rv32FamilyTag::NativeAlu, Rv32FamilyTag::ControlFlow],
        },
        start_pc: START_PC,
        program_words,
        initial_registers,
        initial_memory: Vec::<MemoryWord>::new(),
        transcript_seed: b"rv32im-fibonacci-input-x10-output-x11-v1".to_vec(),
    }
}

#[test]
#[ignore = "custom Fibonacci parity fixture currently exceeds the live RV32IM DEC budget on the safe public-proof schedule"]
fn rv32im_fibonacci_public_proof_binds_expected_output_state() {
    let source = fibonacci_source_case();
    let (_, derived) = build_parity_case_from_source(source.clone(), MAX_STEPS).expect("build derived parity case");
    let input = Rv32imProofInput {
        source,
        max_steps: MAX_STEPS,
    };

    let proof = prove_rv32im_public_proof_with_options(
        &input,
        Rv32imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove rv32im public proof");
    audit_rv32im_public_proof(&proof).expect("audit rv32im public proof");
    audit_rv32im_public_proof_against_input(&input, &proof).expect("proof matches public input");

    assert!(derived.kernel.halted);
    assert_eq!(derived.kernel.final_pc, START_PC + 12 * 4);
    assert_eq!(derived.kernel.final_registers[INPUT_REGISTER], FIB_INPUT);
    assert_eq!(derived.kernel.final_registers[1], EXPECTED_FIB_OUTPUT);
    assert_eq!(derived.kernel.final_registers[2], 21);
    assert_eq!(derived.kernel.final_registers[3], FIB_INPUT);
    assert_eq!(derived.kernel.final_registers[4], FIB_INPUT);
    assert_eq!(derived.kernel.final_registers[5], 21);
    assert_eq!(derived.kernel.final_registers[OUTPUT_REGISTER], EXPECTED_FIB_OUTPUT);
    assert!(derived.kernel.final_memory.is_empty());

    assert_eq!(proof.statement.initial_pc, u64::from(START_PC));
    assert_eq!(proof.statement.final_pc, u64::from(derived.kernel.final_pc));
    assert!(proof.statement.halted);
    assert_eq!(proof.statement.final_state_digest, derived.kernel.final_state_digest);
    assert_eq!(
        proof.claim.accepted.terminal.final_state_digest,
        derived.kernel.final_state_digest
    );
    assert_eq!(
        proof.kernel.kernel_claims.final_state_digest(),
        derived.kernel.final_state_digest
    );
}

#[test]
#[ignore = "custom Fibonacci parity fixture currently exceeds the live RV32IM DEC budget on the safe public-proof schedule"]
fn rv32im_fibonacci_public_proof_rejects_tampered_output_digest() {
    let input = Rv32imProofInput {
        source: fibonacci_source_case(),
        max_steps: MAX_STEPS,
    };

    let proof = prove_rv32im_public_proof_with_options(
        &input,
        Rv32imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove rv32im public proof");
    let mut tampered = proof.clone();
    tampered.statement.final_state_digest = [0xA5; 32];

    assert!(
        audit_rv32im_public_proof(&tampered).is_err(),
        "verification should fail when the proof claims a different output state"
    );
}
