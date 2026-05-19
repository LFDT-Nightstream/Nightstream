use crate::common::proof_cases::{
    accepted_memory, accepted_multiply_high, accepted_test_guard, expect_accepted_audit_failure,
    refresh_stage1_semantic_digests,
};
use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::{
    audit_rv32im_accepted_proof, audit_rv32im_public_proof, audit_rv32im_public_proof_against_input,
};
use neo_fold_prototype::rv32im::layout::{
    RV32IM_PARITY_LOWERING_VERSION_ID, RV32IM_PARITY_PROTOCOL_VERSION_ID, RV32_REGISTER_COUNT,
};
use neo_fold_prototype::rv32im::tables::Rv32FamilyTag;
use neo_fold_prototype::rv32im::{
    encode_ecall, encode_lb, encode_lbu, encode_lh, encode_lhu, encode_lw, prove_rv32im_public_proof_with_options,
    MemoryWord, Rv32Opcode, Rv32imParityCaseManifest, Rv32imParitySourceCase, Rv32imProofInput,
    Rv32imPublicProofOptions,
};

#[test]
fn accepted_stage2_bundle_tracks_register_and_ram_timelines() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_memory();
    audit_rv32im_accepted_proof(&artifact).expect("accepted proof audits");
    assert!(!artifact.stage2.register.writes.is_empty());
    assert!(!artifact.stage2.ram.events.is_empty());
    assert!(!artifact.stage2.temporal.twist_links.is_empty());
}

#[test]
fn accepted_stage2_rejects_tampered_register_history() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    artifact.stage2.register.reads[0].value += 1;
    expect_accepted_audit_failure(&artifact, "stage2 register");
}

#[test]
fn accepted_stage2_rejects_tampered_ram_history() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    artifact.stage2.ram.events[0].next += 1;
    expect_accepted_audit_failure(&artifact, "stage2 RAM");
}

#[test]
fn accepted_stage2_rejects_public_initial_register_history_tamper() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    let addi_row = artifact
        .root_execution
        .execution_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.trace_opcode == Some(Rv32Opcode::Addi) && row.writes_rd)
        .expect("addi write row");
    let addi_index = addi_row.0;
    let addi_trace_index = addi_row.1.trace_index;
    artifact.root_execution.execution_rows[addi_index].rd_before = 9;
    artifact.stage1.sem_inputs[addi_index].rd_before = 9;
    let write = artifact
        .stage2
        .register
        .writes
        .iter_mut()
        .find(|event| event.trace_index == addi_trace_index)
        .expect("addi register write");
    write.previous = 9;
    refresh_stage1_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage2 register history mismatch");
}

#[test]
fn accepted_stage2_rejects_public_initial_memory_history_tamper() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    let store_row = artifact
        .root_execution
        .execution_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.trace_opcode == Some(Rv32Opcode::Sw))
        .expect("store row");
    let store_index = store_row.0;
    let store_trace_index = store_row.1.trace_index;
    artifact.root_execution.execution_rows[store_index].memory_before = Some(9);
    artifact.stage1.sem_inputs[store_index].memory_before = Some(9);
    let ram_event = artifact
        .stage2
        .ram
        .events
        .iter_mut()
        .find(|event| event.trace_index == store_trace_index)
        .expect("store RAM event");
    ram_event.previous = 9;
    let twist = artifact
        .stage2
        .temporal
        .twist_links
        .iter_mut()
        .find(|event| event.trace_index == store_trace_index)
        .expect("store twist link");
    twist.routed_memory_before = Some(9);
    refresh_stage1_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage2 RAM history mismatch");
}

#[test]
#[ignore = "custom narrow-memory public fixture currently exceeds the live RV32IM DEC budget on the safe public-proof schedule"]
fn public_stage2_accepts_narrow_memory_backing_word_history() {
    let _serial = accepted_test_guard();
    let mut registers = [0u32; RV32_REGISTER_COUNT];
    registers[10] = 0x3000;
    let input = Rv32imProofInput {
        source: Rv32imParitySourceCase {
            manifest: Rv32imParityCaseManifest {
                name: "rv32im_stage2_narrow_memory_load_history".into(),
                fixture_id: "rv32im_stage2_narrow_memory_load_history_v1".into(),
                protocol_version_id: RV32IM_PARITY_PROTOCOL_VERSION_ID,
                lowering_version_id: RV32IM_PARITY_LOWERING_VERSION_ID,
                family_tags: vec![Rv32FamilyTag::NarrowMemory, Rv32FamilyTag::ControlFlow],
            },
            start_pc: 0,
            program_words: vec![
                encode_lb(1, 10, 0),
                encode_lbu(2, 10, 1),
                encode_lh(3, 10, 0),
                encode_lhu(4, 10, 2),
                encode_lw(5, 10, 0),
                encode_ecall(),
            ],
            initial_registers: registers,
            initial_memory: vec![MemoryWord {
                addr: 0x3000,
                value: 0x807f_80ff,
            }],
            transcript_seed: b"rv32im-stage2-narrow-memory-load-history-v1".to_vec(),
        },
        max_steps: 16,
    };

    let proof = prove_rv32im_public_proof_with_options(
        &input,
        Rv32imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove narrow-memory public proof");
    audit_rv32im_public_proof(&proof).expect("audit narrow-memory public proof");
    audit_rv32im_public_proof_against_input(&input, &proof).expect("narrow-memory proof matches public input");
}

#[test]
fn accepted_stage2_accepts_multiply_high_temporary_register_reset() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_multiply_high();
    audit_rv32im_accepted_proof(&artifact).expect("accepted proof audits");
}
