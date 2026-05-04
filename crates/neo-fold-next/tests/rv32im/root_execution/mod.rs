use crate::common::proof_cases::{
    accepted_branch, accepted_test_guard, expect_accepted_audit_failure, refresh_step_composition_surface_digest,
};
use neo_fold_next::rv32im::ccs::RV32IM_ROOT_ROW_WIDTH;
use neo_fold_next::rv32im::{
    build_program, decode_instruction, encode_addi, encode_ecall, encode_lh, encode_sll, MemoryWord, Rv32Program,
    Rv32State, Rv32imAcceptedProofArtifact,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn refresh_root_execution_bundle_digest(artifact: &mut Rv32imAcceptedProofArtifact) {
    artifact.root_execution.digest = artifact.root_execution.expected_digest();
    artifact.step_composition.root_execution_digest = artifact.root_execution.digest;
    refresh_step_composition_surface_digest(artifact);
}

fn refresh_row_local_ccs_acceptance_summary(artifact: &mut Rv32imAcceptedProofArtifact) {
    let summary = &mut artifact.root_execution.row_local_ccs_acceptance;
    summary.acceptance_count = summary.acceptances.len() as u64;
    summary.first_acceptance_digest = summary
        .acceptances
        .first()
        .map(|acceptance| acceptance.digest);
    summary.last_acceptance_digest = summary
        .acceptances
        .last()
        .map(|acceptance| acceptance.digest);
    summary.digest = summary.expected_digest();
    refresh_root_execution_bundle_digest(artifact);
}

fn refresh_execution_semantics_refinement_summary(artifact: &mut Rv32imAcceptedProofArtifact) {
    let summary = &mut artifact.root_execution.execution_semantics_refinement;
    summary.refinement_count = summary.refinements.len() as u64;
    summary.first_refinement_digest = summary
        .refinements
        .first()
        .map(|refinement| refinement.digest);
    summary.last_refinement_digest = summary
        .refinements
        .last()
        .map(|refinement| refinement.digest);
    summary.digest = summary.expected_digest();
    refresh_root_execution_bundle_digest(artifact);
}

fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn i_type(imm: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}

fn s_type(imm: u32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | opcode
}

#[test]
fn rv32_decoder_rejects_rv64_only_opcodes() {
    let rv64_only = [
        i_type(1, 0, 0, 1, 0x1b),    // ADDIW
        i_type(0, 0, 3, 1, 0x03),    // LD
        s_type(0, 1, 0, 3, 0x23),    // SD
        i_type(0, 0, 6, 1, 0x03),    // LWU
        r_type(1, 2, 1, 0, 3, 0x3b), // MULW
    ];
    for word in rv64_only {
        assert!(
            decode_instruction(word).is_err(),
            "accepted RV64-only word 0x{word:08x}"
        );
    }
}

#[test]
fn rv32_execution_wraps_register_arithmetic_and_masks_shifts() {
    let program = Rv32Program::new(0, vec![encode_addi(4, 4, 1), encode_sll(3, 1, 2), encode_ecall()]);
    let mut registers = [0u32; 32];
    registers[1] = 1;
    registers[2] = 32;
    registers[4] = u32::MAX;
    let state = Rv32State::new(0, registers, &[]);
    let build = build_program(&program, &state, 3).expect("build RV32 wraparound program");
    assert_eq!(build.final_state.regs[3], 1);
    assert_eq!(build.final_state.regs[4], 0);
    assert_eq!(build.final_state.pc, 12);
}

#[test]
fn rv32_memory_rejects_cross_word_narrow_access() {
    let program = Rv32Program::new(0, vec![encode_lh(1, 10, 3), encode_ecall()]);
    let mut registers = [0u32; 32];
    registers[10] = 0x3000;
    let state = Rv32State::new(
        0,
        registers,
        &[MemoryWord {
            addr: 0x3000,
            value: 0x1122_3344,
        }],
    );
    let err = build_program(&program, &state, 2).expect_err("cross-word halfword load must fail");
    let message = err.to_string();
    assert!(message.contains("not naturally aligned") || message.contains("crosses a 4-byte backing word"));
}

#[test]
fn rv32_root_layout_has_single_word_machine_columns() {
    assert_eq!(RV32IM_ROOT_ROW_WIDTH, 27);
}

#[test]
fn accepted_root_execution_rejects_tampered_semantic_row_values() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.root_execution.semantic_rows[0].values[0] += F::ONE;

    expect_accepted_audit_failure(&artifact, "root execution semantic-row digest mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_binding_row_opening_digest() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.root_execution.prepared_step_bindings.bindings[0].row_opening_digest[0] ^= 1;

    expect_accepted_audit_failure(&artifact, "root execution prepared-step bindings mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_acceptance_public_step_digest_after_rebinding() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    let acceptance = &mut artifact.root_execution.row_local_ccs_acceptance.acceptances[0];
    acceptance.public_step_digest[0] ^= 1;
    acceptance.digest = acceptance.expected_digest();
    refresh_row_local_ccs_acceptance_summary(&mut artifact);

    expect_accepted_audit_failure(&artifact, "root execution row-local CCS acceptance mismatch");
}

#[test]
fn accepted_root_execution_rejects_tampered_refinement_semantic_row_digest_after_rebinding() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    let refinement = &mut artifact
        .root_execution
        .execution_semantics_refinement
        .refinements[0];
    refinement.semantic_row_digest[0] ^= 1;
    refinement.digest = refinement.expected_digest();
    refresh_execution_semantics_refinement_summary(&mut artifact);

    expect_accepted_audit_failure(&artifact, "root execution semantics refinement mismatch");
}
