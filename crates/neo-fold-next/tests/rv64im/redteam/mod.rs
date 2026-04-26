use crate::common::proof_cases::{
    accepted_alu, accepted_branch, accepted_divu, accepted_memory, accepted_test_guard, expect_accepted_audit_failure,
    refresh_accepted_artifact_digest, refresh_soundness_accounting_surface_digest, refresh_stage1_semantic_digests,
    refresh_stage3_semantic_digests,
};
use neo_fold_next::rv64im::audit::audit_rv64im_accepted_proof;
use neo_fold_next::rv64im::Rv64Opcode;

#[test]
fn redteam_semantic_input_substitution_fails_without_rebuild() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.stage1.sem_inputs[0].rs2_value ^= 1;
    expect_accepted_audit_failure(&artifact, "stage1 semantic inputs mismatch");
}

#[test]
fn redteam_bytecode_auth_mismatch_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.stage1.row_bindings[0].fetched_word ^= 1;
    expect_accepted_audit_failure(&artifact, "stage1");
}

#[test]
fn redteam_branch_target_forgery_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    let effect_row = artifact
        .stage1
        .row_bindings
        .iter_mut()
        .find(|row| row.is_effect_row)
        .expect("effect row");
    effect_row.next_pc += 4;
    expect_accepted_audit_failure(&artifact, "stage1");
}

#[test]
fn redteam_register_history_forgery_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    artifact.stage2.register.writes[0].next ^= 1;
    expect_accepted_audit_failure(&artifact, "stage2 register write surface mismatch");
}

#[test]
fn redteam_ram_history_forgery_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    artifact.stage2.ram.events[0].previous ^= 1;
    expect_accepted_audit_failure(&artifact, "stage2 RAM event surface mismatch");
}

#[test]
fn redteam_continuity_evasion_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.stage3.bridge.continuity[0].continuity_holds = false;
    expect_accepted_audit_failure(&artifact, "stage3 continuity surface mismatch");
}

#[test]
fn redteam_transcript_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.transcript.events[0].message.push(0xA5);
    expect_accepted_audit_failure(&artifact, "accepted proof artifact digest mismatch");
}

#[test]
fn redteam_provenance_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.root_execution.row_chunk_routes[0].chunk_index ^= 1;
    expect_accepted_audit_failure(&artifact, "row-to-chunk routing mismatch");
}

#[test]
fn redteam_row_local_ccs_acceptance_replay_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.root_execution.row_local_ccs_acceptance.digest[0] ^= 1;
    artifact.root_execution.digest = artifact.root_execution.expected_digest();
    refresh_accepted_artifact_digest(&mut artifact);
    expect_accepted_audit_failure(&artifact, "row-local CCS acceptance mismatch");
}

#[test]
fn redteam_rebuild_dependence_check_fails_without_audit_mode() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.root_execution.prepared_step_bindings.bindings[0].row_digest = [0x5A; 32];
    assert!(
        audit_rv64im_accepted_proof(&artifact).is_err(),
        "accepted audit must fail without relying on rebuild"
    );
}

#[test]
fn redteam_nonselected_divu_helper_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_divu();
    let helper_row = artifact
        .stage1
        .row_bindings
        .iter_mut()
        .find(|row| row.trace_virtual_opcode.is_none() && !row.is_effect_row && !row.is_commit_row)
        .expect("non-selected divu helper row");
    helper_row.rd_after ^= 1;
    expect_accepted_audit_failure(&artifact, "stage1 row bindings mismatch");
}

#[test]
fn redteam_register_history_against_public_seed_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    let addi_row = artifact
        .root_execution
        .execution_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.trace_opcode == Some(Rv64Opcode::Addi) && row.writes_rd)
        .expect("addi write row");
    let addi_index = addi_row.0;
    let addi_trace_index = addi_row.1.trace_index;
    artifact.root_execution.execution_rows[addi_index].rd_before = 17;
    artifact.stage1.sem_inputs[addi_index].rd_before = 17;
    let write = artifact
        .stage2
        .register
        .writes
        .iter_mut()
        .find(|event| event.trace_index == addi_trace_index)
        .expect("addi write event");
    write.previous = 17;
    refresh_stage1_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage2 register history mismatch");
}

#[test]
fn redteam_memory_history_against_public_seed_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_memory();
    let store_row = artifact
        .root_execution
        .execution_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.trace_opcode == Some(Rv64Opcode::Sd))
        .expect("store row");
    let store_index = store_row.0;
    let store_trace_index = store_row.1.trace_index;
    artifact.root_execution.execution_rows[store_index].memory_before = Some(17);
    artifact.stage1.sem_inputs[store_index].memory_before = Some(17);
    let event = artifact
        .stage2
        .ram
        .events
        .iter_mut()
        .find(|ram| ram.trace_index == store_trace_index)
        .expect("store RAM event");
    event.previous = 17;
    let twist = artifact
        .stage2
        .temporal
        .twist_links
        .iter_mut()
        .find(|link| link.trace_index == store_trace_index)
        .expect("store twist link");
    twist.routed_memory_before = Some(17);
    refresh_stage1_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage2 RAM history mismatch");
}

#[test]
fn redteam_stage3_start_boundary_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_branch();
    artifact.stage3.semantics.initial_pc = artifact.stage3.semantics.initial_pc.wrapping_add(4);
    refresh_stage3_semantic_digests(&mut artifact);
    expect_accepted_audit_failure(&artifact, "stage3 semantic bridge mismatch");
}

#[test]
fn redteam_soundness_accounting_tamper_fails() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.soundness_accounting.scalar_terms[0] = "bogus_term".into();
    refresh_soundness_accounting_surface_digest(&mut artifact);
    expect_accepted_audit_failure(&artifact, "soundness accounting surface mismatch");
}
