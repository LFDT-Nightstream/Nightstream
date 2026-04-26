use crate::common::proof_cases::{
    accepted_alu, accepted_branch, accepted_divu, accepted_multiply_high, accepted_test_guard,
    expect_accepted_audit_failure,
};
use neo_fold_next::rv64im::audit::audit_rv64im_accepted_proof;

#[test]
fn accepted_stage1_bundle_tracks_sem_inputs_and_selected_opening() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_alu();
    audit_rv64im_accepted_proof(&artifact).expect("accepted proof audits");
    assert_eq!(
        artifact.stage1.sem_inputs.len(),
        artifact.root_execution.execution_rows.len()
    );
    assert_eq!(
        artifact.stage1.selected_opening.digest,
        artifact.stage_packages.packages.stage1.digest
    );
}

#[test]
fn accepted_stage1_rejects_tampered_sem_inputs() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_alu();
    artifact.stage1.sem_inputs[0].rs1_value += 1;
    expect_accepted_audit_failure(&artifact, "stage1 semantic inputs mismatch");
}

#[test]
fn accepted_stage1_rejects_tampered_branch_binding() {
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
fn accepted_stage1_rejects_tampered_nonselected_divu_helper_row() {
    let _serial = accepted_test_guard();
    let (mut artifact, _) = accepted_divu();
    let helper_row = artifact
        .stage1
        .row_bindings
        .iter_mut()
        .find(|row| row.trace_virtual_opcode.is_none() && !row.is_effect_row && !row.is_commit_row)
        .expect("non-selected divu helper row");
    helper_row.alu_result ^= 1;
    expect_accepted_audit_failure(&artifact, "stage1 row bindings mismatch");
}

#[test]
fn accepted_stage1_accepts_multiply_high_parity_case() {
    let _serial = accepted_test_guard();
    let (artifact, _) = accepted_multiply_high();
    audit_rv64im_accepted_proof(&artifact).expect("accepted proof audits");
}
