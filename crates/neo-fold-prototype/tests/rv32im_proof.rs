//! Focused tests for the final public RV32IM proof API.

use std::sync::{LazyLock, Mutex, MutexGuard};

use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::{
    audit_rv32im_public_proof, audit_rv32im_public_proof_against_input,
    audit_rv32im_public_proof_with_witness as audit_rv32im_proof,
};
use neo_fold_prototype::rv32im::ccs::RV32IM_ROOT_ROW_WIDTH;
use neo_fold_prototype::rv32im::{
    build_main_lane_surface, build_rv32im_accepted_proof_artifact,
    build_rv32im_kernel_export_source_from_accepted_artifact, parity_source_cases,
    prove_rv32im_audit_proof as prove_rv32im_proof, prove_rv32im_public_proof, prove_rv32im_public_proof_with_options,
    verify_rv32im_kernel_export_source, Rv32imProof, Rv32imProofInput, Rv32imProofWitnessBundle,
    Rv32imPublicProofOptions,
};
use neo_fold_prototype::rv32im::{
    Rv32imKernelClaimProofBundle, Rv32imKernelProofBundle, Rv32imStageClaimDigestBundle, Rv32imStageClaimProofBundle,
    Rv32imStageWitnessProjectionBundle, Rv32imStageWitnessSummaryBundle,
};
use neo_transcript::{Poseidon2Transcript, Transcript};

fn source_case(name: &str) -> neo_fold_prototype::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

fn proof_test_guard() -> MutexGuard<'static, ()> {
    static RV32IM_PROOF_TEST_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    RV32IM_PROOF_TEST_MUTEX
        .lock()
        .expect("serialize rv32im_proof tests")
}

fn rows_per_chunk_7_options() -> Rv32imPublicProofOptions {
    Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(7),
    }
}

static CONTROL_FLOW_AUDIT_PROOF: LazyLock<(Rv32imProofWitnessBundle, Rv32imProof)> = LazyLock::new(|| {
    let input = proof_input("control_flow_jal_skip_ecall");
    prove_rv32im_proof(&input).expect("prove control-flow rv32im audit proof")
});

static CONTROL_FLOW_PUBLIC_PROOF: LazyLock<Rv32imProof> = LazyLock::new(|| {
    let input = proof_input("control_flow_jal_skip_ecall");
    prove_rv32im_public_proof(&input).expect("prove control-flow rv32im public proof")
});

static CONTROL_FLOW_CHUNKED_PUBLIC_PROOF: LazyLock<Rv32imProof> = LazyLock::new(|| {
    let input = proof_input("control_flow_jal_skip_ecall");
    prove_rv32im_public_proof_with_options(&input, rows_per_chunk_7_options())
        .expect("prove control-flow chunked rv32im public proof")
});

static CONTROL_FLOW_ECALL_ONLY_AUDIT_PROOF: LazyLock<(Rv32imProofWitnessBundle, Rv32imProof)> = LazyLock::new(|| {
    let input = proof_input("control_flow_ecall_only");
    prove_rv32im_proof(&input).expect("prove control-flow ecall-only rv32im audit proof")
});

static NATIVE_ADD_CHAIN_AUDIT_PROOF: LazyLock<(Rv32imProofWitnessBundle, Rv32imProof)> = LazyLock::new(|| {
    let input = proof_input("native_add_chain_x0_ecall");
    prove_rv32im_proof(&input).expect("prove native add-chain rv32im audit proof")
});

static NATIVE_LOGIC_COMPARE_AUDIT_PROOF: LazyLock<(Rv32imProofWitnessBundle, Rv32imProof)> = LazyLock::new(|| {
    let input = proof_input("native_logic_compare_chain_ecall");
    prove_rv32im_proof(&input).expect("prove native logic-compare rv32im audit proof")
});

fn stage_witness_summary_digest(summary: &Rv32imStageWitnessSummaryBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_witness_summary_bundle");
    tr.append_u64s(
        b"rv32im/stage_witness_summary_bundle/meta",
        &[
            summary.stage1_row_count,
            summary.stage2_register_read_count,
            summary.stage2_register_write_count,
            summary.stage2_ram_event_count,
            summary.stage2_twist_link_count,
            summary.stage3_continuity_count,
            summary.stage3_halted as u64,
            summary.transcript_event_count,
        ],
    );
    tr.digest32()
}

fn stage_witness_projection_digest(bundle: &Rv32imStageWitnessProjectionBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_witness_summary_proof_bundle");
    tr.append_message(
        b"rv32im/stage_witness_summary_proof_bundle/summary",
        &bundle.summary.digest,
    );
    tr.digest32()
}

fn stage_claim_digest_bundle_digest(summary: &Rv32imStageClaimDigestBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_claim_digest_bundle");
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/claim_bundle_digest",
        &summary.claim_bundle_digest,
    );
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/stage1_digest",
        &summary.stage1_digest,
    );
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/stage2_digest",
        &summary.stage2_digest,
    );
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/stage3_digest",
        &summary.stage3_digest,
    );
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/transcript_digest",
        &summary.transcript_digest,
    );
    tr.append_message(
        b"rv32im/stage_claim_digest_bundle/execution_digest",
        &summary.execution_digest,
    );
    tr.digest32()
}

fn stage_claim_proof_digest(bundle: &Rv32imStageClaimProofBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/stage_claim_proof_bundle");
    tr.append_message(b"summary_digest", &bundle.summary.digest);
    tr.append_message(b"statement_digest", &bundle.packaged.statement.digest);
    tr.append_message(b"proof_digest", &bundle.packaged.proof.proof_digest);
    tr.digest32()
}

fn kernel_claim_proof_digest(bundle: &Rv32imKernelClaimProofBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_claim_proof_bundle");
    tr.append_message(b"summary_digest", &bundle.summary.digest);
    tr.append_message(b"statement_digest", &bundle.packaged.statement.digest);
    tr.append_message(b"proof_digest", &bundle.packaged.proof.proof_digest);
    tr.digest32()
}

fn kernel_proof_bundle_digest(bundle: &Rv32imKernelProofBundle) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/kernel_proof_bundle");
    tr.append_message(b"rv32im/kernel_proof_bundle/root_params_id", &bundle.root_params_id);
    tr.append_message(b"rv32im/kernel_proof_bundle/trace_digest", &bundle.trace.digest);
    tr.append_message(b"rv32im/kernel_proof_bundle/stages_digest", &bundle.stages.digest);
    tr.append_message(
        b"rv32im/kernel_proof_bundle/stage_claims_digest",
        &bundle.stage_claims.digest,
    );
    tr.append_message(
        b"rv32im/kernel_proof_bundle/stage_packages_digest",
        &bundle.stage_packages.digest,
    );
    tr.append_message(
        b"rv32im/kernel_proof_bundle/kernel_opening_digest",
        &bundle.kernel_opening.digest,
    );
    tr.append_message(
        b"rv32im/kernel_proof_bundle/kernel_claims_digest",
        &bundle.kernel_claims.digest,
    );
    tr.append_message(
        b"rv32im/kernel_proof_bundle/root_lane_columns_digest",
        &bundle.root_lane_columns.digest,
    );
    tr.append_message(
        b"rv32im/kernel_proof_bundle/root_lane_commitment_digest",
        &bundle.root_lane_commitment.digest,
    );
    tr.append_message(b"rv32im/kernel_proof_bundle/main_lane_digest", &bundle.main_lane.digest);
    tr.digest32()
}

#[test]
fn rv32im_proof_roundtrip_matches_kernel_export() {
    let _serial = proof_test_guard();
    let (witness, proof) = CONTROL_FLOW_AUDIT_PROOF.clone();
    let verified = audit_rv32im_proof(&proof).expect("audit rv32im proof");
    audit_rv32im_public_proof_against_input(&proof_input("control_flow_jal_skip_ecall"), &proof)
        .expect("proof matches public input");

    assert_ne!(proof.claim.digest, [0; 32]);
    assert_ne!(proof.claim.main_lane.digest, [0; 32]);
    assert_ne!(proof.claim.opening.digest, [0; 32]);
    assert_ne!(proof.claim.joint_opening.digest, [0; 32]);
    assert_ne!(proof.claim.root0.digest, [0; 32]);
    assert_ne!(proof.statement.digest, [0; 32]);
    assert_eq!(verified.digest, witness.digest);
    assert_eq!(verified.trace.digest, witness.trace.digest);
    assert_eq!(verified.stages.digest, witness.stages.digest);
    assert_eq!(
        proof.kernel.trace.execution_row_count(),
        witness.trace.execution_row_count()
    );
    assert_eq!(
        proof.kernel.trace.execution_digest(),
        witness.kernel_claims.execution_digest()
    );
    assert_eq!(
        proof.kernel.stages.stage1_row_count(),
        witness.stages.stage1_row_count()
    );
    assert_eq!(
        proof.kernel.stage_claims.stage1_digest(),
        witness.stage_claims.summary.stage1_digest
    );
    assert_eq!(
        proof.kernel.stage_packages.stage1_digest(),
        witness.stage_packages.summary.stage1_digest
    );
    assert_eq!(verified.kernel_opening.digest, witness.kernel_opening.digest);
    assert_eq!(
        proof.kernel.kernel_claims.root0_digest(),
        witness.kernel_claims.root0_digest()
    );
    assert_eq!(
        proof.claim.accepted.statement.proof_statement_digest,
        proof.statement.digest
    );
    assert_eq!(
        proof.claim.accepted.statement.kernel_opening_digest,
        witness.kernel_opening.digest
    );
    assert_eq!(
        proof.claim.accepted.main_lane.main_lane_bundle_digest,
        proof.kernel.main_lane.digest
    );
    assert_eq!(
        proof.claim.main_lane.binding.main_lane_bundle_digest,
        proof.kernel.main_lane.digest
    );
    assert_eq!(
        proof.statement.root_lane_columns_digest,
        proof.kernel.root_lane_columns.digest
    );
    assert_eq!(proof.kernel.root_lane_commitment, witness.root_lane_commitment);
    let derived_main_lane_surface = build_main_lane_surface(&witness.root_lane_columns);
    assert_eq!(
        derived_main_lane_surface.public_step_count,
        witness.root_lane_columns.time_len
    );
    assert_eq!(derived_main_lane_surface.row_width, RV32IM_ROOT_ROW_WIDTH as u64);
    assert_eq!(
        derived_main_lane_surface.family_digest,
        witness.root_lane_columns.family_digest
    );
    assert_ne!(witness.root_lane_columns.family_digest, [0; 32]);
    assert_ne!(witness.root_lane_columns.object.digest, [0; 32]);
    assert_ne!(witness.root_lane_commitment.digest, [0; 32]);
    assert_eq!(
        witness.root_lane_commitment.commitments.commitment_count,
        RV32IM_ROOT_ROW_WIDTH as u64
    );
    assert_eq!(
        derived_main_lane_surface.object_digest,
        witness.root_lane_columns.object.digest
    );
    assert_eq!(
        derived_main_lane_surface.first_public_step,
        witness.root_lane_columns.first_row
    );
    assert_eq!(
        derived_main_lane_surface.last_public_step,
        witness.root_lane_columns.last_row
    );
    assert!(witness.root_lane_columns.first_row.is_some());
    assert!(witness.root_lane_columns.last_row.is_some());
    assert_eq!(
        witness
            .root_lane_columns
            .first_row
            .as_ref()
            .expect("first public step")
            .id
            .logical_index,
        0
    );
    assert_eq!(
        witness
            .root_lane_columns
            .last_row
            .as_ref()
            .expect("last public step")
            .id
            .logical_index,
        witness.root_lane_columns.time_len.saturating_sub(1)
    );
    assert_eq!(
        proof.claim.opening.stages.stage_claims_digest,
        witness.stage_claims.digest
    );
    assert_eq!(
        proof.claim.joint_opening.binding.proof_statement_digest,
        proof.statement.digest
    );
    assert_eq!(
        proof.claim.root0.terminal.root0_digest,
        witness.kernel_claims.root0_digest()
    );
    assert_eq!(proof.kernel.digest, kernel_proof_bundle_digest(&proof.kernel));
    assert_eq!(proof.statement.kernel_opening_digest, witness.kernel_opening.digest);
    assert_eq!(
        proof.kernel.root_lane_columns.digest,
        proof.statement.root_lane_columns_digest
    );
    assert_eq!(
        proof.kernel.main_lane.root_lane_columns_digest(),
        proof.kernel.root_lane_columns.digest
    );
    assert_eq!(
        proof.kernel.main_lane.root_lane_commitment_digest(),
        proof.kernel.root_lane_commitment.digest
    );
    assert_eq!(
        proof.kernel.main_lane.public_step_count(),
        witness.root_lane_columns.time_len
    );
    assert_eq!(verified.root_lane_columns, witness.root_lane_columns);
    assert_eq!(verified.root_lane_commitment, witness.root_lane_commitment);
    assert_eq!(
        proof.kernel.root_lane_columns.object.digest,
        witness.root_lane_columns.object.digest
    );
    assert_eq!(
        proof.kernel.root_lane_columns.family_digest,
        witness.root_lane_columns.family_digest
    );
    assert_eq!(
        proof.kernel.root_lane_columns.row_width,
        witness.root_lane_columns.row_width
    );
    assert_eq!(
        proof.kernel.root_lane_columns.time_len,
        witness.root_lane_columns.time_len
    );
    assert_eq!(
        proof.kernel.root_lane_columns.first_row,
        derived_main_lane_surface.first_public_step
    );
    assert_eq!(
        proof.kernel.root_lane_columns.last_row,
        derived_main_lane_surface.last_public_step
    );
    assert_eq!(
        proof
            .kernel
            .root_lane_commitment
            .first_selected_row
            .as_ref()
            .map(|reference| reference.value_digest),
        proof
            .kernel
            .root_lane_columns
            .first_row
            .as_ref()
            .map(|reference| reference.value_digest)
    );
    assert_eq!(
        proof
            .kernel
            .root_lane_commitment
            .last_selected_row
            .as_ref()
            .map(|reference| reference.value_digest),
        proof
            .kernel
            .root_lane_columns
            .last_row
            .as_ref()
            .map(|reference| reference.value_digest)
    );
    assert_eq!(
        proof.statement.final_state_digest,
        witness.kernel_claims.final_state_digest()
    );
    assert_eq!(proof.statement.public_step_count, witness.root_lane_columns.time_len);
}

#[test]
fn rv32im_public_proof_defaults_to_whole_trace_root_chunk() {
    let _serial = proof_test_guard();
    let proof = CONTROL_FLOW_PUBLIC_PROOF.clone();
    audit_rv32im_public_proof(&proof).expect("audit public proof");

    assert_eq!(proof.statement.fold_schedule, FoldSchedule::WholeTrace);
    assert_eq!(proof.statement.chunk_count, 1);
    assert_eq!(proof.kernel.main_lane.fold_schedule(), FoldSchedule::WholeTrace);
    assert_eq!(proof.kernel.main_lane.chunk_count(), 1);
    assert_eq!(proof.kernel.main_lane.packaged.statement.chunks.len(), 1);
    assert_eq!(proof.kernel.main_lane.packaged.proof.session.chunks.len(), 1);

    let chunk = &proof.kernel.main_lane.packaged.statement.chunks[0];
    assert_eq!(chunk.start_index, 0);
    assert_eq!(chunk.steps.len() as u64, proof.statement.public_step_count);
}

#[test]
fn rv32im_public_proof_rows_per_chunk_schedule_is_contiguous() {
    let _serial = proof_test_guard();
    let proof = CONTROL_FLOW_CHUNKED_PUBLIC_PROOF.clone();
    audit_rv32im_public_proof(&proof).expect("audit chunked public proof");

    assert_eq!(proof.statement.fold_schedule, FoldSchedule::RowsPerChunk(7));
    assert_eq!(proof.kernel.main_lane.fold_schedule(), FoldSchedule::RowsPerChunk(7));
    assert_eq!(
        proof.statement.chunk_count as usize,
        proof.kernel.main_lane.packaged.statement.chunks.len()
    );

    let mut next_start = 0usize;
    for (idx, chunk) in proof
        .kernel
        .main_lane
        .packaged
        .statement
        .chunks
        .iter()
        .enumerate()
    {
        assert_eq!(chunk.start_index, next_start);
        assert!(!chunk.steps.is_empty());
        if idx + 1 < proof.kernel.main_lane.packaged.statement.chunks.len() {
            assert_eq!(chunk.steps.len(), 7);
        } else {
            assert!(chunk.steps.len() <= 7);
        }
        next_start += chunk.steps.len();
    }

    assert_eq!(next_start as u64, proof.statement.public_step_count);
    assert_eq!(next_start as u64, proof.kernel.main_lane.public_step_count());
    assert_eq!(
        proof.kernel.main_lane.packaged.proof.session.chunks.len(),
        proof.kernel.main_lane.packaged.statement.chunks.len()
    );
}

#[test]
fn rv32im_proof_rejects_tampered_root_main_lane_packaged_proof() {
    let _serial = proof_test_guard();
    let (_, mut proof) = CONTROL_FLOW_AUDIT_PROOF.clone();

    proof.kernel.main_lane.packaged.proof.proof_digest[0] ^= 1;

    audit_rv32im_proof(&proof).expect_err("tampered root main-lane packaged proof must fail");
}

#[test]
fn rv32im_public_proof_rejects_tampered_fold_schedule_and_chunk_layout() {
    let _serial = proof_test_guard();
    let mut tampered_schedule = CONTROL_FLOW_CHUNKED_PUBLIC_PROOF.clone();
    tampered_schedule.statement.fold_schedule = FoldSchedule::WholeTrace;
    audit_rv32im_public_proof(&tampered_schedule).expect_err("tampered public statement fold schedule must fail");

    let mut tampered_chunk_count = CONTROL_FLOW_CHUNKED_PUBLIC_PROOF.clone();
    tampered_chunk_count
        .kernel
        .main_lane
        .packaged
        .statement
        .chunk_count += 1;
    audit_rv32im_public_proof(&tampered_chunk_count).expect_err("tampered packaged chunk count must fail");

    let mut tampered_start_index = CONTROL_FLOW_CHUNKED_PUBLIC_PROOF.clone();
    tampered_start_index
        .kernel
        .main_lane
        .packaged
        .statement
        .chunks[0]
        .start_index += 1;
    audit_rv32im_public_proof(&tampered_start_index).expect_err("tampered chunk start index must fail");
}

#[test]
fn rv32im_proof_rejects_tampered_stage_package_proof() {
    let _serial = proof_test_guard();
    let (_, mut proof) = CONTROL_FLOW_AUDIT_PROOF.clone();

    proof
        .kernel
        .stage_packages
        .packages
        .stage1
        .packaged
        .proof
        .proof_digest[0] ^= 1;

    audit_rv32im_proof(&proof).expect_err("tampered stage packaged proof must fail");
}

#[test]
fn rv32im_proof_rejects_tampered_kernel_opening_proof() {
    let _serial = proof_test_guard();
    let (_, mut proof) = CONTROL_FLOW_AUDIT_PROOF.clone();

    proof
        .kernel
        .kernel_opening
        .opening
        .bindings
        .packaged
        .proof
        .proof_digest[0] ^= 1;

    audit_rv32im_proof(&proof).expect_err("tampered kernel opening packaged proof must fail");
}

#[test]
fn rv32im_kernel_export_source_roundtrip_matches_accepted_artifact() {
    let _serial = proof_test_guard();
    let (_, proof) = CONTROL_FLOW_AUDIT_PROOF.clone();
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build rv32im accepted artifact");
    let source = build_rv32im_kernel_export_source_from_accepted_artifact(&accepted_artifact)
        .expect("build rv32im kernel export source");

    verify_rv32im_kernel_export_source(&source).expect("verify rv32im kernel export source");
    assert_ne!(source.digest, [0; 32]);
    assert_eq!(source.root_execution.digest, accepted_artifact.root_execution.digest);
}

#[test]
fn rv32im_kernel_export_source_rejects_tampered_digest() {
    let _serial = proof_test_guard();
    let (_, proof) = CONTROL_FLOW_AUDIT_PROOF.clone();
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build rv32im accepted artifact");
    let mut source = build_rv32im_kernel_export_source_from_accepted_artifact(&accepted_artifact)
        .expect("build rv32im kernel export source");

    source.digest[0] ^= 1;
    let err = verify_rv32im_kernel_export_source(&source).expect_err("tampered source digest must fail");
    assert!(format!("{err}").contains("kernel export source"));
}

#[test]
fn rv32im_proof_rejects_tampered_stage_claim_packaged_proof() {
    let _serial = proof_test_guard();
    let (_, mut proof) = CONTROL_FLOW_AUDIT_PROOF.clone();
    proof.kernel.stage_claims.packaged.proof.proof_digest[0] ^= 1;
    proof.kernel.stage_claims.digest = stage_claim_proof_digest(&proof.kernel.stage_claims);
    proof.kernel.digest = kernel_proof_bundle_digest(&proof.kernel);

    audit_rv32im_proof(&proof).expect_err("tampered stage-claim packaged proof must fail");
}

#[test]
fn rv32im_proof_rejects_tampered_kernel_claim_packaged_proof() {
    let _serial = proof_test_guard();
    let (_, mut proof) = CONTROL_FLOW_AUDIT_PROOF.clone();
    proof.kernel.kernel_claims.packaged.proof.proof_digest[0] ^= 1;
    proof.kernel.kernel_claims.digest = kernel_claim_proof_digest(&proof.kernel.kernel_claims);
    proof.kernel.digest = kernel_proof_bundle_digest(&proof.kernel);

    audit_rv32im_proof(&proof).expect_err("tampered kernel-claim packaged proof must fail");
}

#[test]
fn rv32im_proof_rejects_proof_from_different_public_input() {
    let _serial = proof_test_guard();
    let expected_input = proof_input("vertical_add_sw_lw_ecall");
    let (_, foreign_proof) = CONTROL_FLOW_ECALL_ONLY_AUDIT_PROOF.clone();
    audit_rv32im_proof(&foreign_proof).expect("audit proof verification");
    audit_rv32im_public_proof_against_input(&expected_input, &foreign_proof)
        .expect_err("proof for a different public input must fail audit consistency");
}

#[test]
fn rv32im_proof_rejects_tampered_kernel_and_main_lane_surfaces() {
    let _serial = proof_test_guard();
    let (_witness, proof) = NATIVE_ADD_CHAIN_AUDIT_PROOF.clone();

    let mut tampered_kernel = proof.clone();
    tampered_kernel
        .kernel
        .kernel_opening
        .bindings
        .bindings_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_kernel).is_err());

    let mut tampered_trace = proof.clone();
    tampered_trace.kernel.trace.execution_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_trace).is_err());

    let mut tampered_trace_shape = proof.clone();
    tampered_trace_shape.kernel.trace.shape.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_trace_shape).is_err());

    let mut tampered_stages = proof.clone();
    tampered_stages
        .kernel
        .stages
        .summary
        .stage3_continuity_count ^= 1;
    assert!(audit_rv32im_proof(&tampered_stages).is_err());

    let mut tampered_stage_summary = proof.clone();
    tampered_stage_summary.kernel.stages.summary.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_stage_summary).is_err());

    let mut tampered_stage_claims = proof.clone();
    tampered_stage_claims
        .kernel
        .stage_claims
        .summary
        .stage1_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_stage_claims).is_err());

    let mut tampered_stage_packages = proof.clone();
    tampered_stage_packages
        .kernel
        .stage_packages
        .summary
        .stage1_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_stage_packages).is_err());

    let mut tampered_stage_claim_summary = proof.clone();
    tampered_stage_claim_summary
        .kernel
        .stage_claims
        .summary
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_stage_claim_summary).is_err());

    let mut tampered_stage_package_summary = proof.clone();
    tampered_stage_package_summary
        .kernel
        .stage_packages
        .summary
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_stage_package_summary).is_err());

    let mut tampered_statement = proof.clone();
    tampered_statement.statement.final_state_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_statement).is_err());

    let mut tampered_claim = proof.clone();
    tampered_claim.claim.accepted.terminal.final_state_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_claim).is_err());

    let mut tampered_accepted_statement_binding = proof.clone();
    tampered_accepted_statement_binding
        .claim
        .accepted
        .statement
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_accepted_statement_binding).is_err());

    let mut tampered_main_lane_claim = proof.clone();
    tampered_main_lane_claim
        .claim
        .main_lane
        .binding
        .main_lane_bundle_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_main_lane_claim).is_err());

    let mut tampered_statement_main_lane_surface = proof.clone();
    tampered_statement_main_lane_surface
        .statement
        .main_lane_surface_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_statement_main_lane_surface).is_err());

    let mut tampered_statement_root_lane_columns = proof.clone();
    tampered_statement_root_lane_columns
        .statement
        .root_lane_columns_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_statement_root_lane_columns).is_err());

    let mut tampered_kernel_root_lane_columns = proof.clone();
    tampered_kernel_root_lane_columns
        .kernel
        .root_lane_columns
        .family_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_kernel_root_lane_columns).is_err());

    let mut tampered_kernel_root_lane_commitment = proof.clone();
    tampered_kernel_root_lane_commitment
        .kernel
        .root_lane_commitment
        .commitments
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_kernel_root_lane_commitment).is_err());

    let mut tampered_main_lane_opening_ref = proof.clone();
    tampered_main_lane_opening_ref
        .kernel
        .root_lane_columns
        .first_row
        .as_mut()
        .expect("first row")
        .value_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_main_lane_opening_ref).is_err());

    let mut tampered_accepted_main_lane_surface = proof.clone();
    tampered_accepted_main_lane_surface
        .claim
        .accepted
        .main_lane
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_accepted_main_lane_surface).is_err());

    let mut tampered_main_lane_claim_binding_digest = proof.clone();
    tampered_main_lane_claim_binding_digest
        .claim
        .main_lane
        .binding
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_main_lane_claim_binding_digest).is_err());

    let mut tampered_opening_claim = proof.clone();
    tampered_opening_claim
        .claim
        .opening
        .terminal
        .execution_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_opening_claim).is_err());

    let mut tampered_opening_stage_claim_binding = proof.clone();
    tampered_opening_stage_claim_binding
        .claim
        .opening
        .stages
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_opening_stage_claim_binding).is_err());

    let mut tampered_opening_terminal_claim_binding = proof.clone();
    tampered_opening_terminal_claim_binding
        .claim
        .opening
        .terminal
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_opening_terminal_claim_binding).is_err());

    let mut tampered_joint_opening_claim = proof.clone();
    tampered_joint_opening_claim
        .claim
        .joint_opening
        .binding
        .main_lane_claim_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_joint_opening_claim).is_err());

    let mut tampered_root0_claim = proof.clone();
    tampered_root0_claim.claim.root0.terminal.root0_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_root0_claim).is_err());

    let mut tampered_joint_opening_claim_binding = proof.clone();
    tampered_joint_opening_claim_binding
        .claim
        .joint_opening
        .binding
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_joint_opening_claim_binding).is_err());

    let mut tampered_root0_stage_claim_binding = proof.clone();
    tampered_root0_stage_claim_binding.claim.root0.stages.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_root0_stage_claim_binding).is_err());

    let mut tampered_root0_terminal_claim_binding = proof.clone();
    tampered_root0_terminal_claim_binding
        .claim
        .root0
        .terminal
        .digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_root0_terminal_claim_binding).is_err());

    let mut tampered_bundle = proof.clone();
    tampered_bundle.claim.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_bundle).is_err());

    let mut tampered_main_lane_surface_digest = proof.clone();
    tampered_main_lane_surface_digest.kernel.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_main_lane_surface_digest).is_err());

    let mut tampered_main_lane_binding = proof.clone();
    tampered_main_lane_binding.kernel.main_lane.binding.digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_main_lane_binding).is_err());

    let mut tampered_kernel_claim_bundle = proof.clone();
    tampered_kernel_claim_bundle
        .kernel
        .kernel_claims
        .summary
        .terminal
        .final_state_digest[0] ^= 1;
    assert!(audit_rv32im_proof(&tampered_kernel_claim_bundle).is_err());
}

#[test]
fn rv32im_proof_rejects_export_surface_mismatches_with_recomputed_digests() {
    let _serial = proof_test_guard();
    let (_witness, proof) = NATIVE_LOGIC_COMPARE_AUDIT_PROOF.clone();

    let mut tampered_stage_summary = proof.clone();
    tampered_stage_summary
        .kernel
        .stages
        .summary
        .stage2_register_read_count += 1;
    tampered_stage_summary.kernel.stages.summary.digest =
        stage_witness_summary_digest(&tampered_stage_summary.kernel.stages.summary);
    tampered_stage_summary.kernel.stages.digest =
        stage_witness_projection_digest(&tampered_stage_summary.kernel.stages);
    tampered_stage_summary.kernel.digest = kernel_proof_bundle_digest(&tampered_stage_summary.kernel);
    assert!(audit_rv32im_proof(&tampered_stage_summary).is_err());

    let mut tampered_stage_claim_summary = proof.clone();
    tampered_stage_claim_summary
        .kernel
        .stage_claims
        .summary
        .stage2_digest[0] ^= 1;
    tampered_stage_claim_summary
        .kernel
        .stage_claims
        .summary
        .digest = stage_claim_digest_bundle_digest(&tampered_stage_claim_summary.kernel.stage_claims.summary);
    tampered_stage_claim_summary.kernel.stage_claims.digest =
        stage_claim_proof_digest(&tampered_stage_claim_summary.kernel.stage_claims);
    tampered_stage_claim_summary.kernel.digest = kernel_proof_bundle_digest(&tampered_stage_claim_summary.kernel);
    assert!(audit_rv32im_proof(&tampered_stage_claim_summary).is_err());
}
