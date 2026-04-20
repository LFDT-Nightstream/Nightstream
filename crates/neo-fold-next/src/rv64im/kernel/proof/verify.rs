//! Owns verifier-side bridge checks between the public RV64IM proof API and the private simple-kernel export.

use super::main_lane_artifact::build_simple_kernel_main_lane_artifact_from_summary;
use super::perf_diagnostics::{Rv64imPublicProofVerifyPerf, SimpleKernelBuildPerf};
use super::proof_api::Rv64imProof;
use super::proof_witness::{
    kernel_claim_summary_bundle_from_claims, kernel_opening_binding_bundle_from_opening,
    stage_claim_summary_bundle_from_claims, stage_package_summary_bundle_from_packages,
    verify_kernel_claim_packaged_proof, verify_stage_claim_packaged_proof,
};
use super::simple::{
    build_public_simple_kernel_output_and_witness_with_perf, verify_root_main_lane_packaged_proof_with_public_rows,
    PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar,
};
use super::stage_artifacts::verify_public_kernel_opening_bundle_with_perf;
use super::stage_package_perf::verify_stage_package_bundle_with_perf;
use super::{
    build_main_lane_surface, AjtaiFamilyKind, RootLaneColumns, RootLaneCommitmentSummaryArtifact,
    Rv64imMainLaneSurface, SelectedOpeningRef, SimpleKernelError, SimpleKernelPublicInput,
};
use std::time::Instant;

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn validate_main_lane_opening_ref(
    surface: &Rv64imMainLaneSurface,
    reference: &SelectedOpeningRef,
    expected_logical_index: u64,
) -> Result<(), SimpleKernelError> {
    if reference.id.object.family != AjtaiFamilyKind::RootMainLaneColumns
        || reference.id.object.commitment_digest != surface.family_digest
        || reference.id.object.digest != surface.object_digest
        || reference.id.logical_index != expected_logical_index
        || reference.id.object.expected_digest() != reference.id.object.digest
        || reference.id.expected_digest() != reference.id.digest
        || reference.expected_digest() != reference.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane surface selected opening ref is inconsistent".into(),
        ));
    }
    Ok(())
}

fn validate_root_lane_columns_opening_ref(
    columns: &RootLaneColumns,
    reference: &SelectedOpeningRef,
    expected_logical_index: u64,
) -> Result<(), SimpleKernelError> {
    if reference.id.object.family != AjtaiFamilyKind::RootMainLaneColumns
        || reference.id.object.commitment_digest != columns.family_digest
        || reference.id.object.digest != columns.object.digest
        || reference.id.logical_index != expected_logical_index
        || reference.id.object.expected_digest() != reference.id.object.digest
        || reference.id.expected_digest() != reference.id.digest
        || reference.expected_digest() != reference.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane columns selected opening ref is inconsistent".into(),
        ));
    }
    Ok(())
}

fn validate_root_lane_commitment_selected_ref(
    summary: &RootLaneCommitmentSummaryArtifact,
    reference: &SelectedOpeningRef,
    expected_logical_index: u64,
) -> Result<(), SimpleKernelError> {
    if reference.id.object.family != AjtaiFamilyKind::RootMainLaneCommittedRows
        || reference.id.object.commitment_digest != summary.commitments.digest
        || reference.id.logical_index != expected_logical_index
        || reference.id.object.expected_digest() != reference.id.object.digest
        || reference.id.expected_digest() != reference.id.digest
        || reference.expected_digest() != reference.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane commitment selected opening ref is inconsistent".into(),
        ));
    }
    Ok(())
}

fn validate_public_claim_digests(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    if proof.claim.accepted.statement.digest != proof.claim.accepted.statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof statement binding digest mismatch".into(),
        ));
    }
    if proof.claim.accepted.main_lane.digest != proof.claim.accepted.main_lane.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof main-lane binding digest mismatch".into(),
        ));
    }
    if proof.claim.accepted.terminal.digest != proof.claim.accepted.terminal.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof terminal binding digest mismatch".into(),
        ));
    }
    if proof.claim.accepted.digest != proof.claim.accepted.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof claim digest mismatch".into(),
        ));
    }
    if proof.claim.main_lane.binding.digest != proof.claim.main_lane.binding.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.main_lane.digest != proof.claim.main_lane.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane claim digest mismatch".into(),
        ));
    }
    if proof.claim.opening.stages.digest != proof.claim.opening.stages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening stage-claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.opening.terminal.digest != proof.claim.opening.terminal.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening terminal-claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.opening.digest != proof.claim.opening.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening claim digest mismatch".into(),
        ));
    }
    if proof.claim.joint_opening.binding.digest != proof.claim.joint_opening.binding.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM joint-opening claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.joint_opening.digest != proof.claim.joint_opening.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM joint-opening claim digest mismatch".into(),
        ));
    }
    if proof.claim.root0.stages.digest != proof.claim.root0.stages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root0 stage-claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.root0.terminal.digest != proof.claim.root0.terminal.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root0 terminal-claim binding digest mismatch".into(),
        ));
    }
    if proof.claim.root0.digest != proof.claim.root0.expected_digest() {
        return Err(SimpleKernelError::Bridge("RV64IM root0 claim digest mismatch".into()));
    }
    if proof.claim.digest != proof.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel claim bundle digest mismatch".into(),
        ));
    }
    if proof.statement.digest != proof.statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM proof statement digest mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_public_bundle_digests(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    if proof.kernel.trace.shape.digest != proof.kernel.trace.shape.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace shape bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.trace.digest != proof.kernel.trace.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace proof bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.stages.summary.digest != proof.kernel.stages.summary.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-witness summary bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.stages.digest != proof.kernel.stages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-witness proof bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.stage_claims.summary.digest != proof.kernel.stage_claims.summary.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-claim digest bundle mismatch".into(),
        ));
    }
    if proof.kernel.stage_claims.digest != proof.kernel.stage_claims.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-claim proof bundle digest mismatch".into(),
        ));
    }
    let expected_stage_claims = stage_claim_summary_bundle_from_claims(&proof.kernel.stage_claims.claims);
    if proof.kernel.stage_claims.summary != expected_stage_claims.summary {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-claim summary bundle does not match the carried stage claims".into(),
        ));
    }
    verify_stage_claim_packaged_proof(&proof.kernel.stage_claims.claims, &proof.kernel.stage_claims.packaged)?;
    if proof.kernel.stage_packages.summary.digest != proof.kernel.stage_packages.summary.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-package digest bundle mismatch".into(),
        ));
    }
    if proof.kernel.stage_packages.digest != proof.kernel.stage_packages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-package proof bundle digest mismatch".into(),
        ));
    }
    let expected_stage_packages = stage_package_summary_bundle_from_packages(&proof.kernel.stage_packages.packages);
    if proof.kernel.stage_packages.summary != expected_stage_packages {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-package summary bundle does not match the carried stage packages".into(),
        ));
    }
    if proof.kernel.kernel_opening.bindings.digest != proof.kernel.kernel_opening.bindings.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening binding bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.kernel_opening.digest != proof.kernel.kernel_opening.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening proof bundle digest mismatch".into(),
        ));
    }
    let expected_kernel_opening_bindings =
        kernel_opening_binding_bundle_from_opening(&proof.kernel.kernel_opening.opening);
    if proof.kernel.kernel_opening.opening_digest != proof.kernel.kernel_opening.opening.digest
        || proof.kernel.kernel_opening.bindings != expected_kernel_opening_bindings
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening summary bundle does not match the carried opening proof".into(),
        ));
    }
    if proof.kernel.kernel_claims.digest != proof.kernel.kernel_claims.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-claim proof bundle digest mismatch".into(),
        ));
    }
    let expected_kernel_claims = kernel_claim_summary_bundle_from_claims(&proof.kernel.kernel_claims.claims);
    if proof.kernel.kernel_claims.summary != expected_kernel_claims.summary {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-claim summary bundle does not match the carried kernel claims".into(),
        ));
    }
    verify_kernel_claim_packaged_proof(&proof.kernel.kernel_claims.claims, &proof.kernel.kernel_claims.packaged)?;
    if proof.kernel.root_lane_columns.object.expected_digest() != proof.kernel.root_lane_columns.object.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane columns object digest mismatch".into(),
        ));
    }
    if proof.kernel.root_lane_columns.digest != proof.kernel.root_lane_columns.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane columns bundle digest mismatch".into(),
        ));
    }
    if proof.kernel.root_lane_commitment.digest != proof.kernel.root_lane_commitment.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane commitment artifact digest mismatch".into(),
        ));
    }
    if let Some(reference) = &proof.kernel.root_lane_commitment.first_selected_row {
        validate_root_lane_commitment_selected_ref(&proof.kernel.root_lane_commitment, reference, 0)?;
    }
    if let Some(reference) = &proof.kernel.root_lane_commitment.last_selected_row {
        let expected_logical_index = proof.kernel.root_lane_commitment.time_len.saturating_sub(1);
        validate_root_lane_commitment_selected_ref(
            &proof.kernel.root_lane_commitment,
            reference,
            expected_logical_index,
        )?;
    }
    if let Some(reference) = &proof.kernel.root_lane_columns.first_row {
        validate_root_lane_columns_opening_ref(&proof.kernel.root_lane_columns, reference, 0)?;
    }
    if let Some(reference) = &proof.kernel.root_lane_columns.last_row {
        let expected_logical_index = proof.kernel.root_lane_columns.time_len.saturating_sub(1);
        validate_root_lane_columns_opening_ref(&proof.kernel.root_lane_columns, reference, expected_logical_index)?;
    }
    let derived_main_lane_surface = build_main_lane_surface(&proof.kernel.root_lane_columns);
    if proof.kernel.main_lane.binding.digest != proof.kernel.main_lane.binding.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane proof binding digest mismatch".into(),
        ));
    }
    if proof.kernel.main_lane.digest != proof.kernel.main_lane.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane proof bundle digest mismatch".into(),
        ));
    }
    if let Some(reference) = &derived_main_lane_surface.first_public_step {
        validate_main_lane_opening_ref(&derived_main_lane_surface, reference, 0)?;
    }
    if let Some(reference) = &derived_main_lane_surface.last_public_step {
        let expected_logical_index = derived_main_lane_surface
            .public_step_count
            .saturating_sub(1);
        validate_main_lane_opening_ref(&derived_main_lane_surface, reference, expected_logical_index)?;
    }
    if proof.kernel.digest != proof.kernel.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel proof bundle digest mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_public_bundle_bindings(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    let derived_main_lane_surface = build_main_lane_surface(&proof.kernel.root_lane_columns);
    let expected_initial_pc = proof
        .witness
        .trace
        .trace
        .execution_rows
        .first()
        .map(|row| row.pc)
        .unwrap_or(proof.statement.final_pc);
    if proof.claim.joint_opening.binding.proof_statement_digest != proof.statement.digest
        || proof.claim.joint_opening.binding.main_lane_claim_digest != proof.claim.main_lane.digest
        || proof
            .claim
            .joint_opening
            .binding
            .kernel_opening_claim_digest
            != proof.claim.opening.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM joint-opening claim does not bind the expected public claims".into(),
        ));
    }
    if proof.claim.accepted.statement.proof_statement_digest != proof.statement.digest
        || proof.claim.accepted.statement.kernel_opening_digest != proof.kernel.kernel_opening.digest
        || proof.claim.accepted.main_lane.main_lane_bundle_digest != proof.kernel.main_lane.digest
        || proof.claim.accepted.terminal.final_state_digest != proof.kernel.kernel_claims.final_state_digest()
        || proof.claim.accepted.terminal.final_pc != proof.statement.final_pc
        || proof.claim.accepted.terminal.halted != proof.statement.halted
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof claim does not bind the expected public statement and proof digests".into(),
        ));
    }
    if proof.statement.stage_claims_digest != proof.kernel.stage_claims.digest
        || proof.statement.fold_schedule != proof.kernel.main_lane.fold_schedule()
        || proof.statement.chunk_count != proof.kernel.main_lane.chunk_count()
        || proof.statement.stage_packages_digest != proof.kernel.stage_packages.digest
        || proof.statement.kernel_opening_digest != proof.kernel.kernel_opening.digest
        || proof.statement.prepared_step_bindings_digest != proof.kernel.kernel_claims.prepared_step_bindings_digest()
        || proof.statement.execution_digest != proof.kernel.trace.execution_digest()
        || proof.statement.final_state_digest != proof.kernel.kernel_claims.final_state_digest()
        || proof.statement.transcript_final_digest != proof.kernel.kernel_claims.transcript_final_digest()
        || proof.statement.initial_pc != expected_initial_pc
        || proof.statement.final_pc != proof.kernel.kernel_claims.final_pc()
        || proof.statement.halted != proof.kernel.kernel_claims.halted()
        || proof.statement.main_lane_surface_digest != derived_main_lane_surface.digest
        || proof.statement.root_lane_columns_digest != proof.kernel.root_lane_columns.digest
        || proof.claim.main_lane.binding.main_lane_bundle_digest != proof.kernel.main_lane.digest
        || derived_main_lane_surface.public_step_count != proof.statement.public_step_count
        || proof.kernel.root_lane_columns.object.family != AjtaiFamilyKind::RootMainLaneColumns
        || proof.kernel.root_lane_columns.object.commitment_digest != proof.kernel.root_lane_columns.family_digest
        || proof.kernel.root_lane_columns.row_width != derived_main_lane_surface.row_width
        || proof.kernel.root_lane_columns.time_len != derived_main_lane_surface.public_step_count
        || proof.kernel.root_lane_columns.time_len != proof.kernel.trace.shape.execution_row_count
        || proof.kernel.root_lane_columns.column_digests.len() as u64 != proof.kernel.root_lane_columns.row_width
        || proof.kernel.root_lane_commitment.time_len != proof.kernel.root_lane_columns.time_len
        || proof
            .kernel
            .root_lane_commitment
            .commitments
            .commitment_count
            != proof.kernel.root_lane_columns.row_width
        || derived_main_lane_surface.object_digest != proof.kernel.root_lane_columns.object.digest
        || derived_main_lane_surface.family_digest != proof.kernel.root_lane_columns.family_digest
        || derived_main_lane_surface.first_public_step != proof.kernel.root_lane_columns.first_row
        || derived_main_lane_surface.last_public_step != proof.kernel.root_lane_columns.last_row
        || proof.kernel.main_lane.root_lane_columns_digest() != proof.kernel.root_lane_columns.digest
        || proof.kernel.main_lane.root_lane_commitment_digest() != proof.kernel.root_lane_commitment.digest
        || proof.kernel.main_lane.chunk_count() == 0
        || proof.kernel.main_lane.public_step_count() != proof.kernel.root_lane_columns.time_len
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane claim does not bind the expected public main-lane proof bundle".into(),
        ));
    }
    if proof.claim.opening.stages.stage_claims_digest != proof.kernel.stage_claims.digest
        || proof.claim.opening.stages.stage_packages_digest != proof.kernel.stage_packages.digest
        || proof.claim.opening.stages.kernel_opening_digest != proof.kernel.kernel_opening.digest
        || proof.claim.opening.terminal.prepared_step_bindings_digest
            != proof.kernel.kernel_claims.prepared_step_bindings_digest()
        || proof.claim.opening.terminal.execution_digest != proof.kernel.kernel_claims.execution_digest()
        || proof.claim.opening.terminal.transcript_final_digest != proof.kernel.kernel_claims.transcript_final_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening claim does not bind the expected stage and kernel terminal digests".into(),
        ));
    }
    if proof.claim.root0.stages.stage1_digest != proof.kernel.kernel_claims.claims.kernel.stage1_digest
        || proof.claim.root0.stages.stage2_digest != proof.kernel.kernel_claims.claims.kernel.stage2_digest
        || proof.claim.root0.stages.stage3_digest != proof.kernel.kernel_claims.claims.kernel.stage3_digest
        || proof.claim.root0.terminal.root0_digest != proof.kernel.kernel_claims.root0_digest()
        || proof.claim.root0.terminal.execution_digest != proof.kernel.kernel_claims.execution_digest()
        || proof.claim.root0.terminal.final_state_digest != proof.kernel.kernel_claims.final_state_digest()
        || proof.claim.root0.terminal.transcript_final_digest != proof.kernel.kernel_claims.transcript_final_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root0 claim does not bind the expected carried kernel claim bundle".into(),
        ));
    }
    if proof.kernel.trace.execution_digest != proof.kernel.kernel_claims.execution_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace proof bundle execution digest does not match the kernel-claim proof bundle".into(),
        ));
    }
    if proof.kernel.stages.summary.stage1_row_count != proof.kernel.trace.shape.execution_row_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-witness proof bundle stage1 count does not match the trace row count".into(),
        ));
    }
    if proof.kernel.stages.summary.stage3_continuity_count != proof.kernel.trace.shape.real_row_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-witness proof bundle continuity count does not match the real trace row count".into(),
        ));
    }
    if proof.kernel.stages.summary.stage3_halted != proof.kernel.kernel_claims.halted() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-witness proof bundle halted flag does not match the kernel-claim proof bundle".into(),
        ));
    }
    Ok(())
}

fn validate_public_witness_bundle_digests(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    if proof.witness.digest != proof.witness.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM proof witness bundle digest mismatch".into(),
        ));
    }
    if proof.witness.trace.shape.digest != proof.witness.trace.shape.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace witness shape bundle digest mismatch".into(),
        ));
    }
    if proof.witness.trace.digest != proof.witness.trace.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace witness bundle digest mismatch".into(),
        ));
    }
    if proof.witness.stages.summary.digest != proof.witness.stages.summary.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage witness summary bundle digest mismatch".into(),
        ));
    }
    if proof.witness.stages.digest != proof.witness.stages.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage witness proof bundle digest mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_public_witness_bindings(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    if proof.witness.root_params_id != proof.kernel.root_params_id
        || proof.witness.root_params_id != proof.statement.root_params_id
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM proof witness bundle root-context binding mismatch".into(),
        ));
    }
    if proof.witness.trace.projection() != proof.kernel.trace {
        return Err(SimpleKernelError::Bridge(
            "RV64IM trace witness bundle does not match the carried public trace projection".into(),
        ));
    }
    if proof.witness.stages.projection_bundle() != proof.kernel.stages {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage witness bundle does not match the carried public stage projection".into(),
        ));
    }
    if proof.witness.stage_claims.summary != proof.kernel.stage_claims.summary
        || proof.witness.stage_claims.digest != proof.kernel.stage_claims.digest
        || proof.witness.stage_claims.claims.digest != proof.kernel.stage_claims.claims.digest
        || proof.witness.stage_claims.packaged.statement.digest != proof.kernel.stage_claims.packaged.statement.digest
        || proof.witness.stage_claims.packaged.proof.proof_digest
            != proof.kernel.stage_claims.packaged.proof.proof_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-claim proof bundle does not match the carried proof witness bundle".into(),
        ));
    }
    if proof.witness.stage_packages.summary != proof.kernel.stage_packages.summary
        || proof.witness.stage_packages.digest != proof.kernel.stage_packages.digest
        || proof.witness.stage_packages.packages.digest != proof.kernel.stage_packages.packages.digest
        || proof.witness.stage_packages.packages.stage1.digest != proof.kernel.stage_packages.packages.stage1.digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage1
            .packaged
            .statement
            .digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage1
                .packaged
                .statement
                .digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage1
            .packaged
            .proof
            .proof_digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage1
                .packaged
                .proof
                .proof_digest
        || proof.witness.stage_packages.packages.stage2.digest != proof.kernel.stage_packages.packages.stage2.digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage2
            .packaged
            .statement
            .digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage2
                .packaged
                .statement
                .digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage2
            .packaged
            .proof
            .proof_digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage2
                .packaged
                .proof
                .proof_digest
        || proof.witness.stage_packages.packages.stage3.digest != proof.kernel.stage_packages.packages.stage3.digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage3
            .packaged
            .statement
            .digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage3
                .packaged
                .statement
                .digest
        || proof
            .witness
            .stage_packages
            .packages
            .stage3
            .packaged
            .proof
            .proof_digest
            != proof
                .kernel
                .stage_packages
                .packages
                .stage3
                .packaged
                .proof
                .proof_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage-package proof bundle does not match the carried proof witness bundle".into(),
        ));
    }
    if proof.witness.kernel_opening.opening_digest != proof.kernel.kernel_opening.opening_digest
        || proof.witness.kernel_opening.bindings != proof.kernel.kernel_opening.bindings
        || proof.witness.kernel_opening.digest != proof.kernel.kernel_opening.digest
        || proof.witness.kernel_opening.opening.digest != proof.kernel.kernel_opening.opening.digest
        || proof.witness.kernel_opening.opening.bindings.digest != proof.kernel.kernel_opening.opening.bindings.digest
        || proof
            .witness
            .kernel_opening
            .opening
            .bindings
            .packaged
            .statement
            .digest
            != proof
                .kernel
                .kernel_opening
                .opening
                .bindings
                .packaged
                .statement
                .digest
        || proof
            .witness
            .kernel_opening
            .opening
            .bindings
            .packaged
            .proof
            .proof_digest
            != proof
                .kernel
                .kernel_opening
                .opening
                .bindings
                .packaged
                .proof
                .proof_digest
        || proof.witness.kernel_opening.opening.prepared_steps.digest
            != proof.kernel.kernel_opening.opening.prepared_steps.digest
        || proof
            .witness
            .kernel_opening
            .opening
            .prepared_steps
            .packaged
            .statement
            .digest
            != proof
                .kernel
                .kernel_opening
                .opening
                .prepared_steps
                .packaged
                .statement
                .digest
        || proof
            .witness
            .kernel_opening
            .opening
            .prepared_steps
            .packaged
            .proof
            .proof_digest
            != proof
                .kernel
                .kernel_opening
                .opening
                .prepared_steps
                .packaged
                .proof
                .proof_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-opening proof bundle does not match the carried proof witness bundle".into(),
        ));
    }
    if proof.witness.kernel_claims.summary != proof.kernel.kernel_claims.summary
        || proof.witness.kernel_claims.digest != proof.kernel.kernel_claims.digest
        || proof.witness.kernel_claims.claims != proof.kernel.kernel_claims.claims
        || proof.witness.kernel_claims.packaged.statement.digest != proof.kernel.kernel_claims.packaged.statement.digest
        || proof.witness.kernel_claims.packaged.proof.proof_digest
            != proof.kernel.kernel_claims.packaged.proof.proof_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM kernel-claim proof bundle does not match the carried proof witness bundle".into(),
        ));
    }
    if proof.witness.root_lane_columns != proof.kernel.root_lane_columns {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane columns do not match the carried proof witness bundle".into(),
        ));
    }
    if proof.witness.root_lane_commitment != proof.kernel.root_lane_commitment {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root-lane commitment does not match the carried proof witness bundle".into(),
        ));
    }
    Ok(())
}

fn public_kernel_from_proof(proof: &Rv64imProof) -> PublicSimpleKernelOutput {
    PublicSimpleKernelOutput {
        trace: proof.kernel.trace.clone(),
        stages: proof.kernel.stages.clone(),
        stage_claims: proof.kernel.stage_claims.claims.clone(),
        stage_packages: proof.kernel.stage_packages.packages.clone(),
        kernel_opening: proof.kernel.kernel_opening.opening.clone(),
        kernel_claims: proof.kernel.kernel_claims.claims.clone(),
        root_lane_columns: proof.kernel.root_lane_columns.clone(),
        root_lane_commitment: proof.kernel.root_lane_commitment.clone(),
    }
}

fn public_sidecar_from_proof(proof: &Rv64imProof) -> PublicSimpleKernelWitnessSidecar {
    PublicSimpleKernelWitnessSidecar {
        trace: proof.witness.trace.trace.clone(),
        stages: proof.witness.stages.stages.clone(),
    }
}

fn rebuild_public_kernel_from_input(
    input: &SimpleKernelPublicInput,
    proof: &Rv64imProof,
) -> Result<
    (
        (PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar),
        SimpleKernelBuildPerf,
    ),
    SimpleKernelError,
> {
    let ((kernel, sidecar), perf) = build_public_simple_kernel_output_and_witness_with_perf(input)?;

    if proof.kernel.trace != kernel.trace {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof trace summary does not match the canonical trace summary".into(),
        ));
    }
    if proof.kernel.stages != kernel.stages {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof stage summary does not match the canonical stage summary".into(),
        ));
    }
    let expected_stage_claims = stage_claim_summary_bundle_from_claims(&kernel.stage_claims);
    if proof.kernel.stage_claims.summary_bundle() != expected_stage_claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof stage-claim summary does not match the canonical stage claims".into(),
        ));
    }
    if proof.kernel.stage_claims.claims != kernel.stage_claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof stage claims do not match the canonical stage claims".into(),
        ));
    }
    let expected_stage_packages = stage_package_summary_bundle_from_packages(&kernel.stage_packages);
    if proof.kernel.stage_packages.summary != expected_stage_packages {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof stage-package summary does not match the canonical stage packages".into(),
        ));
    }
    let expected_kernel_opening_bindings = kernel_opening_binding_bundle_from_opening(&kernel.kernel_opening);
    if proof.kernel.kernel_opening.opening_digest != kernel.kernel_opening.digest
        || proof.kernel.kernel_opening.bindings != expected_kernel_opening_bindings
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof kernel-opening summary does not match the canonical kernel opening".into(),
        ));
    }
    let expected_kernel_claims = kernel_claim_summary_bundle_from_claims(&kernel.kernel_claims);
    if proof.kernel.kernel_claims.summary_bundle() != expected_kernel_claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof kernel-claim summary does not match the canonical kernel claims".into(),
        ));
    }
    if proof.kernel.kernel_claims.claims != kernel.kernel_claims {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof kernel claims do not match the canonical kernel claims".into(),
        ));
    }
    if proof.kernel.root_lane_columns != kernel.root_lane_columns {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof root-lane columns do not match the canonical committed columns".into(),
        ));
    }
    if proof.kernel.root_lane_commitment != kernel.root_lane_commitment {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof root-lane commitment does not match the canonical commitment artifact".into(),
        ));
    }
    let expected_main_lane = build_simple_kernel_main_lane_artifact_from_summary(
        &kernel.root_lane_columns,
        &kernel.root_lane_commitment,
        proof.statement.fold_schedule,
    )?;
    if proof.kernel.main_lane.root_lane_columns_digest() != expected_main_lane.binding.root_lane_columns_digest
        || proof.kernel.main_lane.root_lane_commitment_digest()
            != expected_main_lane.binding.root_lane_commitment_digest
        || proof.kernel.main_lane.fold_schedule() != expected_main_lane.binding.fold_schedule
        || proof.kernel.main_lane.chunk_count() != expected_main_lane.binding.chunk_count
        || proof.kernel.main_lane.public_step_count() != expected_main_lane.binding.public_step_count
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main-lane proof binding does not match the accepted root-lane artifact".into(),
        ));
    }
    Ok(((kernel, sidecar), perf))
}

fn finalize_public_proof_verify_with_perf(
    proof: &Rv64imProof,
    kernel: PublicSimpleKernelOutput,
    sidecar: PublicSimpleKernelWitnessSidecar,
    public_kernel_build: SimpleKernelBuildPerf,
    summary_consistency_ms: f64,
    total_started: Instant,
) -> Result<(PublicSimpleKernelOutput, Rv64imPublicProofVerifyPerf), SimpleKernelError> {
    let root_main_lane_started = Instant::now();
    let root_main_lane = verify_root_main_lane_packaged_proof_with_public_rows(
        &sidecar.trace.execution_rows,
        &proof.kernel.main_lane.packaged,
    )?;
    let root_main_lane_proof_ms = millis_since(root_main_lane_started);

    let stage_package_started = Instant::now();
    verify_stage_package_bundle_with_perf(
        &sidecar.stages.stage1,
        &sidecar.stages.stage2,
        &sidecar.stages.stage3,
        &proof.witness.stage_packages.packages,
        &proof.witness.stage_claims.claims,
    )?;
    let stage_package_verify_ms = millis_since(stage_package_started);

    let kernel_opening_started = Instant::now();
    verify_public_kernel_opening_bundle_with_perf(
        &proof.witness.kernel_opening.opening,
        &proof.witness.stage_claims.claims,
        &proof.witness.stage_packages.packages,
        &proof.witness.kernel_claims.claims,
        &proof.witness.root_lane_commitment,
    )?;
    let kernel_opening_verify_ms = millis_since(kernel_opening_started);

    Ok((
        kernel,
        Rv64imPublicProofVerifyPerf {
            public_claim_digests_ms: 0.0,
            public_bundle_digests_ms: 0.0,
            public_bundle_bindings_ms: 0.0,
            native_stage_bundle_verify_ms: 0.0,
            public_kernel_build,
            root_execution_verify_ms: 0.0,
            root_main_lane_proof_ms,
            root_main_lane,
            stage_package_verify_ms,
            accepted_stage_package: Default::default(),
            accepted_root_execution: Default::default(),
            kernel_opening_verify_ms,
            summary_consistency_ms,
            total_ms: millis_since(total_started),
        },
    ))
}

pub(super) fn verify_kernel_output_from_public_proof_with_perf(
    proof: &Rv64imProof,
) -> Result<
    (
        (PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar),
        Rv64imPublicProofVerifyPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let claim_digests_started = Instant::now();
    validate_public_claim_digests(proof)?;
    let public_claim_digests_ms = millis_since(claim_digests_started);

    let bundle_digests_started = Instant::now();
    validate_public_bundle_digests(proof)?;
    let public_bundle_digests_ms = millis_since(bundle_digests_started);

    let bundle_bindings_started = Instant::now();
    validate_public_bundle_bindings(proof)?;
    let public_bundle_bindings_ms = millis_since(bundle_bindings_started);

    let summary_consistency_started = Instant::now();
    validate_public_witness_bundle_digests(proof)?;
    validate_public_witness_bindings(proof)?;
    let summary_consistency_ms = millis_since(summary_consistency_started);

    let kernel = public_kernel_from_proof(proof);
    let sidecar = public_sidecar_from_proof(proof);
    let (kernel, mut perf) = finalize_public_proof_verify_with_perf(
        proof,
        kernel,
        sidecar.clone(),
        SimpleKernelBuildPerf::default(),
        summary_consistency_ms,
        total_started,
    )?;
    perf.public_claim_digests_ms = public_claim_digests_ms;
    perf.public_bundle_digests_ms = public_bundle_digests_ms;
    perf.public_bundle_bindings_ms = public_bundle_bindings_ms;
    Ok(((kernel, sidecar), perf))
}

pub(super) fn validate_public_proof_against_input_with_perf(
    input: &SimpleKernelPublicInput,
    proof: &Rv64imProof,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    if proof.statement.initial_pc != input.source.start_pc {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof initial pc does not match the public input entrypoint".into(),
        ));
    }
    let ((_, sidecar), mut perf) = verify_kernel_output_from_public_proof_with_perf(proof)?;
    let kernel_build_started = Instant::now();
    let ((_, rebuilt_sidecar), mut public_kernel_build) = rebuild_public_kernel_from_input(input, proof)?;
    public_kernel_build.total_ms = millis_since(kernel_build_started);
    let rebuilt_initial_pc = rebuilt_sidecar
        .trace
        .execution_rows
        .first()
        .map(|row| row.pc)
        .unwrap_or(proof.statement.final_pc);
    if proof.statement.initial_pc != rebuilt_initial_pc {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof initial pc does not match the canonical build from public input".into(),
        ));
    }
    if sidecar.trace != rebuilt_sidecar.trace || sidecar.stages != rebuilt_sidecar.stages {
        return Err(SimpleKernelError::Bridge(
            "RV64IM public proof witness does not match the canonical build from public input".into(),
        ));
    }
    perf.public_kernel_build = public_kernel_build;
    perf.total_ms = millis_since(total_started);
    Ok(perf)
}
