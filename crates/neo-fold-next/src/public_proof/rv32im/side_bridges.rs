//! Owns compact RV32IM Nightstream side-bundle bridge surfaces and their digest bindings.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::rv32im::kernel::{
    build_verified_stage3_claim_from_accepted_artifact, verify_transcript_record, RootLaneCommitmentSummaryArtifact,
    Rv32imAcceptedProofArtifact, Rv32imKernelExportProof, SimpleKernelError, Stage1VerifiedClaims,
    Stage2VerifiedClaims, Stage3VerifiedClaims, VerifiedTranscriptSurface,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideProofBundle {
    pub statement_core_digest: [u8; 32],
    pub transcript: VerifiedTranscriptSurface,
    pub stage1: Stage1VerifiedClaims,
    pub stage2: Stage2VerifiedClaims,
    pub stage3: Stage3VerifiedClaims,
    pub stage_claim_proof_bridge: Rv32imStageClaimProofBridge,
    pub kernel_opening_bridge: Rv32imKernelOpeningBridge,
    pub kernel_claim_bridge: Rv32imKernelClaimBridge,
    pub kernel_claim_proof_bridge: Rv32imKernelClaimProofBridge,
    pub main_lane_bridge: Rv32imMainLaneProofBridge,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelOpeningBridge {
    pub prepared_step_bindings: Rv32imPreparedStepBindingSummaryBridge,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
    pub bindings_opening_statement_digest: [u8; 32],
    pub bindings_opening_digest: [u8; 32],
    pub prepared_steps_opening_statement_digest: [u8; 32],
    pub prepared_steps_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imPreparedStepBindingSummaryBridge {
    pub binding_count: u64,
    pub first_binding_digest: Option<[u8; 32]>,
    pub last_binding_digest: Option<[u8; 32]>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imStageClaimProofBridge {
    pub packaged_statement_digest: [u8; 32],
    pub packaged_proof_digest: [u8; 32],
    pub stage_claim_proof_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelClaimBridge {
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub root0_digest: [u8; 32],
    pub kernel_claim_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imKernelClaimProofBridge {
    pub packaged_statement_digest: [u8; 32],
    pub packaged_proof_digest: [u8; 32],
    pub kernel_claim_proof_bundle_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imMainLaneProofBridge {
    pub main_lane_statement_digest: [u8; 32],
    pub main_lane_proof_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Rv32imSideProofBundle {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_proof_bundle");
        tr.append_message(b"neo.fold.next/nightstream/rv32im/side_proof_bundle/version", b"v1");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/statement_core_digest",
            &self.statement_core_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/transcript",
            &self.transcript.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage1",
            &self.stage1.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage2",
            &self.stage2.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage3",
            &self.stage3.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/stage_claim_proof_bridge",
            &self.stage_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_opening_bridge",
            &self.kernel_opening_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_claim_bridge",
            &self.kernel_claim_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/kernel_claim_proof_bridge",
            &self.kernel_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof_bundle/main_lane_bridge",
            &self.main_lane_bridge.digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelOpeningBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_step_bindings",
            &self.prepared_step_bindings.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/root_lane_commitment",
            &self.root_lane_commitment.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/bindings_opening_statement_digest",
            &self.bindings_opening_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/bindings_opening_digest",
            &self.bindings_opening_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_steps_opening_statement_digest",
            &self.prepared_steps_opening_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_opening_bridge/prepared_steps_opening_digest",
            &self.prepared_steps_opening_digest,
        );
        tr.digest32()
    }
}

impl Rv32imPreparedStepBindingSummaryBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/binding_count",
            &[self.binding_count],
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/first_present",
            &[self.first_binding_digest.is_some() as u64],
        );
        if let Some(digest) = self.first_binding_digest {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/first_binding_digest",
                &digest,
            );
        }
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/last_present",
            &[self.last_binding_digest.is_some() as u64],
        );
        if let Some(digest) = self.last_binding_digest {
            tr.append_message(
                b"neo.fold.next/nightstream/rv32im/prepared_step_binding_summary_bridge/last_binding_digest",
                &digest,
            );
        }
        tr.digest32()
    }
}

impl Rv32imStageClaimProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/packaged_statement_digest",
            &self.packaged_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/packaged_proof_digest",
            &self.packaged_proof_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage_claim_proof_bridge/stage_claim_proof_bundle_digest",
            &self.stage_claim_proof_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelClaimBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage1_digest",
            &self.stage1_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage2_digest",
            &self.stage2_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/stage3_digest",
            &self.stage3_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/root0_digest",
            &self.root0_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_bridge/kernel_claim_bundle_digest",
            &self.kernel_claim_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imKernelClaimProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/packaged_statement_digest",
            &self.packaged_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/packaged_proof_digest",
            &self.packaged_proof_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/kernel_claim_proof_bridge/kernel_claim_proof_bundle_digest",
            &self.kernel_claim_proof_bundle_digest,
        );
        tr.digest32()
    }
}

impl Rv32imMainLaneProofBridge {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge/main_lane_statement_digest",
            &self.main_lane_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/main_lane_proof_bridge/main_lane_proof_digest",
            &self.main_lane_proof_digest,
        );
        tr.digest32()
    }
}

pub(crate) fn validate_rv32im_side_proof_bundle_structure(
    bundle: &Rv32imSideProofBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof bundle digest mismatch".into(),
        ));
    }
    if bundle.transcript.digest != bundle.transcript.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried transcript surface digest mismatch".into(),
        ));
    }
    if bundle.stage1.digest != bundle.stage1.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried stage1 verified-claims digest mismatch".into(),
        ));
    }
    if bundle.stage2.digest != bundle.stage2.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried stage2 verified-claims digest mismatch".into(),
        ));
    }
    if bundle.stage3.digest != bundle.stage3.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried stage3 verified-claims digest mismatch".into(),
        ));
    }
    if bundle.stage_claim_proof_bridge.digest != bundle.stage_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried stage-claim proof bridge digest mismatch".into(),
        ));
    }
    if bundle.kernel_opening_bridge.digest != bundle.kernel_opening_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried kernel-opening bridge digest mismatch".into(),
        ));
    }
    if bundle.kernel_claim_bridge.digest != bundle.kernel_claim_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried kernel-claim bridge digest mismatch".into(),
        ));
    }
    if bundle.kernel_claim_proof_bridge.digest != bundle.kernel_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried kernel-claim proof bridge digest mismatch".into(),
        ));
    }
    if bundle.main_lane_bridge.digest != bundle.main_lane_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried main-lane proof bridge digest mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn build_rv32im_verified_side_claims_from_accepted_artifact_fast(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<
    (
        VerifiedTranscriptSurface,
        Stage1VerifiedClaims,
        Stage2VerifiedClaims,
        Stage3VerifiedClaims,
    ),
    crate::rv32im::kernel::SimpleKernelError,
> {
    if artifact.digest != artifact.expected_digest()
        || artifact.statement.digest != artifact.statement.expected_digest()
        || artifact.stage_claims.digest != artifact.stage_claims.expected_digest()
        || artifact.stage1.digest != artifact.stage1.expected_digest()
        || artifact.stage2.digest != artifact.stage2.expected_digest()
        || artifact.stage3.digest != artifact.stage3.expected_digest()
        || artifact.root_execution.digest != artifact.root_execution.expected_digest()
    {
        return Err(crate::rv32im::kernel::SimpleKernelError::Bridge(
            "RV32IM accepted-artifact side-bundle fast path digest mismatch".into(),
        ));
    }

    let transcript = verify_transcript_record(&artifact.transcript)?;
    if transcript.final_digest != artifact.statement.transcript_final_digest
        || transcript.final_digest != artifact.stage_claims.claims.transcript.commitment.digest
        || transcript.final_digest != artifact.stage_claims.claims.transcript.claim.final_digest
        || transcript.event_count != artifact.stage_claims.claims.transcript.claim.event_count
        || transcript.challenges.kernel_final_mix
            != artifact
                .stage_claims
                .claims
                .transcript
                .claim
                .kernel_final_mix
        || transcript.challenges.stage1_mix != artifact.stage_claims.claims.stage1.claim.mix
        || transcript.challenges.stage2_reg_mix != artifact.stage_claims.claims.stage2.claim.reg_mix
        || transcript.challenges.stage2_ram_mix != artifact.stage_claims.claims.stage2.claim.ram_mix
        || transcript.challenges.stage3_continuity_mix != artifact.stage_claims.claims.stage3.claim.continuity_mix
    {
        return Err(crate::rv32im::kernel::SimpleKernelError::Bridge(
            "RV32IM accepted-artifact side-bundle fast path transcript bindings mismatch".into(),
        ));
    }

    let stage1_surface = &artifact.stage_claims.claims.stage1.claim;
    let stage1_claim = &artifact.stage1.selected_opening.claim;
    if artifact.stage1.bytecode.digest != artifact.stage1.bytecode.expected_digest()
        || artifact.stage1.alu.digest != artifact.stage1.alu.expected_digest()
        || artifact.stage1.branch.digest != artifact.stage1.branch.expected_digest()
        || artifact.stage1.semantics.digest != artifact.stage1.semantics.expected_digest()
        || artifact.stage1.address_correctness.digest != artifact.stage1.address_correctness.expected_digest()
        || artifact.stage1.linkage.digest != artifact.stage1.linkage.expected_digest()
        || artifact.stage1.selected_opening.digest != artifact.stage1.selected_opening.expected_digest()
        || stage1_claim.digest != stage1_claim.expected_digest()
        || artifact.stage1.bytecode.rows_digest != artifact.stage_claims.claims.stage1.rows.rows_digest
        || artifact.stage1.address_correctness.rows_digest != artifact.stage_claims.claims.stage1.rows.rows_digest
        || artifact.stage1.linkage.rows_digest != artifact.stage_claims.claims.stage1.rows.rows_digest
        || artifact.stage1.semantics.row_bindings_digest != artifact.stage_claims.claims.stage1.rows.rows_digest
        || artifact.stage1.alu.sem_inputs_digest != artifact.stage1.branch.sem_inputs_digest
        || artifact.stage1.alu.sem_inputs_digest != artifact.stage1.linkage.sem_inputs_digest
        || artifact.stage1.alu.sem_inputs_digest != artifact.stage1.semantics.sem_inputs_digest
        || stage1_surface.row_count as u64 != stage1_claim.row_count
        || stage1_surface.effect_row_count as u64 != stage1_claim.effect_row_count
        || stage1_surface.commit_row_count as u64 != stage1_claim.commit_row_count
        || stage1_surface.real_row_count as u64 != stage1_claim.real_row_count
        || stage1_surface.preserves_x0_count as u64 != stage1_claim.preserves_x0_count
        || stage1_surface.mix != stage1_claim.mix
        || stage1_claim.rows_family_digest != artifact.stage_claims.claims.stage1.rows.rows_digest
        || stage1_claim.row_count != artifact.stage1.address_correctness.row_count
        || stage1_claim.effect_row_count != artifact.stage1.address_correctness.effect_row_count
        || stage1_claim.commit_row_count != artifact.stage1.address_correctness.commit_row_count
        || stage1_claim.real_row_count != artifact.stage1.address_correctness.real_row_count
        || stage1_claim.preserves_x0_count != artifact.stage1.address_correctness.preserves_x0_count
        || stage1_claim.first_trace_index != artifact.stage1.branch.first_trace_index
        || stage1_claim.first_trace_index != artifact.stage1.linkage.first_trace_index
        || stage1_claim.effect_trace_index != artifact.stage1.alu.effect_trace_index
        || stage1_claim.effect_trace_index != artifact.stage1.linkage.effect_trace_index
        || stage1_claim.commit_trace_index != artifact.stage1.alu.commit_trace_index
        || stage1_claim.commit_trace_index != artifact.stage1.linkage.commit_trace_index
        || stage1_claim.last_trace_index != artifact.stage1.branch.last_trace_index
        || stage1_claim.last_trace_index != artifact.stage1.linkage.last_trace_index
        || stage1_claim.mix != transcript.challenges.stage1_mix
        || artifact.stage1.linkage.mix != transcript.challenges.stage1_mix
    {
        return Err(crate::rv32im::kernel::SimpleKernelError::Bridge(
            "RV32IM accepted-artifact side-bundle fast path stage1 surface mismatch".into(),
        ));
    }
    let stage1 = Stage1VerifiedClaims {
        sem_inputs_digest: artifact.stage1.semantics.sem_inputs_digest,
        rows_digest: artifact.stage_claims.claims.stage1.rows.rows_digest,
        claim: stage1_claim.clone(),
        packaged_statement_digest: artifact.stage1.selected_opening.packaged.statement.digest,
        packaged_digest: artifact.stage1.selected_opening.digest,
        mix: transcript.challenges.stage1_mix,
        digest: [0; 32],
    };
    let stage1 = Stage1VerifiedClaims {
        digest: stage1.expected_digest(),
        ..stage1
    };

    let stage2_surface = &artifact.stage_claims.claims.stage2.claim;
    let stage2_claim = &artifact.stage2.selected_opening.claim;
    let ram_read_count = artifact
        .stage2
        .ram
        .events
        .iter()
        .filter(|event| matches!(event.kind, crate::rv32im::stage2::RamAccessKind::Read))
        .count() as u64;
    let ram_write_count = artifact.stage2.ram.events.len() as u64 - ram_read_count;
    if artifact.stage2.register.digest != artifact.stage2.register.expected_digest()
        || artifact.stage2.ram.digest != artifact.stage2.ram.expected_digest()
        || artifact.stage2.temporal.digest != artifact.stage2.temporal.expected_digest()
        || artifact.stage2.semantics.digest != artifact.stage2.semantics.expected_digest()
        || artifact.stage2.linkage.digest != artifact.stage2.linkage.expected_digest()
        || artifact.stage2.selected_opening.digest != artifact.stage2.selected_opening.expected_digest()
        || stage2_claim.digest != stage2_claim.expected_digest()
        || artifact.stage2.temporal.register_timeline_digest != artifact.stage2.register.timeline_digest
        || artifact.stage2.temporal.ram_timeline_digest != artifact.stage2.ram.timeline_digest
        || artifact.stage2.semantics.register_reads_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_reads_digest
        || artifact.stage2.semantics.register_writes_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_writes_digest
        || artifact.stage2.semantics.ram_events_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .ram_events_digest
        || artifact.stage2.semantics.twist_links_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .twist_links_digest
        || artifact.stage2.linkage.register_reads_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_reads_digest
        || artifact.stage2.linkage.register_writes_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_writes_digest
        || artifact.stage2.linkage.ram_events_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .ram_events_digest
        || artifact.stage2.linkage.twist_links_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .twist_links_digest
        || stage2_surface.register_read_count as u64 != stage2_claim.register_read_count
        || stage2_surface.register_write_count as u64 != stage2_claim.register_write_count
        || stage2_surface.ram_event_count as u64 != stage2_claim.ram_event_count
        || stage2_surface.twist_link_count as u64 != stage2_claim.twist_link_count
        || stage2_surface.ram_read_count as u64 != stage2_claim.ram_read_count
        || stage2_surface.ram_write_count as u64 != stage2_claim.ram_write_count
        || stage2_surface.reg_mix != stage2_claim.reg_mix
        || stage2_surface.ram_mix != stage2_claim.ram_mix
        || stage2_claim.register_reads_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_reads_digest
        || stage2_claim.register_writes_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .register_writes_digest
        || stage2_claim.ram_events_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .ram_events_digest
        || stage2_claim.twist_links_family_digest
            != artifact
                .stage_claims
                .claims
                .stage2
                .families
                .twist_links_digest
        || stage2_claim.register_read_count != artifact.stage2.register.reads.len() as u64
        || stage2_claim.register_write_count != artifact.stage2.register.writes.len() as u64
        || stage2_claim.ram_event_count != artifact.stage2.ram.events.len() as u64
        || stage2_claim.twist_link_count != artifact.stage2.temporal.twist_links.len() as u64
        || stage2_claim.ram_read_count != ram_read_count
        || stage2_claim.ram_write_count != ram_write_count
        || stage2_claim.reg_mix != transcript.challenges.stage2_reg_mix
        || stage2_claim.ram_mix != transcript.challenges.stage2_ram_mix
        || artifact.stage2.linkage.reg_mix != transcript.challenges.stage2_reg_mix
        || artifact.stage2.linkage.ram_mix != transcript.challenges.stage2_ram_mix
    {
        return Err(crate::rv32im::kernel::SimpleKernelError::Bridge(
            "RV32IM accepted-artifact side-bundle fast path stage2 surface mismatch".into(),
        ));
    }
    let stage2 = Stage2VerifiedClaims {
        register_timeline_digest: artifact.stage2.temporal.register_timeline_digest,
        ram_timeline_digest: artifact.stage2.temporal.ram_timeline_digest,
        twist_links_digest: artifact.stage2.temporal.twist_links_digest,
        claim: stage2_claim.clone(),
        packaged_statement_digest: artifact.stage2.selected_opening.packaged.statement.digest,
        packaged_digest: artifact.stage2.selected_opening.digest,
        reg_mix: transcript.challenges.stage2_reg_mix,
        ram_mix: transcript.challenges.stage2_ram_mix,
        digest: [0; 32],
    };
    let stage2 = Stage2VerifiedClaims {
        digest: stage2.expected_digest(),
        ..stage2
    };

    let stage3 = build_verified_stage3_claim_from_accepted_artifact(artifact, &transcript)?;

    Ok((transcript, stage1, stage2, stage3))
}

pub(super) fn build_rv32im_kernel_opening_bridge_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Rv32imKernelOpeningBridge {
    let mut bridge = Rv32imKernelOpeningBridge {
        prepared_step_bindings: build_rv32im_prepared_step_binding_summary_bridge(
            artifact.root_execution.prepared_step_bindings.binding_count,
            artifact
                .root_execution
                .prepared_step_bindings
                .first_binding_digest,
            artifact
                .root_execution
                .prepared_step_bindings
                .last_binding_digest,
        ),
        root_lane_commitment: artifact.root_lane_commitment.clone(),
        bindings_opening_statement_digest: artifact
            .kernel_opening
            .opening
            .bindings
            .packaged
            .statement
            .digest,
        bindings_opening_digest: artifact.kernel_opening.opening.bindings.digest,
        prepared_steps_opening_statement_digest: artifact
            .kernel_opening
            .opening
            .prepared_steps
            .packaged
            .statement
            .digest,
        prepared_steps_opening_digest: artifact.kernel_opening.opening.prepared_steps.digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}

fn build_rv32im_prepared_step_binding_summary_bridge(
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
) -> Rv32imPreparedStepBindingSummaryBridge {
    let mut bridge = Rv32imPreparedStepBindingSummaryBridge {
        binding_count,
        first_binding_digest,
        last_binding_digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}

pub(super) fn build_rv32im_stage_claim_proof_bridge_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Rv32imStageClaimProofBridge {
    let mut bridge = Rv32imStageClaimProofBridge {
        packaged_statement_digest: artifact.stage_claims.packaged.statement.digest,
        packaged_proof_digest: artifact.stage_claims.packaged.proof.proof_digest,
        stage_claim_proof_bundle_digest: artifact.stage_claims.digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}

pub(super) fn build_rv32im_kernel_claim_bridge_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Rv32imKernelClaimBridge {
    let kernel_summary = &artifact.kernel_claims.claims.kernel;
    let mut bridge = Rv32imKernelClaimBridge {
        stage1_digest: kernel_summary.stage1_digest,
        stage2_digest: kernel_summary.stage2_digest,
        stage3_digest: kernel_summary.stage3_digest,
        root0_digest: artifact.kernel_claims.root0_digest(),
        kernel_claim_bundle_digest: artifact.claim.digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}

pub(super) fn build_rv32im_kernel_claim_proof_bridge_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Rv32imKernelClaimProofBridge {
    let mut bridge = Rv32imKernelClaimProofBridge {
        packaged_statement_digest: artifact.kernel_claims.packaged.statement.digest,
        packaged_proof_digest: artifact.kernel_claims.packaged.proof.proof_digest,
        kernel_claim_proof_bundle_digest: artifact.kernel_claims.digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}

pub(super) fn build_rv32im_main_lane_proof_bridge_from_export_proof(
    kernel_export: &Rv32imKernelExportProof,
) -> Rv32imMainLaneProofBridge {
    let mut bridge = Rv32imMainLaneProofBridge {
        main_lane_statement_digest: kernel_export.source.main_lane.packaged.statement.digest,
        main_lane_proof_digest: kernel_export.source.main_lane.packaged.proof.proof_digest,
        digest: [0; 32],
    };
    bridge.digest = bridge.expected_digest();
    bridge
}
