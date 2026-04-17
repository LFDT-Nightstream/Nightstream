//! Owns accepted-proof staged verification and transcript replay for RV64IM.

use crate::finalize::public_chunk_digest;
use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::rv64im::kernel::root_lane_witness::{
    root_execution_public_step_digests, root_execution_row_chunk_routes_digest, root_execution_semantic_rows_digest,
    validate_prepared_step_binding_summary, validate_root_execution_main_lane_chunk_layout,
    validate_root_execution_row_chunk_routes, validate_root_execution_semantic_rows,
    validate_root_execution_semantics_refinement_summary, validate_root_row_local_ccs_acceptance_summary,
};
use crate::rv64im::stage1::{
    build_sem_inputs, build_stage1_semantics_proof, build_stage1_summary, sem_in_digest, sem_in_from_row,
    stage1_row_binding_from_row, stage1_row_bindings_digest, stage1_row_digest, verify_stage1_semantics, SemIn,
    Stage1ProofBundle, Stage1RowBinding, Stage1SemanticsProof,
};
use crate::rv64im::stage2::{
    build_stage2_summary, ram_events_family_digest, ram_timeline_digest, register_reads_family_digest,
    register_timeline_digest, register_writes_family_digest, twist_links_family_digest, twist_links_timeline_digest,
    verify_stage2_semantics_from_events, RamEvent, RegisterReadEvent, RegisterWriteEvent, Stage2ProofBundle,
    Stage2SemanticsProof, Stage2TemporalContext, TwistLinkEvent,
};
use crate::rv64im::stage3::{
    build_stage3_summary, continuity_event_digest, verify_stage3_semantics, ContinuityEvent, PcAdjacentBridge,
    Stage3ProofBundle, Stage3SemanticsProof, Stage3Summary,
};
use crate::rv64im::Rv64ExpandedRow;

use super::artifacts::digest_rows;
use super::perf_diagnostics::{
    AcceptedRootExecutionVerifyPerf, AcceptedStage1VerifyPerf, AcceptedStage2VerifyPerf,
    AcceptedStagePackageVerifyPerf, Rv64imPublicProofVerifyPerf,
};
use super::proof_accepted::Rv64imAcceptedProofArtifact;
use super::proof_api::{Rv64imMainLaneProofBundle, Rv64imProofStatement};
use super::proof_completeness::{build_step_composition_surface, canonical_kernel_soundness_accounting_surface};
use super::proof_witness::{
    kernel_export_claim_proof_from_bundle, verify_kernel_export_claim_packaged_proof,
    verify_stage_claim_packaged_proof, Rv64imKernelExportClaimProof, Rv64imKernelOpeningProofBundle,
    Rv64imStageClaimProofBundle, Rv64imStagePackageProofBundle,
};
use super::root_lane_witness::RootExecutionBundle;
use super::simple::{verify_root_main_lane_packaged_proof_with_verified_public_statement_with_perf, SimpleKernelError};
use super::simple_openings::{
    Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2PackagedOpeningProof, Stage2SelectedOpeningClaim,
    Stage3PackagedOpeningProof, Stage3SelectedOpeningClaim,
};
use super::stage1_canonical::build_stage1_selected_opening_claim_from_rows;
use super::stage2_canonical::build_stage2_selected_opening_claim_from_events;
use super::stage3_canonical::build_stage3_selected_opening_claim;
use super::stage_artifacts::{
    verify_public_kernel_opening_bundle_from_export_parts_with_perf, verify_stage1_packaged_opening_proof,
    verify_stage2_packaged_opening_proof, verify_stage3_packaged_opening_proof,
};
use super::transcript::verify_transcript_record;
use super::{
    RootLaneColumns, RootLaneCommitmentSummaryArtifact, TranscriptChallenges, TranscriptInitialState,
    VerifiedTranscriptSurface,
};
use std::time::Instant;

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

#[derive(Clone, Copy, Debug)]
struct ValidatedStage1SemInputSurface {
    sem_inputs_digest: [u8; 32],
    sequence_count: u64,
    helper_row_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedStage1RowBindingSurface {
    rows_digest: [u8; 32],
    first_trace_index: u64,
    effect_trace_index: u64,
    commit_trace_index: u64,
    last_trace_index: u64,
    row_count: u64,
    effect_row_count: u64,
    commit_row_count: u64,
    real_row_count: u64,
    preserves_x0_count: u64,
}

pub(crate) struct ReusedStage1Verification<'a> {
    sem_inputs: &'a [SemIn],
    sem_inputs_surface: ValidatedStage1SemInputSurface,
    row_bindings_surface: ValidatedStage1RowBindingSurface,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage1VerifiedClaims {
    pub sem_inputs_digest: [u8; 32],
    pub rows_digest: [u8; 32],
    pub claim: Stage1SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub mix: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage2VerifiedClaims {
    pub register_timeline_digest: [u8; 32],
    pub ram_timeline_digest: [u8; 32],
    pub twist_links_digest: [u8; 32],
    pub claim: Stage2SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub reg_mix: u64,
    pub ram_mix: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Stage3VerifiedClaims {
    pub continuity_digest: [u8; 32],
    pub claim: Stage3SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub continuity_mix: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct VerifierClaimAccumulator {
    pub transcript: TranscriptChallenges,
    pub stage1: Option<Stage1VerifiedClaims>,
    pub stage2: Option<Stage2VerifiedClaims>,
    pub stage3: Option<Stage3VerifiedClaims>,
    pub root_execution_digest: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage1ExportProof {
    pub rows: Vec<Stage1RowBinding>,
    pub semantics: Stage1SemanticsProof,
    pub selected_opening: Stage1PackagedOpeningProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage2ExportProof {
    pub register_reads: Vec<RegisterReadEvent>,
    pub register_writes: Vec<RegisterWriteEvent>,
    pub ram_events: Vec<RamEvent>,
    pub twist_links: Vec<TwistLinkEvent>,
    pub register_timeline_digest: [u8; 32],
    pub ram_timeline_digest: [u8; 32],
    pub twist_links_digest: [u8; 32],
    pub temporal_digest: [u8; 32],
    pub semantics: Stage2SemanticsProof,
    pub selected_opening: Stage2PackagedOpeningProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage3ExportProof {
    pub bridge: PcAdjacentBridge,
    pub semantics: Stage3SemanticsProof,
    pub selected_opening: Stage3PackagedOpeningProof,
    pub digest: [u8; 32],
}

pub(crate) struct Rv64imAcceptedProofCoreInputs<'a> {
    pub statement: &'a Rv64imProofStatement,
    pub stage_claims: &'a Rv64imStageClaimProofBundle,
    pub stage_packages: &'a Rv64imStagePackageProofBundle,
    pub kernel_opening: &'a Rv64imKernelOpeningProofBundle,
    pub kernel_claims: Rv64imKernelExportClaimProof,
    pub root_lane_columns: &'a RootLaneColumns,
    pub root_lane_commitment: &'a RootLaneCommitmentSummaryArtifact,
    pub main_lane: &'a Rv64imMainLaneProofBundle,
    pub stage1: Rv64imStage1ExportProof,
    pub stage2: Rv64imStage2ExportProof,
    pub stage3: Rv64imStage3ExportProof,
    pub root_execution: &'a RootExecutionBundle,
}

impl Rv64imStage1ExportProof {
    fn expected_digest_from_parts(
        rows_digest: [u8; 32],
        semantics_digest: [u8; 32],
        selected_opening_digest: [u8; 32],
    ) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_export_proof");
        tr.append_message(b"rv64im/stage1_export_proof/rows_digest", &rows_digest);
        tr.append_message(b"rv64im/stage1_export_proof/semantics", &semantics_digest);
        tr.append_message(b"rv64im/stage1_export_proof/selected_opening", &selected_opening_digest);
        tr.digest32()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        Self::expected_digest_from_parts(
            stage1_row_bindings_digest(&self.rows),
            self.semantics.digest,
            self.selected_opening.digest,
        )
    }
}

impl Rv64imStage2ExportProof {
    fn expected_digest_from_parts(
        register_reads_family_digest: [u8; 32],
        register_writes_family_digest: [u8; 32],
        ram_events_family_digest: [u8; 32],
        twist_links_family_digest: [u8; 32],
        register_timeline_digest: [u8; 32],
        ram_timeline_digest: [u8; 32],
        twist_links_digest: [u8; 32],
        temporal_digest: [u8; 32],
        semantics_digest: [u8; 32],
        selected_opening_digest: [u8; 32],
    ) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage2_export_proof");
        tr.append_message(
            b"rv64im/stage2_export_proof/register_reads_family_digest",
            &register_reads_family_digest,
        );
        tr.append_message(
            b"rv64im/stage2_export_proof/register_writes_family_digest",
            &register_writes_family_digest,
        );
        tr.append_message(
            b"rv64im/stage2_export_proof/ram_events_family_digest",
            &ram_events_family_digest,
        );
        tr.append_message(
            b"rv64im/stage2_export_proof/twist_links_family_digest",
            &twist_links_family_digest,
        );
        tr.append_message(
            b"rv64im/stage2_export_proof/register_timeline_digest",
            &register_timeline_digest,
        );
        tr.append_message(b"rv64im/stage2_export_proof/ram_timeline_digest", &ram_timeline_digest);
        tr.append_message(b"rv64im/stage2_export_proof/twist_links_digest", &twist_links_digest);
        tr.append_message(b"rv64im/stage2_export_proof/temporal_digest", &temporal_digest);
        tr.append_message(b"rv64im/stage2_export_proof/semantics", &semantics_digest);
        tr.append_message(b"rv64im/stage2_export_proof/selected_opening", &selected_opening_digest);
        tr.digest32()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        Self::expected_digest_from_parts(
            register_reads_family_digest(&self.register_reads),
            register_writes_family_digest(&self.register_writes),
            ram_events_family_digest(&self.ram_events),
            twist_links_family_digest(&self.twist_links),
            self.register_timeline_digest,
            self.ram_timeline_digest,
            self.twist_links_digest,
            self.temporal_digest,
            self.semantics.digest,
            self.selected_opening.digest,
        )
    }
}

impl Rv64imStage3ExportProof {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage3_export_proof");
        tr.append_message(b"rv64im/stage3_export_proof/bridge", &self.bridge.digest);
        tr.append_message(b"rv64im/stage3_export_proof/semantics", &self.semantics.digest);
        tr.append_message(
            b"rv64im/stage3_export_proof/selected_opening",
            &self.selected_opening.digest,
        );
        tr.digest32()
    }
}

pub(crate) fn stage1_export_proof_from_bundle(bundle: &Stage1ProofBundle) -> Rv64imStage1ExportProof {
    let proof = Rv64imStage1ExportProof {
        rows: bundle.row_bindings.clone(),
        semantics: bundle.semantics.clone(),
        selected_opening: bundle.selected_opening.clone(),
        digest: [0; 32],
    };
    Rv64imStage1ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

pub(crate) fn stage2_export_proof_from_bundle(bundle: &Stage2ProofBundle) -> Rv64imStage2ExportProof {
    let proof = Rv64imStage2ExportProof {
        register_reads: bundle.register.reads.clone(),
        register_writes: bundle.register.writes.clone(),
        ram_events: bundle.ram.events.clone(),
        twist_links: bundle.temporal.twist_links.clone(),
        register_timeline_digest: bundle.register.timeline_digest,
        ram_timeline_digest: bundle.ram.timeline_digest,
        twist_links_digest: bundle.temporal.twist_links_digest,
        temporal_digest: bundle.temporal.digest,
        semantics: bundle.semantics.clone(),
        selected_opening: bundle.selected_opening.clone(),
        digest: [0; 32],
    };
    Rv64imStage2ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

pub(crate) fn stage3_export_proof_from_bundle(bundle: &Stage3ProofBundle) -> Rv64imStage3ExportProof {
    let proof = Rv64imStage3ExportProof {
        bridge: bundle.bridge.clone(),
        semantics: bundle.semantics.clone(),
        selected_opening: bundle.selected_opening.clone(),
        digest: [0; 32],
    };
    Rv64imStage3ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

#[derive(Clone, Copy, Debug)]
struct DerivedStage2TemporalDigests {
    register_timeline_digest: [u8; 32],
    ram_timeline_digest: [u8; 32],
    twist_links_digest: [u8; 32],
    digest: [u8; 32],
}

fn derived_stage2_temporal_digests_from_events(
    register_reads: &[RegisterReadEvent],
    register_writes: &[RegisterWriteEvent],
    ram_events: &[RamEvent],
    twist_links: &[TwistLinkEvent],
) -> DerivedStage2TemporalDigests {
    let register_timeline_digest = register_timeline_digest(register_reads, register_writes);
    let ram_timeline_digest = ram_timeline_digest(ram_events);
    let twist_links_digest = twist_links_timeline_digest(twist_links);
    DerivedStage2TemporalDigests {
        register_timeline_digest,
        ram_timeline_digest,
        twist_links_digest,
        digest: Stage2TemporalContext::expected_digest_from_parts(
            register_timeline_digest,
            ram_timeline_digest,
            twist_links_digest,
            twist_links.len(),
        ),
    }
}

fn derived_stage3_continuity_digest(events: &[ContinuityEvent]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage3_pc_adjacent_bridge");
    tr.append_u64s(b"rv64im/stage3_pc_adjacent_bridge/len", &[events.len() as u64]);
    for event in events {
        tr.append_message(
            b"rv64im/stage3_pc_adjacent_bridge/event",
            &continuity_event_digest(event),
        );
    }
    tr.digest32()
}

pub(crate) fn derive_stage1_export_proof(
    execution_rows: &[Rv64ExpandedRow],
    stage_packages: &Rv64imStagePackageProofBundle,
) -> Rv64imStage1ExportProof {
    let summary = build_stage1_summary(execution_rows);
    let sem_inputs = build_sem_inputs(execution_rows);
    let proof = Rv64imStage1ExportProof {
        semantics: build_stage1_semantics_proof(&sem_inputs, &summary.rows),
        rows: summary.rows,
        selected_opening: stage_packages.packages.stage1.clone(),
        digest: [0; 32],
    };
    Rv64imStage1ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

pub(crate) fn derive_stage2_export_proof(
    execution_rows: &[Rv64ExpandedRow],
    stage_packages: &Rv64imStagePackageProofBundle,
) -> Rv64imStage2ExportProof {
    let summary = build_stage2_summary(execution_rows);
    let temporal = derived_stage2_temporal_digests_from_events(
        &summary.register_reads,
        &summary.register_writes,
        &summary.ram_events,
        &summary.twist_links,
    );
    let semantics = Stage2SemanticsProof::new(&summary);
    let proof = Rv64imStage2ExportProof {
        register_reads: summary.register_reads,
        register_writes: summary.register_writes,
        ram_events: summary.ram_events,
        twist_links: summary.twist_links,
        register_timeline_digest: temporal.register_timeline_digest,
        ram_timeline_digest: temporal.ram_timeline_digest,
        twist_links_digest: temporal.twist_links_digest,
        temporal_digest: temporal.digest,
        semantics,
        selected_opening: stage_packages.packages.stage2.clone(),
        digest: [0; 32],
    };
    Rv64imStage2ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

pub(crate) fn derive_stage3_export_proof(
    execution_rows: &[Rv64ExpandedRow],
    root_execution: &RootExecutionBundle,
    stage_packages: &Rv64imStagePackageProofBundle,
    initial_pc: u64,
    final_pc: u64,
    stage2_temporal_digest: [u8; 32],
) -> Rv64imStage3ExportProof {
    let summary = build_stage3_summary(execution_rows);
    let bridge = PcAdjacentBridge {
        continuity: summary.continuity.clone(),
        halted: summary.halted,
        continuity_digest: derived_stage3_continuity_digest(&summary.continuity),
        digest: [0; 32],
    };
    let bridge = PcAdjacentBridge {
        digest: bridge.expected_digest(),
        ..bridge
    };
    let proof = Rv64imStage3ExportProof {
        semantics: Stage3SemanticsProof::new(
            bridge.continuity_digest,
            root_execution,
            stage2_temporal_digest,
            initial_pc,
            final_pc,
            &summary.continuity,
        ),
        bridge,
        selected_opening: stage_packages.packages.stage3.clone(),
        digest: [0; 32],
    };
    Rv64imStage3ExportProof {
        digest: proof.expected_digest(),
        ..proof
    }
}

impl<'a> From<&'a Rv64imAcceptedProofArtifact> for Rv64imAcceptedProofCoreInputs<'a> {
    fn from(value: &'a Rv64imAcceptedProofArtifact) -> Self {
        Self {
            statement: &value.statement,
            stage_claims: &value.stage_claims,
            stage_packages: &value.stage_packages,
            kernel_opening: &value.kernel_opening,
            kernel_claims: kernel_export_claim_proof_from_bundle(&value.kernel_claims)
                .expect("RV64IM accepted proof should derive a compact export kernel-claim proof"),
            root_lane_columns: &value.root_lane_columns,
            root_lane_commitment: &value.root_lane_commitment,
            main_lane: &value.main_lane,
            stage1: stage1_export_proof_from_bundle(&value.stage1),
            stage2: stage2_export_proof_from_bundle(&value.stage2),
            stage3: stage3_export_proof_from_bundle(&value.stage3),
            root_execution: &value.root_execution,
        }
    }
}

fn export_terminal_row(inputs: &Rv64imAcceptedProofCoreInputs<'_>) -> Result<(u64, bool), SimpleKernelError> {
    let Some(last_row) = inputs.root_execution.execution_rows.last() else {
        return Err(SimpleKernelError::Bridge(
            "RV64IM export root execution must carry at least one row".into(),
        ));
    };
    Ok((last_row.next_pc, last_row.halted))
}

fn verify_kernel_claim_surface_bindings(inputs: &Rv64imAcceptedProofCoreInputs<'_>) -> Result<(), SimpleKernelError> {
    let _ = export_terminal_row(inputs)?;
    Ok(())
}

impl Stage1VerifiedClaims {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_verified_claims");
        tr.append_message(b"rows_digest", &self.rows_digest);
        tr.append_message(b"sem_inputs_digest", &self.sem_inputs_digest);
        tr.append_message(b"claim_digest", &self.claim.digest);
        tr.append_message(b"packaged_statement_digest", &self.packaged_statement_digest);
        tr.append_message(b"packaged_digest", &self.packaged_digest);
        tr.append_u64s(b"meta", &[self.mix]);
        tr.digest32()
    }
}

impl Stage2VerifiedClaims {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage2_verified_claims");
        tr.append_message(b"register_timeline_digest", &self.register_timeline_digest);
        tr.append_message(b"ram_timeline_digest", &self.ram_timeline_digest);
        tr.append_message(b"twist_links_digest", &self.twist_links_digest);
        tr.append_message(b"claim_digest", &self.claim.digest);
        tr.append_message(b"packaged_statement_digest", &self.packaged_statement_digest);
        tr.append_message(b"packaged_digest", &self.packaged_digest);
        tr.append_u64s(b"meta", &[self.reg_mix, self.ram_mix]);
        tr.digest32()
    }
}

impl Stage3VerifiedClaims {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage3_verified_claims");
        tr.append_message(b"continuity_digest", &self.continuity_digest);
        tr.append_message(b"claim_digest", &self.claim.digest);
        tr.append_message(b"packaged_statement_digest", &self.packaged_statement_digest);
        tr.append_message(b"packaged_digest", &self.packaged_digest);
        tr.append_u64s(b"meta", &[self.continuity_mix]);
        tr.digest32()
    }
}

fn validate_stage1_carried_sem_inputs(
    rows: &[Rv64ExpandedRow],
    sem_inputs: &[SemIn],
) -> Result<ValidatedStage1SemInputSurface, SimpleKernelError> {
    if sem_inputs.len() != rows.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 semantic inputs length mismatch".into(),
        ));
    }

    let mut digest_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_sem_inputs");
    digest_tr.append_u64s(b"rv64im/stage1_sem_inputs/len", &[sem_inputs.len() as u64]);
    let mut sequence_count = 0u64;
    let mut helper_row_count = 0u64;

    for (row, sem_input) in rows.iter().zip(sem_inputs.iter()) {
        let expected = sem_in_from_row(row);
        if sem_input != &expected {
            return Err(SimpleKernelError::Bridge(
                "RV64IM stage1 semantic inputs mismatch".into(),
            ));
        }
        digest_tr.append_message(b"rv64im/stage1_sem_inputs/entry", &sem_in_digest(sem_input));
        sequence_count += u64::from(sem_input.is_first_in_sequence);
        helper_row_count += u64::from(!sem_input.is_effect_row);
    }

    Ok(ValidatedStage1SemInputSurface {
        sem_inputs_digest: digest_tr.digest32(),
        sequence_count,
        helper_row_count,
    })
}

fn build_stage1_sem_inputs_surface(rows: &[Rv64ExpandedRow]) -> (Vec<SemIn>, ValidatedStage1SemInputSurface) {
    let mut sem_inputs = Vec::with_capacity(rows.len());
    let mut digest_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_sem_inputs");
    digest_tr.append_u64s(b"rv64im/stage1_sem_inputs/len", &[rows.len() as u64]);
    let mut sequence_count = 0u64;
    let mut helper_row_count = 0u64;

    for row in rows {
        let sem_input = sem_in_from_row(row);
        sequence_count += u64::from(sem_input.is_first_in_sequence);
        helper_row_count += u64::from(!sem_input.is_effect_row);
        digest_tr.append_message(b"rv64im/stage1_sem_inputs/entry", &sem_in_digest(&sem_input));
        sem_inputs.push(sem_input);
    }

    (
        sem_inputs,
        ValidatedStage1SemInputSurface {
            sem_inputs_digest: digest_tr.digest32(),
            sequence_count,
            helper_row_count,
        },
    )
}

fn validate_stage1_carried_row_bindings(
    rows: &[Rv64ExpandedRow],
    row_bindings: &[Stage1RowBinding],
) -> Result<ValidatedStage1RowBindingSurface, SimpleKernelError> {
    if row_bindings.len() != rows.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 row-binding length mismatch".into(),
        ));
    }

    let mut digest_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage1_rows_family");
    digest_tr.append_u64s(b"rv64im/stage1_rows_family/len", &[row_bindings.len() as u64]);

    let mut first_trace_index = 0u64;
    let mut effect_trace_index = 0u64;
    let mut commit_trace_index = 0u64;
    let mut last_trace_index = 0u64;
    let mut saw_effect = false;
    let mut saw_commit = false;
    let mut effect_row_count = 0u64;
    let mut commit_row_count = 0u64;
    let mut real_row_count = 0u64;
    let mut preserves_x0_count = 0u64;

    for (index, (row, row_binding)) in rows.iter().zip(row_bindings.iter()).enumerate() {
        let expected = stage1_row_binding_from_row(row);
        if row_binding != &expected {
            return Err(SimpleKernelError::Bridge("RV64IM stage1 row bindings mismatch".into()));
        }
        digest_tr.append_message(b"rv64im/stage1_rows_family/row_digest", &stage1_row_digest(row_binding));
        let trace_index = row_binding.trace_index as u64;
        if index == 0 {
            first_trace_index = trace_index;
        }
        last_trace_index = trace_index;
        if row_binding.is_effect_row {
            effect_row_count += 1;
            if !saw_effect {
                effect_trace_index = trace_index;
                saw_effect = true;
            }
        }
        if row_binding.is_commit_row {
            commit_row_count += 1;
            if !saw_commit {
                commit_trace_index = trace_index;
                saw_commit = true;
            }
        }
        real_row_count += u64::from(row_binding.is_real);
        preserves_x0_count += u64::from(row_binding.preserves_x0);
    }

    Ok(ValidatedStage1RowBindingSurface {
        rows_digest: digest_tr.digest32(),
        first_trace_index,
        effect_trace_index,
        commit_trace_index,
        last_trace_index,
        row_count: row_bindings.len() as u64,
        effect_row_count,
        commit_row_count,
        real_row_count,
        preserves_x0_count,
    })
}

fn verify_stage1_native_bundle_surface_with_reuse<'a>(
    artifact: &'a Rv64imAcceptedProofArtifact,
    transcript: &VerifiedTranscriptSurface,
) -> Result<ReusedStage1Verification<'a>, SimpleKernelError> {
    if artifact.stage1.digest != artifact.stage1.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 proof bundle digest mismatch".into(),
        ));
    }

    let sem_inputs =
        validate_stage1_carried_sem_inputs(&artifact.root_execution.execution_rows, &artifact.stage1.sem_inputs)?;
    let row_bindings =
        validate_stage1_carried_row_bindings(&artifact.root_execution.execution_rows, &artifact.stage1.row_bindings)?;

    if artifact.stage1.bytecode.digest != artifact.stage1.bytecode.expected_digest()
        || artifact.stage1.bytecode.rows_digest != row_bindings.rows_digest
        || artifact.stage1.bytecode.packaged_digest != artifact.stage1.selected_opening.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 native bytecode surface mismatch".into(),
        ));
    }

    if artifact.stage1.alu.digest != artifact.stage1.alu.expected_digest()
        || artifact.stage1.alu.sem_inputs_digest != sem_inputs.sem_inputs_digest
        || artifact.stage1.alu.effect_trace_index != row_bindings.effect_trace_index
        || artifact.stage1.alu.commit_trace_index != row_bindings.commit_trace_index
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 native ALU surface mismatch".into(),
        ));
    }

    if artifact.stage1.branch.digest != artifact.stage1.branch.expected_digest()
        || artifact.stage1.branch.sem_inputs_digest != sem_inputs.sem_inputs_digest
        || artifact.stage1.branch.first_trace_index != row_bindings.first_trace_index
        || artifact.stage1.branch.last_trace_index != row_bindings.last_trace_index
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 native branch surface mismatch".into(),
        ));
    }

    if artifact.stage1.address_correctness.digest != artifact.stage1.address_correctness.expected_digest()
        || artifact.stage1.address_correctness.rows_digest != row_bindings.rows_digest
        || artifact.stage1.address_correctness.row_count != row_bindings.row_count
        || artifact.stage1.address_correctness.effect_row_count != row_bindings.effect_row_count
        || artifact.stage1.address_correctness.commit_row_count != row_bindings.commit_row_count
        || artifact.stage1.address_correctness.real_row_count != row_bindings.real_row_count
        || artifact.stage1.address_correctness.preserves_x0_count != row_bindings.preserves_x0_count
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 native address-correctness surface mismatch".into(),
        ));
    }

    if artifact.stage1.linkage.digest != artifact.stage1.linkage.expected_digest()
        || artifact.stage1.linkage.rows_digest != row_bindings.rows_digest
        || artifact.stage1.linkage.sem_inputs_digest != sem_inputs.sem_inputs_digest
        || artifact.stage1.linkage.mix != transcript.challenges.stage1_mix
        || artifact.stage1.linkage.first_trace_index != row_bindings.first_trace_index
        || artifact.stage1.linkage.effect_trace_index != row_bindings.effect_trace_index
        || artifact.stage1.linkage.commit_trace_index != row_bindings.commit_trace_index
        || artifact.stage1.linkage.last_trace_index != row_bindings.last_trace_index
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 native linkage surface mismatch".into(),
        ));
    }

    Ok(ReusedStage1Verification {
        sem_inputs: &artifact.stage1.sem_inputs,
        sem_inputs_surface: sem_inputs,
        row_bindings_surface: row_bindings,
    })
}

fn verify_stage1_with_perf(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
    accumulator: &mut VerifierClaimAccumulator,
    reused: Option<&ReusedStage1Verification<'_>>,
) -> Result<(Stage1VerifiedClaims, AcceptedStage1VerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    if inputs.stage1.semantics.digest != inputs.stage1.semantics.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 semantics proof digest mismatch".into(),
        ));
    }

    let sem_inputs_surface_ms;
    let sem_input_surface;
    let sem_inputs: Cow<'_, [SemIn]>;
    if let Some(reused) = reused {
        sem_inputs_surface_ms = 0.0;
        sem_input_surface = reused.sem_inputs_surface;
        sem_inputs = Cow::Borrowed(reused.sem_inputs);
    } else {
        let sem_inputs_started = Instant::now();
        let (built_sem_inputs, built_sem_input_surface) =
            build_stage1_sem_inputs_surface(&inputs.root_execution.execution_rows);
        sem_inputs_surface_ms = millis_since(sem_inputs_started);
        sem_input_surface = built_sem_input_surface;
        sem_inputs = Cow::Owned(built_sem_inputs);
    }

    let semantics_started = Instant::now();
    verify_stage1_semantics(sem_inputs.as_ref(), &inputs.stage1.rows).map_err(SimpleKernelError::Bridge)?;
    let semantics_verify_ms = millis_since(semantics_started);

    let row_bindings_surface_ms;
    let row_bindings;
    if let Some(reused) = reused {
        row_bindings_surface_ms = 0.0;
        row_bindings = reused.row_bindings_surface;
    } else {
        let row_bindings_started = Instant::now();
        row_bindings =
            validate_stage1_carried_row_bindings(&inputs.root_execution.execution_rows, &inputs.stage1.rows)?;
        row_bindings_surface_ms = millis_since(row_bindings_started);
    }

    let surface_digest_checks_started = Instant::now();
    if inputs.stage1.digest
        != Rv64imStage1ExportProof::expected_digest_from_parts(
            row_bindings.rows_digest,
            inputs.stage1.semantics.digest,
            inputs.stage1.selected_opening.digest,
        )
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 export proof digest mismatch".into(),
        ));
    }
    if inputs.stage1.semantics.sem_inputs_digest != sem_input_surface.sem_inputs_digest
        || inputs.stage1.semantics.row_bindings_digest != row_bindings.rows_digest
        || inputs.stage1.semantics.sequence_count != sem_input_surface.sequence_count
        || inputs.stage1.semantics.helper_row_count != sem_input_surface.helper_row_count
        || row_bindings.rows_digest != inputs.stage_claims.claims.stage1.rows.rows_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 semantic surface digest mismatch".into(),
        ));
    }
    let surface_digest_checks_ms = millis_since(surface_digest_checks_started);

    let selected_opening_started = Instant::now();
    let expected_claim = build_stage1_selected_opening_claim_from_rows(
        &inputs.stage1.rows,
        &inputs.stage_claims.claims.stage1.claim,
        &inputs.stage_claims.claims.stage1.rows,
    )?;
    verify_stage1_packaged_opening_proof(&inputs.stage1.selected_opening, &expected_claim)?;
    if inputs.stage_packages.packages.stage1.digest != inputs.stage1.selected_opening.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage1 selected opening does not match the carried stage package".into(),
        ));
    }
    let selected_opening_ms = millis_since(selected_opening_started);

    let claims = Stage1VerifiedClaims {
        sem_inputs_digest: sem_input_surface.sem_inputs_digest,
        rows_digest: inputs.stage_claims.claims.stage1.rows.rows_digest,
        claim: expected_claim,
        packaged_statement_digest: inputs.stage1.selected_opening.packaged.statement.digest,
        packaged_digest: inputs.stage1.selected_opening.digest,
        mix: accumulator.transcript.stage1_mix,
        digest: [0; 32],
    };
    let claims = Stage1VerifiedClaims {
        digest: claims.expected_digest(),
        ..claims
    };
    accumulator.stage1 = Some(claims.clone());
    Ok((
        claims,
        AcceptedStage1VerifyPerf {
            sem_inputs_surface_ms,
            semantics_verify_ms,
            row_bindings_surface_ms,
            surface_digest_checks_ms,
            selected_opening_ms,
            total_ms: millis_since(total_started),
        },
    ))
}

fn verify_stage2_with_perf(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
    initial_state: &TranscriptInitialState,
    accumulator: &mut VerifierClaimAccumulator,
) -> Result<(Stage2VerifiedClaims, AcceptedStage2VerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    if inputs.stage2.semantics.digest != inputs.stage2.semantics.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage2 export sub-proof digest mismatch".into(),
        ));
    }
    let semantics_started = Instant::now();
    verify_stage2_semantics_from_events(
        &inputs.root_execution.execution_rows,
        &inputs.stage2.register_reads,
        &inputs.stage2.register_writes,
        &inputs.stage2.ram_events,
        &inputs.stage2.twist_links,
        &initial_state.registers,
        &initial_state.memory,
    )
    .map_err(SimpleKernelError::Bridge)?;
    let semantics_ms = millis_since(semantics_started);
    let temporal_started = Instant::now();
    let actual_temporal = derived_stage2_temporal_digests_from_events(
        &inputs.stage2.register_reads,
        &inputs.stage2.register_writes,
        &inputs.stage2.ram_events,
        &inputs.stage2.twist_links,
    );
    let temporal_ms = millis_since(temporal_started);
    let actual_register_timeline_digest = actual_temporal.register_timeline_digest;
    let actual_ram_timeline_digest = actual_temporal.ram_timeline_digest;
    let actual_twist_links_digest = actual_temporal.twist_links_digest;
    let family_digests_started = Instant::now();
    let actual_register_reads_family_digest = register_reads_family_digest(&inputs.stage2.register_reads);
    let actual_register_writes_family_digest = register_writes_family_digest(&inputs.stage2.register_writes);
    let actual_ram_events_family_digest = ram_events_family_digest(&inputs.stage2.ram_events);
    let actual_twist_links_family_digest = twist_links_family_digest(&inputs.stage2.twist_links);
    if inputs.stage2.digest
        != Rv64imStage2ExportProof::expected_digest_from_parts(
            actual_register_reads_family_digest,
            actual_register_writes_family_digest,
            actual_ram_events_family_digest,
            actual_twist_links_family_digest,
            actual_register_timeline_digest,
            actual_ram_timeline_digest,
            actual_twist_links_digest,
            actual_temporal.digest,
            inputs.stage2.semantics.digest,
            inputs.stage2.selected_opening.digest,
        )
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage2 export proof digest mismatch".into(),
        ));
    }
    if inputs.stage2.register_timeline_digest != actual_register_timeline_digest
        || inputs.stage2.ram_timeline_digest != actual_ram_timeline_digest
        || inputs.stage2.twist_links_digest != actual_twist_links_digest
        || inputs.stage2.temporal_digest != actual_temporal.digest
        || inputs.stage2.semantics.register_reads_family_digest != actual_register_reads_family_digest
        || inputs.stage2.semantics.register_writes_family_digest != actual_register_writes_family_digest
        || inputs.stage2.semantics.ram_events_family_digest != actual_ram_events_family_digest
        || inputs.stage2.semantics.twist_links_family_digest != actual_twist_links_family_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage2 semantic surface digest mismatch".into(),
        ));
    }
    if inputs
        .stage_claims
        .claims
        .stage2
        .families
        .register_reads_digest
        != actual_register_reads_family_digest
        || inputs
            .stage_claims
            .claims
            .stage2
            .families
            .register_writes_digest
            != actual_register_writes_family_digest
        || inputs.stage_claims.claims.stage2.families.ram_events_digest != actual_ram_events_family_digest
        || inputs
            .stage_claims
            .claims
            .stage2
            .families
            .twist_links_digest
            != actual_twist_links_family_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage2 linkage digest mismatch".into(),
        ));
    }
    let family_digests_ms = millis_since(family_digests_started);
    let selected_opening_started = Instant::now();
    let expected_claim = build_stage2_selected_opening_claim_from_events(
        &inputs.stage2.register_reads,
        &inputs.stage2.register_writes,
        &inputs.stage2.ram_events,
        &inputs.stage2.twist_links,
        &inputs.stage_claims.claims.stage2.claim,
        &inputs.stage_claims.claims.stage2.families,
    );
    verify_stage2_packaged_opening_proof(&inputs.stage2.selected_opening, &expected_claim)?;
    if inputs.stage_packages.packages.stage2.digest != inputs.stage2.selected_opening.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage2 selected opening does not match the carried stage package".into(),
        ));
    }
    let selected_opening_ms = millis_since(selected_opening_started);
    let claims = Stage2VerifiedClaims {
        register_timeline_digest: actual_register_timeline_digest,
        ram_timeline_digest: actual_ram_timeline_digest,
        twist_links_digest: actual_twist_links_digest,
        claim: expected_claim,
        packaged_statement_digest: inputs.stage2.selected_opening.packaged.statement.digest,
        packaged_digest: inputs.stage2.selected_opening.digest,
        reg_mix: accumulator.transcript.stage2_reg_mix,
        ram_mix: accumulator.transcript.stage2_ram_mix,
        digest: [0; 32],
    };
    let claims = Stage2VerifiedClaims {
        digest: claims.expected_digest(),
        ..claims
    };
    accumulator.stage2 = Some(claims.clone());
    Ok((
        claims,
        AcceptedStage2VerifyPerf {
            semantics_ms,
            temporal_ms,
            family_digests_ms,
            selected_opening_ms,
            total_ms: millis_since(total_started),
        },
    ))
}

pub(crate) fn verify_stage3(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
    accumulator: &mut VerifierClaimAccumulator,
) -> Result<Stage3VerifiedClaims, SimpleKernelError> {
    if inputs.stage3.digest != inputs.stage3.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 export proof digest mismatch".into(),
        ));
    }
    let summary = Stage3Summary {
        continuity: inputs.stage3.bridge.continuity.clone(),
        halted: inputs.stage3.bridge.halted,
    };
    let expected_summary = build_stage3_summary(&inputs.root_execution.execution_rows);
    if summary != expected_summary {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 continuity surface mismatch".into(),
        ));
    }
    if inputs.stage3.bridge.digest != inputs.stage3.bridge.expected_digest()
        || inputs.stage3.semantics.digest != inputs.stage3.semantics.expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 export sub-proof digest mismatch".into(),
        ));
    }
    verify_stage3_semantics(
        &summary.continuity,
        &inputs.root_execution.execution_rows,
        inputs.root_execution,
        inputs.statement.initial_pc,
        inputs.statement.final_pc,
    )
    .map_err(SimpleKernelError::Bridge)?;
    let actual_root_semantic_rows_digest = root_execution_semantic_rows_digest(&inputs.root_execution.semantic_rows);
    let actual_row_chunk_routes_digest =
        root_execution_row_chunk_routes_digest(&inputs.root_execution.row_chunk_routes);
    if inputs.stage3.semantics.continuity_digest != inputs.stage3.bridge.continuity_digest
        || inputs.stage3.semantics.root_semantic_rows_digest != actual_root_semantic_rows_digest
        || inputs.stage3.semantics.row_chunk_routes_digest != actual_row_chunk_routes_digest
        || inputs.stage3.semantics.prepared_step_bindings_digest != inputs.root_execution.prepared_step_bindings.digest
        || inputs.stage3.semantics.stage2_temporal_digest != inputs.stage2.temporal_digest
        || inputs.stage3.semantics.initial_pc != inputs.statement.initial_pc
        || inputs.stage3.semantics.final_pc != inputs.statement.final_pc
        || inputs.stage3.semantics.real_row_count != summary.continuity.len() as u64
        || inputs.stage3.semantics.first_real_step_index
            != summary
                .continuity
                .first()
                .map(|event| event.step_index as u64)
                .unwrap_or(0)
        || inputs.stage3.semantics.last_real_step_index
            != summary
                .continuity
                .last()
                .map(|event| event.step_index as u64)
                .unwrap_or(0)
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 semantic bridge mismatch".into(),
        ));
    }
    let expected_claim = build_stage3_selected_opening_claim(
        &summary,
        &inputs.stage_claims.claims.stage3.claim,
        &inputs.stage_claims.claims.stage3.continuity,
    );
    if expected_claim.continuity_family_digest
        != inputs
            .stage_claims
            .claims
            .stage3
            .continuity
            .continuity_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 continuity claim digest mismatch".into(),
        ));
    }
    verify_stage3_packaged_opening_proof(&inputs.stage3.selected_opening, &expected_claim)?;
    if inputs.stage_packages.packages.stage3.digest != inputs.stage3.selected_opening.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM stage3 selected opening does not match the carried stage package".into(),
        ));
    }
    let claims = Stage3VerifiedClaims {
        continuity_digest: inputs.stage3.bridge.continuity_digest,
        claim: expected_claim,
        packaged_statement_digest: inputs.stage3.selected_opening.packaged.statement.digest,
        packaged_digest: inputs.stage3.selected_opening.digest,
        continuity_mix: accumulator.transcript.stage3_continuity_mix,
        digest: [0; 32],
    };
    let claims = Stage3VerifiedClaims {
        digest: claims.expected_digest(),
        ..claims
    };
    accumulator.stage3 = Some(claims.clone());
    Ok(claims)
}

pub(crate) fn build_verified_stage3_claim_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
    transcript: &VerifiedTranscriptSurface,
) -> Result<Stage3VerifiedClaims, SimpleKernelError> {
    let inputs = Rv64imAcceptedProofCoreInputs::from(artifact);
    let mut accumulator = VerifierClaimAccumulator {
        transcript: transcript.challenges,
        ..VerifierClaimAccumulator::default()
    };
    verify_stage3(&inputs, &mut accumulator)
}

fn verify_transcript_surface_bindings(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
    transcript: &VerifiedTranscriptSurface,
) -> Result<(), SimpleKernelError> {
    if transcript.digest != transcript.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM verified transcript surface digest mismatch".into(),
        ));
    }
    if transcript.final_digest != inputs.statement.transcript_final_digest
        || transcript.final_digest != inputs.stage_claims.claims.transcript.commitment.digest
        || transcript.final_digest != inputs.stage_claims.claims.transcript.claim.final_digest
        || transcript.event_count != inputs.stage_claims.claims.transcript.claim.event_count
        || transcript.challenges.kernel_final_mix != inputs.stage_claims.claims.transcript.claim.kernel_final_mix
        || transcript.challenges.stage1_mix != inputs.stage_claims.claims.stage1.claim.mix
        || transcript.challenges.stage2_reg_mix != inputs.stage_claims.claims.stage2.claim.reg_mix
        || transcript.challenges.stage2_ram_mix != inputs.stage_claims.claims.stage2.claim.ram_mix
        || transcript.challenges.stage3_continuity_mix != inputs.stage_claims.claims.stage3.claim.continuity_mix
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM transcript surface does not match the carried stage-claim or terminal bindings".into(),
        ));
    }
    Ok(())
}

fn verify_root_execution(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
) -> Result<AcceptedRootExecutionVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let preflight_started = Instant::now();
    if inputs.root_execution.digest != inputs.root_execution.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root execution bundle digest mismatch".into(),
        ));
    }
    if inputs.root_execution.family_digest != inputs.root_lane_columns.family_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root execution family digest mismatch".into(),
        ));
    }
    let preflight_ms = millis_since(preflight_started);
    let semantic_rows_started = Instant::now();
    validate_root_execution_semantic_rows(
        &inputs.root_execution.execution_rows,
        &inputs.root_execution.semantic_rows,
        inputs.root_execution.semantic_rows_digest,
    )?;
    let semantic_rows_ms = millis_since(semantic_rows_started);

    let prepared_step_bindings_started = Instant::now();
    validate_prepared_step_binding_summary(
        &inputs.root_execution.execution_rows,
        &inputs.root_execution.semantic_rows,
        inputs.root_lane_columns,
        &inputs.root_execution.prepared_step_bindings,
    )?;
    let prepared_step_bindings_ms = millis_since(prepared_step_bindings_started);

    let kernel_claim_bindings_started = Instant::now();
    verify_kernel_claim_surface_bindings(inputs)?;
    let kernel_claim_bindings_ms = millis_since(kernel_claim_bindings_started);

    let row_chunk_routes_started = Instant::now();
    validate_root_execution_row_chunk_routes(
        &inputs.root_execution.row_chunk_routes,
        inputs.root_execution.row_chunk_routes_digest,
    )?;
    let row_chunk_routes_ms = millis_since(row_chunk_routes_started);

    let public_step_digests = root_execution_public_step_digests(&inputs.main_lane.packaged.statement);

    let row_local_ccs_acceptance_started = Instant::now();
    validate_root_row_local_ccs_acceptance_summary(
        &inputs.root_execution.prepared_step_bindings,
        &inputs.root_execution.row_chunk_routes,
        &public_step_digests,
        &inputs.root_execution.row_local_ccs_acceptance,
    )?;
    let row_local_ccs_acceptance_ms = millis_since(row_local_ccs_acceptance_started);

    let semantics_refinement_started = Instant::now();
    validate_root_execution_semantics_refinement_summary(
        &inputs.root_execution.semantic_rows,
        &inputs.root_execution.prepared_step_bindings,
        &inputs.root_execution.row_local_ccs_acceptance,
        &public_step_digests,
        &inputs.root_execution.execution_semantics_refinement,
    )?;
    let semantics_refinement_ms = millis_since(semantics_refinement_started);

    let statement_chunk_layout_started = Instant::now();
    validate_root_execution_main_lane_chunk_layout(
        &inputs.main_lane.packaged.statement,
        &inputs.root_execution.row_chunk_routes,
    )?;
    let statement_chunk_layout_ms = millis_since(statement_chunk_layout_started);
    Ok(AcceptedRootExecutionVerifyPerf {
        preflight_ms,
        semantic_rows_ms,
        prepared_step_bindings_ms,
        kernel_claim_bindings_ms,
        row_chunk_routes_ms,
        row_local_ccs_acceptance_ms,
        semantics_refinement_ms,
        statement_chunk_layout_ms,
        total_ms: millis_since(total_started),
    })
}

fn verify_step_composition_surface(artifact: &Rv64imAcceptedProofArtifact) -> Result<(), SimpleKernelError> {
    let expected = build_step_composition_surface(
        &artifact.stage1,
        &artifact.stage2,
        &artifact.stage3,
        &artifact.root_execution,
        artifact.statement.initial_pc,
        artifact.statement.final_pc,
    );
    if artifact.step_composition != expected {
        return Err(SimpleKernelError::Bridge(
            "RV64IM step composition surface mismatch".into(),
        ));
    }
    Ok(())
}

fn verify_soundness_accounting_surface(artifact: &Rv64imAcceptedProofArtifact) -> Result<(), SimpleKernelError> {
    let expected = canonical_kernel_soundness_accounting_surface();
    if artifact.soundness_accounting != expected {
        return Err(SimpleKernelError::Bridge(
            "RV64IM soundness accounting surface mismatch".into(),
        ));
    }
    Ok(())
}

pub(crate) fn verify_accepted_proof_core_with_transcript_surface_with_perf(
    inputs: &Rv64imAcceptedProofCoreInputs<'_>,
    transcript: &VerifiedTranscriptSurface,
    reused_stage1: Option<&ReusedStage1Verification<'_>>,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let transcript_started = Instant::now();
    verify_transcript_surface_bindings(inputs, transcript)?;
    let transcript_ms = millis_since(transcript_started);
    let mut accumulator = VerifierClaimAccumulator {
        transcript: transcript.challenges,
        ..VerifierClaimAccumulator::default()
    };

    let bundle_digests_started = Instant::now();
    if inputs.statement.digest != inputs.statement.expected_digest()
        || inputs.stage_claims.digest != inputs.stage_claims.expected_digest()
        || inputs.stage_packages.digest != inputs.stage_packages.expected_digest()
        || inputs.kernel_opening.digest != inputs.kernel_opening.expected_digest()
        || inputs.kernel_claims.digest != inputs.kernel_claims.expected_digest()
        || inputs.root_lane_columns.digest != inputs.root_lane_columns.expected_digest()
        || inputs.root_lane_commitment.digest != inputs.root_lane_commitment.expected_digest()
        || inputs.main_lane.digest != inputs.main_lane.expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof core bundle digest mismatch".into(),
        ));
    }
    let public_bundle_digests_ms = millis_since(bundle_digests_started);

    let bundle_bindings_started = Instant::now();
    verify_stage_claim_packaged_proof(&inputs.stage_claims.claims, &inputs.stage_claims.packaged)?;
    verify_kernel_export_claim_packaged_proof(&inputs.kernel_claims, &inputs.kernel_claims.packaged)?;
    let public_bundle_bindings_ms = millis_since(bundle_bindings_started);

    let stage_package_started = Instant::now();
    let stage1_started = Instant::now();
    let (_, stage1_perf) = verify_stage1_with_perf(inputs, &mut accumulator, reused_stage1)?;
    let stage1_ms = millis_since(stage1_started);
    let stage2_started = Instant::now();
    let (_, stage2_perf) = verify_stage2_with_perf(inputs, &transcript.initial_state, &mut accumulator)?;
    let stage2_ms = millis_since(stage2_started);
    let stage3_started = Instant::now();
    verify_stage3(inputs, &mut accumulator)?;
    let stage3_ms = millis_since(stage3_started);
    let stage_package_verify_ms = millis_since(stage_package_started);

    let root_execution_started = Instant::now();
    let root_execution_perf = verify_root_execution(inputs)?;
    accumulator.root_execution_digest = Some(inputs.root_execution.digest);
    let root_execution_ms = millis_since(root_execution_started);

    let verified_public_chunk_digests = inputs
        .main_lane
        .packaged
        .statement
        .chunks
        .iter()
        .map(public_chunk_digest)
        .collect::<Vec<_>>();

    let root_main_lane_started = Instant::now();
    let root_main_lane = verify_root_main_lane_packaged_proof_with_verified_public_statement_with_perf(
        &inputs.main_lane.packaged,
        &verified_public_chunk_digests,
    )?;
    let root_main_lane_proof_ms = millis_since(root_main_lane_started);

    let kernel_opening_started = Instant::now();
    let (final_pc, halted) = export_terminal_row(inputs)?;
    verify_public_kernel_opening_bundle_from_export_parts_with_perf(
        &inputs.kernel_opening.opening,
        &inputs.stage_claims.claims,
        &inputs.stage_packages.packages,
        inputs.root_execution.prepared_step_bindings.digest,
        inputs.root_execution.prepared_step_bindings.binding_count,
        inputs
            .root_execution
            .prepared_step_bindings
            .first_binding_digest,
        inputs
            .root_execution
            .prepared_step_bindings
            .last_binding_digest,
        digest_rows(&inputs.root_execution.execution_rows),
        inputs.kernel_claims.final_state_digest(),
        transcript.final_digest,
        final_pc,
        halted,
        inputs.root_lane_commitment,
    )?;
    let kernel_opening_verify_ms = millis_since(kernel_opening_started);

    Ok(Rv64imPublicProofVerifyPerf {
        public_claim_digests_ms: 0.0,
        public_bundle_digests_ms,
        public_bundle_bindings_ms,
        native_stage_bundle_verify_ms: 0.0,
        public_kernel_build: Default::default(),
        root_execution_verify_ms: root_execution_ms,
        root_main_lane_proof_ms,
        root_main_lane,
        stage_package_verify_ms,
        accepted_stage_package: AcceptedStagePackageVerifyPerf {
            stage1_ms,
            stage1_breakdown: stage1_perf,
            stage2_ms,
            stage2_breakdown: stage2_perf,
            stage3_ms,
            total_ms: stage_package_verify_ms,
        },
        accepted_root_execution: root_execution_perf,
        kernel_opening_verify_ms,
        summary_consistency_ms: transcript_ms,
        total_ms: millis_since(total_started),
    })
}

pub(crate) fn verify_accepted_proof_artifact_export_core_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof artifact digest mismatch".into(),
        ));
    }
    if artifact.claim.digest != artifact.claim.expected_digest()
        || artifact.statement.digest != artifact.statement.expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof public claim digest mismatch".into(),
        ));
    }
    let transcript_started = Instant::now();
    let transcript = verify_transcript_record(&artifact.transcript)?;
    let transcript_ms = millis_since(transcript_started);
    let mut perf = verify_accepted_proof_core_with_transcript_surface_with_perf(
        &Rv64imAcceptedProofCoreInputs::from(artifact),
        &transcript,
        None,
    )?;
    perf.summary_consistency_ms += transcript_ms;
    perf.total_ms += transcript_ms;
    perf.public_claim_digests_ms = 0.0;
    Ok(perf)
}

pub(crate) fn verify_accepted_proof_artifact_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof artifact digest mismatch".into(),
        ));
    }
    let claim_digests_started = Instant::now();
    if artifact.claim.digest != artifact.claim.expected_digest()
        || artifact.statement.digest != artifact.statement.expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM accepted proof public claim digest mismatch".into(),
        ));
    }
    let public_claim_digests_ms = millis_since(claim_digests_started);
    let transcript_started = Instant::now();
    let transcript = verify_transcript_record(&artifact.transcript)?;
    let transcript_ms = millis_since(transcript_started);
    let native_stage_bundle_started = Instant::now();
    let reused_stage1 = verify_stage1_native_bundle_surface_with_reuse(artifact, &transcript)?;
    let native_stage_bundle_verify_ms = millis_since(native_stage_bundle_started);
    let mut perf = verify_accepted_proof_core_with_transcript_surface_with_perf(
        &Rv64imAcceptedProofCoreInputs::from(artifact),
        &transcript,
        Some(&reused_stage1),
    )?;
    verify_step_composition_surface(artifact)?;
    verify_soundness_accounting_surface(artifact)?;
    perf.native_stage_bundle_verify_ms = native_stage_bundle_verify_ms;
    perf.summary_consistency_ms += transcript_ms;
    perf.public_claim_digests_ms = public_claim_digests_ms;
    perf.total_ms = millis_since(total_started);
    Ok(perf)
}
