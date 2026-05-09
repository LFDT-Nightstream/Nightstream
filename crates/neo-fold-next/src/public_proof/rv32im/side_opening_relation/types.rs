//! Owns side-opening statement and witness data shapes.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::proof::FoldSchedule;
use crate::public_proof::rv32im::side_bridges::{
    Rv32imKernelClaimBridge, Rv32imKernelClaimProofBridge, Rv32imKernelOpeningBridge, Rv32imMainLaneProofBridge,
    Rv32imStageClaimProofBridge,
};
use crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness;
use crate::rv32im::kernel::{
    Rv32imProofStatement, Stage1SelectedOpeningClaim, Stage2SelectedOpeningClaim, Stage3SelectedOpeningClaim,
};
use crate::rv32im::stage1::{stage1_row_digest, Stage1RowBinding};
use crate::rv32im::stage2::{
    ram_event_digest, register_read_event_digest, register_write_event_digest, twist_link_event_digest, RamEvent,
    RegisterReadEvent, RegisterWriteEvent, TwistLinkEvent,
};
use crate::rv32im::stage3::{continuity_event_digest, ContinuityEvent};
use crate::rv32im::{Stage1VerifiedClaims, Stage2VerifiedClaims, Stage3VerifiedClaims};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideTranscriptSummary {
    pub surface_digest: [u8; 32],
    pub event_count: usize,
    pub kernel_final_mix: u64,
}

impl Rv32imSideTranscriptSummary {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_transcript_summary");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_transcript_summary/surface_digest",
            &self.surface_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/side_transcript_summary/meta",
            &[self.event_count as u64, self.kernel_final_mix],
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideOpeningPublicStatementSummary {
    pub proof_statement_digest: [u8; 32],
    pub root_params_id: [u8; 32],
    pub fold_schedule: FoldSchedule,
    pub chunk_count: u64,
    pub stage_claims_digest: [u8; 32],
    pub stage_packages_digest: [u8; 32],
    pub kernel_opening_digest: [u8; 32],
    pub prepared_step_bindings_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub final_state_digest: [u8; 32],
    pub transcript_final_digest: [u8; 32],
    pub main_lane_surface_digest: [u8; 32],
    pub root_lane_columns_digest: [u8; 32],
    pub public_step_count: u64,
    pub initial_pc: u64,
    pub final_pc: u64,
    pub halted: bool,
}

impl Rv32imSideOpeningPublicStatementSummary {
    pub fn from_public_statement(public_statement: &Rv32imProofStatement) -> Self {
        Self {
            proof_statement_digest: public_statement.digest,
            root_params_id: public_statement.root_params_id,
            fold_schedule: public_statement.fold_schedule,
            chunk_count: public_statement.chunk_count,
            stage_claims_digest: public_statement.stage_claims_digest,
            stage_packages_digest: public_statement.stage_packages_digest,
            kernel_opening_digest: public_statement.kernel_opening_digest,
            prepared_step_bindings_digest: public_statement.prepared_step_bindings_digest,
            execution_digest: public_statement.execution_digest,
            final_state_digest: public_statement.final_state_digest,
            transcript_final_digest: public_statement.transcript_final_digest,
            main_lane_surface_digest: public_statement.main_lane_surface_digest,
            root_lane_columns_digest: public_statement.root_lane_columns_digest,
            public_step_count: public_statement.public_step_count,
            initial_pc: public_statement.initial_pc,
            final_pc: public_statement.final_pc,
            halted: public_statement.halted,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr =
            Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/proof_statement_digest",
            &self.proof_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/root_params_id",
            &self.root_params_id,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/chunk_count",
            &[self.chunk_count],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/stage_claims_digest",
            &self.stage_claims_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/stage_packages_digest",
            &self.stage_packages_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/kernel_opening_digest",
            &self.kernel_opening_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/execution_digest",
            &self.execution_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/final_state_digest",
            &self.final_state_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/transcript_final_digest",
            &self.transcript_final_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/main_lane_surface_digest",
            &self.main_lane_surface_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/root_lane_columns_digest",
            &self.root_lane_columns_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/side_opening_public_statement_summary/meta",
            &[
                self.public_step_count,
                self.initial_pc,
                self.final_pc,
                u64::from(self.halted),
            ],
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideStage1Summary {
    pub rows_digest: [u8; 32],
    pub claim: Stage1SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub mix: u64,
    pub digest: [u8; 32],
}

impl Rv32imSideStage1Summary {
    pub fn from_verified_claims(stage1: &Stage1VerifiedClaims) -> Self {
        Self {
            rows_digest: stage1.rows_digest,
            claim: stage1.claim.clone(),
            packaged_statement_digest: stage1.packaged_statement_digest,
            packaged_digest: stage1.packaged_digest,
            mix: stage1.mix,
            digest: stage1.digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideStage2Summary {
    pub claim: Stage2SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub reg_mix: u64,
    pub ram_mix: u64,
    pub digest: [u8; 32],
}

impl Rv32imSideStage2Summary {
    pub fn from_verified_claims(stage2: &Stage2VerifiedClaims) -> Self {
        Self {
            claim: stage2.claim.clone(),
            packaged_statement_digest: stage2.packaged_statement_digest,
            packaged_digest: stage2.packaged_digest,
            reg_mix: stage2.reg_mix,
            ram_mix: stage2.ram_mix,
            digest: stage2.digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideStage3Summary {
    pub claim: Stage3SelectedOpeningClaim,
    pub packaged_statement_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub continuity_mix: u64,
    pub digest: [u8; 32],
}

impl Rv32imSideStage3Summary {
    pub fn from_verified_claims(stage3: &Stage3VerifiedClaims) -> Self {
        Self {
            claim: stage3.claim.clone(),
            packaged_statement_digest: stage3.packaged_statement_digest,
            packaged_digest: stage3.packaged_digest,
            continuity_mix: stage3.continuity_mix,
            digest: stage3.digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideOpeningRelationStatement {
    pub public_summary: Rv32imSideOpeningPublicStatementSummary,
    pub transcript: Rv32imSideTranscriptSummary,
    pub stage1: Rv32imSideStage1Summary,
    pub stage2: Rv32imSideStage2Summary,
    pub stage3: Rv32imSideStage3Summary,
    pub stage_claim_proof_bridge: Rv32imStageClaimProofBridge,
    pub kernel_opening_bridge: Rv32imKernelOpeningBridge,
    pub kernel_claim_bridge: Rv32imKernelClaimBridge,
    pub kernel_claim_proof_bridge: Rv32imKernelClaimProofBridge,
    pub main_lane_bridge: Rv32imMainLaneProofBridge,
}

impl Rv32imSideOpeningRelationStatement {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/public_summary_digest",
            &self.public_summary.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/transcript_digest",
            &self.transcript.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/stage1_digest",
            &self.stage1.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/stage2_digest",
            &self.stage2.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/stage3_digest",
            &self.stage3.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/stage_claim_proof_bridge_digest",
            &self.stage_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/kernel_opening_bridge_digest",
            &self.kernel_opening_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/kernel_claim_bridge_digest",
            &self.kernel_claim_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/kernel_claim_proof_bridge_digest",
            &self.kernel_claim_proof_bridge.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_spartan_statement/main_lane_bridge_digest",
            &self.main_lane_bridge.digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imStage1SelectedRowsWitness {
    pub first: Stage1RowBinding,
    pub effect_position: u64,
    pub effect: Stage1RowBinding,
    pub commit_position: u64,
    pub commit: Stage1RowBinding,
    pub last: Stage1RowBinding,
}

impl Rv32imStage1SelectedRowsWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness/positions",
            &[self.effect_position, self.commit_position],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness/first",
            &stage1_row_digest(&self.first),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness/effect",
            &stage1_row_digest(&self.effect),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness/commit",
            &stage1_row_digest(&self.commit),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/stage1_selected_rows_witness/last",
            &stage1_row_digest(&self.last),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imStage2SelectedEventsWitness {
    pub first_read: Option<RegisterReadEvent>,
    pub last_read: Option<RegisterReadEvent>,
    pub first_write: Option<RegisterWriteEvent>,
    pub last_write: Option<RegisterWriteEvent>,
    pub first_ram: Option<RamEvent>,
    pub last_ram: Option<RamEvent>,
    pub first_twist: Option<TwistLinkEvent>,
    pub last_twist: Option<TwistLinkEvent>,
}

fn append_optional_digest(
    tr: &mut Poseidon2Transcript,
    present_label: &'static [u8],
    digest_label: &'static [u8],
    digest: Option<[u8; 32]>,
) {
    tr.append_u64s(present_label, &[u64::from(digest.is_some())]);
    if let Some(digest) = digest {
        tr.append_message(digest_label, &digest);
    }
}

impl Rv32imStage2SelectedEventsWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness");
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_read_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_read",
            self.first_read.as_ref().map(register_read_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_read_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_read",
            self.last_read.as_ref().map(register_read_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_write_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_write",
            self.first_write.as_ref().map(register_write_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_write_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_write",
            self.last_write.as_ref().map(register_write_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_ram_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_ram",
            self.first_ram.as_ref().map(ram_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_ram_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_ram",
            self.last_ram.as_ref().map(ram_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_twist_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/first_twist",
            self.first_twist.as_ref().map(twist_link_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_twist_present",
            b"neo.fold.next/nightstream/rv32im/stage2_selected_events_witness/last_twist",
            self.last_twist.as_ref().map(twist_link_event_digest),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imStage3SelectedContinuityWitness {
    pub first_continuity: Option<ContinuityEvent>,
    pub last_continuity: Option<ContinuityEvent>,
}

impl Rv32imStage3SelectedContinuityWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/stage3_selected_continuity_witness");
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage3_selected_continuity_witness/first_present",
            b"neo.fold.next/nightstream/rv32im/stage3_selected_continuity_witness/first",
            self.first_continuity.as_ref().map(continuity_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv32im/stage3_selected_continuity_witness/last_present",
            b"neo.fold.next/nightstream/rv32im/stage3_selected_continuity_witness/last",
            self.last_continuity.as_ref().map(continuity_event_digest),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32imSideOpeningRelationWitness {
    pub stage1_selected_rows: Rv32imStage1SelectedRowsWitness,
    pub stage2_selected_events: Rv32imStage2SelectedEventsWitness,
    pub stage3_selected_continuity: Rv32imStage3SelectedContinuityWitness,
    pub stage1_packaged: Rv32imSingleStepPackagedProofWitness,
    pub stage2_packaged: Rv32imSingleStepPackagedProofWitness,
    pub stage3_packaged: Rv32imSingleStepPackagedProofWitness,
    pub bindings_packaged: Rv32imSingleStepPackagedProofWitness,
    pub prepared_steps_packaged: Rv32imSingleStepPackagedProofWitness,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideSelectedOpeningWitness {
    pub stage1_selected_rows: Rv32imStage1SelectedRowsWitness,
    pub stage2_selected_events: Rv32imStage2SelectedEventsWitness,
    pub stage3_selected_continuity: Rv32imStage3SelectedContinuityWitness,
}

impl Rv32imSideOpeningRelationWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage1_selected_rows",
            &self.stage1_selected_rows.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage2_selected_events",
            &self.stage2_selected_events.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage3_selected_continuity",
            &self.stage3_selected_continuity.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage1_packaged",
            &self.stage1_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage2_packaged",
            &self.stage2_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/stage3_packaged",
            &self.stage3_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/bindings_packaged",
            &self.bindings_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_opening_relation_witness/prepared_steps_packaged",
            &self.prepared_steps_packaged.digest(),
        );
        tr.digest32()
    }
}
