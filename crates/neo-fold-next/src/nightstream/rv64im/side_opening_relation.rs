//! Owns the below-export RV64IM side-opening theorem seam.
//!
//! The statement is restricted to already-carried Nightstream surfaces.
//! The witness owns only the selected rows/events needed to justify the
//! carried opening claims, plus compact single-step packaged projections that
//! bind those claims back to the exact stage-package and kernel-opening public
//! steps, without widening the published boundary.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::proof::PublicStep;
use crate::rv64im::kernel::{
    build_claim_packaged_public_step, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, same_public_step, AjtaiFamilyKind, Rv64imAcceptedProofArtifact,
    Rv64imProofStatement, SelectedOpeningRef, SimpleKernelError, SimpleKernelOpeningClaim,
    SimpleKernelStageClaimBundle, Stage1CanonicalRowBundle, Stage1ClaimSurface, Stage1OpeningPoints,
    Stage1SelectedOpeningClaim, Stage2CanonicalFamilyBundle, Stage2ClaimSurface, Stage2OpeningPoints,
    Stage2SelectedOpeningClaim, Stage3CanonicalContinuityBundle, Stage3ClaimSurface, Stage3OpeningPoints,
    Stage3SelectedOpeningClaim, RV64IM_SELECTED_OPENING_LAYOUT_V1,
};
use crate::rv64im::stage1::{stage1_row_digest, Stage1RowBinding};
use crate::rv64im::stage2::{
    ram_event_digest, register_read_event_digest, register_write_event_digest, twist_link_event_digest, RamEvent,
    RegisterReadEvent, RegisterWriteEvent, TwistLinkEvent,
};
use crate::rv64im::stage3::{continuity_event_digest, ContinuityEvent};

use super::compact_surfaces::packaged_opening_proof_digest_from_surfaces;
use super::side_bridges::validate_rv64im_side_proof_bundle_structure;
use super::side_claim_relation::Rv64imSingleStepPackagedProofWitness;
use super::{build_rv64im_kernel_opening_claim_from_side_proof_bundle, Rv64imSideProofBundle};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imSideOpeningRelationStatement {
    pub public_statement: Rv64imProofStatement,
    pub side_bundle: Rv64imSideProofBundle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage1SelectedRowsWitness {
    pub first: Stage1RowBinding,
    pub effect_position: u64,
    pub effect: Stage1RowBinding,
    pub commit_position: u64,
    pub commit: Stage1RowBinding,
    pub last: Stage1RowBinding,
}

impl Rv64imStage1SelectedRowsWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness/positions",
            &[self.effect_position, self.commit_position],
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness/first",
            &stage1_row_digest(&self.first),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness/effect",
            &stage1_row_digest(&self.effect),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness/commit",
            &stage1_row_digest(&self.commit),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/stage1_selected_rows_witness/last",
            &stage1_row_digest(&self.last),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage2SelectedEventsWitness {
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

impl Rv64imStage2SelectedEventsWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness");
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_read_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_read",
            self.first_read.as_ref().map(register_read_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_read_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_read",
            self.last_read.as_ref().map(register_read_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_write_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_write",
            self.first_write.as_ref().map(register_write_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_write_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_write",
            self.last_write.as_ref().map(register_write_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_ram_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_ram",
            self.first_ram.as_ref().map(ram_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_ram_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_ram",
            self.last_ram.as_ref().map(ram_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_twist_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/first_twist",
            self.first_twist.as_ref().map(twist_link_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_twist_present",
            b"neo.fold.next/nightstream/rv64im/stage2_selected_events_witness/last_twist",
            self.last_twist.as_ref().map(twist_link_event_digest),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStage3SelectedContinuityWitness {
    pub first_continuity: Option<ContinuityEvent>,
    pub last_continuity: Option<ContinuityEvent>,
}

impl Rv64imStage3SelectedContinuityWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/stage3_selected_continuity_witness");
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage3_selected_continuity_witness/first_present",
            b"neo.fold.next/nightstream/rv64im/stage3_selected_continuity_witness/first",
            self.first_continuity.as_ref().map(continuity_event_digest),
        );
        append_optional_digest(
            &mut tr,
            b"neo.fold.next/nightstream/rv64im/stage3_selected_continuity_witness/last_present",
            b"neo.fold.next/nightstream/rv64im/stage3_selected_continuity_witness/last",
            self.last_continuity.as_ref().map(continuity_event_digest),
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imSideOpeningRelationWitness {
    pub stage1_selected_rows: Rv64imStage1SelectedRowsWitness,
    pub stage2_selected_events: Rv64imStage2SelectedEventsWitness,
    pub stage3_selected_continuity: Rv64imStage3SelectedContinuityWitness,
    pub stage1_packaged: Rv64imSingleStepPackagedProofWitness,
    pub stage2_packaged: Rv64imSingleStepPackagedProofWitness,
    pub stage3_packaged: Rv64imSingleStepPackagedProofWitness,
    pub bindings_packaged: Rv64imSingleStepPackagedProofWitness,
    pub prepared_steps_packaged: Rv64imSingleStepPackagedProofWitness,
}

impl Rv64imSideOpeningRelationWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage1_selected_rows",
            &self.stage1_selected_rows.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage2_selected_events",
            &self.stage2_selected_events.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage3_selected_continuity",
            &self.stage3_selected_continuity.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage1_packaged",
            &self.stage1_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage2_packaged",
            &self.stage2_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/stage3_packaged",
            &self.stage3_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/bindings_packaged",
            &self.bindings_packaged.digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_opening_relation_witness/prepared_steps_packaged",
            &self.prepared_steps_packaged.digest(),
        );
        tr.digest32()
    }
}

pub fn build_rv64im_side_opening_relation_statement(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<Rv64imSideOpeningRelationStatement, SimpleKernelError> {
    if public_statement.digest != public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation public statement digest mismatch".into(),
        ));
    }
    validate_rv64im_side_proof_bundle_structure(side_bundle)?;
    Ok(Rv64imSideOpeningRelationStatement {
        public_statement: public_statement.clone(),
        side_bundle: side_bundle.clone(),
    })
}

pub fn build_rv64im_side_opening_relation_witness_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Rv64imSideOpeningRelationWitness {
    let stage1_rows = &artifact.stage1.row_bindings;
    let stage2 = &artifact.stage2;
    let stage3 = &artifact.stage3.bridge.continuity;
    let effect_position = stage1_rows
        .iter()
        .position(|row| row.is_effect_row)
        .expect("RV64IM side-opening witness requires a stage1 effect row") as u64;
    let commit_position = stage1_rows
        .iter()
        .position(|row| row.is_commit_row)
        .expect("RV64IM side-opening witness requires a stage1 commit row") as u64;
    Rv64imSideOpeningRelationWitness {
        stage1_selected_rows: Rv64imStage1SelectedRowsWitness {
            first: stage1_rows
                .first()
                .expect("RV64IM side-opening witness requires a first stage1 row")
                .clone(),
            effect_position,
            effect: stage1_rows[effect_position as usize].clone(),
            commit_position,
            commit: stage1_rows[commit_position as usize].clone(),
            last: stage1_rows
                .last()
                .expect("RV64IM side-opening witness requires a last stage1 row")
                .clone(),
        },
        stage2_selected_events: Rv64imStage2SelectedEventsWitness {
            first_read: stage2.register.reads.first().cloned(),
            last_read: stage2.register.reads.last().cloned(),
            first_write: stage2.register.writes.first().cloned(),
            last_write: stage2.register.writes.last().cloned(),
            first_ram: stage2.ram.events.first().cloned(),
            last_ram: stage2.ram.events.last().cloned(),
            first_twist: stage2.temporal.twist_links.first().cloned(),
            last_twist: stage2.temporal.twist_links.last().cloned(),
        },
        stage3_selected_continuity: Rv64imStage3SelectedContinuityWitness {
            first_continuity: stage3.first().cloned(),
            last_continuity: stage3.last().cloned(),
        },
        stage1_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(
            &artifact.stage_packages.packages.stage1.packaged,
        ),
        stage2_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(
            &artifact.stage_packages.packages.stage2.packaged,
        ),
        stage3_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(
            &artifact.stage_packages.packages.stage3.packaged,
        ),
        bindings_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(
            &artifact.kernel_opening.opening.bindings.packaged,
        ),
        prepared_steps_packaged: Rv64imSingleStepPackagedProofWitness::from_packaged(
            &artifact.kernel_opening.opening.prepared_steps.packaged,
        ),
    }
}

pub fn build_rv64im_side_opening_relation_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imSideOpeningRelationStatement, Rv64imSideOpeningRelationWitness), SimpleKernelError> {
    let side_bundle = super::build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    let statement = build_rv64im_side_opening_relation_statement(&artifact.statement, &side_bundle)?;
    let witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(artifact);
    Ok((statement, witness))
}

fn selected_opening_ref(
    family: AjtaiFamilyKind,
    commitment_digest: [u8; 32],
    logical_index: u64,
    value_digest: [u8; 32],
) -> SelectedOpeningRef {
    SelectedOpeningRef::from_parts(
        family,
        commitment_digest,
        RV64IM_SELECTED_OPENING_LAYOUT_V1,
        logical_index,
        value_digest,
    )
}

fn optional_selected_opening_ref(
    family: AjtaiFamilyKind,
    commitment_digest: [u8; 32],
    logical_index: u64,
    value_digest: Option<[u8; 32]>,
) -> Option<SelectedOpeningRef> {
    value_digest.map(|value_digest| selected_opening_ref(family, commitment_digest, logical_index, value_digest))
}

fn build_stage1_selected_opening_claim_from_witness(
    witness: &Rv64imStage1SelectedRowsWitness,
    claim: &Stage1ClaimSurface,
    rows: &Stage1CanonicalRowBundle,
) -> Result<Stage1SelectedOpeningClaim, SimpleKernelError> {
    let row_count = u64::try_from(claim.row_count)
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side-opening relation stage1 row_count overflows u64".into()))?;
    if row_count == 0 {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation stage1 claim must carry at least one row".into(),
        ));
    }
    let claim = Stage1SelectedOpeningClaim {
        rows_family_digest: rows.rows_digest,
        row_count,
        effect_row_count: claim.effect_row_count as u64,
        commit_row_count: claim.commit_row_count as u64,
        real_row_count: claim.real_row_count as u64,
        preserves_x0_count: claim.preserves_x0_count as u64,
        first_trace_index: witness.first.trace_index as u64,
        effect_trace_index: witness.effect.trace_index as u64,
        commit_trace_index: witness.commit.trace_index as u64,
        last_trace_index: witness.last.trace_index as u64,
        mix: claim.mix,
        points: Stage1OpeningPoints {
            first: selected_opening_ref(
                AjtaiFamilyKind::Stage1Rows,
                rows.rows_digest,
                0,
                stage1_row_digest(&witness.first),
            ),
            effect: selected_opening_ref(
                AjtaiFamilyKind::Stage1Rows,
                rows.rows_digest,
                witness.effect_position,
                stage1_row_digest(&witness.effect),
            ),
            commit: selected_opening_ref(
                AjtaiFamilyKind::Stage1Rows,
                rows.rows_digest,
                witness.commit_position,
                stage1_row_digest(&witness.commit),
            ),
            last: selected_opening_ref(
                AjtaiFamilyKind::Stage1Rows,
                rows.rows_digest,
                row_count.saturating_sub(1),
                stage1_row_digest(&witness.last),
            ),
        },
        digest: [0; 32],
    };
    Ok(Stage1SelectedOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    })
}

fn build_stage2_selected_opening_claim_from_witness(
    witness: &Rv64imStage2SelectedEventsWitness,
    claim: &Stage2ClaimSurface,
    families: &Stage2CanonicalFamilyBundle,
) -> Stage2SelectedOpeningClaim {
    let claim = Stage2SelectedOpeningClaim {
        register_reads_family_digest: families.register_reads_digest,
        register_writes_family_digest: families.register_writes_digest,
        ram_events_family_digest: families.ram_events_digest,
        twist_links_family_digest: families.twist_links_digest,
        register_read_count: claim.register_read_count as u64,
        register_write_count: claim.register_write_count as u64,
        ram_event_count: claim.ram_event_count as u64,
        twist_link_count: claim.twist_link_count as u64,
        ram_read_count: claim.ram_read_count as u64,
        ram_write_count: claim.ram_write_count as u64,
        reg_mix: claim.reg_mix,
        ram_mix: claim.ram_mix,
        points: Stage2OpeningPoints {
            first_read: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RegisterReads,
                families.register_reads_digest,
                0,
                witness.first_read.as_ref().map(register_read_event_digest),
            ),
            last_read: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RegisterReads,
                families.register_reads_digest,
                claim.register_read_count.saturating_sub(1) as u64,
                witness.last_read.as_ref().map(register_read_event_digest),
            ),
            first_write: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RegisterWrites,
                families.register_writes_digest,
                0,
                witness
                    .first_write
                    .as_ref()
                    .map(register_write_event_digest),
            ),
            last_write: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RegisterWrites,
                families.register_writes_digest,
                claim.register_write_count.saturating_sub(1) as u64,
                witness.last_write.as_ref().map(register_write_event_digest),
            ),
            first_ram: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RamEvents,
                families.ram_events_digest,
                0,
                witness.first_ram.as_ref().map(ram_event_digest),
            ),
            last_ram: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2RamEvents,
                families.ram_events_digest,
                claim.ram_event_count.saturating_sub(1) as u64,
                witness.last_ram.as_ref().map(ram_event_digest),
            ),
            first_twist: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2TwistLinks,
                families.twist_links_digest,
                0,
                witness.first_twist.as_ref().map(twist_link_event_digest),
            ),
            last_twist: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage2TwistLinks,
                families.twist_links_digest,
                claim.twist_link_count.saturating_sub(1) as u64,
                witness.last_twist.as_ref().map(twist_link_event_digest),
            ),
        },
        digest: [0; 32],
    };
    Stage2SelectedOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn build_stage3_selected_opening_claim_from_witness(
    witness: &Rv64imStage3SelectedContinuityWitness,
    claim: &Stage3ClaimSurface,
    continuity: &Stage3CanonicalContinuityBundle,
) -> Stage3SelectedOpeningClaim {
    let claim = Stage3SelectedOpeningClaim {
        continuity_family_digest: continuity.continuity_digest,
        continuity_count: claim.continuity_count as u64,
        final_step_count: claim.final_step_count as u64,
        halted: claim.halted,
        all_continuity_hold: claim.all_continuity_hold,
        continuity_mix: claim.continuity_mix,
        points: Stage3OpeningPoints {
            first_continuity: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage3Continuity,
                continuity.continuity_digest,
                0,
                witness
                    .first_continuity
                    .as_ref()
                    .map(continuity_event_digest),
            ),
            last_continuity: optional_selected_opening_ref(
                AjtaiFamilyKind::Stage3Continuity,
                continuity.continuity_digest,
                claim.continuity_count.saturating_sub(1) as u64,
                witness
                    .last_continuity
                    .as_ref()
                    .map(continuity_event_digest),
            ),
        },
        digest: [0; 32],
    };
    Stage3SelectedOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn verify_stage_claim_opening_witness(
    label: &str,
    claim_digest: [u8; 32],
    expected_step: &PublicStep,
    witness: &Rv64imSingleStepPackagedProofWitness,
    carried_packaged_digest: [u8; 32],
    bridge_label: &str,
) -> Result<(), SimpleKernelError> {
    if !same_public_step(&witness.step, expected_step) {
        return Err(SimpleKernelError::Bridge(format!(
            "{label} selected-claim package public step mismatch"
        )));
    }
    let statement_digest = super::side_claim_relation::single_step_packaged_statement_digest(
        &witness.step,
        &witness.final_main_claim_digests,
    );
    let expected_packaged_digest =
        packaged_opening_proof_digest_from_surfaces(claim_digest, statement_digest, witness.proof_digest);
    if expected_packaged_digest != carried_packaged_digest {
        return Err(SimpleKernelError::Bridge(bridge_label.into()));
    }
    Ok(())
}

fn verify_compact_stage_opening_claims(
    side_bundle: &Rv64imSideProofBundle,
    witness: &Rv64imSideOpeningRelationWitness,
    stage_claims: &SimpleKernelStageClaimBundle,
) -> Result<(), SimpleKernelError> {
    let stage1_claim = build_stage1_selected_opening_claim_from_witness(
        &witness.stage1_selected_rows,
        &stage_claims.stage1.claim,
        &stage_claims.stage1.rows,
    )?;
    if stage1_claim != side_bundle.stage1.claim {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation stage1 selected rows do not match the carried opening claim".into(),
        ));
    }

    let stage2_claim = build_stage2_selected_opening_claim_from_witness(
        &witness.stage2_selected_events,
        &stage_claims.stage2.claim,
        &stage_claims.stage2.families,
    );
    if stage2_claim != side_bundle.stage2.claim {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation stage2 selected events do not match the carried opening claim".into(),
        ));
    }

    let stage3_claim = build_stage3_selected_opening_claim_from_witness(
        &witness.stage3_selected_continuity,
        &stage_claims.stage3.claim,
        &stage_claims.stage3.continuity,
    );
    if stage3_claim != side_bundle.stage3.claim {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation stage3 selected continuity does not match the carried opening claim".into(),
        ));
    }

    Ok(())
}

fn verify_compact_stage_opening_packages(
    side_bundle: &Rv64imSideProofBundle,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    let stage1_step = build_claim_packaged_public_step("rv64im/stage1", &side_bundle.stage1.claim.claim_words())?;
    verify_stage_claim_opening_witness(
        "rv64im/stage1",
        side_bundle.stage1.claim.digest,
        &stage1_step,
        &witness.stage1_packaged,
        side_bundle.stage1.packaged_digest,
        "RV64IM side-opening relation stage1 package witness does not match the carried side bundle",
    )?;

    let stage2_step = build_claim_packaged_public_step("rv64im/stage2", &side_bundle.stage2.claim.claim_words())?;
    verify_stage_claim_opening_witness(
        "rv64im/stage2",
        side_bundle.stage2.claim.digest,
        &stage2_step,
        &witness.stage2_packaged,
        side_bundle.stage2.packaged_digest,
        "RV64IM side-opening relation stage2 package witness does not match the carried side bundle",
    )?;

    let stage3_step = build_claim_packaged_public_step("rv64im/stage3", &side_bundle.stage3.claim.claim_words())?;
    verify_stage_claim_opening_witness(
        "rv64im/stage3",
        side_bundle.stage3.claim.digest,
        &stage3_step,
        &witness.stage3_packaged,
        side_bundle.stage3.packaged_digest,
        "RV64IM side-opening relation stage3 package witness does not match the carried side bundle",
    )?;

    Ok(())
}

fn verify_compact_kernel_opening_packages(
    side_bundle: &Rv64imSideProofBundle,
    witness: &Rv64imSideOpeningRelationWitness,
    kernel_opening_claim: &SimpleKernelOpeningClaim,
) -> Result<(), SimpleKernelError> {
    let bindings_step = build_kernel_binding_opening_public_step(&kernel_opening_claim.bindings)?;
    verify_stage_claim_opening_witness(
        "rv64im/kernel_opening_bundle/bindings",
        kernel_opening_claim.bindings.digest,
        &bindings_step,
        &witness.bindings_packaged,
        side_bundle.kernel_opening_bridge.bindings_opening_digest,
        "RV64IM side-opening relation binding-opening witness does not match the carried side bundle",
    )?;

    let prepared_steps_step = build_kernel_prepared_step_opening_public_step(&kernel_opening_claim.prepared_steps)?;
    verify_stage_claim_opening_witness(
        "rv64im/kernel_opening_bundle/prepared_steps",
        kernel_opening_claim.prepared_steps.digest,
        &prepared_steps_step,
        &witness.prepared_steps_packaged,
        side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest,
        "RV64IM side-opening relation prepared-step opening witness does not match the carried side bundle",
    )?;

    Ok(())
}

pub(super) fn verify_rv64im_side_opening_witness_against_compact_surfaces(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    let stage_claims =
        super::build_rv64im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    verify_compact_stage_opening_claims(side_bundle, witness, &stage_claims)?;

    super::verify_rv64im_side_stage_packages_surface(side_bundle, public_statement)?;
    verify_compact_stage_opening_packages(side_bundle, witness)?;

    super::verify_rv64im_side_kernel_opening_surface(side_bundle, public_statement)?;
    let kernel_opening_claim = build_rv64im_kernel_opening_claim_from_side_proof_bundle(side_bundle, public_statement)?;
    verify_compact_kernel_opening_packages(side_bundle, witness, &kernel_opening_claim)?;
    Ok(())
}

pub fn verify_rv64im_side_opening_relation(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    if statement.public_statement.digest != statement.public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation public statement digest mismatch".into(),
        ));
    }
    if statement.side_bundle.digest != statement.side_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side-opening relation side-proof bundle digest mismatch".into(),
        ));
    }
    verify_rv64im_side_opening_witness_against_compact_surfaces(
        &statement.public_statement,
        &statement.side_bundle,
        witness,
    )
}
