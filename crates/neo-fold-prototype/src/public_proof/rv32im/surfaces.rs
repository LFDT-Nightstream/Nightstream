//! Verifies compact RV32IM public surfaces derived from the side-proof bundle.

use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::rv32im::kernel::{
    build_public_kernel_opening_claim_from_compact_surfaces, kernel_claim_bundle_from_statement_and_compact_surfaces,
    Rv32imProofStatement, Rv32imStageClaimDigestBundle, SimpleKernelError, SimpleKernelOpeningClaim,
    SimpleKernelStageClaimBundle, Stage1ArtifactSurface, Stage1CanonicalRowBundle, Stage1ClaimSurface,
    Stage2ArtifactSurface, Stage2CanonicalFamilyBundle, Stage2ClaimSurface, Stage3ArtifactSurface,
    Stage3CanonicalContinuityBundle, Stage3ClaimSurface, StageDigestCommitment, TranscriptArtifactSurface,
    TranscriptClaimSurface,
};

use super::compact_surfaces::{kernel_claim_summary_digest_from_surfaces, packaged_claim_proof_digest_from_surfaces};
use super::side_bridges::{validate_rv32im_side_proof_bundle_structure, Rv32imSideProofBundle};

pub(super) fn verify_rv32im_side_kernel_claim_surface(
    side_bundle: &Rv32imSideProofBundle,
    public_statement: &Rv32imProofStatement,
    main_lane_bundle_digest: [u8; 32],
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_claim_bridge.digest != side_bundle.kernel_claim_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof kernel-claim bridge digest mismatch".into(),
        ));
    }
    let expected = kernel_claim_bundle_from_statement_and_compact_surfaces(
        public_statement,
        main_lane_bundle_digest,
        side_bundle.kernel_claim_bridge.stage1_digest,
        side_bundle.kernel_claim_bridge.stage2_digest,
        side_bundle.kernel_claim_bridge.stage3_digest,
        side_bundle.kernel_claim_bridge.root0_digest,
    );
    if side_bundle.kernel_claim_bridge.kernel_claim_bundle_digest != expected.digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream compact kernel-claim surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

pub(super) fn verify_rv32im_side_stage_claim_proof_surface(
    side_bundle: &Rv32imSideProofBundle,
    public_statement: &Rv32imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.stage_claim_proof_bridge.digest != side_bundle.stage_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof stage-claim proof bridge digest mismatch".into(),
        ));
    }
    let claims =
        build_rv32im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    let summary = Rv32imStageClaimDigestBundle::from_claims(&claims);
    let expected = packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv32im/stage_claim_proof_bundle",
        summary.digest,
        side_bundle
            .stage_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.stage_claim_proof_bridge.packaged_proof_digest,
    );
    if side_bundle
        .stage_claim_proof_bridge
        .stage_claim_proof_bundle_digest
        != expected
        || public_statement.stage_claims_digest != expected
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream compact stage-claim proof surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn usize_from_u64(value: u64, label: &'static str) -> Result<usize, SimpleKernelError> {
    usize::try_from(value).map_err(|_| SimpleKernelError::Bridge(format!("RV32IM Nightstream {label} overflows usize")))
}

fn build_stage1_artifact_surface_from_verified_claims(
    stage1: &crate::rv32im::Stage1VerifiedClaims,
) -> Result<Stage1ArtifactSurface, SimpleKernelError> {
    if stage1.claim.digest != stage1.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage1 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage1.claim.mix != stage1.mix {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage1 selected-opening claim mix does not match the carried verified claim".into(),
        ));
    }
    if stage1.claim.rows_family_digest != stage1.rows_digest {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage1 selected-opening claim rows digest does not match the carried verified claim"
                .into(),
        ));
    }

    let rows = Stage1CanonicalRowBundle {
        rows_digest: stage1.rows_digest,
        digest: [0; 32],
    };
    let rows = Stage1CanonicalRowBundle {
        digest: rows.expected_digest(),
        ..rows
    };
    Ok(Stage1ArtifactSurface {
        rows,
        claim: Stage1ClaimSurface {
            row_count: usize_from_u64(stage1.claim.row_count, "stage1 row_count")?,
            effect_row_count: usize_from_u64(stage1.claim.effect_row_count, "stage1 effect_row_count")?,
            commit_row_count: usize_from_u64(stage1.claim.commit_row_count, "stage1 commit_row_count")?,
            real_row_count: usize_from_u64(stage1.claim.real_row_count, "stage1 real_row_count")?,
            preserves_x0_count: usize_from_u64(stage1.claim.preserves_x0_count, "stage1 preserves_x0_count")?,
            mix: stage1.mix,
        },
    })
}

fn build_stage2_artifact_surface_from_verified_claims(
    stage2: &crate::rv32im::Stage2VerifiedClaims,
) -> Result<Stage2ArtifactSurface, SimpleKernelError> {
    if stage2.claim.digest != stage2.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage2 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage2.claim.reg_mix != stage2.reg_mix || stage2.claim.ram_mix != stage2.ram_mix {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage2 selected-opening claim mixes do not match the carried verified claim".into(),
        ));
    }

    let families = Stage2CanonicalFamilyBundle {
        register_reads_digest: stage2.claim.register_reads_family_digest,
        register_writes_digest: stage2.claim.register_writes_family_digest,
        ram_events_digest: stage2.claim.ram_events_family_digest,
        twist_links_digest: stage2.claim.twist_links_family_digest,
        digest: [0; 32],
    };
    let families = Stage2CanonicalFamilyBundle {
        digest: families.expected_digest(),
        ..families
    };
    Ok(Stage2ArtifactSurface {
        families,
        claim: Stage2ClaimSurface {
            register_read_count: usize_from_u64(stage2.claim.register_read_count, "stage2 register_read_count")?,
            register_write_count: usize_from_u64(stage2.claim.register_write_count, "stage2 register_write_count")?,
            ram_event_count: usize_from_u64(stage2.claim.ram_event_count, "stage2 ram_event_count")?,
            twist_link_count: usize_from_u64(stage2.claim.twist_link_count, "stage2 twist_link_count")?,
            ram_read_count: usize_from_u64(stage2.claim.ram_read_count, "stage2 ram_read_count")?,
            ram_write_count: usize_from_u64(stage2.claim.ram_write_count, "stage2 ram_write_count")?,
            reg_mix: stage2.reg_mix,
            ram_mix: stage2.ram_mix,
        },
    })
}

fn build_stage3_artifact_surface_from_verified_claims(
    stage3: &crate::rv32im::Stage3VerifiedClaims,
) -> Result<Stage3ArtifactSurface, SimpleKernelError> {
    if stage3.claim.digest != stage3.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage3 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage3.claim.continuity_mix != stage3.continuity_mix {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream stage3 selected-opening claim mix does not match the carried verified claim".into(),
        ));
    }

    let continuity = Stage3CanonicalContinuityBundle {
        continuity_digest: stage3.claim.continuity_family_digest,
        digest: [0; 32],
    };
    let continuity = Stage3CanonicalContinuityBundle {
        digest: continuity.expected_digest(),
        ..continuity
    };
    Ok(Stage3ArtifactSurface {
        continuity,
        claim: Stage3ClaimSurface {
            continuity_count: usize_from_u64(stage3.claim.continuity_count, "stage3 continuity_count")?,
            final_step_count: usize_from_u64(stage3.claim.final_step_count, "stage3 final_step_count")?,
            halted: stage3.claim.halted,
            all_continuity_hold: stage3.claim.all_continuity_hold,
            continuity_mix: stage3.continuity_mix,
        },
    })
}

fn build_transcript_artifact_surface_from_verified_surface(
    transcript: &crate::rv32im::VerifiedTranscriptSurface,
) -> Result<TranscriptArtifactSurface, SimpleKernelError> {
    if transcript.digest != transcript.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream carried transcript surface digest mismatch".into(),
        ));
    }
    Ok(TranscriptArtifactSurface {
        commitment: StageDigestCommitment {
            digest: transcript.final_digest,
        },
        claim: TranscriptClaimSurface {
            final_digest: transcript.final_digest,
            event_count: transcript.event_count,
            kernel_final_mix: transcript.challenges.kernel_final_mix,
        },
    })
}

pub fn build_rv32im_stage_claim_bundle_from_side_proof_bundle(
    bundle: &Rv32imSideProofBundle,
    execution_digest: [u8; 32],
) -> Result<SimpleKernelStageClaimBundle, SimpleKernelError> {
    validate_rv32im_side_proof_bundle_structure(bundle)?;

    let claims = SimpleKernelStageClaimBundle {
        stage1: build_stage1_artifact_surface_from_verified_claims(&bundle.stage1)?,
        stage2: build_stage2_artifact_surface_from_verified_claims(&bundle.stage2)?,
        stage3: build_stage3_artifact_surface_from_verified_claims(&bundle.stage3)?,
        transcript: build_transcript_artifact_surface_from_verified_surface(&bundle.transcript)?,
        execution_digest,
        digest: [0; 32],
    };
    Ok(SimpleKernelStageClaimBundle {
        digest: claims.expected_digest(),
        ..claims
    })
}

pub(super) fn verify_rv32im_side_kernel_claim_proof_surface(
    side_bundle: &Rv32imSideProofBundle,
    public_statement: &Rv32imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_claim_proof_bridge.digest != side_bundle.kernel_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof kernel-claim proof bridge digest mismatch".into(),
        ));
    }
    let summary_digest = kernel_claim_summary_digest_from_surfaces(
        public_statement.prepared_step_bindings_digest,
        side_bundle.kernel_claim_bridge.root0_digest,
        public_statement.execution_digest,
        public_statement.final_state_digest,
        public_statement.transcript_final_digest,
        public_statement.final_pc,
        public_statement.halted,
    );
    let expected_bundle_digest = packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv32im/kernel_claim_proof_bundle",
        summary_digest,
        side_bundle
            .kernel_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.kernel_claim_proof_bridge.packaged_proof_digest,
    );
    if side_bundle
        .kernel_claim_proof_bridge
        .kernel_claim_proof_bundle_digest
        != expected_bundle_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream compact kernel-claim proof surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn main_lane_proof_binding_digest_from_surfaces(
    root_lane_columns_digest: [u8; 32],
    root_lane_commitment_digest: [u8; 32],
    fold_schedule: crate::proof::FoldSchedule,
    chunk_count: u64,
    public_step_count: u64,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_lane_proof_binding");
    tr.append_message(
        b"rv32im/main_lane_proof_binding/root_lane_columns_digest",
        &root_lane_columns_digest,
    );
    tr.append_message(
        b"rv32im/main_lane_proof_binding/root_lane_commitment_digest",
        &root_lane_commitment_digest,
    );
    tr.append_u64s(
        b"rv32im/main_lane_proof_binding/fold_schedule",
        &fold_schedule.meta_words(),
    );
    tr.append_u64s(
        b"rv32im/main_lane_proof_binding/meta",
        &[chunk_count, public_step_count],
    );
    tr.digest32()
}

fn main_lane_proof_bundle_digest_from_surfaces(
    binding_digest: [u8; 32],
    statement_digest: [u8; 32],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_lane_proof_bundle");
    tr.append_message(b"rv32im/main_lane_proof_bundle/binding_digest", &binding_digest);
    tr.append_message(b"rv32im/main_lane_proof_bundle/statement_digest", &statement_digest);
    tr.append_message(b"rv32im/main_lane_proof_bundle/proof_digest", &proof_digest);
    tr.digest32()
}

pub fn build_rv32im_kernel_opening_claim_from_side_proof_bundle(
    side_bundle: &Rv32imSideProofBundle,
    public_statement: &Rv32imProofStatement,
) -> Result<SimpleKernelOpeningClaim, SimpleKernelError> {
    validate_rv32im_side_proof_bundle_structure(side_bundle)?;
    if side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .digest
        != side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream prepared-step binding summary bridge digest mismatch".into(),
        ));
    }
    if side_bundle
        .kernel_opening_bridge
        .root_lane_commitment
        .digest
        != side_bundle
            .kernel_opening_bridge
            .root_lane_commitment
            .expected_digest()
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream root-lane commitment summary digest mismatch".into(),
        ));
    }
    let binding_summary = &side_bundle.kernel_opening_bridge.prepared_step_bindings;
    if binding_summary.binding_count != public_statement.public_step_count
        || side_bundle
            .kernel_opening_bridge
            .root_lane_commitment
            .time_len
            != public_statement.public_step_count
    {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream kernel-opening provenance summaries do not match the carried public step count".into(),
        ));
    }
    let stage_claims =
        build_rv32im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    let claim = build_public_kernel_opening_claim_from_compact_surfaces(
        &stage_claims,
        side_bundle.stage1.packaged_digest,
        side_bundle.stage2.packaged_digest,
        side_bundle.stage3.packaged_digest,
        public_statement.prepared_step_bindings_digest,
        binding_summary.binding_count,
        binding_summary.first_binding_digest,
        binding_summary.last_binding_digest,
        public_statement.execution_digest,
        public_statement.final_state_digest,
        public_statement.transcript_final_digest,
        public_statement.final_pc,
        public_statement.halted,
        &side_bundle.kernel_opening_bridge.root_lane_commitment,
    );
    Ok(claim)
}

pub(super) fn verify_rv32im_side_main_lane_proof_surface(
    side_bundle: &Rv32imSideProofBundle,
    public_statement: &Rv32imProofStatement,
) -> Result<[u8; 32], SimpleKernelError> {
    let binding_digest = main_lane_proof_binding_digest_from_surfaces(
        public_statement.root_lane_columns_digest,
        side_bundle
            .kernel_opening_bridge
            .root_lane_commitment
            .digest,
        public_statement.fold_schedule,
        public_statement.chunk_count,
        public_statement.public_step_count,
    );
    let expected_bundle_digest = main_lane_proof_bundle_digest_from_surfaces(
        binding_digest,
        side_bundle.main_lane_bridge.main_lane_statement_digest,
        side_bundle.main_lane_bridge.main_lane_proof_digest,
    );
    Ok(expected_bundle_digest)
}
