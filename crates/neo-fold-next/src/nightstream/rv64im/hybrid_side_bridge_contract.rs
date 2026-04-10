//! Owns the backend-facing contract for the current hybrid RV64IM side bridge.
//!
//! This module decides exactly what the backend shell is allowed to assume
//! about the witness-backed side bridge, and produces the single decider
//! relation/target that the backend proof binds to today.

use crate::decider::spartan2::{
    build_spartan2_self_bound_decider_relation, Spartan2DeciderRelation, Spartan2DeciderTarget,
};
use crate::finalize::{digest32_as_fields, FixedShapeChunkSummary};
use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{
    Rv64imAcceptedProofArtifact, Rv64imProofStatement, Rv64imStageClaimDigestBundle, SimpleKernelError,
};

use super::compact_surfaces::{
    kernel_claim_summary_digest_from_surfaces, kernel_opening_binding_bundle_digest_from_surfaces,
    kernel_opening_bundle_digest_from_surfaces, kernel_opening_proof_bundle_digest_from_surfaces,
    packaged_claim_proof_digest_from_surfaces, stage_package_proof_bundle_digest_from_surfaces,
};
use super::witness_backed_side_bridge::{
    build_rv64im_witness_backed_side_bridge_artifact_from_accepted_artifact,
    build_rv64im_witness_backed_side_bridge_statement, verify_rv64im_witness_backed_side_bridge_artifact,
    Rv64imWitnessBackedSideBridgeArtifact, Rv64imWitnessBackedSideBridgeStatement,
};
use super::{
    build_rv64im_kernel_opening_claim_from_side_proof_bundle, build_rv64im_stage_claim_bundle_from_side_proof_bundle,
    Rv64imSideProofBundle,
};

pub(super) type Rv64imHybridSideBridgeDeciderRelation = Spartan2DeciderRelation;

const RV64IM_HYBRID_SIDE_BRIDGE_BASE_COMPONENT_COUNT: usize = 4;
const RV64IM_HYBRID_SIDE_BRIDGE_MAX_CHUNK_TRANSITIONS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rv64imHybridSideBridgeBaseComponents {
    stage_claim_proof_bundle_digest: [u8; 32],
    stage_package_proof_bundle_digest: [u8; 32],
    kernel_opening_proof_bundle_digest: [u8; 32],
    kernel_claim_proof_bundle_digest: [u8; 32],
}

impl Rv64imHybridSideBridgeBaseComponents {
    fn ordered_digests(self) -> Vec<[u8; 32]> {
        vec![
            self.stage_claim_proof_bundle_digest,
            self.stage_package_proof_bundle_digest,
            self.kernel_opening_proof_bundle_digest,
            self.kernel_claim_proof_bundle_digest,
        ]
    }
}

pub(super) struct Rv64imHybridSideBridgeContract {
    relation: Rv64imHybridSideBridgeDeciderRelation,
}

impl Rv64imHybridSideBridgeContract {
    pub(super) fn from_bridge_artifact(
        nightstream_statement: &NightstreamStatement,
        bridge_handoff_digests: &[[u8; 32]],
        public_statement: &Rv64imProofStatement,
        bridge_artifact: &Rv64imWitnessBackedSideBridgeArtifact,
    ) -> Result<Self, SimpleKernelError> {
        let bridge_witness = &bridge_artifact.witness;
        let statement = build_rv64im_witness_backed_side_bridge_statement(
            nightstream_statement,
            bridge_handoff_digests,
            public_statement,
            bridge_witness.side_bundle.digest,
            bridge_witness.opening_artifact.digest,
        )?;
        verify_rv64im_witness_backed_side_bridge_artifact(&statement, bridge_artifact)?;
        Ok(Self {
            relation: build_rv64im_hybrid_side_bridge_relation(&statement, &bridge_witness.side_bundle)?,
        })
    }

    pub(super) fn from_accepted_artifact(
        nightstream_statement: &NightstreamStatement,
        bridge_handoff_digests: &[[u8; 32]],
        public_statement: &Rv64imProofStatement,
        side_bundle: &Rv64imSideProofBundle,
        accepted_artifact: &Rv64imAcceptedProofArtifact,
    ) -> Result<(Rv64imWitnessBackedSideBridgeArtifact, Self), SimpleKernelError> {
        let bridge_artifact = build_rv64im_witness_backed_side_bridge_artifact_from_accepted_artifact(
            nightstream_statement,
            bridge_handoff_digests,
            public_statement,
            side_bundle,
            accepted_artifact,
        )?;
        let statement = build_rv64im_witness_backed_side_bridge_statement(
            nightstream_statement,
            bridge_handoff_digests,
            public_statement,
            bridge_artifact.witness.side_bundle.digest,
            bridge_artifact.witness.opening_artifact.digest,
        )?;
        let base_component_digests = rv64im_hybrid_side_bridge_fast_path_from_accepted_artifact(
            public_statement,
            side_bundle,
            accepted_artifact,
        )?;
        Ok((
            bridge_artifact,
            Self {
                relation: build_rv64im_hybrid_side_bridge_relation_from_base_component_digests(
                    &statement,
                    base_component_digests,
                )?,
            },
        ))
    }

    pub(super) fn relation(&self) -> &Rv64imHybridSideBridgeDeciderRelation {
        &self.relation
    }

    pub(super) fn into_relation(self) -> Rv64imHybridSideBridgeDeciderRelation {
        self.relation
    }

    pub(super) fn target(&self) -> Spartan2DeciderTarget {
        self.relation.target()
    }
}

fn rv64im_hybrid_side_bridge_stage_package_proof_bundle_digest(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<[u8; 32], SimpleKernelError> {
    let digest = stage_package_proof_bundle_digest_from_surfaces(
        side_bundle.stage1.packaged_digest,
        side_bundle.stage2.packaged_digest,
        side_bundle.stage3.packaged_digest,
    );
    if digest != public_statement.stage_packages_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract compact stage-package proof surface does not match the carried RV64IM public statement"
                .into(),
        ));
    }
    Ok(digest)
}

fn rv64im_hybrid_side_bridge_stage_claim_proof_bundle_digest(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<[u8; 32], SimpleKernelError> {
    let claims =
        build_rv64im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    let summary = Rv64imStageClaimDigestBundle {
        claim_bundle_digest: claims.digest,
        stage1_digest: claims.stage1.rows.digest,
        stage2_digest: claims.stage2.families.digest,
        stage3_digest: claims.stage3.continuity.digest,
        transcript_digest: claims.transcript.commitment.digest,
        execution_digest: claims.execution_digest,
        digest: [0; 32],
    };
    let summary = Rv64imStageClaimDigestBundle {
        digest: summary.expected_digest(),
        ..summary
    };
    let digest = packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv64im/stage_claim_proof_bundle",
        summary.digest,
        side_bundle
            .stage_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.stage_claim_proof_bridge.packaged_proof_digest,
    );
    if digest
        != side_bundle
            .stage_claim_proof_bridge
            .stage_claim_proof_bundle_digest
        || digest != public_statement.stage_claims_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract compact stage-claim proof surface does not match the carried RV64IM public statement"
                .into(),
        ));
    }
    Ok(digest)
}

fn rv64im_hybrid_side_bridge_kernel_opening_proof_bundle_digest(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<[u8; 32], SimpleKernelError> {
    let claim = build_rv64im_kernel_opening_claim_from_side_proof_bundle(side_bundle, public_statement)?;
    let opening_digest = kernel_opening_bundle_digest_from_surfaces(
        claim.digest,
        side_bundle.kernel_opening_bridge.bindings_opening_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest,
    );
    let binding_digest = kernel_opening_binding_bundle_digest_from_surfaces(
        claim.digest,
        side_bundle.kernel_opening_bridge.bindings_opening_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest,
    );
    let digest = kernel_opening_proof_bundle_digest_from_surfaces(opening_digest, binding_digest);
    if digest != public_statement.kernel_opening_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract compact kernel-opening proof surface does not match the carried RV64IM public statement"
                .into(),
        ));
    }
    Ok(digest)
}

fn rv64im_hybrid_side_bridge_kernel_claim_proof_bundle_digest(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<[u8; 32], SimpleKernelError> {
    let summary_digest = kernel_claim_summary_digest_from_surfaces(
        public_statement.prepared_step_bindings_digest,
        side_bundle.kernel_claim_bridge.root0_digest,
        public_statement.execution_digest,
        public_statement.final_state_digest,
        public_statement.transcript_final_digest,
        public_statement.final_pc,
        public_statement.halted,
    );
    let digest = packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv64im/kernel_claim_proof_bundle",
        summary_digest,
        side_bundle
            .kernel_claim_proof_bridge
            .packaged_statement_digest,
        side_bundle.kernel_claim_proof_bridge.packaged_proof_digest,
    );
    if digest
        != side_bundle
            .kernel_claim_proof_bridge
            .kernel_claim_proof_bundle_digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract compact kernel-claim proof surface does not match the carried RV64IM public statement"
                .into(),
        ));
    }
    Ok(digest)
}

fn rv64im_hybrid_side_bridge_base_components(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<Rv64imHybridSideBridgeBaseComponents, SimpleKernelError> {
    Ok(Rv64imHybridSideBridgeBaseComponents {
        stage_claim_proof_bundle_digest: rv64im_hybrid_side_bridge_stage_claim_proof_bundle_digest(
            public_statement,
            side_bundle,
        )?,
        stage_package_proof_bundle_digest: rv64im_hybrid_side_bridge_stage_package_proof_bundle_digest(
            public_statement,
            side_bundle,
        )?,
        kernel_opening_proof_bundle_digest: rv64im_hybrid_side_bridge_kernel_opening_proof_bundle_digest(
            public_statement,
            side_bundle,
        )?,
        kernel_claim_proof_bundle_digest: rv64im_hybrid_side_bridge_kernel_claim_proof_bundle_digest(
            public_statement,
            side_bundle,
        )?,
    })
}

fn validate_rv64im_hybrid_side_bridge_contract_statement(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
) -> Result<(), SimpleKernelError> {
    if statement.public_statement.digest != statement.public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract public statement digest mismatch".into(),
        ));
    }
    if statement.nightstream_statement.public_io_digest != statement.public_statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract Nightstream public IO does not match the carried RV64IM statement"
                .into(),
        ));
    }
    if statement.nightstream_statement.fold_schedule != statement.public_statement.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract fold schedule does not match the carried Nightstream statement".into(),
        ));
    }
    if statement.nightstream_statement.chunk_summaries.len() as u64 != statement.public_statement.chunk_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract chunk count does not match the carried Nightstream statement".into(),
        ));
    }
    let public_step_count =
        crate::finalize::fixed_shape_chunk_coverage_terminal_index(&statement.nightstream_statement.chunk_summaries)
            .map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM hybrid side-bridge contract Nightstream chunk summaries are not contiguous: {err}"
                ))
            })?;
    if public_step_count != statement.public_statement.public_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract public-step count does not match the carried Nightstream statement"
                .into(),
        ));
    }
    if statement.bridge_handoff_digests.len() != statement.nightstream_statement.chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract handoff count does not match the carried Nightstream chunk summaries"
                .into(),
        ));
    }
    if statement.nightstream_statement.chunk_summaries.len() > RV64IM_HYBRID_SIDE_BRIDGE_MAX_CHUNK_TRANSITIONS {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM hybrid side-bridge contract chunk count {} exceeds the fixed compiler maximum {}",
            statement.nightstream_statement.chunk_summaries.len(),
            RV64IM_HYBRID_SIDE_BRIDGE_MAX_CHUNK_TRANSITIONS
        )));
    }
    if statement.opening_artifact_digest == [0; 32] {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract opening artifact digest must be nonzero".into(),
        ));
    }
    Ok(())
}

fn build_rv64im_hybrid_side_bridge_relation_from_base_component_digests(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    base_components: Rv64imHybridSideBridgeBaseComponents,
) -> Result<Rv64imHybridSideBridgeDeciderRelation, SimpleKernelError> {
    validate_rv64im_hybrid_side_bridge_contract_statement(statement)?;
    let base_component_digests = base_components.ordered_digests();
    if base_component_digests.len() != RV64IM_HYBRID_SIDE_BRIDGE_BASE_COMPONENT_COUNT {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract base-component layout drifted from the fixed compiler policy".into(),
        ));
    }
    let public_statement_digest = statement.nightstream_statement.core_digest();
    let relation_digest = statement.digest();
    let initial_handle_digest = digest32_as_fields(statement.nightstream_statement.core_digest());
    let padded_chunk_summaries = rv64im_hybrid_side_bridge_padded_chunk_summaries(statement);
    let padded_bridge_handoff_digests = rv64im_hybrid_side_bridge_padded_handoff_digests(statement);

    build_spartan2_self_bound_decider_relation(
        public_statement_digest,
        relation_digest,
        initial_handle_digest,
        statement.nightstream_statement.fold_schedule,
        statement.nightstream_statement.semantic_step_count,
        padded_chunk_summaries,
        base_component_digests,
        padded_bridge_handoff_digests,
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}

fn rv64im_hybrid_side_bridge_padded_chunk_summaries(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
) -> Vec<FixedShapeChunkSummary> {
    let mut chunk_summaries = statement.nightstream_statement.chunk_summaries.clone();
    let padding = FixedShapeChunkSummary {
        start_index: statement.nightstream_statement.semantic_step_count,
        public_step_count: 0,
        public_chunk_digest: [0; 32],
        chunk_relation_digest: [0; 32],
    };
    chunk_summaries.resize(RV64IM_HYBRID_SIDE_BRIDGE_MAX_CHUNK_TRANSITIONS, padding);
    chunk_summaries
}

fn rv64im_hybrid_side_bridge_padded_handoff_digests(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
) -> Vec<[u8; 32]> {
    let mut bridge_handoff_digests = statement.bridge_handoff_digests.clone();
    bridge_handoff_digests.resize(RV64IM_HYBRID_SIDE_BRIDGE_MAX_CHUNK_TRANSITIONS, [0; 32]);
    bridge_handoff_digests
}

fn build_rv64im_hybrid_side_bridge_relation(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    side_bundle: &Rv64imSideProofBundle,
) -> Result<Rv64imHybridSideBridgeDeciderRelation, SimpleKernelError> {
    if side_bundle.digest != side_bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract side-proof bundle digest mismatch".into(),
        ));
    }
    if side_bundle.digest != statement.side_bundle_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract side-proof bundle digest does not match the carried witness-backed side bridge statement"
                .into(),
        ));
    }
    if side_bundle.statement_core_digest != statement.nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge contract side-proof bundle does not match the carried Nightstream statement core"
                .into(),
        ));
    }
    let base_components = rv64im_hybrid_side_bridge_base_components(&statement.public_statement, side_bundle)?;
    build_rv64im_hybrid_side_bridge_relation_from_base_component_digests(statement, base_components)
}

fn guard_rv64im_hybrid_side_bridge_base_components_against_accepted_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(), SimpleKernelError> {
    if public_statement.digest != accepted_artifact.statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path public statement does not match the carried accepted artifact"
                .into(),
        ));
    }
    if public_statement.stage_claims_digest != accepted_artifact.stage_claims.digest
        || public_statement.stage_packages_digest != accepted_artifact.stage_packages.digest
        || public_statement.kernel_opening_digest != accepted_artifact.kernel_opening.digest
        || public_statement.prepared_step_bindings_digest
            != accepted_artifact
                .kernel_claims
                .prepared_step_bindings_digest()
        || public_statement.execution_digest != accepted_artifact.kernel_claims.execution_digest()
        || public_statement.final_state_digest != accepted_artifact.kernel_claims.final_state_digest()
        || public_statement.transcript_final_digest != accepted_artifact.kernel_claims.transcript_final_digest()
        || public_statement.final_pc != accepted_artifact.kernel_claims.final_pc()
        || public_statement.halted != accepted_artifact.kernel_claims.halted()
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path public statement component digests do not match the carried accepted artifact"
                .into(),
        ));
    }

    let stage_package_digest = stage_package_proof_bundle_digest_from_surfaces(
        side_bundle.stage1.packaged_digest,
        side_bundle.stage2.packaged_digest,
        side_bundle.stage3.packaged_digest,
    );
    if stage_package_digest != accepted_artifact.stage_packages.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path stage-package bridge does not match the carried accepted artifact"
                .into(),
        ));
    }
    if side_bundle
        .stage_claim_proof_bridge
        .packaged_statement_digest
        != accepted_artifact.stage_claims.packaged.statement.digest
        || side_bundle.stage_claim_proof_bridge.packaged_proof_digest
            != accepted_artifact.stage_claims.packaged.proof.proof_digest
        || side_bundle
            .stage_claim_proof_bridge
            .stage_claim_proof_bundle_digest
            != accepted_artifact.stage_claims.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path stage-claim bridge does not match the carried accepted artifact"
                .into(),
        ));
    }
    if side_bundle
        .kernel_opening_bridge
        .prepared_step_bindings
        .binding_count
        != accepted_artifact
            .root_execution
            .prepared_step_bindings
            .binding_count
        || side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .first_binding_digest
            != accepted_artifact
                .root_execution
                .prepared_step_bindings
                .first_binding_digest
        || side_bundle
            .kernel_opening_bridge
            .prepared_step_bindings
            .last_binding_digest
            != accepted_artifact
                .root_execution
                .prepared_step_bindings
                .last_binding_digest
        || side_bundle
            .kernel_opening_bridge
            .root_lane_commitment
            .digest
            != accepted_artifact.root_lane_commitment.digest
        || side_bundle.kernel_opening_bridge.bindings_opening_digest
            != accepted_artifact.kernel_opening.opening.bindings.digest
        || side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest
            != accepted_artifact
                .kernel_opening
                .opening
                .prepared_steps
                .digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path kernel-opening bridge does not match the carried accepted artifact"
                .into(),
        ));
    }
    if side_bundle.kernel_claim_bridge.stage1_digest != accepted_artifact.kernel_claims.claims.kernel.stage1_digest
        || side_bundle.kernel_claim_bridge.stage2_digest != accepted_artifact.kernel_claims.claims.kernel.stage2_digest
        || side_bundle.kernel_claim_bridge.stage3_digest != accepted_artifact.kernel_claims.claims.kernel.stage3_digest
        || side_bundle.kernel_claim_bridge.root0_digest != accepted_artifact.kernel_claims.root0_digest()
        || side_bundle.kernel_claim_bridge.kernel_claim_bundle_digest != accepted_artifact.claim.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path kernel-claim bridge does not match the carried accepted artifact"
                .into(),
        ));
    }
    if side_bundle
        .kernel_claim_proof_bridge
        .packaged_statement_digest
        != accepted_artifact.kernel_claims.packaged.statement.digest
        || side_bundle.kernel_claim_proof_bridge.packaged_proof_digest
            != accepted_artifact.kernel_claims.packaged.proof.proof_digest
        || side_bundle
            .kernel_claim_proof_bridge
            .kernel_claim_proof_bundle_digest
            != accepted_artifact.kernel_claims.digest
    {
        return Err(SimpleKernelError::Bridge(
            "RV64IM hybrid side-bridge accepted-artifact fast path kernel-claim proof bridge does not match the carried accepted artifact"
                .into(),
        ));
    }
    Ok(())
}

fn rv64im_hybrid_side_bridge_fast_path_from_accepted_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imHybridSideBridgeBaseComponents, SimpleKernelError> {
    guard_rv64im_hybrid_side_bridge_base_components_against_accepted_artifact(
        public_statement,
        side_bundle,
        accepted_artifact,
    )?;
    Ok(Rv64imHybridSideBridgeBaseComponents {
        stage_claim_proof_bundle_digest: accepted_artifact.stage_claims.digest,
        stage_package_proof_bundle_digest: accepted_artifact.stage_packages.digest,
        kernel_opening_proof_bundle_digest: accepted_artifact.kernel_opening.digest,
        kernel_claim_proof_bundle_digest: accepted_artifact.kernel_claims.digest,
    })
}
