//! Owns the RV64IM published Nightstream proof boundary above the current final/decider seam.

mod authoritative_side;
mod build_perf;
mod compact_surfaces;
mod opening_artifact;
mod side_bridges;
mod side_claim_relation;
mod side_eval_claim_relation;
mod side_opening_relation;
mod side_relation;
mod side_relation_circuit;
mod side_relation_spartan;
mod verify_perf;
mod witness_backed_side_bridge;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

pub use self::authoritative_side::{
    build_rv64im_authoritative_side_public_instance, build_rv64im_authoritative_side_statement,
    build_rv64im_side_proof_container, build_rv64im_side_proof_container_from_accepted_artifact,
    verify_rv64im_side_proof_container, Rv64imAuthoritativeSidePublicInstance, Rv64imAuthoritativeSideStatement,
    Rv64imEvalPublic, Rv64imOpenedObjectPublic, Rv64imSideProofContainer, Rv64imSideSurfacePublic,
    Rv64imSideSurfaceTarget,
};
pub use self::build_perf::{
    build_rv64im_nightstream_from_public_proof_with_perf, build_rv64im_nightstream_from_published_proof_seam_with_perf,
    Rv64imNightstreamBuildPerf, Rv64imNightstreamVerifiedSeamsBuildPerf,
};
use self::compact_surfaces::{
    kernel_claim_summary_digest_from_surfaces, kernel_opening_binding_bundle_digest_from_surfaces,
    kernel_opening_bundle_digest_from_surfaces, kernel_opening_proof_bundle_digest_from_surfaces,
    packaged_claim_proof_digest_from_surfaces, stage_package_proof_bundle_digest_from_surfaces,
};
pub use self::opening_artifact::Rv64imOpeningArtifact;
use self::side_bridges::{
    build_rv64im_kernel_claim_bridge_from_accepted_artifact,
    build_rv64im_kernel_claim_proof_bridge_from_accepted_artifact,
    build_rv64im_kernel_export_source_bridge_from_export_proof,
    build_rv64im_kernel_opening_bridge_from_accepted_artifact,
    build_rv64im_stage_claim_proof_bridge_from_accepted_artifact,
    build_rv64im_verified_side_claims_from_accepted_artifact_fast, validate_rv64im_side_proof_bundle_structure,
};
pub use self::side_bridges::{
    Rv64imKernelClaimBridge, Rv64imKernelClaimProofBridge, Rv64imKernelExportSourceBridge, Rv64imKernelOpeningBridge,
    Rv64imPreparedStepBindingSummaryBridge, Rv64imSideProofBundle, Rv64imStageClaimProofBridge,
};
pub use self::side_relation_spartan::{
    build_rv64im_side_spartan_from_accepted_artifact, debug_check_rv64im_side_spartan_circuit,
    measure_rv64im_side_spartan_circuit_constraints, prove_rv64im_side_spartan, setup_rv64im_side_spartan,
    setup_rv64im_side_spartan_cached, setup_rv64im_side_spartan_from_accepted_artifact, verify_rv64im_side_spartan,
    Rv64imSideSpartanProof, Rv64imSideSpartanProverKey, Rv64imSideSpartanVerifierKey,
};
pub use self::verify_perf::{verify_rv64im_nightstream_with_perf, Rv64imNightstreamVerifyPerf};
pub use self::witness_backed_side_bridge::Rv64imWitnessBackedSideBridgeStatement;

pub mod audit {
    pub use super::opening_artifact::{
        build_rv64im_opening_artifact_from_accepted_artifact, verify_rv64im_opening_artifact_from_accepted_artifact,
        verify_rv64im_opening_artifact_from_side_proof_bundle,
    };
    pub use super::side_claim_relation::{
        build_rv64im_side_claim_relation_from_accepted_artifact, build_rv64im_side_claim_relation_statement,
        build_rv64im_side_claim_relation_witness_from_accepted_artifact, verify_rv64im_side_claim_relation,
        Rv64imSideClaimRelationStatement, Rv64imSideClaimRelationWitness,
    };
    pub use super::side_eval_claim_relation::{
        build_rv64im_phase0_opened_object_bundle_from_claim_witnesses, build_rv64im_side_eval_claim_artifact,
        build_rv64im_side_eval_claim_artifact_from_accepted_artifact,
        build_rv64im_side_eval_claim_relation_from_accepted_artifact, build_rv64im_side_eval_claim_relation_statement,
        build_rv64im_side_eval_claim_relation_statement_from_artifact,
        build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact, verify_rv64im_side_eval_claim_artifact,
        verify_rv64im_side_eval_claim_relation, Rv64imPhase0OpenedObjectBundle, Rv64imPhase0OpenedObjectSummary,
        Rv64imPhase0OpeningTarget, Rv64imPhase0OpeningTargetBundle, Rv64imSideEvalClaimArtifact,
        Rv64imSideEvalClaimRelationStatement, Rv64imSideEvalClaimRelationWitness,
    };
    pub use super::side_opening_relation::{
        build_rv64im_side_opening_relation_from_accepted_artifact, build_rv64im_side_opening_relation_statement,
        build_rv64im_side_opening_relation_witness_from_accepted_artifact, verify_rv64im_side_opening_relation,
        Rv64imSideOpeningRelationStatement, Rv64imSideOpeningRelationWitness,
    };
    pub use super::side_relation::{
        build_rv64im_direct_side_relation_from_accepted_artifact,
        build_rv64im_direct_side_relation_witness_from_accepted_artifact, verify_rv64im_direct_side_relation,
        Rv64imDirectSideRelationWitness,
    };
    pub use super::side_relation_circuit::digests::{
        continuity_event_digest as circuit_continuity_event_digest, digest_u64_words as circuit_digest_u64_words,
        kernel_binding_opening_packaged_statement_digest as circuit_kernel_binding_opening_packaged_statement_digest,
        kernel_prepared_step_opening_packaged_statement_digest as circuit_kernel_prepared_step_opening_packaged_statement_digest,
        ram_event_digest as circuit_ram_event_digest, register_read_event_digest as circuit_register_read_event_digest,
        register_write_event_digest as circuit_register_write_event_digest,
        single_step_packaged_statement_digest as circuit_single_step_packaged_statement_digest,
        stage1_opening_packaged_statement_digest as circuit_stage1_opening_packaged_statement_digest,
        stage1_row_digest as circuit_stage1_row_digest,
        stage2_opening_packaged_statement_digest as circuit_stage2_opening_packaged_statement_digest,
        stage3_opening_packaged_statement_digest as circuit_stage3_opening_packaged_statement_digest,
        twist_link_event_digest as circuit_twist_link_event_digest,
    };
    pub use super::side_relation_circuit::exact_package::{
        exact_vector_packaged_step_digest_from_native_words as circuit_exact_vector_packaged_step_digest_from_native_words,
        exact_vector_packaged_step_digest_from_words as circuit_exact_vector_packaged_step_digest_from_words,
    };
    pub use super::side_relation_circuit::phase0::{
        derive_phase0_point as circuit_derive_phase0_point,
        enforce_commitment_root_and_opened_object_digest as circuit_enforce_phase0_commitment_root_and_opened_object_digest,
        enforce_payload_eq as circuit_enforce_phase0_payload_eq, enforce_point_eq as circuit_enforce_phase0_point_eq,
        evaluate_payload_from_packed_rows as circuit_evaluate_phase0_payload_from_packed_rows,
    };
    pub use super::{
        build_rv64im_kernel_opening_claim_from_side_proof_bundle,
        build_rv64im_side_proof_bundle_from_accepted_artifact, build_rv64im_stage_claim_bundle_from_side_proof_bundle,
        verify_rv64im_side_proof_bundle_from_accepted_artifact,
    };
}
use crate::finalize::fixed_shape_chunk_coverage_terminal_index;
use crate::nightstream::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use crate::rv64im::decider_relation::{
    build_rv64im_decider_relation_from_final_surface, validate_rv64im_decider_relation_surface, Rv64imDeciderRelation,
};
use crate::rv64im::final_relation::{
    verify_rv64im_final_statement_with_output, Rv64imFinalProof, Rv64imFinalStatement,
};
use crate::rv64im::kernel::{
    build_public_kernel_opening_claim_from_compact_surfaces, build_rv64im_kernel_export_proof_from_accepted_artifact,
    build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf,
    kernel_claim_bundle_from_statement_and_compact_surfaces, rv64im_public_chunk_digest, Rv64imAcceptedProofArtifact,
    Rv64imKernelExportProof, Rv64imProof, Rv64imProofStatement, Rv64imStageClaimDigestBundle, SimpleKernelError,
    SimpleKernelOpeningClaim, SimpleKernelStageClaimBundle, Stage1ArtifactSurface, Stage1CanonicalRowBundle,
    Stage1ClaimSurface, Stage2ArtifactSurface, Stage2CanonicalFamilyBundle, Stage2ClaimSurface, Stage3ArtifactSurface,
    Stage3CanonicalContinuityBundle, Stage3ClaimSurface, StageDigestCommitment, TranscriptArtifactSurface,
    TranscriptClaimSurface,
};
use crate::rv64im::{Rv64imSpartan2DeciderProof, Rv64imSpartan2DeciderVerifierKey};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imLinkageClaims {
    pub public_chunk_digests: Vec<[u8; 32]>,
    pub bridge_handoff_digests: Vec<[u8; 32]>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainDeciderProof {
    pub spartan_proof: Rv64imSpartan2DeciderProof,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imMainResidualProof {
    pub public_statement_digest: [u8; 32],
    pub decider_relation: Rv64imDeciderRelation,
    pub bridge_handoff_digests: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imSideDeciderProof {
    pub public_instance_digest: [u8; 32],
    pub proof_container: Rv64imSideProofContainer,
    pub spartan_proof: Rv64imSideSpartanProof,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imLinkageArtifact {
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imNightstreamProof {
    pub main_decider_proof: Rv64imMainDeciderProof,
    pub main_residual_proof: Rv64imMainResidualProof,
    pub side_decider_proof: Rv64imSideDeciderProof,
    pub linkage_artifact: Rv64imLinkageArtifact,
}

impl Rv64imLinkageClaims {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/linkage_claims");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/linkage_claims/version", b"v1");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/linkage_claims/counts",
            &[
                self.public_chunk_digests.len() as u64,
                self.bridge_handoff_digests.len() as u64,
            ],
        );
        for digest in &self.public_chunk_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/linkage_claims/public_chunk_digest",
                digest,
            );
        }
        for digest in &self.bridge_handoff_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/linkage_claims/bridge_handoff_digest",
                digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imMainDeciderProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/main_decider_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/main_decider_proof/version", b"v4");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_decider_proof/spartan_snark_data",
            &self.spartan_proof.snark_data,
        );
        tr.digest32()
    }
}

impl Rv64imMainResidualProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/main_residual_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/main_residual_proof/version", b"v1");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_residual_proof/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/main_residual_proof/decider_relation_digest",
            &self.decider_relation.digest,
        );
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/main_residual_proof/bridge_handoff_count",
            &[self.bridge_handoff_digests.len() as u64],
        );
        for digest in &self.bridge_handoff_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/main_residual_proof/bridge_handoff_digest",
                digest,
            );
        }
        tr.digest32()
    }
}

impl Rv64imSideDeciderProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_decider_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/side_decider_proof/version", b"v2");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_decider_proof/public_instance_digest",
            &self.public_instance_digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_decider_proof/proof_container_digest",
            &self.proof_container.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_decider_proof/spartan_snark_data",
            &self.spartan_proof.snark_data,
        );
        tr.digest32()
    }
}

fn build_rv64im_side_proof_bundle_from_accepted_artifact_and_kernel_export(
    artifact: &Rv64imAcceptedProofArtifact,
    kernel_export: &Rv64imKernelExportProof,
) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
    let (transcript, stage1, stage2, stage3, root_execution_digest) =
        build_rv64im_verified_side_claims_from_accepted_artifact_fast(artifact)?;
    let mut bundle = Rv64imSideProofBundle {
        statement_core_digest: [0; 32],
        transcript,
        stage1,
        stage2,
        stage3,
        stage_claim_proof_bridge: build_rv64im_stage_claim_proof_bridge_from_accepted_artifact(artifact),
        kernel_opening_bridge: build_rv64im_kernel_opening_bridge_from_accepted_artifact(artifact),
        kernel_claim_bridge: build_rv64im_kernel_claim_bridge_from_accepted_artifact(artifact),
        kernel_claim_proof_bridge: build_rv64im_kernel_claim_proof_bridge_from_accepted_artifact(artifact),
        kernel_export_bridge: build_rv64im_kernel_export_source_bridge_from_export_proof(kernel_export),
        semantic_rows_digest: artifact.root_execution.semantic_rows_digest,
        row_local_ccs_acceptance_digest: artifact.root_execution.row_local_ccs_acceptance.digest,
        execution_semantics_refinement_digest: artifact
            .root_execution
            .execution_semantics_refinement
            .digest,
        family_digest: artifact.root_execution.family_digest,
        root_execution_digest,
        digest: [0; 32],
    };
    bundle.digest = bundle.expected_digest();
    Ok(bundle)
}

pub fn build_rv64im_side_proof_bundle_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
    let (_, kernel_export, _) = build_rv64im_kernel_export_proof_from_accepted_artifact(artifact)?;
    build_rv64im_side_proof_bundle_from_accepted_artifact_and_kernel_export(artifact, &kernel_export)
}

pub fn build_rv64im_bound_side_proof_bundle_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
    let bundle = build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    bind_rv64im_side_proof_bundle_to_statement_core(&bundle, statement.core_digest())
}

pub(crate) fn bind_rv64im_side_proof_bundle_to_statement_core(
    bundle: &Rv64imSideProofBundle,
    statement_core_digest: [u8; 32],
) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof bundle digest mismatch".into(),
        ));
    }
    let mut bound = bundle.clone();
    bound.statement_core_digest = statement_core_digest;
    bound.digest = bound.expected_digest();
    Ok(bound)
}

pub fn verify_rv64im_side_proof_bundle_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
    bundle: &Rv64imSideProofBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof bundle digest mismatch".into(),
        ));
    }
    let expected = build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?;
    if &expected != bundle {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof bundle does not match the accepted artifact".into(),
        ));
    }
    Ok(())
}

fn derived_rv64im_row_chunk_routes_digest(
    statement: &NightstreamStatement,
) -> Result<([u8; 32], u64), SimpleKernelError> {
    let public_step_count = fixed_shape_chunk_coverage_terminal_index(&statement.chunk_summaries).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream chunk summaries do not form a contiguous fixed-shape route layout: {err}"
        ))
    })?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root_execution_row_chunk_routes");
    tr.append_u64s(b"rv64im/root_execution_row_chunk_routes/len", &[public_step_count]);
    for (chunk_index, summary) in statement.chunk_summaries.iter().enumerate() {
        for chunk_local_index in 0..summary.public_step_count {
            let logical_index = summary.start_index + chunk_local_index;
            let mut route_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root_execution_row_chunk_route");
            route_tr.append_u64s(
                b"rv64im/root_execution_row_chunk_route/meta",
                &[
                    logical_index,
                    chunk_index as u64,
                    summary.start_index,
                    chunk_local_index,
                ],
            );
            tr.append_message(b"rv64im/root_execution_row_chunk_routes/route", &route_tr.digest32());
        }
    }
    Ok((tr.digest32(), public_step_count))
}

pub(super) fn verify_rv64im_root_execution_surface_against_compact_surfaces(
    statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.transcript.final_digest != public_statement.transcript_final_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof transcript surface does not match the carried public statement".into(),
        ));
    }
    if statement.fold_schedule != public_statement.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream public statement fold schedule does not match the carried Nightstream statement".into(),
        ));
    }
    if statement.chunk_summaries.len() as u64 != public_statement.chunk_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream public statement chunk count does not match the carried Nightstream statement".into(),
        ));
    }

    let (row_chunk_routes_digest, public_step_count) = derived_rv64im_row_chunk_routes_digest(statement)?;
    if public_statement.public_step_count != public_step_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream public statement public-step count does not match the carried fixed-shape chunk summaries".into(),
        ));
    }

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root_execution_bundle");
    tr.append_message(
        b"rv64im/root_execution_bundle/semantic_rows_digest",
        &side_bundle.semantic_rows_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/prepared_step_bindings",
        &public_statement.prepared_step_bindings_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/row_chunk_routes_digest",
        &row_chunk_routes_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/row_local_ccs_acceptance_digest",
        &side_bundle.row_local_ccs_acceptance_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/execution_semantics_refinement_digest",
        &side_bundle.execution_semantics_refinement_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/family_digest",
        &side_bundle.family_digest,
    );
    tr.append_u64s(
        b"rv64im/root_execution_bundle/meta",
        &[
            public_step_count,
            public_step_count,
            public_step_count,
            public_step_count,
            public_step_count,
        ],
    );
    let expected_root_execution_digest = tr.digest32();
    if side_bundle.root_execution_digest != expected_root_execution_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact side-proof root-execution surface does not match the carried statement surfaces"
                .into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_side_kernel_claim_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
    main_lane_bundle_digest: [u8; 32],
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_claim_bridge.digest != side_bundle.kernel_claim_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof kernel-claim bridge digest mismatch".into(),
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
            "RV64IM Nightstream compact kernel-claim surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_side_stage_packages_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    let expected = stage_package_proof_bundle_digest_from_surfaces(
        side_bundle.stage1.packaged_digest,
        side_bundle.stage2.packaged_digest,
        side_bundle.stage3.packaged_digest,
    );
    if public_statement.stage_packages_digest != expected {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact stage-package proof surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn verify_rv64im_side_stage_claim_proof_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.stage_claim_proof_bridge.digest != side_bundle.stage_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof stage-claim proof bridge digest mismatch".into(),
        ));
    }
    let claims =
        build_rv64im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
    let summary = rv64im_stage_claim_digest_bundle_from_claims(&claims);
    let expected = packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv64im/stage_claim_proof_bundle",
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
            "RV64IM Nightstream compact stage-claim proof surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn usize_from_u64(value: u64, label: &'static str) -> Result<usize, SimpleKernelError> {
    usize::try_from(value).map_err(|_| SimpleKernelError::Bridge(format!("RV64IM Nightstream {label} overflows usize")))
}

fn build_stage1_artifact_surface_from_verified_claims(
    stage1: &crate::rv64im::Stage1VerifiedClaims,
) -> Result<Stage1ArtifactSurface, SimpleKernelError> {
    if stage1.claim.digest != stage1.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage1 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage1.claim.mix != stage1.mix {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage1 selected-opening claim mix does not match the carried verified claim".into(),
        ));
    }
    if stage1.claim.rows_family_digest != stage1.rows_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage1 selected-opening claim rows digest does not match the carried verified claim"
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
    stage2: &crate::rv64im::Stage2VerifiedClaims,
) -> Result<Stage2ArtifactSurface, SimpleKernelError> {
    if stage2.claim.digest != stage2.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage2 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage2.claim.reg_mix != stage2.reg_mix || stage2.claim.ram_mix != stage2.ram_mix {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage2 selected-opening claim mixes do not match the carried verified claim".into(),
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
    stage3: &crate::rv64im::Stage3VerifiedClaims,
) -> Result<Stage3ArtifactSurface, SimpleKernelError> {
    if stage3.claim.digest != stage3.claim.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage3 selected-opening claim digest mismatch".into(),
        ));
    }
    if stage3.claim.continuity_mix != stage3.continuity_mix {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream stage3 selected-opening claim mix does not match the carried verified claim".into(),
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
    transcript: &crate::rv64im::VerifiedTranscriptSurface,
) -> Result<TranscriptArtifactSurface, SimpleKernelError> {
    if transcript.digest != transcript.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream carried transcript surface digest mismatch".into(),
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

fn rv64im_stage_claim_digest_bundle_from_claims(claims: &SimpleKernelStageClaimBundle) -> Rv64imStageClaimDigestBundle {
    let summary = Rv64imStageClaimDigestBundle {
        claim_bundle_digest: claims.digest,
        stage1_digest: claims.stage1.rows.digest,
        stage2_digest: claims.stage2.families.digest,
        stage3_digest: claims.stage3.continuity.digest,
        transcript_digest: claims.transcript.commitment.digest,
        execution_digest: claims.execution_digest,
        digest: [0; 32],
    };
    Rv64imStageClaimDigestBundle {
        digest: summary.expected_digest(),
        ..summary
    }
}

pub fn build_rv64im_stage_claim_bundle_from_side_proof_bundle(
    bundle: &Rv64imSideProofBundle,
    execution_digest: [u8; 32],
) -> Result<SimpleKernelStageClaimBundle, SimpleKernelError> {
    validate_rv64im_side_proof_bundle_structure(bundle)?;

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

fn verify_rv64im_side_kernel_claim_proof_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_claim_proof_bridge.digest != side_bundle.kernel_claim_proof_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof kernel-claim proof bridge digest mismatch".into(),
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
        b"neo.fold.next/rv64im/kernel_claim_proof_bundle",
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
            "RV64IM Nightstream compact kernel-claim proof surface does not match the carried public statement".into(),
        ));
    }
    Ok(())
}

fn kernel_export_claim_summary_digest(final_state_digest: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_export_claim_terminal");
    tr.append_message(
        b"rv64im/kernel_export_claim_terminal/final_state_digest",
        &final_state_digest,
    );
    tr.digest32()
}

fn kernel_export_claim_proof_digest_from_surfaces(
    final_state_digest: [u8; 32],
    statement_digest: [u8; 32],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    packaged_claim_proof_digest_from_surfaces(
        b"neo.fold.next/rv64im/kernel_export_claim_proof",
        kernel_export_claim_summary_digest(final_state_digest),
        statement_digest,
        proof_digest,
    )
}

fn kernel_export_main_lane_proof_digest_from_surfaces(statement_digest: [u8; 32], proof_digest: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_export_main_lane_proof");
    tr.append_message(
        b"rv64im/kernel_export_main_lane_proof/statement_digest",
        &statement_digest,
    );
    tr.append_message(b"rv64im/kernel_export_main_lane_proof/proof_digest", &proof_digest);
    tr.digest32()
}

fn kernel_export_source_digest_from_surfaces(
    kernel_claims_digest: [u8; 32],
    main_lane_digest: [u8; 32],
    transcript_digest: [u8; 32],
    root_execution_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_export_source");
    tr.append_message(b"rv64im/kernel_export_source/kernel_claims", &kernel_claims_digest);
    tr.append_message(b"rv64im/kernel_export_source/main_lane", &main_lane_digest);
    tr.append_message(b"rv64im/kernel_export_source/transcript_digest", &transcript_digest);
    tr.append_message(b"rv64im/kernel_export_source/root_execution", &root_execution_digest);
    tr.digest32()
}

fn main_lane_proof_binding_digest_from_surfaces(
    root_lane_columns_digest: [u8; 32],
    root_lane_commitment_digest: [u8; 32],
    fold_schedule: crate::proof::FoldSchedule,
    chunk_count: u64,
    public_step_count: u64,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_proof_binding");
    tr.append_message(
        b"rv64im/main_lane_proof_binding/root_lane_columns_digest",
        &root_lane_columns_digest,
    );
    tr.append_message(
        b"rv64im/main_lane_proof_binding/root_lane_commitment_digest",
        &root_lane_commitment_digest,
    );
    tr.append_u64s(
        b"rv64im/main_lane_proof_binding/fold_schedule",
        &fold_schedule.meta_words(),
    );
    tr.append_u64s(
        b"rv64im/main_lane_proof_binding/meta",
        &[chunk_count, public_step_count],
    );
    tr.digest32()
}

fn main_lane_proof_bundle_digest_from_surfaces(
    binding_digest: [u8; 32],
    statement_digest: [u8; 32],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_lane_proof_bundle");
    tr.append_message(b"rv64im/main_lane_proof_bundle/binding_digest", &binding_digest);
    tr.append_message(b"rv64im/main_lane_proof_bundle/statement_digest", &statement_digest);
    tr.append_message(b"rv64im/main_lane_proof_bundle/proof_digest", &proof_digest);
    tr.digest32()
}

fn verify_rv64im_side_kernel_opening_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_opening_bridge.digest != side_bundle.kernel_opening_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof kernel-opening bridge digest mismatch".into(),
        ));
    }
    let claim = build_rv64im_kernel_opening_claim_from_side_proof_bundle(side_bundle, public_statement)?;
    let opening_bundle_digest = kernel_opening_bundle_digest_from_surfaces(
        claim.digest,
        side_bundle.kernel_opening_bridge.bindings_opening_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest,
    );
    let binding_bundle_digest = kernel_opening_binding_bundle_digest_from_surfaces(
        claim.digest,
        side_bundle.kernel_opening_bridge.bindings_opening_digest,
        side_bundle
            .kernel_opening_bridge
            .prepared_steps_opening_digest,
    );
    let expected_proof_bundle_digest =
        kernel_opening_proof_bundle_digest_from_surfaces(opening_bundle_digest, binding_bundle_digest);
    if public_statement.kernel_opening_digest != expected_proof_bundle_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact kernel-opening proof surface does not match the carried public statement"
                .into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_kernel_opening_claim_from_side_proof_bundle(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<SimpleKernelOpeningClaim, SimpleKernelError> {
    validate_rv64im_side_proof_bundle_structure(side_bundle)?;
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
            "RV64IM Nightstream prepared-step binding summary bridge digest mismatch".into(),
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
            "RV64IM Nightstream root-lane commitment summary digest mismatch".into(),
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
            "RV64IM Nightstream kernel-opening provenance summaries do not match the carried public step count".into(),
        ));
    }
    let stage_claims =
        build_rv64im_stage_claim_bundle_from_side_proof_bundle(side_bundle, public_statement.execution_digest)?;
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

fn verify_rv64im_side_main_lane_proof_surface(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
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
        side_bundle.kernel_export_bridge.main_lane_statement_digest,
        side_bundle.kernel_export_bridge.main_lane_proof_digest,
    );
    Ok(expected_bundle_digest)
}

pub(super) fn verify_rv64im_kernel_export_source_surface_against_compact_surfaces(
    side_bundle: &Rv64imSideProofBundle,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    if side_bundle.kernel_export_bridge.digest != side_bundle.kernel_export_bridge.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side-proof kernel-export bridge digest mismatch".into(),
        ));
    }
    let kernel_claims_digest = kernel_export_claim_proof_digest_from_surfaces(
        public_statement.final_state_digest,
        side_bundle
            .kernel_export_bridge
            .kernel_claim_statement_digest,
        side_bundle.kernel_export_bridge.kernel_claim_proof_digest,
    );
    let main_lane_digest = kernel_export_main_lane_proof_digest_from_surfaces(
        side_bundle.kernel_export_bridge.main_lane_statement_digest,
        side_bundle.kernel_export_bridge.main_lane_proof_digest,
    );
    let expected_source_digest = kernel_export_source_digest_from_surfaces(
        kernel_claims_digest,
        main_lane_digest,
        side_bundle.transcript.expected_digest(),
        side_bundle.root_execution_digest,
    );
    if side_bundle.kernel_export_bridge.kernel_export_source_digest != expected_source_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact kernel-export source surface does not match the carried public statement"
                .into(),
        ));
    }
    Ok(())
}

pub fn rv64im_verifier_context_digest(root_params_id: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/verifier_context");
    tr.append_message(b"neo.fold.next/nightstream/rv64im/verifier_context/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/verifier_context/root_params_id",
        &root_params_id,
    );
    tr.digest32()
}

pub fn build_rv64im_nightstream_statement_from_final(
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    verify_rv64im_final_statement_with_output(statement, proof)?;
    build_rv64im_nightstream_statement_from_relation(
        public_io_digest,
        verifier_context_digest,
        &build_rv64im_decider_relation_from_final_surface(statement, proof)?,
        linkage_root,
        proof_binding_root,
    )
}

pub fn build_rv64im_nightstream_statement_from_relation(
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    relation: &Rv64imDeciderRelation,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    validate_rv64im_decider_relation_surface(relation)?;
    Ok(NightstreamStatement {
        public_io_digest,
        verifier_context_digest,
        fold_schedule: relation.fold_schedule,
        semantic_step_count: relation.semantic_step_count,
        chunk_summaries: relation.chunk_summaries.clone(),
        linkage_root,
        proof_binding_root,
    })
}

pub fn build_rv64im_main_decider_proof(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainDeciderProof, SimpleKernelError> {
    let spartan_proof = crate::rv64im::prove_rv64im_spartan2_decider_cached(statement, proof)?;
    Ok(Rv64imMainDeciderProof { spartan_proof })
}

pub fn verify_rv64im_main_decider_proof(
    vk: &Rv64imSpartan2DeciderVerifierKey,
    main_residual_proof: &Rv64imMainResidualProof,
    main_decider_proof: &Rv64imMainDeciderProof,
) -> Result<(), SimpleKernelError> {
    crate::rv64im::verify_rv64im_spartan2_decider(
        vk,
        main_residual_proof.public_statement_digest,
        &main_residual_proof.decider_relation,
        &main_decider_proof.spartan_proof,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream main decider proof failed verification: {err}"
        ))
    })
}

pub fn build_rv64im_side_decider_proof(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideDeciderProof, SimpleKernelError> {
    let proof_container = build_rv64im_side_proof_container_from_accepted_artifact(
        nightstream_statement,
        &accepted_artifact.statement,
        side_bundle,
        accepted_artifact,
    )?;
    let (side_statement, side_witness) =
        build_rv64im_side_spartan_from_accepted_artifact(nightstream_statement, side_bundle, accepted_artifact)?;
    let keys = setup_rv64im_side_spartan_cached(&side_statement, &side_witness)?;
    let spartan_proof = prove_rv64im_side_spartan(&keys.as_ref().0, &side_statement, &side_witness)?;
    Ok(Rv64imSideDeciderProof {
        public_instance_digest: side_statement.public_instance_digest,
        proof_container,
        spartan_proof,
    })
}

pub fn verify_rv64im_side_decider_proof(
    vk: &Rv64imSideSpartanVerifierKey,
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    side_decider_proof: &Rv64imSideDeciderProof,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_side_proof_container(
        nightstream_statement,
        public_statement,
        &side_decider_proof.proof_container,
    )?;
    if side_decider_proof.proof_container.public_instance.digest != side_decider_proof.public_instance_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream side decider proof public-instance digest does not match the carried proof container"
                .into(),
        ));
    }
    let side_statement = build_rv64im_authoritative_side_statement(
        nightstream_statement,
        &side_decider_proof.proof_container.public_instance,
    )?;
    verify_rv64im_side_spartan(vk, &side_statement, &side_decider_proof.spartan_proof)
}

pub fn build_rv64im_nightstream_linkage_claims(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
    let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
    let mut claims = Rv64imLinkageClaims {
        public_chunk_digests: verified_kernel
            .chunk_handoffs
            .iter()
            .map(|handoff| rv64im_public_chunk_digest(&handoff.public_chunk))
            .collect(),
        bridge_handoff_digests: verified_kernel
            .chunk_handoffs
            .iter()
            .map(|handoff| handoff.bridge_handoff.digest)
            .collect(),
        digest: [0; 32],
    };
    claims.digest = claims.expected_digest();
    Ok(claims)
}

pub fn build_rv64im_nightstream_linkage_claims_from_relation(
    relation: &Rv64imDeciderRelation,
    bridge_handoff_digests: &[[u8; 32]],
) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
    validate_rv64im_decider_relation_surface(relation)?;
    if relation.chunk_summaries.len() != bridge_handoff_digests.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage handoff count does not match the carried decider relation".into(),
        ));
    }
    let mut claims = Rv64imLinkageClaims {
        public_chunk_digests: relation
            .chunk_summaries
            .iter()
            .map(|summary| summary.public_chunk_digest)
            .collect(),
        bridge_handoff_digests: bridge_handoff_digests.to_vec(),
        digest: [0; 32],
    };
    claims.digest = claims.expected_digest();
    Ok(claims)
}

pub fn rv64im_nightstream_linkage_root(
    kernel_export_anchor_digest: [u8; 32],
    linkage_claims: &Rv64imLinkageClaims,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/linkage_root");
    tr.append_message(b"neo.fold.next/nightstream/rv64im/linkage_root/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/linkage_root/kernel_export_anchor_digest",
        &kernel_export_anchor_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/linkage_root/linkage_claims_digest",
        &linkage_claims.digest,
    );
    tr.digest32()
}

pub fn build_rv64im_linkage_artifact(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imLinkageArtifact, SimpleKernelError> {
    Ok(Rv64imLinkageArtifact {
        digest: build_rv64im_nightstream_linkage_claims(statement, proof)?.digest,
    })
}

pub fn verify_rv64im_linkage_artifact(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
    linkage_artifact: &Rv64imLinkageArtifact,
) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
    let expected = build_rv64im_nightstream_linkage_claims(statement, proof)?;
    if linkage_artifact.digest != expected.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage artifact does not match the verified final seam".into(),
        ));
    }
    Ok(expected)
}

pub fn build_rv64im_main_residual_proof(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainResidualProof, SimpleKernelError> {
    verify_rv64im_final_statement_with_output(statement, proof)?;
    let decider_relation = build_rv64im_decider_relation_from_final_surface(statement, proof)?;
    let linkage_claims = build_rv64im_nightstream_linkage_claims(statement, proof)?;
    Ok(Rv64imMainResidualProof {
        public_statement_digest: statement.public_statement_digest,
        decider_relation,
        bridge_handoff_digests: linkage_claims.bridge_handoff_digests,
    })
}

pub fn verify_rv64im_main_residual_proof(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
    residual: &Rv64imMainResidualProof,
) -> Result<(), SimpleKernelError> {
    let expected = build_rv64im_main_residual_proof(statement, proof)?;
    if expected != *residual {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream main residual proof does not match the carried final proof seam".into(),
        ));
    }
    Ok(())
}

fn rv64im_kernel_export_anchor_digest_from_relation(
    relation: &Rv64imDeciderRelation,
) -> Result<[u8; 32], SimpleKernelError> {
    validate_rv64im_decider_relation_surface(relation)?;
    relation
        .base_component_digests
        .first()
        .copied()
        .ok_or_else(|| {
            SimpleKernelError::Bridge(
                "RV64IM Nightstream decider relation is missing the kernel export anchor digest".into(),
            )
        })
}

pub fn build_rv64im_linkage_artifact_from_claims(
    linkage_claims: &Rv64imLinkageClaims,
) -> Result<Rv64imLinkageArtifact, SimpleKernelError> {
    if linkage_claims.digest != linkage_claims.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage claims digest mismatch".into(),
        ));
    }
    Ok(Rv64imLinkageArtifact {
        digest: linkage_claims.digest,
    })
}

pub fn verify_rv64im_linkage_artifact_from_claims(
    linkage_claims: &Rv64imLinkageClaims,
    linkage_artifact: &Rv64imLinkageArtifact,
) -> Result<(), SimpleKernelError> {
    let expected = build_rv64im_linkage_artifact_from_claims(linkage_claims)?;
    if &expected != linkage_artifact {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage artifact does not match the verified linkage claims".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_nightstream_from_public_proof(
    proof: &Rv64imProof,
) -> Result<(NightstreamStatement, Rv64imNightstreamProof), SimpleKernelError> {
    build_rv64im_nightstream_from_public_proof_with_perf(proof).map(|(built, _)| built)
}

fn verify_rv64im_nightstream_carried_boundary(
    statement: &NightstreamStatement,
    proof: &Rv64imNightstreamProof,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_decider_relation_surface(&proof.main_residual_proof.decider_relation).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream main residual proof carries an invalid decider relation: {err}"
        ))
    })?;
    let linkage_claims = build_rv64im_nightstream_linkage_claims_from_relation(
        &proof.main_residual_proof.decider_relation,
        &proof.main_residual_proof.bridge_handoff_digests,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream main residual proof failed to rebuild linkage claims: {err}"
        ))
    })?;
    verify_rv64im_linkage_artifact_from_claims(&linkage_claims, &proof.linkage_artifact)?;

    let linkage_root = rv64im_nightstream_linkage_root(
        rv64im_kernel_export_anchor_digest_from_relation(&proof.main_residual_proof.decider_relation).map_err(
            |err| {
                SimpleKernelError::Bridge(format!(
                    "RV64IM Nightstream main residual proof is missing the kernel export anchor digest: {err}"
                ))
            },
        )?,
        &linkage_claims,
    );
    let mut expected_statement = build_rv64im_nightstream_statement_from_relation(
        proof.main_residual_proof.public_statement_digest,
        statement.verifier_context_digest,
        &proof.main_residual_proof.decider_relation,
        linkage_root,
        [0; 32],
    )?;
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_decider_proof_digest: proof.main_decider_proof.expected_digest(),
        main_residual_proof_digest: proof.main_residual_proof.expected_digest(),
        side_bridge_artifact_digest: proof.side_decider_proof.expected_digest(),
        linkage_artifact_digest: proof.linkage_artifact.digest,
    };
    expected_statement.proof_binding_root =
        nightstream_proof_binding_root(expected_statement.core_digest(), &proof_binding_inputs);
    if &expected_statement != statement {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream statement does not match the verified final seam".into(),
        ));
    }
    Ok(())
}

pub fn verify_rv64im_nightstream(
    statement: &NightstreamStatement,
    proof: &Rv64imNightstreamProof,
    trusted_root_params_id: [u8; 32],
    decider_vk: &Rv64imSpartan2DeciderVerifierKey,
    side_decider_vk: &Rv64imSideSpartanVerifierKey,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_nightstream_with_perf(
        statement,
        proof,
        trusted_root_params_id,
        decider_vk,
        side_decider_vk,
        public_statement,
    )
    .map(|_| ())
}
