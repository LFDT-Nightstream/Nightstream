//! Owns the RV64IM published Nightstream proof boundary above the current final/decider seam.

mod authoritative_side;
mod build_perf;
mod compact_surfaces;
mod opening_artifact;
mod side_bridges;
mod side_claim_relation;
mod side_eval_claim_relation;
mod side_opening_relation;
mod side_opening_spartan;
mod side_relation_circuit;
mod side_relation_spartan;
mod side_runtime_binding;
mod verify_perf;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use self::authoritative_side::build_rv64im_side_opening_public;
pub use self::authoritative_side::{
    build_rv64im_side_binding_statement, validate_rv64im_side_opening_public, verify_rv64im_side_opening_native,
    Rv64imEvalPublic, Rv64imOpenedObjectPublic, Rv64imSideBindingStatement, Rv64imSideOpeningProof,
    Rv64imSideOpeningPublic, Rv64imSideSurfacePublic, Rv64imSideSurfaceTarget,
};
pub use self::build_perf::{
    build_rv64im_nightstream_from_public_proof_with_perf, build_rv64im_nightstream_from_published_proof_seam_with_perf,
    Rv64imNightstreamBuildPerf, Rv64imNightstreamVerifiedSeamsBuildPerf,
};
use self::compact_surfaces::{kernel_claim_summary_digest_from_surfaces, packaged_claim_proof_digest_from_surfaces};
use self::side_bridges::Rv64imSideProofBundle;
use self::side_bridges::{
    build_rv64im_kernel_claim_bridge_from_accepted_artifact,
    build_rv64im_kernel_claim_proof_bridge_from_accepted_artifact,
    build_rv64im_kernel_export_source_bridge_from_export_proof,
    build_rv64im_kernel_opening_bridge_from_accepted_artifact,
    build_rv64im_stage_claim_proof_bridge_from_accepted_artifact,
    build_rv64im_verified_side_claims_from_accepted_artifact_fast, validate_rv64im_side_proof_bundle_structure,
};
use self::side_opening_relation::{
    build_rv64im_side_opening_relation_statement, build_rv64im_side_opening_relation_witness_from_accepted_artifact,
    Rv64imSideOpeningRelationStatement,
};
use self::side_opening_spartan::{
    prove_rv64im_side_opening_spartan, setup_rv64im_side_opening_spartan_cached, verify_rv64im_side_opening_spartan,
};
pub use self::side_opening_spartan::{Rv64imSideOpeningSpartanProof, Rv64imSideOpeningSpartanVerifierKey};
use self::side_relation_spartan::{
    prove_rv64im_side_binding, setup_rv64im_side_binding_cached, verify_rv64im_side_binding,
};
pub use self::side_relation_spartan::{Rv64imSideBindingProof, Rv64imSideBindingVerifierKey};
use self::side_runtime_binding::verify_rv64im_side_opening_statement_against_runtime_surfaces;
pub use self::verify_perf::{verify_rv64im_nightstream_with_perf, Rv64imNightstreamVerifyPerf};
use crate::rv64im::main_proof::Rv64imPublishedStatement;
pub use crate::rv64im::{build_rv64im_main_proof, verify_rv64im_main_proof, Rv64imMainProof};

pub mod audit {
    use crate::nightstream::rv64im::Rv64imLinkageClaims;
    use crate::nightstream::NightstreamStatement;
    use crate::rv64im::audit::Rv64imDeciderRelation;
    use crate::rv64im::final_relation::{Rv64imFinalBuildProof, Rv64imFinalStatement};
    use crate::rv64im::kernel::{Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError};
    use crate::rv64im::main_proof::Rv64imPublishedStatement;

    pub use super::authoritative_side::{
        build_rv64im_side_opening_public, build_rv64im_side_surface_public,
        verify_rv64im_side_surface_public_against_bundle,
    };
    pub use super::opening_artifact::{
        build_rv64im_opening_artifact_from_accepted_artifact, verify_rv64im_opening_artifact_from_accepted_artifact,
        verify_rv64im_opening_artifact_from_side_proof_bundle, Rv64imOpeningArtifact,
    };
    pub use super::side_bridges::{
        Rv64imKernelClaimBridge, Rv64imKernelClaimProofBridge, Rv64imKernelExportSourceBridge,
        Rv64imKernelOpeningBridge, Rv64imPreparedStepBindingSummaryBridge, Rv64imSideProofBundle,
        Rv64imStageClaimProofBridge,
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
    pub fn derive_rv64im_root_execution_digest_from_compact_surfaces(
        statement: &NightstreamStatement,
        public_statement: &Rv64imProofStatement,
        semantic_rows_digest: [u8; 32],
        row_local_ccs_acceptance_digest: [u8; 32],
        execution_semantics_refinement_digest: [u8; 32],
        family_digest: [u8; 32],
    ) -> Result<[u8; 32], SimpleKernelError> {
        super::derive_rv64im_root_execution_digest_from_compact_surfaces(
            statement,
            public_statement,
            semantic_rows_digest,
            row_local_ccs_acceptance_digest,
            execution_semantics_refinement_digest,
            family_digest,
        )
    }
    pub use super::side_opening_spartan::{
        debug_check_rv64im_side_opening_spartan_circuit, debug_compare_rv64im_side_opening_spartan_setup_shape,
        debug_compare_rv64im_side_opening_spartan_statement_owned_shape,
        debug_compare_rv64im_side_opening_spartan_without_packaged_final_main_claims_shape,
        debug_compare_rv64im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape,
        debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape,
        debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape,
        debug_measure_rv64im_side_opening_spartan_circuit_shape, debug_native_stage1_packaged_statement_digest,
        debug_round_trip_rv64im_stage1_packaged_opening_digest_with_reduced_setup,
        debug_setup_rv64im_side_opening_spartan_without_packaged_final_main_claims,
        debug_setup_rv64im_side_opening_spartan_without_stage1_packaged_final_main_claims,
        prove_rv64im_side_opening_spartan, setup_rv64im_side_opening_spartan, setup_rv64im_side_opening_spartan_cached,
        verify_rv64im_side_opening_spartan, Rv64imSideOpeningSpartanCircuitShape, Rv64imSideOpeningSpartanProof,
        Rv64imSideOpeningSpartanProverKey, Rv64imSideOpeningSpartanVerifierKey,
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
    pub use super::side_relation_spartan::{
        debug_check_rv64im_side_binding_circuit, measure_rv64im_side_binding_circuit_constraints,
        prove_rv64im_side_binding, setup_rv64im_side_binding, setup_rv64im_side_binding_cached,
        verify_rv64im_side_binding, Rv64imSideBindingProverKey, Rv64imSideBindingVerifierKey,
    };

    pub fn build_rv64im_side_proof_bundle_from_accepted_artifact(
        artifact: &Rv64imAcceptedProofArtifact,
    ) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
        super::build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)
    }

    pub fn verify_rv64im_side_proof_bundle_from_accepted_artifact(
        artifact: &Rv64imAcceptedProofArtifact,
        bundle: &Rv64imSideProofBundle,
    ) -> Result<(), SimpleKernelError> {
        super::verify_rv64im_side_proof_bundle_from_accepted_artifact(artifact, bundle)
    }

    pub fn build_rv64im_stage_claim_bundle_from_side_proof_bundle(
        bundle: &Rv64imSideProofBundle,
        execution_digest: [u8; 32],
    ) -> Result<crate::rv64im::kernel::SimpleKernelStageClaimBundle, SimpleKernelError> {
        super::build_rv64im_stage_claim_bundle_from_side_proof_bundle(bundle, execution_digest)
    }

    pub fn build_rv64im_kernel_opening_claim_from_side_proof_bundle(
        bundle: &Rv64imSideProofBundle,
        public_statement: &Rv64imProofStatement,
    ) -> Result<crate::rv64im::kernel::SimpleKernelOpeningClaim, SimpleKernelError> {
        super::build_rv64im_kernel_opening_claim_from_side_proof_bundle(bundle, public_statement)
    }

    pub fn build_rv64im_bound_side_proof_bundle_from_accepted_artifact(
        statement: &NightstreamStatement,
        artifact: &Rv64imAcceptedProofArtifact,
    ) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
        super::bind_rv64im_side_proof_bundle_to_statement_core(
            &super::build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?,
            statement.core_digest(),
        )
    }

    pub fn build_rv64im_nightstream_statement_from_final(
        public_io_digest: [u8; 32],
        verifier_context_digest: [u8; 32],
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
        linkage_root: [u8; 32],
        proof_binding_root: [u8; 32],
    ) -> Result<NightstreamStatement, SimpleKernelError> {
        super::build_rv64im_nightstream_statement_from_final(
            public_io_digest,
            verifier_context_digest,
            statement,
            proof,
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
        super::build_rv64im_nightstream_statement_from_relation(
            public_io_digest,
            verifier_context_digest,
            relation,
            linkage_root,
            proof_binding_root,
        )
    }

    pub fn build_rv64im_nightstream_statement_from_published_statement(
        verifier_context_digest: [u8; 32],
        published_statement: &Rv64imPublishedStatement,
        chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
        linkage_root: [u8; 32],
        proof_binding_root: [u8; 32],
    ) -> Result<NightstreamStatement, SimpleKernelError> {
        super::build_rv64im_nightstream_statement_from_published_statement(
            verifier_context_digest,
            published_statement,
            chunk_summaries,
            linkage_root,
            proof_binding_root,
        )
    }

    pub fn build_rv64im_nightstream_linkage_claims(
        statement: &Rv64imFinalStatement,
        proof: &Rv64imFinalBuildProof,
    ) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
        super::build_rv64im_nightstream_linkage_claims(statement, proof)
    }

    pub fn validate_rv64im_nightstream_linkage_claims_against_statement(
        statement: &NightstreamStatement,
        linkage_claims: &Rv64imLinkageClaims,
    ) -> Result<(), SimpleKernelError> {
        super::validate_rv64im_nightstream_linkage_claims_against_statement(statement, linkage_claims)
    }
}
use crate::finalize::fixed_shape_chunk_coverage_terminal_index;
use crate::nightstream::{nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement};
use crate::rv64im::final_relation::{
    verify_rv64im_final_statement_with_output, Rv64imFinalBuildProof, Rv64imFinalStatement,
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imLinkageClaims {
    public_chunk_digests: Vec<[u8; 32]>,
    digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imSideProof {
    opening_public: Rv64imSideOpeningPublic,
    opening_statement: Rv64imSideOpeningRelationStatement,
    opening: Rv64imSideOpeningSpartanProof,
    binding: Rv64imSideBindingProof,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imNightstreamProof {
    main_proof: Rv64imMainProof,
    linkage_claims: Rv64imLinkageClaims,
    side_proof: Rv64imSideProof,
}

impl Rv64imLinkageClaims {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/linkage_claims");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/linkage_claims/version", b"v2");
        tr.append_u64s(
            b"neo.fold.next/nightstream/rv64im/linkage_claims/counts",
            &[self.public_chunk_digests.len() as u64],
        );
        for digest in &self.public_chunk_digests {
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/linkage_claims/public_chunk_digest",
                digest,
            );
        }
        tr.digest32()
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn public_chunk_digests(&self) -> &[[u8; 32]] {
        &self.public_chunk_digests
    }

    pub fn public_chunk_digests_mut(&mut self) -> &mut [[u8; 32]] {
        &mut self.public_chunk_digests
    }

    pub fn digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.digest
    }
}

impl Rv64imSideProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv64im/side_proof/version", b"v7");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_proof/public_digest",
            &self.opening_public.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_proof/opening_statement_digest",
            &self.opening_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_proof/opening_snark_data",
            &self.opening.snark_data,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/side_proof/binding_snark_data",
            &self.binding.snark_data,
        );
        tr.digest32()
    }

    pub fn binding_statement(
        &self,
        nightstream_statement: &NightstreamStatement,
    ) -> Result<Rv64imSideBindingStatement, SimpleKernelError> {
        build_rv64im_side_binding_statement(nightstream_statement, &self.opening_public)
    }

    pub fn opening_public(&self) -> &Rv64imSideOpeningPublic {
        &self.opening_public
    }

    pub fn opening_public_mut(&mut self) -> &mut Rv64imSideOpeningPublic {
        &mut self.opening_public
    }

    pub fn opening_statement(&self) -> &Rv64imSideOpeningRelationStatement {
        &self.opening_statement
    }

    pub fn opening_statement_mut(&mut self) -> &mut Rv64imSideOpeningRelationStatement {
        &mut self.opening_statement
    }

    pub fn opening(&self) -> &Rv64imSideOpeningSpartanProof {
        &self.opening
    }

    pub fn opening_mut(&mut self) -> &mut Rv64imSideOpeningSpartanProof {
        &mut self.opening
    }

    pub fn binding(&self) -> &Rv64imSideBindingProof {
        &self.binding
    }

    pub fn binding_mut(&mut self) -> &mut Rv64imSideBindingProof {
        &mut self.binding
    }
}

impl Rv64imNightstreamProof {
    pub fn main_proof(&self) -> &Rv64imMainProof {
        &self.main_proof
    }

    pub fn main_proof_mut(&mut self) -> &mut Rv64imMainProof {
        &mut self.main_proof
    }

    pub fn linkage_claims(&self) -> &Rv64imLinkageClaims {
        &self.linkage_claims
    }

    pub fn linkage_claims_mut(&mut self) -> &mut Rv64imLinkageClaims {
        &mut self.linkage_claims
    }

    pub fn side_proof(&self) -> &Rv64imSideProof {
        &self.side_proof
    }

    pub fn side_proof_mut(&mut self) -> &mut Rv64imSideProof {
        &mut self.side_proof
    }
}

fn build_rv64im_nightstream_linkage_claims_from_parts(public_chunk_digests: Vec<[u8; 32]>) -> Rv64imLinkageClaims {
    let mut claims = Rv64imLinkageClaims {
        public_chunk_digests,
        digest: [0; 32],
    };
    claims.digest = claims.expected_digest();
    claims
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

fn build_rv64im_side_proof_bundle_from_accepted_artifact(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProofBundle, SimpleKernelError> {
    let (_, kernel_export, _) = build_rv64im_kernel_export_proof_from_accepted_artifact(artifact)?;
    build_rv64im_side_proof_bundle_from_accepted_artifact_and_kernel_export(artifact, &kernel_export)
}

pub fn build_rv64im_bound_side_opening_public_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideOpeningPublic, SimpleKernelError> {
    let side_bundle = bind_rv64im_side_proof_bundle_to_statement_core(
        &build_rv64im_side_proof_bundle_from_accepted_artifact(artifact)?,
        statement.core_digest(),
    )?;
    let opening =
        side_eval_claim_relation::build_rv64im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
            &artifact.statement,
            &side_bundle,
            artifact,
        )?;
    build_rv64im_side_opening_public(&side_bundle, &opening)
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

fn verify_rv64im_side_proof_bundle_from_accepted_artifact(
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

fn derive_rv64im_root_execution_digest_from_compact_surfaces(
    statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    semantic_rows_digest: [u8; 32],
    row_local_ccs_acceptance_digest: [u8; 32],
    execution_semantics_refinement_digest: [u8; 32],
    family_digest: [u8; 32],
) -> Result<[u8; 32], SimpleKernelError> {
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
        &semantic_rows_digest,
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
        &row_local_ccs_acceptance_digest,
    );
    tr.append_message(
        b"rv64im/root_execution_bundle/execution_semantics_refinement_digest",
        &execution_semantics_refinement_digest,
    );
    tr.append_message(b"rv64im/root_execution_bundle/family_digest", &family_digest);
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
    Ok(tr.digest32())
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

fn build_rv64im_stage_claim_bundle_from_side_proof_bundle(
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

fn build_rv64im_kernel_opening_claim_from_side_proof_bundle(
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

pub fn rv64im_verifier_context_digest(root_params_id: [u8; 32]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/verifier_context");
    tr.append_message(b"neo.fold.next/nightstream/rv64im/verifier_context/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/verifier_context/root_params_id",
        &root_params_id,
    );
    tr.digest32()
}

fn build_rv64im_nightstream_statement_from_final(
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    verify_rv64im_final_statement_with_output(statement, proof)?;
    Ok(NightstreamStatement {
        public_io_digest,
        verifier_context_digest,
        fold_schedule: statement.folded.fold_schedule,
        semantic_step_count: statement.folded.semantic_step_count,
        chunk_summaries: proof.chunk_summaries.clone(),
        linkage_root,
        proof_binding_root,
    })
}

fn build_rv64im_nightstream_statement_from_published_statement(
    verifier_context_digest: [u8; 32],
    published_statement: &Rv64imPublishedStatement,
    chunk_summaries: &[crate::finalize::FixedShapeChunkSummary],
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    Ok(NightstreamStatement {
        public_io_digest: published_statement.expected_digest(),
        verifier_context_digest,
        fold_schedule: published_statement.fold_schedule(),
        semantic_step_count: published_statement.step_count(),
        chunk_summaries: chunk_summaries.to_vec(),
        linkage_root,
        proof_binding_root,
    })
}

pub fn build_rv64im_nightstream_statement_from_main_proof(
    verifier_context_digest: [u8; 32],
    main_proof: &Rv64imMainProof,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    build_rv64im_nightstream_statement_from_published_statement(
        verifier_context_digest,
        main_proof.published_statement(),
        main_proof.chunk_summaries(),
        linkage_root,
        proof_binding_root,
    )
}

fn build_rv64im_nightstream_statement_from_relation(
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    relation: &crate::rv64im::audit::Rv64imDeciderRelation,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
) -> Result<NightstreamStatement, SimpleKernelError> {
    crate::rv64im::audit::validate_rv64im_decider_relation_surface(relation)?;
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

fn build_rv64im_side_proof_from_bundle(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProof, SimpleKernelError> {
    let opening =
        side_eval_claim_relation::build_rv64im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
            &accepted_artifact.statement,
            side_bundle,
            accepted_artifact,
        )?;
    let claim_witnesses = side_eval_claim_relation::rebuild_phase0_claim_witnesses_from_artifact(&opening)?;
    let opening_statement = build_rv64im_side_opening_relation_statement(&accepted_artifact.statement, side_bundle)?;
    let opening_witness = build_rv64im_side_opening_relation_witness_from_accepted_artifact(accepted_artifact);
    let opening_keys = setup_rv64im_side_opening_spartan_cached(&opening_statement, &opening_witness)?;
    let opening_final =
        prove_rv64im_side_opening_spartan(&opening_keys.as_ref().0, &opening_statement, &opening_witness)?;
    let public = build_rv64im_side_opening_public(side_bundle, &opening)?;
    let side_statement = build_rv64im_side_binding_statement(nightstream_statement, &public)?;
    let keys = setup_rv64im_side_binding_cached(&side_statement, &public)?;
    let binding = prove_rv64im_side_binding(&keys.as_ref().0, &side_statement, &public, &claim_witnesses)?;
    Ok(Rv64imSideProof {
        opening_public: public,
        opening_statement,
        opening: opening_final,
        binding,
    })
}

pub fn build_rv64im_side_proof(
    nightstream_statement: &NightstreamStatement,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imSideProof, SimpleKernelError> {
    let side_bundle = bind_rv64im_side_proof_bundle_to_statement_core(
        &build_rv64im_side_proof_bundle_from_accepted_artifact(accepted_artifact)?,
        nightstream_statement.core_digest(),
    )?;
    build_rv64im_side_proof_from_bundle(nightstream_statement, &side_bundle, accepted_artifact)
}

pub fn verify_rv64im_side_proof(
    opening_vk: &Rv64imSideOpeningSpartanVerifierKey,
    vk: &Rv64imSideBindingVerifierKey,
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv64imProofStatement,
    side_proof: &Rv64imSideProof,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_side_opening_public(nightstream_statement, &side_proof.opening_public)?;
    verify_rv64im_side_opening_statement_against_runtime_surfaces(
        nightstream_statement,
        public_statement,
        &side_proof.opening_public,
        &side_proof.opening_statement,
    )?;
    verify_rv64im_side_opening_spartan(opening_vk, &side_proof.opening_statement, &side_proof.opening)?;
    let side_statement = side_proof.binding_statement(nightstream_statement)?;
    verify_rv64im_side_binding(vk, &side_statement, &side_proof.binding)
}

fn build_rv64im_nightstream_linkage_claims(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<Rv64imLinkageClaims, SimpleKernelError> {
    let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
    Ok(build_rv64im_nightstream_linkage_claims_from_parts(
        verified_kernel
            .chunk_handoffs
            .iter()
            .map(|handoff| rv64im_public_chunk_digest(&handoff.public_chunk))
            .collect(),
    ))
}

pub fn rv64im_nightstream_linkage_root(
    public_statement_anchor_digest: [u8; 32],
    linkage_claims: &Rv64imLinkageClaims,
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/linkage_root");
    tr.append_message(b"neo.fold.next/nightstream/rv64im/linkage_root/version", b"v1");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/linkage_root/public_statement_anchor_digest",
        &public_statement_anchor_digest,
    );
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/linkage_root/linkage_claims_digest",
        &linkage_claims.digest(),
    );
    tr.digest32()
}

fn validate_rv64im_nightstream_linkage_claims_against_statement(
    statement: &NightstreamStatement,
    linkage_claims: &Rv64imLinkageClaims,
) -> Result<(), SimpleKernelError> {
    if linkage_claims.digest() != linkage_claims.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage claims digest mismatch".into(),
        ));
    }
    if linkage_claims.public_chunk_digests.len() != statement.chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream linkage claims public-chunk count does not match the carried statement".into(),
        ));
    }
    for (expected, carried) in statement
        .chunk_summaries
        .iter()
        .map(|summary| summary.public_chunk_digest)
        .zip(linkage_claims.public_chunk_digests.iter().copied())
    {
        if expected != carried {
            return Err(SimpleKernelError::Bridge(
                "RV64IM Nightstream linkage claims public-chunk digests do not match the carried statement".into(),
            ));
        }
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
    validate_rv64im_nightstream_linkage_claims_against_statement(statement, &proof.linkage_claims)?;
    proof.main_proof.validate_final_surface()?;
    let linkage_root = rv64im_nightstream_linkage_root(proof.main_proof.linkage_anchor_digest(), &proof.linkage_claims);
    let mut expected_statement = build_rv64im_nightstream_statement_from_main_proof(
        statement.verifier_context_digest,
        &proof.main_proof,
        linkage_root,
        [0; 32],
    )?;
    let proof_binding_inputs = NightstreamProofBindingInputs {
        main_proof_digest: proof.main_proof.binding_digest(),
        side_proof_digest: proof.side_proof.expected_digest(),
        linkage_binding_digest: proof.linkage_claims.digest(),
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
    side_opening_vk: &Rv64imSideOpeningSpartanVerifierKey,
    side_binding_vk: &Rv64imSideBindingVerifierKey,
    public_statement: &Rv64imProofStatement,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_nightstream_with_perf(
        statement,
        proof,
        trusted_root_params_id,
        side_opening_vk,
        side_binding_vk,
        public_statement,
    )
    .map(|_| ())
}
