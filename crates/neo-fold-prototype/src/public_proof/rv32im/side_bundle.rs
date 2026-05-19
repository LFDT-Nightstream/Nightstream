//! Builds and binds the RV32IM side-proof bundle carried by the public proof boundary.

use crate::public_proof::NightstreamStatement;
use crate::rv32im::kernel::{
    build_rv32im_kernel_export_proof_from_accepted_artifact, Rv32imAcceptedProofArtifact, Rv32imKernelExportProof,
    SimpleKernelError,
};

use super::authoritative_side::{build_rv32im_side_opening_public, Rv32imSideOpeningPublic};
use super::side_bridges::{
    build_rv32im_kernel_claim_bridge_from_accepted_artifact,
    build_rv32im_kernel_claim_proof_bridge_from_accepted_artifact,
    build_rv32im_kernel_opening_bridge_from_accepted_artifact, build_rv32im_main_lane_proof_bridge_from_export_proof,
    build_rv32im_stage_claim_proof_bridge_from_accepted_artifact,
    build_rv32im_verified_side_claims_from_accepted_artifact_fast, Rv32imSideProofBundle,
};
use super::side_eval_claim_relation;

pub(super) fn build_rv32im_side_proof_bundle_from_accepted_artifact_and_kernel_export(
    artifact: &Rv32imAcceptedProofArtifact,
    kernel_export: &Rv32imKernelExportProof,
) -> Result<Rv32imSideProofBundle, SimpleKernelError> {
    let (transcript, stage1, stage2, stage3) = build_rv32im_verified_side_claims_from_accepted_artifact_fast(artifact)?;
    let mut bundle = Rv32imSideProofBundle {
        statement_core_digest: [0; 32],
        transcript,
        stage1,
        stage2,
        stage3,
        stage_claim_proof_bridge: build_rv32im_stage_claim_proof_bridge_from_accepted_artifact(artifact),
        kernel_opening_bridge: build_rv32im_kernel_opening_bridge_from_accepted_artifact(artifact),
        kernel_claim_bridge: build_rv32im_kernel_claim_bridge_from_accepted_artifact(artifact),
        kernel_claim_proof_bridge: build_rv32im_kernel_claim_proof_bridge_from_accepted_artifact(artifact),
        main_lane_bridge: build_rv32im_main_lane_proof_bridge_from_export_proof(kernel_export),
        digest: [0; 32],
    };
    bundle.digest = bundle.expected_digest();
    Ok(bundle)
}

pub fn build_rv32im_side_proof_bundle_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideProofBundle, SimpleKernelError> {
    let (_, kernel_export, _) = build_rv32im_kernel_export_proof_from_accepted_artifact(artifact)?;
    build_rv32im_side_proof_bundle_from_accepted_artifact_and_kernel_export(artifact, &kernel_export)
}

pub fn build_rv32im_bound_side_proof_bundle_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideProofBundle, SimpleKernelError> {
    bind_rv32im_side_proof_bundle_to_statement_core(
        &build_rv32im_side_proof_bundle_from_accepted_artifact(artifact)?,
        statement.core_digest(),
    )
}

pub fn build_rv32im_bound_side_opening_public_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideOpeningPublic, SimpleKernelError> {
    let side_bundle = bind_rv32im_side_proof_bundle_to_statement_core(
        &build_rv32im_side_proof_bundle_from_accepted_artifact(artifact)?,
        statement.core_digest(),
    )?;
    let opening =
        side_eval_claim_relation::build_rv32im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
            &artifact.statement,
            &side_bundle,
            artifact,
        )?;
    build_rv32im_side_opening_public(&side_bundle, &opening)
}

pub(crate) fn bind_rv32im_side_proof_bundle_to_statement_core(
    bundle: &Rv32imSideProofBundle,
    statement_core_digest: [u8; 32],
) -> Result<Rv32imSideProofBundle, SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof bundle digest mismatch".into(),
        ));
    }
    let mut bound = bundle.clone();
    bound.statement_core_digest = statement_core_digest;
    bound.digest = bound.expected_digest();
    Ok(bound)
}

pub fn verify_rv32im_side_proof_bundle_from_accepted_artifact(
    artifact: &Rv32imAcceptedProofArtifact,
    bundle: &Rv32imSideProofBundle,
) -> Result<(), SimpleKernelError> {
    if bundle.digest != bundle.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof bundle digest mismatch".into(),
        ));
    }
    let expected = build_rv32im_side_proof_bundle_from_accepted_artifact(artifact)?;
    if &expected != bundle {
        return Err(SimpleKernelError::Bridge(
            "RV32IM Nightstream side-proof bundle does not match the accepted artifact".into(),
        ));
    }
    Ok(())
}
