//! Owns the compact Nightstream opening artifact boundary above the Phase 0 and convergence carriers.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::side_eval_claim_relation::{
    build_rv64im_phase0_binding_surface_from_side_bundle,
    build_rv64im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle,
    build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle,
    build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle,
    verify_rv64im_side_eval_claim_artifact, Rv64imSideEvalClaimArtifact,
};
use super::Rv64imSideProofBundle;
use crate::rv64im::kernel::{
    build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local,
    verify_rv64im_opening_convergence_artifact, Rv64imAcceptedProofArtifact, Rv64imEvalClaimBundle,
    Rv64imOpeningConvergenceArtifact, Rv64imProofStatement, SimpleKernelError,
};
use crate::rv64im::FamilyEvalClaimWitness;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imOpeningArtifact {
    pub phase0_artifact: Rv64imSideEvalClaimArtifact,
    pub convergence_artifact: Rv64imOpeningConvergenceArtifact,
    pub digest: [u8; 32],
}

impl Rv64imOpeningArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/opening_artifact");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/opening_artifact/phase0_artifact_digest",
            &self.phase0_artifact.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/opening_artifact/convergence_artifact_digest",
            &self.convergence_artifact.digest,
        );
        tr.digest32()
    }
}

pub fn build_rv64im_opening_artifact_from_accepted_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imOpeningArtifact, SimpleKernelError> {
    let claim_witnesses =
        build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(side_bundle, artifact)?
            .claim_witnesses;
    let phase0_artifact = build_rv64im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
        public_statement,
        side_bundle,
        artifact,
    )?;
    let phase0_binding_surface = build_rv64im_phase0_binding_surface_from_side_bundle(side_bundle);
    let convergence_artifact =
        build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local(
            &phase0_binding_surface,
            &phase0_artifact.eval_claim_bundle,
            &claim_witnesses,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM Nightstream opening convergence artifact build failed: {err}"
            ))
        })?;
    build_rv64im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
        &phase0_artifact,
        &convergence_artifact,
    )
}

pub(super) fn build_rv64im_opening_artifact_from_claim_witnesses_and_side_bundle(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<Rv64imOpeningArtifact, SimpleKernelError> {
    let phase0_artifact = build_rv64im_side_eval_claim_artifact_from_claim_witnesses_and_side_bundle(
        public_statement,
        side_bundle,
        claim_witnesses,
    )?;
    let phase0_binding_surface = build_rv64im_phase0_binding_surface_from_side_bundle(side_bundle);
    let convergence_artifact =
        build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local(
            &phase0_binding_surface,
            &phase0_artifact.eval_claim_bundle,
            claim_witnesses,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM Nightstream opening convergence artifact build failed: {err}"
            ))
        })?;
    build_rv64im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
        &phase0_artifact,
        &convergence_artifact,
    )
}

pub fn verify_rv64im_opening_artifact_from_accepted_artifact(
    public_statement: &Rv64imProofStatement,
    side_bundle: &Rv64imSideProofBundle,
    artifact: &Rv64imAcceptedProofArtifact,
    proof: &Rv64imOpeningArtifact,
) -> Result<(), SimpleKernelError> {
    let expected = build_rv64im_opening_artifact_from_accepted_artifact(public_statement, side_bundle, artifact)?;
    if &expected != proof {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening artifact does not match the accepted artifact".into(),
        ));
    }
    Ok(())
}

pub(super) fn build_rv64im_opening_artifact_from_trusted_local_phase0_and_convergence_artifacts(
    phase0_artifact: &Rv64imSideEvalClaimArtifact,
    convergence_artifact: &Rv64imOpeningConvergenceArtifact,
) -> Result<Rv64imOpeningArtifact, SimpleKernelError> {
    if convergence_artifact.phase0_digest != phase0_artifact.eval_claim_bundle.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening convergence artifact Phase 0 digest does not match the carried Phase 0 claim bundle"
                .into(),
        ));
    }
    let artifact = Rv64imOpeningArtifact {
        phase0_artifact: phase0_artifact.clone(),
        convergence_artifact: convergence_artifact.clone(),
        digest: [0; 32],
    };
    Ok(Rv64imOpeningArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    })
}

fn validate_rv64im_opening_artifact_internal(proof: &Rv64imOpeningArtifact) -> Result<(), SimpleKernelError> {
    if proof.phase0_artifact.digest != proof.phase0_artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening artifact Phase 0 artifact digest mismatch".into(),
        ));
    }
    verify_rv64im_opening_convergence_artifact(&proof.convergence_artifact).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream opening convergence artifact verification failed: {err}"
        ))
    })?;
    let phase0_bundle = build_rv64im_eval_claim_bundle_from_convergence_artifact(&proof.convergence_artifact)?;
    if phase0_bundle.digest != proof.phase0_artifact.eval_claim_bundle.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening convergence artifact Phase 0 digest does not match the carried Phase 0 claim bundle"
                .into(),
        ));
    }
    let expected_digest = proof.expected_digest();
    if proof.digest != expected_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening artifact is not self-consistent".into(),
        ));
    }
    Ok(())
}

pub(super) fn verify_rv64im_opening_artifact_against_compact_surfaces(
    public_statement: &Rv64imProofStatement,
    bundle: &Rv64imSideProofBundle,
    proof: &Rv64imOpeningArtifact,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_opening_artifact_internal(proof)?;
    verify_rv64im_side_eval_claim_artifact(public_statement, bundle, &proof.phase0_artifact).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream opening artifact does not match the verified compact Phase 0 opening surface: {err}"
        ))
    })?;
    let phase0_bundle = build_rv64im_eval_claim_bundle_from_convergence_artifact(&proof.convergence_artifact)?;
    if proof.phase0_artifact.eval_claim_bundle.digest != phase0_bundle.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream compact opening artifact Phase 0 digest does not match the carried convergence artifact"
                .into(),
        ));
    }
    Ok(())
}

pub fn verify_rv64im_opening_artifact_from_side_proof_bundle(
    public_statement: &Rv64imProofStatement,
    bundle: &Rv64imSideProofBundle,
    proof: &Rv64imOpeningArtifact,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_opening_artifact_against_compact_surfaces(public_statement, bundle, proof)
}

fn build_rv64im_eval_claim_bundle_from_convergence_artifact(
    convergence_artifact: &Rv64imOpeningConvergenceArtifact,
) -> Result<Rv64imEvalClaimBundle, SimpleKernelError> {
    let bundle = Rv64imEvalClaimBundle::new(
        convergence_artifact
            .phase1_results
            .iter()
            .flat_map(|result| result.bucket.claims.clone())
            .collect(),
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM Nightstream opening convergence artifact cannot canonicalize the Phase 0 claim bundle: {err}"
        ))
    })?;
    if bundle.digest != convergence_artifact.phase0_digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM Nightstream opening convergence artifact Phase 0 digest does not match the carried Phase 1 buckets"
                .into(),
        ));
    }
    Ok(bundle)
}
