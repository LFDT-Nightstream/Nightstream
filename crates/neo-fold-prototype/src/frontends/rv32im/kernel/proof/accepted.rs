//! Owns the accepted-proof and audit-only artifact split for RV32IM public proofs.

use crate::rv32im::stage1::{build_stage1_proof_bundle, Stage1ProofBundle};
use crate::rv32im::stage2::{build_stage2_proof_bundle, Stage2ProofBundle};
use crate::rv32im::stage3::{build_stage3_proof_bundle, Stage3ProofBundle};
use neo_transcript::{Poseidon2Transcript, Transcript};

use super::proof_api::{Rv32imKernelClaimBundle, Rv32imMainLaneProofBundle, Rv32imProof, Rv32imProofStatement};
use super::proof_completeness::{
    build_step_composition_surface, canonical_kernel_soundness_accounting_surface, KernelSoundnessAccountingSurface,
    StepCompositionSurface,
};
use super::proof_witness::{
    Rv32imKernelClaimProofBundle, Rv32imKernelOpeningProofBundle, Rv32imProofWitnessBundle,
    Rv32imStageClaimProofBundle, Rv32imStagePackageProofBundle,
};
use super::root_lane_witness::{
    build_root_execution_row_chunk_routes, build_root_execution_semantic_row_values,
    build_root_execution_semantic_rows_from_values, build_root_execution_semantics_refinement_summary,
    build_root_row_local_ccs_acceptance_summary, root_execution_public_step_digests,
    root_execution_row_chunk_routes_digest, root_execution_semantic_rows_digest, RootExecutionBundle,
};
use super::simple::{
    build_prepared_step_binding_summary, PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar, SimpleKernelError,
};
use super::{RootLaneColumns, RootLaneCommitmentSummaryArtifact, TranscriptRecord};

#[derive(Clone, Debug)]
pub struct Rv32imAcceptedProofArtifact {
    pub claim: Rv32imKernelClaimBundle,
    pub statement: Rv32imProofStatement,
    pub stage_claims: Rv32imStageClaimProofBundle,
    pub stage_packages: Rv32imStagePackageProofBundle,
    pub kernel_opening: Rv32imKernelOpeningProofBundle,
    pub kernel_claims: Rv32imKernelClaimProofBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
    pub main_lane: Rv32imMainLaneProofBundle,
    pub transcript: TranscriptRecord,
    pub stage1: Stage1ProofBundle,
    pub stage2: Stage2ProofBundle,
    pub stage3: Stage3ProofBundle,
    pub root_execution: RootExecutionBundle,
    pub step_composition: StepCompositionSurface,
    pub soundness_accounting: KernelSoundnessAccountingSurface,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct Rv32imAuditBundle {
    pub witness: Rv32imProofWitnessBundle,
    pub digest: [u8; 32],
}

fn build_root_execution_bundle(
    kernel: &PublicSimpleKernelOutput,
    sidecar: &PublicSimpleKernelWitnessSidecar,
    main_lane: &Rv32imMainLaneProofBundle,
) -> Result<RootExecutionBundle, SimpleKernelError> {
    let semantic_row_values = build_root_execution_semantic_row_values(&sidecar.trace.execution_rows);
    let semantic_rows =
        build_root_execution_semantic_rows_from_values(&sidecar.trace.execution_rows, &semantic_row_values);
    let public_step_digests = root_execution_public_step_digests(&main_lane.packaged.statement);
    let row_chunk_routes = build_root_execution_row_chunk_routes(&main_lane.packaged.statement);
    let prepared_step_bindings = build_prepared_step_binding_summary(
        &sidecar.trace.execution_rows,
        &semantic_row_values,
        &kernel.root_lane_columns,
        true,
    )?;
    let row_local_ccs_acceptance =
        build_root_row_local_ccs_acceptance_summary(&prepared_step_bindings, &row_chunk_routes, &public_step_digests)?;
    let execution_semantics_refinement = build_root_execution_semantics_refinement_summary(
        &semantic_rows,
        &prepared_step_bindings,
        &row_local_ccs_acceptance,
        &public_step_digests,
    )?;
    let bundle = RootExecutionBundle {
        execution_rows: sidecar.trace.execution_rows.clone(),
        semantic_rows_digest: root_execution_semantic_rows_digest(&semantic_rows),
        semantic_rows,
        prepared_step_bindings,
        row_chunk_routes_digest: root_execution_row_chunk_routes_digest(&row_chunk_routes),
        row_chunk_routes,
        row_local_ccs_acceptance,
        execution_semantics_refinement,
        family_digest: kernel.root_lane_columns.family_digest,
        digest: [0; 32],
    };
    Ok(RootExecutionBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

impl Rv32imAcceptedProofArtifact {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/accepted_proof_artifact");
        tr.append_message(b"rv32im/accepted_proof_artifact/claim", &self.claim.digest);
        tr.append_message(b"rv32im/accepted_proof_artifact/statement", &self.statement.digest);
        tr.append_message(
            b"rv32im/accepted_proof_artifact/stage_claims",
            &self.stage_claims.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/stage_packages",
            &self.stage_packages.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/kernel_opening",
            &self.kernel_opening.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/kernel_claims",
            &self.kernel_claims.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/root_lane_columns",
            &self.root_lane_columns.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/root_lane_commitment",
            &self.root_lane_commitment.digest,
        );
        tr.append_message(b"rv32im/accepted_proof_artifact/main_lane", &self.main_lane.digest);
        tr.append_message(b"rv32im/accepted_proof_artifact/stage1", &self.stage1.digest);
        tr.append_message(b"rv32im/accepted_proof_artifact/stage2", &self.stage2.digest);
        tr.append_message(b"rv32im/accepted_proof_artifact/stage3", &self.stage3.digest);
        tr.append_message(
            b"rv32im/accepted_proof_artifact/root_execution",
            &self.root_execution.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/step_composition",
            &self.step_composition.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/soundness_accounting",
            &self.soundness_accounting.digest,
        );
        tr.append_message(
            b"rv32im/accepted_proof_artifact/transcript_digest",
            &self.transcript.expected_digest(),
        );
        tr.digest32()
    }
}

impl Rv32imAuditBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/audit_bundle");
        tr.append_message(b"rv32im/audit_bundle/witness", &self.witness.digest);
        tr.digest32()
    }
}

pub(crate) fn accepted_proof_artifact_from_public_proof(
    proof: &Rv32imProof,
) -> Result<Rv32imAcceptedProofArtifact, SimpleKernelError> {
    let kernel = PublicSimpleKernelOutput {
        trace: proof.kernel.trace.clone(),
        stages: proof.kernel.stages.clone(),
        stage_claims: proof.kernel.stage_claims.claims.clone(),
        stage_packages: proof.kernel.stage_packages.packages.clone(),
        kernel_opening: proof.kernel.kernel_opening.opening.clone(),
        kernel_claims: proof.kernel.kernel_claims.claims.clone(),
        root_lane_columns: proof.kernel.root_lane_columns.clone(),
        root_lane_commitment: proof.kernel.root_lane_commitment.clone(),
    };
    let sidecar = PublicSimpleKernelWitnessSidecar {
        trace: proof.witness.trace.trace.clone(),
        stages: proof.witness.stages.stages.clone(),
    };
    accepted_proof_artifact_from_prover_materials(
        &proof.claim,
        &proof.statement,
        &kernel,
        &sidecar,
        &proof.kernel.main_lane,
        &proof.kernel.stage_claims,
        &proof.kernel.stage_packages,
        &proof.kernel.kernel_opening,
        &proof.kernel.kernel_claims,
        &proof.kernel.root_lane_columns,
        &proof.kernel.root_lane_commitment,
    )
}

pub(crate) fn accepted_proof_artifact_from_prover_materials(
    claim: &Rv32imKernelClaimBundle,
    statement: &Rv32imProofStatement,
    kernel: &PublicSimpleKernelOutput,
    sidecar: &PublicSimpleKernelWitnessSidecar,
    main_lane: &Rv32imMainLaneProofBundle,
    stage_claims: &Rv32imStageClaimProofBundle,
    stage_packages: &Rv32imStagePackageProofBundle,
    kernel_opening: &Rv32imKernelOpeningProofBundle,
    kernel_claims: &Rv32imKernelClaimProofBundle,
    root_lane_columns: &RootLaneColumns,
    root_lane_commitment: &RootLaneCommitmentSummaryArtifact,
) -> Result<Rv32imAcceptedProofArtifact, SimpleKernelError> {
    let stage1 = build_stage1_proof_bundle(
        &sidecar.trace.execution_rows,
        &sidecar.stages.stage1,
        &stage_claims.claims.stage1,
        &stage_packages.packages.stage1,
    );
    let stage2 = build_stage2_proof_bundle(
        &sidecar.stages.stage2,
        &stage_claims.claims.stage2,
        &stage_packages.packages.stage2,
    );
    let root_execution = build_root_execution_bundle(kernel, sidecar, main_lane)?;
    let stage3 = build_stage3_proof_bundle(
        &sidecar.stages.stage3,
        &stage_claims.claims.stage3,
        &root_execution,
        stage2.temporal.digest,
        statement.initial_pc,
        statement.final_pc,
        &stage_packages.packages.stage3,
    );
    let step_composition = build_step_composition_surface(
        &stage1,
        &stage2,
        &stage3,
        &root_execution,
        statement.initial_pc,
        statement.final_pc,
    );
    let soundness_accounting = canonical_kernel_soundness_accounting_surface();
    let artifact = Rv32imAcceptedProofArtifact {
        claim: claim.clone(),
        statement: statement.clone(),
        stage_claims: stage_claims.clone(),
        stage_packages: stage_packages.clone(),
        kernel_opening: kernel_opening.clone(),
        kernel_claims: kernel_claims.clone(),
        root_lane_columns: root_lane_columns.clone(),
        root_lane_commitment: root_lane_commitment.clone(),
        main_lane: main_lane.clone(),
        transcript: sidecar.stages.transcript.clone(),
        stage1,
        stage2,
        stage3,
        root_execution,
        step_composition,
        soundness_accounting,
        digest: [0; 32],
    };
    Ok(Rv32imAcceptedProofArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    })
}

pub(crate) fn audit_bundle_from_public_proof(proof: &Rv32imProof) -> Rv32imAuditBundle {
    let bundle = Rv32imAuditBundle {
        witness: proof.witness.clone(),
        digest: [0; 32],
    };
    Rv32imAuditBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}
