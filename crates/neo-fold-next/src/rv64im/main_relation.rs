//! Owns the RV64IM main relation boundary between the final folded statement and generic decider backends.
//!
//! This module owns the public/private theorem surface for the main RV64IM
//! relation. Generic decider backends may compile this relation, but they do
//! not own its meaning.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::decider::spartan2::{build_spartan2_decider_relation, Spartan2DeciderRelation};
use crate::finalize::digest32_as_fields;
use crate::rv64im::final_relation::{
    final_proof_component_digests, recursive_seed, validate_rv64im_final_statement_surface,
    validate_rv64im_final_statement_surface_with_component_digests, verify_rv64im_final_statement_with_output,
    Rv64imFinalProof, Rv64imFinalProofComponentDigests, Rv64imFinalStatement,
};
use crate::rv64im::kernel::{
    build_rv64im_accepted_proof_artifact, Rv64imKernelExportRelationResult, Rv64imProof, SimpleKernelError,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv64imMainRelationStatement {
    pub final_statement: Rv64imFinalStatement,
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRelationWitness {
    pub final_proof: Rv64imFinalProof,
}

#[derive(Clone, Debug)]
pub struct Rv64imMainRelationArtifact {
    pub statement: Rv64imMainRelationStatement,
    pub witness: Rv64imMainRelationWitness,
    pub digest: [u8; 32],
}

impl Rv64imMainRelationStatement {
    pub fn digest(&self) -> [u8; 32] {
        self.final_statement.digest
    }
}

impl Rv64imMainRelationWitness {
    pub fn digest(&self) -> [u8; 32] {
        self.final_proof.proof_digest
    }
}

impl Rv64imMainRelationArtifact {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/main_relation_artifact");
        tr.append_message(
            b"neo.fold.next/rv64im/main_relation_artifact/statement_digest",
            &self.statement.digest(),
        );
        tr.append_message(
            b"neo.fold.next/rv64im/main_relation_artifact/witness_digest",
            &self.witness.digest(),
        );
        tr.digest32()
    }
}

pub fn build_rv64im_main_relation(proof: &Rv64imProof) -> Result<Rv64imMainRelationArtifact, SimpleKernelError> {
    let artifact = build_rv64im_accepted_proof_artifact(proof)?;
    let (statement, final_proof) = crate::rv64im::final_relation::prove_rv64im_final_statement_from_accepted(&artifact)?;
    build_rv64im_main_relation_from_final(&statement, &final_proof)
}

pub fn validate_rv64im_main_relation_surface(
    statement: &Rv64imMainRelationStatement,
    witness: &Rv64imMainRelationWitness,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_final_statement_surface(&statement.final_statement, &witness.final_proof)
}

pub fn verify_rv64im_main_relation(
    statement: &Rv64imMainRelationStatement,
    witness: &Rv64imMainRelationWitness,
) -> Result<(), SimpleKernelError> {
    verify_rv64im_main_relation_with_output(statement, witness)?;
    Ok(())
}

pub fn build_rv64im_main_relation_from_final(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imMainRelationArtifact, SimpleKernelError> {
    let verified_kernel = verify_rv64im_final_statement_with_output(statement, proof)?;
    let component_digests = final_proof_component_digests(proof);
    build_rv64im_main_relation_from_verified_final_with_component_digests(
        statement,
        proof,
        &verified_kernel,
        &component_digests,
    )
}

pub fn build_rv64im_main_relation_backend_relation(
    statement: &Rv64imMainRelationStatement,
    witness: &Rv64imMainRelationWitness,
) -> Result<Spartan2DeciderRelation, SimpleKernelError> {
    let verified_kernel = verify_rv64im_main_relation_with_output(statement, witness)?;
    let component_digests = final_proof_component_digests(&witness.final_proof);
    let artifact = build_rv64im_main_relation_from_verified_final_with_component_digests(
        &statement.final_statement,
        &witness.final_proof,
        &verified_kernel,
        &component_digests,
    )?;
    build_rv64im_main_relation_backend_relation_from_verified_artifact_with_component_digests(
        &artifact,
        &verified_kernel,
        &component_digests,
    )
}

pub fn build_rv64im_main_relation_backend_relation_from_artifact(
    artifact: &Rv64imMainRelationArtifact,
) -> Result<Spartan2DeciderRelation, SimpleKernelError> {
    let verified_kernel = verify_rv64im_main_relation_with_output(&artifact.statement, &artifact.witness)?;
    let component_digests = final_proof_component_digests(&artifact.witness.final_proof);
    build_rv64im_main_relation_backend_relation_from_verified_artifact_with_component_digests(
        artifact,
        &verified_kernel,
        &component_digests,
    )
}

pub(crate) fn verify_rv64im_main_relation_with_output(
    statement: &Rv64imMainRelationStatement,
    witness: &Rv64imMainRelationWitness,
) -> Result<Rv64imKernelExportRelationResult, SimpleKernelError> {
    verify_rv64im_final_statement_with_output(&statement.final_statement, &witness.final_proof)
}

pub(crate) fn build_rv64im_main_relation_from_verified_final_with_component_digests(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
    verified_kernel: &Rv64imKernelExportRelationResult,
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imMainRelationArtifact, SimpleKernelError> {
    validate_rv64im_final_statement_surface_with_component_digests(statement, proof, component_digests)?;
    if statement.folded.chunk_count as usize != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match the verified kernel export handoffs".into(),
        ));
    }
    let mut artifact = Rv64imMainRelationArtifact {
        statement: Rv64imMainRelationStatement {
            final_statement: statement.clone(),
        },
        witness: Rv64imMainRelationWitness {
            final_proof: proof.clone(),
        },
        digest: [0; 32],
    };
    artifact.digest = artifact.expected_digest();
    Ok(artifact)
}

pub(crate) fn build_rv64im_main_relation_backend_relation_from_verified_artifact_with_component_digests(
    artifact: &Rv64imMainRelationArtifact,
    verified_kernel: &Rv64imKernelExportRelationResult,
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Spartan2DeciderRelation, SimpleKernelError> {
    let statement = &artifact.statement.final_statement;
    let proof = &artifact.witness.final_proof;
    if artifact.digest != artifact.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation artifact digest mismatch".into(),
        ));
    }
    if statement.folded.chunk_count as usize != verified_kernel.chunk_handoffs.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match verified kernel export handoffs".into(),
        ));
    }
    if statement.folded.chunk_count as usize != proof.chunk_summaries.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match final proof chunk summaries".into(),
        ));
    }
    if statement.folded.chunk_count as usize != component_digests.chunk_transition_digests.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM main relation chunk count does not match final proof replay witness".into(),
        ));
    }

    build_spartan2_decider_relation(
        statement.digest,
        statement.folded.digest,
        proof.proof_digest,
        digest32_as_fields(recursive_seed()),
        digest32_as_fields(statement.folded.final_accumulator.terminal_handle.0),
        statement.folded.fold_schedule,
        statement.folded.semantic_step_count,
        proof.chunk_summaries.clone(),
        vec![component_digests.kernel_export_proof_digest],
        component_digests.chunk_transition_digests.clone(),
    )
    .map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}
