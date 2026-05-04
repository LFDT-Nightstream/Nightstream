//! Owns audit-only RV32IM escape hatches grouped by responsibility.
//!
//! `chunk_step` owns one-step chunk relation construction helpers.
//! `decider` owns the main-recursion SNARK audit surface.
//! `main_recursion` owns native F', NIFS, and recursive-step Spartan audit helpers.

use super::final_relation::{Rv32imFinalBuildProof, Rv32imFinalStatement, Rv32imFoldedProof, Rv32imFoldedStatement};
use super::kernel::{
    Rv32imAcceptedProofArtifact, Rv32imAuditBundle, Rv32imProof, Rv32imProofInput, Rv32imProofWitnessBundle,
    Rv32imPublicProofVerifyPerf, SimpleKernelError,
};

pub mod chunk_step;
pub mod decider;
pub mod main_recursion;

pub use chunk_step::*;
pub use decider::*;
pub use main_recursion::*;

pub fn audit_rv32im_public_proof(proof: &Rv32imProof) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof(proof)
}

pub fn audit_rv32im_public_proof_with_perf(
    proof: &Rv32imProof,
) -> Result<Rv32imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof_with_perf(proof)
}

pub fn audit_rv32im_accepted_proof(artifact: &Rv32imAcceptedProofArtifact) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv32im_accepted_proof(artifact)
}

pub fn audit_rv32im_accepted_proof_with_perf(
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv32im_accepted_proof_with_perf(artifact)
}

pub fn audit_rv32im_public_proof_against_input(
    input: &Rv32imProofInput,
    proof: &Rv32imProof,
) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof_against_input(input, proof)
}

pub fn audit_rv32im_public_proof_against_input_with_perf(
    input: &Rv32imProofInput,
    proof: &Rv32imProof,
) -> Result<Rv32imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof_against_input_with_perf(input, proof)
}

pub fn audit_rv32im_accepted_proof_against_input(
    input: &Rv32imProofInput,
    artifact: &Rv32imAcceptedProofArtifact,
    audit: &Rv32imAuditBundle,
) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv32im_accepted_proof_against_input(input, artifact, audit)
}

pub fn audit_rv32im_accepted_proof_against_input_with_perf(
    input: &Rv32imProofInput,
    artifact: &Rv32imAcceptedProofArtifact,
    audit: &Rv32imAuditBundle,
) -> Result<Rv32imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv32im_accepted_proof_against_input_with_perf(input, artifact, audit)
}

pub fn audit_rv32im_public_proof_with_witness(
    proof: &Rv32imProof,
) -> Result<Rv32imProofWitnessBundle, SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof_with_witness(proof)
}

pub fn audit_rv32im_public_proof_with_witness_and_perf(
    proof: &Rv32imProof,
) -> Result<(Rv32imProofWitnessBundle, Rv32imPublicProofVerifyPerf), SimpleKernelError> {
    super::kernel::audit_rv32im_public_proof_with_witness_and_perf(proof)
}

pub fn audit_check_rv32im_folded_statement_replay(
    folded: &Rv32imFoldedStatement,
    proof: &Rv32imFoldedProof,
) -> Result<(), SimpleKernelError> {
    super::final_relation::audit_check_rv32im_folded_statement_replay(folded, proof)
}

pub fn audit_check_rv32im_final_statement_replay(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    super::final_relation::audit_check_rv32im_final_statement_replay(statement, proof)
}
