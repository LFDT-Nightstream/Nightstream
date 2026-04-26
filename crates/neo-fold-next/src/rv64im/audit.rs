//! Owns audit-only RV64IM escape hatches grouped by responsibility.
//!
//! `chunk_step` owns one-step chunk relation construction helpers.
//! `decider` owns the main-recursion SNARK audit surface.
//! `main_recursion` owns native F', NIFS, and recursive-step Spartan audit helpers.

use super::final_relation::{Rv64imFinalBuildProof, Rv64imFinalStatement, Rv64imFoldedProof, Rv64imFoldedStatement};
use super::kernel::{
    Rv64imAcceptedProofArtifact, Rv64imAuditBundle, Rv64imProof, Rv64imProofInput, Rv64imProofWitnessBundle,
    Rv64imPublicProofVerifyPerf, SimpleKernelError,
};

pub mod chunk_step;
pub mod decider;
pub mod main_recursion;

pub use chunk_step::*;
pub use decider::*;
pub use main_recursion::*;

pub fn audit_rv64im_public_proof(proof: &Rv64imProof) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof(proof)
}

pub fn audit_rv64im_public_proof_with_perf(
    proof: &Rv64imProof,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof_with_perf(proof)
}

pub fn audit_rv64im_accepted_proof(artifact: &Rv64imAcceptedProofArtifact) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv64im_accepted_proof(artifact)
}

pub fn audit_rv64im_accepted_proof_with_perf(
    artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv64im_accepted_proof_with_perf(artifact)
}

pub fn audit_rv64im_public_proof_against_input(
    input: &Rv64imProofInput,
    proof: &Rv64imProof,
) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof_against_input(input, proof)
}

pub fn audit_rv64im_public_proof_against_input_with_perf(
    input: &Rv64imProofInput,
    proof: &Rv64imProof,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof_against_input_with_perf(input, proof)
}

pub fn audit_rv64im_accepted_proof_against_input(
    input: &Rv64imProofInput,
    artifact: &Rv64imAcceptedProofArtifact,
    audit: &Rv64imAuditBundle,
) -> Result<(), SimpleKernelError> {
    super::kernel::audit_rv64im_accepted_proof_against_input(input, artifact, audit)
}

pub fn audit_rv64im_accepted_proof_against_input_with_perf(
    input: &Rv64imProofInput,
    artifact: &Rv64imAcceptedProofArtifact,
    audit: &Rv64imAuditBundle,
) -> Result<Rv64imPublicProofVerifyPerf, SimpleKernelError> {
    super::kernel::audit_rv64im_accepted_proof_against_input_with_perf(input, artifact, audit)
}

pub fn audit_rv64im_public_proof_with_witness(
    proof: &Rv64imProof,
) -> Result<Rv64imProofWitnessBundle, SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof_with_witness(proof)
}

pub fn audit_rv64im_public_proof_with_witness_and_perf(
    proof: &Rv64imProof,
) -> Result<(Rv64imProofWitnessBundle, Rv64imPublicProofVerifyPerf), SimpleKernelError> {
    super::kernel::audit_rv64im_public_proof_with_witness_and_perf(proof)
}

pub fn audit_check_rv64im_folded_statement_replay(
    folded: &Rv64imFoldedStatement,
    proof: &Rv64imFoldedProof,
) -> Result<(), SimpleKernelError> {
    super::final_relation::audit_check_rv64im_folded_statement_replay(folded, proof)
}

pub fn audit_check_rv64im_final_statement_replay(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    super::final_relation::audit_check_rv64im_final_statement_replay(statement, proof)
}
