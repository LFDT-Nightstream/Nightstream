//! Builds and verifies the RV32IM side proof SNARK pair.

use crate::public_proof::NightstreamStatement;
use crate::rv32im::kernel::{Rv32imAcceptedProofArtifact, Rv32imProofStatement, SimpleKernelError};

use super::authoritative_side::{
    build_rv32im_side_binding_statement, build_rv32im_side_opening_public, validate_rv32im_side_opening_public,
};
use super::proof::Rv32imSideProof;
use super::side_bridges::Rv32imSideProofBundle;
use super::side_bundle::{
    bind_rv32im_side_proof_bundle_to_statement_core, build_rv32im_side_proof_bundle_from_accepted_artifact,
};
use super::side_eval_claim_relation;
use super::side_opening_relation::{
    build_rv32im_side_opening_relation_statement, build_rv32im_side_opening_relation_witness_from_accepted_artifact,
};
use super::side_opening_spartan::{
    prove_rv32im_side_opening_spartan, setup_rv32im_side_opening_spartan_cached, verify_rv32im_side_opening_spartan,
    Rv32imSideOpeningSpartanVerifierKey,
};
use super::side_relation_spartan::{
    prove_rv32im_side_binding, setup_rv32im_side_binding_cached, verify_rv32im_side_binding,
    Rv32imSideBindingVerifierKey,
};
use super::side_runtime_binding::verify_rv32im_side_opening_statement_against_runtime_surfaces;

pub(super) fn build_rv32im_side_proof_from_bundle(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv32imSideProofBundle,
    accepted_artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideProof, SimpleKernelError> {
    let opening =
        side_eval_claim_relation::build_rv32im_side_eval_claim_artifact_from_accepted_artifact_and_side_bundle(
            &accepted_artifact.statement,
            side_bundle,
            accepted_artifact,
        )?;
    let claim_witnesses = side_eval_claim_relation::rebuild_phase0_claim_witnesses_from_artifact(&opening)?;
    let opening_statement = build_rv32im_side_opening_relation_statement(&accepted_artifact.statement, side_bundle)?;
    let opening_witness = build_rv32im_side_opening_relation_witness_from_accepted_artifact(accepted_artifact);
    let opening_keys = setup_rv32im_side_opening_spartan_cached(&opening_statement, &opening_witness)?;
    let opening_final =
        prove_rv32im_side_opening_spartan(&opening_keys.as_ref().0, &opening_statement, &opening_witness)?;
    let public = build_rv32im_side_opening_public(side_bundle, &opening)?;
    let side_statement =
        build_rv32im_side_binding_statement(nightstream_statement, &accepted_artifact.statement, &public)?;
    let keys = setup_rv32im_side_binding_cached(&side_statement, &public)?;
    let binding = prove_rv32im_side_binding(&keys.as_ref().0, &side_statement, &public, &claim_witnesses)?;
    Ok(Rv32imSideProof::from_parts(
        public,
        opening_statement,
        opening_final,
        binding,
    ))
}

pub fn build_rv32im_side_proof(
    nightstream_statement: &NightstreamStatement,
    accepted_artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideProof, SimpleKernelError> {
    let side_bundle = bind_rv32im_side_proof_bundle_to_statement_core(
        &build_rv32im_side_proof_bundle_from_accepted_artifact(accepted_artifact)?,
        nightstream_statement.core_digest(),
    )?;
    build_rv32im_side_proof_from_bundle(nightstream_statement, &side_bundle, accepted_artifact)
}

pub fn verify_rv32im_side_proof(
    opening_vk: &Rv32imSideOpeningSpartanVerifierKey,
    vk: &Rv32imSideBindingVerifierKey,
    nightstream_statement: &NightstreamStatement,
    public_statement: &Rv32imProofStatement,
    side_proof: &Rv32imSideProof,
) -> Result<(), SimpleKernelError> {
    validate_rv32im_side_opening_public(nightstream_statement, side_proof.opening_public())?;
    verify_rv32im_side_opening_statement_against_runtime_surfaces(
        nightstream_statement,
        public_statement,
        side_proof.opening_public(),
        side_proof.opening_statement(),
    )?;
    verify_rv32im_side_opening_spartan(opening_vk, side_proof.opening_statement(), side_proof.opening())?;
    let side_statement = side_proof.binding_statement(nightstream_statement, public_statement)?;
    verify_rv32im_side_binding(vk, &side_statement, side_proof.binding())
}
