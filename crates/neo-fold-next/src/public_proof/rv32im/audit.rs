//! Exposes RV32IM Nightstream audit helpers without making the root module noisy.

use crate::public_proof::NightstreamStatement;
use crate::rv32im::kernel::{Rv32imAcceptedProofArtifact, SimpleKernelError};
use crate::rv32im::FamilyEvalClaimWitness;

use super::side_bundle::bind_rv32im_side_proof_bundle_to_statement_core;

pub use super::authoritative_side::{
    build_rv32im_side_opening_public, build_rv32im_side_surface_public,
    verify_rv32im_side_surface_public_against_bundle,
};
pub use super::opening_artifact::{
    build_rv32im_opening_artifact_from_accepted_artifact, verify_rv32im_opening_artifact_from_accepted_artifact,
    verify_rv32im_opening_artifact_from_side_proof_bundle, Rv32imOpeningArtifact,
};
pub use super::proof::rv32im_main_nightstream_proof_digest;
pub use super::side_bridges::{
    Rv32imKernelClaimBridge, Rv32imKernelClaimProofBridge, Rv32imKernelOpeningBridge, Rv32imMainLaneProofBridge,
    Rv32imPreparedStepBindingSummaryBridge, Rv32imSideProofBundle, Rv32imStageClaimProofBridge,
};
pub use super::side_bundle::{
    build_rv32im_side_proof_bundle_from_accepted_artifact, verify_rv32im_side_proof_bundle_from_accepted_artifact,
};
pub use super::side_claim_relation::{
    build_rv32im_side_claim_relation_from_accepted_artifact, build_rv32im_side_claim_relation_statement,
    build_rv32im_side_claim_relation_witness_from_accepted_artifact, verify_rv32im_side_claim_relation,
    Rv32imSideClaimRelationStatement, Rv32imSideClaimRelationWitness,
};
pub use super::side_eval_claim_relation::{
    build_rv32im_phase0_opened_object_bundle_from_claim_witnesses, build_rv32im_side_eval_claim_artifact,
    build_rv32im_side_eval_claim_artifact_from_accepted_artifact,
    build_rv32im_side_eval_claim_relation_from_accepted_artifact, build_rv32im_side_eval_claim_relation_statement,
    build_rv32im_side_eval_claim_relation_statement_from_artifact,
    build_rv32im_side_eval_claim_relation_witness_from_accepted_artifact, verify_rv32im_side_eval_claim_artifact,
    verify_rv32im_side_eval_claim_relation, Rv32imPhase0OpenedObjectBundle, Rv32imPhase0OpenedObjectSummary,
    Rv32imPhase0OpeningTarget, Rv32imPhase0OpeningTargetBundle, Rv32imSideEvalClaimArtifact,
    Rv32imSideEvalClaimRelationStatement, Rv32imSideEvalClaimRelationWitness,
};
pub use super::side_opening_relation::{
    build_rv32im_side_opening_relation_from_accepted_artifact, build_rv32im_side_opening_relation_statement,
    build_rv32im_side_opening_relation_witness_from_accepted_artifact, verify_rv32im_side_opening_relation,
    Rv32imSideOpeningRelationStatement, Rv32imSideOpeningRelationWitness,
};
pub use super::side_opening_spartan::{
    debug_check_rv32im_side_opening_spartan_circuit, debug_compare_rv32im_side_opening_spartan_setup_shape,
    debug_compare_rv32im_side_opening_spartan_statement_owned_shape,
    debug_compare_rv32im_side_opening_spartan_without_packaged_final_main_claims_shape,
    debug_compare_rv32im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape,
    debug_compare_rv32im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape,
    debug_compare_rv32im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape,
    debug_measure_rv32im_side_opening_spartan_circuit_shape, debug_native_stage1_packaged_statement_digest,
    debug_round_trip_rv32im_stage1_packaged_opening_digest_with_reduced_setup,
    debug_setup_rv32im_side_opening_spartan_without_packaged_final_main_claims,
    debug_setup_rv32im_side_opening_spartan_without_stage1_packaged_final_main_claims,
    prove_rv32im_side_opening_spartan, setup_rv32im_side_opening_spartan, setup_rv32im_side_opening_spartan_cached,
    verify_rv32im_side_opening_spartan, Rv32imSideOpeningSpartanCircuitShape, Rv32imSideOpeningSpartanProof,
    Rv32imSideOpeningSpartanProverKey, Rv32imSideOpeningSpartanVerifierKey,
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
    debug_check_rv32im_side_binding_circuit, measure_rv32im_side_binding_circuit_constraints,
    prove_rv32im_side_binding, setup_rv32im_side_binding, setup_rv32im_side_binding_cached, verify_rv32im_side_binding,
    Rv32imSideBindingProverKey, Rv32imSideBindingVerifierKey,
};
pub use super::statement::{
    build_rv32im_nightstream_statement_from_final, build_rv32im_nightstream_statement_from_published_statement,
};
pub use super::surfaces::{
    build_rv32im_kernel_opening_claim_from_side_proof_bundle, build_rv32im_stage_claim_bundle_from_side_proof_bundle,
};

pub fn build_rv32im_bound_side_proof_bundle_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Rv32imSideProofBundle, SimpleKernelError> {
    bind_rv32im_side_proof_bundle_to_statement_core(
        &build_rv32im_side_proof_bundle_from_accepted_artifact(artifact)?,
        statement.core_digest(),
    )
}

pub fn build_rv32im_bound_phase0_claim_witnesses_from_accepted_artifact(
    statement: &NightstreamStatement,
    artifact: &Rv32imAcceptedProofArtifact,
) -> Result<Vec<FamilyEvalClaimWitness>, SimpleKernelError> {
    let side_bundle = bind_rv32im_side_proof_bundle_to_statement_core(
        &build_rv32im_side_proof_bundle_from_accepted_artifact(artifact)?,
        statement.core_digest(),
    )?;
    Ok(
        super::side_eval_claim_relation::build_rv32im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(
            &side_bundle,
            artifact,
        )?
        .claim_witnesses,
    )
}
