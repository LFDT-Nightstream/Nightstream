//! Owns RV32IM kernel opening claims, reductions, and convergence proofs.

use super::{canonical_openings, proof_accepted, proof_staged_verify, simple, simple_openings, stage_artifacts};

#[path = "accumulate.rs"]
mod opening_accumulate;
#[path = "batch.rs"]
mod opening_batch;
#[path = "claim_reduction.rs"]
mod opening_claim_reduction;
#[path = "claim_reduction_error.rs"]
mod opening_claim_reduction_error;
#[path = "eval_claim_witness.rs"]
mod opening_eval_claim_witness;
#[path = "eval_claims.rs"]
mod opening_eval_claims;
#[path = "final_opening.rs"]
mod opening_final;
#[path = "manifest.rs"]
mod opening_manifest;
#[path = "payload_semantics.rs"]
mod opening_payload_semantics;
#[path = "phase0_binding_surface.rs"]
mod opening_phase0_binding_surface;
#[path = "point_derivation.rs"]
mod opening_point_derivation;
#[path = "verify.rs"]
mod opening_verify;

pub use super::simple_openings::{
    KernelBindingOpeningClaim, KernelBindingOpeningPoints, KernelBindingPackagedOpeningProof,
    KernelPreparedStepOpeningClaim, KernelPreparedStepOpeningPoints, KernelPreparedStepPackagedOpeningProof,
    OpeningPointLabel, SimpleKernelOpeningBundle, SimpleKernelOpeningClaim, SimpleKernelStagePackageBundle,
    Stage1OpeningPoints, Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2OpeningPoints,
    Stage2PackagedOpeningProof, Stage2SelectedOpeningClaim, Stage3OpeningPoints, Stage3PackagedOpeningProof,
    Stage3SelectedOpeningClaim,
};
pub(crate) use super::stage_artifacts::{
    build_claim_packaged_public_step, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, build_public_kernel_opening_claim_from_compact_surfaces,
    RV32IM_SELECTED_OPENING_LAYOUT_V1,
};
pub use super::stage_artifacts::{
    SimpleKernelStageClaimBundle, Stage1ArtifactSurface, Stage1ClaimSurface, Stage2ArtifactSurface, Stage2ClaimSurface,
    Stage3ArtifactSurface, Stage3ClaimSurface, StageDigestCommitment, TranscriptArtifactSurface,
    TranscriptClaimSurface,
};
pub use opening_accumulate::{
    build_phase2_collapse_result, verify_phase2_collapse_result, Phase2CollapseError, Phase2CollapseRecord,
    Phase2CollapseResult, ReducedEvalClaim,
};
pub use opening_batch::{build_rv32im_opening_bundle_from_accepted_artifact, Rv32imOpeningBundle};
pub use opening_claim_reduction::{
    build_claim_reduction_buckets, build_claim_reduction_results_from_witnesses, domain_for_schema,
    phase1_claim_digest, phase1_unified_claim_digest, verify_claim_reduction_result_with_binding_surface,
    verify_claim_reduction_results_with_binding_surface, ClaimReductionBucket, ClaimReductionProof,
    ClaimReductionResult, QuadraticRoundPoly,
};
pub use opening_claim_reduction_error::ClaimReductionError;
pub(crate) use opening_eval_claim_witness::{
    build_commitment_vector, build_rv32im_eval_claim_bundle_from_claim_witnesses_trusted_local,
    build_rv32im_eval_claim_witnesses_from_accepted_artifact_with_perf, phase0_binding_digest,
};
pub use opening_eval_claim_witness::{
    build_rv32im_eval_claim_bundle_from_accepted_artifact, build_rv32im_eval_claim_bundle_from_claim_witnesses,
    build_rv32im_eval_claim_witnesses_from_accepted_artifact, build_stage1_claim_witnesses,
    build_stage2_claim_witnesses, build_stage3_claim_witness, verify_rv32im_eval_claim_bundle_from_accepted_artifact,
    FamilyEvalClaimWitness, OpenedAjtaiObjectWitness, PackedColumnOracleRef, RealAjtaiCommitmentVector,
};
pub use opening_eval_claims::{
    phase0_family_order, CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId, FamilyEvalPayload,
    FamilyEvalSchemaId, OpenedAjtaiObjectId, OpeningClaimAccumulator, PackedColumnEval, Rv32imEvalClaimBundle,
};
pub(crate) use opening_final::{
    build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local,
    build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf,
    rebuild_opened_object_witness_from_projection,
};
pub use opening_final::{
    build_rv32im_opening_convergence_artifact_from_proof, build_rv32im_opening_convergence_artifact_from_witnesses,
    build_rv32im_opening_convergence_proof_from_witnesses, verify_rv32im_opening_convergence_artifact,
    verify_rv32im_opening_convergence_artifact_from_proof, verify_rv32im_opening_convergence_proof, AjtaiOpeningProof,
    FinalOpeningError, FinalOpeningTarget, OpenedAjtaiCommitmentPublic, ProjectedFinalOpeningTarget,
    RealAjtaiCommitmentVectorPublic, Rv32imOpeningConvergenceArtifact, Rv32imOpeningConvergenceProof,
};
pub use opening_manifest::{
    opening_claims_from_carriers, stage1_opening_witness_carriers, stage1_opening_witness_carriers_from_claim_surface,
    stage2_opening_witness_carriers, stage2_opening_witness_carriers_from_claim_surface,
    stage3_opening_witness_carriers, stage3_opening_witness_carriers_from_claim_surface, Rv32imOpeningWitnessCarrier,
};
pub use opening_payload_semantics::{
    encode_packed_column_evals_k, encode_words_to_field_evals_k, phase0_full_width_for_schema,
    phase0_word_count_for_schema, reconstruct_words_from_field_evals, unpack_column_evals_k,
};
pub use opening_phase0_binding_surface::{
    build_rv32im_phase0_binding_surface_from_accepted_artifact, Rv32imPhase0BindingSurface, Rv32imPhase0BindingTarget,
};
pub use opening_point_derivation::derive_phase0_point;
pub use opening_verify::verify_rv32im_opening_bundle_from_accepted_artifact;
