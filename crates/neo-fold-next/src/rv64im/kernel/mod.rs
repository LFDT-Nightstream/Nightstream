//! Owns the RV64IM parity-slice kernel artifacts and transcript logging.

mod canonical_openings;
mod perf_diagnostics;
mod simple;
mod transcript;

#[path = "parity/cases.rs"]
mod artifacts;
#[path = "main_lane/family.rs"]
mod lane_family;
#[path = "main_lane/artifact.rs"]
mod main_lane_artifact;
#[path = "main_lane/surface.rs"]
mod main_lane_surface;
#[path = "openings/accumulate.rs"]
mod opening_accumulate;
#[path = "openings/batch.rs"]
mod opening_batch;
#[path = "openings/claim_reduction.rs"]
mod opening_claim_reduction;
#[path = "openings/eval_claim_witness.rs"]
mod opening_eval_claim_witness;
#[path = "openings/eval_claims.rs"]
mod opening_eval_claims;
#[path = "openings/final_opening.rs"]
mod opening_final;
#[path = "openings/manifest.rs"]
mod opening_manifest;
#[path = "openings/payload_semantics.rs"]
mod opening_payload_semantics;
#[path = "openings/phase0_binding_surface.rs"]
mod opening_phase0_binding_surface;
#[path = "openings/point_derivation.rs"]
mod opening_point_derivation;
#[path = "openings/verify.rs"]
mod opening_verify;
#[path = "proof/accepted.rs"]
mod proof_accepted;
#[path = "proof/api.rs"]
mod proof_api;
#[path = "proof/bridge.rs"]
mod proof_bridge;
#[path = "proof/completeness.rs"]
mod proof_completeness;
#[path = "proof/export_relation.rs"]
mod proof_export_relation;
#[path = "proof/staged_verify.rs"]
mod proof_staged_verify;
#[path = "proof/verify.rs"]
mod proof_verify;
#[path = "proof/witness.rs"]
mod proof_witness;
#[path = "root_lane/columns.rs"]
mod root_lane_columns;
#[path = "root_lane/commitment.rs"]
mod root_lane_commitment;
#[path = "root_lane/witness.rs"]
mod root_lane_witness;
#[path = "openings/simple.rs"]
mod simple_openings;
#[path = "stages/stage1.rs"]
mod stage1_canonical;
#[path = "stages/stage2.rs"]
mod stage2_canonical;
#[path = "stages/stage3.rs"]
mod stage3_canonical;
#[path = "openings/stage_artifacts.rs"]
mod stage_artifacts;
#[path = "stages/package_perf.rs"]
mod stage_package_perf;

pub use artifacts::{
    aligned_memory_focus_manifest, build_aligned_memory_focus_parity_case, build_all_parity_cases,
    build_control_flow_beq_parity_case, build_control_flow_bge_parity_case, build_control_flow_bgeu_parity_case,
    build_control_flow_blt_parity_case, build_control_flow_bltu_parity_case, build_control_flow_bne_parity_case,
    build_control_flow_focus_parity_case, build_control_flow_jal_parity_case, build_control_flow_jalr_parity_case,
    build_multiply_high_parity_case, build_multiply_low_parity_case, build_narrow_memory_load_parity_case,
    build_narrow_memory_store_parity_case, build_native_alu_focus_parity_case, build_native_logic_compare_parity_case,
    build_native_shift_parity_case, build_native_upper_parity_case, build_native_word_arith_parity_case,
    build_native_word_shift_parity_case, build_parity_case_from_source, build_signed_divrem_parity_case,
    build_unsigned_divrem_parity_case, build_vertical_slice_parity_case, control_flow_beq_manifest,
    control_flow_bge_manifest, control_flow_bgeu_manifest, control_flow_blt_manifest, control_flow_bltu_manifest,
    control_flow_bne_manifest, control_flow_focus_manifest, control_flow_jal_manifest, control_flow_jalr_manifest,
    multiply_high_manifest, multiply_low_manifest, narrow_memory_load_manifest, narrow_memory_store_manifest,
    native_alu_focus_manifest, native_logic_compare_manifest, native_shift_manifest, native_upper_manifest,
    native_word_arith_manifest, native_word_shift_manifest, parity_source_cases, signed_divrem_manifest,
    unsigned_divrem_manifest, vertical_slice_manifest, Rv64imKernelSummary, Rv64imParityCaseManifest,
    Rv64imParityDerivedCase, Rv64imParitySourceCase,
};
pub(crate) use artifacts::{
    family_word, opcode_word, ram_access_kind_word, register_read_role_word, trace_virtual_opcode_word,
};
pub use canonical_openings::{
    AjtaiFamilyKind, AjtaiObjectId, AjtaiOpeningId, OpeningAccumulator, OpeningAccumulatorStats, OpeningAliasError,
    SelectedOpeningRef,
};
pub use lane_family::{
    build_main_lane_family_summary, prepared_step_digest, public_step_digest, public_step_family_digest,
    same_public_step, MainLaneFamilySummary,
};
pub(crate) use lane_family::{prepared_step_digests, public_step_digests};
pub use main_lane_artifact::{
    build_simple_kernel_main_lane_artifact, validate_simple_kernel_main_lane_artifact, SimpleKernelMainLaneArtifact,
    SimpleKernelMainLaneBinding,
};
pub use main_lane_surface::{build_main_lane_surface, Rv64imMainLaneSurface};
pub use opening_accumulate::{
    build_phase2_collapse_result, verify_phase2_collapse_result, Phase2CollapseError, Phase2CollapseRecord,
    Phase2CollapseResult, ReducedEvalClaim,
};
pub use opening_batch::{build_rv64im_opening_bundle_from_accepted_artifact, Rv64imOpeningBundle};
pub use opening_claim_reduction::{
    build_claim_reduction_buckets, build_claim_reduction_results_from_witnesses, domain_for_schema,
    phase1_claim_digest, phase1_unified_claim_digest, verify_claim_reduction_result_with_binding_surface,
    verify_claim_reduction_results_with_binding_surface, ClaimReductionBucket, ClaimReductionError,
    ClaimReductionProof, ClaimReductionResult, QuadraticRoundPoly,
};
pub use opening_eval_claim_witness::{
    build_rv64im_eval_claim_bundle_from_accepted_artifact, build_rv64im_eval_claim_bundle_from_claim_witnesses,
    build_rv64im_eval_claim_witnesses_from_accepted_artifact, build_stage1_claim_witnesses,
    build_stage2_claim_witnesses, build_stage3_claim_witness, verify_rv64im_eval_claim_bundle_from_accepted_artifact,
    FamilyEvalClaimWitness, OpenedAjtaiObjectWitness, PackedColumnOracleRef, RealAjtaiCommitmentVector,
};
pub(crate) use opening_eval_claim_witness::{
    build_rv64im_eval_claim_bundle_from_claim_witnesses_trusted_local,
    build_rv64im_eval_claim_witnesses_from_accepted_artifact_with_perf, phase0_binding_digest,
};
pub use opening_eval_claims::{
    phase0_family_order, CommitmentContextId, EvalClaimError, FamilyEvalClaim, FamilyEvalClaimId, FamilyEvalPayload,
    FamilyEvalSchemaId, OpenedAjtaiObjectId, OpeningClaimAccumulator, PackedColumnEval, Rv64imEvalClaimBundle,
};
pub(crate) use opening_final::{
    build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local,
    build_rv64im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf,
    rebuild_opened_object_witness_from_projection,
};
pub use opening_final::{
    build_rv64im_opening_convergence_artifact_from_proof, build_rv64im_opening_convergence_artifact_from_witnesses,
    build_rv64im_opening_convergence_proof_from_witnesses, verify_rv64im_opening_convergence_artifact,
    verify_rv64im_opening_convergence_artifact_from_proof, verify_rv64im_opening_convergence_proof, AjtaiOpeningProof,
    FinalOpeningError, FinalOpeningTarget, OpenedAjtaiCommitmentPublic, ProjectedFinalOpeningTarget,
    RealAjtaiCommitmentVectorPublic, Rv64imOpeningConvergenceArtifact, Rv64imOpeningConvergenceProof,
};
pub use opening_manifest::{
    opening_claims_from_carriers, stage1_opening_witness_carriers, stage1_opening_witness_carriers_from_claim_surface,
    stage2_opening_witness_carriers, stage2_opening_witness_carriers_from_claim_surface,
    stage3_opening_witness_carriers, stage3_opening_witness_carriers_from_claim_surface, Rv64imOpeningWitnessCarrier,
};
pub use opening_payload_semantics::{
    encode_packed_column_evals_k, encode_words_to_field_evals_k, phase0_full_width_for_schema,
    phase0_word_count_for_schema, reconstruct_words_from_field_evals, unpack_column_evals_k,
};
pub use opening_phase0_binding_surface::{
    build_rv64im_phase0_binding_surface_from_accepted_artifact, Rv64imPhase0BindingSurface, Rv64imPhase0BindingTarget,
};
pub use opening_point_derivation::{derive_phase0_point, derive_phase0_point_from_seed, phase0_point_seed};
pub use opening_verify::verify_rv64im_opening_bundle_from_accepted_artifact;
pub use perf_diagnostics::{
    ExactStageVectorBuildPerf, KernelOpeningBundleBuildPerf, KernelOpeningBundleVerifyPerf, PackagedOpeningBuildPerf,
    PackagedSimpleKernelVerifyPerf, RootMainLanePackagedProofProvePerf, RootMainLanePackagedProofVerifyPerf,
    RootMainLaneRunProofProvePerf, RootMainLaneRunProofVerifyPerf, Rv64imProofProvePerf, Rv64imPublicProofVerifyPerf,
    SimpleKernelBuildPerf, SimpleKernelVerifyPerf, StageClaimBundleBuildPerf, StagePackageBundleBuildPerf,
    StagePackageBundleVerifyPerf,
};
pub(crate) use proof_accepted::accepted_proof_artifact_from_prover_materials;
pub use proof_accepted::{Rv64imAcceptedProofArtifact, Rv64imAuditBundle};
pub(crate) use proof_api::prove_rv64im_public_proof_prover_seam_with_perf;
pub use proof_api::{
    audit_rv64im_accepted_proof_against_input, audit_rv64im_accepted_proof_against_input_with_perf,
    build_rv64im_accepted_proof_artifact, build_rv64im_audit_bundle, build_rv64im_audit_witness_bundle,
    prove_rv64im_accepted_proof, prove_rv64im_accepted_proof_with_options,
    prove_rv64im_accepted_proof_with_options_and_perf, prove_rv64im_accepted_proof_with_perf, prove_rv64im_audit_proof,
    prove_rv64im_audit_proof_with_perf, prove_rv64im_public_proof, prove_rv64im_public_proof_with_options,
    prove_rv64im_public_proof_with_options_and_perf, prove_rv64im_public_proof_with_perf,
    validate_rv64im_public_proof_against_input, validate_rv64im_public_proof_against_input_with_perf,
    verify_rv64im_accepted_proof, verify_rv64im_accepted_proof_with_perf, verify_rv64im_audit_proof,
    verify_rv64im_audit_proof_with_perf, verify_rv64im_public_proof, verify_rv64im_public_proof_with_perf,
    Rv64imAcceptedProofClaim, Rv64imAcceptedProofMainLaneBinding, Rv64imAcceptedProofStatementBinding,
    Rv64imAcceptedProofTerminalBinding, Rv64imJointOpeningClaim, Rv64imKernelClaimBundle, Rv64imKernelOpeningClaim,
    Rv64imKernelProofBundle, Rv64imMainLaneClaim, Rv64imMainLaneClaimBinding, Rv64imMainLaneProofBinding,
    Rv64imMainLaneProofBundle, Rv64imMainLaneProofSummaryBundle, Rv64imProof, Rv64imProofInput, Rv64imProofStatement,
    Rv64imPublicProofOptions, Rv64imRoot0Claim,
};
pub(crate) use proof_bridge::kernel_claim_bundle_from_statement_and_compact_surfaces;
pub use proof_completeness::{KernelSoundnessAccountingSurface, StepCompositionSurface};
pub(crate) use proof_export_relation::{
    build_rv64im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs,
    build_rv64im_kernel_export_proof_from_accepted_artifact,
    build_rv64im_kernel_export_proof_from_carried_accepted_artifact, rv64im_public_chunk_digest,
    verify_rv64im_kernel_export_proof_with_output, verify_rv64im_kernel_export_proof_with_relation_output,
    Rv64imKernelExportRelationResult,
};
pub use proof_export_relation::{
    build_rv64im_kernel_export_relation, build_rv64im_kernel_export_source_from_accepted_artifact,
    build_rv64im_kernel_export_witness, verify_rv64im_kernel_export_relation, verify_rv64im_kernel_export_source,
    verify_rv64im_kernel_export_witness, Rv64imChunkBridgeHandoff, Rv64imChunkExportSurface,
    Rv64imKernelChunkExportWitness, Rv64imKernelExportMainLaneProof, Rv64imKernelExportProof,
    Rv64imKernelExportRelation, Rv64imKernelExportSource, Rv64imKernelExportWitness, Rv64imPreparedStepBridgeBinding,
    Rv64imVerifiedKernelChunkHandoff,
};
pub(crate) use proof_staged_verify::build_verified_stage3_claim_from_accepted_artifact;
pub use proof_staged_verify::{
    Rv64imStage1ExportProof, Rv64imStage2ExportProof, Rv64imStage3ExportProof, Stage1VerifiedClaims,
    Stage2VerifiedClaims, Stage3VerifiedClaims, VerifierClaimAccumulator,
};
pub(crate) use proof_witness::{
    build_kernel_claim_packaged_public_step_from_compact_surfaces, build_stage_claim_packaged_public_step,
};
pub use proof_witness::{
    Rv64imKernelClaimProofBundle, Rv64imKernelClaimSummaryBundle, Rv64imKernelClaimSummaryProofBundle,
    Rv64imKernelClaimTerminalBundle, Rv64imKernelExportClaimProof, Rv64imKernelOpeningBindingBundle,
    Rv64imKernelOpeningProofBundle, Rv64imKernelOpeningSummaryBundle, Rv64imProofWitnessBundle,
    Rv64imStageClaimDigestBundle, Rv64imStageClaimProofBundle, Rv64imStageClaimSummaryProofBundle,
    Rv64imStagePackageDigestBundle, Rv64imStagePackageProofBundle, Rv64imStagePackageSummaryProofBundle,
    Rv64imStageWitnessProjectionBundle, Rv64imStageWitnessProofBundle, Rv64imStageWitnessSummaryBundle,
    Rv64imTraceProjectionBundle, Rv64imTraceProofBundle, Rv64imTraceShapeBundle,
};
pub use root_lane_columns::{build_root_lane_columns, RootLaneColumns};
pub use root_lane_commitment::{
    build_root_lane_commitment_artifact, verify_root_lane_commitment_artifact, RootLaneCommitmentArtifact,
    RootLaneCommitmentSet, RootLaneCommitmentSetSummary, RootLaneCommitmentSummaryArtifact, RootLaneOpeningProof,
};
pub use root_lane_witness::{
    RootExecutionBundle, RootExecutionSemanticsRefinement, RootExecutionSemanticsRefinementSummary,
    RootRowLocalCcsAcceptance, RootRowLocalCcsAcceptanceSummary, RootSemanticRow, RowChunkRoute,
};
pub use simple::{
    build_simple_kernel_witness, build_simple_kernel_witness_with_perf, prove_packaged_simple_kernel,
    prove_packaged_simple_kernel_with_perf, prove_root_main_lane_packaged_proof_with_perf,
    prove_root_main_lane_run_proof_with_perf, prove_simple_kernel, rv64im_ajtai_mixers, rv64im_exact_stage_pp_seed,
    rv64im_simple_kernel_pp_seed, rv64im_simple_root_context_id, rv64im_simple_root_params,
    verify_packaged_simple_kernel, verify_packaged_simple_kernel_with_perf,
    verify_root_main_lane_packaged_proof_with_public_rows, verify_root_main_lane_run_proof_with_public_rows,
    verify_simple_kernel, verify_simple_kernel_with_perf, PreparedStepBinding, PreparedStepBindingSummary,
    SimpleKernelAuditOutput, SimpleKernelError, SimpleKernelKernelClaimBundle, SimpleKernelOutput,
    SimpleKernelPackagedProof, SimpleKernelProof, SimpleKernelProverInput, SimpleKernelPublicInput,
    SimpleKernelStageWitnessBundle, SimpleKernelTraceWitness, SimpleKernelVerifierInput,
};
pub(crate) use simple::{rv64im_cached_root_main_lane_context, rv64im_cached_root_main_lane_optimized_cache};
pub use simple_openings::{
    KernelBindingOpeningClaim, KernelBindingOpeningPoints, KernelBindingPackagedOpeningProof,
    KernelPreparedStepOpeningClaim, KernelPreparedStepOpeningPoints, KernelPreparedStepPackagedOpeningProof,
    OpeningPointLabel, SimpleKernelOpeningBundle, SimpleKernelOpeningClaim, SimpleKernelStagePackageBundle,
    Stage1OpeningPoints, Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2OpeningPoints,
    Stage2PackagedOpeningProof, Stage2SelectedOpeningClaim, Stage3OpeningPoints, Stage3PackagedOpeningProof,
    Stage3SelectedOpeningClaim,
};
pub use stage1_canonical::Stage1CanonicalRowBundle;
pub use stage2_canonical::Stage2CanonicalFamilyBundle;
pub use stage3_canonical::Stage3CanonicalContinuityBundle;
pub(crate) use stage_artifacts::{
    build_claim_packaged_public_step, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, build_public_kernel_opening_claim_from_compact_surfaces,
    RV64IM_SELECTED_OPENING_LAYOUT_V1,
};
pub use stage_artifacts::{
    SimpleKernelStageClaimBundle, Stage1ArtifactSurface, Stage1ClaimSurface, Stage2ArtifactSurface, Stage2ClaimSurface,
    Stage3ArtifactSurface, Stage3ClaimSurface, StageDigestCommitment, TranscriptArtifactSurface,
    TranscriptClaimSurface,
};
pub(crate) use transcript::verify_transcript_record;
pub use transcript::{
    TranscriptChallenges, TranscriptCursorSnapshot, TranscriptEventKind, TranscriptEventRecord, TranscriptInitialState,
    TranscriptRecord, VerifiedTranscriptSurface,
};
