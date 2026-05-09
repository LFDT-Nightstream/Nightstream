//! Owns the RV32IM parity-slice kernel artifacts and transcript logging.

mod canonical_openings;
mod openings;
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
    build_native_rv32_shift_mask_parity_case, build_native_rv32_wrap_parity_case, build_native_shift_parity_case,
    build_native_upper_parity_case, build_parity_case_from_source, build_signed_divrem_parity_case,
    build_unsigned_divrem_parity_case, build_vertical_slice_parity_case, control_flow_beq_manifest,
    control_flow_bge_manifest, control_flow_bgeu_manifest, control_flow_blt_manifest, control_flow_bltu_manifest,
    control_flow_bne_manifest, control_flow_focus_manifest, control_flow_jal_manifest, control_flow_jalr_manifest,
    multiply_high_manifest, multiply_low_manifest, narrow_memory_load_manifest, narrow_memory_store_manifest,
    native_alu_focus_manifest, native_logic_compare_manifest, native_rv32_shift_mask_manifest,
    native_rv32_wrap_manifest, native_shift_manifest, native_upper_manifest, parity_source_cases,
    signed_divrem_manifest, unsigned_divrem_manifest, vertical_slice_manifest, Rv32imKernelSummary,
    Rv32imParityCaseManifest, Rv32imParityDerivedCase, Rv32imParitySourceCase,
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
pub use main_lane_surface::{build_main_lane_surface, Rv32imMainLaneSurface};
pub use openings::*;
pub(crate) use openings::{
    build_claim_packaged_public_step, build_commitment_vector, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, build_public_kernel_opening_claim_from_compact_surfaces,
    build_rv32im_eval_claim_bundle_from_claim_witnesses_trusted_local,
    build_rv32im_eval_claim_witnesses_from_accepted_artifact_with_perf,
    build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local,
    build_rv32im_opening_convergence_artifact_from_phase0_bundle_and_witnesses_trusted_local_with_perf,
    phase0_binding_digest, rebuild_opened_object_witness_from_projection, RV32IM_SELECTED_OPENING_LAYOUT_V1,
};
pub use perf_diagnostics::{
    ExactStageVectorBuildPerf, KernelOpeningBundleBuildPerf, KernelOpeningBundleVerifyPerf, PackagedOpeningBuildPerf,
    PackagedSimpleKernelVerifyPerf, RootMainLanePackagedProofProvePerf, RootMainLanePackagedProofVerifyPerf,
    RootMainLaneRunProofProvePerf, RootMainLaneRunProofVerifyPerf, Rv32imProofProvePerf, Rv32imPublicProofVerifyPerf,
    SimpleKernelBuildPerf, SimpleKernelVerifyPerf, StageClaimBundleBuildPerf, StagePackageBundleBuildPerf,
    StagePackageBundleVerifyPerf,
};
pub(crate) use proof_accepted::accepted_proof_artifact_from_prover_materials;
pub use proof_accepted::{Rv32imAcceptedProofArtifact, Rv32imAuditBundle};
pub(crate) use proof_api::prove_rv32im_public_proof_prover_seam_with_perf;
pub use proof_api::{
    audit_rv32im_accepted_proof, audit_rv32im_accepted_proof_against_input,
    audit_rv32im_accepted_proof_against_input_with_perf, audit_rv32im_accepted_proof_with_perf,
    audit_rv32im_public_proof, audit_rv32im_public_proof_against_input,
    audit_rv32im_public_proof_against_input_with_perf, audit_rv32im_public_proof_with_perf,
    audit_rv32im_public_proof_with_witness, audit_rv32im_public_proof_with_witness_and_perf,
};
pub use proof_api::{
    build_rv32im_accepted_proof_artifact, build_rv32im_audit_bundle, build_rv32im_audit_witness_bundle,
    prove_rv32im_accepted_proof, prove_rv32im_accepted_proof_with_options,
    prove_rv32im_accepted_proof_with_options_and_perf, prove_rv32im_accepted_proof_with_perf, prove_rv32im_audit_proof,
    prove_rv32im_audit_proof_with_perf, prove_rv32im_public_proof, prove_rv32im_public_proof_with_options,
    prove_rv32im_public_proof_with_options_and_perf, prove_rv32im_public_proof_with_perf, Rv32imAcceptedProofClaim,
    Rv32imAcceptedProofMainLaneBinding, Rv32imAcceptedProofStatementBinding, Rv32imAcceptedProofTerminalBinding,
    Rv32imJointOpeningClaim, Rv32imKernelClaimBundle, Rv32imKernelOpeningClaim, Rv32imKernelProofBundle,
    Rv32imMainLaneClaim, Rv32imMainLaneClaimBinding, Rv32imMainLaneProofBinding, Rv32imMainLaneProofBundle,
    Rv32imMainLaneProofSummaryBundle, Rv32imProof, Rv32imProofInput, Rv32imProofStatement, Rv32imPublicProofOptions,
    Rv32imRoot0Claim,
};
pub(crate) use proof_bridge::kernel_claim_bundle_from_statement_and_compact_surfaces;
pub use proof_completeness::{KernelSoundnessAccountingSurface, StepCompositionSurface};
pub(crate) use proof_export_relation::{
    build_rv32im_kernel_export_build_output_from_carried_accepted_artifact_with_source_and_chunk_inputs,
    build_rv32im_kernel_export_proof_from_accepted_artifact,
    build_rv32im_kernel_export_proof_from_carried_accepted_artifact, rv32im_public_chunk_digest,
    verify_rv32im_kernel_export_proof_with_output, verify_rv32im_kernel_export_proof_with_relation_output,
    Rv32imKernelExportRelationResult,
};
pub use proof_export_relation::{
    build_rv32im_kernel_export_relation, build_rv32im_kernel_export_source_from_accepted_artifact,
    build_rv32im_kernel_export_witness, verify_rv32im_kernel_export_relation, verify_rv32im_kernel_export_source,
    verify_rv32im_kernel_export_witness, Rv32imChunkBridgeHandoff, Rv32imChunkExportSurface,
    Rv32imKernelChunkExportWitness, Rv32imKernelExportMainLaneProof, Rv32imKernelExportProof,
    Rv32imKernelExportRelation, Rv32imKernelExportSource, Rv32imKernelExportWitness, Rv32imPreparedStepBridgeBinding,
    Rv32imVerifiedKernelChunkHandoff,
};
pub(crate) use proof_staged_verify::build_verified_stage3_claim_from_accepted_artifact;
pub use proof_staged_verify::{
    Rv32imStage1ExportProof, Rv32imStage2ExportProof, Rv32imStage3ExportProof, Stage1VerifiedClaims,
    Stage2VerifiedClaims, Stage3VerifiedClaims, VerifierClaimAccumulator,
};
pub(crate) use proof_witness::{
    build_kernel_claim_packaged_public_step_from_compact_surfaces, build_stage_claim_packaged_public_step,
};
pub use proof_witness::{
    Rv32imKernelClaimProofBundle, Rv32imKernelClaimSummaryBundle, Rv32imKernelClaimSummaryProofBundle,
    Rv32imKernelClaimTerminalBundle, Rv32imKernelExportClaimProof, Rv32imKernelOpeningBindingBundle,
    Rv32imKernelOpeningProofBundle, Rv32imKernelOpeningSummaryBundle, Rv32imProofWitnessBundle,
    Rv32imStageClaimDigestBundle, Rv32imStageClaimProofBundle, Rv32imStageClaimSummaryProofBundle,
    Rv32imStagePackageDigestBundle, Rv32imStagePackageProofBundle, Rv32imStagePackageSummaryProofBundle,
    Rv32imStageWitnessProjectionBundle, Rv32imStageWitnessProofBundle, Rv32imStageWitnessSummaryBundle,
    Rv32imTraceProjectionBundle, Rv32imTraceProofBundle, Rv32imTraceShapeBundle,
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
    prove_root_main_lane_run_proof_with_perf, prove_simple_kernel, rv32im_ajtai_mixers, rv32im_exact_stage_pp_seed,
    rv32im_simple_kernel_pp_seed, rv32im_simple_root_context_id, rv32im_simple_root_context_id_for_step_cap,
    rv32im_simple_root_k_rho_for_step_cap, rv32im_simple_root_params, rv32im_simple_root_params_for_step_cap,
    verify_packaged_simple_kernel, verify_packaged_simple_kernel_with_perf,
    verify_root_main_lane_packaged_proof_with_public_rows, verify_root_main_lane_run_proof_with_public_rows,
    verify_simple_kernel, verify_simple_kernel_with_perf, PreparedStepBinding, PreparedStepBindingSummary,
    SimpleKernelAuditOutput, SimpleKernelError, SimpleKernelKernelClaimBundle, SimpleKernelOutput,
    SimpleKernelPackagedProof, SimpleKernelProof, SimpleKernelProverInput, SimpleKernelPublicInput,
    SimpleKernelStageWitnessBundle, SimpleKernelTraceWitness, SimpleKernelVerifierInput,
};
pub(crate) use simple::{
    rv32im_cached_root_main_lane_context, rv32im_cached_root_main_lane_optimized_cache,
    rv32im_root_main_lane_context_for_claim_count, rv32im_root_main_lane_context_for_step_cap,
    rv32im_simple_root_context_id_for_schedule,
};
pub use stage1_canonical::Stage1CanonicalRowBundle;
pub use stage2_canonical::Stage2CanonicalFamilyBundle;
pub use stage3_canonical::Stage3CanonicalContinuityBundle;
pub(crate) use transcript::verify_transcript_record;
pub use transcript::{
    TranscriptChallenges, TranscriptCursorSnapshot, TranscriptEventKind, TranscriptEventRecord, TranscriptInitialState,
    TranscriptRecord, VerifiedTranscriptSurface,
};
