//! Owns the RV64IM frontend parity slice: machine layer, staged summaries, and kernel artifacts.

pub mod builder;
pub mod ccs;
mod chunk_relation;
pub mod decider;
pub mod decider_relation;
pub mod execute;
pub mod final_relation;
pub mod isa;
pub mod kernel;
pub mod layout;
pub mod lower;
pub mod main_relation;
mod perf_case;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod tables;
mod trace_expand;

pub use builder::{build_program, Rv64ProgramBuild};
pub use decider::{
    build_rv64im_published_proof_seam, build_rv64im_published_proof_seam_with_perf,
    build_rv64im_spartan2_decider_target, prove_rv64im_public_proof_and_published_seam_with_options_and_perf,
    prove_rv64im_public_proof_and_published_seam_with_perf, prove_rv64im_spartan2_decider,
    prove_rv64im_spartan2_decider_for_target, prove_rv64im_spartan2_decider_for_target_with_perf,
    prove_rv64im_spartan2_decider_from_public_proof, setup_rv64im_spartan2_decider,
    setup_rv64im_spartan2_decider_for_target, setup_rv64im_spartan2_decider_from_public_proof,
    verify_rv64im_spartan2_decider, verify_rv64im_spartan2_decider_for_target,
    verify_rv64im_spartan2_decider_from_public_proof, Rv64imPublicProofAndSeamBuildPerf, Rv64imPublishedProofSeam,
    Rv64imPublishedProofSeamBuildPerf,
};
pub use decider_relation::{
    build_rv64im_decider_relation, validate_rv64im_decider_relation_surface, verify_rv64im_decider_relation,
    Rv64imDeciderRelation,
};
pub use isa::{
    decode_instruction, encode_add, encode_addi, encode_addiw, encode_addw, encode_and, encode_andi, encode_auipc,
    encode_beq, encode_bge, encode_bgeu, encode_blt, encode_bltu, encode_bne, encode_div, encode_divu, encode_divuw,
    encode_divw, encode_ecall, encode_fence, encode_jal, encode_jalr, encode_lb, encode_lbu, encode_ld, encode_lh,
    encode_lhu, encode_lui, encode_lw, encode_lwu, encode_mul, encode_mulh, encode_mulhsu, encode_mulhu, encode_mulw,
    encode_or, encode_ori, encode_rem, encode_remu, encode_remuw, encode_remw, encode_sb, encode_sd, encode_sh,
    encode_sll, encode_slli, encode_slliw, encode_sllw, encode_slt, encode_slti, encode_sltiu, encode_sltu, encode_sra,
    encode_srai, encode_sraiw, encode_sraw, encode_srl, encode_srli, encode_srliw, encode_srlw, encode_sub,
    encode_subw, encode_sw, encode_xor, encode_xori, MemoryWord, Rv64BuildError, Rv64DecodedInstruction, Rv64Opcode,
    Rv64Program, Rv64State,
};
pub use kernel::{
    aligned_memory_focus_manifest, audit_rv64im_accepted_proof_against_input,
    audit_rv64im_accepted_proof_against_input_with_perf, build_aligned_memory_focus_parity_case,
    build_all_parity_cases, build_claim_reduction_buckets, build_claim_reduction_results_from_witnesses,
    build_control_flow_beq_parity_case, build_control_flow_bge_parity_case, build_control_flow_bgeu_parity_case,
    build_control_flow_blt_parity_case, build_control_flow_bltu_parity_case, build_control_flow_bne_parity_case,
    build_control_flow_focus_parity_case, build_control_flow_jal_parity_case, build_control_flow_jalr_parity_case,
    build_main_lane_surface, build_multiply_high_parity_case, build_multiply_low_parity_case,
    build_narrow_memory_load_parity_case, build_narrow_memory_store_parity_case, build_native_alu_focus_parity_case,
    build_native_logic_compare_parity_case, build_native_shift_parity_case, build_native_upper_parity_case,
    build_native_word_arith_parity_case, build_native_word_shift_parity_case, build_parity_case_from_source,
    build_phase2_collapse_result, build_rv64im_accepted_proof_artifact, build_rv64im_audit_bundle,
    build_rv64im_audit_witness_bundle, build_rv64im_eval_claim_bundle_from_accepted_artifact,
    build_rv64im_eval_claim_bundle_from_claim_witnesses, build_rv64im_eval_claim_witnesses_from_accepted_artifact,
    build_rv64im_kernel_export_relation, build_rv64im_kernel_export_source_from_accepted_artifact,
    build_rv64im_kernel_export_witness, build_rv64im_opening_bundle_from_accepted_artifact,
    build_rv64im_opening_convergence_artifact_from_proof, build_rv64im_opening_convergence_artifact_from_witnesses,
    build_rv64im_opening_convergence_proof_from_witnesses, build_rv64im_phase0_binding_surface_from_accepted_artifact,
    build_signed_divrem_parity_case, build_simple_kernel_witness, build_simple_kernel_witness_with_perf,
    build_stage1_claim_witnesses, build_stage2_claim_witnesses, build_stage3_claim_witness,
    build_unsigned_divrem_parity_case, build_vertical_slice_parity_case, control_flow_beq_manifest,
    control_flow_bge_manifest, control_flow_bgeu_manifest, control_flow_blt_manifest, control_flow_bltu_manifest,
    control_flow_bne_manifest, control_flow_focus_manifest, control_flow_jal_manifest, control_flow_jalr_manifest,
    derive_phase0_point, derive_phase0_point_from_seed, domain_for_schema, encode_packed_column_evals_k,
    encode_words_to_field_evals_k, multiply_high_manifest, multiply_low_manifest, narrow_memory_load_manifest,
    narrow_memory_store_manifest, native_alu_focus_manifest, native_logic_compare_manifest, native_shift_manifest,
    native_upper_manifest, native_word_arith_manifest, native_word_shift_manifest, parity_source_cases,
    phase0_family_order, phase0_full_width_for_schema, phase0_point_seed, phase0_word_count_for_schema,
    phase1_claim_digest, phase1_unified_claim_digest, prepared_step_digest, prove_packaged_simple_kernel,
    prove_packaged_simple_kernel_with_perf, prove_root_main_lane_packaged_proof_with_perf,
    prove_root_main_lane_run_proof_with_perf, prove_rv64im_accepted_proof, prove_rv64im_accepted_proof_with_options,
    prove_rv64im_accepted_proof_with_options_and_perf, prove_rv64im_accepted_proof_with_perf, prove_rv64im_audit_proof,
    prove_rv64im_audit_proof_with_perf, prove_rv64im_public_proof, prove_rv64im_public_proof_with_options,
    prove_rv64im_public_proof_with_options_and_perf, prove_rv64im_public_proof_with_perf, prove_simple_kernel,
    public_step_digest, public_step_family_digest, reconstruct_words_from_field_evals, rv64im_ajtai_mixers,
    rv64im_simple_root_context_id, rv64im_simple_root_params, signed_divrem_manifest, unpack_column_evals_k,
    unsigned_divrem_manifest, validate_rv64im_public_proof_against_input,
    validate_rv64im_public_proof_against_input_with_perf, verify_claim_reduction_result_with_binding_surface,
    verify_claim_reduction_results_with_binding_surface, verify_packaged_simple_kernel,
    verify_packaged_simple_kernel_with_perf, verify_phase2_collapse_result,
    verify_root_main_lane_packaged_proof_with_public_rows, verify_root_main_lane_run_proof_with_public_rows,
    verify_rv64im_accepted_proof, verify_rv64im_accepted_proof_with_perf, verify_rv64im_audit_proof,
    verify_rv64im_audit_proof_with_perf, verify_rv64im_eval_claim_bundle_from_accepted_artifact,
    verify_rv64im_kernel_export_relation, verify_rv64im_kernel_export_source, verify_rv64im_kernel_export_witness,
    verify_rv64im_opening_bundle_from_accepted_artifact, verify_rv64im_opening_convergence_artifact,
    verify_rv64im_opening_convergence_artifact_from_proof, verify_rv64im_opening_convergence_proof,
    verify_rv64im_public_proof, verify_rv64im_public_proof_with_perf, verify_simple_kernel,
    verify_simple_kernel_with_perf, vertical_slice_manifest, AjtaiFamilyKind, AjtaiObjectId, AjtaiOpeningId,
    AjtaiOpeningProof, ClaimReductionBucket, ClaimReductionError, ClaimReductionProof, ClaimReductionResult,
    CommitmentContextId, EvalClaimError, ExactStageVectorBuildPerf, FamilyEvalClaim, FamilyEvalClaimId,
    FamilyEvalClaimWitness, FamilyEvalPayload, FamilyEvalSchemaId, FinalOpeningError, FinalOpeningTarget,
    KernelBindingOpeningClaim, KernelBindingOpeningPoints, KernelBindingPackagedOpeningProof,
    KernelOpeningBundleBuildPerf, KernelOpeningBundleVerifyPerf, KernelPreparedStepOpeningClaim,
    KernelPreparedStepOpeningPoints, KernelPreparedStepPackagedOpeningProof, KernelSoundnessAccountingSurface,
    MainLaneFamilySummary, OpenedAjtaiCommitmentPublic, OpenedAjtaiObjectId, OpenedAjtaiObjectWitness,
    OpeningAccumulator, OpeningAccumulatorStats, OpeningAliasError, OpeningClaimAccumulator, OpeningPointLabel,
    PackagedOpeningBuildPerf, PackagedSimpleKernelVerifyPerf, PackedColumnEval, PackedColumnOracleRef,
    Phase2CollapseError, Phase2CollapseRecord, Phase2CollapseResult, PreparedStepBinding, PreparedStepBindingSummary,
    ProjectedFinalOpeningTarget, QuadraticRoundPoly, RealAjtaiCommitmentVector, RealAjtaiCommitmentVectorPublic,
    ReducedEvalClaim, RootExecutionBundle, RootLaneColumns, RootLaneCommitmentSetSummary,
    RootLaneCommitmentSummaryArtifact, RootMainLaneRunProofProvePerf, RootMainLaneRunProofVerifyPerf, RootSemanticRow,
    RowChunkRoute, Rv64imAcceptedProofArtifact, Rv64imAcceptedProofClaim, Rv64imAcceptedProofMainLaneBinding,
    Rv64imAcceptedProofStatementBinding, Rv64imAcceptedProofTerminalBinding, Rv64imAuditBundle,
    Rv64imChunkBridgeHandoff, Rv64imChunkExportSurface, Rv64imEvalClaimBundle, Rv64imJointOpeningClaim,
    Rv64imKernelChunkExportWitness, Rv64imKernelClaimBundle, Rv64imKernelClaimProofBundle,
    Rv64imKernelClaimSummaryBundle, Rv64imKernelClaimSummaryProofBundle, Rv64imKernelClaimTerminalBundle,
    Rv64imKernelExportClaimProof, Rv64imKernelExportMainLaneProof, Rv64imKernelExportProof, Rv64imKernelExportRelation,
    Rv64imKernelExportSource, Rv64imKernelExportWitness, Rv64imKernelOpeningBindingBundle, Rv64imKernelOpeningClaim,
    Rv64imKernelOpeningProofBundle, Rv64imKernelOpeningSummaryBundle, Rv64imKernelProofBundle, Rv64imKernelSummary,
    Rv64imMainLaneClaim, Rv64imMainLaneClaimBinding, Rv64imMainLaneProofBinding, Rv64imMainLaneProofBundle,
    Rv64imMainLaneProofSummaryBundle, Rv64imMainLaneSurface, Rv64imOpeningBundle, Rv64imOpeningConvergenceArtifact,
    Rv64imOpeningConvergenceProof, Rv64imOpeningWitnessCarrier, Rv64imParityCaseManifest, Rv64imParityDerivedCase,
    Rv64imParitySourceCase, Rv64imPhase0BindingSurface, Rv64imPhase0BindingTarget, Rv64imPreparedStepBridgeBinding,
    Rv64imProof, Rv64imProofInput, Rv64imProofProvePerf, Rv64imProofStatement, Rv64imProofWitnessBundle,
    Rv64imPublicProofOptions, Rv64imPublicProofVerifyPerf, Rv64imRoot0Claim, Rv64imStage1ExportProof,
    Rv64imStage2ExportProof, Rv64imStage3ExportProof, Rv64imStageClaimDigestBundle, Rv64imStageClaimProofBundle,
    Rv64imStageClaimSummaryProofBundle, Rv64imStagePackageDigestBundle, Rv64imStagePackageProofBundle,
    Rv64imStagePackageSummaryProofBundle, Rv64imStageWitnessProjectionBundle, Rv64imStageWitnessProofBundle,
    Rv64imStageWitnessSummaryBundle, Rv64imTraceProjectionBundle, Rv64imTraceProofBundle, Rv64imTraceShapeBundle,
    Rv64imVerifiedKernelChunkHandoff, SelectedOpeningRef, SimpleKernelAuditOutput, SimpleKernelBuildPerf,
    SimpleKernelError, SimpleKernelKernelClaimBundle, SimpleKernelMainLaneArtifact, SimpleKernelMainLaneBinding,
    SimpleKernelOpeningBundle, SimpleKernelOpeningClaim, SimpleKernelOutput, SimpleKernelPackagedProof,
    SimpleKernelProof, SimpleKernelProverInput, SimpleKernelPublicInput, SimpleKernelStageClaimBundle,
    SimpleKernelStagePackageBundle, SimpleKernelStageWitnessBundle, SimpleKernelTraceWitness,
    SimpleKernelVerifierInput, SimpleKernelVerifyPerf, Stage1ArtifactSurface, Stage1CanonicalRowBundle,
    Stage1ClaimSurface, Stage1OpeningPoints, Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim,
    Stage1VerifiedClaims, Stage2ArtifactSurface, Stage2CanonicalFamilyBundle, Stage2ClaimSurface, Stage2OpeningPoints,
    Stage2PackagedOpeningProof, Stage2SelectedOpeningClaim, Stage2VerifiedClaims, Stage3ArtifactSurface,
    Stage3CanonicalContinuityBundle, Stage3ClaimSurface, Stage3OpeningPoints, Stage3PackagedOpeningProof,
    Stage3SelectedOpeningClaim, Stage3VerifiedClaims, StageClaimBundleBuildPerf, StageDigestCommitment,
    StagePackageBundleBuildPerf, StagePackageBundleVerifyPerf, StepCompositionSurface, TranscriptArtifactSurface,
    TranscriptChallenges, TranscriptClaimSurface, TranscriptCursorSnapshot, TranscriptEventKind, TranscriptEventRecord,
    TranscriptInitialState, TranscriptRecord, VerifiedTranscriptSurface, VerifierClaimAccumulator,
};
pub use lower::{Rv64ExpandedRow, Rv64TraceOpcode, Rv64TraceVirtualOpcode};
pub use main_relation::{
    build_rv64im_main_relation, build_rv64im_main_relation_backend_relation,
    build_rv64im_main_relation_backend_relation_from_artifact, build_rv64im_main_relation_from_final,
    validate_rv64im_main_relation_surface, verify_rv64im_main_relation, Rv64imMainRelationArtifact,
    Rv64imMainRelationStatement, Rv64imMainRelationWitness,
};
pub use perf_case::{
    build_mixed_opcode_perf_source_case, mixed_opcode_perf_expected_x1, RV64IM_MIXED_OPCODE_PERF_BLOCK_LEN,
    RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
pub use stage1::{
    build_sem_inputs, build_stage1_proof_bundle, sem_in_digest, sem_in_from_row, sem_inputs_digest, AluShoutProof,
    BranchShoutProof, BytecodeShoutProof, SemIn, Stage1AddressCorrectnessProof, Stage1LinkageProof, Stage1ProofBundle,
};
pub use stage2::{
    build_stage2_proof_bundle, RamTwistProof, RegisterTwistProof, Stage2LinkageProof, Stage2ProofBundle,
    Stage2SemanticsProof, Stage2TemporalContext,
};
pub use stage3::{
    build_stage3_proof_bundle, PcAdjacentBridge, Stage3LinkageProof, Stage3ProofBundle, Stage3SemanticsProof,
};
