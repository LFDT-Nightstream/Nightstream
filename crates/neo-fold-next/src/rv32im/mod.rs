//! Owns the RV32IM frontend parity slice: machine layer, staged summaries, and kernel artifacts.

pub mod audit;
pub mod builder;
pub mod ccs;
pub mod chunk_fold_step;
mod chunk_relation;
mod chunk_step_ivc;
mod chunk_step_relation;
pub mod claim_tree;
pub mod construction2;
mod construction2_default;
mod decider;
mod encoded_public_input;
pub mod execute;
pub mod f_prime;
mod f_prime_accumulator;
mod f_prime_side;
pub mod final_relation;
pub mod isa;
pub mod ivc;
pub(crate) mod ivc_snark;
pub mod kernel;
pub mod layout;
pub mod lower;
pub mod main_proof;
pub mod main_recursion;
#[allow(dead_code)]
pub mod main_relation_circuit;
pub(crate) mod main_relation_spartan;
pub(crate) mod main_relation_trace;
mod nifs;
mod perf_case;
pub mod recursion_shape;
mod recursion_spartan;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod tables;
mod trace_expand;

pub use builder::{build_program, Rv32ProgramBuild};
pub use chunk_fold_step::{
    adapt_rv32im_chunk_to_fresh_ccs, rv32im_chunk_fold_seed, Rv32imAccumulatorHandle, Rv32imChunkFoldCarry,
    Rv32imChunkFoldFresh, Rv32imChunkStepPublic,
};
pub use chunk_step_ivc::{
    build_rv32im_chunk_step_ivc_relations, rv32im_chunk_step_ivc_initial_state, Rv32imChunkStepIvcRelation,
    Rv32imChunkStepIvcStatement, Rv32imChunkStepIvcWitness,
};
pub use claim_tree::{
    build_rv32im_claim_digests, rv32im_claim_tree_opening_from_digests, rv32im_claim_tree_root_from_claims,
    rv32im_claim_tree_root_from_digests, verify_rv32im_claim_tree_opening, Rv32imClaimDigestFields,
    Rv32imClaimMerkleOpening,
};
pub use construction2::{
    build_rv32im_main_recursion_construction2_default_fresh_instance,
    build_rv32im_main_recursion_construction2_default_pair,
    build_rv32im_main_recursion_construction2_f_prime_ccs_shape,
    build_rv32im_main_recursion_construction2_fresh_instance,
    build_rv32im_main_recursion_construction2_fresh_instance_with_input,
    build_rv32im_main_recursion_construction2_input_state_image,
    build_rv32im_main_recursion_construction2_output_state_image, build_rv32im_main_recursion_construction2_x_i,
    Rv32imMainRecursionConstruction2Commitment, Rv32imMainRecursionConstruction2FPrimeCcsShape,
    Rv32imMainRecursionConstruction2FreshInstance, Rv32imMainRecursionConstruction2PublicBoundary,
    Rv32imMainRecursionConstruction2StateImage,
};
pub use construction2_default::{
    build_rv32im_main_recursion_construction2_canonical_full_width,
    build_rv32im_main_recursion_construction2_canonical_shape,
    build_rv32im_main_recursion_construction2_default_full_width_from_ccs_shape,
    build_rv32im_main_recursion_construction2_default_full_width_from_relations,
    build_rv32im_main_recursion_construction2_default_pair_for_full_width, Rv32imMainRecursionConstruction2DefaultPair,
};
pub use decider::{
    prove_rv32im_public_proof_and_published_seam_with_perf, Rv32imPublicProofAndSeamBuildPerf,
    Rv32imPublishedProofSeam, Rv32imPublishedProofSeamBuildPerf,
};
pub use f_prime::{
    build_rv32im_main_recursion_f_prime_advices, build_rv32im_main_recursion_f_prime_advices_single_step,
    build_rv32im_main_recursion_f_prime_advices_single_step_with_perf,
    build_rv32im_main_recursion_f_prime_advices_with_perf,
    build_rv32im_main_recursion_f_prime_advices_with_side_opening_public,
    build_rv32im_main_recursion_f_prime_advices_with_side_opening_public_single_step,
    build_rv32im_main_recursion_f_prime_public_output, build_rv32im_main_recursion_side_lane_from_side_opening_public,
    build_rv32im_main_recursion_verifier_key_fs, build_rv32im_main_recursion_verifier_key_fs_for_step_cap,
    debug_trace_rv32im_main_recursion_f_prime_advices_single_step_build, evaluate_rv32im_main_recursion_f_prime_advice,
    verify_rv32im_main_recursion_f_prime_public_output, Rv32imEncodedPublicInput,
    Rv32imMainRecursionBackendStepStatement, Rv32imMainRecursionFPrimeAdvice, Rv32imMainRecursionFPrimeAdviceBuildPerf,
    Rv32imMainRecursionFPrimeAdviceStepBuildPerf, Rv32imMainRecursionFPrimeInput,
    Rv32imMainRecursionFPrimePublicOutput, Rv32imMainRecursionFPrimeStepImage, Rv32imMainRecursionPhiSide,
    Rv32imMainRecursionSideClaim, Rv32imMainRecursionSideLaneWitness, Rv32imMainRecursionStepStatement,
    Rv32imVerifierKeyFs, RV32IM_MAIN_RECURSION_PHI_SIDE_ACTIVE, RV32IM_MAIN_RECURSION_SIDE_LANE_ACTIVE,
    RV32IM_MAIN_RECURSION_SIDE_WITNESS_ACTIVE,
};
pub use isa::{
    decode_instruction, encode_add, encode_addi, encode_and, encode_andi, encode_auipc, encode_beq, encode_bge,
    encode_bgeu, encode_blt, encode_bltu, encode_bne, encode_div, encode_divu, encode_ecall, encode_fence, encode_jal,
    encode_jalr, encode_lb, encode_lbu, encode_lh, encode_lhu, encode_lui, encode_lw, encode_mul, encode_mulh,
    encode_mulhsu, encode_mulhu, encode_or, encode_ori, encode_rem, encode_remu, encode_sb, encode_sh, encode_sll,
    encode_slli, encode_slt, encode_slti, encode_sltiu, encode_sltu, encode_sra, encode_srai, encode_srl, encode_srli,
    encode_sub, encode_sw, encode_xor, encode_xori, MemoryWord, Rv32BuildError, Rv32DecodedInstruction, Rv32Opcode,
    Rv32Program, Rv32State,
};
pub use ivc::Rv32imIvcPublicImage;
pub use ivc_snark::{
    setup_rv32im_ivc_snark_cached, setup_rv32im_ivc_snark_cached_with_trace, setup_rv32im_ivc_snark_from_final,
    setup_rv32im_ivc_snark_from_final_cached, Rv32imIvcRecursionSnarkSetupShape, Rv32imIvcSnark, Rv32imIvcSnarkKeyPair,
    Rv32imIvcSnarkProof, Rv32imIvcSnarkProverKey, Rv32imIvcSnarkVerifierKey, Rv32imTerminalFPrimeCommittedStepProof,
};
pub use kernel::{
    aligned_memory_focus_manifest, build_aligned_memory_focus_parity_case, build_all_parity_cases,
    build_claim_reduction_buckets, build_claim_reduction_results_from_witnesses, build_control_flow_beq_parity_case,
    build_control_flow_bge_parity_case, build_control_flow_bgeu_parity_case, build_control_flow_blt_parity_case,
    build_control_flow_bltu_parity_case, build_control_flow_bne_parity_case, build_control_flow_focus_parity_case,
    build_control_flow_jal_parity_case, build_control_flow_jalr_parity_case, build_main_lane_surface,
    build_multiply_high_parity_case, build_multiply_low_parity_case, build_narrow_memory_load_parity_case,
    build_narrow_memory_store_parity_case, build_native_alu_focus_parity_case, build_native_logic_compare_parity_case,
    build_native_rv32_shift_mask_parity_case, build_native_rv32_wrap_parity_case, build_native_shift_parity_case,
    build_native_upper_parity_case, build_parity_case_from_source, build_phase2_collapse_result,
    build_rv32im_accepted_proof_artifact, build_rv32im_audit_bundle, build_rv32im_audit_witness_bundle,
    build_rv32im_eval_claim_bundle_from_accepted_artifact, build_rv32im_eval_claim_bundle_from_claim_witnesses,
    build_rv32im_eval_claim_witnesses_from_accepted_artifact, build_rv32im_kernel_export_relation,
    build_rv32im_kernel_export_source_from_accepted_artifact, build_rv32im_kernel_export_witness,
    build_rv32im_opening_bundle_from_accepted_artifact, build_rv32im_opening_convergence_artifact_from_proof,
    build_rv32im_opening_convergence_artifact_from_witnesses, build_rv32im_opening_convergence_proof_from_witnesses,
    build_rv32im_phase0_binding_surface_from_accepted_artifact, build_signed_divrem_parity_case,
    build_simple_kernel_witness, build_simple_kernel_witness_with_perf, build_stage1_claim_witnesses,
    build_stage2_claim_witnesses, build_stage3_claim_witness, build_unsigned_divrem_parity_case,
    build_vertical_slice_parity_case, control_flow_beq_manifest, control_flow_bge_manifest, control_flow_bgeu_manifest,
    control_flow_blt_manifest, control_flow_bltu_manifest, control_flow_bne_manifest, control_flow_focus_manifest,
    control_flow_jal_manifest, control_flow_jalr_manifest, derive_phase0_point, domain_for_schema,
    encode_packed_column_evals_k, encode_words_to_field_evals_k, multiply_high_manifest, multiply_low_manifest,
    narrow_memory_load_manifest, narrow_memory_store_manifest, native_alu_focus_manifest,
    native_logic_compare_manifest, native_rv32_shift_mask_manifest, native_rv32_wrap_manifest, native_shift_manifest,
    native_upper_manifest, parity_source_cases, phase0_family_order, phase0_full_width_for_schema,
    phase0_word_count_for_schema, phase1_claim_digest, phase1_unified_claim_digest, prepared_step_digest,
    prove_packaged_simple_kernel, prove_packaged_simple_kernel_with_perf,
    prove_root_main_lane_packaged_proof_with_perf, prove_root_main_lane_run_proof_with_perf,
    prove_rv32im_accepted_proof, prove_rv32im_accepted_proof_with_options,
    prove_rv32im_accepted_proof_with_options_and_perf, prove_rv32im_accepted_proof_with_perf, prove_rv32im_audit_proof,
    prove_rv32im_audit_proof_with_perf, prove_rv32im_public_proof, prove_rv32im_public_proof_with_options,
    prove_rv32im_public_proof_with_options_and_perf, prove_rv32im_public_proof_with_perf, prove_simple_kernel,
    public_step_digest, public_step_family_digest, reconstruct_words_from_field_evals, rv32im_ajtai_mixers,
    rv32im_simple_root_context_id, rv32im_simple_root_context_id_for_step_cap, rv32im_simple_root_k_rho_for_step_cap,
    rv32im_simple_root_params, rv32im_simple_root_params_for_step_cap, signed_divrem_manifest, unpack_column_evals_k,
    unsigned_divrem_manifest, verify_claim_reduction_result_with_binding_surface,
    verify_claim_reduction_results_with_binding_surface, verify_packaged_simple_kernel,
    verify_packaged_simple_kernel_with_perf, verify_phase2_collapse_result,
    verify_root_main_lane_packaged_proof_with_public_rows, verify_root_main_lane_run_proof_with_public_rows,
    verify_rv32im_eval_claim_bundle_from_accepted_artifact, verify_rv32im_kernel_export_relation,
    verify_rv32im_kernel_export_source, verify_rv32im_kernel_export_witness,
    verify_rv32im_opening_bundle_from_accepted_artifact, verify_rv32im_opening_convergence_artifact,
    verify_rv32im_opening_convergence_artifact_from_proof, verify_rv32im_opening_convergence_proof,
    verify_simple_kernel, verify_simple_kernel_with_perf, vertical_slice_manifest, AjtaiFamilyKind, AjtaiObjectId,
    AjtaiOpeningId, AjtaiOpeningProof, ClaimReductionBucket, ClaimReductionError, ClaimReductionProof,
    ClaimReductionResult, CommitmentContextId, EvalClaimError, ExactStageVectorBuildPerf, FamilyEvalClaim,
    FamilyEvalClaimId, FamilyEvalClaimWitness, FamilyEvalPayload, FamilyEvalSchemaId, FinalOpeningError,
    FinalOpeningTarget, KernelBindingOpeningClaim, KernelBindingOpeningPoints, KernelBindingPackagedOpeningProof,
    KernelOpeningBundleBuildPerf, KernelOpeningBundleVerifyPerf, KernelPreparedStepOpeningClaim,
    KernelPreparedStepOpeningPoints, KernelPreparedStepPackagedOpeningProof, KernelSoundnessAccountingSurface,
    MainLaneFamilySummary, OpenedAjtaiCommitmentPublic, OpenedAjtaiObjectId, OpenedAjtaiObjectWitness,
    OpeningAccumulator, OpeningAccumulatorStats, OpeningAliasError, OpeningClaimAccumulator, OpeningPointLabel,
    PackagedOpeningBuildPerf, PackagedSimpleKernelVerifyPerf, PackedColumnEval, PackedColumnOracleRef,
    Phase2CollapseError, Phase2CollapseRecord, Phase2CollapseResult, PreparedStepBinding, PreparedStepBindingSummary,
    ProjectedFinalOpeningTarget, QuadraticRoundPoly, RealAjtaiCommitmentVector, RealAjtaiCommitmentVectorPublic,
    ReducedEvalClaim, RootExecutionBundle, RootLaneColumns, RootLaneCommitmentSetSummary,
    RootLaneCommitmentSummaryArtifact, RootMainLaneRunProofProvePerf, RootMainLaneRunProofVerifyPerf, RootSemanticRow,
    RowChunkRoute, Rv32imAcceptedProofArtifact, Rv32imAcceptedProofClaim, Rv32imAcceptedProofMainLaneBinding,
    Rv32imAcceptedProofStatementBinding, Rv32imAcceptedProofTerminalBinding, Rv32imAuditBundle,
    Rv32imChunkBridgeHandoff, Rv32imChunkExportSurface, Rv32imEvalClaimBundle, Rv32imJointOpeningClaim,
    Rv32imKernelChunkExportWitness, Rv32imKernelClaimBundle, Rv32imKernelClaimProofBundle,
    Rv32imKernelClaimSummaryBundle, Rv32imKernelClaimSummaryProofBundle, Rv32imKernelClaimTerminalBundle,
    Rv32imKernelExportClaimProof, Rv32imKernelExportMainLaneProof, Rv32imKernelExportProof, Rv32imKernelExportRelation,
    Rv32imKernelExportSource, Rv32imKernelExportWitness, Rv32imKernelOpeningBindingBundle, Rv32imKernelOpeningClaim,
    Rv32imKernelOpeningProofBundle, Rv32imKernelOpeningSummaryBundle, Rv32imKernelProofBundle, Rv32imKernelSummary,
    Rv32imMainLaneClaim, Rv32imMainLaneClaimBinding, Rv32imMainLaneProofBinding, Rv32imMainLaneProofBundle,
    Rv32imMainLaneProofSummaryBundle, Rv32imMainLaneSurface, Rv32imOpeningBundle, Rv32imOpeningConvergenceArtifact,
    Rv32imOpeningConvergenceProof, Rv32imOpeningWitnessCarrier, Rv32imParityCaseManifest, Rv32imParityDerivedCase,
    Rv32imParitySourceCase, Rv32imPhase0BindingSurface, Rv32imPhase0BindingTarget, Rv32imPreparedStepBridgeBinding,
    Rv32imProof, Rv32imProofInput, Rv32imProofProvePerf, Rv32imProofStatement, Rv32imProofWitnessBundle,
    Rv32imPublicProofOptions, Rv32imPublicProofVerifyPerf, Rv32imRoot0Claim, Rv32imStage1ExportProof,
    Rv32imStage2ExportProof, Rv32imStage3ExportProof, Rv32imStageClaimDigestBundle, Rv32imStageClaimProofBundle,
    Rv32imStageClaimSummaryProofBundle, Rv32imStagePackageDigestBundle, Rv32imStagePackageProofBundle,
    Rv32imStagePackageSummaryProofBundle, Rv32imStageWitnessProjectionBundle, Rv32imStageWitnessProofBundle,
    Rv32imStageWitnessSummaryBundle, Rv32imTraceProjectionBundle, Rv32imTraceProofBundle, Rv32imTraceShapeBundle,
    Rv32imVerifiedKernelChunkHandoff, SelectedOpeningRef, SimpleKernelAuditOutput, SimpleKernelBuildPerf,
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
pub use lower::{Rv32ExpandedRow, Rv32TraceOpcode, Rv32TraceVirtualOpcode};
pub use main_proof::{Rv32imAccumulatorPublicStatement, Rv32imCompressedMainProof, Rv32imPublishedStatement};
pub use main_relation_spartan::debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint;
pub use main_relation_spartan::debug_measure_rv32im_main_relation_state_in_prefix_fingerprints;
pub use perf_case::{
    build_mixed_opcode_perf_source_case, mixed_opcode_perf_expected_x1, RV32IM_MIXED_OPCODE_PERF_BLOCK_LEN,
    RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
};
pub use recursion_shape::{
    build_rv32im_recursion_shape, build_rv32im_recursion_shape_for_step_cap, ProtocolVersion, RecursionShape,
    ShapeError,
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
