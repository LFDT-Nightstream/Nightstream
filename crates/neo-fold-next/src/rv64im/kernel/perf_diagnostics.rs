//! Owns compact performance-diagnostics records for the RV64IM kernel path.

use crate::proof::{RunProvePerf, RunVerifyPerf};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactStageVectorBuildPerf {
    pub flatten_u64_words: usize,
    pub field_limb_width: usize,
    pub packed_rows: usize,
    pub packed_cols: usize,
    pub flatten_ms: f64,
    pub limb_encode_ms: f64,
    pub context_setup_ms: f64,
    pub ccs_encode_ms: f64,
    pub ajtai_commit_ms: f64,
    pub opening_manifest_ms: f64,
    pub opening_prove_ms: f64,
}

impl ExactStageVectorBuildPerf {
    pub fn total_ms(&self) -> f64 {
        self.flatten_ms
            + self.limb_encode_ms
            + self.context_setup_ms
            + self.ccs_encode_ms
            + self.ajtai_commit_ms
            + self.opening_manifest_ms
            + self.opening_prove_ms
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StageClaimBundleBuildPerf {
    pub stage1: ExactStageVectorBuildPerf,
    pub stage2: ExactStageVectorBuildPerf,
    pub stage3: ExactStageVectorBuildPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PackagedOpeningBuildPerf {
    pub selected_labels: usize,
    pub claim_words: usize,
    pub package_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StagePackageBundleBuildPerf {
    pub stage1: PackagedOpeningBuildPerf,
    pub stage2: PackagedOpeningBuildPerf,
    pub stage3: PackagedOpeningBuildPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOpeningBundleBuildPerf {
    pub bindings: PackagedOpeningBuildPerf,
    pub prepared_steps: PackagedOpeningBuildPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimpleKernelBuildPerf {
    pub root_lane_witness_ms: f64,
    pub root_lane_columns_ms: f64,
    pub root_lane_commitment_ms: f64,
    pub public_steps_ms: f64,
    pub prepared_steps_ms: f64,
    pub prepared_step_bindings_ms: f64,
    pub stage_claim_bundle: StageClaimBundleBuildPerf,
    pub stage_package_bundle: StagePackageBundleBuildPerf,
    pub kernel_opening_bundle: KernelOpeningBundleBuildPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StagePackageBundleVerifyPerf {
    pub stage1_ms: f64,
    pub stage2_ms: f64,
    pub stage3_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptedStage2VerifyPerf {
    pub semantics_ms: f64,
    pub temporal_ms: f64,
    pub family_digests_ms: f64,
    pub selected_opening_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptedStage1VerifyPerf {
    pub sem_inputs_surface_ms: f64,
    pub semantics_verify_ms: f64,
    pub row_bindings_surface_ms: f64,
    pub surface_digest_checks_ms: f64,
    pub selected_opening_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptedStagePackageVerifyPerf {
    pub stage1_ms: f64,
    pub stage1_breakdown: AcceptedStage1VerifyPerf,
    pub stage2_ms: f64,
    pub stage2_breakdown: AcceptedStage2VerifyPerf,
    pub stage3_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptedRootExecutionVerifyPerf {
    pub preflight_ms: f64,
    pub semantic_rows_ms: f64,
    pub prepared_step_bindings_ms: f64,
    pub kernel_claim_bindings_ms: f64,
    pub row_chunk_routes_ms: f64,
    pub row_local_ccs_acceptance_ms: f64,
    pub semantics_refinement_ms: f64,
    pub statement_chunk_layout_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOpeningBundleVerifyPerf {
    pub claim_rebuild_ms: f64,
    pub bindings_ms: f64,
    pub prepared_steps_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimpleKernelVerifyPerf {
    pub expected_core_ms: f64,
    pub trace_match_ms: f64,
    pub stages_match_ms: f64,
    pub stage_claims_match_ms: f64,
    pub kernel_claims_match_ms: f64,
    pub root_lane_columns_match_ms: f64,
    pub root_lane_commitment_match_ms: f64,
    pub stage_package_bundle: StagePackageBundleVerifyPerf,
    pub kernel_opening_bundle: KernelOpeningBundleVerifyPerf,
    pub total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PackagedSimpleKernelVerifyPerf {
    pub simple_kernel: SimpleKernelVerifyPerf,
    pub main_lane_artifact_match_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imProofProvePerf {
    pub shared_trace_ms: f64,
    pub simple_kernel: SimpleKernelBuildPerf,
    pub parallel_overlap_ms: f64,
    pub main_lane_ms: f64,
    pub root_main_lane: RootMainLanePackagedProofProvePerf,
    pub public_export_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Rv64imPublicProofVerifyPerf {
    pub public_claim_digests_ms: f64,
    pub public_bundle_digests_ms: f64,
    pub public_bundle_bindings_ms: f64,
    pub native_stage_bundle_verify_ms: f64,
    pub public_kernel_build: SimpleKernelBuildPerf,
    pub root_execution_verify_ms: f64,
    pub root_main_lane_proof_ms: f64,
    pub root_main_lane: RootMainLanePackagedProofVerifyPerf,
    pub stage_package_verify_ms: f64,
    pub accepted_stage_package: AcceptedStagePackageVerifyPerf,
    pub accepted_root_execution: AcceptedRootExecutionVerifyPerf,
    pub kernel_opening_verify_ms: f64,
    pub summary_consistency_ms: f64,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RootMainLanePackagedProofProvePerf {
    pub prepare_steps_ms: f64,
    pub session: RunProvePerf,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RootMainLaneRunProofProvePerf {
    pub prepare_steps_ms: f64,
    pub session: RunProvePerf,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RootMainLanePackagedProofVerifyPerf {
    pub prepare_public_steps_ms: f64,
    pub public_chunk_match_ms: f64,
    pub packaged_statement_digest_ms: f64,
    pub packaged_chunk_digests_ms: f64,
    pub packaged_final_main_claim_digests_ms: f64,
    pub packaged_statement_hash_ms: f64,
    pub packaged_schedule_checks_ms: f64,
    pub packaged_proof_digest_ms: f64,
    pub packaged_final_claim_match_ms: f64,
    pub packaged_total_ms: f64,
    pub session: RunVerifyPerf,
    pub total_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RootMainLaneRunProofVerifyPerf {
    pub prepare_public_steps_ms: f64,
    pub session: RunVerifyPerf,
    pub total_ms: f64,
}
