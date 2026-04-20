use crate::finalize::package_session_proof;
use crate::proof::{
    partition_public_steps, Carry, ChunkInput, FoldSchedule, PackagedProof, PublicChunk, PublicStep, RunProof,
    RunProvePerf, RunVerifyPerf, StepInput,
};
use crate::prover::{CommitmentMixers, ShardProver};
use crate::run::{
    prove_chunks_from_slice_with_perf_and_cache, verify_packaged_with_detailed_perf_and_cache,
    verify_packaged_with_precomputed_chunk_digests_and_detailed_perf_and_cache,
};
use crate::rv64im::ccs::{rv64im_root_main_lane_ccs, semantic_row_from_execution_row, RV64IM_ROOT_ROW_WIDTH};
use crate::rv64im::isa::Rv64BuildError;
use crate::rv64im::lower::Rv64ExpandedRow;
use crate::rv64im::stage1::Stage1Summary;
use crate::rv64im::stage2::Stage2Summary;
use crate::rv64im::stage3::Stage3Summary;
use crate::verifier::ShardVerifier;
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};
use neo_ajtai::{s_mul_add_from_rot_col, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::{traits::SModuleHomomorphism, CcsStructure, Mat};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{ops::Deref, sync::OnceLock, time::Instant};

use super::{
    build_parity_case_from_source,
    main_lane_artifact::{
        build_simple_kernel_main_lane_artifact, validate_simple_kernel_main_lane_artifact, SimpleKernelMainLaneArtifact,
    },
    perf_diagnostics::{
        PackagedSimpleKernelVerifyPerf, RootMainLanePackagedProofProvePerf, RootMainLanePackagedProofVerifyPerf,
        RootMainLaneRunProofProvePerf, RootMainLaneRunProofVerifyPerf, Rv64imProofProvePerf, SimpleKernelBuildPerf,
        SimpleKernelVerifyPerf,
    },
    proof_witness::{
        stage_witness_projection_bundle_from_summaries, trace_projection_bundle_from_rows,
        Rv64imStageWitnessProjectionBundle, Rv64imTraceProjectionBundle,
    },
    root_lane_columns::{build_root_lane_columns_from_public_witness, build_root_lane_columns_from_witness},
    root_lane_commitment::{
        build_root_lane_commitment_artifact_from_witness,
        build_root_lane_commitment_summary_artifact_from_public_witness,
    },
    root_lane_witness::{
        build_root_lane_witness, next_power_of_two_len, root_lane_column_digest, root_lane_family_digest,
        root_lane_row_digest, RootLanePublicWitness, RootLaneWitness,
    },
    simple_openings::{SimpleKernelOpeningBundle, SimpleKernelStagePackageBundle},
    stage_artifacts::{
        build_kernel_opening_bundle_with_perf, build_public_kernel_opening_bundle_with_perf,
        build_stage_claim_bundle_from_parts, build_stage_claim_bundle_from_parts_with_perf,
        verify_kernel_opening_bundle_with_perf, SimpleKernelStageClaimBundle,
    },
    stage_package_perf::{
        build_public_stage_package_bundle_with_perf, build_stage_package_bundle_with_perf,
        verify_stage_package_bundle_with_perf,
    },
    RootLaneColumns, RootLaneCommitmentArtifact, RootLaneCommitmentSummaryArtifact, Rv64imKernelSummary,
    Rv64imParityCaseManifest, Rv64imParityDerivedCase, Rv64imParitySourceCase, TranscriptRecord,
};

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn allow_parallel_step_build(count: usize) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads() > 1 && rayon::current_thread_index().is_none() && count >= 8
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = count;
        false
    }
}

pub(super) const SIMPLE_KERNEL_PP_SEED: [u8; 32] = [
    0x40, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
// Keep the live RV64IM family on the conservative fixed parameter while the
// accepted-artifact bound is still being characterized.
pub(super) const SIMPLE_KERNEL_K_RHO: u32 = 48;
pub(super) const SIMPLE_KERNEL_B: u64 = 1u64 << SIMPLE_KERNEL_K_RHO;
// Ajtai public parameters are global per dimension bucket, so exact stage surfaces share one seed.
pub(super) const EXACT_STAGE_PP_SEED: [u8; 32] = SIMPLE_KERNEL_PP_SEED;
const ROOT_MAIN_LANE_STEP_LABEL: &str = "";

pub fn rv64im_simple_kernel_pp_seed() -> [u8; 32] {
    SIMPLE_KERNEL_PP_SEED
}

pub fn rv64im_exact_stage_pp_seed() -> [u8; 32] {
    EXACT_STAGE_PP_SEED
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelPublicInput {
    pub source: Rv64imParitySourceCase,
    pub max_steps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelProverInput {
    pub public: SimpleKernelPublicInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelVerifierInput {
    pub public: SimpleKernelPublicInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedStepBinding {
    pub trace_index: usize,
    pub row_digest: [u8; 32],
    pub row_opening_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedStepBindingSummary {
    pub bindings: Vec<PreparedStepBinding>,
    pub binding_count: u64,
    pub first_binding_digest: Option<[u8; 32]>,
    pub last_binding_digest: Option<[u8; 32]>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelTraceWitness {
    pub manifest: Rv64imParityCaseManifest,
    pub execution_rows: Vec<Rv64ExpandedRow>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelStageWitnessBundle {
    pub stage1: Stage1Summary,
    pub stage2: Stage2Summary,
    pub stage3: Stage3Summary,
    pub transcript: TranscriptRecord,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimpleKernelKernelClaimBundle {
    pub kernel: Rv64imKernelSummary,
    pub prepared_step_bindings: PreparedStepBindingSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelOutput {
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelAuditOutput {
    pub kernel: SimpleKernelOutput,
    pub prepared_steps: Vec<StepInput>,
}

impl Deref for SimpleKernelAuditOutput {
    type Target = SimpleKernelOutput;

    fn deref(&self) -> &Self::Target {
        &self.kernel
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PublicSimpleKernelOutput {
    pub trace: Rv64imTraceProjectionBundle,
    pub stages: Rv64imStageWitnessProjectionBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelProof {
    pub root_params_id: [u8; 32],
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
    pub stage_claims: SimpleKernelStageClaimBundle,
    pub stage_packages: SimpleKernelStagePackageBundle,
    pub kernel_opening: SimpleKernelOpeningBundle,
    pub kernel_claims: SimpleKernelKernelClaimBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentArtifact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleKernelPackagedProof {
    pub kernel: SimpleKernelProof,
    pub main_lane: SimpleKernelMainLaneArtifact,
}

#[derive(Debug)]
pub enum SimpleKernelError {
    Build(String),
    Bridge(String),
    Proof(String),
}

impl core::fmt::Display for SimpleKernelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Build(s) => write!(f, "build failed: {s}"),
            Self::Bridge(s) => write!(f, "bridge failed: {s}"),
            Self::Proof(s) => write!(f, "proof failed: {s}"),
        }
    }
}

impl std::error::Error for SimpleKernelError {}

impl From<Rv64BuildError> for SimpleKernelError {
    fn from(value: Rv64BuildError) -> Self {
        Self::Build(value.to_string())
    }
}

impl From<PiCcsError> for SimpleKernelError {
    fn from(value: PiCcsError) -> Self {
        Self::Proof(value.to_string())
    }
}

struct SimpleKernelRootContext {
    params: NeoParams,
    log: AjtaiSModule,
}

pub(super) struct SimpleKernelExpectedSeed {
    trace: SimpleKernelTraceWitness,
    stages: SimpleKernelStageWitnessBundle,
    stage_claims: SimpleKernelStageClaimBundle,
    kernel_claims: SimpleKernelKernelClaimBundle,
    root_lane_columns: RootLaneColumns,
    root_lane_commitment: RootLaneCommitmentArtifact,
    root_lane_witness: RootLaneWitness,
}

struct SimpleKernelBuildSeed {
    trace: SimpleKernelTraceWitness,
    stages: SimpleKernelStageWitnessBundle,
    stage_claims: SimpleKernelStageClaimBundle,
    stage_packages: SimpleKernelStagePackageBundle,
    kernel_opening: SimpleKernelOpeningBundle,
    kernel_claims: SimpleKernelKernelClaimBundle,
    root_lane_columns: RootLaneColumns,
    root_lane_commitment: RootLaneCommitmentArtifact,
}

struct PublicSimpleKernelBuildSeed {
    trace: Rv64imTraceProjectionBundle,
    stages: Rv64imStageWitnessProjectionBundle,
    stage_claims: SimpleKernelStageClaimBundle,
    stage_packages: SimpleKernelStagePackageBundle,
    kernel_opening: SimpleKernelOpeningBundle,
    kernel_claims: SimpleKernelKernelClaimBundle,
    root_lane_columns: RootLaneColumns,
    root_lane_commitment: RootLaneCommitmentSummaryArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct PublicSimpleKernelWitnessSidecar {
    pub trace: SimpleKernelTraceWitness,
    pub stages: SimpleKernelStageWitnessBundle,
}

pub(crate) fn selected_opening_ref_digest(
    object_digest: [u8; 32],
    logical_index: u64,
    value_digest: [u8; 32],
) -> [u8; 32] {
    let mut opening_id = Poseidon2Transcript::new(b"neo.fold.next/rv64im/ajtai_opening_id");
    opening_id.append_message(b"rv64im/ajtai_opening_id/object_digest", &object_digest);
    opening_id.append_u64s(b"rv64im/ajtai_opening_id/logical_index", &[logical_index]);
    let opening_id_digest = opening_id.digest32();

    let mut selected_opening = Poseidon2Transcript::new(b"neo.fold.next/rv64im/selected_opening_ref");
    selected_opening.append_message(b"rv64im/selected_opening_ref/opening_id", &opening_id_digest);
    selected_opening.append_message(b"rv64im/selected_opening_ref/value_digest", &value_digest);
    selected_opening.digest32()
}

pub(crate) fn prepared_step_binding_digest(
    logical_index: usize,
    trace_index: usize,
    semantic_row: &[F; RV64IM_ROOT_ROW_WIDTH],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/prepared_step_binding");
    tr.append_u64s(
        b"rv64im/prepared_step_binding/meta",
        &[logical_index as u64, trace_index as u64],
    );
    tr.append_fields(b"rv64im/prepared_step_binding/semantic_row", semantic_row);
    tr.digest32()
}

impl SimpleKernelRootContext {
    fn new() -> Result<Self, SimpleKernelError> {
        let params = rv64im_simple_root_params();
        let m = commit_cols_for_full_width(RV64IM_ROOT_ROW_WIDTH);
        set_global_pp_seeded(D, params.kappa as usize, m, SIMPLE_KERNEL_PP_SEED)
            .map_err(|err| SimpleKernelError::Bridge(format!("canonical RV64IM root seed setup failed: {err}")))?;
        let log = AjtaiSModule::from_global_for_dims(D, m)
            .map_err(|err| SimpleKernelError::Bridge(format!("canonical RV64IM root module failed: {err}")))?;
        Ok(Self { params, log })
    }

    fn params(&self) -> &NeoParams {
        &self.params
    }

    fn log(&self) -> &AjtaiSModule {
        &self.log
    }
}

fn cached_simple_kernel_root_context() -> Result<&'static SimpleKernelRootContext, SimpleKernelError> {
    static ROOT_CONTEXT: OnceLock<Result<SimpleKernelRootContext, String>> = OnceLock::new();
    ROOT_CONTEXT
        .get_or_init(|| {
            SimpleKernelRootContext::new().map_err(|err| match err {
                SimpleKernelError::Build(msg) | SimpleKernelError::Bridge(msg) | SimpleKernelError::Proof(msg) => msg,
            })
        })
        .as_ref()
        .map_err(|err| SimpleKernelError::Bridge(err.clone()))
}

fn cached_root_main_lane_ccs() -> Result<&'static CcsStructure<F>, SimpleKernelError> {
    static ROOT_MAIN_LANE_CCS: OnceLock<Result<CcsStructure<F>, String>> = OnceLock::new();
    ROOT_MAIN_LANE_CCS
        .get_or_init(rv64im_root_main_lane_ccs)
        .as_ref()
        .map_err(|err| SimpleKernelError::Proof(err.clone()))
}

fn cached_root_main_lane_optimized_cache() -> Result<&'static OptimizedStructureCache, SimpleKernelError> {
    static ROOT_MAIN_LANE_OPTIMIZED_CACHE: OnceLock<Result<OptimizedStructureCache, String>> = OnceLock::new();
    let ccs = cached_root_main_lane_ccs()?;
    ROOT_MAIN_LANE_OPTIMIZED_CACHE
        .get_or_init(|| OptimizedStructureCache::build(ccs).map_err(|err| err.to_string()))
        .as_ref()
        .map_err(|err| SimpleKernelError::Proof(err.clone()))
}

pub(crate) fn rv64im_cached_root_main_lane_context(
) -> Result<(&'static NeoParams, &'static AjtaiSModule, &'static CcsStructure<F>), SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    Ok((root_context.params(), root_context.log(), ccs))
}

pub(crate) fn rv64im_cached_root_main_lane_optimized_cache(
) -> Result<&'static OptimizedStructureCache, SimpleKernelError> {
    cached_root_main_lane_optimized_cache()
}

pub fn rv64im_simple_root_params() -> NeoParams {
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(RV64IM_ROOT_ROW_WIDTH).expect("valid RV64IM root params");
    params.k_rho = SIMPLE_KERNEL_K_RHO;
    params.B = SIMPLE_KERNEL_B;
    params
}

pub fn rv64im_simple_root_context_id() -> [u8; 32] {
    let params = rv64im_simple_root_params();
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/root_context");
    tr.append_u64s(
        b"rv64im/root_context/values",
        &[
            params.q,
            params.eta as u64,
            params.d as u64,
            params.kappa as u64,
            params.m,
            params.b as u64,
            params.k_rho as u64,
            params.B,
            params.T as u64,
            params.s as u64,
            params.lambda as u64,
            RV64IM_ROOT_ROW_WIDTH as u64,
            commit_cols_for_full_width(RV64IM_ROOT_ROW_WIDTH) as u64,
        ],
    );
    tr.append_message(b"rv64im/root_context/seed", &SIMPLE_KERNEL_PP_SEED);
    tr.digest32()
}

fn root_encode_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV64IM_ROOT_ROW_WIDTH],
) -> Result<(Vec<F>, Mat<F>), SimpleKernelError> {
    let witness = semantic_row[1..].to_vec();
    let packed = encode_vector_for_full_width(root_context.params(), RV64IM_ROOT_ROW_WIDTH, semantic_row)
        .map_err(|err| SimpleKernelError::Bridge(format!("root encoding failed for row {trace_index}: {err}")))?;
    Ok((witness, packed))
}

fn build_prepared_step_from_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV64IM_ROOT_ROW_WIDTH],
) -> Result<StepInput, SimpleKernelError> {
    let (witness, z_mat) = root_encode_semantic_row(root_context, trace_index, semantic_row)?;
    Ok(StepInput {
        // Root main-lane position is already bound by chunk ordering, so per-row labels only bloat traces.
        label: ROOT_MAIN_LANE_STEP_LABEL.into(),
        mcs: neo_ccs::CcsClaim {
            c: root_context.log().commit(&z_mat),
            x: vec![F::ONE],
            m_in: 1,
        },
        witness: neo_ccs::CcsWitness { w: witness, Z: z_mat },
    })
}

fn build_public_step_from_semantic_row(
    root_context: &SimpleKernelRootContext,
    trace_index: usize,
    semantic_row: &[F; RV64IM_ROOT_ROW_WIDTH],
) -> Result<PublicStep, SimpleKernelError> {
    let z_mat = encode_vector_for_full_width(root_context.params(), RV64IM_ROOT_ROW_WIDTH, semantic_row)
        .map_err(|err| SimpleKernelError::Bridge(format!("root encoding failed for row {trace_index}: {err}")))?;
    Ok(PublicStep {
        label: ROOT_MAIN_LANE_STEP_LABEL.into(),
        mcs: neo_ccs::CcsClaim {
            c: root_context.log().commit(&z_mat),
            x: vec![F::ONE],
            m_in: 1,
        },
    })
}

fn build_prepared_steps_from_root_lane_witness(
    root_context: &SimpleKernelRootContext,
    rows: &[Rv64ExpandedRow],
    root_lane_witness: &RootLaneWitness,
) -> Result<Vec<StepInput>, SimpleKernelError> {
    if rows.len() != root_lane_witness.semantic_rows.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "root lane semantic row count {} != execution row count {}",
            root_lane_witness.semantic_rows.len(),
            rows.len()
        )));
    }
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .zip(root_lane_witness.semantic_rows.par_iter())
            .map(|(row, semantic_row)| {
                build_prepared_step_from_semantic_row(root_context, row.trace_index, semantic_row)
            })
            .collect();
    }
    rows.iter()
        .zip(root_lane_witness.semantic_rows.iter())
        .map(|(row, semantic_row)| build_prepared_step_from_semantic_row(root_context, row.trace_index, semantic_row))
        .collect()
}

pub(super) fn build_prepared_steps_from_execution_rows(
    rows: &[Rv64ExpandedRow],
) -> Result<Vec<StepInput>, SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .map(|row| {
                let semantic_row = semantic_row_from_execution_row(row);
                build_prepared_step_from_semantic_row(root_context, row.trace_index, &semantic_row)
            })
            .collect();
    }
    let mut steps = Vec::with_capacity(rows.len());
    for row in rows {
        let semantic_row = semantic_row_from_execution_row(row);
        steps.push(build_prepared_step_from_semantic_row(
            root_context,
            row.trace_index,
            &semantic_row,
        )?);
    }
    Ok(steps)
}

pub(super) fn build_public_steps_from_execution_rows(
    rows: &[Rv64ExpandedRow],
) -> Result<Vec<PublicStep>, SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    if allow_parallel_step_build(rows.len()) {
        return rows
            .par_iter()
            .map(|row| {
                let semantic_row = semantic_row_from_execution_row(row);
                build_public_step_from_semantic_row(root_context, row.trace_index, &semantic_row)
            })
            .collect();
    }
    let mut steps = Vec::with_capacity(rows.len());
    for row in rows {
        let semantic_row = semantic_row_from_execution_row(row);
        steps.push(build_public_step_from_semantic_row(
            root_context,
            row.trace_index,
            &semantic_row,
        )?);
    }
    Ok(steps)
}

fn same_public_step(lhs: &PublicStep, rhs: &PublicStep) -> bool {
    lhs.label == rhs.label
        && lhs.mcs.m_in == rhs.mcs.m_in
        && lhs.mcs.x == rhs.mcs.x
        && lhs.mcs.c.d == rhs.mcs.c.d
        && lhs.mcs.c.kappa == rhs.mcs.c.kappa
        && lhs.mcs.c.data == rhs.mcs.c.data
}

fn same_public_chunk(lhs: &PublicChunk, rhs: &PublicChunk) -> bool {
    lhs.start_index == rhs.start_index
        && lhs.steps.len() == rhs.steps.len()
        && lhs
            .steps
            .iter()
            .zip(rhs.steps.iter())
            .all(|(lhs, rhs)| same_public_step(lhs, rhs))
}

fn root_main_lane_packaged_verify_perf(
    prepare_public_steps_ms: f64,
    public_chunk_match_ms: f64,
    packaged_verify: crate::finalize::PackagedVerifyPerf,
    total_ms: f64,
) -> RootMainLanePackagedProofVerifyPerf {
    RootMainLanePackagedProofVerifyPerf {
        prepare_public_steps_ms,
        public_chunk_match_ms,
        packaged_statement_digest_ms: packaged_verify.statement_digest_ms,
        packaged_chunk_digests_ms: packaged_verify.chunk_digests_ms,
        packaged_final_main_claim_digests_ms: packaged_verify.final_main_claim_digests_ms,
        packaged_statement_hash_ms: packaged_verify.statement_hash_ms,
        packaged_schedule_checks_ms: packaged_verify.schedule_checks_ms,
        packaged_proof_digest_ms: packaged_verify.proof_digest_ms,
        packaged_final_claim_match_ms: packaged_verify.final_claim_match_ms,
        packaged_total_ms: packaged_verify.total_ms,
        session: packaged_verify.session,
        total_ms,
    }
}

fn root_main_lane_chunk_len(schedule: FoldSchedule, row_count: usize) -> Result<usize, SimpleKernelError> {
    schedule.validate()?;
    Ok(match schedule {
        FoldSchedule::WholeTrace => row_count.max(1),
        FoldSchedule::RowsPerChunk(rows) => rows,
    })
}

pub fn prove_root_main_lane_packaged_proof_with_perf(
    rows: &[Rv64ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(PackagedProof, RootMainLanePackagedProofProvePerf), SimpleKernelError> {
    let (packaged, _, perf) = prove_root_main_lane_packaged_proof_with_inputs_and_perf(rows, schedule)?;
    Ok((packaged, perf))
}

pub(crate) fn prove_root_main_lane_packaged_proof_with_inputs_and_perf(
    rows: &[Rv64ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(PackagedProof, Vec<ChunkInput>, RootMainLanePackagedProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let prepare_steps_started = Instant::now();
    let steps = build_prepared_steps_from_execution_rows(rows)?;
    let prepare_steps_ms = millis_since(prepare_steps_started);
    let chunk_inputs = crate::proof::partition_step_inputs(schedule, steps)?;
    let public_chunks = chunk_inputs
        .iter()
        .map(ChunkInput::public)
        .collect::<Vec<_>>();
    let (session, session_perf) = prove_chunks_from_slice_with_perf_and_cache(
        FoldingMode::Optimized,
        schedule,
        root_context.params(),
        ccs,
        &chunk_inputs,
        root_context.log(),
        rv64im_ajtai_mixers(),
        None,
    )?;
    let packaged = package_session_proof(public_chunks, session)?;
    Ok((
        packaged,
        chunk_inputs,
        RootMainLanePackagedProofProvePerf {
            prepare_steps_ms,
            session: session_perf,
            total_ms: millis_since(total_started),
        },
    ))
}

pub fn prove_root_main_lane_run_proof_with_perf(
    rows: &[Rv64ExpandedRow],
    schedule: FoldSchedule,
) -> Result<(RunProof, RootMainLaneRunProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let chunk_len = root_main_lane_chunk_len(schedule, rows.len())?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = Carry::default();
    let mut proof = RunProof {
        fold_schedule: schedule,
        ..RunProof::default()
    };
    let mut session = RunProvePerf::default();
    let mut prepare_steps_ms = 0.0;
    let mut start_index = 0usize;
    while start_index < rows.len() {
        let end_index = (start_index + chunk_len).min(rows.len());
        let prepare_steps_started = Instant::now();
        let steps = build_prepared_steps_from_execution_rows(&rows[start_index..end_index])?;
        prepare_steps_ms += millis_since(prepare_steps_started);
        let chunk = ChunkInput { start_index, steps };
        let (proved, chunk_perf) = ShardProver::prove_chunk_with_perf(
            FoldingMode::Optimized,
            &mut tr,
            root_context.params(),
            ccs,
            &chunk,
            &main_carry,
            root_context.log(),
            rv64im_ajtai_mixers(),
            Some(&optimized_cache),
        )?;
        main_carry = proved.next_main;
        proof.chunks.push(proved.proof);
        session.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
        start_index = end_index;
    }
    proof.final_main_claims = main_carry.claims;
    session.total_ms = millis_since(total_started);
    Ok((
        proof,
        RootMainLaneRunProofProvePerf {
            prepare_steps_ms,
            session,
            total_ms: millis_since(total_started),
        },
    ))
}

pub fn verify_root_main_lane_packaged_proof_with_public_rows(
    rows: &[Rv64ExpandedRow],
    packaged: &PackagedProof,
) -> Result<RootMainLanePackagedProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let prepare_public_steps_started = Instant::now();
    let public_steps = build_public_steps_from_execution_rows(rows)?;
    let prepare_public_steps_ms = millis_since(prepare_public_steps_started);
    let public_chunk_match_started = Instant::now();
    let expected_chunks = partition_public_steps(packaged.statement.fold_schedule, public_steps)?;
    if packaged.statement.chunks.len() != expected_chunks.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM root main-lane packaged proof chunk count {} != expected chunk count {}",
            packaged.statement.chunks.len(),
            expected_chunks.len()
        )));
    }
    for (idx, (actual, expected)) in packaged
        .statement
        .chunks
        .iter()
        .zip(expected_chunks.iter())
        .enumerate()
    {
        if !same_public_chunk(actual, expected) {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM root main-lane packaged proof public chunk {idx} mismatch"
            )));
        }
    }
    let public_chunk_match_ms = millis_since(public_chunk_match_started);
    let (_, packaged_verify) = verify_packaged_with_detailed_perf_and_cache(
        FoldingMode::Optimized,
        root_context.params(),
        ccs,
        packaged,
        rv64im_ajtai_mixers(),
        None,
    )?;
    Ok(root_main_lane_packaged_verify_perf(
        prepare_public_steps_ms,
        public_chunk_match_ms,
        packaged_verify,
        millis_since(total_started),
    ))
}

pub(super) fn verify_root_main_lane_packaged_proof_with_verified_public_statement_with_perf(
    packaged: &PackagedProof,
    public_chunk_digests: &[[F; 4]],
) -> Result<RootMainLanePackagedProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let (_, packaged_verify) = verify_packaged_with_precomputed_chunk_digests_and_detailed_perf_and_cache(
        FoldingMode::Optimized,
        root_context.params(),
        ccs,
        packaged,
        public_chunk_digests,
        rv64im_ajtai_mixers(),
        Some(optimized_cache),
    )?;
    Ok(root_main_lane_packaged_verify_perf(
        0.0,
        0.0,
        packaged_verify,
        millis_since(total_started),
    ))
}

pub fn verify_root_main_lane_run_proof_with_public_rows(
    rows: &[Rv64ExpandedRow],
    proof: &RunProof,
) -> Result<RootMainLaneRunProofVerifyPerf, SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    let chunk_len = root_main_lane_chunk_len(proof.fold_schedule, rows.len())?;
    let optimized_cache = cached_root_main_lane_optimized_cache()?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/session");
    let mut main_carry = &[][..];
    let mut session = RunVerifyPerf::default();
    let mut prepare_public_steps_ms = 0.0;
    let mut start_index = 0usize;
    for (chunk_index, chunk_proof) in proof.chunks.iter().enumerate() {
        let end_index = (start_index + chunk_len).min(rows.len());
        let prepare_public_steps_started = Instant::now();
        let steps = build_public_steps_from_execution_rows(&rows[start_index..end_index])?;
        prepare_public_steps_ms += millis_since(prepare_public_steps_started);
        let chunk = PublicChunk { start_index, steps };
        let (next_main, chunk_perf) = ShardVerifier::verify_chunk_with_perf(
            FoldingMode::Optimized,
            &mut tr,
            root_context.params(),
            ccs,
            &chunk,
            main_carry,
            chunk_proof,
            rv64im_ajtai_mixers(),
            Some(&optimized_cache),
        )?;
        main_carry = next_main;
        session.chunks.push(chunk_perf);
        tr.append_message(b"neo.fold.next/chunk_done", &[1]);
        start_index = end_index;
        if chunk_index + 1 == proof.chunks.len() && start_index != rows.len() {
            return Err(SimpleKernelError::Bridge(
                "RV64IM root main-lane run proof ended before covering all rows".into(),
            ));
        }
    }
    if start_index != rows.len() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root main-lane run proof chunk count does not cover the provided rows".into(),
        ));
    }
    if main_carry != proof.final_main_claims.as_slice() {
        return Err(SimpleKernelError::Proof(
            "RV64IM root main-lane run proof final carried claims mismatch".into(),
        ));
    }
    session.total_ms = millis_since(total_started);
    Ok(RootMainLaneRunProofVerifyPerf {
        prepare_public_steps_ms,
        session,
        total_ms: millis_since(total_started),
    })
}

fn build_prepared_step_binding_summary_from_trace_row_digests(
    rows: &[Rv64ExpandedRow],
    semantic_rows: &[[F; RV64IM_ROOT_ROW_WIDTH]],
    root_lane_columns: &RootLaneColumns,
    materialize_bindings: bool,
) -> Result<PreparedStepBindingSummary, SimpleKernelError> {
    if rows.len() != semantic_rows.len() {
        return Err(SimpleKernelError::Bridge(format!(
            "prepared step row count {} != semantic row count {}",
            rows.len(),
            semantic_rows.len(),
        )));
    }

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/prepared_step_binding_summary");
    tr.append_u64s(b"rv64im/prepared_step_binding_summary/len", &[rows.len() as u64]);
    let mut bindings = if materialize_bindings {
        Vec::with_capacity(rows.len())
    } else {
        Vec::new()
    };
    let mut first_binding_digest = None;
    let mut last_binding_digest = None;
    for (logical_index, (row, semantic_row)) in rows.iter().zip(semantic_rows.iter()).enumerate() {
        let row_digest = root_lane_row_digest(logical_index as u64, semantic_row);
        let binding_digest = prepared_step_binding_digest(logical_index, row.trace_index, semantic_row);
        if first_binding_digest.is_none() {
            first_binding_digest = Some(binding_digest);
        }
        last_binding_digest = Some(binding_digest);
        tr.append_message(b"rv64im/prepared_step_binding_summary/binding_digest", &binding_digest);
        if materialize_bindings {
            let row_opening_digest =
                selected_opening_ref_digest(root_lane_columns.object.digest, logical_index as u64, row_digest);
            bindings.push(PreparedStepBinding {
                trace_index: row.trace_index,
                row_digest,
                row_opening_digest,
                digest: binding_digest,
            });
        }
    }
    Ok(PreparedStepBindingSummary {
        bindings,
        binding_count: rows.len() as u64,
        first_binding_digest,
        last_binding_digest,
        digest: tr.digest32(),
    })
}

pub(crate) fn build_prepared_step_binding_summary(
    rows: &[Rv64ExpandedRow],
    semantic_rows: &[[F; RV64IM_ROOT_ROW_WIDTH]],
    root_lane_columns: &RootLaneColumns,
    materialize_bindings: bool,
) -> Result<PreparedStepBindingSummary, SimpleKernelError> {
    build_prepared_step_binding_summary_from_trace_row_digests(
        rows,
        semantic_rows,
        root_lane_columns,
        materialize_bindings,
    )
}

pub(super) fn build_public_root_lane_witness_and_binding_summary(
    rows: &[Rv64ExpandedRow],
) -> (RootLanePublicWitness, PreparedStepBindingSummary) {
    let time_len = rows.len();
    let padded_time_len = next_power_of_two_len(time_len);
    let mut columns = (0..RV64IM_ROOT_ROW_WIDTH)
        .map(|_| Vec::with_capacity(time_len))
        .collect::<Vec<_>>();
    let mut binding_tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/prepared_step_binding_summary");
    binding_tr.append_u64s(b"rv64im/prepared_step_binding_summary/len", &[time_len as u64]);

    let mut first_row_digest = None;
    let mut last_row_digest = None;
    let mut first_binding_digest = None;
    let mut last_binding_digest = None;
    for (logical_index, row) in rows.iter().enumerate() {
        let semantic_row = semantic_row_from_execution_row(row);
        let row_digest = root_lane_row_digest(logical_index as u64, &semantic_row);
        if logical_index == 0 {
            first_row_digest = Some(row_digest);
        }
        if logical_index + 1 == time_len {
            last_row_digest = Some(row_digest);
        }
        let binding_digest = prepared_step_binding_digest(logical_index, row.trace_index, &semantic_row);
        if first_binding_digest.is_none() {
            first_binding_digest = Some(binding_digest);
        }
        last_binding_digest = Some(binding_digest);
        binding_tr.append_message(b"rv64im/prepared_step_binding_summary/binding_digest", &binding_digest);
        for (column_index, value) in semantic_row.iter().enumerate() {
            columns[column_index].push(*value);
        }
    }

    let column_digests = columns
        .iter()
        .enumerate()
        .map(|(column_index, values)| root_lane_column_digest(column_index as u64, values))
        .collect::<Vec<_>>();
    let family_digest = root_lane_family_digest(&column_digests);

    (
        RootLanePublicWitness {
            columns,
            time_len,
            padded_time_len,
            first_row_digest,
            last_row_digest,
            column_digests,
            family_digest,
        },
        PreparedStepBindingSummary {
            bindings: Vec::new(),
            binding_count: time_len as u64,
            first_binding_digest,
            last_binding_digest,
            digest: binding_tr.digest32(),
        },
    )
}

fn trace_witness_from_derived(derived: &Rv64imParityDerivedCase) -> SimpleKernelTraceWitness {
    SimpleKernelTraceWitness {
        manifest: derived.manifest.clone(),
        execution_rows: derived.execution_rows.clone(),
    }
}

fn stage_witness_bundle_from_derived(derived: &Rv64imParityDerivedCase) -> SimpleKernelStageWitnessBundle {
    SimpleKernelStageWitnessBundle {
        stage1: derived.stage1.clone(),
        stage2: derived.stage2.clone(),
        stage3: derived.stage3.clone(),
        transcript: derived.transcript.clone(),
    }
}

fn kernel_claim_bundle_from_parts(
    derived: &Rv64imParityDerivedCase,
    prepared_step_bindings: PreparedStepBindingSummary,
) -> SimpleKernelKernelClaimBundle {
    SimpleKernelKernelClaimBundle {
        kernel: derived.kernel.clone(),
        prepared_step_bindings,
    }
}

fn build_simple_kernel_expected_seed(
    public: &SimpleKernelPublicInput,
    materialize_bindings: bool,
) -> Result<SimpleKernelExpectedSeed, SimpleKernelError> {
    let (_, derived) = build_parity_case_from_source(public.source.clone(), public.max_steps)?;
    let root_context = SimpleKernelRootContext::new()?;
    let root_lane_witness = build_root_lane_witness(&derived.execution_rows);
    let root_lane_columns = build_root_lane_columns_from_witness(&root_lane_witness);
    let root_lane_commitment =
        build_root_lane_commitment_artifact_from_witness(root_context.params(), &root_lane_witness)?;
    let prepared_step_bindings = build_prepared_step_binding_summary(
        &derived.execution_rows,
        &root_lane_witness.semantic_rows,
        &root_lane_columns,
        materialize_bindings,
    )?;
    let trace = trace_witness_from_derived(&derived);
    let stages = stage_witness_bundle_from_derived(&derived);
    let stage_claims = build_stage_claim_bundle_from_parts(
        &stages.stage1,
        &stages.stage2,
        &stages.stage3,
        stages.transcript.events.len(),
        &derived.kernel,
    )?;
    let kernel_claims = kernel_claim_bundle_from_parts(&derived, prepared_step_bindings);
    Ok(SimpleKernelExpectedSeed {
        trace,
        stages,
        stage_claims,
        kernel_claims,
        root_lane_columns,
        root_lane_commitment,
        root_lane_witness,
    })
}

fn build_simple_kernel_seed_and_witness_with_perf(
    public: &SimpleKernelPublicInput,
    materialize_bindings: bool,
) -> Result<((SimpleKernelBuildSeed, RootLaneWitness), SimpleKernelBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (_, derived) = build_parity_case_from_source(public.source.clone(), public.max_steps)?;
    let root_context = cached_simple_kernel_root_context()?;

    let root_lane_witness_started = Instant::now();
    let root_lane_witness = build_root_lane_witness(&derived.execution_rows);
    let root_lane_witness_ms = millis_since(root_lane_witness_started);

    let root_lane_columns_started = Instant::now();
    let root_lane_columns = build_root_lane_columns_from_witness(&root_lane_witness);
    let root_lane_columns_ms = millis_since(root_lane_columns_started);

    let root_lane_commitment_started = Instant::now();
    let root_lane_commitment =
        build_root_lane_commitment_artifact_from_witness(root_context.params(), &root_lane_witness)?;
    let root_lane_commitment_ms = millis_since(root_lane_commitment_started);

    let bindings_started = Instant::now();
    let prepared_step_bindings = build_prepared_step_binding_summary(
        &derived.execution_rows,
        &root_lane_witness.semantic_rows,
        &root_lane_columns,
        materialize_bindings,
    )?;
    let prepared_step_bindings_ms = millis_since(bindings_started);

    let trace = trace_witness_from_derived(&derived);
    let stages = stage_witness_bundle_from_derived(&derived);
    let (stage_claims, stage_claim_bundle) = build_stage_claim_bundle_from_parts_with_perf(
        &stages.stage1,
        &stages.stage2,
        &stages.stage3,
        stages.transcript.events.len(),
        &derived.kernel,
    )?;
    let kernel_claims = kernel_claim_bundle_from_parts(&derived, prepared_step_bindings);
    let (stage_packages, stage_package_bundle) =
        build_stage_package_bundle_with_perf(&stages.stage1, &stages.stage2, &stages.stage3, &stage_claims)?;
    let (kernel_opening, kernel_opening_bundle) =
        build_kernel_opening_bundle_with_perf(&stage_claims, &stage_packages, &kernel_claims, &root_lane_commitment)?;
    Ok((
        (
            SimpleKernelBuildSeed {
                trace,
                stages,
                stage_claims,
                stage_packages,
                kernel_opening,
                kernel_claims,
                root_lane_columns,
                root_lane_commitment,
            },
            root_lane_witness,
        ),
        SimpleKernelBuildPerf {
            root_lane_witness_ms,
            root_lane_columns_ms,
            root_lane_commitment_ms,
            public_steps_ms: 0.0,
            prepared_steps_ms: 0.0,
            prepared_step_bindings_ms,
            stage_claim_bundle,
            stage_package_bundle,
            kernel_opening_bundle,
            total_ms: millis_since(total_started),
        },
    ))
}

fn build_simple_kernel_seed_with_perf(
    public: &SimpleKernelPublicInput,
) -> Result<(SimpleKernelBuildSeed, SimpleKernelBuildPerf), SimpleKernelError> {
    let ((seed, _root_lane_witness), perf) = build_simple_kernel_seed_and_witness_with_perf(public, false)?;
    Ok((seed, perf))
}

pub fn build_simple_kernel_witness(
    public: &SimpleKernelPublicInput,
) -> Result<SimpleKernelAuditOutput, SimpleKernelError> {
    Ok(build_simple_kernel_witness_with_perf(public)?.0)
}

pub fn build_simple_kernel_witness_with_perf(
    public: &SimpleKernelPublicInput,
) -> Result<(SimpleKernelAuditOutput, SimpleKernelBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let ((seed, root_lane_witness), mut perf) = build_simple_kernel_seed_and_witness_with_perf(public, true)?;
    let prepared_steps_started = Instant::now();
    let root_context = SimpleKernelRootContext::new()?;
    let prepared_steps =
        build_prepared_steps_from_root_lane_witness(&root_context, &seed.trace.execution_rows, &root_lane_witness)?;
    perf.prepared_steps_ms = millis_since(prepared_steps_started);
    perf.total_ms = millis_since(total_started);
    Ok((
        SimpleKernelAuditOutput {
            kernel: SimpleKernelOutput {
                trace: seed.trace,
                stages: seed.stages,
                stage_claims: seed.stage_claims,
                stage_packages: seed.stage_packages,
                kernel_opening: seed.kernel_opening,
                kernel_claims: seed.kernel_claims,
                root_lane_columns: seed.root_lane_columns,
                root_lane_commitment: seed.root_lane_commitment,
            },
            prepared_steps,
        },
        perf,
    ))
}

fn build_packaged_simple_kernel_output_with_perf(
    public: &SimpleKernelPublicInput,
) -> Result<(SimpleKernelOutput, SimpleKernelBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (seed, mut perf) = build_simple_kernel_seed_with_perf(public)?;
    perf.total_ms = millis_since(total_started);
    Ok((
        SimpleKernelOutput {
            trace: seed.trace,
            stages: seed.stages,
            stage_claims: seed.stage_claims,
            stage_packages: seed.stage_packages,
            kernel_opening: seed.kernel_opening,
            kernel_claims: seed.kernel_claims,
            root_lane_columns: seed.root_lane_columns,
            root_lane_commitment: seed.root_lane_commitment,
        },
        perf,
    ))
}

fn build_public_simple_kernel_seed_from_derived_with_perf(
    derived: &Rv64imParityDerivedCase,
) -> Result<(PublicSimpleKernelBuildSeed, SimpleKernelBuildPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let root_context = cached_simple_kernel_root_context()?;

    let root_lane_witness_started = Instant::now();
    let (root_lane_witness, prepared_step_bindings) =
        build_public_root_lane_witness_and_binding_summary(&derived.execution_rows);
    let root_lane_witness_ms = millis_since(root_lane_witness_started);

    let root_lane_columns_started = Instant::now();
    let root_lane_columns = build_root_lane_columns_from_public_witness(&root_lane_witness);
    let root_lane_columns_ms = millis_since(root_lane_columns_started);

    let root_lane_commitment_started = Instant::now();
    let root_lane_commitment =
        build_root_lane_commitment_summary_artifact_from_public_witness(root_context.params(), &root_lane_witness)?;
    let root_lane_commitment_ms = millis_since(root_lane_commitment_started);

    let prepared_step_bindings_ms = 0.0;

    let trace = trace_projection_bundle_from_rows(
        &derived.manifest,
        &derived.execution_rows,
        derived.kernel.execution_digest,
    );
    let stages = stage_witness_projection_bundle_from_summaries(
        &derived.stage1,
        &derived.stage2,
        &derived.stage3,
        &derived.transcript,
    );
    let (stage_claims, stage_claim_bundle) = build_stage_claim_bundle_from_parts_with_perf(
        &derived.stage1,
        &derived.stage2,
        &derived.stage3,
        derived.transcript.events.len(),
        &derived.kernel,
    )?;
    let kernel_claims = kernel_claim_bundle_from_parts(&derived, prepared_step_bindings);
    let (stage_packages, stage_package_bundle) =
        build_public_stage_package_bundle_with_perf(&derived.stage1, &derived.stage2, &derived.stage3, &stage_claims)?;
    let (kernel_opening, kernel_opening_bundle) = build_public_kernel_opening_bundle_with_perf(
        &stage_claims,
        &stage_packages,
        &kernel_claims,
        &root_lane_commitment,
    )?;
    Ok((
        PublicSimpleKernelBuildSeed {
            trace,
            stages,
            stage_claims,
            stage_packages,
            kernel_opening,
            kernel_claims,
            root_lane_columns,
            root_lane_commitment,
        },
        SimpleKernelBuildPerf {
            root_lane_witness_ms,
            root_lane_columns_ms,
            root_lane_commitment_ms,
            public_steps_ms: 0.0,
            prepared_steps_ms: 0.0,
            prepared_step_bindings_ms,
            stage_claim_bundle,
            stage_package_bundle,
            kernel_opening_bundle,
            total_ms: millis_since(total_started),
        },
    ))
}

pub(super) fn simple_kernel_output_from_expected_seed(
    expected: SimpleKernelExpectedSeed,
    stage_packages: SimpleKernelStagePackageBundle,
    kernel_opening: SimpleKernelOpeningBundle,
) -> SimpleKernelOutput {
    SimpleKernelOutput {
        trace: expected.trace,
        stages: expected.stages,
        stage_claims: expected.stage_claims,
        stage_packages,
        kernel_opening,
        kernel_claims: expected.kernel_claims,
        root_lane_columns: expected.root_lane_columns,
        root_lane_commitment: expected.root_lane_commitment,
    }
}

fn public_simple_kernel_output_from_seed(seed: PublicSimpleKernelBuildSeed) -> PublicSimpleKernelOutput {
    PublicSimpleKernelOutput {
        trace: seed.trace,
        stages: seed.stages,
        stage_claims: seed.stage_claims,
        stage_packages: seed.stage_packages,
        kernel_opening: seed.kernel_opening,
        kernel_claims: seed.kernel_claims,
        root_lane_columns: seed.root_lane_columns,
        root_lane_commitment: seed.root_lane_commitment,
    }
}

pub(super) fn build_public_simple_kernel_output_and_witness_with_perf(
    public: &SimpleKernelPublicInput,
) -> Result<
    (
        (PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar),
        SimpleKernelBuildPerf,
    ),
    SimpleKernelError,
> {
    let total_started = Instant::now();
    let (_, derived) = build_parity_case_from_source(public.source.clone(), public.max_steps)?;
    let ((output, sidecar), mut perf) = build_public_simple_kernel_output_and_witness_from_derived_with_perf(&derived)?;
    perf.total_ms = millis_since(total_started);
    Ok(((output, sidecar), perf))
}

pub(super) fn build_public_simple_kernel_output_and_witness_from_derived_with_perf(
    derived: &Rv64imParityDerivedCase,
) -> Result<
    (
        (PublicSimpleKernelOutput, PublicSimpleKernelWitnessSidecar),
        SimpleKernelBuildPerf,
    ),
    SimpleKernelError,
> {
    let (seed, perf) = build_public_simple_kernel_seed_from_derived_with_perf(derived)?;
    let sidecar = PublicSimpleKernelWitnessSidecar {
        trace: trace_witness_from_derived(derived),
        stages: stage_witness_bundle_from_derived(derived),
    };
    Ok(((public_simple_kernel_output_from_seed(seed), sidecar), perf))
}

pub(super) fn simple_kernel_proof_from_output(output: &SimpleKernelOutput) -> SimpleKernelProof {
    SimpleKernelProof {
        root_params_id: rv64im_simple_root_context_id(),
        trace: output.trace.clone(),
        stages: output.stages.clone(),
        stage_claims: output.stage_claims.clone(),
        stage_packages: output.stage_packages.clone(),
        kernel_opening: output.kernel_opening.clone(),
        kernel_claims: output.kernel_claims.clone(),
        root_lane_columns: output.root_lane_columns.clone(),
        root_lane_commitment: output.root_lane_commitment.clone(),
    }
}

pub fn prove_simple_kernel(
    input: &SimpleKernelProverInput,
) -> Result<(SimpleKernelAuditOutput, SimpleKernelProof), SimpleKernelError> {
    let output = build_simple_kernel_witness(&input.public)?;
    let proof = simple_kernel_proof_from_output(&output.kernel);
    Ok((output, proof))
}

pub fn verify_simple_kernel(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
) -> Result<SimpleKernelAuditOutput, SimpleKernelError> {
    Ok(verify_simple_kernel_with_perf(input, proof)?.0)
}

pub fn verify_simple_kernel_with_perf(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
) -> Result<(SimpleKernelAuditOutput, SimpleKernelVerifyPerf), SimpleKernelError> {
    let (expected, perf) = verify_simple_kernel_seed_with_perf(input, proof)?;
    let root_context = SimpleKernelRootContext::new()?;
    let prepared_steps = build_prepared_steps_from_root_lane_witness(
        &root_context,
        &expected.trace.execution_rows,
        &expected.root_lane_witness,
    )?;
    Ok((
        SimpleKernelAuditOutput {
            kernel: simple_kernel_output_from_expected_seed(
                expected,
                proof.stage_packages.clone(),
                proof.kernel_opening.clone(),
            ),
            prepared_steps,
        },
        perf,
    ))
}

pub(super) fn verify_simple_kernel_core_seed_with_perf(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
) -> Result<(SimpleKernelExpectedSeed, SimpleKernelVerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let expected_root = rv64im_simple_root_context_id();
    if proof.root_params_id != expected_root {
        return Err(SimpleKernelError::Bridge("RV64IM root context id mismatch".into()));
    }
    let expected_started = Instant::now();
    let expected = build_simple_kernel_expected_seed(&input.public, true)?;
    let expected_core_ms = millis_since(expected_started);
    let trace_match_started = Instant::now();
    if proof.trace != expected.trace {
        return Err(SimpleKernelError::Bridge("RV64IM kernel trace witness mismatch".into()));
    }
    let trace_match_ms = millis_since(trace_match_started);
    let stages_match_started = Instant::now();
    if proof.stages != expected.stages {
        return Err(SimpleKernelError::Bridge("RV64IM stage witness bundle mismatch".into()));
    }
    let stages_match_ms = millis_since(stages_match_started);
    let stage_claims_match_started = Instant::now();
    if proof.stage_claims != expected.stage_claims {
        return Err(SimpleKernelError::Bridge("RV64IM stage claim bundle mismatch".into()));
    }
    let stage_claims_match_ms = millis_since(stage_claims_match_started);
    let kernel_claims_match_started = Instant::now();
    if proof.kernel_claims != expected.kernel_claims {
        return Err(SimpleKernelError::Bridge("RV64IM kernel claim bundle mismatch".into()));
    }
    let kernel_claims_match_ms = millis_since(kernel_claims_match_started);
    let root_lane_columns_match_started = Instant::now();
    if proof.root_lane_columns != expected.root_lane_columns {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root lane column family mismatch".into(),
        ));
    }
    let root_lane_columns_match_ms = millis_since(root_lane_columns_match_started);
    let root_lane_commitment_match_started = Instant::now();
    if proof.root_lane_commitment != expected.root_lane_commitment {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root lane commitment artifact mismatch".into(),
        ));
    }
    let root_lane_commitment_match_ms = millis_since(root_lane_commitment_match_started);
    Ok((
        expected,
        SimpleKernelVerifyPerf {
            expected_core_ms,
            trace_match_ms,
            stages_match_ms,
            stage_claims_match_ms,
            kernel_claims_match_ms,
            root_lane_columns_match_ms,
            root_lane_commitment_match_ms,
            stage_package_bundle: Default::default(),
            kernel_opening_bundle: Default::default(),
            total_ms: millis_since(total_started),
        },
    ))
}

fn verify_simple_kernel_seed_with_perf(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
) -> Result<(SimpleKernelExpectedSeed, SimpleKernelVerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (expected, mut perf) = verify_simple_kernel_core_seed_with_perf(input, proof)?;
    let stage_package_bundle = verify_stage_package_bundle_with_perf(
        &expected.stages.stage1,
        &expected.stages.stage2,
        &expected.stages.stage3,
        &proof.stage_packages,
        &expected.stage_claims,
    )?;
    let kernel_opening_bundle = verify_kernel_opening_bundle_with_perf(
        &proof.kernel_opening,
        &expected.stage_claims,
        &proof.stage_packages,
        &expected.kernel_claims,
        &expected.root_lane_commitment,
    )?;
    perf.stage_package_bundle = stage_package_bundle;
    perf.kernel_opening_bundle = kernel_opening_bundle;
    perf.total_ms = millis_since(total_started);
    Ok((expected, perf))
}

fn verify_packaged_simple_kernel_seed_with_perf(
    input: &SimpleKernelVerifierInput,
    proof: &SimpleKernelProof,
) -> Result<(SimpleKernelExpectedSeed, SimpleKernelVerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let expected_root = rv64im_simple_root_context_id();
    if proof.root_params_id != expected_root {
        return Err(SimpleKernelError::Bridge("RV64IM root context id mismatch".into()));
    }
    let expected_started = Instant::now();
    let expected = build_simple_kernel_expected_seed(&input.public, false)?;
    let expected_core_ms = millis_since(expected_started);
    let trace_match_started = Instant::now();
    if proof.trace != expected.trace {
        return Err(SimpleKernelError::Bridge("RV64IM kernel trace witness mismatch".into()));
    }
    let trace_match_ms = millis_since(trace_match_started);
    let stages_match_started = Instant::now();
    if proof.stages != expected.stages {
        return Err(SimpleKernelError::Bridge("RV64IM stage witness bundle mismatch".into()));
    }
    let stages_match_ms = millis_since(stages_match_started);
    let stage_claims_match_started = Instant::now();
    if proof.stage_claims != expected.stage_claims {
        return Err(SimpleKernelError::Bridge("RV64IM stage claim bundle mismatch".into()));
    }
    let stage_claims_match_ms = millis_since(stage_claims_match_started);
    let kernel_claims_match_started = Instant::now();
    if proof.kernel_claims != expected.kernel_claims {
        return Err(SimpleKernelError::Bridge("RV64IM kernel claim bundle mismatch".into()));
    }
    let kernel_claims_match_ms = millis_since(kernel_claims_match_started);
    let root_lane_columns_match_started = Instant::now();
    if proof.root_lane_columns != expected.root_lane_columns {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root lane column family mismatch".into(),
        ));
    }
    let root_lane_columns_match_ms = millis_since(root_lane_columns_match_started);
    let root_lane_commitment_match_started = Instant::now();
    if proof.root_lane_commitment != expected.root_lane_commitment {
        return Err(SimpleKernelError::Bridge(
            "RV64IM root lane commitment artifact mismatch".into(),
        ));
    }
    let root_lane_commitment_match_ms = millis_since(root_lane_commitment_match_started);
    let stage_package_bundle = verify_stage_package_bundle_with_perf(
        &expected.stages.stage1,
        &expected.stages.stage2,
        &expected.stages.stage3,
        &proof.stage_packages,
        &expected.stage_claims,
    )?;
    let kernel_opening_bundle = verify_kernel_opening_bundle_with_perf(
        &proof.kernel_opening,
        &expected.stage_claims,
        &proof.stage_packages,
        &expected.kernel_claims,
        &expected.root_lane_commitment,
    )?;
    Ok((
        expected,
        SimpleKernelVerifyPerf {
            expected_core_ms,
            trace_match_ms,
            stages_match_ms,
            stage_claims_match_ms,
            kernel_claims_match_ms,
            root_lane_columns_match_ms,
            root_lane_commitment_match_ms,
            stage_package_bundle,
            kernel_opening_bundle,
            total_ms: millis_since(total_started),
        },
    ))
}

fn first_rot_col_from_matrix(mat: &Mat<F>) -> [F; D] {
    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    coeffs
}

fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs.iter()) {
        let first_rot_col = first_rot_col_from_matrix(rho);
        s_mul_add_from_rot_col(&mut acc, &first_rot_col, c);
    }
    acc
}

fn combine_b_pows(cs: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    let base = F::from_u64(b as u64);
    let mut pow = F::ONE;
    for c in cs {
        scale_commitment_add_inplace(&mut acc, pow, c);
        pow *= base;
    }
    acc
}

pub fn rv64im_ajtai_mixers(
) -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment> {
    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

pub fn prove_packaged_simple_kernel(
    input: &SimpleKernelProverInput,
) -> Result<(SimpleKernelOutput, SimpleKernelPackagedProof), SimpleKernelError> {
    Ok(prove_packaged_simple_kernel_with_perf(input)?.0)
}

pub fn prove_packaged_simple_kernel_with_perf(
    input: &SimpleKernelProverInput,
) -> Result<((SimpleKernelOutput, SimpleKernelPackagedProof), Rv64imProofProvePerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (output, simple_kernel) = build_packaged_simple_kernel_output_with_perf(&input.public)?;
    let kernel = simple_kernel_proof_from_output(&output);
    let main_lane_started = Instant::now();
    let main_lane = build_simple_kernel_main_lane_artifact(
        &output.root_lane_columns,
        &output.root_lane_commitment,
        FoldSchedule::WholeTrace,
    )?;
    Ok((
        (output, SimpleKernelPackagedProof { kernel, main_lane }),
        Rv64imProofProvePerf {
            shared_trace_ms: 0.0,
            simple_kernel,
            parallel_overlap_ms: 0.0,
            main_lane_ms: millis_since(main_lane_started),
            root_main_lane: RootMainLanePackagedProofProvePerf::default(),
            public_export_ms: 0.0,
            total_ms: millis_since(total_started),
        },
    ))
}

pub fn verify_packaged_simple_kernel(
    input: &SimpleKernelVerifierInput,
    packaged: &SimpleKernelPackagedProof,
) -> Result<SimpleKernelOutput, SimpleKernelError> {
    Ok(verify_packaged_simple_kernel_with_perf(input, packaged)?.0)
}

pub fn verify_packaged_simple_kernel_with_perf(
    input: &SimpleKernelVerifierInput,
    packaged: &SimpleKernelPackagedProof,
) -> Result<(SimpleKernelOutput, PackagedSimpleKernelVerifyPerf), SimpleKernelError> {
    let total_started = Instant::now();
    let (expected, simple_kernel) = verify_packaged_simple_kernel_seed_with_perf(input, &packaged.kernel)?;
    let main_lane_artifact_match_started = Instant::now();
    validate_simple_kernel_main_lane_artifact(
        &expected.root_lane_columns,
        &expected.root_lane_commitment,
        &packaged.main_lane,
    )?;
    let main_lane_artifact_match_ms = millis_since(main_lane_artifact_match_started);
    Ok((
        simple_kernel_output_from_expected_seed(
            expected,
            packaged.kernel.stage_packages.clone(),
            packaged.kernel.kernel_opening.clone(),
        ),
        PackagedSimpleKernelVerifyPerf {
            simple_kernel,
            main_lane_artifact_match_ms,
            total_ms: millis_since(total_started),
        },
    ))
}
