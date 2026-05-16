//! Owns the RV32IM simple-kernel root context and cached root main-lane parameters.
//!
//! This file does not build witnesses or prove/verify sessions. It only fixes the
//! parameter family, Ajtai module, CCS structure, optimized cache, and context IDs
//! shared by the simple-kernel flows.

use crate::proof::FoldSchedule;
use crate::rv32im::ccs::{rv32im_root_main_lane_ccs, RV32IM_ROOT_ROW_WIDTH};
use crate::witness_layout::commit_cols_for_full_width;
use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::CcsStructure;
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::OptimizedStructureCache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use std::sync::OnceLock;

use super::types::SimpleKernelError;

pub(crate) const SIMPLE_KERNEL_PP_SEED: [u8; 32] = [
    0x40, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

// Single-step RV32IM uses the lean base DEC width. Wider chunk step-caps derive
// a larger `k_rho` up front as part of the declared proof family.
const SIMPLE_KERNEL_BASE_K_RHO: u32 = 16;

// Ajtai public parameters are global per dimension bucket, so exact stage surfaces share one seed.
pub(crate) const EXACT_STAGE_PP_SEED: [u8; 32] = SIMPLE_KERNEL_PP_SEED;

pub fn rv32im_simple_kernel_pp_seed() -> [u8; 32] {
    SIMPLE_KERNEL_PP_SEED
}

pub fn rv32im_exact_stage_pp_seed() -> [u8; 32] {
    EXACT_STAGE_PP_SEED
}

pub(super) struct SimpleKernelRootContext {
    params: NeoParams,
    log: AjtaiSModule,
}

impl SimpleKernelRootContext {
    pub(super) fn new() -> Result<Self, SimpleKernelError> {
        Self::new_for_step_cap(1)
    }

    pub(super) fn new_for_step_cap(step_cap: usize) -> Result<Self, SimpleKernelError> {
        let params = rv32im_simple_root_params_for_step_cap(step_cap);
        let m = commit_cols_for_full_width(RV32IM_ROOT_ROW_WIDTH);
        set_global_pp_seeded(D, params.kappa as usize, m, SIMPLE_KERNEL_PP_SEED)
            .map_err(|err| SimpleKernelError::Bridge(format!("canonical RV32IM root seed setup failed: {err}")))?;
        let log = AjtaiSModule::from_global_for_dims(D, m)
            .map_err(|err| SimpleKernelError::Bridge(format!("canonical RV32IM root module failed: {err}")))?;
        Ok(Self { params, log })
    }

    pub(super) fn params(&self) -> &NeoParams {
        &self.params
    }

    pub(super) fn log(&self) -> &AjtaiSModule {
        &self.log
    }
}

fn rv32im_simple_root_params_with_k_rho(k_rho: u32) -> NeoParams {
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(RV32IM_ROOT_ROW_WIDTH).expect("valid RV32IM root params");
    params.k_rho = k_rho;
    params.B = 1u64 << params.k_rho;
    params
}

pub(super) fn cached_simple_kernel_root_context() -> Result<&'static SimpleKernelRootContext, SimpleKernelError> {
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

pub(super) fn cached_root_main_lane_ccs() -> Result<&'static CcsStructure<F>, SimpleKernelError> {
    static ROOT_MAIN_LANE_CCS: OnceLock<Result<CcsStructure<F>, String>> = OnceLock::new();
    ROOT_MAIN_LANE_CCS
        .get_or_init(rv32im_root_main_lane_ccs)
        .as_ref()
        .map_err(|err| SimpleKernelError::Proof(err.clone()))
}

pub(super) fn cached_root_main_lane_optimized_cache() -> Result<&'static OptimizedStructureCache, SimpleKernelError> {
    static ROOT_MAIN_LANE_OPTIMIZED_CACHE: OnceLock<Result<OptimizedStructureCache, String>> = OnceLock::new();
    let ccs = cached_root_main_lane_ccs()?;
    ROOT_MAIN_LANE_OPTIMIZED_CACHE
        .get_or_init(|| OptimizedStructureCache::build(ccs).map_err(|err| err.to_string()))
        .as_ref()
        .map_err(|err| SimpleKernelError::Proof(err.clone()))
}

pub(crate) fn rv32im_cached_root_main_lane_context(
) -> Result<(&'static NeoParams, &'static AjtaiSModule, &'static CcsStructure<F>), SimpleKernelError> {
    let root_context = cached_simple_kernel_root_context()?;
    let ccs = cached_root_main_lane_ccs()?;
    Ok((root_context.params(), root_context.log(), ccs))
}

pub(crate) fn rv32im_root_main_lane_context_for_step_cap(
    step_cap: usize,
) -> Result<(NeoParams, &'static AjtaiSModule, &'static CcsStructure<F>), SimpleKernelError> {
    let (_, log, ccs) = rv32im_cached_root_main_lane_context()?;
    Ok((rv32im_simple_root_params_for_step_cap(step_cap), log, ccs))
}

pub(crate) fn rv32im_root_main_lane_context_for_claim_count(
    claim_count: usize,
) -> Result<(NeoParams, &'static AjtaiSModule, &'static CcsStructure<F>), SimpleKernelError> {
    let k_rho = u32::try_from(claim_count).map_err(|_| {
        SimpleKernelError::Bridge(
            "RV32IM carried claim count does not fit into the local root-parameter selector".into(),
        )
    })?;
    let (_, log, ccs) = rv32im_cached_root_main_lane_context()?;
    Ok((rv32im_simple_root_params_with_k_rho(k_rho), log, ccs))
}

pub(crate) fn rv32im_cached_root_main_lane_optimized_cache(
) -> Result<&'static OptimizedStructureCache, SimpleKernelError> {
    cached_root_main_lane_optimized_cache()
}

fn ceil_log2_usize(value: usize) -> u32 {
    if value <= 1 {
        0
    } else {
        usize::BITS - (value - 1).leading_zeros()
    }
}

pub fn rv32im_simple_root_k_rho_for_step_cap(step_cap: usize) -> u32 {
    let widened = ceil_log2_usize(step_cap.saturating_add(1)).saturating_sub(2);
    SIMPLE_KERNEL_BASE_K_RHO + widened
}

pub fn rv32im_simple_root_params() -> NeoParams {
    rv32im_simple_root_params_for_step_cap(1)
}

pub fn rv32im_simple_root_params_for_step_cap(step_cap: usize) -> NeoParams {
    rv32im_simple_root_params_with_k_rho(rv32im_simple_root_k_rho_for_step_cap(step_cap))
}

pub fn rv32im_simple_root_context_id() -> [u8; 32] {
    rv32im_simple_root_context_id_for_step_cap(1)
}

pub fn rv32im_simple_root_context_id_for_step_cap(step_cap: usize) -> [u8; 32] {
    let params = rv32im_simple_root_params_for_step_cap(step_cap);
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_context");
    tr.append_u64s(
        b"rv32im/root_context/values",
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
            RV32IM_ROOT_ROW_WIDTH as u64,
            commit_cols_for_full_width(RV32IM_ROOT_ROW_WIDTH) as u64,
        ],
    );
    tr.append_message(b"rv32im/root_context/seed", &SIMPLE_KERNEL_PP_SEED);
    tr.digest32()
}

pub(crate) fn rv32im_root_step_cap_for_schedule(
    schedule: FoldSchedule,
    public_step_count: usize,
) -> Result<usize, SimpleKernelError> {
    schedule.validate()?;
    Ok(match schedule {
        FoldSchedule::WholeTrace => public_step_count.max(1),
        FoldSchedule::RowsPerChunk(rows) => rows,
    })
}

pub(crate) fn rv32im_simple_root_context_id_for_schedule(
    schedule: FoldSchedule,
    public_step_count: usize,
) -> Result<[u8; 32], SimpleKernelError> {
    Ok(rv32im_simple_root_context_id_for_step_cap(
        rv32im_root_step_cap_for_schedule(schedule, public_step_count)?,
    ))
}
