//! Type-compatible unavailable backend for non-Apple targets and shaderless builds.

#![allow(dead_code)]

use super::*;

pub struct MetalSession;

pub struct MetalAjtaiLowNormPlan;

pub struct MetalKxChainPlan;

pub struct MetalPoseidonUniformPlan;

#[derive(Clone)]
pub(crate) struct MetalWitnessMasks;

impl MetalSession {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn ownership_id(&self) -> u64 {
        0
    }

    pub fn goldilocks_ops(&self, _lhs: &[u64], _rhs: &[u64]) -> Result<Vec<GoldilocksOps>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn goldilocks_ops_variant(
        &self,
        _lhs: &[u64],
        _rhs: &[u64],
        _variant: GoldilocksMulVariant,
    ) -> Result<Vec<GoldilocksOps>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn kx_mul_add_chain(
        &self,
        _initial: &[KWords],
        _multipliers: &[KWords],
        _rounds: usize,
    ) -> Result<(Vec<KWords>, MetalRunStats), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn prepare_kx_chain(
        &self,
        _initial: &[KWords],
        _multipliers: &[KWords],
    ) -> Result<MetalKxChainPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn kx_mul_add_chain_with_plan(
        &self,
        _plan: &MetalKxChainPlan,
        _rounds: usize,
    ) -> Result<(Vec<KWords>, MetalRunStats), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_permute(&self, _states: &[PoseidonState]) -> Result<Vec<PoseidonState>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_hash(&self, _inputs: &[Vec<u64>]) -> Result<Vec<PoseidonDigest>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_hash_variant(
        &self,
        _inputs: &[Vec<u64>],
        _variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_hash_uniform(
        &self,
        _fields: &[u64],
        _fields_per_hash: usize,
        _variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn prepare_poseidon2_uniform(
        &self,
        _fields: &[u64],
        _fields_per_hash: usize,
    ) -> Result<MetalPoseidonUniformPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn poseidon2_hash_uniform_with_plan(
        &self,
        _plan: &MetalPoseidonUniformPlan,
        _variant: PoseidonHashVariant,
    ) -> Result<Vec<PoseidonDigest>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn ajtai_mat_vec(
        &self,
        _matrix: &[u64],
        _rows: usize,
        _cols: usize,
        _message: &[u64],
    ) -> Result<Vec<u64>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn ajtai_low_norm_mat_vec(
        &self,
        _matrix: &[u64],
        _rows: usize,
        _cols: usize,
        _message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn prepare_ajtai_low_norm(
        &self,
        _matrix: &[u64],
        _rows: usize,
        _cols: usize,
    ) -> Result<MetalAjtaiLowNormPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_ajtai_low_norm_seeded(
        &self,
        _seed: [u8; 32],
        _rows: usize,
        _cols: usize,
    ) -> Result<MetalAjtaiLowNormPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn ajtai_low_norm_with_plan(
        &self,
        _plan: &MetalAjtaiLowNormPlan,
        _message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn ajtai_low_norm_many_from_masks(
        &self,
        _plan: &MetalAjtaiLowNormPlan,
        _masks: &MetalWitnessMasks,
        _count: usize,
    ) -> Result<(Vec<u64>, Duration), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn ajtai_lane_commitments_from_masks(
        &self,
        _ops_plan: &MetalAjtaiLowNormPlan,
        _mem_plan: &MetalAjtaiLowNormPlan,
        _masks: &MetalWitnessMasks,
        _count: usize,
        _full_cols: usize,
        _ranges: &neo_fold_clean::paper::relations::LaneRanges,
    ) -> Result<(Vec<u64>, Duration), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn sis_accumulator_digest(
        &self,
        _config: neo_fold_clean::paper::reductions::accumulator_sis_circuit::SisAccumulatorConfig,
        _fields: &[neo_math::F],
    ) -> Result<[neo_math::F; 4], MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn sis_accumulator_digest_resident(
        &self,
        _config: neo_fold_clean::paper::reductions::accumulator_sis_circuit::SisAccumulatorConfig,
        _fields: &[neo_math::F],
    ) -> Result<[neo_math::F; 4], MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn fold_k_table(&self, _table: &[KWords], _challenge: KWords) -> Result<Vec<KWords>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn fold_k_tables(&self, _tables: &[Vec<KWords>], _challenge: KWords) -> Result<Vec<Vec<KWords>>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn fold_k_table_full(
        &self,
        _table: &[KWords],
        _challenges: &[KWords],
    ) -> Result<(KWords, MetalRunStats), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn device_info(&self) -> Result<MetalDeviceInfo, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub fn activity(&self) -> MetalActivity {
        MetalActivity::default()
    }

    pub fn reset_activity(&self) {}

    pub(crate) fn prepare_witness_masks(
        &self,
        _words: &[u64],
        _witness_count: usize,
        _blocks: usize,
        _active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        Err(MetalError::Unavailable)
    }
}
