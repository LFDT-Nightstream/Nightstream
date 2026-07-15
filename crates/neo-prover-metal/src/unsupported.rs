//! Type-compatible unavailable backend for non-Apple targets and shaderless builds.

use super::*;

pub struct MetalSession;

pub struct MetalAjtaiLowNormPlan;

pub struct MetalKxChainPlan;

pub struct MetalPoseidonUniformPlan;

pub(crate) struct MetalResidentWitness;

pub(crate) struct MetalResidentWitnessSnapshot;

impl MetalResidentWitnessSnapshot {
    pub(crate) fn materialize(&self) -> Result<Vec<neo_ccs::Mat<neo_math::F>>, &'static str> {
        Err("Metal resident witnesses are unavailable")
    }
}

#[derive(Clone)]
pub(crate) struct MetalWitnessMasks;

impl MetalWitnessMasks {
    pub(crate) fn matches(&self, _witness_count: usize, _blocks: usize) -> bool {
        false
    }

    pub(crate) fn matches_nc(&self, _witness_count: usize, _blocks: usize, _active_rows: usize) -> bool {
        false
    }
}

pub(crate) struct MetalResidentChildren;

pub(crate) struct MetalDecFormPlan;

pub(crate) struct MetalDecPublicProjection<'a> {
    pub active_rows: usize,
    pub s_col: &'a [neo_math::K],
}

pub(crate) struct MetalAjtaiRingForms;

pub(crate) struct MetalFeOraclePlan;

pub(crate) struct MetalDeferredEvalTable;

pub(crate) struct MetalDeferredMcsRowTables;

impl MetalDecFormPlan {
    pub(crate) fn matches(&self, _cache: &neo_reductions::superneo_eval::SuperneoEvalCache) -> bool {
        false
    }
}

impl MetalFeOraclePlan {
    pub(crate) fn matches(&self, _cache: &neo_reductions::superneo_eval::SuperneoEvalCache) -> bool {
        false
    }

    pub(crate) fn supports_resident_eval(&self) -> bool {
        false
    }

    pub(crate) fn explicit_coefficients(&self) -> usize {
        0
    }

    pub(crate) fn explicit_row_list_histogram(&self) -> [usize; 8] {
        [0; 8]
    }

    pub(crate) fn max_explicit_row_entries(&self) -> usize {
        0
    }
}

impl MetalDeferredMcsRowTables {
    pub(crate) fn matches(&self, _mcs_idx: usize, _n_pad: usize, _table_count: usize) -> bool {
        false
    }

    pub(crate) fn seeded_build(&self) -> Duration {
        Duration::ZERO
    }

    pub(crate) fn seeded_patch_entries(&self) -> usize {
        0
    }

    pub(crate) fn seeded_patch_bytes(&self) -> usize {
        0
    }
}

pub(crate) struct MetalDecMaterial {
    pub child_mask_words: Vec<u64>,
    pub child_nonzero: Vec<bool>,
    pub y_words: Vec<u64>,
    pub y_zcol_words: Vec<u64>,
    pub y_zcol_gpu: Duration,
    pub commitment_words: Vec<u64>,
    pub resident_children: MetalResidentChildren,
}

pub(crate) struct MetalFeSumcheckPlan;

pub(crate) struct MetalNcSumcheckPlan;

impl MetalNcSumcheckPlan {
    pub(crate) fn active_witness_count(&self) -> usize {
        0
    }
}

pub(crate) struct MetalFeSumcheckInputs<'a> {
    pub tables: &'a [MetalFeTableInput<'a>],
    pub shape: &'a [u64],
    pub mcs_headers: &'a [u64],
    pub mcs_table_indices: &'a [u64],
    pub gammas: &'a [u64],
    pub term_headers: &'a [u64],
    pub term_variables: &'a [u64],
    pub table_count: usize,
    pub coefficient_count: usize,
}

pub(crate) enum MetalFeTableInput<'a> {
    Host(&'a [neo_math::K]),
    TensorPoint(&'a [neo_math::K]),
    DeferredMcs {
        tables: &'a MetalDeferredMcsRowTables,
        table: usize,
    },
    DeferredEval(&'a MetalDeferredEvalTable),
}

pub(crate) struct MetalNcSumcheckInputs<'a> {
    pub eq_point: &'a [u64],
    pub digits: MetalNcDigitInput<'a>,
    pub resident_masks: Option<&'a MetalWitnessMasks>,
    pub weights: &'a [u64],
    pub witness_count: usize,
    pub rows: usize,
    pub width: usize,
    pub dense: bool,
}

pub(crate) enum MetalNcDigitInput<'a> {
    Table(&'a [u64]),
    SignedMasks {
        words: &'a [u64],
        blocks: usize,
        active_rows: usize,
    },
}

pub(crate) struct MetalNcFinalState {
    pub eq_beta: KWords,
    pub digit_words: Vec<u64>,
    pub width: usize,
    pub dense: bool,
}

pub(crate) struct MetalSumcheckTrace {
    pub coeffs: Vec<Vec<KWords>>,
    pub challenges: Vec<KWords>,
    pub transcript_state: [u64; 8],
    pub transcript_absorbed: usize,
}

pub(crate) struct MetalNcSumcheckTrace {
    pub rounds: MetalSumcheckTrace,
    pub final_state: MetalNcFinalState,
}

impl MetalSession {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::Unavailable)
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

    pub(crate) fn prepare_fe_sumcheck(
        &self,
        _inputs: MetalFeSumcheckInputs<'_>,
    ) -> Result<MetalFeSumcheckPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_fe_oracle(
        &self,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
    ) -> Result<MetalFeOraclePlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_mcs_row_tables(
        &self,
        _plan: &MetalFeOraclePlan,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        _mcs_idx: usize,
        _matrix_indices: &[usize],
        _z_blocks: &neo_reductions::superneo_eval::SuperneoZBlocks,
        _witness_masks: Option<&MetalWitnessMasks>,
        _n_eff: usize,
        _n_pad: usize,
    ) -> Result<MetalDeferredMcsRowTables, MetalError> {
        Err(MetalError::Unavailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_carried_eval_table(
        &self,
        _plan: &MetalFeOraclePlan,
        _resident_id: u64,
        _carried_coeffs: &[neo_math::K],
        _weights: &[neo_math::K; neo_math::D],
        _mat_coeffs: &[neo_math::K],
        _n_eff: usize,
        _n_pad: usize,
    ) -> Result<MetalDeferredEvalTable, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn fe_sumcheck_round(
        &self,
        _plan: &mut MetalFeSumcheckPlan,
        _shape: &[u64],
        _fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn fe_sumcheck_trace(
        &self,
        _plan: &mut MetalFeSumcheckPlan,
        _base_shape: &[u64],
        _transcript_state: [u64; 8],
        _transcript_absorbed: usize,
        _rounds: usize,
    ) -> Result<MetalSumcheckTrace, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_nc_sumcheck(
        &self,
        _inputs: MetalNcSumcheckInputs<'_>,
    ) -> Result<MetalNcSumcheckPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_witness_masks(
        &self,
        _words: &[u64],
        _witness_count: usize,
        _blocks: usize,
        _active_rows: usize,
    ) -> Result<MetalWitnessMasks, MetalError> {
        Err(MetalError::Unavailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_witness_masks_with_resident_id(
        &self,
        _fresh_words: &[u64],
        _fresh_count: usize,
        _input_count: usize,
        _blocks: usize,
        _active_rows: usize,
        _resident_id: u64,
    ) -> Result<MetalWitnessMasks, MetalError> {
        Err(MetalError::Unavailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compose_witness_masks_from_device(
        &self,
        _fresh: &MetalWitnessMasks,
        _fresh_count: usize,
        _input_count: usize,
        _blocks: usize,
        _active_rows: usize,
        _resident_id: Option<u64>,
    ) -> Result<MetalWitnessMasks, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn enqueue_rlc_witness_mix_from_signed_masks(
        &self,
        _rhos: &[i8],
        _plan: &MetalNcSumcheckPlan,
        _input_count: usize,
        _cols: usize,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        Err(MetalError::Unavailable)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_rlc_witness_mix_from_signed_masks_with_resident_id(
        &self,
        _rhos: &[i8],
        _plan: &MetalNcSumcheckPlan,
        _fresh_count: usize,
        _input_count: usize,
        _cols: usize,
        _resident_id: u64,
    ) -> Result<Option<MetalResidentWitness>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn nc_sumcheck_round(
        &self,
        _plan: &mut MetalNcSumcheckPlan,
        _shape: &[u64],
        _fold_challenge: Option<KWords>,
    ) -> Result<Vec<KWords>, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn nc_sumcheck_trace(
        &self,
        _plan: &mut MetalNcSumcheckPlan,
        _transcript_state: [u64; 8],
        _transcript_absorbed: usize,
        _rounds: usize,
    ) -> Result<MetalNcSumcheckTrace, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn finalize_nc_sumcheck(
        &self,
        _plan: &mut MetalNcSumcheckPlan,
        _fold_challenge: Option<KWords>,
    ) -> Result<MetalNcFinalState, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn mix_rlc_witnesses_resident(
        &self,
        _rhos: &[u64],
        _witnesses: &[u64],
        _input_count: usize,
        _cols: usize,
    ) -> Result<MetalResidentWitness, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn mix_rlc_witnesses_with_resident_id(
        &self,
        _rhos: &[u64],
        _fresh_witnesses: &[u64],
        _fresh_count: usize,
        _input_count: usize,
        _cols: usize,
        _resident_id: u64,
    ) -> Result<MetalResidentWitness, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn resident_running_shape(&self, _id: u64) -> Option<(usize, usize)> {
        None
    }

    pub(crate) fn retain_running_children(&self, _children: MetalResidentChildren) -> u64 {
        0
    }

    pub(crate) fn split_dec_base2_with_ring_forms(
        &self,
        _parent: &MetalResidentWitness,
        _child_count: usize,
        _form_rows: usize,
        _form_words: &[u64],
        _public_projection: Option<MetalDecPublicProjection<'_>>,
        _commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn eval_ajtai_y_from_signed_masks(
        &self,
        _plan: &MetalDecFormPlan,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        _chi_r: &[neo_math::K],
        _n_eff: usize,
        _mask_words: &[u64],
        _resident_masks: Option<&MetalWitnessMasks>,
        _witness_count: usize,
    ) -> Result<(Vec<u64>, MetalAjtaiRingForms, MetalAjtaiYProfile), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn eval_ajtai_y_from_signed_masks_and_row_challenges(
        &self,
        _plan: &MetalDecFormPlan,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        _row_challenges: &[neo_math::K],
        _n_eff: usize,
        _mask_words: &[u64],
        _resident_masks: Option<&MetalWitnessMasks>,
        _witness_count: usize,
    ) -> Result<(Vec<u64>, MetalAjtaiRingForms, MetalAjtaiYProfile), MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn split_dec_base2_with_prebuilt_ring_forms(
        &self,
        _parent: &MetalResidentWitness,
        _child_count: usize,
        _plan: &MetalDecFormPlan,
        _forms: &MetalAjtaiRingForms,
        _row_challenges: &[neo_math::K],
        _n_eff: usize,
        _public_projection: Option<MetalDecPublicProjection<'_>>,
        _commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_dec_ring_forms(
        &self,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        _oracle: &MetalFeOraclePlan,
    ) -> Result<MetalDecFormPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn split_dec_base2_with_ring_form_plan(
        &self,
        _parent: &mut MetalResidentWitness,
        _child_count: usize,
        _plan: &MetalDecFormPlan,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
        _chi_r: &[neo_math::K],
        _n_eff: usize,
        _public_projection: Option<MetalDecPublicProjection<'_>>,
        _commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        Err(MetalError::Unavailable)
    }
}
