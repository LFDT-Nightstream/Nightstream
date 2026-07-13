//! Native Metal prover backend for SuperNeo.
//!
//! The protocol crates remain authoritative for transcript order, proof
//! assembly, and verification. This crate owns Metal device state and the
//! accelerator implementation of the `NifsProverAdapter` execution surface.

pub mod poseidon2;

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature requires an Apple target");

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod device;

#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub use device::{MetalDevice, MetalError as MetalDeviceError, MetalTranscriptOp, MetalTranscriptOutput};

use std::time::Duration;

use thiserror::Error;

mod adapter;
pub use adapter::{MetalNifsProfile, MetalNifsProver};
mod fold_output;
mod sumcheck;

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
mod session;
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
pub use session::{MetalAjtaiLowNormPlan, MetalKxChainPlan, MetalPoseidonUniformPlan, MetalSession};
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
pub(crate) use session::{
    MetalDecFormPlan, MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalNcFinalState, MetalNcSumcheckInputs,
    MetalNcSumcheckPlan, MetalNcSumcheckTrace, MetalResidentWitness, MetalSumcheckTrace,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoldilocksOps {
    pub add: u64,
    pub sub: u64,
    pub mul: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KWords {
    pub c0: u64,
    pub c1: u64,
}

pub type PoseidonState = [u64; 8];
pub type PoseidonDigest = [u64; 4];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoldilocksMulVariant {
    Limb32,
    Native64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoseidonHashVariant {
    Scalar,
    SimdGroup,
}

impl KWords {
    pub const fn new(c0: u64, c1: u64) -> Self {
        Self { c0, c1 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MetalRunStats {
    pub elements: usize,
    pub dispatches: usize,
    pub elapsed: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub unified_memory: bool,
    pub recommended_working_set_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalActivity {
    pub command_buffers: u64,
    pub dispatches: u64,
    pub host_waits: u64,
    pub allocated_bytes: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    pub current_allocated_bytes: u64,
}

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal is unavailable on this target or its precompiled shader library was not built")]
    Unavailable,
    #[error("Metal device creation failed")]
    Device,
    #[error("Metal command queue creation failed")]
    Queue,
    #[error("Metal shader library load failed: {0}")]
    Library(String),
    #[error("Metal shader function {0} is missing")]
    Function(&'static str),
    #[error("Metal compute pipeline creation failed: {0}")]
    Pipeline(String),
    #[error("Metal buffer allocation failed for {bytes} bytes")]
    Buffer { bytes: usize },
    #[error("Metal command buffer creation failed")]
    CommandBuffer,
    #[error("Metal compute encoder creation failed")]
    Encoder,
    #[error("Metal command execution failed: {0}")]
    Execution(String),
    #[error("Metal input shape mismatch: {0}")]
    Shape(&'static str),
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub struct MetalSession;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub struct MetalAjtaiLowNormPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub struct MetalKxChainPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub struct MetalPoseidonUniformPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalResidentWitness;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
impl MetalResidentWitness {
    pub(crate) fn cols(&self) -> usize {
        0
    }
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalResidentChildren;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalDecFormPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
impl MetalDecFormPlan {
    pub(crate) fn matches(&self, _cache: &neo_reductions::superneo_eval::SuperneoEvalCache) -> bool {
        false
    }
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalDecMaterial {
    pub child_mask_words: Vec<u64>,
    pub child_nonzero: Vec<bool>,
    pub y_words: Vec<u64>,
    pub commitment_words: Vec<u64>,
    pub resident_children: MetalResidentChildren,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalFeSumcheckPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalNcSumcheckPlan;

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
#[allow(dead_code)]
pub(crate) struct MetalFeSumcheckInputs<'a> {
    pub tables: &'a [u64],
    pub shape: &'a [u64],
    pub mcs_headers: &'a [u64],
    pub mcs_table_indices: &'a [u64],
    pub gammas: &'a [u64],
    pub term_headers: &'a [u64],
    pub term_variables: &'a [u64],
    pub table_count: usize,
    pub coefficient_count: usize,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
#[allow(dead_code)]
pub(crate) struct MetalNcSumcheckInputs<'a> {
    pub eq_table: &'a [u64],
    pub digit_values: &'a [u64],
    pub weights: &'a [u64],
    pub witness_count: usize,
    pub rows: usize,
    pub width: usize,
    pub dense: bool,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalNcFinalState {
    pub eq_beta: KWords,
    pub digit_words: Vec<u64>,
    pub width: usize,
    pub dense: bool,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalSumcheckTrace {
    pub coeffs: Vec<Vec<KWords>>,
    pub challenges: Vec<KWords>,
    pub transcript_state: [u64; 8],
    pub transcript_absorbed: usize,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) struct MetalNcSumcheckTrace {
    pub rounds: MetalSumcheckTrace,
    pub final_state: MetalNcFinalState,
}

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
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

    pub fn ajtai_low_norm_with_plan(
        &self,
        _plan: &MetalAjtaiLowNormPlan,
        _message: &[i8],
    ) -> Result<Vec<u64>, MetalError> {
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
        _rhos: &[i8],
        _witnesses: &[u64],
        _input_count: usize,
        _cols: usize,
    ) -> Result<MetalResidentWitness, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn mix_rlc_witnesses_with_resident_id(
        &self,
        _rhos: &[i8],
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
        _commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn prepare_dec_ring_forms(
        &self,
        _cache: &neo_reductions::superneo_eval::SuperneoEvalCache,
    ) -> Result<MetalDecFormPlan, MetalError> {
        Err(MetalError::Unavailable)
    }

    pub(crate) fn split_dec_base2_with_ring_form_plan(
        &self,
        _parent: &MetalResidentWitness,
        _child_count: usize,
        _plan: &MetalDecFormPlan,
        _chi_words: &[u64],
        _n_eff: usize,
        _commitment_plan: &MetalAjtaiLowNormPlan,
    ) -> Result<MetalDecMaterial, MetalError> {
        Err(MetalError::Unavailable)
    }
}
