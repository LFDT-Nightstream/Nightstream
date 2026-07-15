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
pub use adapter::{
    MetalAjtaiProfile, MetalFeProfile, MetalFreshProfile, MetalNcProfile, MetalNifsProfile, MetalNifsProver,
    MetalPiCcsProfile, MetalPiDecProfile, MetalPiRlcProfile, MetalResidencyProfile,
};
mod fold_output;
mod sumcheck;

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
mod session;
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
pub use session::{MetalAjtaiLowNormPlan, MetalKxChainPlan, MetalPoseidonUniformPlan, MetalSession};
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
pub(crate) use session::{
    MetalAjtaiRingForms, MetalDecFormPlan, MetalDecPublicProjection, MetalDeferredEvalTable, MetalDeferredMcsRowTables,
    MetalFeOraclePlan, MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalFeTableInput, MetalNcDigitInput,
    MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan, MetalNcSumcheckTrace, MetalResidentWitness,
    MetalResidentWitnessSnapshot, MetalSumcheckTrace, MetalWitnessMasks,
};

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
mod unsupported;
#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub use unsupported::{MetalAjtaiLowNormPlan, MetalKxChainPlan, MetalPoseidonUniformPlan, MetalSession};
#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) use unsupported::{
    MetalAjtaiRingForms, MetalDecFormPlan, MetalDecPublicProjection, MetalDeferredEvalTable, MetalDeferredMcsRowTables,
    MetalFeOraclePlan, MetalFeSumcheckInputs, MetalFeSumcheckPlan, MetalFeTableInput, MetalNcDigitInput,
    MetalNcFinalState, MetalNcSumcheckInputs, MetalNcSumcheckPlan, MetalNcSumcheckTrace, MetalResidentWitness,
    MetalResidentWitnessSnapshot, MetalSumcheckTrace, MetalWitnessMasks,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MetalAjtaiYProfile {
    pub seeded_build: Duration,
    pub device_eval: Duration,
    pub tensor_gpu: Duration,
    pub form_gpu: Duration,
    pub tail_gpu: Duration,
    pub seeded_patch_entries: usize,
    pub seeded_patch_bytes: usize,
    pub form_blocks: usize,
    pub form_bytes: usize,
    pub explicit_coefficients: usize,
    pub signed_unit_coefficients: usize,
    pub explicit_form_list_histogram: [usize; 8],
    pub max_explicit_form_list_entries: usize,
    pub parallel_form_lists: usize,
    pub parallel_form_entries: usize,
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
