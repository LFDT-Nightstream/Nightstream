//! Native Metal prover backend for SuperNeo.
//!
//! The protocol crates remain authoritative for transcript order, proof
//! assembly, and verification. This crate owns Metal device state and the
//! accelerator implementation of the `NifsProverAdapter` execution surface.

pub mod poseidon2;

#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
compile_error!("the `metal` feature requires an Apple target");

use std::time::Duration;

use thiserror::Error;

mod adapter;
pub use adapter::MetalNifsProver;

#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
mod session;
#[cfg(all(target_vendor = "apple", neo_metal_shaders))]
pub use session::{MetalAjtaiLowNormPlan, MetalKxChainPlan, MetalPoseidonUniformPlan, MetalSession};

#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
mod unsupported;
#[cfg(not(all(target_vendor = "apple", neo_metal_shaders)))]
pub use unsupported::{MetalAjtaiLowNormPlan, MetalKxChainPlan, MetalPoseidonUniformPlan, MetalSession};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoldilocksOps {
    pub add: u64,
    pub sub: u64,
    pub mul: u64,
}

/// One quadratic-extension element in the canonical `[c0, c1]` basis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KWords {
    pub c0: u64,
    pub c1: u64,
}

/// Canonical Goldilocks words for one width-eight Poseidon2 state.
pub type PoseidonState = [u64; 8];
/// Canonical Goldilocks words for one four-element Poseidon2 digest.
pub type PoseidonDigest = [u64; 4];

/// Goldilocks multiplication kernels exposed for parity and profiling checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoldilocksMulVariant {
    /// Portable multiplication assembled from 32-bit limbs.
    Limb32,
    /// Metal's native 64-bit multiply-high instruction.
    Native64,
}

/// Poseidon2 scheduling strategies with identical protocol output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoseidonHashVariant {
    /// One Metal thread owns an entire permutation.
    Scalar,
    /// One eight-lane SIMD tile owns an entire permutation.
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

/// Cumulative host/device activity since session creation or the last reset.
///
/// `current_allocated_bytes` is an instantaneous device reading; the other
/// byte counters record explicit allocations and CPU-visible data movement.
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

#[cfg(any(test, all(target_vendor = "apple", neo_metal_shaders)))]
pub(crate) fn oracle_error(error: MetalError) -> neo_reductions::PiCcsError {
    match error {
        MetalError::Shape(reason) => neo_reductions::PiCcsError::InvalidInput(reason.into()),
        error => neo_reductions::PiCcsError::BackendFailure {
            backend: "metal",
            reason: error.to_string(),
        },
    }
}

#[cfg(test)]
#[path = "../tests/unit/oracle_errors.rs"]
mod oracle_errors;
