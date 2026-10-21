//! Proving and terminal row arithmetic. Circuit identity and transcript order stay fixed.

use std::borrow::Borrow;

use neo_ajtai::{nightstream_fprime_setup, Commitment};
use neo_ccs::Mat;
use neo_math::F;

pub(crate) mod crosscheck;
#[cfg(feature = "metal")]
pub(crate) mod metal;
pub(crate) mod paper_exact;

/// The owner's memory ceiling for the CPU engines.
pub(crate) const CPU_MEMORY_LIMIT_BYTES: usize = 16_000_000_000;

/// Arithmetic implementation used for proving and terminal row checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Engine {
    /// Direct paper formulas for reference checks.
    PaperExact,
    /// Optimized native CPU arithmetic.
    #[default]
    Optimized,
    /// Run PaperExact and Optimized in parallel and require identical results.
    /// This has the reference engine's cost and is intended for small circuits.
    Crosscheck,
    /// Apple Metal device arithmetic.
    Metal,
    /// NVIDIA CUDA device arithmetic.
    Cuda,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("PaperExact/Optimized cross-check mismatch: {boundary}")]
    CrosscheckMismatch { boundary: &'static str },
    #[error("{engine:?} engine failed: {reason}")]
    Failure { engine: Engine, reason: String },
    #[error("{engine:?} engine is unavailable: {reason}")]
    Unavailable {
        engine: Engine,
        reason: &'static str,
    },
}

pub(crate) enum Backend {
    #[cfg(feature = "metal")]
    Metal(std::sync::Arc<std::sync::Mutex<neo_prover_metal::MetalRowProver>>),
    PaperExact,
    Optimized,
    Crosscheck,
}

impl Backend {
    pub(crate) fn commit<W: Borrow<Mat<F>>>(&self, witnesses: &[W]) -> Result<Vec<Commitment>, EngineError> {
        let failure = |reason: String| EngineError::Failure {
            engine: self.engine(),
            reason,
        };
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .lock()
                .map_err(|_| failure("device lock was poisoned".into()))?
                .commit_production_prefixes(witnesses)
                .map_err(|error| failure(error.to_string())),
            _ => match witnesses {
                [witness] => nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix(witness.borrow())
                    .map(|commitment| vec![commitment])
                    .map_err(|error| failure(error.to_string())),
                _ => nightstream_fprime_setup::commit_production_signed_unit_prefix_matrices(witnesses)
                    .map_err(|error| failure(error.to_string())),
            },
        }
    }

    pub(crate) fn new(engine: Engine) -> Result<Self, EngineError> {
        match engine {
            Engine::Optimized => Ok(Self::Optimized),
            Engine::PaperExact => Ok(Self::PaperExact),
            Engine::Crosscheck => Ok(Self::Crosscheck),
            #[cfg(feature = "metal")]
            Engine::Metal => neo_prover_metal::MetalRowProver::new()
                .map(|device| Self::Metal(std::sync::Arc::new(std::sync::Mutex::new(device))))
                .map_err(|error| match error {
                    neo_prover_metal::MetalError::Unavailable => EngineError::Unavailable {
                        engine,
                        reason: "Metal requires an Apple device and compiled shaders",
                    },
                    error => EngineError::Failure {
                        engine,
                        reason: error.to_string(),
                    },
                }),
            #[cfg(not(feature = "metal"))]
            Engine::Metal => Err(EngineError::Unavailable {
                engine,
                reason: "build nightstream with the metal feature",
            }),
            Engine::Cuda => Err(EngineError::Unavailable {
                engine,
                #[cfg(feature = "cuda")]
                reason: neo_prover_cuda::CANONICAL_KERNEL_UNAVAILABLE,
                #[cfg(not(feature = "cuda"))]
                reason: "build nightstream with the cuda feature; its canonical kernel is not implemented",
            }),
        }
    }

    /// The owner's memory ceiling for the engine's working storage: 24 GB for
    /// Metal, so a prover can run on a current iPhone, and 16 GB otherwise.
    pub(crate) fn memory_limit_bytes(&self) -> usize {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(_) => neo_prover_metal::BUFFER_LIMIT_BYTES,
            _ => CPU_MEMORY_LIMIT_BYTES,
        }
    }

    /// Device bytes that keeping the commitment key over `columns` columns
    /// takes; `None` for engines that expand no key on a device.
    pub(crate) fn commitment_key_bytes(&self, columns: usize) -> Option<usize> {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(_) => Some(neo_prover_metal::MetalRowProver::commitment_key_bytes(columns)),
            _ => {
                let _ = columns;
                None
            }
        }
    }

    /// Keep the commitment key on the device after the first commitment.
    pub(crate) fn keep_commitment_key(&mut self) {
        #[cfg(feature = "metal")]
        if let Self::Metal(device) = self {
            device
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .keep_commitment_key();
        }
    }

    /// Free a kept commitment key until the next commitment.
    pub(crate) fn release_commitment_key_rows(&self) {
        #[cfg(feature = "metal")]
        if let Self::Metal(device) = self {
            device
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .release_commitment_key_rows();
        }
    }

    /// Where the compression prover's batch loops run. Only Metal has a
    /// device; every other engine runs them on the CPU.
    pub(crate) fn accelerator(&self) -> neo_spartan::Accelerator {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(device) => neo_spartan::Accelerator::new(std::sync::Arc::new(metal::CompressionDevice::new(
                std::sync::Arc::clone(device),
            ))),
            _ => neo_spartan::Accelerator::cpu(),
        }
    }

    pub(crate) fn engine(&self) -> Engine {
        match self {
            #[cfg(feature = "metal")]
            Self::Metal(_) => Engine::Metal,
            Self::Optimized => Engine::Optimized,
            Self::PaperExact => Engine::PaperExact,
            Self::Crosscheck => Engine::Crosscheck,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/engines/parity.rs"]
mod parity;
