//! Prover engine selection. Circuit identity and verification do not depend on it.

pub(crate) mod crosscheck;
#[cfg(feature = "metal")]
pub(crate) mod metal;
pub(crate) mod paper_exact;

/// Arithmetic implementation used for an active C/R/D fold.
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

pub(crate) enum Prover {
    #[cfg(feature = "metal")]
    Metal(Box<std::sync::Mutex<neo_prover_metal::MetalRowProver>>),
    PaperExact,
    Optimized,
    Crosscheck,
}

impl Prover {
    pub(crate) fn new(engine: Engine) -> Result<Self, EngineError> {
        match engine {
            Engine::Optimized => Ok(Self::Optimized),
            Engine::PaperExact => Ok(Self::PaperExact),
            Engine::Crosscheck => Ok(Self::Crosscheck),
            #[cfg(feature = "metal")]
            Engine::Metal => neo_prover_metal::MetalRowProver::new()
                .map(|device| Self::Metal(Box::new(std::sync::Mutex::new(device))))
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
