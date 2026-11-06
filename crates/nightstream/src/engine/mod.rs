//! The engine that proves and verifies. Steps prove and verify on the CPU
//! path (`Engine::Optimized`); the device engines are unavailable.

/// Arithmetic implementation used for proving and verifying.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Engine {
    /// Optimized native CPU arithmetic.
    #[default]
    Optimized,
    /// Apple Metal device arithmetic.
    Metal,
    /// NVIDIA CUDA device arithmetic.
    Cuda,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("{engine:?} engine is unavailable: {reason}")]
    Unavailable {
        engine: Engine,
        reason: &'static str,
    },
}
