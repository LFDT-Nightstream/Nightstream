use std::borrow::Borrow;

use neo_ajtai::{nightstream_fprime_setup, Commitment};
use neo_ccs::{CcsStructure, Mat, V1_2Evaluations};
use neo_math::{F, K};
use neo_reductions::optimized_engine::PaperJointOracleBackend;
use neo_reductions::superneo_eval::{eval_real_v1_2_openings_from_rows_reusing, MatrixRows, SuperneoZBlocks};
use neo_reductions::PiCcsError;

#[cfg(feature = "metal")]
pub(crate) mod metal;

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

#[derive(Clone, Debug, thiserror::Error)]
pub enum EngineError {
    #[error("{engine:?} engine failed: {reason}")]
    Failure { engine: Engine, reason: String },
    #[error("{engine:?} engine is unavailable: {reason}")]
    Unavailable {
        engine: Engine,
        reason: &'static str,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("fixed-key commitment input: {0}")]
    Input(String),
    #[error(transparent)]
    Engine(#[from] EngineError),
}

pub(crate) enum Backend {
    Optimized,
    #[cfg(feature = "metal")]
    Metal(metal::Device),
}

impl Backend {
    pub(crate) fn new(engine: Engine) -> Result<Self, EngineError> {
        match engine {
            Engine::Optimized => Ok(Self::Optimized),
            #[cfg(feature = "metal")]
            Engine::Metal => metal::Device::new().map(Self::Metal),
            #[cfg(not(feature = "metal"))]
            Engine::Metal => Err(EngineError::Unavailable {
                engine,
                reason: "build nightstream with the metal feature",
            }),
            Engine::Cuda => Err(EngineError::Unavailable {
                engine,
                reason: "the CUDA engine is not connected to this crate",
            }),
        }
    }

    pub(crate) fn engine(&self) -> Engine {
        match self {
            Self::Optimized => Engine::Optimized,
            #[cfg(feature = "metal")]
            Self::Metal(_) => Engine::Metal,
        }
    }

    pub(crate) fn commit<W: Borrow<Mat<F>>>(&self, witnesses: &[W]) -> Result<Vec<Commitment>, CommitError> {
        match self {
            Self::Optimized => match witnesses {
                [witness] => nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix(witness.borrow())
                    .map(|commitment| vec![commitment])
                    .map_err(|error| CommitError::Input(error.to_string())),
                _ => nightstream_fprime_setup::commit_production_signed_unit_prefix_matrices(witnesses)
                    .map_err(|error| CommitError::Input(error.to_string())),
            },
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.commit(witnesses),
        }
    }

    pub(crate) fn oracle(&self, packed: bool) -> Result<Oracle<'_>, EngineError> {
        if packed {
            #[cfg(feature = "metal")]
            if let Self::Metal(device) = self {
                return Ok(Oracle::Packed(device.clone()));
            }
            return Ok(Oracle::Cpu(std::marker::PhantomData));
        }
        Ok(match self {
            Self::Optimized => Oracle::Cpu(std::marker::PhantomData),
            #[cfg(feature = "metal")]
            Self::Metal(device) => Oracle::Metal(device.lock()?),
        })
    }

    pub(crate) fn child_openings(
        &self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        children: &[Mat<F>],
        point: &[K],
        width: usize,
    ) -> Result<Vec<V1_2Evaluations<K>>, DeviceError> {
        match self {
            Self::Optimized => {
                let blocks = children
                    .iter()
                    .map(|witness| SuperneoZBlocks::from_witness_mat(witness, width))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(DeviceError::Rejected)?;
                eval_real_v1_2_openings_from_rows_reusing(rows, point, &blocks, workspace_bytes, Vec::new(), None, None)
                    .map_err(DeviceError::Rejected)
            }
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .lock()?
                .child_openings(rows, workspace_bytes, children, point, width)
                .map_err(|error| self.classify(error)),
        }
    }

    pub(crate) fn first_failing_row(
        &self,
        rows: &dyn MatrixRows,
        workspace_bytes: usize,
        structure: &CcsStructure<F>,
        witness: &Mat<F>,
    ) -> Option<Result<Option<usize>, DeviceError>> {
        match self {
            Self::Optimized => {
                let _ = (rows, workspace_bytes, structure, witness);
                None
            }
            #[cfg(feature = "metal")]
            Self::Metal(device) => Some(match device.lock() {
                Ok(prover) => prover
                    .first_unsatisfied_row(rows, workspace_bytes, structure, witness)
                    .map_err(|error| self.classify(error)),
                Err(error) => Err(error.into()),
            }),
        }
    }

    pub(crate) fn classify(&self, error: PiCcsError) -> DeviceError {
        match error {
            PiCcsError::BackendFailure { reason, .. } => DeviceError::Engine(EngineError::Failure {
                engine: self.engine(),
                reason,
            }),
            error => DeviceError::Rejected(error),
        }
    }
}

pub(crate) enum Oracle<'a> {
    Cpu(std::marker::PhantomData<&'a ()>),
    #[cfg(feature = "metal")]
    Packed(metal::Device),
    #[cfg(feature = "metal")]
    Metal(std::sync::MutexGuard<'a, neo_prover_metal::MetalRowProver>),
}

impl Oracle<'_> {
    pub(crate) fn backend(&mut self) -> Option<&mut dyn PaperJointOracleBackend> {
        match self {
            Self::Cpu(_) => None,
            #[cfg(feature = "metal")]
            Self::Metal(prover) => Some(&mut **prover),
            #[cfg(feature = "metal")]
            Self::Packed(device) => Some(device),
        }
    }
}

#[derive(Debug)]
pub(crate) enum DeviceError {
    Engine(EngineError),
    Rejected(PiCcsError),
}

impl From<EngineError> for DeviceError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

#[cfg(all(test, feature = "metal"))]
#[path = "../../tests/unit/packed_engine.rs"]
mod packed_engine_tests;
