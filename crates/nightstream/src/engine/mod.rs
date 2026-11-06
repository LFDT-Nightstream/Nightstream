//! The engine that proves and verifies. `Engine::Optimized` runs on the CPU;
//! `Engine::Metal` (feature `metal`) runs the commitments, the PiCCS round
//! oracle, the PiDEC openings and the compression loops on the device. Every
//! device result equals the CPU result; the transcript order, the proof
//! assembly and every public check stay on the host.

use std::borrow::Borrow;

use neo_ajtai::{nightstream_fprime_setup, Commitment};
use neo_ccs::{CcsStructure, Mat, V1_2Evaluations};
use neo_math::{F, K};
use neo_reductions::optimized_engine::PaperJointOracleBackend;
use neo_reductions::superneo_eval::{eval_real_v1_2_openings_from_rows, MatrixRows, SuperneoZBlocks};
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

/// A fixed-key commitment failed: a witness is not a signed-unit matrix of
/// the key's shape, or the engine failed.
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("fixed-key commitment input: {0}")]
    Input(String),
    #[error(transparent)]
    Engine(#[from] EngineError),
}

/// The selected engine's device state.
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
                reason: "the CUDA engine is not connected to nightstream yet",
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

    /// The production-key commitment of each signed-unit witness.
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

    /// The engine's PiCCS round oracle, held for one PiCCS proof.
    pub(crate) fn oracle(&self) -> Result<Oracle<'_>, EngineError> {
        Ok(match self {
            Self::Optimized => Oracle::Cpu(std::marker::PhantomData),
            #[cfg(feature = "metal")]
            Self::Metal(device) => Oracle::Metal(device.lock()?),
        })
    }

    /// The `Eval_K` and `Eval_A` openings at `point` of each PiDEC child.
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
                eval_real_v1_2_openings_from_rows(rows, point, &blocks, workspace_bytes).map_err(DeviceError::Rejected)
            }
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .lock()?
                .child_openings(rows, workspace_bytes, children, point, width)
                .map_err(|error| self.classify(error)),
        }
    }

    /// The first row of `structure` over `rows` that the signed-unit
    /// `witness` fails, checked on the device; `None` when the engine checks
    /// rows on the CPU.
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

    /// Free a kept commitment key until the next commitment.
    pub(crate) fn release_commitment_key(&self) -> Result<(), EngineError> {
        match self {
            Self::Optimized => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.release_commitment_key(),
        }
    }

    /// Where the compression prover's batch loops run.
    pub(crate) fn accelerator(&self) -> neo_spartan::Accelerator {
        match self {
            Self::Optimized => neo_spartan::Accelerator::cpu(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.accelerator(),
        }
    }

    /// A device failure judged nothing, so it is an engine failure; any other
    /// error judged the input.
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

/// A held PiCCS round oracle: the CPU's, or the device session's.
pub(crate) enum Oracle<'a> {
    Cpu(std::marker::PhantomData<&'a ()>),
    #[cfg(feature = "metal")]
    Metal(std::sync::MutexGuard<'a, neo_prover_metal::MetalRowProver>),
}

impl Oracle<'_> {
    /// The device backend, or `None` for the CPU oracle.
    pub(crate) fn backend(&mut self) -> Option<&mut dyn PaperJointOracleBackend> {
        match self {
            Self::Cpu(_) => None,
            #[cfg(feature = "metal")]
            Self::Metal(prover) => Some(&mut **prover),
        }
    }
}

/// An error of an engine call: the engine failed, or the input was rejected.
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
