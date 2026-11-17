use std::borrow::Borrow;
use std::sync::{Arc, Mutex, MutexGuard};

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use neo_prover_metal::MetalRowProver;
use neo_reductions::PiCcsError;

use super::{CommitError, DeviceError, Engine, EngineError};

#[derive(Clone)]
pub(crate) struct Device(Arc<Mutex<MetalRowProver>>);

impl Device {
    pub(crate) fn new() -> Result<Self, EngineError> {
        let mut prover = MetalRowProver::new().map_err(|error| EngineError::Failure {
            engine: Engine::Metal,
            reason: error.to_string(),
        })?;
        prover.keep_commitment_key();
        Ok(Self(Arc::new(Mutex::new(prover))))
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, MetalRowProver>, EngineError> {
        self.0.lock().map_err(|_| poisoned())
    }

    pub(crate) fn commit<W: Borrow<Mat<F>>>(&self, witnesses: &[W]) -> Result<Vec<Commitment>, CommitError> {
        self.lock()?
            .commit_production_prefixes(witnesses)
            .map_err(|error| match error {
                PiCcsError::BackendFailure { reason, .. } => EngineError::Failure {
                    engine: Engine::Metal,
                    reason,
                }
                .into(),
                error => CommitError::Input(error.to_string()),
            })
    }

    pub(crate) fn mix_rlc_witnesses(&self, rhos: &[Mat<F>], witnesses: &[&Mat<F>]) -> Result<Mat<F>, DeviceError> {
        self.lock()?
            .mix_rlc_witnesses(rhos, witnesses)
            .map_err(|error| match error {
                PiCcsError::BackendFailure { reason, .. } => DeviceError::Engine(EngineError::Failure {
                    engine: Engine::Metal,
                    reason,
                }),
                error => DeviceError::Rejected(error),
            })
    }
}

fn poisoned() -> EngineError {
    EngineError::Failure {
        engine: Engine::Metal,
        reason: POISONED.into(),
    }
}

const POISONED: &str = "the device session lock was poisoned";
