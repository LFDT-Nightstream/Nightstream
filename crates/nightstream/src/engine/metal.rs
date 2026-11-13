//! The Metal device session (`neo_prover_metal::MetalRowProver`) behind one
//! lock, and its compression device. The prover keeps the commitment key on
//! the device after the first commitment; the key and every other device
//! buffer stay below `neo_prover_metal::BUFFER_LIMIT_BYTES`.

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

    /// The device session. A panic during a device call poisons the lock and
    /// leaves the session in an unknown state, so that is an engine failure.
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

    /// Π_RLC's parent witness `Σ ρ_i·Z_i`.
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

    /// Free the kept commitment key until the next commitment: layer 1 and
    /// the shrink layer make no Ajtai commitment.
    pub(crate) fn release_commitment_key(&self) -> Result<(), EngineError> {
        self.lock()?.release_commitment_key_rows();
        Ok(())
    }

    pub(crate) fn accelerator(&self) -> neo_spartan::Accelerator {
        neo_spartan::Accelerator::new(Arc::new(CompressionDevice(self.0.clone())))
    }
}

fn poisoned() -> EngineError {
    EngineError::Failure {
        engine: Engine::Metal,
        reason: POISONED.into(),
    }
}

const POISONED: &str = "the device session lock was poisoned";

/// The Metal device of the compression prover: the backend's own session.
struct CompressionDevice(Arc<Mutex<MetalRowProver>>);

impl CompressionDevice {
    /// The device session, or `None` once a panic poisoned it: a poisoned
    /// session is never used again.
    fn lock(&self) -> Option<MutexGuard<'_, MetalRowProver>> {
        self.0.lock().ok()
    }
}

impl neo_spartan::Device for CompressionDevice {
    fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; 4]>>> {
        self.lock()?.merkle_layers(rows, width, fill).ok()
    }

    fn merkle_leaves(&self, values: &[u64], width: usize) -> Option<Vec<[u64; 4]>> {
        self.lock()?.merkle_leaves(values, width).ok()
    }

    fn key_weights(&self, powers: &[[u64; 3]], blocks: usize) -> Option<Vec<[u64; 3]>> {
        self.lock()?.production_key_weights(powers, blocks).ok()
    }

    fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<bool, String> {
        let Some(device) = self.lock() else {
            return Ok(false);
        };
        match device.dft_columns(values, width, generator) {
            Ok(()) => Ok(true),
            // Only execution touches `values`; any earlier failure leaves it
            // unchanged, and the CPU runs the DFT.
            Err(error @ neo_prover_metal::MetalError::Execution(_)) => Err(error.to_string()),
            Err(_) => Ok(false),
        }
    }

    fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Option<Box<dyn neo_spartan::GkrLayer + 'a>> {
        let layer = self.lock()?.gkr_layer(trees, eq).ok()?;
        Some(Box::new(GkrOnDevice {
            device: self.0.clone(),
            layer,
        }))
    }
}

/// One GKR layer's rounds on the backend's device.
struct GkrOnDevice<'a> {
    device: Arc<Mutex<MetalRowProver>>,
    layer: neo_prover_metal::MetalGkrLayer<'a>,
}

impl neo_spartan::GkrLayer for GkrOnDevice<'_> {
    fn message(&mut self, weights: &[[u64; 3]], mu: [u64; 3], degree: usize) -> Result<Vec<[u64; 3]>, String> {
        let device = self.device.lock().map_err(|_| POISONED.to_string())?;
        device
            .gkr_message(&self.layer, weights, mu, degree)
            .map_err(|error| error.to_string())
    }

    fn fold(&mut self, r: [u64; 3]) -> Result<(), String> {
        let device = self.device.lock().map_err(|_| POISONED.to_string())?;
        device
            .gkr_fold(&mut self.layer, r)
            .map_err(|error| error.to_string())
    }
}
