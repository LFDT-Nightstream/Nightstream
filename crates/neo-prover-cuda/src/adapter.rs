//! CUDA availability boundary for the selected one-joint NIFS protocol.
//!
//! CUDA does not yet own a one-joint `PaddedRowIdentity` kernel. Construction
//! fails until that kernel exists. This module never reports CPU work as CUDA
//! work.

use std::sync::Arc;

use cuda_core::CudaContext;
use neo_fold_clean::paper::nifs::{Error, NifsProof, NifsProverAdapter, NifsProverRequest};
use neo_fold_clean::RunningInstance;

const MISSING_KERNEL: &str = "the canonical one-joint CUDA NIFS kernel is not implemented";

/// CUDA prover for the canonical one-joint protocol.
///
/// Construction returns [`Error::BackendUnavailable`] until the device kernel
/// implements the same protocol as the optimized CPU and PaperExact provers.
pub struct CudaNifsProver {
    _context: Arc<CudaContext>,
}

impl CudaNifsProver {
    /// Open CUDA device zero and construct the CUDA NIFS prover.
    pub fn new() -> Result<Self, Error> {
        require_kernel()?;
        let context = CudaContext::new(0).map_err(|error| Error::BackendFailure {
            backend: "cuda",
            phase: "initialization",
            reason: error.to_string(),
        })?;
        Ok(Self { _context: context })
    }

    /// Construct the CUDA NIFS prover in a caller-owned context.
    pub fn new_on_context(context: Arc<CudaContext>) -> Result<Self, Error> {
        require_kernel()?;
        Ok(Self { _context: context })
    }
}

impl NifsProverAdapter for CudaNifsProver {
    fn prove(&mut self, _request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), Error> {
        Err(unavailable())
    }
}

fn require_kernel() -> Result<(), Error> {
    Err(unavailable())
}

fn unavailable() -> Error {
    Error::BackendUnavailable {
        backend: "cuda",
        reason: MISSING_KERNEL,
    }
}
