//! CUDA availability wrapper for the selected one-joint NIFS protocol.
//!
//! CUDA does not yet own a one-joint `PaddedRowIdentity` kernel. The adapter
//! therefore uses the canonical host prover. This prevents accelerator
//! selection from changing protocol messages or proof bytes.

use std::sync::Arc;

use cuda_core::CudaContext;
use neo_fold_clean::paper::nifs::{
    AcceleratorCrosscheckNifsProver, Error, NifsProverAdapter, NifsProverOutput, NifsProverRequest,
    OptimizedCpuNifsProver, OptimizedNifsProverAdapter,
};

/// CUDA-selected prover with canonical one-joint protocol execution.
pub struct CudaNifsProver {
    _context: Arc<CudaContext>,
    cpu: OptimizedCpuNifsProver,
}

impl CudaNifsProver {
    /// Open CUDA device zero and construct a canonical NIFS prover.
    pub fn new() -> Result<Self, Error> {
        let context = CudaContext::new(0).map_err(|error| Error::BackendFailure {
            backend: "cuda",
            phase: "initialization",
            reason: error.to_string(),
        })?;
        Ok(Self::new_on_context(context))
    }

    /// Construct a prover in a caller-owned CUDA context.
    pub fn new_on_context(context: Arc<CudaContext>) -> Self {
        Self {
            _context: context,
            cpu: OptimizedCpuNifsProver,
        }
    }

    /// Wrap this complete CUDA selection in an optimized-CPU NIFS crosscheck.
    pub fn crosschecked(self) -> AcceleratorCrosscheckNifsProver<Self> {
        AcceleratorCrosscheckNifsProver::new(self)
    }
}

impl NifsProverAdapter for CudaNifsProver {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error> {
        self.cpu.prove(request)
    }
}

impl OptimizedNifsProverAdapter for CudaNifsProver {}
