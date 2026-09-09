//! Prover backend selection and reusable accelerator-session ownership.
//!
//! The protocol and proof format remain backend-independent. This module owns
//! only the prover-side choice between optimized CPU, PaperExact, Metal, and
//! CUDA.

use neo_fold_clean::paper::nifs::{NifsProverAdapter, OptimizedCpuNifsProver, PaperExactNifsProver};

use crate::nebula::{prove_with_nifs_adapter, WasmNebulaError, WasmNebulaPreprocessing, WasmNebulaProof};
use crate::WasmVmStep;

/// Prover implementation selected for the next proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmProverBackend {
    CpuOptimized,
    PaperExact,
    Cuda,
    Metal,
}

impl WasmProverBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOptimized => "cpu-optimized",
            Self::PaperExact => "paper-exact",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

/// Reusable WASM prover with one owned reference, CPU, or accelerator session.
///
/// Use [`WasmProver::auto`] for ordinary proving. Use an explicit constructor
/// for benchmarks, diagnostics, or deployments that require one backend.
///
/// All backends produce the canonical [`WasmNebulaProof`]. Verification does
/// not select a backend.
///
/// ```ignore
/// let proof = neo_wasm::prove(&prep, &trace)?;
/// neo_wasm::verify(&prep, &proof, final_state)?;
///
/// let mut cpu = neo_wasm::WasmProver::cpu_optimized();
/// let proof = cpu.prove(&prep, &trace)?;
/// # Ok::<(), neo_wasm::WasmNebulaError>(())
/// ```
pub struct WasmProver {
    backend: WasmProverBackend,
    adapter: Box<dyn NifsProverAdapter>,
    automatic: bool,
    fallback_reason: Option<String>,
}

impl WasmProver {
    /// Select the best enabled accelerator and fall back to optimized CPU.
    ///
    /// CUDA has priority when the `cuda` feature is enabled. Metal is next on
    /// Apple targets when the default `metal` feature is enabled.
    pub fn auto() -> Self {
        #[cfg(feature = "cuda")]
        match neo_prover_cuda::CudaNifsProver::new() {
            Ok(cuda) => return Self::new(WasmProverBackend::Cuda, cuda, true),
            Err(cuda_error) => {
                return Self::auto_without_cuda(Some(format!("CUDA initialization failed: {cuda_error}")));
            }
        }

        #[cfg(not(feature = "cuda"))]
        Self::auto_without_cuda(None)
    }

    fn auto_without_cuda(fallback_reason: Option<String>) -> Self {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        match neo_prover_metal::MetalNifsProver::new() {
            Ok(metal) => return Self::new(WasmProverBackend::Metal, metal, true),
            Err(metal_error) => {
                let mut cpu = Self::cpu_optimized();
                cpu.automatic = true;
                cpu.fallback_reason = Some(match fallback_reason {
                    Some(cuda_reason) => {
                        format!("{cuda_reason}; Metal initialization failed: {metal_error}")
                    }
                    None => format!("Metal initialization failed: {metal_error}"),
                });
                return cpu;
            }
        }

        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        {
            let mut cpu = Self::cpu_optimized();
            cpu.automatic = true;
            cpu.fallback_reason = fallback_reason;
            cpu
        }
    }

    /// Create a reusable optimized CPU prover.
    pub fn cpu_optimized() -> Self {
        Self::new(WasmProverBackend::CpuOptimized, OptimizedCpuNifsProver, false)
    }

    /// Create the direct PaperExact reference prover.
    ///
    /// PaperExact has exponential cost and is intended for small protocol
    /// checks. Automatic selection never chooses it.
    pub fn paper_exact() -> Self {
        Self::new(WasmProverBackend::PaperExact, PaperExactNifsProver, false)
    }

    /// Create a reusable Metal prover or return an explicit availability error.
    pub fn metal() -> Result<Self, WasmNebulaError> {
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        {
            let metal = neo_prover_metal::MetalNifsProver::new().map_err(|error| {
                WasmNebulaError::ProverBackendUnavailable {
                    backend: WasmProverBackend::Metal.as_str(),
                    reason: error.to_string(),
                }
            })?;
            Ok(Self::new(WasmProverBackend::Metal, metal, false))
        }

        #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
        Err(WasmNebulaError::ProverBackendUnavailable {
            backend: WasmProverBackend::Metal.as_str(),
            reason: "the Metal backend requires an Apple target and the `metal` feature".to_owned(),
        })
    }

    /// Create a reusable CUDA prover or return an explicit availability error.
    pub fn cuda() -> Result<Self, WasmNebulaError> {
        #[cfg(feature = "cuda")]
        {
            let cuda =
                neo_prover_cuda::CudaNifsProver::new().map_err(|error| WasmNebulaError::ProverBackendUnavailable {
                    backend: WasmProverBackend::Cuda.as_str(),
                    reason: error.to_string(),
                })?;
            Ok(Self::new(WasmProverBackend::Cuda, cuda, false))
        }

        #[cfg(not(feature = "cuda"))]
        Err(WasmNebulaError::ProverBackendUnavailable {
            backend: WasmProverBackend::Cuda.as_str(),
            reason: "the CUDA backend is not enabled in this build".to_owned(),
        })
    }

    /// Return the backend that the next proof will use.
    ///
    /// Automatic fallback updates this value to `CpuOptimized`.
    pub fn backend(&self) -> WasmProverBackend {
        self.backend
    }

    /// Why automatic selection moved to CPU, if it did.
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// Prove one trace and retain reusable backend state for the next call.
    pub fn prove(
        &mut self,
        prep: &WasmNebulaPreprocessing,
        trace: &[WasmVmStep],
    ) -> Result<WasmNebulaProof, WasmNebulaError> {
        let result = prove_with_nifs_adapter(prep, self.adapter.as_mut(), trace);
        if self.automatic && self.backend != WasmProverBackend::CpuOptimized {
            if let Err(error) = &result {
                if is_backend_error(error) {
                    self.fallback_to_cpu(format!("{} proving failed: {error}", self.backend.as_str()));
                    return prove_with_nifs_adapter(prep, self.adapter.as_mut(), trace);
                }
            }
        }
        result
    }

    fn new(backend: WasmProverBackend, adapter: impl NifsProverAdapter + 'static, automatic: bool) -> Self {
        Self {
            backend,
            adapter: Box::new(adapter),
            automatic,
            fallback_reason: None,
        }
    }

    fn fallback_to_cpu(&mut self, reason: String) {
        self.backend = WasmProverBackend::CpuOptimized;
        self.adapter = Box::new(OptimizedCpuNifsProver);
        self.fallback_reason = Some(reason);
    }
}

fn is_backend_error(error: &WasmNebulaError) -> bool {
    match error {
        WasmNebulaError::Chain(error) => match error {
            neo_fold_clean::frontends::nebula::NebulaFPrimeChainError::Lifecycle(error) => {
                is_lifecycle_backend_error(error)
            }
            _ => false,
        },
        WasmNebulaError::Lifecycle(error) => is_lifecycle_backend_error(error),
        _ => false,
    }
}

fn is_lifecycle_backend_error(error: &neo_fold_clean::lifecycle::Error) -> bool {
    use neo_fold_clean::{
        engine::optimized,
        lifecycle,
        paper::{construction2, nifs, pi_ccs, pi_dec, pi_rlc},
    };

    let lifecycle::Error::Construction2(construction2::Error::Nifs(error)) = error else {
        return false;
    };
    match error {
        nifs::Error::BackendUnavailable { .. } | nifs::Error::BackendFailure { .. } => true,
        nifs::Error::PiCcs(pi_ccs::Error::Engine(error))
        | nifs::Error::PiDec(pi_dec::Error::Engine(error))
        | nifs::Error::PiRlc(pi_rlc::Error::Engine(error)) => matches!(
            error,
            optimized::Error::Reductions(neo_reductions::PiCcsError::BackendFailure { .. })
        ),
        _ => false,
    }
}

#[cfg(test)]
#[path = "../tests/unit/prover_errors.rs"]
mod prover_errors;
