//! Prover backend selection and reusable accelerator-session ownership.
//!
//! The protocol and proof format remain backend-independent. This module owns
//! only the prover-side choice between CPU, Metal, and CUDA.

use neo_fold_clean::paper::nifs::{CpuNifsProver, NifsProverAdapter};

use crate::nebula::{prove_with_nifs_adapter, WasmNebulaError, WasmNebulaPreprocessing, WasmNebulaProof};
use crate::WasmVmStep;

/// Prover implementation selected for the next proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasmProverBackend {
    Cpu,
    Metal,
    Cuda,
}

impl WasmProverBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

/// Reusable WASM prover with one owned CPU or accelerator session.
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
/// let mut cpu = neo_wasm::WasmProver::cpu();
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
    /// Select the best enabled accelerator and fall back to CPU when needed.
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
                let mut cpu = Self::cpu();
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
            let mut cpu = Self::cpu();
            cpu.automatic = true;
            cpu.fallback_reason = fallback_reason;
            cpu
        }
    }

    /// Create a reusable canonical CPU prover.
    pub fn cpu() -> Self {
        Self::new(WasmProverBackend::Cpu, CpuNifsProver, false)
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
    /// Automatic fallback updates this value to `Cpu`.
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
        if self.automatic && !self.supports(prep) {
            self.fallback_to_cpu("the selected accelerator does not support this proof shape".to_owned());
        }

        let result = prove_with_nifs_adapter(prep, self.adapter.as_mut(), trace);
        if self.automatic && self.backend != WasmProverBackend::Cpu {
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

    fn supports(&self, prep: &WasmNebulaPreprocessing) -> bool {
        if self.backend != WasmProverBackend::Cuda {
            return true;
        }
        !neo_fold_clean::paper::construction2::running::uses_pending_accumulator_family(
            prep.inner().relation().structure(),
        )
    }

    fn fallback_to_cpu(&mut self, reason: String) {
        self.backend = WasmProverBackend::Cpu;
        self.adapter = Box::new(CpuNifsProver);
        self.fallback_reason = Some(reason);
    }
}

fn is_backend_error(error: &WasmNebulaError) -> bool {
    match error {
        WasmNebulaError::Chain(error) => match error {
            neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeChainError::Lifecycle(error) => {
                is_lifecycle_backend_error(error)
            }
            _ => false,
        },
        WasmNebulaError::Lifecycle(error) => is_lifecycle_backend_error(error),
        _ => false,
    }
}

fn is_lifecycle_backend_error(error: &neo_fold_clean::lifecycle::Error) -> bool {
    matches!(
        error,
        neo_fold_clean::lifecycle::Error::Construction2(neo_fold_clean::paper::construction2::Error::Nifs(
            neo_fold_clean::paper::nifs::Error::BackendUnavailable { .. }
                | neo_fold_clean::paper::nifs::Error::BackendFailure { .. }
        ))
    )
}
