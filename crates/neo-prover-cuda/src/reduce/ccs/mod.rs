//! Device Π_CCS sumcheck oracles: the FE (row) and NC (column) channel
//! tables live on the GPU; each round returns univariate coefficients (tiny
//! D2H) and folds all tables after the host transcript samples the
//! challenge.
//!
//! Owns the device table layouts and round flow. Does not own the sumcheck
//! semantics: tables come from the canonical CPU oracles' snapshots, and
//! every round must stay field-identical to `neo-reductions`.

mod digest;
mod fe;
mod fe_cooperative;
mod fe_oracle_workspace;
mod fe_phase;
mod fe_rows;
mod fe_y_eval;
mod nc;
mod nc_phase;
mod nc_phase_download;
mod nc_workspace;
mod oracle_plan;
mod output;
mod phase;
mod public_challenges;

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use thiserror::Error;

use crate::device::Device;
use crate::kernels::ajtai::{load_ajtai_kernels, AjtaiKernelModule};
use crate::kernels::csr::{load_csr_kernels, CsrKernelModule};
use crate::kernels::pi_ccs_digest::{load_ccs_digest_kernels, CcsDigestKernelModule};
use crate::kernels::pi_ccs_fe::{load_fe_kernels, FeKernelModule};
use crate::kernels::pi_ccs_nc::{load_nc_kernels, NcKernelModule};
use crate::kernels::pi_ccs_output::{load_ccs_output_kernels, CcsOutputKernelModule};
use crate::kernels::pi_ccs_tail::{load_fe_tail_kernels, FeTailKernelModule};
use crate::kernels::pi_rlc::{load_rlc_kernels, RlcKernelModule};
use crate::kernels::poseidon2::{load_poseidon2_kernels, Poseidon2KernelModule};
use crate::kernels::sumcheck_common::{load_sumcheck_common, SumcheckCommonModule};
use crate::transcript::upload_round_constants;

pub(crate) use digest::accumulator_digest_from_surfaces;
pub use digest::{DevicePiCcsOutputsDigest, PiCcsOutputDigestShell};
pub use fe::{DeviceAjtaiYEval, DeviceFeBackend, DeviceFeOracle, DeviceFeTailRound};
pub(crate) use fe_oracle_workspace::FeOracleWorkspace;
pub(crate) use fe_phase::{FePhaseGraphKey, FePhaseWorkspace, PiCcsPhaseGraphKey};
pub(crate) use fe_rows::DeviceFeRowProofLogArchive;
pub use nc::{DeviceNcBackend, DeviceNcFinalState, DeviceNcOracle};
pub(crate) use nc::{NcPhaseGraphKey, PendingNcPhase};
pub(crate) use nc_phase::NcPhaseWorkspace;
pub(crate) use nc_phase_download::NcPhaseSummary;
pub(crate) use nc_workspace::NcOracleWorkspace;
pub use output::DevicePiCcsKSurfaces;
pub(crate) use output::DevicePublicX;
pub use phase::DevicePiCcsPhaseBackend;
pub(crate) use phase::DevicePiCcsProofLogExporter;
pub(crate) use public_challenges::DevicePublicChallenges;

#[derive(Debug, Error)]
pub enum CcsDeviceError {
    #[error("CUDA driver error: {0:?}")]
    Driver(cuda_core::DriverError),
    #[error("kernel module load failed: {0:?}")]
    ModuleLoad(cuda_host::EmbeddedModuleError),
    #[error("unsupported oracle shape: {0}")]
    Shape(&'static str),
}

impl From<cuda_core::DriverError> for CcsDeviceError {
    fn from(e: cuda_core::DriverError) -> Self {
        Self::Driver(e)
    }
}

/// The three Π_CCS kernel modules, loaded once per device and shared by
/// every prove (module loads cost ~1.6ms each).
pub struct SumcheckKernels {
    pub(crate) fe: FeKernelModule,
    pub(crate) nc: NcKernelModule,
    pub(crate) common: SumcheckCommonModule,
    /// Ajtai-tail FE kernels used by whole-phase FE scheduling.
    pub(crate) tail: FeTailKernelModule,
    /// Ring mat-vec kernels, shared with the commit / Π_DEC paths — the
    /// Ajtai-phase `Y_eval` is the same forms × witness-planes product.
    pub(crate) ring: AjtaiKernelModule,
    /// CSR-table kernels (forms build, row tables, weighted eval table).
    pub(crate) csr: CsrKernelModule,
    /// Poseidon2 transcript kernels/constants for device-driven challenges.
    pub(crate) poseidon: Poseidon2KernelModule,
    pub(crate) poseidon_rc: DeviceBuffer<u64>,
    /// Π_RLC output-surface kernels.
    pub(crate) rlc: RlcKernelModule,
    /// Pi_CCS-to-Pi_RLC device output-surface packing kernels.
    pub(crate) output: CcsOutputKernelModule,
    /// Pi_CCS output-digest preimage kernels.
    pub(crate) digest: CcsDigestKernelModule,
}

impl SumcheckKernels {
    /// The shared ring-algebra module, for callers driving ring launches
    /// directly (Π_RLC mix, Π_DEC).
    pub fn ring(&self) -> &AjtaiKernelModule {
        &self.ring
    }

    pub fn rlc(&self) -> &RlcKernelModule {
        &self.rlc
    }
}

impl SumcheckKernels {
    pub fn load(device: &Device) -> Result<Self, CcsDeviceError> {
        let ctx: &Arc<cuda_core::CudaContext> = device.ctx();
        Ok(Self {
            fe: load_fe_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            nc: load_nc_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            common: load_sumcheck_common(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            tail: load_fe_tail_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            ring: load_ajtai_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            csr: load_csr_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            poseidon: load_poseidon2_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            poseidon_rc: upload_round_constants(device)?,
            rlc: load_rlc_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            output: load_ccs_output_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
            digest: load_ccs_digest_kernels(ctx).map_err(CcsDeviceError::ModuleLoad)?,
        })
    }
}
