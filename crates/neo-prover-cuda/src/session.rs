//! Device-resident session state: everything that persists across folds.
//!
//! Owns the CUDA device, kernel modules, the Ajtai PP upload, the static
//! CSR matrices, scratch, and the retained child planes. Owns no protocol
//! flow — the adapter orchestrates; the reductions compute.

use std::sync::Arc;

use cuda_core::{CudaContext, DeviceBuffer};
use neo_ajtai::{AjtaiSModule, Commitment, PP};
use neo_fold_clean::paper::nifs::Error;
use neo_fold_clean::RunningInstance;
use neo_math::D;
use neo_reductions::optimized_engine::OptimizedStructureCache;

use crate::commit::DeviceAjtai;
use crate::device::Device;
use crate::fold_output::{DeviceCommitments, DeviceFoldOutput};
use crate::kernels::ajtai::RingMatVecScratch;
use crate::reduce::ccs::{FeOracleWorkspace, FePhaseWorkspace, NcOracleWorkspace, NcPhaseWorkspace, SumcheckKernels};
use crate::reduce::dec::DeviceDec;
use crate::ring_forms::{DeviceBarMatrices, DeviceRowMatrices};

pub(crate) fn backend_unavailable(reason: &'static str) -> Error {
    Error::BackendUnavailable {
        backend: "cuda",
        reason,
    }
}

pub struct DeviceSession {
    pub(crate) device: Device,
    pub(crate) ajtai: Option<DeviceAjtai>,
    /// The exact host PP the device copy was uploaded from. Holding the Arc
    /// keeps its allocation alive, so pointer identity below cannot alias a
    /// different PP of the same shape.
    pub(crate) ajtai_source: Option<Arc<PP<neo_math::Rq>>>,
    pub(crate) kernels: Option<SumcheckKernels>,
    pub(crate) dec: Option<DeviceDec>,
    /// Π_CCS Ajtai-phase bar matrices, persisted across folds (the upload is
    /// the dominant one-time cost; the fingerprint re-check is per fold).
    pub(crate) bar_matrices: Option<DeviceBarMatrices>,
    /// Row-major orig CSR for the f-var row tables, same lifetime.
    pub(crate) row_matrices: Option<DeviceRowMatrices>,
    /// Whole-FE trace buffers retained across folds so CUDA graph capture can
    /// eventually reuse stable addresses for logs, points, and tail scratch.
    pub(crate) fe_phase_workspace: Option<FePhaseWorkspace>,
    /// FE row-oracle buffers retained across folds so a captured graph can
    /// reuse stable row-table and round-scratch addresses.
    pub(crate) fe_oracle_workspace: Option<FeOracleWorkspace>,
    /// NC column-oracle buffers retained across folds. Whole-Pi_CCS graph
    /// capture needs these addresses to become as stable as the FE side.
    pub(crate) nc_oracle_workspace: Option<NcOracleWorkspace>,
    /// NC transcript/prolog/log buffers retained across folds. These are
    /// separate from the oracle workspace so graph capture can name the
    /// phase boundary without owning folded digit state.
    pub(crate) nc_phase_workspace: Option<NcPhaseWorkspace>,
    /// FE Ajtai-tail mat-vec scratch retained across folds. Whole-FE graph
    /// replay captures these scratch addresses.
    pub(crate) fe_ring_scratch: Option<RingMatVecScratch>,
    /// Current fold witness planes, retained as a grow-only session buffer so
    /// graph replay never refers to a dropped fold-local allocation.
    pub(crate) fold_planes: Option<DeviceBuffer<u64>>,
    /// Fresh CCS commitments produced by this adapter's `build_fresh_instances`.
    /// They are consumed by the next `prove` call only when their host
    /// commitments match the fresh claims exactly.
    pub(crate) cached_fresh_commitments: Option<CachedDeviceCommitments>,
    /// The previous fold's Π_DEC split planes, retained when the caller
    /// staged the output as the next fold's running instance.
    pub(crate) cached_running_planes: Option<CachedRunningPlanes>,
}

/// Host commitments plus their resident device words. The host side is only
/// an identity check; hot prover consumers should use `words`.
pub(crate) struct CachedDeviceCommitments {
    pub(crate) host: Vec<Commitment>,
    pub(crate) device: Arc<DeviceCommitments>,
    pub(crate) planes: Option<CachedDevicePlanes>,
}

/// Device-resident witness planes bound to a cache's host commitments.
///
/// The planes use the fold-plane layout (`[count][cols * D]`). They are
/// consumed only after the corresponding host commitments match the fresh
/// claims exactly, so this cache is a data-movement shortcut, not authority.
pub(crate) struct CachedDevicePlanes {
    pub(crate) words: DeviceBuffer<u64>,
    pub(crate) plane_len: usize,
    pub(crate) count: usize,
}

/// Retained Π_DEC split planes: the previous fold's child witnesses, still
/// device-resident in the fold-planes layout (`[k][cols * D]`).
pub(crate) struct CachedRunningPlanes {
    pub(crate) planes: DeviceBuffer<u64>,
    /// Words per plane (`cols * D`) at retention time.
    pub(crate) plane_len: usize,
    /// The child Ajtai commitments these planes were split into — binding
    /// identity for the staging contract behind `cache_output_for_next_step`:
    /// a running instance carrying exactly these commitments carries exactly
    /// these witnesses.
    pub(crate) commitments: CachedDeviceCommitments,
}

pub(crate) struct DecProverParts<'a> {
    pub(crate) device: &'a Device,
    pub(crate) kernels: &'a SumcheckKernels,
    pub(crate) ajtai: &'a mut DeviceAjtai,
    pub(crate) dec: &'a mut DeviceDec,
    pub(crate) bar_matrices: &'a mut Option<DeviceBarMatrices>,
}

pub(crate) struct AjtaiCommitParts<'a> {
    pub(crate) device: &'a Device,
    pub(crate) ajtai: &'a mut DeviceAjtai,
}

impl DeviceSession {
    pub(crate) fn new() -> Result<Self, Error> {
        let device;
        crate::perf_timed!("session.device", {
            device = Device::open().map_err(|_| backend_unavailable("failed to open CUDA device 0"))?;
        });
        Ok(Self::from_device(device))
    }

    pub(crate) fn new_on_context(ctx: Arc<CudaContext>) -> Result<Self, Error> {
        let device;
        crate::perf_timed!("session.device", {
            device = Device::from_context(ctx).map_err(|_| backend_unavailable("failed to create CUDA stream"))?;
        });
        Ok(Self::from_device(device))
    }

    fn from_device(device: Device) -> Self {
        Self {
            device,
            ajtai: None,
            ajtai_source: None,
            kernels: None,
            dec: None,
            bar_matrices: None,
            row_matrices: None,
            fe_phase_workspace: None,
            fe_oracle_workspace: None,
            nc_oracle_workspace: None,
            nc_phase_workspace: None,
            fe_ring_scratch: None,
            fold_planes: None,
            cached_fresh_commitments: None,
            cached_running_planes: None,
        }
    }

    pub(crate) fn take_cached_fresh_commitments(
        &mut self,
        commitments: &[Commitment],
    ) -> Option<CachedDeviceCommitments> {
        let cached = self.cached_fresh_commitments.take()?;
        let matches = cached.host.len() == commitments.len()
            && cached
                .host
                .iter()
                .zip(commitments)
                .all(|(cached, claim)| cached == claim);
        matches.then_some(cached)
    }

    /// The retained split planes, but only if they demonstrably correspond
    /// to `running`: the claim commitments must equal the cached children's
    /// commitments exactly (binding), and the widths must match. A mismatch
    /// simply drops the cache and falls back to a full upload.
    pub(crate) fn take_cached_running_planes(
        &mut self,
        output: Option<&DeviceFoldOutput>,
        running: &RunningInstance,
        cols: usize,
    ) -> Result<Option<CachedRunningPlanes>, Error> {
        let Some(cached) = self.cached_running_planes.take() else {
            return Ok(None);
        };
        let commitment_authority_matches = match output {
            Some(output) => Arc::ptr_eq(&cached.commitments.device, output.child_commitments()),
            None => {
                cached.commitments.device.materialize()?
                    == running
                        .claims
                        .iter()
                        .map(|claim| claim.c.clone())
                        .collect::<Vec<_>>()
            }
        };
        let matches = commitment_authority_matches
            && cached.commitments.device.count() == running.claims.len()
            && running.witnesses.len() == running.claims.len()
            && cached.plane_len == cols * D
            && output.is_none_or(|output| output.child_count() == running.claims.len());
        Ok(matches.then_some(cached))
    }

    /// Load the sumcheck + ring kernel modules and the Π_DEC module once
    /// per session.
    pub(crate) fn ensure_kernels_loaded(&mut self) -> Result<&SumcheckKernels, Error> {
        if self.kernels.is_none() {
            let kernels = SumcheckKernels::load(&self.device)
                .map_err(|_| backend_unavailable("failed to load sumcheck kernels"))?;
            self.kernels = Some(kernels);
        }
        if self.dec.is_none() {
            let dec = DeviceDec::new(&self.device).map_err(|_| backend_unavailable("failed to load Π_DEC kernels"))?;
            self.dec = Some(dec);
        }
        self.kernels()
    }

    pub(crate) fn kernels(&self) -> Result<&SumcheckKernels, Error> {
        self.kernels
            .as_ref()
            .ok_or_else(|| backend_unavailable("sumcheck kernels not loaded"))
    }

    /// Materialize and upload the Ajtai PP matching this session's `log` on
    /// first use (or when the PP changes); afterwards `self.ajtai` is
    /// populated and reused across folds. Identity is the materialized PP
    /// allocation itself (`materialize_pp` is an Arc clone after first
    /// load), so two same-shape PPs from different seeds cannot alias.
    pub(crate) fn ensure_pp_uploaded(&mut self, log: &AjtaiSModule) -> Result<&mut DeviceAjtai, Error> {
        let pp = log
            .materialize_pp()
            .map_err(|_| backend_unavailable("failed to materialize Ajtai PP for device upload"))?;
        if self
            .ajtai_source
            .as_ref()
            .is_some_and(|held| Arc::ptr_eq(held, &pp))
            && self
                .ajtai
                .as_ref()
                .is_some_and(|a| a.matches_z_dims(log.dims()))
        {
            return self.ajtai_mut();
        }
        let uploaded = DeviceAjtai::upload(&self.device, &pp)
            .map_err(|_| backend_unavailable("failed to upload Ajtai PP to device"))?;
        self.ajtai = Some(uploaded);
        self.ajtai_source = Some(pp);
        self.ajtai_mut()
    }

    /// Upload the static SuperNeo matrix views for this preprocessing context.
    ///
    /// This is setup work, not fold work: once the bar and row CSR views match
    /// `cache`, Π_CCS and Π_DEC reuse them across every fold in the session.
    pub(crate) fn ensure_structure_uploaded(&mut self, cache: &OptimizedStructureCache) -> Result<(), Error> {
        let superneo = cache.superneo();
        let needs_bar = self
            .bar_matrices
            .as_ref()
            .map_or(true, |matrices| !matrices.matches(superneo));
        if needs_bar {
            self.bar_matrices = Some(
                DeviceBarMatrices::upload(&self.device, superneo)
                    .map_err(|_| backend_unavailable("failed to upload CUDA SuperNeo bar matrices"))?,
            );
        }

        let needs_row = self
            .row_matrices
            .as_ref()
            .map_or(true, |matrices| !matrices.matches(superneo));
        if needs_row {
            self.row_matrices = Some(
                DeviceRowMatrices::upload(&self.device, superneo)
                    .map_err(|_| backend_unavailable("failed to upload CUDA SuperNeo row matrices"))?,
            );
        }
        Ok(())
    }

    pub(crate) fn ajtai_mut(&mut self) -> Result<&mut DeviceAjtai, Error> {
        self.ajtai
            .as_mut()
            .ok_or_else(|| backend_unavailable("Ajtai PP not uploaded"))
    }

    pub(crate) fn ajtai_commit_parts(&mut self) -> Result<AjtaiCommitParts<'_>, Error> {
        let Self { device, ajtai, .. } = self;
        Ok(AjtaiCommitParts {
            device,
            ajtai: ajtai
                .as_mut()
                .ok_or_else(|| backend_unavailable("Ajtai PP not uploaded"))?,
        })
    }

    pub(crate) fn dec_prover_parts(&mut self) -> Result<DecProverParts<'_>, Error> {
        let Self {
            device,
            ajtai,
            dec,
            bar_matrices,
            kernels,
            ..
        } = self;
        Ok(DecProverParts {
            device,
            kernels: kernels
                .as_ref()
                .ok_or_else(|| backend_unavailable("sumcheck kernels not loaded"))?,
            ajtai: ajtai
                .as_mut()
                .ok_or_else(|| backend_unavailable("Ajtai PP not uploaded"))?,
            dec: dec
                .as_mut()
                .ok_or_else(|| backend_unavailable("Π_DEC kernels not loaded"))?,
            bar_matrices,
        })
    }
}
