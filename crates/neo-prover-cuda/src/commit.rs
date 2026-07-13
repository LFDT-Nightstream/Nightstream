//! Host orchestration for device Ajtai commitments.
//!
//! Owns the uploaded public-parameter buffer (one H2D per PP, reused across
//! commits) and the assignment → commitment flow. Does not own PP generation
//! or the commitment semantics (`neo-ajtai` stays canonical).

use std::sync::Arc;

use cuda_core::{DeviceBuffer, DriverError};
use cuda_host::EmbeddedModuleError;
use neo_ajtai::{Commitment, PP};
use neo_fold_clean::paper::nifs::{Error as NifsError, NifsFreshInstancesRequest};
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::{CcsInstance, CcsWitness};
use neo_math::{Rq, D, F};
use p3_field::PrimeField64;
use thiserror::Error;

use crate::device::{upload_u64_device_buffer, Device};
use crate::field::f_from_device_word;
use crate::fold_output::DeviceCommitments;
use crate::kernels::ajtai::{
    load_ajtai_kernels, ring_mat_vec, ring_mat_vec_active_flags_into, ring_mat_vec_into, AjtaiKernelModule,
    RingMatVecScratch, RING_D,
};
use crate::ring_layout;
use crate::session::{backend_unavailable, CachedDeviceCommitments, CachedDevicePlanes, DeviceSession};

#[derive(Debug, Error)]
pub enum AjtaiDeviceError {
    #[error("CUDA driver error: {0:?}")]
    Driver(DriverError),
    #[error("kernel module load failed: {0:?}")]
    ModuleLoad(EmbeddedModuleError),
    #[error("assignment length {got} exceeds PP capacity {capacity}")]
    AssignmentTooLong { got: usize, capacity: usize },
}

impl From<DriverError> for AjtaiDeviceError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

/// Device-resident Ajtai public parameters plus the ring mat-vec kernels.
pub struct DeviceAjtai {
    kappa: usize,
    cols: usize,
    pp_dev: DeviceBuffer<u64>,
    module: AjtaiKernelModule,
    /// Reused ring mat-vec stage buffers (see `RingMatVecScratch`).
    scratch: RingMatVecScratch,
}

impl DeviceAjtai {
    /// Upload a materialized PP once. `pp.d` must equal the compiled ring
    /// degree; the layout is `[kappa][cols][D]` canonical coefficient words.
    pub fn upload(device: &Device, pp: &PP<Rq>) -> Result<Self, AjtaiDeviceError> {
        assert_eq!(pp.d, D, "PP ring degree must match neo_math::D");
        assert_eq!(D, RING_D, "kernel ring degree out of sync with neo_math::D");
        let mut words = Vec::with_capacity(pp.kappa * pp.m * D);
        for row in &pp.m_rows {
            assert_eq!(row.len(), pp.m, "PP row length must equal pp.m");
            for el in row {
                words.extend(el.0.iter().map(|c| c.as_canonical_u64()));
            }
        }
        let pp_dev = upload_u64_device_buffer(device.stream(), &words)?;
        let module = load_ajtai_kernels(device.ctx()).map_err(AjtaiDeviceError::ModuleLoad)?;
        Ok(Self {
            kappa: pp.kappa,
            cols: pp.m,
            pp_dev,
            module,
            scratch: RingMatVecScratch::new(),
        })
    }

    /// True when this uploaded PP commits Z matrices of the given `(d, cols)`
    /// shape — the tuple `AjtaiSModule::dims` reports.
    pub fn matches_z_dims(&self, dims: (usize, usize)) -> bool {
        dims == (D, self.cols)
    }

    /// Commit to an assignment z ∈ F^len (len ≤ cols·D). The flat ring-column
    /// layout equals the padded assignment vector, so no packing pass runs.
    pub fn commit_assignment(&mut self, device: &Device, z: &[F]) -> Result<Commitment, AjtaiDeviceError> {
        if z.len() > self.cols * D {
            return Err(AjtaiDeviceError::AssignmentTooLong {
                got: z.len(),
                capacity: self.cols * D,
            });
        }
        let words = ring_layout::assignment_to_words(z, self.cols);
        let z_dev = upload_u64_device_buffer(device.stream(), &words)?;
        let out = self.commit_device_columns(device, &z_dev, 0)?;
        self.download_commitment(device, &out)
    }

    /// Commit to ring columns already resident on device, starting at word
    /// `z_offset`.
    pub fn commit_device_columns(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        z_offset: usize,
    ) -> Result<DeviceBuffer<u64>, AjtaiDeviceError> {
        Ok(ring_mat_vec(
            &self.module,
            device.stream(),
            &mut self.scratch,
            &self.pp_dev,
            self.kappa,
            self.cols,
            z_dev,
            z_offset,
            1,
            0,
        )?)
    }

    /// Commit one resident message whose coefficients are all `-1`, `0`, or
    /// `1`. The signed-unit mask path avoids general field multiplication;
    /// fixed-seed SIS messages use exactly this alphabet.
    pub fn commit_signed_unit_device_columns(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        active_flag: &DeviceBuffer<u64>,
    ) -> Result<DeviceBuffer<u64>, AjtaiDeviceError> {
        let mut out = DeviceBuffer::zeroed(device.stream(), self.kappa * D)?;
        ring_mat_vec_active_flags_into(
            &self.module,
            device.stream(),
            &mut self.scratch,
            &self.pp_dev,
            self.kappa,
            self.cols,
            z_dev,
            0,
            1,
            0,
            active_flag,
            2,
            &mut out,
        )?;
        Ok(out)
    }

    /// Commit to `planes` consecutive ring-column planes (`plane_stride`
    /// words apart, e.g. the digit planes of a Π_DEC split) in one launch,
    /// returning the downloaded commitments in plane order.
    pub fn commit_planes(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        planes: usize,
        plane_stride: usize,
    ) -> Result<Vec<Commitment>, AjtaiDeviceError> {
        let (commitments, _) = self.commit_planes_with_device_output(device, z_dev, planes, plane_stride)?;
        Ok(commitments)
    }

    /// Commit to device-resident planes and keep the batched device output.
    ///
    /// The output layout is `[plane][kappa][D]`, exactly the commitment input
    /// layout Π_RLC consumes for resident commitment mixing.
    pub fn commit_planes_with_device_output(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        planes: usize,
        plane_stride: usize,
    ) -> Result<(Vec<Commitment>, DeviceBuffer<u64>), AjtaiDeviceError> {
        let out = self.commit_planes_device(device, z_dev, planes, plane_stride)?;
        let commitments = self.download_commitments(device, &out, planes)?;
        Ok((commitments, out))
    }

    /// Commit to device-resident planes without downloading the commitments.
    pub fn commit_planes_device(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        planes: usize,
        plane_stride: usize,
    ) -> Result<DeviceBuffer<u64>, AjtaiDeviceError> {
        self.commit_planes_device_at(device, z_dev, 0, planes, plane_stride)
    }

    /// Commit the same whole-column slice from every plane. `z_offset` is a
    /// word offset inside plane zero and `plane_stride` advances to the same
    /// slice in the next plane. Nebula uses this for its L-ALIGN lane ranges.
    pub fn commit_planes_device_at(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        z_offset: usize,
        planes: usize,
        plane_stride: usize,
    ) -> Result<DeviceBuffer<u64>, AjtaiDeviceError> {
        Ok(ring_mat_vec(
            &self.module,
            device.stream(),
            &mut self.scratch,
            &self.pp_dev,
            self.kappa,
            self.cols,
            z_dev,
            z_offset,
            planes,
            plane_stride,
        )?)
    }

    /// Prepare scratch for a graph-captured plane commitment of this shape.
    ///
    /// The captured body must not allocate; callers own output/input buffer
    /// stability, while this object owns the mat-vec scratch capacity.
    pub fn prepare_commit_planes(&mut self, device: &Device, planes: usize) -> Result<(), AjtaiDeviceError> {
        Ok(self
            .scratch
            .prepare_mat_vec(device.stream(), self.kappa, self.cols, planes)?)
    }

    /// Commit to device-resident planes into caller-owned output.
    ///
    /// This is the graph-safe form: the caller owns the commitment-word
    /// buffer lifetime, while this object only supplies the uploaded PP and
    /// reusable mat-vec scratch.
    pub fn commit_planes_device_into(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        planes: usize,
        plane_stride: usize,
        out: &mut DeviceBuffer<u64>,
    ) -> Result<(), AjtaiDeviceError> {
        Ok(ring_mat_vec_into(
            &self.module,
            device.stream(),
            &mut self.scratch,
            &self.pp_dev,
            self.kappa,
            self.cols,
            z_dev,
            0,
            planes,
            plane_stride,
            out,
        )?)
    }

    /// Commit every plane while zeroing inactive planes from a device flag
    /// surface inside the mat-vec kernel. Used by Π_DEC to avoid a host
    /// active-count join before child commitments.
    pub fn commit_planes_device_flags_into(
        &mut self,
        device: &Device,
        z_dev: &DeviceBuffer<u64>,
        planes: usize,
        plane_stride: usize,
        active_flags: &DeviceBuffer<u64>,
        digit_base: u32,
        out: &mut DeviceBuffer<u64>,
    ) -> Result<(), AjtaiDeviceError> {
        Ok(ring_mat_vec_active_flags_into(
            &self.module,
            device.stream(),
            &mut self.scratch,
            &self.pp_dev,
            self.kappa,
            self.cols,
            z_dev,
            0,
            planes,
            plane_stride,
            active_flags,
            digit_base,
            out,
        )?)
    }

    /// Download `planes` consecutive `kappa * D` commitment words.
    pub fn download_commitments(
        &self,
        device: &Device,
        out: &DeviceBuffer<u64>,
        planes: usize,
    ) -> Result<Vec<Commitment>, AjtaiDeviceError> {
        let words = out.to_host_vec(device.stream())?;
        device.sync()?;
        let plane_words = self.kappa * D;
        Ok((0..planes)
            .map(|p| {
                let mut commitment = Commitment::zeros(D, self.kappa);
                for (slot, word) in commitment.data.iter_mut().zip(&words[p * plane_words..]) {
                    *slot = f_from_device_word(*word);
                }
                commitment
            })
            .collect())
    }

    /// Download a `kappa * D` commit output into the canonical `Commitment`
    /// (column-major, matching `Commitment::data`). Synchronizes the stream.
    pub fn download_commitment(
        &self,
        device: &Device,
        out: &DeviceBuffer<u64>,
    ) -> Result<Commitment, AjtaiDeviceError> {
        let words = out.to_host_vec(device.stream())?;
        device.sync()?;
        let mut commitment = Commitment::zeros(D, self.kappa);
        for (slot, word) in commitment.data.iter_mut().zip(words) {
            *slot = f_from_device_word(word);
        }
        Ok(commitment)
    }

    pub fn module(&self) -> &AjtaiKernelModule {
        &self.module
    }

    pub fn kappa(&self) -> usize {
        self.kappa
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Build canonical fresh CCS instances while retaining their assignment
/// planes and commitments for the first fold. Invalid inputs deliberately
/// return `None` so the canonical CPU constructor remains the authority.
pub(crate) fn build_fresh_instances(
    session: &mut DeviceSession,
    request: NifsFreshInstancesRequest<'_>,
) -> Result<Option<Vec<CcsInstance>>, NifsError> {
    let b = request.pp.b();
    let valid = request.assignments.iter().all(|z| {
        z.len() == request.s.m
            && request.m_in <= z.len()
            && z.iter().all(|v| neo_math::balanced::within_nc_bound(*v, b))
    });
    if !valid {
        session.cached_fresh_commitments = None;
        return Ok(None);
    }

    let cols = request.s.m.div_ceil(D);
    session.ensure_pp_uploaded(request.log)?;
    let mut instances = Vec::with_capacity(request.assignments.len());
    let fresh_cache;
    {
        let parts = session.ajtai_commit_parts()?;
        if parts.ajtai.cols() != cols {
            session.cached_fresh_commitments = None;
            return Ok(None);
        }

        let commitments;
        let commitment_words;
        let assignments_dev;
        crate::perf_timed!("fold.commit.fresh", {
            let mut assignment_words = Vec::with_capacity(request.assignments.len() * cols * D);
            for z in request.assignments {
                assignment_words.extend(ring_layout::assignment_to_words(z, cols));
            }
            assignments_dev = upload_u64_device_buffer(parts.device.stream(), &assignment_words)
                .map_err(|_| backend_unavailable("fresh assignment upload failed"))?;
            (commitments, commitment_words) = parts
                .ajtai
                .commit_planes_with_device_output(parts.device, &assignments_dev, request.assignments.len(), cols * D)
                .map_err(|_| backend_unavailable("device batched Ajtai commit failed"))?;
        });
        for (z, c) in request.assignments.iter().zip(commitments) {
            instances.push(CcsInstance {
                claim: CcsClaim {
                    adv: None,
                    c,
                    x: z[..request.m_in].to_vec(),
                    m_in: request.m_in,
                },
                witness: CcsWitness {
                    // `Z` is the authoritative packed assignment and already
                    // contains the private suffix. Match the canonical CPU
                    // constructor without retaining that suffix twice.
                    w: Vec::new(),
                    Z: ring_layout::assignment_to_mat(z, cols),
                },
            });
        }
        let device_commitments = Arc::new(DeviceCommitments::new(
            Arc::clone(parts.device.stream()),
            commitment_words,
            request.assignments.len(),
            D,
            parts.ajtai.kappa(),
        )?);
        fresh_cache = CachedDeviceCommitments {
            host: instances
                .iter()
                .map(|instance| instance.claim.c.clone())
                .collect(),
            device: device_commitments,
            planes: Some(CachedDevicePlanes {
                words: assignments_dev,
                plane_len: cols * D,
                count: request.assignments.len(),
            }),
        };
    }
    session.cached_fresh_commitments = Some(fresh_cache);
    Ok(Some(instances))
}
