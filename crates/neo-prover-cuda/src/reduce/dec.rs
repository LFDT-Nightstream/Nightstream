//! Device Π_DEC: split the parent witness into k digit planes on the GPU,
//! evaluate child openings via the shared ring mat-vec kernels, and commit
//! child planes — all from one parent-witness upload.
//!
//! Owns the device flow and CE-claim assembly. Does not own the DEC
//! semantics: forms come from the canonical SuperNeo eval cache, ct/X/y_zcol
//! use `neo_reductions::common` helpers, and outputs must be field-identical
//! to `neo_fold_clean::paper::pi_dec::prove`.

use std::sync::{Arc, Mutex};

use cuda_core::{CudaStream, DeviceBuffer, PinnedHostBuffer};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::paper::pi_dec::{Children, Proof};
use neo_fold_clean::{CeClaim, DecMixer, Params, Structure};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

use crate::commit::{AjtaiDeviceError, DeviceAjtai};
use crate::device::{copy_host_to_device, uninit_u64_device_buffer, Device};
use crate::field::{f_from_device_word, k_from_device_words};
use crate::graph::{CaptureError, CapturedGraph};
use crate::kernels::ajtai::ring_mat_vec_active_flags_into;
use crate::kernels::ajtai::{ring_mat_vec_into, RingMatVecScratch};
use crate::kernels::csr::{launch_tensor_point_k, load_csr_kernels, CsrKernelModule};
use crate::kernels::pi_dec::{
    dec_y_zcol_partials_words, launch_dec_accumulate_status, launch_dec_scatter_active_words, launch_dec_split,
    launch_dec_y_zcol, launch_dec_y_zcol_active_flags_with_partials, load_dec_kernels, DecKernelModule,
};
use crate::reduce::ccs::{CcsDeviceError, DevicePiCcsKSurfaces, DevicePublicX, SumcheckKernels};
use crate::ring_forms::DeviceBarMatrices;
use crate::ring_layout;

static FULL_WITNESS_EXPORT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug, Error)]
pub enum DecDeviceError {
    #[error(transparent)]
    Ajtai(#[from] AjtaiDeviceError),
    #[error("CUDA driver error: {0:?}")]
    Driver(cuda_core::DriverError),
    #[error("kernel module load failed: {0:?}")]
    ModuleLoad(cuda_host::EmbeddedModuleError),
    #[error("parent witness value out of DEC range; use the CPU path for its error surface")]
    SplitOutOfRange,
    #[error("unsupported DEC shape: {0}")]
    Shape(&'static str),
    #[error("Π_DEC public self-check failed (ok_y={ok_y}, ok_x={ok_x}, ok_c={ok_c})")]
    PublicCheckFailed { ok_y: bool, ok_x: bool, ok_c: bool },
}

/// Session-wide device authority for deferred Π_DEC range failures.
///
/// Every resident split ORs its trailing error flag into this word on the
/// prover stream. Deferred proof/claim materialization downloads it once at
/// egress, preserving the CPU error contract without a join in every fold.
pub(crate) struct DeferredDecStatus {
    stream: Arc<CudaStream>,
    word: Mutex<DeviceBuffer<u64>>,
}

impl DeferredDecStatus {
    fn new(device: &Device) -> Result<Self, DecDeviceError> {
        Ok(Self {
            stream: Arc::clone(device.stream()),
            word: Mutex::new(DeviceBuffer::zeroed(device.stream(), 1)?),
        })
    }

    fn accumulate(&self, module: &DecKernelModule, flags: &DeviceBuffer<u64>, k: usize) -> Result<(), DecDeviceError> {
        let mut word = self
            .word
            .lock()
            .map_err(|_| DecDeviceError::Shape("deferred DEC status lock poisoned"))?;
        launch_dec_accumulate_status(module, &self.stream, flags, k, &mut word)?;
        Ok(())
    }

    pub(crate) fn check(&self) -> Result<(), DecDeviceError> {
        let word = self
            .word
            .lock()
            .map_err(|_| DecDeviceError::Shape("deferred DEC status lock poisoned"))?;
        let status = word.to_host_vec(&self.stream)?;
        if status.first().copied().unwrap_or(1) != 0 {
            return Err(DecDeviceError::SplitOutOfRange);
        }
        Ok(())
    }
}

impl From<cuda_core::DriverError> for DecDeviceError {
    fn from(e: cuda_core::DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CcsDeviceError> for DecDeviceError {
    fn from(e: CcsDeviceError) -> Self {
        match e {
            CcsDeviceError::Driver(d) => Self::Driver(d),
            CcsDeviceError::ModuleLoad(m) => Self::ModuleLoad(m),
            CcsDeviceError::Shape(s) => Self::Shape(s),
        }
    }
}

pub struct DeviceDec {
    module: DecKernelModule,
    csr: CsrKernelModule,
    /// Pinned landing buffer for the k digit planes' download — pageable
    /// D2H of ~50MB costs ~3x the pinned path. Sized on first prove,
    /// regrown only when a larger structure appears.
    planes_host: Option<PinnedHostBuffer<u64>>,
    /// Reused ring mat-vec stage buffers for the child y_ring evals.
    ring_scratch: RingMatVecScratch,
    /// Reused Π_DEC form-building buffers for `r -> chi_r -> bar(M)^T chi_r`.
    forms_workspace: DecFormsWorkspace,
    /// Reused split flags. Keeping this pointer stable lets downstream DEC
    /// graph captures read the device-owned active-child schedule.
    split_workspace: DecSplitWorkspace,
    /// Reused child-output buffers. These are temporary within one fold, but
    /// their addresses must persist across folds before DEC graph capture is
    /// trustworthy.
    child_workspace: DecChildWorkspace,
    /// Sticky split failure authority shared by all deferred outputs from
    /// this session.
    deferred_status: Arc<DeferredDecStatus>,
}

/// How much private child-witness material Π_DEC should return to the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecOutputMode {
    /// Materialize full child witnesses. Required for terminal proof export
    /// and strict reduction parity gates.
    Full,
    /// Materialize public child claims but leave private witness planes on
    /// device for the next fold.
    ResidentOnly,
}

impl DecOutputMode {
    fn downloads_full_witnesses(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Which part of the prover-side DEC recomposition check runs inside
/// [`DeviceDec::prove`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecRecompositionMode {
    /// Check y, X, and c against the fully materialized parent claim.
    Full,
    /// Check c now; the caller must check y and X once those parent public
    /// surfaces have been exported from pending device computations.
    DeferYAndX,
    /// Check y and c now; the caller must check X once the parent public
    /// surface has been exported from its pending device computation.
    DeferX,
    /// Check y and X now; the caller must check c once the parent commitment
    /// has been exported from its pending device computation.
    DeferCommitment,
    /// Check y now; the caller must check X and c once their pending device
    /// computations have been exported.
    DeferXAndCommitment,
    /// Check nothing except shape now; the caller must check y, X, and c once
    /// their pending device computations have been exported.
    DeferYAndXAndCommitment,
}

/// Outputs of one device Π_DEC fold.
///
/// `children` and `proof` are the current host-visible protocol objects.
/// `split` and `child_commitment_words` are the resident surfaces the CUDA
/// adapter can carry into the next fold.
pub struct DecFoldOutput {
    pub children: Children,
    pub proof: Proof,
    pub split: SplitPlanes,
    pub child_commitment_words: DeviceBuffer<u64>,
    pub child_surfaces: Option<DevicePiCcsKSurfaces>,
    pub(crate) child_public_x: Option<DevicePublicX>,
    pub(crate) deferred_status: Option<Arc<DeferredDecStatus>>,
}

impl DeviceDec {
    pub fn new(device: &Device) -> Result<Self, DecDeviceError> {
        let module = load_dec_kernels(device.ctx()).map_err(DecDeviceError::ModuleLoad)?;
        let csr = load_csr_kernels(device.ctx()).map_err(DecDeviceError::ModuleLoad)?;
        let deferred_status = Arc::new(DeferredDecStatus::new(device)?);
        Ok(Self {
            module,
            csr,
            planes_host: None,
            ring_scratch: RingMatVecScratch::new(),
            forms_workspace: DecFormsWorkspace::default(),
            split_workspace: DecSplitWorkspace::default(),
            child_workspace: DecChildWorkspace::default(),
            deferred_status,
        })
    }

    fn planes_host_for(&mut self, device: &Device, len: usize) -> Result<&mut PinnedHostBuffer<u64>, DecDeviceError> {
        if self.planes_host.as_ref().is_none_or(|p| p.len() < len) {
            self.planes_host = Some(PinnedHostBuffer::zeroed(device.ctx(), len)?);
        }
        Ok(self.planes_host.as_mut().expect("allocated above"))
    }

    /// GPU Π_DEC prover step. Mirrors `neo_fold_clean::paper::pi_dec::prove`:
    /// split → per-child y_ring eval + commit → claim assembly → the same
    /// three reconstruction self-checks the CPU prover runs.
    /// `bar_matrices` is the session's device-resident bar upload, shared
    /// with the Π_CCS Ajtai phase (same structure cache → same fingerprint);
    /// this call uploads it on first use.
    ///
    /// The returned [`SplitPlanes`] holds the k digit planes still resident
    /// on device (`[k][blocks * D]`, byte-equal to `mat_to_words` of the
    /// child witnesses) — a session that feeds the children back as the next
    /// fold's running witnesses can reuse them instead of re-uploading.
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        ajtai: &mut DeviceAjtai,
        bar_matrices: &mut Option<DeviceBarMatrices>,
        pp: &Params,
        s: &Structure,
        cache: &OptimizedStructureCache,
        combine: DecMixer,
        parent: &CeClaim,
        parent_witness: DecParentWitness<'_>,
        reusable_split_planes: Option<DeviceBuffer<u64>>,
        pi_ccs_forms: Option<&DeviceBuffer<u64>>,
        output_mode: DecOutputMode,
        recomposition_mode: DecRecompositionMode,
    ) -> Result<DecFoldOutput, DecDeviceError> {
        let shape = DecShape::for_inputs(pp, s, parent, &parent_witness)?;
        let want_nc = !(parent.s_col.is_empty() && parent.y_zcol.is_empty());
        let (mut split, children);
        let (claims, witnesses);
        if let Some(forms) = pi_ccs_forms {
            let expected = 2 * shape.t_mats * shape.blocks * D;
            if forms.len() < expected {
                return Err(DecDeviceError::Shape(
                    "retained Pi_CCS forms are shorter than the DEC shape",
                ));
            }
        } else {
            perf_timed!("fold.superneo.pi_dec.open_children.forms", {
                device_ring_forms(
                    device,
                    &self.csr,
                    &mut self.forms_workspace,
                    bar_matrices,
                    cache,
                    parent,
                    s,
                    &shape,
                )?;
            });
        }
        perf_timed!("fold.superneo.pi_dec.split", {
            split = split_parent(
                device,
                &self.module,
                &mut self.split_workspace,
                &parent_witness,
                reusable_split_planes,
                &shape,
                &self.deferred_status,
            )?;
        });
        let forms_dev = match pi_ccs_forms {
            Some(forms) => forms,
            None => self.forms_workspace.forms()?,
        };
        children = eval_and_commit_children(
            device,
            kernels,
            &self.module,
            &self.csr,
            &mut self.ring_scratch,
            &mut self.child_workspace,
            ajtai,
            forms_dev,
            &split,
            &shape,
            s,
            want_nc.then_some(parent.s_col.as_slice()),
            matches!(output_mode, DecOutputMode::ResidentOnly),
        )?;
        let deferred_status = if output_mode.downloads_full_witnesses() {
            perf_timed!("fold.superneo.pi_dec.split.status", {
                split.check_status(device, shape.k)?;
            });
            None
        } else {
            Some(Arc::clone(&self.deferred_status))
        };
        let child_public_x = if matches!(output_mode, DecOutputMode::ResidentOnly) {
            Some(DevicePublicX::pack_from_planes(
                device,
                kernels.rlc(),
                split.planes(),
                shape.k,
                shape.len,
                parent.m_in,
            )?)
        } else {
            None
        };
        split.release_flags(&mut self.split_workspace);
        if output_mode.downloads_full_witnesses() {
            let planes_len = shape.k * shape.len;
            perf_timed!("fold.superneo.pi_dec.emit.planes", {
                let _export_guard = FULL_WITNESS_EXPORT_LOCK
                    .lock()
                    .expect("full witness export lock poisoned");
                let planes_host = self.planes_host_for(device, planes_len)?;
                split
                    .planes()
                    .copy_to_pinned_host(device.stream(), planes_host)?;
            });
            let planes_words = &self
                .planes_host
                .as_ref()
                .expect("downloaded above")
                .as_slice()[..planes_len];
            perf_timed!("fold.superneo.pi_dec.emit.assemble", {
                (claims, witnesses) = assemble_children(pp, s, parent, &shape, &children, planes_words)?;
            });
        } else {
            perf_timed!("fold.superneo.pi_dec.emit.assemble", {
                claims = assemble_claim_shells(pp, s, parent, &shape, ajtai.kappa(), &children)?;
                witnesses = Vec::new();
            });
        }

        let commitments: Vec<Commitment> = claims.iter().map(|c| c.c.clone()).collect();
        perf_timed!("fold.superneo.pi_dec.check_recompose", {
            verify_reconstruction(parent, &claims, &commitments, combine, &shape, recomposition_mode)?;
        });
        Ok(DecFoldOutput {
            children: Children {
                claims: claims.clone(),
                witnesses,
            },
            proof: Proof { children: claims },
            split,
            child_commitment_words: children.commitment_words,
            child_surfaces: children.resident_surfaces,
            child_public_x,
            deferred_status,
        })
    }
}

/// The Π_DEC parent witness: a host `Mat` (uploaded by the split) or an
/// already device-resident buffer in the standard `cols * D` word layout
/// (e.g. the Π_RLC mix output, consumed without a host round-trip).
pub enum DecParentWitness<'a> {
    Host(&'a Mat<F>),
    Device(&'a DeviceBuffer<u64>),
}

/// Dimensions of one Π_DEC step, validated once up front.
struct DecShape {
    k: usize,
    b: u32,
    big_b: u64,
    blocks: usize,
    /// Words per digit plane (`blocks * D`).
    len: usize,
    d_pad: usize,
    t_mats: usize,
}

impl DecShape {
    fn for_inputs(
        pp: &Params,
        s: &Structure,
        parent: &CeClaim,
        parent_witness: &DecParentWitness<'_>,
    ) -> Result<Self, DecDeviceError> {
        let k = pp.k_rho() as usize;
        let b = pp.b();
        let big_b = (b as u128)
            .checked_pow(k as u32)
            .filter(|v| *v <= i64::MAX as u128)
            .ok_or(DecDeviceError::Shape("b^k exceeds i64 range"))?;
        let blocks = s.m.div_ceil(D);
        let witness_ok = match parent_witness {
            DecParentWitness::Host(mat) => mat.rows() == D && mat.cols() == blocks,
            DecParentWitness::Device(buf) => buf.len() == blocks * D,
        };
        if !witness_ok {
            return Err(DecDeviceError::Shape("parent witness must be D x ceil(m/D)"));
        }
        let d_pad = parent.y_ring.first().map(Vec::len).unwrap_or(0);
        if d_pad < D || !d_pad.is_power_of_two() {
            return Err(DecDeviceError::Shape("parent y_ring rows must be d_pad >= D"));
        }
        Ok(Self {
            k,
            b,
            big_b: big_b as u64,
            blocks,
            len: blocks * D,
            d_pad,
            t_mats: s.t(),
        })
    }
}

/// The k digit planes, resident on device after the split kernel.
pub struct SplitPlanes {
    planes: DeviceBuffer<u64>,
    activity_flags: Option<DeviceBuffer<u64>>,
    active_count: usize,
    flags_only: bool,
}

impl SplitPlanes {
    /// The `[k][blocks * D]` plane words, still on device.
    pub fn planes(&self) -> &DeviceBuffer<u64> {
        &self.planes
    }

    fn activity_flags(&self) -> &DeviceBuffer<u64> {
        self.activity_flags
            .as_ref()
            .expect("split activity flags released after DEC child materialization")
    }

    fn active_count(&self) -> usize {
        self.active_count
    }

    fn uses_flag_schedule(&self) -> bool {
        self.flags_only
    }

    fn check_status(&self, device: &Device, k: usize) -> Result<(), DecDeviceError> {
        if !self.flags_only {
            return Ok(());
        }
        let flags = self.activity_flags().to_host_vec(device.stream())?;
        device.sync()?;
        if flags.get(k).copied().unwrap_or(1) != 0 {
            return Err(DecDeviceError::SplitOutOfRange);
        }
        Ok(())
    }

    fn release_flags(&mut self, workspace: &mut DecSplitWorkspace) {
        workspace.activity_flags = self.activity_flags.take();
    }

    pub fn into_planes(self) -> DeviceBuffer<u64> {
        self.planes
    }
}

/// Everything measured on device for the children, downloaded to host —
/// except the digit planes themselves, which land in the session's pinned
/// buffer. Zero digit planes yield zero y words and zero commitments, which
/// is exactly what the CPU reduction constructs for them — so every child
/// is handled uniformly.
struct ChildDeviceResults {
    /// `2t * D` y_ring coefficient words per child, plane-major.
    y_words: Option<Vec<u64>>,
    commitments: Vec<Commitment>,
    /// `[child][kappa][D]` Ajtai commitments, still device-resident.
    commitment_words: DeviceBuffer<u64>,
    /// `D` K words per child when the NC channel is active.
    y_zcol_words: Option<Vec<u64>>,
    resident_surfaces: Option<DevicePiCcsKSurfaces>,
}

struct MaterializedChildSurfaces {
    y_words: Vec<u64>,
    commitments: Vec<Commitment>,
    y_zcol_words: Option<Vec<u64>>,
}

/// Reused device storage for the Π_DEC ring-form build. The protocol value is
/// still `parent.r`; this object only owns the hot-path transfer/table buffers.
#[derive(Default)]
struct DecFormsWorkspace {
    challenge_words: Vec<u64>,
    challenge_dev: Option<DeviceBuffer<u64>>,
    chi_dev: Option<DeviceBuffer<u64>>,
    forms: Option<DeviceBuffer<u64>>,
}

#[derive(Default)]
struct DecSplitWorkspace {
    activity_flags: Option<DeviceBuffer<u64>>,
}

#[derive(Default)]
struct DecChildWorkspace {
    compact_planes: Option<DeviceBuffer<u64>>,
    active_commit_words: Option<DeviceBuffer<u64>>,
    active_commit_graph: Option<DecActiveCommitGraph>,
    y_ring_out: Option<DeviceBuffer<u64>>,
    y_zcol_challenges: Option<DeviceBuffer<u64>>,
    y_zcol_chi: Option<DeviceBuffer<u64>>,
    y_zcol_partials: Option<DeviceBuffer<u64>>,
    y_zcol_out: Option<DeviceBuffer<u64>>,
    canonical_y: Option<DeviceBuffer<u64>>,
    canonical_y_zcol: Option<DeviceBuffer<u64>>,
    commitments_host: Option<PinnedHostBuffer<u64>>,
    y_host: Option<PinnedHostBuffer<u64>>,
    y_zcol_host: Option<PinnedHostBuffer<u64>>,
}

struct DecActiveCommitGraph {
    key: DecActiveCommitGraphKey,
    graph: CapturedGraph,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecActiveCommitGraphKey {
    active_count: usize,
    plane_stride: usize,
    kappa: usize,
    cols: usize,
    input_ptr: u64,
    output_ptr: u64,
}

impl DecFormsWorkspace {
    fn build(
        &mut self,
        device: &Device,
        csr: &CsrKernelModule,
        bar: &DeviceBarMatrices,
        challenges: &[K],
        n_eff: usize,
    ) -> Result<(), DecDeviceError> {
        self.challenge_words.clear();
        self.challenge_words.reserve(challenges.len() * 2);
        for value in challenges {
            let (re, im) = value.to_limbs_u64();
            self.challenge_words.extend([re, im]);
        }

        let stream = device.stream();
        if self
            .challenge_dev
            .as_ref()
            .is_none_or(|buffer| buffer.len() < self.challenge_words.len())
        {
            self.challenge_dev = Some(uninit_u64_device_buffer(stream, self.challenge_words.len())?);
        }
        let challenge_dev = self.challenge_dev.as_ref().expect("allocated above");
        copy_host_to_device(stream, challenge_dev, &self.challenge_words)?;

        let chi_words = tensor_point_len(challenges)? * 2;
        if self
            .chi_dev
            .as_ref()
            .is_none_or(|buffer| buffer.len() < chi_words)
        {
            self.chi_dev = Some(uninit_u64_device_buffer(stream, chi_words)?);
        }

        let form_words = bar.form_words();
        if self
            .forms
            .as_ref()
            .is_none_or(|buffer| buffer.len() < form_words)
        {
            self.forms = Some(uninit_u64_device_buffer(stream, form_words)?);
        }

        bar.build_forms_from_device_challenges_into_concurrent(
            device,
            csr,
            challenge_dev,
            challenges.len(),
            n_eff,
            self.chi_dev.as_mut().expect("allocated above"),
            self.forms.as_mut().expect("allocated above"),
        )?;
        Ok(())
    }

    fn forms(&self) -> Result<&DeviceBuffer<u64>, DecDeviceError> {
        self.forms
            .as_ref()
            .ok_or(DecDeviceError::Shape("DEC forms workspace has not been built"))
    }
}

fn take_child_buffer(
    slot: &mut Option<DeviceBuffer<u64>>,
    stream: &std::sync::Arc<CudaStream>,
    len: usize,
) -> Result<DeviceBuffer<u64>, DecDeviceError> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        return Ok(uninit_u64_device_buffer(stream, len)?);
    }
    Ok(slot.take().expect("checked above"))
}

fn store_child_buffer(slot: &mut Option<DeviceBuffer<u64>>, buffer: DeviceBuffer<u64>) {
    *slot = Some(buffer);
}

fn child_host_buffer<'a>(
    slot: &'a mut Option<PinnedHostBuffer<u64>>,
    device: &Device,
    len: usize,
) -> Result<&'a mut PinnedHostBuffer<u64>, DecDeviceError> {
    if slot.as_ref().is_none_or(|buffer| buffer.len() < len) {
        *slot = Some(PinnedHostBuffer::zeroed(device.ctx(), len)?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

/// Build the fold's ring linear forms (`bar(M_j)^T · χ_r`) on device as a
/// `[2t][blocks][D]` matrix: row 2j = re(form_j), row 2j+1 = im. The static
/// bar matrices upload once per structure and are reused across folds.
fn device_ring_forms(
    device: &Device,
    csr: &CsrKernelModule,
    workspace: &mut DecFormsWorkspace,
    bar_matrices: &mut Option<DeviceBarMatrices>,
    cache: &OptimizedStructureCache,
    parent: &CeClaim,
    s: &Structure,
    shape: &DecShape,
) -> Result<(), DecDeviceError> {
    let superneo = cache.superneo();
    if superneo.matrix_caches().len() != shape.t_mats {
        return Err(DecDeviceError::Shape("ring form count must equal structure.t"));
    }
    let cached = bar_matrices
        .as_ref()
        .is_some_and(|bar| bar.matches(superneo));
    if !cached {
        perf_timed!("session.structure", {
            *bar_matrices = Some(DeviceBarMatrices::upload(device, superneo)?);
        });
    }
    let bar = bar_matrices.as_ref().expect("bar matrices uploaded above");
    if bar.blocks() != shape.blocks {
        return Err(DecDeviceError::Shape("bar block count must match the DEC shape"));
    }
    let n_eff = core::cmp::min(s.n, 1usize << parent.r.len());
    workspace.build(device, csr, bar, &parent.r, n_eff)
}

fn split_parent(
    device: &Device,
    module: &DecKernelModule,
    workspace: &mut DecSplitWorkspace,
    parent_witness: &DecParentWitness<'_>,
    reusable_planes: Option<DeviceBuffer<u64>>,
    shape: &DecShape,
    deferred_status: &DeferredDecStatus,
) -> Result<SplitPlanes, DecDeviceError> {
    let uploaded;
    let z_dev = match parent_witness {
        DecParentWitness::Host(mat) => {
            let words = ring_layout::mat_to_words(mat);
            uploaded = DeviceBuffer::from_host(device.stream(), &words)?;
            &uploaded
        }
        DecParentWitness::Device(buf) => *buf,
    };
    // `dec_split` writes every child-plane slot for each parent word.
    // Avoid a separate full-plane memset before the split.
    let planes_len = shape.k * shape.len;
    let mut planes = match reusable_planes {
        Some(planes) if planes.len() == planes_len => planes,
        _ => uninit_u64_device_buffer(device.stream(), planes_len)?,
    };
    let mut activity_flags = take_child_buffer(&mut workspace.activity_flags, device.stream(), shape.k + 1)?;
    activity_flags.zero_async(device.stream())?;
    launch_dec_split(
        module,
        device.stream(),
        z_dev,
        shape.len,
        shape.k,
        shape.b,
        shape.big_b,
        &mut planes,
        &mut activity_flags,
    )?;
    deferred_status.accumulate(module, &activity_flags, shape.k)?;
    Ok(SplitPlanes {
        planes,
        activity_flags: Some(activity_flags),
        active_count: shape.k,
        flags_only: true,
    })
}

/// Evaluate every child's y_ring (ring mat-vec against the forms) and
/// commit every digit plane — one batched launch each — then download
/// planes, y words, and commitments.
#[allow(clippy::too_many_arguments)]
fn eval_and_commit_children(
    device: &Device,
    kernels: &SumcheckKernels,
    dec_module: &DecKernelModule,
    csr: &CsrKernelModule,
    ring_scratch: &mut RingMatVecScratch,
    child_workspace: &mut DecChildWorkspace,
    ajtai: &mut DeviceAjtai,
    forms_dev: &DeviceBuffer<u64>,
    split: &SplitPlanes,
    shape: &DecShape,
    s: &Structure,
    s_col: Option<&[K]>,
    retain_surfaces: bool,
) -> Result<ChildDeviceResults, DecDeviceError> {
    let stream = device.stream();
    let active_count = split.active_count();
    if active_count == 0 {
        return zero_child_results(device, kernels, ajtai.kappa(), shape, s_col.is_some(), retain_surfaces);
    }
    let mut compact_planes = compact_active_planes(device, dec_module, child_workspace, split, shape)?;
    let active_planes = compact_planes.as_ref().unwrap_or(split.planes());
    let y_zcol_stream = match s_col {
        Some(_) => Some(stream.fork()?),
        None => None,
    };
    let mut y_zcol_buffers;
    perf_timed!("fold.superneo.pi_dec.open_children.y_zcol", {
        y_zcol_buffers = match (s_col, y_zcol_stream.as_ref()) {
            (Some(s_col), Some(y_zcol_stream)) => {
                let chi_len = tensor_point_len(s_col)?;
                if chi_len < s.m {
                    return Err(DecDeviceError::Shape("chi(s_col) shorter than CCS width"));
                }
                let s_col_words = k_slice_words(s_col);
                let s_col_dev =
                    take_child_buffer(&mut child_workspace.y_zcol_challenges, y_zcol_stream, s_col_words.len())?;
                copy_host_to_device(y_zcol_stream, &s_col_dev, &s_col_words)?;
                // tensor_point_k writes every K limb before dec_y_zcol reads
                // the table; zeroing this 2^|s_col| buffer is pure hot-path
                // memset churn.
                let mut chi_dev = take_child_buffer(&mut child_workspace.y_zcol_chi, y_zcol_stream, chi_len * 2)?;
                launch_tensor_point_k(csr, y_zcol_stream, &s_col_dev, s_col.len(), &mut chi_dev)?;
                let mut out = take_child_buffer(&mut child_workspace.y_zcol_out, y_zcol_stream, active_count * D * 2)?;
                if split.uses_flag_schedule() {
                    let partial_words = dec_y_zcol_partials_words(s.m, active_count);
                    let mut partials =
                        take_child_buffer(&mut child_workspace.y_zcol_partials, y_zcol_stream, partial_words)?;
                    launch_dec_y_zcol_active_flags_with_partials(
                        dec_module,
                        y_zcol_stream,
                        active_planes,
                        split.activity_flags(),
                        &chi_dev,
                        s.m,
                        shape.len,
                        active_count,
                        &mut partials,
                        &mut out,
                    )?;
                    store_child_buffer(&mut child_workspace.y_zcol_partials, partials);
                } else {
                    launch_dec_y_zcol(
                        dec_module,
                        y_zcol_stream,
                        active_planes,
                        &chi_dev,
                        s.m,
                        shape.len,
                        active_count,
                        &mut out,
                    )?;
                }
                Some((s_col_dev, chi_dev, out))
            }
            _ => None,
        };
    });
    let mut y_out;
    perf_timed!("fold.superneo.pi_dec.open_children.y_ring", {
        y_out = take_child_buffer(
            &mut child_workspace.y_ring_out,
            stream,
            active_count * 2 * shape.t_mats * D,
        )?;
        if split.uses_flag_schedule() {
            ring_mat_vec_active_flags_into(
                ajtai.module(),
                stream,
                ring_scratch,
                forms_dev,
                2 * shape.t_mats,
                shape.blocks,
                active_planes,
                0,
                active_count,
                shape.len,
                split.activity_flags(),
                shape.b,
                &mut y_out,
            )?;
        } else {
            ring_mat_vec_into(
                ajtai.module(),
                stream,
                ring_scratch,
                forms_dev,
                2 * shape.t_mats,
                shape.blocks,
                active_planes,
                0,
                active_count,
                shape.len,
                &mut y_out,
            )?;
        }
        #[cfg(feature = "perf-timers")]
        device.sync()?;
    });
    let commitment_words;
    perf_timed!("fold.superneo.pi_dec.commit_children", {
        commitment_words = commit_active_planes(
            ajtai,
            device,
            dec_module,
            child_workspace,
            active_planes,
            split,
            shape,
            active_count,
        )?;
    });
    let materialized;
    let resident_surfaces;
    perf_timed!("fold.superneo.pi_dec.emit.download", {
        if retain_surfaces {
            if let Some(y_zcol_stream) = y_zcol_stream.as_ref() {
                stream.join(y_zcol_stream)?;
            }
            let y_zcol = y_zcol_buffers.as_ref().map(|(_, _, out)| out);
            resident_surfaces = Some(DevicePiCcsKSurfaces::pack_raw(
                device,
                kernels,
                &y_out,
                y_zcol,
                shape.k,
                shape.t_mats,
                shape.d_pad,
            )?);
            materialized = MaterializedChildSurfaces {
                y_words: Vec::new(),
                commitments: Vec::new(),
                y_zcol_words: None,
            };
        } else {
            resident_surfaces = None;
            materialized = materialize_child_claim_surfaces(
                device,
                dec_module,
                child_workspace,
                ajtai,
                split,
                shape,
                active_count,
                &y_out,
                y_zcol_stream.as_ref(),
                y_zcol_buffers.take(),
                &commitment_words,
            )?;
        }
    });
    store_child_buffer(&mut child_workspace.y_ring_out, y_out);
    if let Some(buffer) = compact_planes.take() {
        store_child_buffer(&mut child_workspace.compact_planes, buffer);
    }
    if let Some((s_col_dev, chi_dev, y_zcol_out)) = y_zcol_buffers {
        store_child_buffer(&mut child_workspace.y_zcol_challenges, s_col_dev);
        store_child_buffer(&mut child_workspace.y_zcol_chi, chi_dev);
        store_child_buffer(&mut child_workspace.y_zcol_out, y_zcol_out);
    }
    Ok(ChildDeviceResults {
        y_words: (!retain_surfaces).then_some(materialized.y_words),
        commitments: materialized.commitments,
        commitment_words,
        y_zcol_words: materialized.y_zcol_words,
        resident_surfaces,
    })
}

#[allow(clippy::too_many_arguments)]
fn materialize_child_claim_surfaces(
    device: &Device,
    dec_module: &DecKernelModule,
    child_workspace: &mut DecChildWorkspace,
    ajtai: &DeviceAjtai,
    split: &SplitPlanes,
    shape: &DecShape,
    active_count: usize,
    y_out: &DeviceBuffer<u64>,
    y_zcol_stream: Option<&std::sync::Arc<CudaStream>>,
    y_zcol_buffers: Option<(DeviceBuffer<u64>, DeviceBuffer<u64>, DeviceBuffer<u64>)>,
    commitment_words: &DeviceBuffer<u64>,
) -> Result<MaterializedChildSurfaces, DecDeviceError> {
    let stream = device.stream();
    let commitments = download_child_commitments(device, child_workspace, ajtai.kappa(), commitment_words, shape.k)?;
    let y_words_per_child = 2 * shape.t_mats * D;
    let mut canonical_y_words = canonicalize_active_words(
        dec_module,
        stream,
        &mut child_workspace.canonical_y,
        y_out,
        split.activity_flags(),
        y_words_per_child,
        active_count,
        shape.k,
    )?;
    let y_source = canonical_y_words.as_ref().unwrap_or(y_out);
    let y_len = shape.k * y_words_per_child;
    let y_host = child_host_buffer(&mut child_workspace.y_host, device, y_len)?;
    y_source.copy_to_pinned_host(stream, y_host)?;
    let y_words = y_host.as_slice()[..y_len].to_vec();
    if let Some(buffer) = canonical_y_words.take() {
        store_child_buffer(&mut child_workspace.canonical_y, buffer);
    }

    if let Some(y_zcol_stream) = y_zcol_stream {
        stream.join(y_zcol_stream)?;
    }
    device.sync()?;
    let mut canonical_y_zcol_words = match y_zcol_buffers.as_ref().map(|(_, _, out)| out) {
        Some(out) => Some(canonicalize_active_words(
            dec_module,
            stream,
            &mut child_workspace.canonical_y_zcol,
            out,
            split.activity_flags(),
            D * 2,
            active_count,
            shape.k,
        )?),
        None => None,
    };
    let y_zcol_words = match (
        y_zcol_buffers.as_ref().map(|(_, _, out)| out),
        canonical_y_zcol_words.as_ref(),
    ) {
        (Some(_), Some(Some(canonical))) => {
            let len = shape.k * D * 2;
            let host = child_host_buffer(&mut child_workspace.y_zcol_host, device, len)?;
            canonical.copy_to_pinned_host(stream, host)?;
            Some(host.as_slice()[..len].to_vec())
        }
        (Some(out), Some(None)) => {
            let len = shape.k * D * 2;
            let host = child_host_buffer(&mut child_workspace.y_zcol_host, device, len)?;
            out.copy_to_pinned_host(stream, host)?;
            Some(host.as_slice()[..len].to_vec())
        }
        _ => None,
    };
    if let Some(Some(buffer)) = canonical_y_zcol_words.take() {
        store_child_buffer(&mut child_workspace.canonical_y_zcol, buffer);
    }
    if let Some((s_col_dev, chi_dev, y_zcol_out)) = y_zcol_buffers {
        store_child_buffer(&mut child_workspace.y_zcol_challenges, s_col_dev);
        store_child_buffer(&mut child_workspace.y_zcol_chi, chi_dev);
        store_child_buffer(&mut child_workspace.y_zcol_out, y_zcol_out);
    }
    Ok(MaterializedChildSurfaces {
        y_words,
        commitments,
        y_zcol_words,
    })
}

fn download_child_commitments(
    device: &Device,
    child_workspace: &mut DecChildWorkspace,
    kappa: usize,
    words: &DeviceBuffer<u64>,
    planes: usize,
) -> Result<Vec<Commitment>, DecDeviceError> {
    let plane_words = kappa * D;
    let len = planes * plane_words;
    let host = child_host_buffer(&mut child_workspace.commitments_host, device, len)?;
    words.copy_to_pinned_host(device.stream(), host)?;
    let words = &host.as_slice()[..len];
    Ok((0..planes)
        .map(|p| {
            let mut commitment = Commitment::zeros(D, kappa);
            for (slot, word) in commitment
                .data
                .iter_mut()
                .zip(&words[p * plane_words..(p + 1) * plane_words])
            {
                *slot = f_from_device_word(*word);
            }
            commitment
        })
        .collect())
}

fn compact_active_planes(
    _device: &Device,
    _dec_module: &DecKernelModule,
    _child_workspace: &mut DecChildWorkspace,
    split: &SplitPlanes,
    shape: &DecShape,
) -> Result<Option<DeviceBuffer<u64>>, DecDeviceError> {
    if split.uses_flag_schedule() || split.active_count() == shape.k {
        return Ok(None);
    }
    Err(DecDeviceError::Shape(
        "compact active DEC scheduling requires a device-owned active-count path",
    ))
}

#[allow(clippy::too_many_arguments)]
fn commit_active_planes(
    ajtai: &mut DeviceAjtai,
    device: &Device,
    dec_module: &DecKernelModule,
    child_workspace: &mut DecChildWorkspace,
    active_planes: &DeviceBuffer<u64>,
    split: &SplitPlanes,
    shape: &DecShape,
    active_count: usize,
) -> Result<DeviceBuffer<u64>, DecDeviceError> {
    let word_len = ajtai.kappa() * D;
    if split.uses_flag_schedule() {
        let mut words = uninit_u64_device_buffer(device.stream(), shape.k * word_len)?;
        ajtai.commit_planes_device_flags_into(
            device,
            active_planes,
            shape.k,
            shape.len,
            split.activity_flags(),
            shape.b,
            &mut words,
        )?;
        return Ok(words);
    }
    if active_count == shape.k {
        return Ok(ajtai.commit_planes_device(device, active_planes, active_count, shape.len)?);
    }

    let stream = device.stream();
    let mut active_words = take_child_buffer(
        &mut child_workspace.active_commit_words,
        stream,
        active_count * word_len,
    )?;
    commit_active_planes_into_workspace(
        ajtai,
        device,
        child_workspace,
        active_planes,
        active_count,
        shape.len,
        &mut active_words,
    )?;
    let mut words = uninit_u64_device_buffer(stream, shape.k * word_len)?;
    launch_dec_scatter_active_words(
        dec_module,
        stream,
        &active_words,
        split.activity_flags(),
        word_len,
        active_count,
        shape.k,
        &mut words,
    )?;
    store_child_buffer(&mut child_workspace.active_commit_words, active_words);
    Ok(words)
}

fn commit_active_planes_into_workspace(
    ajtai: &mut DeviceAjtai,
    device: &Device,
    child_workspace: &mut DecChildWorkspace,
    active_planes: &DeviceBuffer<u64>,
    active_count: usize,
    plane_stride: usize,
    active_words: &mut DeviceBuffer<u64>,
) -> Result<(), DecDeviceError> {
    let stream = device.stream();
    ajtai.prepare_commit_planes(device, active_count)?;
    let key = DecActiveCommitGraphKey {
        active_count,
        plane_stride,
        kappa: ajtai.kappa(),
        cols: ajtai.cols(),
        input_ptr: active_planes.cu_deviceptr(),
        output_ptr: active_words.cu_deviceptr(),
    };
    if let Some(graph) = child_workspace
        .active_commit_graph
        .as_ref()
        .filter(|graph| graph.key == key)
    {
        graph.graph.launch(stream)?;
        return Ok(());
    }

    let graph = CapturedGraph::capture_checked(stream, || -> Result<(), AjtaiDeviceError> {
        ajtai.commit_planes_device_into(device, active_planes, active_count, plane_stride, active_words)
    })
    .map_err(|error| match error {
        CaptureError::Body(error) => DecDeviceError::from(error),
        CaptureError::Driver(error) => DecDeviceError::from(error),
    })?;
    graph.launch(stream)?;
    child_workspace.active_commit_graph = Some(DecActiveCommitGraph { key, graph });
    Ok(())
}

fn canonicalize_active_words(
    dec_module: &DecKernelModule,
    stream: &std::sync::Arc<CudaStream>,
    slot: &mut Option<DeviceBuffer<u64>>,
    active_words: &DeviceBuffer<u64>,
    activity_flags: &DeviceBuffer<u64>,
    words_per_child: usize,
    active_count: usize,
    k: usize,
) -> Result<Option<DeviceBuffer<u64>>, DecDeviceError> {
    if active_count == k {
        return Ok(None);
    }
    let mut canonical = take_child_buffer(slot, stream, k * words_per_child)?;
    launch_dec_scatter_active_words(
        dec_module,
        stream,
        active_words,
        activity_flags,
        words_per_child,
        active_count,
        k,
        &mut canonical,
    )?;
    Ok(Some(canonical))
}

fn zero_child_results(
    device: &Device,
    kernels: &SumcheckKernels,
    kappa: usize,
    shape: &DecShape,
    include_y_zcol: bool,
    retain_surfaces: bool,
) -> Result<ChildDeviceResults, DecDeviceError> {
    let y_words = DeviceBuffer::zeroed(device.stream(), shape.k * 2 * shape.t_mats * D)?;
    let y_zcol_words = if include_y_zcol {
        Some(DeviceBuffer::zeroed(device.stream(), shape.k * D * 2)?)
    } else {
        None
    };
    let resident_surfaces = retain_surfaces
        .then(|| {
            DevicePiCcsKSurfaces::pack_raw(
                device,
                kernels,
                &y_words,
                y_zcol_words.as_ref(),
                shape.k,
                shape.t_mats,
                shape.d_pad,
            )
        })
        .transpose()?;
    Ok(ChildDeviceResults {
        y_words: (!retain_surfaces).then(|| vec![0; shape.k * 2 * shape.t_mats * D]),
        commitments: vec![Commitment::zeros(D, kappa); shape.k],
        commitment_words: DeviceBuffer::zeroed(device.stream(), shape.k * kappa * D)?,
        y_zcol_words: (!retain_surfaces && include_y_zcol).then(|| vec![0; shape.k * D * 2]),
        resident_surfaces,
    })
}

fn tensor_point_len(point: &[K]) -> Result<usize, DecDeviceError> {
    1usize
        .checked_shl(point.len() as u32)
        .ok_or(DecDeviceError::Shape("tensor point challenge count overflow"))
}

fn k_slice_words(values: &[K]) -> Vec<u64> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        let (c0, c1) = value.to_limbs_u64();
        out.extend([c0, c1]);
    }
    out
}

/// Build the k child claims and witnesses from the downloaded device
/// results, exactly as the CPU `dec_reduction` assembles them. Children are
/// independent, so they build in parallel; the indexed collect keeps the
/// canonical child order.
fn assemble_children(
    pp: &Params,
    s: &Structure,
    parent: &CeClaim,
    shape: &DecShape,
    results: &ChildDeviceResults,
    planes: &[u64],
) -> Result<(Vec<CeClaim>, Vec<Mat<F>>), DecDeviceError> {
    let y_words_per_child = 2 * shape.t_mats * D;
    let y_words = results
        .y_words
        .as_ref()
        .ok_or(DecDeviceError::Shape("full child assembly is missing y surfaces"))?;

    let built = (0..shape.k)
        .into_par_iter()
        .map(|i| {
            let witness = ring_layout::mat_from_words(&planes[i * shape.len..(i + 1) * shape.len], shape.blocks);
            let y_ring = y_ring_from_words(&y_words[i * y_words_per_child..(i + 1) * y_words_per_child], shape);
            let ct = neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_ring, pp.inner(), s.m);
            let x = neo_reductions::common::project_x_from_witness_mat(&witness, s.m, parent.m_in)
                .map_err(|_| DecDeviceError::Shape("X projection failed"))?;
            let y_zcol = match &results.y_zcol_words {
                Some(words) => y_zcol_from_words(words, i, shape),
                None => Vec::new(),
            };
            let claim = child_claim(parent, i, results.commitments[i].clone(), x, y_ring, ct, y_zcol);
            Ok((claim, witness))
        })
        .collect::<Result<Vec<_>, DecDeviceError>>()?;
    Ok(built.into_iter().unzip())
}

/// Build child claims without materializing private child witnesses.
///
/// Public `X` is installed by the adapter after its independent parent-X
/// projection completes. Keeping zero shells here lets DEC child work and the
/// parent projection stay concurrent without downloading the split planes.
fn assemble_claim_shells(
    pp: &Params,
    s: &Structure,
    parent: &CeClaim,
    shape: &DecShape,
    kappa: usize,
    results: &ChildDeviceResults,
) -> Result<Vec<CeClaim>, DecDeviceError> {
    let y_words_per_child = 2 * shape.t_mats * D;

    (0..shape.k)
        .into_par_iter()
        .map(|i| {
            let y_ring = results.y_words.as_ref().map_or_else(
                || vec![vec![K::ZERO; shape.d_pad]; shape.t_mats],
                |words| y_ring_from_words(&words[i * y_words_per_child..(i + 1) * y_words_per_child], shape),
            );
            let ct = if results.y_words.is_some() {
                neo_reductions::common::ct_from_y_ring_for_ccs_m(&y_ring, pp.inner(), s.m)
            } else {
                vec![K::ZERO; shape.t_mats]
            };
            let y_zcol = match &results.y_zcol_words {
                Some(words) => y_zcol_from_words(words, i, shape),
                None => Vec::new(),
            };
            Ok(child_claim(
                parent,
                i,
                results
                    .commitments
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| Commitment::zeros(D, kappa)),
                Mat::zero(D, parent.m_in, F::ZERO),
                y_ring,
                ct,
                y_zcol,
            ))
        })
        .collect()
}

/// One child CE claim: the parent's shared fields plus this child's own
/// commitment, projection, and evaluations. Child 0 inherits the parent's
/// aux openings; the rest get zeros, as in the CPU reduction.
fn child_claim(
    parent: &CeClaim,
    index: usize,
    c: Commitment,
    x: Mat<F>,
    y_ring: Vec<Vec<K>>,
    ct: Vec<K>,
    y_zcol: Vec<K>,
) -> CeClaim {
    CeClaim {
        adv: None,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X: x,
        r: parent.r.clone(),
        s_col: parent.s_col.clone(),
        y_ring,
        ct,
        aux_openings: if index == 0 {
            parent.aux_openings.clone()
        } else {
            vec![K::ZERO; parent.aux_openings.len()]
        },
        y_zcol,
        m_in: parent.m_in,
        fold_digest: parent.fold_digest,
    }
}

/// Interleaved `(re, im)` coefficient rows → padded K rows, one per matrix.
fn y_ring_from_words(y_words: &[u64], shape: &DecShape) -> Vec<Vec<K>> {
    (0..shape.t_mats)
        .map(|j| {
            let re = &y_words[(2 * j) * D..(2 * j + 1) * D];
            let im = &y_words[(2 * j + 1) * D..(2 * j + 2) * D];
            let mut row = vec![K::ZERO; shape.d_pad];
            for c in 0..D {
                row[c] = k_from_device_words(re[c], im[c]);
            }
            row
        })
        .collect()
}

fn y_zcol_from_words(words: &[u64], child: usize, shape: &DecShape) -> Vec<K> {
    let mut yz = vec![K::ZERO; shape.d_pad.max(D)];
    for (rho, slot) in yz.iter_mut().take(D).enumerate() {
        let base = (child * D + rho) * 2;
        *slot = k_from_device_words(words[base], words[base + 1]);
    }
    yz.truncate(shape.d_pad);
    yz
}

/// The three prover-side reconstruction checks (`y`, `X`, `c`), identical to
/// the CPU engine's: children must recombine to the parent under Σ b^{i-1}.
fn verify_reconstruction(
    parent: &CeClaim,
    claims: &[CeClaim],
    commitments: &[Commitment],
    combine: DecMixer,
    shape: &DecShape,
    mode: DecRecompositionMode,
) -> Result<(), DecDeviceError> {
    let ok_y = match mode {
        DecRecompositionMode::DeferYAndX | DecRecompositionMode::DeferYAndXAndCommitment => true,
        DecRecompositionMode::Full
        | DecRecompositionMode::DeferX
        | DecRecompositionMode::DeferCommitment
        | DecRecompositionMode::DeferXAndCommitment => {
            let b_k = K::from(F::from_u64(shape.b as u64));
            (0..shape.t_mats).all(|j| {
                (0..shape.d_pad).all(|t| {
                    let mut acc = K::ZERO;
                    let mut pow = K::ONE;
                    for child in claims {
                        acc += pow * child.y_ring[j][t];
                        pow *= b_k;
                    }
                    acc == parent.y_ring[j][t]
                })
            })
        }
    };

    let ok_x = match mode {
        DecRecompositionMode::DeferX
        | DecRecompositionMode::DeferXAndCommitment
        | DecRecompositionMode::DeferYAndX
        | DecRecompositionMode::DeferYAndXAndCommitment => true,
        DecRecompositionMode::Full | DecRecompositionMode::DeferCommitment => {
            let b_f = F::from_u64(shape.b as u64);
            (0..D).all(|rho| {
                (0..parent.m_in).all(|c| {
                    let mut acc = F::ZERO;
                    let mut pow = F::ONE;
                    for child in claims {
                        acc += pow * child.X[(rho, c)];
                        pow *= b_f;
                    }
                    acc == parent.X[(rho, c)]
                })
            })
        }
    };

    let ok_c = match mode {
        DecRecompositionMode::Full | DecRecompositionMode::DeferX | DecRecompositionMode::DeferYAndX => {
            combine(commitments, shape.b) == parent.c
        }
        DecRecompositionMode::DeferCommitment
        | DecRecompositionMode::DeferXAndCommitment
        | DecRecompositionMode::DeferYAndXAndCommitment => true,
    };
    if ok_y && ok_x && ok_c {
        Ok(())
    } else {
        Err(DecDeviceError::PublicCheckFailed { ok_y, ok_x, ok_c })
    }
}

/// Finish the deferred y-side DEC recomposition check once the parent
/// y-surfaces are available on the host.
pub fn verify_y_recomposition(parent_y_ring: &[Vec<K>], claims: &[CeClaim], b: u32) -> Result<(), DecDeviceError> {
    let b_k = K::from(F::from_u64(b as u64));
    let ok_y = parent_y_ring.iter().enumerate().all(|(j, parent_row)| {
        (0..parent_row.len()).all(|t| {
            let mut acc = K::ZERO;
            let mut pow = K::ONE;
            for child in claims {
                acc += pow * child.y_ring[j][t];
                pow *= b_k;
            }
            acc == parent_row[t]
        })
    });
    if ok_y {
        Ok(())
    } else {
        Err(DecDeviceError::PublicCheckFailed {
            ok_y: false,
            ok_x: true,
            ok_c: true,
        })
    }
}

/// Finish the deferred X-side DEC recomposition check once the parent public
/// input projection is available on the host.
pub fn verify_x_recomposition(parent_x: &Mat<F>, claims: &[CeClaim], b: u32) -> Result<(), DecDeviceError> {
    let b_f = F::from_u64(b as u64);
    let ok_x = (0..D).all(|rho| {
        (0..parent_x.cols()).all(|c| {
            let mut acc = F::ZERO;
            let mut pow = F::ONE;
            for child in claims {
                acc += pow * child.X[(rho, c)];
                pow *= b_f;
            }
            acc == parent_x[(rho, c)]
        })
    });
    if ok_x {
        Ok(())
    } else {
        Err(DecDeviceError::PublicCheckFailed {
            ok_y: true,
            ok_x: false,
            ok_c: true,
        })
    }
}

/// Finish the deferred c-side DEC recomposition check once the parent
/// commitment is available on the host.
pub fn verify_commitment_recomposition(
    parent_c: &Commitment,
    claims: &[CeClaim],
    combine: DecMixer,
    b: u32,
) -> Result<(), DecDeviceError> {
    let commitments: Vec<Commitment> = claims.iter().map(|claim| claim.c.clone()).collect();
    if combine(&commitments, b) == *parent_c {
        Ok(())
    } else {
        Err(DecDeviceError::PublicCheckFailed {
            ok_y: true,
            ok_x: true,
            ok_c: false,
        })
    }
}
