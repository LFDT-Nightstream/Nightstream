//! Ring-algebra kernels over Rq = Fq[X]/Φ₈₁: the mat-vec
//! `out[p][r] = Σ_j mat[r][j] ⊗ z[p][j]` — the shared core of the Ajtai
//! commit (mat = PP) and the Π_DEC y_ring evaluation (mat = ring linear
//! forms) — plus the Π_RLC witness mix (per-column ring products against
//! the ρ elements). CSR-table kernels live in `kernels::csr`.
//!
//! Contract: bit-identical to `neo_ajtai::commit::commit_row_major`. The
//! product accumulates unreduced 2D-1 coefficients and applies the same Φ₈₁
//! reduction as `neo_math::ring::reduce_mod_phi_81` once at the end (the
//! reduction is linear, so late reduction is exact).
//!
//! Thread shape: stage 1 gives every *unreduced output coefficient* of every
//! (plane, row, column-chunk) its own thread, so the accumulator is one
//! register — a per-thread `[Gl; 107]` array would live in local memory and
//! throttle the whole kernel. Stage 2 sums chunks; stage 3 reduces mod Φ₈₁.
//!
//! Layouts (flat u64, canonical Goldilocks):
//! - `mat`: `[rows][cols][D]` row-major ring coefficients.
//! - `z`:   `[planes][cols][D]` column-major ring columns.
//! - `out`: `[planes][rows][D]` (`Commitment.data` layout per plane).

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::device::uninit_u64_device_buffer;
use crate::graph::GraphAllocations;
use crate::kernels::goldilocks::{mul_low_norm, reduce_192, Gl, GOLDILOCKS_MODULUS};

/// Ring degree; must equal `neo_math::D` (asserted host-side on upload).
pub const RING_D: usize = 54;
/// Unreduced product width `2D - 1`.
const PROD: usize = 2 * RING_D - 1;
const PHI_MID: usize = 27;

/// Columns each stage-1 thread walks. Larger chunks shrink the partials
/// buffer; the coefficient-parallel thread shape keeps occupancy high even
/// for few-row mats.
///
/// Perf notes (2026-07-02, two rejected experiments — do not retry naively):
/// (1) E=4 coefficient grouping with named accumulators: dec commits
/// 74→91ms. (2) Shared-memory (mat, z) tile per (plane, row, chunk) block:
/// dec 81→93ms, fold 310→320ms — the tile costs occupancy while the old
/// shape's reads were already warp-broadcast (mat) / coalesced (z). The
/// kernel sits near its EMULATED-64-BIT-MUL compute bound (~30 SASS ops
/// per Goldilocks mul); real wins need fewer ring multiplications
/// (Karatsuba/Toom over the coefficient windows), not access-pattern work.
pub(crate) const CHUNK_COLS: usize = 32;

pub use ajtai_kernels::LoadedModule as AjtaiKernelModule;

pub fn load_ajtai_kernels(ctx: &Arc<CudaContext>) -> Result<AjtaiKernelModule, EmbeddedModuleError> {
    ajtai_kernels::load(ctx)
}

/// Reusable stage buffers for [`ring_mat_vec`]. The unreduced partials run
/// ~50MB at real scale, and per-call `cuMemFree` of them dominated the
/// fold's CUDA API time (nsys 2026-07-02: 316ms/run, 52.8%). One scratch
/// per sequential owner; sized on first use and regrown when a larger launch
/// appears. The stage kernels fully overwrite the active prefix before any
/// read, so reuse must not enqueue a device memset.
#[derive(Default)]
pub struct RingMatVecScratch {
    binary_masks: Option<DeviceBuffer<u64>>,
    partials: Option<DeviceBuffer<u64>>,
    sums: Option<DeviceBuffer<u64>>,
}

impl RingMatVecScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate stage buffers before CUDA graph capture. The captured body
    /// will still zero the reused buffers, but it must not allocate them.
    pub fn prepare_mat_vec(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: usize,
        cols: usize,
        planes: usize,
    ) -> Result<(), DriverError> {
        let groups = planes * rows;
        let num_chunks = cols.div_ceil(CHUNK_COLS);
        ensure_scratch_capacity(&mut self.binary_masks, stream, planes * cols * 2)?;
        ensure_scratch_capacity(&mut self.partials, stream, groups * num_chunks * PROD)?;
        ensure_scratch_capacity(&mut self.sums, stream, groups * PROD)?;
        Ok(())
    }

    /// Prepare scratch for a dense-row matrix whose column chunks contain
    /// only statically touched block indices.
    pub fn prepare_sparse_mat_vec(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: usize,
        chunks: usize,
        planes: usize,
    ) -> Result<(), DriverError> {
        ensure_scratch_capacity(&mut self.partials, stream, planes * chunks * PROD)?;
        ensure_scratch_capacity(&mut self.sums, stream, planes * rows * PROD)?;
        Ok(())
    }

    pub(crate) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        if let Some(binary_masks) = &self.binary_masks {
            allocations.push(binary_masks);
        }
        if let Some(partials) = &self.partials {
            allocations.push(partials);
        }
        if let Some(sums) = &self.sums {
            allocations.push(sums);
        }
    }
}

fn scratch_buffer<'a>(
    slot: &'a mut Option<DeviceBuffer<u64>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<&'a mut DeviceBuffer<u64>, DriverError> {
    if slot.as_ref().is_none_or(|b| b.len() < len) {
        crate::perf_timed!("session.buffers", {
            *slot = Some(uninit_u64_device_buffer(stream, len)?);
        });
    }
    Ok(slot.as_mut().expect("allocated above"))
}

fn ensure_scratch_capacity(
    slot: &mut Option<DeviceBuffer<u64>>,
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<(), DriverError> {
    if slot.as_ref().is_none_or(|b| b.len() < len) {
        crate::perf_timed!("session.buffers", {
            *slot = Some(uninit_u64_device_buffer(stream, len)?);
        });
    }
    Ok(())
}

/// Ring mat-vec over `planes` input vectors (`plane_stride` words apart in
/// `z`, starting at `z_offset`) in one three-stage launch sequence.
/// Returns the `planes * rows * D` output words.
#[allow(clippy::too_many_arguments)]
pub fn ring_mat_vec(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    scratch: &mut RingMatVecScratch,
    mat: &DeviceBuffer<u64>,
    rows: usize,
    cols: usize,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    planes: usize,
    plane_stride: usize,
) -> Result<DeviceBuffer<u64>, DriverError> {
    let groups = planes * rows;
    let mut out = uninit_u64_device_buffer(stream, groups * RING_D)?;
    ring_mat_vec_into(
        module,
        stream,
        scratch,
        mat,
        rows,
        cols,
        z,
        z_offset,
        planes,
        plane_stride,
        &mut out,
    )?;
    Ok(out)
}

/// Same computation as [`ring_mat_vec`], writing into caller-owned output.
/// Whole-phase CUDA graph capture uses this form so device addresses are
/// stable across folds.
#[allow(clippy::too_many_arguments)]
pub fn ring_mat_vec_into(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    scratch: &mut RingMatVecScratch,
    mat: &DeviceBuffer<u64>,
    rows: usize,
    cols: usize,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    planes: usize,
    plane_stride: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let num_chunks = cols.div_ceil(CHUNK_COLS);
    let groups = planes * rows;
    assert!(
        out.len() >= groups * RING_D,
        "ring_mat_vec output too small: {} < {}",
        out.len(),
        groups * RING_D
    );

    let partials = scratch_buffer(&mut scratch.partials, stream, groups * num_chunks * PROD)?;
    module.mat_vec_coeff_partials(
        stream,
        LaunchConfig::for_num_elems((groups * num_chunks * PROD) as u32),
        mat,
        z,
        z_offset as u32,
        planes as u32,
        plane_stride as u32,
        rows as u32,
        cols as u32,
        num_chunks as u32,
        partials,
    )?;

    let sums = scratch_buffer(&mut scratch.sums, stream, groups * PROD)?;
    module.mat_vec_sum_chunks(
        stream,
        LaunchConfig::for_num_elems((groups * PROD) as u32),
        scratch.partials.as_ref().expect("stage 1 above"),
        groups as u32,
        num_chunks as u32,
        sums,
    )?;

    module.mat_vec_reduce_phi81(
        stream,
        LaunchConfig::for_num_elems(groups as u32),
        scratch.sums.as_ref().expect("stage 2 above"),
        groups as u32,
        out,
    )?;
    Ok(())
}

/// Ring mat-vec over dense form rows using a static touched-block schedule.
///
/// `mat[row][dense_col][D]` keeps its canonical dense layout while the
/// schedule skips form blocks that are structurally zero.
#[allow(clippy::too_many_arguments)]
pub fn ring_mat_vec_sparse_rows(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    scratch: &mut RingMatVecScratch,
    mat: &DeviceBuffer<u64>,
    rows: usize,
    dense_cols: usize,
    entry_blocks: &DeviceBuffer<u64>,
    chunk_rows: &DeviceBuffer<u64>,
    chunk_entry_starts: &DeviceBuffer<u64>,
    chunk_entry_lens: &DeviceBuffer<u64>,
    row_chunk_offsets: &DeviceBuffer<u64>,
    chunks: usize,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    planes: usize,
    plane_stride: usize,
) -> Result<DeviceBuffer<u64>, DriverError> {
    let mut out = uninit_u64_device_buffer(stream, planes * rows * RING_D)?;
    ring_mat_vec_sparse_rows_into(
        module,
        stream,
        scratch,
        mat,
        rows,
        dense_cols,
        entry_blocks,
        chunk_rows,
        chunk_entry_starts,
        chunk_entry_lens,
        row_chunk_offsets,
        chunks,
        z,
        z_offset,
        planes,
        plane_stride,
        &mut out,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn ring_mat_vec_sparse_rows_into(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    scratch: &mut RingMatVecScratch,
    mat: &DeviceBuffer<u64>,
    rows: usize,
    dense_cols: usize,
    entry_blocks: &DeviceBuffer<u64>,
    chunk_rows: &DeviceBuffer<u64>,
    chunk_entry_starts: &DeviceBuffer<u64>,
    chunk_entry_lens: &DeviceBuffer<u64>,
    row_chunk_offsets: &DeviceBuffer<u64>,
    chunks: usize,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    planes: usize,
    plane_stride: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let groups = planes * rows;
    assert_eq!(chunk_rows.len(), chunks, "sparse mat-vec chunk row count mismatch");
    assert_eq!(
        chunk_entry_starts.len(),
        chunks,
        "sparse mat-vec chunk start count mismatch"
    );
    assert_eq!(
        chunk_entry_lens.len(),
        chunks,
        "sparse mat-vec chunk length count mismatch"
    );
    assert_eq!(
        row_chunk_offsets.len(),
        rows + 1,
        "sparse mat-vec row offset count mismatch"
    );
    assert!(out.len() >= groups * RING_D, "sparse ring_mat_vec output too small");

    let partials = scratch_buffer(&mut scratch.partials, stream, planes * chunks * PROD)?;
    module.mat_vec_coeff_partials_sparse_rows(
        stream,
        LaunchConfig::for_num_elems((planes * chunks * PROD) as u32),
        mat,
        z,
        entry_blocks,
        chunk_rows,
        chunk_entry_starts,
        chunk_entry_lens,
        z_offset as u32,
        planes as u32,
        plane_stride as u32,
        rows as u32,
        dense_cols as u32,
        chunks as u32,
        partials,
    )?;

    let sums = scratch_buffer(&mut scratch.sums, stream, groups * PROD)?;
    module.mat_vec_sum_sparse_chunks(
        stream,
        LaunchConfig::for_num_elems((groups * PROD) as u32),
        scratch.partials.as_ref().expect("sparse stage 1 above"),
        row_chunk_offsets,
        planes as u32,
        rows as u32,
        chunks as u32,
        sums,
    )?;

    module.mat_vec_reduce_phi81(
        stream,
        LaunchConfig::for_num_elems(groups as u32),
        scratch.sums.as_ref().expect("sparse stage 2 above"),
        groups as u32,
        out,
    )?;
    Ok(())
}

/// Same ring mat-vec, but inactive planes are zeroed from a device flag
/// surface instead of being compacted by the host. This keeps DEC child
/// scheduling device-owned while avoiding dense work for zero digit planes.
#[allow(clippy::too_many_arguments)]
pub fn ring_mat_vec_active_flags_into(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    scratch: &mut RingMatVecScratch,
    mat: &DeviceBuffer<u64>,
    rows: usize,
    cols: usize,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    planes: usize,
    plane_stride: usize,
    active_flags: &DeviceBuffer<u64>,
    digit_base: u32,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let num_chunks = cols.div_ceil(CHUNK_COLS);
    let groups = planes * rows;
    assert!(
        out.len() >= groups * RING_D,
        "ring_mat_vec output too small: {} < {}",
        out.len(),
        groups * RING_D
    );

    scratch_buffer(&mut scratch.partials, stream, groups * num_chunks * PROD)?;
    if digit_base == 2 {
        scratch_buffer(&mut scratch.binary_masks, stream, planes * cols * 2)?;
        module.mat_vec_binary_masks(
            stream,
            LaunchConfig::for_num_elems((planes * cols) as u32),
            z,
            active_flags,
            z_offset as u32,
            planes as u32,
            plane_stride as u32,
            cols as u32,
            scratch
                .binary_masks
                .as_mut()
                .expect("binary mask scratch above"),
        )?;
        module.mat_vec_coeff_partials_binary_active_flags(
            stream,
            LaunchConfig::for_num_elems((groups * num_chunks * PROD) as u32),
            mat,
            scratch.binary_masks.as_ref().expect("binary masks above"),
            active_flags,
            planes as u32,
            rows as u32,
            cols as u32,
            num_chunks as u32,
            scratch.partials.as_mut().expect("partials scratch above"),
        )?;
    } else {
        module.mat_vec_coeff_partials_active_flags(
            stream,
            LaunchConfig::for_num_elems((groups * num_chunks * PROD) as u32),
            mat,
            z,
            active_flags,
            z_offset as u32,
            planes as u32,
            plane_stride as u32,
            rows as u32,
            cols as u32,
            num_chunks as u32,
            scratch.partials.as_mut().expect("partials scratch above"),
        )?;
    }

    let sums = scratch_buffer(&mut scratch.sums, stream, groups * PROD)?;
    module.mat_vec_sum_chunks(
        stream,
        LaunchConfig::for_num_elems((groups * PROD) as u32),
        scratch.partials.as_ref().expect("stage 1 above"),
        groups as u32,
        num_chunks as u32,
        sums,
    )?;

    module.mat_vec_reduce_phi81(
        stream,
        LaunchConfig::for_num_elems(groups as u32),
        scratch.sums.as_ref().expect("stage 2 above"),
        groups as u32,
        out,
    )?;
    Ok(())
}

/// RLC witness mix on device: `out[:, c] = Σ_i ρ_i ⊗ Z_i[:, c]` over Rq,
/// where `rhos` holds each ρ's ring coefficients (`[k1][D]`, the rotation
/// matrix's first column) and `zs` holds the k1 witnesses back to back in
/// the standard column-major layout (`[k1][cols][D]`). Returns `cols * D`
/// words in the same standard layout — a device-resident witness.
pub fn launch_rlc_mix(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    rhos: &DeviceBuffer<u64>,
    zs: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<DeviceBuffer<u64>, DriverError> {
    Ok(launch_rlc_mix_retained(module, stream, rhos, zs, k1, cols)?.into_out())
}

/// In-flight RLC mix whose unreduced reduction buffer must stay alive until
/// the output has been consumed. Dropping that buffer immediately after
/// enqueue can force the driver to synchronize before the mix is actually
/// useful to the caller.
pub struct PendingRlcMix {
    _sums: DeviceBuffer<u64>,
    out: DeviceBuffer<u64>,
}

impl PendingRlcMix {
    pub fn out(&self) -> &DeviceBuffer<u64> {
        &self.out
    }

    pub fn into_out(self) -> DeviceBuffer<u64> {
        self.out
    }

    pub(crate) fn into_parts(self) -> (DeviceBuffer<u64>, DeviceBuffer<u64>) {
        (self._sums, self.out)
    }
}

pub fn launch_rlc_mix_retained(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    rhos: &DeviceBuffer<u64>,
    zs: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<PendingRlcMix, DriverError> {
    let mut sums = uninit_u64_device_buffer(stream, cols * PROD)?;
    module.rlc_mix_partials(
        stream,
        LaunchConfig::for_num_elems((cols * PROD) as u32),
        rhos,
        zs,
        k1 as u32,
        cols as u32,
        &mut sums,
    )?;

    let mut out = uninit_u64_device_buffer(stream, cols * RING_D)?;
    module.mat_vec_reduce_phi81(
        stream,
        LaunchConfig::for_num_elems(cols as u32),
        &sums,
        cols as u32,
        &mut out,
    )?;
    Ok(PendingRlcMix { _sums: sums, out })
}

/// `dst[dst_offset .. dst_offset + src.len()] = src`, device-to-device.
/// cuda-oxide's buffer API has no offset copies, so composing one buffer
/// from resident pieces (fold planes from retained Π_DEC splits) goes
/// through this kernel; a copy is trivially bit-exact.
pub fn launch_plane_copy(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    dst_offset: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.plane_copy(
        stream,
        LaunchConfig::for_num_elems(src.len() as u32),
        src,
        dst_offset as u32,
        src.len() as u32,
        dst,
    )
}

/// `dst[dst_offset .. dst_offset + count] = src[src_offset .. src_offset + count]`.
pub fn launch_plane_copy_slice(
    module: &AjtaiKernelModule,
    stream: &Arc<CudaStream>,
    src: &DeviceBuffer<u64>,
    src_offset: usize,
    dst_offset: usize,
    count: usize,
    dst: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if count == 0 {
        return Ok(());
    }
    module.plane_copy_slice(
        stream,
        LaunchConfig::for_num_elems(count as u32),
        src,
        src_offset as u32,
        dst_offset as u32,
        count as u32,
        dst,
    )
}

#[cuda_module]
pub mod ajtai_kernels {
    use super::*;

    /// One thread per (plane, row, chunk, unreduced coefficient `e`):
    /// accumulate `Σ_{j ∈ chunk} Σ_{s+t=e} mat[row][j][s] · z[plane][j][t]`
    /// into a single register.
    #[kernel]
    pub fn mat_vec_coeff_partials(
        mat: &[u64],
        z: &[u64],
        z_offset: u32,
        planes: u32,
        plane_stride: u32,
        rows: u32,
        cols: u32,
        num_chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let rows = rows as usize;
        let cols = cols as usize;
        let num_chunks = num_chunks as usize;
        if idx >= planes * rows * num_chunks * PROD {
            return;
        }
        let e = idx % PROD;
        let rest = idx / PROD;
        let chunk = rest % num_chunks;
        let group = rest / num_chunks;
        let plane = group / rows;
        let row = group % rows;

        let col_start = chunk * CHUNK_COLS;
        let col_end = if col_start + CHUNK_COLS < cols {
            col_start + CHUNK_COLS
        } else {
            cols
        };
        let s_start = if e >= RING_D { e - (RING_D - 1) } else { 0 };
        let s_end = if e < RING_D { e + 1 } else { RING_D };
        let z_plane_base = z_offset as usize + plane * plane_stride as usize;

        // Lazy accumulation: raw 128-bit products land in a 192-bit
        // register triple per sign (named scalars — structs/arrays would
        // spill to local memory), reduced once at the end. Exact integer
        // arithmetic, so the sum is value-equal to reduce-per-term. The
        // sign split folds balanced low-norm z values to their magnitude
        // (a·z ≡ ∓a·m for z = q ∓ m) with no range fallback needed.
        let mut pos_lo = 0u64;
        let mut pos_mid = 0u64;
        let mut pos_hi = 0u64;
        let mut neg_lo = 0u64;
        let mut neg_mid = 0u64;
        let mut neg_hi = 0u64;
        for j in col_start..col_end {
            let a_base = (row * cols + j) * RING_D;
            let z_base = z_plane_base + j * RING_D;
            if a_base + RING_D > mat.len() || z_base + RING_D > z.len() {
                return;
            }
            for s in s_start..s_end {
                let a = mat[a_base + s];
                let zv = z[z_base + e - s];
                if zv == 0 {
                    continue;
                }
                let negz = zv > GOLDILOCKS_MODULUS / 2;
                let m = if negz { GOLDILOCKS_MODULUS - zv } else { zv };
                let p = (a as u128) * (m as u128);
                let p_lo = p as u64;
                let p_mid = (p >> 64) as u64;
                if negz {
                    let (lo, c1) = neg_lo.overflowing_add(p_lo);
                    let (mid, c2) = neg_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    neg_hi += (c2 as u64) + (c3 as u64);
                    neg_lo = lo;
                    neg_mid = mid;
                } else {
                    let (lo, c1) = pos_lo.overflowing_add(p_lo);
                    let (mid, c2) = pos_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    pos_hi += (c2 as u64) + (c3 as u64);
                    pos_lo = lo;
                    pos_mid = mid;
                }
            }
        }
        let acc = reduce_192(pos_lo, pos_mid, pos_hi) - reduce_192(neg_lo, neg_mid, neg_hi);

        let slot = (group * num_chunks + chunk) * PROD + e;
        if slot >= partials.len() {
            return;
        }
        unsafe {
            *partials.get_unchecked_mut(slot) = acc.as_canonical_u64();
        }
    }

    /// Dense form rows multiplied only at statically touched block indices.
    /// One thread still owns one unreduced coefficient of one row chunk; the
    /// chunk's entries name original dense columns for both `mat` and `z`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn mat_vec_coeff_partials_sparse_rows(
        mat: &[u64],
        z: &[u64],
        entry_blocks: &[u64],
        chunk_rows: &[u64],
        chunk_entry_starts: &[u64],
        chunk_entry_lens: &[u64],
        z_offset: u32,
        planes: u32,
        plane_stride: u32,
        rows: u32,
        dense_cols: u32,
        chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let rows = rows as usize;
        let dense_cols = dense_cols as usize;
        let chunks = chunks as usize;
        if idx >= planes * chunks * PROD {
            return;
        }
        let e = idx % PROD;
        let rest = idx / PROD;
        let chunk = rest % chunks;
        let plane = rest / chunks;
        if chunk >= chunk_rows.len() || chunk >= chunk_entry_starts.len() || chunk >= chunk_entry_lens.len() {
            return;
        }
        let row = chunk_rows[chunk] as usize;
        if row >= rows {
            return;
        }
        let entry_start = chunk_entry_starts[chunk] as usize;
        let entry_end = entry_start + chunk_entry_lens[chunk] as usize;
        if entry_end > entry_blocks.len() {
            return;
        }
        let s_start = if e >= RING_D { e - (RING_D - 1) } else { 0 };
        let s_end = if e < RING_D { e + 1 } else { RING_D };
        let z_plane_base = z_offset as usize + plane * plane_stride as usize;

        let mut pos_lo = 0u64;
        let mut pos_mid = 0u64;
        let mut pos_hi = 0u64;
        let mut neg_lo = 0u64;
        let mut neg_mid = 0u64;
        let mut neg_hi = 0u64;
        for entry in entry_start..entry_end {
            let col = entry_blocks[entry] as usize;
            if col >= dense_cols {
                return;
            }
            let a_base = (row * dense_cols + col) * RING_D;
            let z_base = z_plane_base + col * RING_D;
            if a_base + RING_D > mat.len() || z_base + RING_D > z.len() {
                return;
            }
            for s in s_start..s_end {
                let a = mat[a_base + s];
                let zv = z[z_base + e - s];
                if zv == 0 {
                    continue;
                }
                let negz = zv > GOLDILOCKS_MODULUS / 2;
                let m = if negz { GOLDILOCKS_MODULUS - zv } else { zv };
                let p = (a as u128) * (m as u128);
                let p_lo = p as u64;
                let p_mid = (p >> 64) as u64;
                if negz {
                    let (lo, c1) = neg_lo.overflowing_add(p_lo);
                    let (mid, c2) = neg_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    neg_hi += (c2 as u64) + (c3 as u64);
                    neg_lo = lo;
                    neg_mid = mid;
                } else {
                    let (lo, c1) = pos_lo.overflowing_add(p_lo);
                    let (mid, c2) = pos_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    pos_hi += (c2 as u64) + (c3 as u64);
                    pos_lo = lo;
                    pos_mid = mid;
                }
            }
        }
        let acc = reduce_192(pos_lo, pos_mid, pos_hi) - reduce_192(neg_lo, neg_mid, neg_hi);
        let slot = (plane * chunks + chunk) * PROD + e;
        if slot < partials.len() {
            unsafe {
                *partials.get_unchecked_mut(slot) = acc.as_canonical_u64();
            }
        }
    }

    /// Flagged variant used by Π_DEC. Inactive child planes still get their
    /// partial slots fully written, but skip the expensive ring multiply loop.
    #[kernel]
    pub fn mat_vec_coeff_partials_active_flags(
        mat: &[u64],
        z: &[u64],
        active_flags: &[u64],
        z_offset: u32,
        planes: u32,
        plane_stride: u32,
        rows: u32,
        cols: u32,
        num_chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let rows = rows as usize;
        let cols = cols as usize;
        let num_chunks = num_chunks as usize;
        if idx >= planes * rows * num_chunks * PROD {
            return;
        }
        let e = idx % PROD;
        let rest = idx / PROD;
        let chunk = rest % num_chunks;
        let group = rest / num_chunks;
        let plane = group / rows;
        let slot = (group * num_chunks + chunk) * PROD + e;
        if slot >= partials.len() {
            return;
        }
        if plane >= active_flags.len() || active_flags[plane] == 0 {
            unsafe {
                *partials.get_unchecked_mut(slot) = 0;
            }
            return;
        }
        let row = group % rows;

        let col_start = chunk * CHUNK_COLS;
        let col_end = if col_start + CHUNK_COLS < cols {
            col_start + CHUNK_COLS
        } else {
            cols
        };
        let s_start = if e >= RING_D { e - (RING_D - 1) } else { 0 };
        let s_end = if e < RING_D { e + 1 } else { RING_D };
        let z_plane_base = z_offset as usize + plane * plane_stride as usize;

        let mut pos_lo = 0u64;
        let mut pos_mid = 0u64;
        let mut pos_hi = 0u64;
        let mut neg_lo = 0u64;
        let mut neg_mid = 0u64;
        let mut neg_hi = 0u64;
        for j in col_start..col_end {
            let a_base = (row * cols + j) * RING_D;
            let z_base = z_plane_base + j * RING_D;
            if a_base + RING_D > mat.len() || z_base + RING_D > z.len() {
                return;
            }
            for s in s_start..s_end {
                let a = mat[a_base + s];
                let zv = z[z_base + e - s];
                if zv == 0 {
                    continue;
                }
                let negz = zv > GOLDILOCKS_MODULUS / 2;
                let m = if negz { GOLDILOCKS_MODULUS - zv } else { zv };
                let p = (a as u128) * (m as u128);
                let p_lo = p as u64;
                let p_mid = (p >> 64) as u64;
                if negz {
                    let (lo, c1) = neg_lo.overflowing_add(p_lo);
                    let (mid, c2) = neg_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    neg_hi += (c2 as u64) + (c3 as u64);
                    neg_lo = lo;
                    neg_mid = mid;
                } else {
                    let (lo, c1) = pos_lo.overflowing_add(p_lo);
                    let (mid, c2) = pos_mid.overflowing_add(p_mid);
                    let (mid, c3) = mid.overflowing_add(c1 as u64);
                    pos_hi += (c2 as u64) + (c3 as u64);
                    pos_lo = lo;
                    pos_mid = mid;
                }
            }
        }
        let acc = reduce_192(pos_lo, pos_mid, pos_hi) - reduce_192(neg_lo, neg_mid, neg_hi);
        unsafe {
            *partials.get_unchecked_mut(slot) = acc.as_canonical_u64();
        }
    }

    /// Compact every base-2 ring into positive/negative coefficient masks.
    /// The same split ring is reused for every matrix row and unreduced
    /// coefficient, so paying this scan once avoids re-reading zeros in the
    /// convolution kernel below.
    #[kernel]
    pub fn mat_vec_binary_masks(
        z: &[u64],
        active_flags: &[u64],
        z_offset: u32,
        planes: u32,
        plane_stride: u32,
        cols: u32,
        mut masks: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let cols = cols as usize;
        if idx >= planes * cols || 2 * idx + 1 >= masks.len() {
            return;
        }
        let plane = idx / cols;
        if plane >= active_flags.len() || active_flags[plane] == 0 {
            unsafe {
                *masks.get_unchecked_mut(2 * idx) = 0;
                *masks.get_unchecked_mut(2 * idx + 1) = 0;
            }
            return;
        }
        let col = idx % cols;
        let z_base = z_offset as usize + plane * plane_stride as usize + col * RING_D;
        if z_base + RING_D > z.len() {
            return;
        }
        let mut positive = 0u64;
        let mut negative = 0u64;
        for coeff in 0..RING_D {
            let value = z[z_base + coeff];
            if value == GOLDILOCKS_MODULUS - 1 {
                negative |= 1u64 << coeff;
            } else if value != 0 {
                positive |= 1u64 << coeff;
            }
        }
        unsafe {
            *masks.get_unchecked_mut(2 * idx) = positive;
            *masks.get_unchecked_mut(2 * idx + 1) = negative;
        }
    }

    /// Base-2 Π_DEC specialization. Split-plane entries are represented by
    /// two masks per ring, so each convolution thread visits only the
    /// nonzero `{-1, +1}` coefficients.
    #[kernel]
    pub fn mat_vec_coeff_partials_binary_active_flags(
        mat: &[u64],
        masks: &[u64],
        active_flags: &[u64],
        planes: u32,
        rows: u32,
        cols: u32,
        num_chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let rows = rows as usize;
        let cols = cols as usize;
        let num_chunks = num_chunks as usize;
        if idx >= planes * rows * num_chunks * PROD {
            return;
        }
        let e = idx % PROD;
        let rest = idx / PROD;
        let chunk = rest % num_chunks;
        let group = rest / num_chunks;
        let plane = group / rows;
        let slot = (group * num_chunks + chunk) * PROD + e;
        if slot >= partials.len() {
            return;
        }
        if plane >= active_flags.len() || active_flags[plane] == 0 {
            unsafe {
                *partials.get_unchecked_mut(slot) = 0;
            }
            return;
        }
        let row = group % rows;

        let col_start = chunk * CHUNK_COLS;
        let col_end = if col_start + CHUNK_COLS < cols {
            col_start + CHUNK_COLS
        } else {
            cols
        };
        let t_start = if e >= RING_D { e - (RING_D - 1) } else { 0 };
        let t_end = if e < RING_D { e } else { RING_D - 1 };
        let valid_coeffs = (!0u64 << t_start) & ((1u64 << (t_end + 1)) - 1);

        let mut pos_lo = 0u64;
        let mut pos_hi = 0u64;
        let mut neg_lo = 0u64;
        let mut neg_hi = 0u64;
        for j in col_start..col_end {
            let a_base = (row * cols + j) * RING_D;
            let mask_base = 2 * (plane * cols + j);
            if a_base + RING_D > mat.len() || mask_base + 1 >= masks.len() {
                return;
            }
            let mut positive = masks[mask_base] & valid_coeffs;
            while positive != 0 {
                let t = positive.trailing_zeros() as usize;
                positive &= positive - 1;
                let a = mat[a_base + e - t];
                let (lo, carry) = pos_lo.overflowing_add(a);
                pos_lo = lo;
                pos_hi += carry as u64;
            }
            let mut negative = masks[mask_base + 1] & valid_coeffs;
            while negative != 0 {
                let t = negative.trailing_zeros() as usize;
                negative &= negative - 1;
                let a = mat[a_base + e - t];
                let (lo, carry) = neg_lo.overflowing_add(a);
                neg_lo = lo;
                neg_hi += carry as u64;
            }
        }
        let acc = reduce_192(pos_lo, pos_hi, 0) - reduce_192(neg_lo, neg_hi, 0);
        unsafe {
            *partials.get_unchecked_mut(slot) = acc.as_canonical_u64();
        }
    }

    /// One thread per (group, coefficient): fold the per-chunk partials.
    #[kernel]
    pub fn mat_vec_sum_chunks(partials: &[u64], groups: u32, num_chunks: u32, mut sums: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let groups = groups as usize;
        let num_chunks = num_chunks as usize;
        if idx >= groups * PROD {
            return;
        }
        let group = idx / PROD;
        let e = idx % PROD;
        let mut acc = Gl::ZERO;
        for chunk in 0..num_chunks {
            let slot = (group * num_chunks + chunk) * PROD + e;
            if slot >= partials.len() {
                return;
            }
            acc = acc + Gl::from_u64(partials[slot]);
        }
        if idx >= sums.len() {
            return;
        }
        unsafe {
            *sums.get_unchecked_mut(idx) = acc.as_canonical_u64();
        }
    }

    /// Fold variable sparse-chunk ranges into the canonical dense row sums.
    #[kernel]
    pub fn mat_vec_sum_sparse_chunks(
        partials: &[u64],
        row_chunk_offsets: &[u64],
        planes: u32,
        rows: u32,
        chunks: u32,
        mut sums: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let planes = planes as usize;
        let rows = rows as usize;
        let chunks = chunks as usize;
        if idx >= planes * rows * PROD {
            return;
        }
        let e = idx % PROD;
        let group = idx / PROD;
        let plane = group / rows;
        let row = group % rows;
        if row + 1 >= row_chunk_offsets.len() {
            return;
        }
        let start = row_chunk_offsets[row] as usize;
        let end = row_chunk_offsets[row + 1] as usize;
        if end > chunks {
            return;
        }
        let mut acc = Gl::ZERO;
        for chunk in start..end {
            let slot = (plane * chunks + chunk) * PROD + e;
            if slot >= partials.len() {
                return;
            }
            acc = acc + Gl::from_u64(partials[slot]);
        }
        if idx < sums.len() {
            unsafe {
                *sums.get_unchecked_mut(idx) = acc.as_canonical_u64();
            }
        }
    }

    /// One thread per group: Φ₈₁-reduce the 2D-1 unreduced coefficients to
    /// D, mirroring `neo_math::ring::reduce_mod_phi_81`.
    #[kernel]
    pub fn mat_vec_reduce_phi81(sums: &[u64], groups: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        if idx >= groups as usize {
            return;
        }
        let base = idx * PROD;
        if base + PROD > sums.len() {
            return;
        }
        let mut acc = [Gl::ZERO; PROD];
        for e in 0..PROD {
            acc[e] = Gl::from_u64(sums[base + e]);
        }
        for e in (RING_D..PROD).rev() {
            let t = acc[e];
            acc[e] = Gl::ZERO;
            acc[e - RING_D] = acc[e - RING_D] - t;
            let idx_27 = e - PHI_MID;
            if idx_27 < RING_D {
                acc[idx_27] = acc[idx_27] - t;
            } else {
                acc[idx_27 - RING_D] = acc[idx_27 - RING_D] + t;
                if idx_27 - PHI_MID < RING_D {
                    acc[idx_27 - PHI_MID] = acc[idx_27 - PHI_MID] + t;
                }
            }
        }
        let out_base = idx * RING_D;
        for (r, coeff) in acc.iter().take(RING_D).enumerate() {
            let slot = out_base + r;
            if slot >= out.len() {
                return;
            }
            unsafe {
                *out.get_unchecked_mut(slot) = coeff.as_canonical_u64();
            }
        }
    }

    /// One thread per word: offset device-to-device copy.
    #[kernel]
    pub fn plane_copy(src: &[u64], dst_offset: u32, count: u32, mut dst: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        if idx >= count as usize || idx >= src.len() {
            return;
        }
        let slot = dst_offset as usize + idx;
        if slot >= dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(slot) = src[idx];
        }
    }

    #[kernel]
    pub fn plane_copy_slice(src: &[u64], src_offset: u32, dst_offset: u32, count: u32, mut dst: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        if idx >= count as usize {
            return;
        }
        let src_slot = src_offset as usize + idx;
        let dst_slot = dst_offset as usize + idx;
        if src_slot >= src.len() || dst_slot >= dst.len() {
            return;
        }
        unsafe {
            *dst.get_unchecked_mut(dst_slot) = src[src_slot];
        }
    }

    /// One thread per (column, unreduced coefficient `e`): accumulate the
    /// RLC ring products `Σ_i Σ_{s+t=e} rho[i][s] · z_i[col][t]` into a
    /// single register, writing the unreduced sums for `mat_vec_reduce_phi81`.
    #[kernel]
    pub fn rlc_mix_partials(rhos: &[u64], zs: &[u64], k1: u32, cols: u32, mut sums: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let k1 = k1 as usize;
        let cols = cols as usize;
        if idx >= cols * PROD {
            return;
        }
        let col = idx / PROD;
        let e = idx % PROD;
        let s_start = if e >= RING_D { e - (RING_D - 1) } else { 0 };
        let s_end = if e < RING_D { e + 1 } else { RING_D };
        let wit_stride = cols * RING_D;

        let mut acc = Gl::ZERO;
        for i in 0..k1 {
            let rho_base = i * RING_D;
            let z_base = i * wit_stride + col * RING_D;
            if rho_base + RING_D > rhos.len() || z_base + RING_D > zs.len() {
                return;
            }
            for s in s_start..s_end {
                acc = acc + mul_low_norm(Gl::from_u64(rhos[rho_base + s]), zs[z_base + e - s]);
            }
        }
        if idx >= sums.len() {
            return;
        }
        unsafe {
            *sums.get_unchecked_mut(idx) = acc.as_canonical_u64();
        }
    }
}
