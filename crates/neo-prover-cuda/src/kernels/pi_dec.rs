//! Π_DEC digit-split kernel.
//!
//! Contract: bit-identical to `neo_reductions::common::
//! split_b_matrix_k_with_nonzero_flags` (balanced digits, base-2 fast path,
//! truncated `%` for general b). Out-of-range parent values set the error
//! flag instead of erroring; the host must check it and fall back to the CPU
//! path, which owns the detailed error message.
//!
//! Layouts (flat u64, canonical Goldilocks):
//! - `z`: `[m][D]` column-major parent witness ring columns.
//! - `planes`: `[k][m*D]` child digit planes (fully written by launch).
//! - `flags`: `[k]` nonzero-plane markers plus one trailing error word
//!   (`flags[k]`), written racily but monotonically (0 → 1 only).

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

use crate::device::uninit_u64_device_buffer;
use crate::kernels::ajtai::RING_D as RING_LANES;
use crate::kernels::goldilocks::{Gl, Kx, GOLDILOCKS_MODULUS};

pub use dec_kernels::LoadedModule as DecKernelModule;

const Y_ZCOL_CHUNK_COLS: usize = 8;

pub fn load_dec_kernels(ctx: &Arc<cuda_core::CudaContext>) -> Result<DecKernelModule, cuda_host::EmbeddedModuleError> {
    dec_kernels::load(ctx)
}

pub fn dec_y_zcol_partials_words(m: usize, k: usize) -> usize {
    let lane_cols = m.div_ceil(RING_LANES);
    let chunks = lane_cols.div_ceil(Y_ZCOL_CHUNK_COLS).max(1);
    if chunks <= 1 {
        0
    } else {
        k * RING_LANES * chunks * 2
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_dec_split(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    z: &DeviceBuffer<u64>,
    len: usize,
    k: usize,
    b: u32,
    big_b: u64,
    planes: &mut DeviceBuffer<u64>,
    flags: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_split(
        stream,
        LaunchConfig::for_num_elems(len as u32),
        z,
        len as u32,
        k as u32,
        b,
        big_b,
        planes,
        flags,
    )
}

/// Merge the current split's trailing error flag into a session-resident
/// sticky status word. Stream ordering makes the split writes visible before
/// this kernel and preserves every prior fold's status without a host join.
pub fn launch_dec_accumulate_status(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    flags: &DeviceBuffer<u64>,
    k: usize,
    sticky_status: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_accumulate_status(stream, LaunchConfig::for_num_elems(1), flags, k as u32, sticky_status)
}

/// Per-(child, lane) NC opening: `y_zcol[child][rho] = Σ_{col ≡ rho (mod D),
/// col < m} χ_s[col] · plane[child][col]` — the same strided weighted sum
/// `compute_y_zcol_from_witness` walks, in the same ascending-column order.
pub fn launch_dec_y_zcol(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    chi_s: &DeviceBuffer<u64>,
    m: usize,
    plane_stride: usize,
    k: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let lane_cols = m.div_ceil(RING_LANES);
    let chunks = lane_cols.div_ceil(Y_ZCOL_CHUNK_COLS).max(1);
    if chunks == 1 {
        return module.dec_y_zcol(
            stream,
            LaunchConfig::for_num_elems((k * RING_LANES) as u32),
            planes,
            chi_s,
            m as u32,
            plane_stride as u32,
            k as u32,
            out,
        );
    }

    let mut partials = uninit_u64_device_buffer(stream, k * RING_LANES * chunks * 2)?;
    module.dec_y_zcol_partials(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES * chunks) as u32),
        planes,
        chi_s,
        m as u32,
        plane_stride as u32,
        k as u32,
        chunks as u32,
        &mut partials,
    )?;
    module.dec_y_zcol_reduce(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES) as u32),
        &partials,
        k as u32,
        chunks as u32,
        out,
    )
}

pub fn launch_dec_y_zcol_active_flags(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    active_flags: &DeviceBuffer<u64>,
    chi_s: &DeviceBuffer<u64>,
    m: usize,
    plane_stride: usize,
    k: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let lane_cols = m.div_ceil(RING_LANES);
    let chunks = lane_cols.div_ceil(Y_ZCOL_CHUNK_COLS).max(1);
    if chunks == 1 {
        return module.dec_y_zcol_active_flags(
            stream,
            LaunchConfig::for_num_elems((k * RING_LANES) as u32),
            planes,
            active_flags,
            chi_s,
            m as u32,
            plane_stride as u32,
            k as u32,
            out,
        );
    }

    let mut partials = uninit_u64_device_buffer(stream, k * RING_LANES * chunks * 2)?;
    module.dec_y_zcol_partials_active_flags(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES * chunks) as u32),
        planes,
        active_flags,
        chi_s,
        m as u32,
        plane_stride as u32,
        k as u32,
        chunks as u32,
        &mut partials,
    )?;
    module.dec_y_zcol_reduce(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES) as u32),
        &partials,
        k as u32,
        chunks as u32,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_dec_y_zcol_active_flags_with_partials(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    active_flags: &DeviceBuffer<u64>,
    chi_s: &DeviceBuffer<u64>,
    m: usize,
    plane_stride: usize,
    k: usize,
    partials: &mut DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let lane_cols = m.div_ceil(RING_LANES);
    let chunks = lane_cols.div_ceil(Y_ZCOL_CHUNK_COLS).max(1);
    if chunks == 1 {
        return module.dec_y_zcol_active_flags(
            stream,
            LaunchConfig::for_num_elems((k * RING_LANES) as u32),
            planes,
            active_flags,
            chi_s,
            m as u32,
            plane_stride as u32,
            k as u32,
            out,
        );
    }

    assert!(
        partials.len() >= k * RING_LANES * chunks * 2,
        "DEC y_zcol partials buffer too small"
    );
    module.dec_y_zcol_partials_active_flags(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES * chunks) as u32),
        planes,
        active_flags,
        chi_s,
        m as u32,
        plane_stride as u32,
        k as u32,
        chunks as u32,
        partials,
    )?;
    module.dec_y_zcol_reduce(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES) as u32),
        partials,
        k as u32,
        chunks as u32,
        out,
    )
}

/// Pack only the public `X` surface from split child planes.
///
/// Output layout is child-major row-major `Mat<F>` data:
/// `[child][row][x_col]`, with columns outside `ceil(m_in / D)` left zero.
/// This mirrors `project_x_from_witness_mat` without downloading full
/// private child witnesses.
pub fn launch_dec_pack_public_x(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    m_in: usize,
    plane_stride: usize,
    k: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_pack_public_x(
        stream,
        LaunchConfig::for_num_elems((k * RING_LANES * m_in) as u32),
        planes,
        m_in as u32,
        plane_stride as u32,
        k as u32,
        out,
    )
}

pub fn launch_dec_build_active_index(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    flags: &DeviceBuffer<u64>,
    k: usize,
    active_indices: &mut DeviceBuffer<u64>,
    active_count: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_build_active_index(
        stream,
        LaunchConfig::for_num_elems(1),
        flags,
        k as u32,
        active_indices,
        active_count,
    )
}

pub fn launch_dec_compact_active_planes(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    active_indices: &DeviceBuffer<u64>,
    plane_stride: usize,
    active_count: usize,
    compact: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_compact_active_planes(
        stream,
        LaunchConfig::for_num_elems((active_count * plane_stride) as u32),
        planes,
        active_indices,
        plane_stride as u32,
        active_count as u32,
        compact,
    )
}

pub fn launch_dec_scatter_active_words(
    module: &DecKernelModule,
    stream: &Arc<CudaStream>,
    active_words: &DeviceBuffer<u64>,
    activity_flags: &DeviceBuffer<u64>,
    words_per_child: usize,
    active_count: usize,
    k: usize,
    canonical_words: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.dec_scatter_active_words(
        stream,
        LaunchConfig::for_num_elems((k * words_per_child) as u32),
        active_words,
        activity_flags,
        words_per_child as u32,
        active_count as u32,
        k as u32,
        canonical_words,
    )
}

#[cuda_module]
pub mod dec_kernels {
    use super::*;

    /// One thread per (child, lane): ascending-column strided weighted sum
    /// of the child's digit plane against χ_s. Used for tiny inputs where
    /// chunking would add more launch overhead than parallel work.
    #[kernel]
    pub fn dec_y_zcol(planes: &[u64], chi_s: &[u64], m: u32, plane_stride: u32, k: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let m = m as usize;
        let plane_stride = plane_stride as usize;
        if idx >= (k as usize) * RING_LANES {
            return;
        }
        let child = idx / RING_LANES;
        let rho = idx % RING_LANES;
        let base = child * plane_stride;
        let mut acc = Kx::ZERO;
        let mut col = rho;
        while col < m {
            if base + col >= planes.len() || 2 * col + 2 > chi_s.len() {
                return;
            }
            let value = Gl::from_u64(planes[base + col]);
            let weight = Kx::from_words(chi_s[2 * col], chi_s[2 * col + 1]);
            acc = acc + weight.scale_base(value);
            col += RING_LANES;
        }
        let at = 2 * idx;
        if at + 2 > out.len() {
            return;
        }
        let words = acc.as_words();
        unsafe {
            *out.get_unchecked_mut(at) = words[0];
            *out.get_unchecked_mut(at + 1) = words[1];
        }
    }

    #[kernel]
    pub fn dec_y_zcol_active_flags(
        planes: &[u64],
        active_flags: &[u64],
        chi_s: &[u64],
        m: u32,
        plane_stride: u32,
        k: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let m = m as usize;
        let plane_stride = plane_stride as usize;
        if idx >= (k as usize) * RING_LANES {
            return;
        }
        let child = idx / RING_LANES;
        let at = 2 * idx;
        if at + 2 > out.len() {
            return;
        }
        if child >= active_flags.len() || active_flags[child] == 0 {
            unsafe {
                *out.get_unchecked_mut(at) = 0;
                *out.get_unchecked_mut(at + 1) = 0;
            }
            return;
        }

        let rho = idx % RING_LANES;
        let base = child * plane_stride;
        let mut acc = Kx::ZERO;
        let mut col = rho;
        while col < m {
            if base + col >= planes.len() || 2 * col + 2 > chi_s.len() {
                return;
            }
            let value = Gl::from_u64(planes[base + col]);
            let weight = Kx::from_words(chi_s[2 * col], chi_s[2 * col + 1]);
            acc = acc + weight.scale_base(value);
            col += RING_LANES;
        }
        let words = acc.as_words();
        unsafe {
            *out.get_unchecked_mut(at) = words[0];
            *out.get_unchecked_mut(at + 1) = words[1];
        }
    }

    /// One thread per (child, lane, column chunk). This exposes the column
    /// axis of the NC opening so large DEC shapes do not run as a few hundred
    /// long-loop threads on a 128-SM GPU.
    #[kernel]
    pub fn dec_y_zcol_partials(
        planes: &[u64],
        chi_s: &[u64],
        m: u32,
        plane_stride: u32,
        k: u32,
        chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let chunks = chunks as usize;
        if chunks == 0 {
            return;
        }
        let total = (k as usize) * RING_LANES * chunks;
        if idx >= total {
            return;
        }

        let chunk = idx % chunks;
        let lane_idx = idx / chunks;
        let child = lane_idx / RING_LANES;
        let rho = lane_idx % RING_LANES;
        let m = m as usize;
        let plane_stride = plane_stride as usize;
        let base = child * plane_stride;
        let mut acc = Kx::ZERO;
        let start = chunk * Y_ZCOL_CHUNK_COLS;
        let end = start + Y_ZCOL_CHUNK_COLS;
        let mut lane_col = start;
        let mut col = rho + start * RING_LANES;
        while lane_col < end && col < m {
            if base + col >= planes.len() || 2 * col + 2 > chi_s.len() {
                return;
            }
            let value = Gl::from_u64(planes[base + col]);
            let weight = Kx::from_words(chi_s[2 * col], chi_s[2 * col + 1]);
            acc = acc + weight.scale_base(value);
            lane_col += 1;
            col += RING_LANES;
        }

        let at = 2 * idx;
        if at + 2 > partials.len() {
            return;
        }
        let words = acc.as_words();
        unsafe {
            *partials.get_unchecked_mut(at) = words[0];
            *partials.get_unchecked_mut(at + 1) = words[1];
        }
    }

    #[kernel]
    pub fn dec_y_zcol_partials_active_flags(
        planes: &[u64],
        active_flags: &[u64],
        chi_s: &[u64],
        m: u32,
        plane_stride: u32,
        k: u32,
        chunks: u32,
        mut partials: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let chunks = chunks as usize;
        if chunks == 0 {
            return;
        }
        let total = (k as usize) * RING_LANES * chunks;
        if idx >= total {
            return;
        }

        let chunk = idx % chunks;
        let lane_idx = idx / chunks;
        let child = lane_idx / RING_LANES;
        let at = 2 * idx;
        if at + 2 > partials.len() {
            return;
        }
        if child >= active_flags.len() || active_flags[child] == 0 {
            unsafe {
                *partials.get_unchecked_mut(at) = 0;
                *partials.get_unchecked_mut(at + 1) = 0;
            }
            return;
        }

        let rho = lane_idx % RING_LANES;
        let m = m as usize;
        let plane_stride = plane_stride as usize;
        let base = child * plane_stride;
        let mut acc = Kx::ZERO;
        let start = chunk * Y_ZCOL_CHUNK_COLS;
        let end = start + Y_ZCOL_CHUNK_COLS;
        let mut lane_col = start;
        let mut col = rho + start * RING_LANES;
        while lane_col < end && col < m {
            if base + col >= planes.len() || 2 * col + 2 > chi_s.len() {
                return;
            }
            let value = Gl::from_u64(planes[base + col]);
            let weight = Kx::from_words(chi_s[2 * col], chi_s[2 * col + 1]);
            acc = acc + weight.scale_base(value);
            lane_col += 1;
            col += RING_LANES;
        }

        let words = acc.as_words();
        unsafe {
            *partials.get_unchecked_mut(at) = words[0];
            *partials.get_unchecked_mut(at + 1) = words[1];
        }
    }

    /// Reduce `dec_y_zcol_partials` in ascending chunk order for each
    /// (child, lane) output.
    #[kernel]
    pub fn dec_y_zcol_reduce(partials: &[u64], k: u32, chunks: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let chunks = chunks as usize;
        if idx >= (k as usize) * RING_LANES || chunks == 0 {
            return;
        }

        let mut acc = Kx::ZERO;
        let base = idx * chunks;
        for chunk in 0..chunks {
            let at = 2 * (base + chunk);
            if at + 2 > partials.len() {
                return;
            }
            acc = acc + Kx::from_words(partials[at], partials[at + 1]);
        }
        let at = 2 * idx;
        if at + 2 > out.len() {
            return;
        }
        let words = acc.as_words();
        unsafe {
            *out.get_unchecked_mut(at) = words[0];
            *out.get_unchecked_mut(at + 1) = words[1];
        }
    }

    /// One thread per `(child, row, x_col)` in the public `X` matrix.
    #[kernel]
    pub fn dec_pack_public_x(planes: &[u64], m_in: u32, plane_stride: u32, k: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d().get();
        let m_in = m_in as usize;
        let plane_stride = plane_stride as usize;
        let total = (k as usize) * RING_LANES * m_in;
        if idx >= total || idx >= out.len() || m_in == 0 {
            return;
        }

        let x_col = idx % m_in;
        let row = (idx / m_in) % RING_LANES;
        let child = idx / (RING_LANES * m_in);
        let required_cols = m_in.div_ceil(RING_LANES);
        if x_col >= required_cols {
            return;
        }

        let src = child * plane_stride + x_col * RING_LANES + row;
        if src >= planes.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(idx) = planes[src];
        }
    }

    /// Build a compact active-child index surface from the split flags.
    ///
    /// `k` is tiny for the current SuperNeo profile, so a single thread is
    /// enough and avoids atomics. The summary writes `[active_count,
    /// split_out_of_range]`, so the host can mirror the small terminal status
    /// without downloading the whole flags surface.
    #[kernel]
    pub fn dec_build_active_index(
        flags: &[u64],
        k: u32,
        mut active_indices: DisjointSlice<u64>,
        mut active_summary: DisjointSlice<u64>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let k = k as usize;
        let mut count = 0usize;
        for child in 0..k {
            if child < flags.len() && flags[child] != 0 {
                if count < active_indices.len() {
                    unsafe {
                        *active_indices.get_unchecked_mut(count) = child as u64;
                    }
                }
                count += 1;
            }
        }
        if !active_summary.is_empty() {
            unsafe {
                *active_summary.get_unchecked_mut(0) = count as u64;
            }
        }
        if active_summary.len() > 1 {
            let out_of_range = if k < flags.len() { flags[k] } else { 1 };
            unsafe {
                *active_summary.get_unchecked_mut(1) = out_of_range;
            }
        }
    }

    /// Preserve split failures across resident folds without downloading the
    /// activity surface. The sticky word is materialized only at proof/audit
    /// egress.
    #[kernel]
    pub fn dec_accumulate_status(flags: &[u64], k: u32, mut sticky_status: DisjointSlice<u64>) {
        if thread::index_1d().get() != 0 || sticky_status.is_empty() {
            return;
        }
        let k = k as usize;
        let current = unsafe { *sticky_status.get_unchecked_mut(0) };
        let split_error = if k < flags.len() { flags[k] } else { 1 };
        unsafe {
            *sticky_status.get_unchecked_mut(0) = current | split_error;
        }
    }

    /// Compact active child planes using the device-built index surface.
    #[kernel]
    pub fn dec_compact_active_planes(
        planes: &[u64],
        active_indices: &[u64],
        plane_stride: u32,
        active_count: u32,
        mut compact: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let plane_stride = plane_stride as usize;
        let active_count = active_count as usize;
        let total = active_count * plane_stride;
        if idx >= total || idx >= compact.len() || plane_stride == 0 {
            return;
        }

        let active_child = idx / plane_stride;
        let offset = idx % plane_stride;
        if active_child >= active_indices.len() {
            return;
        }
        let canonical_child = active_indices[active_child] as usize;
        let src = canonical_child * plane_stride + offset;
        if src >= planes.len() {
            return;
        }
        unsafe {
            *compact.get_unchecked_mut(idx) = planes[src];
        }
    }

    /// Scatter active child output words back to canonical child order.
    #[kernel]
    pub fn dec_scatter_active_words(
        active_words: &[u64],
        activity_flags: &[u64],
        words_per_child: u32,
        active_count: u32,
        k: u32,
        mut canonical_words: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let words_per_child = words_per_child as usize;
        let active_count = active_count as usize;
        let k = k as usize;
        let total = k * words_per_child;
        if idx >= total || words_per_child == 0 || idx >= canonical_words.len() {
            return;
        }

        let canonical_child = idx / words_per_child;
        let offset = idx % words_per_child;
        if canonical_child >= activity_flags.len() || activity_flags[canonical_child] == 0 {
            unsafe {
                *canonical_words.get_unchecked_mut(idx) = 0;
            }
            return;
        }

        let mut active_child = 0usize;
        for child in 0..canonical_child {
            if child < activity_flags.len() && activity_flags[child] != 0 {
                active_child += 1;
            }
        }
        if active_child >= active_count {
            return;
        }
        let src = active_child * words_per_child + offset;
        let value = if src < active_words.len() { active_words[src] } else { 0 };
        unsafe {
            *canonical_words.get_unchecked_mut(idx) = value;
        }
    }

    /// One thread per parent witness element: extract k balanced base-b
    /// digits, write every per-child plane slot for this element.
    #[kernel]
    pub fn dec_split(
        z: &[u64],
        len: u32,
        k: u32,
        b: u32,
        big_b: u64,
        mut planes: DisjointSlice<u64>,
        mut flags: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let len = len as usize;
        let k = k as usize;
        if idx >= len {
            return;
        }

        for child in 0..k {
            let slot = child * len + idx;
            if slot < planes.len() {
                unsafe {
                    *planes.get_unchecked_mut(slot) = 0;
                }
            }
        }

        if idx >= z.len() {
            return;
        }
        let u = z[idx];
        if u == 0 {
            return;
        }

        // Map the canonical word to the smaller-magnitude signed representative
        // within (-B, B), exactly as the CPU split does.
        let neg_mag = GOLDILOCKS_MODULUS - u;
        let pos_ok = u < big_b;
        let neg_ok = neg_mag < big_b;
        let mut v: i64 = if pos_ok && (!neg_ok || u <= neg_mag) {
            u as i64
        } else if neg_ok {
            -(neg_mag as i64)
        } else {
            if k < flags.len() {
                unsafe {
                    *flags.get_unchecked_mut(k) = 1;
                }
            }
            return;
        };

        let b_i64 = b as i64;
        for i in 0..k {
            if v == 0 {
                break;
            }
            let (r, q) = if b == 2 {
                if (v & 1) == 0 {
                    (0, v >> 1)
                } else if v > 0 {
                    (1, (v - 1) >> 1)
                } else {
                    (-1, (v + 1) >> 1)
                }
            } else {
                let r = v % b_i64;
                ((r), (v - r) / b_i64)
            };
            if r != 0 {
                let word = if r >= 0 {
                    r as u64
                } else {
                    GOLDILOCKS_MODULUS - ((-r) as u64)
                };
                let slot = i * len + idx;
                if slot >= planes.len() || i >= flags.len() {
                    return;
                }
                unsafe {
                    *planes.get_unchecked_mut(slot) = word;
                    *flags.get_unchecked_mut(i) = 1;
                }
            }
            v = q;
        }
        if v != 0 && k < flags.len() {
            unsafe {
                *flags.get_unchecked_mut(k) = 1;
            }
        }
    }
}
