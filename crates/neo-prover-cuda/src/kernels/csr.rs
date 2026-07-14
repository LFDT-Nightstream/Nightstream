//! CSR-table kernels over the static SuperNeo matrix data: the per-fold
//! ring-linear-forms build (bar CSR), the f-var row tables and the
//! carried-ME weighted eval table (orig CSR).
//!
//! Contract: bit-identical to `SuperneoEvalCache::build_ring_linear_forms`,
//! `SuperneoMatrixCache::row_dot_with_blocks`, and
//! `eval_weighted_row_table` respectively — the host owns the CSR uploads
//! (`crate::ring_forms`) and hands buffers here.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};
use cuda_host::EmbeddedModuleError;

use crate::kernels::ajtai::RING_D;
use crate::kernels::goldilocks::{mul_low_norm, Gl, Kx};

pub use csr_kernels::LoadedModule as CsrKernelModule;

pub fn load_csr_kernels(ctx: &Arc<CudaContext>) -> Result<CsrKernelModule, EmbeddedModuleError> {
    csr_kernels::load(ctx)
}

/// Build one matrix's ring-linear-form rows on device:
/// `forms[2j+half][blk][lane] = Σ_{e ∈ blk} chi[row_e].half · bar_e[lane]`,
/// mirroring `build_ring_linear_form_split_chi`'s aggregation. Entries with
/// `row ≥ row_cap` are dead, exactly as the CPU's row bound.
#[allow(clippy::too_many_arguments)]
pub fn launch_forms_from_bar_csr(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    chi: &DeviceBuffer<u64>,
    block_offsets: &DeviceBuffer<u64>,
    entry_rows: &DeviceBuffer<u64>,
    entry_bars: &DeviceBuffer<u64>,
    blocks: usize,
    row_cap: usize,
    out_base: usize,
    forms: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let launch = LaunchConfig {
        grid_dim: ((2 * blocks) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    module.forms_from_bar_csr(
        stream,
        launch,
        chi,
        block_offsets,
        entry_rows,
        entry_bars,
        blocks as u32,
        row_cap as u32,
        out_base as u32,
        forms,
    )
}

/// Build a K-valued tensor/equality table on device:
/// `out[idx] = Π_i if bit_i(idx) { r_i } else { 1 - r_i }`.
/// The output layout is K words, `[idx].re, [idx].im`, matching the
/// host `tensor_point_parallel::<K>` table consumed by the CSR form builder.
pub fn launch_tensor_point_k(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    challenges: &DeviceBuffer<u64>,
    challenge_count: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    launch_tensor_point_k_at(module, stream, challenges, challenge_count, 0, out)
}

/// Build a K-valued tensor/equality table into `out[out_offset_words..]`.
pub fn launch_tensor_point_k_at(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    challenges: &DeviceBuffer<u64>,
    challenge_count: usize,
    out_offset_words: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let points = 1usize
        .checked_shl(challenge_count as u32)
        .expect("tensor point challenge count overflow");
    module.tensor_point_k(
        stream,
        LaunchConfig::for_num_elems(points as u32),
        challenges,
        challenge_count as u32,
        out_offset_words as u32,
        out,
    )
}

/// Row tables from the orig CSR: `out[row] = Σ_{e ∈ row} <orig_e, z[blk_e]>`
/// over base-field words, one K (re, 0) pair per row. `out` must be zeroed
/// and `2 * n_pad` long; rows ≥ `row_cap` stay zero.
pub fn launch_row_table_from_csr(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    row_offsets: &DeviceBuffer<u64>,
    entry_blks: &DeviceBuffer<u64>,
    entry_origs: &DeviceBuffer<u64>,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    row_cap: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.row_table_from_csr(
        stream,
        LaunchConfig::for_num_elems(row_cap as u32),
        row_offsets,
        entry_blks,
        entry_origs,
        z,
        z_offset as u32,
        row_cap as u32,
        out,
    )
}

/// Build many row tables in one launch over `(table, row)`.
///
/// The static CSR is flattened across matrices. `matrix_indices` selects the
/// matrices needed by the FE oracle, and `out` is packed as
/// `[table][row].{re,im}` with `2 * table_count * n_pad` words. Rows outside a
/// selected matrix or past `n_eff` are written as zero, so the destination does
/// not need a pre-zeroing memset.
#[allow(clippy::too_many_arguments)]
pub fn launch_packed_row_tables_from_csr(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    matrix_indices: &DeviceBuffer<u64>,
    row_offset_bases: &DeviceBuffer<u64>,
    entry_bases: &DeviceBuffer<u64>,
    matrix_rows: &DeviceBuffer<u64>,
    row_offsets: &DeviceBuffer<u64>,
    entry_blks: &DeviceBuffer<u64>,
    entry_origs: &DeviceBuffer<u64>,
    z: &DeviceBuffer<u64>,
    z_offset: usize,
    table_count: usize,
    n_eff: usize,
    n_pad: usize,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    let threads = table_count
        .checked_mul(n_pad)
        .expect("packed row table launch size overflow");
    if threads == 0 {
        return Ok(());
    }
    module.packed_row_tables_from_csr(
        stream,
        LaunchConfig::for_num_elems(threads as u32),
        matrix_indices,
        row_offset_bases,
        entry_bases,
        matrix_rows,
        row_offsets,
        entry_blks,
        entry_origs,
        z,
        z_offset as u32,
        table_count as u32,
        n_eff as u32,
        n_pad as u32,
        out,
    )
}

/// Per-(block, local) weighted basis dots: fold the four basis-form × plane
/// dot products into one K value per slot, `QK = (rr + 7·ii, ir + ri)` —
/// the u² = 7 extension collapse of `K(rr, ir) + u·K(ri, ii)`.
#[allow(clippy::too_many_arguments)]
pub fn launch_weighted_basis_dots(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    basis_re: &DeviceBuffer<u64>,
    basis_im: &DeviceBuffer<u64>,
    z_re: &DeviceBuffer<u64>,
    z_im: &DeviceBuffer<u64>,
    blocks: usize,
    qk: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.weighted_basis_dots(
        stream,
        LaunchConfig::for_num_elems((blocks * RING_D) as u32),
        basis_re,
        basis_im,
        z_re,
        z_im,
        blocks as u32,
        qk,
    )
}

/// Accumulate one matrix's weighted row contributions into the eval table:
/// `out[row] += coeff · Σ_{e ∈ row} Σ_l orig_e[l] · QK[blk_e][l]`.
/// Launches for successive matrices are stream-ordered, so the K adds hit
/// `out` in the CPU's matrix order.
#[allow(clippy::too_many_arguments)]
pub fn launch_weighted_row_table(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    row_offsets: &DeviceBuffer<u64>,
    entry_blks: &DeviceBuffer<u64>,
    entry_origs: &DeviceBuffer<u64>,
    qk: &DeviceBuffer<u64>,
    row_cap: usize,
    coeff_c0: u64,
    coeff_c1: u64,
    out: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.weighted_row_table(
        stream,
        LaunchConfig::for_num_elems(row_cap as u32),
        row_offsets,
        entry_blks,
        entry_origs,
        qk,
        row_cap as u32,
        coeff_c0,
        coeff_c1,
        out,
    )
}

/// Carried-witness linear combination on device:
/// `out_re/out_im[w] = Σ_i re/im(coeffs[i]) · planes[plane_offset + i·plane_stride + w]`.
/// The plane values are low-norm witness columns, so the small-magnitude
/// product path applies; field sums are exact in any order, making the
/// result value-identical to the host `linear_combination_real`.
#[allow(clippy::too_many_arguments)]
pub fn launch_plane_lin_comb(
    module: &CsrKernelModule,
    stream: &Arc<CudaStream>,
    planes: &DeviceBuffer<u64>,
    coeffs: &DeviceBuffer<u64>,
    k: usize,
    plane_offset: usize,
    plane_stride: usize,
    len: usize,
    out_re: &mut DeviceBuffer<u64>,
    out_im: &mut DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.plane_lin_comb(
        stream,
        LaunchConfig::for_num_elems(len as u32),
        planes,
        coeffs,
        k as u32,
        plane_offset as u32,
        plane_stride as u32,
        len as u32,
        out_re,
        out_im,
    )
}

#[cuda_module]
pub mod csr_kernels {
    use super::*;

    /// One thread per output word: the carried complex combination of k
    /// real low-norm planes.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn plane_lin_comb(
        planes: &[u64],
        coeffs: &[u64],
        k: u32,
        plane_offset: u32,
        plane_stride: u32,
        len: u32,
        mut out_re: DisjointSlice<u64>,
        mut out_im: DisjointSlice<u64>,
    ) {
        let w = thread::index_1d().get();
        let len = len as usize;
        if w >= len {
            return;
        }
        let mut re = Gl::ZERO;
        let mut im = Gl::ZERO;
        for i in 0..k as usize {
            let at = plane_offset as usize + i * plane_stride as usize + w;
            if at >= planes.len() || 2 * i + 1 >= coeffs.len() {
                return;
            }
            let z = planes[at];
            re = re + mul_low_norm(Gl::from_u64(coeffs[2 * i]), z);
            im = im + mul_low_norm(Gl::from_u64(coeffs[2 * i + 1]), z);
        }
        if w >= out_re.len() || w >= out_im.len() {
            return;
        }
        unsafe {
            *out_re.get_unchecked_mut(w) = re.as_canonical_u64();
            *out_im.get_unchecked_mut(w) = im.as_canonical_u64();
        }
    }

    /// One thread per table point; the challenge count is small enough for a
    /// direct product loop, and this avoids a full host-built chi table upload.
    #[kernel]
    pub fn tensor_point_k(
        challenges: &[u64],
        challenge_count: u32,
        out_offset_words: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let challenge_count = challenge_count as usize;
        let out_base = out_offset_words as usize + 2 * idx;
        if idx >= (1usize << challenge_count) || out_base + 1 >= out.len() {
            return;
        }

        let mut value = Kx::ONE;
        for bit in 0..challenge_count {
            let at = 2 * bit;
            if at + 1 >= challenges.len() {
                return;
            }
            let r = Kx::from_words(challenges[at], challenges[at + 1]);
            value = value * if ((idx >> bit) & 1) == 0 { Kx::ONE - r } else { r };
        }

        let words = value.as_words();
        unsafe {
            *out.get_unchecked_mut(out_base) = words[0];
            *out.get_unchecked_mut(out_base + 1) = words[1];
        }
    }

    /// One thread per (re/im half, block, lane): aggregate this block's
    /// bar entries weighted by the χ coefficient of their row.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn forms_from_bar_csr(
        chi: &[u64],
        block_offsets: &[u64],
        entry_rows: &[u64],
        entry_bars: &[u64],
        blocks: u32,
        row_cap: u32,
        out_base: u32,
        mut forms: DisjointSlice<u64>,
    ) {
        const PARTITIONS: usize = 2;
        const PARTITION_THREADS: usize = 64;
        static mut PARTIALS: SharedArray<u64, { PARTITIONS * RING_D }> = SharedArray::UNINIT;

        let blocks = blocks as usize;
        let row_cap = row_cap as usize;
        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let partition = tid / PARTITION_THREADS;
        let lane = tid % PARTITION_THREADS;
        if block >= 2 * blocks {
            return;
        }
        let half = block / blocks;
        let blk = block % blocks;
        if blk + 1 >= block_offsets.len() || partition >= PARTITIONS {
            return;
        }

        let mut acc = Gl::ZERO;
        let start = block_offsets[blk] as usize;
        let end = block_offsets[blk + 1] as usize;
        if lane < RING_D {
            let mut e = start + partition;
            while e < end {
                if e >= entry_rows.len() || (e * RING_D + lane) >= entry_bars.len() {
                    break;
                }
                let row = entry_rows[e] as usize;
                if row < row_cap && 2 * row + half < chi.len() {
                    let weight = Gl::from_u64(chi[2 * row + half]);
                    acc = acc + weight * Gl::from_u64(entry_bars[e * RING_D + lane]);
                }
                e += PARTITIONS;
            }
            unsafe {
                PARTIALS[partition * RING_D + lane] = acc.as_canonical_u64();
            }
        }
        thread::sync_threads();

        if partition == 0 && lane < RING_D {
            let acc = unsafe { Gl::from_u64(PARTIALS[lane]) + Gl::from_u64(PARTIALS[RING_D + lane]) };
            let out_at = out_base as usize + half * blocks * RING_D + blk * RING_D + lane;
            if out_at < forms.len() {
                unsafe {
                    *forms.get_unchecked_mut(out_at) = acc.as_canonical_u64();
                }
            }
        }
    }

    /// One thread per (block, local): four basis-form dots against the
    /// carried planes, collapsed to one K per slot via u² = 7.
    #[kernel]
    pub fn weighted_basis_dots(
        basis_re: &[u64],
        basis_im: &[u64],
        z_re: &[u64],
        z_im: &[u64],
        blocks: u32,
        mut qk: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= (blocks as usize) * RING_D {
            return;
        }
        let blk = idx / RING_D;
        let local = idx % RING_D;
        let form_base = local * RING_D;
        let z_base = blk * RING_D;
        if form_base + RING_D > basis_re.len()
            || form_base + RING_D > basis_im.len()
            || z_base + RING_D > z_re.len()
            || z_base + RING_D > z_im.len()
        {
            return;
        }
        let mut rr = Gl::ZERO;
        let mut ir = Gl::ZERO;
        let mut ri = Gl::ZERO;
        let mut ii = Gl::ZERO;
        for lane in 0..RING_D {
            let form_re = Gl::from_u64(basis_re[form_base + lane]);
            let form_im = Gl::from_u64(basis_im[form_base + lane]);
            let zr = Gl::from_u64(z_re[z_base + lane]);
            let zi = Gl::from_u64(z_im[z_base + lane]);
            rr = rr + form_re * zr;
            ir = ir + form_im * zr;
            ri = ri + form_re * zi;
            ii = ii + form_im * zi;
        }
        let seven = Gl::from_u64(7);
        let re = rr + seven * ii;
        let im = ir + ri;
        let at = idx * 2;
        if at + 2 > qk.len() {
            return;
        }
        unsafe {
            *qk.get_unchecked_mut(at) = re.as_canonical_u64();
            *qk.get_unchecked_mut(at + 1) = im.as_canonical_u64();
        }
    }

    /// One thread per row: fold the row's orig entries through the QK slots
    /// and accumulate `coeff · y_alpha` into the eval table.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn weighted_row_table(
        row_offsets: &[u64],
        entry_blks: &[u64],
        entry_origs: &[u64],
        qk: &[u64],
        row_cap: u32,
        coeff_c0: u64,
        coeff_c1: u64,
        mut out: DisjointSlice<u64>,
    ) {
        let row = thread::index_1d().get();
        if row >= row_cap as usize || row + 1 >= row_offsets.len() {
            return;
        }
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        let mut acc_re = Gl::ZERO;
        let mut acc_im = Gl::ZERO;
        for e in start..end {
            if e >= entry_blks.len() || (e + 1) * RING_D > entry_origs.len() {
                return;
            }
            let qk_base = (entry_blks[e] as usize) * RING_D * 2;
            if qk_base + RING_D * 2 > qk.len() {
                return;
            }
            for l in 0..RING_D {
                let orig = Gl::from_u64(entry_origs[e * RING_D + l]);
                acc_re = acc_re + orig * Gl::from_u64(qk[qk_base + 2 * l]);
                acc_im = acc_im + orig * Gl::from_u64(qk[qk_base + 2 * l + 1]);
            }
        }
        let at = 2 * row;
        if at + 2 > out.len() {
            return;
        }
        let coeff = Kx::from_words(coeff_c0, coeff_c1);
        let y_alpha = Kx::from_components(acc_re, acc_im);
        // One thread per row per launch; launches are stream-ordered, so
        // this read-modify-write accumulates in the CPU's matrix order.
        unsafe {
            let prev = Kx::from_words(*out.get_unchecked_mut(at), *out.get_unchecked_mut(at + 1));
            let words = (prev + coeff * y_alpha).as_words();
            *out.get_unchecked_mut(at) = words[0];
            *out.get_unchecked_mut(at + 1) = words[1];
        }
    }

    /// One thread per row: dot the row's orig entries against the real
    /// witness plane — `row_dot_with_blocks` for a real packed witness.
    #[kernel]
    pub fn row_table_from_csr(
        row_offsets: &[u64],
        entry_blks: &[u64],
        entry_origs: &[u64],
        z: &[u64],
        z_offset: u32,
        row_cap: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let row = thread::index_1d().get();
        if row >= row_cap as usize || row + 1 >= row_offsets.len() {
            return;
        }
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        let mut acc = Gl::ZERO;
        for e in start..end {
            if e >= entry_blks.len() || (e + 1) * RING_D > entry_origs.len() {
                return;
            }
            let z_base = z_offset as usize + (entry_blks[e] as usize) * RING_D;
            if z_base + RING_D > z.len() {
                return;
            }
            for lane in 0..RING_D {
                acc = acc + Gl::from_u64(entry_origs[e * RING_D + lane]) * Gl::from_u64(z[z_base + lane]);
            }
        }
        let at = 2 * row;
        if at + 2 > out.len() {
            return;
        }
        unsafe {
            *out.get_unchecked_mut(at) = acc.as_canonical_u64();
            *out.get_unchecked_mut(at + 1) = 0;
        }
    }

    /// One thread per `(selected matrix, row)`: build all requested real
    /// witness row tables into one packed output buffer.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn packed_row_tables_from_csr(
        matrix_indices: &[u64],
        row_offset_bases: &[u64],
        entry_bases: &[u64],
        matrix_rows: &[u64],
        row_offsets: &[u64],
        entry_blks: &[u64],
        entry_origs: &[u64],
        z: &[u64],
        z_offset: u32,
        table_count: u32,
        n_eff: u32,
        n_pad: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d().get();
        let table_count = table_count as usize;
        let n_pad = n_pad as usize;
        if table_count == 0 || n_pad == 0 || idx >= table_count * n_pad {
            return;
        }

        let table = idx / n_pad;
        let row = idx % n_pad;
        let out_at = idx * 2;
        if out_at + 1 >= out.len() {
            return;
        }

        let mut acc = Gl::ZERO;
        let matrix = if table < matrix_indices.len() {
            matrix_indices[table] as usize
        } else {
            unsafe {
                *out.get_unchecked_mut(out_at) = 0;
                *out.get_unchecked_mut(out_at + 1) = 0;
            }
            return;
        };

        let active_rows = if matrix < matrix_rows.len() {
            (matrix_rows[matrix] as usize)
                .min(n_eff as usize)
                .min(n_pad)
        } else {
            0
        };
        if row < active_rows && matrix < row_offset_bases.len() && matrix < entry_bases.len() {
            let row_base = row_offset_bases[matrix] as usize;
            let entry_base = entry_bases[matrix] as usize;
            if row_base + row + 1 < row_offsets.len() {
                let start = entry_base + row_offsets[row_base + row] as usize;
                let end = entry_base + row_offsets[row_base + row + 1] as usize;
                for e in start..end {
                    if e >= entry_blks.len() || (e + 1) * RING_D > entry_origs.len() {
                        unsafe {
                            *out.get_unchecked_mut(out_at) = 0;
                            *out.get_unchecked_mut(out_at + 1) = 0;
                        }
                        return;
                    }
                    let z_base = z_offset as usize + (entry_blks[e] as usize) * RING_D;
                    if z_base + RING_D > z.len() {
                        unsafe {
                            *out.get_unchecked_mut(out_at) = 0;
                            *out.get_unchecked_mut(out_at + 1) = 0;
                        }
                        return;
                    }
                    for lane in 0..RING_D {
                        acc = acc + Gl::from_u64(entry_origs[e * RING_D + lane]) * Gl::from_u64(z[z_base + lane]);
                    }
                }
            }
        }

        unsafe {
            *out.get_unchecked_mut(out_at) = acc.as_canonical_u64();
            *out.get_unchecked_mut(out_at + 1) = 0;
        }
    }
}
