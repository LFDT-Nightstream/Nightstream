//! Device-resident static SuperNeo matrix data, uploaded once per structure.
//!
//! Owns two views of the `SuperneoMatrixCache` CSR: the block-reordered bar
//! entries ([`DeviceBarMatrices`], for the per-prove ring-linear-forms build,
//! field-identical to `SuperneoEvalCache::build_ring_linear_forms`) and the
//! row-major orig entries ([`DeviceRowMatrices`], for the f-var row tables,
//! field-identical to `SuperneoMatrixCache::row_dot_with_blocks`).

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::SuperneoEvalCache;
use p3_field::PrimeField64;

use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer, zeroed_u64_device_buffer, Device};
use crate::graph::GraphAllocations;
use crate::kernels::ajtai::RING_D;
use crate::kernels::csr::{
    launch_forms_from_bar_csr, launch_packed_row_tables_from_csr, launch_row_table_from_csr, launch_tensor_point_k,
    launch_weighted_basis_dots, launch_weighted_row_table, CsrKernelModule,
};
use crate::reduce::ccs::CcsDeviceError;

/// One matrix's bar entries on device, grouped by block.
struct BarMatrix {
    block_offsets: DeviceBuffer<u64>,
    entry_rows: DeviceBuffer<u64>,
    entry_bars: DeviceBuffer<u64>,
    rows: usize,
}

/// The static bar matrices of one CCS structure, resident on device.
pub struct DeviceBarMatrices {
    matrices: Vec<BarMatrix>,
    blocks: usize,
    sparse_form_layout: SparseFormLayout,
    /// Identity of the source cache so a backend can reuse the upload
    /// across proves of the same structure.
    fingerprint: CacheFingerprint,
}

/// Static touched-block schedule for multiplying the dense form carrier.
///
/// Form values remain in `[row][block][D]` order so DEC can reuse them.
/// Y-eval uses this schedule to skip blocks the source CSR cannot populate.
pub(crate) struct SparseFormLayout {
    pub(crate) entry_blocks: DeviceBuffer<u64>,
    pub(crate) chunk_rows: DeviceBuffer<u64>,
    pub(crate) chunk_entry_starts: DeviceBuffer<u64>,
    pub(crate) chunk_entry_lens: DeviceBuffer<u64>,
    pub(crate) row_chunk_offsets: DeviceBuffer<u64>,
    pub(crate) rows: usize,
    pub(crate) chunks: usize,
}

impl DeviceBarMatrices {
    pub fn upload(device: &Device, cache: &SuperneoEvalCache) -> Result<Self, CcsDeviceError> {
        assert_eq!(D, RING_D, "kernel lane count out of sync with neo_math::D");
        let mats = cache.matrix_caches();
        if mats.is_empty() {
            return Err(CcsDeviceError::Shape("structure has no matrices"));
        }
        let (_, cols, _, _) = mats[0].bar_shape();
        let blocks = cols.div_ceil(D);

        let mut matrices = Vec::with_capacity(mats.len());
        let mut total_entries = 0usize;
        let mut sparse_entry_blocks = Vec::new();
        let mut sparse_chunk_rows = Vec::new();
        let mut sparse_chunk_entry_starts = Vec::new();
        let mut sparse_chunk_entry_lens = Vec::new();
        let mut sparse_row_chunk_offsets = vec![0u64];
        for mat in mats {
            let (rows, mat_cols, row_offsets, entry_count) = mat.bar_shape();
            if mat_cols != cols {
                return Err(CcsDeviceError::Shape("matrices must share the column count"));
            }
            total_entries += entry_count;

            // Reorder the row-major CSR by block so each output block owns a
            // contiguous entry range.
            let mut per_block = vec![0usize; blocks + 1];
            for i in 0..entry_count {
                let (blk, _) = mat.bar_entry(i);
                per_block[blk + 1] += 1;
            }
            for blk in 0..blocks {
                per_block[blk + 1] += per_block[blk];
            }
            let block_offsets: Vec<u64> = per_block.iter().map(|&o| o as u64).collect();
            let active_blocks = block_offsets
                .windows(2)
                .enumerate()
                .filter_map(|(block, pair)| (pair[0] != pair[1]).then_some(block as u64))
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let row = sparse_row_chunk_offsets.len() - 1;
                let entry_base = sparse_entry_blocks.len();
                sparse_entry_blocks.extend_from_slice(&active_blocks);
                for local_start in (0..active_blocks.len()).step_by(crate::kernels::ajtai::CHUNK_COLS) {
                    sparse_chunk_rows.push(row as u64);
                    sparse_chunk_entry_starts.push((entry_base + local_start) as u64);
                    sparse_chunk_entry_lens.push(
                        active_blocks
                            .len()
                            .saturating_sub(local_start)
                            .min(crate::kernels::ajtai::CHUNK_COLS) as u64,
                    );
                }
                sparse_row_chunk_offsets.push(sparse_chunk_rows.len() as u64);
            }

            let mut cursor = per_block;
            let mut entry_rows = vec![0u64; entry_count];
            // Entry-major (`bars[e * D + lane]`): warp threads are adjacent
            // lanes of one block, so their loads coalesce at each entry.
            let mut entry_bars = vec![0u64; entry_count * D];
            for row in 0..rows {
                for i in row_offsets[row]..row_offsets[row + 1] {
                    let (blk, bar) = mat.bar_entry(i);
                    let slot = cursor[blk];
                    cursor[blk] += 1;
                    entry_rows[slot] = row as u64;
                    for (lane, coeff) in bar.0.iter().enumerate() {
                        entry_bars[slot * D + lane] = coeff.as_canonical_u64();
                    }
                }
            }

            let stream = device.stream();
            matrices.push(BarMatrix {
                block_offsets: upload_u64_device_buffer(stream, &block_offsets)?,
                entry_rows: upload_u64_device_buffer(stream, &entry_rows)?,
                entry_bars: upload_u64_device_buffer(stream, &entry_bars)?,
                rows,
            });
        }
        let stream = device.stream();
        let sparse_form_layout = SparseFormLayout {
            entry_blocks: upload_u64_device_buffer(stream, &sparse_entry_blocks)?,
            chunk_rows: upload_u64_device_buffer(stream, &sparse_chunk_rows)?,
            chunk_entry_starts: upload_u64_device_buffer(stream, &sparse_chunk_entry_starts)?,
            chunk_entry_lens: upload_u64_device_buffer(stream, &sparse_chunk_entry_lens)?,
            row_chunk_offsets: upload_u64_device_buffer(stream, &sparse_row_chunk_offsets)?,
            rows: 2 * matrices.len(),
            chunks: sparse_chunk_rows.len(),
        };
        device.sync()?;
        Ok(Self {
            matrices,
            blocks,
            sparse_form_layout,
            fingerprint: fingerprint_of(cache, blocks, total_entries),
        })
    }

    /// True when this upload came from a cache with the same identity and
    /// shape — safe to reuse across proves of one structure.
    pub fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        let mats = cache.matrix_caches();
        let total_entries = mats.iter().map(|m| m.bar_shape().3).sum();
        self.fingerprint == fingerprint_of(cache, self.blocks, total_entries)
    }

    pub(crate) fn sparse_form_layout(&self) -> &SparseFormLayout {
        &self.sparse_form_layout
    }

    /// Build `[2t][blocks][D]` form rows on device at the given χ table,
    /// bit-identical to `build_ring_linear_forms` densified.
    pub fn build_forms(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        chi_r: &[K],
        n_eff: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        let mut chi_words = vec![0u64; chi_r.len() * 2];
        for (i, value) in chi_r.iter().enumerate() {
            let (re, im) = value.to_limbs_u64();
            chi_words[2 * i] = re;
            chi_words[2 * i + 1] = im;
        }
        let stream = device.stream();
        let chi_dev = upload_u64_device_buffer(stream, &chi_words)?;
        self.build_forms_from_device_chi(device, csr, &chi_dev, chi_r.len(), n_eff)
    }

    /// Build forms after generating the `χ_r` table on device from the small
    /// challenge vector. This is for hot prover paths where uploading the full
    /// table is more expensive than the table itself.
    pub fn build_forms_from_challenges(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        challenges: &[K],
        n_eff: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        let mut challenge_words = vec![0u64; challenges.len() * 2];
        for (i, value) in challenges.iter().enumerate() {
            let (re, im) = value.to_limbs_u64();
            challenge_words[2 * i] = re;
            challenge_words[2 * i + 1] = im;
        }
        let stream = device.stream();
        let challenge_dev = upload_u64_device_buffer(stream, &challenge_words)?;
        self.build_forms_from_device_challenges(device, csr, &challenge_dev, challenges.len(), n_eff)
    }

    /// Build forms from a device-resident challenge vector. This is the
    /// whole-phase Π_CCS path: row challenges sampled by the device transcript
    /// can feed the Ajtai forms without a host round-trip.
    pub fn build_forms_from_device_challenges(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        challenges: &DeviceBuffer<u64>,
        challenge_count: usize,
        n_eff: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        let chi_len = chi_len(challenge_count)?;
        let stream = device.stream();
        let mut chi_dev = zeroed_u64_device_buffer(stream, 2 * chi_len)?;
        let mut forms = uninit_u64_buffer(stream, self.form_words())?;
        self.build_forms_from_device_challenges_into(
            device,
            csr,
            challenges,
            challenge_count,
            n_eff,
            &mut chi_dev,
            &mut forms,
        )?;
        Ok(forms)
    }

    /// Caller-owned-buffer form for whole-phase CUDA graph capture. The
    /// caller retains `chi_dev` and `forms` so the graph body has stable
    /// device addresses across folds.
    pub fn build_forms_from_device_challenges_into(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        challenges: &DeviceBuffer<u64>,
        challenge_count: usize,
        n_eff: usize,
        chi_dev: &mut DeviceBuffer<u64>,
        forms: &mut DeviceBuffer<u64>,
    ) -> Result<(), CcsDeviceError> {
        let chi_len = chi_len(challenge_count)?;
        if challenges.len() < 2 * challenge_count {
            return Err(CcsDeviceError::Shape("device challenge buffer too small"));
        }
        if chi_dev.len() < 2 * chi_len {
            return Err(CcsDeviceError::Shape("device chi buffer too small"));
        }
        let stream = device.stream();
        launch_tensor_point_k(csr, stream, challenges, challenge_count, chi_dev)?;
        self.build_forms_from_device_chi_into(device, csr, chi_dev, chi_len, n_eff, forms)
    }

    /// Same output as [`Self::build_forms_from_device_challenges_into`], but
    /// each matrix form row pair is enqueued on an independent forked stream.
    ///
    /// Use only outside CUDA graph capture. The per-matrix outputs are
    /// disjoint slices of `forms`; the final join makes later work on the
    /// prover stream observe every matrix write.
    pub(crate) fn build_forms_from_device_challenges_into_concurrent(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        challenges: &DeviceBuffer<u64>,
        challenge_count: usize,
        n_eff: usize,
        chi_dev: &mut DeviceBuffer<u64>,
        forms: &mut DeviceBuffer<u64>,
    ) -> Result<(), CcsDeviceError> {
        let chi_len = chi_len(challenge_count)?;
        if challenges.len() < 2 * challenge_count {
            return Err(CcsDeviceError::Shape("device challenge buffer too small"));
        }
        if chi_dev.len() < 2 * chi_len {
            return Err(CcsDeviceError::Shape("device chi buffer too small"));
        }
        let stream = device.stream();
        launch_tensor_point_k(csr, stream, challenges, challenge_count, chi_dev)?;
        self.build_forms_from_device_chi_into_concurrent(device, csr, chi_dev, chi_len, n_eff, forms)
    }

    fn build_forms_from_device_chi(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        chi_dev: &DeviceBuffer<u64>,
        chi_len: usize,
        n_eff: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        let stream = device.stream();
        let mut forms = uninit_u64_buffer(stream, self.form_words())?;
        self.build_forms_from_device_chi_into(device, csr, chi_dev, chi_len, n_eff, &mut forms)?;
        Ok(forms)
    }

    fn build_forms_from_device_chi_into(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        chi_dev: &DeviceBuffer<u64>,
        chi_len: usize,
        n_eff: usize,
        forms: &mut DeviceBuffer<u64>,
    ) -> Result<(), CcsDeviceError> {
        let stream = device.stream();
        let row_len = self.blocks * D;
        if forms.len() < self.form_words() {
            return Err(CcsDeviceError::Shape("device forms buffer too small"));
        }
        for (j, matrix) in self.matrices.iter().enumerate() {
            let row_cap = matrix.rows.min(n_eff).min(chi_len);
            launch_forms_from_bar_csr(
                csr,
                stream,
                chi_dev,
                &matrix.block_offsets,
                &matrix.entry_rows,
                &matrix.entry_bars,
                self.blocks,
                row_cap,
                (2 * j) * row_len,
                forms,
            )?;
        }
        Ok(())
    }

    fn build_forms_from_device_chi_into_concurrent(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        chi_dev: &DeviceBuffer<u64>,
        chi_len: usize,
        n_eff: usize,
        forms: &mut DeviceBuffer<u64>,
    ) -> Result<(), CcsDeviceError> {
        let stream = device.stream();
        let row_len = self.blocks * D;
        if forms.len() < self.form_words() {
            return Err(CcsDeviceError::Shape("device forms buffer too small"));
        }
        if self.matrices.len() <= 1 {
            return self.build_forms_from_device_chi_into(device, csr, chi_dev, chi_len, n_eff, forms);
        }

        let mut forked_streams = Vec::with_capacity(self.matrices.len() - 1);
        for _ in 1..self.matrices.len() {
            forked_streams.push(stream.fork()?);
        }

        for (j, matrix) in self.matrices.iter().enumerate() {
            let launch_stream = if j == 0 { stream } else { &forked_streams[j - 1] };
            let row_cap = matrix.rows.min(n_eff).min(chi_len);
            launch_forms_from_bar_csr(
                csr,
                launch_stream,
                chi_dev,
                &matrix.block_offsets,
                &matrix.entry_rows,
                &matrix.entry_bars,
                self.blocks,
                row_cap,
                (2 * j) * row_len,
                forms,
            )?;
        }

        for forked in &forked_streams {
            stream.join(forked)?;
        }
        Ok(())
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn form_words(&self) -> usize {
        2 * self.matrices.len() * self.blocks * D
    }

    pub(crate) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        for matrix in &self.matrices {
            allocations.push(&matrix.block_offsets);
            allocations.push(&matrix.entry_rows);
            allocations.push(&matrix.entry_bars);
        }
    }
}

fn chi_len(challenge_count: usize) -> Result<usize, CcsDeviceError> {
    1usize
        .checked_shl(challenge_count as u32)
        .ok_or(CcsDeviceError::Shape(
            "challenge count exceeds addressable tensor table",
        ))
}

fn uninit_u64_buffer(stream: &Arc<CudaStream>, len: usize) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    Ok(uninit_u64_device_buffer(stream, len)?)
}

/// Identity of a source cache, shared by both static uploads so a session
/// can reuse them across proves of one structure. The Poseidon2 CCS matrix
/// digest is the authoritative identity; the pointer and shape counts are
/// cheap first discriminants (a pointer reused after reallocation cannot
/// alias a different structure past the digest).
type CacheFingerprint = (usize, [F; 4], usize, usize, usize);

fn fingerprint_of(cache: &SuperneoEvalCache, blocks: usize, total_entries: usize) -> CacheFingerprint {
    (
        cache as *const SuperneoEvalCache as usize,
        cache.mat_digest(),
        cache.matrix_caches().len(),
        blocks,
        total_entries,
    )
}

/// One matrix's orig entries on device, in row-major CSR order.
struct RowMatrix {
    row_offsets: DeviceBuffer<u64>,
    entry_blks: DeviceBuffer<u64>,
    entry_origs: DeviceBuffer<u64>,
    rows: usize,
}

/// The row-major orig CSR of one CCS structure, resident on device, for
/// per-witness f-var row tables (`m_j(row) = (M_j z)[row]`).
pub struct DeviceRowMatrices {
    matrices: Vec<RowMatrix>,
    row_offset_bases: DeviceBuffer<u64>,
    entry_bases: DeviceBuffer<u64>,
    matrix_rows: DeviceBuffer<u64>,
    flat_row_offsets: DeviceBuffer<u64>,
    flat_entry_blks: DeviceBuffer<u64>,
    flat_entry_origs: DeviceBuffer<u64>,
    blocks: usize,
    fingerprint: CacheFingerprint,
}

impl DeviceRowMatrices {
    pub fn upload(device: &Device, cache: &SuperneoEvalCache) -> Result<Self, CcsDeviceError> {
        assert_eq!(D, RING_D, "kernel lane count out of sync with neo_math::D");
        let mats = cache.matrix_caches();
        if mats.is_empty() {
            return Err(CcsDeviceError::Shape("structure has no matrices"));
        }
        let (_, cols, _, _) = mats[0].bar_shape();
        let blocks = cols.div_ceil(D);

        let mut matrices = Vec::with_capacity(mats.len());
        let mut row_offset_bases = Vec::with_capacity(mats.len());
        let mut entry_bases = Vec::with_capacity(mats.len());
        let mut matrix_rows = Vec::with_capacity(mats.len());
        let mut flat_row_offsets = Vec::new();
        let mut flat_entry_blks = Vec::new();
        let mut flat_entry_origs = Vec::new();
        let mut total_entries = 0usize;
        for mat in mats {
            let (rows, mat_cols, row_offsets, entry_count) = mat.bar_shape();
            if mat_cols != cols {
                return Err(CcsDeviceError::Shape("matrices must share the column count"));
            }
            total_entries += entry_count;

            let offsets: Vec<u64> = row_offsets.iter().map(|&o| o as u64).collect();
            let mut entry_blks = vec![0u64; entry_count];
            let mut entry_origs = vec![0u64; entry_count * D];
            for i in 0..entry_count {
                let (blk, orig) = mat.orig_entry(i);
                entry_blks[i] = blk as u64;
                for (lane, coeff) in orig.0.iter().enumerate() {
                    entry_origs[i * D + lane] = coeff.as_canonical_u64();
                }
            }

            row_offset_bases.push(flat_row_offsets.len() as u64);
            entry_bases.push(flat_entry_blks.len() as u64);
            matrix_rows.push(rows as u64);
            flat_row_offsets.extend(offsets.iter().copied());
            flat_entry_blks.extend(entry_blks.iter().copied());
            flat_entry_origs.extend(entry_origs.iter().copied());

            let stream = device.stream();
            matrices.push(RowMatrix {
                row_offsets: upload_u64_device_buffer(stream, &offsets)?,
                entry_blks: upload_u64_device_buffer(stream, &entry_blks)?,
                entry_origs: upload_u64_device_buffer(stream, &entry_origs)?,
                rows,
            });
        }
        let stream = device.stream();
        let row_offset_bases = upload_u64_device_buffer(stream, &row_offset_bases)?;
        let entry_bases = upload_u64_device_buffer(stream, &entry_bases)?;
        let matrix_rows = upload_u64_device_buffer(stream, &matrix_rows)?;
        let flat_row_offsets = upload_u64_device_buffer(stream, &flat_row_offsets)?;
        let flat_entry_blks = upload_u64_device_buffer(stream, &flat_entry_blks)?;
        let flat_entry_origs = upload_u64_device_buffer(stream, &flat_entry_origs)?;
        device.sync()?;
        Ok(Self {
            matrices,
            row_offset_bases,
            entry_bases,
            matrix_rows,
            flat_row_offsets,
            flat_entry_blks,
            flat_entry_origs,
            blocks,
            fingerprint: fingerprint_of(cache, blocks, total_entries),
        })
    }

    /// True when this upload came from a cache with the same identity and
    /// shape — safe to reuse across proves of one structure.
    pub fn matches(&self, cache: &SuperneoEvalCache) -> bool {
        let mats = cache.matrix_caches();
        let total_entries = mats.iter().map(|m| m.bar_shape().3).sum();
        self.fingerprint == fingerprint_of(cache, self.blocks, total_entries)
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn matrix_count(&self) -> usize {
        self.matrices.len()
    }

    /// Build several real-witness row tables as one packed device buffer.
    ///
    /// Output layout is `[matrix_indices position][row].{re,im}`. This is the
    /// Pi_CCS oracle path: it exposes the independent table axis to CUDA in a
    /// single launch instead of serializing one launch per FE variable table.
    #[allow(clippy::too_many_arguments)]
    pub fn build_packed_row_tables_device(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        matrix_indices: &[usize],
        z_planes: &DeviceBuffer<u64>,
        z_offset: usize,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<(DeviceBuffer<u64>, DeviceBuffer<u64>), CcsDeviceError> {
        if matrix_indices.iter().any(|&j| j >= self.matrices.len()) {
            return Err(CcsDeviceError::Shape("f-var index out of range"));
        }
        let stream = device.stream();
        let index_words: Vec<u64> = matrix_indices.iter().map(|&j| j as u64).collect();
        let indices = upload_u64_device_buffer(stream, &index_words)?;
        let mut out = uninit_u64_buffer(stream, matrix_indices.len() * n_pad * 2)?;
        launch_packed_row_tables_from_csr(
            csr,
            stream,
            &indices,
            &self.row_offset_bases,
            &self.entry_bases,
            &self.matrix_rows,
            &self.flat_row_offsets,
            &self.flat_entry_blks,
            &self.flat_entry_origs,
            z_planes,
            z_offset,
            matrix_indices.len(),
            n_eff,
            n_pad,
            &mut out,
        )?;
        Ok((out, indices))
    }

    /// The carried-ME eval table: `out[row] = Σ_j mat_coeffs[j] · y_alpha_j`
    /// with the per-entry weighted projection collapsed to QK slots first.
    /// Returns `2 * n_pad` words, bit-identical to `eval_weighted_row_table`
    /// on complex carried planes.
    #[allow(clippy::too_many_arguments)]
    pub fn build_weighted_table(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        basis_re_words: &[u64],
        basis_im_words: &[u64],
        z_re_words: &[u64],
        z_im_words: &[u64],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Vec<u64>, CcsDeviceError> {
        if mat_coeffs.len() != self.matrices.len() {
            return Err(CcsDeviceError::Shape("matrix coefficient count mismatch"));
        }
        let plane_len = self.blocks * D;
        if z_re_words.len() != plane_len || z_im_words.len() != plane_len {
            return Err(CcsDeviceError::Shape("carried plane length mismatch"));
        }
        let stream = device.stream();
        let z_re = upload_u64_device_buffer(stream, z_re_words)?;
        let z_im = upload_u64_device_buffer(stream, z_im_words)?;
        return self.build_weighted_table_from_device(
            device,
            csr,
            basis_re_words,
            basis_im_words,
            &z_re,
            &z_im,
            mat_coeffs,
            n_eff,
            n_pad,
        );
    }

    /// [`Self::build_weighted_table`] against carried planes already on
    /// device (e.g. the backend-computed carried combination).
    #[allow(clippy::too_many_arguments)]
    pub fn build_weighted_table_from_device(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        basis_re_words: &[u64],
        basis_im_words: &[u64],
        z_re: &DeviceBuffer<u64>,
        z_im: &DeviceBuffer<u64>,
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Vec<u64>, CcsDeviceError> {
        let out = self.build_weighted_table_device(
            device,
            csr,
            basis_re_words,
            basis_im_words,
            z_re,
            z_im,
            mat_coeffs,
            n_eff,
            n_pad,
        )?;
        let words = out.to_host_vec(device.stream())?;
        device.sync()?;
        Ok(words)
    }

    /// Device-resident variant of [`Self::build_weighted_table_from_device`].
    /// The returned table has `2 * n_pad` words and may be consumed directly
    /// by the FE row oracle without a host round-trip.
    #[allow(clippy::too_many_arguments)]
    pub fn build_weighted_table_device(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        basis_re_words: &[u64],
        basis_im_words: &[u64],
        z_re: &DeviceBuffer<u64>,
        z_im: &DeviceBuffer<u64>,
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        if mat_coeffs.len() != self.matrices.len() {
            return Err(CcsDeviceError::Shape("matrix coefficient count mismatch"));
        }
        let plane_len = self.blocks * D;
        if z_re.len() != plane_len || z_im.len() != plane_len {
            return Err(CcsDeviceError::Shape("carried plane length mismatch"));
        }
        let stream = device.stream();
        let basis_re = upload_u64_device_buffer(stream, basis_re_words)?;
        let basis_im = upload_u64_device_buffer(stream, basis_im_words)?;
        let mut qk = zeroed_u64_device_buffer(stream, plane_len * 2)?;
        launch_weighted_basis_dots(csr, stream, &basis_re, &basis_im, z_re, z_im, self.blocks, &mut qk)?;

        let mut out = zeroed_u64_device_buffer(stream, 2 * n_pad)?;
        for (matrix, coeff) in self.matrices.iter().zip(mat_coeffs) {
            let (c0, c1) = coeff.to_limbs_u64();
            let row_cap = matrix.rows.min(n_eff).min(n_pad);
            launch_weighted_row_table(
                csr,
                stream,
                &matrix.row_offsets,
                &matrix.entry_blks,
                &matrix.entry_origs,
                &qk,
                row_cap,
                c0,
                c1,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Row table for matrix `j` against a real witness plane already on
    /// device: `2 * n_pad` words, `(re, 0)` per row, zero past
    /// `min(rows, n_eff)` — bit-identical to `row_dot_with_blocks`.
    pub fn build_row_table(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        j: usize,
        z_plane: &DeviceBuffer<u64>,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Vec<u64>, CcsDeviceError> {
        let out = self.build_row_table_device(device, csr, j, z_plane, 0, n_eff, n_pad)?;
        let words = out.to_host_vec(device.stream())?;
        device.sync()?;
        Ok(words)
    }

    /// Device-resident variant of [`Self::build_row_table`]. The source
    /// plane may be a slice inside a larger resident plane buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn build_row_table_device(
        &self,
        device: &Device,
        csr: &CsrKernelModule,
        j: usize,
        z_planes: &DeviceBuffer<u64>,
        z_offset: usize,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        self.build_row_table_device_on_stream(device.stream(), csr, j, z_planes, z_offset, n_eff, n_pad)
    }

    /// [`Self::build_row_table_device`] on an explicitly chosen stream.
    ///
    /// Pi_CCS oracle planning uses this to enqueue independent `F` row-table
    /// branches concurrently, then joins before FE consumes the tables.
    #[allow(clippy::too_many_arguments)]
    pub fn build_row_table_device_on_stream(
        &self,
        stream: &Arc<CudaStream>,
        csr: &CsrKernelModule,
        j: usize,
        z_planes: &DeviceBuffer<u64>,
        z_offset: usize,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        let matrix = self
            .matrices
            .get(j)
            .ok_or(CcsDeviceError::Shape("f-var index out of range"))?;
        let mut out = zeroed_u64_device_buffer(stream, 2 * n_pad)?;
        let row_cap = matrix.rows.min(n_eff).min(n_pad);
        launch_row_table_from_csr(
            csr,
            stream,
            &matrix.row_offsets,
            &matrix.entry_blks,
            &matrix.entry_origs,
            z_planes,
            z_offset,
            row_cap,
            &mut out,
        )?;
        Ok(out)
    }
}
