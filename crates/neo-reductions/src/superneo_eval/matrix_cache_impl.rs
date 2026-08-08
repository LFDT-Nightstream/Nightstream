//! Operations over one cached SuperNeo matrix.

use super::*;

impl SuperneoMatrixCache {
    #[inline]
    pub(super) fn compact_row_offsets(&mut self) {
        self.row_offsets.compact(self.rows + 1);
    }

    #[inline]
    pub(super) fn compact_dense_blocks(&mut self) {
        self.dense_orig.finish();
    }

    #[inline]
    pub(super) fn dense_block(&self, index: usize) -> Rq {
        self.dense_orig.expanded(index)
    }

    #[inline]
    pub(super) fn row_blocks_for(&self, row: usize) -> &[CompactRowBlock] {
        if self.identity {
            return &[];
        }
        &self.row_blocks[self.row_offsets.range(row)]
    }

    #[inline]
    pub(super) fn expanded_block(&self, block: CompactRowBlock) -> RowBlock {
        let orig = if let Some((local, coefficient)) = block.single_parts() {
            let mut coefficients = [F::ZERO; D];
            coefficients[local] = coefficient;
            Rq(coefficients)
        } else {
            self.dense_block(block.dense_index().expect("compact dense block"))
        };
        RowBlock {
            blk: block.block(),
            bar: Rq(neo_math::superneo_bar_block(orig.0)),
            orig,
        }
    }

    #[inline]
    pub(super) fn expanded_row_blocks(&self, row: usize) -> Vec<RowBlock> {
        if self.identity {
            let mut orig = [F::ZERO; D];
            orig[row % D] = F::ONE;
            return vec![RowBlock {
                blk: row / D,
                bar: Rq(neo_math::superneo_bar_block(orig)),
                orig: Rq(orig),
            }];
        }
        self.row_blocks_for(row)
            .iter()
            .copied()
            .map(|block| self.expanded_block(block))
            .collect()
    }

    #[inline]
    pub(super) fn compact_dot_real(&self, block: CompactRowBlock, input: &SuperneoZBlocks, block_index: usize) -> F {
        if let Some((local, coefficient)) = block.single_parts() {
            coefficient * input.real_coefficient(block_index, local)
        } else {
            input.real_dot(
                &self.dense_block(block.dense_index().expect("compact dense block")),
                block_index,
            )
        }
    }

    #[inline]
    pub fn compile_weighted_rows(&self, weights: &[K; D]) -> SuperneoWeightedMatrixCache {
        let (basis_re_forms, basis_im_forms) = weighted_projection_basis_forms_from_k(weights);
        self.compile_weighted_rows_with_basis(&basis_re_forms, &basis_im_forms)
    }

    #[inline]
    pub(super) fn compile_weighted_rows_with_basis(
        &self,
        basis_re_forms: &[Rq; D],
        basis_im_forms: &[Rq; D],
    ) -> SuperneoWeightedMatrixCache {
        if self.identity {
            let local_forms = (0..D)
                .map(|local| {
                    let re_form = basis_re_forms[local];
                    let im_form = basis_im_forms[local];
                    (!is_all_zero(&re_form.0) || !is_all_zero(&im_form.0)).then_some((re_form, im_form))
                })
                .collect::<Vec<_>>();

            let mut row_offsets = Vec::with_capacity(self.rows + 1);
            let mut row_blocks = Vec::with_capacity(self.rows);
            row_offsets.push(0);
            for row in 0..self.rows {
                let local = row % D;
                if let Some((re_form, im_form)) = local_forms[local] {
                    row_blocks.push(WeightedRowBlock {
                        blk: row / D,
                        re_form,
                        im_form,
                    });
                }
                row_offsets.push(row_blocks.len());
            }

            return SuperneoWeightedMatrixCache {
                rows: self.rows,
                cols: self.cols,
                row_offsets,
                row_blocks,
            };
        }

        let compile_rows = |row_start: usize, row_end: usize| {
            let mut offsets = Vec::with_capacity(row_end - row_start + 1);
            let mut blocks = Vec::new();
            offsets.push(0);
            for row in row_start..row_end {
                for rb in self.row_blocks_including_seeded(row) {
                    let re_form = weighted_projection_form_from_orig(&rb.orig, &basis_re_forms);
                    let im_form = weighted_projection_form_from_orig(&rb.orig, &basis_im_forms);
                    if is_all_zero(&re_form.0) && is_all_zero(&im_form.0) {
                        continue;
                    }
                    blocks.push(WeightedRowBlock {
                        blk: rb.blk,
                        re_form,
                        im_form,
                    });
                }
                offsets.push(blocks.len());
            }
            (offsets, blocks)
        };

        const CHUNK_ROWS: usize = 2048;
        let chunk_count = self.rows.div_ceil(CHUNK_ROWS);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let chunks: Vec<(Vec<usize>, Vec<WeightedRowBlock>)> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk| {
                let row_start = chunk * CHUNK_ROWS;
                let row_end = core::cmp::min(row_start + CHUNK_ROWS, self.rows);
                compile_rows(row_start, row_end)
            })
            .collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let chunks: Vec<(Vec<usize>, Vec<WeightedRowBlock>)> = (0..chunk_count)
            .map(|chunk| {
                let row_start = chunk * CHUNK_ROWS;
                let row_end = core::cmp::min(row_start + CHUNK_ROWS, self.rows);
                compile_rows(row_start, row_end)
            })
            .collect();

        let total_blocks = chunks.iter().map(|(_, blocks)| blocks.len()).sum();
        let mut row_offsets = Vec::with_capacity(self.rows + 1);
        let mut row_blocks = Vec::with_capacity(total_blocks);
        row_offsets.push(0);
        let mut offset_base = 0usize;
        for (offsets, blocks) in chunks {
            for local_end in offsets.iter().skip(1) {
                row_offsets.push(offset_base + *local_end);
            }
            offset_base += blocks.len();
            row_blocks.extend(blocks);
        }

        SuperneoWeightedMatrixCache {
            rows: self.rows,
            cols: self.cols,
            row_offsets,
            row_blocks,
        }
    }

    #[inline]
    pub fn row_dot_ring_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::row_dot_ring_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return [K::ZERO; D];
        }

        let mut row_re = [F::ZERO; D];
        let mut row_im = [F::ZERO; D];

        for rb in self.row_blocks_including_seeded(row) {
            if !z_blocks.block_nonzero(rb.blk) {
                continue;
            }
            if z_blocks.real_nonzero(rb.blk) {
                let prod_re = z_blocks.real_mul(&rb.bar, rb.blk);
                for i in 0..D {
                    row_re[i] += prod_re.0[i];
                }
            }
            if !z_blocks.imag_all_zero && z_blocks.im_nonzero[rb.blk] {
                let prod_im = rb.bar.mul(&z_blocks.im[rb.blk]);
                for i in 0..D {
                    row_im[i] += prod_im.0[i];
                }
            }
        }

        let mut out = [K::ZERO; D];
        if z_blocks.imag_all_zero {
            for i in 0..D {
                out[i] = K::from_coeffs([row_re[i], F::ZERO]);
            }
            return out;
        }
        for i in 0..D {
            out[i] = K::from_coeffs([row_re[i], row_im[i]]);
        }
        out
    }

    #[inline]
    pub fn row_dot_ring_weighted_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks, weights: &[K; D]) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::row_dot_ring_weighted_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        if z_blocks.imag_all_zero {
            let mut acc = K::ZERO;
            for rb in self.row_blocks_including_seeded(row) {
                if !z_blocks.real_nonzero(rb.blk) {
                    continue;
                }
                let prod_re = z_blocks.real_mul(&rb.bar, rb.blk);
                for i in 0..D {
                    let v = prod_re.0[i];
                    if v != F::ZERO {
                        acc += weights[i].scale_base(v);
                    }
                }
            }
            return acc;
        }

        let row_coeffs = self.row_dot_ring_with_blocks(row, z_blocks);
        let mut acc = K::ZERO;
        for i in 0..D {
            let coeff = row_coeffs[i];
            if weights[i] != K::ZERO && coeff != K::ZERO {
                acc += weights[i] * coeff;
            }
        }
        acc
    }

    #[inline]
    pub fn row_dot_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::row_dot_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc_re = F::ZERO;
        let mut acc_im = F::ZERO;

        for rb in self.row_blocks_including_seeded(row) {
            if !z_blocks.block_nonzero(rb.blk) {
                continue;
            }
            if z_blocks.real_nonzero(rb.blk) {
                acc_re += z_blocks.real_dot(&rb.orig, rb.blk);
            }
            if !z_blocks.imag_all_zero && z_blocks.im_nonzero[rb.blk] {
                acc_im += coeff_dot(&rb.orig, &z_blocks.im[rb.blk]);
            }
        }
        if z_blocks.imag_all_zero {
            return K::from_coeffs([acc_re, F::ZERO]);
        }
        K::from_coeffs([acc_re, acc_im])
    }

    #[inline]
    pub fn row_dot(&self, row: usize, z: &[K]) -> K {
        assert_eq!(
            self.cols,
            z.len(),
            "SuperneoMatrixCache::row_dot: column/vector length mismatch"
        );
        let z_blocks = SuperneoZBlocks::from_z(z);
        self.row_dot_with_blocks(row, &z_blocks)
    }

    #[inline]
    pub fn eval_mle_with_blocks(&self, z_blocks: &SuperneoZBlocks, chi_r: &[K], n_eff: usize) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::eval_mle_with_blocks: block count mismatch"
        );
        let row_cap = min(min(self.rows, n_eff), chi_r.len());
        let mut acc = K::ZERO;
        for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
            if w == K::ZERO {
                continue;
            }
            acc += w * self.row_dot_with_blocks(row, z_blocks);
        }
        acc
    }

    #[inline]
    pub fn eval_mle(&self, z: &[K], chi_r: &[K], n_eff: usize) -> K {
        assert_eq!(
            self.cols,
            z.len(),
            "SuperneoMatrixCache::eval_mle: column/vector length mismatch"
        );
        let z_blocks = SuperneoZBlocks::from_z(z);
        self.eval_mle_with_blocks(&z_blocks, chi_r, n_eff)
    }

    /// Evaluate `\widetilde{(M z)}(r)` in ring-coefficient form.
    pub fn eval_mle_ring_with_blocks(&self, z_blocks: &SuperneoZBlocks, chi_r: &[K], n_eff: usize) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks: block count mismatch"
        );
        let row_cap = min(min(self.rows, n_eff), chi_r.len());
        if z_blocks.imag_all_zero {
            let mut out_re = [F::ZERO; D];
            let mut out_im = [F::ZERO; D];
            for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
                if w == K::ZERO {
                    continue;
                }
                let [w_re, w_im] = w.as_coeffs();
                for rb in self.row_blocks_including_seeded(row) {
                    if !z_blocks.real_nonzero(rb.blk) {
                        continue;
                    }
                    let prod_re = z_blocks.real_mul(&rb.bar, rb.blk);
                    for i in 0..D {
                        let v = prod_re.0[i];
                        out_re[i] += w_re * v;
                        out_im[i] += w_im * v;
                    }
                }
            }
            let mut out = [K::ZERO; D];
            for i in 0..D {
                out[i] = K::from_coeffs([out_re[i], out_im[i]]);
            }
            return out;
        }

        let mut out = [K::ZERO; D];
        for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
            if w == K::ZERO {
                continue;
            }
            let row_coeffs = self.row_dot_ring_with_blocks(row, z_blocks);
            for i in 0..D {
                out[i] += w * row_coeffs[i];
            }
        }
        out
    }

    #[inline]
    pub(super) fn eval_mle_ring_with_blocks_split_chi_scratch(
        &self,
        z_blocks: &SuperneoZBlocks,
        chi_re: &[F],
        chi_im: &[F],
        n_eff: usize,
        scratch: &mut RingEvalScratch,
    ) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks_split_chi: block count mismatch"
        );
        debug_assert_eq!(
            chi_re.len(),
            chi_im.len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks_split_chi: chi coeff length mismatch"
        );
        let block_count = z_blocks.block_len();
        debug_assert_eq!(self.cols.div_ceil(D), block_count);
        self.accumulate_ring_form_split_chi(chi_re, chi_im, n_eff, scratch);
        let out = eval_ring_scratch_real_z_blocks(scratch, z_blocks);
        scratch.clear_active();
        out
    }

    #[inline]
    pub fn eval_mle_ring(&self, z: &[K], chi_r: &[K], n_eff: usize) -> [K; D] {
        assert_eq!(
            self.cols,
            z.len(),
            "SuperneoMatrixCache::eval_mle_ring: column/vector length mismatch"
        );
        let z_blocks = SuperneoZBlocks::from_z(z);
        self.eval_mle_ring_with_blocks(&z_blocks, chi_r, n_eff)
    }

    /// Build sparse `v = M^T · χ_r` once, so repeated evals at the same `r` are cheap.
    #[inline]
    pub fn build_linear_form(&self, chi_r: &[K], n_eff: usize) -> SuperneoLinearForm {
        let row_cap = min(min(self.rows, n_eff), chi_r.len());
        let mut dense = vec![K::ZERO; self.cols];
        for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
            if w == K::ZERO {
                continue;
            }
            for rb in self.row_blocks_including_seeded(row) {
                let base = rb.blk * D;
                for i in 0..D {
                    let a = rb.orig.0[i];
                    if a != F::ZERO {
                        dense[base + i] += w.scale_base_k(K::from(a));
                    }
                }
            }
        }
        let nz = dense
            .into_iter()
            .enumerate()
            .filter_map(|(c, v)| (v != K::ZERO).then_some((c, v)))
            .collect();
        SuperneoLinearForm { cols: self.cols, nz }
    }

    /// Build a sparse ring-linear form for repeated real-only packed witness evals.
    #[inline]
    pub fn build_ring_linear_form(&self, chi_r: &[K], n_eff: usize) -> SuperneoRingLinearForm {
        let (chi_re, chi_im) = split_chi_coeffs(chi_r, n_eff);
        self.build_ring_linear_form_split_chi(&chi_re, &chi_im, n_eff)
    }

    #[inline]
    pub(super) fn build_ring_linear_form_split_chi(
        &self,
        chi_re: &[F],
        chi_im: &[F],
        n_eff: usize,
    ) -> SuperneoRingLinearForm {
        let block_count = self.cols.div_ceil(D);
        let mut scratch = RingEvalScratch::new(block_count);
        self.build_ring_linear_form_split_chi_with_scratch(chi_re, chi_im, n_eff, &mut scratch)
    }

    #[inline]
    pub(super) fn build_ring_linear_form_split_chi_with_scratch(
        &self,
        chi_re: &[F],
        chi_im: &[F],
        n_eff: usize,
        scratch: &mut RingEvalScratch,
    ) -> SuperneoRingLinearForm {
        self.accumulate_ring_form_split_chi(chi_re, chi_im, n_eff, scratch);

        let mut entries = Vec::with_capacity(scratch.active_blocks.len());
        for &blk in &scratch.active_blocks {
            let re_form = scratch.agg_re[blk];
            let im_form = scratch.agg_im[blk];
            let re_nonzero = !is_all_zero(&re_form.0);
            let im_nonzero = !is_all_zero(&im_form.0);
            if re_nonzero || im_nonzero {
                entries.push(SuperneoRingLinearBlock {
                    blk,
                    re_form,
                    im_form,
                    re_nonzero,
                    im_nonzero,
                });
            }
        }

        scratch.clear_active();

        SuperneoRingLinearForm {
            cols: self.cols,
            entries,
        }
    }
}
