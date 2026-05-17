//! SuperNeo transformed-matrix evaluators for reductions integration.
//!
//! These helpers evaluate multilinear extensions using `bar(M)` plus ring
//! constant-term products. They are intentionally isolated so engines can adopt
//! them incrementally without changing protocol wiring in one large patch.

use core::cmp::min;

use neo_ccs::{CcsMatrix, CcsStructure, Mat};
use neo_math::KExtensions;
use neo_math::{ct, superneo_bar_block, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[inline]
fn matrix_entry<Ff>(mat: &CcsMatrix<Ff>, row: usize, col: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if row >= mat.rows() || col >= mat.cols() {
        return Ff::ZERO;
    }
    match mat {
        CcsMatrix::Identity { .. } => {
            if row == col {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(csc) => {
            let s = csc.col_ptr[col];
            let e = csc.col_ptr[col + 1];
            match csc.row_idx[s..e].binary_search(&row) {
                Ok(idx) => csc.vals[s + idx],
                Err(_) => Ff::ZERO,
            }
        }
    }
}

/// Row dot-product using a transformed row `bar(a)` and SuperNeo's `ct` product.
///
/// Returns `sum_t ct( cf_inv(bar(a_t)) * cf_inv(z_t) )` over `D`-coefficient blocks.
pub fn superneo_row_dot_transformed_matrix(mat_bar: &CcsMatrix<F>, row: usize, z: &[K]) -> K {
    assert_eq!(
        mat_bar.cols(),
        z.len(),
        "superneo_row_dot_transformed_matrix: column/vector length mismatch"
    );
    if row >= mat_bar.rows() {
        return K::ZERO;
    }

    let blocks = z.len().div_ceil(D);
    let mut acc_re = F::ZERO;
    let mut acc_im = F::ZERO;

    for blk in 0..blocks {
        let base = blk * D;
        let mut a_bar = [F::ZERO; D];
        let mut z_re = [F::ZERO; D];
        let mut z_im = [F::ZERO; D];

        for i in 0..D {
            a_bar[i] = matrix_entry(mat_bar, row, base + i);
            if base + i < z.len() {
                let [re, im] = z[base + i].as_coeffs();
                z_re[i] = re;
                z_im[i] = im;
            }
        }

        let a_ring = Rq(a_bar);
        acc_re += ct(&a_ring.mul(&Rq(z_re)));
        acc_im += ct(&a_ring.mul(&Rq(z_im)));
    }

    K::from_coeffs([acc_re, acc_im])
}

#[inline]
fn as_base_field<Ff>(v: Ff) -> F
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    K::from(v).real()
}

/// Cached SuperNeo row-lifted representation for one matrix.
///
/// For each row, stores sparse transformed blocks `(block_idx, bar(a_block))`.
#[derive(Clone, Copy, Debug)]
struct RowBlock {
    blk: usize,
    /// SuperNeo transformed coefficients for this block.
    bar: Rq,
    /// Original row coefficients for this block.
    orig: Rq,
}

#[derive(Clone, Debug)]
pub struct SuperneoMatrixCache {
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    row_blocks: Vec<RowBlock>,
    identity: bool,
}

#[derive(Clone, Copy, Debug)]
struct WeightedRowBlock {
    blk: usize,
    re_form: Rq,
    im_form: Rq,
}

/// Fixed weighted projection of one matrix's row-ring outputs.
#[derive(Clone, Debug)]
pub struct SuperneoWeightedMatrixCache {
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    row_blocks: Vec<WeightedRowBlock>,
}

#[derive(Clone, Debug)]
struct RingEvalScratch {
    agg_re: Vec<Rq>,
    agg_im: Vec<Rq>,
    touched: Vec<bool>,
    active_blocks: Vec<usize>,
}

impl RingEvalScratch {
    #[inline]
    fn new(block_count: usize) -> Self {
        Self {
            agg_re: vec![Rq::zero(); block_count],
            agg_im: vec![Rq::zero(); block_count],
            touched: vec![false; block_count],
            active_blocks: Vec::new(),
        }
    }

    #[inline]
    fn ensure_block_count(&mut self, block_count: usize) {
        if self.agg_re.len() == block_count {
            return;
        }
        self.agg_re.resize(block_count, Rq::zero());
        self.agg_im.resize(block_count, Rq::zero());
        self.touched.resize(block_count, false);
        self.active_blocks.clear();
    }

    #[inline]
    fn clear_active(&mut self) {
        for &blk in &self.active_blocks {
            self.agg_re[blk] = Rq::zero();
            self.agg_im[blk] = Rq::zero();
            self.touched[blk] = false;
        }
        self.active_blocks.clear();
    }
}

/// Precomputed linear form `v = M^T · χ_r` in sparse `(col, value)` form.
///
/// This lets callers evaluate `\tilde{(M z)}(r)` as a single sparse dot product
/// without scanning all rows for each `z`.
#[derive(Clone, Debug)]
pub struct SuperneoLinearForm {
    cols: usize,
    nz: Vec<(usize, K)>,
}

impl SuperneoLinearForm {
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn eval_vec_k(&self, z: &[K]) -> K {
        assert_eq!(
            z.len(),
            self.cols,
            "SuperneoLinearForm::eval_vec_k: column/vector length mismatch"
        );
        let mut acc = K::ZERO;
        for &(c, v) in &self.nz {
            acc += z[c] * v;
        }
        acc
    }

    /// Evaluate packed SuperNeo witness coefficients and return Ajtai digit lanes.
    ///
    /// For packed witnesses, logical column `c` belongs to digit lane `rho = c % D`.
    /// This computes all `D` lane sums in one pass over the sparse linear form.
    #[inline]
    pub fn eval_packed_digits_k(&self, z: &[K]) -> [K; D] {
        assert_eq!(
            z.len(),
            self.cols,
            "SuperneoLinearForm::eval_packed_digits_k: column/vector length mismatch"
        );
        let mut out = [K::ZERO; D];
        for &(c, v) in &self.nz {
            out[c % D] += z[c] * v;
        }
        out
    }

    #[inline]
    pub fn eval_vec_base_f<Ff>(&self, z_row: &[Ff]) -> K
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
    {
        assert_eq!(
            z_row.len(),
            self.cols,
            "SuperneoLinearForm::eval_vec_base_f: column/vector length mismatch"
        );
        let mut acc = K::ZERO;
        for &(c, v) in &self.nz {
            acc += v.scale_base_k(K::from(z_row[c]));
        }
        acc
    }

    #[inline]
    pub fn eval_vec_base_f_with<Ff, G>(&self, mut get: G) -> K
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
        G: FnMut(usize) -> Ff,
    {
        let mut acc = K::ZERO;
        for &(c, v) in &self.nz {
            acc += v.scale_base_k(K::from(get(c)));
        }
        acc
    }
}

/// Precomputed ring-linear form `bar(M)^T · chi_r`, grouped by D-column block.
///
/// This lets DEC evaluate many digit witnesses at the same row challenge without
/// rescanning every matrix row for each child.
#[derive(Clone, Debug)]
pub struct SuperneoRingLinearForm {
    cols: usize,
    entries: Vec<SuperneoRingLinearBlock>,
}

#[derive(Clone, Debug)]
struct SuperneoRingLinearBlock {
    blk: usize,
    re_form: Rq,
    im_form: Rq,
}

impl SuperneoRingLinearForm {
    #[inline]
    pub fn eval_real_z_blocks(&self, z_blocks: &SuperneoZBlocks) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoRingLinearForm::eval_real_z_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoRingLinearForm::eval_real_z_blocks expects real-only witness blocks"
        );

        let mut out_re = [F::ZERO; D];
        let mut out_im = [F::ZERO; D];
        for entry in &self.entries {
            let z_re = &z_blocks.re[entry.blk];
            if !is_all_zero(&entry.re_form.0) {
                let prod_re = entry.re_form.mul(z_re);
                for i in 0..D {
                    out_re[i] += prod_re.0[i];
                }
            }
            if !is_all_zero(&entry.im_form.0) {
                let prod_im = entry.im_form.mul(z_re);
                for i in 0..D {
                    out_im[i] += prod_im.0[i];
                }
            }
        }

        let mut out = [K::ZERO; D];
        for i in 0..D {
            out[i] = K::from_coeffs([out_re[i], out_im[i]]);
        }
        out
    }
}

/// Pre-split `z` blocks for repeated SuperNeo row dot-products.
#[derive(Clone, Debug)]
pub struct SuperneoZBlocks {
    re: Vec<Rq>,
    im: Vec<Rq>,
    imag_all_zero: bool,
}

impl SuperneoZBlocks {
    #[inline]
    pub fn with_block_len(blocks: usize) -> Self {
        Self {
            re: vec![Rq([F::ZERO; D]); blocks],
            im: Vec::new(),
            imag_all_zero: true,
        }
    }

    #[inline]
    pub fn from_z(z: &[K]) -> Self {
        let blocks = z.len().div_ceil(D);
        let mut re = Vec::with_capacity(blocks);
        let mut im = Vec::with_capacity(blocks);
        let mut imag_all_zero = true;
        for blk in 0..blocks {
            let base = blk * D;
            let mut zr = [F::ZERO; D];
            let mut zi = [F::ZERO; D];
            for i in 0..D {
                if base + i < z.len() {
                    let [r, im_part] = z[base + i].as_coeffs();
                    zr[i] = r;
                    zi[i] = im_part;
                    imag_all_zero &= im_part == F::ZERO;
                }
            }
            re.push(Rq(zr));
            im.push(Rq(zi));
        }
        Self { re, im, imag_all_zero }
    }

    #[inline]
    pub fn from_witness_mat<Ff>(z: &Mat<Ff>, expected_m: usize) -> Result<Self, crate::PiCcsError>
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
    {
        crate::common::validate_superneo_witness_mat(z, expected_m)?;
        let blocks = expected_m.div_ceil(D);
        let mut re = Vec::with_capacity(blocks);
        for blk in 0..blocks {
            let mut zr = [F::ZERO; D];
            for (i, cell) in zr.iter_mut().enumerate() {
                *cell = as_base_field(z[(i, blk)]);
            }
            re.push(Rq(zr));
        }
        Ok(Self {
            re,
            im: Vec::new(),
            imag_all_zero: true,
        })
    }

    #[inline]
    pub fn from_base_row_f<Ff>(row: &[Ff]) -> Self
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
    {
        let blocks = row.len().div_ceil(D);
        let mut re = Vec::with_capacity(blocks);
        for blk in 0..blocks {
            let base = blk * D;
            let mut zr = [F::ZERO; D];
            for i in 0..D {
                if base + i < row.len() {
                    zr[i] = as_base_field(row[base + i]);
                }
            }
            re.push(Rq(zr));
        }
        Self {
            re,
            im: Vec::new(),
            imag_all_zero: true,
        }
    }

    #[inline]
    pub fn load_base_row_f<Ff>(&mut self, row: &[Ff])
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
    {
        let blocks = row.len().div_ceil(D);
        if self.re.len() != blocks {
            *self = Self::with_block_len(blocks);
        }
        self.imag_all_zero = true;
        self.im.clear();
        for blk in 0..blocks {
            let base = blk * D;
            for i in 0..D {
                self.re[blk].0[i] = if base + i < row.len() {
                    as_base_field(row[base + i])
                } else {
                    F::ZERO
                };
            }
        }
    }

    #[inline]
    pub fn imag_all_zero(&self) -> bool {
        self.imag_all_zero
    }
}

impl SuperneoMatrixCache {
    #[inline]
    fn row_blocks_for(&self, row: usize) -> &[RowBlock] {
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        &self.row_blocks[start..end]
    }

    #[inline]
    pub fn compile_weighted_rows(&self, weights: &[K; D]) -> SuperneoWeightedMatrixCache {
        let mut weight_re = [F::ZERO; D];
        let mut weight_im = [F::ZERO; D];
        for (i, weight) in weights.iter().enumerate() {
            let [re, im] = weight.as_coeffs();
            weight_re[i] = re;
            weight_im[i] = im;
        }
        let weight_re = Rq(weight_re);
        let weight_im = Rq(weight_im);
        let basis_re_forms = weighted_projection_basis_forms(&weight_re);
        let basis_im_forms = weighted_projection_basis_forms(&weight_im);

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
                for rb in self.row_blocks_for(row) {
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

    /// Evaluate one matrix row against packed witness blocks and return all `D` ring coefficients.
    #[inline]
    pub fn row_dot_ring_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoMatrixCache::row_dot_ring_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return [K::ZERO; D];
        }

        let mut row_re = [F::ZERO; D];
        let mut row_im = [F::ZERO; D];

        for rb in self.row_blocks_for(row) {
            let prod_re = rb.bar.mul(&z_blocks.re[rb.blk]);
            for i in 0..D {
                row_re[i] += prod_re.0[i];
            }
            if !z_blocks.imag_all_zero {
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
            z_blocks.re.len(),
            "SuperneoMatrixCache::row_dot_ring_weighted_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        if z_blocks.imag_all_zero {
            let mut acc = K::ZERO;
            for rb in self.row_blocks_for(row) {
                let prod_re = rb.bar.mul(&z_blocks.re[rb.blk]);
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
            z_blocks.re.len(),
            "SuperneoMatrixCache::row_dot_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc_re = F::ZERO;
        let mut acc_im = F::ZERO;

        for rb in self.row_blocks_for(row) {
            acc_re += coeff_dot(&rb.orig, &z_blocks.re[rb.blk]);
            if !z_blocks.imag_all_zero {
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
            z_blocks.re.len(),
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
    ///
    /// Returns the `D` coefficients of the ring element in `K`.
    pub fn eval_mle_ring_with_blocks(&self, z_blocks: &SuperneoZBlocks, chi_r: &[K], n_eff: usize) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks: block count mismatch"
        );
        let row_cap = min(min(self.rows, n_eff), chi_r.len());
        if z_blocks.imag_all_zero {
            let mut out_re = [F::ZERO; D];
            let mut out_im = [F::ZERO; D];
            let z_re = &z_blocks.re;
            for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
                if w == K::ZERO {
                    continue;
                }
                let [w_re, w_im] = w.as_coeffs();
                for rb in self.row_blocks_for(row) {
                    let prod_re = rb.bar.mul(&z_re[rb.blk]);
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
    fn eval_mle_ring_with_blocks_split_chi_scratch(
        &self,
        z_blocks: &SuperneoZBlocks,
        chi_re: &[F],
        chi_im: &[F],
        n_eff: usize,
        scratch: &mut RingEvalScratch,
    ) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks_split_chi: block count mismatch"
        );
        debug_assert_eq!(
            chi_re.len(),
            chi_im.len(),
            "SuperneoMatrixCache::eval_mle_ring_with_blocks_split_chi: chi coeff length mismatch"
        );
        let row_cap = min(min(self.rows, n_eff), chi_re.len());
        let block_count = z_blocks.re.len();
        scratch.ensure_block_count(block_count);
        for row in 0..row_cap {
            let w_re = chi_re[row];
            let w_im = chi_im[row];
            if w_re == F::ZERO && w_im == F::ZERO {
                continue;
            }
            for rb in self.row_blocks_for(row) {
                let blk = rb.blk;
                if !scratch.touched[blk] {
                    scratch.touched[blk] = true;
                    scratch.active_blocks.push(blk);
                }
                add_scaled_rq(&mut scratch.agg_re[blk], &rb.bar, w_re);
                add_scaled_rq(&mut scratch.agg_im[blk], &rb.bar, w_im);
            }
        }
        let mut out_re = [F::ZERO; D];
        let mut out_im = [F::ZERO; D];
        let z_re = &z_blocks.re;
        for &blk in &scratch.active_blocks {
            if !is_all_zero(&scratch.agg_re[blk].0) {
                let prod_re = scratch.agg_re[blk].mul(&z_re[blk]);
                for i in 0..D {
                    out_re[i] += prod_re.0[i];
                }
            }
            if !is_all_zero(&scratch.agg_im[blk].0) {
                let prod_im = scratch.agg_im[blk].mul(&z_re[blk]);
                for i in 0..D {
                    out_im[i] += prod_im.0[i];
                }
            }
        }
        scratch.clear_active();
        let mut out = [K::ZERO; D];
        for i in 0..D {
            out[i] = K::from_coeffs([out_re[i], out_im[i]]);
        }
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
            for rb in self.row_blocks_for(row) {
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
    fn build_ring_linear_form_split_chi(&self, chi_re: &[F], chi_im: &[F], n_eff: usize) -> SuperneoRingLinearForm {
        let row_cap = min(min(self.rows, n_eff), chi_re.len());
        let block_count = self.cols.div_ceil(D);
        let mut scratch = RingEvalScratch::new(block_count);

        for row in 0..row_cap {
            let w_re = chi_re[row];
            let w_im = chi_im[row];
            if w_re == F::ZERO && w_im == F::ZERO {
                continue;
            }
            for rb in self.row_blocks_for(row) {
                let blk = rb.blk;
                if !scratch.touched[blk] {
                    scratch.touched[blk] = true;
                    scratch.active_blocks.push(blk);
                }
                add_scaled_rq(&mut scratch.agg_re[blk], &rb.bar, w_re);
                add_scaled_rq(&mut scratch.agg_im[blk], &rb.bar, w_im);
            }
        }

        let mut entries = Vec::with_capacity(scratch.active_blocks.len());
        for &blk in &scratch.active_blocks {
            let re_form = scratch.agg_re[blk];
            let im_form = scratch.agg_im[blk];
            if !is_all_zero(&re_form.0) || !is_all_zero(&im_form.0) {
                entries.push(SuperneoRingLinearBlock { blk, re_form, im_form });
            }
        }

        SuperneoRingLinearForm {
            cols: self.cols,
            entries,
        }
    }
}

impl SuperneoWeightedMatrixCache {
    #[inline]
    pub fn row_dot_real_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoWeightedMatrixCache::row_dot_real_with_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoWeightedMatrixCache::row_dot_real_with_blocks expects real-only witness blocks"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc_re = F::ZERO;
        let mut acc_im = F::ZERO;
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        for rb in &self.row_blocks[start..end] {
            let z_re = &z_blocks.re[rb.blk];
            acc_re += coeff_dot(&rb.re_form, z_re);
            acc_im += coeff_dot(&rb.im_form, z_re);
        }
        K::from_coeffs([acc_re, acc_im])
    }
}

/// Cached SuperNeo row-lifted representation for all CCS matrices.
#[derive(Clone, Debug)]
pub struct SuperneoEvalCache {
    mats: Vec<SuperneoMatrixCache>,
}

impl SuperneoEvalCache {
    #[inline]
    pub fn matrix(&self, j: usize) -> Option<&SuperneoMatrixCache> {
        self.mats.get(j)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mats.len()
    }

    #[inline]
    pub fn build_linear_forms(&self, chi_r: &[K], n_eff: usize) -> Vec<SuperneoLinearForm> {
        self.mats
            .iter()
            .map(|m| m.build_linear_form(chi_r, n_eff))
            .collect()
    }

    #[inline]
    pub fn build_ring_linear_forms(&self, chi_r: &[K], n_eff: usize) -> Vec<SuperneoRingLinearForm> {
        let (chi_re, chi_im) = split_chi_coeffs(chi_r, n_eff);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            self.mats
                .par_iter()
                .map(|m| m.build_ring_linear_form_split_chi(&chi_re, &chi_im, n_eff))
                .collect()
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            self.mats
                .iter()
                .map(|m| m.build_ring_linear_form_split_chi(&chi_re, &chi_im, n_eff))
                .collect()
        }
    }

    #[inline]
    pub fn build_weighted_matrix_caches(&self, weights: &[K; D]) -> Vec<SuperneoWeightedMatrixCache> {
        self.mats
            .iter()
            .map(|m| m.compile_weighted_rows(weights))
            .collect()
    }
}

#[inline]
fn is_all_zero(arr: &[F; D]) -> bool {
    arr.iter().all(|&v| v == F::ZERO)
}

#[inline]
fn split_chi_coeffs(chi_r: &[K], n_eff: usize) -> (Vec<F>, Vec<F>) {
    let row_cap = min(n_eff, chi_r.len());
    let mut chi_re = Vec::with_capacity(row_cap);
    let mut chi_im = Vec::with_capacity(row_cap);
    for &w in chi_r.iter().take(row_cap) {
        let [re, im] = w.as_coeffs();
        chi_re.push(re);
        chi_im.push(im);
    }
    (chi_re, chi_im)
}

#[inline]
fn add_scaled_rq(dst: &mut Rq, src: &Rq, scale: F) {
    if scale == F::ZERO {
        return;
    }
    for i in 0..D {
        dst.0[i] += scale * src.0[i];
    }
}

#[inline]
fn coeff_dot(lhs: &Rq, rhs: &Rq) -> F {
    let mut acc = F::ZERO;
    for i in 0..D {
        acc += lhs.0[i] * rhs.0[i];
    }
    acc
}

#[inline]
fn weighted_projection_basis_forms(weights: &Rq) -> [Rq; D] {
    let mut forms = [Rq([F::ZERO; D]); D];
    for (local, slot) in forms.iter_mut().enumerate() {
        let mut e = [F::ZERO; D];
        e[local] = F::ONE;
        let bar = Rq(superneo_bar_block(e));
        *slot = weighted_projection_form(&bar, weights);
    }
    forms
}

#[inline]
fn weighted_projection_form_from_orig(orig: &Rq, basis_forms: &[Rq; D]) -> Rq {
    let mut out = [F::ZERO; D];
    for (local, &coeff) in orig.0.iter().enumerate() {
        if coeff == F::ZERO {
            continue;
        }
        let form = &basis_forms[local].0;
        for i in 0..D {
            out[i] += coeff * form[i];
        }
    }
    Rq(out)
}

#[inline]
fn weighted_projection_form(lhs: &Rq, weights: &Rq) -> Rq {
    let mut out = [F::ZERO; D];
    for (basis, out_cell) in out.iter_mut().enumerate() {
        *out_cell = coeff_dot_mul_by_monomial(lhs, weights, basis);
    }
    Rq(out)
}

#[inline]
fn coeff_dot_mul_by_monomial(lhs: &Rq, weights: &Rq, basis: usize) -> F {
    if basis == 0 {
        return coeff_dot(weights, lhs);
    }

    let mut acc = F::ZERO;
    for i in 0..D {
        let coeff = lhs.0[i];
        if coeff == F::ZERO {
            continue;
        }

        let new_deg = i + basis;
        if new_deg < D {
            acc += weights.0[new_deg] * coeff;
        } else if new_deg < D + D / 2 {
            let reduced_deg = new_deg - D;
            acc -= weights.0[reduced_deg] * coeff;
            acc -= weights.0[reduced_deg + D / 2] * coeff;
        } else {
            let deg1 = new_deg - D / 2;
            let deg2 = new_deg - D;
            if deg2 < D {
                acc -= weights.0[deg2] * coeff;
            }
            if deg1 >= D {
                let deg1_red = deg1 - D;
                if deg1_red < D {
                    acc += weights.0[deg1_red] * coeff;
                    if deg1_red + D / 2 < D {
                        acc += weights.0[deg1_red + D / 2] * coeff;
                    }
                }
            } else {
                acc -= weights.0[deg1] * coeff;
            }
        }
    }
    acc
}

fn build_matrix_cache<Ff>(mat: &CcsMatrix<Ff>) -> SuperneoMatrixCache
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let rows = mat.rows();
    let cols = mat.cols().div_ceil(D) * D;
    match mat {
        CcsMatrix::Identity { .. } => {
            // Transform basis vectors once: bar(e_local).
            let mut basis_bar = [Rq([F::ZERO; D]); D];
            let mut basis_orig = [Rq([F::ZERO; D]); D];
            for (local, out) in basis_bar.iter_mut().enumerate().take(D) {
                let mut e = [F::ZERO; D];
                e[local] = F::ONE;
                basis_orig[local] = Rq(e);
                *out = Rq(superneo_bar_block(e));
            }
            let mut row_offsets = Vec::with_capacity(rows + 1);
            let mut row_blocks = Vec::with_capacity(rows);
            row_offsets.push(0);
            for row in 0..rows {
                let blk = row / D;
                let local = row % D;
                row_blocks.push(RowBlock {
                    blk,
                    bar: basis_bar[local],
                    orig: basis_orig[local],
                });
                row_offsets.push(row_blocks.len());
            }
            SuperneoMatrixCache {
                rows,
                cols,
                row_offsets,
                row_blocks,
                identity: true,
            }
        }
        CcsMatrix::Csc(csc) => {
            let mut counts = vec![0usize; rows];
            let mut last_blk_by_row = vec![usize::MAX; rows];
            for c in 0..csc.ncols {
                let blk = c / D;
                let s = csc.col_ptr[c];
                let e = csc.col_ptr[c + 1];
                for k in s..e {
                    let row = csc.row_idx[k];
                    let v = as_base_field(csc.vals[k]);
                    if v == F::ZERO {
                        continue;
                    }
                    if last_blk_by_row[row] != blk {
                        counts[row] += 1;
                        last_blk_by_row[row] = blk;
                    }
                }
            }

            let total_blocks: usize = counts.iter().sum();
            let mut row_offsets = Vec::with_capacity(rows + 1);
            row_offsets.push(0);
            for count in counts {
                row_offsets.push(row_offsets.last().copied().unwrap() + count);
            }

            let empty_block = RowBlock {
                blk: 0,
                bar: Rq([F::ZERO; D]),
                orig: Rq([F::ZERO; D]),
            };
            let mut row_blocks = vec![empty_block; total_blocks];
            let mut cursor = row_offsets[..rows].to_vec();
            last_blk_by_row.fill(usize::MAX);

            for c in 0..csc.ncols {
                let blk = c / D;
                let local = c % D;
                let s = csc.col_ptr[c];
                let e = csc.col_ptr[c + 1];
                for k in s..e {
                    let row = csc.row_idx[k];
                    let v = as_base_field(csc.vals[k]);
                    if v == F::ZERO {
                        continue;
                    }
                    if last_blk_by_row[row] == blk {
                        row_blocks[cursor[row] - 1].orig.0[local] += v;
                    } else {
                        let idx = cursor[row];
                        cursor[row] += 1;
                        last_blk_by_row[row] = blk;
                        let mut block = [F::ZERO; D];
                        block[local] = v;
                        row_blocks[idx] = RowBlock {
                            blk,
                            bar: Rq([F::ZERO; D]),
                            orig: Rq(block),
                        };
                    }
                }
            }

            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            row_blocks.par_iter_mut().for_each(|rb| {
                rb.bar.0 = superneo_bar_block(rb.orig.0);
            });
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            for rb in &mut row_blocks {
                rb.bar.0 = superneo_bar_block(rb.orig.0);
            }

            SuperneoMatrixCache {
                rows,
                cols,
                row_offsets,
                row_blocks,
                identity: false,
            }
        }
    }
}

/// Build a cached SuperNeo row-lift representation when shape-compatible.
///
/// Returns `None` when the CCS width is not compatible with the `D`-sized block lift.
pub fn build_superneo_eval_cache<Ff>(s: &CcsStructure<Ff>) -> Option<SuperneoEvalCache>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if !is_superneo_compatible_shape(s.m) {
        return None;
    }
    let mats = s.matrices.iter().map(build_matrix_cache).collect();
    Some(SuperneoEvalCache { mats })
}

/// Evaluate `\tilde{(M_j z)}(r)` for all matrices using cached SuperNeo rows.
pub fn eval_all_mats_cached_with_blocks(
    cache: &SuperneoEvalCache,
    z_blocks: &SuperneoZBlocks,
    chi_r: &[K],
    n_eff: usize,
) -> Vec<K> {
    let mut out = Vec::with_capacity(cache.mats.len());
    for m in &cache.mats {
        out.push(m.eval_mle_with_blocks(z_blocks, chi_r, n_eff));
    }
    out
}

/// Evaluate `\tilde{(M_j z)}(r)` for all matrices using cached SuperNeo rows.
pub fn eval_all_mats_cached(cache: &SuperneoEvalCache, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K> {
    let z_blocks = SuperneoZBlocks::from_z(z);
    eval_all_mats_cached_with_blocks(cache, &z_blocks, chi_r, n_eff)
}

/// Evaluate `\widetilde{(M_j z)}(r)` for all matrices in ring-coefficient form.
pub fn eval_all_mats_ring_cached_with_blocks(
    cache: &SuperneoEvalCache,
    z_blocks: &SuperneoZBlocks,
    chi_r: &[K],
    n_eff: usize,
) -> Vec<[K; D]> {
    if z_blocks.imag_all_zero {
        let (chi_re, chi_im) = split_chi_coeffs(chi_r, n_eff);
        return eval_all_mats_ring_cached_with_split_chi(cache, z_blocks, &chi_re, &chi_im, n_eff);
    }

    let mut out = Vec::with_capacity(cache.mats.len());
    for m in &cache.mats {
        out.push(m.eval_mle_ring_with_blocks(z_blocks, chi_r, n_eff));
    }
    out
}

pub fn eval_all_mats_ring_cached_with_split_chi(
    cache: &SuperneoEvalCache,
    z_blocks: &SuperneoZBlocks,
    chi_re: &[F],
    chi_im: &[F],
    n_eff: usize,
) -> Vec<[K; D]> {
    let mut out = Vec::with_capacity(cache.mats.len());
    let mut scratch = RingEvalScratch::new(z_blocks.re.len());
    for m in &cache.mats {
        out.push(m.eval_mle_ring_with_blocks_split_chi_scratch(z_blocks, chi_re, chi_im, n_eff, &mut scratch));
    }
    out
}

/// Evaluate `\widetilde{(M_j z)}(r)` for all matrices in ring-coefficient form.
pub fn eval_all_mats_ring_cached(cache: &SuperneoEvalCache, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<[K; D]> {
    let z_blocks = SuperneoZBlocks::from_z(z);
    eval_all_mats_ring_cached_with_blocks(cache, &z_blocks, chi_r, n_eff)
}

/// Row dot-product via on-the-fly SuperNeo lift from an original matrix row.
///
/// For each `D`-sized block of row coefficients `a`, this computes `bar(a)` and accumulates
/// `ct(cf_inv(bar(a)) * cf_inv(z_block))` (both real and imaginary channels).
pub fn superneo_row_dot_from_original<Ff>(mat: &CcsMatrix<Ff>, row: usize, z: &[K]) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    assert_eq!(
        mat.cols(),
        z.len(),
        "superneo_row_dot_from_original: column/vector length mismatch"
    );
    if row >= mat.rows() {
        return K::ZERO;
    }

    let blocks = z.len().div_ceil(D);
    let mut acc_re = F::ZERO;
    let mut acc_im = F::ZERO;

    for blk in 0..blocks {
        let base = blk * D;
        let mut a = [F::ZERO; D];
        let mut z_re = [F::ZERO; D];
        let mut z_im = [F::ZERO; D];

        for i in 0..D {
            a[i] = as_base_field(matrix_entry(mat, row, base + i));
            if base + i < z.len() {
                let [re, im] = z[base + i].as_coeffs();
                z_re[i] = re;
                z_im[i] = im;
            }
        }

        let a_bar = superneo_bar_block(a);
        let a_ring = Rq(a_bar);
        acc_re += ct(&a_ring.mul(&Rq(z_re)));
        acc_im += ct(&a_ring.mul(&Rq(z_im)));
    }

    K::from_coeffs([acc_re, acc_im])
}

/// Evaluate `\tilde{(M z)}(r)` using transformed matrix rows and `ct` products.
///
/// `mat_bar` must be the SuperNeo transform of the original matrix `M`.
pub fn eval_mle_transformed_matrix(mat_bar: &CcsMatrix<F>, z: &[K], chi_r: &[K], n_eff: usize) -> K {
    let row_cap = min(min(mat_bar.rows(), n_eff), chi_r.len());
    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * superneo_row_dot_transformed_matrix(mat_bar, row, z);
    }
    acc
}

/// Evaluate `\tilde{(M z)}(r)` by lifting original rows through the SuperNeo `bar` transform.
pub fn eval_mle_superneo_from_original<Ff>(mat: &CcsMatrix<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let row_cap = min(min(mat.rows(), n_eff), chi_r.len());
    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(row_cap).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * superneo_row_dot_from_original(mat, row, z);
    }
    acc
}

/// Evaluate `\tilde{(M z)}(r)` directly from `M` (baseline path).
pub fn eval_mle_direct_matrix<Ff>(mat: &CcsMatrix<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> K
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    assert_eq!(
        mat.cols(),
        z.len(),
        "eval_mle_direct_matrix: column/vector length mismatch"
    );
    let row_cap = min(mat.rows(), n_eff);
    let mut mz = vec![K::ZERO; row_cap];
    mat.add_mul_into(z, &mut mz, row_cap);

    let mut acc = K::ZERO;
    for (row, &w) in chi_r.iter().take(min(row_cap, chi_r.len())).enumerate() {
        if w == K::ZERO {
            continue;
        }
        acc += w * mz[row];
    }
    acc
}

#[inline]
pub fn is_superneo_compatible_shape(cols: usize) -> bool {
    cols > 0
}

/// Default heuristic for enabling cached SuperNeo evaluators in hot paths.
///
/// SuperNeo-only mode uses one canonical full-coefficient evaluator path, so enable
/// the cache whenever the CCS shape is compatible.
pub fn should_enable_superneo_cache_default<Ff>(s: &CcsStructure<Ff>, _b: u32) -> bool {
    is_superneo_compatible_shape(s.m) && !s.matrices.is_empty()
}

/// Evaluate all transformed matrices in `s_bar` at `(z, r)`.
pub fn eval_all_mats_transformed(s_bar: &CcsStructure<F>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K> {
    let mut out = Vec::with_capacity(s_bar.matrices.len());
    for m in &s_bar.matrices {
        out.push(eval_mle_transformed_matrix(m, z, chi_r, n_eff));
    }
    out
}

/// Evaluate all original matrices in `s` at `(z, r)` (baseline path).
pub fn eval_all_mats_direct<Ff>(s: &CcsStructure<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut out = Vec::with_capacity(s.matrices.len());
    for m in &s.matrices {
        out.push(eval_mle_direct_matrix(m, z, chi_r, n_eff));
    }
    out
}

/// Evaluate all matrices in `s` at `(z, r)` using SuperNeo row lifting.
pub fn eval_all_mats_superneo<Ff>(s: &CcsStructure<Ff>, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy + Send + Sync,
    K: From<Ff>,
{
    let mut out = Vec::with_capacity(s.matrices.len());
    for m in &s.matrices {
        out.push(eval_mle_superneo_from_original(m, z, chi_r, n_eff));
    }
    out
}
