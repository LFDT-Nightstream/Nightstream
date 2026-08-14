use core::cmp::min;
use neo_ccs::{CcsMatrix, Mat, SeededPhi81LinearBlock};
use neo_math::{ct, KExtensions, Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

mod artifact;
mod authority;
mod baseline;
mod cache;
mod compact;
mod digit;
mod geometric;
mod matrix_cache_impl;
mod parallel;
mod row_block;
mod seeded;
mod weighted;
mod weighted_table;

pub use artifact::{
    SuperneoCacheArtifactError, SuperneoCacheArtifactLimits, SuperneoCacheArtifactReceipt,
    VerifiedSuperneoCacheArtifact,
};
pub use authority::{check_ccs_relation_zero_cached, SuperneoCachedRelationError};
pub use baseline::{
    eval_all_mats_direct, eval_all_mats_superneo, eval_all_mats_transformed, eval_mle_direct_matrix,
    eval_mle_superneo_from_original, eval_mle_transformed_matrix, is_superneo_compatible_shape,
    should_enable_superneo_cache_default, superneo_row_dot_from_original,
};
pub use cache::build_superneo_eval_cache;
#[doc(hidden)]
pub use compact::{SuperneoCompactDeviceParts, SuperneoCompactRowOffsets};
use digit::{
    accumulate_by_digit_block, accumulate_by_signed_unit_masks, accumulate_pair_by_digit_block,
    accumulate_pair_by_signed_unit_masks, mul_by_digit_block, mul_by_signed_unit_masks,
};
use row_block::{CompactRowBlock, DenseRowBlock, COMPACT_SINGLE_BLOCK_MASK};
use weighted::{weighted_projection_basis_forms_from_k, weighted_projection_form_from_orig};

/// The per-lane weighted projection basis forms `(re, im)` derived from the
/// chi-alpha weights. Device backends use the same forms to build their row
/// tables without materializing the CPU table first.
pub fn weighted_projection_basis_forms(weights: &[K; D]) -> ([Rq; D], [Rq; D]) {
    weighted_projection_basis_forms_from_k(weights)
}

#[inline]
fn matrix_entry<Ff: Field + PrimeCharacteristicRing + Copy>(mat: &CcsMatrix<Ff>, row: usize, col: usize) -> Ff {
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
            let range = csc.column_range(col);
            match csc.row_idx[range.clone()].binary_search(&(row as u32)) {
                Ok(idx) => csc.vals[range.start + idx],
                Err(_) => Ff::ZERO,
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let range = csc.column_range(col);
            let mut value = match csc.row_idx[range.clone()].binary_search(&(row as u32)) {
                Ok(idx) => csc.vals[range.start + idx],
                Err(_) => Ff::ZERO,
            };
            for block in blocks {
                value += block.entry::<Ff>(row, col);
            }
            for run in geometric_runs {
                value += run.entry(row, col);
            }
            value
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("direct matrix access is unavailable for verifier-artifact matrices")
        }
    }
}

/// Row dot-product using a transformed row `bar(a)` and SuperNeo's `ct` product.
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
    Ff: Field + PrimeCharacteristicRing + Copy + Sync,
    K: From<Ff>,
{
    K::from(v).real()
}
#[derive(Clone, Copy, Debug)]
struct RowBlock {
    blk: usize,
    bar: Rq,
    orig: Rq,
}

#[derive(Clone, Debug, Default)]
enum RowOffsetStore {
    #[default]
    Empty,
    U16Chunked {
        chunk_offsets: Vec<u32>,
        local_offsets: Vec<u16>,
    },
    U24(Vec<u8>),
    U32(Vec<u32>),
}

impl RowOffsetStore {
    const CHUNK_ROWS: usize = 256;

    fn from_dense(offsets: Vec<u32>) -> Self {
        if offsets.is_empty() {
            return Self::Empty;
        }
        let mut chunk_offsets = Vec::with_capacity(offsets.len().div_ceil(Self::CHUNK_ROWS));
        let mut local_offsets = Vec::with_capacity(offsets.len());
        let mut fits_u16 = true;
        for chunk in offsets.chunks(Self::CHUNK_ROWS) {
            let base = chunk[0];
            chunk_offsets.push(base);
            for &offset in chunk {
                let Some(local) = offset
                    .checked_sub(base)
                    .and_then(|value| u16::try_from(value).ok())
                else {
                    fits_u16 = false;
                    break;
                };
                local_offsets.push(local);
            }
            if !fits_u16 {
                break;
            }
        }
        if fits_u16 {
            return Self::U16Chunked {
                chunk_offsets,
                local_offsets,
            };
        }
        if offsets.last().copied().unwrap_or(0) <= 0x00ff_ffff {
            let mut packed = Vec::with_capacity(offsets.len() * 3);
            for offset in offsets {
                let bytes = offset.to_le_bytes();
                packed.extend_from_slice(&bytes[..3]);
            }
            Self::U24(packed)
        } else {
            Self::U32(offsets)
        }
    }

    #[inline]
    fn get(&self, index: usize) -> u32 {
        match self {
            Self::Empty => 0,
            Self::U16Chunked {
                chunk_offsets,
                local_offsets,
            } => {
                let chunk = index / Self::CHUNK_ROWS;
                chunk_offsets[chunk] + u32::from(local_offsets[index])
            }
            Self::U24(bytes) => {
                let start = index * 3;
                u32::from_le_bytes([bytes[start], bytes[start + 1], bytes[start + 2], 0])
            }
            Self::U32(offsets) => offsets[index],
        }
    }

    #[inline]
    fn range(&self, row: usize) -> core::ops::Range<usize> {
        self.get(row) as usize..self.get(row + 1) as usize
    }

    fn take_dense(&mut self, len: usize) -> Vec<u32> {
        match core::mem::take(self) {
            Self::Empty => Vec::new(),
            Self::U16Chunked {
                chunk_offsets,
                local_offsets,
            } => local_offsets
                .into_iter()
                .enumerate()
                .map(|(index, local)| chunk_offsets[index / Self::CHUNK_ROWS] + u32::from(local))
                .collect(),
            Self::U24(bytes) => (0..len)
                .map(|index| {
                    let start = index * 3;
                    u32::from_le_bytes([bytes[start], bytes[start + 1], bytes[start + 2], 0])
                })
                .collect(),
            Self::U32(offsets) => offsets,
        }
    }

    fn compact(&mut self, len: usize) {
        let dense = self.take_dense(len);
        *self = Self::from_dense(dense);
    }

    #[cfg(feature = "perf-timers")]
    fn compact_bytes(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::U16Chunked {
                chunk_offsets,
                local_offsets,
            } => chunk_offsets.len() * core::mem::size_of::<u32>() + local_offsets.len() * core::mem::size_of::<u16>(),
            Self::U24(bytes) => bytes.len(),
            Self::U32(offsets) => offsets.len() * core::mem::size_of::<u32>(),
        }
    }
}

#[derive(Clone, Debug)]
enum DenseBlockStore {
    Building(Vec<Rq>),
    Compact {
        offsets: Vec<u32>,
        locals: Vec<u8>,
        coefficients: Vec<F>,
    },
}

impl DenseBlockStore {
    fn finish(&mut self) {
        let DenseBlockStore::Building(blocks) = core::mem::replace(self, DenseBlockStore::Building(Vec::new())) else {
            return;
        };
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        let mut locals = Vec::new();
        let mut coefficients = Vec::new();
        offsets.push(0);
        for block in blocks {
            for (local, coefficient) in block.0.into_iter().enumerate() {
                if coefficient != F::ZERO {
                    locals.push(local as u8);
                    coefficients.push(coefficient);
                }
            }
            offsets.push(u32::try_from(locals.len()).expect("dense-block coefficient count exceeds u32"));
        }
        *self = DenseBlockStore::Compact {
            offsets,
            locals,
            coefficients,
        };
    }

    fn expanded(&self, index: usize) -> Rq {
        match self {
            DenseBlockStore::Building(blocks) => blocks[index],
            DenseBlockStore::Compact {
                offsets,
                locals,
                coefficients,
            } => {
                let mut out = Rq([F::ZERO; D]);
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                for entry in start..end {
                    out.0[locals[entry] as usize] = coefficients[entry];
                }
                out
            }
        }
    }

    #[cfg(feature = "perf-timers")]
    fn compact_bytes(&self) -> usize {
        match self {
            DenseBlockStore::Building(blocks) => blocks.len() * core::mem::size_of::<Rq>(),
            DenseBlockStore::Compact {
                offsets,
                locals,
                coefficients,
            } => {
                offsets.len() * core::mem::size_of::<u32>()
                    + locals.len() * core::mem::size_of::<u8>()
                    + coefficients.len() * core::mem::size_of::<F>()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuperneoMatrixCache {
    rows: usize,
    cols: usize,
    row_offsets: RowOffsetStore,
    row_blocks: Vec<CompactRowBlock>,
    dense_row_blocks: Vec<DenseRowBlock>,
    dense_orig: DenseBlockStore,
    geometric_row_offsets: RowOffsetStore,
    geometric_runs: Vec<[u64; 3]>,
    identity: bool,
    seeded_phi81_blocks: Vec<SeededPhi81LinearBlock>,
}

impl SuperneoMatrixCache {
    /// CSR shape of the explicit bar-transformed entries. Seeded Phi81 blocks
    /// remain represented separately and are not included in this entry view.
    pub fn bar_shape(&self) -> (usize, usize, Vec<usize>, usize) {
        if self.identity {
            return (self.rows, self.cols, (0..=self.rows).collect(), self.rows);
        }
        (
            self.rows,
            self.cols,
            (0..=self.rows)
                .map(|row| self.row_offsets.get(row) as usize)
                .collect(),
            self.row_blocks.len(),
        )
    }

    /// Explicit entry `i` in row order: `(block, bar ring element)`.
    pub fn bar_entry(&self, i: usize) -> (usize, Rq) {
        if self.identity {
            let mut orig = [F::ZERO; D];
            orig[i % D] = F::ONE;
            return (i / D, Rq(neo_math::superneo_bar_block(orig)));
        }
        let row_block = self.expanded_block(self.row_blocks[i]);
        (row_block.blk, row_block.bar)
    }

    /// Explicit entry `i` in row order: `(block, original ring row)`.
    pub fn orig_entry(&self, i: usize) -> (usize, Rq) {
        if self.identity {
            let mut orig = [F::ZERO; D];
            orig[i % D] = F::ONE;
            return (i / D, Rq(orig));
        }
        let row_block = self.expanded_block(self.row_blocks[i]);
        (row_block.blk, row_block.orig)
    }
}
#[derive(Clone, Copy, Debug)]
struct WeightedRowBlock {
    blk: usize,
    re_form: Rq,
    im_form: Rq,
}
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
    re_nonzero: bool,
    im_nonzero: bool,
}

impl SuperneoRingLinearForm {
    /// Dense (re, im) coefficient planes over all column blocks, laid out as
    /// `[block][D]`. CUDA evaluates this form as a flat ring mat-vec rather
    /// than walking the sparse entry list.
    pub fn to_dense_block_coeffs(&self) -> (Vec<F>, Vec<F>) {
        let blocks = self.cols.div_ceil(D);
        let mut re = vec![F::ZERO; blocks * D];
        let mut im = vec![F::ZERO; blocks * D];
        for entry in &self.entries {
            let base = entry.blk * D;
            re[base..base + D].copy_from_slice(&entry.re_form.0);
            im[base..base + D].copy_from_slice(&entry.im_form.0);
        }
        (re, im)
    }

    #[inline]
    pub fn eval_real_z_blocks(&self, z_blocks: &SuperneoZBlocks) -> [K; D] {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoRingLinearForm::eval_real_z_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoRingLinearForm::eval_real_z_blocks expects real-only witness blocks"
        );

        let mut out_re = [F::ZERO; D];
        let mut out_im = [F::ZERO; D];
        for entry in &self.entries {
            if !z_blocks.real_nonzero(entry.blk) {
                continue;
            }
            match (entry.re_nonzero, entry.im_nonzero) {
                (true, true) => {
                    z_blocks.accumulate_real_pair(&mut out_re, &mut out_im, &entry.re_form, &entry.im_form, entry.blk);
                }
                (true, false) => z_blocks.accumulate_real(&mut out_re, &entry.re_form, entry.blk),
                (false, true) => z_blocks.accumulate_real(&mut out_im, &entry.im_form, entry.blk),
                (false, false) => {}
            }
        }

        let mut out = [K::ZERO; D];
        for i in 0..D {
            out[i] = K::from_coeffs([out_re[i], out_im[i]]);
        }
        out
    }
}

/// Evaluate independent ring-linear forms against the same packed witness.
pub fn eval_ring_linear_forms_real_z_blocks(
    forms: &[SuperneoRingLinearForm],
    z_blocks: &SuperneoZBlocks,
) -> Vec<[K; D]> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if forms.len() > 1 && rayon::current_num_threads() > 1 {
            const CHUNK_ENTRIES: usize = 2048;
            let total_entries = forms.iter().map(|form| form.entries.len()).sum::<usize>();
            if total_entries >= CHUNK_ENTRIES * 4 {
                let mut chunks = Vec::with_capacity(total_entries.div_ceil(CHUNK_ENTRIES));
                for (form_idx, form) in forms.iter().enumerate() {
                    for start in (0..form.entries.len()).step_by(CHUNK_ENTRIES) {
                        let end = core::cmp::min(start + CHUNK_ENTRIES, form.entries.len());
                        chunks.push((form_idx, start, end));
                    }
                }
                let partials: Vec<(usize, [F; D], [F; D])> = chunks
                    .par_iter()
                    .map(|&(form_idx, start, end)| {
                        let mut out_re = [F::ZERO; D];
                        let mut out_im = [F::ZERO; D];
                        for entry in &forms[form_idx].entries[start..end] {
                            if !z_blocks.real_nonzero(entry.blk) {
                                continue;
                            }
                            match (entry.re_nonzero, entry.im_nonzero) {
                                (true, true) => z_blocks.accumulate_real_pair(
                                    &mut out_re,
                                    &mut out_im,
                                    &entry.re_form,
                                    &entry.im_form,
                                    entry.blk,
                                ),
                                (true, false) => z_blocks.accumulate_real(&mut out_re, &entry.re_form, entry.blk),
                                (false, true) => z_blocks.accumulate_real(&mut out_im, &entry.im_form, entry.blk),
                                (false, false) => {}
                            }
                        }
                        (form_idx, out_re, out_im)
                    })
                    .collect();
                let mut acc_re = vec![[F::ZERO; D]; forms.len()];
                let mut acc_im = vec![[F::ZERO; D]; forms.len()];
                for (form_idx, out_re, out_im) in partials {
                    for i in 0..D {
                        acc_re[form_idx][i] += out_re[i];
                        acc_im[form_idx][i] += out_im[i];
                    }
                }
                return acc_re
                    .into_iter()
                    .zip(acc_im)
                    .map(|(re, im)| {
                        let mut out = [K::ZERO; D];
                        for i in 0..D {
                            out[i] = K::from_coeffs([re[i], im[i]]);
                        }
                        out
                    })
                    .collect();
            }
            if rayon::current_thread_index().is_none() {
                return forms
                    .par_iter()
                    .map(|form| form.eval_real_z_blocks(z_blocks))
                    .collect();
            }
        }
    }
    forms
        .iter()
        .map(|form| form.eval_real_z_blocks(z_blocks))
        .collect()
}

#[derive(Clone, Debug)]
enum RealBlockStorage {
    Zero {
        len: usize,
    },
    Dense {
        blocks: Vec<Rq>,
        nonzero: Vec<bool>,
    },
    SignedUnit {
        positive: Vec<u64>,
        negative: Vec<u64>,
    },
}

impl RealBlockStorage {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Zero { len } => *len,
            Self::Dense { blocks, .. } => blocks.len(),
            Self::SignedUnit { positive, .. } => positive.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SuperneoZBlocks {
    re: RealBlockStorage,
    im: Vec<Rq>,
    im_nonzero: Vec<bool>,
    imag_all_zero: bool,
}

impl SuperneoZBlocks {
    #[inline]
    pub fn with_block_len(blocks: usize) -> Self {
        Self {
            re: RealBlockStorage::Zero { len: blocks },
            im: Vec::new(),
            im_nonzero: vec![false; blocks],
            imag_all_zero: true,
        }
    }

    #[inline]
    pub fn from_z(z: &[K]) -> Self {
        let blocks = z.len().div_ceil(D);
        let mut re = Vec::with_capacity(blocks);
        let mut im = Vec::with_capacity(blocks);
        let mut re_nonzero = Vec::with_capacity(blocks);
        let mut im_nonzero = Vec::with_capacity(blocks);
        let mut imag_all_zero = true;
        for blk in 0..blocks {
            let base = blk * D;
            let mut zr = [F::ZERO; D];
            let mut zi = [F::ZERO; D];
            let mut re_block_nonzero = false;
            let mut im_block_nonzero = false;
            for i in 0..D {
                if base + i < z.len() {
                    let [r, im_part] = z[base + i].as_coeffs();
                    zr[i] = r;
                    zi[i] = im_part;
                    re_block_nonzero |= r != F::ZERO;
                    im_block_nonzero |= im_part != F::ZERO;
                    imag_all_zero &= im_part == F::ZERO;
                }
            }
            re.push(Rq(zr));
            im.push(Rq(zi));
            re_nonzero.push(re_block_nonzero);
            im_nonzero.push(im_block_nonzero);
        }
        Self {
            re: RealBlockStorage::Dense {
                blocks: re,
                nonzero: re_nonzero,
            },
            im,
            im_nonzero,
            imag_all_zero,
        }
    }

    #[inline]
    pub fn linear_combination_real(blocks: &[Self], coeffs: &[K]) -> Self {
        assert_eq!(
            blocks.len(),
            coeffs.len(),
            "SuperneoZBlocks::linear_combination_real: block/coeff length mismatch"
        );
        let Some(first) = blocks.first() else {
            return Self {
                re: RealBlockStorage::Zero { len: 0 },
                im: Vec::new(),
                im_nonzero: Vec::new(),
                imag_all_zero: true,
            };
        };
        let block_count = first.block_len();
        let mut re = vec![Rq([F::ZERO; D]); block_count];
        let mut im = vec![Rq([F::ZERO; D]); block_count];
        let mut imag_all_zero = true;

        for (z, &coeff) in blocks.iter().zip(coeffs.iter()) {
            assert_eq!(
                z.block_len(),
                block_count,
                "SuperneoZBlocks::linear_combination_real: inconsistent block count"
            );
            debug_assert!(
                z.imag_all_zero,
                "SuperneoZBlocks::linear_combination_real expects real packed witnesses"
            );
            if coeff == K::ZERO {
                continue;
            }
            let [coeff_re, coeff_im] = coeff.as_coeffs();
            for blk in 0..block_count {
                for lane in 0..D {
                    let v = z.real_coefficient(blk, lane);
                    if v == F::ZERO {
                        continue;
                    }
                    re[blk].0[lane] += v * coeff_re;
                    if coeff_im != F::ZERO {
                        im[blk].0[lane] += v * coeff_im;
                        imag_all_zero = false;
                    }
                }
            }
        }
        if imag_all_zero {
            im.clear();
        }
        let re_nonzero = re.iter().map(|block| !is_all_zero(&block.0)).collect();
        let im_nonzero = if imag_all_zero {
            vec![false; block_count]
        } else {
            im.iter().map(|block| !is_all_zero(&block.0)).collect()
        };
        Self {
            re: RealBlockStorage::Dense {
                blocks: re,
                nonzero: re_nonzero,
            },
            im,
            im_nonzero,
            imag_all_zero,
        }
    }

    #[inline]
    pub fn from_witness_mat<Ff>(z: &Mat<Ff>, expected_m: usize) -> Result<Self, crate::PiCcsError>
    where
        Ff: Field + PrimeCharacteristicRing + Copy,
        K: From<Ff>,
    {
        crate::common::validate_superneo_witness_mat(z, expected_m)?;
        let blocks = expected_m.div_ceil(D);
        if z.virtual_constant_value()
            .is_some_and(|value| *value == Ff::ZERO)
        {
            return Ok(Self::with_block_len(blocks));
        }
        if let Some((positive, negative)) = z.packed_signed_unit_column_masks() {
            debug_assert_eq!(positive.len(), blocks);
            debug_assert_eq!(negative.len(), blocks);
            return Ok(Self {
                re: RealBlockStorage::SignedUnit {
                    positive: positive.to_vec(),
                    negative: negative.to_vec(),
                },
                im: Vec::new(),
                im_nonzero: vec![false; blocks],
                imag_all_zero: true,
            });
        }
        let mut positive = Vec::with_capacity(blocks);
        let mut negative = Vec::with_capacity(blocks);
        let neg_one = F::ZERO - F::ONE;
        let mut signed_unit = true;
        for blk in 0..blocks {
            let mut positive_mask = 0u64;
            let mut negative_mask = 0u64;
            for i in 0..D {
                let value = as_base_field(z[(i, blk)]);
                if value == F::ONE {
                    positive_mask |= 1u64 << i;
                } else if value == neg_one {
                    negative_mask |= 1u64 << i;
                } else if value != F::ZERO {
                    signed_unit = false;
                    break;
                }
            }
            if !signed_unit {
                break;
            }
            positive.push(positive_mask);
            negative.push(negative_mask);
        }
        let re = if signed_unit {
            RealBlockStorage::SignedUnit { positive, negative }
        } else {
            let mut dense = Vec::with_capacity(blocks);
            let mut nonzero = Vec::with_capacity(blocks);
            for blk in 0..blocks {
                let mut zr = [F::ZERO; D];
                let mut block_nonzero = false;
                for (i, cell) in zr.iter_mut().enumerate() {
                    *cell = as_base_field(z[(i, blk)]);
                    block_nonzero |= *cell != F::ZERO;
                }
                dense.push(Rq(zr));
                nonzero.push(block_nonzero);
            }
            RealBlockStorage::Dense { blocks: dense, nonzero }
        };
        Ok(Self {
            re,
            im: Vec::new(),
            im_nonzero: vec![false; blocks],
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
        let mut re_nonzero = Vec::with_capacity(blocks);
        for blk in 0..blocks {
            let base = blk * D;
            let mut zr = [F::ZERO; D];
            let mut block_nonzero = false;
            for i in 0..D {
                if base + i < row.len() {
                    zr[i] = as_base_field(row[base + i]);
                    block_nonzero |= zr[i] != F::ZERO;
                }
            }
            re.push(Rq(zr));
            re_nonzero.push(block_nonzero);
        }
        Self {
            re: RealBlockStorage::Dense {
                blocks: re,
                nonzero: re_nonzero,
            },
            im: Vec::new(),
            im_nonzero: vec![false; blocks],
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
        let mut re = vec![Rq([F::ZERO; D]); blocks];
        let mut re_nonzero = vec![false; blocks];
        self.imag_all_zero = true;
        self.im.clear();
        if self.im_nonzero.len() != blocks {
            self.im_nonzero.resize(blocks, false);
        }
        for blk in 0..blocks {
            let base = blk * D;
            let mut block_nonzero = false;
            for i in 0..D {
                re[blk].0[i] = if base + i < row.len() {
                    as_base_field(row[base + i])
                } else {
                    F::ZERO
                };
                block_nonzero |= re[blk].0[i] != F::ZERO;
            }
            re_nonzero[blk] = block_nonzero;
            self.im_nonzero[blk] = false;
        }
        self.re = RealBlockStorage::Dense {
            blocks: re,
            nonzero: re_nonzero,
        };
    }

    #[inline]
    pub fn imag_all_zero(&self) -> bool {
        self.imag_all_zero
    }

    /// Real coefficient plane as canonical words in `[block][D]` layout.
    pub fn re_plane_words(&self) -> Vec<u64> {
        let mut words = vec![0; self.re.len() * D];
        for block in 0..self.re.len() {
            for lane in 0..D {
                words[block * D + lane] = self.real_coefficient(block, lane).as_canonical_u64();
            }
        }
        words
    }

    /// Imaginary coefficient plane in `[block][D]` layout.
    pub fn im_plane_words(&self) -> Vec<u64> {
        if self.imag_all_zero {
            return vec![0; self.re.len() * D];
        }
        Self::plane_words(&self.im)
    }

    fn plane_words(rings: &[Rq]) -> Vec<u64> {
        let mut words = vec![0; rings.len() * D];
        for (block, ring) in rings.iter().enumerate() {
            for (lane, coefficient) in ring.0.iter().enumerate() {
                words[block * D + lane] = coefficient.as_canonical_u64();
            }
        }
        words
    }

    #[inline]
    pub(crate) fn block_nonzero(&self, blk: usize) -> bool {
        self.real_nonzero(blk) || (!self.imag_all_zero && self.im_nonzero[blk])
    }

    #[inline]
    fn block_len(&self) -> usize {
        self.re.len()
    }

    #[inline]
    fn real_nonzero(&self, block: usize) -> bool {
        match &self.re {
            RealBlockStorage::Zero { .. } => false,
            RealBlockStorage::Dense { nonzero, .. } => nonzero[block],
            RealBlockStorage::SignedUnit { positive, negative } => (positive[block] | negative[block]) != 0,
        }
    }

    #[inline]
    fn real_coefficient(&self, block: usize, local: usize) -> F {
        match &self.re {
            RealBlockStorage::Zero { .. } => F::ZERO,
            RealBlockStorage::Dense { blocks, .. } => blocks[block].0[local],
            RealBlockStorage::SignedUnit { positive, negative } => {
                let bit = 1u64 << local;
                if positive[block] & bit != 0 {
                    F::ONE
                } else if negative[block] & bit != 0 {
                    F::ZERO - F::ONE
                } else {
                    F::ZERO
                }
            }
        }
    }

    #[inline]
    fn real_dot(&self, form: &Rq, block: usize) -> F {
        match &self.re {
            RealBlockStorage::Zero { .. } => F::ZERO,
            RealBlockStorage::Dense { blocks, .. } => coeff_dot(form, &blocks[block]),
            RealBlockStorage::SignedUnit { positive, negative } => {
                signed_unit_dot(form, positive[block], negative[block])
            }
        }
    }

    #[inline]
    fn real_mul(&self, form: &Rq, block: usize) -> Rq {
        match &self.re {
            RealBlockStorage::Zero { .. } => Rq::zero(),
            RealBlockStorage::Dense { blocks, .. } => mul_by_digit_block(form, &blocks[block]),
            RealBlockStorage::SignedUnit { positive, negative } => {
                mul_by_signed_unit_masks(form, positive[block], negative[block])
            }
        }
    }

    #[inline]
    fn accumulate_real(&self, out: &mut [F; D], form: &Rq, block: usize) {
        match &self.re {
            RealBlockStorage::Zero { .. } => {}
            RealBlockStorage::Dense { blocks, .. } => accumulate_by_digit_block(out, form, &blocks[block]),
            RealBlockStorage::SignedUnit { positive, negative } => {
                accumulate_by_signed_unit_masks(out, form, positive[block], negative[block]);
            }
        }
    }

    #[inline]
    fn accumulate_real_pair(&self, out_re: &mut [F; D], out_im: &mut [F; D], re_form: &Rq, im_form: &Rq, block: usize) {
        match &self.re {
            RealBlockStorage::Zero { .. } => {}
            RealBlockStorage::Dense { blocks, .. } => {
                accumulate_pair_by_digit_block(out_re, out_im, re_form, im_form, &blocks[block]);
            }
            RealBlockStorage::SignedUnit { positive, negative } => {
                accumulate_pair_by_signed_unit_masks(
                    out_re,
                    out_im,
                    re_form,
                    im_form,
                    positive[block],
                    negative[block],
                );
            }
        }
    }
}

#[inline]
fn signed_unit_dot(form: &Rq, mut positive: u64, mut negative: u64) -> F {
    let mut out = F::ZERO;
    while positive != 0 {
        let index = positive.trailing_zeros() as usize;
        out += form.0[index];
        positive &= positive - 1;
    }
    while negative != 0 {
        let index = negative.trailing_zeros() as usize;
        out -= form.0[index];
        negative &= negative - 1;
    }
    out
}

impl SuperneoWeightedMatrixCache {
    #[inline]
    pub fn row_dot_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> K {
        if z_blocks.imag_all_zero {
            return self.row_dot_real_with_blocks(row, z_blocks);
        }
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoWeightedMatrixCache::row_dot_with_blocks: block count mismatch"
        );
        debug_assert_eq!(
            z_blocks.block_len(),
            z_blocks.im.len(),
            "SuperneoWeightedMatrixCache::row_dot_with_blocks: complex block length mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc = K::ZERO;
        let extension_generator = K::from_coeffs([F::ZERO, F::ONE]);
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        for rb in &self.row_blocks[start..end] {
            if !z_blocks.block_nonzero(rb.blk) {
                continue;
            }
            let (rr, ir) = if z_blocks.real_nonzero(rb.blk) {
                (
                    z_blocks.real_dot(&rb.re_form, rb.blk),
                    z_blocks.real_dot(&rb.im_form, rb.blk),
                )
            } else {
                (F::ZERO, F::ZERO)
            };
            let (ri, ii) = if !z_blocks.imag_all_zero && z_blocks.im_nonzero[rb.blk] {
                let z_im = &z_blocks.im[rb.blk];
                (coeff_dot(&rb.re_form, z_im), coeff_dot(&rb.im_form, z_im))
            } else {
                (F::ZERO, F::ZERO)
            };
            acc += K::from_coeffs([rr, ir]) + extension_generator * K::from_coeffs([ri, ii]);
        }
        acc
    }

    #[inline]
    pub fn row_dot_real_with_blocks(&self, row: usize, z_blocks: &SuperneoZBlocks) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
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
            if !z_blocks.real_nonzero(rb.blk) {
                continue;
            }
            acc_re += z_blocks.real_dot(&rb.re_form, rb.blk);
            acc_im += z_blocks.real_dot(&rb.im_form, rb.blk);
        }
        K::from_coeffs([acc_re, acc_im])
    }
}

/// Cached SuperNeo row-lifted representation for all CCS matrices.
#[derive(Clone, Debug)]
pub struct SuperneoEvalCache {
    mats: Vec<SuperneoMatrixCache>,
    explicit_matrix_masks: Option<Vec<u16>>,
}

impl SuperneoEvalCache {
    #[inline]
    pub fn matrix(&self, j: usize) -> Option<&SuperneoMatrixCache> {
        self.mats.get(j)
    }

    /// Per-matrix explicit bar caches, in CCS matrix order.
    pub fn matrix_caches(&self) -> &[SuperneoMatrixCache] {
        &self.mats
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
        let block_count = self
            .mats
            .iter()
            .map(|matrix| matrix.cols.div_ceil(D))
            .max()
            .unwrap_or(0);
        let mut scratch = RingEvalScratch::new(block_count);
        let mut forms = Vec::with_capacity(self.mats.len());
        for matrix in &self.mats {
            forms.push(matrix.build_ring_linear_form_split_chi_with_scratch(&chi_re, &chi_im, n_eff, &mut scratch));
        }
        forms
    }

    /// Evaluate every matrix ring form against real-only packed witnesses while
    /// retaining only one matrix's accumulator scratch at a time.
    #[inline]
    pub fn eval_ring_linear_forms_for_real_z_blocks(
        &self,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[SuperneoZBlocks],
    ) -> Vec<Vec<[K; D]>> {
        let matrix_count = self.mats.len();
        let mut out = vec![vec![[K::ZERO; D]; matrix_count]; witnesses.len()];
        if witnesses.is_empty() || matrix_count == 0 {
            return out;
        }

        let (chi_re, chi_im) = split_chi_coeffs(chi_r, n_eff);
        let block_count = self
            .mats
            .iter()
            .map(|matrix| matrix.cols.div_ceil(D))
            .max()
            .unwrap_or(0);
        let mut scratch = RingEvalScratch::new(block_count);
        for (matrix_index, matrix) in self.mats.iter().enumerate() {
            matrix.accumulate_ring_form_split_chi(&chi_re, &chi_im, n_eff, &mut scratch);
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let matrix_values: Vec<[K; D]> = if witnesses.len() > 1 && rayon::current_num_threads() > 1 {
                witnesses
                    .par_iter()
                    .map(|witness| eval_ring_scratch_real_z_blocks(&scratch, witness))
                    .collect()
            } else {
                witnesses
                    .iter()
                    .map(|witness| eval_ring_scratch_real_z_blocks(&scratch, witness))
                    .collect()
            };
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let matrix_values: Vec<[K; D]> = witnesses
                .iter()
                .map(|witness| eval_ring_scratch_real_z_blocks(&scratch, witness))
                .collect();
            scratch.clear_active();
            for (witness_index, value) in matrix_values.into_iter().enumerate() {
                out[witness_index][matrix_index] = value;
            }
        }
        out
    }

    #[inline]
    pub fn build_weighted_matrix_caches(&self, weights: &[K; D]) -> Vec<SuperneoWeightedMatrixCache> {
        let (basis_re_forms, basis_im_forms) = weighted_projection_basis_forms_from_k(weights);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            self.mats
                .par_iter()
                .map(|m| m.compile_weighted_rows_with_basis(&basis_re_forms, &basis_im_forms))
                .collect()
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            self.mats
                .iter()
                .map(|m| m.compile_weighted_rows_with_basis(&basis_re_forms, &basis_im_forms))
                .collect()
        }
    }
}

#[inline]
fn eval_ring_scratch_real_z_blocks(scratch: &RingEvalScratch, z_blocks: &SuperneoZBlocks) -> [K; D] {
    debug_assert!(
        z_blocks.imag_all_zero,
        "ring scratch evaluation expects real-only witness blocks"
    );
    if let Some(out) = parallel::eval_active_blocks(&scratch.active_blocks, &scratch.agg_re, &scratch.agg_im, z_blocks)
    {
        return out;
    }

    let mut out_re = [F::ZERO; D];
    let mut out_im = [F::ZERO; D];
    for &blk in &scratch.active_blocks {
        if !z_blocks.real_nonzero(blk) {
            continue;
        }
        let re_nonzero = !is_all_zero(&scratch.agg_re[blk].0);
        let im_nonzero = !is_all_zero(&scratch.agg_im[blk].0);
        match (re_nonzero, im_nonzero) {
            (true, true) => z_blocks.accumulate_real_pair(
                &mut out_re,
                &mut out_im,
                &scratch.agg_re[blk],
                &scratch.agg_im[blk],
                blk,
            ),
            (true, false) => z_blocks.accumulate_real(&mut out_re, &scratch.agg_re[blk], blk),
            (false, true) => z_blocks.accumulate_real(&mut out_im, &scratch.agg_im[blk], blk),
            (false, false) => {}
        }
    }
    core::array::from_fn(|index| K::from_coeffs([out_re[index], out_im[index]]))
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
pub(super) fn coeff_dot(lhs: &Rq, rhs: &Rq) -> F {
    let mut acc = F::ZERO;
    for i in 0..D {
        acc += lhs.0[i] * rhs.0[i];
    }
    acc
}

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
pub fn eval_all_mats_cached(cache: &SuperneoEvalCache, z: &[K], chi_r: &[K], n_eff: usize) -> Vec<K> {
    let z_blocks = SuperneoZBlocks::from_z(z);
    eval_all_mats_cached_with_blocks(cache, &z_blocks, chi_r, n_eff)
}
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
    if !z_blocks.imag_all_zero {
        let len = core::cmp::min(chi_re.len(), chi_im.len());
        let chi_r = (0..len)
            .map(|idx| K::from_coeffs([chi_re[idx], chi_im[idx]]))
            .collect::<Vec<_>>();
        return eval_all_mats_ring_cached_with_blocks(cache, z_blocks, &chi_r, n_eff);
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    {
        if cache.mats.len() > 1
            && z_blocks.block_len() >= 1024
            && rayon::current_thread_index().is_none()
            && rayon::current_num_threads() > 1
        {
            return cache
                .mats
                .par_iter()
                .map(|m| {
                    let mut scratch = RingEvalScratch::new(z_blocks.block_len());
                    m.eval_mle_ring_with_blocks_split_chi_scratch(z_blocks, chi_re, chi_im, n_eff, &mut scratch)
                })
                .collect();
        }
    }
    let mut out = Vec::with_capacity(cache.mats.len());
    let mut scratch = RingEvalScratch::new(z_blocks.block_len());
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
