//! Compact explicit-matrix export and seeded-only ring forms for accelerators.

use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::{
    split_chi_coeffs, DenseBlockStore, RealBlockStorage, RingEvalScratch, SuperneoEvalCache, SuperneoMatrixCache,
    SuperneoRingLinearBlock, SuperneoRingLinearForm, SuperneoZBlocks,
};

impl SuperneoZBlocks {
    /// Packed positive/negative masks when the real plane is signed-unit.
    pub fn signed_unit_masks(&self) -> Option<(&[u64], &[u64])> {
        match &self.re {
            RealBlockStorage::SignedUnit { positive, negative } => Some((positive, negative)),
            _ => None,
        }
    }
}

impl SuperneoMatrixCache {
    /// Whether this matrix has a compact seeded Phi81 component.
    pub fn has_compact_seeded_phi81_blocks(&self) -> bool {
        !self.seeded_phi81_blocks.is_empty()
    }

    /// Compact seeded Phi81 blocks retained outside the explicit matrix CSR.
    pub fn compact_seeded_phi81_blocks(&self) -> &[neo_ccs::SeededPhi81LinearBlock] {
        &self.seeded_phi81_blocks
    }

    /// `(rows, scalar columns, identity)` for the explicit matrix component.
    pub fn compact_explicit_shape(&self) -> (usize, usize, bool) {
        (self.rows, self.cols, self.identity)
    }

    /// Number of scalar coefficients in the compact explicit matrix.
    pub fn compact_explicit_coefficient_count(&self) -> usize {
        if self.identity {
            return 0;
        }
        self.row_blocks
            .iter()
            .map(|block| {
                block.single_parts().map_or_else(
                    || self.dense_coefficient_count(block.dense_index().expect("dense compact block")),
                    |_| 1,
                )
            })
            .sum()
    }

    /// Sorted ring-column blocks touched by compact seeded Phi81 maps.
    pub fn compact_seeded_column_blocks(&self) -> Vec<usize> {
        let mut blocks = Vec::new();
        for seeded in &self.seeded_phi81_blocks {
            for &start in seeded.word_starts() {
                let end = start + seeded.word_width();
                blocks.extend(start / D..end.div_ceil(D));
            }
        }
        blocks.sort_unstable();
        blocks.dedup();
        blocks
    }

    /// Visit the nonzero original coefficients in one explicit row.
    ///
    /// Identity matrices are reported by [`Self::compact_explicit_shape`] and
    /// intentionally have no stored coefficients.
    pub fn for_each_compact_explicit_row_coefficient(&self, row: usize, mut visit: impl FnMut(u32, u8, F)) {
        if self.identity || row >= self.rows {
            return;
        }
        for block in self.row_blocks_for(row).iter().copied() {
            if let Some((local, coefficient)) = block.single_parts() {
                visit(block.blk, local as u8, coefficient);
                continue;
            }
            let dense = block.dense_index().expect("dense compact block");
            match &self.dense_orig {
                DenseBlockStore::Building(blocks) => {
                    for (local, &coefficient) in blocks[dense].0.iter().enumerate() {
                        if coefficient != F::ZERO {
                            visit(block.blk, local as u8, coefficient);
                        }
                    }
                }
                DenseBlockStore::Compact {
                    offsets,
                    locals,
                    coefficients,
                } => {
                    for entry in offsets[dense] as usize..offsets[dense + 1] as usize {
                        visit(block.blk, locals[entry], coefficients[entry]);
                    }
                }
            }
        }
    }

    fn dense_coefficient_count(&self, dense: usize) -> usize {
        match &self.dense_orig {
            DenseBlockStore::Building(blocks) => blocks[dense]
                .0
                .iter()
                .filter(|&&coefficient| coefficient != F::ZERO)
                .count(),
            DenseBlockStore::Compact { offsets, .. } => (offsets[dense + 1] - offsets[dense]) as usize,
        }
    }
}

impl SuperneoEvalCache {
    /// Build only compact seeded-Phi81 contributions to `bar(M)^T * chi_r`.
    pub fn build_seeded_ring_linear_forms(&self, chi_r: &[K], n_eff: usize) -> Vec<SuperneoRingLinearForm> {
        let (chi_re, chi_im) = split_chi_coeffs(chi_r, n_eff);
        self.build_seeded_ring_linear_forms_with(|matrix, scratch| {
            matrix.accumulate_seeded_ring_form_split_chi(&chi_re, &chi_im, n_eff, scratch);
        })
    }

    /// Build seeded-only forms directly from the row challenge vector,
    /// evaluating just the rows touched by compact seeded blocks.
    pub fn build_seeded_ring_linear_forms_from_row_challenges(
        &self,
        row_challenges: &[K],
        n_eff: usize,
    ) -> Vec<SuperneoRingLinearForm> {
        self.build_seeded_ring_linear_forms_with(|matrix, scratch| {
            matrix.accumulate_seeded_ring_form_row_challenges(row_challenges, n_eff, scratch);
        })
    }

    fn build_seeded_ring_linear_forms_with(
        &self,
        mut accumulate: impl FnMut(&SuperneoMatrixCache, &mut RingEvalScratch),
    ) -> Vec<SuperneoRingLinearForm> {
        let block_count = self
            .mats
            .iter()
            .map(|matrix| matrix.cols.div_ceil(D))
            .max()
            .unwrap_or(0);
        let mut scratch = RingEvalScratch::new(block_count);
        let mut forms = Vec::with_capacity(self.mats.len());
        for matrix in &self.mats {
            accumulate(matrix, &mut scratch);
            let entries = scratch
                .active_blocks
                .iter()
                .filter_map(|&blk| {
                    let re_form = scratch.agg_re[blk];
                    let im_form = scratch.agg_im[blk];
                    let re_nonzero = re_form.0.iter().any(|&value| value != F::ZERO);
                    let im_nonzero = im_form.0.iter().any(|&value| value != F::ZERO);
                    (re_nonzero || im_nonzero).then_some(SuperneoRingLinearBlock {
                        blk,
                        re_form,
                        im_form,
                        re_nonzero,
                        im_nonzero,
                    })
                })
                .collect();
            scratch.clear_active();
            forms.push(SuperneoRingLinearForm {
                cols: matrix.cols,
                entries,
            });
        }
        forms
    }
}

impl SuperneoRingLinearForm {
    /// Sparse `(block, real coefficients, imaginary coefficients)` export.
    pub fn to_sparse_block_coeffs(&self) -> Vec<(usize, [F; D], [F; D])> {
        self.entries
            .iter()
            .map(|entry| (entry.blk, entry.re_form.0, entry.im_form.0))
            .collect()
    }
}
