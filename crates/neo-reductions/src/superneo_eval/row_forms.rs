//! Row dot products and weighted ring forms of one explicit matrix cache.

use core::cmp::min;

use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::{scratch::ScratchPart, RingEvalScratch, SuperneoMatrixCache, SuperneoZBlocks};

impl SuperneoMatrixCache {
    /// Fill `out[row] = (M * z)[row]` for a real packed witness from the
    /// prebuilt row cache.
    pub fn fill_row_dots_real_with_blocks(&self, out: &mut [K], z_blocks: &SuperneoZBlocks) {
        self.fill_row_dots_real_from(0, out, z_blocks);
    }

    /// Row dots for rows `first_row..first_row + out.len()`, so disjoint row
    /// ranges of one matrix can be filled concurrently.
    pub fn fill_row_dots_real_from(&self, first_row: usize, out: &mut [K], z_blocks: &SuperneoZBlocks) {
        debug_assert!(first_row + out.len() <= self.rows, "row-dot output exceeds matrix rows");
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::fill_row_dots_real_with_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoMatrixCache::fill_row_dots_real_with_blocks expects a real witness"
        );
        let rows = first_row..first_row + out.len();

        if !self.identity && self.row_blocks.is_empty() && self.geometric_runs.is_empty() {
            out.fill(K::ZERO);
            return;
        }

        if self.identity {
            for (row, out_row) in rows.clone().zip(out.iter_mut()) {
                let block = row / D;
                let local = row % D;
                *out_row = if z_blocks.real_nonzero(block) {
                    K::from(z_blocks.real_coefficient(block, local))
                } else {
                    K::ZERO
                };
            }
            return;
        }

        for (row, out_row) in rows.zip(out.iter_mut()) {
            let mut acc = self.geometric_dot_real(row, z_blocks);
            for block in self.row_blocks_for(row).iter().copied() {
                let block_index = self.row_block_index(block);
                if z_blocks.real_nonzero(block_index) {
                    acc += self.compact_dot_real(block, z_blocks, block_index);
                }
            }
            *out_row = K::from(acc);
        }
    }

    /// Base-field form of [`Self::fill_row_dots_real_with_blocks`]. The
    /// initial row tables contain no extension component, so retaining only
    /// this limb halves their peak storage before the first sumcheck fold.
    pub fn fill_row_dots_base_with_blocks(&self, out: &mut [F], z_blocks: &SuperneoZBlocks) {
        debug_assert!(out.len() <= self.rows, "row-dot output exceeds matrix rows");
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::fill_row_dots_base_with_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoMatrixCache::fill_row_dots_base_with_blocks expects a real witness"
        );

        if !self.identity && self.row_blocks.is_empty() && self.geometric_runs.is_empty() {
            out.fill(F::ZERO);
            return;
        }

        if self.identity {
            for (row, out_row) in out.iter_mut().enumerate() {
                let block = row / D;
                let local = row % D;
                *out_row = if z_blocks.real_nonzero(block) {
                    z_blocks.real_coefficient(block, local)
                } else {
                    F::ZERO
                };
            }
            return;
        }

        for (row, out_row) in out.iter_mut().enumerate() {
            let mut acc = self.geometric_dot_real(row, z_blocks);
            for block in self.row_blocks_for(row).iter().copied() {
                let block_index = self.row_block_index(block);
                if z_blocks.real_nonzero(block_index) {
                    acc += self.compact_dot_real(block, z_blocks, block_index);
                }
            }
            *out_row = acc;
        }
    }

    pub(super) fn accumulate_ring_form_split_chi(
        &self,
        chi_re: &[F],
        chi_im: &[F],
        n_eff: usize,
        scratch: &mut RingEvalScratch,
    ) {
        debug_assert_eq!(chi_re.len(), chi_im.len(), "chi coefficient length mismatch");
        debug_assert!(scratch.active_blocks.is_empty(), "ring-form scratch must start empty");
        let row_cap = min(min(self.rows, n_eff), chi_re.len());
        self.accumulate_original_ring_form_with(row_cap, scratch, |row| K::from_coeffs([chi_re[row], chi_im[row]]));

        scratch.bar_active();
    }

    pub(super) fn accumulate_original_ring_form_with(
        &self,
        row_cap: usize,
        scratch: &mut RingEvalScratch,
        weight: impl Fn(usize) -> K,
    ) {
        scratch.ensure_block_count(self.cols.div_ceil(D));
        self.accumulate_original_ring_form_part(row_cap, &mut scratch.part(), weight);
    }

    /// Accumulate the weighted rows and apply `bar`, with one disjoint block
    /// range per worker thread.
    pub(super) fn accumulate_barred_ring_form_parallel(
        &self,
        row_cap: usize,
        scratch: &mut RingEvalScratch,
        weight: impl Fn(usize) -> K + Sync,
    ) {
        scratch.ensure_block_count(self.cols.div_ceil(D));
        scratch.fill_parts(|part| {
            self.accumulate_original_ring_form_part(row_cap, part, &weight);
            part.bar_active();
        });
    }

    /// Accumulate only the entries whose block lies in `part`.
    fn accumulate_original_ring_form_part(
        &self,
        row_cap: usize,
        part: &mut ScratchPart<'_>,
        weight: impl Fn(usize) -> K,
    ) {
        debug_assert!(row_cap <= self.rows);
        if self.row_blocks.is_empty() && self.geometric_runs.is_empty() {
            if self.identity {
                let blocks = part.blocks();
                let start = blocks.start.saturating_mul(D).min(row_cap);
                let end = blocks.end.saturating_mul(D).min(row_cap);
                for row in start..end {
                    let [real, imaginary] = weight(row).as_coeffs();
                    if real != F::ZERO || imaginary != F::ZERO {
                        part.add_coefficient(row / D, row % D, real, imaginary);
                    }
                }
            }
            return;
        }
        // Runs with one shared ratio become start and end events; a single
        // sweep turns them into run sums before any other entry is added.
        let event_ratio = self.event_ratio();
        if let Some(ratio) = event_ratio {
            for row in 0..row_cap {
                let [w_re, w_im] = weight(row).as_coeffs();
                if w_re != F::ZERO || w_im != F::ZERO {
                    self.add_geometric_events(row, w_re, w_im, ratio, part);
                }
            }
            part.resolve_geometric(ratio);
        }
        let blocks = part.blocks();
        for row in 0..row_cap {
            let [w_re, w_im] = weight(row).as_coeffs();
            if w_re == F::ZERO && w_im == F::ZERO {
                continue;
            }
            if self.identity {
                let block = row / D;
                if blocks.contains(&block) {
                    part.add_coefficient(block, row % D, w_re, w_im);
                }
                continue;
            }
            for compact in self.row_blocks_for(row).iter().copied() {
                let block = self.row_block_index(compact);
                if !blocks.contains(&block) {
                    continue;
                }
                if let Some((_, local, coefficient)) = compact.single_parts() {
                    part.add_coefficient(block, local, w_re * coefficient, w_im * coefficient);
                } else {
                    let orig = self.dense_block(self.dense_pattern_index(compact));
                    part.add_scaled(block, &orig, w_re, w_im);
                }
            }
            self.accumulate_geometric_ring_form_row(row, w_re, w_im, event_ratio, part);
        }
    }
}
