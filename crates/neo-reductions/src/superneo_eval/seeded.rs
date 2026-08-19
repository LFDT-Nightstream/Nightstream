//! Exact streaming contractions for compact seeded Phi81 matrix blocks.

use core::cmp::min;
use std::collections::HashMap;

use neo_math::{superneo_bar_block, Rq, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{add_scaled_rq, RingEvalScratch, SuperneoMatrixCache, SuperneoZBlocks};

const SEEDED_WORK_COLUMNS: usize = 1024;

pub(super) fn seeded_work_ranges(block: &neo_ccs::SeededPhi81LinearBlock, output: usize) -> Vec<(usize, usize, usize)> {
    let mut work = Vec::new();
    for chunk in 0..block.chunk_seeds_by_row()[output].len() {
        let chunk_len = block.original_chunk_len(chunk);
        for local_start in (0..chunk_len).step_by(SEEDED_WORK_COLUMNS) {
            work.push((
                chunk,
                local_start,
                core::cmp::min(chunk_len, local_start + SEEDED_WORK_COLUMNS),
            ));
        }
    }
    work
}

impl SuperneoMatrixCache {
    pub(super) fn row_blocks_including_seeded(&self, row: usize) -> Vec<super::RowBlock> {
        let mut out = self.expanded_row_blocks(row);
        for block in &self.seeded_phi81_blocks {
            if row < block.row_start() || row >= block.row_end() {
                continue;
            }
            let mut grouped = std::collections::BTreeMap::<usize, [F; D]>::new();
            block.for_each_row_term::<F, _>(row, |column, coefficient| {
                grouped.entry(column / D).or_insert([F::ZERO; D])[column % D] += coefficient;
            });
            out.extend(
                grouped
                    .into_iter()
                    .map(|(blk, coefficients)| super::RowBlock {
                        blk,
                        bar: Rq(superneo_bar_block(coefficients)),
                        orig: Rq(coefficients),
                    }),
            );
        }
        out
    }

    /// Fill `out[row] = (M * z)[row]` for a real packed witness.
    ///
    /// Explicit CSC rows use the prebuilt row cache. Compact seeded blocks
    /// are streamed once per logical input column, updating all `D` output
    /// coordinates together.
    pub fn fill_row_dots_real_with_blocks(&self, out: &mut [K], z_blocks: &SuperneoZBlocks) {
        debug_assert!(out.len() <= self.rows, "row-dot output exceeds matrix rows");
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.block_len(),
            "SuperneoMatrixCache::fill_row_dots_real_with_blocks: block count mismatch"
        );
        debug_assert!(
            z_blocks.imag_all_zero,
            "SuperneoMatrixCache::fill_row_dots_real_with_blocks expects a real witness"
        );

        if self.identity {
            for (row, out_row) in out.iter_mut().enumerate() {
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

        for (row, out_row) in out.iter_mut().enumerate() {
            let mut acc = F::ZERO;
            for block in self.row_blocks_for(row).iter().copied() {
                let block_index = block.block();
                if z_blocks.real_nonzero(block_index) {
                    acc += self.compact_dot_real(block, z_blocks, block_index);
                }
            }
            *out_row = K::from(acc);
        }

        for block in &self.seeded_phi81_blocks {
            let transformed_basis = block
                .has_superneo_transformed_columns()
                .then(seeded_transformed_column_basis);
            block.for_each_original_column_rotation::<F, _>(|output, column, rotation| {
                let row_start = block.row_start() + output * D;
                if row_start >= out.len() {
                    return;
                }
                let blk = column / D;
                if !z_blocks.real_nonzero(blk) {
                    return;
                }
                let local = column % D;
                let input = match &transformed_basis {
                    Some(basis) => z_blocks.real_dot(&basis[local], blk),
                    None => z_blocks.real_coefficient(blk, local),
                };
                if input == F::ZERO {
                    return;
                }
                let coordinate_count = min(D, out.len() - row_start);
                for coordinate in 0..coordinate_count {
                    let coefficient = rotation[coordinate];
                    if coefficient != F::ZERO {
                        out[row_start + coordinate] += K::from(coefficient * input);
                    }
                }
            });
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
            let mut acc = F::ZERO;
            for block in self.row_blocks_for(row).iter().copied() {
                let block_index = block.block();
                if z_blocks.real_nonzero(block_index) {
                    acc += self.compact_dot_real(block, z_blocks, block_index);
                }
            }
            *out_row = acc;
        }

        for block in &self.seeded_phi81_blocks {
            let transformed_basis = block
                .has_superneo_transformed_columns()
                .then(seeded_transformed_column_basis);
            block.for_each_original_column_rotation::<F, _>(|output, column, rotation| {
                let row_start = block.row_start() + output * D;
                if row_start >= out.len() {
                    return;
                }
                let block_index = column / D;
                if !z_blocks.real_nonzero(block_index) {
                    return;
                }
                let local = column % D;
                let input = match &transformed_basis {
                    Some(basis) => z_blocks.real_dot(&basis[local], block_index),
                    None => z_blocks.real_coefficient(block_index, local),
                };
                if input == F::ZERO {
                    return;
                }
                let coordinate_count = min(D, out.len() - row_start);
                for coordinate in 0..coordinate_count {
                    let coefficient = rotation[coordinate];
                    if coefficient != F::ZERO {
                        out[row_start + coordinate] += coefficient * input;
                    }
                }
            });
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
        scratch.ensure_block_count(self.cols.div_ceil(D));

        for row in 0..row_cap {
            let w_re = chi_re[row];
            let w_im = chi_im[row];
            if w_re == F::ZERO && w_im == F::ZERO {
                continue;
            }
            if self.identity {
                let block = row / D;
                let local = row % D;
                touch_ring_block(scratch, block);
                scratch.agg_re[block].0[local] += w_re;
                scratch.agg_im[block].0[local] += w_im;
                continue;
            }
            for compact in self.row_blocks_for(row).iter().copied() {
                let block = compact.block();
                touch_ring_block(scratch, block);
                if let Some((local, coefficient)) = compact.single_parts() {
                    scratch.agg_re[block].0[local] += w_re * coefficient;
                    scratch.agg_im[block].0[local] += w_im * coefficient;
                } else {
                    let orig = self.dense_block(compact.dense_index().expect("dense compact block"));
                    add_scaled_rq(&mut scratch.agg_re[block], &orig, w_re);
                    add_scaled_rq(&mut scratch.agg_im[block], &orig, w_im);
                }
            }
        }

        for index in 0..scratch.active_blocks.len() {
            let block = scratch.active_blocks[index];
            scratch.agg_re[block].0 = superneo_bar_block(scratch.agg_re[block].0);
            scratch.agg_im[block].0 = superneo_bar_block(scratch.agg_im[block].0);
        }

        for block in &self.seeded_phi81_blocks {
            let already_transformed = block.has_superneo_transformed_columns();
            for output in 0..block.kappa() {
                let row_start = block.row_start() + output * D;
                if row_start >= row_cap {
                    break;
                }
                let coordinate_count = min(D, row_cap - row_start);
                let work = seeded_work_ranges(block, output);
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let partials: Vec<Vec<(usize, Rq, Rq)>> = work
                    .into_par_iter()
                    .map(|(chunk, local_start, local_end)| {
                        seeded_ring_form_work(
                            block,
                            output,
                            chunk,
                            local_start,
                            local_end,
                            row_start,
                            coordinate_count,
                            chi_re,
                            chi_im,
                            already_transformed,
                        )
                    })
                    .collect();
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let partials: Vec<Vec<(usize, Rq, Rq)>> = work
                    .into_iter()
                    .map(|(chunk, local_start, local_end)| {
                        seeded_ring_form_work(
                            block,
                            output,
                            chunk,
                            local_start,
                            local_end,
                            row_start,
                            coordinate_count,
                            chi_re,
                            chi_im,
                            already_transformed,
                        )
                    })
                    .collect();
                for partial in partials {
                    for (blk, re, im) in partial {
                        touch_ring_block(scratch, blk);
                        add_scaled_rq(&mut scratch.agg_re[blk], &re, F::ONE);
                        add_scaled_rq(&mut scratch.agg_im[blk], &im, F::ONE);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn seeded_ring_form_work(
    block: &neo_ccs::SeededPhi81LinearBlock,
    output: usize,
    chunk: usize,
    local_start: usize,
    local_end: usize,
    row_start: usize,
    coordinate_count: usize,
    chi_re: &[F],
    chi_im: &[F],
    already_transformed: bool,
) -> Vec<(usize, Rq, Rq)> {
    let mut aggregates = HashMap::<usize, ([F; D], [F; D])>::new();
    block.for_each_original_chunk_range_column_rotation::<F, _>(
        output,
        chunk,
        local_start,
        local_end,
        |column, rotation| {
            let mut weight_re = F::ZERO;
            let mut weight_im = F::ZERO;
            for coordinate in 0..coordinate_count {
                let coefficient = rotation[coordinate];
                if coefficient != F::ZERO {
                    weight_re += chi_re[row_start + coordinate] * coefficient;
                    weight_im += chi_im[row_start + coordinate] * coefficient;
                }
            }
            if weight_re == F::ZERO && weight_im == F::ZERO {
                return;
            }
            let entry = aggregates
                .entry(column / D)
                .or_insert_with(|| ([F::ZERO; D], [F::ZERO; D]));
            let local = column % D;
            entry.0[local] += weight_re;
            entry.1[local] += weight_im;
        },
    );
    aggregates
        .into_iter()
        .map(|(blk, (re, im))| {
            let mut re = superneo_bar_block(re);
            let mut im = superneo_bar_block(im);
            if already_transformed {
                re = superneo_bar_block(re);
                im = superneo_bar_block(im);
            }
            (blk, Rq(re), Rq(im))
        })
        .collect()
}

#[inline]
fn touch_ring_block(scratch: &mut RingEvalScratch, blk: usize) {
    if !scratch.touched[blk] {
        scratch.touched[blk] = true;
        scratch.active_blocks.push(blk);
    }
}

fn seeded_transformed_column_basis() -> [Rq; D] {
    core::array::from_fn(|local| {
        let mut basis = [F::ZERO; D];
        basis[local] = F::ONE;
        Rq(superneo_bar_block(basis))
    })
}

pub(super) fn seeded_matrix_column_basis(already_transformed: bool) -> [Rq; D] {
    if already_transformed {
        seeded_transformed_column_basis()
    } else {
        core::array::from_fn(|local| {
            let mut basis = [F::ZERO; D];
            basis[local] = F::ONE;
            Rq(basis)
        })
    }
}
