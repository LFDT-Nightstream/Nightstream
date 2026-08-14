use super::seeded::{seeded_matrix_column_basis, seeded_work_ranges};
use super::{Rq, SuperneoEvalCache, SuperneoMatrixCache, SuperneoZBlocks};
use neo_math::{superneo_bar_block, KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

impl SuperneoMatrixCache {
    #[inline]
    fn row_dot_ring_weighted_projected_explicit(&self, row: usize, identity_projection: &[K]) -> K {
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc = K::ZERO;
        for block in self.row_blocks_for(row).iter().copied() {
            if let Some((block, local, coefficient)) = block.single_parts() {
                let projected = identity_projection[block * D + local];
                acc += if coefficient == F::ONE {
                    projected
                } else {
                    projected.scale_base(coefficient)
                };
            } else {
                let orig = self.dense_block(self.dense_pattern_index(block));
                acc += projected_linear_form(&orig, self.row_block_index(block), identity_projection);
            }
        }
        acc += self.geometric_weighted_projection(row, identity_projection);
        acc
    }
}

#[inline]
fn single_nonzero(orig: &Rq) -> Option<(usize, F)> {
    let mut found = None;
    for (local, &coefficient) in orig.0.iter().enumerate() {
        if coefficient == F::ZERO {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some((local, coefficient));
    }
    found
}

#[inline]
fn projected_linear_form(orig: &Rq, block: usize, identity_projection: &[K]) -> K {
    if let Some((local, coefficient)) = single_nonzero(orig) {
        let projected = identity_projection[block * D + local];
        return if coefficient == F::ONE {
            projected
        } else {
            projected.scale_base(coefficient)
        };
    }

    let mut out = K::ZERO;
    let projected = &identity_projection[block * D..(block + 1) * D];
    for (value, &coefficient) in projected.iter().zip(&orig.0) {
        if coefficient == F::ONE {
            out += *value;
        } else if coefficient != F::ZERO {
            out += value.scale_base(coefficient);
        }
    }
    out
}

impl SuperneoEvalCache {
    #[inline]
    fn eval_weighted_explicit_row(&self, row: usize, mat_coeffs: &[K], identity_projection: &[K]) -> K {
        let mut row_acc = K::ZERO;
        let mut add_matrix = |matrix: usize| {
            if self.mats[matrix].identity {
                return;
            }
            let coeff = mat_coeffs[matrix];
            if coeff == K::ZERO {
                return;
            }
            let y_alpha = self.mats[matrix].row_dot_ring_weighted_projected_explicit(row, identity_projection);
            if y_alpha != K::ZERO {
                row_acc += coeff * y_alpha;
            }
        };

        if let Some(masks) = &self.explicit_matrix_masks {
            let mut mask = masks[row];
            while mask != 0 {
                let matrix = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                add_matrix(matrix);
            }
        } else {
            for matrix in 0..self.mats.len() {
                add_matrix(matrix);
            }
        }
        row_acc
    }

    pub fn eval_weighted_row_table(
        &self,
        z_blocks: &SuperneoZBlocks,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Vec<K> {
        #[cfg(feature = "perf-timers")]
        let total_start = std::time::Instant::now();
        assert_eq!(
            self.mats.len(),
            mat_coeffs.len(),
            "eval_weighted_row_table: matrix coefficient count mismatch"
        );
        let identity_projection = weighted_identity_projection(z_blocks, weights);
        let mut out = vec![K::ZERO; n_pad];
        let identity_coeff = self
            .mats
            .iter()
            .zip(mat_coeffs)
            .filter(|(matrix, _)| matrix.identity)
            .fold(K::ZERO, |sum, (_, &coefficient)| sum + coefficient);
        let has_identity_contribution = identity_coeff != K::ZERO;
        assert!(
            !has_identity_contribution || n_eff <= identity_projection.len(),
            "eval_weighted_row_table: identity rows exceed witness projection"
        );
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            out.par_iter_mut()
                .take(n_eff)
                .enumerate()
                .for_each(|(row, out_r)| {
                    let identity = if has_identity_contribution {
                        identity_coeff * identity_projection[row]
                    } else {
                        K::ZERO
                    };
                    *out_r = identity + self.eval_weighted_explicit_row(row, mat_coeffs, &identity_projection);
                });
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            for (row, out_r) in out.iter_mut().take(n_eff).enumerate() {
                let identity = if has_identity_contribution {
                    identity_coeff * identity_projection[row]
                } else {
                    K::ZERO
                };
                *out_r = identity + self.eval_weighted_explicit_row(row, mat_coeffs, &identity_projection);
            }
        }
        #[cfg(feature = "perf-timers")]
        let explicit_elapsed = total_start.elapsed();
        self.add_seeded_weighted_rows(&mut out[..n_eff], mat_coeffs, &identity_projection);
        #[cfg(feature = "perf-timers")]
        eprintln!(
            "SuperneoEvalCache::eval_weighted_row_table: explicit {:.2?} seeded {:.2?} total {:.2?}",
            explicit_elapsed,
            total_start.elapsed() - explicit_elapsed,
            total_start.elapsed(),
        );
        out
    }

    fn add_seeded_weighted_rows(&self, out: &mut [K], mat_coeffs: &[K], identity_projection: &[K]) {
        let plain_basis = seeded_matrix_column_basis(false);
        let transformed_basis = seeded_matrix_column_basis(true);

        for (matrix, &matrix_coeff) in self.mats.iter().zip(mat_coeffs) {
            if matrix_coeff == K::ZERO {
                continue;
            }
            for block in &matrix.seeded_phi81_blocks {
                let column_basis = if block.has_superneo_transformed_columns() {
                    &transformed_basis
                } else {
                    &plain_basis
                };
                for output in 0..block.kappa() {
                    let row_start = block.row_start() + output * D;
                    if row_start >= out.len() {
                        break;
                    }
                    let work = seeded_work_ranges(block, output);
                    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                    let contribution = work
                        .into_par_iter()
                        .map(|(chunk, local_start, local_end)| {
                            seeded_weighted_chunk(
                                block,
                                output,
                                chunk,
                                local_start,
                                local_end,
                                column_basis,
                                identity_projection,
                                matrix_coeff,
                            )
                        })
                        .reduce(
                            || [K::ZERO; D],
                            |mut left, right| {
                                for coordinate in 0..D {
                                    left[coordinate] += right[coordinate];
                                }
                                left
                            },
                        );
                    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                    let contribution = work
                        .into_iter()
                        .map(|(chunk, local_start, local_end)| {
                            seeded_weighted_chunk(
                                block,
                                output,
                                chunk,
                                local_start,
                                local_end,
                                column_basis,
                                identity_projection,
                                matrix_coeff,
                            )
                        })
                        .fold([K::ZERO; D], |mut left, right| {
                            for coordinate in 0..D {
                                left[coordinate] += right[coordinate];
                            }
                            left
                        });
                    let coordinate_count = core::cmp::min(D, out.len() - row_start);
                    for coordinate in 0..coordinate_count {
                        out[row_start + coordinate] += contribution[coordinate];
                    }
                }
            }
        }
    }
}

fn weighted_identity_projection(z_blocks: &SuperneoZBlocks, weights: &[K; D]) -> Vec<K> {
    let mut weight_re = [F::ZERO; D];
    let mut weight_im = [F::ZERO; D];
    for (index, weight) in weights.iter().enumerate() {
        [weight_re[index], weight_im[index]] = weight.as_coeffs();
    }
    let bar_weight_re = Rq(superneo_bar_block(weight_re));
    let bar_weight_im = Rq(superneo_bar_block(weight_im));
    let extension_generator = K::from_coeffs([F::ZERO, F::ONE]);

    let mut out = vec![K::ZERO; z_blocks.block_len() * D];
    let fill_block = |block: usize, output: &mut [K]| {
        if !z_blocks.block_nonzero(block) {
            return;
        }
        let (rr, ir) = if z_blocks.real_nonzero(block) {
            (
                z_blocks.real_mul(&bar_weight_re, block),
                z_blocks.real_mul(&bar_weight_im, block),
            )
        } else {
            (Rq::zero(), Rq::zero())
        };
        let (ri, ii) = if z_blocks.im_nonzero[block] {
            (
                bar_weight_re.mul(&z_blocks.im[block]),
                bar_weight_im.mul(&z_blocks.im[block]),
            )
        } else {
            (Rq::zero(), Rq::zero())
        };
        for (local, slot) in output.iter_mut().enumerate() {
            *slot = K::from_coeffs([rr.0[local], ir.0[local]])
                + extension_generator * K::from_coeffs([ri.0[local], ii.0[local]]);
        }
    };

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    out.par_chunks_mut(D)
        .enumerate()
        .for_each(|(block, output)| fill_block(block, output));
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    out.chunks_mut(D)
        .enumerate()
        .for_each(|(block, output)| fill_block(block, output));
    out
}

fn seeded_weighted_chunk(
    block: &neo_ccs::SeededPhi81LinearBlock,
    output: usize,
    chunk: usize,
    local_start: usize,
    local_end: usize,
    column_basis: &[Rq; D],
    identity_projection: &[K],
    matrix_coeff: K,
) -> [K; D] {
    let mut out = [K::ZERO; D];
    block.for_each_original_chunk_range_column_rotation::<F, _>(
        output,
        chunk,
        local_start,
        local_end,
        |column, rotation| {
            let blk = column / D;
            let contribution =
                matrix_coeff * projected_linear_form(&column_basis[column % D], blk, identity_projection);
            if contribution == K::ZERO {
                return;
            }
            for coordinate in 0..D {
                let coefficient = rotation[coordinate];
                if coefficient != F::ZERO {
                    out[coordinate] += contribution.scale_base(coefficient);
                }
            }
        },
    );
    out
}
