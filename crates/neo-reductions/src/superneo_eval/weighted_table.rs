use super::weighted::weighted_projection_basis_forms_from_k;
use super::{coeff_dot, is_all_zero, Rq, SuperneoEvalCache, SuperneoMatrixCache, SuperneoZBlocks};
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

impl SuperneoMatrixCache {
    #[inline]
    fn row_dot_ring_weighted_projected_with_blocks(
        &self,
        row: usize,
        z_blocks: &SuperneoZBlocks,
        basis_re_forms: &[Rq; D],
        basis_im_forms: &[Rq; D],
    ) -> K {
        debug_assert_eq!(
            self.cols.div_ceil(D),
            z_blocks.re.len(),
            "SuperneoMatrixCache::row_dot_ring_weighted_projected_with_blocks: block count mismatch"
        );
        if row >= self.rows {
            return K::ZERO;
        }

        let mut acc = K::ZERO;
        let extension_generator = K::from_coeffs([F::ZERO, F::ONE]);
        for rb in self.row_blocks_for(row) {
            if !z_blocks.block_nonzero(rb.blk) {
                continue;
            }

            let (re_form, im_form) = weighted_projection_pair_from_orig(&rb.orig, basis_re_forms, basis_im_forms);
            if is_all_zero(&re_form.0) && is_all_zero(&im_form.0) {
                continue;
            }

            let (rr, ir) = if z_blocks.re_nonzero[rb.blk] {
                let z_re = &z_blocks.re[rb.blk];
                (coeff_dot(&re_form, z_re), coeff_dot(&im_form, z_re))
            } else {
                (F::ZERO, F::ZERO)
            };
            let (ri, ii) = if z_blocks.im_nonzero[rb.blk] {
                let z_im = &z_blocks.im[rb.blk];
                (coeff_dot(&re_form, z_im), coeff_dot(&im_form, z_im))
            } else {
                (F::ZERO, F::ZERO)
            };
            acc += K::from_coeffs([rr, ir]) + extension_generator * K::from_coeffs([ri, ii]);
        }
        acc
    }
}

#[inline]
fn weighted_projection_pair_from_orig(orig: &Rq, basis_re_forms: &[Rq; D], basis_im_forms: &[Rq; D]) -> (Rq, Rq) {
    let neg_one = F::ZERO - F::ONE;
    let mut first = None;
    let mut multiple = false;
    for (local, &coeff) in orig.0.iter().enumerate() {
        if coeff == F::ZERO {
            continue;
        }
        if first.is_none() {
            first = Some((local, coeff));
        } else {
            multiple = true;
            break;
        }
    }

    match (first, multiple) {
        (None, _) => return (Rq([F::ZERO; D]), Rq([F::ZERO; D])),
        (Some((local, coeff)), false) => {
            return (
                scale_weighted_projection_form(basis_re_forms[local], coeff, neg_one),
                scale_weighted_projection_form(basis_im_forms[local], coeff, neg_one),
            );
        }
        _ => {}
    }

    let mut out_re = [F::ZERO; D];
    let mut out_im = [F::ZERO; D];
    for (local, &coeff) in orig.0.iter().enumerate() {
        if coeff == F::ZERO {
            continue;
        }
        add_scaled_form(&mut out_re, &basis_re_forms[local].0, coeff, neg_one);
        add_scaled_form(&mut out_im, &basis_im_forms[local].0, coeff, neg_one);
    }
    (Rq(out_re), Rq(out_im))
}

#[inline]
fn scale_weighted_projection_form(mut form: Rq, coeff: F, neg_one: F) -> Rq {
    if coeff == F::ONE {
        return form;
    }
    if coeff == neg_one {
        for slot in &mut form.0 {
            *slot = F::ZERO - *slot;
        }
    } else {
        for slot in &mut form.0 {
            *slot *= coeff;
        }
    }
    form
}

#[inline]
fn add_scaled_form(out: &mut [F; D], form: &[F; D], coeff: F, neg_one: F) {
    if coeff == F::ONE {
        for i in 0..D {
            out[i] += form[i];
        }
    } else if coeff == neg_one {
        for i in 0..D {
            out[i] -= form[i];
        }
    } else {
        for i in 0..D {
            out[i] += coeff * form[i];
        }
    }
}

impl SuperneoEvalCache {
    pub fn eval_weighted_row_table(
        &self,
        z_blocks: &SuperneoZBlocks,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Vec<K> {
        assert_eq!(
            self.mats.len(),
            mat_coeffs.len(),
            "eval_weighted_row_table: matrix coefficient count mismatch"
        );
        let mut out = vec![K::ZERO; n_pad];
        if z_blocks.imag_all_zero {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                out.par_iter_mut()
                    .take(n_eff)
                    .enumerate()
                    .for_each(|(row, out_r)| {
                        let mut row_acc = K::ZERO;
                        for (j, mat_cache) in self.mats.iter().enumerate() {
                            let coeff = mat_coeffs[j];
                            if coeff == K::ZERO {
                                continue;
                            }
                            let y_alpha = mat_cache.row_dot_ring_weighted_with_blocks(row, z_blocks, weights);
                            if y_alpha != K::ZERO {
                                row_acc += coeff * y_alpha;
                            }
                        }
                        *out_r = row_acc;
                    });
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                for (row, out_r) in out.iter_mut().take(n_eff).enumerate() {
                    let mut row_acc = K::ZERO;
                    for j in 0..self.mats.len() {
                        let coeff = mat_coeffs[j];
                        if coeff == K::ZERO {
                            continue;
                        }
                        let y_alpha = mat_cache.row_dot_ring_weighted_with_blocks(row, z_blocks, weights);
                        if y_alpha != K::ZERO {
                            row_acc += coeff * y_alpha;
                        }
                    }
                    *out_r = row_acc;
                }
            }
            return out;
        }

        let (basis_re_forms, basis_im_forms) = weighted_projection_basis_forms_from_k(weights);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            out.par_iter_mut()
                .take(n_eff)
                .enumerate()
                .for_each(|(row, out_r)| {
                    let mut row_acc = K::ZERO;
                    for j in 0..self.mats.len() {
                        let coeff = mat_coeffs[j];
                        if coeff == K::ZERO {
                            continue;
                        }
                        let y_alpha = self.mats[j].row_dot_ring_weighted_projected_with_blocks(
                            row,
                            z_blocks,
                            &basis_re_forms,
                            &basis_im_forms,
                        );
                        if y_alpha != K::ZERO {
                            row_acc += coeff * y_alpha;
                        }
                    }
                    *out_r = row_acc;
                });
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            for (row, out_r) in out.iter_mut().take(n_eff).enumerate() {
                let mut row_acc = K::ZERO;
                for j in 0..self.mats.len() {
                    let coeff = mat_coeffs[j];
                    if coeff == K::ZERO {
                        continue;
                    }
                    let y_alpha = self.mats[j].row_dot_ring_weighted_projected_with_blocks(
                        row,
                        z_blocks,
                        &basis_re_forms,
                        &basis_im_forms,
                    );
                    if y_alpha != K::ZERO {
                        row_acc += coeff * y_alpha;
                    }
                }
                *out_r = row_acc;
            }
        }
        out
    }
}
