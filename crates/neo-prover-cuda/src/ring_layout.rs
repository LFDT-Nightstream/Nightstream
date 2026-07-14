//! The flat ring-column layout device kernels consume.
//!
//! Owns the mapping between the three representations of a witness:
//! an assignment `z ∈ F^len`, its `Mat<F>` ring matrix (D rows × cols,
//! entry `(rho, blk)` = `z[blk·D + rho]`, mirroring
//! `CcsInstance::from_low_norm_assignment`), and the flat column-major
//! coefficient words (`words[blk·D + rho]`) uploaded to the GPU. The flat
//! layout equals the assignment vector zero-padded to `cols·D`.

use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::optimized_engine::oracle::SuperneoRingLinearForm;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::kernels::goldilocks::GOLDILOCKS_MODULUS;

/// Assignment → zero-padded column-major words (`cols * D` long).
pub fn assignment_to_words(z: &[F], cols: usize) -> Vec<u64> {
    let mut words = vec![0u64; cols * D];
    for (word, value) in words.iter_mut().zip(z) {
        *word = value.as_canonical_u64();
    }
    words
}

/// Assignment → ring matrix.
pub fn assignment_to_mat(z: &[F], cols: usize) -> Mat<F> {
    let mut out = Mat::zero(D, cols, F::ZERO);
    for (column, value) in z.iter().enumerate() {
        out[(column % D, column / D)] = *value;
    }
    out
}

/// Ring matrix → column-major words.
pub fn mat_to_words(mat: &Mat<F>) -> Vec<u64> {
    let cols = mat.cols();
    let mut words = vec![0u64; cols * D];
    for blk in 0..cols {
        for rho in 0..D {
            words[blk * D + rho] = mat[(rho, blk)].as_canonical_u64();
        }
    }
    words
}

/// Column-major words → ring matrix.
pub fn mat_from_words(words: &[u64], cols: usize) -> Mat<F> {
    debug_assert_eq!(words.len(), cols * D);
    let mut out = Mat::zero(D, cols, F::ZERO);
    for blk in 0..cols {
        for rho in 0..D {
            out[(rho, blk)] = f_from_word(words[blk * D + rho]);
        }
    }
    out
}

fn f_from_word(word: u64) -> F {
    let canonical = if word >= GOLDILOCKS_MODULUS {
        word - GOLDILOCKS_MODULUS
    } else {
        word
    };
    F::from_u64(canonical)
}

/// Ring linear forms → the `[2t][blocks][D]` word matrix the ring mat-vec
/// kernels consume: row `2j` holds re(form_j), row `2j+1` holds im(form_j).
/// Returns the words and the per-row length (`blocks * D`).
pub fn forms_to_words(forms: &[SuperneoRingLinearForm]) -> (Vec<u64>, usize) {
    let dense: Vec<(Vec<F>, Vec<F>)> = forms
        .iter()
        .map(SuperneoRingLinearForm::to_dense_block_coeffs)
        .collect();
    let row_len = dense.first().map(|(re, _)| re.len()).unwrap_or(0);
    let mut out = vec![0u64; 2 * dense.len() * row_len];
    for (j, (re, im)) in dense.iter().enumerate() {
        debug_assert_eq!(re.len(), row_len, "forms must share the block count");
        for (slot, value) in out[(2 * j) * row_len..].iter_mut().zip(re) {
            *slot = value.as_canonical_u64();
        }
        for (slot, value) in out[(2 * j + 1) * row_len..].iter_mut().zip(im) {
            *slot = value.as_canonical_u64();
        }
    }
    (out, row_len)
}
