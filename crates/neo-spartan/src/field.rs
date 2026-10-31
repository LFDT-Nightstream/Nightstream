//! Layer-1 field types and the conversion of workspace values into them.
//!
//! Owns: the Goldilocks and cubic-extension aliases in Plonky3 0.8, the
//! conversions from workspace values (Plonky3 0.5 field values by `gl` and
//! `re_im`, raw words such as key coefficients and Poseidon2 constants by
//! `from_word`), the conversions to and from `neo_verified::field`, which
//! read a Plonky3 value by its canonical word, the in-place slice views over
//! their words (the crate's only `unsafe` code outside `scratch.rs`),
//! multilinear helpers, and the point-order conversion. Does not own: words
//! that enter a `Backend` (`circuit`), device outputs, or words that the
//! prover reads back from its own files.

use neo_math::{KExtensions, F, K};
use neo_verified::field as verified;
use p3_field::PrimeField64;
use p3_field_v08::extension::CubicTrinomialExtensionField;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64 as _};
use p3_multilinear_util_v08::point::Point;
use rayon::prelude::*;

/// Goldilocks in Plonky3 0.8 types.
pub(crate) type Gl = p3_goldilocks_v08::Goldilocks;

/// Layer-1 challenge and WHIR extension field, `Gl[w] / (w^3 - w - 1)`.
/// It does not contain `K`: `K` values enter only through `re_im`.
pub(crate) type Ext = CubicTrinomialExtensionField<Gl>;

// The slice views below rely on these layouts. Plonky3 0.8 declares both
// types `repr(transparent)`; a change of layout fails the build here.
const _: () = assert!(size_of::<Gl>() == size_of::<u64>() && align_of::<Gl>() == align_of::<u64>());
const _: () = assert!(size_of::<Ext>() == 3 * size_of::<Gl>() && align_of::<Ext>() == align_of::<Gl>());

/// The base coordinates of `values`, three per element, in place.
pub(crate) fn as_base(values: &[Ext]) -> &[Gl] {
    // SAFETY: `Ext` is `repr(transparent)` over `[Gl; 3]` (layout checked above).
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Gl>(), 3 * values.len()) }
}

/// The base coordinates of `values`, three per element, in place.
pub(crate) fn as_base_mut(values: &mut [Ext]) -> &mut [Gl] {
    // SAFETY: as in `as_base`; every `[Gl; 3]` is a valid `Ext`.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Gl>(), 3 * values.len()) }
}

/// The words of `values` in place, not necessarily canonical.
pub(crate) fn as_words(values: &[Gl]) -> &[u64] {
    // SAFETY: `Gl` is `repr(transparent)` over `u64` (layout checked above).
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u64>(), values.len()) }
}

/// The words of `values` in place, not necessarily canonical.
pub(crate) fn as_words_mut(values: &mut [Gl]) -> &mut [u64] {
    // SAFETY: as in `as_words`; every `u64`, canonical or not, is a valid `Gl`.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u64>(), values.len()) }
}

pub(crate) fn gl(value: F) -> Gl {
    Gl::new(value.as_canonical_u64())
}

/// A workspace word as a field element, reduced to canonical form.
pub(crate) fn from_word(word: u64) -> Gl {
    from_verified(verified::reduce(word))
}

/// `value` in `neo_verified`, by its canonical word: a Plonky3 value may
/// hold any word.
pub(crate) fn to_verified(value: Gl) -> verified::Gl {
    verified::reduce(value.as_canonical_u64())
}

pub(crate) fn from_verified(value: verified::Gl) -> Gl {
    Gl::new(verified::gl_word(value))
}

/// The coordinates of `K = F[u] / (u^2 - 7)` in the basis `(1, u)`.
pub(crate) fn re_im(value: K) -> [Gl; 2] {
    value.as_coeffs().map(gl)
}

/// The three base coordinates of an `Ext` value.
pub(crate) fn ext_words(value: Ext) -> [Gl; 3] {
    let slice = <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(&value);
    [slice[0], slice[1], slice[2]]
}

/// The base-field coordinates of an `Ext` table, one column per basis element.
pub(crate) fn coordinates(table: &[Ext]) -> Vec<Gl> {
    let mut columns = vec![Gl::ZERO; 3 * table.len()];
    for (index, value) in table.iter().enumerate() {
        for (c, &coefficient) in <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(value)
            .iter()
            .enumerate()
        {
            columns[c * table.len() + index] = coefficient;
        }
    }
    columns
}

/// The signed integer `value` as a field element.
pub(crate) fn signed(value: i64) -> Gl {
    let magnitude = Gl::from_u64(value.unsigned_abs());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// `eq(point, x)` for every `x < 2^point.len()`; index bit `t` pairs with
/// `point[t]` (low bit first, as everywhere in Nightstream).
pub(crate) fn eq_table(point: &[Ext]) -> Vec<Ext> {
    let mut table = vec![Ext::ONE];
    for &coordinate in point {
        let mut next = vec![Ext::ZERO; 2 * table.len()];
        let (low, high) = next.split_at_mut(table.len());
        low.par_iter_mut()
            .zip(high.par_iter_mut())
            .zip(table.par_iter())
            .for_each(|((low, high), &value)| {
                // value · (1 - c) = value - value · c.
                *high = value * coordinate;
                *low = value - *high;
            });
        table = next;
    }
    table
}

/// Bind the lowest index variable of `table` to `r`.
pub(crate) fn fold_low(table: &mut Vec<Ext>, r: Ext) {
    *table = table
        .par_chunks_exact(2)
        .map(|pair| pair[0] + r * (pair[1] - pair[0]))
        .collect();
}

/// Plonky3 orders point coordinates from the most significant index bit;
/// Nightstream orders them from the least significant bit.
pub(crate) fn p3_point(low_bit_first: &[Ext]) -> Point<Ext> {
    Point::new(low_bit_first.iter().rev().copied().collect())
}
