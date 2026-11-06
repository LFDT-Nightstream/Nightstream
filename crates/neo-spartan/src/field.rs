//! Layer-1 field types and the conversion of workspace values into them.
//!
//! Owns: the Goldilocks and cubic-extension aliases in Plonky3 0.8, the
//! conversions from workspace values (Plonky3 0.5 field values by `gl` and
//! `re_im`, raw words such as key coefficients and Poseidon2 constants by
//! `from_word`), the in-place slice views over their words (the crate's
//! only `unsafe` code outside `scratch.rs`), multilinear helpers, and the
//! point-order conversion. Does not own: words that enter a `Backend`
//! (`circuit`), device outputs, or words that the prover reads back from
//! its own files.

use neo_math::{KExtensions, F, K};
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
    Gl::new(if word < Gl::ORDER_U64 {
        word
    } else {
        word - Gl::ORDER_U64
    })
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
///
/// Built in place: on macOS a table freed per coordinate stays resident in
/// the allocator's cache.
pub(crate) fn eq_table(point: &[Ext]) -> Vec<Ext> {
    let mut table = vec![Ext::ZERO; 1 << point.len()];
    table[0] = Ext::ONE;
    for (t, &coordinate) in point.iter().enumerate() {
        let (low, high) = table[..2 << t].split_at_mut(1 << t);
        low.par_iter_mut().zip(high).for_each(|(low, high)| {
            // value · (1 - c) = value - value · c.
            *high = *low * coordinate;
            *low -= *high;
        });
    }
    table
}

/// Bind the lowest index variable of `table` to `r`, in place.
///
/// Entry `i` reads entries `2i` and `2i + 1`, which only entries `2i` and
/// `2i + 1` overwrite. So the entries in `[2^g, 2^(g+1))` fold in parallel
/// once every entry below them has folded.
pub(crate) fn fold_low(table: &mut Vec<Ext>, r: Ext) {
    let half = table.len() / 2;
    if half == 0 {
        table.clear();
        return;
    }
    let (low, high) = (table[0], table[1]);
    table[0] = low + r * (high - low);
    let mut start = 1;
    while start < half {
        let end = (2 * start).min(half);
        let (done, rest) = table.split_at_mut(2 * start);
        done[start..end]
            .par_iter_mut()
            .zip(rest[..2 * (end - start)].par_chunks_exact(2))
            .for_each(|(value, pair)| *value = pair[0] + r * (pair[1] - pair[0]));
        start = end;
    }
    table.truncate(half);
}

/// Plonky3 orders point coordinates from the most significant index bit;
/// Nightstream orders them from the least significant bit.
pub(crate) fn p3_point(low_bit_first: &[Ext]) -> Point<Ext> {
    Point::new(low_bit_first.iter().rev().copied().collect())
}
