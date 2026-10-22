//! Layer-1 field types and the conversion of workspace values into them.
//!
//! Owns: the Goldilocks and cubic-extension aliases in Plonky3 0.8, the
//! conversions from workspace values (Plonky3 0.5 field values by `gl` and
//! `re_im`, raw words such as key coefficients and Poseidon2 constants by
//! `from_word`), multilinear helpers, and the point-order conversion. Does
//! not own: words that enter a `Backend` (`circuit`), device outputs, or
//! words that the prover reads back from its own files.

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
