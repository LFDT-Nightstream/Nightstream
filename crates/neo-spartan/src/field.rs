//! Layer-1 field types and the only crossing between Plonky3 generations.
//!
//! Owns: the Goldilocks and cubic-extension aliases in Plonky3 0.8, the
//! canonical-u64 conversion from workspace values, multilinear helpers, and
//! the point-order conversion. Workspace values are Plonky3 0.5; nothing else
//! converts.

use neo_math::{KExtensions, F, K};
use p3_field::PrimeField64;
use p3_field_v08::extension::CubicTrinomialExtensionField;
use p3_field_v08::PrimeCharacteristicRing;
use p3_multilinear_util_v08::point::Point;

/// Goldilocks in Plonky3 0.8 types.
pub(crate) type Gl = p3_goldilocks_v08::Goldilocks;

/// Layer-1 challenge and WHIR extension field, `Gl[w] / (w^3 - w - 1)`.
/// It does not contain `K`: `K` values enter only through `re_im`.
pub(crate) type Ext = CubicTrinomialExtensionField<Gl>;

pub(crate) fn gl(value: F) -> Gl {
    Gl::new(value.as_canonical_u64())
}

/// The coordinates of `K = F[u] / (u^2 - 7)` in the basis `(1, u)`.
pub(crate) fn re_im(value: K) -> [Gl; 2] {
    value.as_coeffs().map(gl)
}

/// `weights[0]·Re(value) + weights[1]·Im(value)`: one F-linear projection of
/// a `K` value. A `K` equation over F-valued data holds exactly when both
/// coordinates hold, so distinct weights per coordinate lose nothing.
pub(crate) fn project(value: K, weights: [Ext; 2]) -> Ext {
    let [re, im] = re_im(value);
    weights[0] * re + weights[1] * im
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
        let mut next = Vec::with_capacity(2 * table.len());
        next.extend(table.iter().map(|&value| value * (Ext::ONE - coordinate)));
        next.extend(table.iter().map(|&value| value * coordinate));
        table = next;
    }
    table
}

/// `eq(a, b) = Π (a_t b_t + (1 - a_t)(1 - b_t))`.
pub(crate) fn eq_eval(a: &[Ext], b: &[Ext]) -> Ext {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y + (Ext::ONE - x) * (Ext::ONE - y))
        .product()
}

/// Bind the lowest index variable of `table` to `r`.
pub(crate) fn fold_low(table: &mut Vec<Ext>, r: Ext) {
    let half = table.len() / 2;
    for k in 0..half {
        table[k] = table[2 * k] + r * (table[2 * k + 1] - table[2 * k]);
    }
    table.truncate(half);
}

/// Plonky3 orders point coordinates from the most significant index bit;
/// Nightstream orders them from the least significant bit.
pub(crate) fn p3_point(low_bit_first: &[Ext]) -> Point<Ext> {
    Point::new(low_bit_first.iter().rev().copied().collect())
}
