//! `K = F[u] / (u^2 - 7)` over any backend. A value is `[re, im]`.

use neo_math::{KExtensions, K};
use neo_spartan::{Backend, Error};
use p3_field::PrimeField64;

pub(super) type Kw<B> = [<B as Backend>::F; 2];

pub(super) fn constant<B: Backend>(b: &mut B, value: K) -> Kw<B> {
    value
        .as_coeffs()
        .map(|word| b.constant(word.as_canonical_u64()))
}

pub(super) fn zero<B: Backend>(b: &mut B) -> Kw<B> {
    [b.constant(0), b.constant(0)]
}

pub(super) fn one<B: Backend>(b: &mut B) -> Kw<B> {
    [b.constant(1), b.constant(0)]
}

pub(super) fn add<B: Backend>(b: &mut B, x: Kw<B>, y: Kw<B>) -> Kw<B> {
    [b.add(x[0], y[0]), b.add(x[1], y[1])]
}

pub(super) fn sub<B: Backend>(b: &mut B, x: Kw<B>, y: Kw<B>) -> Kw<B> {
    [b.sub(x[0], y[0]), b.sub(x[1], y[1])]
}

/// `(x0 + x1 u)(y0 + y1 u) = x0 y0 + 7 x1 y1 + (x0 y1 + x1 y0) u`.
pub(super) fn mul<B: Backend>(b: &mut B, x: Kw<B>, y: Kw<B>) -> Kw<B> {
    let rr = b.mul(x[0], y[0]);
    let ii = b.mul(x[1], y[1]);
    let seven = b.scale(ii, 7);
    let ri = b.mul(x[0], y[1]);
    let ir = b.mul(x[1], y[0]);
    [b.add(rr, seven), b.add(ri, ir)]
}

/// `x^exponent` by squaring.
pub(super) fn power<B: Backend>(b: &mut B, x: Kw<B>, exponent: usize) -> Kw<B> {
    let mut result = one(b);
    let mut base = x;
    let mut e = exponent;
    while e != 0 {
        if e & 1 == 1 {
            result = mul(b, result, base);
        }
        e >>= 1;
        if e != 0 {
            base = mul(b, base, base);
        }
    }
    result
}

/// `Σ_i values_i·x^i` by Horner.
pub(super) fn horner<B: Backend>(b: &mut B, values: &[Kw<B>], x: Kw<B>) -> Kw<B> {
    let mut total = zero(b);
    for &value in values.iter().rev() {
        let scaled = mul(b, total, x);
        total = add(b, scaled, value);
    }
    total
}

/// `Π_t ((1 - a_t)(1 - b_t) + a_t b_t)`.
pub(super) fn equality<B: Backend>(b: &mut B, x: &[Kw<B>], y: &[Kw<B>]) -> Kw<B> {
    assert_eq!(x.len(), y.len());
    let mut total = one(b);
    for (&a, &c) in x.iter().zip(y) {
        let both = mul(b, a, c);
        let twice = add(b, both, both);
        let unit = one(b);
        let shifted = add(b, twice, unit);
        let sum = add(b, a, c);
        let factor = sub(b, shifted, sum);
        total = mul(b, total, factor);
    }
    total
}

pub(super) fn assert_equal<B: Backend>(b: &mut B, x: Kw<B>, y: Kw<B>, what: &'static str) -> Result<(), Error> {
    b.assert_equal(x[0], y[0], what)?;
    b.assert_equal(x[1], y[1], what)
}
