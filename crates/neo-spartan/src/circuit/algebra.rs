//! Field helpers over any backend: extension constants and products, `K`
//! and `K ⊗ Ext` values, eq tables and interpolation. Points are low bit
//! first.

use p3_field_v08::{Field, PrimeCharacteristicRing, PrimeField64};

use super::Backend;
use crate::field::{ext_words, Ext, Gl};

/// A constant from a layer-1 field value.
pub(crate) fn constant<B: Backend>(b: &mut B, value: Gl) -> B::F {
    b.constant(value.as_canonical_u64())
}

/// A proof word from a layer-1 field value.
pub(crate) fn private<B: Backend>(b: &mut B, value: Gl) -> B::F {
    b.private(value.as_canonical_u64())
}

/// `a·by` for a layer-1 field constant.
pub(crate) fn scale<B: Backend>(b: &mut B, a: B::F, by: Gl) -> B::F {
    b.scale(a, by.as_canonical_u64())
}

/// An extension constant from a layer-1 value.
pub(crate) fn ext_constant<B: Backend>(b: &mut B, value: Ext) -> B::E {
    b.ext_constant(ext_words(value).map(|word| word.as_canonical_u64()))
}

pub(crate) fn ext_zero<B: Backend>(b: &mut B) -> B::E {
    ext_constant(b, Ext::ZERO)
}

pub(crate) fn ext_one<B: Backend>(b: &mut B) -> B::E {
    ext_constant(b, Ext::ONE)
}

/// An extension proof value, as three proof words.
pub(crate) fn private_ext<B: Backend>(b: &mut B, value: Ext) -> B::E {
    let words = ext_words(value).map(|word| private(b, word));
    b.ext(words)
}

/// A base word as an extension value.
pub(crate) fn lift<B: Backend>(b: &mut B, value: B::F) -> B::E {
    let zero = b.constant(0);
    b.ext([value, zero, zero])
}

pub(crate) fn ext_scale_constant<B: Backend>(b: &mut B, a: B::E, by: Gl) -> B::E {
    let by = constant(b, by);
    b.ext_scale(a, by)
}

/// `Σ_i a_i·b_i`.
pub(crate) fn dot<B: Backend>(b: &mut B, x: &[B::E], y: &[B::E]) -> B::E {
    assert_eq!(x.len(), y.len());
    let mut total = ext_zero(b);
    for (&x, &y) in x.iter().zip(y) {
        let term = b.ext_mul(x, y);
        total = b.ext_add(total, term);
    }
    total
}

/// `1 - x`.
pub(crate) fn one_minus<B: Backend>(b: &mut B, x: B::E) -> B::E {
    let one = ext_one(b);
    b.ext_sub(one, x)
}

/// `eq(x, y) = x·y + (1 - x)(1 - y)` for one coordinate.
pub(crate) fn eq1<B: Backend>(b: &mut B, x: B::E, y: B::E) -> B::E {
    let both = b.ext_mul(x, y);
    let twice = b.ext_add(both, both);
    let one = ext_one(b);
    let shifted = b.ext_add(twice, one);
    let sum = b.ext_add(x, y);
    b.ext_sub(shifted, sum)
}

/// `Π_t eq(a_t, b_t)`.
pub(crate) fn eq_eval<B: Backend>(b: &mut B, x: &[B::E], y: &[B::E]) -> B::E {
    assert_eq!(x.len(), y.len());
    let mut total = ext_one(b);
    for (&x, &y) in x.iter().zip(y) {
        let factor = eq1(b, x, y);
        total = b.ext_mul(total, factor);
    }
    total
}

/// `eq(point, i)` for every `i < 2^point.len()`; index bit `t` pairs with `point[t]`.
pub(crate) fn eq_table<B: Backend>(b: &mut B, point: &[B::E]) -> Vec<B::E> {
    let mut table = vec![ext_one(b)];
    for &x in point {
        let low = one_minus(b, x);
        let mut next = Vec::with_capacity(2 * table.len());
        for &value in &table {
            next.push(b.ext_mul(value, low));
        }
        for &value in &table {
            next.push(b.ext_mul(value, x));
        }
        table = next;
    }
    table
}

/// `eq(point, index)` for a known `index`.
pub(crate) fn eq_at<B: Backend>(b: &mut B, point: &[B::E], index: usize) -> B::E {
    let mut total = ext_one(b);
    for (t, &x) in point.iter().enumerate() {
        let factor = if (index >> t) & 1 == 1 { x } else { one_minus(b, x) };
        total = b.ext_mul(total, factor);
    }
    total
}

/// The polynomial through `(i, values[i])` for `i < values.len()`, at `x`.
pub(crate) fn interpolate<B: Backend>(b: &mut B, values: &[B::E], x: B::E) -> B::E {
    let n = values.len();
    let mut total = ext_zero(b);
    for (i, &value) in values.iter().enumerate() {
        let mut numerator = ext_one(b);
        let mut denominator = Gl::ONE;
        for j in 0..n {
            if i != j {
                let node = ext_constant(b, Ext::from(Gl::from_usize(j)));
                let factor = b.ext_sub(x, node);
                numerator = b.ext_mul(numerator, factor);
                denominator *= Gl::from_usize(i) - Gl::from_usize(j);
            }
        }
        let weight = ext_scale_constant(b, numerator, denominator.inverse());
        let term = b.ext_mul(value, weight);
        total = b.ext_add(total, term);
    }
    total
}

/// `Σ_i c_i x^i` by Horner over extension coefficients.
pub(crate) fn horner<B: Backend>(b: &mut B, coefficients: &[B::E], x: B::E) -> B::E {
    let mut total = ext_zero(b);
    for &c in coefficients.iter().rev() {
        let scaled = b.ext_mul(total, x);
        total = b.ext_add(scaled, c);
    }
    total
}

/// A `K = F[u] / (u^2 - 7)` value: `[re, im]`.
pub(crate) type K<B> = [<B as Backend>::F; 2];

/// A `K ⊗ Ext = Ext[u] / (u^2 - 7)` value.
pub(crate) struct Kx<B: Backend> {
    pub(crate) re: B::E,
    pub(crate) im: B::E,
}

impl<B: Backend> Clone for Kx<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for Kx<B> {}

impl<B: Backend> Kx<B> {
    pub(crate) fn zero(b: &mut B) -> Self {
        let zero = ext_zero(b);
        Self { re: zero, im: zero }
    }

    pub(crate) fn one(b: &mut B) -> Self {
        let (one, zero) = (ext_one(b), ext_zero(b));
        Self { re: one, im: zero }
    }

    pub(crate) fn from_k(b: &mut B, value: K<B>) -> Self {
        Self {
            re: lift(b, value[0]),
            im: lift(b, value[1]),
        }
    }

    pub(crate) fn add(self, b: &mut B, other: Self) -> Self {
        Self {
            re: b.ext_add(self.re, other.re),
            im: b.ext_add(self.im, other.im),
        }
    }

    pub(crate) fn mul(self, b: &mut B, other: Self) -> Self {
        let rr = b.ext_mul(self.re, other.re);
        let ii = b.ext_mul(self.im, other.im);
        let seven = ext_scale_constant(b, ii, Gl::from_u8(7));
        let ri = b.ext_mul(self.re, other.im);
        let ir = b.ext_mul(self.im, other.re);
        Self {
            re: b.ext_add(rr, seven),
            im: b.ext_add(ri, ir),
        }
    }

    pub(crate) fn scale(self, b: &mut B, by: B::E) -> Self {
        Self {
            re: b.ext_mul(self.re, by),
            im: b.ext_mul(self.im, by),
        }
    }

    /// `1 - self`.
    pub(crate) fn one_minus(self, b: &mut B) -> Self {
        let re = one_minus(b, self.re);
        let zero = ext_zero(b);
        Self {
            re,
            im: b.ext_sub(zero, self.im),
        }
    }
}

/// `generator^index` from the little-endian bits of `index`.
pub(crate) fn power_from_bits<B: Backend>(b: &mut B, generator: Gl, bits: &[B::F]) -> B::F {
    let mut total = b.constant(1);
    let mut base = generator;
    for &bit in bits {
        let one = b.constant(1);
        let factor_minus = scale(b, bit, base - Gl::ONE);
        let factor = b.add(one, factor_minus);
        total = b.mul(total, factor);
        base = base.square();
    }
    total
}
