//! Field helpers over any backend: constants and proof words from field values, extension
//! products, `K` and `K ⊗ Ext` values, eq tables and interpolation. Points are low bit first.
//!
//! `formal/compression-lean/CompressionLeanRust/Algebra.lean` proves each helper, run on the
//! Lean copies' `Build`, equal to its copy in `Ports/Circuit/Algebra.lean`.

use crate::backend::Backend;
use crate::field::{ext_of_gl, gl_inv, gl_mul, gl_sub, gl_word, reduce, Ext, Gl, EXT_ONE, EXT_ZERO, GL_ONE};

/// A constant from a field value.
pub fn constant<B: Backend>(b: &mut B, value: Gl) -> B::F {
    b.constant(gl_word(value))
}

/// A proof word from a field value.
pub fn private<B: Backend>(b: &mut B, value: Gl) -> B::F {
    b.private(gl_word(value))
}

/// `a·by` for a field constant.
pub fn scale<B: Backend>(b: &mut B, a: B::F, by: Gl) -> B::F {
    b.scale(a, gl_word(by))
}

/// An extension constant.
pub fn ext_constant<B: Backend>(b: &mut B, value: Ext) -> B::E {
    b.ext_constant([gl_word(value.c0), gl_word(value.c1), gl_word(value.c2)])
}

pub fn ext_zero<B: Backend>(b: &mut B) -> B::E {
    ext_constant(b, EXT_ZERO)
}

pub fn ext_one<B: Backend>(b: &mut B) -> B::E {
    ext_constant(b, EXT_ONE)
}

/// An extension proof value, as three proof words.
pub fn private_ext<B: Backend>(b: &mut B, value: Ext) -> B::E {
    let x = private(b, value.c0);
    let y = private(b, value.c1);
    let z = private(b, value.c2);
    b.ext([x, y, z])
}

/// A base word as an extension value.
pub fn lift_word<B: Backend>(b: &mut B, value: B::F) -> B::E {
    let zero = b.constant(0);
    b.ext([value, zero, zero])
}

pub fn ext_scale_constant<B: Backend>(b: &mut B, a: B::E, by: Gl) -> B::E {
    let by = constant(b, by);
    b.ext_scale(a, by)
}

/// `Σ_i x_i·y_i` over the shorter of `x` and `y`.
pub fn dot<B: Backend>(b: &mut B, x: &[B::E], y: &[B::E]) -> B::E {
    let mut total = ext_zero(b);
    let mut i = 0;
    while i < x.len() && i < y.len() {
        let term = b.ext_mul(x[i], y[i]);
        total = b.ext_add(total, term);
        i = i.wrapping_add(1);
    }
    total
}

/// `1 - x`.
pub fn one_minus<B: Backend>(b: &mut B, x: B::E) -> B::E {
    let one = ext_one(b);
    b.ext_sub(one, x)
}

/// `eq(x, y) = x·y + (1 - x)(1 - y)` for one coordinate.
pub fn eq1<B: Backend>(b: &mut B, x: B::E, y: B::E) -> B::E {
    let both = b.ext_mul(x, y);
    let twice = b.ext_add(both, both);
    let one = ext_one(b);
    let shifted = b.ext_add(twice, one);
    let sum = b.ext_add(x, y);
    b.ext_sub(shifted, sum)
}

/// `Π_t eq(x_t, y_t)` over the shorter of `x` and `y`.
pub fn eq_eval<B: Backend>(b: &mut B, x: &[B::E], y: &[B::E]) -> B::E {
    let mut total = ext_one(b);
    let mut t = 0;
    while t < x.len() && t < y.len() {
        let factor = eq1(b, x[t], y[t]);
        total = b.ext_mul(total, factor);
        t = t.wrapping_add(1);
    }
    total
}

/// `eq(point, i)` for every `i < 2^point.len()`; index bit `t` pairs with `point[t]`.
pub fn eq_table<B: Backend>(b: &mut B, point: &[B::E]) -> Vec<B::E> {
    let mut table = Vec::from([ext_one(b)]);
    let mut t = 0;
    while t < point.len() {
        let x = point[t];
        let low = one_minus(b, x);
        let mut next = Vec::new();
        let mut i = 0;
        while i < table.len() {
            next.push(b.ext_mul(table[i], low));
            i = i.wrapping_add(1);
        }
        let mut i = 0;
        while i < table.len() {
            next.push(b.ext_mul(table[i], x));
            i = i.wrapping_add(1);
        }
        table = next;
        t = t.wrapping_add(1);
    }
    table
}

/// Bit `t` of `index`.
pub fn index_bit(index: usize, t: usize) -> bool {
    t < usize::BITS as usize && !index.wrapping_shr(t as u32).is_multiple_of(2)
}

/// `eq(point, index)` for a known `index`.
pub fn eq_at<B: Backend>(b: &mut B, point: &[B::E], index: usize) -> B::E {
    let mut total = ext_one(b);
    let mut t = 0;
    while t < point.len() {
        let factor = if index_bit(index, t) {
            point[t]
        } else {
            one_minus(b, point[t])
        };
        total = b.ext_mul(total, factor);
        t = t.wrapping_add(1);
    }
    total
}

/// The polynomial through `(i, values[i])` for `i < values.len()`, at `x`.
pub fn interpolate<B: Backend>(b: &mut B, values: &[B::E], x: B::E) -> B::E {
    let n = values.len();
    let mut total = ext_zero(b);
    let mut i = 0;
    while i < n {
        let mut numerator = ext_one(b);
        let mut denominator = GL_ONE;
        let mut j = 0;
        while j < n {
            if i != j {
                let node = ext_constant(b, ext_of_gl(reduce(j as u64)));
                let factor = b.ext_sub(x, node);
                numerator = b.ext_mul(numerator, factor);
                denominator = gl_mul(denominator, gl_sub(reduce(i as u64), reduce(j as u64)));
            }
            j = j.wrapping_add(1);
        }
        let weight = ext_scale_constant(b, numerator, gl_inv(denominator));
        let term = b.ext_mul(values[i], weight);
        total = b.ext_add(total, term);
        i = i.wrapping_add(1);
    }
    total
}

/// `Σ_i c_i x^i` by Horner over extension coefficients.
pub fn horner<B: Backend>(b: &mut B, coefficients: &[B::E], x: B::E) -> B::E {
    let mut total = ext_zero(b);
    let mut i = coefficients.len();
    while 0 < i {
        i = i.wrapping_sub(1);
        let scaled = b.ext_mul(total, x);
        total = b.ext_add(scaled, coefficients[i]);
    }
    total
}

/// A `K = F[u] / (u^2 - 7)` value: `[re, im]`.
pub type K<B> = [<B as Backend>::F; 2];

/// A `K ⊗ Ext = Ext[u] / (u^2 - 7)` value.
pub struct Kx<B: Backend> {
    pub re: B::E,
    pub im: B::E,
}

impl<B: Backend> Clone for Kx<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for Kx<B> {}

impl<B: Backend> Kx<B> {
    pub fn zero(b: &mut B) -> Self {
        let zero = ext_zero(b);
        Self { re: zero, im: zero }
    }

    pub fn one(b: &mut B) -> Self {
        let one = ext_one(b);
        let zero = ext_zero(b);
        Self { re: one, im: zero }
    }

    pub fn from_k(b: &mut B, value: K<B>) -> Self {
        let re = lift_word(b, value[0]);
        let im = lift_word(b, value[1]);
        Self { re, im }
    }

    pub fn add(self, b: &mut B, other: Self) -> Self {
        let re = b.ext_add(self.re, other.re);
        let im = b.ext_add(self.im, other.im);
        Self { re, im }
    }

    pub fn mul(self, b: &mut B, other: Self) -> Self {
        let rr = b.ext_mul(self.re, other.re);
        let ii = b.ext_mul(self.im, other.im);
        let seven = ext_scale_constant(b, ii, reduce(7));
        let ri = b.ext_mul(self.re, other.im);
        let ir = b.ext_mul(self.im, other.re);
        let re = b.ext_add(rr, seven);
        let im = b.ext_add(ri, ir);
        Self { re, im }
    }

    pub fn scale(self, b: &mut B, by: B::E) -> Self {
        let re = b.ext_mul(self.re, by);
        let im = b.ext_mul(self.im, by);
        Self { re, im }
    }

    /// `1 - self`.
    pub fn one_minus(self, b: &mut B) -> Self {
        let re = one_minus(b, self.re);
        let zero = ext_zero(b);
        let im = b.ext_sub(zero, self.im);
        Self { re, im }
    }
}

/// `generator^index` from the little-endian bits of `index`.
pub fn power_from_bits<B: Backend>(b: &mut B, generator: Gl, bits: &[B::F]) -> B::F {
    let mut total = b.constant(1);
    let mut base = generator;
    let mut i = 0;
    while i < bits.len() {
        let one = b.constant(1);
        let factor_minus = scale(b, bits[i], gl_sub(base, GL_ONE));
        let factor = b.add(one, factor_minus);
        total = b.mul(total, factor);
        base = gl_mul(base, base);
        i = i.wrapping_add(1);
    }
    total
}
