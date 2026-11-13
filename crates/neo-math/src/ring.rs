//! Ring layer: `R_q = F_q[X]/(X^64 + 1)`, the 128th cyclotomic ring.

use crate::balanced::to_balanced_i128;
use crate::Fq;
use neo_params::nightstream_p64_k16;
use p3_field::PrimeCharacteristicRing;
use std::array;
use std::ops::{Add, Mul, Sub};
use std::sync::OnceLock;

pub const ETA: usize = nightstream_p64_k16::ETA;
pub const D: usize = nightstream_p64_k16::D;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rq(pub [Fq; D]);

impl Rq {
    #[inline]
    pub fn zero() -> Self {
        Self([Fq::ZERO; D])
    }
    #[inline]
    pub fn one() -> Self {
        let mut c = [Fq::ZERO; D];
        c[0] = Fq::ONE;
        Self(c)
    }

    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = [Fq::ZERO; D];
        for (out_elem, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *out_elem = a + b;
        }
        Self(out)
    }

    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = [Fq::ZERO; D];
        for (out_elem, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *out_elem = a - b;
        }
        Self(out)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = (&self.0, &rhs.0);
        let products = |a: &[Fq], b: &[Fq]| Fq::sum_of_products(a.iter().copied().zip(b.iter().rev().copied()));
        // X^D = −1: the products with i + j = k + D enter coefficient k with a minus sign.
        Self(array::from_fn(|k| {
            products(&a[..=k], &b[..=k]) - products(&a[k + 1..], &b[k + 1..])
        }))
    }

    /// The product with `X^j`. `X^D = −1`, so `X^ETA = 1`.
    pub fn mul_by_monomial(&self, j: usize) -> Self {
        let mut out = [Fq::ZERO; D];
        for (i, &coefficient) in self.0.iter().enumerate() {
            let degree = i + j % ETA;
            out[degree % D] = if (degree / D).is_multiple_of(2) {
                coefficient
            } else {
                -coefficient
            };
        }
        Self(out)
    }

    pub fn from_field_coeffs(coeffs: &[Fq]) -> Self {
        let mut ring_coeffs = [Fq::ZERO; D];
        for (i, &c) in coeffs.iter().enumerate().take(D) {
            ring_coeffs[i] = c;
        }
        Self(ring_coeffs)
    }

    pub fn from_field_scalar(scalar: Fq) -> Self {
        let mut ring_coeffs = [Fq::ZERO; D];
        ring_coeffs[0] = scalar;
        Self(ring_coeffs)
    }

    pub fn field_coeffs(&self) -> Vec<Fq> {
        self.0.to_vec()
    }

    pub fn random_small(rng: &mut impl rand::Rng, bound: u64) -> Self {
        let mut coeffs = [Fq::ZERO; D];
        coeffs.iter_mut().for_each(|c| {
            let val = rng.random_range(0..=bound);
            *c = Fq::from_u64(val);
        });
        Self(coeffs)
    }

    pub fn random_uniform(rng: &mut impl rand::Rng) -> Self {
        let mut coeffs = [Fq::ZERO; D];
        coeffs.iter_mut().for_each(|c| {
            *c = Fq::from_u64(rng.random::<u64>());
        });
        Self(coeffs)
    }

    pub fn norm_inf(&self) -> u64 {
        let n = inf_norm(self);
        debug_assert!(n <= u64::MAX as u128, "inf_norm exceeds u64 range");
        n as u64
    }

    pub fn mul_sparse_bits(&self, bits: &[(usize, bool)]) -> Self {
        let mut result = Self::zero();
        for &(index, bit) in bits {
            if bit {
                let shifted = self.mul_by_monomial(index);
                result = result + shifted;
            }
        }
        result
    }
}

#[inline]
pub fn ct(a: &Rq) -> Fq {
    a.0[0]
}

/// The matrix of `superneo_bar_block`.
pub fn superneo_bar_matrix() -> &'static [[Fq; D]; D] {
    static M: OnceLock<[[Fq; D]; D]> = OnceLock::new();
    M.get_or_init(|| {
        let mut matrix = [[Fq::ZERO; D]; D];
        for column in 0..D {
            let mut basis = [Fq::ZERO; D];
            basis[column] = Fq::ONE;
            for (row, entry) in superneo_bar_block(basis).into_iter().enumerate() {
                matrix[row][column] = entry;
            }
        }
        matrix
    })
}

/// The transform of SuperNeo v1.2 Section 5 with `ct(a · bar(b)) = ⟨a, b⟩`.
/// In `F_q[X]/(X^D + 1)`, `ct(X^i · X^j)` is 1 for `i + j = 0`, −1 for
/// `i + j = D`, and 0 otherwise. So `bar(b) = b_0 − Σ_{j ≥ 1} b_{D−j} X^j`.
#[inline]
pub fn superneo_bar_block(v: [Fq; D]) -> [Fq; D] {
    array::from_fn(|j| if j == 0 { v[0] } else { -v[D - j] })
}

pub fn superneo_bar_vec(v: &[Fq]) -> Vec<Fq> {
    assert!(
        v.len().is_multiple_of(D),
        "superneo_bar_vec expects length multiple of D"
    );
    let mut out = vec![Fq::ZERO; v.len()];
    for (blk_idx, chunk) in v.chunks_exact(D).enumerate() {
        let mut block = [Fq::ZERO; D];
        block.copy_from_slice(chunk);
        let transformed = superneo_bar_block(block);
        let dst = &mut out[blk_idx * D..(blk_idx + 1) * D];
        dst.copy_from_slice(&transformed);
    }
    out
}

#[inline]
pub fn cf(a: Rq) -> [Fq; D] {
    a.0
}

#[inline]
pub fn cf_inv(v: [Fq; D]) -> Rq {
    Rq(v)
}

pub fn inf_norm(a: &Rq) -> u128 {
    a.0.iter()
        .map(|&c| to_balanced_i128(c).unsigned_abs())
        .max()
        .unwrap_or(0)
}

#[inline]
pub fn rot_apply_vec(a: &Rq, v: &[Fq; D]) -> [Fq; D] {
    let prod = a.mul(&cf_inv(*v));
    cf(prod)
}

impl Add for Rq {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::add(&self, &rhs)
    }
}

impl Sub for Rq {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(&self, &rhs)
    }
}

impl Mul for Rq {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(&self, &rhs)
    }
}
