//! Ring layer: R_q = F_q\[X\]/(Phi_eta) with eta=81, Phi_eta = X^54 + X^27 + 1.
//! MUST: cf/cf^{-1}, ||a||_∞, rot(a) S-action on vectors; constant-time schoolbook mul.

use crate::Fq;
use neo_params::goldilocks_paper_b2;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use std::ops::{Add, Mul, Sub};
use std::sync::OnceLock;

/// Cyclotomic parameter eta and derived dimension d = deg(Phi_eta).
pub const ETA: usize = goldilocks_paper_b2::ETA;
/// Degree d = 54 for Phi_{81}(X) = X^54 + X^27 + 1 (used throughout Neo).
pub const D: usize = goldilocks_paper_b2::D;
pub const PHI_MID_DEGREE: usize = goldilocks_paper_b2::PHI_MID_DEGREE;

/// A ring element a(X) ∈ R_q is represented by its coefficient vector (length D).
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

    /// MUST: constant-time coefficient-wise add.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = [Fq::ZERO; D];
        for (out_elem, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *out_elem = a + b;
        }
        Self(out)
    }

    /// MUST: constant-time coefficient-wise sub.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = [Fq::ZERO; D];
        for (out_elem, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *out_elem = a - b;
        }
        Self(out)
    }

    /// MUST: constant-time 3-way split mul with reduction mod Phi_{81}(X) = X^54 + X^27 + 1.
    /// No branches on secret data; loops run fixed D and fixed split sizes.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut tmp = mul_3way_karatsuba_54(&self.0, &rhs.0);
        reduce_mod_phi_81(&mut tmp);
        let mut out = [Fq::ZERO; D];
        out.copy_from_slice(&tmp[0..D]);
        Self(out)
    }

    /// Multiply by monomial X^j mod Phi_81 (fast rotation)
    pub fn mul_by_monomial(&self, j: usize) -> Self {
        if j == 0 {
            return *self;
        }

        let mut out = [Fq::ZERO; D];
        for i in 0..D {
            let new_deg = i + j;
            if new_deg < D {
                out[new_deg] = self.0[i];
            } else if new_deg < D + PHI_MID_DEGREE {
                // X^new_deg = X^(new_deg-54) * X^54 = X^(new_deg-54) * (-X^27 - 1)
                let reduced_deg = new_deg - D;
                out[reduced_deg] -= self.0[i]; // -X^(new_deg-54)
                out[reduced_deg + PHI_MID_DEGREE] -= self.0[i]; // -X^(new_deg-54+27) = -X^(new_deg-27)
            } else {
                // new_deg >= D + 27, so new_deg - 27 >= D
                // X^new_deg = -X^(new_deg-27) - X^(new_deg-54)
                let deg1 = new_deg - PHI_MID_DEGREE;
                let deg2 = new_deg - D;
                if deg2 < D {
                    out[deg2] -= self.0[i];
                }
                if deg1 >= D {
                    // deg1 = new_deg - 27, need to reduce X^deg1 further
                    let deg1_red = deg1 - D;
                    if deg1_red < D {
                        out[deg1_red] += self.0[i]; // -(-X^(deg1_red)) = +X^(deg1_red)
                        if deg1_red + PHI_MID_DEGREE < D {
                            out[deg1_red + PHI_MID_DEGREE] += self.0[i];
                        }
                    }
                } else {
                    out[deg1] -= self.0[i];
                }
            }
        }
        Self(out)
    }

    /// Create ring element from field coefficients
    pub fn from_field_coeffs(coeffs: &[Fq]) -> Self {
        let mut ring_coeffs = [Fq::ZERO; D];
        for (i, &c) in coeffs.iter().enumerate().take(D) {
            ring_coeffs[i] = c;
        }
        Self(ring_coeffs)
    }

    /// Create ring element from scalar field element
    pub fn from_field_scalar(scalar: Fq) -> Self {
        let mut ring_coeffs = [Fq::ZERO; D];
        ring_coeffs[0] = scalar;
        Self(ring_coeffs)
    }

    /// Get coefficients as field element vector
    pub fn field_coeffs(&self) -> Vec<Fq> {
        self.0.to_vec()
    }

    /// Random ring element with small coefficients
    pub fn random_small(rng: &mut impl rand::Rng, bound: u64) -> Self {
        let mut coeffs = [Fq::ZERO; D];
        coeffs.iter_mut().for_each(|c| {
            let val = rng.random_range(0..=bound);
            *c = Fq::from_u64(val);
        });
        Self(coeffs)
    }

    /// Random ring element (uniform over field elements)
    pub fn random_uniform(rng: &mut impl rand::Rng) -> Self {
        let mut coeffs = [Fq::ZERO; D];
        coeffs.iter_mut().for_each(|c| {
            *c = Fq::from_u64(rng.random::<u64>());
        });
        Self(coeffs)
    }

    /// Infinity norm over centered representatives.
    pub fn norm_inf(&self) -> u64 {
        let n = inf_norm(self);
        debug_assert!(n <= u64::MAX as u128, "inf_norm exceeds u64 range");
        n as u64
    }

    /// Pay-per-bit multiplication by sparse vector (Neo's key optimization)
    /// Only processes set bits, avoiding full O(d^2) when input is sparse
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

/// Constant-term extraction `ct: R_q -> F_q`.
#[inline]
pub fn ct(a: &Rq) -> Fq {
    a.0[0]
}

/// SuperNeo inner-product transform matrix `M` for `bar(v) = M * v`.
///
/// This matrix is built once (deterministically) as the inverse of the constant-term
/// Gram matrix `G`, where `G[i,j] = ct(X^i * X^j mod Φ_81)`.
///
/// The resulting invariant is:
/// `ct(cf_inv(superneo_bar_block(a)) * cf_inv(b)) == <a, b>`.
pub fn superneo_bar_matrix() -> &'static [[Fq; D]; D] {
    static M: OnceLock<[[Fq; D]; D]> = OnceLock::new();
    M.get_or_init(build_superneo_bar_matrix)
}

/// Apply the SuperNeo `bar` transform to one `d`-coefficient block.
#[inline]
pub fn superneo_bar_block(v: [Fq; D]) -> [Fq; D] {
    let m = superneo_bar_matrix();
    let mut out = [Fq::ZERO; D];
    for row in 0..D {
        let mut acc = Fq::ZERO;
        for col in 0..D {
            acc += m[row][col] * v[col];
        }
        out[row] = acc;
    }
    out
}

/// Apply the SuperNeo `bar` transform block-wise over a field vector.
///
/// Panics if `v.len()` is not a multiple of `D`.
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

/// Reduce polynomial in-place modulo Φ₈₁(X) = X^54 + X^27 + 1.
///
/// **Internal implementation detail** - not part of the public API.
///
/// **Precondition**: `coeffs` holds coefficients for degrees 0..(2*D-2) with D=54.
///
/// Implements the cyclotomic reduction X^i ≡ -X^(i-54) - X^(i-27) for i≥54
/// in a single downward pass, avoiding double-counting corner cases.
///
/// This is specific to η=81, giving the 54th cyclotomic polynomial
/// Φ₈₁(X) = X^54 + X^27 + 1 = ∏(X - ζ₈₁^k) where gcd(k,81)=1.
pub(crate) fn reduce_mod_phi_81(coeffs: &mut [Fq; 2 * D - 1]) {
    for i in (D..(2 * D - 1)).rev() {
        let t = coeffs[i];
        coeffs[i] = Fq::ZERO;
        coeffs[i - D] -= t; // X^i = X^(i-54) * X^54 = X^(i-54) * (-X^27 - 1)
        let idx_27 = i - PHI_MID_DEGREE;
        if idx_27 < D {
            coeffs[idx_27] -= t; // -X^(i-27)
        } else {
            // idx_27 >= D, need recursive reduction
            coeffs[idx_27 - D] += t; // -(-X^(idx_27-54)) = +X^(idx_27-54)
            if idx_27 - PHI_MID_DEGREE < D {
                coeffs[idx_27 - PHI_MID_DEGREE] += t; // -(-X^(idx_27-27)) = +X^(idx_27-27)
            }
        }
    }
}

fn build_superneo_bar_matrix() -> [[Fq; D]; D] {
    let g = build_ct_gram_matrix();
    let m = invert_matrix(g).expect("ct Gram matrix must be invertible");

    // Internal sanity check: M^T G == I.
    let mut mtg = [[Fq::ZERO; D]; D];
    for i in 0..D {
        for j in 0..D {
            let mut acc = Fq::ZERO;
            for r in 0..D {
                acc += m[r][i] * g[r][j];
            }
            mtg[i][j] = acc;
        }
    }
    for (i, row) in mtg.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            let want = if i == j { Fq::ONE } else { Fq::ZERO };
            assert_eq!(*entry, want, "SuperNeo bar matrix sanity check failed at ({i},{j})");
        }
    }

    m
}

fn build_ct_gram_matrix() -> [[Fq; D]; D] {
    let mut g = [[Fq::ZERO; D]; D];
    let neg_one = Fq::ZERO - Fq::ONE;
    for i in 0..D {
        for j in 0..D {
            g[i][j] = match i + j {
                0 => Fq::ONE,
                D => neg_one,
                ETA => Fq::ONE,
                _ => Fq::ZERO,
            };
        }
    }
    g
}

fn invert_matrix(mut a: [[Fq; D]; D]) -> Option<[[Fq; D]; D]> {
    let mut inv = [[Fq::ZERO; D]; D];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = Fq::ONE;
    }

    for col in 0..D {
        let mut pivot = None;
        for (r, row) in a.iter().enumerate().skip(col) {
            if row[col] != Fq::ZERO {
                pivot = Some(r);
                break;
            }
        }
        let pivot = pivot?;
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
        }

        let piv_inv = a[col][col].inverse();
        for c in 0..D {
            a[col][c] *= piv_inv;
            inv[col][c] *= piv_inv;
        }

        for r in 0..D {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == Fq::ZERO {
                continue;
            }
            for c in 0..D {
                a[r][c] -= factor * a[col][c];
                inv[r][c] -= factor * inv[col][c];
            }
        }
    }

    Some(inv)
}

/// Test-only wrapper for reduce_mod_phi_81
/// Exposes the internal reduction function for testing cyclotomic properties
/// Available for both unit tests and integration tests
#[doc(hidden)]
pub fn test_reduce_mod_phi_81(coeffs: &mut [Fq; 2 * D - 1]) {
    reduce_mod_phi_81(coeffs);
}

/// MUST: coefficient embedding cf : R_q → F_q^d (just the coefficients).
#[inline]
pub fn cf(a: Rq) -> [Fq; D] {
    a.0
}

/// MUST: inverse map cf^{-1} : F_q^d → R_q.
#[inline]
pub fn cf_inv(v: [Fq; D]) -> Rq {
    Rq(v)
}

/// MUST: infinity norm ||a||_∞ := max_i |cf(a)_i| over centered reps.
/// (Uses u128 modulus; audit-friendly explicit modulus in `field`.)
pub fn inf_norm(a: &Rq) -> u128 {
    let p: u128 = crate::field::GOLDILOCKS_MODULUS;
    let half = (p - 1) / 2;
    let mut m = 0u128;
    for &c in a.0.iter() {
        let x = c.as_canonical_u64() as u128;
        let centered = if x <= half { x } else { p - x };
        if centered > m {
            m = centered;
        }
    }
    m
}

/// MUST: S-action "rot(a)" applied to a vector v ∈ F_q^d as cf(a * cf^{-1}(v)).
#[inline]
pub fn rot_apply_vec(a: &Rq, v: &[Fq; D]) -> [Fq; D] {
    let prod = a.mul(&cf_inv(*v));
    cf(prod)
}

// Arithmetic trait implementations
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

const KARATSUBA_SPLIT: usize = D / 3;
const KARATSUBA_CHUNK_OUT: usize = 2 * KARATSUBA_SPLIT - 1;
const KARATSUBA_TWO: Fq = Fq::TWO;
const KARATSUBA_FOUR: Fq = Fq::new(4);
const KARATSUBA_SIXTEEN: Fq = Fq::new(16);

#[inline]
fn inv_two() -> Fq {
    static INV_TWO: OnceLock<Fq> = OnceLock::new();
    *INV_TWO.get_or_init(|| Fq::from_u64(2).inverse())
}

#[inline]
fn inv_six() -> Fq {
    static INV_SIX: OnceLock<Fq> = OnceLock::new();
    *INV_SIX.get_or_init(|| Fq::from_u64(6).inverse())
}

#[inline]
fn add_chunk<const N: usize>(lhs: &[Fq; N], rhs: &[Fq; N]) -> [Fq; N] {
    let mut out = [Fq::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] + rhs[i];
    }
    out
}

#[inline]
fn sub_chunk<const N: usize>(lhs: &[Fq; N], rhs: &[Fq; N]) -> [Fq; N] {
    let mut out = [Fq::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] - rhs[i];
    }
    out
}

#[inline]
fn scale_chunk<const N: usize>(lhs: &[Fq; N], scale: Fq) -> [Fq; N] {
    let mut out = [Fq::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] * scale;
    }
    out
}

#[inline]
fn add_scaled_chunk<const N: usize>(lhs: &[Fq; N], rhs: &[Fq; N], scale: Fq) -> [Fq; N] {
    let mut out = [Fq::ZERO; N];
    for i in 0..N {
        out[i] = lhs[i] + rhs[i] * scale;
    }
    out
}

#[inline]
fn sub_assign_chunk<const N: usize>(dst: &mut [Fq; N], src: &[Fq; N]) {
    for i in 0..N {
        dst[i] -= src[i];
    }
}

#[inline]
fn add_assign_chunk_at<const N: usize>(dst: &mut [Fq; 2 * D - 1], offset: usize, src: &[Fq; N]) {
    for i in 0..N {
        dst[offset + i] += src[i];
    }
}

#[inline]
fn mul_schoolbook_chunk(lhs: &[Fq; KARATSUBA_SPLIT], rhs: &[Fq; KARATSUBA_SPLIT]) -> [Fq; KARATSUBA_CHUNK_OUT] {
    let mut out = [Fq::ZERO; KARATSUBA_CHUNK_OUT];
    for i in 0..KARATSUBA_SPLIT {
        let ai = lhs[i];
        for j in 0..KARATSUBA_SPLIT {
            out[i + j] += ai * rhs[j];
        }
    }
    out
}

#[inline]
fn mul_3way_karatsuba_54(lhs: &[Fq; D], rhs: &[Fq; D]) -> [Fq; 2 * D - 1] {
    let mut a0 = [Fq::ZERO; KARATSUBA_SPLIT];
    let mut a1 = [Fq::ZERO; KARATSUBA_SPLIT];
    let mut a2 = [Fq::ZERO; KARATSUBA_SPLIT];
    let mut b0 = [Fq::ZERO; KARATSUBA_SPLIT];
    let mut b1 = [Fq::ZERO; KARATSUBA_SPLIT];
    let mut b2 = [Fq::ZERO; KARATSUBA_SPLIT];
    a0.copy_from_slice(&lhs[0..KARATSUBA_SPLIT]);
    a1.copy_from_slice(&lhs[KARATSUBA_SPLIT..2 * KARATSUBA_SPLIT]);
    a2.copy_from_slice(&lhs[2 * KARATSUBA_SPLIT..3 * KARATSUBA_SPLIT]);
    b0.copy_from_slice(&rhs[0..KARATSUBA_SPLIT]);
    b1.copy_from_slice(&rhs[KARATSUBA_SPLIT..2 * KARATSUBA_SPLIT]);
    b2.copy_from_slice(&rhs[2 * KARATSUBA_SPLIT..3 * KARATSUBA_SPLIT]);

    let half = inv_two();
    let sixth = inv_six();

    let a01 = add_chunk(&a0, &a1);
    let b01 = add_chunk(&b0, &b1);
    let a012 = add_chunk(&a01, &a2);
    let b012 = add_chunk(&b01, &b2);
    let am1 = add_chunk(&sub_chunk(&a0, &a1), &a2);
    let bm1 = add_chunk(&sub_chunk(&b0, &b1), &b2);
    let a2eval = add_scaled_chunk(&add_scaled_chunk(&a0, &a1, KARATSUBA_TWO), &a2, KARATSUBA_FOUR);
    let b2eval = add_scaled_chunk(&add_scaled_chunk(&b0, &b1, KARATSUBA_TWO), &b2, KARATSUBA_FOUR);
    let p0 = mul_schoolbook_chunk(&a0, &b0);
    let p1 = mul_schoolbook_chunk(&a012, &b012);
    let pm1 = mul_schoolbook_chunk(&am1, &bm1);
    let p2 = mul_schoolbook_chunk(&a2eval, &b2eval);
    let p4 = mul_schoolbook_chunk(&a2, &b2);

    let c0 = p0;
    let c4 = p4;
    let mut c2 = scale_chunk(&add_chunk(&p1, &pm1), half);
    sub_assign_chunk(&mut c2, &c0);
    sub_assign_chunk(&mut c2, &c4);

    let s = scale_chunk(&sub_chunk(&p1, &pm1), half);

    let mut c3 = p2;
    sub_assign_chunk(&mut c3, &c0);
    for i in 0..KARATSUBA_CHUNK_OUT {
        c3[i] -= c2[i] * KARATSUBA_FOUR;
        c3[i] -= c4[i] * KARATSUBA_SIXTEEN;
    }
    c3 = scale_chunk(&sub_chunk(&c3, &scale_chunk(&s, KARATSUBA_TWO)), sixth);

    let mut c1 = s;
    sub_assign_chunk(&mut c1, &c3);

    let mut out = [Fq::ZERO; 2 * D - 1];
    add_assign_chunk_at(&mut out, 0, &c0);
    add_assign_chunk_at(&mut out, KARATSUBA_SPLIT, &c1);
    add_assign_chunk_at(&mut out, 2 * KARATSUBA_SPLIT, &c2);
    add_assign_chunk_at(&mut out, 3 * KARATSUBA_SPLIT, &c3);
    add_assign_chunk_at(&mut out, 4 * KARATSUBA_SPLIT, &c4);
    out
}
