//! Ring layer: R_q = F_q[X]/(Phi_eta) with eta=81, Phi_eta = X^54 + X^27 + 1.
//! MUST: cf/cf^{-1}, ||a||_∞, rot(a) S-action on vectors/matrices; constant-time schoolbook mul.
//! SHOULD: fast-mul hook (API-level; can be swapped to NTT later).

use crate::norms::{NeoMathError, Norms};
use crate::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::DenseMatrix;
use std::ops::{Add, Mul, Sub};

/// Cyclotomic parameter eta and derived dimension d = deg(Phi_eta).
pub const ETA: usize = 81;
/// Degree d = 54 for Phi_{81}(X) = X^54 + X^27 + 1 (used throughout Neo).
pub const D: usize = 54;

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

    /// MUST: constant-time schoolbook mul with reduction mod Phi_{81}(X) = X^54 + X^27 + 1.
    /// No branches on secret data; loops run fixed D and 2D-1 iterations.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut tmp = [Fq::ZERO; 2 * D - 1];
        for i in 0..D {
            let ai = self.0[i];
            for j in 0..D {
                tmp[i + j] += ai * rhs.0[j];
            }
        }
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
            } else if new_deg < D + 27 {
                // X^new_deg = X^(new_deg-54) * X^54 = X^(new_deg-54) * (-X^27 - 1)
                let reduced_deg = new_deg - D;
                out[reduced_deg] -= self.0[i]; // -X^(new_deg-54)
                out[reduced_deg + 27] -= self.0[i]; // -X^(new_deg-54+27) = -X^(new_deg-27)
            } else {
                // new_deg >= D + 27, so new_deg - 27 >= D
                // X^new_deg = -X^(new_deg-27) - X^(new_deg-54)
                let deg1 = new_deg - 27;
                let deg2 = new_deg - D;
                if deg2 < D {
                    out[deg2] -= self.0[i];
                }
                if deg1 >= D {
                    // deg1 = new_deg - 27, need to reduce X^deg1 further
                    let deg1_red = deg1 - D;
                    if deg1_red < D {
                        out[deg1_red] += self.0[i]; // -(-X^(deg1_red)) = +X^(deg1_red)
                        if deg1_red + 27 < D {
                            out[deg1_red + 27] += self.0[i];
                        }
                    }
                } else {
                    out[deg1] -= self.0[i];
                }
            }
        }
        Self(out)
    }

    /// SHOULD: placeholder "fast" multiply; currently calls `mul`.
    #[inline]
    pub fn mul_fast(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }

    // Direct field-based methods (replacing ModInt backward compatibility)

    /// Create ring element from field coefficients
    pub fn from_field_coeffs(coeffs: Vec<Fq>) -> Self {
        let mut ring_coeffs = [Fq::ZERO; D];
        for (i, c) in coeffs.into_iter().enumerate().take(D) {
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

    /// Infinity norm over centered representatives (backward compatibility)
    pub fn norm_inf(&self) -> u64 {
        inf_norm(self) as u64
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
        let idx_27 = i - 27;
        if idx_27 < D {
            coeffs[idx_27] -= t; // -X^(i-27)
        } else {
            // idx_27 >= D, need recursive reduction
            coeffs[idx_27 - D] += t; // -(-X^(idx_27-54)) = +X^(idx_27-54)
            if idx_27 - 27 < D {
                coeffs[idx_27 - 27] += t; // -(-X^(idx_27-27)) = +X^(idx_27-27)
            }
        }
    }
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

/// SHOULD: left action on a dense d×m matrix (columns are vectors in F_q^d).
pub(crate) fn rot_apply_matrix(a: &Rq, z: &DenseMatrix<Fq>) -> Result<DenseMatrix<Fq>, NeoMathError> {
    let norms = Norms::default();
    norms.must(z.width > 0 && z.values.len() % z.width == 0, "matrix shape")?;
    let h = z.values.len() / z.width;
    norms.must(h == D, "matrix height must be d")?;
    let mut out = DenseMatrix::default(z.width, h);
    for col in 0..z.width {
        let mut colv = [Fq::ZERO; D];
        colv.iter_mut().enumerate().for_each(|(r, elem)| {
            *elem = z.values[r * z.width + col];
        });
        let newc = rot_apply_vec(a, &colv);
        newc.iter().enumerate().for_each(|(r, &val)| {
            out.values[r * z.width + col] = val;
        });
    }
    Ok(out)
}

// Implement arithmetic traits for backward compatibility
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
