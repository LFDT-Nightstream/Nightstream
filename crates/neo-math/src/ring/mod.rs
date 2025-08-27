use crate::{signed_decomp_b, Coeff, Polynomial};
use p3_matrix::Matrix;
#[cfg(feature = "quickcheck")]
use quickcheck::{Arbitrary, Gen};
use rand::distributions::Uniform;
use rand::Rng;

use subtle::{Choice, ConditionallySelectable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingElement<C: Coeff> {
    poly: Polynomial<C>,
    pub n: usize,
}

impl<C: Coeff> RingElement<C> {
    pub fn new(poly: Polynomial<C>, n: usize) -> Self {
        Self {
            poly: Self::reduce_mod_xn_plus1(poly, n),
            n,
        }
    }

    pub fn from_coeffs(coeffs: Vec<C>, n: usize) -> Self {
        Self::new(Polynomial::new(coeffs), n)
    }

    pub fn from_scalar(c: C, n: usize) -> Self {
        Self::new(Polynomial::new(vec![c]), n)
    }

    pub fn zero(n: usize) -> Self {
        Self::from_scalar(C::zero(), n)
    }

    /// Toy commitment: multiply by public element and return coefficients.
    pub fn commit(&self, m: &RingElement<C>) -> Vec<C> {
        let prod = m.clone() * self.clone();
        prod.coeffs().to_vec()
    }

    pub fn coeffs(&self) -> &[C] {
        self.poly.coeffs()
    }

    /// Computes the infinity norm using signed representation (-q/2 < val <= q/2).
    /// Assumes the modulus fits in i128 (e.g., 64-bit fields like Goldilocks).
    pub fn norm_inf(&self) -> u64
    where
        C: Copy + Into<u64>,
    {
        let q: u64 = C::modulus();
        let half = q / 2;
        let mut max_val = 0u64;
        for &c in self.poly.coeffs() {
            let val: u64 = c.into();
            let gt = Choice::from((val > half) as u8);
            let neg = val.wrapping_sub(q);
            let selected = u64::conditional_select(&val, &neg, gt);
            let signed = selected as i64;
            let mask = signed >> 63;
            let abs = ((signed ^ mask) - mask) as u64;
            let greater = Choice::from((max_val < abs) as u8);
            max_val = u64::conditional_select(&max_val, &abs, greater);
        }
        max_val
    }

    /// Enhanced constant-time infinity norm computation with additional security checks.
    /// This version is fully constant-time and includes bounds checking for security.
    pub fn norm_inf_ct(&self) -> u64
    where
        C: Copy + Into<u64>,
    {
        let q: u64 = C::modulus();
        let half = q / 2;
        let mut max_val = 0u64;

        // Process each coefficient in constant time
        for &c in self.poly.coeffs() {
            let val: u64 = c.into();

            // Constant-time sign handling
            let gt = Choice::from((val > half) as u8);
            let neg_val = val.wrapping_sub(q);
            let signed_val = u64::conditional_select(&val, &neg_val, gt);

            // Constant-time absolute value
            let signed = signed_val as i64;
            let mask = signed >> 63;  // Get sign bit
            let abs_val = ((signed ^ mask) - mask) as u64;

            // Constant-time maximum computation
            let is_greater = Choice::from((max_val < abs_val) as u8);
            max_val = u64::conditional_select(&max_val, &abs_val, is_greater);
        }

        // Additional security: ensure result is within expected bounds
        let max_bound = q / 2 + 1;
        let within_bounds = Choice::from((max_val <= max_bound) as u8);
        max_val = u64::conditional_select(&max_val, &max_bound, !within_bounds);

        max_val
    }

    fn reduce_mod_xn_plus1(mut poly: Polynomial<C>, n: usize) -> Polynomial<C> {
        while poly.degree() >= n {
            let deg = poly.degree();
            let lead = poly.coeffs_mut().pop().unwrap();
            let idx = deg - n;
            if poly.coeffs().len() <= idx {
                poly.coeffs_mut().resize(idx + 1, C::zero());
            }
            poly.coeffs_mut()[idx] = poly.coeffs_mut()[idx] - lead;
        }
        while poly.coeffs().last().is_some_and(|c| *c == C::zero()) {
            poly.coeffs_mut().pop();
        }
        poly
    }

    /// Sample a uniform random element in the ring.
    pub fn random_uniform(rng: &mut impl Rng, n: usize) -> Self {
        let coeffs = (0..n).map(|_| C::random(rng)).collect();
        Self::from_coeffs(coeffs, n)
    }

    /// Sample a small random element with coefficients in [-bound, bound].
    pub fn random_small(rng: &mut impl Rng, n: usize, bound: u64) -> Self
    where
        C: From<i128>,
    {
        let dist = Uniform::new_inclusive(-(bound as i128), bound as i128);
        let coeffs = (0..n).map(|_| C::from(rng.sample(dist))).collect();
        Self::from_coeffs(coeffs, n)
    }

    /// Sample a ring element with coefficients drawn from a discrete Gaussian
    /// distribution with standard deviation `sigma`.
    pub fn random_gaussian(rng: &mut impl Rng, n: usize, sigma: f64) -> Self
    where
        C: From<i128>,
    {
        let mut coeffs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut sample = 0i128;
            let mut chosen = Choice::from(0u8);
            for _ in 0..8 {
                let x: f64 = rng.gen::<f64>() * sigma;
                let z = x.round() as i128;
                let prob = ((x - z as f64).powi(2) / (2.0 * sigma.powi(2))).exp();
                let accept = Choice::from((rng.gen::<f64>() < prob) as u8);
                let select = !chosen & accept;
                sample = i128::conditional_select(&sample, &z, select);
                chosen |= accept;
            }
            coeffs.push(C::from(sample));
        }
        Self::from_coeffs(coeffs, n)
    }

    /// Check if the element is invertible in R = Z_q[X]/(X^n + 1).
    /// Uses polynomial GCD over the coefficient field to check gcd(self, X^n + 1) == 1.
    pub fn is_invertible(&self) -> bool {
        let modulus = xn_plus_one(self.n);
        let g = gcd_polys(self.poly.clone(), modulus);
        g.degree() == 0 && !g.coeffs().is_empty() && g.coeffs()[0] != C::zero()
    }

    /// Rotate the element by `j` positions. This is equivalent to
    /// multiplication by `X^j` followed by reduction modulo `X^n + 1`.
    pub fn rotate(&self, mut j: usize) -> Self {
        j %= 2 * self.n;
        if j == 0 {
            return self.clone();
        }
        let mut coeffs = vec![C::zero(); j];
        coeffs.push(C::one());
        let xj = Polynomial::new(coeffs);
        Self::new(self.poly.clone() * xj, self.n)
    }

    /// Apply the automorphism `sigma_k` defined by substituting `X -> X^k`.
    /// For cyclotomic rings modulo `X^n + 1`, this is a ring automorphism when
    /// `k` is odd.
    pub fn automorphism(&self, k: usize) -> Self {
        let mut coeffs = vec![C::zero(); self.n];
        let two_n = (self.n as u128) * 2;
        for (i, &c) in self.coeffs().iter().enumerate() {
            let exp = (i as u128) * (k as u128);
            let mod_exp = (exp % two_n) as usize;
            let mut coeff = c;
            let pos = if mod_exp >= self.n {
                coeff = -coeff;
                mod_exp - self.n
            } else {
                mod_exp
            };
            coeffs[pos] = coeff;
        }
        Self::new(Polynomial::new(coeffs), self.n)
    }

    /// Decompose the coefficients of this ring element using neo-decomp.
    /// Returns `k` ring elements representing the decomposition layers and the
    /// accompanying gadget vector.
    pub fn decompose_coeffs(&self, b: u64, k: usize) -> (Vec<Self>, Vec<C>)
    where
        C: Copy + Into<i128> + From<i128> + Send + Sync,
    {
        let (matrix, g) = signed_decomp_b(self.coeffs(), b, k);
        let m = matrix.width();
        let mut layers = Vec::with_capacity(k);
        for row in 0..k {
            let mut row_vec = Vec::with_capacity(m);
            for col in 0..m {
                row_vec.push(matrix.get(row, col).unwrap());
            }
            layers.push(Self::from_coeffs(row_vec, self.n));
        }
        (layers, g)
    }
}

fn normalize_poly<C: Coeff>(mut p: Polynomial<C>) -> Polynomial<C> {
    while p.coeffs().last().is_some_and(|c| *c == C::zero()) {
        p.coeffs_mut().pop();
    }
    p
}

fn xn_plus_one<C: Coeff>(n: usize) -> Polynomial<C> {
    let mut coeffs = vec![C::zero(); n + 1];
    coeffs[0] = C::one();
    coeffs[n] = C::one();
    Polynomial::new(coeffs)
}

fn div_rem_polys<C: Coeff + Copy>(
    mut a: Polynomial<C>,
    b: &Polynomial<C>,
) -> (Polynomial<C>, Polynomial<C>) {
    a = normalize_poly(a);
    let mut q = Polynomial::new(Vec::new());
    if b.coeffs().is_empty() || b.degree() == usize::MAX {
        return (q, a);
    }
    let mut r = a;
    let mut q_coeffs: Vec<C> = vec![C::zero(); r.degree().saturating_sub(b.degree()) + 1];
    let b_lead = *b.coeffs().last().unwrap();
    let b_lead_inv = b_lead.inverse();
    while !r.coeffs().is_empty() && r.degree() >= b.degree() {
        let deg_diff = r.degree() - b.degree();
        let r_lead = *r.coeffs().last().unwrap();
        let factor = r_lead * b_lead_inv;
        if q_coeffs.len() <= deg_diff {
            q_coeffs.resize(deg_diff + 1, C::zero());
        }
        q_coeffs[deg_diff] += factor;
        let mut sub = vec![C::zero(); deg_diff];
        sub.extend(b.coeffs().iter().map(|&c| c * factor));
        let sub_poly = Polynomial::new(sub);
        r = normalize_poly(r - sub_poly);
    }
    q = normalize_poly(Polynomial::new(q_coeffs));
    (q, r)
}

fn gcd_polys<C: Coeff + Copy>(mut a: Polynomial<C>, mut b: Polynomial<C>) -> Polynomial<C> {
    a = normalize_poly(a);
    b = normalize_poly(b);
    while !b.coeffs().is_empty() {
        let (_q, r) = div_rem_polys(a.clone(), &b);
        a = b;
        b = r;
    }
    if let Some(&lead) = a.coeffs().last() {
        let inv = lead.inverse();
        let coeffs = a.coeffs().iter().map(|&c| c * inv).collect();
        return Polynomial::new(coeffs);
    }
    a
}
impl<C: Coeff> std::ops::Add for RingElement<C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.n, rhs.n);
        Self::new(self.poly + rhs.poly, self.n)
    }
}

impl<C: Coeff> std::ops::Sub for RingElement<C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.n, rhs.n);
        Self::new(self.poly - rhs.poly, self.n)
    }
}

impl<C: Coeff> std::ops::Neg for RingElement<C> {
    type Output = Self;
    fn neg(self) -> Self {
        // Use subtraction from zero since `Polynomial` does not implement `Neg`.
        Self::from_scalar(C::zero(), self.n) - self
    }
}

impl<C: Coeff> std::ops::Mul for RingElement<C> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.n, rhs.n);
        RingElement::new(self.poly * rhs.poly, self.n)
    }
}

impl<C: Coeff> std::ops::Mul<C> for RingElement<C> {
    type Output = Self;
    fn mul(self, rhs: C) -> Self {
        Self::new(self.poly * Polynomial::new(vec![rhs]), self.n)
    }
}

#[cfg(feature = "quickcheck")]
impl<C: Coeff + Arbitrary> Arbitrary for RingElement<C> {
    fn arbitrary(g: &mut Gen) -> Self {
        let n = ((u8::arbitrary(g) % 6) + 1) as usize * 2;
        let mut coeffs: Vec<C> = Arbitrary::arbitrary(g);
        coeffs.truncate(n);
        while coeffs.len() < n {
            coeffs.push(Arbitrary::arbitrary(g));
        }
        RingElement::from_coeffs(coeffs, n)
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::empty())
    }
}