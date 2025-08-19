//! Simple polynomial arithmetic over generic coefficient types.

pub use neo_modint::Coeff;
#[cfg(feature = "quickcheck")]
use quickcheck::{Arbitrary, Gen};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial<C: Coeff> {
    coeffs: Vec<C>,
}

impl<C: Coeff> Polynomial<C> {
    /// Create a polynomial from the given coefficients (low degree first).
    pub fn new(mut coeffs: Vec<C>) -> Self {
        while coeffs.last().is_some_and(|c| *c == C::zero()) {
            coeffs.pop();
        }
        Self { coeffs }
    }

    /// Degree of the polynomial (0 for constant zero).
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn coeffs(&self) -> &[C] {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut Vec<C> {
        &mut self.coeffs
    }

    /// Evaluate the polynomial at `x`.
    pub fn eval(&self, x: C) -> C {
        let mut acc = C::zero();
        for coeff in self.coeffs.iter().rev() {
            acc = acc * x + *coeff;
        }
        acc
    }

    /// Polynomial division: returns (quotient, remainder) such that self = divisor * quotient + remainder, deg(remainder) < deg(divisor).
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        if divisor.coeffs.is_empty() || (divisor.degree() == 0 && divisor.coeffs[0] == C::zero()) {
            panic!("division by zero polynomial");
        }
        let lead_d = *divisor.coeffs.last().unwrap();
        if lead_d == C::zero() {
            panic!("divisor has zero leading coefficient");
        }
        let lead_inv = lead_d.inverse();
        if self.degree() < divisor.degree() {
            return (Self::new(vec![]), self.clone());
        }
        let mut quotient = vec![C::zero(); self.degree() - divisor.degree() + 1];
        let mut remainder = self.coeffs.clone();
        while remainder.len() >= divisor.coeffs.len() {
            let lead_r = *remainder.last().unwrap();
            let factor = lead_r * lead_inv;
            let q_deg = remainder.len() - divisor.coeffs.len();
            quotient[q_deg] = factor;
            for (j, &d) in divisor.coeffs.iter().enumerate() {
                let idx = q_deg + j;
                remainder[idx] -= factor * d;
            }
            while remainder.last().copied() == Some(C::zero()) && !remainder.is_empty() {
                remainder.pop();
            }
        }
        (Self::new(quotient), Self::new(remainder))
    }

    /// Interpolate a polynomial from provided points and evaluations using naive Lagrange.
    pub fn interpolate(points: &[C], evals: &[C]) -> Self {
        assert_eq!(points.len(), evals.len());
        let n = points.len();
        let mut poly = Self::new(vec![C::zero()]);
        for i in 0..n {
            let mut basis = Self::new(vec![C::one()]);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let denom = points[i] - points[j];
                let denom_inv = denom.inverse();
                let lin = Self::new(vec![-points[j] * denom_inv, denom_inv]);
                basis = basis * lin;
            }
            let scaled = basis * Self::new(vec![evals[i]]);
            poly = poly + scaled;
        }
        poly
    }
}

impl<C: Coeff> std::ops::Add for Polynomial<C> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        self.coeffs.resize(len, C::zero());
        for (i, &b) in rhs.coeffs.iter().enumerate() {
            self.coeffs[i] += b;
        }
        Self::new(self.coeffs)
    }
}

impl<C: Coeff> std::ops::Sub for Polynomial<C> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        let len = self.coeffs.len().max(rhs.coeffs.len());
        self.coeffs.resize(len, C::zero());
        for (i, &b) in rhs.coeffs.iter().enumerate() {
            self.coeffs[i] -= b;
        }
        Self::new(self.coeffs)
    }
}

pub fn naive_mul<C: Coeff>(a: &[C], b: &[C]) -> Vec<C> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![C::zero(); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn add_slices<C: Coeff>(a: &[C], b: &[C]) -> Vec<C> {
    let n = a.len().max(b.len());
    let mut out = vec![C::zero(); n];
    for (i, &ai) in a.iter().enumerate() {
        out[i] = ai;
    }
    for (i, &bi) in b.iter().enumerate() {
        out[i] += bi;
    }
    out
}

pub fn karatsuba_mul<C: Coeff>(a: &[C], b: &[C]) -> Vec<C> {
    let n = a.len().max(b.len());
    if n <= 64 {
        return naive_mul(a, b);
    }
    let m = n / 2;
    let (a0, a1) = if a.len() > m { (&a[..m], &a[m..]) } else { (a, &[][..]) };
    let (b0, b1) = if b.len() > m { (&b[..m], &b[m..]) } else { (b, &[][..]) };

    let p0 = karatsuba_mul(a0, b0);
    let p2 = karatsuba_mul(a1, b1);
    let a01 = add_slices(a0, a1);
    let b01 = add_slices(b0, b1);
    let mut p1 = karatsuba_mul(&a01, &b01);
    let max_sub_len = p0.len().max(p2.len());
    if p1.len() < max_sub_len {
        p1.resize(max_sub_len, C::zero());
    }
    p1.iter_mut().zip(&p0).for_each(|(p, &o)| *p -= o);
    p1.iter_mut().zip(&p2).for_each(|(p, &o)| *p -= o);
    while p1.last() == Some(&C::zero()) {
        p1.pop();
    }

    let mut res = vec![C::zero(); a.len() + b.len() - 1];
    for (i, &c) in p0.iter().enumerate() {
        res[i] += c;
    }
    for (i, &c) in p1.iter().enumerate() {
        if i + m >= res.len() {
            res.resize(i + m + 1, C::zero());
        }
        res[i + m] += c;
    }
    for (i, &c) in p2.iter().enumerate() {
        if i + 2 * m >= res.len() {
            res.resize(i + 2 * m + 1, C::zero());
        }
        res[i + 2 * m] += c;
    }
    res
}

impl<C: Coeff> std::ops::Mul for Polynomial<C> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let coeffs = karatsuba_mul(&self.coeffs, &rhs.coeffs);
        Self::new(coeffs)
    }
}

#[cfg(feature = "quickcheck")]
impl<C: Coeff + Arbitrary> Arbitrary for Polynomial<C> {
    fn arbitrary(g: &mut Gen) -> Self {
        let coeffs: Vec<C> = Arbitrary::arbitrary(g);
        Polynomial::new(coeffs)
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(std::iter::empty())
    }
}