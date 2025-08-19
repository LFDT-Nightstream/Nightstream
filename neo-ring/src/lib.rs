use neo_decomp::signed_decomp_b;
use neo_modint::Coeff;
use neo_poly::Polynomial;
use p3_matrix::Matrix;
#[cfg(test)]
use quickcheck::{Arbitrary, Gen};
use rand::distr::Uniform;
use rand::Rng;
use rand_distr::StandardNormal;
use subtle::{Choice, ConditionallySelectable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingElement<C: Coeff> {
    poly: Polynomial<C>,
    n: usize,
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
        let dist = Uniform::new_inclusive(-(bound as i128), bound as i128).unwrap();
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
                let x: f64 = rng.sample::<f64, _>(StandardNormal) * sigma;
                let z = x.round() as i128;
                let prob = ((x - z as f64).powi(2) / (2.0 * sigma.powi(2))).exp();
                let accept = Choice::from((rng.random::<f64>() < prob) as u8);
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

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use neo_decomp::reconstruct_decomp;
    use neo_modint::ModInt;
    use neo_poly::Polynomial;
    use p3_matrix::dense::RowMajorMatrix;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[test]
    fn add_mul_closure_modint() {
        let mut rng = rand::rng();
        let n = 4;
        for _ in 0..10 {
            let a = RingElement::<ModInt>::random_uniform(&mut rng, n);
            let b = RingElement::<ModInt>::random_uniform(&mut rng, n);
            let c = a.clone() * b.clone() + a.clone();
            assert!(c.coeffs().len() <= n);
        }
    }

    #[test]
    fn test_norm_inf() {
        let n = 2;
        let coeffs = vec![ModInt::from_u64(1), ModInt::from_u64(ModInt::modulus() - 1)];
        let re = RingElement::from_coeffs(coeffs, n);
        assert_eq!(re.norm_inf(), 1);
    }

    #[test]
    fn test_random_small() {
        let mut rng = rand::rng();
        let n = 4;
        let bound = 3;
        let re = RingElement::<ModInt>::random_small(&mut rng, n, bound);
        for &c in re.coeffs() {
            let val: i128 = c.into();
            assert!(val.abs() <= bound as i128);
        }
    }

    #[test]
    fn test_reduction_specific() {
        let n = 4;
        // X^4 should reduce to -1
        let poly = Polynomial::new(vec![
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(1),
        ]);
        let re = RingElement::new(poly, n);
        let expected = vec![ModInt::from_u64(ModInt::modulus() - 1)];
        assert_eq!(re.coeffs(), expected.as_slice());

        // X^5 should reduce to -X
        let poly_high = Polynomial::new(vec![
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(0),
            ModInt::from_u64(1),
        ]);
        let re_high = RingElement::new(poly_high, n);
        let expected_high = vec![ModInt::from_u64(0), ModInt::from_u64(ModInt::modulus() - 1)];
        assert_eq!(re_high.coeffs(), expected_high.as_slice());
    }

    #[test]
    fn test_zero_and_identity() {
        let n = 8;
        let zero = RingElement::<ModInt>::from_scalar(ModInt::zero(), n);
        let one = RingElement::<ModInt>::from_scalar(ModInt::one(), n);
        let a = RingElement::<ModInt>::random_uniform(&mut rand::rng(), n);

        assert_eq!(a.clone() + zero.clone(), a);
        assert_eq!(a.clone() * zero.clone(), zero);
        assert_eq!(a.clone() * one.clone(), a);
    }

    #[test]
    fn test_norm_edge_cases() {
        let n = 2;
        let coeffs_max = vec![
            ModInt::from_u64(ModInt::modulus() - 1),
            ModInt::from_u64(ModInt::modulus() - 1),
        ];
        let re_max = RingElement::from_coeffs(coeffs_max, n);
        assert_eq!(re_max.norm_inf(), 1);

        let coeffs_mixed = vec![
            ModInt::from_u64(1),
            ModInt::from_u64(ModInt::modulus() / 2 + 1),
        ];
        let re_mixed = RingElement::from_coeffs(coeffs_mixed, n);
        assert_eq!(re_mixed.norm_inf(), (ModInt::modulus() / 2));

        let zero = RingElement::<ModInt>::from_scalar(ModInt::zero(), n);
        assert_eq!(zero.norm_inf(), 0);
    }

    #[test]
    fn test_rotate() {
        let n = 4;
        let coeffs = vec![
            ModInt::from_u64(1),
            ModInt::from_u64(2),
            ModInt::from_u64(3),
            ModInt::from_u64(4),
        ];
        let re = RingElement::from_coeffs(coeffs, n);
        let rotated = re.rotate(1);
        let expected = vec![
            ModInt::from_u64(ModInt::modulus() - 4),
            ModInt::from_u64(1),
            ModInt::from_u64(2),
            ModInt::from_u64(3),
        ];
        assert_eq!(rotated.coeffs(), expected.as_slice());
    }

    #[test]
    fn test_automorphism() {
        let n = 4;
        let coeffs = vec![
            ModInt::from_u64(1),
            ModInt::from_u64(2),
            ModInt::from_u64(3),
            ModInt::from_u64(4),
        ];
        let re = RingElement::from_coeffs(coeffs, n);
        let auto = re.automorphism(3); // 3 is odd
        let expected = vec![
            ModInt::from_u64(1),
            ModInt::from_u64(4),
            ModInt::from_u64(ModInt::modulus() - 3),
            ModInt::from_u64(2),
        ];
        assert_eq!(auto.coeffs(), expected.as_slice());
    }

    #[test]
    fn test_decompose_coeffs() {
        let n = 2;
        let b = 3u64;
        let k = 4; // enough to represent the coefficients exactly
        let coeffs = vec![ModInt::from_u64(10), ModInt::from_u64(20)];
        let re = RingElement::from_coeffs(coeffs, n);
        let (decomp, g) = re.decompose_coeffs(b, k);
        let m = re.coeffs().len();
        let mut data = Vec::with_capacity(k * m);
        for layer in decomp.iter() {
            let mut v = layer.coeffs().to_vec();
            v.resize(m, ModInt::zero());
            data.extend_from_slice(&v);
        }
        let matrix = RowMajorMatrix::new(data, m);
        let recon_coeffs = reconstruct_decomp(&matrix, &g);
        let recon = RingElement::from_coeffs(recon_coeffs, n);
        assert_eq!(recon, re);
        for layer in &decomp {
            assert!(layer.norm_inf() <= b / 2);
        }
    }

    #[test]
    fn test_sampling_distribution() {
        let mut rng = rand::rng();
        let n = 16;
        let bound = 4;
        let samples: Vec<_> = (0..1000)
            .map(|_| RingElement::<ModInt>::random_small(&mut rng, n, bound))
            .collect();
        let max_sampled = samples.iter().map(|r| r.norm_inf()).max().unwrap();
        assert!(max_sampled <= bound);

        let mut counts = [0u32; 9];
        for re in samples.iter().take(1000) {
            for &c in re.coeffs() {
                let val: i128 = c.into();
                counts[(val + 4) as usize] += 1;
            }
        }
        for &count in &counts {
            assert!(count > 0, "Bias: zero count in bin");
        }
    }

    #[test]
    #[should_panic]
    fn test_mismatched_n_panics() {
        let a = RingElement::<ModInt>::random_uniform(&mut rand::rng(), 4);
        let b = RingElement::<ModInt>::random_uniform(&mut rand::rng(), 8);
        let _ = a + b;
    }

    #[test]
    fn test_large_n_mul() {
        let n = 64;
        let a = RingElement::<ModInt>::random_uniform(&mut rand::rng(), n);
        let b = RingElement::<ModInt>::random_uniform(&mut rand::rng(), n);
        let c = a * b;
        assert!(c.coeffs().len() <= n);
    }

    // Property-based tests using QuickCheck
    #[quickcheck]
    fn prop_add_associative(
        a: RingElement<ModInt>,
        b: RingElement<ModInt>,
        c: RingElement<ModInt>,
    ) -> TestResult {
        if a.n != b.n || b.n != c.n {
            return TestResult::discard();
        }
        TestResult::from_bool((a.clone() + b.clone()) + c.clone() == a.clone() + (b + c))
    }

    #[quickcheck]
    fn prop_mul_associative(
        a: RingElement<ModInt>,
        b: RingElement<ModInt>,
        c: RingElement<ModInt>,
    ) -> TestResult {
        if a.n != b.n || b.n != c.n {
            return TestResult::discard();
        }
        TestResult::from_bool((a.clone() * b.clone()) * c.clone() == a.clone() * (b * c))
    }

    #[quickcheck]
    fn prop_distributivity(
        a: RingElement<ModInt>,
        b: RingElement<ModInt>,
        c: RingElement<ModInt>,
    ) -> TestResult {
        if a.n != b.n || b.n != c.n {
            return TestResult::discard();
        }
        TestResult::from_bool(a.clone() * (b.clone() + c.clone()) == a.clone() * b.clone() + a * c)
    }

    #[quickcheck]
    fn prop_add_identity(a: RingElement<ModInt>) -> bool {
        let zero = RingElement::from_scalar(ModInt::zero(), a.n);
        a.clone() + zero.clone() == a && zero + a.clone() == a
    }

    #[quickcheck]
    fn prop_mul_identity(a: RingElement<ModInt>) -> bool {
        let one = RingElement::from_scalar(ModInt::one(), a.n);
        a.clone() * one.clone() == a && one * a.clone() == a
    }

    #[quickcheck]
    fn prop_reduction_invariant(poly: Polynomial<ModInt>, n: u8) -> bool {
        let n = (n as usize % 12) + 1;
        let re = RingElement::new(poly, n);
        re.coeffs().len() <= n
    }

    #[quickcheck]
    fn prop_norm_signed(a: RingElement<ModInt>) -> bool {
        let norm = a.norm_inf();
        a.coeffs().iter().all(|&c| {
            let val: i128 = c.into();
            let q = ModInt::modulus() as i128;
            let signed = if val > q / 2 { val - q } else { val };
            signed.abs() <= norm as i128
        })
    }

    #[quickcheck]
    fn prop_decompose_coeffs_reconstruct(re: RingElement<ModInt>) -> bool {
        let b = 3u64;
        let k = 45; // sufficiently large for 64-bit values
        let (decomp, g) = re.decompose_coeffs(b, k);
        let m = re.coeffs().len();
        let mut data = Vec::with_capacity(k * m);
        for layer in decomp.iter() {
            let mut v = layer.coeffs().to_vec();
            v.resize(m, ModInt::zero());
            data.extend_from_slice(&v);
        }
        let matrix = RowMajorMatrix::new(data, m);
        let recon = RingElement::from_coeffs(reconstruct_decomp(&matrix, &g), re.n);
        recon == re
    }
}
