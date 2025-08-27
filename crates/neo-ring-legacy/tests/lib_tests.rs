use neo_decomp::reconstruct_decomp;
use neo_modint::{Coeff, ModInt};
use neo_poly::Polynomial;
use neo_ring::*;
use p3_matrix::dense::RowMajorMatrix;

#[cfg(feature = "quickcheck")]
use quickcheck::TestResult;
#[cfg(feature = "quickcheck")]
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

#[cfg(feature = "quickcheck")]
mod quickcheck_tests {
    use super::*;

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
