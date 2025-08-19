use neo_fields::F;
use neo_modint::ModInt;
use neo_poly::*;
use p3_field::PrimeCharacteristicRing;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use rand::Rng;

#[test]
fn mul_and_eval() {
    let mut rng = rand::rng();
    let a0 = F::from_u64(rng.random());
    let a1 = F::from_u64(rng.random());
    let b0 = F::from_u64(rng.random());
    let b1 = F::from_u64(rng.random());

    let a = Polynomial::new(vec![a0, a1]);
    let b = Polynomial::new(vec![b0, b1]);
    let prod = a.clone() * b.clone();
    for _ in 0..10 {
        let x = F::from_u64(rng.random());
        assert_eq!(prod.eval(x), a.eval(x) * b.eval(x));
    }
}

#[test]
fn karatsuba_matches_naive() {
    let mut rng = rand::rng();
    let a: Vec<ModInt> = (0..130).map(|_| ModInt::random(&mut rng)).collect();
    let b: Vec<ModInt> = (0..130).map(|_| ModInt::random(&mut rng)).collect();
    let pa = Polynomial::new(a.clone());
    let pb = Polynomial::new(b.clone());
    let prod_fast = pa * pb;
    let prod_slow = Polynomial::new(naive_mul(&a, &b));
    assert_eq!(prod_fast, prod_slow);
}

#[test]
fn test_div_rem() {
    let p = Polynomial::new(vec![
        ModInt::from_u64(1),
        ModInt::from_u64(2),
        ModInt::from_u64(3),
        ModInt::from_u64(4),
    ]);
    let d = Polynomial::new(vec![ModInt::from_u64(1), ModInt::from_u64(1)]);
    let (q, r) = p.div_rem(&d);
    let expected_q = Polynomial::new(vec![
        ModInt::from_u64(3),
        ModInt::from(-1_i128),
        ModInt::from_u64(4),
    ]);
    let expected_r = Polynomial::new(vec![ModInt::from(-2_i128)]);
    assert_eq!(q, expected_q);
    assert_eq!(r, expected_r);
}

#[test]
fn test_interpolate() {
    let points = vec![
        ModInt::from_u64(0),
        ModInt::from_u64(1),
        ModInt::from_u64(2),
    ];
    let evals = vec![
        ModInt::from_u64(0),
        ModInt::from_u64(1),
        ModInt::from_u64(4),
    ];
    let poly = Polynomial::interpolate(&points, &evals);
    assert_eq!(poly.eval(ModInt::from_u64(3)), ModInt::from_u64(9));
}

#[quickcheck]
fn prop_add_associative(
    a: Polynomial<ModInt>,
    b: Polynomial<ModInt>,
    c: Polynomial<ModInt>,
) -> bool {
    (a.clone() + b.clone()) + c.clone() == a + (b + c)
}

#[quickcheck]
fn prop_mul_associative(
    a: Polynomial<ModInt>,
    b: Polynomial<ModInt>,
    c: Polynomial<ModInt>,
) -> bool {
    (a.clone() * b.clone()) * c.clone() == a * (b * c)
}

#[quickcheck]
fn prop_distributive(
    a: Polynomial<ModInt>,
    b: Polynomial<ModInt>,
    c: Polynomial<ModInt>,
) -> bool {
    a.clone() * (b.clone() + c.clone()) == a.clone() * b + a * c
}

#[quickcheck]
fn prop_div_rem(p: Polynomial<ModInt>, d: Polynomial<ModInt>) -> TestResult {
    if d.coeffs().is_empty() || d.coeffs().last().copied() == Some(ModInt::zero()) {
        return TestResult::discard();
    }
    let (q, r) = p.div_rem(&d);
    let recon = q * d.clone() + r.clone();
    if d.coeffs().len() == 1 {
        TestResult::from_bool(p == recon && r.coeffs().is_empty())
    } else {
        TestResult::from_bool(p == recon && r.degree() < d.degree())
    }
}
