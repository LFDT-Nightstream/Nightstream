use neo_poly::{karatsuba_mul, Coeff};
use neo_modint::ModInt;
use neo_poly::Polynomial;
use neo_fields::{from_base, ExtF, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_karatsuba_in_place_no_alloc_panic_large() {
    let a = vec![ModInt::one(); 1024];
    let b = vec![ModInt::one(); 1024];
    let _ = karatsuba_mul(&a, &b);
}

#[test]
fn test_interpolate_large() {
    let n = 64;
    let points: Vec<ExtF> = (0..n)
        .map(|i| from_base(F::from_u64(i as u64)))
        .collect();
    let evals: Vec<ExtF> = points.iter().map(|&p| p * p).collect();
    let poly = Polynomial::interpolate(&points, &evals);
    let x = from_base(F::from_u64(n as u64));
    assert_eq!(poly.eval(x), x * x);
}
