use neo_math::ring::{cf, Rq};
use neo_math::{Fq, D};
use p3_field::PrimeCharacteristicRing;

#[test]
fn mul_by_monomial_is_consistent() {
    let mut a = [Fq::ZERO; D];
    for i in 0..D {
        a[i] = Fq::from_u64(i as u64 + 1);
    }
    let a = Rq(a);

    for j in 0..82 {
        let xj = {
            let mut x = Rq::one();
            for _ in 0..j {
                x = x.mul_by_monomial(1);
            }
            x
        };
        let ref_mul = a.mul(&xj);
        let fast = a.mul_by_monomial(j);
        assert_eq!(cf(ref_mul), cf(fast), "Monomial multiplication inconsistent at j={}", j);
    }

    for &j in &[0, 1, 27, 53, 54, 55, 81] {
        let xj = {
            let mut x = Rq::one();
            for _ in 0..j {
                x = x.mul_by_monomial(1);
            }
            x
        };
        let ref_mul = a.mul(&xj);
        let fast = a.mul_by_monomial(j);
        assert_eq!(cf(ref_mul), cf(fast), "Monomial multiplication inconsistent at j={}", j);
    }
}

#[test]
fn mul_by_monomial_zero_is_identity() {
    let mut a = [Fq::ZERO; D];
    for i in 0..D {
        a[i] = Fq::from_u64(i as u64 + 42);
    }
    let a = Rq(a);

    let result = a.mul_by_monomial(0);
    assert_eq!(cf(a), cf(result), "mul_by_monomial(0) should be identity");
}

#[test]
fn cyclotomic_reduction_consistency() {
    let mut x_power_d = [Fq::ZERO; D];
    x_power_d[0] = Fq::ONE;
    let x_power_d = Rq(x_power_d);

    let result_x_d = x_power_d.mul_by_monomial(D);

    let mut expected = [Fq::ZERO; D];
    expected[0] = -Fq::ONE;
    expected[27] = -Fq::ONE;
    let expected = Rq(expected);

    assert_eq!(cf(result_x_d), cf(expected), "X^54 should reduce to -X^27 - 1");

    let one = Rq::one();

    let x_0 = one.mul_by_monomial(0);
    assert_eq!(cf(x_0), cf(one), "X^0 should be identity");

    for power in [1, 27, 53, 54, 81] {
        let via_monomial = one.mul_by_monomial(power);

        let mut via_multiplication = Rq::one();
        let x = {
            let mut x_coeffs = [Fq::ZERO; D];
            x_coeffs[1] = Fq::ONE;
            Rq(x_coeffs)
        };

        for _ in 0..power {
            via_multiplication = via_multiplication.mul(&x);
        }

        assert_eq!(
            cf(via_monomial),
            cf(via_multiplication),
            "Monomial and repeated multiplication should be equivalent for X^{}",
            power
        );
    }
}
