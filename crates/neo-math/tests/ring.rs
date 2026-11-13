//! `R_q = F_q[X]/(X^64 + 1)` against a schoolbook reference.

use neo_math::ring::{inf_norm, rot_apply_vec};
use neo_math::{cf, cf_inv, ct, Fq, Rq, SAction, D, ETA};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// The product in `F_q[X]`, then `X^(D + i) = −X^i`.
fn reference_mul(a: &Rq, b: &Rq) -> Rq {
    let mut wide = [Fq::ZERO; 2 * D - 1];
    for i in 0..D {
        for j in 0..D {
            wide[i + j] += a.0[i] * b.0[j];
        }
    }
    Rq(core::array::from_fn(|i| {
        if i + D < wide.len() {
            wide[i] - wide[i + D]
        } else {
            wide[i]
        }
    }))
}

fn monomial(degree: usize) -> Rq {
    let mut coefficients = [Fq::ZERO; D];
    coefficients[degree] = Fq::ONE;
    Rq(coefficients)
}

fn elements() -> Vec<Rq> {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut out = vec![
        Rq::zero(),
        Rq::one(),
        Rq([Fq::NEG_ONE; D]),
        Rq([Fq::ONE; D]),
        monomial(1),
        monomial(D - 1),
        Rq(core::array::from_fn(|i| {
            if i % 2 == 0 {
                Fq::NEG_ONE
            } else {
                Fq::from_u64(1 << 63)
            }
        })),
    ];
    out.extend((0..12).map(|_| Rq::random_uniform(&mut rng)));
    out
}

#[test]
fn dimensions_match_the_128th_cyclotomic_ring() {
    assert_eq!((ETA, D), (128, 64));
}

#[test]
fn multiplication_matches_the_schoolbook_reference() {
    let elements = elements();
    for a in &elements {
        for b in &elements {
            assert_eq!(a.mul(b), reference_mul(a, b));
            assert_eq!(*a * *b, a.mul(b));
        }
    }
}

#[test]
fn x_to_the_64_is_minus_one() {
    let minus_one = Rq::zero() - Rq::one();
    assert_eq!(monomial(D / 2).mul(&monomial(D / 2)), minus_one);
    assert_eq!(monomial(1).mul(&monomial(D - 1)), minus_one);
    assert_eq!(Rq::one().mul_by_monomial(D), minus_one);
    assert_eq!(Rq::one().mul_by_monomial(ETA), Rq::one());
}

#[test]
fn ring_axioms_hold() {
    let elements = elements();
    for (i, a) in elements.iter().enumerate() {
        let b = &elements[(i + 3) % elements.len()];
        let c = &elements[(i + 8) % elements.len()];
        assert_eq!(a.mul(b), b.mul(a));
        assert_eq!(a.mul(b).mul(c), a.mul(&b.mul(c)));
        assert_eq!(a.mul(&(*b + *c)), a.mul(b) + a.mul(c));
        assert_eq!(a.mul(&Rq::one()), *a);
        assert_eq!(*a + *b - *b, *a);
    }
}

#[test]
fn monomial_multiplication_matches_repeated_multiplication_by_x() {
    for a in elements() {
        let mut expected = a;
        for j in 0..=2 * ETA + 1 {
            assert_eq!(a.mul_by_monomial(j), expected, "j={j}");
            if j < D {
                assert_eq!(a.mul(&monomial(j)), expected, "j={j}");
            }
            expected = reference_mul(&expected, &monomial(1));
        }
    }
}

#[test]
fn sparse_multiplication_matches_the_sum_of_monomial_products() {
    let mut rng = StdRng::seed_from_u64(7);
    for a in elements() {
        let bits: Vec<(usize, bool)> = (0..D).map(|i| (i, rng.random())).collect();
        let dense = Rq(core::array::from_fn(|i| Fq::from_bool(bits[i].1)));
        assert_eq!(a.mul_sparse_bits(&bits), a.mul(&dense));
    }
}

#[test]
fn rotation_matrix_columns_are_the_products_with_monomials() {
    for a in elements().into_iter().skip(7).take(3) {
        let matrix = SAction::from_ring(a).to_matrix();
        for j in 0..D {
            let column: [Fq; D] = core::array::from_fn(|i| matrix.get(i, j).unwrap());
            assert_eq!(column, cf(a.mul(&monomial(j))), "column {j}");
            assert_eq!(rot_apply_vec(&a, &cf(monomial(j))), column);
        }
    }
}

#[test]
fn coefficient_maps_and_constant_term_agree() {
    for a in elements() {
        assert_eq!(cf_inv(cf(a)), a);
        assert_eq!(ct(&a), cf(a)[0]);
        assert_eq!(Rq::from_field_coeffs(&a.field_coeffs()), a);
    }
    assert_eq!(cf(Rq::from_field_scalar(Fq::TWO))[0], Fq::TWO);
}

#[test]
fn infinity_norm_uses_centered_representatives() {
    assert_eq!(Rq::zero().norm_inf(), 0);
    assert_eq!(Rq([Fq::NEG_ONE; D]).norm_inf(), 1);
    let mut coefficients = [Fq::ONE; D];
    coefficients[D - 1] = -Fq::from_u64(1_000);
    assert_eq!(inf_norm(&Rq(coefficients)), 1_000);

    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..10 {
        let (a, b) = (Rq::random_small(&mut rng, 1_000), Rq::random_small(&mut rng, 1_000));
        assert!(a.norm_inf() <= 1_000);
        assert!((a + b).norm_inf() <= a.norm_inf() + b.norm_inf());
    }
}
