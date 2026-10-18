use neo_math::ring::rot_apply_vec;
use neo_math::{cf, cf_inv, Fq, Rq, SAction, D};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use rand::RngCore;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn ring_s_action_isomorphism() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..20 {
        let a = Rq::random_uniform(&mut rng);
        let b = Rq::random_uniform(&mut rng);

        let ab = a.mul(&b);
        let cf_ab = cf(ab);

        let cf_b = cf(b);
        let rot_a_cf_b = rot_apply_vec(&a, &cf_b);

        assert_eq!(cf_ab, rot_a_cf_b, "Ring isomorphism failed: cf(a*b) != rot(a)·cf(b)");
    }
}

#[test]
fn power_of_x_columns() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);
        let s_action = SAction::from_ring(a);
        let rot_a_matrix = s_action.to_matrix();

        for j in 0..D {
            let x_j = {
                let mut coeffs = [Fq::ZERO; D];
                coeffs[j] = Fq::ONE;
                cf_inv(coeffs)
            };
            let a_times_x_j = a.mul(&x_j);
            let cf_result = cf(a_times_x_j);

            let mut col_j = [Fq::ZERO; D];
            for i in 0..D {
                col_j[i] = rot_a_matrix.get(i, j).unwrap_or(Fq::ZERO);
            }

            assert_eq!(
                cf_result, col_j,
                "Power of X property failed: cf(a*X^{}) != column {} of rot(a)",
                j, j
            );
        }
    }
}

#[test]
fn monomial_multiplication_regression() {
    let mut coeffs = [Fq::ZERO; D];
    coeffs[0] = Fq::ONE;
    coeffs[1] = Fq::from_u64(2);
    let p = Rq::from_field_coeffs(&coeffs);

    let px = p.mul_by_monomial(1);
    let px_coeffs = px.field_coeffs();

    assert_eq!(px_coeffs[0], Fq::ZERO);
    assert_eq!(px_coeffs[1], Fq::ONE);
    assert_eq!(px_coeffs[2], Fq::from_u64(2));

    let px27 = p.mul_by_monomial(27);

    assert!(px27.field_coeffs().iter().any(|&c| c != Fq::ZERO));
}

#[test]
fn cyclotomic_phi_81_relation() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..10 {
        let p = Rq::random_uniform(&mut rng);

        let x54 = Rq::one().mul_by_monomial(54);

        let minus_x27_minus_1 = {
            let mut coeffs = [Fq::ZERO; D];
            coeffs[0] = -Fq::ONE;
            coeffs[27] = -Fq::ONE;
            Rq::from_field_coeffs(&coeffs)
        };

        assert_eq!(
            x54.field_coeffs(),
            minus_x27_minus_1.field_coeffs(),
            "X^54 must equal -X^27 - 1 in the cyclotomic ring"
        );

        let p_times_x54 = p.mul(&x54);
        let p_times_x27 = p.mul_by_monomial(27);
        let expected = Rq::zero() - p_times_x27 - p;

        assert_eq!(
            p_times_x54.field_coeffs(),
            expected.field_coeffs(),
            "p * X^54 must equal -p * X^27 - p for the cyclotomic relation"
        );
    }
}

#[test]
fn sparse_vs_dense_multiplication() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..10 {
        let a = Rq::random_small(&mut rng, 100);

        let mut sparse_bits = Vec::new();
        let mut dense_poly = Rq::zero();

        for i in 0..D {
            if (rng.next_u32() % 2) == 0 && sparse_bits.len() < 5 {
                sparse_bits.push((i, true));
                let mut x_i_coeffs = [Fq::ZERO; D];
                x_i_coeffs[i] = Fq::ONE;
                dense_poly = dense_poly + cf_inv(x_i_coeffs);
            }
        }

        if !sparse_bits.is_empty() {
            let sparse_result = a.mul_sparse_bits(&sparse_bits);
            let dense_result = a.mul(&dense_poly);

            assert_eq!(
                sparse_result.field_coeffs(),
                dense_result.field_coeffs(),
                "Sparse multiplication doesn't match dense multiplication"
            );
        }
    }
}

#[test]
fn ring_arithmetic_properties() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);
        let b = Rq::random_uniform(&mut rng);
        let c = Rq::random_uniform(&mut rng);

        let ab_c = a.mul(&b).mul(&c);
        let a_bc = a.mul(&b.mul(&c));
        assert_eq!(cf(ab_c), cf(a_bc), "Multiplication not associative");

        let bc = b + c;
        let a_bc_dist = a.mul(&bc);
        let ab_ac = a.mul(&b) + a.mul(&c);
        assert_eq!(
            cf(a_bc_dist),
            cf(ab_ac),
            "Multiplication not distributive over addition"
        );

        let ab = a.mul(&b);
        let ba = b.mul(&a);
        assert_eq!(cf(ab), cf(ba), "Multiplication not commutative");
    }
}

#[test]
fn infinity_norm_properties() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    assert_eq!(Rq::zero().norm_inf(), 0);

    assert_eq!(Rq::one().norm_inf(), 1);

    for _ in 0..10 {
        let a = Rq::random_small(&mut rng, 1000);
        let b = Rq::random_small(&mut rng, 1000);
        let sum = a + b;

        let norm_sum = sum.norm_inf();
        let norm_a = a.norm_inf();
        let norm_b = b.norm_inf();

        assert!(
            norm_sum <= norm_a + norm_b,
            "Triangle inequality violated: ||a+b|| = {} > {} + {} = ||a|| + ||b||",
            norm_sum,
            norm_a,
            norm_b
        );
    }
}

#[test]
fn rotation_step_vs_ring_multiply() {
    let mut rng = StdRng::seed_from_u64(0x5eed_u64);

    for _ in 0..10 {
        let a = Rq::random_uniform(&mut rng);

        let x_poly = {
            let mut coeffs = vec![Fq::ZERO; neo_math::D];
            coeffs[1] = Fq::ONE;
            Rq::from_field_coeffs(&coeffs)
        };
        let cf_x = cf(x_poly);
        let rot_a_cf_x = rot_apply_vec(&a, &cf_x);

        let a_times_x = a.mul_by_monomial(1);
        let cf_a_times_x = cf(a_times_x);

        assert_eq!(rot_a_cf_x, cf_a_times_x, "rot(a)·cf(X) should equal cf(a·X)");
    }

    println!("✅ Rotation step agrees with ring multiply by X");
}
