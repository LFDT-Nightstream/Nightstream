use neo_math::{cf, Fq, Rq, SAction, D};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

#[test]
fn test_rq_s_isomorphism() {
    let a = test_ring_element(42);
    let b = test_ring_element(73);

    let rot_a = SAction::from_ring(a);
    let v_b = cf(b);

    let lhs = rot_a.apply_vec(&v_b);
    let rhs = cf(a.mul(&b));

    assert_eq!(lhs, rhs, "R_q ≅ S isomorphism failed: rot(a) * cf(b) ≠ cf(a * b)");
}

#[test]
fn test_rotation_matrix_columns() {
    let a = test_ring_element(123);
    let rot_a = SAction::from_ring(a);
    let matrix = rot_a.to_matrix();

    let mut x_power = Rq::one();
    for j in 0..D {
        let expected_col = cf(a.mul(&x_power));
        expected_col.iter().enumerate().for_each(|(i, &expected)| {
            let matrix_val = matrix.get(i, j).expect("Matrix index should be valid");
            assert_eq!(
                matrix_val, expected,
                "Column {j} row {i} mismatch: got {matrix_val:?}, expected {expected:?}"
            );
        });
        x_power = x_power.mul_by_monomial(1);
    }
}

#[test]
fn test_monomial_multiplication() {
    let a = test_ring_element(456);
    let x = monomial_x();

    let lhs = x.mul(&a);
    let rhs = a.mul_by_monomial(1);

    assert_eq!(lhs, rhs, "X * a ≠ a.mul_by_monomial(1)");
}

#[test]
fn test_cyclotomic_reduction_identity() {
    let x54 = monomial_x_power(54);
    let neg_x27_minus_1 = neg_x27_minus_one();

    assert_eq!(x54, neg_x27_minus_1, "X^54 ≢ -X^27 - 1 mod Phi_81");
}

#[test]
fn test_pay_per_bit_sparse() {
    let a = test_ring_element(789);

    let sparse_bits = vec![(0, true), (3, true), (7, true)];
    let sparse_result = a.mul_sparse_bits(&sparse_bits);

    let full_result = a.add(&a.mul_by_monomial(3)).add(&a.mul_by_monomial(7));

    assert_eq!(sparse_result, full_result, "Sparse bit multiplication failed");
}

fn test_ring_element(seed: u64) -> Rq {
    let mut coeffs = [Fq::ZERO; D];
    let mut x = seed;
    coeffs.iter_mut().for_each(|elem| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        *elem = Fq::from_u64(x % 1000);
    });
    Rq(coeffs)
}

fn monomial_x() -> Rq {
    let mut coeffs = [Fq::ZERO; D];
    coeffs[1] = Fq::ONE;
    Rq(coeffs)
}

fn monomial_x_power(power: usize) -> Rq {
    if power == 0 {
        return Rq::one();
    }
    Rq::one().mul_by_monomial(power)
}

fn neg_x27_minus_one() -> Rq {
    let mut coeffs = [Fq::ZERO; D];
    coeffs[0] = -Fq::ONE;
    coeffs[27] = -Fq::ONE;
    Rq(coeffs)
}
