use super::*;

fn polynomial_division(
    left: &[u64; PHI81_RING_DEGREE],
    right: &[u64; PHI81_RING_DEGREE],
) -> ([u64; PHI81_RING_DEGREE], [u64; PHI81_RING_DEGREE]) {
    let mut product = [0; 2 * PHI81_RING_DEGREE];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            product[i + j] = add_mod(product[i + j], mul_mod(a, b));
        }
    }
    let mut quotient = [0; PHI81_RING_DEGREE];
    for degree in (PHI81_RING_DEGREE..product.len()).rev() {
        let coefficient = product[degree];
        quotient[degree - PHI81_RING_DEGREE] = coefficient;
        for shift in [0, 27, 54] {
            product[degree - PHI81_RING_DEGREE + shift] =
                sub_mod(product[degree - PHI81_RING_DEGREE + shift], coefficient);
        }
    }
    assert!(product[PHI81_RING_DEGREE..].iter().all(|&value| value == 0));
    let remainder = std::array::from_fn(|degree| product[degree]);
    (quotient, remainder)
}

#[test]
fn phi81_quotient_matches_division_for_every_basis_product() {
    for left_degree in 0..PHI81_RING_DEGREE {
        for right_degree in 0..PHI81_RING_DEGREE {
            let mut left = [0; PHI81_RING_DEGREE];
            let mut right = [0; PHI81_RING_DEGREE];
            left[left_degree] = 1;
            right[right_degree] = 1;
            let actual = phi81_quotient(&left, &right);
            assert_eq!(actual, polynomial_division(&left, &right).0);
            assert_eq!(actual[53], 0, "degree-106 products have no degree-53 quotient term");
        }
    }
}

#[test]
fn phi81_quotient_preserves_full_field_coefficients() {
    let left = std::array::from_fn(|degree| sub_mod(0, (degree * degree + 1) as u64));
    let right = std::array::from_fn(|degree| sub_mod(0, (degree * 17 + 2) as u64));
    assert_eq!(phi81_quotient(&left, &right), polynomial_division(&left, &right).0);
}
