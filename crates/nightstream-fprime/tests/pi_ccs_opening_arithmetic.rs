//! Compare delayed integer reduction with the original modular evaluator.

#[allow(dead_code)]
#[path = "support/pi_ccs_opening.rs"]
mod opening;
#[allow(dead_code)]
#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod reference;

use opening::{Extension, Ring, DEGREE};

fn modular_product(left: &Ring, right: &[u8; DEGREE]) -> Ring {
    let mut coefficients = [Extension::ZERO; 2 * DEGREE - 1];
    for (power, &value) in right.iter().enumerate() {
        match value {
            0 => {}
            1 => {
                for (index, &coefficient) in left.iter().enumerate() {
                    coefficients[index + power] += coefficient;
                }
            }
            255 => {
                for (index, &coefficient) in left.iter().enumerate() {
                    coefficients[index + power] += -coefficient;
                }
            }
            _ => panic!("signed unit required"),
        }
    }
    for power in (DEGREE..coefficients.len()).rev() {
        let coefficient = coefficients[power];
        coefficients[power - DEGREE] += -coefficient;
        coefficients[power - DEGREE / 2] += -coefficient;
    }
    std::array::from_fn(|index| coefficients[index])
}

#[test]
fn integer_reduction_matches_every_signed_basis_product() {
    for left_power in 0..DEGREE {
        for right_power in 0..DEGREE {
            for lane in 0..2 {
                let mut left = [Extension::ZERO; DEGREE];
                let mut words = [0; 2];
                words[lane] = reference::GOLDILOCKS_MODULUS - 1;
                left[left_power] = Extension::checked(words).unwrap();
                for sign in [1, 255] {
                    let mut right = [0; DEGREE];
                    right[right_power] = sign;
                    assert_eq!(opening::multiply_signed(&left, &right), modular_product(&left, &right));
                }
            }
        }
    }
}

#[test]
fn integer_reduction_matches_dense_field_boundaries() {
    let modulus = reference::GOLDILOCKS_MODULUS;
    for word in [0, 1, modulus - 1] {
        let left = [Extension::checked([word, modulus - 1 - word]).unwrap(); DEGREE];
        for right in [
            [0; DEGREE],
            [1; DEGREE],
            [255; DEGREE],
            std::array::from_fn(|i| [0, 1, 255][i % 3]),
        ] {
            assert_eq!(opening::multiply_signed(&left, &right), modular_product(&left, &right));
        }
    }
    for shift in 0..DEGREE {
        let left = std::array::from_fn(|index| {
            let real = ((u128::from(modulus - 1) * (index + shift + 1) as u128) % u128::from(modulus)) as u64;
            let imaginary = ((u128::from(modulus - 1) * (DEGREE + shift - index) as u128) % u128::from(modulus)) as u64;
            Extension::checked([real, imaginary]).unwrap()
        });
        let right = std::array::from_fn(|index| [0, 1, 255][(index + shift) % 3]);
        assert_eq!(opening::multiply_signed(&left, &right), modular_product(&left, &right));
    }
}

#[test]
#[should_panic(expected = "opening carrier is not a signed unit")]
fn integer_reduction_rejects_an_unbounded_digit() {
    let mut right = [0; DEGREE];
    right[0] = 2;
    opening::multiply_signed(&[Extension::ONE; DEGREE], &right);
}
