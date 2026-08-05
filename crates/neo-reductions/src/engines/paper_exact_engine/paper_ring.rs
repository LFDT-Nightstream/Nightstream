//! Direct arithmetic for the concrete SuperNeo quotient ring.
//!
//! This module owns a literal schoolbook multiplication and a local
//! construction of the paper's one-sided bar transform. It does not use the
//! production ring multiplier, the global bar cache, or rotation matrices.

use neo_math::{Fq, D, ETA};
use p3_field::{Field, PrimeCharacteristicRing};

use neo_math::K;

const PHI_MID_DEGREE: usize = 27;

/// Concrete arithmetic data for `R_K = K[X]/(X^54 + X^27 + 1)`.
pub(super) struct PaperRing {
    bar: [[Fq; D]; D],
}

impl PaperRing {
    pub(super) fn new() -> Self {
        Self {
            bar: build_bar_matrix(),
        }
    }

    pub(super) fn bar_block(&self, input: [Fq; D]) -> [Fq; D] {
        let mut output = [Fq::ZERO; D];
        for (row, slot) in output.iter_mut().enumerate() {
            for (column, &value) in input.iter().enumerate() {
                *slot += self.bar[row][column] * value;
            }
        }
        output
    }

    pub(super) fn transformed_product(&self, matrix: [Fq; D], assignment: [K; D]) -> [K; D] {
        let transformed: [K; D] = self.bar_block(matrix).map(K::from);
        self.multiply_extension(transformed, assignment)
    }

    pub(super) fn multiply_extension(&self, left: [K; D], right: [K; D]) -> [K; D] {
        multiply_mod_phi81(left, right)
    }

    pub(super) fn multiply_base(&self, left: [Fq; D], right: [Fq; D]) -> [Fq; D] {
        let mut product = [Fq::ZERO; 2 * D - 1];
        for (left_degree, &left_coefficient) in left.iter().enumerate() {
            for (right_degree, &right_coefficient) in right.iter().enumerate() {
                product[left_degree + right_degree] += left_coefficient * right_coefficient;
            }
        }
        for degree in (D..2 * D - 1).rev() {
            let coefficient = product[degree];
            product[degree] = Fq::ZERO;
            product[degree - D] -= coefficient;
            product[degree - PHI_MID_DEGREE] -= coefficient;
        }
        core::array::from_fn(|degree| product[degree])
    }
}

fn multiply_mod_phi81(left: [K; D], right: [K; D]) -> [K; D] {
    let mut product = [K::ZERO; 2 * D - 1];
    for (left_degree, &left_coefficient) in left.iter().enumerate() {
        for (right_degree, &right_coefficient) in right.iter().enumerate() {
            product[left_degree + right_degree] += left_coefficient * right_coefficient;
        }
    }

    // X^54 = -X^27 - 1. A descending pass is sufficient because every
    // replacement has lower degree than the term that it replaces.
    for degree in (D..2 * D - 1).rev() {
        let coefficient = product[degree];
        product[degree] = K::ZERO;
        product[degree - D] -= coefficient;
        product[degree - PHI_MID_DEGREE] -= coefficient;
    }

    core::array::from_fn(|degree| product[degree])
}

fn build_bar_matrix() -> [[Fq; D]; D] {
    let mut gram = [[Fq::ZERO; D]; D];
    for (row, values) in gram.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = match row + column {
                0 => Fq::ONE,
                D => -Fq::ONE,
                ETA => Fq::ONE,
                _ => Fq::ZERO,
            };
        }
    }
    invert(gram).expect("the paper bar Gram matrix must be invertible")
}

fn invert(mut matrix: [[Fq; D]; D]) -> Option<[[Fq; D]; D]> {
    let mut inverse = [[Fq::ZERO; D]; D];
    for (index, row) in inverse.iter_mut().enumerate() {
        row[index] = Fq::ONE;
    }

    for column in 0..D {
        let pivot = (column..D).find(|&row| matrix[row][column] != Fq::ZERO)?;
        if pivot != column {
            matrix.swap(pivot, column);
            inverse.swap(pivot, column);
        }

        let scale = matrix[column][column].inverse();
        for entry in 0..D {
            matrix[column][entry] *= scale;
            inverse[column][entry] *= scale;
        }

        for row in 0..D {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in 0..D {
                matrix[row][entry] -= factor * matrix[column][entry];
                inverse[row][entry] -= factor * inverse[column][entry];
            }
        }
    }
    Some(inverse)
}
