//! Independent SuperNeo v1_1 Section 5 opening arithmetic.
//! The base field comes from the independent row checker. No production
//! extension, bar transform, ring product, or MLE evaluator is used.

use super::reference::{Field, Result};

pub const DEGREE: usize = 54;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extension(pub [Field; 2]);

impl Extension {
    pub const ZERO: Self = Self([Field::ZERO; 2]);
    pub const ONE: Self = Self([Field::ONE, Field::ZERO]);

    pub fn checked(words: [u64; 2]) -> Result<Self> {
        Ok(Self([
            Field::checked(words[0], "extension real")?,
            Field::checked(words[1], "extension second")?,
        ]))
    }

    pub fn words(self) -> [u64; 2] {
        self.0.map(Field::canonical)
    }

    pub fn scale(self, value: Field) -> Self {
        Self(self.0.map(|coefficient| coefficient * value))
    }
}

impl std::ops::Add for Extension {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl std::ops::AddAssign for Extension {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Neg for Extension {
    type Output = Self;
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl std::ops::Mul for Extension {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Spec.Algebra.K.mul: u^2 = 7.
        let seven = Field::checked(7, "quadratic nonresidue").expect("seven is canonical");
        Self([
            self.0[0] * rhs.0[0] + seven * self.0[1] * rhs.0[1],
            self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0],
        ])
    }
}

pub type Ring = [Extension; DEGREE];

/// Inverse of T[i,j] = ct(X^(i+j)). For Phi81, T has nonzero
/// entries only at i+j = 0, 54, 81, with coefficients 1, -1, 1.
/// Solving these sparse equations gives the following dual basis map.
pub fn transform(values: &Ring) -> Ring {
    std::array::from_fn(|row| match row {
        0 => values[0],
        1..27 => -(values[27 - row] + values[54 - row]),
        _ => -values[54 - row],
    })
}

pub fn multiply_signed(left: &Ring, right: &[u8]) -> Ring {
    assert_eq!(right.len(), DEGREE);
    // Each basis product contributes at most four signed terms after the
    // two possible polynomial-reduction steps. Integer accumulation fits
    // i128; reducing modulo p only at the end preserves the field product.
    const _: () = assert!(
        4 * DEGREE as u128 * DEGREE as u128 * (super::reference::GOLDILOCKS_MODULUS as u128) < i128::MAX as u128
    );
    let left = left.map(|value| value.words().map(i128::from));
    let mut coefficients = [[0i128; 2]; 2 * DEGREE - 1];
    for (power, &value) in right.iter().enumerate() {
        match value {
            0 => {}
            1 => {
                for (index, &coefficient) in left.iter().enumerate() {
                    for lane in 0..2 {
                        coefficients[index + power][lane] += coefficient[lane];
                    }
                }
            }
            255 => {
                for (index, &coefficient) in left.iter().enumerate() {
                    for lane in 0..2 {
                        coefficients[index + power][lane] -= coefficient[lane];
                    }
                }
            }
            _ => panic!("opening carrier is not a signed unit"),
        }
    }
    for power in (DEGREE..coefficients.len()).rev() {
        let coefficient = coefficients[power];
        for lane in 0..2 {
            coefficients[power - DEGREE][lane] -= coefficient[lane];
            coefficients[power - DEGREE / 2][lane] -= coefficient[lane];
        }
    }
    let modulus = i128::from(super::reference::GOLDILOCKS_MODULUS);
    std::array::from_fn(|index| {
        Extension::checked(coefficients[index].map(|value| value.rem_euclid(modulus) as u64))
            .expect("integer remainder is a canonical field word")
    })
}

pub fn evaluate_block(weights: &Ring, carrier: &[u8]) -> Ring {
    multiply_signed(&transform(weights), carrier)
}

pub fn add_rings(left: Ring, right: Ring) -> Ring {
    std::array::from_fn(|index| left[index] + right[index])
}

pub struct EqualityTensor {
    low: Vec<Extension>,
    high: Vec<Extension>,
    split: usize,
}

fn tensor(point: &[Extension]) -> Vec<Extension> {
    let mut values = vec![Extension::ONE];
    for &coordinate in point {
        let before = values.len();
        for index in 0..before {
            values.push(values[index] * coordinate);
            values[index] = values[index] * (Extension::ONE + -coordinate);
        }
    }
    values
}

impl EqualityTensor {
    pub fn new(point: &[Extension]) -> Self {
        // Equal halves minimize the two stored tensor factors for this
        // 28-variable profile; the split is derived from the point length.
        assert_eq!(point.len(), 28);
        let split = point.len() / 2;
        Self {
            low: tensor(&point[..split]),
            high: tensor(&point[split..]),
            split,
        }
    }

    pub fn at(&self, row: usize) -> Extension {
        self.low[row & ((1 << self.split) - 1)] * self.high[row >> self.split]
    }
}
