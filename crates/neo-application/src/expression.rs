//! Linear expressions for application R1CS construction.

use crate::ColumnId;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Debug, Default)]
pub struct LinearExpression {
    pub terms: Vec<(usize, F)>,
}

impl LinearExpression {
    pub fn evaluate(&self, values: &[F]) -> F {
        self.terms
            .iter()
            .fold(F::ZERO, |sum, &(column, coefficient)| {
                sum + values[column] * coefficient
            })
    }

    pub fn zero() -> Self {
        Self::default()
    }

    pub fn var(column: impl Into<ColumnId>) -> Self {
        let column = column.into().0;
        Self {
            terms: vec![(column, F::ONE)],
        }
    }

    pub fn from_bits(bits: &[ColumnId]) -> Self {
        let mut coefficient = F::ONE;
        let mut out = Self::zero();
        for &bit in bits {
            out.terms.push((bit.0, coefficient));
            coefficient += coefficient;
        }
        out
    }

    pub fn add_scaled(mut self, other: &Self, scale: F) -> Self {
        self.terms.extend(
            other
                .terms
                .iter()
                .map(|&(column, coefficient)| (column, coefficient * scale)),
        );
        self
    }

    pub fn scaled(mut self, scale: F) -> Self {
        for (_, coefficient) in &mut self.terms {
            *coefficient *= scale;
        }
        self
    }

    pub fn plus(self, other: &Self) -> Self {
        self.add_scaled(other, F::ONE)
    }

    pub fn minus(self, other: &Self) -> Self {
        self.add_scaled(other, -F::ONE)
    }
}
