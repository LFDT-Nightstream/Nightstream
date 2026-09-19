//! Retains application expression order for the existing package wire identity.

use std::{ops::Range, sync::Arc};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::{json, Value};

use super::{ApplicationError, Variable};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expression {
    Variable(Variable),
    Constant(Goldilocks),
    Add(Arc<Self>, Arc<Self>),
    Multiply(Arc<Self>, Arc<Self>),
}

impl Expression {
    pub(super) fn validate(&self, count: usize, outputs: Range<usize>, causal: bool) -> Result<(), ApplicationError> {
        match self {
            Self::Variable(variable) => {
                if variable.index() >= count {
                    return Err(ApplicationError::VariableOutOfScope(variable.index()));
                }
                if causal && outputs.contains(&variable.index()) {
                    return Err(ApplicationError::OutputDependency(variable.index()));
                }
                Ok(())
            }
            Self::Constant(_) => Ok(()),
            Self::Add(left, right) | Self::Multiply(left, right) => {
                left.validate(count, outputs.clone(), causal)?;
                right.validate(count, outputs, causal)
            }
        }
    }

    pub(crate) fn encode(&self, columns: &[usize]) -> Value {
        match self {
            Self::Variable(variable) => json!([0, columns[variable.index()]]),
            Self::Constant(value) => json!([1, value.as_canonical_u64()]),
            Self::Add(left, right) => json!([2, left.encode(columns), right.encode(columns)]),
            Self::Multiply(left, right) => json!([3, left.encode(columns), right.encode(columns)]),
        }
    }

    pub(crate) fn affine(&self, columns: &[usize]) -> Value {
        let (constant, terms) = self.affine_parts(columns);
        json!([
            constant.as_canonical_u64(),
            terms
                .into_iter()
                .map(|(column, value)| (column, value.as_canonical_u64()))
                .collect::<Vec<_>>()
        ])
    }

    fn affine_parts(&self, columns: &[usize]) -> (Goldilocks, Vec<(usize, Goldilocks)>) {
        match self {
            Self::Variable(variable) => (Goldilocks::ZERO, vec![(columns[variable.index()], Goldilocks::ONE)]),
            Self::Constant(value) => (*value, Vec::new()),
            Self::Add(left, right) => {
                let (left_constant, mut terms) = left.affine_parts(columns);
                let (right_constant, right_terms) = right.affine_parts(columns);
                terms.extend(right_terms);
                (left_constant + right_constant, terms)
            }
            Self::Multiply(left, right) => {
                let (coefficient, expression) = match (&**left, &**right) {
                    (Self::Constant(coefficient), expression) | (expression, Self::Constant(coefficient)) => {
                        (*coefficient, expression)
                    }
                    _ => unreachable!("Affine only constructs products with a scalar"),
                };
                let (constant, terms) = expression.affine_parts(columns);
                (
                    coefficient * constant,
                    terms
                        .into_iter()
                        .map(|(column, value)| (column, coefficient * value))
                        .collect(),
                )
            }
        }
    }
}
