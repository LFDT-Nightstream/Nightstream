//! Rust-owned application constraints and causal witnesses. `Circuit::compile`
//! puts the application in place of Lean's application in the step packages.

mod builder;
mod expression;
mod poseidon2;

pub(crate) use expression::Expression;

pub use builder::{ApplicationBuilder, ApplicationCircuit, ApplicationError, ApplicationWitness};
pub(crate) use poseidon2::{check_blocks, permutation};
pub use poseidon2::{poseidon2_hash_chain_step, poseidon2_hash_chain_v1};

use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;

/// An application-local variable. The step packages assign its column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Variable(pub(super) usize);

impl Variable {
    pub fn index(self) -> usize {
        self.0
    }
}

/// A canonical affine field expression with ordered, nonzero coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Affine {
    constant: Fq,
    terms: Vec<(Variable, Fq)>,
    pub(crate) expression: Arc<Expression>,
}

impl Affine {
    pub fn constant(value: Fq) -> Self {
        Self {
            constant: value,
            terms: Vec::new(),
            expression: Arc::new(Expression::Constant(value)),
        }
    }

    pub fn variable(variable: Variable) -> Self {
        Self {
            constant: Fq::ZERO,
            terms: vec![(variable, Fq::ONE)],
            expression: Arc::new(Expression::Variable(variable)),
        }
    }

    pub fn constant_term(&self) -> Fq {
        self.constant
    }

    pub fn terms(&self) -> &[(Variable, Fq)] {
        &self.terms
    }

    pub(crate) fn as_variable(&self) -> Option<Variable> {
        match self.terms.as_slice() {
            [(variable, coefficient)] if self.constant == Fq::ZERO && *coefficient == Fq::ONE => Some(*variable),
            _ => None,
        }
    }

    pub(super) fn flatten_if_nested_shared(self) -> Self {
        if !self.expression.has_nested_shared_operations() {
            return self;
        }
        let expression = self.terms.iter().fold(
            Arc::new(Expression::Constant(self.constant)),
            |sum, (variable, coefficient)| {
                Arc::new(Expression::Add(
                    sum,
                    Arc::new(Expression::Multiply(
                        Arc::new(Expression::Constant(*coefficient)),
                        Arc::new(Expression::Variable(*variable)),
                    )),
                ))
            },
        );
        Self { expression, ..self }
    }
}

impl From<Variable> for Affine {
    fn from(value: Variable) -> Self {
        Self::variable(value)
    }
}

impl From<Fq> for Affine {
    fn from(value: Fq) -> Self {
        Self::constant(value)
    }
}

impl Add for Affine {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self.expression = Arc::new(Expression::Add(self.expression, rhs.expression));
        self.constant += rhs.constant;
        let mut merged = Vec::with_capacity(self.terms.len() + rhs.terms.len());
        let mut left = self.terms.into_iter().peekable();
        let mut right = rhs.terms.into_iter().peekable();
        while let (Some(&(a, _)), Some(&(b, _))) = (left.peek(), right.peek()) {
            match a.cmp(&b) {
                std::cmp::Ordering::Less => merged.push(left.next().unwrap()),
                std::cmp::Ordering::Greater => merged.push(right.next().unwrap()),
                std::cmp::Ordering::Equal => {
                    let (_, coefficient) = left.next().unwrap();
                    let coefficient = coefficient + right.next().unwrap().1;
                    if coefficient != Fq::ZERO {
                        merged.push((a, coefficient));
                    }
                }
            }
        }
        merged.extend(left);
        merged.extend(right);
        self.terms = merged;
        self
    }
}

impl Sub for Affine {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + rhs * -Fq::ONE
    }
}

impl Mul<Fq> for Affine {
    type Output = Self;

    fn mul(mut self, rhs: Fq) -> Self {
        self.expression = Arc::new(Expression::Multiply(
            Arc::new(Expression::Constant(rhs)),
            self.expression,
        ));
        self.constant *= rhs;
        for (_, coefficient) in &mut self.terms {
            *coefficient *= rhs;
        }
        self.terms
            .retain(|(_, coefficient)| *coefficient != Fq::ZERO);
        self
    }
}

struct R1csRow {
    pub(super) a: Affine,
    pub(super) b: Affine,
    pub(super) c: Affine,
}

#[cfg(test)]
#[path = "../../tests/application/builder.rs"]
mod builder_tests;

#[cfg(test)]
#[path = "../../tests/application/poseidon2.rs"]
mod poseidon2_tests;
