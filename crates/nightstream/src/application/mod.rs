//! Rust-owned application constraints and causal witnesses. The shared recursive
//! verifier and physical column allocation are owned by the assembler.

mod builder;
mod expression;
mod poseidon2;

pub(crate) use expression::Expression;

pub use builder::{ApplicationBuilder, ApplicationCircuit, ApplicationError, ApplicationWitness};
pub use poseidon2::{poseidon2_hash_chain, poseidon2_hash_chain_step, poseidon2_hash_chain_v1};

use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

/// An application-local variable. The assembler assigns its physical column.
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
    constant: Goldilocks,
    terms: Vec<(Variable, Goldilocks)>,
    pub(crate) expression: Arc<Expression>,
}

impl Affine {
    pub fn constant(value: Goldilocks) -> Self {
        Self {
            constant: value,
            terms: Vec::new(),
            expression: Arc::new(Expression::Constant(value)),
        }
    }

    pub fn variable(variable: Variable) -> Self {
        Self {
            constant: Goldilocks::ZERO,
            terms: vec![(variable, Goldilocks::ONE)],
            expression: Arc::new(Expression::Variable(variable)),
        }
    }

    pub fn constant_term(&self) -> Goldilocks {
        self.constant
    }

    pub fn terms(&self) -> &[(Variable, Goldilocks)] {
        &self.terms
    }

    pub(super) fn as_variable(&self) -> Option<Variable> {
        match self.terms.as_slice() {
            [(variable, coefficient)] if self.constant == Goldilocks::ZERO && *coefficient == Goldilocks::ONE => {
                Some(*variable)
            }
            _ => None,
        }
    }

    // Preserve the saved recipe syntax for trees. Nested shared operations must use
    // the already canonical coefficients instead of expanding a DAG as a tree.
    pub(super) fn compact_shared(self) -> Self {
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

impl From<Goldilocks> for Affine {
    fn from(value: Goldilocks) -> Self {
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
                    if coefficient != Goldilocks::ZERO {
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
        self + rhs * -Goldilocks::ONE
    }
}

impl Mul<Goldilocks> for Affine {
    type Output = Self;

    fn mul(mut self, rhs: Goldilocks) -> Self {
        self.expression = Arc::new(Expression::Multiply(
            Arc::new(Expression::Constant(rhs)),
            self.expression,
        ));
        self.constant *= rhs;
        for (_, coefficient) in &mut self.terms {
            *coefficient *= rhs;
        }
        self.terms
            .retain(|(_, coefficient)| *coefficient != Goldilocks::ZERO);
        self
    }
}

struct R1csRow {
    pub(super) a: Affine,
    pub(super) b: Affine,
    pub(super) c: Affine,
}
