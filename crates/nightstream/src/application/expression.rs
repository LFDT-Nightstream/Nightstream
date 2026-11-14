//! Retains application expression order for the existing package wire identity.

use std::{
    collections::HashSet,
    hash::{BuildHasherDefault, Hasher},
    ops::Range,
    sync::Arc,
};

use nightstream_fprime::ApplicationRecipeNode;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{ApplicationError, Variable};

/// A set of nodes that a walk has met. The walks only ask whether they met a
/// node before, so a node's address needs no keyed hash: one multiply mixes
/// it, and the rotation brings the mixed high bits to the low bits that pick
/// the bucket.
type Nodes<T> = HashSet<T, BuildHasherDefault<NodeHasher>>;

#[derive(Default)]
struct NodeHasher(u64);

impl Hasher for NodeHasher {
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Expression {
    Variable(Variable),
    Constant(Goldilocks),
    Add(Arc<Self>, Arc<Self>),
    Multiply(Arc<Self>, Arc<Self>),
}

impl Expression {
    pub(super) fn validate(&self, count: usize, outputs: Range<usize>, causal: bool) -> Result<(), ApplicationError> {
        let mut seen = Nodes::default();
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            if !seen.insert(expression as *const Self) {
                continue;
            }
            match expression {
                Self::Variable(variable) => {
                    if variable.index() >= count {
                        return Err(ApplicationError::VariableOutOfScope(variable.index()));
                    }
                    if causal && outputs.contains(&variable.index()) {
                        return Err(ApplicationError::OutputDependency(variable.index()));
                    }
                }
                Self::Constant(_) => {}
                Self::Add(left, right) | Self::Multiply(left, right) => {
                    pending.push(right);
                    pending.push(left);
                }
            }
        }
        Ok(())
    }

    pub(super) fn has_nested_shared_operations(&self) -> bool {
        let mut seen = Nodes::default();
        let mut shared = Nodes::default();
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            if let Self::Add(left, right) | Self::Multiply(left, right) = expression {
                let pointer = expression as *const Self;
                if !seen.insert(pointer) {
                    shared.insert(pointer);
                    continue;
                }
                pending.push(right);
                pending.push(left);
            }
        }
        let mut visited = Nodes::default();
        let mut pending = vec![(self, false)];
        while let Some((expression, shared_parent)) = pending.pop() {
            if let Self::Add(left, right) | Self::Multiply(left, right) = expression {
                let pointer = expression as *const Self;
                if !visited.insert((pointer, shared_parent)) {
                    continue;
                }
                let is_shared = shared.contains(&pointer);
                if shared_parent && is_shared {
                    return true;
                }
                pending.push((right, shared_parent || is_shared));
                pending.push((left, shared_parent || is_shared));
            }
        }
        false
    }

    pub(super) fn nodes(&self) -> RecipeNodes<'_> {
        RecipeNodes { stack: vec![self] }
    }

    pub(super) fn ordered_terms(&self) -> OrderedTerms<'_> {
        OrderedTerms {
            stack: vec![(self, Goldilocks::ONE)],
        }
    }

    pub(super) fn term_count(&self) -> Result<usize, ApplicationError> {
        self.ordered_terms().try_fold(0usize, |count, _| {
            count
                .checked_add(1)
                .ok_or(ApplicationError::DimensionOverflow)
        })
    }
}

pub(super) struct OrderedTerms<'a> {
    stack: Vec<(&'a Expression, Goldilocks)>,
}

impl Iterator for OrderedTerms<'_> {
    type Item = (Variable, Goldilocks);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((expression, scale)) = self.stack.pop() {
            match expression {
                Expression::Variable(variable) => return Some((*variable, scale)),
                Expression::Constant(_) => {}
                Expression::Add(left, right) => {
                    self.stack.push((right, scale));
                    self.stack.push((left, scale));
                }
                Expression::Multiply(left, right) => {
                    let (coefficient, expression) = match (&**left, &**right) {
                        (Expression::Constant(coefficient), expression)
                        | (expression, Expression::Constant(coefficient)) => (*coefficient, expression),
                        _ => unreachable!("Affine only constructs products with a scalar"),
                    };
                    self.stack.push((expression, scale * coefficient));
                }
            }
        }
        None
    }
}

pub(super) struct RecipeNodes<'a> {
    stack: Vec<&'a Expression>,
}

impl Iterator for RecipeNodes<'_> {
    type Item = ApplicationRecipeNode;
    fn next(&mut self) -> Option<Self::Item> {
        Some(match self.stack.pop()? {
            Expression::Variable(variable) => ApplicationRecipeNode::Variable(variable.index()),
            Expression::Constant(value) => ApplicationRecipeNode::Constant(value.as_canonical_u64()),
            Expression::Add(left, right) => {
                self.stack.push(right);
                self.stack.push(left);
                ApplicationRecipeNode::Add
            }
            Expression::Multiply(left, right) => {
                self.stack.push(right);
                self.stack.push(left);
                ApplicationRecipeNode::Multiply
            }
        })
    }
}
