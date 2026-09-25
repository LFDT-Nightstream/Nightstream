//! Retains application expression order for the existing package wire identity.

use std::{collections::HashSet, ops::Range, sync::Arc};

use nightstream_fprime::ApplicationRecipeNode;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

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
        let mut seen = HashSet::new();
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
        let mut seen = HashSet::new();
        let mut shared = HashSet::new();
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
        // Flat sharing has no repeated branching along a path. Preserve that
        // existing recipe syntax; nested sharing can expand exponentially.
        let mut visited = HashSet::new();
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

// These cursors borrow the caller's live expression while it is appended. No
// tree or term vector survives in the builder or sealed record owner.
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
