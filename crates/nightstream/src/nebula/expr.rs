//! Lean `Circuit.Expr` and `Layout/R1CS.lean`.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::application::{Affine, ApplicationBuilder, ApplicationError, Variable};

/// `Circuit.Expr`: `sub` and `neg` are sugar, as in Lean.
#[derive(Clone, Debug)]
pub enum Expr {
    Var(Variable),
    Const(F),
    Add(Rc<Expr>, Rc<Expr>),
    Mul(Rc<Expr>, Rc<Expr>),
}

impl Expr {
    pub fn constant(value: u64) -> Self {
        Self::Const(F::from_u64(value))
    }

    pub(super) fn variable_outside(&self, scope: &BTreeSet<Variable>) -> Option<Variable> {
        match self {
            Self::Var(variable) => (!scope.contains(variable)).then_some(*variable),
            Self::Const(_) => None,
            Self::Add(left, right) | Self::Mul(left, right) => left
                .variable_outside(scope)
                .or_else(|| right.variable_outside(scope)),
        }
    }

    /// Lean `AuxRelocation.rename`, in place.
    pub(super) fn rename(&mut self, aliases: &BTreeMap<Variable, Variable>) {
        match self {
            Self::Var(variable) => *variable = *aliases.get(variable).unwrap_or(variable),
            Self::Const(_) => {}
            Self::Add(left, right) | Self::Mul(left, right) => {
                Rc::make_mut(left).rename(aliases);
                Rc::make_mut(right).rename(aliases);
            }
        }
    }
}

/// One row of an application (Lean `PortApp.rows`): an `Expr`, or the rank-one
/// row `[A, B, C]` of linear forms whose `Expr` is `affine C - affine A *
/// affine B`, with `affine` the left fold of `+ const c * e` from `const 0`.
/// A rank-one row keeps no `Expr` tree, and it lowers to the rows of its `Expr`.
#[derive(Clone, Debug)]
pub enum Row {
    Expr(Expr),
    RankOne([Vec<(Expr, F)>; 3]),
}

impl From<Expr> for Row {
    fn from(expression: Expr) -> Self {
        Self::Expr(expression)
    }
}

fn affine(form: &[(Expr, F)]) -> Expr {
    form.iter()
        .fold(Expr::constant(0), |sum, (word, coefficient)| {
            sum + Expr::Const(*coefficient) * word.clone()
        })
}

/// `lowerAffine` of `affine form`.
fn lower_form(form: &[(Expr, F)]) -> Option<Affine> {
    form.iter().try_fold(zero(), |sum, (word, coefficient)| {
        Some(sum + lower_affine(word)? * *coefficient)
    })
}

impl Row {
    /// The Lean `Expr` of the row.
    pub fn expr(&self) -> Expr {
        match self {
            Self::Expr(expression) => expression.clone(),
            Self::RankOne([a, b, c]) => affine(c) - affine(a) * affine(b),
        }
    }

    pub(super) fn variable_outside(&self, scope: &BTreeSet<Variable>) -> Option<Variable> {
        match self {
            Self::Expr(expression) => expression.variable_outside(scope),
            Self::RankOne([a, b, c]) => [c, a, b]
                .into_iter()
                .flatten()
                .find_map(|(word, _)| word.variable_outside(scope)),
        }
    }

    pub(super) fn rename(&mut self, aliases: &BTreeMap<Variable, Variable>) {
        match self {
            Self::Expr(expression) => expression.rename(aliases),
            Self::RankOne(forms) => forms
                .iter_mut()
                .flatten()
                .for_each(|(word, _)| word.rename(aliases)),
        }
    }

    /// `lowerAffine` of the row.
    pub(super) fn lower_affine(&self) -> Option<Affine> {
        match self {
            Self::Expr(expression) => lower_affine(expression),
            Self::RankOne(forms) if forms[0].is_empty() || forms[1].is_empty() => {
                rank_one_direct(forms).map(|(form, _, _)| form)
            }
            Self::RankOne(_) => None,
        }
    }

    /// `rankOneConstraint` of the row.
    pub(super) fn rank_one(&self) -> Option<Sides> {
        match self {
            Self::Expr(expression) => rank_one_row(expression),
            Self::RankOne(forms) => {
                let [a, b, c] = forms.each_ref().map(|form| lower_form(form));
                Some((a?, b?, c?))
            }
        }
    }

    /// `lowerConstraint` of the row.
    pub(super) fn lower(&self, builder: &mut ApplicationBuilder) -> Result<usize, ApplicationError> {
        match self {
            Self::Expr(expression) => lower_constraint(builder, expression),
            Self::RankOne(forms) => match rank_one_direct(forms) {
                Some((a, b, c)) => builder.constrain(a, b, c),
                None => lower_constraint(builder, &self.expr()),
            },
        }
    }
}

/// `directConstraint` of `affine C - affine A * affine B`, which is never
/// `var - …`. `affine A` is `const 0` only when `A` is empty. Then
/// `affineConstraint` reads the row, and the expression of `lowerAffine` of
/// `const 0 * affine B` keeps the variables of `B`, which the records keep
/// with coefficient 0. Else `rankOneConstraint` reads the row.
fn rank_one_direct(forms: &[Vec<(Expr, F)>; 3]) -> Option<Sides> {
    let [a, b, c] = forms.each_ref().map(|form| lower_form(form));
    let (a, b, c) = (a?, b?, c?);
    Some(match (forms[0].is_empty(), forms[1].is_empty()) {
        (true, _) => (c + b * F::ZERO * F::NEG_ONE, one(), zero()),
        (false, true) => (c + a * F::ZERO * F::NEG_ONE, one(), zero()),
        (false, false) => (a, b, c),
    })
}

impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Rc::new(self), Rc::new(rhs))
    }
}

/// `Expr.neg e = const (-1) * e`.
impl Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        Self::Const(F::NEG_ONE) * self
    }
}

/// `Expr.sub a b = a + neg b`.
impl Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

/// `Circuit.Quadratic.KExpr`.
#[derive(Clone, Debug)]
pub(super) struct KExpr {
    pub(super) c0: Expr,
    pub(super) c1: Expr,
}

impl KExpr {
    pub(super) fn zero() -> Self {
        Self::of(Expr::constant(0), Expr::constant(0))
    }

    pub(super) fn of(c0: Expr, c1: Expr) -> Self {
        Self { c0, c1 }
    }

    /// `embedE`.
    pub(super) fn embed(x: Expr) -> Self {
        Self::of(x, Expr::constant(0))
    }

    pub(super) fn sub(self, rhs: Self) -> Self {
        Self::of(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }

    pub(super) fn mul(self, rhs: Self) -> Self {
        Self::of(
            self.c0.clone() * rhs.c0.clone() + Expr::constant(2) * self.c1.clone() * rhs.c1.clone(),
            self.c0 * rhs.c1 + self.c1 * rhs.c0,
        )
    }

    /// `KExpr.equalities`.
    pub(super) fn equalities(self, rhs: Self) -> [Expr; 2] {
        [self.c0 - rhs.c0, self.c1 - rhs.c1]
    }
}

fn one() -> Affine {
    Affine::constant(F::ONE)
}

fn zero() -> Affine {
    Affine::constant(F::ZERO)
}

/// `lowerAffine`.
pub(super) fn lower_affine(expression: &Expr) -> Option<Affine> {
    match expression {
        Expr::Var(variable) => Some((*variable).into()),
        Expr::Const(value) => Some(Affine::constant(*value)),
        Expr::Add(left, right) => Some(lower_affine(left)? + lower_affine(right)?),
        Expr::Mul(left, right) => match (&**left, &**right) {
            (Expr::Const(value), right) => Some(lower_affine(right)? * *value),
            (left, Expr::Const(value)) => Some(lower_affine(left)? * *value),
            _ => None,
        },
    }
}

/// Lean `Row`: the sides `(A, B, C)` of `A · B = C`.
pub(super) type Sides = (Affine, Affine, Affine);

/// `directRecipeRow`.
fn recipe_row(output: Variable, recipe: &Expr) -> Option<Sides> {
    if let Some(lowered) = lower_affine(recipe) {
        return Some((lowered, one(), output.into()));
    }
    match recipe {
        Expr::Mul(left, right) => Some((lower_affine(left)?, lower_affine(right)?, output.into())),
        Expr::Add(product, rest) => match &**product {
            Expr::Mul(left, right) => Some((
                lower_affine(left)?,
                lower_affine(right)?,
                Affine::from(output) - lower_affine(rest)?,
            )),
            _ => None,
        },
        _ => None,
    }
}

fn affine_row(expression: &Expr) -> Option<Sides> {
    Some((lower_affine(expression)?, one(), zero()))
}

/// `rankOneConstraint`.
pub(super) fn rank_one_row(expression: &Expr) -> Option<Sides> {
    match expression {
        Expr::Add(constant, scaled) => match &**scaled {
            Expr::Mul(coefficient, product) => match (&**coefficient, &**product) {
                (Expr::Const(value), Expr::Mul(left, right)) if *value == F::NEG_ONE => {
                    Some((lower_affine(left)?, lower_affine(right)?, lower_affine(constant)?))
                }
                _ => None,
            },
            _ => None,
        },
        Expr::Mul(left, right) => Some((lower_affine(left)?, lower_affine(right)?, zero())),
        _ => None,
    }
}

/// `directConstraint`.
pub(super) fn direct_row(expression: &Expr) -> Option<Sides> {
    if let Expr::Add(output, scaled) = expression {
        if let (Expr::Var(output), Expr::Mul(coefficient, recipe)) = (&**output, &**scaled) {
            if let Expr::Const(value) = &**coefficient {
                if *value == F::NEG_ONE {
                    return recipe_row(*output, recipe).or_else(|| affine_row(expression));
                }
                return affine_row(expression);
            }
        }
    }
    affine_row(expression).or_else(|| rank_one_row(expression))
}

/// `lowerExpression`.
fn lower_expression(builder: &mut ApplicationBuilder, expression: &Expr) -> Result<Affine, ApplicationError> {
    Ok(match expression {
        Expr::Var(variable) => (*variable).into(),
        Expr::Const(value) => Affine::constant(*value),
        Expr::Add(left, right) => {
            let left = lower_expression(builder, left)?;
            left + lower_expression(builder, right)?
        }
        Expr::Mul(left, right) => {
            let left = lower_expression(builder, left)?;
            let right = lower_expression(builder, right)?;
            builder.multiply(left, right)?.into()
        }
    })
}

/// `lowerConstraint`.
pub(super) fn lower_constraint(builder: &mut ApplicationBuilder, expression: &Expr) -> Result<usize, ApplicationError> {
    match direct_row(expression) {
        Some((a, b, c)) => builder.constrain(a, b, c),
        None => {
            let value = lower_expression(builder, expression)?;
            builder.constrain(value, one(), zero())
        }
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/rank_one.rs"]
mod tests;
