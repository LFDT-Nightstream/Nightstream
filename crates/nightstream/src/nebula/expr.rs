//! Lean's `Circuit.Expr` and its constraint lowering (`Layout/R1CS.lean`),
//! case for case, so a Rust circuit has the rows of the Lean-proved one. The
//! tree shape matters: a generic constraint allocates one fresh variable per
//! `mul` node, constant factors included. An application writes its rows as
//! `Expr` trees over a `View`. It does not own any row list.

use std::{
    collections::BTreeSet,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

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

    /// The first variable of the tree that is not in `scope`.
    pub(super) fn variable_outside(&self, scope: &BTreeSet<Variable>) -> Option<Variable> {
        match self {
            Self::Var(variable) => (!scope.contains(variable)).then_some(*variable),
            Self::Const(_) => None,
            Self::Add(left, right) | Self::Mul(left, right) => left
                .variable_outside(scope)
                .or_else(|| right.variable_outside(scope)),
        }
    }
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

/// `Circuit.Quadratic.KExpr`: an element of `F[X]/(X² − 7)`.
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

    /// `embedE x = (x, 0)`.
    pub(super) fn embed(x: Expr) -> Self {
        Self::of(x, Expr::constant(0))
    }

    pub(super) fn add(self, rhs: Self) -> Self {
        Self::of(self.c0 + rhs.c0, self.c1 + rhs.c1)
    }

    pub(super) fn sub(self, rhs: Self) -> Self {
        Self::of(self.c0 - rhs.c0, self.c1 - rhs.c1)
    }

    pub(super) fn mul(self, rhs: Self) -> Self {
        Self::of(
            self.c0.clone() * rhs.c0.clone() + Expr::constant(7) * self.c1.clone() * rhs.c1.clone(),
            self.c0 * rhs.c1 + self.c1 * rhs.c0,
        )
    }

    /// `KExpr.equalities`: one zero row per coefficient.
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

/// `lowerAffine`: commits to the first `mul` pattern that matches.
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

type Row = (Affine, Affine, Affine);

/// `directRecipeRow`: an affine recipe, a product, or a product plus an affine rest.
fn recipe_row(output: Variable, recipe: &Expr) -> Option<Row> {
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

fn affine_row(expression: &Expr) -> Option<Row> {
    Some((lower_affine(expression)?, one(), zero()))
}

/// `rankOneConstraint`: `c − l · r` or `l · r` with affine parts.
fn rank_one_row(expression: &Expr) -> Option<Row> {
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

/// `directConstraint`: `v − recipe` as a recipe row, else an affine or
/// rank-one row.
fn direct_row(expression: &Expr) -> Option<Row> {
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

/// `lowerExpression`: one fresh variable per `mul` node, left before right.
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

/// `lowerConstraint` for the zero assertion `expression = 0`; returns its last row.
pub(super) fn lower_constraint(builder: &mut ApplicationBuilder, expression: &Expr) -> Result<usize, ApplicationError> {
    match direct_row(expression) {
        Some((a, b, c)) => builder.constrain(a, b, c),
        None => {
            let value = lower_expression(builder, expression)?;
            builder.constrain(value, one(), zero())
        }
    }
}
