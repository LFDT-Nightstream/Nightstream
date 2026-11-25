use neo_math::Fq;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::package::PackageError;

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawWitnessBatch(pub(super) u64, pub(super) Vec<Value>, pub(super) Vec<Value>);

#[derive(Clone, Debug, PartialEq)]
pub(super) enum WitnessExpr {
    Var(usize),
    Const(Fq),
    Add(Box<WitnessExpr>, Box<WitnessExpr>),
    Mul(Box<WitnessExpr>, Box<WitnessExpr>),
}

#[derive(Clone, Debug)]
enum WitnessHint {
    Bit(usize),
    InverseOrZero,
    QuotientFive,
    RemainderFive,
}

#[derive(Clone, Debug)]
pub(super) struct WitnessBatch {
    pub(super) start: usize,
    recipes: Vec<WitnessExpr>,
    hints: Vec<WitnessHint>,
    sources: Vec<(usize, HintSource, bool)>,
}

impl WitnessBatch {
    pub(super) fn end(&self) -> usize {
        self.start + self.recipes.len() + self.hints.len()
    }

    /// Lean `StepPackage.bitColumns`.
    pub(super) fn bit_columns(&self) -> impl Iterator<Item = usize> + '_ {
        let first = self.start + self.recipes.len();
        self.hints
            .iter()
            .enumerate()
            .filter(|(_, hint)| matches!(hint, WitnessHint::Bit(..)))
            .map(move |(index, _)| first + index)
    }

    pub(super) fn map_columns(&self, map: &impl Fn(usize) -> Option<usize>) -> Option<Self> {
        let start = map(self.start)?;
        (map(self.end() - 1)? == start + self.recipes.len() + self.hints.len() - 1).then_some(())?;
        Some(Self {
            start,
            recipes: self
                .recipes
                .iter()
                .map(|recipe| recipe.map_columns(map))
                .collect::<Option<_>>()?,
            hints: self.hints.clone(),
            sources: self
                .sources
                .iter()
                .map(|(offset, expr, _)| {
                    let expr = expr.map_columns(map)?;
                    let reusable = expr.reads_before(start + self.recipes.len() + offset);
                    Some((*offset, expr, reusable))
                })
                .collect::<Option<_>>()?,
        })
    }
}

#[derive(Clone, Debug)]
enum HintSource {
    Linear(Fq, Vec<(usize, Fq)>),
    Expression(WitnessExpr),
}

impl HintSource {
    fn compile(expr: WitnessExpr) -> Self {
        fn collect(expr: &WitnessExpr, scale: Fq, constant: &mut Fq, terms: &mut Vec<(usize, Fq)>) -> bool {
            match expr {
                WitnessExpr::Const(x) => *constant += scale * *x,
                WitnessExpr::Var(c) => terms.push((*c, scale)),
                WitnessExpr::Add(a, b) => {
                    return collect(a, scale, constant, terms) && collect(b, scale, constant, terms)
                }
                WitnessExpr::Mul(a, b) => {
                    return match (&**a, &**b) {
                        (WitnessExpr::Const(v), e) | (e, WitnessExpr::Const(v)) => {
                            collect(e, scale * *v, constant, terms)
                        }
                        _ => false,
                    }
                }
            };
            true
        }
        let mut constant = Fq::ZERO;
        let mut terms = Vec::new();
        if collect(&expr, Fq::ONE, &mut constant, &mut terms) {
            Self::Linear(constant, terms)
        } else {
            Self::Expression(expr)
        }
    }
    fn map_columns(&self, map: &impl Fn(usize) -> Option<usize>) -> Option<Self> {
        match self {
            Self::Expression(e) => Some(Self::Expression(e.map_columns(map)?)),
            Self::Linear(c, terms) => Some(Self::Linear(
                *c,
                terms
                    .iter()
                    .map(|&(i, v)| Some((map(i)?, v)))
                    .collect::<Option<_>>()?,
            )),
        }
    }
    fn reads_before(&self, limit: usize) -> bool {
        match self {
            Self::Expression(e) => e.reads_before(limit),
            Self::Linear(_, terms) => terms.iter().all(|(i, _)| *i < limit),
        }
    }
    fn eval(&self, assignment: &[Fq]) -> Fq {
        match self {
            Self::Expression(e) => e.eval(assignment),
            Self::Linear(c, terms) => {
                *c + Fq::sum_of_products(
                    terms
                        .iter()
                        .map(|&(index, scale)| (assignment[index], scale)),
                )
            }
        }
    }
}

impl WitnessHint {
    fn eval(&self, value: Fq) -> Fq {
        match self {
            Self::Bit(index) => Fq::from_u64(
                value
                    .as_canonical_u64()
                    .checked_shr(*index as u32)
                    .unwrap_or(0)
                    & 1,
            ),
            Self::InverseOrZero if value != Fq::ZERO => value.inverse(),
            Self::InverseOrZero => Fq::ZERO,
            Self::QuotientFive => Fq::from_u64(value.as_canonical_u64() / 5),
            Self::RemainderFive => Fq::from_u64(value.as_canonical_u64() % 5),
        }
    }
}

impl WitnessExpr {
    fn reads_before(&self, limit: usize) -> bool {
        match self {
            Self::Var(column) => *column < limit,
            Self::Const(_) => true,
            Self::Add(l, r) | Self::Mul(l, r) => l.reads_before(limit) && r.reads_before(limit),
        }
    }

    fn map_columns(&self, map: &impl Fn(usize) -> Option<usize>) -> Option<Self> {
        Some(match self {
            Self::Var(column) => Self::Var(map(*column)?),
            Self::Const(value) => Self::Const(*value),
            Self::Add(left, right) => Self::Add(Box::new(left.map_columns(map)?), Box::new(right.map_columns(map)?)),
            Self::Mul(left, right) => Self::Mul(Box::new(left.map_columns(map)?), Box::new(right.map_columns(map)?)),
        })
    }

    fn eval(&self, assignment: &[Fq]) -> Fq {
        self.eval_with(&|column| assignment[column])
    }

    pub(super) fn eval_with(&self, value: &impl Fn(usize) -> Fq) -> Fq {
        match self {
            Self::Var(column) => value(*column),
            Self::Const(value) => *value,
            Self::Add(left, right) => left.eval_with(value) + right.eval_with(value),
            Self::Mul(left, right) => left.eval_with(value) * right.eval_with(value),
        }
    }
}

pub(super) fn validate_witness_batch(
    raw: RawWitnessBatch,
    witness_start: usize,
    private_column_count: usize,
    constant_column: usize,
    total_column_count: usize,
) -> Result<WitnessBatch, PackageError> {
    let RawWitnessBatch(start, recipes, hints) = raw;
    let start = usize::try_from(start).map_err(|_| PackageError::Malformed("witness batch start"))?;
    let output_length = recipes
        .len()
        .checked_add(hints.len())
        .ok_or(PackageError::Malformed("witness batch range overflow"))?;
    let end = start
        .checked_add(output_length)
        .ok_or(PackageError::Malformed("witness batch range overflow"))?;
    if output_length == 0 || start < witness_start || end > private_column_count {
        return Err(PackageError::Malformed("witness batch range"));
    }
    let recipes = recipes
        .iter()
        .enumerate()
        .map(|(offset, expression)| decode_expr(expression, start + offset, constant_column, total_column_count))
        .collect::<Result<Vec<_>, _>>()?;
    let hint_start = start + recipes.len();
    let mut sources = Vec::<(usize, WitnessExpr, bool)>::new();
    let hints = hints
        .iter()
        .enumerate()
        .map(|(offset, hint)| {
            let (kind, expr) = decode_hint(hint, hint_start + offset, constant_column, total_column_count)?;
            if sources.last().is_none_or(|(_, prev, _)| prev != &expr) {
                let reusable = expr.reads_before(hint_start + offset);
                sources.push((offset, expr, reusable));
            }
            Ok(kind)
        })
        .collect::<Result<_, PackageError>>()?;
    Ok(WitnessBatch {
        start,
        recipes,
        hints,
        sources: sources
            .into_iter()
            .map(|(offset, expr, reusable)| (offset, HintSource::compile(expr), reusable))
            .collect(),
    })
}

pub(super) fn validate_witness_batch_order(batches: &[WitnessBatch]) -> Result<(), PackageError> {
    if batches
        .windows(2)
        .any(|pair| pair[0].start >= pair[1].start || pair[0].end() > pair[1].start)
    {
        return Err(PackageError::Malformed("witness batch order"));
    }
    Ok(())
}

pub(super) fn execute_witness_batch(batch: &WitnessBatch, assignment: &mut [Fq]) {
    for (offset, recipe) in batch.recipes.iter().enumerate() {
        assignment[batch.start + offset] = recipe.eval(assignment);
    }
    let hint_start = batch.start + batch.recipes.len();
    for (group, (start, expr, reusable)) in batch.sources.iter().enumerate() {
        let end = batch
            .sources
            .get(group + 1)
            .map_or(batch.hints.len(), |(next, _, _)| *next);
        let saved = reusable.then(|| expr.eval(assignment));
        for offset in *start..end {
            let value = saved.unwrap_or_else(|| expr.eval(assignment));
            assignment[hint_start + offset] = batch.hints[offset].eval(value);
        }
    }
}

fn decode_hint(
    value: &Value,
    target: usize,
    constant_column: usize,
    total_column_count: usize,
) -> Result<(WitnessHint, WitnessExpr), PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Malformed("witness hint"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Malformed("witness hint tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, source, index]) => {
            let index = index
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(PackageError::Malformed("witness bit index"))?;
            Ok((
                WitnessHint::Bit(index),
                decode_expr(source, target, constant_column, total_column_count)?,
            ))
        }
        (1, [_, source]) => Ok((
            WitnessHint::InverseOrZero,
            decode_expr(source, target, constant_column, total_column_count)?,
        )),
        (2, [_, source]) => Ok((
            WitnessHint::QuotientFive,
            decode_expr(source, target, constant_column, total_column_count)?,
        )),
        (3, [_, source]) => Ok((
            WitnessHint::RemainderFive,
            decode_expr(source, target, constant_column, total_column_count)?,
        )),
        _ => Err(PackageError::Malformed("witness hint")),
    }
}

fn decode_expr(
    value: &Value,
    target: usize,
    constant_column: usize,
    total_column_count: usize,
) -> Result<WitnessExpr, PackageError> {
    decode_expr_with(value, &|column| {
        if column >= total_column_count || column == constant_column || (column < constant_column && column >= target) {
            Err(PackageError::Malformed("noncausal witness expression"))
        } else {
            Ok(())
        }
    })
}

fn decode_expr_with(
    value: &Value,
    validate_column: &impl Fn(usize) -> Result<(), PackageError>,
) -> Result<WitnessExpr, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Malformed("witness expression"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Malformed("witness expression tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, column]) => {
            let column = column
                .as_u64()
                .and_then(|column| usize::try_from(column).ok())
                .ok_or(PackageError::Malformed("witness expression column"))?;
            validate_column(column)?;
            Ok(WitnessExpr::Var(column))
        }
        (1, [_, constant]) => {
            let constant = constant
                .as_u64()
                .ok_or(PackageError::Malformed("witness expression constant"))?;
            if constant >= Fq::ORDER_U64 {
                return Err(PackageError::NonCanonicalField {
                    location: "witness expression constant",
                    value: constant,
                });
            }
            Ok(WitnessExpr::Const(Fq::from_u64(constant)))
        }
        (2, [_, left, right]) => Ok(WitnessExpr::Add(
            Box::new(decode_expr_with(left, validate_column)?),
            Box::new(decode_expr_with(right, validate_column)?),
        )),
        (3, [_, left, right]) => Ok(WitnessExpr::Mul(
            Box::new(decode_expr_with(left, validate_column)?),
            Box::new(decode_expr_with(right, validate_column)?),
        )),
        _ => Err(PackageError::Malformed("witness expression")),
    }
}

#[cfg(test)]
#[path = "../tests/unit/witness_hints.rs"]
mod tests;
