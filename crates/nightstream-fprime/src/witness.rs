//! Decoder and evaluator for the Lean-owned symbolic witness-expression IR.

use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::package::{PackageError, GOLDILOCKS_MODULUS};

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawWitnessBatch(pub(super) u64, pub(super) Vec<Value>, pub(super) Vec<Value>);

#[derive(Clone, Debug)]
pub(super) enum WitnessExpr {
    Var(usize),
    Const(Goldilocks),
    Add(Box<WitnessExpr>, Box<WitnessExpr>),
    Mul(Box<WitnessExpr>, Box<WitnessExpr>),
}

#[derive(Clone, Debug)]
enum WitnessHint {
    Bit(WitnessExpr, usize),
    InverseOrZero(WitnessExpr),
    QuotientFive(WitnessExpr),
    RemainderFive(WitnessExpr),
}

#[derive(Clone, Debug)]
pub(super) struct WitnessBatch {
    pub(super) start: usize,
    recipes: Vec<WitnessExpr>,
    hints: Vec<WitnessHint>,
}

impl WitnessBatch {
    pub(super) fn end(&self) -> usize {
        self.start + self.recipes.len() + self.hints.len()
    }
}

impl WitnessHint {
    fn eval(&self, assignment: &[Goldilocks]) -> Goldilocks {
        match self {
            Self::Bit(source, index) => {
                let value = source.eval(assignment).as_canonical_u64();
                Goldilocks::from_u64(value.checked_shr(*index as u32).unwrap_or(0) & 1)
            }
            Self::InverseOrZero(source) => {
                let value = source.eval(assignment);
                if value == Goldilocks::ZERO {
                    Goldilocks::ZERO
                } else {
                    value.inverse()
                }
            }
            Self::QuotientFive(source) => Goldilocks::from_u64(source.eval(assignment).as_canonical_u64() / 5),
            Self::RemainderFive(source) => Goldilocks::from_u64(source.eval(assignment).as_canonical_u64() % 5),
        }
    }
}

impl WitnessExpr {
    fn eval(&self, assignment: &[Goldilocks]) -> Goldilocks {
        self.eval_with(&|column| assignment[column])
    }

    pub(super) fn eval_with(&self, value: &impl Fn(usize) -> Goldilocks) -> Goldilocks {
        match self {
            Self::Var(column) => value(*column),
            Self::Const(value) => *value,
            Self::Add(left, right) => left.eval_with(value) + right.eval_with(value),
            Self::Mul(left, right) => left.eval_with(value) * right.eval_with(value),
        }
    }
}

pub(super) fn decode_template_expr(
    value: &Value,
    input_count: usize,
    output_input: usize,
) -> Result<WitnessExpr, PackageError> {
    decode_expr_with(value, &|column| {
        if column >= input_count || column == output_input {
            Err(PackageError::Invalid("compact output recipe input"))
        } else {
            Ok(())
        }
    })
}

pub(super) fn validate_witness_batch(
    raw: RawWitnessBatch,
    witness_start: usize,
    private_column_count: usize,
    constant_column: usize,
    total_column_count: usize,
) -> Result<WitnessBatch, PackageError> {
    let RawWitnessBatch(start, recipes, hints) = raw;
    let start = usize::try_from(start).map_err(|_| PackageError::Invalid("witness batch start"))?;
    let output_length = recipes
        .len()
        .checked_add(hints.len())
        .ok_or(PackageError::Invalid("witness batch range overflow"))?;
    let end = start
        .checked_add(output_length)
        .ok_or(PackageError::Invalid("witness batch range overflow"))?;
    if output_length == 0 || start < witness_start || end > private_column_count {
        return Err(PackageError::Invalid("witness batch range"));
    }
    let recipes = recipes
        .iter()
        .enumerate()
        .map(|(offset, expression)| decode_expr(expression, start + offset, constant_column, total_column_count))
        .collect::<Result<Vec<_>, _>>()?;
    let hint_start = start + recipes.len();
    let hints = hints
        .iter()
        .enumerate()
        .map(|(offset, hint)| decode_hint(hint, hint_start + offset, constant_column, total_column_count))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WitnessBatch { start, recipes, hints })
}

pub(super) fn validate_witness_batch_order(batches: &[WitnessBatch]) -> Result<(), PackageError> {
    if batches
        .windows(2)
        .any(|pair| pair[0].start >= pair[1].start || pair[0].end() > pair[1].start)
    {
        return Err(PackageError::Invalid("witness batch order"));
    }
    Ok(())
}

pub(super) fn validate_witness_coverage(
    witness_start: usize,
    witness_length: usize,
    mut intervals: Vec<(usize, usize)>,
) -> Result<(), PackageError> {
    intervals.sort_unstable();
    let mut cursor = witness_start;
    for (start, end) in intervals {
        if start != cursor || end <= start {
            return Err(PackageError::Invalid("witness column coverage"));
        }
        cursor = end;
    }
    let witness_end = witness_start
        .checked_add(witness_length)
        .ok_or(PackageError::Invalid("witness range overflow"))?;
    if cursor != witness_end {
        return Err(PackageError::Invalid("witness column count coverage"));
    }
    Ok(())
}

pub(super) fn execute_witness_batch(batch: &WitnessBatch, assignment: &mut [Goldilocks]) {
    for (offset, recipe) in batch.recipes.iter().enumerate() {
        assignment[batch.start + offset] = recipe.eval(assignment);
    }
    let hint_start = batch.start + batch.recipes.len();
    for (offset, hint) in batch.hints.iter().enumerate() {
        assignment[hint_start + offset] = hint.eval(assignment);
    }
}

fn decode_hint(
    value: &Value,
    target: usize,
    constant_column: usize,
    total_column_count: usize,
) -> Result<WitnessHint, PackageError> {
    let fields = value
        .as_array()
        .ok_or(PackageError::Invalid("witness hint"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Invalid("witness hint tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, source, index]) => {
            let index = index
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(PackageError::Invalid("witness bit index"))?;
            Ok(WitnessHint::Bit(
                decode_expr(source, target, constant_column, total_column_count)?,
                index,
            ))
        }
        (1, [_, source]) => Ok(WitnessHint::InverseOrZero(decode_expr(
            source,
            target,
            constant_column,
            total_column_count,
        )?)),
        (2, [_, source]) => Ok(WitnessHint::QuotientFive(decode_expr(
            source,
            target,
            constant_column,
            total_column_count,
        )?)),
        (3, [_, source]) => Ok(WitnessHint::RemainderFive(decode_expr(
            source,
            target,
            constant_column,
            total_column_count,
        )?)),
        _ => Err(PackageError::Invalid("witness hint")),
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
            Err(PackageError::Invalid("noncausal witness expression"))
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
        .ok_or(PackageError::Invalid("witness expression"))?;
    let tag = fields
        .first()
        .and_then(Value::as_u64)
        .ok_or(PackageError::Invalid("witness expression tag"))?;
    match (tag, fields.as_slice()) {
        (0, [_, column]) => {
            let column = column
                .as_u64()
                .and_then(|column| usize::try_from(column).ok())
                .ok_or(PackageError::Invalid("witness expression column"))?;
            validate_column(column)?;
            Ok(WitnessExpr::Var(column))
        }
        (1, [_, constant]) => {
            let constant = constant
                .as_u64()
                .ok_or(PackageError::Invalid("witness expression constant"))?;
            if constant >= GOLDILOCKS_MODULUS {
                return Err(PackageError::NonCanonicalField {
                    location: "witness expression constant",
                    value: constant,
                });
            }
            Ok(WitnessExpr::Const(Goldilocks::from_u64(constant)))
        }
        (2, [_, left, right]) => Ok(WitnessExpr::Add(
            Box::new(decode_expr_with(left, validate_column)?),
            Box::new(decode_expr_with(right, validate_column)?),
        )),
        (3, [_, left, right]) => Ok(WitnessExpr::Mul(
            Box::new(decode_expr_with(left, validate_column)?),
            Box::new(decode_expr_with(right, validate_column)?),
        )),
        _ => Err(PackageError::Invalid("witness expression")),
    }
}
