use std::ops::Range;
use std::sync::{Arc, LazyLock};

use neo_math::Fq;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::package::PackageError;

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct RawWitnessBatch(pub(super) u64, pub(super) Vec<Value>, pub(super) Vec<Value>);

#[derive(Clone, Debug)]
pub(super) enum WitnessExpr {
    Var(usize),
    Const(Fq),
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

type Batches = Arc<Vec<WitnessBatch>>;
type RestoreBatches = Box<dyn FnOnce() -> Option<Batches> + Send + Sync>;

#[derive(Clone, Debug)]
pub(super) enum WitnessProgram {
    Ready(Batches),
    Deferred {
        batches: Arc<LazyLock<Option<Batches>, RestoreBatches>>,
        bit_columns: Arc<[usize]>,
        count: usize,
    },
}

impl WitnessProgram {
    pub(super) fn new(batches: Vec<WitnessBatch>) -> Self {
        Self::Ready(Arc::new(batches))
    }

    pub(super) fn batches(&self) -> Result<&Batches, PackageError> {
        match self {
            Self::Ready(batches) => Ok(batches),
            Self::Deferred { batches, .. } => batches.as_ref().as_ref().ok_or_else(Self::invalid),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Ready(batches) => batches.len(),
            Self::Deferred { count, .. } => *count,
        }
    }

    pub(super) fn bit_columns(&self) -> Vec<usize> {
        match self {
            Self::Ready(batches) => batches.iter().flat_map(WitnessBatch::bit_columns).collect(),
            Self::Deferred { bit_columns, .. } => bit_columns.to_vec(),
        }
    }

    pub(super) fn remap(
        &self,
        replaced: Range<usize>,
        map: impl Fn(usize) -> Option<usize> + Send + Sync + 'static,
        defer: bool,
    ) -> Result<Self, PackageError> {
        let source = Arc::clone(self.batches()?);
        let keep = move |batch: &&WitnessBatch| !replaced.contains(&batch.start);
        let mut count = 0;
        let mut bit_columns = Vec::new();
        if defer {
            for batch in source.iter().filter(&keep) {
                let start = batch.mapped_start(&map).ok_or_else(Self::invalid)?;
                if !batch.references_map(&map) {
                    return Err(Self::invalid());
                }
                bit_columns.extend(
                    batch
                        .bit_columns()
                        .map(|column| start + (column - batch.start)),
                );
                count += 1;
            }
        }
        let restore = move || {
            source
                .iter()
                .filter(keep)
                .map(|batch| batch.map_columns(&map))
                .collect::<Option<Vec<_>>>()
                .map(Arc::new)
        };
        if defer {
            Ok(Self::Deferred {
                batches: Arc::new(LazyLock::new(Box::new(restore))),
                bit_columns: bit_columns.into(),
                count,
            })
        } else {
            restore().map(Self::Ready).ok_or_else(Self::invalid)
        }
    }

    fn invalid() -> PackageError {
        PackageError::Malformed("bounded-depth witness batch reads the application")
    }
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

    fn mapped_start(&self, map: &impl Fn(usize) -> Option<usize>) -> Option<usize> {
        let start = map(self.start)?;
        (map(self.end() - 1)? == start + self.recipes.len() + self.hints.len() - 1).then_some(start)
    }

    fn references_map(&self, map: &impl Fn(usize) -> Option<usize>) -> bool {
        self.recipes
            .iter()
            .chain(self.hints.iter().map(|hint| match hint {
                WitnessHint::Bit(input, _)
                | WitnessHint::InverseOrZero(input)
                | WitnessHint::QuotientFive(input)
                | WitnessHint::RemainderFive(input) => input,
            }))
            .all(|input| input.references_map(map))
    }

    fn map_columns(&self, map: &impl Fn(usize) -> Option<usize>) -> Option<Self> {
        Some(Self {
            start: self.mapped_start(map)?,
            recipes: self
                .recipes
                .iter()
                .map(|recipe| recipe.map_columns(map))
                .collect::<Option<_>>()?,
            hints: self
                .hints
                .iter()
                .map(|hint| {
                    Some(match hint {
                        WitnessHint::Bit(source, index) => WitnessHint::Bit(source.map_columns(map)?, *index),
                        WitnessHint::InverseOrZero(source) => WitnessHint::InverseOrZero(source.map_columns(map)?),
                        WitnessHint::QuotientFive(source) => WitnessHint::QuotientFive(source.map_columns(map)?),
                        WitnessHint::RemainderFive(source) => WitnessHint::RemainderFive(source.map_columns(map)?),
                    })
                })
                .collect::<Option<_>>()?,
        })
    }
}

impl WitnessHint {
    fn eval(&self, assignment: &[Fq]) -> Fq {
        match self {
            Self::Bit(source, index) => {
                let value = source.eval(assignment).as_canonical_u64();
                Fq::from_u64(value.checked_shr(*index as u32).unwrap_or(0) & 1)
            }
            Self::InverseOrZero(source) => {
                let value = source.eval(assignment);
                if value == Fq::ZERO {
                    Fq::ZERO
                } else {
                    value.inverse()
                }
            }
            Self::QuotientFive(source) => Fq::from_u64(source.eval(assignment).as_canonical_u64() / 5),
            Self::RemainderFive(source) => Fq::from_u64(source.eval(assignment).as_canonical_u64() % 5),
        }
    }
}

impl WitnessExpr {
    fn references_map(&self, map: &impl Fn(usize) -> Option<usize>) -> bool {
        match self {
            Self::Var(column) => map(*column).is_some(),
            Self::Const(_) => true,
            Self::Add(left, right) | Self::Mul(left, right) => left.references_map(map) && right.references_map(map),
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
        return Err(PackageError::Malformed("witness batch order"));
    }
    Ok(())
}

pub(super) fn execute_witness_batch(batch: &WitnessBatch, assignment: &mut [Fq]) {
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
#[path = "../tests/unit/bounded_depth_witness_template.rs"]
mod tests;
