//! Strict decoding and execution of the Lean-owned final-assignment transport.
//!
//! The decoded plan retains every compact recipe, expression, block, and
//! affine source run. Execution reads a Rust-produced physical assignment and
//! constructs the exact balanced logical assignment without a Lean runtime.

use serde_json::Value;

use crate::WitnessAssignment;

use super::{Layout, PackageError, GOLDILOCKS_MODULUS};

const TRANSPORT_SCHEMA: usize = 4;
const FIELD_COORDINATES: usize = 41;
const OUTPUT_DIGEST_WORDS: usize = 4;
const PHI81_INVOCATIONS: usize = 52_326;
const PHI81_RING_DEGREE: usize = 54;
const CENTERED_HALF_MODULUS: u64 = (GOLDILOCKS_MODULUS - 1) / 2;

mod wide;

/// The package selects one exact retained-value transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedAssignmentPlan {
    program: wide::Plan,
}

impl LoadedAssignmentPlan {
    pub(super) fn execute(
        &self,
        layout: &Layout,
        assignment: &WitnessAssignment,
    ) -> Result<LogicalAssignment, PackageError> {
        self.program.execute(layout, assignment)
    }
}

pub(super) fn decode(
    value: &Value,
    physical_width: usize,
    logical_public_width: usize,
    logical_width: usize,
) -> Result<LoadedAssignmentPlan, PackageError> {
    let schema = value
        .as_array()
        .and_then(|fields| fields.first())
        .ok_or(PackageError::Invalid("assignment transport plan"))?;
    if word(schema, "assignment transport schema")? != TRANSPORT_SCHEMA {
        return Err(PackageError::Invalid("assignment transport schema version"));
    }
    let program = wide::decode(value, physical_width, logical_public_width, logical_width)?;
    Ok(LoadedAssignmentPlan { program })
}

/// Exact balanced logical assignment produced by the schema-4 transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalAssignment {
    values: Vec<i8>,
}

impl LogicalAssignment {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn value(&self, column: usize) -> Result<u64, PackageError> {
        match self.values.get(column).copied() {
            Some(-1) => Ok(GOLDILOCKS_MODULUS - 1),
            Some(0) => Ok(0),
            Some(1) => Ok(1),
            Some(_) => Err(PackageError::Invalid("logical assignment coordinate")),
            None => Err(PackageError::Invalid("logical assignment column")),
        }
    }

    pub fn balanced_values(&self) -> &[i8] {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotKind {
    Bit,
    Centered,
    Field,
}

impl SlotKind {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        match word(value, "assignment slot kind")? {
            0 => Ok(Self::Bit),
            1 => Ok(Self::Centered),
            2 => Ok(Self::Field),
            _ => Err(PackageError::Invalid("assignment slot kind")),
        }
    }

    const fn coordinate_width(self) -> usize {
        match self {
            Self::Bit | Self::Centered => 1,
            Self::Field => FIELD_COORDINATES,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Expression {
    Column(usize),
    Constant(u64),
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
}

impl Expression {
    fn decode(value: &Value, physical_width: usize) -> Result<Self, PackageError> {
        let fields = value
            .as_array()
            .ok_or(PackageError::Invalid("assignment expression"))?;
        let tag = fields
            .first()
            .and_then(Value::as_u64)
            .ok_or(PackageError::Invalid("assignment expression tag"))?;
        match (tag, fields.as_slice()) {
            (0, [_, column]) => {
                let column = word(column, "assignment expression column")?;
                if column >= physical_width {
                    return Err(PackageError::Invalid("assignment expression column bound"));
                }
                Ok(Self::Column(column))
            }
            (1, [_, constant]) => {
                let constant = constant
                    .as_u64()
                    .ok_or(PackageError::Invalid("assignment expression constant"))?;
                if constant >= GOLDILOCKS_MODULUS {
                    return Err(PackageError::NonCanonicalField {
                        location: "assignment expression constant",
                        value: constant,
                    });
                }
                Ok(Self::Constant(constant))
            }
            (2, [_, left, right]) => Ok(Self::Add(
                Box::new(Self::decode(left, physical_width)?),
                Box::new(Self::decode(right, physical_width)?),
            )),
            (3, [_, left, right]) => Ok(Self::Multiply(
                Box::new(Self::decode(left, physical_width)?),
                Box::new(Self::decode(right, physical_width)?),
            )),
            _ => Err(PackageError::Invalid("assignment expression")),
        }
    }

    fn evaluate(&self, physical: &PhysicalAssignment<'_>) -> Result<u64, PackageError> {
        match self {
            Self::Column(column) => physical.value(*column),
            Self::Constant(value) => Ok(*value),
            Self::Add(left, right) => Ok(add_mod(left.evaluate(physical)?, right.evaluate(physical)?)),
            Self::Multiply(left, right) => Ok(mul_mod(left.evaluate(physical)?, right.evaluate(physical)?)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    first: usize,
    step: usize,
    count: usize,
    last: usize,
}

fn decode_source_runs(value: &Value, expected_count: usize, domain_width: usize) -> Result<Vec<Run>, PackageError> {
    let run_values = value
        .as_array()
        .ok_or(PackageError::Invalid("assignment source runs"))?;
    let mut runs = Vec::with_capacity(run_values.len());
    let mut covered = 0usize;
    for value in run_values {
        let fields = exact_array(value, 3, "assignment source run")?;
        let first = word(&fields[0], "assignment source run first")?;
        let step = word(&fields[1], "assignment source run step")?;
        let count = word(&fields[2], "assignment source run count")?;
        if count == 0 || (count == 1 && step != 0) {
            return Err(PackageError::Invalid("noncanonical assignment source run"));
        }
        let last = first
            .checked_add(
                step.checked_mul(count - 1)
                    .ok_or(PackageError::Invalid("assignment source run overflow"))?,
            )
            .ok_or(PackageError::Invalid("assignment source run overflow"))?;
        if last >= domain_width {
            return Err(PackageError::Invalid("assignment source domain bound"));
        }
        covered = covered
            .checked_add(count)
            .ok_or(PackageError::Invalid("assignment source run count overflow"))?;
        runs.push(Run {
            first,
            step,
            count,
            last,
        });
    }
    if covered != expected_count {
        return Err(PackageError::Invalid("assignment source run coverage"));
    }
    validate_canonical_run_boundaries(&runs)?;

    Ok(runs)
}

fn source_run_at(runs: &[Run], slot: usize) -> Result<usize, PackageError> {
    let mut run_start = 0usize;
    for run in runs {
        let run_end = run_start
            .checked_add(run.count)
            .ok_or(PackageError::Invalid("assignment source run count overflow"))?;
        if slot < run_end {
            return run
                .first
                .checked_add(
                    run.step
                        .checked_mul(slot - run_start)
                        .ok_or(PackageError::Invalid("assignment source run overflow"))?,
                )
                .ok_or(PackageError::Invalid("assignment source run overflow"));
        }
        run_start = run_end;
    }
    Err(PackageError::Invalid("assignment source run coverage"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Phi81FamilyShape {
    source_count: usize,
    block_count: usize,
    cell_count: usize,
    first_invocation: usize,
}

impl Phi81FamilyShape {
    fn invocation(
        self,
        ring_degree: usize,
        source: usize,
        block: usize,
        lane: usize,
        cell: usize,
    ) -> Result<usize, PackageError> {
        if source >= self.source_count || block >= self.block_count || lane >= ring_degree || cell >= self.cell_count {
            return Err(PackageError::Invalid("Phi81 invocation coordinate"));
        }
        let source_offset = source
            .checked_mul(self.block_count)
            .and_then(|offset| offset.checked_mul(ring_degree))
            .and_then(|offset| offset.checked_mul(self.cell_count))
            .ok_or(PackageError::Invalid("Phi81 invocation index overflow"))?;
        let block_offset = block
            .checked_mul(ring_degree)
            .and_then(|offset| offset.checked_mul(self.cell_count))
            .ok_or(PackageError::Invalid("Phi81 invocation index overflow"))?;
        let lane_offset = lane
            .checked_mul(self.cell_count)
            .ok_or(PackageError::Invalid("Phi81 invocation index overflow"))?;
        self.first_invocation
            .checked_add(source_offset)
            .and_then(|index| index.checked_add(block_offset))
            .and_then(|index| index.checked_add(lane_offset))
            .and_then(|index| index.checked_add(cell))
            .ok_or(PackageError::Invalid("Phi81 invocation index overflow"))
    }
}

fn decode_expressions(
    value: &Value,
    expected_count: usize,
    physical_width: usize,
) -> Result<Vec<Expression>, PackageError> {
    let expressions = exact_array(value, expected_count, "assignment expressions")?;
    expressions
        .iter()
        .map(|expression| Expression::decode(expression, physical_width))
        .collect()
}

#[derive(Clone, Copy)]
struct PhysicalAssignment<'a> {
    private_values: &'a [u64],
    public_values: &'a [u64],
    constant_column: usize,
    total_columns: usize,
}

impl<'a> PhysicalAssignment<'a> {
    fn new(layout: &Layout, assignment: &'a WitnessAssignment, physical_width: usize) -> Result<Self, PackageError> {
        let total_columns = layout
            .constant_column
            .checked_add(1)
            .and_then(|width| width.checked_add(layout.public_column_count))
            .ok_or(PackageError::Invalid("physical assignment width overflow"))?;
        if layout.private_column_count != layout.constant_column
            || total_columns != layout.total_column_count
            || physical_width != layout.total_column_count
            || assignment.private_values().len() != layout.private_column_count
            || assignment.public_values().len() != layout.public_column_count
        {
            return Err(PackageError::Invalid("physical assignment dimensions"));
        }
        Ok(Self {
            private_values: assignment.private_values(),
            public_values: assignment.public_values(),
            constant_column: layout.constant_column,
            total_columns,
        })
    }

    fn value(&self, column: usize) -> Result<u64, PackageError> {
        if column >= self.total_columns {
            return Err(PackageError::Invalid("physical assignment column"));
        }
        let value = if column < self.constant_column {
            *self
                .private_values
                .get(column)
                .ok_or(PackageError::Invalid("physical private assignment column"))?
        } else if column == self.constant_column {
            1
        } else {
            *self
                .public_values
                .get(column - self.constant_column - 1)
                .ok_or(PackageError::Invalid("physical public assignment column"))?
        };
        if value >= GOLDILOCKS_MODULUS {
            return Err(PackageError::NonCanonicalField {
                location: "physical assignment",
                value,
            });
        }
        Ok(value)
    }
}

/// Canonical quotient of the raw product by X^54 + X^27 + 1.
fn phi81_quotient(left: &[u64; PHI81_RING_DEGREE], right: &[u64; PHI81_RING_DEGREE]) -> [u64; PHI81_RING_DEGREE] {
    let mut high_product = [0u64; 2 * PHI81_RING_DEGREE - 1];
    for (left_degree, &left_value) in left.iter().enumerate() {
        for (right_degree, &right_value) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            if degree >= PHI81_RING_DEGREE {
                high_product[degree] = add_mod(high_product[degree], mul_mod(left_value, right_value));
            }
        }
    }
    std::array::from_fn(|degree| {
        sub_mod(
            high_product.get(degree + 54).copied().unwrap_or(0),
            high_product.get(degree + 81).copied().unwrap_or(0),
        )
    })
}

fn append_public(
    digest: [u64; OUTPUT_DIGEST_WORDS],
    logical_public_width: usize,
    output: &mut Vec<i8>,
) -> Result<(), PackageError> {
    output.push(1);
    for word in digest {
        for bit in 0..u64::BITS {
            output.push(((word >> bit) & 1) as i8);
        }
    }
    let padding = logical_public_width
        .checked_sub(output.len())
        .ok_or(PackageError::Invalid("logical public assignment width"))?;
    output.extend(std::iter::repeat_n(0, padding));
    Ok(())
}

fn encode_slot(kind: SlotKind, value: u64, output: &mut Vec<i8>) -> Result<(), PackageError> {
    if value >= GOLDILOCKS_MODULUS {
        return Err(PackageError::NonCanonicalField {
            location: "logical assignment source",
            value,
        });
    }
    let start = output.len();
    match kind {
        SlotKind::Bit => match value {
            0 | 1 => output.push(value as i8),
            _ => return Err(PackageError::Invalid("bit assignment source")),
        },
        SlotKind::Centered => match value {
            0 => output.push(0),
            1 => output.push(1),
            value if value == GOLDILOCKS_MODULUS - 1 => output.push(-1),
            _ => return Err(PackageError::Invalid("centered assignment source")),
        },
        SlotKind::Field => encode_field(value, output),
    }
    if output.len() - start != kind.coordinate_width() {
        return Err(PackageError::Invalid("assignment slot coordinate width"));
    }
    Ok(())
}

fn encode_field(value: u64, output: &mut Vec<i8>) {
    let negative = value > CENTERED_HALF_MODULUS;
    let mut magnitude = if negative { GOLDILOCKS_MODULUS - value } else { value };
    for _ in 0..FIELD_COORDINATES {
        let remainder = magnitude % 3;
        let unsigned = match remainder {
            0 => 0,
            1 => 1,
            _ => -1,
        };
        output.push(if negative { -unsigned } else { unsigned });
        magnitude = magnitude / 3 + u64::from(remainder == 2);
    }
    debug_assert_eq!(magnitude, 0);
}

fn add_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) + u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn mul_mod(left: u64, right: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(GOLDILOCKS_MODULUS)) as u64
}

fn neg_mod(value: u64) -> u64 {
    if value == 0 {
        0
    } else {
        GOLDILOCKS_MODULUS - value
    }
}

fn sub_mod(left: u64, right: u64) -> u64 {
    add_mod(left, neg_mod(right))
}

fn validate_canonical_run_boundaries(runs: &[Run]) -> Result<(), PackageError> {
    for pair in runs.windows(2) {
        let left_last = pair[0].last;
        let right = pair[1];
        let mergeable = if right.count == 1 {
            left_last <= right.first
        } else {
            left_last.checked_add(right.step) == Some(right.first)
        };
        if mergeable {
            return Err(PackageError::Invalid("noncanonical assignment source runs"));
        }
    }
    Ok(())
}

fn exact_array<'a>(value: &'a Value, expected_len: usize, location: &'static str) -> Result<&'a [Value], PackageError> {
    let values = value.as_array().ok_or(PackageError::Invalid(location))?;
    if values.len() != expected_len {
        return Err(PackageError::Invalid(location));
    }
    Ok(values)
}

fn word(value: &Value, location: &'static str) -> Result<usize, PackageError> {
    value
        .as_u64()
        .and_then(|word| usize::try_from(word).ok())
        .ok_or(PackageError::Invalid(location))
}

#[cfg(test)]
#[path = "../../tests/unit/phi81_quotient_assignment.rs"]
mod quotient_tests;

#[cfg(test)]
#[path = "../../tests/unit/wide_witness_parity.rs"]
mod wide_witness_tests;
