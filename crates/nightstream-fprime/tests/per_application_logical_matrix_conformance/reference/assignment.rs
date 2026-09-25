//! Independent construction of the final balanced logical assignment.
//!
//! This module reads only raw sealed bytes and raw physical assignment
//! slices. It does not import the production package, witness, or matrix APIs.

use serde::{de::IgnoredAny, Deserialize};
use serde_json::Value;
use std::{ops::Range, sync::OnceLock};

use super::{array, exact_array, field, word, Field, Result, GOLDILOCKS_MODULUS};

const SEALED_SCHEMA: usize = 6;
const INNER_SCHEMA: usize = 8;
const TRANSPORT_SCHEMA: usize = 4;
pub(super) const BLOCK_COUNT: usize = 76;
const PHYSICAL_ROWS: usize = 27_724_114;
const PHYSICAL_COLUMNS: usize = 27_867_239;
const PHYSICAL_PUBLIC: usize = 278;
const LOGICAL_PUBLIC: usize = 270;
const LOGICAL_WIDTH: usize = 137_341_846;
const CARRIER_WIDTH: usize = 137_341_872;
const FIELD_COORDINATES: usize = 41;
const OUTPUT_DIGEST_WORDS: usize = 4;
const PHI81_INVOCATIONS: usize = 52_326;
const PHI81_QUOTIENT_VALUES: usize = PHI81_INVOCATIONS;
const CHALLENGE_SOURCES: usize = 17;
const CHALLENGE_COEFFICIENTS: usize = 54;
const CHALLENGE_BITS: usize = 3;
const OUTPUT_DIGEST_BLOCK: usize = 17;
const PHI81_QUOTIENT_BLOCK: usize = 75;
const CENTERED_HALF_MODULUS: u64 = (GOLDILOCKS_MODULUS - 1) / 2;

#[derive(Deserialize)]
struct RawSealed(u64, RawCircuit, IgnoredAny, IgnoredAny, Value, IgnoredAny, u64);

#[derive(Deserialize)]
struct RawCircuit(
    u64,
    IgnoredAny,
    IgnoredAny,
    RawLayout,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
);

#[derive(Deserialize)]
struct RawLayout(u64, u64, u64, u64, u64, IgnoredAny, IgnoredAny);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotKind {
    Bit,
    Centered,
    Field,
}

impl SlotKind {
    fn decode(value: &Value) -> Result<Self> {
        match word(value, "assignment slot kind")? {
            0 => Ok(Self::Bit),
            1 => Ok(Self::Centered),
            2 => Ok(Self::Field),
            _ => Err("unknown assignment slot kind".into()),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::Bit | Self::Centered => 1,
            Self::Field => FIELD_COORDINATES,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Run {
    first: usize,
    step: usize,
    count: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct BlockPlan {
    index: usize,
    kind: SlotKind,
    slot_count: usize,
    runs: Vec<Run>,
}

fn decode_runs(value: &Value, expected_count: usize) -> Result<Vec<Run>> {
    let mut end = 0usize;
    let runs = array(value, "assignment source runs")?
        .iter()
        .map(|run| {
            let fields = exact_array(run, 3, "assignment source run")?;
            let first = word(&fields[0], "assignment source run first")?;
            let step = word(&fields[1], "assignment source run step")?;
            let count = word(&fields[2], "assignment source run count")?;
            if count == 0 || (count == 1 && step != 0) {
                return Err("noncanonical assignment source run".into());
            }
            first
                .checked_add(
                    step.checked_mul(count - 1)
                        .ok_or_else(|| "assignment source run offset overflow".to_string())?,
                )
                .ok_or_else(|| "assignment source run value overflow".to_string())?;
            end = end
                .checked_add(count)
                .ok_or_else(|| "assignment source run count overflow".to_string())?;
            Ok(Run {
                first,
                step,
                count,
                end,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if end != expected_count {
        return Err(format!(
            "assignment source runs cover {end} slots, expected {expected_count}"
        ));
    }
    Ok(runs)
}

fn source_at(runs: &[Run], slot: usize) -> Result<usize> {
    let position = runs.partition_point(|run| run.end <= slot);
    let run = runs
        .get(position)
        .ok_or_else(|| "assignment source-run gap".to_string())?;
    let run_start = run.end - run.count;
    run.first
        .checked_add(
            run.step
                .checked_mul(slot - run_start)
                .ok_or_else(|| "assignment source offset overflow".to_string())?,
        )
        .ok_or_else(|| "assignment source value overflow".to_string())
}

impl BlockPlan {
    fn decode(value: &Value, index: usize) -> Result<Self> {
        let fields = exact_array(value, 3, "assignment block plan")?;
        let kind = SlotKind::decode(&fields[0])?;
        let slot_count = word(&fields[1], "assignment block slot count")?;
        let runs = decode_runs(&fields[2], slot_count)?;
        if runs
            .iter()
            .any(|run| run.first + run.step * (run.count - 1) >= PHYSICAL_COLUMNS + PHI81_QUOTIENT_VALUES)
        {
            return Err(format!("assignment block {index} source is out of range"));
        }
        Ok(Self {
            index,
            kind,
            slot_count,
            runs,
        })
    }

    fn source(&self, slot: usize) -> Result<usize> {
        if slot >= self.slot_count {
            return Err(format!("assignment block {} slot {slot} is out of range", self.index));
        }
        source_at(&self.runs, slot)
    }

    fn sources(&self) -> impl Iterator<Item = Result<usize>> + '_ {
        self.runs.iter().flat_map(|run| {
            (0..run.count).map(|offset| {
                run.first
                    .checked_add(
                        run.step
                            .checked_mul(offset)
                            .ok_or_else(|| "assignment source offset overflow".to_string())?,
                    )
                    .ok_or_else(|| "assignment source value overflow".to_string())
            })
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Family {
    source_count: usize,
    block_count: usize,
    cell_count: usize,
    first_invocation: usize,
}

impl Family {
    fn invocation_count(self) -> Result<usize> {
        self.source_count
            .checked_mul(self.block_count)
            .and_then(|count| count.checked_mul(54))
            .and_then(|count| count.checked_mul(self.cell_count))
            .ok_or_else(|| "Phi81 family invocation count overflow".to_string())
    }

    fn invocation(self, source: usize, block: usize, lane: usize, cell: usize) -> Result<usize> {
        if source >= self.source_count || block >= self.block_count || lane >= 54 || cell >= self.cell_count {
            return Err("Phi81 invocation coordinate is out of range".into());
        }
        self.first_invocation
            .checked_add(source * self.block_count * 54 * self.cell_count)
            .and_then(|value| value.checked_add(block * 54 * self.cell_count))
            .and_then(|value| value.checked_add(lane * self.cell_count))
            .and_then(|value| value.checked_add(cell))
            .ok_or_else(|| "Phi81 invocation index overflow".to_string())
    }
}

#[derive(Clone, Debug)]
struct Phi81Plan {
    families: Vec<Family>,
    value_sources: Vec<Run>,
    challenge_bit_sources: Vec<Run>,
}

impl Phi81Plan {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 3, "Phi81 assignment plan")?;
        let expected_shapes = [[17, 22, 1], [17, 5, 1], [17, 1, 2], [17, 14, 2]];
        let shape_values = exact_array(&fields[0], expected_shapes.len(), "Phi81 family shapes")?;
        let mut first_invocation = 0usize;
        let mut families = Vec::with_capacity(expected_shapes.len());
        for (value, expected) in shape_values.iter().zip(expected_shapes) {
            let shape = exact_array(value, 3, "Phi81 family shape")?
                .iter()
                .map(|value| word(value, "Phi81 family dimension"))
                .collect::<Result<Vec<_>>>()?;
            if shape != expected {
                return Err("unexpected Phi81 family shape".into());
            }
            let family = Family {
                source_count: shape[0],
                block_count: shape[1],
                cell_count: shape[2],
                first_invocation,
            };
            first_invocation = first_invocation
                .checked_add(family.invocation_count()?)
                .ok_or_else(|| "Phi81 invocation total overflow".to_string())?;
            families.push(family);
        }
        if first_invocation != PHI81_INVOCATIONS {
            return Err("unexpected Phi81 invocation count".into());
        }
        let value_sources = decode_runs(&fields[1], PHI81_INVOCATIONS)?;
        let challenge_bit_sources =
            decode_runs(&fields[2], CHALLENGE_SOURCES * CHALLENGE_COEFFICIENTS * CHALLENGE_BITS)?;
        if value_sources
            .iter()
            .chain(&challenge_bit_sources)
            .any(|run| run.first + run.step * (run.count - 1) >= PHYSICAL_COLUMNS)
        {
            return Err("Phi81 operand source is outside the physical assignment".into());
        }
        Ok(Self {
            families,
            value_sources,
            challenge_bit_sources,
        })
    }
}

struct Transport {
    blocks: Vec<BlockPlan>,
    phi81: Phi81Plan,
    output_digest_expressions: Vec<Value>,
}

impl Transport {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 4, "assignment transport plan")?;
        if word(&fields[0], "assignment transport schema")? != TRANSPORT_SCHEMA {
            return Err("unexpected assignment transport schema".into());
        }
        let block_values = exact_array(&fields[1], BLOCK_COUNT, "assignment block plans")?;
        let blocks = block_values
            .iter()
            .enumerate()
            .map(|(index, value)| BlockPlan::decode(value, index))
            .collect::<Result<Vec<_>>>()?;
        let phi81 = Phi81Plan::decode(&fields[2])?;
        let output_digest_expressions =
            exact_array(&fields[3], OUTPUT_DIGEST_WORDS, "output-digest expressions")?.to_vec();
        Ok(Self {
            blocks,
            phi81,
            output_digest_expressions,
        })
    }

    fn block(&self, index: usize) -> Result<&BlockPlan> {
        self.blocks
            .get(index)
            .ok_or_else(|| format!("missing assignment block {index}"))
    }
}

struct Physical<'a> {
    private: &'a [u64],
    public: &'a [u64],
    constant_column: usize,
    total_columns: usize,
    unavailable_private: Option<Range<usize>>,
}

impl Physical<'_> {
    fn value(&self, column: usize) -> Result<u64> {
        if column >= self.total_columns {
            return Err(format!("physical assignment column {column} is out of range"));
        }
        if self
            .unavailable_private
            .as_ref()
            .is_some_and(|range| range.contains(&column))
        {
            return Err(format!("physical private column {column} is unavailable"));
        }
        let value = if column < self.constant_column {
            self.private
                .get(column)
                .copied()
                .ok_or_else(|| format!("physical private column {column} is unavailable"))?
        } else if column == self.constant_column {
            1
        } else {
            let public = column - self.constant_column - 1;
            self.public
                .get(public)
                .copied()
                .ok_or_else(|| format!("physical public column {public} is unavailable"))?
        };
        if value >= GOLDILOCKS_MODULUS {
            return Err(format!("physical assignment column {column} is noncanonical"));
        }
        Ok(value)
    }
}

struct Domains<'a> {
    physical: Physical<'a>,
    quotients: Vec<u64>,
}

impl Domains<'_> {
    fn retained(&self, index: usize) -> Result<u64> {
        if index < PHYSICAL_COLUMNS {
            return self.physical.value(index);
        }
        let index = index - PHYSICAL_COLUMNS;
        self.quotients
            .get(index)
            .copied()
            .ok_or_else(|| "retained assignment source is out of range".into())
    }
}

#[derive(Clone, Debug)]
pub struct LogicalAssignment {
    values: Vec<i8>,
    block_ranges: Vec<Range<usize>>,
}

impl LogicalAssignment {
    pub fn decode(sealed_bytes: &[u8], private_values: &[u64], public_values: &[u64]) -> Result<Self> {
        if sealed_bytes.last() != Some(&b'\n') {
            return Err("sealed package is not newline terminated".into());
        }
        let RawSealed(
            outer_schema,
            RawCircuit(inner_schema, _, _, layout, _, _, _, _, _, _, _, _, _, _),
            _,
            _,
            raw_transport,
            _,
            logical_public,
        ) = serde_json::from_slice(sealed_bytes).map_err(|error| format!("independent assignment decode: {error}"))?;
        let RawLayout(rows, private, constant, public, total, _, _) = layout;
        if usize::try_from(outer_schema).ok() != Some(SEALED_SCHEMA)
            || usize::try_from(inner_schema).ok() != Some(INNER_SCHEMA)
            || usize::try_from(rows).ok() != Some(PHYSICAL_ROWS)
            || usize::try_from(total).ok() != Some(PHYSICAL_COLUMNS)
            || usize::try_from(public).ok() != Some(PHYSICAL_PUBLIC)
            || usize::try_from(logical_public).ok() != Some(LOGICAL_PUBLIC)
        {
            return Err("unexpected physical assignment envelope".into());
        }
        let private = usize::try_from(private).map_err(|_| "physical private count exceeds usize")?;
        let constant = usize::try_from(constant).map_err(|_| "physical constant column exceeds usize")?;
        if private != constant
            || private.checked_add(1 + PHYSICAL_PUBLIC) != Some(PHYSICAL_COLUMNS)
            || private_values.len() != private
            || public_values.len() != PHYSICAL_PUBLIC
        {
            return Err("raw physical assignment dimensions do not match the package".into());
        }
        let physical = Physical {
            private: private_values,
            public: public_values,
            constant_column: constant,
            total_columns: PHYSICAL_COLUMNS,
            unavailable_private: None,
        };
        for column in 0..PHYSICAL_COLUMNS {
            physical.value(column)?;
        }

        let transport = Transport::decode(&raw_transport)?;
        let quotients = derive_phi81_quotients(&transport, &physical)?;
        let output_digest = transport
            .output_digest_expressions
            .iter()
            .map(|expression| evaluate_expression(expression, &physical))
            .collect::<Result<Vec<_>>>()?;
        let output_digest: [u64; OUTPUT_DIGEST_WORDS] = output_digest
            .try_into()
            .map_err(|_| "output digest word count".to_string())?;
        let domains = Domains { physical, quotients };
        validate_derived_block_sources(&transport, &domains, output_digest)?;

        let mut values = Vec::with_capacity(LOGICAL_WIDTH);
        append_public(output_digest, &mut values);
        let mut block_ranges = Vec::with_capacity(BLOCK_COUNT);
        for block in &transport.blocks {
            let start = values.len();
            for source in block.sources() {
                encode_slot(block.kind, domains.retained(source?)?, &mut values)?;
            }
            block_ranges.push(start..values.len());
        }
        if values.len() != LOGICAL_WIDTH {
            return Err(format!(
                "logical assignment has width {}, expected {LOGICAL_WIDTH}",
                values.len()
            ));
        }
        Ok(Self { values, block_ranges })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn value(&self, column: usize) -> Result<Field> {
        let value = *self
            .values
            .get(column)
            .ok_or_else(|| format!("logical assignment column {column} is out of range"))?;
        match value {
            -1 => Field::checked(GOLDILOCKS_MODULUS - 1, "balanced logical coordinate"),
            0 => Ok(Field::ZERO),
            1 => Ok(Field::ONE),
            _ => Err("logical assignment coordinate is not balanced".into()),
        }
    }

    pub fn carrier_value(&self, column: usize) -> Result<Field> {
        if column < self.values.len() {
            self.value(column)
        } else if column < CARRIER_WIDTH {
            Ok(Field::ZERO)
        } else {
            Err(format!("carrier assignment column {column} is out of range"))
        }
    }

    pub fn balanced_values(&self) -> &[i8] {
        &self.values
    }

    pub fn nonempty_block_count(&self) -> usize {
        self.block_ranges
            .iter()
            .filter(|range| !range.is_empty())
            .count()
    }

    pub fn block_is_nonempty(&self, block: usize) -> bool {
        self.block_ranges
            .get(block)
            .is_some_and(|range| !range.is_empty())
    }

    pub fn block_for_column(&self, column: usize) -> Option<usize> {
        let position = self
            .block_ranges
            .partition_point(|range| range.end <= column);
        self.block_ranges
            .get(position)
            .filter(|range| range.contains(&column))
            .map(|_| position)
    }

    pub fn mutation_delta(&self, column: usize) -> Result<Field> {
        let current = *self
            .values
            .get(column)
            .ok_or_else(|| format!("logical mutation column {column} is out of range"))?;
        let replacement = if current == 0 { 1 } else { 0 };
        let current = self.value(column)?;
        let replacement = match replacement {
            0 => Field::ZERO,
            1 => Field::ONE,
            _ => unreachable!("balanced mutation replacement"),
        };
        Ok(replacement + -current)
    }
}

/// A fail-closed logical view over an available physical-assignment prefix.
///
/// Values are decoded from the raw schema-4 transport only when a logical
/// row requests them. No value is invented for an unavailable private suffix.
pub struct PartialLogicalAssignment<'a> {
    transport: Transport,
    physical: Physical<'a>,
    block_ranges: Vec<Range<usize>>,
    quotients: OnceLock<Result<Vec<u64>>>,
    output_digest: OnceLock<Result<[u64; OUTPUT_DIGEST_WORDS]>>,
}

impl<'a> PartialLogicalAssignment<'a> {
    pub fn decode(sealed_bytes: &[u8], private_prefix: &'a [u64], public_values: &'a [u64]) -> Result<Self> {
        if sealed_bytes.last() != Some(&b'\n') {
            return Err("sealed package is not newline terminated".into());
        }
        let RawSealed(
            outer_schema,
            RawCircuit(inner_schema, _, _, layout, _, _, _, _, _, _, _, _, _, _),
            _,
            _,
            raw_transport,
            _,
            logical_public,
        ) = serde_json::from_slice(sealed_bytes)
            .map_err(|error| format!("independent partial-assignment decode: {error}"))?;
        let RawLayout(rows, private, constant, public, total, _, _) = layout;
        if usize::try_from(outer_schema).ok() != Some(SEALED_SCHEMA)
            || usize::try_from(inner_schema).ok() != Some(INNER_SCHEMA)
            || usize::try_from(rows).ok() != Some(PHYSICAL_ROWS)
            || usize::try_from(total).ok() != Some(PHYSICAL_COLUMNS)
            || usize::try_from(public).ok() != Some(PHYSICAL_PUBLIC)
            || usize::try_from(logical_public).ok() != Some(LOGICAL_PUBLIC)
        {
            return Err("unexpected partial physical-assignment envelope".into());
        }
        let private = usize::try_from(private).map_err(|_| "physical private count exceeds usize")?;
        let constant = usize::try_from(constant).map_err(|_| "physical constant column exceeds usize")?;
        if private != constant
            || private.checked_add(1 + PHYSICAL_PUBLIC) != Some(PHYSICAL_COLUMNS)
            || private_prefix.len() > private
            || public_values.len() != PHYSICAL_PUBLIC
        {
            return Err("partial physical-assignment dimensions do not match the package".into());
        }
        if private_prefix
            .iter()
            .chain(public_values)
            .any(|value| *value >= GOLDILOCKS_MODULUS)
        {
            return Err("partial physical assignment contains a noncanonical value".into());
        }

        let transport = Transport::decode(&raw_transport)?;
        let mut cursor = LOGICAL_PUBLIC;
        let mut block_ranges = Vec::with_capacity(BLOCK_COUNT);
        for block in &transport.blocks {
            let width = block
                .slot_count
                .checked_mul(block.kind.width())
                .ok_or_else(|| "logical assignment block width overflow".to_string())?;
            let end = cursor
                .checked_add(width)
                .ok_or_else(|| "logical assignment block end overflow".to_string())?;
            block_ranges.push(cursor..end);
            cursor = end;
        }
        if cursor != LOGICAL_WIDTH {
            return Err(format!(
                "logical assignment has width {cursor}, expected {LOGICAL_WIDTH}"
            ));
        }

        Ok(Self {
            transport,
            physical: Physical {
                private: private_prefix,
                public: public_values,
                constant_column: constant,
                total_columns: PHYSICAL_COLUMNS,
                unavailable_private: None,
            },
            block_ranges,
            quotients: OnceLock::new(),
            output_digest: OnceLock::new(),
        })
    }

    pub fn len(&self) -> usize {
        LOGICAL_WIDTH
    }

    /// Keep the sealed pilot's proof-input gap and non-pilot public context
    /// unavailable. This uses the same independent schema-4 transport.
    pub fn decode_pilot(sealed_bytes: &[u8], private_prefix: &'a [u64], public_values: &'a [u64]) -> Result<Self> {
        if private_prefix.len() != 14_751_526 {
            return Err("pilot physical-assignment prefix has the wrong length".into());
        }
        let mut assignment = Self::decode(sealed_bytes, private_prefix, public_values)?;
        assignment.physical.unavailable_private = Some(98_786..128_074);
        assignment.physical.public = &public_values[..274];
        Ok(assignment)
    }

    pub fn value(&self, column: usize) -> Result<Field> {
        if column >= LOGICAL_WIDTH {
            return Err(format!("logical assignment column {column} is out of range"));
        }
        if column == 0 {
            return Ok(Field::ONE);
        }
        if column < 257 {
            let bit = column - 1;
            let digest = self.output_digest()?;
            return Field::checked((digest[bit / 64] >> (bit % 64)) & 1, "logical public digest bit");
        }
        if column < LOGICAL_PUBLIC {
            return Ok(Field::ZERO);
        }

        let position = self
            .block_ranges
            .partition_point(|range| range.end <= column);
        let range = self
            .block_ranges
            .get(position)
            .ok_or_else(|| format!("logical assignment column {column} has no block"))?;
        let block = self
            .transport
            .blocks
            .get(position)
            .ok_or_else(|| format!("logical assignment block {position} is missing"))?;
        if !range.contains(&column) {
            return Err(format!("logical assignment column {column} is outside its block"));
        }
        let coordinate = column - range.start;
        let width = block.kind.width();
        let slot = coordinate / width;
        let digit = coordinate % width;
        let source = block.source(slot)?;
        let value = self.retained(source)?;
        encode_slot_coordinate(block.kind, value, digit)
    }

    fn retained(&self, index: usize) -> Result<u64> {
        if index < PHYSICAL_COLUMNS {
            return self.physical.value(index);
        }
        let index = index - PHYSICAL_COLUMNS;
        if index >= PHI81_QUOTIENT_VALUES {
            return Err("retained assignment source is out of range".into());
        }
        let quotients = match self
            .quotients
            .get_or_init(|| derive_phi81_quotients(&self.transport, &self.physical))
        {
            Ok(values) => values,
            Err(error) => return Err(error.clone()),
        };
        quotients
            .get(index)
            .copied()
            .ok_or_else(|| "retained Phi81 quotient source is out of range".into())
    }

    fn output_digest(&self) -> Result<[u64; OUTPUT_DIGEST_WORDS]> {
        match self.output_digest.get_or_init(|| {
            self.transport
                .output_digest_expressions
                .iter()
                .map(|expression| evaluate_expression(expression, &self.physical))
                .collect::<Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| "output digest word count".to_string())
        }) {
            Ok(output) => Ok(*output),
            Err(error) => Err(error.clone()),
        }
    }
}

fn encode_slot_coordinate(kind: SlotKind, value: u64, coordinate: usize) -> Result<Field> {
    if value >= GOLDILOCKS_MODULUS || coordinate >= kind.width() {
        return Err("logical assignment coordinate source is invalid".into());
    }
    let balanced = match kind {
        SlotKind::Bit => match value {
            0 => 0,
            1 => 1,
            _ => return Err("bit assignment source is not zero or one".into()),
        },
        SlotKind::Centered => match value {
            0 => 0,
            1 => 1,
            value if value == GOLDILOCKS_MODULUS - 1 => -1,
            _ => return Err("centered assignment source is outside {-1,0,1}".into()),
        },
        SlotKind::Field => {
            let negative = value > CENTERED_HALF_MODULUS;
            let magnitude = if negative { GOLDILOCKS_MODULUS - value } else { value };
            let power = 3u128.pow(coordinate as u32);
            let rounded = (u128::from(magnitude) + (power - 1) / 2) / power;
            let digit = match rounded % 3 {
                0 => 0,
                1 => 1,
                2 => -1,
                _ => unreachable!("remainder modulo three"),
            };
            if negative {
                -digit
            } else {
                digit
            }
        }
    };
    match balanced {
        -1 => Field::checked(GOLDILOCKS_MODULUS - 1, "balanced logical coordinate"),
        0 => Ok(Field::ZERO),
        1 => Ok(Field::ONE),
        _ => Err("logical assignment coordinate is not balanced".into()),
    }
}

fn derive_challenges(
    plan: &Phi81Plan,
    physical: &Physical<'_>,
) -> Result<[[u64; CHALLENGE_COEFFICIENTS]; CHALLENGE_SOURCES]> {
    let mut challenges = [[0; CHALLENGE_COEFFICIENTS]; CHALLENGE_SOURCES];
    for (source, coefficients) in challenges.iter_mut().enumerate() {
        for (lane, coefficient) in coefficients.iter_mut().enumerate() {
            let mut digit = 0;
            for bit in 0..CHALLENGE_BITS {
                let slot = (source * CHALLENGE_COEFFICIENTS + lane) * CHALLENGE_BITS + bit;
                let value = physical.value(source_at(&plan.challenge_bit_sources, slot)?)?;
                if value > 1 {
                    return Err(format!("challenge bit {source}/{lane}/{bit} is not zero or one"));
                }
                digit += value << bit;
            }
            if digit > 4 {
                return Err(format!("challenge digit {source}/{lane} is outside 0..=4"));
            }
            *coefficient = sub_mod(digit, 2);
        }
    }
    Ok(challenges)
}

fn derive_phi81_quotients(transport: &Transport, physical: &Physical<'_>) -> Result<Vec<u64>> {
    let plan = &transport.phi81;
    let challenges = derive_challenges(plan, physical)?;
    let output = transport.block(PHI81_QUOTIENT_BLOCK)?;
    if output.kind != SlotKind::Field || output.slot_count != PHI81_QUOTIENT_VALUES {
        return Err("Phi81 quotient-output block has the wrong slot count".into());
    }
    let mut quotients = vec![0; PHI81_QUOTIENT_VALUES];
    for family in &plan.families {
        for source in 0..family.source_count {
            let left = &challenges[source];
            for block in 0..family.block_count {
                for cell in 0..family.cell_count {
                    let mut right = [0u64; 54];
                    for (degree, coefficient) in right.iter_mut().enumerate() {
                        let slot = family.invocation(source, block, degree, cell)?;
                        *coefficient = physical.value(source_at(&plan.value_sources, slot)?)?;
                    }
                    // Independent monic long division, not the consumer's closed coefficient formula.
                    let mut product = [0u64; 108];
                    for (i, &a) in left.iter().enumerate() {
                        for (j, &b) in right.iter().enumerate() {
                            product[i + j] = add_mod(product[i + j], mul_mod(a, b));
                        }
                    }
                    for degree in (54..108).rev() {
                        let coefficient = product[degree];
                        let lane = degree - 54;
                        quotients[family.invocation(source, block, lane, cell)?] = coefficient;
                        for shift in [0, 27, 54] {
                            product[lane + shift] = sub_mod(product[lane + shift], coefficient);
                        }
                    }
                }
            }
        }
    }
    Ok(quotients)
}

fn validate_derived_block_sources(
    transport: &Transport,
    domains: &Domains<'_>,
    output_digest: [u64; OUTPUT_DIGEST_WORDS],
) -> Result<()> {
    let quotient = transport.block(PHI81_QUOTIENT_BLOCK)?;
    for slot in 0..quotient.slot_count {
        if domains.retained(quotient.source(slot)?)? != domains.quotients[slot] {
            return Err("Phi81 quotient source map does not select the derived value".into());
        }
    }
    let digest = transport.block(OUTPUT_DIGEST_BLOCK)?;
    if digest.kind != SlotKind::Field || digest.slot_count != OUTPUT_DIGEST_WORDS {
        return Err("output-digest block has the wrong slot count".into());
    }
    for (slot, expected) in output_digest.into_iter().enumerate() {
        if domains.retained(digest.source(slot)?)? != expected {
            return Err("output-digest source map disagrees with its expression".into());
        }
    }
    Ok(())
}

fn evaluate_expression(value: &Value, physical: &Physical<'_>) -> Result<u64> {
    let fields = array(value, "assignment expression")?;
    match fields {
        [tag, argument] if word(tag, "assignment expression opcode")? == 0 => {
            physical.value(word(argument, "assignment expression column")?)
        }
        [tag, argument] if word(tag, "assignment expression opcode")? == 1 => {
            Ok(field(argument, "assignment expression constant")?.canonical())
        }
        [tag, left, right] if word(tag, "assignment expression opcode")? == 2 => Ok(add_mod(
            evaluate_expression(left, physical)?,
            evaluate_expression(right, physical)?,
        )),
        [tag, left, right] if word(tag, "assignment expression opcode")? == 3 => Ok(mul_mod(
            evaluate_expression(left, physical)?,
            evaluate_expression(right, physical)?,
        )),
        _ => Err("invalid assignment expression".into()),
    }
}

fn append_public(digest: [u64; OUTPUT_DIGEST_WORDS], output: &mut Vec<i8>) {
    output.push(1);
    for word in digest {
        for bit in 0..64 {
            output.push(((word >> bit) & 1) as i8);
        }
    }
    output.extend(std::iter::repeat_n(0, 13));
    debug_assert_eq!(output.len(), LOGICAL_PUBLIC);
}

fn encode_slot(kind: SlotKind, value: u64, output: &mut Vec<i8>) -> Result<()> {
    if value >= GOLDILOCKS_MODULUS {
        return Err("assignment source value is noncanonical".into());
    }
    let start = output.len();
    match kind {
        SlotKind::Bit => match value {
            0 | 1 => output.push(value as i8),
            _ => return Err("bit assignment source is not zero or one".into()),
        },
        SlotKind::Centered => match value {
            0 => output.push(0),
            1 => output.push(1),
            value if value == GOLDILOCKS_MODULUS - 1 => output.push(-1),
            _ => return Err("centered assignment source is outside {-1,0,1}".into()),
        },
        SlotKind::Field => encode_field(value, output),
    }
    if output.len() - start != kind.width() {
        return Err("assignment slot encoder produced the wrong width".into());
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
        magnitude = magnitude / 3 + usize::from(remainder == 2) as u64;
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
