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
const TRANSPORT_SCHEMA: usize = 1;
const BLOCK_COUNT: usize = 33;
const PHYSICAL_COLUMNS: usize = 29_344_425;
const PHYSICAL_PUBLIC: usize = 278;
const LOGICAL_PUBLIC: usize = 270;
const LOGICAL_WIDTH: usize = 254_260_583;
const CARRIER_WIDTH: usize = 254_260_620;
const FIELD_COORDINATES: usize = 41;
const OUTPUT_DIGEST_WORDS: usize = 4;
const PAYLOAD_VALUES: usize = 30_416;
const PHI81_INVOCATIONS: usize = 52_326;
const PHI81_GROUPS: usize = 33;
const PHI81_GROUP_VALUES: usize = PHI81_INVOCATIONS * PHI81_GROUPS;
const FIRST54_PRODUCTS: usize = 1_088;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceDomain {
    Retained,
    Payload,
    Physical,
}

impl SourceDomain {
    fn decode(value: &Value) -> Result<Self> {
        match word(value, "assignment source domain")? {
            0 => Ok(Self::Retained),
            1 => Ok(Self::Payload),
            2 => Ok(Self::Physical),
            _ => Err("unknown assignment source domain".into()),
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
    opcode: usize,
    kind: SlotKind,
    slot_count: usize,
    domain: SourceDomain,
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
    fn decode(value: &Value, expected_opcode: usize) -> Result<Self> {
        let fields = exact_array(value, 5, "assignment block plan")?;
        let opcode = word(&fields[0], "assignment block opcode")?;
        if opcode != expected_opcode {
            return Err(format!(
                "assignment block opcode {opcode} is out of order; expected {expected_opcode}"
            ));
        }
        let kind = SlotKind::decode(&fields[1])?;
        let slot_count = word(&fields[2], "assignment block slot count")?;
        let domain = SourceDomain::decode(&fields[3])?;
        let expected_domain = match opcode {
            12 => SourceDomain::Payload,
            31..=32 => SourceDomain::Physical,
            _ => SourceDomain::Retained,
        };
        if domain != expected_domain {
            return Err(format!("assignment block {opcode} uses the wrong source domain"));
        }
        let runs = decode_runs(&fields[4], slot_count)?;
        Ok(Self {
            opcode,
            kind,
            slot_count,
            domain,
            runs,
        })
    }

    fn source(&self, slot: usize) -> Result<usize> {
        if slot >= self.slot_count {
            return Err(format!("assignment block {} slot {slot} is out of range", self.opcode));
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
    challenge_opcode: usize,
    challenge_slot_base: usize,
    challenge_source_stride: usize,
    challenge_shift: u64,
    value_sources: Vec<Run>,
    group_opcode: usize,
}

impl Phi81Plan {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 15, "Phi81 assignment plan")?;
        let constants = fields[..8]
            .iter()
            .enumerate()
            .map(|(index, value)| word(value, &format!("Phi81 constant {index}")))
            .collect::<Result<Vec<_>>>()?;
        if constants != [54, 27, 81, 106, 3, 162, 5, 33] {
            return Err("unexpected Phi81 assignment constants".into());
        }
        let expected_shapes = [[17, 22, 1], [17, 5, 1], [17, 1, 2], [17, 14, 2]];
        let shape_values = exact_array(&fields[8], expected_shapes.len(), "Phi81 family shapes")?;
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
        let tail = fields[9..13]
            .iter()
            .chain([&fields[14]])
            .enumerate()
            .map(|(index, value)| word(value, &format!("Phi81 selector {index}")))
            .collect::<Result<Vec<_>>>()?;
        if tail != [7, 3402, 3456, 2, 3] {
            return Err("unexpected Phi81 assignment selectors".into());
        }
        let value_sources = decode_runs(&fields[13], PHI81_INVOCATIONS)?;
        if value_sources
            .iter()
            .any(|run| run.first + run.step * (run.count - 1) >= PHYSICAL_COLUMNS)
        {
            return Err("Phi81 operand source is outside the physical assignment".into());
        }
        Ok(Self {
            families,
            challenge_opcode: tail[0],
            challenge_slot_base: tail[1],
            challenge_source_stride: tail[2],
            challenge_shift: tail[3] as u64,
            value_sources,
            group_opcode: tail[4],
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct First54Plan {
    candidate_count: usize,
    reject_opcode: usize,
    symbol_opcode: usize,
    output_opcode: usize,
}

impl First54Plan {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 4, "first54 assignment plan")?;
        let values = fields
            .iter()
            .map(|value| word(value, "first54 assignment selector"))
            .collect::<Result<Vec<_>>>()?;
        if values != [FIRST54_PRODUCTS, 4, 5, 8] {
            return Err("unexpected first54 assignment plan".into());
        }
        Ok(Self {
            candidate_count: values[0],
            reject_opcode: values[1],
            symbol_opcode: values[2],
            output_opcode: values[3],
        })
    }
}

struct Transport {
    blocks: Vec<BlockPlan>,
    phi81: Phi81Plan,
    first54: First54Plan,
    payload_expressions: Vec<Value>,
    output_digest_expressions: Vec<Value>,
}

impl Transport {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 8, "assignment transport plan")?;
        if word(&fields[0], "assignment transport schema")? != TRANSPORT_SCHEMA {
            return Err("unexpected assignment transport schema".into());
        }
        let block_values = exact_array(&fields[1], BLOCK_COUNT, "assignment block plans")?;
        let blocks = block_values
            .iter()
            .enumerate()
            .map(|(opcode, value)| BlockPlan::decode(value, opcode))
            .collect::<Result<Vec<_>>>()?;
        let phi81 = Phi81Plan::decode(&fields[2])?;
        let first54 = First54Plan::decode(&fields[3])?;
        if word(&fields[4], "payload block opcode")? != 12 || word(&fields[6], "output-digest block opcode")? != 26 {
            return Err("unexpected derived assignment block selector".into());
        }
        let payload_expressions = exact_array(&fields[5], PAYLOAD_VALUES, "payload expressions")?.to_vec();
        let output_digest_expressions =
            exact_array(&fields[7], OUTPUT_DIGEST_WORDS, "output-digest expressions")?.to_vec();
        Ok(Self {
            blocks,
            phi81,
            first54,
            payload_expressions,
            output_digest_expressions,
        })
    }

    fn block(&self, opcode: usize) -> Result<&BlockPlan> {
        self.blocks
            .get(opcode)
            .filter(|block| block.opcode == opcode)
            .ok_or_else(|| format!("missing assignment block opcode {opcode}"))
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
    groups: Vec<u64>,
    products: Vec<u64>,
    payload: Vec<u64>,
}

impl Domains<'_> {
    fn retained(&self, index: usize) -> Result<u64> {
        if index < PHYSICAL_COLUMNS {
            return self.physical.value(index);
        }
        let index = index - PHYSICAL_COLUMNS;
        if index < self.groups.len() {
            return Ok(self.groups[index]);
        }
        let index = index - self.groups.len();
        self.products
            .get(index)
            .copied()
            .ok_or_else(|| "retained assignment source is out of range".into())
    }

    fn value(&self, domain: SourceDomain, index: usize) -> Result<u64> {
        match domain {
            SourceDomain::Retained => self.retained(index),
            SourceDomain::Payload => self
                .payload
                .get(index)
                .copied()
                .ok_or_else(|| "payload assignment source is out of range".into()),
            SourceDomain::Physical => self.physical.value(index),
        }
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
            || usize::try_from(rows).ok() != Some(29_225_729)
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
        let groups = derive_phi81_groups(&transport, &physical)?;
        let products = derive_first54_products(&transport, &physical)?;
        let payload = transport
            .payload_expressions
            .iter()
            .map(|expression| evaluate_expression(expression, &physical))
            .collect::<Result<Vec<_>>>()?;
        let output_digest = transport
            .output_digest_expressions
            .iter()
            .map(|expression| evaluate_expression(expression, &physical))
            .collect::<Result<Vec<_>>>()?;
        let output_digest: [u64; OUTPUT_DIGEST_WORDS] = output_digest
            .try_into()
            .map_err(|_| "output digest word count".to_string())?;
        let domains = Domains {
            physical,
            groups,
            products,
            payload,
        };
        validate_derived_block_sources(&transport, &domains, output_digest)?;

        let mut values = Vec::with_capacity(LOGICAL_WIDTH);
        append_public(output_digest, &mut values);
        let mut block_ranges = Vec::with_capacity(BLOCK_COUNT);
        for block in &transport.blocks {
            let start = values.len();
            for source in block.sources() {
                encode_slot(block.kind, domains.value(block.domain, source?)?, &mut values)?;
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
/// Values are decoded from the raw schema-6 transport only when a logical
/// row requests them. No value is invented for an unavailable private suffix.
pub struct PartialLogicalAssignment<'a> {
    transport: Transport,
    physical: Physical<'a>,
    block_ranges: Vec<Range<usize>>,
    groups: OnceLock<Result<Vec<u64>>>,
    products: OnceLock<Result<Vec<u64>>>,
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
            || usize::try_from(rows).ok() != Some(29_225_729)
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
            groups: OnceLock::new(),
            products: OnceLock::new(),
            output_digest: OnceLock::new(),
        })
    }

    pub fn len(&self) -> usize {
        LOGICAL_WIDTH
    }

    /// Keep the sealed pilot's proof-input gap and non-pilot public context
    /// unavailable. This uses the same independent schema-6 transport.
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
        let value = match block.domain {
            SourceDomain::Retained => self.retained(source)?,
            SourceDomain::Payload => self.payload(source)?,
            SourceDomain::Physical => self.physical.value(source)?,
        };
        encode_slot_coordinate(block.kind, value, digit)
    }

    fn retained(&self, index: usize) -> Result<u64> {
        if index < PHYSICAL_COLUMNS {
            return self.physical.value(index);
        }
        let index = index - PHYSICAL_COLUMNS;
        if index < PHI81_GROUP_VALUES {
            let groups = match self
                .groups
                .get_or_init(|| derive_phi81_groups(&self.transport, &self.physical))
            {
                Ok(values) => values,
                Err(error) => return Err(error.clone()),
            };
            return groups
                .get(index)
                .copied()
                .ok_or_else(|| "retained Phi81 group source is out of range".into());
        }
        let index = index - PHI81_GROUP_VALUES;
        let products = match self
            .products
            .get_or_init(|| derive_first54_products(&self.transport, &self.physical))
        {
            Ok(values) => values,
            Err(error) => return Err(error.clone()),
        };
        products
            .get(index)
            .copied()
            .ok_or_else(|| "retained first54 product source is out of range".into())
    }

    fn payload(&self, index: usize) -> Result<u64> {
        let expression = self
            .transport
            .payload_expressions
            .get(index)
            .ok_or_else(|| "payload assignment source is out of range".to_string())?;
        evaluate_expression(expression, &self.physical)
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

fn raw_block_value(block: &BlockPlan, slot: usize, physical: &Physical<'_>) -> Result<u64> {
    if block.domain != SourceDomain::Retained {
        return Err(format!(
            "derived assignment block {} does not use retained sources",
            block.opcode
        ));
    }
    let source = block.source(slot)?;
    if source >= PHYSICAL_COLUMNS {
        return Err(format!(
            "derived assignment block {} source is not in the physical base",
            block.opcode
        ));
    }
    physical.value(source)
}

fn derive_phi81_groups(transport: &Transport, physical: &Physical<'_>) -> Result<Vec<u64>> {
    let plan = &transport.phi81;
    let challenge = transport.block(plan.challenge_opcode)?;
    let output = transport.block(plan.group_opcode)?;
    if output.slot_count != PHI81_GROUP_VALUES {
        return Err("Phi81 group-output block has the wrong slot count".into());
    }
    let mut groups = Vec::with_capacity(PHI81_GROUP_VALUES);
    for family in &plan.families {
        for source in 0..family.source_count {
            for block in 0..family.block_count {
                for lane in 0..54 {
                    for cell in 0..family.cell_count {
                        let invocation = family.invocation(source, block, lane, cell)?;
                        if invocation != groups.len() / PHI81_GROUPS {
                            return Err("Phi81 invocation order mismatch".into());
                        }
                        for group in 0..PHI81_GROUPS {
                            let mut sum = 0u64;
                            for raw_term in group * 5..((group + 1) * 5).min(162) {
                                let section = raw_term / 54;
                                let convolution_source = raw_term % 54;
                                let (degree, sign) = match section {
                                    0 => (lane, 1i8),
                                    1 => (lane + if lane < 27 { 54 } else { 27 }, -1i8),
                                    2 if lane + 81 <= 106 => (lane + 81, 1i8),
                                    2 => continue,
                                    _ => return Err("Phi81 raw term is out of range".into()),
                                };
                                if convolution_source > degree || degree - convolution_source >= 54 {
                                    continue;
                                }
                                let challenge_slot = plan
                                    .challenge_slot_base
                                    .checked_add(source * plan.challenge_source_stride)
                                    .and_then(|slot| slot.checked_add(convolution_source))
                                    .ok_or_else(|| "Phi81 challenge slot overflow".to_string())?;
                                let challenge_value = raw_block_value(challenge, challenge_slot, physical)?;
                                let shifted_challenge = sub_mod(challenge_value, plan.challenge_shift);
                                let value_lane = degree - convolution_source;
                                let value_slot = family.invocation(source, block, value_lane, cell)?;
                                let product = mul_mod(
                                    shifted_challenge,
                                    physical.value(source_at(&plan.value_sources, value_slot)?)?,
                                );
                                sum = add_mod(sum, if sign < 0 { neg_mod(product) } else { product });
                            }
                            groups.push(sum);
                        }
                    }
                }
            }
        }
    }
    if groups.len() != PHI81_GROUP_VALUES {
        return Err("Phi81 derived group count mismatch".into());
    }
    Ok(groups)
}

fn derive_first54_products(transport: &Transport, physical: &Physical<'_>) -> Result<Vec<u64>> {
    let plan = transport.first54;
    let reject = transport.block(plan.reject_opcode)?;
    let symbol = transport.block(plan.symbol_opcode)?;
    let output = transport.block(plan.output_opcode)?;
    if output.slot_count != plan.candidate_count {
        return Err("first54 output block has the wrong slot count".into());
    }
    (0..plan.candidate_count)
        .map(|candidate| {
            let reject = raw_block_value(reject, candidate, physical)?;
            let symbol = raw_block_value(symbol, candidate, physical)?;
            Ok(mul_mod(sub_mod(1, reject), symbol))
        })
        .collect()
}

fn validate_derived_block_sources(
    transport: &Transport,
    domains: &Domains<'_>,
    output_digest: [u64; OUTPUT_DIGEST_WORDS],
) -> Result<()> {
    let group = transport.block(transport.phi81.group_opcode)?;
    for slot in 0..group.slot_count {
        if domains.value(group.domain, group.source(slot)?)? != domains.groups[slot] {
            return Err("Phi81 group source map does not select the derived value".into());
        }
    }
    let product = transport.block(transport.first54.output_opcode)?;
    for slot in 0..product.slot_count {
        if domains.value(product.domain, product.source(slot)?)? != domains.products[slot] {
            return Err("first54 product source map does not select the derived value".into());
        }
    }
    let payload = transport.block(12)?;
    if payload.slot_count != PAYLOAD_VALUES {
        return Err("payload block has the wrong slot count".into());
    }
    for slot in 0..payload.slot_count {
        if domains.value(payload.domain, payload.source(slot)?)? != domains.payload[slot] {
            return Err("payload source map does not select the derived value".into());
        }
    }
    let digest = transport.block(26)?;
    if digest.slot_count != OUTPUT_DIGEST_WORDS {
        return Err("output-digest block has the wrong slot count".into());
    }
    for (slot, expected) in output_digest.into_iter().enumerate() {
        if domains.value(digest.domain, digest.source(slot)?)? != expected {
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
