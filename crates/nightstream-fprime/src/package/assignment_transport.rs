//! Strict decoding and execution of the Lean-owned final-assignment transport.
//!
//! The decoded plan retains every compact recipe, expression, block, and
//! affine source run. Execution reads a Rust-produced physical assignment and
//! constructs the exact balanced logical assignment without a Lean runtime.

use serde_json::Value;

use crate::WitnessAssignment;

use super::{Layout, PackageError, GOLDILOCKS_MODULUS};

const TRANSPORT_SCHEMA: usize = 1;
pub(super) const BLOCK_COUNT: usize = 45;
const FIELD_COORDINATES: usize = 41;
const PAYLOAD_VALUE_COUNT: usize = 30_416;
const OUTPUT_DIGEST_WORDS: usize = 4;
const PHI81_INVOCATIONS: usize = 52_326;
const PHI81_GROUPS: usize = 33;
const PHI81_GROUP_VALUES: usize = PHI81_INVOCATIONS * PHI81_GROUPS;
const FIRST54_PRODUCTS: usize = 1_088;
const CENTERED_HALF_MODULUS: u64 = (GOLDILOCKS_MODULUS - 1) / 2;

const PRODUCT_GROUP_BLOCK: usize = 3;
const FIRST54_REJECT_BLOCK: usize = 4;
const FIRST54_SYMBOL_BLOCK: usize = 5;
const FIRST54_VALUE_BLOCK: usize = 7;
const FIRST54_PRODUCT_BLOCK: usize = 8;
const PRODUCT_INPUT_BLOCK: usize = 9;
const PAYLOAD_BLOCK: usize = 13;
const OUTPUT_DIGEST_BLOCK: usize = 31;

/// Lean-authored block order for the final logical assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedAssignmentPlan {
    blocks: Vec<BlockPlan>,
    phi81: Phi81Recipe,
    first54: First54Recipe,
    payload_block: usize,
    payload_expressions: Vec<Expression>,
    output_digest_block: usize,
    output_digest_expressions: Vec<Expression>,
    physical_width: usize,
    logical_public_width: usize,
    logical_width: usize,
}

impl LoadedAssignmentPlan {
    pub fn kind_codes(&self) -> [u8; BLOCK_COUNT] {
        std::array::from_fn(|opcode| {
            debug_assert_eq!(self.blocks[opcode].opcode, opcode);
            opcode as u8
        })
    }

    pub(super) fn execute(
        &self,
        layout: &Layout,
        assignment: &WitnessAssignment,
    ) -> Result<LogicalAssignment, PackageError> {
        let physical = PhysicalAssignment::new(layout, assignment, self.physical_width)?;
        for column in 0..self.physical_width {
            physical.value(column)?;
        }

        let groups = derive_phi81_groups(self, &physical)?;
        let products = derive_first54_products(self, &physical)?;
        let payload = self
            .payload_expressions
            .iter()
            .map(|expression| expression.evaluate(&physical))
            .collect::<Result<Vec<_>, _>>()?;
        let output_digest: [u64; OUTPUT_DIGEST_WORDS] = self
            .output_digest_expressions
            .iter()
            .map(|expression| expression.evaluate(&physical))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| PackageError::Invalid("output digest word count"))?;
        let domains = Domains {
            physical,
            groups,
            products,
            payload,
        };
        validate_derived_block_sources(self, &domains, output_digest)?;

        let mut values = Vec::with_capacity(self.logical_width);
        append_public(output_digest, self.logical_public_width, &mut values)?;
        for block in &self.blocks {
            for source in block.sources() {
                encode_slot(block.kind, domains.value(block.domain, source?)?, &mut values)?;
            }
        }
        if values.len() != self.logical_width {
            return Err(PackageError::Invalid("logical assignment width"));
        }
        Ok(LogicalAssignment { values })
    }

    fn block(&self, opcode: usize) -> Result<&BlockPlan, PackageError> {
        block(&self.blocks, opcode)
    }
}

/// Exact balanced logical assignment produced by the schema-6 transport.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceDomain {
    Retained,
    Payload,
    Physical,
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

    fn direct_column(&self) -> Option<usize> {
        match self {
            Self::Column(column) => Some(*column),
            Self::Constant(_) | Self::Add(_, _) | Self::Multiply(_, _) => None,
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

impl SourceDomain {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        match word(value, "assignment source domain")? {
            0 => Ok(Self::Retained),
            1 => Ok(Self::Payload),
            2 => Ok(Self::Physical),
            _ => Err(PackageError::Invalid("assignment source domain")),
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct BlockPlan {
    opcode: usize,
    kind: SlotKind,
    slot_count: usize,
    domain: SourceDomain,
    runs: Vec<Run>,
}

impl BlockPlan {
    fn decode(
        value: &Value,
        expected_opcode: usize,
        physical_width: usize,
        retained_width: usize,
    ) -> Result<Self, PackageError> {
        let fields = exact_array(value, 5, "assignment block plan")?;
        let opcode = word(&fields[0], "assignment block opcode")?;
        if opcode != expected_opcode {
            return Err(PackageError::Invalid("assignment block order"));
        }
        let kind = SlotKind::decode(&fields[1])?;
        let slot_count = word(&fields[2], "assignment block slot count")?;
        let domain = SourceDomain::decode(&fields[3])?;
        let expected_domain = match opcode {
            PAYLOAD_BLOCK => SourceDomain::Payload,
            41..=44 => SourceDomain::Physical,
            _ => SourceDomain::Retained,
        };
        if domain != expected_domain {
            return Err(PackageError::Invalid("assignment block source domain"));
        }
        let domain_width = match domain {
            SourceDomain::Retained => retained_width,
            SourceDomain::Payload => PAYLOAD_VALUE_COUNT,
            SourceDomain::Physical => physical_width,
        };

        let run_values = fields[4]
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
        if covered != slot_count {
            return Err(PackageError::Invalid("assignment source run coverage"));
        }
        validate_canonical_run_boundaries(&runs)?;

        Ok(Self {
            opcode,
            kind,
            slot_count,
            domain,
            runs,
        })
    }

    fn source(&self, slot: usize) -> Result<usize, PackageError> {
        if slot >= self.slot_count {
            return Err(PackageError::Invalid("assignment block source slot"));
        }
        let mut run_start = 0usize;
        for run in &self.runs {
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

    fn sources(&self) -> impl Iterator<Item = Result<usize, PackageError>> + '_ {
        self.runs.iter().flat_map(|run| {
            (0..run.count).map(move |offset| {
                run.first
                    .checked_add(
                        run.step
                            .checked_mul(offset)
                            .ok_or(PackageError::Invalid("assignment source run overflow"))?,
                    )
                    .ok_or(PackageError::Invalid("assignment source run overflow"))
            })
        })
    }

    fn require_field_count(&self, count: usize) -> Result<(), PackageError> {
        self.require_kind_count(SlotKind::Field, count)
    }

    fn require_kind_count(&self, kind: SlotKind, count: usize) -> Result<(), PackageError> {
        if self.kind != kind || self.slot_count != count {
            return Err(PackageError::Invalid("derived assignment block shape"));
        }
        Ok(())
    }

    fn require_physical_sources(&self, physical_width: usize) -> Result<(), PackageError> {
        if self.domain != SourceDomain::Retained || self.runs.iter().any(|run| run.last >= physical_width) {
            return Err(PackageError::Invalid("derived assignment physical source"));
        }
        Ok(())
    }

    fn require_exact_range(&self, first: usize, count: usize) -> Result<(), PackageError> {
        if self.slot_count != count {
            return Err(PackageError::Invalid("derived assignment source count"));
        }
        if count == 0 {
            if self.runs.is_empty() {
                return Ok(());
            }
            return Err(PackageError::Invalid("derived assignment source range"));
        }
        if self.runs.len() != 1 {
            return Err(PackageError::Invalid("derived assignment source range"));
        }
        let run = self.runs[0];
        let expected_step = usize::from(count > 1);
        if run.first != first || run.step != expected_step || run.count != count {
            return Err(PackageError::Invalid("derived assignment source range"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Phi81FamilyShape {
    source_count: usize,
    block_count: usize,
    cell_count: usize,
    first_invocation: usize,
}

impl Phi81FamilyShape {
    fn invocation_count(self, ring_degree: usize) -> Result<usize, PackageError> {
        self.source_count
            .checked_mul(self.block_count)
            .and_then(|count| count.checked_mul(ring_degree))
            .and_then(|count| count.checked_mul(self.cell_count))
            .ok_or(PackageError::Invalid("Phi81 invocation count overflow"))
    }

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct Phi81Recipe {
    ring_degree: usize,
    middle_degree: usize,
    fold_offset: usize,
    twice_cutoff: usize,
    raw_convolution_count: usize,
    raw_term_count: usize,
    group_width: usize,
    group_count: usize,
    family_shapes: Vec<Phi81FamilyShape>,
    challenge_block: usize,
    challenge_slot_base: usize,
    challenge_source_stride: usize,
    challenge_shift: u64,
    value_block: usize,
    group_output_block: usize,
}

impl Phi81Recipe {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 15, "Phi81 assignment recipe")?;
        let expected_constants = [54, 27, 81, 106, 3, 162, 5, PHI81_GROUPS];
        let constants = fields[..8]
            .iter()
            .map(|value| word(value, "Phi81 assignment constant"))
            .collect::<Result<Vec<_>, _>>()?;
        if constants != expected_constants {
            return Err(PackageError::Invalid("Phi81 assignment constants"));
        }

        let expected_shapes = [[17, 22, 1], [17, 5, 1], [17, 1, 2], [17, 14, 2]];
        let shapes = exact_array(&fields[8], expected_shapes.len(), "Phi81 family shapes")?;
        let mut first_invocation = 0usize;
        let mut family_shapes = Vec::with_capacity(expected_shapes.len());
        for (value, expected) in shapes.iter().zip(expected_shapes) {
            let shape = exact_array(value, 3, "Phi81 family shape")?;
            let decoded = [
                word(&shape[0], "Phi81 family source count")?,
                word(&shape[1], "Phi81 family block count")?,
                word(&shape[2], "Phi81 family cell count")?,
            ];
            if decoded != expected {
                return Err(PackageError::Invalid("Phi81 family shape"));
            }
            let family = Phi81FamilyShape {
                source_count: decoded[0],
                block_count: decoded[1],
                cell_count: decoded[2],
                first_invocation,
            };
            first_invocation = first_invocation
                .checked_add(family.invocation_count(constants[0])?)
                .ok_or(PackageError::Invalid("Phi81 invocation count overflow"))?;
            family_shapes.push(family);
        }
        if first_invocation != PHI81_INVOCATIONS {
            return Err(PackageError::Invalid("Phi81 invocation count"));
        }

        let selectors = [
            word(&fields[9], "Phi81 challenge block")?,
            word(&fields[10], "Phi81 challenge slot base")?,
            word(&fields[11], "Phi81 challenge source stride")?,
            word(&fields[12], "Phi81 challenge shift")?,
            word(&fields[13], "Phi81 value block")?,
            word(&fields[14], "Phi81 group output block")?,
        ];
        if selectors
            != [
                FIRST54_VALUE_BLOCK,
                3_402,
                3_456,
                2,
                PRODUCT_INPUT_BLOCK,
                PRODUCT_GROUP_BLOCK,
            ]
        {
            return Err(PackageError::Invalid("Phi81 assignment selectors"));
        }
        Ok(Self {
            ring_degree: constants[0],
            middle_degree: constants[1],
            fold_offset: constants[2],
            twice_cutoff: constants[3],
            raw_convolution_count: constants[4],
            raw_term_count: constants[5],
            group_width: constants[6],
            group_count: constants[7],
            family_shapes,
            challenge_block: selectors[0],
            challenge_slot_base: selectors[1],
            challenge_source_stride: selectors[2],
            challenge_shift: selectors[3] as u64,
            value_block: selectors[4],
            group_output_block: selectors[5],
        })
    }

    fn invocation_count(&self) -> Result<usize, PackageError> {
        self.family_shapes.iter().try_fold(0usize, |count, family| {
            count
                .checked_add(family.invocation_count(self.ring_degree)?)
                .ok_or(PackageError::Invalid("Phi81 invocation count overflow"))
        })
    }

    fn validate(&self, blocks: &[BlockPlan], physical_width: usize) -> Result<(), PackageError> {
        let challenge = block(blocks, self.challenge_block)?;
        let value = block(blocks, self.value_block)?;
        let output = block(blocks, self.group_output_block)?;

        let invocation_count = self.invocation_count()?;
        let group_value_count = invocation_count
            .checked_mul(self.group_count)
            .ok_or(PackageError::Invalid("Phi81 group count overflow"))?;

        challenge.require_physical_sources(physical_width)?;
        challenge.require_field_count(58_752)?;
        value.require_physical_sources(physical_width)?;
        value.require_field_count(invocation_count)?;
        output.require_field_count(group_value_count)?;
        output.require_exact_range(physical_width, group_value_count)?;

        let final_source = self
            .family_shapes
            .iter()
            .map(|family| family.source_count)
            .max()
            .and_then(|count| count.checked_sub(1))
            .ok_or(PackageError::Invalid("Phi81 family source count"))?;
        let final_lane = self
            .ring_degree
            .checked_sub(1)
            .ok_or(PackageError::Invalid("Phi81 ring degree"))?;
        let final_challenge_slot = self
            .challenge_slot_base
            .checked_add(
                final_source
                    .checked_mul(self.challenge_source_stride)
                    .ok_or(PackageError::Invalid("Phi81 challenge slot overflow"))?,
            )
            .and_then(|slot| slot.checked_add(final_lane))
            .ok_or(PackageError::Invalid("Phi81 challenge slot overflow"))?;
        challenge.source(final_challenge_slot)?;
        value.source(
            invocation_count
                .checked_sub(1)
                .ok_or(PackageError::Invalid("Phi81 invocation count"))?,
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct First54Recipe {
    candidate_count: usize,
    reject_block: usize,
    symbol_block: usize,
    output_block: usize,
}

impl First54Recipe {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 4, "First54 assignment recipe")?;
        let values = [
            word(&fields[0], "First54 candidate count")?,
            word(&fields[1], "First54 reject block")?,
            word(&fields[2], "First54 symbol block")?,
            word(&fields[3], "First54 output block")?,
        ];
        if values
            != [
                FIRST54_PRODUCTS,
                FIRST54_REJECT_BLOCK,
                FIRST54_SYMBOL_BLOCK,
                FIRST54_PRODUCT_BLOCK,
            ]
        {
            return Err(PackageError::Invalid("First54 assignment recipe"));
        }
        Ok(Self {
            candidate_count: values[0],
            reject_block: values[1],
            symbol_block: values[2],
            output_block: values[3],
        })
    }

    fn validate(
        &self,
        blocks: &[BlockPlan],
        physical_width: usize,
        product_source_start: usize,
    ) -> Result<(), PackageError> {
        let reject = block(blocks, self.reject_block)?;
        let symbol = block(blocks, self.symbol_block)?;
        let output = block(blocks, self.output_block)?;
        reject.require_kind_count(SlotKind::Bit, self.candidate_count)?;
        symbol.require_field_count(self.candidate_count)?;
        reject.require_physical_sources(physical_width)?;
        symbol.require_physical_sources(physical_width)?;
        output.require_field_count(self.candidate_count)?;
        output.require_exact_range(product_source_start, self.candidate_count)?;
        Ok(())
    }
}

/// Decode and validate the exact eight-field assignment transport.
pub(super) fn decode(
    value: &Value,
    physical_width: usize,
    logical_public_width: usize,
    logical_width: usize,
) -> Result<LoadedAssignmentPlan, PackageError> {
    let fields = exact_array(value, 8, "assignment transport plan")?;
    if word(&fields[0], "assignment transport schema")? != TRANSPORT_SCHEMA {
        return Err(PackageError::Invalid("assignment transport schema version"));
    }

    let phi81 = Phi81Recipe::decode(&fields[2])?;
    let first54 = First54Recipe::decode(&fields[3])?;
    let group_value_count = phi81
        .invocation_count()?
        .checked_mul(phi81.group_count)
        .ok_or(PackageError::Invalid("Phi81 group count overflow"))?;
    if group_value_count != PHI81_GROUP_VALUES {
        return Err(PackageError::Invalid("Phi81 group value count"));
    }
    let retained_width = physical_width
        .checked_add(group_value_count)
        .and_then(|width| width.checked_add(first54.candidate_count))
        .ok_or(PackageError::Invalid("assignment retained source width overflow"))?;
    let raw_blocks = exact_array(&fields[1], BLOCK_COUNT, "assignment block plans")?;
    let blocks = raw_blocks
        .iter()
        .enumerate()
        .map(|(opcode, value)| BlockPlan::decode(value, opcode, physical_width, retained_width))
        .collect::<Result<Vec<_>, _>>()?;

    let mut encoded_width = logical_public_width;
    for block in &blocks {
        encoded_width = encoded_width
            .checked_add(
                block
                    .slot_count
                    .checked_mul(block.kind.coordinate_width())
                    .ok_or(PackageError::Invalid("assignment coordinate width overflow"))?,
            )
            .ok_or(PackageError::Invalid("assignment coordinate width overflow"))?;
    }
    if encoded_width != logical_width {
        return Err(PackageError::Invalid("assignment coordinate width"));
    }

    phi81.validate(&blocks, physical_width)?;
    let product_source_start = physical_width
        .checked_add(group_value_count)
        .ok_or(PackageError::Invalid("First54 product source start overflow"))?;
    first54.validate(&blocks, physical_width, product_source_start)?;

    let payload_block = word(&fields[4], "payload block selector")?;
    if payload_block != PAYLOAD_BLOCK {
        return Err(PackageError::Invalid("payload block selector"));
    }
    let payload = block(&blocks, payload_block)?;
    payload.require_field_count(PAYLOAD_VALUE_COUNT)?;
    payload.require_exact_range(0, PAYLOAD_VALUE_COUNT)?;
    let payload_expressions = decode_expressions(&fields[5], PAYLOAD_VALUE_COUNT, physical_width)?;

    let output_digest_block = word(&fields[6], "output digest block selector")?;
    if output_digest_block != OUTPUT_DIGEST_BLOCK {
        return Err(PackageError::Invalid("output digest block selector"));
    }
    let digest = block(&blocks, output_digest_block)?;
    digest.require_field_count(OUTPUT_DIGEST_WORDS)?;
    let output_digest_expressions = decode_expressions(&fields[7], OUTPUT_DIGEST_WORDS, physical_width)?;
    for (slot, expression) in output_digest_expressions.iter().enumerate() {
        let source = digest.source(slot)?;
        if expression.direct_column() != Some(source) {
            return Err(PackageError::Invalid("output digest expression source"));
        }
    }

    Ok(LoadedAssignmentPlan {
        blocks,
        phi81,
        first54,
        payload_block,
        payload_expressions,
        output_digest_block,
        output_digest_expressions,
        physical_width,
        logical_public_width,
        logical_width,
    })
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

struct Domains<'a> {
    physical: PhysicalAssignment<'a>,
    groups: Vec<u64>,
    products: Vec<u64>,
    payload: Vec<u64>,
}

impl Domains<'_> {
    fn retained(&self, index: usize) -> Result<u64, PackageError> {
        if index < self.physical.total_columns {
            return self.physical.value(index);
        }
        let index = index - self.physical.total_columns;
        if let Some(value) = self.groups.get(index) {
            return Ok(*value);
        }
        let index = index - self.groups.len();
        self.products
            .get(index)
            .copied()
            .ok_or(PackageError::Invalid("retained assignment source"))
    }

    fn value(&self, domain: SourceDomain, index: usize) -> Result<u64, PackageError> {
        match domain {
            SourceDomain::Retained => self.retained(index),
            SourceDomain::Payload => self
                .payload
                .get(index)
                .copied()
                .ok_or(PackageError::Invalid("payload assignment source")),
            SourceDomain::Physical => self.physical.value(index),
        }
    }
}

fn raw_block_value(block: &BlockPlan, slot: usize, physical: &PhysicalAssignment<'_>) -> Result<u64, PackageError> {
    if block.domain != SourceDomain::Retained {
        return Err(PackageError::Invalid("derived assignment block domain"));
    }
    let source = block.source(slot)?;
    if source >= physical.total_columns {
        return Err(PackageError::Invalid("derived assignment physical source"));
    }
    physical.value(source)
}

fn derive_phi81_groups(
    transport: &LoadedAssignmentPlan,
    physical: &PhysicalAssignment<'_>,
) -> Result<Vec<u64>, PackageError> {
    let recipe = &transport.phi81;
    let challenge = transport.block(recipe.challenge_block)?;
    let value = transport.block(recipe.value_block)?;
    let output = transport.block(recipe.group_output_block)?;
    let invocation_count = recipe.invocation_count()?;
    let expected_count = invocation_count
        .checked_mul(recipe.group_count)
        .ok_or(PackageError::Invalid("Phi81 group count overflow"))?;
    if recipe.ring_degree == 0 || recipe.group_count == 0 || output.slot_count != expected_count {
        return Err(PackageError::Invalid("Phi81 group output count"));
    }

    let mut groups = Vec::with_capacity(expected_count);
    for family in &recipe.family_shapes {
        for source in 0..family.source_count {
            for block_index in 0..family.block_count {
                for lane in 0..recipe.ring_degree {
                    for cell in 0..family.cell_count {
                        let invocation = family.invocation(recipe.ring_degree, source, block_index, lane, cell)?;
                        if invocation != groups.len() / recipe.group_count {
                            return Err(PackageError::Invalid("Phi81 invocation order"));
                        }
                        for group in 0..recipe.group_count {
                            let raw_start = group
                                .checked_mul(recipe.group_width)
                                .ok_or(PackageError::Invalid("Phi81 raw term range overflow"))?;
                            let raw_end = group
                                .checked_add(1)
                                .and_then(|group| group.checked_mul(recipe.group_width))
                                .map(|end| end.min(recipe.raw_term_count))
                                .ok_or(PackageError::Invalid("Phi81 raw term range overflow"))?;
                            let mut sum = 0u64;
                            for raw_term in raw_start..raw_end {
                                let section = raw_term / recipe.ring_degree;
                                if section >= recipe.raw_convolution_count {
                                    return Err(PackageError::Invalid("Phi81 raw convolution"));
                                }
                                let convolution_source = raw_term % recipe.ring_degree;
                                let (degree, negative) = match section {
                                    0 => (lane, false),
                                    1 => (
                                        lane.checked_add(if lane < recipe.middle_degree {
                                            recipe.ring_degree
                                        } else {
                                            recipe.middle_degree
                                        })
                                        .ok_or(PackageError::Invalid("Phi81 folded degree overflow"))?,
                                        true,
                                    ),
                                    2 => {
                                        let degree = lane
                                            .checked_add(recipe.fold_offset)
                                            .ok_or(PackageError::Invalid("Phi81 folded degree overflow"))?;
                                        if degree > recipe.twice_cutoff {
                                            continue;
                                        }
                                        (degree, false)
                                    }
                                    _ => return Err(PackageError::Invalid("Phi81 raw convolution")),
                                };
                                if convolution_source > degree || degree - convolution_source >= recipe.ring_degree {
                                    continue;
                                }
                                let challenge_slot = source
                                    .checked_mul(recipe.challenge_source_stride)
                                    .and_then(|offset| recipe.challenge_slot_base.checked_add(offset))
                                    .and_then(|slot| slot.checked_add(convolution_source))
                                    .ok_or(PackageError::Invalid("Phi81 challenge slot overflow"))?;
                                let challenge_value = raw_block_value(challenge, challenge_slot, physical)?;
                                let shifted_challenge = sub_mod(challenge_value, recipe.challenge_shift);
                                let value_lane = degree - convolution_source;
                                let value_slot =
                                    family.invocation(recipe.ring_degree, source, block_index, value_lane, cell)?;
                                let product = mul_mod(shifted_challenge, raw_block_value(value, value_slot, physical)?);
                                sum = add_mod(sum, if negative { neg_mod(product) } else { product });
                            }
                            groups.push(sum);
                        }
                    }
                }
            }
        }
    }
    if groups.len() != expected_count {
        return Err(PackageError::Invalid("Phi81 derived group count"));
    }
    Ok(groups)
}

fn derive_first54_products(
    transport: &LoadedAssignmentPlan,
    physical: &PhysicalAssignment<'_>,
) -> Result<Vec<u64>, PackageError> {
    let recipe = &transport.first54;
    let reject = transport.block(recipe.reject_block)?;
    let symbol = transport.block(recipe.symbol_block)?;
    let output = transport.block(recipe.output_block)?;
    if output.slot_count != recipe.candidate_count {
        return Err(PackageError::Invalid("First54 output count"));
    }
    (0..recipe.candidate_count)
        .map(|candidate| {
            let reject = raw_block_value(reject, candidate, physical)?;
            let symbol = raw_block_value(symbol, candidate, physical)?;
            Ok(mul_mod(sub_mod(1, reject), symbol))
        })
        .collect()
}

fn validate_derived_block_sources(
    transport: &LoadedAssignmentPlan,
    domains: &Domains<'_>,
    output_digest: [u64; OUTPUT_DIGEST_WORDS],
) -> Result<(), PackageError> {
    let group = transport.block(transport.phi81.group_output_block)?;
    for slot in 0..group.slot_count {
        let expected = *domains
            .groups
            .get(slot)
            .ok_or(PackageError::Invalid("Phi81 group output count"))?;
        if domains.value(group.domain, group.source(slot)?)? != expected {
            return Err(PackageError::Invalid("Phi81 group source map"));
        }
    }

    let product = transport.block(transport.first54.output_block)?;
    for slot in 0..product.slot_count {
        let expected = *domains
            .products
            .get(slot)
            .ok_or(PackageError::Invalid("First54 output count"))?;
        if domains.value(product.domain, product.source(slot)?)? != expected {
            return Err(PackageError::Invalid("First54 product source map"));
        }
    }

    let payload = transport.block(transport.payload_block)?;
    if payload.slot_count != transport.payload_expressions.len() {
        return Err(PackageError::Invalid("payload output count"));
    }
    for slot in 0..payload.slot_count {
        let expected = *domains
            .payload
            .get(slot)
            .ok_or(PackageError::Invalid("payload output count"))?;
        if domains.value(payload.domain, payload.source(slot)?)? != expected {
            return Err(PackageError::Invalid("payload source map"));
        }
    }

    let digest = transport.block(transport.output_digest_block)?;
    if digest.slot_count != output_digest.len() {
        return Err(PackageError::Invalid("output digest block count"));
    }
    for (slot, expected) in output_digest.into_iter().enumerate() {
        if domains.value(digest.domain, digest.source(slot)?)? != expected {
            return Err(PackageError::Invalid("output digest source map"));
        }
    }
    Ok(())
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

fn block(blocks: &[BlockPlan], opcode: usize) -> Result<&BlockPlan, PackageError> {
    blocks
        .get(opcode)
        .filter(|block| block.opcode == opcode)
        .ok_or(PackageError::Invalid("assignment block selector"))
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
