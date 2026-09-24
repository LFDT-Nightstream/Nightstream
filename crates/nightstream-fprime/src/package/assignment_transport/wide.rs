//! Execute the wide sampler's Lean-authored retained-value schedule.
//! Source runs select physical values or the derived Phi81 quotient suffix.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Values {
    kind: SlotKind,
    count: usize,
    runs: Vec<Run>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    blocks: Vec<Values>,
    families: Vec<Phi81FamilyShape>,
    values: Vec<Run>,
    challenges: Vec<Run>,
    digest: Vec<Expression>,
    physical_width: usize,
    public_width: usize,
    logical_width: usize,
}

impl Plan {
    fn quotients(&self, physical: &PhysicalAssignment<'_>) -> Result<Vec<u64>, PackageError> {
        let mut quotients = vec![0; PHI81_INVOCATIONS];
        for family in &self.families {
            for source in 0..family.source_count {
                let mut left = [0; PHI81_RING_DEGREE];
                for (lane, coefficient) in left.iter_mut().enumerate() {
                    let index = (source * PHI81_RING_DEGREE + lane) * 3;
                    let mut digit = 0;
                    for bit in 0..3 {
                        let value = physical.value(source_run_at(&self.challenges, index + bit)?)?;
                        if value > 1 {
                            return Err(PackageError::Invalid("wide challenge bit"));
                        }
                        digit += value << bit;
                    }
                    if digit > 4 {
                        return Err(PackageError::Invalid("wide challenge digit"));
                    }
                    *coefficient = sub_mod(digit, 2);
                }
                for block in 0..family.block_count {
                    for cell in 0..family.cell_count {
                        let mut right = [0; PHI81_RING_DEGREE];
                        for (lane, coefficient) in right.iter_mut().enumerate() {
                            let invocation = family.invocation(PHI81_RING_DEGREE, source, block, lane, cell)?;
                            *coefficient = physical.value(source_run_at(&self.values, invocation)?)?;
                        }
                        for (lane, coefficient) in phi81_quotient(&left, &right).into_iter().enumerate() {
                            let invocation = family.invocation(PHI81_RING_DEGREE, source, block, lane, cell)?;
                            quotients[invocation] = coefficient;
                        }
                    }
                }
            }
        }
        Ok(quotients)
    }

    pub(super) fn execute(
        &self,
        layout: &Layout,
        assignment: &WitnessAssignment,
    ) -> Result<LogicalAssignment, PackageError> {
        let physical = PhysicalAssignment::new(layout, assignment, self.physical_width)?;
        let quotients = self.quotients(&physical)?;
        let digest: [u64; OUTPUT_DIGEST_WORDS] = self
            .digest
            .iter()
            .map(|expression| expression.evaluate(&physical))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| PackageError::Invalid("output digest word count"))?;
        let mut values = Vec::with_capacity(self.logical_width);
        append_public(digest, self.public_width, &mut values)?;
        for block in &self.blocks {
            for run in &block.runs {
                for ordinal in 0..run.count {
                    let source = run.first + ordinal * run.step;
                    let value = if source < self.physical_width {
                        physical.value(source)?
                    } else {
                        *quotients
                            .get(source - self.physical_width)
                            .ok_or(PackageError::Invalid("wide quotient source"))?
                    };
                    encode_slot(block.kind, value, &mut values)?;
                }
            }
        }
        if values.len() != self.logical_width {
            return Err(PackageError::Invalid("wide assignment coordinate width"));
        }
        Ok(LogicalAssignment { values })
    }
}

pub(super) fn decode(
    value: &Value,
    physical_width: usize,
    public_width: usize,
    logical_width: usize,
) -> Result<Plan, PackageError> {
    let fields = exact_array(value, 4, "wide assignment transport plan")?;
    let recipe = exact_array(&fields[2], 3, "wide quotient recipe")?;
    let shapes = recipe[0]
        .as_array()
        .ok_or(PackageError::Invalid("wide quotient families"))?;
    let mut families = Vec::with_capacity(shapes.len());
    let mut invocation_count = 0usize;
    let mut source_count = None;
    for shape in shapes {
        let words = exact_array(shape, 3, "wide quotient family")?;
        let family = Phi81FamilyShape {
            source_count: word(&words[0], "wide quotient sources")?,
            block_count: word(&words[1], "wide quotient blocks")?,
            cell_count: word(&words[2], "wide quotient cells")?,
            first_invocation: invocation_count,
        };
        if family.source_count == 0
            || family.block_count == 0
            || family.cell_count == 0
            || source_count.is_some_and(|count| count != family.source_count)
        {
            return Err(PackageError::Invalid("wide quotient family shape"));
        }
        source_count = Some(family.source_count);
        invocation_count = invocation_count
            .checked_add(
                family
                    .source_count
                    .checked_mul(family.block_count)
                    .and_then(|count| count.checked_mul(PHI81_RING_DEGREE))
                    .and_then(|count| count.checked_mul(family.cell_count))
                    .ok_or(PackageError::Invalid("wide quotient count overflow"))?,
            )
            .ok_or(PackageError::Invalid("wide quotient count overflow"))?;
        families.push(family);
    }
    if invocation_count != PHI81_INVOCATIONS || source_count != Some(17) {
        return Err(PackageError::Invalid("wide quotient profile"));
    }
    let values = decode_source_runs(&recipe[1], invocation_count, physical_width)?;
    let challenges = decode_source_runs(&recipe[2], 17 * PHI81_RING_DEGREE * 3, physical_width)?;
    let retained_width = physical_width
        .checked_add(invocation_count)
        .ok_or(PackageError::Invalid("wide source width overflow"))?;
    let raw_blocks = fields[1]
        .as_array()
        .ok_or(PackageError::Invalid("wide retained blocks"))?;
    let mut blocks = Vec::with_capacity(raw_blocks.len());
    let mut encoded_width = public_width;
    for raw in raw_blocks {
        let words = exact_array(raw, 3, "wide retained block")?;
        let kind = SlotKind::decode(&words[0])?;
        let count = word(&words[1], "wide retained slot count")?;
        let runs = decode_source_runs(&words[2], count, retained_width)?;
        encoded_width = encoded_width
            .checked_add(
                count
                    .checked_mul(kind.coordinate_width())
                    .ok_or(PackageError::Invalid("wide coordinate width overflow"))?,
            )
            .ok_or(PackageError::Invalid("wide coordinate width overflow"))?;
        blocks.push(Values { kind, count, runs });
    }
    if encoded_width != logical_width {
        return Err(PackageError::Invalid("wide assignment coordinate width"));
    }
    let digest = decode_expressions(&fields[3], OUTPUT_DIGEST_WORDS, physical_width)?;
    Ok(Plan {
        blocks,
        families,
        values,
        challenges,
        digest,
        physical_width,
        public_width,
        logical_width,
    })
}
