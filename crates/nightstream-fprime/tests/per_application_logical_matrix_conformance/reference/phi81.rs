//! Independent polynomial-evaluation interpreter for the Phi81 quotient opcode.

use serde_json::Value;

use super::matrix::SourceSubstitution;
use super::{
    checked_add, checked_mul, decode_list, empty_row, exact_array, word, Field, Form, Result, RetainedBlock, RowForms,
};

const RING_DEGREE: usize = 54;
const ROWS_PER_RING: usize = 108;

#[derive(Clone, Copy, Debug)]
struct Family {
    source_count: usize,
    block_count: usize,
    cell_count: usize,
}

impl Family {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 3, "Phi81 family")?;
        Ok(Self {
            source_count: word(&fields[0], "Phi81 source count")?,
            block_count: word(&fields[1], "Phi81 block count")?,
            cell_count: word(&fields[2], "Phi81 cell count")?,
        })
    }

    fn rings_per_source(self) -> Result<usize> {
        checked_mul(self.block_count, self.cell_count, "Phi81 rings per source")
    }

    fn ring_count(self) -> Result<usize> {
        checked_mul(self.source_count, self.rings_per_source()?, "Phi81 ring count")
    }

    fn private_count(self) -> Result<usize> {
        checked_mul(self.rings_per_source()?, RING_DEGREE, "Phi81 private count")
    }
}

#[derive(Clone, Copy, Debug)]
struct Descriptor {
    family: Family,
    family_offset: usize,
    source: usize,
    block: usize,
    cell: usize,
}

impl Descriptor {
    fn invocation_at_lane(self, lane: usize) -> Result<usize> {
        let block = checked_mul(
            self.block,
            checked_mul(RING_DEGREE, self.family.cell_count, "Phi81 block coordinates")?,
            "Phi81 block coordinate",
        )?;
        let lane = checked_add(
            checked_mul(lane, self.family.cell_count, "Phi81 lane coordinate")?,
            self.cell,
            "Phi81 lane coordinate",
        )?;
        checked_add(
            self.family_offset,
            checked_add(
                checked_mul(self.source, self.family.private_count()?, "Phi81 source coordinate")?,
                checked_add(block, lane, "Phi81 private coordinate")?,
                "Phi81 invocation coordinate",
            )?,
            "Phi81 invocation coordinate",
        )
    }
}

struct RingForms {
    left: [Form; RING_DEGREE],
    right: [Form; RING_DEGREE],
    quotient: [Form; RING_DEGREE],
    difference: [Form; RING_DEGREE],
}

#[derive(Clone, Debug)]
pub struct Block {
    families: Vec<Family>,
    one_column: usize,
    challenge: RetainedBlock,
    challenge_slot_start: usize,
    challenge_source_stride: usize,
    input: SourceSubstitution,
    output: RetainedBlock,
    quotient: RetainedBlock,
}

impl Block {
    pub fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 8, "Phi81 matrix block")?;
        let block = Self {
            families: decode_list(&fields[0], Family::decode, "Phi81 families")?,
            one_column: word(&fields[1], "Phi81 one column")?,
            challenge: RetainedBlock::decode(&fields[2])?,
            challenge_slot_start: word(&fields[3], "Phi81 challenge slot start")?,
            challenge_source_stride: word(&fields[4], "Phi81 challenge source stride")?,
            input: SourceSubstitution::decode(&fields[5], logical_width)?,
            output: RetainedBlock::decode(&fields[6])?,
            quotient: RetainedBlock::decode(&fields[7])?,
        };
        if block.one_column != 0 || block.one_column >= logical_width {
            return Err("Phi81 one column is not logical column zero".into());
        }
        for retained in [&block.challenge, &block.output, &block.quotient] {
            retained.validate(logical_width)?;
        }
        Ok(block)
    }

    pub fn row_count(&self) -> Result<usize> {
        let rings = self.families.iter().try_fold(0usize, |count, family| {
            checked_add(count, family.ring_count()?, "Phi81 ring count")
        })?;
        checked_mul(rings, ROWS_PER_RING, "Phi81 row count")
    }

    pub fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms> {
        if ordinal >= self.row_count()? || self.one_column >= logical_width {
            return Err("Phi81 matrix row is out of range".into());
        }
        let descriptor = self.descriptor(ordinal / ROWS_PER_RING)?;
        let forms = self.ring_forms(logical_width, descriptor)?;
        self.evaluation_row(&forms, ordinal % ROWS_PER_RING)
    }

    pub fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(usize, RowForms) -> Result<()>,
    ) -> Result<()> {
        if start > end || end > self.row_count()? {
            return Err("Phi81 matrix row range is out of bounds".into());
        }
        if start == end {
            return Ok(());
        }
        if self.one_column >= logical_width {
            return Err("Phi81 one column is out of range".into());
        }
        for ring in start / ROWS_PER_RING..=(end - 1) / ROWS_PER_RING {
            let ring_start = checked_mul(ring, ROWS_PER_RING, "Phi81 ring row")?;
            let forms = self.ring_forms(logical_width, self.descriptor(ring)?)?;
            for ordinal in start.max(ring_start)..end.min(ring_start + ROWS_PER_RING) {
                visit(ordinal, self.evaluation_row(&forms, ordinal - ring_start)?)?;
            }
        }
        Ok(())
    }

    fn descriptor(&self, mut index: usize) -> Result<Descriptor> {
        let mut family_offset = 0usize;
        for &family in &self.families {
            let count = family.ring_count()?;
            if index < count {
                let per_source = family.rings_per_source()?;
                if per_source == 0 || family.cell_count == 0 {
                    return Err("zero Phi81 family geometry".into());
                }
                let coordinate = index % per_source;
                return Ok(Descriptor {
                    family,
                    family_offset,
                    source: index / per_source,
                    block: coordinate / family.cell_count,
                    cell: coordinate % family.cell_count,
                });
            }
            family_offset = checked_add(
                family_offset,
                checked_mul(count, RING_DEGREE, "Phi81 family width")?,
                "Phi81 family offset",
            )?;
            index -= count;
        }
        Err("Phi81 ring is out of range".into())
    }

    fn ring_forms(&self, logical_width: usize, descriptor: Descriptor) -> Result<RingForms> {
        let challenge_base = checked_add(
            self.challenge_slot_start,
            checked_mul(
                descriptor.source,
                self.challenge_source_stride,
                "Phi81 challenge source",
            )?,
            "Phi81 challenge base",
        )?;
        let negative_two = -Field::checked(2, "Phi81 centering")?;
        Ok(RingForms {
            left: fixed_state(|lane| {
                Ok(self
                    .challenge
                    .form(
                        logical_width,
                        checked_add(challenge_base, lane, "Phi81 challenge lane")?,
                    )?
                    .append(Form::singleton(self.one_column, negative_two)))
            })?,
            right: fixed_state(|lane| {
                self.input
                    .form(logical_width, descriptor.invocation_at_lane(lane)?)
            })?,
            quotient: fixed_state(|lane| {
                self.quotient
                    .form(logical_width, descriptor.invocation_at_lane(lane)?)
            })?,
            difference: fixed_state(|lane| {
                let invocation = descriptor.invocation_at_lane(lane)?;
                let output = self.output.form(logical_width, invocation)?;
                if descriptor.source == 0 {
                    Ok(output)
                } else {
                    let prior = invocation
                        .checked_sub(descriptor.family.private_count()?)
                        .ok_or_else(|| "Phi81 prior output underflow".to_string())?;
                    Ok(output.append(self.output.form(logical_width, prior)?.scaled(-Field::ONE)))
                }
            })?,
        })
    }

    fn evaluation_row(&self, forms: &RingForms, point: usize) -> Result<RowForms> {
        let point = Field::checked(point as u64, "Phi81 evaluation node")?;
        let mut powers = [Field::ONE; 55];
        for degree in 1..powers.len() {
            powers[degree] = powers[degree - 1] * point;
        }
        let phi81 = powers[54] + powers[27] + Field::ONE;
        let mut row = empty_row();
        for (degree, &power) in powers[..54].iter().enumerate() {
            row[0] = row[0]
                .clone()
                .append(forms.left[degree].clone().scaled(power));
            row[2] = row[2]
                .clone()
                .append(forms.right[degree].clone().scaled(power));
            row[4] = row[4]
                .clone()
                .append(forms.difference[degree].clone().scaled(power))
                .append(forms.quotient[degree].clone().scaled(phi81 * power));
        }
        row[7] = Form::singleton(self.one_column, Field::ONE);
        Ok(row)
    }
}

fn fixed_state(mut load: impl FnMut(usize) -> Result<Form>) -> Result<[Form; RING_DEGREE]> {
    let forms = (0..RING_DEGREE)
        .map(&mut load)
        .collect::<Result<Vec<_>>>()?;
    Ok(forms.try_into().expect("Phi81 state has 54 forms"))
}
