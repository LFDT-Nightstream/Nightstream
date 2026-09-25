//! Phi81 quotient matrix blocks, with 108 evaluations per complete ring product.

use std::ops::ControlFlow;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::ColumnProjection;
use super::{
    array, checked_add, checked_mul, checked_wire_form, decode_entries, decode_list, exact_array, owned_row, template,
    usize_atom, Entry, Form, PackageError, RetainedBlock, RowForms, RowView, SourceSubstitution,
};

const RING_DEGREE: usize = 54;
const ROWS_PER_RING: usize = 2 * RING_DEGREE;

#[derive(Clone, Copy, Debug)]
struct Family {
    source_count: usize,
    block_count: usize,
    cell_count: usize,
}

impl Family {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 3, "Phi81 product family")?;
        Ok(Self {
            source_count: usize_atom(&fields[0], "Phi81 source count")?,
            block_count: usize_atom(&fields[1], "Phi81 block count")?,
            cell_count: usize_atom(&fields[2], "Phi81 cell count")?,
        })
    }

    fn private_count(self) -> Result<usize, PackageError> {
        checked_mul(
            self.block_count,
            checked_mul(RING_DEGREE, self.cell_count, "Phi81 family private count")?,
            "Phi81 family private count",
        )
    }

    fn invocation_count(self) -> Result<usize, PackageError> {
        checked_mul(
            self.source_count,
            self.private_count()?,
            "Phi81 family invocation count",
        )
    }

    fn ring_count(self) -> Result<usize, PackageError> {
        checked_mul(
            self.source_count,
            checked_mul(self.block_count, self.cell_count, "Phi81 ring count")?,
            "Phi81 ring count",
        )
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
    fn invocation_at_lane(self, lane: usize) -> Result<usize, PackageError> {
        let coordinate = checked_add(
            checked_mul(
                self.block,
                checked_mul(RING_DEGREE, self.family.cell_count, "Phi81 coordinate")?,
                "Phi81 coordinate",
            )?,
            checked_add(
                checked_mul(lane, self.family.cell_count, "Phi81 coordinate")?,
                self.cell,
                "Phi81 coordinate",
            )?,
            "Phi81 coordinate",
        )?;
        checked_add(
            self.family_offset,
            checked_add(
                checked_mul(self.source, self.family.private_count()?, "Phi81 lane invocation")?,
                coordinate,
                "Phi81 lane invocation",
            )?,
            "Phi81 lane invocation",
        )
    }
}

#[derive(Clone, Debug)]
enum Challenge {
    Retained {
        block: RetainedBlock,
        slot_start: usize,
    },
    Direct(Vec<Vec<Entry>>),
}

#[derive(Clone, Debug)]
pub(super) struct Block {
    families: Vec<Family>,
    one_column: usize,
    challenge: Challenge,
    challenge_source_stride: usize,
    input: SourceSubstitution,
    output: RetainedBlock,
    quotient: RetainedBlock,
}

impl Block {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = array(value, "Phi81 product block")?;
        let (challenge, stride, input, output, quotient) = match fields {
            [_, _, block, start, stride, input, output, quotient] => (
                Challenge::Retained {
                    block: RetainedBlock::decode(block)?,
                    slot_start: usize_atom(start, "Phi81 challenge slot start")?,
                },
                stride,
                input,
                output,
                quotient,
            ),
            [_, _, forms, stride, input, output, quotient] => (
                Challenge::Direct(decode_list(forms, decode_entries)?),
                stride,
                input,
                output,
                quotient,
            ),
            _ => return Err(PackageError::Invalid("Phi81 product block")),
        };
        Ok(Self {
            families: decode_list(&fields[0], Family::decode)?,
            one_column: usize_atom(&fields[1], "Phi81 one column")?,
            challenge,
            challenge_source_stride: usize_atom(stride, "Phi81 challenge source stride")?,
            input: SourceSubstitution::decode(input)?,
            output: RetainedBlock::decode(output)?,
            quotient: RetainedBlock::decode(quotient)?,
        })
    }

    pub(super) fn map_columns(&mut self, projection: &ColumnProjection) -> Result<(), PackageError> {
        self.one_column = projection.column(self.one_column)?;
        match &mut self.challenge {
            Challenge::Retained { block, .. } => projection.retained(block)?,
            Challenge::Direct(forms) => {
                for form in forms {
                    projection.entries(form)?;
                }
            }
        }
        self.input.map_columns(projection)?;
        projection.retained(&mut self.output)?;
        projection.retained(&mut self.quotient)
    }

    fn ring_count(&self) -> Result<usize, PackageError> {
        self.families.iter().try_fold(0usize, |sum, family| {
            sum.checked_add(family.ring_count()?)
                .ok_or(PackageError::Invalid("Phi81 ring count"))
        })
    }

    pub(super) fn row_count(&self) -> Result<usize, PackageError> {
        checked_mul(self.ring_count()?, ROWS_PER_RING, "Phi81 product row count")
    }

    pub(super) fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if ordinal >= self.row_count()? {
            return Err(PackageError::Invalid("Phi81 product row ordinal"));
        }
        let mut result = None;
        let _ = self.visit_rows_until(logical_width, ordinal, ordinal + 1, |row| {
            result = Some(owned_row(row));
            Ok(ControlFlow::Break(()))
        })?;
        result.ok_or(PackageError::Invalid("Phi81 template row count"))
    }

    pub(super) fn visit_rows_until(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(RowView<'_>) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        if start > end || end > self.row_count()? {
            return Err(PackageError::Invalid("Phi81 product row range"));
        }
        if start == end {
            return Ok(ControlFlow::Continue(()));
        }
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("Phi81 one column"));
        }

        let mut scratch = template::RowScratch::default();
        let first_invocation = start / ROWS_PER_RING;
        let last_invocation = (end - 1) / ROWS_PER_RING;
        for invocation in first_invocation..=last_invocation {
            let invocation_start = checked_mul(invocation, ROWS_PER_RING, "Phi81 product row")?;
            let local_start = start.saturating_sub(invocation_start).min(ROWS_PER_RING);
            let local_end = end.saturating_sub(invocation_start).min(ROWS_PER_RING);
            let descriptor = self.descriptor(invocation)?;
            let inputs = self.invocation_inputs(logical_width, descriptor)?;
            if scratch
                .visit_rows_until(
                    "phi81-product-v1",
                    0,
                    &inputs,
                    logical_width,
                    local_start..local_end,
                    &mut visit,
                )?
                .is_break()
            {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn descriptor(&self, mut index: usize) -> Result<Descriptor, PackageError> {
        let mut family_offset = 0usize;
        for &family in &self.families {
            let ring_count = family.ring_count()?;
            if index < ring_count {
                let rings_per_source = checked_mul(family.block_count, family.cell_count, "Phi81 rings per source")?;
                if rings_per_source == 0 || family.cell_count == 0 {
                    return Err(PackageError::Invalid("Phi81 family geometry"));
                }
                let source = index / rings_per_source;
                let coordinate = index % rings_per_source;
                return Ok(Descriptor {
                    family,
                    family_offset,
                    source,
                    block: coordinate / family.cell_count,
                    cell: coordinate % family.cell_count,
                });
            }
            family_offset = checked_add(family_offset, family.invocation_count()?, "Phi81 family offset")?;
            index -= ring_count;
        }
        Err(PackageError::Invalid("Phi81 family descriptor"))
    }

    fn challenge_state(
        &self,
        logical_width: usize,
        descriptor: Descriptor,
    ) -> Result<[Form; RING_DEGREE], PackageError> {
        let source_base = checked_mul(descriptor.source, self.challenge_source_stride, "Phi81 challenge slot")?;
        fixed_ring_state(|lane| {
            let index = checked_add(source_base, lane, "Phi81 challenge slot")?;
            match &self.challenge {
                Challenge::Retained { block, slot_start } => {
                    block.form(logical_width, checked_add(*slot_start, index, "Phi81 challenge slot")?)
                }
                Challenge::Direct(forms) => {
                    let entries = forms
                        .get(index)
                        .ok_or(PackageError::Invalid("Phi81 direct challenge table"))?;
                    let centered = checked_wire_form(entries, logical_width)?;
                    // The saved Lean template takes an uncentered digit and subtracts two.
                    // The new wire form is already centered; cancel that template offset.
                    Ok(centered.append(Form::singleton(self.one_column, Goldilocks::from_u64(2))))
                }
            }
        })
    }

    fn input_state(&self, logical_width: usize, descriptor: Descriptor) -> Result<[Form; RING_DEGREE], PackageError> {
        fixed_ring_state(|lane| {
            self.input
                .form(logical_width, descriptor.invocation_at_lane(lane)?)
        })
    }

    fn invocation_inputs(&self, logical_width: usize, descriptor: Descriptor) -> Result<Vec<Form>, PackageError> {
        let mut inputs = Vec::with_capacity(1 + 5 * RING_DEGREE);
        inputs.push(Form::singleton(self.one_column, Goldilocks::ONE));
        inputs.extend(self.challenge_state(logical_width, descriptor)?);
        inputs.extend(self.input_state(logical_width, descriptor)?);
        inputs.extend(fixed_ring_state(|lane| {
            self.quotient
                .form(logical_width, descriptor.invocation_at_lane(lane)?)
        })?);
        inputs.extend(fixed_ring_state(|lane| {
            if descriptor.source == 0 {
                Ok(Form::default())
            } else {
                self.output.form(
                    logical_width,
                    descriptor
                        .invocation_at_lane(lane)?
                        .checked_sub(descriptor.family.private_count()?)
                        .ok_or(PackageError::Invalid("Phi81 prior output slot"))?,
                )
            }
        })?);
        inputs.extend(fixed_ring_state(|lane| {
            self.output
                .form(logical_width, descriptor.invocation_at_lane(lane)?)
        })?);
        Ok(inputs)
    }
}

fn fixed_ring_state(
    mut load: impl FnMut(usize) -> Result<Form, PackageError>,
) -> Result<[Form; RING_DEGREE], PackageError> {
    let forms = (0..RING_DEGREE)
        .map(&mut load)
        .collect::<Result<Vec<_>, _>>()?;
    forms
        .try_into()
        .map_err(|_| PackageError::Invalid("Phi81 ring state"))
}
