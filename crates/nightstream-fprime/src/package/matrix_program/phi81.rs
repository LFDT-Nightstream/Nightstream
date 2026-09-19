//! Direct 34-row Phi81 product-family matrix blocks.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::{
    checked_add, checked_mul, decode_list, exact_array, template, usize_atom, Form, PackageError, RetainedBlock,
    RowForms, SourceSubstitution,
};

const RING_DEGREE: usize = 54;
const GROUP_COUNT: usize = 33;
const ROWS_PER_INVOCATION: usize = 34;

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
}

#[derive(Clone, Copy, Debug)]
struct Descriptor {
    family: Family,
    family_offset: usize,
    source: usize,
    block: usize,
    lane: usize,
    cell: usize,
    local_invocation: usize,
}

impl Descriptor {
    fn invocation(self) -> Result<usize, PackageError> {
        checked_add(self.family_offset, self.local_invocation, "Phi81 invocation")
    }

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
pub(super) struct Block {
    families: Vec<Family>,
    one_column: usize,
    challenge: RetainedBlock,
    challenge_slot_start: usize,
    challenge_source_stride: usize,
    input: SourceSubstitution,
    output: RetainedBlock,
    group: RetainedBlock,
}

impl Block {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 8, "Phi81 product block")?;
        Ok(Self {
            families: decode_list(&fields[0], Family::decode)?,
            one_column: usize_atom(&fields[1], "Phi81 one column")?,
            challenge: RetainedBlock::decode(&fields[2])?,
            challenge_slot_start: usize_atom(&fields[3], "Phi81 challenge slot start")?,
            challenge_source_stride: usize_atom(&fields[4], "Phi81 challenge source stride")?,
            input: SourceSubstitution::decode(&fields[5])?,
            output: RetainedBlock::decode(&fields[6])?,
            group: RetainedBlock::decode(&fields[7])?,
        })
    }

    fn invocation_count(&self) -> Result<usize, PackageError> {
        self.families.iter().try_fold(0usize, |sum, family| {
            sum.checked_add(family.invocation_count()?)
                .ok_or(PackageError::Invalid("Phi81 invocation count"))
        })
    }

    pub(super) fn row_count(&self) -> Result<usize, PackageError> {
        checked_mul(self.invocation_count()?, ROWS_PER_INVOCATION, "Phi81 product row count")
    }

    pub(super) fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if ordinal >= self.row_count()? {
            return Err(PackageError::Invalid("Phi81 product row ordinal"));
        }
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("Phi81 one column"));
        }
        let descriptor = self.descriptor(ordinal / ROWS_PER_INVOCATION)?;
        let local_row = ordinal % ROWS_PER_INVOCATION;
        self.invocation_rows(logical_width, descriptor, local_row, local_row + 1)?
            .pop()
            .ok_or(PackageError::Invalid("Phi81 template row count"))
    }

    pub(super) fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(RowForms) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        if start > end || end > self.row_count()? {
            return Err(PackageError::Invalid("Phi81 product row range"));
        }
        if start == end {
            return Ok(());
        }
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("Phi81 one column"));
        }

        let first_invocation = start / ROWS_PER_INVOCATION;
        let last_invocation = (end - 1) / ROWS_PER_INVOCATION;
        for invocation in first_invocation..=last_invocation {
            let invocation_start = checked_mul(invocation, ROWS_PER_INVOCATION, "Phi81 product row")?;
            let local_start = start
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            let local_end = end
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            self.visit_invocation_rows(logical_width, invocation, local_start, local_end, &mut visit)?;
        }
        Ok(())
    }

    fn visit_invocation_rows(
        &self,
        logical_width: usize,
        invocation: usize,
        local_start: usize,
        local_end: usize,
        visit: &mut impl FnMut(RowForms) -> Result<(), PackageError>,
    ) -> Result<(), PackageError> {
        if local_start > local_end || local_end > ROWS_PER_INVOCATION {
            return Err(PackageError::Invalid("Phi81 invocation row range"));
        }
        let descriptor = self.descriptor(invocation)?;
        for row in self.invocation_rows(logical_width, descriptor, local_start, local_end)? {
            visit(row)?;
        }
        Ok(())
    }

    fn descriptor(&self, mut index: usize) -> Result<Descriptor, PackageError> {
        let mut family_offset = 0usize;
        for &family in &self.families {
            let invocation_count = family.invocation_count()?;
            if index < invocation_count {
                let private_count = family.private_count()?;
                if private_count == 0 || family.cell_count == 0 {
                    return Err(PackageError::Invalid("Phi81 family geometry"));
                }
                let source = index / private_count;
                let coordinate = index % private_count;
                let lane_cell_count = checked_mul(RING_DEGREE, family.cell_count, "Phi81 family coordinate")?;
                return Ok(Descriptor {
                    family,
                    family_offset,
                    source,
                    block: coordinate / lane_cell_count,
                    lane: (coordinate % lane_cell_count) / family.cell_count,
                    cell: coordinate % family.cell_count,
                    local_invocation: index,
                });
            }
            family_offset = checked_add(family_offset, invocation_count, "Phi81 family offset")?;
            index -= invocation_count;
        }
        Err(PackageError::Invalid("Phi81 family descriptor"))
    }

    fn challenge_state(
        &self,
        logical_width: usize,
        descriptor: Descriptor,
    ) -> Result<[Form; RING_DEGREE], PackageError> {
        let source_base = checked_add(
            self.challenge_slot_start,
            checked_mul(descriptor.source, self.challenge_source_stride, "Phi81 challenge slot")?,
            "Phi81 challenge slot",
        )?;
        fixed_ring_state(|lane| {
            self.challenge
                .form(logical_width, checked_add(source_base, lane, "Phi81 challenge slot")?)
        })
    }

    fn input_state(&self, logical_width: usize, descriptor: Descriptor) -> Result<[Form; RING_DEGREE], PackageError> {
        fixed_ring_state(|lane| {
            self.input
                .form(logical_width, descriptor.invocation_at_lane(lane)?)
        })
    }

    fn invocation_rows(
        &self,
        logical_width: usize,
        descriptor: Descriptor,
        start: usize,
        end: usize,
    ) -> Result<Vec<RowForms>, PackageError> {
        let invocation = descriptor.invocation()?;
        let mut inputs = Vec::with_capacity(1 + 2 * RING_DEGREE + GROUP_COUNT + 2);
        inputs.push(Form::singleton(self.one_column, Goldilocks::ONE));
        inputs.extend(self.challenge_state(logical_width, descriptor)?);
        inputs.extend(self.input_state(logical_width, descriptor)?);
        let group_base = checked_mul(invocation, GROUP_COUNT, "Phi81 group output slot")?;
        for group in 0..GROUP_COUNT {
            inputs.push(self.group.form(
                logical_width,
                checked_add(group_base, group, "Phi81 group output slot")?,
            )?);
        }
        let prior = if descriptor.source == 0 {
            Form::default()
        } else {
            self.output.form(
                logical_width,
                invocation
                    .checked_sub(descriptor.family.private_count()?)
                    .ok_or(PackageError::Invalid("Phi81 prior output slot"))?,
            )?
        };
        inputs.push(prior);
        inputs.push(self.output.form(logical_width, invocation)?);
        template::rows("phi81-product-v1", descriptor.lane, &inputs, logical_width, start..end)
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
