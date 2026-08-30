//! Direct 34-row Phi81 product-family matrix blocks.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::{
    checked_add, checked_mul, decode_list, empty_row, exact_array, usize_atom, Form, PackageError, RetainedBlock,
    RowForms,
};

const RING_DEGREE: usize = 54;
const MIDDLE_DEGREE: usize = 27;
const TERMS_PER_GROUP: usize = 5;
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
    input: RetainedBlock,
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
            input: RetainedBlock::decode(&fields[5])?,
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
        let challenge = self.challenge_state(logical_width, descriptor)?;
        let input = self.input_state(logical_width, descriptor)?;
        let left: [Form; RING_DEGREE] = std::array::from_fn(|lane| {
            challenge[lane]
                .clone()
                .append(Form::singleton(self.one_column, -Goldilocks::from_u64(2)))
        });

        if local_row < GROUP_COUNT {
            self.product_row(logical_width, descriptor, &left, &input, local_row)
        } else {
            self.final_row(logical_width, descriptor)
        }
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

    fn product_row(
        &self,
        logical_width: usize,
        descriptor: Descriptor,
        left: &[Form; RING_DEGREE],
        right: &[Form; RING_DEGREE],
        group: usize,
    ) -> Result<RowForms, PackageError> {
        let terms = product_terms(left, right, descriptor.lane);
        let first = checked_mul(group, TERMS_PER_GROUP, "Phi81 group term")?;
        let mut row = empty_row();
        let left_ports = [0, 3, 6, 9, 11];
        let right_ports = [2, 5, 8, 10, 12];
        for lane in 0..TERMS_PER_GROUP {
            if let Some(term) = terms.get(first + lane) {
                row[left_ports[lane]] = term.0.clone();
                row[right_ports[lane]] = term.1.clone();
            }
        }
        row[4] = self.group.form(
            logical_width,
            checked_add(
                checked_mul(descriptor.invocation()?, GROUP_COUNT, "Phi81 group output slot")?,
                group,
                "Phi81 group output slot",
            )?,
        )?;
        row[7] = Form::singleton(self.one_column, Goldilocks::ONE);
        Ok(row)
    }

    fn final_row(&self, logical_width: usize, descriptor: Descriptor) -> Result<RowForms, PackageError> {
        let invocation = descriptor.invocation()?;
        let output = self.output.form(logical_width, invocation)?;
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
        let mut group_sum = Form::default();
        let group_base = checked_mul(invocation, GROUP_COUNT, "Phi81 group output slot")?;
        for group in 0..GROUP_COUNT {
            group_sum = group_sum.append(self.group.form(
                logical_width,
                checked_add(group_base, group, "Phi81 group output slot")?,
            )?);
        }
        let difference = output
            .append(prior.scaled(-Goldilocks::ONE))
            .append(group_sum.scaled(-Goldilocks::ONE));
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Goldilocks::ONE);
        row[4] = difference;
        Ok(row)
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

fn product_terms(left: &[Form; RING_DEGREE], right: &[Form; RING_DEGREE], lane: usize) -> Vec<(Form, Form)> {
    let folded_degree = if lane < MIDDLE_DEGREE {
        lane + RING_DEGREE
    } else {
        lane + MIDDLE_DEGREE
    };
    let twice = if lane + 81 <= 106 {
        Goldilocks::ONE
    } else {
        Goldilocks::ZERO
    };
    let mut terms = Vec::with_capacity(3 * RING_DEGREE);
    append_raw_terms(&mut terms, Goldilocks::ONE, left, right, lane);
    append_raw_terms(&mut terms, -Goldilocks::ONE, left, right, folded_degree);
    append_raw_terms(&mut terms, twice, left, right, lane + 81);
    terms
}

fn append_raw_terms(
    terms: &mut Vec<(Form, Form)>,
    coefficient: Goldilocks,
    left: &[Form; RING_DEGREE],
    right: &[Form; RING_DEGREE],
    degree: usize,
) {
    for source in 0..RING_DEGREE {
        if source <= degree && degree - source < RING_DEGREE {
            terms.push((left[source].scaled(coefficient), right[degree - source].clone()));
        } else {
            terms.push((Form::default(), Form::default()));
        }
    }
}
