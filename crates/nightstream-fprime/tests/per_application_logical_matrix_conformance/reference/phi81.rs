//! Independent interpreter for the 34-row Phi81 product opcode.

use serde_json::Value;

use super::{
    checked_add, checked_mul, decode_list, empty_row, exact_array, word, Field, Form, Result, RetainedBlock, RowForms,
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
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 3, "Phi81 family")?;
        Ok(Self {
            source_count: word(&fields[0], "Phi81 source count")?,
            block_count: word(&fields[1], "Phi81 block count")?,
            cell_count: word(&fields[2], "Phi81 cell count")?,
        })
    }

    fn private_count(self) -> Result<usize> {
        checked_mul(
            self.block_count,
            checked_mul(RING_DEGREE, self.cell_count, "Phi81 private lanes")?,
            "Phi81 private count",
        )
    }

    fn invocation_count(self) -> Result<usize> {
        checked_mul(self.source_count, self.private_count()?, "Phi81 invocation count")
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
    fn invocation(self) -> Result<usize> {
        checked_add(self.family_offset, self.local_invocation, "Phi81 invocation")
    }

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

#[derive(Clone, Debug)]
pub struct Block {
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
    pub fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 8, "Phi81 matrix block")?;
        let block = Self {
            families: decode_list(&fields[0], Family::decode, "Phi81 families")?,
            one_column: word(&fields[1], "Phi81 one column")?,
            challenge: RetainedBlock::decode(&fields[2])?,
            challenge_slot_start: word(&fields[3], "Phi81 challenge slot start")?,
            challenge_source_stride: word(&fields[4], "Phi81 challenge source stride")?,
            input: RetainedBlock::decode(&fields[5])?,
            output: RetainedBlock::decode(&fields[6])?,
            group: RetainedBlock::decode(&fields[7])?,
        };
        if block.one_column != 0 || block.one_column >= logical_width {
            return Err("Phi81 one column is not logical column zero".into());
        }
        for retained in [&block.challenge, &block.input, &block.output, &block.group] {
            retained.validate(logical_width)?;
        }
        Ok(block)
    }

    pub fn row_count(&self) -> Result<usize> {
        let invocations = self.families.iter().try_fold(0usize, |count, family| {
            checked_add(count, family.invocation_count()?, "Phi81 invocation count")
        })?;
        checked_mul(invocations, ROWS_PER_INVOCATION, "Phi81 row count")
    }

    pub fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms> {
        if ordinal >= self.row_count()? || self.one_column >= logical_width {
            return Err("Phi81 matrix row is out of range".into());
        }
        let descriptor = self.descriptor(ordinal / ROWS_PER_INVOCATION)?;
        let local_row = ordinal % ROWS_PER_INVOCATION;
        if local_row == GROUP_COUNT {
            return self.final_row(logical_width, descriptor);
        }

        let challenge = self.challenge_state(logical_width, descriptor)?;
        let input = self.input_state(logical_width, descriptor)?;
        let left: [Form; RING_DEGREE] = std::array::from_fn(|lane| {
            challenge[lane].clone().append(Form::singleton(
                self.one_column,
                -Field::checked(2, "Phi81 two").expect("two is canonical"),
            ))
        });
        self.product_row(logical_width, descriptor, &left, &input, local_row)
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

        let first_invocation = start / ROWS_PER_INVOCATION;
        let last_invocation = (end - 1) / ROWS_PER_INVOCATION;
        for invocation in first_invocation..=last_invocation {
            let invocation_start = checked_mul(invocation, ROWS_PER_INVOCATION, "Phi81 matrix row")?;
            let local_start = start
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            let local_end = end
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            let descriptor = self.descriptor(invocation)?;
            let product_state = if local_start < local_end.min(GROUP_COUNT) {
                let challenge = self.challenge_state(logical_width, descriptor)?;
                let input = self.input_state(logical_width, descriptor)?;
                let negative_two = -Field::checked(2, "Phi81 two")?;
                let left = std::array::from_fn(|lane| {
                    challenge[lane]
                        .clone()
                        .append(Form::singleton(self.one_column, negative_two))
                });
                Some((left, input))
            } else {
                None
            };

            for local_row in local_start..local_end {
                let row = if local_row < GROUP_COUNT {
                    let (left, input) = product_state
                        .as_ref()
                        .ok_or_else(|| "missing Phi81 product state".to_string())?;
                    self.product_row(logical_width, descriptor, left, input, local_row)?
                } else {
                    self.final_row(logical_width, descriptor)?
                };
                visit(invocation_start + local_row, row)?;
            }
        }
        Ok(())
    }

    fn descriptor(&self, mut index: usize) -> Result<Descriptor> {
        let mut family_offset = 0usize;
        for &family in &self.families {
            let count = family.invocation_count()?;
            if index < count {
                let private_count = family.private_count()?;
                let lane_cells = checked_mul(RING_DEGREE, family.cell_count, "Phi81 lane cells")?;
                if private_count == 0 || lane_cells == 0 {
                    return Err("zero Phi81 family geometry".into());
                }
                let source = index / private_count;
                let coordinate = index % private_count;
                return Ok(Descriptor {
                    family,
                    family_offset,
                    source,
                    block: coordinate / lane_cells,
                    lane: (coordinate % lane_cells) / family.cell_count,
                    cell: coordinate % family.cell_count,
                    local_invocation: index,
                });
            }
            family_offset = checked_add(family_offset, count, "Phi81 family offset")?;
            index -= count;
        }
        Err("Phi81 invocation is out of range".into())
    }

    fn challenge_state(&self, logical_width: usize, descriptor: Descriptor) -> Result<[Form; RING_DEGREE]> {
        let base = checked_add(
            self.challenge_slot_start,
            checked_mul(
                descriptor.source,
                self.challenge_source_stride,
                "Phi81 challenge source",
            )?,
            "Phi81 challenge base",
        )?;
        fixed_state(|lane| {
            self.challenge
                .form(logical_width, checked_add(base, lane, "Phi81 challenge lane")?)
        })
    }

    fn input_state(&self, logical_width: usize, descriptor: Descriptor) -> Result<[Form; RING_DEGREE]> {
        fixed_state(|lane| {
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
    ) -> Result<RowForms> {
        let mut row = empty_row();
        let left_ports = [0, 3, 6, 9, 11];
        let right_ports = [2, 5, 8, 10, 12];
        let first = checked_mul(group, TERMS_PER_GROUP, "Phi81 group term")?;
        for offset in 0..TERMS_PER_GROUP {
            let term = first + offset;
            if term < 3 * RING_DEGREE {
                let (a, b) = convolution_term(left, right, descriptor.lane, term);
                row[left_ports[offset]] = a;
                row[right_ports[offset]] = b;
            }
        }
        row[4] = self.group.form(
            logical_width,
            checked_add(
                checked_mul(descriptor.invocation()?, GROUP_COUNT, "Phi81 group base")?,
                group,
                "Phi81 group slot",
            )?,
        )?;
        row[7] = Form::singleton(self.one_column, Field::ONE);
        Ok(row)
    }

    fn final_row(&self, logical_width: usize, descriptor: Descriptor) -> Result<RowForms> {
        let invocation = descriptor.invocation()?;
        let output = self.output.form(logical_width, invocation)?;
        let prior = if descriptor.source == 0 {
            Form::default()
        } else {
            self.output.form(
                logical_width,
                invocation
                    .checked_sub(descriptor.family.private_count()?)
                    .ok_or_else(|| "Phi81 prior output underflow".to_string())?,
            )?
        };
        let base = checked_mul(invocation, GROUP_COUNT, "Phi81 group base")?;
        let mut groups = Form::default();
        for group in 0..GROUP_COUNT {
            groups = groups.append(
                self.group
                    .form(logical_width, checked_add(base, group, "Phi81 group slot")?)?,
            );
        }
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Field::ONE);
        row[4] = output
            .append(prior.scaled(-Field::ONE))
            .append(groups.scaled(-Field::ONE));
        Ok(row)
    }
}

fn fixed_state(mut load: impl FnMut(usize) -> Result<Form>) -> Result<[Form; RING_DEGREE]> {
    let forms = (0..RING_DEGREE)
        .map(&mut load)
        .collect::<Result<Vec<_>>>()?;
    Ok(forms.try_into().expect("Phi81 state has 54 forms"))
}

fn convolution_term(left: &[Form; RING_DEGREE], right: &[Form; RING_DEGREE], lane: usize, term: usize) -> (Form, Form) {
    let section = term / RING_DEGREE;
    let source = term % RING_DEGREE;
    let folded = if lane < MIDDLE_DEGREE {
        lane + RING_DEGREE
    } else {
        lane + MIDDLE_DEGREE
    };
    let (degree, coefficient) = match section {
        0 => (lane, Field::ONE),
        1 => (folded, -Field::ONE),
        2 if lane + 81 <= 106 => (lane + 81, Field::ONE),
        2 => return (Form::default(), Form::default()),
        _ => unreachable!("three Phi81 convolution sections"),
    };
    if source <= degree && degree - source < RING_DEGREE {
        (left[source].clone().scaled(coefficient), right[degree - source].clone())
    } else {
        (Form::default(), Form::default())
    }
}
