//! Poseidon2 input-state programs carried by the Lean matrix package.

use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::{
    array, checked_add, checked_mul, decode_list, exact_array, field_atom, usize_atom, Form, PackageError,
    RetainedBlock,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InvocationTag {
    Absorb,
    SqueezeFirst,
    SqueezeSecond,
}

impl InvocationTag {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        match usize_atom(value, "Poseidon2 invocation tag")? {
            0 => Ok(Self::Absorb),
            1 => Ok(Self::SqueezeFirst),
            2 => Ok(Self::SqueezeSecond),
            _ => Err(PackageError::Invalid("Poseidon2 invocation tag")),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    invocation_start: usize,
    invocation_count: usize,
    lane_start: usize,
    lane_count: usize,
}

impl Region {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 4, "Poseidon2 input region")?;
        Ok(Self {
            invocation_start: usize_atom(&fields[0], "Poseidon2 invocation start")?,
            invocation_count: usize_atom(&fields[1], "Poseidon2 invocation count")?,
            lane_start: usize_atom(&fields[2], "Poseidon2 lane start")?,
            lane_count: usize_atom(&fields[3], "Poseidon2 lane count")?,
        })
    }

    fn offsets(self, invocation: usize, lane: usize) -> Option<(usize, usize)> {
        let invocation_offset = invocation.checked_sub(self.invocation_start)?;
        let lane_offset = lane.checked_sub(self.lane_start)?;
        (invocation_offset < self.invocation_count && lane_offset < self.lane_count)
            .then_some((invocation_offset, lane_offset))
    }
}

#[derive(Clone, Debug)]
enum Term {
    Retained {
        block: RetainedBlock,
        slot_base: usize,
        invocation_stride: usize,
        lane_stride: usize,
    },
    Constant(Goldilocks),
    External {
        block: RetainedBlock,
        slot_base: usize,
        invocation_stride: usize,
    },
    TaggedRetained {
        block: RetainedBlock,
        tags: Vec<InvocationTag>,
        required: InvocationTag,
        slot_base: usize,
        invocation_stride: usize,
        lane_stride: usize,
    },
    OptionalConstant {
        values: Vec<Option<Goldilocks>>,
        lane_count: usize,
    },
}

impl Term {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = array(value, "Poseidon2 input term")?;
        match fields.first().and_then(Value::as_u64) {
            Some(0) if fields.len() == 5 => Ok(Self::Retained {
                block: RetainedBlock::decode(&fields[1])?,
                slot_base: usize_atom(&fields[2], "Poseidon2 slot base")?,
                invocation_stride: usize_atom(&fields[3], "Poseidon2 invocation stride")?,
                lane_stride: usize_atom(&fields[4], "Poseidon2 lane stride")?,
            }),
            Some(1) if fields.len() == 2 => Ok(Self::Constant(field_atom(&fields[1], "Poseidon2 input constant")?)),
            Some(2) if fields.len() == 4 => Ok(Self::External {
                block: RetainedBlock::decode(&fields[1])?,
                slot_base: usize_atom(&fields[2], "Poseidon2 external slot base")?,
                invocation_stride: usize_atom(&fields[3], "Poseidon2 external invocation stride")?,
            }),
            Some(3) if fields.len() == 7 => Ok(Self::TaggedRetained {
                block: RetainedBlock::decode(&fields[1])?,
                tags: decode_list(&fields[2], InvocationTag::decode)?,
                required: InvocationTag::decode(&fields[3])?,
                slot_base: usize_atom(&fields[4], "Poseidon2 tagged slot base")?,
                invocation_stride: usize_atom(&fields[5], "Poseidon2 tagged invocation stride")?,
                lane_stride: usize_atom(&fields[6], "Poseidon2 tagged lane stride")?,
            }),
            Some(4) if fields.len() == 3 => Ok(Self::OptionalConstant {
                values: decode_optional_constants(&fields[1])?,
                lane_count: usize_atom(&fields[2], "Poseidon2 optional lane count")?,
            }),
            _ => Err(PackageError::Invalid("Poseidon2 input term")),
        }
    }

    fn form(
        &self,
        logical_width: usize,
        one_column: usize,
        invocation_offset: usize,
        lane_offset: usize,
    ) -> Result<Form, PackageError> {
        match self {
            Self::Retained {
                block,
                slot_base,
                invocation_stride,
                lane_stride,
            } => block.form(
                logical_width,
                affine_index(
                    *slot_base,
                    invocation_offset,
                    *invocation_stride,
                    lane_offset,
                    *lane_stride,
                    "Poseidon2 retained slot",
                )?,
            ),
            Self::Constant(coefficient) => {
                constant_form(logical_width, one_column, *coefficient, "Poseidon2 one column")
            }
            Self::External {
                block,
                slot_base,
                invocation_stride,
            } => block.external_form(
                logical_width,
                checked_add(
                    *slot_base,
                    checked_mul(invocation_offset, *invocation_stride, "Poseidon2 external slot")?,
                    "Poseidon2 external slot",
                )?,
                lane_offset,
            ),
            Self::TaggedRetained {
                block,
                tags,
                required,
                slot_base,
                invocation_stride,
                lane_stride,
            } => {
                let actual = tags
                    .get(invocation_offset)
                    .ok_or(PackageError::Invalid("Poseidon2 invocation tag table"))?;
                if actual == required {
                    block.form(
                        logical_width,
                        affine_index(
                            *slot_base,
                            invocation_offset,
                            *invocation_stride,
                            lane_offset,
                            *lane_stride,
                            "Poseidon2 tagged retained slot",
                        )?,
                    )
                } else {
                    Ok(Form::default())
                }
            }
            Self::OptionalConstant { values, lane_count } => {
                let index = checked_add(
                    checked_mul(invocation_offset, *lane_count, "Poseidon2 optional constant index")?,
                    lane_offset,
                    "Poseidon2 optional constant index",
                )?;
                match values
                    .get(index)
                    .ok_or(PackageError::Invalid("Poseidon2 optional constant table"))?
                {
                    None => Ok(Form::default()),
                    Some(coefficient) => constant_form(logical_width, one_column, *coefficient, "Poseidon2 one column"),
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Rule {
    region: Region,
    term: Term,
}

impl Rule {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "Poseidon2 input rule")?;
        Ok(Self {
            region: Region::decode(&fields[0])?,
            term: Term::decode(&fields[1])?,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct Program {
    rules: Vec<Rule>,
}

impl Program {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        Ok(Self {
            rules: decode_list(value, Rule::decode)?,
        })
    }

    fn form(
        &self,
        logical_width: usize,
        one_column: usize,
        invocation: usize,
        lane: usize,
    ) -> Result<Form, PackageError> {
        let mut accumulated = Form::default();
        for rule in &self.rules {
            if let Some((invocation_offset, lane_offset)) = rule.region.offsets(invocation, lane) {
                accumulated =
                    accumulated.append(
                        rule.term
                            .form(logical_width, one_column, invocation_offset, lane_offset)?,
                    );
            }
        }
        Ok(accumulated)
    }

    pub(super) fn state(
        &self,
        logical_width: usize,
        one_column: usize,
        invocation: usize,
    ) -> Result<[Form; 8], PackageError> {
        let lanes = (0..8)
            .map(|lane| self.form(logical_width, one_column, invocation, lane))
            .collect::<Result<Vec<_>, _>>()?;
        lanes
            .try_into()
            .map_err(|_| PackageError::Invalid("Poseidon2 input state"))
    }
}

fn decode_optional_constants(value: &Value) -> Result<Vec<Option<Goldilocks>>, PackageError> {
    array(value, "Poseidon2 optional constant table")?
        .iter()
        .map(|item| {
            let fields = array(item, "Poseidon2 optional constant")?;
            match fields {
                [tag] if tag.as_u64() == Some(0) => Ok(None),
                [tag, coefficient] if tag.as_u64() == Some(1) => {
                    field_atom(coefficient, "Poseidon2 optional constant").map(Some)
                }
                _ => Err(PackageError::Invalid("Poseidon2 optional constant")),
            }
        })
        .collect()
}

fn affine_index(
    base: usize,
    first: usize,
    first_stride: usize,
    second: usize,
    second_stride: usize,
    location: &'static str,
) -> Result<usize, PackageError> {
    checked_add(
        checked_add(base, checked_mul(first, first_stride, location)?, location)?,
        checked_mul(second, second_stride, location)?,
        location,
    )
}

fn constant_form(
    logical_width: usize,
    one_column: usize,
    coefficient: Goldilocks,
    location: &'static str,
) -> Result<Form, PackageError> {
    if one_column >= logical_width {
        return Err(PackageError::Invalid(location));
    }
    Ok(Form::singleton(one_column, coefficient))
}
