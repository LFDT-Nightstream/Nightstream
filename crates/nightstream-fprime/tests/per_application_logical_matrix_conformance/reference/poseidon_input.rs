//! Independent interpreter for serialized Poseidon2 input-state rules.

use serde_json::Value;

use super::{
    array, checked_add, checked_mul, decode_list, exact_array, field, word, Field, Form, Result, RetainedBlock,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InvocationTag {
    Absorb,
    SqueezeFirst,
    SqueezeSecond,
}

impl InvocationTag {
    fn decode(value: &Value) -> Result<Self> {
        match word(value, "Poseidon2 invocation tag")? {
            0 => Ok(Self::Absorb),
            1 => Ok(Self::SqueezeFirst),
            2 => Ok(Self::SqueezeSecond),
            _ => Err("unknown Poseidon2 invocation tag".into()),
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
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 4, "Poseidon2 input region")?;
        Ok(Self {
            invocation_start: word(&fields[0], "Poseidon2 invocation start")?,
            invocation_count: word(&fields[1], "Poseidon2 invocation count")?,
            lane_start: word(&fields[2], "Poseidon2 lane start")?,
            lane_count: word(&fields[3], "Poseidon2 lane count")?,
        })
    }

    fn offsets(self, invocation: usize, lane: usize) -> Option<(usize, usize)> {
        let invocation = invocation.checked_sub(self.invocation_start)?;
        let lane = lane.checked_sub(self.lane_start)?;
        (invocation < self.invocation_count && lane < self.lane_count).then_some((invocation, lane))
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
    Constant(Field),
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
        values: Vec<Option<Field>>,
        lane_count: usize,
    },
}

impl Term {
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = array(value, "Poseidon2 input term")?;
        match fields.first().and_then(Value::as_u64) {
            Some(0) if fields.len() == 5 => {
                let block = RetainedBlock::decode(&fields[1])?;
                block.validate(logical_width)?;
                Ok(Self::Retained {
                    block,
                    slot_base: word(&fields[2], "Poseidon2 retained slot base")?,
                    invocation_stride: word(&fields[3], "Poseidon2 retained invocation stride")?,
                    lane_stride: word(&fields[4], "Poseidon2 retained lane stride")?,
                })
            }
            Some(1) if fields.len() == 2 => Ok(Self::Constant(field(&fields[1], "Poseidon2 input constant")?)),
            Some(2) if fields.len() == 4 => {
                let block = RetainedBlock::decode(&fields[1])?;
                block.validate(logical_width)?;
                Ok(Self::External {
                    block,
                    slot_base: word(&fields[2], "Poseidon2 external slot base")?,
                    invocation_stride: word(&fields[3], "Poseidon2 external invocation stride")?,
                })
            }
            Some(3) if fields.len() == 7 => {
                let block = RetainedBlock::decode(&fields[1])?;
                block.validate(logical_width)?;
                Ok(Self::TaggedRetained {
                    block,
                    tags: decode_list(&fields[2], InvocationTag::decode, "Poseidon2 invocation tags")?,
                    required: InvocationTag::decode(&fields[3])?,
                    slot_base: word(&fields[4], "Poseidon2 tagged slot base")?,
                    invocation_stride: word(&fields[5], "Poseidon2 tagged invocation stride")?,
                    lane_stride: word(&fields[6], "Poseidon2 tagged lane stride")?,
                })
            }
            Some(4) if fields.len() == 3 => Ok(Self::OptionalConstant {
                values: decode_optional_constants(&fields[1])?,
                lane_count: word(&fields[2], "Poseidon2 optional lane count")?,
            }),
            _ => Err("unknown Poseidon2 input term opcode".into()),
        }
    }

    fn form(&self, logical_width: usize, one_column: usize, invocation: usize, lane: usize) -> Result<Form> {
        match self {
            Self::Retained {
                block,
                slot_base,
                invocation_stride,
                lane_stride,
            } => block.form(
                logical_width,
                affine_index(*slot_base, invocation, *invocation_stride, lane, *lane_stride)?,
            ),
            Self::Constant(coefficient) => constant(logical_width, one_column, *coefficient),
            Self::External {
                block,
                slot_base,
                invocation_stride,
            } => block.external_form(
                logical_width,
                checked_add(
                    *slot_base,
                    checked_mul(invocation, *invocation_stride, "Poseidon2 external slot")?,
                    "Poseidon2 external slot",
                )?,
                lane,
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
                    .get(invocation)
                    .ok_or_else(|| "Poseidon2 invocation tag is out of range".to_string())?;
                if actual == required {
                    block.form(
                        logical_width,
                        affine_index(*slot_base, invocation, *invocation_stride, lane, *lane_stride)?,
                    )
                } else {
                    Ok(Form::default())
                }
            }
            Self::OptionalConstant { values, lane_count } => {
                let index = checked_add(
                    checked_mul(invocation, *lane_count, "Poseidon2 optional constant")?,
                    lane,
                    "Poseidon2 optional constant",
                )?;
                match values
                    .get(index)
                    .ok_or_else(|| "Poseidon2 optional constant is out of range".to_string())?
                {
                    Some(coefficient) => constant(logical_width, one_column, *coefficient),
                    None => Ok(Form::default()),
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
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 2, "Poseidon2 input rule")?;
        Ok(Self {
            region: Region::decode(&fields[0])?,
            term: Term::decode(&fields[1], logical_width)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    rules: Vec<Rule>,
}

impl Program {
    pub fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        Ok(Self {
            rules: decode_list(value, |rule| Rule::decode(rule, logical_width), "Poseidon2 input rules")?,
        })
    }

    pub fn state(&self, logical_width: usize, one_column: usize, invocation: usize) -> Result<[Form; 8]> {
        let mut state: [Form; 8] = std::array::from_fn(|_| Form::default());
        for lane in 0..8 {
            let mut form = Form::default();
            for rule in &self.rules {
                if let Some((invocation, lane)) = rule.region.offsets(invocation, lane) {
                    form = form.append(
                        rule.term
                            .form(logical_width, one_column, invocation, lane)?,
                    );
                }
            }
            state[lane] = form;
        }
        Ok(state)
    }
}

fn decode_optional_constants(value: &Value) -> Result<Vec<Option<Field>>> {
    array(value, "Poseidon2 optional constants")?
        .iter()
        .map(|item| {
            let fields = array(item, "Poseidon2 optional constant")?;
            match fields {
                [tag] if tag.as_u64() == Some(0) => Ok(None),
                [tag, coefficient] if tag.as_u64() == Some(1) => {
                    Ok(Some(field(coefficient, "Poseidon2 optional constant")?))
                }
                _ => Err("invalid Poseidon2 optional constant".into()),
            }
        })
        .collect()
}

fn affine_index(base: usize, first: usize, first_stride: usize, second: usize, second_stride: usize) -> Result<usize> {
    checked_add(
        checked_add(
            base,
            checked_mul(first, first_stride, "Poseidon2 input slot")?,
            "Poseidon2 input slot",
        )?,
        checked_mul(second, second_stride, "Poseidon2 input slot")?,
        "Poseidon2 input slot",
    )
}

fn constant(logical_width: usize, one_column: usize, coefficient: Field) -> Result<Form> {
    if one_column >= logical_width {
        return Err("Poseidon2 one column is out of range".into());
    }
    Ok(Form::singleton(one_column, coefficient))
}
