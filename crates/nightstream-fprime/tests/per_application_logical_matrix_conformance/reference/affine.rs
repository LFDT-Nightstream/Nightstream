//! Independent affine sparse-form rules for multiplication blocks.

use serde_json::Value;

use super::{
    array, checked_add, checked_mul, decode_list, exact_array, field, word, Field, Form, Result, RetainedBlock,
};

#[derive(Clone, Copy, Debug)]
pub struct Coordinate {
    pub major: usize,
    pub middle: usize,
    pub minor: usize,
}

#[derive(Clone, Copy, Debug)]
struct Region {
    major_start: usize,
    major_count: usize,
    middle_start: usize,
    middle_count: usize,
    minor_start: usize,
    minor_count: usize,
}

impl Region {
    fn decode(value: &Value) -> Result<Self> {
        let fields = exact_array(value, 6, "affine region")?;
        Ok(Self {
            major_start: word(&fields[0], "affine major start")?,
            major_count: word(&fields[1], "affine major count")?,
            middle_start: word(&fields[2], "affine middle start")?,
            middle_count: word(&fields[3], "affine middle count")?,
            minor_start: word(&fields[4], "affine minor start")?,
            minor_count: word(&fields[5], "affine minor count")?,
        })
    }

    fn offsets(self, coordinate: Coordinate) -> Option<Coordinate> {
        let major = coordinate.major.checked_sub(self.major_start)?;
        let middle = coordinate.middle.checked_sub(self.middle_start)?;
        let minor = coordinate.minor.checked_sub(self.minor_start)?;
        (major < self.major_count && middle < self.middle_count && minor < self.minor_count).then_some(Coordinate {
            major,
            middle,
            minor,
        })
    }
}

#[derive(Clone, Debug)]
enum Term {
    Retained {
        block: RetainedBlock,
        slot_base: usize,
        major_stride: usize,
        middle_stride: usize,
        minor_stride: usize,
        coefficient: Field,
    },
    Constant(Field),
}

impl Term {
    fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = array(value, "affine term")?;
        match fields.first().and_then(Value::as_u64) {
            Some(0) if fields.len() == 7 => {
                let block = RetainedBlock::decode(&fields[1])?;
                block.validate(logical_width)?;
                Ok(Self::Retained {
                    block,
                    slot_base: word(&fields[2], "affine slot base")?,
                    major_stride: word(&fields[3], "affine major stride")?,
                    middle_stride: word(&fields[4], "affine middle stride")?,
                    minor_stride: word(&fields[5], "affine minor stride")?,
                    coefficient: field(&fields[6], "affine coefficient")?,
                })
            }
            Some(1) if fields.len() == 2 => Ok(Self::Constant(field(&fields[1], "affine constant")?)),
            _ => Err("unknown affine term opcode".into()),
        }
    }

    fn form(&self, logical_width: usize, one_column: usize, offsets: Coordinate) -> Result<Form> {
        match self {
            Self::Retained {
                block,
                slot_base,
                major_stride,
                middle_stride,
                minor_stride,
                coefficient,
            } => {
                let slot = checked_add(
                    checked_add(
                        checked_add(
                            *slot_base,
                            checked_mul(offsets.major, *major_stride, "affine major slot")?,
                            "affine slot",
                        )?,
                        checked_mul(offsets.middle, *middle_stride, "affine middle slot")?,
                        "affine slot",
                    )?,
                    checked_mul(offsets.minor, *minor_stride, "affine minor slot")?,
                    "affine slot",
                )?;
                Ok(block.form(logical_width, slot)?.scaled(*coefficient))
            }
            Self::Constant(coefficient) => {
                if one_column >= logical_width {
                    return Err("affine one column is out of range".into());
                }
                Ok(Form::singleton(one_column, *coefficient))
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
        let fields = exact_array(value, 2, "affine rule")?;
        Ok(Self {
            region: Region::decode(&fields[0])?,
            term: Term::decode(&fields[1], logical_width)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct AffineProgram {
    rules: Vec<Rule>,
}

impl AffineProgram {
    pub fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        Ok(Self {
            rules: decode_list(value, |rule| Rule::decode(rule, logical_width), "affine rules")?,
        })
    }

    pub fn form(&self, logical_width: usize, one_column: usize, coordinate: Coordinate) -> Result<Form> {
        let mut result = Form::default();
        for rule in &self.rules {
            if let Some(offsets) = rule.region.offsets(coordinate) {
                result = result.append(rule.term.form(logical_width, one_column, offsets)?);
            }
        }
        Ok(result)
    }
}
