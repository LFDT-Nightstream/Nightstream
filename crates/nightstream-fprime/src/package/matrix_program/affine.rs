//! Affine sparse-form rules used by multiplication-grid matrix blocks.

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::{
    checked_add, checked_mul, decode_list, exact_array, field_atom, usize_atom, Form, PackageError, RetainedBlock,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Coordinate {
    pub(super) major: usize,
    pub(super) middle: usize,
    pub(super) minor: usize,
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
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 6, "affine-grid region")?;
        Ok(Self {
            major_start: usize_atom(&fields[0], "affine major start")?,
            major_count: usize_atom(&fields[1], "affine major count")?,
            middle_start: usize_atom(&fields[2], "affine middle start")?,
            middle_count: usize_atom(&fields[3], "affine middle count")?,
            minor_start: usize_atom(&fields[4], "affine minor start")?,
            minor_count: usize_atom(&fields[5], "affine minor count")?,
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
        coefficient: Goldilocks,
    },
    Constant(Goldilocks),
}

impl Term {
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = super::array(value, "affine-grid term")?;
        match fields.first().and_then(Value::as_u64) {
            Some(0) if fields.len() == 7 => Ok(Self::Retained {
                block: RetainedBlock::decode(&fields[1])?,
                slot_base: usize_atom(&fields[2], "affine slot base")?,
                major_stride: usize_atom(&fields[3], "affine major stride")?,
                middle_stride: usize_atom(&fields[4], "affine middle stride")?,
                minor_stride: usize_atom(&fields[5], "affine minor stride")?,
                coefficient: field_atom(&fields[6], "affine coefficient")?,
            }),
            Some(1) if fields.len() == 2 => Ok(Self::Constant(field_atom(&fields[1], "affine constant")?)),
            _ => Err(PackageError::Invalid("affine-grid term")),
        }
    }

    fn form(&self, logical_width: usize, one_column: usize, offsets: Coordinate) -> Result<Form, PackageError> {
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
                            checked_mul(offsets.major, *major_stride, "affine retained slot")?,
                            "affine retained slot",
                        )?,
                        checked_mul(offsets.middle, *middle_stride, "affine retained slot")?,
                        "affine retained slot",
                    )?,
                    checked_mul(offsets.minor, *minor_stride, "affine retained slot")?,
                    "affine retained slot",
                )?;
                let form = block.form(logical_width, slot)?;
                if coefficient.as_canonical_u64() == 1 {
                    Ok(form)
                } else {
                    Ok(form.scaled(*coefficient))
                }
            }
            Self::Constant(coefficient) => {
                if one_column >= logical_width {
                    return Err(PackageError::Invalid("affine one column"));
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
    fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 2, "affine-grid rule")?;
        Ok(Self {
            region: Region::decode(&fields[0])?,
            term: Term::decode(&fields[1])?,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct AffineProgram {
    rules: Vec<Rule>,
}

impl AffineProgram {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        Ok(Self {
            rules: decode_list(value, Rule::decode)?,
        })
    }

    pub(super) fn form(
        &self,
        logical_width: usize,
        one_column: usize,
        coordinate: Coordinate,
    ) -> Result<Form, PackageError> {
        let mut accumulated = Form::default();
        for rule in &self.rules {
            if let Some(offsets) = rule.region.offsets(coordinate) {
                accumulated = accumulated.append(rule.term.form(logical_width, one_column, offsets)?);
            }
        }
        Ok(accumulated)
    }
}
