//! A step package's logical columns (Lean `Export/Generic/SlotLayout`,
//! `ProductionAssignment.logicalAssignment`): the 270 public columns, then
//! one low-norm slot for each live source column in the package's runs. A
//! field slot holds the 41 balanced-ternary digits of its value; a bit or
//! centered slot holds the value itself. A source column without a slot is
//! not a logical column.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::expand::mix;
use super::StepPackage;
use crate::package::{encode_field, PackageError};

const PUBLIC: usize = 270;
const FIELD_DIGITS: usize = 41;
const NO_SLOT: u32 = u32::MAX;

/// A slot's kind (`LowNormSlot.Kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Bit,
    Centered,
    Field,
}

impl Kind {
    fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Bit,
            1 => Self::Centered,
            _ => Self::Field,
        }
    }

    pub fn width(self) -> usize {
        match self {
            Self::Bit | Self::Centered => 1,
            Self::Field => FIELD_DIGITS,
        }
    }
}

/// Where each source column lives among the logical columns.
pub struct LogicalLayout {
    offsets: Vec<u32>,
    kinds: Vec<Kind>,
    slots: Vec<usize>,
    width: usize,
}

impl LogicalLayout {
    /// The number of logical columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The first logical column of `column` and its kind; the public columns
    /// are themselves, and a column without a slot is `None`.
    pub fn slot(&self, column: usize) -> Option<(usize, Kind)> {
        if column < PUBLIC {
            return Some((column, Kind::Centered));
        }
        let offset = *self.offsets.get(column)?;
        (offset != NO_SLOT).then(|| (offset as usize, self.kinds[column]))
    }

    /// The honest logical assignment of the source columns `columns`.
    pub fn assignment(&self, columns: &[Goldilocks]) -> Result<Vec<Goldilocks>, PackageError> {
        if columns.len() < PUBLIC {
            return Err(PackageError::Input("option-B source columns"));
        }
        let mut logical = vec![Goldilocks::ZERO; self.width];
        logical[..PUBLIC].copy_from_slice(&columns[..PUBLIC]);
        let mut digits = [0i8; FIELD_DIGITS];
        for &column in &self.slots {
            let offset = self.offsets[column] as usize;
            let value = *columns
                .get(column)
                .ok_or(PackageError::Input("option-B source columns"))?;
            match self.kinds[column] {
                Kind::Bit | Kind::Centered => logical[offset] = value,
                Kind::Field => {
                    encode_field(value.as_canonical_u64(), &mut digits);
                    for (target, &digit) in logical[offset..offset + FIELD_DIGITS]
                        .iter_mut()
                        .zip(&digits)
                    {
                        *target = Goldilocks::from_i64(i64::from(digit));
                    }
                }
            }
        }
        Ok(logical)
    }

    /// Lean `Parity.logicalSum`: the public columns and the digits of each
    /// slot below `limit`, each times `mix(1, logical column)`.
    pub fn checksum(&self, logical: &[Goldilocks], limit: usize) -> u64 {
        let mut total = (0..PUBLIC).fold(Goldilocks::ZERO, |sum, column| {
            sum + logical[column] * mix(1, column as u64)
        });
        for &column in self.slots.iter().filter(|&&column| column < limit) {
            let offset = self.offsets[column] as usize;
            for index in offset..offset + self.kinds[column].width() {
                total += logical[index] * mix(1, index as u64);
            }
        }
        total.as_canonical_u64()
    }
}

impl StepPackage {
    /// The logical layout of the package's slot runs (`SlotLayout.layout`).
    pub fn layout(&self) -> Result<LogicalLayout, PackageError> {
        let size = self
            .slots
            .iter()
            .map(|&(start, count, _)| start + count)
            .max()
            .unwrap_or(PUBLIC);
        let mut offsets = vec![NO_SLOT; size];
        let mut kinds = vec![Kind::Field; size];
        let mut slots = Vec::new();
        let mut width = PUBLIC;
        for &(start, count, code) in &self.slots {
            if start < PUBLIC {
                return Err(PackageError::Malformed("option-B slot over a public column"));
            }
            let kind = Kind::from_code(code);
            for column in start..start + count {
                if offsets[column] != NO_SLOT {
                    return Err(PackageError::Malformed("option-B column with two slots"));
                }
                offsets[column] = u32::try_from(width).map_err(|_| PackageError::TooLarge("option-B logical width"))?;
                kinds[column] = kind;
                slots.push(column);
                width += kind.width();
            }
        }
        Ok(LogicalLayout {
            offsets,
            kinds,
            slots,
            width,
        })
    }
}

/// A geometric run of a logical row: `count` columns from `column`, with
/// coefficients `coefficient · ratio^i`. A field slot's 41 digits recompose
/// its value with ratio 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub column: usize,
    pub count: usize,
    pub coefficient: Goldilocks,
    pub ratio: Goldilocks,
}

impl LogicalLayout {
    /// Calls `visit` with each run of port `port` of plan row `row` read
    /// through the layout. A row that reads a column without a slot is
    /// malformed (`Frame.Framed` keeps every row off the dead locals).
    pub fn visit_runs(
        &self,
        rows: &super::PlanRows,
        row: usize,
        port: usize,
        mut visit: impl FnMut(Run),
    ) -> Result<(), PackageError> {
        for (column, coefficient) in rows.port(row, port) {
            let (offset, kind) = self
                .slot(column)
                .ok_or(PackageError::Malformed("option-B row reads a column without a slot"))?;
            visit(Run {
                column: offset,
                count: kind.width(),
                coefficient,
                ratio: if kind == Kind::Field {
                    Goldilocks::from_u64(3)
                } else {
                    Goldilocks::ONE
                },
            });
        }
        Ok(())
    }

    /// Port `port` of plan row `row` on the logical assignment `logical`.
    pub fn evaluate(
        &self,
        rows: &super::PlanRows,
        row: usize,
        port: usize,
        logical: &[Goldilocks],
    ) -> Result<Goldilocks, PackageError> {
        let mut total = Goldilocks::ZERO;
        self.visit_runs(rows, row, port, |run| {
            let mut coefficient = run.coefficient;
            for value in &logical[run.column..run.column + run.count] {
                total += coefficient * *value;
                coefficient *= run.ratio;
            }
        })?;
        Ok(total)
    }
}
