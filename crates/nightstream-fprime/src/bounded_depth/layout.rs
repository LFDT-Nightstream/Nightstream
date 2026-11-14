//! A step package's logical columns (Lean `Export/Generic/SlotLayout`,
//! `ProductionAssignment.logicalAssignment`): the 270 public columns, then
//! one low-norm slot for each source column in the package's runs. A field
//! slot holds the 41 balanced-ternary digits of its value; a bit or centered
//! slot holds the value itself. A source column without a slot is not a
//! logical column. Base runs follow `StepPackage.runsOf`. Native application
//! runs also use `BooleanInference` and the fresh norm with `PackedBoolean`.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use rayon::prelude::*;

use super::expand::mix;
use super::{Linear, Piece, Row, StepPackage};
use crate::package::{encode_field, PackageError};

const PUBLIC: usize = 270;
const FIELD_DIGITS: usize = 41;
const FIELD_RATIO: u64 = 3;
const NO_SLOT: u32 = u32::MAX;
const RING_DEGREE: usize = 54;
const SBOX_COLUMNS: usize = 150;
const BLOCK_OUTPUTS: usize = 1080;
const BLOCK_LOCALS: usize = 1096;

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

pub struct SlotSpans {
    pub starts: Vec<u32>,
    /// 0 for a source column without a slot.
    pub widths: Vec<u8>,
    /// The ratio of a field slot's digits.
    pub ratio: Goldilocks,
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

    /// Lean `SlotLayout.layout`.
    pub fn spans(&self) -> SlotSpans {
        let (starts, widths) = (0..self.offsets.len())
            .map(|column| {
                self.slot(column)
                    .map_or((0, 0), |(offset, kind)| (offset as u32, kind.width() as u8))
            })
            .unzip();
        SlotSpans {
            starts,
            widths,
            ratio: Goldilocks::from_u64(FIELD_RATIO),
        }
    }

    /// The honest logical assignment of the source columns `columns`.
    pub fn assignment(&self, columns: &[Goldilocks]) -> Result<Vec<Goldilocks>, PackageError> {
        if columns.len() < PUBLIC {
            return Err(PackageError::Input("bounded-depth source columns"));
        }
        let mut logical = vec![Goldilocks::ZERO; self.width];
        let (public, mut rest) = logical.split_at_mut(PUBLIC);
        public.copy_from_slice(&columns[..PUBLIC]);
        // The slots are in logical order, so each range of slots owns one
        // range of logical columns, and the ranges fill on all cores.
        let range = self
            .slots
            .len()
            .div_ceil(4 * rayon::current_num_threads())
            .max(1);
        let mut parts = Vec::new();
        for slots in self.slots.chunks(range) {
            let width: usize = slots.iter().map(|&column| self.kinds[column].width()).sum();
            let (part, tail) = rest.split_at_mut(width);
            parts.push((slots, part));
            rest = tail;
        }
        parts.into_par_iter().try_for_each(|(slots, part)| {
            let mut digits = [0i8; FIELD_DIGITS];
            let mut offset = 0;
            for &column in slots {
                let value = *columns
                    .get(column)
                    .ok_or(PackageError::Input("bounded-depth source columns"))?;
                let kind = self.kinds[column];
                match kind {
                    Kind::Bit | Kind::Centered => part[offset] = value,
                    Kind::Field => {
                        encode_field(value.as_canonical_u64(), &mut digits);
                        for (target, &digit) in part[offset..offset + FIELD_DIGITS].iter_mut().zip(&digits) {
                            *target = Goldilocks::from_i64(i64::from(digit));
                        }
                    }
                }
                offset += kind.width();
            }
            Ok::<_, PackageError>(())
        })?;
        Ok(logical)
    }

    pub fn source_assignment(&self, logical: &[Goldilocks]) -> Result<Vec<Goldilocks>, PackageError> {
        if logical.len() != self.width {
            return Err(PackageError::Input("bounded-depth logical assignment width"));
        }
        let mut columns = vec![Goldilocks::ZERO; self.offsets.len().max(PUBLIC)];
        columns[..PUBLIC].copy_from_slice(&logical[..PUBLIC]);
        for &column in &self.slots {
            let offset = self.offsets[column] as usize;
            let width = self.kinds[column].width();
            let mut coefficient = Goldilocks::ONE;
            let mut value = Goldilocks::ZERO;
            for &digit in &logical[offset..offset + width] {
                value += coefficient * digit;
                coefficient *= Goldilocks::from_u64(FIELD_RATIO);
            }
            columns[column] = value;
        }
        Ok(columns)
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

/// Lean `SlotLayout.forcedColumn?`: the column `c` that `row` forces to the
/// values 0 or 1. The row's terms read only `c` and the constant column 0,
/// and with `x` at `c` it is `(a₀ + a₁x)(b₀ + b₁x) = c₀ + c₁x` with
/// `a₁b₁ ≠ 0`, holding at `x = 0` and at `x = 1`.
pub(super) fn forced_column(row: &Row) -> Option<usize> {
    let sides = [&row.a, &row.b, &row.c];
    let terms = || sides.iter().flat_map(|side| side.terms.iter());
    let column = terms()
        .map(|&(column, _)| column)
        .find(|&column| column != 0)?;
    if !terms().all(|&(other, _)| other == 0 || other == column) {
        return None;
    }
    let at = |side: &Linear, target: usize| {
        side.terms
            .iter()
            .filter(|&&(other, _)| other == target)
            .fold(Goldilocks::ZERO, |sum, &(_, coefficient)| sum + coefficient)
    };
    let [low_a, low_b, low_c] = sides.map(|side| side.constant + at(side, 0));
    let [high_a, high_b, high_c] = sides.map(|side| at(side, column));
    (high_a * high_b != Goldilocks::ZERO
        && low_a * low_b == low_c
        && (low_a + high_a) * (low_b + high_b) == low_c + high_c)
        .then_some(column)
}

impl StepPackage {
    /// The columns of the package's `bit` hints (Lean `StepPackage.bitColumns`).
    pub(crate) fn bit_columns(&self) -> Vec<usize> {
        self.witness
            .iter()
            .flat_map(|batch| batch.bit_columns())
            .collect()
    }

    /// The slot runs (`StepPackage.runsOf`): a bit slot for each candidate in
    /// `bits` that a row forces or `inferred` names, a field slot for each
    /// other read column.
    pub(super) fn rule_slots(&self, bits: &[usize], inferred: &[usize]) -> Vec<(usize, usize, u8)> {
        const BIT: u8 = 0;
        const FIELD: u8 = 2;
        let end = self.source_end();
        let sbox_end = self.sbox_start + SBOX_COLUMNS * self.block_count();
        let mark = |mask: &mut Vec<bool>, columns: &mut dyn Iterator<Item = usize>| {
            for column in columns {
                if let Some(marked) = mask.get_mut(column) {
                    *marked = true;
                }
            }
        };
        let linear = |lanes: &[Linear]| {
            lanes
                .iter()
                .flat_map(|lane| lane.terms.iter().map(|&(column, _)| column))
                .collect::<Vec<_>>()
        };
        let mut reads = vec![false; end];
        let mut outputs = vec![false; end];
        let mut forced = vec![false; end];
        for piece in &self.pieces {
            match piece {
                Piece::Rows(rows) => {
                    mark(
                        &mut reads,
                        &mut rows
                            .iter()
                            .flat_map(|row| [&row.a, &row.b, &row.c])
                            .flat_map(|side| side.terms.iter().map(|&(column, _)| column)),
                    );
                    mark(&mut forced, &mut rows.iter().filter_map(forced_column));
                }
                Piece::Ring {
                    cells,
                    start,
                    quotient,
                    lanes,
                } => mark(
                    &mut reads,
                    &mut (*start..start + RING_DEGREE * cells)
                        .chain(*quotient..quotient + RING_DEGREE * cells)
                        .chain(linear(lanes)),
                ),
                Piece::Block { start, lanes } => {
                    mark(&mut reads, &mut linear(lanes).into_iter());
                    mark(&mut outputs, &mut (start + BLOCK_OUTPUTS..start + BLOCK_LOCALS));
                }
            }
        }
        let mut candidates = vec![false; end];
        mark(&mut candidates, &mut bits.iter().copied());
        mark(&mut forced, &mut inferred.iter().copied());
        let mut runs: Vec<(usize, usize, u8)> = Vec::new();
        for column in self.public_columns..end {
            if self.field_lowering.dead.contains(&column) {
                continue;
            }
            if !((reads[column] && !outputs[column]) || (self.sbox_start..sbox_end).contains(&column)) {
                continue;
            }
            let kind = if candidates[column] && forced[column] {
                BIT
            } else {
                FIELD
            };
            match runs.last_mut() {
                Some((start, count, last)) if *last == kind && *start + *count == column => *count += 1,
                _ => runs.push((column, 1, kind)),
            }
        }
        runs
    }

    /// The logical layout of the package's slot runs (`SlotLayout.layout`).
    pub(super) fn layout(&self) -> Result<LogicalLayout, PackageError> {
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
                return Err(PackageError::Malformed("bounded-depth slot over a public column"));
            }
            let kind = Kind::from_code(code);
            for column in start..start + count {
                if offsets[column] != NO_SLOT {
                    return Err(PackageError::Malformed("bounded-depth column with two slots"));
                }
                offsets[column] =
                    u32::try_from(width).map_err(|_| PackageError::TooLarge("bounded-depth logical width"))?;
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
            let (offset, kind) = self.slot(column).ok_or(PackageError::Malformed(
                "bounded-depth row reads a column without a slot",
            ))?;
            visit(Run {
                column: offset,
                count: kind.width(),
                coefficient,
                ratio: if kind == Kind::Field {
                    Goldilocks::from_u64(FIELD_RATIO)
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
