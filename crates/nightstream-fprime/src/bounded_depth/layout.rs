//! Lean `Export/Generic/SlotLayout`, `ProductionAssignment.logicalAssignment`.

use neo_ccs::Mat;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::expand::mix;
use super::{Linear, Piece, Row, StepPackage};
use crate::package::{
    balanced_ternary_digits, PackageError, BLOCK_LOCALS, BLOCK_OUTPUTS, PUBLIC_WORDS as PUBLIC, RING_DEGREE,
    SBOX_ROWS as SBOX_COLUMNS,
};

const FIELD_DIGITS: usize = 41;
const FIELD_RATIO: u64 = 3;

/// Lean `LowNormSlot.Kind`.
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

pub struct LogicalLayout {
    spans: SlotSpans,
    run_starts: Box<[u32]>,
    centered: Vec<u64>,
    width: usize,
}

struct LayoutRun {
    column: usize,
    offset: usize,
    end: usize,
    width: usize,
}

impl LayoutRun {
    fn count(&self) -> usize {
        (self.end - self.offset) / self.width
    }
}

pub struct SlotSpans {
    pub starts: Vec<u32>,
    pub widths: Vec<u8>,
    pub ratio: Fq,
}

impl LogicalLayout {
    fn runs_from(&self, first: usize) -> impl Iterator<Item = LayoutRun> + '_ {
        let starts = &self.run_starts[first..];
        let ends = starts
            .iter()
            .skip(1)
            .map(|&column| self.spans.starts[column as usize] as usize);
        starts
            .iter()
            .zip(ends.chain(std::iter::once(self.width)))
            .map(|(&column, end)| {
                let column = column as usize;
                LayoutRun {
                    column,
                    offset: self.spans.starts[column] as usize,
                    end,
                    width: usize::from(self.spans.widths[column]),
                }
            })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn slot(&self, column: usize) -> Option<(usize, Kind)> {
        if column < PUBLIC {
            return Some((column, Kind::Centered));
        }
        let centered = || {
            self.centered
                .get(column / 64)
                .is_some_and(|word| word & (1 << (column % 64)) != 0)
        };
        let kind = match *self.spans.widths.get(column)? {
            0 => return None,
            1 if centered() => Kind::Centered,
            1 => Kind::Bit,
            _ => Kind::Field,
        };
        Some((self.spans.starts[column] as usize, kind))
    }

    /// Lean `SlotLayout.layout`.
    pub fn spans(&self) -> &SlotSpans {
        &self.spans
    }

    pub fn assignment(&self, columns: &[Fq]) -> Result<Vec<Fq>, PackageError> {
        if columns.len() < PUBLIC {
            return Err(PackageError::Input("bounded-depth source columns"));
        }
        let mut logical = vec![Fq::ZERO; self.width];
        logical[..PUBLIC].copy_from_slice(&columns[..PUBLIC]);
        for run in self.runs_from(0) {
            let first = run.column;
            let values = columns
                .get(first..first + run.count())
                .ok_or(PackageError::Input("bounded-depth source columns"))?;
            let digits = &mut logical[run.offset..run.end];
            match run.width {
                1 => digits.copy_from_slice(values),
                _ => {
                    digits
                        .par_chunks_exact_mut(FIELD_DIGITS)
                        .zip(values)
                        .with_min_len(1024)
                        .for_each(|(chunk, &value)| {
                            let mut trits = [0i8; FIELD_DIGITS];
                            balanced_ternary_digits(value.as_canonical_u64(), &mut trits);
                            for (target, digit) in chunk.iter_mut().zip(trits) {
                                *target = Fq::from_i64(i64::from(digit));
                            }
                        });
                }
            }
        }
        Ok(logical)
    }

    pub fn signed_assignment(&self, columns: &[Fq]) -> Result<Mat<Fq>, PackageError> {
        if columns.len() < PUBLIC {
            return Err(PackageError::Input("bounded-depth source columns"));
        }
        let blocks = self.width.div_ceil(RING_DEGREE);
        let task_blocks = blocks.div_ceil(4 * rayon::current_num_threads()).max(1);
        let (mut positive, mut negative) = (vec![0u64; blocks], vec![0u64; blocks]);
        positive
            .par_chunks_mut(task_blocks)
            .zip(negative.par_chunks_mut(task_blocks))
            .enumerate()
            .try_for_each(|(task, (positive, negative))| {
                let start = task * task_blocks * RING_DEGREE;
                let end = start + positive.len() * RING_DEGREE;
                let mut put = |offset: usize, (plus, minus): (u64, u64)| {
                    let dropped = start.saturating_sub(offset);
                    let offset = offset.saturating_sub(start);
                    let block = offset / RING_DEGREE;
                    let shift = offset % RING_DEGREE;
                    positive[block] |= (plus >> dropped) << shift;
                    negative[block] |= (minus >> dropped) << shift;
                    if shift != 0 && block + 1 < positive.len() {
                        positive[block + 1] |= (plus >> dropped) >> (RING_DEGREE - shift);
                        negative[block + 1] |= (minus >> dropped) >> (RING_DEGREE - shift);
                    }
                };
                for offset in start..end.min(PUBLIC) {
                    put(offset, unit_masks(columns[offset])?);
                }
                let first = self
                    .run_starts
                    .partition_point(|&column| self.spans.starts[column as usize] as usize <= start)
                    .saturating_sub(1);
                for run in self.runs_from(first).take_while(|run| run.offset < end) {
                    let width = run.width;
                    let first = start.saturating_sub(run.offset) / width;
                    let last = (end - run.offset).div_ceil(width).min(run.count());
                    let column = run.column;
                    let values = columns
                        .get(column + first..column + last)
                        .ok_or(PackageError::Input("bounded-depth source columns"))?;
                    let mut offset = run.offset + first * width;
                    match run.width {
                        1 => {
                            for &value in values {
                                put(offset, unit_masks(value)?);
                                offset += 1;
                            }
                        }
                        _ => {
                            for &value in values {
                                put(offset, balanced_ternary_masks(value.as_canonical_u64()));
                                offset += FIELD_DIGITS;
                            }
                        }
                    }
                }
                Ok::<_, PackageError>(())
            })?;
        Mat::from_owned_column_masks(RING_DEGREE, blocks, positive, negative).map_err(PackageError::Malformed)
    }

    pub fn source_assignment(&self, logical: &[Fq]) -> Result<Vec<Fq>, PackageError> {
        if logical.len() != self.width {
            return Err(PackageError::Input("bounded-depth logical assignment width"));
        }
        let mut columns = vec![Fq::ZERO; self.spans.starts.len().max(PUBLIC)];
        columns[..PUBLIC].copy_from_slice(&logical[..PUBLIC]);
        for run in self.runs_from(0) {
            let digits = &logical[run.offset..run.end];
            let start = run.column;
            let target = &mut columns[start..start + run.count()];
            for (value, digits) in target.iter_mut().zip(digits.chunks_exact(run.width)) {
                *value = digits
                    .iter()
                    .rev()
                    .fold(Fq::ZERO, |sum, &digit| sum * self.spans.ratio + digit);
            }
        }
        Ok(columns)
    }

    /// Lean `Parity.logicalSum`.
    pub fn checksum(&self, logical: &[Fq], limit: usize) -> u64 {
        let mut total = (0..PUBLIC).fold(Fq::ZERO, |sum, column| sum + logical[column] * mix(1, column as u64));
        for run in self.runs_from(0) {
            let count = limit.saturating_sub(run.column).min(run.count());
            for (index, &value) in logical[run.offset..run.offset + count * run.width]
                .iter()
                .enumerate()
            {
                total += value * mix(1, (run.offset + index) as u64);
            }
        }
        total.as_canonical_u64()
    }
}

const TRITS_PER_CHUNK: u32 = 6;
const TERNARY_RADIX: usize = 3usize.pow(TRITS_PER_CHUNK);
const TERNARY_MASKS: [[u8; 2]; TERNARY_RADIX] = {
    let mut table = [[0; 2]; TERNARY_RADIX];
    let mut index = 0;
    while index < TERNARY_RADIX {
        let mut value = index;
        let mut trit = 0;
        while trit < TRITS_PER_CHUNK {
            let digit = value % 3;
            if digit != 0 {
                table[index][digit - 1] |= 1 << trit;
            }
            value = value / 3 + (digit == 2) as usize;
            trit += 1;
        }
        index += 1;
    }
    table
};

fn balanced_ternary_masks(value: u64) -> (u64, u64) {
    let negative = value > Fq::ORDER_U64 / 2;
    let mut magnitude = if negative { Fq::ORDER_U64 - value } else { value };
    let mut positive = 0;
    let mut minus = 0;
    let mut shift = 0;
    while magnitude != 0 {
        let chunk = (magnitude % TERNARY_RADIX as u64) as usize;
        let [p, n] = TERNARY_MASKS[chunk];
        positive |= u64::from(p) << shift;
        minus |= u64::from(n) << shift;
        magnitude = magnitude / TERNARY_RADIX as u64 + u64::from(chunk > TERNARY_RADIX / 2);
        shift += TRITS_PER_CHUNK;
    }
    if negative {
        (minus, positive)
    } else {
        (positive, minus)
    }
}

fn unit_masks(value: Fq) -> Result<(u64, u64), PackageError> {
    match value {
        Fq::ZERO => Ok((0, 0)),
        Fq::ONE => Ok((1, 0)),
        Fq::NEG_ONE => Ok((0, 1)),
        _ => Err(PackageError::Input("a signed-unit column is outside {-1, 0, 1}")),
    }
}

/// Lean `SlotLayout.forcedColumn?`.
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
            .fold(Fq::ZERO, |sum, &(_, coefficient)| sum + coefficient)
    };
    let [low_a, low_b, low_c] = sides.map(|side| side.constant + at(side, 0));
    let [high_a, high_b, high_c] = sides.map(|side| at(side, column));
    (high_a * high_b != Fq::ZERO && low_a * low_b == low_c && (low_a + high_a) * (low_b + high_b) == low_c + high_c)
        .then_some(column)
}

impl StepPackage {
    /// Lean `StepPackage.bitColumns`.
    pub(crate) fn bit_columns(&self) -> Vec<usize> {
        self.witness
            .iter()
            .flat_map(|batch| batch.bit_columns())
            .collect()
    }

    /// Lean `StepPackage.runsOf`.
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

    /// Lean `SlotLayout.layout`.
    pub(super) fn layout(&self) -> Result<LogicalLayout, PackageError> {
        let size = self
            .slots
            .iter()
            .map(|&(start, count, _)| start + count)
            .max()
            .unwrap_or(PUBLIC);
        let mut starts = vec![0; size];
        let mut widths = vec![0; size];
        for (column, (start, width)) in starts.iter_mut().zip(&mut widths).take(PUBLIC).enumerate() {
            *start = column as u32;
            *width = 1;
        }
        let mut centered = Vec::new();
        let mut run_starts = Vec::new();
        let mut width = PUBLIC;
        for &(start, count, code) in &self.slots {
            if start < PUBLIC {
                return Err(PackageError::Malformed("bounded-depth slot over a public column"));
            }
            let kind = Kind::from_code(code);
            if count == 0 {
                continue;
            }
            let last = start + count - 1;
            u32::try_from(last).map_err(|_| PackageError::TooLarge("bounded-depth source column"))?;
            run_starts.push(start as u32);
            if kind == Kind::Centered && centered.len() <= last / 64 {
                centered.resize(last / 64 + 1, 0);
            }
            for column in start..start + count {
                if widths[column] != 0 {
                    return Err(PackageError::Malformed("bounded-depth column with two slots"));
                }
                starts[column] =
                    u32::try_from(width).map_err(|_| PackageError::TooLarge("bounded-depth logical width"))?;
                if kind == Kind::Centered {
                    centered[column / 64] |= 1u64 << (column % 64);
                }
                widths[column] = kind.width() as u8;
                width += kind.width();
            }
        }
        Ok(LogicalLayout {
            spans: SlotSpans {
                starts,
                widths,
                ratio: Fq::from_u64(FIELD_RATIO),
            },
            centered,
            run_starts: run_starts.into_boxed_slice(),
            width,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub column: usize,
    pub count: usize,
    pub coefficient: Fq,
    pub ratio: Fq,
}

impl LogicalLayout {
    /// Lean `Frame.Framed`.
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
                    Fq::from_u64(FIELD_RATIO)
                } else {
                    Fq::ONE
                },
            });
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        rows: &super::PlanRows,
        row: usize,
        port: usize,
        logical: &[Fq],
    ) -> Result<Fq, PackageError> {
        let mut total = Fq::ZERO;
        self.visit_runs(rows, row, port, |run| {
            total += run.coefficient * run_value(&logical[run.column..run.column + run.count], run.ratio);
        })?;
        Ok(total)
    }
}

fn run_value(digits: &[Fq], ratio: Fq) -> Fq {
    if ratio == Fq::from_u64(FIELD_RATIO) && digits.len() <= FIELD_DIGITS {
        let trits = digits.iter().rev().try_fold(0i128, |sum, &digit| {
            let trit = match digit {
                Fq::ZERO => 0,
                Fq::ONE => 1,
                Fq::NEG_ONE => -1,
                _ => return None,
            };
            Some(3 * sum + trit)
        });
        if let Some(sum) = trits {
            return Fq::from_i128(sum);
        }
    }
    digits
        .iter()
        .rev()
        .fold(Fq::ZERO, |sum, &digit| sum * ratio + digit)
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_signed_assignment.rs"]
mod signed_assignment_tests;
