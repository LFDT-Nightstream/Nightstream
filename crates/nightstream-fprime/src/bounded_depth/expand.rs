//! A step package's plan rows (`CircuitRelation.plan`): one row for each row
//! of `Package.relationRows`, then 150 S-box rows for each block, block-major.
//! Native applications use the experimental `NativePower` rewrite.
//! The source view retains the original application rows.
//! A row has four port forms over the logical columns: A, B, C and the S-box
//! input. An R1CS row fills A, B and C, and its constant reads column 0. A
//! port reads, in place of an output local of a block, the local's form over
//! the block's last 16 S-box columns (`CircuitRelation.outputForms`).

use std::collections::HashMap;
use std::ops::Range;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::{Linear, Piece, Row, StepPackage};
use crate::package::{external_outputs, permutation_rows, Entries, PackageError};

const PORTS: usize = 4;
const RING_DEGREE: usize = 54;
const RING_MIDDLE_DEGREE: u64 = 27;
const RING_NODES: usize = 108;
const BLOCK_LANES: usize = 16;
const SBOX_ROWS: usize = 150;
const BLOCK_OUTPUTS: usize = 1080;
const FINAL_SBOX: usize = 134;

/// Plan rows in compressed sparse form: the entries of port `port` of row
/// `row` are `offsets[4 * row + port]..offsets[4 * row + port + 1]`, in
/// column order with one entry for each column. Zero coefficients are not
/// stored.
pub struct PlanRows {
    ordinary: usize,
    piece_ranges: Vec<Range<usize>>,
    rings: Vec<Range<usize>>,
    offsets: Vec<usize>,
    columns: Vec<u32>,
    coefficients: Vec<Goldilocks>,
}

impl PlanRows {
    /// Drops the spare capacity that the push-grown vectors keep.
    fn shrink_to_fit(&mut self) {
        self.offsets.shrink_to_fit();
        self.columns.shrink_to_fit();
        self.coefficients.shrink_to_fit();
    }

    pub fn piece_ranges(&self) -> &[Range<usize>] {
        &self.piece_ranges
    }

    pub fn row_count(&self) -> usize {
        (self.offsets.len() - 1) / PORTS
    }

    /// The source-piece rows before the native block S-box rows.
    pub fn ordinary_row_count(&self) -> usize {
        self.ordinary
    }

    pub fn entry_count(&self) -> usize {
        self.columns.len()
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn columns(&self) -> &[u32] {
        &self.columns
    }

    pub fn coefficients(&self) -> &[Goldilocks] {
        &self.coefficients
    }

    /// The rows of each ring piece.
    pub fn ring_rows(&self) -> &[Range<usize>] {
        &self.rings
    }

    pub fn sbox_rows(&self) -> Range<usize> {
        self.ordinary..self.row_count()
    }

    pub fn evaluate(&self, row: usize, port: usize, columns: &[Goldilocks]) -> Goldilocks {
        self.port(row, port)
            .fold(Goldilocks::ZERO, |sum, (column, coefficient)| {
                sum + coefficient * columns[column]
            })
    }

    /// The row's constraint `A * B - C + S^7` on `columns`.
    pub fn residual(&self, row: usize, columns: &[Goldilocks]) -> Goldilocks {
        self.evaluate(row, 0, columns) * self.evaluate(row, 1, columns) - self.evaluate(row, 2, columns)
            + self.evaluate(row, 3, columns).exp_u64(7)
    }

    pub fn port(&self, row: usize, port: usize) -> impl Iterator<Item = (usize, Goldilocks)> + '_ {
        let slot = PORTS * row + port;
        let range = self.offsets[slot]..self.offsets[slot + 1];
        self.columns[range.clone()]
            .iter()
            .zip(&self.coefficients[range])
            .map(|(&column, &coefficient)| (column as usize, coefficient))
    }

    /// Lean `Parity.summary`'s checksum of the relation rows: the sum over
    /// every entry of its coefficient times `mix(4 * row + port, column)`.
    pub fn ordinary_checksum(&self) -> u64 {
        let mut total = Goldilocks::ZERO;
        for slot in 0..PORTS * self.ordinary {
            let unit = slot as u64;
            for index in self.offsets[slot]..self.offsets[slot + 1] {
                total += self.coefficients[index] * mix(unit, u64::from(self.columns[index]));
            }
        }
        total.as_canonical_u64()
    }
}

pub(super) fn mix(unit: u64, column: u64) -> Goldilocks {
    let point = Goldilocks::from_u64(unit * 16_777_259 + column + 1);
    point * point * point + Goldilocks::from_u64(unit)
}

struct Builder {
    width: usize,
    /// `CircuitRelation.outputForms`: the form of each output local.
    forms: HashMap<usize, Entries>,
    /// The entries of the port that is open.
    port: Entries,
    rows: PlanRows,
}

impl Builder {
    /// Adds a term to the open port (`Substitute.term`).
    fn push(&mut self, column: usize, coefficient: Goldilocks) -> Result<(), PackageError> {
        if column >= self.width {
            return Err(PackageError::Malformed("bounded-depth plan column beyond the width"));
        }
        match self.forms.get(&column) {
            Some(form) => self.port.extend(
                form.iter()
                    .map(|&(column, factor)| (column, factor * coefficient)),
            ),
            None => self.port.push((column, coefficient)),
        }
        Ok(())
    }

    /// Closes the open port: one entry for each column, in column order.
    fn end_port(&mut self) {
        self.port.sort_by_key(|&(column, _)| column);
        let mut entries = self.port.drain(..).peekable();
        while let Some((column, mut coefficient)) = entries.next() {
            while let Some((_, more)) = entries.next_if(|&(next, _)| next == column) {
                coefficient += more;
            }
            if coefficient != Goldilocks::ZERO {
                self.rows.columns.push(column as u32);
                self.rows.coefficients.push(coefficient);
            }
        }
        self.rows.offsets.push(self.rows.columns.len());
    }

    fn push_linear(&mut self, linear: &Linear) -> Result<(), PackageError> {
        self.push(0, linear.constant)?;
        for &(column, coefficient) in &linear.terms {
            self.push(column, coefficient)?;
        }
        self.end_port();
        Ok(())
    }

    fn push_row(&mut self, row: &Row) -> Result<(), PackageError> {
        self.push_linear(&row.a)?;
        self.push_linear(&row.b)?;
        self.push_linear(&row.c)?;
        self.end_port();
        Ok(())
    }

    fn push_power(&mut self, input: &Linear, output: usize) -> Result<(), PackageError> {
        self.end_port();
        self.end_port();
        self.push(output, Goldilocks::ONE)?;
        self.end_port();
        self.push_linear(input)
    }

    /// `PieceRows.ringRows`: for each cell and each of the 108 nodes,
    /// `challenge * value = output - prior + Φ81 * quotient` at the node.
    fn push_ring(&mut self, cells: usize, start: usize, quotient: usize, lanes: &[Linear]) -> Result<(), PackageError> {
        if lanes.len() != RING_DEGREE + 2 * RING_DEGREE * cells {
            return Err(PackageError::Malformed("bounded-depth ring lane count"));
        }
        let value = |lane: usize, cell: usize| RING_DEGREE + lane * cells + cell;
        let prior = |lane: usize, cell: usize| RING_DEGREE + RING_DEGREE * cells + lane * cells + cell;
        for cell in 0..cells {
            for node in 0..RING_NODES {
                let point = Goldilocks::from_u64(node as u64);
                let powers: Vec<Goldilocks> =
                    std::iter::successors(Some(Goldilocks::ONE), |power| Some(*power * point))
                        .take(RING_DEGREE)
                        .collect();
                let modulus = point.exp_u64(RING_DEGREE as u64) + point.exp_u64(RING_MIDDLE_DEGREE) + Goldilocks::ONE;
                let evaluation = |lane_of: &dyn Fn(usize) -> Linear| {
                    (0..RING_DEGREE).fold(Linear::default(), |sum, lane| {
                        sum.plus(lane_of(lane).scale(powers[lane]))
                    })
                };
                let variable = |column: usize| Linear {
                    constant: Goldilocks::ZERO,
                    terms: vec![(column, Goldilocks::ONE)],
                };
                let a = evaluation(&|lane| lanes[lane].clone());
                let b = evaluation(&|lane| lanes[value(lane, cell)].clone());
                let c = evaluation(&|lane| variable(start + lane * cells + cell))
                    .plus(evaluation(&|lane| lanes[prior(lane, cell)].clone()).scale(-Goldilocks::ONE))
                    .plus(evaluation(&|lane| variable(quotient + cell * RING_DEGREE + lane)).scale(modulus));
                self.push_row(&Row { a, b, c })?;
            }
        }
        Ok(())
    }

    fn push_entries(&mut self, entries: &[(usize, Goldilocks)]) -> Result<(), PackageError> {
        for &(column, coefficient) in entries {
            self.push(column, coefficient)?;
        }
        self.end_port();
        Ok(())
    }
}

impl StepPackage {
    pub(super) fn output_forms(&self) -> Result<HashMap<usize, Entries>, PackageError> {
        let mut forms = HashMap::new();
        for (index, block) in self.blocks().enumerate() {
            let sbox = self.sbox_start + index * SBOX_ROWS;
            let finals = (0..BLOCK_LANES)
                .map(|lane| sbox + FINAL_SBOX + lane)
                .collect::<Vec<_>>();
            for (lane, form) in external_outputs(&finals, self.width)?
                .into_iter()
                .enumerate()
            {
                forms.insert(block.start() + BLOCK_OUTPUTS + lane, form);
            }
        }
        Ok(forms)
    }

    pub(super) fn resolved_forms(&self) -> Result<HashMap<usize, Entries>, PackageError> {
        let mut forms = self.output_forms()?;
        for (&column, (_, linear)) in &self.field_lowering.affine {
            let mut entries = vec![(0, linear.constant)];
            for &(input, coefficient) in &linear.terms {
                if let Some(form) = forms.get(&input) {
                    entries.extend(
                        form.iter()
                            .map(|&(source, value)| (source, value * coefficient)),
                    );
                } else {
                    entries.push((input, coefficient));
                }
            }
            forms.insert(column, entries);
        }
        Ok(forms)
    }

    /// Expands the package into its circuit's plan rows
    /// (`Package.relationRows`): the pieces' rows in order (a block
    /// contributes none), then the blocks' S-box rows from the shared formula
    /// library. Native applications use the experimental `NativePower`
    /// rewrite. Each port substitutes native output and affine forms.
    pub(super) fn plan_rows(&self) -> Result<PlanRows, PackageError> {
        self.rows(true)
    }

    /// The original source rows before the native rewrites, where
    /// a port reads an output local itself. The final program checks these
    /// rows and each block as a permutation (`FinishRelation.package_finish`).
    pub fn source_rows(&self) -> Result<PlanRows, PackageError> {
        self.rows(false)
    }

    fn rows(&self, with_forms: bool) -> Result<PlanRows, PackageError> {
        let mut blocks = Vec::new();
        for piece in &self.pieces {
            if let Piece::Block { start, lanes } = piece {
                if lanes.len() != BLOCK_LANES {
                    return Err(PackageError::Malformed("bounded-depth block lane count"));
                }
                blocks.push((*start, lanes.as_slice()));
            }
        }
        let forms = if with_forms {
            self.resolved_forms()?
        } else {
            HashMap::new()
        };
        let mut builder = Builder {
            width: self.width,
            forms,
            port: Vec::new(),
            rows: PlanRows {
                ordinary: 0,
                piece_ranges: Vec::new(),
                rings: Vec::new(),
                offsets: vec![0],
                columns: Vec::new(),
                coefficients: Vec::new(),
            },
        };
        let word_edits = self.word_packing.edits();
        for (index, piece) in self.pieces.iter().enumerate() {
            let first = builder.rows.row_count();
            match piece {
                Piece::Rows(rows) => {
                    if index == 0 && !with_forms && self.native.is_some() {
                        for row in self.packed_booleans.original_rows(rows)? {
                            builder.push_row(row)?;
                        }
                    } else {
                        for (row_index, row) in rows.iter().enumerate() {
                            if index == 0 && with_forms {
                                if let Some(replacements) = word_edits.get(&row_index) {
                                    for replacement in replacements {
                                        builder.push_row(replacement)?;
                                    }
                                    continue;
                                }
                                if let Some(power) = self.field_lowering.powers.get(&row_index) {
                                    builder.push_power(&power.input, power.powers[3])?;
                                }
                                if self.field_lowering.removed.contains(&row_index)
                                    || self.guard_fusion.removed.contains(&row_index)
                                {
                                    continue;
                                }
                            }
                            let row = if index == 0 && with_forms {
                                self.guard_fusion
                                    .replacements
                                    .get(&row_index)
                                    .unwrap_or(row)
                            } else {
                                row
                            };
                            builder.push_row(row)?;
                        }
                    }
                }
                Piece::Ring {
                    cells,
                    start,
                    quotient,
                    lanes,
                } => {
                    let first = builder.rows.row_count();
                    builder.push_ring(*cells, *start, *quotient, lanes)?;
                    builder.rows.rings.push(first..builder.rows.row_count());
                }
                Piece::Block { .. } => {}
            }
            builder
                .rows
                .piece_ranges
                .push(first..builder.rows.row_count());
        }
        builder.rows.ordinary = builder.rows.row_count();
        for (index, &(_, lanes)) in blocks.iter().enumerate() {
            // A lane's entries pass through a port, so they carry the output
            // forms with one entry for each column.
            let mut lane_forms: Vec<Entries> = Vec::with_capacity(BLOCK_LANES);
            for lane in lanes {
                builder.push_linear(lane)?;
                let end = builder.rows.columns.len();
                let begin = builder.rows.offsets[builder.rows.offsets.len() - 2];
                lane_forms.push(
                    builder.rows.columns[begin..end]
                        .iter()
                        .zip(&builder.rows.coefficients[begin..end])
                        .map(|(&column, &coefficient)| (column as usize, coefficient))
                        .collect(),
                );
                builder.rows.columns.truncate(begin);
                builder.rows.coefficients.truncate(begin);
                builder.rows.offsets.pop();
            }
            let rows = &mut builder;
            permutation_rows(&lane_forms, self.sbox_start + index * SBOX_ROWS, self.width, |ports| {
                for port in &ports {
                    rows.push_entries(port)?;
                }
                Ok(())
            })?;
        }
        builder.rows.shrink_to_fit();
        Ok(builder.rows)
    }
}
