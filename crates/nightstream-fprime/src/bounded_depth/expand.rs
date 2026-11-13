//! A step package's plan rows (`CircuitRelation.plan`): one row for each row
//! of `Package.relationRows`, then 150 S-box rows for each block, block-major.
//! A row has four port forms over the logical columns: A, B, C and the S-box
//! input. An R1CS row fills A, B and C, and its constant reads column 0. A
//! port reads, in place of an output local of a block, the local's form over
//! the block's last 16 S-box columns (`CircuitRelation.outputForms`).

use std::collections::HashMap;
use std::ops::Range;

use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{Linear, Piece, Row, StepPackage};
use crate::package::{
    external_outputs, permutation_rows, Entries, PackageError, BLOCK_OUTPUTS, RING_DEGREE, RING_NODES, SBOX_ROWS,
};

const PORTS: usize = 4;
const BLOCK_LANES: usize = neo_ccs::crypto::poseidon2::WIDTH;
/// The S-box rows of the last full round, whose outputs the output forms read.
const FINAL_SBOX: usize = SBOX_ROWS - BLOCK_LANES;

/// Plan rows in compressed sparse form: the entries of port `port` of row
/// `row` are `offsets[4 * row + port]..offsets[4 * row + port + 1]`, in
/// column order with one entry for each column. Zero coefficients are not
/// stored.
pub struct PlanRows {
    ordinary: usize,
    rings: Vec<Range<usize>>,
    offsets: Vec<usize>,
    columns: Vec<u32>,
    coefficients: Vec<Fq>,
}

impl PlanRows {
    pub fn row_count(&self) -> usize {
        (self.offsets.len() - 1) / PORTS
    }

    /// The rows of `Package.relationRows`, before the S-box rows.
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

    pub fn coefficients(&self) -> &[Fq] {
        &self.coefficients
    }

    /// The rows of each ring piece.
    pub fn ring_rows(&self) -> &[Range<usize>] {
        &self.rings
    }

    pub fn sbox_rows(&self) -> Range<usize> {
        self.ordinary..self.row_count()
    }

    pub fn evaluate(&self, row: usize, port: usize, columns: &[Fq]) -> Fq {
        self.port(row, port)
            .fold(Fq::ZERO, |sum, (column, coefficient)| {
                sum + coefficient * columns[column]
            })
    }

    /// The row's constraint `A * B - C + S^7` on `columns`.
    pub fn residual(&self, row: usize, columns: &[Fq]) -> Fq {
        self.evaluate(row, 0, columns) * self.evaluate(row, 1, columns) - self.evaluate(row, 2, columns)
            + self.evaluate(row, 3, columns).exp_u64(7)
    }

    pub fn port(&self, row: usize, port: usize) -> impl Iterator<Item = (usize, Fq)> + '_ {
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
        let mut total = Fq::ZERO;
        for slot in 0..PORTS * self.ordinary {
            let unit = slot as u64;
            for index in self.offsets[slot]..self.offsets[slot + 1] {
                total += self.coefficients[index] * mix(unit, u64::from(self.columns[index]));
            }
        }
        total.as_canonical_u64()
    }
}

pub(super) fn mix(unit: u64, column: u64) -> Fq {
    let point = Fq::from_u64(unit * 16_777_259 + column + 1);
    point * point * point + Fq::from_u64(unit)
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
    fn push(&mut self, column: usize, coefficient: Fq) -> Result<(), PackageError> {
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
            if coefficient != Fq::ZERO {
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

    /// `PieceRows.ringRows`: for each cell and each node,
    /// `challenge * value = output - prior + (X^64 + 1) * quotient` at the node.
    fn push_ring(&mut self, cells: usize, start: usize, quotient: usize, lanes: &[Linear]) -> Result<(), PackageError> {
        if lanes.len() != RING_DEGREE + 2 * RING_DEGREE * cells {
            return Err(PackageError::Malformed("bounded-depth ring lane count"));
        }
        let value = |lane: usize, cell: usize| RING_DEGREE + lane * cells + cell;
        let prior = |lane: usize, cell: usize| RING_DEGREE + RING_DEGREE * cells + lane * cells + cell;
        for cell in 0..cells {
            for node in 0..RING_NODES {
                let point = Fq::from_u64(node as u64);
                let powers: Vec<Fq> = std::iter::successors(Some(Fq::ONE), |power| Some(*power * point))
                    .take(RING_DEGREE)
                    .collect();
                let modulus = point.exp_u64(RING_DEGREE as u64) + Fq::ONE;
                let evaluation = |lane_of: &dyn Fn(usize) -> Linear| {
                    (0..RING_DEGREE).fold(Linear::default(), |sum, lane| {
                        sum.plus(lane_of(lane).scale(powers[lane]))
                    })
                };
                let variable = |column: usize| Linear {
                    constant: Fq::ZERO,
                    terms: vec![(column, Fq::ONE)],
                };
                let a = evaluation(&|lane| lanes[lane].clone());
                let b = evaluation(&|lane| lanes[value(lane, cell)].clone());
                let c = evaluation(&|lane| variable(start + lane * cells + cell))
                    .plus(evaluation(&|lane| lanes[prior(lane, cell)].clone()).scale(-Fq::ONE))
                    .plus(evaluation(&|lane| variable(quotient + cell * RING_DEGREE + lane)).scale(modulus));
                self.push_row(&Row { a, b, c })?;
            }
        }
        Ok(())
    }

    fn push_entries(&mut self, entries: &[(usize, Fq)]) -> Result<(), PackageError> {
        for &(column, coefficient) in entries {
            self.push(column, coefficient)?;
        }
        self.end_port();
        Ok(())
    }
}

impl StepPackage {
    /// Expands the package into its circuit's plan rows
    /// (`Package.relationRows`): the pieces' rows in order (a block
    /// contributes none), then the blocks' S-box rows from the shared formula
    /// library. Each port reads the form of an output local in place of the
    /// local.
    pub fn plan_rows(&self) -> Result<PlanRows, PackageError> {
        self.rows(true)
    }

    /// The same rows on the source columns, where a port reads an output
    /// local itself.
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
        let mut forms = HashMap::new();
        if with_forms {
            for (index, &(start, _)) in blocks.iter().enumerate() {
                let sbox = self.sbox_start + index * SBOX_ROWS;
                let finals: Vec<usize> = (0..BLOCK_LANES)
                    .map(|lane| sbox + FINAL_SBOX + lane)
                    .collect();
                for (lane, form) in external_outputs(&finals, self.width)?
                    .into_iter()
                    .enumerate()
                {
                    forms.insert(start + BLOCK_OUTPUTS + lane, form);
                }
            }
        }
        let mut builder = Builder {
            width: self.width,
            forms,
            port: Vec::new(),
            rows: PlanRows {
                ordinary: 0,
                rings: Vec::new(),
                offsets: vec![0],
                columns: Vec::new(),
                coefficients: Vec::new(),
            },
        };
        for piece in &self.pieces {
            match piece {
                Piece::Rows(rows) => {
                    for row in rows {
                        builder.push_row(row)?;
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
        Ok(builder.rows)
    }
}
