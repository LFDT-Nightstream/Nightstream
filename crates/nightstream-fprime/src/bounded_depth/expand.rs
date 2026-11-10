//! A step package's plan rows (`CircuitRelation.plan`): one row for each row
//! of `Package.relationRows`, then 150 S-box rows for each block, block-major.
//! A row has four port forms over the logical columns: A, B, C and the S-box
//! input. An R1CS row fills A, B and C, and its constant reads column 0.

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
/// `row` are `offsets[4 * row + port]..offsets[4 * row + port + 1]`. Zero
/// coefficients are not stored.
pub struct PlanRows {
    ordinary: usize,
    binding: usize,
    rings: Vec<Range<usize>>,
    offsets: Vec<usize>,
    columns: Vec<u32>,
    coefficients: Vec<Goldilocks>,
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

    pub fn coefficients(&self) -> &[Goldilocks] {
        &self.coefficients
    }

    /// The rows of each ring piece.
    pub fn ring_rows(&self) -> &[Range<usize>] {
        &self.rings
    }

    /// The blocks' binding rows, the last relation rows.
    pub fn binding_rows(&self) -> Range<usize> {
        self.binding..self.ordinary
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
    rows: PlanRows,
}

impl Builder {
    fn push(&mut self, column: usize, coefficient: Goldilocks) -> Result<(), PackageError> {
        if column >= self.width {
            return Err(PackageError::Malformed("bounded-depth plan column beyond the width"));
        }
        if coefficient != Goldilocks::ZERO {
            self.rows.columns.push(column as u32);
            self.rows.coefficients.push(coefficient);
        }
        Ok(())
    }

    fn end_port(&mut self) {
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
    /// Expands the package into its circuit's plan rows: the pieces' rows in
    /// order (a block contributes none), the blocks' binding rows, then the
    /// blocks' S-box rows from the shared formula library.
    pub fn plan_rows(&self) -> Result<PlanRows, PackageError> {
        let mut builder = Builder {
            width: self.width,
            rows: PlanRows {
                ordinary: 0,
                binding: 0,
                rings: Vec::new(),
                offsets: vec![0],
                columns: Vec::new(),
                coefficients: Vec::new(),
            },
        };
        let mut blocks = Vec::new();
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
                Piece::Block { start, lanes } => {
                    if lanes.len() != BLOCK_LANES {
                        return Err(PackageError::Malformed("bounded-depth block lane count"));
                    }
                    blocks.push((*start, lanes.as_slice()));
                }
            }
        }
        builder.rows.binding = builder.rows.row_count();
        for (index, &(start, _)) in blocks.iter().enumerate() {
            let sbox = self.sbox_start + index * SBOX_ROWS;
            let finals: Vec<usize> = (0..BLOCK_LANES)
                .map(|lane| sbox + FINAL_SBOX + lane)
                .collect();
            for (lane, output) in external_outputs(&finals, self.width)?
                .into_iter()
                .enumerate()
            {
                let mut a = vec![(start + BLOCK_OUTPUTS + lane, Goldilocks::ONE)];
                a.extend(
                    output
                        .into_iter()
                        .map(|(column, coefficient)| (column, -coefficient)),
                );
                builder.push_entries(&a)?;
                builder.push_entries(&[(0, Goldilocks::ONE)])?;
                builder.push_entries(&[])?;
                builder.end_port();
            }
        }
        builder.rows.ordinary = builder.rows.row_count();
        for (index, &(_, lanes)) in blocks.iter().enumerate() {
            let forms: Vec<Entries> = lanes
                .iter()
                .map(|lane| {
                    std::iter::once((0, lane.constant))
                        .chain(lane.terms.iter().copied())
                        .collect()
                })
                .collect();
            let rows = &mut builder;
            permutation_rows(&forms, self.sbox_start + index * SBOX_ROWS, self.width, |ports| {
                for port in &ports {
                    rows.push_entries(port)?;
                }
                Ok(())
            })?;
        }
        Ok(builder.rows)
    }
}
