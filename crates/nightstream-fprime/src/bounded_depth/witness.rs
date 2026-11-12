//! A step package's witness program (`StepPackage.Package`): from the shared
//! columns, the circuit's locals in column order (the witness batches and the
//! native ranges), then the fresh columns of the relation piece by piece.

use neo_ccs::crypto::poseidon2_goldilocks::permute_state;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::expand::{mix, PlanRows};
use super::{Linear, Piece, StepPackage};
use crate::package::PackageError;
use crate::witness::{execute_witness_batch, WitnessBatch};

const RING_DEGREE: usize = 54;
const BLOCK_LOCALS: usize = 1096;
const BLOCK_OUTPUTS: usize = 1080;

/// Something that fills columns from `start` in column order.
enum Step<'a> {
    Batch(&'a WitnessBatch),
    Block {
        start: usize,
        lanes: &'a [Linear],
    },
    Ring {
        cells: usize,
        start: usize,
        lanes: &'a [Linear],
    },
}

impl Step<'_> {
    fn start(&self) -> usize {
        match self {
            Self::Batch(batch) => batch.start,
            Self::Block { start, .. } | Self::Ring { start, .. } => *start,
        }
    }
}

fn evaluate(linear: &Linear, columns: &[Goldilocks]) -> Goldilocks {
    linear
        .terms
        .iter()
        .fold(linear.constant, |sum, &(column, coefficient)| {
            sum + coefficient * columns[column]
        })
}

/// The schoolbook product of two ring elements, all 107 coefficients.
fn product(left: &[Goldilocks], right: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut full = vec![Goldilocks::ZERO; 2 * RING_DEGREE - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            full[i + j] += a * b;
        }
    }
    full
}

/// Divides by Φ81 = X^54 + X^27 + 1: the quotient and the remainder.
fn divide(mut full: Vec<Goldilocks>) -> (Vec<Goldilocks>, Vec<Goldilocks>) {
    let mut quotient = vec![Goldilocks::ZERO; RING_DEGREE];
    for degree in (RING_DEGREE..full.len()).rev() {
        let lead = full[degree];
        quotient[degree - RING_DEGREE] = lead;
        full[degree] = Goldilocks::ZERO;
        full[degree - RING_DEGREE + 27] -= lead;
        full[degree - RING_DEGREE] -= lead;
    }
    full.truncate(RING_DEGREE);
    (quotient, full)
}

/// A ring piece's challenge, value and prior of `cell`, as lane values.
fn ring_operands(cells: usize, cell: usize, lanes: &[Linear], columns: &[Goldilocks]) -> [Vec<Goldilocks>; 3] {
    let lane = |index: usize| evaluate(&lanes[index], columns);
    [
        (0..RING_DEGREE).map(lane).collect(),
        (0..RING_DEGREE)
            .map(|index| lane(RING_DEGREE + index * cells + cell))
            .collect(),
        (0..RING_DEGREE)
            .map(|index| lane(RING_DEGREE + RING_DEGREE * cells + index * cells + cell))
            .collect(),
    ]
}

impl StepPackage {
    pub fn source_assignment(&self, logical: &[Goldilocks]) -> Result<Vec<Goldilocks>, PackageError> {
        if logical.first() != Some(&Goldilocks::ONE) {
            return Err(PackageError::Input("bounded-depth constant column"));
        }
        let mut columns = self.layout()?.source_assignment(logical)?;
        columns.resize(self.source_end(), Goldilocks::ZERO);
        for (output, form) in self.output_forms()? {
            columns[output] = form
                .iter()
                .fold(Goldilocks::ZERO, |sum, &(column, coefficient)| {
                    sum + coefficient * columns[column]
                });
        }
        self.field_lowering.reconstruct(&mut columns)?;
        Ok(columns)
    }

    /// The circuit's columns below `first_fresh` from `shared`, the columns
    /// below the first local: column 0 is the constant 1. A block's 1,080 dead
    /// locals stay 0, because no row reads them (`Frame.Framed`).
    pub fn circuit_columns(&self, shared: &[Goldilocks]) -> Result<Vec<Goldilocks>, PackageError> {
        if shared.first() != Some(&Goldilocks::ONE) || shared.len() > self.first_fresh {
            return Err(PackageError::Input("bounded-depth shared columns"));
        }
        if shared.len() != self.application.local_start {
            return Err(PackageError::Input("bounded-depth shared column count"));
        }
        let mut columns = vec![Goldilocks::ZERO; self.first_fresh];
        columns[..shared.len()].copy_from_slice(shared);
        if let Some(records) = &self.native {
            self.fill_application(records, &mut columns)?;
            for block in self.blocks() {
                columns[block.start()..block.outputs()].fill(Goldilocks::ZERO);
            }
        }
        let mut steps: Vec<_> = self.witness.iter().map(Step::Batch).collect();
        for piece in &self.pieces {
            match piece {
                Piece::Block { start, lanes } => steps.push(Step::Block { start: *start, lanes }),
                Piece::Ring {
                    cells, start, lanes, ..
                } => steps.push(Step::Ring {
                    cells: *cells,
                    start: *start,
                    lanes,
                }),
                Piece::Rows(_) => {}
            }
        }
        steps.sort_by_key(Step::start);
        for step in &steps {
            match step {
                Step::Batch(batch) => execute_witness_batch(batch, &mut columns),
                Step::Block { start, lanes } => {
                    let state: [Goldilocks; 16] = lanes
                        .iter()
                        .map(|lane| evaluate(lane, &columns))
                        .collect::<Vec<_>>()
                        .try_into()
                        .map_err(|_| PackageError::Malformed("bounded-depth block lane count"))?;
                    let outputs = start + BLOCK_OUTPUTS;
                    columns[outputs..start + BLOCK_LOCALS].copy_from_slice(&permute_state(state));
                }
                Step::Ring { cells, start, lanes } => {
                    for cell in 0..*cells {
                        let [challenge, value, prior] = ring_operands(*cells, cell, lanes, &columns);
                        let (_, remainder) = divide(product(&challenge, &value));
                        for lane in 0..RING_DEGREE {
                            columns[start + lane * cells + cell] = prior[lane] + remainder[lane];
                        }
                    }
                }
            }
        }
        Ok(columns)
    }

    /// Extends the circuit's columns with the relation's fresh columns: each
    /// block's S-box outputs from `rows`, then each lowering column `f` from
    /// its row `a * b = f` and each ring piece's quotients. The S-box columns
    /// come first, because a row can read the form of a block output, which
    /// is over S-box columns. A block's lanes read only circuit columns and
    /// outputs of blocks before it. Finally, fill the companions, parity
    /// pairs and zero-product sums (`SPEC.md` sections 7.9 and 7.10).
    pub fn relation_columns(
        &self,
        mut columns: Vec<Goldilocks>,
        rows: &PlanRows,
    ) -> Result<Vec<Goldilocks>, PackageError> {
        let width = self.source_end();
        columns.resize(width, Goldilocks::ZERO);
        for row in rows.ordinary_row_count()..rows.row_count() {
            let [(column, _)] = rows.port(row, 2).collect::<Vec<_>>()[..] else {
                return Err(PackageError::Malformed("bounded-depth S-box row output"));
            };
            columns[column] = rows.evaluate(row, 3, &columns).exp_u64(7);
        }
        if rows.piece_ranges().len() != self.pieces.len() {
            return Err(PackageError::Malformed("bounded-depth witness row pieces"));
        }
        let mut next = self.first_fresh;
        for (piece, range) in self.pieces.iter().zip(rows.piece_ranges()) {
            match piece {
                Piece::Rows(_) => {
                    for row in range.clone() {
                        if let [(column, coefficient)] = rows.port(row, 2).collect::<Vec<_>>()[..] {
                            if column == next && coefficient == Goldilocks::ONE {
                                columns[column] = rows.evaluate(row, 0, &columns) * rows.evaluate(row, 1, &columns);
                                next += 1;
                            }
                        }
                    }
                }
                Piece::Ring {
                    cells, quotient, lanes, ..
                } => {
                    if *quotient != next {
                        return Err(PackageError::Malformed("bounded-depth ring quotient column"));
                    }
                    for cell in 0..*cells {
                        let [challenge, value, _] = ring_operands(*cells, cell, lanes, &columns);
                        let (coefficients, _) = divide(product(&challenge, &value));
                        let target = quotient + cell * RING_DEGREE;
                        columns[target..target + RING_DEGREE].copy_from_slice(&coefficients);
                    }
                    next += RING_DEGREE * cells;
                }
                Piece::Block { .. } => {}
            }
        }
        if next != self.sbox_start {
            return Err(PackageError::Malformed("bounded-depth fresh columns end"));
        }
        for (index, &source) in self.packed_booleans.sources.iter().enumerate() {
            columns[self.packed_booleans.start + index] = Goldilocks::ONE - columns[source];
        }
        self.field_lowering.reconstruct(&mut columns)?;
        for (index, product) in self.packed_booleans.products.iter().enumerate() {
            let column = self.packed_booleans.product_start() + 2 * index;
            columns[column] =
                columns[product.left] + columns[product.right] - Goldilocks::from_u64(2) * columns[product.output];
            columns[column + 1] = Goldilocks::ONE - columns[column];
        }
        for (index, zero) in self.packed_booleans.zeros.iter().enumerate() {
            columns[self.packed_booleans.zero_start() + index] = zero.left.value(&columns) + zero.right.value(&columns);
        }
        Ok(columns)
    }
}

impl StepPackage {
    /// Lean `Parity.witnessSum`: each live column below `first_fresh` times
    /// `mix(0, column)`; a block's 1,080 dead locals are left out.
    pub fn witness_checksum(&self, columns: &[Goldilocks]) -> u64 {
        let mut dead = vec![false; self.first_fresh];
        for block in self.blocks() {
            dead[block.start()..block.outputs()].fill(true);
        }
        columns[..self.first_fresh]
            .iter()
            .enumerate()
            .filter(|(column, _)| !dead[*column])
            .fold(Goldilocks::ZERO, |sum, (column, &value)| {
                sum + value * mix(0, column as u64)
            })
            .as_canonical_u64()
    }
}
