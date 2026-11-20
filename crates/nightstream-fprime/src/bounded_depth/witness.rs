//! Lean `StepPackage.Package`.

use neo_ccs::crypto::poseidon2::{permute_state, INSTANCE_16, SBOX_DEGREE};
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::expand::{mix, PlanRows};
use super::{Linear, Piece, StepPackage};
use crate::package::{PackageError, BLOCK_LOCALS, BLOCK_OUTPUTS, RING_DEGREE, SBOX_ROWS};
use crate::witness::{execute_witness_batch, WitnessBatch};

enum Step<'a> {
    Batch(&'a WitnessBatch),
    Block {
        index: usize,
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

fn evaluate(linear: &Linear, columns: &[Fq]) -> Fq {
    linear
        .terms
        .iter()
        .fold(linear.constant, |sum, &(column, coefficient)| {
            sum + coefficient * columns[column]
        })
}

fn product(left: &[Fq], right: &[Fq]) -> Vec<Fq> {
    let mut full = vec![Fq::ZERO; 2 * RING_DEGREE - 1];
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            full[i + j] += a * b;
        }
    }
    full
}

/// Lean `QuotientProduct.quotient_last_zero`.
fn divide_by_ring_modulus(mut full: Vec<Fq>) -> (Vec<Fq>, Vec<Fq>) {
    let mut quotient = full.split_off(RING_DEGREE);
    for (low, &high) in full.iter_mut().zip(&quotient) {
        *low -= high;
    }
    quotient.resize(RING_DEGREE, Fq::ZERO);
    (quotient, full)
}

fn ring_operands(cells: usize, cell: usize, lanes: &[Linear], columns: &[Fq]) -> [Vec<Fq>; 3] {
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

impl super::ExpandedPackage {
    pub fn assignment_columns(&self, shared: &[Fq], locals: Option<&[Fq]>) -> Result<Vec<Fq>, PackageError> {
        let package = &self.package;
        package.check_shared_columns(shared)?;
        let mut columns = vec![Fq::ZERO; package.source_end()];
        let (inputs, sboxes) = columns.split_at_mut(package.sbox_start);
        package.fill_circuit_columns(shared, locals, &mut inputs[..package.first_fresh], Some(sboxes))?;
        package.finish_relation_columns(&mut columns, &self.rows)?;
        Ok(columns)
    }

    pub fn source_assignment(&self, logical: &[Fq]) -> Result<Vec<Fq>, PackageError> {
        if logical.first() != Some(&Fq::ONE) {
            return Err(PackageError::Input("bounded-depth constant column"));
        }
        let package = &self.package;
        let mut columns = self.layout.source_assignment(logical)?;
        columns.resize(package.source_end(), Fq::ZERO);
        for (output, form) in package.output_forms()? {
            columns[output] = form.iter().fold(Fq::ZERO, |sum, &(column, coefficient)| {
                sum + coefficient * columns[column]
            });
        }
        package.field_lowering.reconstruct(&mut columns)?;
        Ok(columns)
    }
}

impl StepPackage {
    /// Lean `Frame.Framed`.
    pub fn circuit_columns(&self, shared: &[Fq]) -> Result<Vec<Fq>, PackageError> {
        self.circuit_columns_with(shared, None)
    }

    pub fn circuit_columns_with(&self, shared: &[Fq], locals: Option<&[Fq]>) -> Result<Vec<Fq>, PackageError> {
        self.check_shared_columns(shared)?;
        let mut columns = vec![Fq::ZERO; self.first_fresh];
        self.fill_circuit_columns(shared, locals, &mut columns, None)?;
        Ok(columns)
    }

    fn check_shared_columns(&self, shared: &[Fq]) -> Result<(), PackageError> {
        if shared.first() != Some(&Fq::ONE) || shared.len() > self.first_fresh {
            return Err(PackageError::Input("bounded-depth shared columns"));
        }
        if shared.len() != self.application.local_start {
            return Err(PackageError::Input("bounded-depth shared column count"));
        }
        Ok(())
    }

    fn fill_circuit_columns(
        &self,
        shared: &[Fq],
        locals: Option<&[Fq]>,
        columns: &mut [Fq],
        mut sboxes: Option<&mut [Fq]>,
    ) -> Result<(), PackageError> {
        columns[..shared.len()].copy_from_slice(shared);
        if let Some(records) = &self.native {
            match locals {
                Some(locals) => columns
                    .get_mut(self.application.local_start..)
                    .and_then(|columns| columns.get_mut(..records.recipe_count()))
                    .filter(|columns| columns.len() == locals.len())
                    .ok_or(PackageError::Input("bounded-depth application locals"))?
                    .copy_from_slice(locals),
                None => self.fill_application(records, columns)?,
            }
            for block in self.blocks() {
                columns[block.start()..block.first_output()].fill(Fq::ZERO);
            }
        }
        let mut steps: Vec<_> = self.witness.iter().map(Step::Batch).collect();
        let mut block_index = 0;
        for piece in &self.pieces {
            match piece {
                Piece::Block { start, lanes } => {
                    steps.push(Step::Block {
                        index: block_index,
                        start: *start,
                        lanes,
                    });
                    block_index += 1;
                }
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
                Step::Batch(batch) => execute_witness_batch(batch, columns),
                Step::Block { index, start, lanes } => {
                    let state: [Fq; 16] = lanes
                        .iter()
                        .map(|lane| evaluate(lane, columns))
                        .collect::<Vec<_>>()
                        .try_into()
                        .map_err(|_| PackageError::Malformed("bounded-depth block lane count"))?;
                    let outputs = start + BLOCK_OUTPUTS;
                    let output = match sboxes.as_deref_mut() {
                        Some(values) => {
                            let mut target = &mut values[index * SBOX_ROWS..(index + 1) * SBOX_ROWS];
                            INSTANCE_16.permute_with_sboxes(state, |round| {
                                let (words, rest) = std::mem::take(&mut target).split_at_mut(round.len());
                                words.copy_from_slice(round);
                                target = rest;
                            })
                        }
                        None => permute_state(state),
                    };
                    columns[outputs..start + BLOCK_LOCALS].copy_from_slice(&output);
                }
                Step::Ring { cells, start, lanes } => {
                    for cell in 0..*cells {
                        let [challenge, value, prior] = ring_operands(*cells, cell, lanes, columns);
                        let (_, remainder) = divide_by_ring_modulus(product(&challenge, &value));
                        for lane in 0..RING_DEGREE {
                            columns[start + lane * cells + cell] = prior[lane] + remainder[lane];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// `SPEC.md` sections 7.9 and 7.10.
    pub fn relation_columns(&self, mut columns: Vec<Fq>, rows: &PlanRows) -> Result<Vec<Fq>, PackageError> {
        let width = self.source_end();
        columns.resize(width, Fq::ZERO);
        for row in rows.ordinary_row_count()..rows.row_count() {
            let Some((column, _)) = rows.single(row, 2) else {
                return Err(PackageError::Malformed("bounded-depth S-box row output"));
            };
            columns[column] = rows.evaluate(row, 3, &columns).exp_u64(SBOX_DEGREE);
        }
        self.finish_relation_columns(&mut columns, rows)?;
        Ok(columns)
    }

    fn finish_relation_columns(&self, columns: &mut [Fq], rows: &PlanRows) -> Result<(), PackageError> {
        if rows.piece_ranges().len() != self.pieces.len() {
            return Err(PackageError::Malformed("bounded-depth witness row pieces"));
        }
        let mut next = self.first_fresh;
        for (piece, range) in self.pieces.iter().zip(rows.piece_ranges()) {
            match piece {
                Piece::Rows(_) => {
                    for row in range.clone() {
                        if let Some((column, coefficient)) = rows.single(row, 2) {
                            if column == next && coefficient == Fq::ONE {
                                columns[column] = rows.evaluate(row, 0, columns) * rows.evaluate(row, 1, columns);
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
                        let [challenge, value, _] = ring_operands(*cells, cell, lanes, columns);
                        let (coefficients, _) = divide_by_ring_modulus(product(&challenge, &value));
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
            columns[self.packed_booleans.start + index] = Fq::ONE - columns[source];
        }
        self.field_lowering.reconstruct(columns)?;
        for (index, product) in self.packed_booleans.products.iter().enumerate() {
            let column = self.packed_booleans.product_start() + 2 * index;
            columns[column] =
                columns[product.left] + columns[product.right] - Fq::from_u64(2) * columns[product.output];
            columns[column + 1] = Fq::ONE - columns[column];
        }
        for (index, zero) in self.packed_booleans.zeros.iter().enumerate() {
            columns[self.packed_booleans.zero_start() + index] = zero.left.value(columns) + zero.right.value(columns);
        }
        Ok(())
    }
}

impl StepPackage {
    /// Lean `Parity.witnessSum`.
    pub fn witness_checksum(&self, columns: &[Fq]) -> u64 {
        let mut dead = vec![false; self.first_fresh];
        for block in self.blocks() {
            dead[block.start()..block.first_output()].fill(true);
        }
        columns[..self.first_fresh]
            .iter()
            .enumerate()
            .filter(|(column, _)| !dead[*column])
            .fold(Fq::ZERO, |sum, (column, &value)| sum + value * mix(0, column as u64))
            .as_canonical_u64()
    }
}
