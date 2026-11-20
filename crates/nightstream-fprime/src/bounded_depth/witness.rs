//! Lean `StepPackage.Package`.

use neo_ccs::crypto::poseidon2::{permute_state, INSTANCE_16, SBOX_DEGREE, WIDTH};
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::expand::{mix, PlanRows};
use super::{Linear, Piece, StepPackage};
use crate::package::{PackageError, BLOCK_LOCALS, BLOCK_OUTPUTS, RING_DEGREE, SBOX_ROWS};
use crate::witness::{execute_witness_batch, WitnessBatch};

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
        if shared.first() != Some(&Fq::ONE) || shared.len() > self.first_fresh {
            return Err(PackageError::Input("bounded-depth shared columns"));
        }
        if shared.len() != self.application.local_start {
            return Err(PackageError::Input("bounded-depth shared column count"));
        }
        let mut columns = vec![Fq::ZERO; self.first_fresh];
        columns[..shared.len()].copy_from_slice(shared);
        if let Some(records) = &self.native {
            match locals {
                Some(locals) => columns
                    .get_mut(self.application.local_start..)
                    .and_then(|columns| columns.get_mut(..records.recipe_count()))
                    .filter(|columns| columns.len() == locals.len())
                    .ok_or(PackageError::Input("bounded-depth application locals"))?
                    .copy_from_slice(locals),
                None => self.fill_application(records, &mut columns)?,
            }
            for block in self.blocks() {
                columns[block.start()..block.first_output()].fill(Fq::ZERO);
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
                    let state: [Fq; 16] = lanes
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
                        let (_, remainder) = divide_by_ring_modulus(product(&challenge, &value));
                        for lane in 0..RING_DEGREE {
                            columns[start + lane * cells + cell] = prior[lane] + remainder[lane];
                        }
                    }
                }
            }
        }
        Ok(columns)
    }

    /// `SPEC.md` sections 7.9 and 7.10.
    pub fn relation_columns(&self, columns: Vec<Fq>, rows: &PlanRows) -> Result<Vec<Fq>, PackageError> {
        self.assign_relation_columns(columns, rows, &self.fresh_product_rows(rows), None)
    }

    pub(super) fn fresh_product_rows(&self, rows: &PlanRows) -> Box<[usize]> {
        let mut products = Vec::new();
        let mut next = self.first_fresh;
        for (piece, range) in self.pieces.iter().zip(rows.piece_ranges()) {
            match piece {
                Piece::Rows(_) => {
                    for row in range.clone() {
                        if rows.single(row, 2) == Some((next, Fq::ONE)) {
                            products.push(row);
                            next += 1;
                        }
                    }
                }
                Piece::Ring { cells, .. } => next += RING_DEGREE * cells,
                Piece::Block { .. } => {}
            }
        }
        products.into_boxed_slice()
    }

    pub(super) fn assign_relation_columns(
        &self,
        mut columns: Vec<Fq>,
        rows: &PlanRows,
        product_rows: &[usize],
        sbox_start: Option<usize>,
    ) -> Result<Vec<Fq>, PackageError> {
        columns.resize(self.source_end(), Fq::ZERO);
        if let Some(start) = sbox_start.filter(|_| columns.first() == Some(&Fq::ONE)) {
            for (block, first) in rows.sbox_rows().step_by(SBOX_ROWS).enumerate() {
                let output = start + block * SBOX_ROWS;
                let state: [Fq; WIDTH] = std::array::from_fn(|lane| {
                    let value = rows
                        .evaluate(first + lane, 3, &columns)
                        .exp_u64(SBOX_DEGREE);
                    columns[output + lane] = value;
                    value
                });
                let mut remaining = &mut columns[output + WIDTH..output + SBOX_ROWS];
                INSTANCE_16.permute_after_first_sboxes(state, |values| {
                    let (out, rest) = std::mem::take(&mut remaining).split_at_mut(values.len());
                    out.copy_from_slice(values);
                    remaining = rest;
                });
            }
        } else {
            for row in rows.sbox_rows() {
                let Some((column, _)) = rows.single(row, 2) else {
                    return Err(PackageError::Malformed("bounded-depth S-box row output"));
                };
                columns[column] = rows.evaluate(row, 3, &columns).exp_u64(SBOX_DEGREE);
            }
        }
        if rows.piece_ranges().len() != self.pieces.len() {
            return Err(PackageError::Malformed("bounded-depth witness row pieces"));
        }
        let mut next = self.first_fresh;
        let mut products = product_rows.iter().copied().peekable();
        for (piece, range) in self.pieces.iter().zip(rows.piece_ranges()) {
            match piece {
                Piece::Rows(_) => {
                    while let Some(row) = products.next_if(|&row| row < range.end) {
                        columns[next] = rows.evaluate(row, 0, &columns) * rows.evaluate(row, 1, &columns);
                        next += 1;
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
        self.field_lowering.reconstruct(&mut columns)?;
        for (index, product) in self.packed_booleans.products.iter().enumerate() {
            let column = self.packed_booleans.product_start() + 2 * index;
            columns[column] =
                columns[product.left] + columns[product.right] - Fq::from_u64(2) * columns[product.output];
            columns[column + 1] = Fq::ONE - columns[column];
        }
        for (index, zero) in self.packed_booleans.zeros.iter().enumerate() {
            columns[self.packed_booleans.zero_start() + index] = zero.left.value(&columns) + zero.right.value(&columns);
        }
        Ok(columns)
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
