//! Lean `CircuitRelation.plan`, `Package.relationRows`, `CircuitRelation.outputForms`.

use std::collections::HashMap;
use std::ops::Range;

use neo_ccs::crypto::poseidon2::SBOX_DEGREE;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{Linear, Piece, Row, StepPackage};
use crate::package::{
    external_outputs, permutation_rows, Entries, PackageError, BLOCK_OUTPUTS, RING_DEGREE, RING_NODES, SBOX_ROWS,
};

const PORTS: usize = 4;
const BLOCK_LANES: usize = neo_ccs::crypto::poseidon2::WIDTH;
const LAST_FULL_ROUND_SBOX_START: usize = SBOX_ROWS - BLOCK_LANES;

pub struct PlanRows {
    ordinary: usize,
    piece_ranges: Vec<Range<usize>>,
    rings: Vec<Range<usize>>,
    offsets: Vec<usize>,
    columns: Vec<u32>,
    coefficients: Vec<Fq>,
}

impl PlanRows {
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

    pub fn ring_rows(&self) -> &[Range<usize>] {
        &self.rings
    }

    pub fn sbox_rows(&self) -> Range<usize> {
        self.ordinary..self.row_count()
    }

    pub fn evaluate(&self, row: usize, port: usize, columns: &[Fq]) -> Fq {
        Fq::sum_of_products(
            self.port(row, port)
                .map(|(column, coefficient)| (coefficient, columns[column])),
        )
    }

    /// The entry of port `port` of `row` when the port has exactly one.
    pub fn single(&self, row: usize, port: usize) -> Option<(usize, Fq)> {
        let slot = PORTS * row + port;
        (self.offsets[slot + 1] == self.offsets[slot] + 1).then(|| {
            let index = self.offsets[slot];
            (self.columns[index] as usize, self.coefficients[index])
        })
    }

    pub fn residual(&self, row: usize, columns: &[Fq]) -> Fq {
        self.evaluate(row, 0, columns) * self.evaluate(row, 1, columns) - self.evaluate(row, 2, columns)
            + self.evaluate(row, 3, columns).exp_u64(SBOX_DEGREE)
    }

    pub fn port(&self, row: usize, port: usize) -> impl Iterator<Item = (usize, Fq)> + '_ {
        let slot = PORTS * row + port;
        let range = self.offsets[slot]..self.offsets[slot + 1];
        self.columns[range.clone()]
            .iter()
            .zip(&self.coefficients[range])
            .map(|(&column, &coefficient)| (column as usize, coefficient))
    }

    /// Lean `Parity.summary`.
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
    /// Lean `CircuitRelation.outputForms`.
    forms: HashMap<usize, Entries>,
    port: Entries,
    rows: PlanRows,
}

impl Builder {
    /// Lean `Substitute.term`.
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

    fn push_power(&mut self, input: &Linear, output: usize) -> Result<(), PackageError> {
        self.end_port();
        self.end_port();
        self.push(output, Fq::ONE)?;
        self.end_port();
        self.push_linear(input)
    }

    /// Lean `PieceRows.ringRows`.
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
    pub(super) fn output_forms(&self) -> Result<HashMap<usize, Entries>, PackageError> {
        let mut forms = HashMap::new();
        for (index, block) in self.blocks().enumerate() {
            let sbox = self.sbox_start + index * SBOX_ROWS;
            let finals = (0..BLOCK_LANES)
                .map(|lane| sbox + LAST_FULL_ROUND_SBOX_START + lane)
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

    /// Lean `Package.relationRows`.
    pub(super) fn plan_rows(&self) -> Result<PlanRows, PackageError> {
        self.rows(true)
    }

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
