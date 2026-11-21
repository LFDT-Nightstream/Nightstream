//! Lean `CircuitRelation.plan`, `Package.relationRows`, `CircuitRelation.outputForms`.

use std::collections::HashMap;
use std::ops::Range;

use neo_ccs::{
    crypto::poseidon2::SBOX_DEGREE,
    sparse::{FactoredEntries, FactoredPorts, SharedPort},
};
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{FrameCodec, Linear, Piece, Row, StepPackage};
use crate::package::{
    external_outputs, permutation_rows, Entries, PackageError, BLOCK_OUTPUTS, RING_DEGREE, RING_NODES, SBOX_ROWS,
};

const PORTS: usize = 4;
const BLOCK_LANES: usize = neo_ccs::crypto::poseidon2::WIDTH;
const LAST_FULL_ROUND_SBOX_START: usize = SBOX_ROWS - BLOCK_LANES;

fn reserve_span(end: &mut usize, len: usize) -> u32 {
    let start = *end;
    *end += len;
    start as u32
}

fn shared_span<'a, T: Eq + std::hash::Hash>(
    forms: &mut HashMap<&'a [T], u32>,
    values: &'a [T],
    end: &mut usize,
) -> u32 {
    if forms.len() == 1 << 16 {
        forms.clear();
    }
    *forms
        .entry(values)
        .or_insert_with(|| reserve_span(end, values.len()))
}

fn compact_spans<T: Copy>(
    values: &mut Vec<T>,
    offsets: &[usize],
    ports: &[SharedPort],
    start: impl Fn(&SharedPort) -> u32,
) {
    let mut end = 0;
    for (bounds, port) in offsets.windows(2).zip(ports) {
        if start(port) as usize == end {
            if bounds[0] != end {
                values.copy_within(bounds[0]..bounds[1], end);
            }
            end += port.len as usize;
        }
    }
    values.truncate(end);
}

pub struct PlanRows {
    ordinary: usize,
    piece_ranges: Vec<Range<usize>>,
    rings: Vec<Range<usize>>,
    offsets: Vec<usize>,
    columns: Vec<u32>,
    coefficients: Vec<Fq>,
    shared_ports: Option<(Box<[SharedPort]>, usize)>,
}

impl PlanRows {
    pub(super) fn pack<C: FrameCodec>(&mut self) -> impl FnOnce(&mut Self) -> Result<(), PackageError> + Send + Sync {
        use super::compact::Frame;

        let columns = Frame::<C>::pack_u32(std::mem::take(&mut self.columns).into_iter());
        let coefficients = Frame::<C>::pack_fields(&std::mem::take(&mut self.coefficients));
        let ports = self.shared_ports.take().map(|(ports, entries)| {
            (
                entries,
                [
                    Frame::<C>::pack_u32(ports.iter().map(|port| port.slot_start)),
                    Frame::<C>::pack_u32(ports.iter().map(|port| port.coefficient_start)),
                    Frame::<C>::pack_u32(ports.iter().map(|port| port.len)),
                ],
            )
        });
        move |rows| {
            rows.columns = columns.unpack_u32()?;
            rows.coefficients = coefficients.unpack_fields()?;
            if let Some((entries, [slots, coefficients, lengths])) = ports {
                let count = slots.count;
                let (mut slots, mut coefficients, mut lengths) =
                    (slots.reader()?, coefficients.reader()?, lengths.reader()?);
                let mut restored = Vec::with_capacity(count);
                for _ in 0..count {
                    restored.push(SharedPort {
                        slot_start: slots.u32()?,
                        coefficient_start: coefficients.u32()?,
                        len: lengths.u32()?,
                    });
                }
                slots.finish()?;
                coefficients.finish()?;
                lengths.finish()?;
                rows.shared_ports = Some((restored.into_boxed_slice(), entries));
            }
            Ok(())
        }
    }

    fn shrink_to_fit(&mut self) {
        self.share_ports();
        self.offsets.shrink_to_fit();
        self.columns.shrink_to_fit();
        self.coefficients.shrink_to_fit();
    }

    pub fn piece_ranges(&self) -> &[Range<usize>] {
        &self.piece_ranges
    }

    pub fn row_count(&self) -> usize {
        self.shared_ports
            .as_ref()
            .map_or_else(|| self.offsets.len() - 1, |(ports, _)| ports.len())
            / PORTS
    }

    pub fn ordinary_row_count(&self) -> usize {
        self.ordinary
    }

    pub fn entry_count(&self) -> usize {
        self.entries().entry_count()
    }

    pub fn entries(&self) -> FactoredEntries<'_> {
        FactoredEntries {
            ports: match &self.shared_ports {
                Some((ports, entries)) => FactoredPorts::Shared {
                    ports,
                    entries: *entries,
                },
                None => FactoredPorts::Offsets(&self.offsets),
            },
            slots: &self.columns,
            coefficients: &self.coefficients,
        }
    }

    fn share_ports(&mut self) {
        let port_count = self.offsets.len() - 1;
        let entries = self.columns.len();
        let overhead = (port_count as u64) * (size_of::<SharedPort>() - size_of::<usize>()) as u64;
        if entries > u32::MAX as usize || (entries as u64) * (size_of::<u32>() + size_of::<Fq>()) as u64 <= overhead {
            return;
        }
        let mut slot_forms = HashMap::new();
        let mut coefficient_forms = HashMap::new();
        let mut ports = Vec::new();
        if ports.try_reserve_exact(port_count).is_err() {
            return;
        }
        let mut slot_count = 0;
        let mut coefficient_count = 0;
        for bounds in self.offsets.windows(2) {
            let range = bounds[0]..bounds[1];
            let columns = &self.columns[range.clone()];
            let slot_start = if columns.len() < 4 {
                reserve_span(&mut slot_count, columns.len())
            } else {
                shared_span(&mut slot_forms, columns, &mut slot_count)
            };
            let coefficient_start = shared_span(
                &mut coefficient_forms,
                &self.coefficients[range],
                &mut coefficient_count,
            );
            ports.push(SharedPort {
                slot_start,
                coefficient_start,
                len: columns.len() as u32,
            });
        }
        let saved = (entries - slot_count) as u64 * size_of::<u32>() as u64
            + (entries - coefficient_count) as u64 * size_of::<Fq>() as u64;
        if saved > overhead {
            compact_spans(&mut self.columns, &self.offsets, &ports, |port| port.slot_start);
            compact_spans(&mut self.coefficients, &self.offsets, &ports, |port| {
                port.coefficient_start
            });
            self.shared_ports = Some((ports.into_boxed_slice(), entries));
            self.offsets = Vec::new();
        }
    }

    pub fn ring_rows(&self) -> &[Range<usize>] {
        &self.rings
    }

    pub fn sbox_rows(&self) -> Range<usize> {
        self.ordinary..self.row_count()
    }

    #[inline]
    pub fn evaluate(&self, row: usize, port: usize, columns: &[Fq]) -> Fq {
        let (slots, coefficients) = self.entries().port(PORTS * row + port);
        match (slots, coefficients) {
            ([], []) => Fq::ZERO,
            ([slot], [coefficient]) => {
                let value = columns[*slot as usize];
                match *coefficient {
                    Fq::ONE => value,
                    Fq::NEG_ONE => -value,
                    coefficient => coefficient * value,
                }
            }
            _ => Fq::sum_of_products(
                slots
                    .iter()
                    .zip(coefficients)
                    .map(|(&slot, &coefficient)| (coefficient, columns[slot as usize])),
            ),
        }
    }

    pub fn single(&self, row: usize, port: usize) -> Option<(usize, Fq)> {
        let entries = self.entries();
        let port = PORTS * row + port;
        (entries.port_len(port) == 1).then(|| {
            let (slots, coefficients) = entries.port(port);
            (slots[0] as usize, coefficients[0])
        })
    }

    pub fn residual(&self, row: usize, columns: &[Fq]) -> Fq {
        let entries = self.entries();
        let index = PORTS * row;
        let evaluate = |port: usize| self.evaluate(row, port, columns);
        let output = evaluate(2);
        if entries.port_len(index + 3) != 0 {
            return evaluate(0) * evaluate(1) - output + evaluate(3).exp_u64(SBOX_DEGREE);
        }
        let (first, second) = if entries.port_len(index) <= entries.port_len(index + 1) {
            (0, 1)
        } else {
            (1, 0)
        };
        let factor = evaluate(first);
        if factor == Fq::ZERO {
            return -output;
        }
        factor * evaluate(second) - output
    }

    pub fn port(&self, row: usize, port: usize) -> impl Iterator<Item = (usize, Fq)> + '_ {
        let (columns, coefficients) = self.entries().port(PORTS * row + port);
        columns
            .iter()
            .zip(coefficients)
            .map(|(&column, &coefficient)| (column as usize, coefficient))
    }

    /// Lean `Parity.summary`.
    pub fn ordinary_checksum(&self) -> u64 {
        let mut total = Fq::ZERO;
        for slot in 0..PORTS * self.ordinary {
            let unit = slot as u64;
            let (columns, coefficients) = self.entries().port(slot);
            for (&column, &coefficient) in columns.iter().zip(coefficients) {
                total += coefficient * mix(unit, u64::from(column));
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
                shared_ports: None,
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

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_source_rows.rs"]
mod source_rows_tests;
