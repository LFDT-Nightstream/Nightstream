use neo_ccs::SparsePoly;
use neo_math::{KExtensions, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::ops::Range;

use super::prefix;
use crate::engines::pi_ccs_joint::gamma_power;
use crate::superneo_eval::{fill, EqualityWeights, MatrixRows, MatrixShape, MatrixWindow, SuperneoZBlocks};
use crate::PiCcsError;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const ROW_DOT_CHUNK: usize = 1 << 14;

const SEVENTH_POWER_BINOMIAL: [u8; 8] = [1, 7, 21, 35, 35, 21, 7, 1];

pub(super) fn is_poseidon2_gate(polynomial: &SparsePoly<F>) -> bool {
    let [ab, c, s] = polynomial.terms() else {
        return false;
    };
    polynomial.arity() == 4
        && ab.exps == [1, 1, 0, 0]
        && ab.coeff == F::ONE
        && c.exps == [0, 0, 1, 0]
        && c.coeff == -F::ONE
        && s.exps == [0, 0, 0, 7]
        && s.coeff == F::ONE
}

fn gate_evaluations(mut result: Vec<K>, points: &[K], use_coefficients: bool) -> Vec<K> {
    if use_coefficients {
        let mut coefficients = [K::ZERO; SEVENTH_POWER_BINOMIAL.len()];
        coefficients.copy_from_slice(&result[..SEVENTH_POWER_BINOMIAL.len()]);
        for (output, &point) in result.iter_mut().zip(points) {
            *output = coefficients
                .iter()
                .rev()
                .fold(K::ZERO, |value, &coefficient| value * point + coefficient);
        }
    }
    result
}

struct Values {
    words: Vec<K>,
    rows: usize,
}

pub(super) struct ApplicationTables {
    shape: MatrixShape,
    fresh_count: usize,
    rows: usize,
    challenges: Vec<K>,
    resident: Option<Values>,
    complete: Option<MatrixWindow>,
    next_gate: Option<[K; 8]>,
    workspace_bytes: usize,
    peak_bytes: usize,
}

fn invalid(reason: &str) -> PiCcsError {
    PiCcsError::InvalidInput(reason.into())
}

fn bytes(words: usize) -> Result<usize, PiCcsError> {
    words
        .checked_mul(size_of::<K>())
        .ok_or_else(|| invalid("CPU application workspace size overflow"))
}

impl ApplicationTables {
    pub(super) fn new(
        source: &dyn MatrixRows,
        witnesses: &[SuperneoZBlocks],
        workspace_bytes: usize,
    ) -> Result<Self, PiCcsError> {
        let shape = source.shape();
        if shape.rows == 0
            || shape.matrices == 0
            || shape.columns == 0
            || !shape.columns.is_multiple_of(D)
            || witnesses.is_empty()
            || witnesses
                .iter()
                .any(|witness| witness.block_len() != shape.columns / D || !witness.imag_all_zero())
        {
            return Err(invalid("CPU application source or witness shape"));
        }
        witnesses
            .len()
            .checked_mul(shape.matrices)
            .ok_or_else(|| invalid("CPU application table count overflow"))?;
        Ok(Self {
            rows: shape.rows,
            shape,
            fresh_count: witnesses.len(),
            challenges: Vec::new(),
            resident: None,
            complete: None,
            next_gate: None,
            workspace_bytes,
            peak_bytes: 0,
        })
    }

    pub(super) fn resume(&mut self, point: &[K], rows: usize, words: Vec<K>) {
        debug_assert_eq!(words.len(), rows * 4);
        self.resident = Some(Values { rows, words });
        self.rows = rows;
        self.challenges = point.to_vec();
    }

    pub(super) fn fresh_count(&self) -> usize {
        self.fresh_count
    }

    fn complete_bytes(&self) -> usize {
        self.complete
            .as_ref()
            .map_or(0, MatrixWindow::storage_bytes)
    }

    pub(super) fn take_complete_window(&mut self) -> Option<MatrixWindow> {
        self.complete.take()
    }

    fn row_bytes(&self) -> Result<usize, PiCcsError> {
        bytes(self.fresh_count * self.shape.matrices)
    }

    fn record(&mut self, live_bytes: usize) -> Result<(), PiCcsError> {
        if live_bytes > self.workspace_bytes {
            return Err(invalid("CPU application values exceed the supplied workspace"));
        }
        self.peak_bytes = self.peak_bytes.max(live_bytes);
        Ok(())
    }

    fn values(
        &mut self,
        source: &dyn MatrixRows,
        witnesses: &[SuperneoZBlocks],
        row_start: usize,
        rows: usize,
        scratch_bytes: usize,
    ) -> Result<Values, PiCcsError> {
        let word_count = (self.fresh_count * self.shape.matrices)
            .checked_mul(rows)
            .ok_or_else(|| invalid("CPU application value count overflow"))?;
        let value_bytes = bytes(word_count)?;
        let external = scratch_bytes
            .checked_add(value_bytes)
            .ok_or_else(|| invalid("CPU application workspace size overflow"))?;
        self.record(external)?;
        let mut output = Values {
            words: vec![K::ZERO; word_count],
            rows,
        };
        if self.challenges.is_empty()
            && row_start == 0
            && rows == self.shape.rows
            && self.fresh_count == 1
            && self.shape.matrices == 4
        {
            if let Some(prepared) = source.take_checked_fresh_rows().filter(|prepared| {
                prepared.len() == rows
                    && prepared
                        .capacity()
                        .checked_mul(size_of::<[F; 4]>())
                        .and_then(|storage| storage.checked_add(external))
                        .is_some_and(|live| live <= self.workspace_bytes)
            }) {
                self.record(external + prepared.capacity() * size_of::<[F; 4]>())?;
                for (matrix, values) in output.words.chunks_mut(rows).enumerate() {
                    fill(values, |row| K::from(prepared[row][matrix]));
                }
                return Ok(output);
            }
        }
        if self.challenges.is_empty() {
            if let Some(factored) = source.factored()? {
                let slot_bytes = factored.slot_count() * size_of::<F>();
                self.record(external + witnesses.len() * slot_bytes)?;
                let slots: Vec<_> = witnesses
                    .iter()
                    .map(|witness| factored.slot_values(witness))
                    .collect();
                let count = (row_start + rows).min(self.shape.rows) - row_start;
                for (table, values) in output.words.chunks_mut(rows).enumerate() {
                    let (matrix, slots) = (table % self.shape.matrices, &slots[table / self.shape.matrices]);
                    fill(&mut values[..count], |local| {
                        K::from(factored.row_value(matrix, row_start + local, slots))
                    });
                }
                return Ok(output);
            }
        }
        let span = 1usize
            .checked_shl(self.challenges.len() as u32)
            .ok_or_else(|| invalid("CPU application replay span overflow"))?;
        let mut start = row_start
            .checked_mul(span)
            .ok_or_else(|| invalid("CPU application replay offset overflow"))?;
        let end = row_start
            .checked_add(rows)
            .and_then(|end| end.checked_mul(span))
            .ok_or_else(|| invalid("CPU application replay end overflow"))?
            .min(self.shape.rows);
        let payload_per_row = if self.challenges.is_empty() {
            0
        } else {
            size_of::<F>() + size_of::<K>()
        };
        while start < end {
            let window = MatrixWindow::load_next_with_payload(
                source,
                start..end,
                self.workspace_bytes - external,
                payload_per_row,
            )?;
            let original = window.rows();
            if original.start != start || original.end <= start || original.end > end {
                return Err(invalid("CPU application row window returned a different range"));
            }
            self.record(external + window.workspace_peak_bytes())?;
            if self.challenges.is_empty() {
                let matrices = window.cache().matrix_caches();
                let range = original.start - row_start..original.end - row_start;
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                output
                    .words
                    .par_chunks_mut(rows)
                    .enumerate()
                    .for_each(|(table, values)| {
                        let matrix = &matrices[table % self.shape.matrices];
                        let witness = &witnesses[table / self.shape.matrices];
                        values[range.clone()]
                            .par_chunks_mut(ROW_DOT_CHUNK)
                            .enumerate()
                            .for_each(|(chunk, values)| {
                                matrix.fill_row_dots_real_from(chunk * ROW_DOT_CHUNK, values, witness)
                            });
                    });
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                output
                    .words
                    .chunks_mut(rows)
                    .enumerate()
                    .for_each(|(table, values)| {
                        matrices[table % self.shape.matrices].fill_row_dots_real_with_blocks(
                            &mut values[range.clone()],
                            &witnesses[table / self.shape.matrices],
                        )
                    });
                start = original.end;
                if original == (0..self.shape.rows) {
                    self.complete = Some(window);
                }
                continue;
            }
            let mut base = vec![F::ZERO; original.len()];
            let prior_weights = original
                .clone()
                .map(|row| {
                    self.challenges
                        .iter()
                        .enumerate()
                        .fold(K::ONE, |weight, (bit, &challenge)| {
                            weight
                                * if (row >> bit) & 1 == 0 {
                                    K::ONE - challenge
                                } else {
                                    challenge
                                }
                        })
                })
                .collect::<Vec<_>>();
            for (fresh, witness) in witnesses.iter().enumerate() {
                for matrix in 0..self.shape.matrices {
                    window
                        .cache()
                        .matrix(matrix)
                        .ok_or_else(|| invalid("CPU application matrix is missing from its row window"))?
                        .fill_row_dots_base_with_blocks(&mut base, witness);
                    let offset = (fresh * self.shape.matrices + matrix) * rows;
                    for (local, &value) in base.iter().enumerate() {
                        let row = (original.start + local) / span - row_start;
                        let value = K::from(value);
                        output.words[offset + row] += value * prior_weights[local];
                    }
                }
            }
            start = original.end;
        }
        Ok(output)
    }

    fn accumulate_pairs(
        &self,
        values: &Values,
        row_start: usize,
        pairs: Range<usize>,
        polynomial: &SparsePoly<F>,
        gamma: K,
        points: &[K],
        weights: &EqualityWeights,
        coordinates: &mut [K],
        result: &mut [K],
    ) {
        for pair in pairs {
            let equality = weights.at(row_start / 2 + pair);
            for (output, &point) in result.iter_mut().zip(points) {
                for fresh in 0..self.fresh_count {
                    for (matrix, coordinate) in coordinates.iter_mut().enumerate() {
                        let offset = (fresh * self.shape.matrices + matrix) * values.rows;
                        let (low, high) = prefix::pair(&values.words[offset..offset + values.rows], pair);
                        *coordinate = prefix::interpolate(low, high, point);
                    }
                    let value: K = polynomial
                        .terms()
                        .iter()
                        .map(|term| {
                            let mut value = K::from(term.coeff);
                            for (&coordinate, &exponent) in coordinates.iter().zip(&term.exps) {
                                if exponent == 0 {
                                    continue;
                                }
                                if coordinate == K::ZERO {
                                    return K::ZERO;
                                }
                                for _ in 0..exponent {
                                    value *= coordinate;
                                }
                            }
                            value
                        })
                        .sum();
                    *output += equality * gamma_power(gamma, fresh) * value;
                }
            }
        }
    }

    fn accumulate_gate_coefficients<T: Field>(
        values: &Values,
        row_start: usize,
        pairs: Range<usize>,
        gamma: K,
        weights: &EqualityWeights,
        coordinate: impl Fn(K) -> T,
        scale: impl Fn(K, T) -> K,
        result: &mut [K],
    ) {
        let mut source_weight = K::ONE;
        for source in values.words.chunks_exact(4 * values.rows) {
            let tables: [&[K]; 4] =
                std::array::from_fn(|matrix| &source[matrix * values.rows..(matrix + 1) * values.rows]);
            let mut quadratic = [K::ZERO; 3];
            let mut seventh_power = [K::ZERO; SEVENTH_POWER_BINOMIAL.len()];
            for pair in pairs.clone() {
                let [[a, da], [b, db], [c, dc], [s, ds]] = tables.map(|table| {
                    let (low, high) = prefix::pair(table, pair);
                    let low = coordinate(low);
                    [low, coordinate(high) - low]
                });
                let weight = weights.at(row_start / 2 + pair);

                if (a != T::ZERO || da != T::ZERO) && (b != T::ZERO || db != T::ZERO) {
                    quadratic[0] += scale(weight, a * b - c);
                    quadratic[1] += scale(weight, a * db + da * b - dc);
                    quadratic[2] += scale(weight, da * db);
                } else {
                    if c != T::ZERO {
                        quadratic[0] -= scale(weight, c);
                    }
                    if dc != T::ZERO {
                        quadratic[1] -= scale(weight, dc);
                    }
                }

                if s == T::ZERO {
                    if ds != T::ZERO {
                        seventh_power[7] += scale(weight, ds.exp_const_u64::<7>());
                    }
                } else if ds == T::ZERO {
                    seventh_power[0] += scale(weight, s.exp_const_u64::<7>());
                } else {
                    let s2 = s.square();
                    let ds2 = ds.square();
                    let s3 = s2 * s;
                    let ds3 = ds2 * ds;
                    let s4 = s2.square();
                    let ds4 = ds2.square();
                    let coefficients = [
                        s4 * s3,
                        s4 * s2 * ds,
                        s4 * s * ds2,
                        s4 * ds3,
                        s3 * ds4,
                        s2 * ds4 * ds,
                        s * ds4 * ds2,
                        ds4 * ds3,
                    ];
                    for (total, coefficient) in seventh_power.iter_mut().zip(coefficients) {
                        *total += scale(weight, coefficient);
                    }
                }
            }
            for (index, &binomial) in SEVENTH_POWER_BINOMIAL.iter().enumerate() {
                let low = quadratic.get(index).copied().unwrap_or(K::ZERO);
                let high = seventh_power[index].scale_base(F::from_u8(binomial));
                result[index] += source_weight * (low + high);
            }
            source_weight *= gamma;
        }
    }

    fn accumulate(
        &self,
        values: &Values,
        row_start: usize,
        polynomial: &SparsePoly<F>,
        gamma: K,
        points: &[K],
        weights: &EqualityWeights,
        use_coefficients: bool,
        coordinates: &mut [K],
        result: &mut [K],
    ) -> Result<usize, PiCcsError> {
        let scratch_words = coordinates
            .len()
            .checked_add(result.len())
            .ok_or_else(|| invalid("CPU application evaluation scratch overflow"))?;
        let live_bytes = values
            .words
            .capacity()
            .checked_add(scratch_words)
            .ok_or_else(|| invalid("CPU application evaluation scratch overflow"))
            .and_then(bytes)?
            .checked_add(self.complete_bytes())
            .ok_or_else(|| invalid("CPU application evaluation scratch overflow"))?;
        let available = self
            .workspace_bytes
            .checked_sub(live_bytes)
            .ok_or_else(|| invalid("CPU application values exceed the supplied workspace"))?;
        let pairs = values.rows.div_ceil(2);
        let sum_pairs = |pairs: Range<usize>, coordinates: &mut [K], result: &mut [K]| {
            if !use_coefficients {
                self.accumulate_pairs(
                    values,
                    row_start,
                    pairs,
                    polynomial,
                    gamma,
                    points,
                    weights,
                    coordinates,
                    result,
                );
            } else if self.challenges.is_empty() {
                Self::accumulate_gate_coefficients(
                    values,
                    row_start,
                    pairs,
                    gamma,
                    weights,
                    |value| value.real(),
                    K::scale_base,
                    result,
                );
            } else {
                Self::accumulate_gate_coefficients(
                    values,
                    row_start,
                    pairs,
                    gamma,
                    weights,
                    |value| value,
                    |left, right| left * right,
                    result,
                );
            }
        };
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            let workers = rayon::current_num_threads()
                .min((pairs * points.len()).div_ceil(prefix::MIN_JOB_PAIRS))
                .min(available / bytes(scratch_words)?);
            if workers > 1 {
                let mut scratch = vec![K::ZERO; workers * scratch_words];
                let peak = live_bytes
                    .checked_add(bytes(scratch.capacity())?)
                    .ok_or_else(|| invalid("CPU application worker scratch overflow"))?;
                if peak > self.workspace_bytes {
                    return Err(invalid("CPU application worker scratch exceeds the supplied workspace"));
                }
                scratch
                    .par_chunks_mut(scratch_words)
                    .enumerate()
                    .for_each(|(worker, scratch)| {
                        let start = worker * (pairs / workers) + worker.min(pairs % workers);
                        let end = start + pairs / workers + usize::from(worker < pairs % workers);
                        let (coordinates, result) = scratch.split_at_mut(self.shape.matrices);
                        sum_pairs(start..end, coordinates, result);
                    });
                for scratch in scratch.chunks(scratch_words) {
                    for (total, &value) in result.iter_mut().zip(&scratch[self.shape.matrices..]) {
                        *total += value;
                    }
                }
                return Ok(peak);
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let _ = available;
        sum_pairs(0..pairs, coordinates, result);
        Ok(live_bytes)
    }

    pub(super) fn evals_at(
        &mut self,
        source: &dyn MatrixRows,
        witnesses: &[SuperneoZBlocks],
        polynomial: &SparsePoly<F>,
        gamma: K,
        points: &[K],
        weights: &EqualityWeights,
    ) -> Result<Vec<K>, PiCcsError> {
        let shape = source.shape();
        let zero: F = polynomial
            .terms()
            .iter()
            .filter(|term| term.exps.iter().all(|&exponent| exponent == 0))
            .map(|term| term.coeff)
            .sum();
        if (shape.rows, shape.columns, shape.matrices) != (self.shape.rows, self.shape.columns, self.shape.matrices)
            || witnesses.len() != self.fresh_count
            || polynomial.arity() != self.shape.matrices
            || zero != F::ZERO
        {
            return Err(invalid("CPU application evaluation shape or zero padding"));
        }
        let scratch_bytes = points
            .len()
            .checked_add(self.shape.matrices)
            .ok_or_else(|| invalid("CPU application evaluation scratch overflow"))
            .and_then(bytes)?;
        let available = self
            .workspace_bytes
            .checked_sub(scratch_bytes)
            .ok_or_else(|| invalid("CPU application evaluation scratch exceeds the supplied workspace"))?;
        if self.resident.as_ref().is_some_and(|values| {
            values
                .words
                .capacity()
                .checked_mul(size_of::<K>())
                .is_none_or(|bytes| bytes > available)
        }) {
            self.next_gate = None;
            self.resident = None;
        }
        let resident = self
            .resident
            .as_ref()
            .map_or(0, |values| values.words.capacity() * size_of::<K>());
        if scratch_bytes + resident + self.complete_bytes() > self.workspace_bytes {
            self.complete = None;
        }
        let retained = resident + self.complete_bytes();
        self.record(scratch_bytes + retained)?;
        let mut result = vec![K::ZERO; points.len()];
        if points.is_empty()
            || polynomial.terms().iter().all(|term| term.coeff == F::ZERO)
            || witnesses.iter().all(SuperneoZBlocks::is_zero)
        {
            self.next_gate = None;
            self.resident = None;
            self.complete = None;
            return Ok(result);
        }
        let use_coefficients = points.len() >= SEVENTH_POWER_BINOMIAL.len() && is_poseidon2_gate(polynomial);
        let mut coordinates = vec![K::ZERO; self.shape.matrices];
        let row_bytes = self.row_bytes()?;
        if self.resident.is_none() && self.rows <= available / row_bytes {
            match self.values(source, witnesses, 0, self.rows, scratch_bytes) {
                Ok(values) => self.resident = Some(values),
                Err(PiCcsError::MatrixWorkspace { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        if let Some(values) = &self.resident {
            if let Some(coefficients) = self.next_gate.filter(|_| use_coefficients) {
                return Ok(points
                    .iter()
                    .map(|&point| crate::sumcheck::poly_eval_k(&coefficients, point))
                    .collect());
            }
            let peak = self.accumulate(
                values,
                0,
                polynomial,
                gamma,
                points,
                weights,
                use_coefficients,
                &mut coordinates,
                &mut result,
            )?;
            self.record(peak)?;
            return Ok(gate_evaluations(result, points, use_coefficients));
        }
        let mut row_start = 0;
        let mut window_rows = (available / row_bytes).min(self.rows);
        while row_start < self.rows {
            let remaining = self.rows - row_start;
            let mut rows = window_rows.min(remaining);
            if rows != remaining {
                rows -= rows % 2;
            }
            if rows == 0 || (rows == 1 && remaining > 1) {
                return Err(invalid("CPU application workspace cannot hold a complete row pair"));
            }
            let values = match self.values(source, witnesses, row_start, rows, scratch_bytes) {
                Ok(values) => values,
                Err(PiCcsError::MatrixWorkspace { required, available }) if rows <= 2 => {
                    return Err(PiCcsError::MatrixWorkspace { required, available });
                }
                Err(PiCcsError::MatrixWorkspace { .. }) => {
                    window_rows = rows.div_ceil(2);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let peak = self.accumulate(
                &values,
                row_start,
                polynomial,
                gamma,
                points,
                weights,
                use_coefficients,
                &mut coordinates,
                &mut result,
            )?;
            self.record(peak)?;
            row_start += rows;
            window_rows = rows;
        }
        Ok(gate_evaluations(result, points, use_coefficients))
    }

    pub(super) fn fold_next_gate(&mut self, challenge: K, polynomial: &SparsePoly<F>, weights: &EqualityWeights) {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if self.fresh_count == 1 && is_poseidon2_gate(polynomial) {
            if let Some(values) = &mut self.resident {
                self.next_gate = Some(bind_gate(values, challenge, weights));
                self.challenges.push(challenge);
                self.rows = self.rows.div_ceil(2);
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let _ = (polynomial, weights);
        self.fold(challenge);
    }

    pub(super) fn fold(&mut self, challenge: K) {
        self.next_gate = None;
        if let Some(values) = &mut self.resident {
            let rows = values.rows;
            let next = rows.div_ceil(2);
            let fold_table = |table: &mut [K]| {
                for row in 0..next {
                    let low = table[2 * row];
                    let high = table.get(2 * row + 1).copied().unwrap_or(K::ZERO);
                    table[row] = prefix::interpolate(low, high, challenge);
                }
            };
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            values
                .words
                .par_chunks_mut(rows)
                .with_min_len(prefix::MIN_JOB_PAIRS.div_ceil(next))
                .for_each(fold_table);
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            values.words.chunks_mut(rows).for_each(fold_table);
            for table in 1..self.fresh_count * self.shape.matrices {
                values
                    .words
                    .copy_within(table * rows..table * rows + next, table * next);
            }
            values
                .words
                .truncate(self.fresh_count * self.shape.matrices * next);
            values.rows = next;
        }
        self.challenges.push(challenge);
        self.rows = self.rows.div_ceil(2);
    }

    pub(super) fn clear(&mut self) {
        self.next_gate = None;
        self.resident = None;
        self.complete = None;
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
fn bind_gate(values: &mut Values, challenge: K, weights: &EqualityWeights) -> [K; 8] {
    const MATRICES: usize = 4;
    let rows = values.rows;
    let next = rows.div_ceil(2);
    let pairs = rows.div_ceil(4);
    let (low, high) = weights.factors();
    let workers = rayon::current_num_threads().min(pairs.div_ceil(1024));
    let groups_per_worker = pairs.div_ceil(low.len() * workers);
    let span = groups_per_worker * low.len() * 4;
    let (a, remaining) = values.words.split_at_mut(rows);
    let (b, remaining) = remaining.split_at_mut(rows);
    let (c, s) = remaining.split_at_mut(rows);
    let coefficients = a
        .par_chunks_mut(span)
        .zip(b.par_chunks_mut(span))
        .zip(c.par_chunks_mut(span))
        .zip(s.par_chunks_mut(span))
        .enumerate()
        .map(|(worker, (((a, b), c), s))| {
            let mut tables = [a, b, c, s];
            let mut coefficients = [K::ZERO; 8];
            let pairs = tables[0].len().div_ceil(4);
            for first in (0..pairs).step_by(low.len()) {
                let mut tile = [K::ZERO; 8];
                let count = (pairs - first).min(low.len());
                for (index, &weight) in low[..count].iter().enumerate() {
                    let bound = tables
                        .each_mut()
                        .map(|table| prefix::bind_four(table, first + index, challenge));
                    for (sum, term) in tile.iter_mut().zip(gate_terms(bound)) {
                        *sum += weight * term;
                    }
                }
                let weight = high[worker * groups_per_worker + first / low.len()];
                coefficients = prefix::add_coefficients(coefficients, tile.map(|value| value * weight));
            }
            coefficients
        })
        .reduce(|| [K::ZERO; 8], prefix::add_coefficients);
    for table in 0..MATRICES {
        let start = table * rows;
        prefix::compact_bound_chunks(&mut values.words[start..start + rows], span);
        if table != 0 {
            values.words.copy_within(start..start + next, table * next);
        }
    }
    values.rows = next;
    values.words.truncate(MATRICES * next);
    coefficients
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
fn gate_terms([[a, a1], [b, b1], [c, c1], [s, s1]]: [[K; 2]; 4]) -> [K; 8] {
    let (da, db, ds) = (a1 - a, b1 - b, s1 - s);
    let low = a * b - c;
    let quadratic = da * db;
    let linear = a1 * b1 - c1 - low - quadratic;
    let mut terms = [low, linear, quadratic, K::ZERO, K::ZERO, K::ZERO, K::ZERO, K::ZERO];
    if s == K::ZERO {
        terms[7] = ds.exp_const_u64::<7>();
    } else if ds == K::ZERO {
        terms[0] += s.exp_const_u64::<7>();
    } else {
        let s2 = s.square();
        let d2 = ds.square();
        let s3 = s * s2;
        let d3 = ds * d2;
        let s4 = s2.square();
        let d4 = d2.square();
        let powers = [
            s4 * s3,
            s4 * s2 * ds,
            s4 * s * d2,
            s4 * d3,
            s3 * d4,
            s2 * d4 * ds,
            s * d4 * d2,
            d4 * d3,
        ];
        for ((term, value), binomial) in terms.iter_mut().zip(powers).zip(SEVENTH_POWER_BINOMIAL) {
            *term += value.scale_base(F::from_u8(binomial));
        }
    }
    terms
}

#[cfg(test)]
#[path = "../../../tests/unit/application_tables.rs"]
mod tests;
