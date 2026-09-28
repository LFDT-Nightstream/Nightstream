//! Fresh matrix values, retained only when their live payload fits the caller's
//! workspace. Replay reads original row windows and applies every prior coin.

use neo_ccs::SparsePoly;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use std::ops::Range;

use super::prefix;
use crate::engines::pi_ccs_joint::gamma_power;
use crate::superneo_eval::{EqualityWeights, MatrixRows, MatrixShape, MatrixWindow, SuperneoZBlocks};
use crate::PiCcsError;

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
            || shape.columns % D != 0
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
            workspace_bytes,
            peak_bytes: 0,
        })
    }

    pub(super) fn fresh_count(&self) -> usize {
        self.fresh_count
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
                // Each table owns its output slice. Write real values directly
                // into it; parallel construction needs no worker row buffers.
                let matrices = window.cache().matrix_caches();
                let range = original.start - row_start..original.end - row_start;
                let fill = |(table, values): (usize, &mut [K])| {
                    matrices[table % self.shape.matrices].fill_row_dots_real_with_blocks(
                        &mut values[range.clone()],
                        &witnesses[table / self.shape.matrices],
                    );
                };
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                output.words.par_chunks_mut(rows).enumerate().for_each(fill);
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                output.words.chunks_mut(rows).enumerate().for_each(fill);
                start = original.end;
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    fn accumulate(
        &self,
        values: &Values,
        row_start: usize,
        polynomial: &SparsePoly<F>,
        gamma: K,
        points: &[K],
        weights: &EqualityWeights,
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
            .and_then(bytes)?;
        let available = self
            .workspace_bytes
            .checked_sub(live_bytes)
            .ok_or_else(|| invalid("CPU application values exceed the supplied workspace"))?;
        let pairs = values.rows.div_ceil(2);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            let workers = rayon::current_num_threads()
                .min(pairs)
                .min(available / bytes(scratch_words)?);
            if workers > 1 {
                // Allocate each worker's complete payload before dispatch. Rayon
                // cannot create additional result or coordinate vectors.
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
                        self.accumulate_pairs(
                            values,
                            row_start,
                            start..end,
                            polynomial,
                            gamma,
                            points,
                            weights,
                            coordinates,
                            result,
                        );
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
        self.accumulate_pairs(
            values,
            row_start,
            0..pairs,
            polynomial,
            gamma,
            points,
            weights,
            coordinates,
            result,
        );
        Ok(live_bytes)
    }

    #[allow(clippy::too_many_arguments)]
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
            self.resident = None;
        }
        let retained = self
            .resident
            .as_ref()
            .map_or(0, |values| values.words.capacity() * size_of::<K>());
        self.record(scratch_bytes + retained)?;
        let mut result = vec![K::ZERO; points.len()];
        if points.is_empty()
            || polynomial.terms().iter().all(|term| term.coeff == F::ZERO)
            || witnesses.iter().all(SuperneoZBlocks::is_zero)
        {
            self.resident = None;
            return Ok(result);
        }
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
            let peak = self.accumulate(
                values,
                0,
                polynomial,
                gamma,
                points,
                weights,
                &mut coordinates,
                &mut result,
            )?;
            self.record(peak)?;
            return Ok(result);
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
                &mut coordinates,
                &mut result,
            )?;
            self.record(peak)?;
            row_start += rows;
            window_rows = rows;
        }
        Ok(result)
    }

    pub(super) fn fold(&mut self, challenge: K) {
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
            values.words.par_chunks_mut(rows).for_each(fold_table);
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            values.words.chunks_mut(rows).for_each(fold_table);
            // All original table reads are complete before prefixes move into
            // earlier slices of the same allocation.
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
        self.resident = None;
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/application_tables.rs"]
mod tests;
