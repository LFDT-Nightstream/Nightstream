//! The prover of the argument: commit `W`, then prove.

use std::time::{Duration, Instant};

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::EqualityWeights;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::akita::{self, prover::Prover};
use crate::layout::{Layout, Slice, LOOKUPS, LOOKUP_VARS};
use crate::lookup::{code, Inverses, OUTSIDE};
use crate::sumcheck::{self, Tables};
use crate::transcript::{absorb_bytes, absorb_extension, challenges as draw, session};
use crate::weights::{self, Challenges};
use crate::{
    absorb_statements, challenges, claimed_sum, inverse_point, layout, lookup_challenges, Error, Proof, Statement,
    INVERSE_SESSION, WITNESS_SESSION,
};

const CELLS_PER_TASK: usize = 1 << 16;

/// Where the prover's time goes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProverTimes {
    pub commit_witness: Duration,
    pub commit_inverse: Duration,
    /// The public weights and the other tables of the sum-check.
    pub weights: Duration,
    pub sumcheck: Duration,
    pub openings: Duration,
}

/// `W` and its commitment, before the claims are complete: a caller that
/// proves a norm itself binds `commitment` first (`Norm::Caller`).
pub struct Committed {
    layout: Layout,
    akita: Prover,
    witness: Vec<F>,
    /// The table code of each cell of the witnesses that have a table.
    codes: Vec<u16>,
    committed: akita::prover::Committed,
    commit_witness: Duration,
}

/// Block `j`, lane `h` of `witness` to cell `64j + h`.
fn write_cells(cells: &mut [F], witness: &Mat<F>) {
    let blocks = witness.cols();
    if let Some(&value) = witness.virtual_constant_value() {
        cells.fill(value);
    } else if let Some((positive, negative)) = witness.packed_signed_unit_column_masks() {
        cells
            .par_chunks_mut(D)
            .zip(positive.par_iter().zip(negative))
            .for_each(|(block, (&positive, &negative))| {
                for (lane, cell) in block.iter_mut().enumerate() {
                    *cell = F::from_u64(positive >> lane & 1) - F::from_u64(negative >> lane & 1);
                }
            });
    } else {
        let lanes = witness.as_slice();
        cells
            .par_chunks_mut(D)
            .enumerate()
            .for_each(|(block, cells)| {
                for (lane, cell) in cells.iter_mut().enumerate() {
                    *cell = lanes[lane * blocks + block];
                }
            });
    }
}

/// Commit `W` for `layout`: each witness, the table code of each cell with a
/// table, and each table's counts. A cell outside its table gets no count.
pub fn commit(layout: Layout, witnesses: &[&Mat<F>]) -> Result<Committed, Error> {
    let start = Instant::now();
    if witnesses.len() != layout.cells.len()
        || witnesses
            .iter()
            .zip(&layout.cells)
            .any(|(witness, cells)| witness.rows() != D || witness.cols() != cells.blocks)
    {
        return Err(Error::Statement("witness shape"));
    }
    let mut witness = vec![F::ZERO; layout.witness_len];
    for (cells, source) in layout.cells.iter().zip(witnesses) {
        write_cells(&mut witness[cells.witness.range()][..cells.blocks * D], source);
    }
    let mut codes = vec![OUTSIDE; layout.span_len];
    for (_, slice, bound, counts) in layout.tables() {
        let codes = &mut codes[slice.range()];
        codes
            .par_iter_mut()
            .zip(&witness[slice.range()])
            .for_each(|(code_of, &cell)| *code_of = code(cell, bound));
        let values = 2 * bound as usize + 1;
        let histogram = codes
            .par_chunks(CELLS_PER_TASK)
            .fold(
                || vec![0u64; values],
                |mut histogram, codes| {
                    for &code in codes.iter().filter(|&&code| code != OUTSIDE) {
                        histogram[usize::from(code)] += 1;
                    }
                    histogram
                },
            )
            .reduce(
                || vec![0u64; values],
                |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(left, right)| *left += right);
                    left
                },
            );
        for (cell, count) in witness[counts.range()].iter_mut().zip(histogram) {
            *cell = F::from_u64(count);
        }
    }
    let akita = Prover::new(layout.witness_vars.max(layout.inverse_vars))?;
    let committed = akita.commit(
        layout.witness_vars,
        witness.par_iter().copied().map(akita::base).collect(),
    )?;
    Ok(Committed {
        layout,
        akita,
        witness,
        codes,
        committed,
        commit_witness: start.elapsed(),
    })
}

/// The slices of the witnesses with a table, with the inverses of each.
fn table_inverses(layout: &Layout, taus: &[F; LOOKUPS]) -> Result<Vec<(Slice, Inverses)>, Error> {
    layout
        .tables()
        .map(|(_, slice, bound, _)| Ok((slice, Inverses::new(bound, taus)?)))
        .collect()
}

impl Committed {
    /// The bytes of the commitment of `W`.
    pub fn commitment(&self) -> &[u8] {
        &self.committed.bytes
    }

    /// Column `k` of `V`: on each cell with a table, the inverse of
    /// challenge `k`.
    fn write_inverse_column(&self, tables: &[(Slice, Inverses)], k: usize, column: &mut [akita::Base]) {
        for (slice, inverses) in tables {
            column[slice.range()]
                .par_iter_mut()
                .zip(&self.codes[slice.range()])
                .for_each(|(cell, &code)| *cell = akita::base(inverses.at(k, code)));
        }
    }

    /// The tables of the sum-check on the cells with a table: the weight of
    /// the inverse check, the weight of the sum, and the two combinations of
    /// the inverses.
    fn lookup_tables(&self, tables: &[(Slice, Inverses)], challenges: &Challenges) -> [Vec<K>; 4] {
        let mut check = vec![K::ZERO; self.codes.len()];
        let mut sum = vec![K::ZERO; self.codes.len()];
        let mut inverse = vec![K::ZERO; self.codes.len()];
        let mut scaled = vec![K::ZERO; self.codes.len()];
        let scaled_psi: [K; LOOKUPS] = std::array::from_fn(|k| challenges.psi[k] * challenges.taus[k]);
        for (table, (slice, inverses)) in tables.iter().enumerate() {
            let equality = EqualityWeights::new(&challenges.beta[..slice.vars]);
            let (xi, lambda) = (challenges.xi[table], challenges.lambda[table]);
            check[slice.range()]
                .par_iter_mut()
                .enumerate()
                .for_each(|(cell, weight)| *weight = xi * equality.at(cell));
            sum[slice.range()].fill(lambda);
            for (output, combined) in [
                (&mut inverse, inverses.combine(&challenges.psi)),
                (&mut scaled, inverses.combine(&scaled_psi)),
            ] {
                output[slice.range()]
                    .par_iter_mut()
                    .zip(&self.codes[slice.range()])
                    .for_each(|(cell, &code)| *cell = combined.get(usize::from(code)).copied().unwrap_or(K::ZERO));
            }
        }
        [check, sum, inverse, scaled]
    }

    /// Each column of `V` at `rho`, from the codes: the cells of one code
    /// share one inverse, so the sum of their `eq` weights is sufficient.
    fn inverse_values(&self, tables: &[(Slice, Inverses)], rho: &[K]) -> [K; 1 << LOOKUP_VARS] {
        let mut values = [K::ZERO; 1 << crate::layout::LOOKUP_VARS];
        for (slice, inverses) in tables {
            let equality = EqualityWeights::new(&rho[..slice.vars]);
            let codes = &self.codes[slice.range()];
            let bins = || vec![K::ZERO; inverses.values()];
            let weights = codes
                .par_chunks(CELLS_PER_TASK)
                .enumerate()
                .fold(bins, |mut weights, (task, codes)| {
                    for (cell, &code) in codes.iter().enumerate() {
                        if code != OUTSIDE {
                            weights[usize::from(code)] += equality.at(task * CELLS_PER_TASK + cell);
                        }
                    }
                    weights
                })
                .reduce(bins, |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(left, right)| *left += right);
                    left
                });
            let selector = slice.selector(&rho[..self.layout.span_vars]);
            for (k, value) in values.iter_mut().enumerate().take(LOOKUPS) {
                *value += selector
                    * weights
                        .iter()
                        .enumerate()
                        .map(|(code, &weight)| weight * inverses.at(k, code as u16))
                        .sum::<K>();
            }
        }
        values
    }

    /// Prove `statements`, whose witnesses are the committed ones in the
    /// same order. A witness that does not satisfy its claim gives a proof
    /// that the verifier rejects.
    pub fn prove(
        self,
        transcript: &mut Poseidon2Transcript,
        statements: &[Statement<'_>],
    ) -> Result<(Proof, ProverTimes), Error> {
        if layout(statements)? != self.layout {
            return Err(Error::Statement("the claims differ from the committed layout"));
        }
        let mut times = ProverTimes {
            commit_witness: self.commit_witness,
            ..ProverTimes::default()
        };
        absorb_statements(transcript, statements);
        absorb_bytes(transcript, self.commitment());
        let taus = lookup_challenges(transcript);

        let start = Instant::now();
        let tables = table_inverses(&self.layout, &taus)?;
        let mut inverse = vec![akita::base(F::ZERO); self.layout.inverse_len()];
        for k in 0..LOOKUPS {
            let column = &mut inverse[k << self.layout.span_vars..][..self.layout.span_len];
            self.write_inverse_column(&tables, k, column);
        }
        let inverse = self.akita.commit(self.layout.inverse_vars, inverse)?;
        times.commit_inverse = start.elapsed();
        absorb_bytes(transcript, &inverse.bytes);
        let challenges = challenges(transcript, &self.layout, taus);

        let start = Instant::now();
        // The lookup tables are ready before the pass over the Ajtai key ends.
        let (weights, (witness, [check, sum, combined, scaled])) = rayon::join(
            || weights::table(statements, &self.layout, &challenges),
            || {
                let witness = self.witness.par_iter().map(|&cell| K::from(cell)).collect();
                (witness, self.lookup_tables(&tables, &challenges))
            },
        );
        let sum_tables = Tables {
            witness,
            weights: weights?,
            check,
            sum,
            inverse: combined,
            scaled_inverse: scaled,
        };
        times.weights = start.elapsed();

        let start = Instant::now();
        let (rounds, rho, witness_value) = sumcheck::prove(
            transcript,
            sum_tables,
            claimed_sum(statements, &challenges),
            self.layout.witness_vars,
        );
        let inverse_values = self.inverse_values(&tables, &rho);
        times.sumcheck = start.elapsed();
        absorb_extension(transcript, &[witness_value]);
        absorb_extension(transcript, &inverse_values);
        let column = draw(transcript, LOOKUP_VARS);

        let start = Instant::now();
        let inverse_value = weights::equality(&column, inverse_values.len())
            .iter()
            .zip(&inverse_values)
            .map(|(&weight, &value)| weight * value)
            .sum();
        let witness_commitment = self.committed.bytes.clone();
        let inverse_commitment = inverse.bytes.clone();
        let (akita, layout, committed) = (&self.akita, &self.layout, self.committed);
        let (witness_opening, inverse_opening) = rayon::join(
            || akita.open(committed, &rho, witness_value, &session(transcript, WITNESS_SESSION)),
            || {
                akita.open(
                    inverse,
                    &inverse_point(layout, &rho, &column),
                    inverse_value,
                    &session(transcript, INVERSE_SESSION),
                )
            },
        );
        let (witness_opening, inverse_opening) = (witness_opening?, inverse_opening?);
        times.openings = start.elapsed();
        Ok((
            Proof {
                witness_commitment,
                inverse_commitment,
                rounds,
                witness_value,
                inverse_values,
                witness_opening,
                inverse_opening,
            },
            times,
        ))
    }
}

#[cfg(test)]
#[path = "../tests/unit/cheating_prover.rs"]
mod tests;
