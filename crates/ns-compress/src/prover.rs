use std::time::{Duration, Instant};

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::EqualityWeights;
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::akita::{self, Polynomial};
use crate::key::{self, ProverKey};
use crate::lane_prefix::{LaneStart, Masks};
use crate::layout::{Layout, Slice, LANE_VARS, LOOKUPS, LOOKUP_VARS};
use crate::lookup::{code, Inverses, OUTSIDE};
use crate::sumcheck::{self, Lookup, Start, ZeroPrefix};
use crate::transcript::{absorb_bytes, absorb_extension, challenges as draw, session};
use crate::weights::{self, Challenges, Uses};
use crate::{
    absorb_statements, challenges, claimed_sum, inverse_point, layout, lookup_challenges, Error, Proof, Statement,
    SESSION,
};

const CELLS_PER_TASK: usize = 1 << 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct ProverTimes {
    pub commit_witness: Duration,
    pub commit_inverse: Duration,
    pub weights: Duration,
    pub sumcheck: Duration,
    pub key: Duration,
    pub opening: Duration,
}

pub struct Committed<'k> {
    key: &'k ProverKey,
    layout: Layout,
    witness: Vec<F>,
    codes: Vec<u16>,
    zero_prefix: Option<(Slice, u16)>,
    packed_fresh: Option<Mat<F>>,
    committed: akita::prover::Committed,
    commit_witness: Duration,
}

fn zero_prefix(layout: &Layout, witnesses: &[&Mat<F>]) -> Option<(Slice, u16)> {
    let [zero, fresh] = layout.cells.as_slice() else {
        return None;
    };
    let (bound, _) = zero.table?;
    let slice = zero.witness;
    (slice.offset == 0
        && fresh.table.is_none()
        && fresh.blocks == zero.blocks
        && fresh.witness.offset == slice.len()
        && witnesses[0].virtual_constant_value() == Some(&F::ZERO))
    .then(|| (slice, code(F::ZERO, bound)))
}

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

pub fn commit<'k>(key: &'k ProverKey, layout: Layout, witnesses: &[&Mat<F>]) -> Result<Committed<'k>, Error> {
    let start = Instant::now();
    if layout.key_block_vars() != key.verification_key().blocks().trailing_zeros() as usize {
        return Err(Error::Statement("the layout names another key"));
    }
    if witnesses.len() != layout.cells.len()
        || witnesses
            .iter()
            .zip(&layout.cells)
            .any(|(witness, cells)| witness.rows() != D || witness.cols() != cells.blocks)
    {
        return Err(Error::Statement("witness shape"));
    }
    let zero_prefix = zero_prefix(&layout, witnesses);
    let packed_fresh = zero_prefix.and_then(|_| {
        let fresh = witnesses[1];
        fresh
            .packed_signed_unit_column_masks()
            .map(|_| (*fresh).clone())
    });
    let prefix = zero_prefix.map_or(0, |(slice, _)| slice.len());
    let mut witness = vec![F::ZERO; layout.witness_len - prefix];
    for (cells, source) in layout.cells.iter().zip(witnesses) {
        if cells.witness.offset >= prefix {
            write_cells(
                &mut witness[cells.witness.offset - prefix..][..cells.blocks * D],
                source,
            );
        }
    }
    let mut codes = vec![OUTSIDE; layout.table_witness_len - prefix];
    for (_, slice, bound, counts) in layout.tables() {
        if let Some((_, code)) = zero_prefix {
            witness[counts.offset - prefix + usize::from(code)] = F::from_u64(slice.len() as u64);
            continue;
        }
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
    let cells = witness.par_iter().copied().map(akita::base);
    let coefficients = if prefix == 0 {
        cells.collect()
    } else {
        (0..prefix)
            .into_par_iter()
            .map(|_| akita::base(F::ZERO))
            .chain(cells)
            .collect()
    };
    let committed = key
        .akita
        .commit(layout.shape(), Polynomial::Witness, coefficients)?;
    Ok(Committed {
        key,
        layout,
        witness,
        codes,
        zero_prefix,
        packed_fresh,
        committed,
        commit_witness: start.elapsed(),
    })
}

fn table_inverses(layout: &Layout, taus: &[F; LOOKUPS]) -> Result<Vec<(Slice, Inverses)>, Error> {
    layout
        .tables()
        .map(|(_, slice, bound, _)| Ok((slice, Inverses::new(bound, taus)?)))
        .collect()
}

fn lookups(tables: &[(Slice, Inverses)], challenges: &Challenges) -> Vec<Lookup> {
    let scaled_psi: [K; LOOKUPS] = std::array::from_fn(|k| challenges.psi[k] * challenges.taus[k]);
    tables
        .iter()
        .zip(challenges.xi.iter().zip(&challenges.lambda))
        .map(|((slice, inverses), (&xi, &lambda))| Lookup {
            slice: *slice,
            xi,
            lambda,
            inverse: inverses.combine(&challenges.psi),
            scaled: inverses.combine(&scaled_psi),
        })
        .collect()
}

impl Committed<'_> {
    pub fn commitment_to_bind(&self) -> &[u8] {
        &self.committed.bytes
    }

    fn write_inverse_column(&self, tables: &[(Slice, Inverses)], k: usize, column: &mut [akita::Base]) {
        for (slice, inverses) in tables {
            column[slice.range()]
                .par_iter_mut()
                .zip(&self.codes[slice.range()])
                .for_each(|(cell, &code)| *cell = akita::base(inverses.at(k, code)));
        }
    }

    fn inverse_values(&self, tables: &[(Slice, Inverses)], rho: &[K]) -> [K; 1 << LOOKUP_VARS] {
        if let Some((_, code)) = self.zero_prefix {
            let mut values = [K::ZERO; 1 << LOOKUP_VARS];
            for (k, value) in values.iter_mut().enumerate().take(LOOKUPS) {
                *value = K::from(tables[0].1.at(k, code));
            }
            return values;
        }
        let mut values = [K::ZERO; 1 << crate::layout::LOOKUP_VARS];
        for (slice, inverses) in tables {
            let equality = EqualityWeights::new(&rho[..slice.vars]);
            let codes = &self.codes[slice.range()];
            let bins = || vec![K::ZERO; inverses.len()];
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
            let selector = slice.selector(&rho[..self.layout.table_witness_vars]);
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

    pub fn prove(
        mut self,
        transcript: &mut Poseidon2Transcript,
        statements: &[Statement<'_>],
    ) -> Result<(Proof, ProverTimes), Error> {
        let verification = self.key.verification_key();
        if layout(statements, verification.blocks())? != self.layout {
            return Err(Error::Statement("the claims differ from the committed layout"));
        }
        let mut times = ProverTimes {
            commit_witness: self.commit_witness,
            ..ProverTimes::default()
        };
        absorb_statements(transcript, verification, statements);
        absorb_bytes(transcript, self.commitment_to_bind());
        let taus = lookup_challenges(transcript);

        let start = Instant::now();
        let tables = table_inverses(&self.layout, &taus)?;
        let inverse = if let Some((slice, code)) = self.zero_prefix {
            let columns: [akita::Base; LOOKUPS] = std::array::from_fn(|k| akita::base(tables[0].1.at(k, code)));
            (0..self.layout.inverse_len())
                .into_par_iter()
                .with_min_len(CELLS_PER_TASK)
                .map(|cell| columns[cell >> slice.vars])
                .collect()
        } else {
            let mut inverse = vec![akita::base(F::ZERO); self.layout.inverse_len()];
            for k in 0..LOOKUPS {
                let column = &mut inverse[k << self.layout.table_witness_vars..][..self.layout.table_witness_len];
                self.write_inverse_column(&tables, k, column);
            }
            inverse
        };
        let inverse = self
            .key
            .akita
            .commit(self.layout.shape(), Polynomial::Inverse, inverse)?;
        times.commit_inverse = start.elapsed();
        absorb_bytes(transcript, &inverse.bytes);
        let challenges = challenges(transcript, &self.layout, taus);
        let (row_point, scale) = key::row_point(challenges.omega)?;

        let start = Instant::now();
        let masks = self
            .packed_fresh
            .as_ref()
            .and_then(Mat::packed_signed_unit_column_masks)
            .and_then(Masks::suitable);
        let witness = std::mem::take(&mut self.witness);
        let (witness, weights, folded, lane_inputs) = if let (Some((slice, _)), Some(masks)) = (self.zero_prefix, masks)
        {
            let cells = self.layout.cells[1];
            let (source, folded) = rayon::join(
                || weights::LaneSource::new(&statements[1], &challenges, cells.blocks),
                || self.key.fold(challenges.omega),
            );
            let mut witness = witness;
            let tail = witness.split_off(slice.len());
            drop(witness);
            let weights = weights::count_tail(&self.layout, &challenges, cells.witness.range().end)?;
            (tail, weights, folded, Some((masks, source?, slice.len() / D)))
        } else {
            let (weights, folded) = weights::table(
                statements,
                &self.layout,
                &challenges,
                self.key,
                self.zero_prefix.is_some(),
            )?;
            (witness, weights, folded, None)
        };
        let lane_start =
            lane_inputs.map(|(masks, source, padded)| LaneStart::new(masks, &folded, source, padded, challenges.delta));
        times.weights = start.elapsed();

        let start = Instant::now();
        let mut prefix_key = None;
        let claim = claimed_sum(statements, &challenges);
        let rounds = self.layout.witness_vars;
        let (rounds, rho, witness_value) = if let Some((slice, code)) = self.zero_prefix {
            let inverse = |weights: &[K; LOOKUPS]| {
                (0..LOOKUPS)
                    .map(|k| weights[k] * tables[0].1.at(k, code))
                    .sum()
            };
            let lookup = ZeroPrefix {
                beta: &challenges.beta[..slice.vars],
                xi: challenges.xi[0],
                lambda: challenges.lambda[0],
                inverse: inverse(&challenges.psi),
                scaled: inverse(&std::array::from_fn(|k| challenges.psi[k] * challenges.taus[k])),
            };
            sumcheck::prove_with_zero_prefix(transcript, witness, weights, lane_start, claim, rounds, lookup, |rho| {
                let (weight, key) = weights::first_slice_at(&statements[0], &challenges, &folded, rho)?;
                prefix_key = Some(key);
                Ok(weight)
            })?
        } else {
            let start = Start {
                witness,
                weights,
                codes: &self.codes,
                beta: &challenges.beta,
                lookups: lookups(&tables, &challenges),
            };
            sumcheck::prove(transcript, start, claim, rounds)
        };
        let inverse_values = self.inverse_values(&tables, &rho);
        times.sumcheck = start.elapsed();

        let start = Instant::now();
        let uses = Uses::new(&self.layout, &challenges, &rho);
        let blocks = key::block_weights(&uses);
        let key_part = match prefix_key {
            Some(key) => key * uses.claims.iter().map(|claim| claim.coefficient).sum::<K>(),
            None => weights::key_part(&uses.form, &blocks, &folded),
        };
        absorb_extension(transcript, &[witness_value]);
        absorb_extension(transcript, &inverse_values);
        absorb_extension(transcript, &[key_part]);
        let column = draw(transcript, LOOKUP_VARS);
        let (key_rounds, mut key_point, folded_value) = sumcheck::prove_product(
            transcript,
            folded,
            &uses.form,
            &blocks,
            key_part,
            LANE_VARS + self.layout.key_block_vars(),
        );
        key_point.extend(row_point);
        let key_polynomial_value = folded_value * scale.inverse();
        absorb_extension(transcript, &[key_polynomial_value]);
        times.key = start.elapsed();

        let start = Instant::now();
        let inverse_value = weights::equality(&column, inverse_values.len())
            .iter()
            .zip(&inverse_values)
            .map(|(&weight, &value)| weight * value)
            .sum();
        let witness_commitment = self.committed.bytes.clone();
        let inverse_commitment = inverse.bytes.clone();
        let opening = self.key.akita.open(
            [
                (self.key.committed.clone(), &key_point, key_polynomial_value),
                (self.committed, &rho, witness_value),
                (inverse, &inverse_point(&self.layout, &rho, &column), inverse_value),
            ],
            &session(transcript, SESSION),
        )?;
        times.opening = start.elapsed();
        Ok((
            Proof {
                witness_commitment,
                inverse_commitment,
                rounds,
                witness_value,
                inverse_values,
                key_part,
                key_rounds,
                key_polynomial_value,
                opening,
            },
            times,
        ))
    }
}

#[cfg(test)]
#[path = "../tests/unit/cheating_prover.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/unit/transcript_order.rs"]
mod transcript_order;
