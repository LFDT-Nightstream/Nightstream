use std::sync::Arc;
use std::time::{Duration, Instant};

use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::{Field, PrimeCharacteristicRing};
use rayon::prelude::*;

use crate::akita::{self, Polynomial};
use crate::key::{self, ProverKey};
use crate::lane_prefix::{LaneStart, Masks};
use crate::layout::{Layout, Slice, LANE_VARS, LOOKUPS};
use crate::lookup::{code, OUTSIDE};
use crate::sumcheck::{self, lookup::Slot};
use crate::transcript::{absorb_bytes, absorb_extension, session};
use crate::weights::{self, Uses};
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
    codes: Arc<Vec<u16>>,
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

fn signed_cell(positive: u64, negative: u64, lane: usize) -> F {
    F::from_u64(positive >> lane & 1) - F::from_u64(negative >> lane & 1)
}

fn write_signed_cells<T: Send>(cells: &mut [T], (positive, negative): (&[u64], &[u64]), map: impl Fn(F) -> T + Sync) {
    cells
        .par_chunks_mut(D)
        .zip(positive.par_iter().zip(negative))
        .with_min_len(CELLS_PER_TASK / D)
        .for_each(|(block, (&positive, &negative))| {
            for (lane, cell) in block.iter_mut().enumerate() {
                *cell = map(signed_cell(positive, negative, lane));
            }
        });
}

fn write_cells(cells: &mut [F], witness: &Mat<F>) {
    let blocks = witness.cols();
    if let Some(&value) = witness.virtual_constant_value() {
        cells.fill(value);
    } else if let Some(masks) = witness.packed_signed_unit_column_masks() {
        write_signed_cells(cells, masks, |value| value);
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
    if let Some((slice, code)) = zero_prefix {
        let fresh = witnesses[1];
        if let Some(columns) = fresh
            .packed_signed_unit_column_masks()
            .filter(|&masks| Masks::suitable(masks).is_some())
        {
            let end = layout.cells[1].witness.range().end;
            let (_, counts) = layout.cells[0].table.expect("the zero prefix has a table");
            let count_cell = counts.offset + usize::from(code);
            let count = F::from_u64(slice.len() as u64);
            let mut witness = vec![F::ZERO; layout.witness_len - end];
            witness[count_cell - end] = count;
            let committed = key.akita.commit_signed_witness(
                layout.shape(),
                layout.cells[1].witness.offset,
                columns,
                (count_cell, akita::base(count)),
            )?;
            return Ok(Committed {
                key,
                layout,
                witness,
                codes: Arc::default(),
                zero_prefix,
                packed_fresh: Some(fresh.clone()),
                committed,
                commit_witness: start.elapsed(),
            });
        }
    }
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
        codes: Arc::new(codes),
        zero_prefix,
        packed_fresh: None,
        committed,
        commit_witness: start.elapsed(),
    })
}

fn table_inverses(layout: &Layout, taus: &[F; LOOKUPS]) -> Result<Vec<Vec<F>>, Error> {
    layout
        .tables()
        .map(|(_, _, bound, _)| {
            let denominators: Vec<F> = (0..=2 * bound)
                .map(|i| {
                    let value = F::from_u64(i) - F::from_u64(bound);
                    taus.iter().map(|pole| *pole - value).product()
                })
                .collect();
            if denominators.contains(&F::ZERO) {
                return Err(Error::Challenge);
            }
            Ok(p3_field::batch_multiplicative_inverse(&denominators))
        })
        .collect()
}

impl Committed<'_> {
    pub fn commitment_to_bind(&self) -> &[u8] {
        &self.committed.bytes
    }

    pub fn prove(
        self,
        transcript: &mut Poseidon2Transcript,
        statements: &[Statement<'_>],
    ) -> Result<(Proof, ProverTimes), Error> {
        self.prove_with_backend(transcript, statements, false)
    }

    pub fn prove_with_backend(
        mut self,
        transcript: &mut Poseidon2Transcript,
        statements: &[Statement<'_>],
        metal_kernels: bool,
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
            self.key.akita.commit_inverse_runs(
                self.layout.shape(),
                vec![(slice.len(), akita::base(tables[0][usize::from(code)]))],
            )?
        } else if let [table] = tables.as_slice() {
            self.key.akita.commit_lookup_column(
                self.layout.shape(),
                1 << self.layout.table_witness_vars,
                Arc::clone(&self.codes),
                table.iter().copied().map(akita::base).collect(),
                metal_kernels,
            )?
        } else {
            let mut column = vec![akita::base(F::ZERO); self.layout.table_witness_len];
            for ((_, slice, _, _), table) in self.layout.tables().zip(&tables) {
                column[slice.range()]
                    .par_iter_mut()
                    .zip(&self.codes[slice.range()])
                    .for_each(|(cell, &code)| {
                        *cell = akita::base(table.get(usize::from(code)).copied().unwrap_or(F::ZERO));
                    });
            }
            self.key
                .akita
                .commit(self.layout.shape(), Polynomial::Inverse, column)?
        };
        times.commit_inverse = start.elapsed();
        absorb_bytes(transcript, &inverse.bytes);
        let challenges = challenges(transcript, &self.layout, taus);
        let (row_point, scale) = key::row_point(challenges.omega)?;

        let start = Instant::now();
        let masks = self.packed_fresh.as_ref().map(|fresh| {
            Masks::new(
                fresh
                    .packed_signed_unit_column_masks()
                    .expect("the lane witness was packed"),
            )
        });
        let witness = std::mem::take(&mut self.witness);
        let active_blocks = self
            .layout
            .cells
            .iter()
            .map(|cell| cell.blocks)
            .max()
            .expect("a witness claim");
        let fold_key = || {
            self.key
                .fold(challenges.omega, active_blocks, metal_kernels)
        };
        let (witness, weights, folded, lane_inputs) = if let (Some((slice, _)), Some(masks)) = (self.zero_prefix, masks)
        {
            let cells = self.layout.cells[1];
            let (source, folded) = rayon::join(
                || weights::LaneSource::new(&statements[1], &challenges, cells.blocks),
                fold_key,
            );
            let weights = weights::count_tail(&self.layout, &challenges, cells.witness.range().end)?;
            (witness, weights, folded, Some((masks, source?, slice.len() / D)))
        } else {
            let (weights, folded) = weights::table(
                statements,
                &self.layout,
                &challenges,
                fold_key,
                self.zero_prefix.is_some(),
            )?;
            (witness, weights, folded, None)
        };
        let lane_start =
            lane_inputs.map(|(masks, source, padded)| LaneStart::new(masks, &folded, source, padded, challenges.delta));
        times.weights = start.elapsed();

        let start = Instant::now();
        let mut prefix_key = None;
        let slots = self
            .layout
            .tables()
            .zip(&tables)
            .enumerate()
            .map(|(i, ((_, slice, bound, _), pool))| {
                let (codes, values) = if self.zero_prefix.is_some() {
                    (&[][..], &[][..])
                } else {
                    (&self.codes[slice.range()][..], &witness[slice.range()][..])
                };
                Slot::new(
                    slice,
                    bound,
                    codes,
                    pool,
                    [challenges.xi[i], challenges.lambda[i]],
                    values,
                    &challenges.beta,
                )
            })
            .collect();
        let restore = self.zero_prefix.map(|_| {
            |rho: &[K]| {
                let (weight, key) = weights::first_slice_at(&statements[0], &challenges, &folded, rho)?;
                prefix_key = Some(key);
                Ok(weight)
            }
        });
        let (rounds, rho, witness_value, shared_inverse) = sumcheck::lookup::prove(
            transcript,
            witness,
            weights,
            slots,
            &challenges,
            &self.layout,
            claimed_sum(statements, &challenges),
            lane_start,
            restore,
        )?;
        times.sumcheck = start.elapsed();

        let start = Instant::now();
        let uses = Uses::new(&self.layout, &challenges, &rho);
        let blocks = key::block_weights(&uses);
        let key_part = match prefix_key {
            Some(key) => key * uses.claims.iter().map(|claim| claim.coefficient).sum::<K>(),
            None => weights::key_part(&uses.form, &blocks, &folded),
        };
        absorb_extension(transcript, &[witness_value]);
        absorb_extension(transcript, &[shared_inverse]);
        absorb_extension(transcript, &[key_part]);

        let (key_rounds, mut key_point, folded_value) = sumcheck::prove_product(
            transcript,
            folded,
            &uses.form,
            &blocks,
            key_part,
            LANE_VARS + self.layout.key_block_vars(),
            |lanes, folded| {
                self.key
                    .fold_remaining_blocks(challenges.omega, lanes, folded)
            },
        );
        key_point.extend(row_point);
        let key_polynomial_value = folded_value * scale.inverse();
        absorb_extension(transcript, &[key_polynomial_value]);
        times.key = start.elapsed();

        let start = Instant::now();
        let witness_commitment = self.committed.bytes.clone();
        let inverse_commitment = inverse.bytes.clone();
        let opening = self.key.akita.open(
            [
                (self.key.committed.clone(), &key_point, key_polynomial_value),
                (self.committed, &rho, witness_value),
                (inverse, &inverse_point(&self.layout, &rho), shared_inverse),
            ],
            &session(transcript, SESSION),
            metal_kernels,
        )?;
        times.opening = start.elapsed();
        Ok((
            Proof {
                witness_commitment,
                inverse_commitment,
                rounds,
                witness_value,
                shared_inverse,
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
