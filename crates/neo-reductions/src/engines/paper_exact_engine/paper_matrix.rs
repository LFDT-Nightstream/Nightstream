//! Literal expansion of verifier-owned CCS matrix descriptions.
//!
//! Compact descriptors are public storage formats. This module expands them
//! from their seeds and scalar parameters. It does not call their production
//! entry evaluators.

use neo_ccs::{CcsMatrix, SeededPhi81LinearBlock};
use neo_math::{Fq, D};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::paper_ring::PaperRing;

pub(super) fn matrix_entry<Ff>(matrix: &CcsMatrix<Ff>, row: usize, column: usize, ring: &PaperRing) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    if row >= matrix.rows() || column >= matrix.cols() {
        return Ff::ZERO;
    }
    match matrix {
        CcsMatrix::Identity { .. } => {
            if row == column {
                Ff::ONE
            } else {
                Ff::ZERO
            }
        }
        CcsMatrix::Csc(csc) => csc_entry(csc, row, column),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut value = csc_entry(csc, row, column);
            for block in blocks {
                value += seeded_entry(block, row, column, ring);
            }
            for run in geometric_runs {
                if row == run.row() && column >= run.column_start() && column < run.column_start() + run.len() {
                    let mut coefficient = *run.initial();
                    for _ in run.column_start()..column {
                        coefficient *= *run.ratio();
                    }
                    value += coefficient;
                }
            }
            value
        }
    }
}

fn csc_entry<Ff>(matrix: &neo_ccs::CscMat<Ff>, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut value = Ff::ZERO;
    for entry in matrix.column_range(column) {
        if matrix.row_index(entry) == row {
            value += matrix.vals[entry];
        }
    }
    value
}

fn seeded_entry<Ff>(block: &SeededPhi81LinearBlock, row: usize, column: usize, ring: &PaperRing) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    if !block.has_superneo_transformed_columns() {
        return seeded_original_entry(block, row, column);
    }

    let column_block = column / D;
    let mut original = [Fq::ZERO; D];
    for (lane, coefficient) in original.iter_mut().enumerate() {
        *coefficient =
            Fq::from_u64(seeded_original_entry::<Ff>(block, row, column_block * D + lane).as_canonical_u64());
    }
    Ff::from_u64(ring.bar_block(original)[column % D].as_canonical_u64())
}

fn seeded_original_entry<Ff>(block: &SeededPhi81LinearBlock, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    if row < block.row_start() || row >= block.row_start() + D * block.kappa() {
        return Ff::ZERO;
    }
    let local_row = row - block.row_start();
    let output = local_row / D;
    let coordinate = local_row % D;
    let mut value = Ff::ZERO;

    for (chunk, &seed) in block.chunk_seeds_by_row()[output].iter().enumerate() {
        let start = chunk * block.chunk_size();
        let end = core::cmp::min(block.message_cols(), start + block.chunk_size());
        let mut rng = ChaCha8Rng::from_seed(seed);
        for message_column in start..end {
            let mut rotation = sample_coefficients::<Ff>(&mut rng);
            for message_row in 0..D {
                let bit = message_row * block.message_cols() + message_column;
                if bit < block.word_starts().len() * block.word_width() {
                    let candidate = block.word_starts()[bit / block.word_width()] + bit % block.word_width();
                    if candidate == column {
                        value += rotation[coordinate];
                    }
                }
                rotation = rotate_phi81(rotation);
            }
        }
    }
    value
}

fn sample_coefficients<Ff>(rng: &mut ChaCha8Rng) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let mut bytes = [0u8; D * 8];
    rng.fill_bytes(&mut bytes);
    core::array::from_fn(|index| {
        let start = index * 8;
        let sample = u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("eight-byte coefficient"),
        );
        Ff::from_u64(if sample < Fq::ORDER_U64 {
            sample
        } else {
            sample_goldilocks(rng)
        })
    })
}

fn sample_goldilocks(rng: &mut ChaCha8Rng) -> u64 {
    loop {
        let sample = rng.next_u64();
        if sample < Fq::ORDER_U64 {
            return sample;
        }
    }
}

fn rotate_phi81<Ff>(current: [Ff; D]) -> [Ff; D]
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    let last = current[D - 1];
    let mut next = [Ff::ZERO; D];
    next[0] = -last;
    next[1..].copy_from_slice(&current[..D - 1]);
    next[27] -= last;
    next
}
