use std::{borrow::Borrow, ops::Range};

#[cfg(not(target_arch = "wasm32"))]
mod crt;
mod expanded;
mod key;

pub use expanded::ExpandedKey;
#[doc(hidden)]
pub use key::expanded_key_bytes;
pub use key::{kept_coefficients, KEY_WORDS};

use neo_ccs::Mat;
use neo_math::{
    balanced::to_balanced_i128,
    ring::D,
    signed_sums::{SignedShiftSums, SplitRing, LIMB_BOUND, MAX_EXACT_TERMS},
    Fq,
};
use p3_field::PrimeCharacteristicRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};

use crate::{AjtaiError, AjtaiResult, Commitment};

pub const SETUP_ID: &[u8] = b"nightstream-ajtai-shake128-reject64-v1";
pub const ELEMENT_INPUT_BYTES: usize = SETUP_ID.len() + 32 + 4 + 8;
pub const ELEMENT_OUTPUT_BYTES: usize = 8 * KEY_WORDS;
pub const PRODUCTION_VERIFIER_ROWS: u64 = neo_params::nightstream_p64_k14::KAPPA as u64;
/// Lean `Poseidon2HashChainV1Setup.messageColumns_eq` and
/// `Poseidon2HashChainV1Package.logicalWidth`.
pub const PRODUCTION_MESSAGE_COLUMNS: u64 = 41_106_674_u64.div_ceil(D as u64);
pub const PRODUCTION_CARRIER_WIDTH: usize = PRODUCTION_MESSAGE_COLUMNS as usize * D;
/// The whole ring columns in the 254,260,620 coefficients of the matrix of
/// `decisions/fprime-ajtai-public-seed-msis-assumption.md`, below the width `2^28` of
/// `decisions/field-2p64-59-ring-x64.md`.
pub const MAX_MESSAGE_COLUMNS: u64 = 254_260_620 / D as u64;
pub const MAX_CARRIER_WIDTH: usize = MAX_MESSAGE_COLUMNS as usize * D;
pub const PRODUCTION_SEED: [u8; 32] = [
    252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146, 215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231,
    189, 76, 31, 30, 55, 23, 2,
];

const _: () = assert!(MAX_MESSAGE_COLUMNS == 3_972_822);
const _: () = assert!(PRODUCTION_MESSAGE_COLUMNS <= MAX_MESSAGE_COLUMNS);
const _: () = assert!(MAX_MESSAGE_COLUMNS <= MAX_EXACT_TERMS);
const _: () = assert!(MAX_MESSAGE_COLUMNS as u128 * D as u128 * LIMB_BOUND < (1_u128 << 63));

pub fn element_input(seed: &[u8; 32], row: u32, block: u64) -> [u8; ELEMENT_INPUT_BYTES] {
    let mut input = [0_u8; ELEMENT_INPUT_BYTES];
    let (setup_id, fields) = input.split_at_mut(SETUP_ID.len());
    setup_id.copy_from_slice(SETUP_ID);
    fields[..32].copy_from_slice(seed);
    fields[32..36].copy_from_slice(&row.to_le_bytes());
    fields[36..].copy_from_slice(&block.to_le_bytes());
    input
}

pub fn element_bytes(seed: &[u8; 32], row: u32, block: u64) -> [u8; ELEMENT_OUTPUT_BYTES] {
    let mut xof = Shake128::default();
    xof.update(&element_input(seed, row, block));
    let mut bytes = [0_u8; ELEMENT_OUTPUT_BYTES];
    key::count_expanded(bytes.len());
    xof.finalize_xof().read(&mut bytes);
    bytes
}

pub fn coefficient(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> u64 {
    coefficient_block(seed, row, block)[lane as usize]
}

pub fn production_key_pair(row: u32, columns: [u64; 2]) -> [[u64; D]; 2] {
    key::coefficient_pair(&PRODUCTION_SEED, row, columns)
}

pub fn coefficient_block(seed: &[u8; 32], row: u32, block: u64) -> [u64; D] {
    let bytes = element_bytes(seed, row, block);
    kept_coefficients(&core::array::from_fn(|word| {
        u64::from_le_bytes(
            bytes[8 * word..8 * word + 8]
                .try_into()
                .expect("eight-byte word"),
        )
    }))
}

pub struct SignedBlock {
    index: u64,
    positive: u64,
    negative: u64,
}

impl SignedBlock {
    pub fn index(&self) -> u64 {
        self.index
    }

    pub fn positive(&self) -> u64 {
        self.positive
    }

    pub fn negative(&self) -> u64 {
        self.negative
    }
}

/// SPEC.md Algorithm 1.
fn with_elements<T>(
    row: u32,
    items: &[T],
    column: impl Fn(&T) -> u64,
    key: Option<&ExpandedKey>,
    mut add: impl FnMut(&T, &SplitRing),
) {
    let expanded = match key {
        Some(key) => {
            let (kept, expanded) = items.split_at(items.partition_point(|item| column(item) < key.blocks()));
            for item in kept {
                add(item, &key.element(row, column(item)));
            }
            expanded
        }
        None => items,
    };
    for pair in expanded.chunks(2) {
        let columns = [column(&pair[0]), column(pair.last().expect("nonempty pair"))];
        for (item, element) in pair
            .iter()
            .zip(key::split_pair(&PRODUCTION_SEED, row, columns))
        {
            add(item, &element);
        }
    }
}

fn signed_sums(row: u32, blocks: &[SignedBlock], key: Option<&ExpandedKey>) -> SignedShiftSums {
    let mut sums = SignedShiftSums::zero();
    with_elements(row, blocks, SignedBlock::index, key, |block, element| {
        sums.add_signed_units(element, block.positive, block.negative);
    });
    sums
}

const TASK_COLUMNS: u64 = 1 << 12;

fn column_slice<'a>(blocks: &'a [SignedBlock], columns: &Range<u64>) -> &'a [SignedBlock] {
    let start = blocks.partition_point(|block| block.index < columns.start);
    let end = blocks.partition_point(|block| block.index < columns.end);
    &blocks[start..end]
}

fn row_totals(
    column_count: u64,
    sums_for: impl Fn(u32, Range<u64>) -> Vec<SignedShiftSums> + Sync,
) -> Vec<Vec<SignedShiftSums>> {
    let ranges = column_count.div_ceil(TASK_COLUMNS);
    let tasks = 0..PRODUCTION_VERIFIER_ROWS * ranges;
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let tasks = tasks.into_par_iter();
    let partial: Vec<_> = tasks
        .map(|task| {
            let start = task % ranges * TASK_COLUMNS;
            sums_for((task / ranges) as u32, start..(start + TASK_COLUMNS).min(column_count))
        })
        .collect();
    partial
        .chunks(ranges as usize)
        .map(|row| {
            let mut total = row[0].clone();
            for part in &row[1..] {
                for (total, part) in total.iter_mut().zip(part) {
                    total.add(part);
                }
            }
            total
        })
        .collect()
}

pub fn commit_production_signed_units(carrier: &[i8]) -> AjtaiResult<Commitment> {
    if carrier.len() != PRODUCTION_CARRIER_WIDTH {
        return Err(AjtaiError::SizeMismatch {
            expected: PRODUCTION_CARRIER_WIDTH,
            actual: carrier.len(),
        });
    }
    let mut blocks = Vec::new();
    for (index, coordinates) in carrier.chunks_exact(D).enumerate() {
        let mut positive = 0_u64;
        let mut negative = 0_u64;
        for (lane, value) in coordinates.iter().copied().enumerate() {
            match value {
                0 => {}
                1 => positive |= 1_u64 << lane,
                -1 => negative |= 1_u64 << lane,
                _ => {
                    return Err(AjtaiError::RangeViolation {
                        value: i128::from(value),
                        bound: 2,
                    })
                }
            }
        }
        if positive != 0 || negative != 0 {
            blocks.push(SignedBlock {
                index: index as u64,
                positive,
                negative,
            });
        }
    }
    Ok(commit_signed_blocks(&blocks))
}

pub fn commit_production_signed_unit_matrix(witness: &Mat<Fq>) -> AjtaiResult<Commitment> {
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    if witness.rows() != D || witness.cols() != columns {
        return Err(AjtaiError::InvalidDimensions(format!(
            "production signed-unit matrix must be {D}x{columns}, got {}x{}",
            witness.rows(),
            witness.cols()
        )));
    }
    commit_production_signed_unit_prefix_matrix(witness)
}

/// Lean `AjtaiSetupV1.Prefix.commit_zeroExtend`.
pub fn commit_production_signed_unit_prefix_matrix(witness: &Mat<Fq>) -> AjtaiResult<Commitment> {
    let blocks = signed_unit_prefix_blocks(witness)?;
    Ok(commit_signed_blocks(&blocks))
}

pub fn signed_unit_prefix_blocks(witness: &Mat<Fq>) -> AjtaiResult<Vec<SignedBlock>> {
    let columns = witness.cols();
    if witness.rows() != D || columns == 0 || columns > MAX_MESSAGE_COLUMNS as usize {
        return Err(AjtaiError::InvalidDimensions(format!(
            "production key prefix requires {D} rows and 1..={} columns, got {}x{}",
            MAX_MESSAGE_COLUMNS,
            witness.rows(),
            columns
        )));
    }
    if witness.virtual_constant_value() == Some(&Fq::ZERO) {
        return Ok(Vec::new());
    }
    let masks = witness.packed_signed_unit_column_masks();
    let mut blocks = Vec::new();
    for index in 0..columns {
        let (positive, negative) = if let Some((positive, negative)) = masks {
            (positive[index], negative[index])
        } else {
            let mut positive = 0_u64;
            let mut negative = 0_u64;
            for lane in 0..D {
                let value = witness[(lane, index)];
                if value == Fq::ONE {
                    positive |= 1_u64 << lane;
                } else if value == -Fq::ONE {
                    negative |= 1_u64 << lane;
                } else if value != Fq::ZERO {
                    return Err(AjtaiError::RangeViolation {
                        value: to_balanced_i128(value),
                        bound: 2,
                    });
                }
            }
            (positive, negative)
        };
        if positive != 0 || negative != 0 {
            blocks.push(SignedBlock {
                index: index as u64,
                positive,
                negative,
            });
        }
    }
    Ok(blocks)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid signed-unit witness {witness_index}: {source}")]
pub struct SignedUnitBatchError {
    witness_index: usize,
    #[source]
    source: AjtaiError,
}

impl SignedUnitBatchError {
    pub fn witness_index(&self) -> usize {
        self.witness_index
    }
    pub fn error(&self) -> &AjtaiError {
        &self.source
    }
    pub fn into_error(self) -> AjtaiError {
        self.source
    }
}

pub fn commit_production_signed_unit_prefix_matrices<W: Borrow<Mat<Fq>>>(
    witnesses: &[W],
) -> Result<Vec<Commitment>, SignedUnitBatchError> {
    let blocks = witnesses
        .iter()
        .enumerate()
        .map(|(witness_index, witness)| {
            signed_unit_prefix_blocks(witness.borrow()).map_err(|source| SignedUnitBatchError { witness_index, source })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut commitments: Vec<_> = (0..witnesses.len())
        .map(|_| Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize))
        .collect();
    if blocks.iter().all(Vec::is_empty) {
        return Ok(commitments);
    }
    let column_count = blocks
        .iter()
        .filter_map(|blocks| blocks.last())
        .map(|block| block.index + 1)
        .max()
        .expect("a nonempty witness");
    let mut present = vec![false; column_count as usize];
    for block in blocks.iter().flatten() {
        present[block.index as usize] = true;
    }
    let columns: Vec<_> = present
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index as u64))
        .collect();
    let key = ExpandedKey::shared();
    let totals = row_totals(column_count, |row, range| {
        let blocks: Vec<_> = blocks
            .iter()
            .map(|blocks| column_slice(blocks, &range))
            .collect();
        let start = columns.partition_point(|&column| column < range.start);
        let end = columns.partition_point(|&column| column < range.end);
        batch_sums(row, &columns[start..end], &blocks, key.as_deref())
    });
    for (row, totals) in totals.into_iter().enumerate() {
        for (commitment, total) in commitments.iter_mut().zip(totals) {
            commitment.col_mut(row).copy_from_slice(&total.reduce());
        }
    }
    Ok(commitments)
}

fn batch_sums(row: u32, columns: &[u64], blocks: &[&[SignedBlock]], key: Option<&ExpandedKey>) -> Vec<SignedShiftSums> {
    let mut sums = vec![SignedShiftSums::zero(); blocks.len()];
    let mut positions = vec![0usize; blocks.len()];
    with_elements(
        row,
        columns,
        |&column| column,
        key,
        |&column, element| {
            for ((witness, position), sum) in blocks.iter().zip(&mut positions).zip(&mut sums) {
                if let Some(block) = witness.get(*position).filter(|block| block.index == column) {
                    sum.add_signed_units(element, block.positive, block.negative);
                    *position += 1;
                }
            }
        },
    );
    sums
}

fn commit_signed_blocks(blocks: &[SignedBlock]) -> Commitment {
    let key = (!blocks.is_empty()).then(ExpandedKey::shared).flatten();
    commit_signed_blocks_with_key(blocks, key.as_deref())
}

fn commit_signed_blocks_with_key(blocks: &[SignedBlock], key: Option<&ExpandedKey>) -> Commitment {
    let mut commitment = Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize);
    let Some(last) = blocks.last() else {
        return commitment;
    };
    let columns = last.index + 1;

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(key) = key {
        let count = key.crt_tiles().min(columns.div_ceil(TASK_COLUMNS) as usize);
        let tiles: Vec<_> = (0..count)
            .into_par_iter()
            .map(|tile| {
                let start = tile as u64 * TASK_COLUMNS;
                let slice = column_slice(blocks, &(start..start + TASK_COLUMNS));
                if crt::use_tile(slice) {
                    key.crt_tile(tile).map(|key| (key, start, slice))
                } else {
                    None
                }
            })
            .collect();
        let selected = tiles.iter().flatten().copied().collect::<Vec<_>>();
        if !selected.is_empty() {
            let (scalar, transformed) = rayon::join(
                || {
                    row_totals(columns, |row, columns| {
                        let tile = (columns.start / TASK_COLUMNS) as usize;
                        let sums = if tiles.get(tile).is_some_and(Option::is_some) {
                            SignedShiftSums::zero()
                        } else {
                            signed_sums(row, column_slice(blocks, &columns), Some(key))
                        };
                        vec![sums]
                    })
                },
                || crt::sum_tiles(&selected),
            );
            for (row, (output, scalar)) in commitment
                .data_mut()
                .chunks_exact_mut(D)
                .zip(scalar)
                .enumerate()
            {
                for ((output, scalar), transformed) in output
                    .iter_mut()
                    .zip(scalar[0].reduce())
                    .zip(&transformed[row])
                {
                    *output = scalar + *transformed;
                }
            }
            return commitment;
        }
    }

    let totals = row_totals(columns, |row, columns| {
        vec![signed_sums(row, column_slice(blocks, &columns), key)]
    });
    for (output, totals) in commitment.data_mut().chunks_mut(D).zip(totals) {
        output.copy_from_slice(&totals[0].reduce());
    }
    commitment
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "../tests/unit/nightstream_crt.rs"]
mod crt_tests;

pub fn authority_words(verifier_rows: u64, message_columns: u64, seed: &[u8; 32]) -> Vec<u64> {
    let mut words = Vec::with_capacity(1 + SETUP_ID.len() + 3 + seed.len());
    words.push(SETUP_ID.len() as u64);
    words.extend(SETUP_ID.iter().copied().map(u64::from));
    words.extend([verifier_rows, message_columns, seed.len() as u64]);
    words.extend(seed.iter().copied().map(u64::from));
    words
}

pub fn production_authority_words() -> Vec<u64> {
    authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED)
}
