//! Nightstream F-prime's verifier-owned indexed Ajtai setup.
//!
//! This module owns the Rust implementation of
//! `nightstream-ajtai-shake128-wide256-v1`. Key element `(row, block)` is
//! SHAKE128 of `setup ID ‖ seed ‖ row_u32_le ‖ block_u64_le`; lane `L` is
//! output bytes `32L..32L + 32`, reduced modulo Goldilocks as one
//! little-endian integer. Lean owns its semantics and authority framing.

use std::{borrow::Borrow, ops::Range};

mod key;

use neo_ccs::Mat;
use neo_math::{balanced::to_balanced_i128, ring::D, signed_sums::SignedShiftSums};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128,
};

use crate::{AjtaiError, AjtaiResult, Commitment};

const GOLDILOCKS_MODULUS: u128 = 18_446_744_069_414_584_321;

pub const SETUP_ID: &[u8] = b"nightstream-ajtai-shake128-wide256-v1";
/// One key element's SHAKE128 input: setup ID, seed, row and block.
pub const ELEMENT_INPUT_BYTES: usize = SETUP_ID.len() + 32 + 4 + 8;
pub const PRODUCTION_VERIFIER_ROWS: u64 = 22;
// Lean authority: Poseidon2HashChainV1Setup.messageColumns_eq.
pub const PRODUCTION_MESSAGE_COLUMNS: u64 = 1_106_153;
pub const PRODUCTION_CARRIER_WIDTH: usize = PRODUCTION_MESSAGE_COLUMNS as usize * D;
// Approved public-seed MSIS matrix; applications bind their exact key prefix.
// Lean authority: Poseidon2HashChainV1Setup.approvedMsis_carrierWidth.
pub const MAX_MESSAGE_COLUMNS: u64 = 4_708_530;
pub const MAX_CARRIER_WIDTH: usize = MAX_MESSAGE_COLUMNS as usize * D;
pub const PRODUCTION_SEED: [u8; 32] = [
    252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146, 215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231,
    189, 76, 31, 30, 55, 23, 2,
];

const _: () = assert!(D == 54);
const _: () = assert!(PRODUCTION_MESSAGE_COLUMNS <= MAX_MESSAGE_COLUMNS);
// A raw convolution degree has at most 54 terms from each message column.
// Each term adds one bounded integer limb of a key coefficient, so every signed
// half-sum fits in i64 before any field reduction.
const _: () = assert!(MAX_MESSAGE_COLUMNS as u128 * D as u128 * ((1_u128 << 32) + 3) < (1_u128 << 63));

/// The SHAKE128 input of one key element. Every field has a fixed length,
/// so distinct `(seed, row, block)` values give distinct inputs.
pub fn element_input(seed: &[u8; 32], row: u32, block: u64) -> [u8; ELEMENT_INPUT_BYTES] {
    let mut input = [0_u8; ELEMENT_INPUT_BYTES];
    let (setup_id, fields) = input.split_at_mut(SETUP_ID.len());
    setup_id.copy_from_slice(SETUP_ID);
    fields[..32].copy_from_slice(seed);
    fields[32..36].copy_from_slice(&row.to_le_bytes());
    fields[36..].copy_from_slice(&block.to_le_bytes());
    input
}

/// The SHAKE128 output of one key element: 32 bytes for each lane.
pub fn element_bytes(seed: &[u8; 32], row: u32, block: u64) -> [u8; 32 * D] {
    let mut xof = Shake128::default();
    xof.update(&element_input(seed, row, block));
    let mut bytes = [0_u8; 32 * D];
    xof.finalize_xof().read(&mut bytes);
    bytes
}

/// Reference coefficient for `lane < 54`: the lane's 32 output bytes as one
/// little-endian integer, reduced by division modulo the Goldilocks prime.
pub fn coefficient(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> u64 {
    let bytes = element_bytes(seed, row, block);
    let chunk = &bytes[32 * lane as usize..32 * (lane as usize + 1)];
    chunk.iter().rev().fold(0_u128, |value, byte| {
        (value * 256 + u128::from(*byte)) % GOLDILOCKS_MODULUS
    }) as u64
}

/// The 54 coefficients of one exact indexed key element.
pub fn coefficient_block(seed: &[u8; 32], row: u32, block: u64) -> [u64; D] {
    let bytes = element_bytes(seed, row, block);
    core::array::from_fn(|lane| {
        reduce_words(core::array::from_fn(|word| {
            let start = 32 * lane + 4 * word;
            u32::from_le_bytes(bytes[start..start + 4].try_into().expect("four-byte word"))
        }))
    })
}

/// Reduce eight little-endian 32-bit words, one 256-bit integer, modulo
/// Goldilocks.
#[inline(always)]
fn reduce_words(words: [u32; 8]) -> u64 {
    let words = words.map(i64::from);
    // For x = 2^32, x^2 = x - 1 and x^6 = 1 modulo Goldilocks.
    // Each signed sum has magnitude at most 3 * (2^32 - 1), so i64
    // arithmetic is exact. The scalar coefficient keeps the division reference.
    let a = words[0] - words[2] - words[3] + words[5] + words[6];
    let b = words[1] + words[2] - words[4] - words[5] + words[7];
    (Goldilocks::from_i64(a) + Goldilocks::from_i64(b) * Goldilocks::from_u64(1_u64 << 32)).as_canonical_u64()
}

/// One nonzero block of a validated signed-unit production-key prefix.
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

fn signed_sums(row: u32, blocks: &[SignedBlock]) -> SignedShiftSums {
    let mut sums = SignedShiftSums::zero();
    for pair in blocks.chunks(2) {
        let columns = [pair[0].index, pair.last().expect("nonempty pair").index];
        let keys = key::split_pair(&PRODUCTION_SEED, row, columns);
        for (block, key) in pair.iter().zip(keys) {
            sums.add_signed_units(&key, block.positive, block.negative);
        }
    }
    sums
}

/// Key columns per commitment task. The 22 key rows alone leave workers
/// idle, so each row is also split into column ranges.
const TASK_COLUMNS: u64 = 1 << 16;

/// The blocks whose key column lies in `columns`.
fn column_slice<'a>(blocks: &'a [SignedBlock], columns: &Range<u64>) -> &'a [SignedBlock] {
    let start = blocks.partition_point(|block| block.index < columns.start);
    let end = blocks.partition_point(|block| block.index < columns.end);
    &blocks[start..end]
}

/// Run `sums_for` for every key row and column range, then add each row's
/// partial sums. Raw sums are exact integers, so the split does not change
/// the reduced commitment.
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

/// Commit the complete signed-unit carrier with the fixed production key.
///
/// Coordinates are contiguous 54-lane ring blocks. The caller supplies the
/// full carrier, including any application alignment zeros. Every coordinate
/// is checked before key expansion. Zero ring blocks contribute nothing and
/// are skipped by their exact indexed key address. No dense key is stored.
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

/// Commit a complete signed-unit witness matrix with the fixed production key.
/// Validated column masks and virtual zero need no dense conversion. Every
/// fallback coefficient is range-checked before any key element is expanded.
pub fn commit_production_signed_unit_matrix(witness: &Mat<Goldilocks>) -> AjtaiResult<Commitment> {
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

/// Commit a complete signed-unit carrier under a prefix of the production key.
///
/// The seed, row count and every retained key address stay unchanged. The
/// carrier must contain whole degree-54 blocks and cannot exceed the approved
/// matrix. Its exact block count must be bound by the caller's verifier context.
/// No coordinates are inserted, and no larger witness is allocated.
///
/// Lean contract: `AjtaiSetupV1.Prefix.commit_zeroExtend` identifies this
/// commitment with the full-key commitment after suffix zero-extension.
pub fn commit_production_signed_unit_prefix_matrix(witness: &Mat<Goldilocks>) -> AjtaiResult<Commitment> {
    let blocks = signed_unit_prefix_blocks(witness)?;
    Ok(commit_signed_blocks(&blocks))
}

/// Validate every prefix coordinate and return nonzero blocks in key order.
/// Device backends use the same shape and norm checks as the CPU commitment.
pub fn signed_unit_prefix_blocks(witness: &Mat<Goldilocks>) -> AjtaiResult<Vec<SignedBlock>> {
    let columns = witness.cols();
    if witness.rows() != D || columns == 0 || columns > MAX_MESSAGE_COLUMNS as usize {
        return Err(AjtaiError::InvalidDimensions(format!(
            "production key prefix requires {D} rows and 1..={} columns, got {}x{}",
            MAX_MESSAGE_COLUMNS,
            witness.rows(),
            columns
        )));
    }
    if witness.virtual_constant_value() == Some(&Goldilocks::ZERO) {
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
                if value == Goldilocks::ONE {
                    positive |= 1_u64 << lane;
                } else if value == -Goldilocks::ONE {
                    negative |= 1_u64 << lane;
                } else if value != Goldilocks::ZERO {
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

/// The first invalid witness in an ordered fixed-key commitment batch.
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

/// Commit signed-unit witnesses under prefixes of the same fixed production key.
///
/// Every matrix receives the scalar prefix dimension and value checks before any
/// key expansion. Outputs retain input order; an empty batch has no commitments.
/// Different valid prefix widths use their original indexed key addresses.
/// Each nonzero block in the union is expanded once per key row, then shared by
/// all witnesses that use it. No dense key or extended witness is allocated.
pub fn commit_production_signed_unit_prefix_matrices<W: Borrow<Mat<Goldilocks>>>(
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
        .map(|_| Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize))
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
    // The union is independent of the key row. Compute it once instead of
    // merging witness columns again for each of the 22 key rows.
    let mut present = vec![false; column_count as usize];
    for block in blocks.iter().flatten() {
        present[block.index as usize] = true;
    }
    let columns: Vec<_> = present
        .into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index as u64))
        .collect();
    let totals = row_totals(column_count, |row, range| {
        let blocks: Vec<_> = blocks
            .iter()
            .map(|blocks| column_slice(blocks, &range))
            .collect();
        let start = columns.partition_point(|&column| column < range.start);
        let end = columns.partition_point(|&column| column < range.end);
        batch_sums(row, &columns[start..end], &blocks)
    });
    for (row, totals) in totals.into_iter().enumerate() {
        for (commitment, total) in commitments.iter_mut().zip(totals) {
            commitment.col_mut(row).copy_from_slice(&total.reduce());
        }
    }
    Ok(commitments)
}

/// Each key block is expanded once per row, then shared by every witness
/// that uses its column.
fn batch_sums(row: u32, columns: &[u64], blocks: &[&[SignedBlock]]) -> Vec<SignedShiftSums> {
    let mut sums = vec![SignedShiftSums::zero(); blocks.len()];
    let mut positions = vec![0usize; blocks.len()];
    for pair in columns.chunks(2) {
        let keys = key::split_pair(&PRODUCTION_SEED, row, [pair[0], *pair.last().expect("nonempty pair")]);
        for (&column, key) in pair.iter().zip(keys) {
            for ((witness, position), sum) in blocks.iter().zip(&mut positions).zip(&mut sums) {
                if let Some(block) = witness.get(*position).filter(|block| block.index == column) {
                    sum.add_signed_units(&key, block.positive, block.negative);
                    *position += 1;
                }
            }
        }
    }
    sums
}

fn commit_signed_blocks(blocks: &[SignedBlock]) -> Commitment {
    let mut commitment = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    let Some(last) = blocks.last() else {
        return commitment;
    };
    let totals = row_totals(last.index + 1, |row, columns| {
        vec![signed_sums(row, column_slice(blocks, &columns))]
    });
    for (output, totals) in commitment.data.chunks_mut(D).zip(totals) {
        output.copy_from_slice(&totals[0].reduce());
    }
    commitment
}

/// Canonical raw authority words before Poseidon2 context hashing.
pub fn authority_words(verifier_rows: u64, message_columns: u64, seed: &[u8; 32]) -> Vec<u64> {
    let mut words = Vec::with_capacity(1 + SETUP_ID.len() + 3 + seed.len());
    words.push(SETUP_ID.len() as u64);
    words.extend(SETUP_ID.iter().copied().map(u64::from));
    words.extend([verifier_rows, message_columns, seed.len() as u64]);
    words.extend(seed.iter().copied().map(u64::from));
    words
}

/// The sole verifier-owned setup authority for the Stage 1 hash-chain
/// package. Callers cannot select its dimensions or seed.
pub fn production_authority_words() -> Vec<u64> {
    authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED)
}
