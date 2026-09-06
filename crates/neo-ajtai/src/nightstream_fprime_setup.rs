//! Nightstream F-prime's verifier-owned indexed Ajtai setup.
//!
//! This module owns the Rust implementation of
//! `nightstream-ajtai-chacha20-wide256-v1`. Lean owns its semantics and
//! authority framing.

use neo_math::ring::D;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use crate::{AjtaiError, AjtaiResult, Commitment};

const GOLDILOCKS_MODULUS: u128 = 18_446_744_069_414_584_321;
const WORD_RADIX: u128 = 1_u128 << 32;

pub const SETUP_ID: &[u8] = b"nightstream-ajtai-chacha20-wide256-v1";
pub const PRODUCTION_VERIFIER_ROWS: u64 = 22;
pub const PRODUCTION_MESSAGE_COLUMNS: u64 = 4_750_596;
pub const PRODUCTION_CARRIER_WIDTH: usize = PRODUCTION_MESSAGE_COLUMNS as usize * D;
pub const PRODUCTION_SEED: [u8; 32] = [
    252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146, 215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231,
    189, 76, 31, 30, 55, 23, 2,
];

const _: () = assert!(D == 54);
// A raw convolution degree has at most 54 terms from each message column.
// Each sign partition therefore fits in 92 bits, before any field reduction.
const _: () = assert!(PRODUCTION_MESSAGE_COLUMNS as u128 * D as u128 * (GOLDILOCKS_MODULUS - 1) < (1_u128 << 92));

fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

/// One RFC-8439 block with nonce `row_u32_le || block_u64_le`.
pub fn block_words(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> [u32; 16] {
    let mut state = [0_u32; 16];
    state[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
    for (word, bytes) in state[4..12].iter_mut().zip(seed.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().expect("four-byte key word"));
    }
    state[12] = lane;
    state[13] = row;
    state[14] = block as u32;
    state[15] = (block >> 32) as u32;

    let initial = state;
    for _ in 0..10 {
        quarter_round(&mut state, 0, 4, 8, 12);
        quarter_round(&mut state, 1, 5, 9, 13);
        quarter_round(&mut state, 2, 6, 10, 14);
        quarter_round(&mut state, 3, 7, 11, 15);
        quarter_round(&mut state, 0, 5, 10, 15);
        quarter_round(&mut state, 1, 6, 11, 12);
        quarter_round(&mut state, 2, 7, 8, 13);
        quarter_round(&mut state, 3, 4, 9, 14);
    }
    for (word, original) in state.iter_mut().zip(initial) {
        *word = word.wrapping_add(original);
    }
    state
}

/// Reduce the first 256 ChaCha20 output bits modulo the Goldilocks prime.
pub fn coefficient(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> u64 {
    let words = block_words(seed, row, block, lane);
    let reduced = words[..8].iter().rev().fold(0_u128, |value, word| {
        (value * WORD_RADIX + u128::from(*word)) % GOLDILOCKS_MODULUS
    });
    reduced as u64
}

/// Stream the 54 coefficients of one exact indexed key element.
///
/// The RNG's high counter word holds the RFC-8439 nonce row, and its stream
/// identifier holds the nonce block. One full 64-byte block is consumed per
/// lane; only its first 256 bits enter that lane's wide reduction.
pub fn coefficient_block(seed: &[u8; 32], row: u32, block: u64) -> [u64; D] {
    let mut rng = ChaCha20Rng::from_seed(*seed);
    rng.set_stream(block);
    rng.set_word_pos(u128::from(row) << 36);
    let mut bytes = [0_u8; D * 64];
    rng.fill_bytes(&mut bytes);
    core::array::from_fn(|lane| {
        bytes[lane * 64..lane * 64 + 32]
            .chunks_exact(4)
            .rev()
            .fold(0_u128, |value, word| {
                let word = u32::from_le_bytes(word.try_into().expect("four-byte coefficient word"));
                (value * WORD_RADIX + u128::from(word)) % GOLDILOCKS_MODULUS
            }) as u64
    })
}

struct SignedBlock {
    index: u64,
    positive: u64,
    negative: u64,
}

fn add_shifted_coefficients(sum: &mut [u128; 2 * D - 1], mut positions: u64, coefficients: &[u64; D]) {
    while positions != 0 {
        let shift = positions.trailing_zeros() as usize;
        for (lane, coefficient) in coefficients.iter().copied().enumerate() {
            sum[shift + lane] += u128::from(coefficient);
        }
        positions &= positions - 1;
    }
}

fn commit_row(row: u32, blocks: &[SignedBlock], output: &mut [Goldilocks]) {
    let mut positive = [0_u128; 2 * D - 1];
    let mut negative = [0_u128; 2 * D - 1];
    for block in blocks {
        let coefficients = coefficient_block(&PRODUCTION_SEED, row, block.index);
        add_shifted_coefficients(&mut positive, block.positive, &coefficients);
        add_shifted_coefficients(&mut negative, block.negative, &coefficients);
    }
    let raw: [Goldilocks; 2 * D - 1] = core::array::from_fn(|degree| {
        let value = (positive[degree] % GOLDILOCKS_MODULUS + GOLDILOCKS_MODULUS
            - negative[degree] % GOLDILOCKS_MODULUS)
            % GOLDILOCKS_MODULUS;
        Goldilocks::from_u64(value as u64)
    });
    // X^54 = -X^27 - 1 and X^81 = 1. Reduce only after summing all blocks.
    for lane in 0..D {
        output[lane] = if lane < D / 2 {
            let high = if lane + 81 < raw.len() {
                raw[lane + 81]
            } else {
                Goldilocks::ZERO
            };
            raw[lane] - raw[lane + D] + high
        } else {
            raw[lane] - raw[lane + D / 2]
        };
    }
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
    let mut commitment = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let rows = commitment.data.par_chunks_mut(D);
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let rows = commitment.data.chunks_mut(D);
    rows.enumerate()
        .for_each(|(row, output)| commit_row(row as u32, &blocks, output));
    Ok(commitment)
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
