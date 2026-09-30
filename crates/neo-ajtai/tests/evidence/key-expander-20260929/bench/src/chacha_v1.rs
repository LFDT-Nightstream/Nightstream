//! The production key expander `nightstream-ajtai-chacha20-wide256-v1`, copied
//! unchanged from `crates/neo-ajtai/src/nightstream_fprime_setup.rs` at
//! 5bd5873f4 so that this dated evidence keeps measuring it after the
//! production setup changed.

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

pub const D: usize = 54;
const GOLDILOCKS_MODULUS: u128 = 18_446_744_069_414_584_321;
const WORD_RADIX: u128 = 1_u128 << 32;

pub const PRODUCTION_VERIFIER_ROWS: u64 = 22;
pub const PRODUCTION_MESSAGE_COLUMNS: u64 = 3_221_095;
pub const PRODUCTION_SEED: [u8; 32] = [
    252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146, 215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231,
    189, 76, 31, 30, 55, 23, 2,
];

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
    // All 54 lane blocks are evaluated together, one per array lane, so the
    // compiler can vectorize the ChaCha20 rounds across lanes.
    let mut state = [[0_u32; D]; 16];
    for (word, constant) in state[..4]
        .iter_mut()
        .zip([0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574])
    {
        *word = [constant; D];
    }
    for (word, bytes) in state[4..12].iter_mut().zip(seed.chunks_exact(4)) {
        *word = [u32::from_le_bytes(bytes.try_into().expect("four-byte key word")); D];
    }
    state[12] = core::array::from_fn(|lane| lane as u32);
    state[13] = [row; D];
    state[14] = [block as u32; D];
    state[15] = [(block >> 32) as u32; D];
    let words = first_words(&state);
    core::array::from_fn(|lane| reduce_first_words(core::array::from_fn(|word| words[word][lane])))
}

/// The first eight output words of each lane's ChaCha20 block.
#[inline(always)]
fn first_words(initial: &[[u32; D]; 16]) -> [[u32; D]; 8] {
    let mut state = *initial;
    for _ in 0..10 {
        quarter_round_lanes(&mut state, 0, 4, 8, 12);
        quarter_round_lanes(&mut state, 1, 5, 9, 13);
        quarter_round_lanes(&mut state, 2, 6, 10, 14);
        quarter_round_lanes(&mut state, 3, 7, 11, 15);
        quarter_round_lanes(&mut state, 0, 5, 10, 15);
        quarter_round_lanes(&mut state, 1, 6, 11, 12);
        quarter_round_lanes(&mut state, 2, 7, 8, 13);
        quarter_round_lanes(&mut state, 3, 4, 9, 14);
    }
    core::array::from_fn(|word| core::array::from_fn(|lane| state[word][lane].wrapping_add(initial[word][lane])))
}

#[inline(always)]
fn quarter_round_lanes(state: &mut [[u32; D]; 16], a: usize, b: usize, c: usize, d: usize) {
    for lane in 0..D {
        state[a][lane] = state[a][lane].wrapping_add(state[b][lane]);
        state[d][lane] = (state[d][lane] ^ state[a][lane]).rotate_left(16);
        state[c][lane] = state[c][lane].wrapping_add(state[d][lane]);
        state[b][lane] = (state[b][lane] ^ state[c][lane]).rotate_left(12);
        state[a][lane] = state[a][lane].wrapping_add(state[b][lane]);
        state[d][lane] = (state[d][lane] ^ state[a][lane]).rotate_left(8);
        state[c][lane] = state[c][lane].wrapping_add(state[d][lane]);
        state[b][lane] = (state[b][lane] ^ state[c][lane]).rotate_left(7);
    }
}

/// Reduce the first 256 output bits of one block modulo Goldilocks.
#[inline(always)]
fn reduce_first_words(words: [u32; 8]) -> u64 {
    let words = words.map(i64::from);
    // For x = 2^32, x^2 = x - 1 and x^6 = 1 modulo Goldilocks.
    // Each signed sum has magnitude at most 3 * (2^32 - 1), so i64
    // arithmetic is exact. The scalar coefficient keeps the division reference.
    let a = words[0] - words[2] - words[3] + words[5] + words[6];
    let b = words[1] + words[2] - words[4] - words[5] + words[7];
    (Goldilocks::from_i64(a) + Goldilocks::from_i64(b) * Goldilocks::from_u64(1_u64 << 32)).as_canonical_u64()
}
