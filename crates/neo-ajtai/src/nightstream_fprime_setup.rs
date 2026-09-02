//! Nightstream F-prime's verifier-owned indexed Ajtai setup.
//!
//! This module owns the Rust implementation of
//! `nightstream-ajtai-chacha20-wide256-v1`. Lean owns its semantics and
//! authority framing.

const GOLDILOCKS_MODULUS: u128 = 18_446_744_069_414_584_321;
const WORD_RADIX: u128 = 1_u128 << 32;

pub const SETUP_ID: &[u8] = b"nightstream-ajtai-chacha20-wide256-v1";
pub const PRODUCTION_VERIFIER_ROWS: u64 = 22;
pub const PRODUCTION_MESSAGE_COLUMNS: u64 = 4_900_509;
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
