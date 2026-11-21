use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use keccak::{Backend, BackendClosure, Keccak, ParState1600};
use neo_math::{field::Q, signed_sums::SplitRing};

use super::{element_input, D, ELEMENT_INPUT_BYTES};

/// The SHAKE128 rate of 168 bytes in 64-bit words.
const RATE_WORDS: usize = 21;
/// Lean `AjtaiSetupV1.keyWords`.
pub const KEY_WORDS: usize = 4 * RATE_WORDS;
const _: () = assert!(ELEMENT_INPUT_BYTES < 8 * RATE_WORDS);
const _: () = assert!(D <= KEY_WORDS);

/// Lean `AjtaiSetupV1.keptCoefficientNat`.
pub fn kept_coefficients(words: &[u64; KEY_WORDS]) -> [u64; D] {
    let mut kept: [u64; D] = core::array::from_fn(|lane| words[lane]);
    if kept
        .iter()
        .fold(false, |rejected, &word| rejected | (word >= Q))
    {
        kept = [0; D];
        for (coefficient, &word) in kept.iter_mut().zip(words.iter().filter(|&&word| word < Q)) {
            *coefficient = word;
        }
    }
    kept
}

// ARM SHA3 processes two independent 64-bit lanes in each vector register.
pub(super) fn split_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [SplitRing; 2] {
    word_pair(seed, row, columns).map(|words| SplitRing::from_u64(&kept_coefficients(&words)))
}

pub(super) fn coefficient_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[u64; D]; 2] {
    word_pair(seed, row, columns).map(|words| kept_coefficients(&words))
}

#[repr(align(128))]
struct CacheLinePaddedCount(AtomicU64);

static COUNTS: [CacheLinePaddedCount; 64] = [const { CacheLinePaddedCount(AtomicU64::new(0)) }; 64];
static THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static COUNT: usize = THREADS.fetch_add(1, Ordering::Relaxed) % COUNTS.len();
}

pub(super) fn count_expanded(bytes: usize) {
    COUNT.with(|&count| COUNTS[count].0.fetch_add(bytes as u64, Ordering::Relaxed));
}

pub fn expanded_key_bytes() -> u64 {
    COUNTS
        .iter()
        .map(|count| count.0.load(Ordering::Relaxed))
        .sum()
}

fn word_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[u64; KEY_WORDS]; 2] {
    count_expanded(2 * 8 * KEY_WORDS);
    let mut words = [[0u64; KEY_WORDS]; 2];
    Keccak::new().with_backend(KeyPair {
        seed,
        row,
        columns,
        output: &mut words,
    });
    words
}

struct KeyPair<'a> {
    seed: &'a [u8; 32],
    row: u32,
    columns: [u64; 2],
    output: &'a mut [[u64; KEY_WORDS]; 2],
}

impl BackendClosure for KeyPair<'_> {
    fn call_once<B: Backend>(self) {
        let mut states = ParState1600::<B>::default();
        let width = states.len();
        let permute = B::get_par_f1600();
        for (columns, outputs) in self
            .columns
            .chunks(width)
            .zip(self.output.chunks_mut(width))
        {
            states.fill([0; 25]);
            for (state, &column) in states.iter_mut().zip(columns) {
                let input = element_input(self.seed, self.row, column);
                let (words, tail) = input.as_chunks::<8>();
                for (output, &word) in state.iter_mut().zip(words) {
                    *output = u64::from_le_bytes(word);
                }
                let mut last = [0; 8];
                last[..tail.len()].copy_from_slice(tail);
                // SHAKE128 domain suffix.
                last[tail.len()] = 0x1f;
                state[words.len()] = u64::from_le_bytes(last);
                state[RATE_WORDS - 1] ^= 0x80 << 56;
            }
            for start in (0..KEY_WORDS).step_by(RATE_WORDS) {
                permute(&mut states);
                for (state, output) in states.iter().zip(outputs.iter_mut()) {
                    output[start..start + RATE_WORDS].copy_from_slice(&state[..RATE_WORDS]);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/key_pair.rs"]
mod tests;
