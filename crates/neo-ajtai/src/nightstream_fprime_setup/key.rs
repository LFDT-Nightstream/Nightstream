//! Exact indexed SHAKE128 expansion with 8-byte rejection sampling. Two
//! independent key elements share the CPU's parallel permutation; scalar
//! backends keep the same framing.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use keccak::{Backend, BackendClosure, Keccak, ParState1600};
use neo_math::{field::Q, signed_sums::SplitRing};

use super::{element_input, D, ELEMENT_INPUT_BYTES};

/// The SHAKE128 rate of 168 bytes in 64-bit words.
const RATE_WORDS: usize = 21;
/// Lean `AjtaiSetupV1.keyWords`: one key element reads four blocks of the rate.
pub const KEY_WORDS: usize = 4 * RATE_WORDS;
const _: () = assert!(ELEMENT_INPUT_BYTES < 8 * RATE_WORDS);
const _: () = assert!(D <= KEY_WORDS);

/// Lean `AjtaiSetupV1.keptCoefficientNat`: coefficient `L` is the `L`-th word
/// below `q`, and 0 when fewer than `L + 1` words are below `q`.
pub fn kept_coefficients(words: &[u64; KEY_WORDS]) -> [u64; D] {
    let mut kept: [u64; D] = core::array::from_fn(|lane| words[lane]);
    // 59 of the 2^64 words are `q` or more, so the first `D` words are kept
    // for all but about one element in 2^52.
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

/// One counter for each thread, on its own cache lines: a thread that expands
/// key elements writes no line that another thread reads.
#[repr(align(128))]
struct Count(AtomicU64);

static COUNTS: [Count; 64] = [const { Count(AtomicU64::new(0)) }; 64];
static THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static COUNT: usize = THREADS.fetch_add(1, Ordering::Relaxed) % COUNTS.len();
}

/// Each expansion of key elements records its SHAKE128 output bytes here.
pub(super) fn count_expanded(bytes: usize) {
    COUNT.with(|&count| COUNTS[count].0.fetch_add(bytes as u64, Ordering::Relaxed));
}

/// The SHAKE128 output bytes of the key elements that this process expanded.
/// A test reads it before and after a call to show that the call expands no
/// key element.
pub fn expanded_key_bytes() -> u64 {
    COUNTS
        .iter()
        .map(|count| count.0.load(Ordering::Relaxed))
        .sum()
}

/// The first `KEY_WORDS` 64-bit words of the SHAKE128 output of each element.
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
                for (index, byte) in input.into_iter().enumerate() {
                    state[index / 8] ^= u64::from(byte) << (8 * (index % 8));
                }
                // SHAKE domain suffix and multi-rate padding, unchanged from sha3::Shake128.
                state[ELEMENT_INPUT_BYTES / 8] ^= 0x1f << (8 * (ELEMENT_INPUT_BYTES % 8));
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
