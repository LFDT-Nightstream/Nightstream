//! Exact indexed SHAKE128 expansion. Two independent key elements share
//! the CPU's parallel permutation; scalar backends keep the same framing.

use keccak::{Backend, BackendClosure, Keccak, ParState1600};
use neo_math::{field::Q, signed_sums::SplitRing, Fq};
use p3_field::PrimeField64;

use super::{element_input, D, ELEMENT_INPUT_BYTES};

/// The SHAKE128 rate of 168 bytes, and the 32 bytes of each lane, in 64-bit words.
const RATE_WORDS: usize = 21;
const OUTPUT_WORDS: usize = 4 * D;
const _: () = assert!(ELEMENT_INPUT_BYTES < 8 * RATE_WORDS);

/// `2^64 mod q`, and `2^(64·i) mod q` for the four words of a coefficient.
const FOLD: u64 = Q.wrapping_neg();
const WORD_WEIGHTS: [u64; 4] = [1, FOLD, FOLD * FOLD, FOLD * FOLD * FOLD];
/// The weighted sum of four words is below `2^64` times this sum.
const _: () = assert!(WORD_WEIGHTS[0] + WORD_WEIGHTS[1] + WORD_WEIGHTS[2] + WORD_WEIGHTS[3] < 1 << 32);

// ARM SHA3 processes two independent 64-bit lanes in each vector register.
pub(super) fn split_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [SplitRing; 2] {
    word_pair(seed, row, columns).map(|words| {
        SplitRing::from_wide256(core::array::from_fn(|lane| {
            core::array::from_fn(|half| (words[4 * lane + half / 2] >> (32 * (half % 2))) as u32)
        }))
    })
}

pub(super) fn coefficient_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[u64; D]; 2] {
    word_pair(seed, row, columns).map(|words| {
        core::array::from_fn(|lane| {
            let wide: u128 = words[4 * lane..][..4]
                .iter()
                .zip(WORD_WEIGHTS)
                .map(|(&word, weight)| u128::from(word) * u128::from(weight))
                .sum();
            (Fq::new(wide as u64) + Fq::new((wide >> 64) as u64 * FOLD)).as_canonical_u64()
        })
    })
}

/// The output of each element as the sponge's 64-bit words: lane `L` is the
/// little-endian integer of words `4L..4L + 4`.
fn word_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[u64; OUTPUT_WORDS]; 2] {
    let mut words = [[0u64; OUTPUT_WORDS]; 2];
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
    output: &'a mut [[u64; OUTPUT_WORDS]; 2],
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
            for start in (0..OUTPUT_WORDS).step_by(RATE_WORDS) {
                permute(&mut states);
                let length = RATE_WORDS.min(OUTPUT_WORDS - start);
                for (state, output) in states.iter().zip(outputs.iter_mut()) {
                    output[start..start + length].copy_from_slice(&state[..length]);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/key_pair.rs"]
mod tests;
