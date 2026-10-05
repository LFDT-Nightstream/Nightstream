//! Exact indexed SHAKE128 expansion. Two independent key elements share
//! the CPU's parallel permutation; scalar backends keep the same framing.

use keccak::{Backend, BackendClosure, Keccak, ParState1600};
use neo_math::signed_sums::SplitRing;

use super::{element_input, D, ELEMENT_INPUT_BYTES};

const RATE: usize = 168;
const OUTPUT_BYTES: usize = 32 * D;
const _: () = assert!(ELEMENT_INPUT_BYTES < RATE);

// ARM SHA3 processes two independent 64-bit lanes in each vector register.
// Keeping this batch small also bounds the scalar backend's stack storage.
pub(super) fn split_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [SplitRing; 2] {
    word_pair(seed, row, columns).map(SplitRing::from_wide256)
}

#[cfg(test)]
pub(super) fn coefficient_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[u64; D]; 2] {
    word_pair(seed, row, columns).map(|words| words.map(super::reduce_words))
}

fn word_pair(seed: &[u8; 32], row: u32, columns: [u64; 2]) -> [[[u32; 8]; D]; 2] {
    let mut bytes = [[0u8; OUTPUT_BYTES]; 2];
    Keccak::new().with_backend(KeyPair {
        seed,
        row,
        columns,
        output: &mut bytes,
    });
    bytes.map(|bytes| {
        core::array::from_fn(|lane| {
            core::array::from_fn(|word| {
                let start = 32 * lane + 4 * word;
                u32::from_le_bytes(bytes[start..start + 4].try_into().expect("four-byte word"))
            })
        })
    })
}

struct KeyPair<'a> {
    seed: &'a [u8; 32],
    row: u32,
    columns: [u64; 2],
    output: &'a mut [[u8; OUTPUT_BYTES]; 2],
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
                // SHAKE domain suffix and multi-rate padding, unchanged from
                // sha3::Shake128. Every key input fits in one rate block.
                state[ELEMENT_INPUT_BYTES / 8] ^= 0x1f << (8 * (ELEMENT_INPUT_BYTES % 8));
                state[RATE / 8 - 1] ^= 0x80 << 56;
            }
            for start in (0..OUTPUT_BYTES).step_by(RATE) {
                permute(&mut states);
                let length = RATE.min(OUTPUT_BYTES - start);
                for (state, output) in states.iter().zip(outputs.iter_mut()) {
                    for (bytes, word) in output[start..start + length].chunks_exact_mut(8).zip(state) {
                        bytes.copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/key_pair.rs"]
mod tests;
