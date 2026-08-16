//! Rust mirror of the Lean seeded-Phi81 sampler semantics.
//!
//! Owns: a term-for-term transcription of the Lean sampler chain
//! (`SeededPhi81Sampler.sampleVector`/`sampleOutput` cursor rules, Phi81
//! rotation, bit-column mapping, zero elision) with `rand_chacha` as the
//! stream, so production `SeededPhi81LinearBlock` rows can be replayed
//! against the Lean algorithm on this side of the boundary.
//!
//! Does not own: the Lean sampler itself (the Lean tests pin both sides to
//! the same committed conformance fixtures), the production block metadata,
//! or any removal authority. A mismatch here is a conformance failure, never
//! permission to edit either side.

use neo_ccs::SeededPhi81LinearBlock;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;
const DIMENSION: usize = 54;

/// `ChaCha8.u64s seed wordStart count`: little-endian `u64`s assembled from
/// two consecutive `u32` stream words starting at `word_start`.
fn stream_u64s(seed: [u8; 32], word_start: u128, count: usize) -> Vec<u64> {
    let mut rng = ChaCha8Rng::from_seed(seed);
    rng.set_word_pos(word_start);
    (0..count).map(|_| rng.next_u64()).collect()
}

/// `SeededPhi81Sampler.nextAccepted`: skip rejected candidates, two `u32`
/// words per candidate, bounded by `fuel`.
fn next_accepted(seed: [u8; 32], mut word_position: u128, fuel: usize) -> Option<(u64, u128)> {
    for _ in 0..fuel {
        let candidate = stream_u64s(seed, word_position, 1)[0];
        if candidate < GOLDILOCKS_P {
            return Some((candidate, word_position + 2));
        }
        word_position += 2;
    }
    None
}

/// `SeededPhi81Sampler.sampleVector`: 54 raw candidates, then sequential
/// replacements from the stream tail. Returns the vector, the tail cursor,
/// and the number of rejected raw candidates.
fn sample_vector(seed: [u8; 32], fuel: usize, word_position: u128) -> Option<(Vec<u64>, u128, usize)> {
    let raw = stream_u64s(seed, word_position, DIMENSION);
    let mut cursor = word_position + 2 * DIMENSION as u128;
    let mut values = Vec::with_capacity(DIMENSION);
    let mut rejected = 0;
    for candidate in raw {
        if candidate < GOLDILOCKS_P {
            values.push(candidate);
        } else {
            let (value, next_cursor) = next_accepted(seed, cursor, fuel)?;
            values.push(value);
            cursor = next_cursor;
            rejected += 1;
        }
    }
    Some((values, cursor, rejected))
}

/// `SeededPhi81Sampler.chunkMessageCount`.
fn chunk_message_count(message_cols: usize, chunk_size: usize, chunk_index: usize) -> usize {
    let start = chunk_index * chunk_size;
    if start < message_cols {
        chunk_size.min(message_cols - start)
    } else {
        0
    }
}

/// `SeededPhi81Sampler.sampleOutput`: every chunk seed starts at stream
/// position zero; vectors chain through the returned cursor inside a chunk.
fn sample_output(
    seeds: &[[u8; 32]],
    message_cols: usize,
    chunk_size: usize,
    fuel: usize,
) -> Option<(Vec<Vec<u64>>, usize)> {
    let mut vectors = Vec::with_capacity(message_cols);
    let mut rejected = 0;
    for (chunk_index, &seed) in seeds.iter().enumerate() {
        let mut cursor = 0u128;
        for _ in 0..chunk_message_count(message_cols, chunk_size, chunk_index) {
            let (vector, next_cursor, vector_rejected) = sample_vector(seed, fuel, cursor)?;
            vectors.push(vector);
            cursor = next_cursor;
            rejected += vector_rejected;
        }
    }
    Some((vectors, rejected))
}

/// Rows and rejection count of one mirrored block replay.
pub struct MirrorReplay {
    /// Term lists indexed `output * 54 + coordinate` from the block start.
    pub rows: Vec<Vec<(usize, u64)>>,
    /// Rejected raw candidates across every chunk draw of the block.
    pub rejected_words: usize,
}

/// `SeededPhi81.rotatePhi81`: multiplication by `X` modulo
/// `Phi_81 = X^54 + X^27 + 1` over canonical Goldilocks residues.
fn rotate_phi81(current: &[u64; DIMENSION]) -> [u64; DIMENSION] {
    let last = current[DIMENSION - 1] % GOLDILOCKS_P;
    let mut next = [0u64; DIMENSION];
    next[0] = if last == 0 { 0 } else { GOLDILOCKS_P - last };
    next[1..DIMENSION].copy_from_slice(&current[..DIMENSION - 1]);
    let reduced = next[27] % GOLDILOCKS_P;
    next[27] = ((reduced as u128 + (GOLDILOCKS_P - last) as u128) % GOLDILOCKS_P as u128) as u64;
    next
}

/// `SeededPhi81.Block.bitColumn`.
fn bit_column(word_starts: &[usize], word_width: usize, bit_index: usize) -> Option<usize> {
    if word_width == 0 || bit_index >= word_starts.len() * word_width {
        return None;
    }
    Some(word_starts[bit_index / word_width] + bit_index % word_width)
}

/// Replay one production block through the Lean sampler semantics. Rows are
/// indexed `output * 54 + coordinate` relative to the block start; each term
/// list follows the Lean order (message column outer, message row inner) and
/// elides zeros exactly like `Block.terms`. Returns `None` when a replacement
/// search exhausts `fuel` (the fail-closed sampler outcome).
pub fn mirror_block(block: &SeededPhi81LinearBlock, fuel: usize) -> Option<MirrorReplay> {
    let mut rows = vec![Vec::new(); DIMENSION * block.kappa()];
    let mut rejected_words = 0;
    for (output, seeds) in block.chunk_seeds_by_row().iter().enumerate() {
        let (base_rotations, rejected) = sample_output(seeds, block.message_cols(), block.chunk_size(), fuel)?;
        if base_rotations.len() != block.message_cols() {
            return None;
        }
        rejected_words += rejected;
        for (message_col, base) in base_rotations.iter().enumerate() {
            let mut rotation: [u64; DIMENSION] = base.as_slice().try_into().ok()?;
            for message_row in 0..DIMENSION {
                if let Some(column) = bit_column(
                    block.word_starts(),
                    block.word_width(),
                    message_row * block.message_cols() + message_col,
                ) {
                    for (coordinate, &coefficient) in rotation.iter().enumerate() {
                        if coefficient != 0 {
                            rows[output * DIMENSION + coordinate].push((column, coefficient));
                        }
                    }
                }
                rotation = rotate_phi81(&rotation);
            }
        }
    }
    Some(MirrorReplay { rows, rejected_words })
}
