//! The v1.1 fold transcript of `neo_transcript::Poseidon2Transcript` over any
//! backend: an additive sponge on the workspace Poseidon2 permutation.
//!
//! Owns: `absorb_v1_1`, `read_pair_v1_1`, `squeeze_digest_v1_1` and
//! `state_prefix_v1_1` as backend operations, and the layer-1 handoff. A test
//! pins each one to `Poseidon2Transcript`.

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, WIDTH};
use p3_field_v08::PrimeField64;

use super::Backend;
use crate::field::gl;

/// The fold transcript state; it never holds a partial chunk.
pub struct FoldTranscript<B: Backend> {
    state: [B::F; WIDTH],
}

impl<B: Backend> Clone for FoldTranscript<B> {
    fn clone(&self) -> Self {
        Self { state: self.state }
    }
}

impl<B: Backend> FoldTranscript<B> {
    /// The zero state (`new_v1_1`, `reset_v1_1`).
    pub fn new(b: &mut B) -> Self {
        let zero = b.constant(0);
        Self { state: [zero; WIDTH] }
    }

    /// A transcript at a known state, for a caller that holds one.
    pub fn from_state(state: [B::F; WIDTH]) -> Self {
        Self { state }
    }

    /// Add each word into the rate lanes; permute after every complete or
    /// partial chunk.
    pub fn absorb(&mut self, b: &mut B, words: &[B::F]) {
        for chunk in words.chunks(RATE) {
            for (lane, &word) in chunk.iter().enumerate() {
                self.state[lane] = b.add(self.state[lane], word);
            }
            self.state = b.permute(self.state);
        }
    }

    /// Absorb constant words.
    pub fn absorb_constants(&mut self, b: &mut B, words: &[u64]) {
        let words: Vec<B::F> = words.iter().map(|&word| b.constant(word)).collect();
        self.absorb(b, &words);
    }

    /// Lanes `2·pair` and `2·pair + 1`; the state does not change.
    pub fn read_pair(&self, pair: usize) -> [B::F; 2] {
        assert!(pair < RATE / 2, "a read pair lies in the rate lanes");
        [self.state[2 * pair], self.state[2 * pair + 1]]
    }

    /// The first four lanes, then one permutation.
    pub fn squeeze_digest(&mut self, b: &mut B) -> [B::F; DIGEST_LEN] {
        let digest = self.prefix();
        self.state = b.permute(self.state);
        digest
    }

    /// The first four lanes; the state does not change.
    pub fn prefix(&self) -> [B::F; DIGEST_LEN] {
        std::array::from_fn(|lane| self.state[lane])
    }

    /// Name layer 1 in the transcript and return the layer-1 seed
    /// (`hash::seed`).
    pub(crate) fn handoff(mut self, b: &mut B) -> [B::F; DIGEST_LEN] {
        let chunk = neo_transcript::domain_chunk_v1_1(crate::hash::DOMAIN).map(|word| gl(word).as_canonical_u64());
        self.absorb_constants(b, &chunk);
        self.squeeze_digest(b)
    }
}
