//! What the argument writes to and reads from the caller's Poseidon2
//! transcript. `absorb_v1_2` has no frame of its own, so each message here
//! has a length that the statement fixes, or carries its length.

use neo_math::{from_complex, KExtensions, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

pub(crate) fn challenge(transcript: &mut Poseidon2Transcript) -> K {
    let [real, imaginary] = transcript.read_coin_v1_2().value;
    from_complex(real, imaginary)
}

pub(crate) fn challenges(transcript: &mut Poseidon2Transcript, count: usize) -> Vec<K> {
    (0..count).map(|_| challenge(transcript)).collect()
}

pub(crate) fn absorb_extension(transcript: &mut Poseidon2Transcript, values: &[K]) {
    let words: Vec<F> = values.iter().flat_map(|value| value.as_coeffs()).collect();
    transcript.absorb_v1_2(&words);
}

/// `bytes` as field words: the byte count, then each seven bytes as one
/// little-endian word. A word is below `2^56`, so it is canonical, and the
/// count fixes the bytes of the last word.
pub fn byte_words(bytes: &[u8]) -> Vec<F> {
    let mut words = vec![F::from_u64(bytes.len() as u64)];
    words.extend(bytes.chunks(7).map(|chunk| {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        F::from_u64(u64::from_le_bytes(word))
    }));
    words
}

pub(crate) fn absorb_bytes(transcript: &mut Poseidon2Transcript, bytes: &[u8]) {
    transcript.absorb_v1_2(&byte_words(bytes));
}

/// The transcript's state as the session bytes of an Akita opening, after
/// `label`: the opening is bound to every message before it.
pub(crate) fn session(transcript: &Poseidon2Transcript, label: &[u8]) -> Vec<u8> {
    [label, &transcript.state_prefix_v1_2()].concat()
}
