//! Canonical host-to-Metal Poseidon2 parameter layout.
//!
//! Round constants come from `neo-ccs`; the Metal shaders contain no
//! transcript constants of their own.

pub const WIDTH: usize = 8;
pub const RATE: usize = 4;
pub const DIGEST_LEN: usize = 4;
pub const EXTERNAL_HALF_ROUNDS: usize = 4;
pub const INTERNAL_ROUNDS: usize = 22;

pub const RC_INITIAL: usize = 0;
pub const RC_INTERNAL: usize = RC_INITIAL + EXTERNAL_HALF_ROUNDS * WIDTH;
pub const RC_TERMINAL: usize = RC_INTERNAL + INTERNAL_ROUNDS;
pub const RC_DIAG: usize = RC_TERMINAL + EXTERNAL_HALF_ROUNDS * WIDTH;
pub const RC_WORDS: usize = RC_DIAG + WIDTH;

/// Flatten the canonical Poseidon2 parameters into the device ABI shared by
/// CUDA and Metal: initial external rows, internal constants, terminal
/// external rows, then the internal-matrix diagonal.
pub fn round_constant_words() -> Vec<u64> {
    let constants = neo_ccs::crypto::poseidon2_goldilocks::round_constants();
    assert_eq!(constants.initial.len(), EXTERNAL_HALF_ROUNDS);
    assert_eq!(constants.internal.len(), INTERNAL_ROUNDS);
    assert_eq!(constants.terminal.len(), EXTERNAL_HALF_ROUNDS);

    let mut words = vec![0u64; RC_WORDS];
    for (round, row) in constants.initial.iter().enumerate() {
        words[RC_INITIAL + WIDTH * round..][..WIDTH].copy_from_slice(row);
    }
    words[RC_INTERNAL..RC_INTERNAL + INTERNAL_ROUNDS].copy_from_slice(&constants.internal);
    for (round, row) in constants.terminal.iter().enumerate() {
        words[RC_TERMINAL + WIDTH * round..][..WIDTH].copy_from_slice(row);
    }
    words[RC_DIAG..RC_DIAG + WIDTH].copy_from_slice(&constants.diag);
    words
}
