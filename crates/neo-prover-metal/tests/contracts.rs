use neo_prover_metal::poseidon2::{
    round_constant_words, EXTERNAL_HALF_ROUNDS, INTERNAL_ROUNDS, RC_DIAG, RC_INITIAL, RC_INTERNAL, RC_TERMINAL,
    RC_WORDS, WIDTH,
};

#[test]
fn poseidon2_round_constants_use_the_canonical_device_layout() {
    let canonical = neo_ccs::crypto::poseidon2_goldilocks::round_constants();
    let words = round_constant_words();

    assert_eq!(words.len(), RC_WORDS);
    assert_eq!(canonical.initial.len(), EXTERNAL_HALF_ROUNDS);
    assert_eq!(canonical.internal.len(), INTERNAL_ROUNDS);
    assert_eq!(canonical.terminal.len(), EXTERNAL_HALF_ROUNDS);
    for (round, row) in canonical.initial.iter().enumerate() {
        assert_eq!(&words[RC_INITIAL + WIDTH * round..][..WIDTH], row);
    }
    assert_eq!(&words[RC_INTERNAL..RC_INTERNAL + INTERNAL_ROUNDS], canonical.internal);
    for (round, row) in canonical.terminal.iter().enumerate() {
        assert_eq!(&words[RC_TERMINAL + WIDTH * round..][..WIDTH], row);
    }
    assert_eq!(&words[RC_DIAG..RC_DIAG + WIDTH], &canonical.diag);
}
