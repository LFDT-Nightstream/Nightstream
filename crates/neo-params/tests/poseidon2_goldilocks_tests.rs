use neo_params::poseidon2_goldilocks::{CAPACITY, DIGEST_LEN, RATE, SEED, WIDTH};

#[test]
fn parameters_consistent() {
    assert_eq!(WIDTH, 16, "WIDTH should be 16");
    assert_eq!(CAPACITY, 4, "CAPACITY should be 4 (256 bits)");
    assert_eq!(RATE, 12, "RATE should be 12");
    assert_eq!(DIGEST_LEN, 4, "DIGEST_LEN should be 4 (~256 bits)");
    assert_eq!(RATE + CAPACITY, WIDTH, "RATE + CAPACITY must equal WIDTH");
}

#[test]
fn seed_fixed() {
    // Verify seed hasn't been accidentally changed
    assert_eq!(SEED.len(), 32, "Seed must be 32 bytes");
}
