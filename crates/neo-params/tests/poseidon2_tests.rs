use neo_params::poseidon2::{CAPACITY, DIGEST_LEN, RATE, WIDTH};

#[test]
fn sponge_shape_is_consistent() {
    assert_eq!(WIDTH, 16);
    assert_eq!(CAPACITY, 4);
    assert_eq!(RATE + CAPACITY, WIDTH);
    assert_eq!(DIGEST_LEN, CAPACITY);
}
