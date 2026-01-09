use neo_ajtai::compute_pp_id_digest_v1;

#[test]
fn pp_id_digest_v1_test_vector() {
    let d = 54usize;
    let m = 1usize << 20;
    let kappa = 16usize;
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&42u64.to_le_bytes());

    let got = compute_pp_id_digest_v1(d, m, kappa, seed);

    // NOTE: This is a regression test vector. If this changes, bump the domain/version.
    let expected = [
        36, 153, 9, 133, 6, 101, 10, 3, 107, 93, 72, 147, 201, 72, 183, 163, 13, 121, 10, 142,
        252, 121, 99, 26, 90, 177, 232, 59, 90, 100, 81, 132,
    ];
    assert_eq!(got, expected);
}
