use neo_ajtai::prg::expand_row_v2 as expand_row;

#[test]
fn ajtai_prg_determinism_v2() {
    let seed = [42u8; 32];
    let r0 = expand_row(&seed, 0, 10);
    let r0_b = expand_row(&seed, 0, 10);
    assert_eq!(r0, r0_b, "same seed+row_idx must produce identical row");

    let r1 = expand_row(&seed, 1, 10);
    assert_ne!(r0, r1, "different row_idx must produce different rows");

    let mut seed2 = seed;
    seed2[0] ^= 1;
    let r0_seed2 = expand_row(&seed2, 0, 10);
    assert_ne!(r0, r0_seed2, "different seed must produce different rows");
}

#[test]
fn ajtai_prg_length_v2() {
    let seed = [7u8; 32];
    for len in [1usize, 2, 3, 4, 5, 8, 9, 16, 17] {
        let row = expand_row(&seed, 123, len);
        assert_eq!(row.len(), len);
    }
}

#[test]
fn ajtai_prg_rejects_distinct_seeds_with_the_same_field_encoding() {
    let seed_zero = [0u8; 32];
    let mut seed_modulus = seed_zero;
    seed_modulus[..8].copy_from_slice(&0xffff_ffff_0000_0001u64.to_le_bytes());

    assert_ne!(seed_zero, seed_modulus, "the setup seeds are byte-distinct");

    let row_zero = expand_row(&seed_zero, 7, 8);
    let row_modulus = expand_row(&seed_modulus, 7, 8);
    assert_ne!(
        row_zero, row_modulus,
        "distinct Ajtai setup seeds must not collide before Poseidon2 hashing"
    );
}
