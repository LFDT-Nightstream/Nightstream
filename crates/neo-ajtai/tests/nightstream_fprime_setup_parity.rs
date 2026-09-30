//! Lean/Rust parity for the Nightstream F-prime indexed Ajtai setup.

use neo_ajtai::nightstream_fprime_setup::{
    authority_words, coefficient, element_bytes, element_input, production_authority_words, PRODUCTION_MESSAGE_COLUMNS,
    PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS, SETUP_ID,
};
use serde_json::Value;
use sha3::{
    digest::{ExtendableOutput, XofReader},
    Shake128,
};

/// FIPS 202 example: the first 32 bytes of SHAKE128 of the empty message.
const SHAKE128_EMPTY: [u8; 32] = [
    0x7f, 0x9c, 0x2b, 0xa4, 0xe8, 0x8f, 0x82, 0x7d, 0x61, 0x60, 0x45, 0x50, 0x76, 0x05, 0x85, 0x3e, 0xd7, 0x3b, 0x80,
    0x93, 0xf6, 0xef, 0xbc, 0x88, 0xeb, 0x1a, 0x6e, 0xac, 0xfa, 0x66, 0xef, 0x26,
];

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../formal/nightstream-fprime/artifacts/nightstream-fprime-ajtai-setup-v1-parity.json"
);
fn test_seed() -> [u8; 32] {
    core::array::from_fn(|index| index as u8)
}

fn array(value: &Value) -> &[Value] {
    value.as_array().expect("fixture value must be an array")
}

fn nat(value: &Value) -> u64 {
    value.as_u64().expect("fixture atom must be a u64")
}

fn nat_list(value: &Value) -> Vec<u64> {
    array(value).iter().map(nat).collect()
}

#[test]
fn lean_setup_vectors_match_rust_and_fips202() {
    check_lean_setup_fixture(&std::fs::read(FIXTURE_PATH).expect("read Lean setup fixture"));
}

#[test]
#[ignore = "current Lean setup fixture path as JSON on stdin; run under the 300-second cap"]
fn external_lean_setup_vectors_match_current_rust() {
    let path: std::path::PathBuf = serde_json::from_reader(std::io::stdin()).expect("external Lean setup fixture path");
    check_lean_setup_fixture(&std::fs::read(path).expect("read external Lean setup fixture"));
}

fn check_lean_setup_fixture(bytes: &[u8]) {
    let fixture: Value = serde_json::from_slice(bytes).expect("decode Lean setup fixture");
    let root = array(&fixture);
    assert_eq!(root.len(), 8);
    assert_eq!(nat(&root[0]), 4, "setup fixture schema");
    assert_eq!(
        nat_list(&root[1]),
        SETUP_ID.iter().copied().map(u64::from).collect::<Vec<_>>()
    );
    assert_eq!(
        nat_list(&root[2]),
        test_seed()
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>()
    );

    let mut empty = [0_u8; 32];
    Shake128::default().finalize_xof().read(&mut empty);
    assert_eq!(empty, SHAKE128_EMPTY, "FIPS 202 SHAKE128 example");
    assert_eq!(nat_list(&root[3]), SHAKE128_EMPTY.map(u64::from));

    // Fixed-length fields: setup ID, seed, row little-endian, block little-endian.
    let mut expected_input = SETUP_ID.to_vec();
    expected_input.extend(test_seed());
    expected_input.extend([1, 2, 3, 4]);
    expected_input.extend([5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(
        element_input(&test_seed(), 0x0403_0201, 0x0c0b_0a09_0807_0605).to_vec(),
        expected_input
    );

    assert_eq!(nat_list(&root[4]), PRODUCTION_SEED.map(u64::from));
    let cases = [
        (0_u32, 0_u64, 0_u32),
        (0, 0, 53),
        (1, 32_768, 17),
        (21, PRODUCTION_MESSAGE_COLUMNS - 1, 53),
    ];
    let expected = array(&root[5]);
    assert_eq!(expected.len(), cases.len());
    for (entry, (row, block, lane)) in expected.iter().zip(cases) {
        let entry = array(entry);
        assert_eq!(entry.len(), 4);
        assert_eq!(
            [nat(&entry[0]), nat(&entry[1]), nat(&entry[2])],
            [u64::from(row), block, u64::from(lane)]
        );
        assert_eq!(nat(&entry[3]), coefficient(&PRODUCTION_SEED, row, block, lane));
    }
    assert_eq!(nat_list(&root[6]), production_authority_words());
    assert_eq!(production_authority_words().len(), 73);

    // One complete element: every lane, including the lanes that cross a
    // 168-byte rate boundary.
    assert_eq!(
        nat_list(&root[7]),
        element_bytes(&PRODUCTION_SEED, 1, 32_768).map(u64::from)
    );

    let mut changed_seed = test_seed();
    changed_seed[0] ^= 1;
    assert_ne!(coefficient(&test_seed(), 0, 0, 0), coefficient(&changed_seed, 0, 0, 0));
    assert_ne!(
        authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED),
        authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &changed_seed)
    );
    assert_ne!(
        authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED),
        authority_words(
            PRODUCTION_VERIFIER_ROWS + 1,
            PRODUCTION_MESSAGE_COLUMNS,
            &PRODUCTION_SEED
        )
    );
    assert_ne!(
        authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED),
        authority_words(
            PRODUCTION_VERIFIER_ROWS,
            PRODUCTION_MESSAGE_COLUMNS + 1,
            &PRODUCTION_SEED
        )
    );
}
