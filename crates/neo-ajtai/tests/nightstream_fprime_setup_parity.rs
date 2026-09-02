//! Lean/Rust parity for the Nightstream F-prime indexed Ajtai setup.

use neo_ajtai::nightstream_fprime_setup::{
    authority_words, block_words, coefficient, production_authority_words, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED,
    PRODUCTION_VERIFIER_ROWS, SETUP_ID,
};
use serde_json::Value;

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
fn lean_setup_vectors_match_rust_and_rfc8439() {
    let fixture: Value = serde_json::from_slice(&std::fs::read(FIXTURE_PATH).expect("read Lean setup fixture"))
        .expect("decode Lean setup fixture");
    let root = array(&fixture);
    assert_eq!(root.len(), 7);
    assert_eq!(nat(&root[0]), 3, "setup fixture schema");
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

    let rfc_words = [
        0xe4e7_f110,
        0x1559_3bd1,
        0x1fdd_0f50,
        0xc471_20a3,
        0xc7f4_d1c7,
        0x0368_c033,
        0x9aaa_2204,
        0x4e6c_d4c3,
        0x4664_82d2,
        0x09aa_9f07,
        0x05d7_c214,
        0xa202_8bd9,
        0xd19c_12b5,
        0xb94e_16de,
        0xe883_d0cb,
        0x4e3c_50a2,
    ];
    let rust_rfc = block_words(&test_seed(), 0x0900_0000, 0x4a00_0000, 1);
    assert_eq!(rust_rfc, rfc_words, "RFC 8439 Section 2.3.2 block");
    assert_eq!(nat_list(&root[3]), rust_rfc.map(u64::from));

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
