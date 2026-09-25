use neo_ajtai::nightstream_fprime_setup::{
    authority_words, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use neo_math::D;

use crate::lifecycle::validate_key_prefix;

#[test]
fn preparation_requires_the_exact_selected_key_prefix_authority() {
    let manifest: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../artifacts/shared-verifier-v1.json")).unwrap();
    let dimensions = manifest["geometry"]["logical_width"].as_array().unwrap();
    // The real addition application has four message words, four local values,
    // and eight rows. The Lean manifest owns the shared verifier dimensions.
    let addition_width = dimensions
        .iter()
        .zip([1usize, 4, 4, 8])
        .map(|(coefficient, value)| usize::try_from(coefficient.as_u64().unwrap()).unwrap() * value)
        .sum::<usize>();
    let golden_width = usize::try_from(
        manifest["selected_reference"]["logical_width"]
            .as_u64()
            .unwrap(),
    )
    .unwrap();
    for width in [addition_width, golden_width, PRODUCTION_CARRIER_WIDTH] {
        let columns = width.div_ceil(D) as u64;
        let authority = authority_words(PRODUCTION_VERIFIER_ROWS, columns, &PRODUCTION_SEED);
        validate_key_prefix(width, &authority).unwrap();
        for (rows, wrong_columns) in [
            (PRODUCTION_VERIFIER_ROWS + 1, columns),
            (PRODUCTION_VERIFIER_ROWS, columns + 1),
        ] {
            let wrong = authority_words(rows, wrong_columns, &PRODUCTION_SEED);
            assert!(validate_key_prefix(width, &wrong).is_err());
        }
        let mut changed_seed = PRODUCTION_SEED;
        changed_seed[0] ^= 1;
        let wrong = authority_words(PRODUCTION_VERIFIER_ROWS, columns, &changed_seed);
        assert!(validate_key_prefix(width, &wrong).is_err());
    }
    let full_authority = authority_words(PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS, &PRODUCTION_SEED);
    assert!(validate_key_prefix(addition_width, &full_authority).is_err());
    for width in [0, PRODUCTION_CARRIER_WIDTH + 1, usize::MAX] {
        assert!(validate_key_prefix(width, &full_authority).is_err());
    }
}
