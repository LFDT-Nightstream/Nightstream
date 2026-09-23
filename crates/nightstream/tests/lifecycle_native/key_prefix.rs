use neo_ajtai::nightstream_fprime_setup::{
    authority_words, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use neo_math::D;

use crate::lifecycle::validate_key_prefix;

#[test]
fn preparation_requires_the_exact_selected_key_prefix_authority() {
    // ApplicationRetainedGeometry.completeLogicalWidth_eq_applicationCounts:
    // four private words and four addition outputs require eight retained words.
    let addition_width: usize = 171_902_203 + 41 * 8;
    let golden_width: usize = 171_902_203 + 41 * 7_700;
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
