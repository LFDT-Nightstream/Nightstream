use serde_json::json;

use super::{
    execute_invocation, execute_output, validate_template, validate_template_row, CompactInputRange,
    CompactRowInvocation, RawCompactTemplateRow,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

#[test]
fn exact_assertion_accepts_a_nonzero_rhs() {
    let raw: RawCompactTemplateRow =
        serde_json::from_value(json!([[0], [0, [[[0, 0], 1]]], [1, []], [0, [[[0, 1], 1]]]]))
            .expect("compact assertion row");

    validate_template_row(raw, 0, 2, 0).expect("exact A * B = C assertion");
}

#[test]
fn exact_assertion_preserves_a_zero_template_coefficient() {
    let raw: RawCompactTemplateRow =
        serde_json::from_value(json!([[0], [0, [[[0, 0], 0]]], [1, []], [0, []]])).expect("compact assertion row");

    validate_template_row(raw, 0, 1, 0).expect("exact zero-scaled term");
}

#[test]
fn ccs_product_output_preserves_scratch() {
    let template = validate_template(
        serde_json::from_value(json!([
            3,
            1,
            2,
            [3, [0, 0], [0, 1]],
            [
                [[1, 0], [0, [[[0, 0], 1]]], [0, [[[0, 1], 1]]], [0, [[[1, 0], 1]]]],
                [[0], [0, [[[1, 0], 1]]], [1, []], [0, [[[0, 2], 1]]]]
            ]
        ]))
        .unwrap(),
    )
    .unwrap();
    let templates = vec![template];
    let invocation = CompactRowInvocation {
        phase: 7,
        template_index: 0,
        row_start: 0,
        local_start: 3,
        input_ranges: vec![CompactInputRange {
            input_start: 0,
            input_count: 3,
            column_start: 0,
            column_stride: 1,
        }],
        output_column: 2,
    };
    let initial = [7, 11, 0, 19].map(Goldilocks::from_u64);
    let mut full = initial;
    let mut direct = initial;
    execute_invocation(&invocation, &templates, &mut full).unwrap();
    execute_output(&invocation, &templates, &mut direct);
    assert_eq!(&direct[..3], &full[..3]);
    assert_eq!(direct[2], Goldilocks::from_u64(77));
    assert_eq!(direct[3], initial[3]);
    assert_eq!(full[3], Goldilocks::from_u64(77));
}
