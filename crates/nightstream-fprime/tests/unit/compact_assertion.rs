use serde_json::json;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

use super::{
    execute_ccs_invocation, execute_invocation, validate_template, validate_template_row, CompactInputRange,
    CompactRowInvocation, PackageError, RawCompactTemplateRow,
};

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
fn ccs_product_outputs_preserve_scratch_and_first54_keeps_its_row_checks() {
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
    let product_templates = super::super::plan::COMBINATION_TEMPLATE_COUNT;
    let mut templates = vec![template; product_templates + 1];
    let mut invocation = CompactRowInvocation {
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
    for index in 0..product_templates {
        invocation.template_index = index;
        let mut full = initial;
        let mut direct = initial;
        execute_invocation(&invocation, &templates, &mut full).unwrap();
        execute_ccs_invocation(&invocation, &templates, &mut direct).unwrap();
        assert_eq!(&direct[..3], &full[..3]);
        assert_eq!(direct[2], Goldilocks::from_u64(77));
        assert_eq!(direct[3], initial[3]);
        assert_eq!(full[3], Goldilocks::from_u64(77));
    }
    invocation.template_index = product_templates;
    let mut direct = initial;
    execute_ccs_invocation(&invocation, &templates, &mut direct).unwrap();
    assert_eq!(direct[3], Goldilocks::from_u64(77));
    templates[product_templates]
        .rows
        .last_mut()
        .unwrap()
        .c
        .constant = Goldilocks::ONE;
    assert!(matches!(
        execute_ccs_invocation(&invocation, &templates, &mut direct),
        Err(PackageError::Invalid("unsatisfied compact row"))
    ));
}
