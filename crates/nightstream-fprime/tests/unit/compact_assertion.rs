use serde_json::json;

use super::{validate_template_row, RawCompactTemplateRow};

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
