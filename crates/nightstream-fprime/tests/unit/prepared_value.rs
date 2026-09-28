use super::super::{MAX_FIXED_SOURCE_NODES, NATIVE_REFERENCE_NODES};
use super::*;
use serde_json::json;

#[test]
fn numeric_array_decode_accepts_exact_nodes_and_rejects_one_short() {
    let value = json!([0, [1, 2], []]);
    let encoded = serde_json::to_vec(&value).unwrap();
    let nodes = 1 + 1 + (1 + 2) + 1;
    assert_eq!(decode(encoded.as_slice(), nodes).unwrap(), value);
    validate(&value, nodes).unwrap();
    assert!(decode(encoded.as_slice(), nodes - 1).is_err());
    assert!(validate(&value, nodes - 1).is_err());
    assert!(decode(b"[]".as_slice(), 0).is_err());
    assert_eq!(decode(u64::MAX.to_string().as_bytes(), 1).unwrap(), json!(u64::MAX));
}

#[test]
fn fixed_values_reject_non_natural_types_and_trailing_json() {
    for encoded in ["null", "true", "-1", "1.5", "\"text\"", "{}"] {
        assert!(decode(encoded.as_bytes(), 1).is_err(), "accepted {encoded}");
        let value: Value = serde_json::from_str(encoded).unwrap();
        assert!(validate(&value, 1).is_err(), "accepted {encoded}");
    }
    assert!(decode(b"[] []".as_slice(), 2).is_err());
}

fn assert_bound(value: &Value, exact_nodes: usize) {
    let encoded = serde_json::to_vec(value).unwrap();
    assert_eq!(decode(encoded.as_slice(), exact_nodes).unwrap(), *value);
    validate(value, exact_nodes).unwrap();
    assert!(decode(encoded.as_slice(), exact_nodes - 1).is_err());
    assert!(validate(value, exact_nodes - 1).is_err());
}

#[test]
fn duplicate_zero_template_terms_consume_the_real_reference_headroom() {
    // RawTemplateCombination = [constant, [[column_reference, coefficient], ...]].
    // This base has eight nodes; every duplicate zero term adds five.
    let mut combination = json!([0, [[[0, 0], 1]]]);
    let base_nodes = 8;
    assert_bound(&combination, base_nodes);
    let headroom = MAX_FIXED_SOURCE_NODES - NATIVE_REFERENCE_NODES;
    let copies = headroom / 5 + 1;
    let terms = combination[1].as_array_mut().unwrap();
    for _ in 0..copies {
        terms.push(json!([[0, 0], 0]));
    }
    let nodes = base_nodes + copies * 5;
    assert_bound(&combination, nodes);
    let encoded = serde_json::to_vec(&combination).unwrap();
    assert!(decode(encoded.as_slice(), base_nodes + headroom).is_err());
    assert!(validate(&combination, base_nodes + headroom).is_err());
}

#[test]
fn unused_compact_templates_consume_the_real_reference_headroom() {
    // One unused compact template with no locals and a constant assertion.
    // Its twenty nodes are retained even though no invocation refers to it.
    let template = json!([2, 0, 1, [0, 0], [[[0], [0, []], [1, []], [0, []]]]]);
    let mut templates = json!([template]);
    let base_nodes = 1 + 20;
    assert_bound(&templates, base_nodes);
    let headroom = MAX_FIXED_SOURCE_NODES - NATIVE_REFERENCE_NODES;
    let copies = headroom / 20 + 1;
    for _ in 0..copies {
        templates.as_array_mut().unwrap().push(template.clone());
    }
    let nodes = base_nodes + copies * 20;
    assert_bound(&templates, nodes);
    let encoded = serde_json::to_vec(&templates).unwrap();
    assert!(decode(encoded.as_slice(), base_nodes + headroom).is_err());
    assert!(validate(&templates, base_nodes + headroom).is_err());
}
