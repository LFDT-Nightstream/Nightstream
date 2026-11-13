use nightstream_fprime::components::{FormulaLibrary, SparseForm};
use serde_json::{json, Value};

fn library() -> Value {
    let mut row = vec![json!([]); 4];
    row[0] = json!([[2, 1]]);
    json!({
        "format": "nightstream.matrix-templates",
        "version": 1,
        "profile": [18446744073709551557u64, 2, 16, 65536, 64, 28, 4, 4],
        "components": [{
            "id": "linear-example", "input_count": 2,
            "ports": [
                {"name": "input", "role": "input", "start": 0, "count": 1},
                {"name": "output", "role": "output", "start": 1, "count": 1}
            ],
            "variants": [{"linear_forms": [[[0, 1], [1, 2]]], "rows": [row], "output_registers": [2]}],
            "definitions": ["test fixture"], "contracts": []
        }]
    })
}

fn decode(value: &Value) -> Result<FormulaLibrary, nightstream_fprime::components::ComponentError> {
    FormulaLibrary::from_json(&serde_json::to_vec(value).unwrap())
}

#[test]
fn normalizes_equal_columns_and_field_cancellation() {
    let form = SparseForm::new([(4, 1), (1, 7), (4, 18446744073709551556), (1, 3)]).unwrap();
    assert_eq!(form.entries().collect::<Vec<_>>(), vec![(1, 10)]);
    assert!(SparseForm::new([(0, 18446744073709551557)]).is_err());
}

#[test]
fn rejects_bad_references_even_when_the_coefficient_is_zero() {
    for replacement in [json!([[2, 1]]), json!([[2, 0]]), json!([[3, 1]])] {
        let mut value = library();
        value["components"][0]["variants"][0]["linear_forms"][0] = replacement;
        assert!(decode(&value).is_err());
    }
    let mut value = library();
    value["components"][0]["variants"][0]["rows"][0][0] = json!([[3, 0]]);
    assert!(decode(&value).is_err());
    let mut value = library();
    value["components"][0]["variants"][0]["output_registers"] = json!([3]);
    assert!(decode(&value).is_err());
}

#[test]
fn rejects_incomplete_or_ambiguous_component_contracts() {
    let mut wrong_profile = library();
    wrong_profile["profile"][2] = json!(14);
    let mut wrong_version = library();
    wrong_version["version"] = json!(2);
    let mut overlap = library();
    overlap["components"][0]["ports"][1]["start"] = json!(0);
    let mut missing = library();
    missing["components"][0]["ports"]
        .as_array_mut()
        .unwrap()
        .pop();
    let mut duplicate_name = library();
    duplicate_name["components"][0]["ports"][1]["name"] = json!("input");
    let mut unknown = library();
    unknown["components"][0]["rowz"] = json!([]);
    let mut coefficient = library();
    coefficient["components"][0]["variants"][0]["linear_forms"][0][0][1] = json!(18446744073709551557u64);
    let mut row_width = library();
    row_width["components"][0]["variants"][0]["rows"][0]
        .as_array_mut()
        .unwrap()
        .pop();
    let mut duplicate_id = library();
    let component = duplicate_id["components"][0].clone();
    duplicate_id["components"]
        .as_array_mut()
        .unwrap()
        .push(component);
    for value in [
        wrong_profile,
        wrong_version,
        overlap,
        missing,
        duplicate_name,
        unknown,
        coefficient,
        row_width,
        duplicate_id,
    ] {
        assert!(decode(&value).is_err(), "accepted malformed component: {value}");
    }
}
