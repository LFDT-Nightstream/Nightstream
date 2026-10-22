//! The commitment serde form keeps the field names that the Lean replay
//! scripts read, and decoding rejects any shape the constructor rejects.

use neo_ajtai::Commitment;
use neo_math::D;

#[test]
fn serde_keeps_field_names_and_round_trips() {
    let commitment = Commitment::zeros(2);
    let value = serde_json::to_value(&commitment).expect("encode");
    assert_eq!(value["d"], D);
    assert_eq!(value["kappa"], 2);
    assert_eq!(value["data"].as_array().expect("data").len(), 2 * D);
    let back: Commitment = serde_json::from_value(value).expect("decode");
    assert_eq!(back, commitment);
}

#[test]
fn decode_rejects_a_payload_that_does_not_match_the_shape() {
    let mut value = serde_json::to_value(Commitment::zeros(2)).expect("encode");
    value["data"].as_array_mut().expect("data").pop();
    assert!(serde_json::from_value::<Commitment>(value).is_err(), "short payload");

    let mut value = serde_json::to_value(Commitment::zeros(1)).expect("encode");
    value["d"] = serde_json::json!(D - 1);
    assert!(serde_json::from_value::<Commitment>(value).is_err(), "wrong column height");
}
