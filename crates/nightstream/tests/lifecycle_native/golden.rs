//! Compare a complete native fold with fresh Lean output using the current verifier.

use super::*;
use serde::Deserialize;
use std::io::{Read, Write};

#[path = "golden_dec.rs"]
mod dec;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
enum Request {
    Compare {
        package: PathBuf,
        directory: PathBuf,
        lean: PathBuf,
    },
    Encode {
        lean: PathBuf,
        output: PathBuf,
    },
}

#[test]
#[ignore = "stdin runner used by check_lean_fold.py; each call is capped at 300 seconds"]
fn native_checker() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let request: Request = serde_json::from_str(&input).unwrap();
    match request {
        Request::Compare {
            package,
            directory,
            lean,
        } => {
            let actual = read(directory.join("actual_result.json"));
            let expected = read(lean);
            let wire = fs::read(directory.join("proof.native")).unwrap();
            compare(&actual, &wire, &expected);
            dec::check_saved(&package, &actual, &expected);
            println!("actual_selected_nifs_Lean_comparison=passed");
        }
        Request::Encode { lean, output } => {
            let bytes = encode_wire(&read(lean));
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)
                .unwrap()
                .write_all(&bytes)
                .unwrap();
            println!("encoded_lean_nifs_bytes={} provenance=not_checked", bytes.len());
        }
    }
}

#[test]
fn retained_fold_matches_every_lean_field_and_rejects_pidec_mutations() {
    let actual =
        read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs/actual_result.json"));
    let wire =
        fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs/proof.native"))
            .unwrap();
    let lean = read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"));
    compare(&actual, &wire, &lean);
    dec::check_saved(
        &artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
        &actual,
        &lean,
    );
    let mut changed = wire;
    *changed.last_mut().unwrap() ^= 1;
    assert!(std::panic::catch_unwind(|| check_wire(&changed, &lean)).is_err());
}

fn compare(actual: &Value, wire: &[u8], expected: &Value) {
    assert_eq!(actual["schema"], 1);
    assert_eq!(expected.as_array().expect("full Lean C/R/D result").len(), 10);
    assert_eq!(expected[0], 1);
    assert_eq!(expected[5][0], 1);
    assert_eq!(expected[7][0], 1);
    assert_eq!(expected[9][0], 1);
    assert_eq!(expected[9][16][0], 1);
    assert_eq!(actual["package_identity"], expected[6][6]);
    assert_eq!(actual["pi_ccs_input"], expected[1]);
    assert_eq!(actual["pi_ccs_phase"], expected[5]);
    assert_eq!(
        actual["pi_rlc_parent"],
        json!([
            expected[7][3],
            expected[7][4],
            expected[7][5],
            expected[7][6],
            expected[7][7],
            1,
        ])
    );
    assert_eq!(actual["children"], expected[9][16][1]);
    assert_eq!(actual["outgoing_state"], expected[7][9]);
    assert_eq!(actual["outgoing_state"], expected[9][14]);
    assert_eq!(actual["absorbed"], 0);
    check_wire(wire, expected);
}

fn word(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn numbers(output: &mut Vec<u8>, values: &Value) {
    match values {
        Value::Number(value) => word(output, value.as_u64().expect("canonical natural word")),
        Value::Array(values) => values.iter().for_each(|value| numbers(output, value)),
        _ => panic!("numeric Lean proof fields"),
    }
}

fn evaluation(output: &mut Vec<u8>, values: &Value) {
    assert_eq!(values.as_array().expect("full ring evaluation").len(), D);
    word(output, D.next_power_of_two() as u64);
    numbers(output, values);
    for _ in D..D.next_power_of_two() {
        word(output, 0);
        word(output, 0);
    }
}

fn claim(output: &mut Vec<u8>, value: &Value, digest: &Value) {
    assert_eq!(value.as_array().expect("full Lean claim").len(), 6);
    assert_eq!(value[0].as_array().unwrap().len(), 22 * D);
    for n in [D, 22, 22 * D] {
        word(output, n as u64);
    }
    numbers(output, &value[0]);
    assert_eq!(value[1].as_array().unwrap().len(), 270);
    word(output, D as u64);
    word(output, 5);
    for row in 0..D {
        for column in 0..5 {
            numbers(output, &value[1][column * D + row]);
        }
    }
    assert_eq!(value[2].as_array().unwrap().len(), 28);
    word(output, 28);
    numbers(output, &value[2]);
    evaluation(output, &value[3]);
    assert_eq!(value[4].as_array().unwrap().len(), 14);
    word(output, 14);
    for family in value[4].as_array().unwrap() {
        evaluation(output, family);
    }
    word(output, 270);
    numbers(output, digest);
    output.push(0); // No auxiliary lane commitments in this selected profile.
}

/// Encode raw Lean result fields in the documented native wire format.
/// This uses raw numeric fields and does not call the native proof encoder.
fn encode_wire(reference: &Value) -> Vec<u8> {
    let mut expected = b"NS-NIFS-PROOF".to_vec();
    word(&mut expected, 1);
    let mut ccs = Vec::new();
    for value in [1102, 1, 28, 10] {
        word(&mut ccs, value);
    }
    assert_eq!(reference[1][3].as_array().unwrap().len(), 28);
    for round in reference[1][3].as_array().unwrap() {
        assert_eq!(round.as_array().unwrap().len(), 10);
        numbers(&mut ccs, round);
    }
    word(&mut expected, ccs.len() as u64);
    expected.extend(ccs);
    let digest = json!(&reference[5][14].as_array().unwrap()[..4]);
    word(&mut expected, 17);
    for source in 0..17 {
        let value = json!([
            reference[5][10][source],
            reference[5][11][source],
            reference[5][6],
            reference[5][12][source],
            reference[5][13][source],
            0
        ]);
        claim(&mut expected, &value, &digest);
    }
    let parent = json!([
        reference[7][3],
        reference[7][4],
        reference[7][5],
        reference[7][6],
        reference[7][7],
        1
    ]);
    claim(&mut expected, &parent, &digest);
    word(&mut expected, 16);
    assert_eq!(reference[9][13].as_array().unwrap().len(), 16);
    for child in reference[9][13].as_array().unwrap() {
        claim(&mut expected, child, &digest);
    }
    expected
}

fn check_wire(actual: &[u8], reference: &Value) {
    let expected = encode_wire(reference);
    assert!(
        actual == expected,
        "every native NIFS proof byte equals the independent Lean-field encoding"
    );
    println!("complete_nifs_wire=passed bytes={}", actual.len());
}
