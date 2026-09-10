//! Check complete NIFS replay and its wire bytes on the selected nonzero input.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::{
    construction2::RunningInstance,
    nifs::{self, NifsProof},
    params::Params,
    relations::{ajtai_dec_mixer, ajtai_rlc_mixer},
};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};

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

/// Encode the independent Lean result in the documented native wire format.
/// This uses raw numeric fields and does not call the native proof encoder.
pub(super) fn check_wire(actual: &[u8], reference: &Value) {
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
    assert!(
        actual == expected,
        "every native NIFS proof byte equals the independent Lean-field encoding"
    );
    println!("complete_nifs_wire=passed bytes={}", actual.len());
}

fn rejected(
    name: &str,
    params: &Params,
    structure: &CcsStructure<F>,
    fresh: &CcsClaim<Commitment, F>,
    running: &RunningInstance,
    proof: &NifsProof,
) {
    let mut transcript = Transcript::session();
    let result = nifs::verify(
        &mut transcript,
        params,
        structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(fresh),
        running,
        proof,
    );
    assert!(result.is_err(), "complete NIFS accepted mutation {name}");
    println!("nifs_mutation={name} rejected={:?}", result.unwrap_err());
}

pub(super) fn check(
    params: &Params,
    structure: &CcsStructure<F>,
    fresh: &CcsClaim<Commitment, F>,
    running: &RunningInstance,
    proof: &NifsProof,
) {
    let mut changed = running.clone();
    changed.parent_authority = None;
    rejected("missing_prior_parent", params, structure, fresh, &changed, proof);
    for family in 0..structure.t() + 4 {
        let mut changed = running.clone();
        let parent = changed.parent_authority.as_mut().unwrap();
        let name = match family {
            0 => {
                parent.c.data[0] += F::ONE;
                "prior_parent_commitment".to_owned()
            }
            1 => {
                parent.X[(0, 0)] += F::ONE;
                "prior_parent_public".to_owned()
            }
            2 => {
                parent.r[0] += K::ONE;
                "prior_parent_point".to_owned()
            }
            3 => {
                parent.eval_k[0] += K::ONE;
                "prior_parent_eval_K".to_owned()
            }
            matrix => {
                parent.eval_a[matrix - 4][0] += K::ONE;
                format!("prior_parent_eval_A{}", matrix - 4)
            }
        };
        rejected(&name, params, structure, fresh, &changed, proof);
    }
    let mut changed = running.clone();
    changed.parent_authority.as_mut().unwrap().fold_digest[0] ^= 1;
    rejected("prior_parent_frame_digest", params, structure, fresh, &changed, proof);
    for source in 0..running.claims.len() {
        let mut changed = running.clone();
        changed.claims[source].c.data[0] += F::ONE;
        rejected(
            &format!("prior_child_{source}_commitment"),
            params,
            structure,
            fresh,
            &changed,
            proof,
        );
    }
    let mut changed = proof.clone();
    changed.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    rejected("C_round_message", params, structure, fresh, running, &changed);
    let mut changed = proof.clone();
    changed.pi_ccs.outputs[0].eval_k[0] += K::ONE;
    rejected("C_output_evaluation", params, structure, fresh, running, &changed);
    let mut changed = proof.clone();
    changed.pi_ccs.outputs.swap(0, 1);
    rejected("C_output_order", params, structure, fresh, running, &changed);
    let mut changed = proof.clone();
    changed.pi_rlc.combined.c.data[0] += F::ONE;
    rejected("R_output", params, structure, fresh, running, &changed);
    let mut changed = proof.clone();
    changed.pi_dec.children[0].c.data[0] += F::ONE;
    rejected("D_output", params, structure, fresh, running, &changed);
    let mut changed = proof.clone();
    changed.pi_dec.children.pop();
    rejected("D_child_count", params, structure, fresh, running, &changed);
    let mut changed = fresh.clone();
    changed.x[1] += F::ONE;
    rejected("fresh_public_input", params, structure, &changed, running, proof);
}
