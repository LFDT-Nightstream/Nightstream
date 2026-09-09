//! Exact verifier-key relation artifact checks.

use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::{RelationArtifactError, VerifierKeyRelationArtifact};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Value};

fn preprocessing() -> neo_fold_clean::Preprocessing {
    let mut a = Mat::zero(1, D, F::ZERO);
    a[(0, 0)] = F::ONE;
    let r1cs = R1cs {
        a,
        b: Mat::zero(1, D, F::ZERO),
        c: Mat::zero(1, D, F::ZERO),
        m_in: D,
    };
    direct_ccs::preprocess_seeded(&r1cs, 17).expect("artifact preprocessing")
}

fn other_preprocessing() -> neo_fold_clean::Preprocessing {
    let mut a = Mat::zero(1, D, F::ZERO);
    a[(0, 1)] = F::ONE;
    direct_ccs::preprocess_seeded(
        &R1cs {
            a,
            b: Mat::zero(1, D, F::ZERO),
            c: Mat::zero(1, D, F::ZERO),
            m_in: D,
        },
        17,
    )
    .expect("same-shaped other preprocessing")
}

fn mutated(bytes: &[u8], change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut value: Value = serde_json::from_slice(bytes).expect("artifact JSON");
    change(&mut value);
    serde_json::to_vec(&value).expect("mutated artifact JSON")
}

#[test]
fn relation_artifact_round_trips_the_complete_verifier_owned_structure() {
    let prep = preprocessing();
    let bytes = VerifierKeyRelationArtifact::to_json_vec(&prep).expect("export relation artifact");
    let receipt = VerifierKeyRelationArtifact::validate_json(&prep, &bytes).expect("validate relation artifact");

    assert_eq!(receipt.logical_rows(), 1);
    assert_eq!(receipt.assignment_fields(), D as u64);
    assert_eq!(receipt.padded_rows(), 64);
    assert_eq!(receipt.row_variables(), 6);
    assert_eq!(receipt.public_field_width(), Some(D as u64));
    assert_eq!(receipt.semantic_matrix_count(), 3);
    assert_eq!(receipt.joint_matrix_count(), 4);
    assert_eq!(receipt.polynomial_degree(), 2);
    assert_eq!(
        receipt.structure_digest(),
        prep.structure_digest().map(|field| {
            use p3_field::PrimeField64;
            field.as_canonical_u64()
        })
    );
    assert_eq!(receipt.verifier_key_digest(), prep.verifier_key().digest());
}

#[test]
fn relation_artifact_rejects_header_binding_matrix_and_encoding_mutations() {
    let prep = preprocessing();
    let bytes = VerifierKeyRelationArtifact::to_json_vec(&prep).expect("export relation artifact");
    let corruptions = [
        mutated(&bytes, |value| value["relation"]["logical_rows"] = json!(2)),
        mutated(&bytes, |value| value["binding"]["structure_digest"][0] = json!(1)),
        mutated(&bytes, |value| {
            value["structure"]["matrices"]
                .as_array_mut()
                .expect("matrix array")
                .swap(0, 1)
        }),
        mutated(&bytes, |value| value["unrecognized"] = json!(true)),
        {
            let mut noncanonical = bytes.clone();
            noncanonical.push(b'\n');
            noncanonical
        },
    ];

    for corrupted in corruptions {
        assert!(
            VerifierKeyRelationArtifact::validate_json(&prep, &corrupted).is_err(),
            "mutated relation artifact must fail closed"
        );
    }
}

#[test]
fn relation_artifact_rejects_another_same_shaped_verifier_key() {
    let prep = preprocessing();
    let bytes = VerifierKeyRelationArtifact::to_json_vec(&prep).expect("export relation artifact");
    let other = other_preprocessing();

    let error = VerifierKeyRelationArtifact::validate_json(&other, &bytes)
        .expect_err("same shape must not grant relation authority");
    assert!(matches!(
        error,
        RelationArtifactError::Mismatch("key binding") | RelationArtifactError::Mismatch("complete matrix payload")
    ));
}
