//! Exact verifier-key relation artifact checks.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::{RelationArtifactError, VerifierKeyRelationArtifact};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

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
    assert_eq!(receipt.verifier_key_digest(), prep.vk.digest());
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

#[test]
fn rust_origin_relation_artifact_evidence_is_emitted_for_independent_checks() {
    let root = repo_root();
    let output = root.join("formal/nightstream-lean/.lake/build/rust-origin");
    fs::create_dir_all(&output).expect("create Rust-origin evidence directory");

    let prep = preprocessing();
    let other = other_preprocessing();
    let authoritative = VerifierKeyRelationArtifact::to_json_vec(&prep).expect("authoritative relation artifact");
    let cases = vec![
        evidence_case("honest", "none", &prep, authoritative.clone()),
        evidence_case(
            "logical_rows",
            "relation.logical_rows += 1",
            &prep,
            mutated(&authoritative, |value| value["relation"]["logical_rows"] = json!(2)),
        ),
        evidence_case(
            "binding_digest",
            "binding.structure_digest[0] += 1",
            &prep,
            mutated(&authoritative, |value| {
                value["binding"]["structure_digest"][0] = json!(1)
            }),
        ),
        evidence_case(
            "matrix_order",
            "structure.matrices[0..2] swapped",
            &prep,
            mutated(&authoritative, |value| {
                value["structure"]["matrices"]
                    .as_array_mut()
                    .expect("matrix array")
                    .swap(0, 1)
            }),
        ),
        evidence_case(
            "source_kind",
            "source.kind replaced",
            &prep,
            mutated(&authoritative, |value| {
                value["source"]["kind"] = json!("unrecognized-source")
            }),
        ),
        evidence_case(
            "unknown_field",
            "unknown top-level field added",
            &prep,
            mutated(&authoritative, |value| value["unrecognized"] = json!(true)),
        ),
        {
            let mut candidate = authoritative.clone();
            candidate.push(b'\n');
            evidence_case("noncanonical", "trailing newline added", &prep, candidate)
        },
        evidence_case(
            "other_verifier_key",
            "live verifier key replaced",
            &other,
            authoritative.clone(),
        ),
    ];
    assert_eq!(
        cases
            .iter()
            .filter(|case| case["rust_accepted"] == true)
            .count(),
        1,
        "only the exact verifier-owned artifact may pass"
    );

    let replay = relation_artifact_lean_replay(&cases);
    let replay_path = output.join("relation-artifact-replay.lean");
    fs::write(&replay_path, &replay).expect("write relation artifact Lean replay");

    let binary = std::env::current_exe().expect("resolve evidence producer binary");
    let model_path = root.join("formal/nightstream-lean/Nightstream/Assurance/RelationArtifactBinding.lean");
    let mut evidence = json!({
        "schema_version": 1,
        "contract_id": "nightstream-superneo-v1",
        "contract_hash": hash_file(&root.join("protocol-contract/superneo-v1.md")),
        "profile_id": "verifier-key-relation-artifact-v1",
        "origin": "rust-execution",
        "rust_revision": command_text(&root, "git", &["rev-parse", "HEAD"]),
        "source_tree_hash": source_tree_hash(&root),
        "producer": {
            "crate": "neo-fold-clean",
            "binary": binary.to_string_lossy(),
            "binary_hash": hash_file(&binary),
            "cargo_lock_hash": hash_file(&root.join("Cargo.lock")),
            "command": std::env::args().collect::<Vec<_>>(),
            "rustc": command_text(&root, "rustc", &["-vV"]),
        },
        "semantic_target": {
            "model_id": "nightstream-relation-artifact-exact-binding-v1",
            "model_hash": hash_file(&model_path),
            "predicate": "Nightstream.Assurance.RelationArtifactBinding.ExactValidation",
            "checker": "formal/nightstream-lean/.lake/build/rust-origin/relation-artifact-replay.lean",
            "replay_hash": hash_bytes(replay.as_bytes()),
        },
        "authoritative_artifact_json": String::from_utf8(authoritative).expect("artifact JSON is UTF-8"),
        "cases": cases,
    });
    let content_hash = hash_value(&evidence);
    evidence
        .as_object_mut()
        .expect("evidence is an object")
        .insert("content_hash".to_owned(), Value::String(content_hash));
    fs::write(
        output.join("relation-artifact-evidence.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&evidence).expect("serialize relation artifact evidence")
        ),
    )
    .expect("write relation artifact evidence");
}

fn evidence_case(name: &str, mutation: &str, live: &neo_fold_clean::Preprocessing, candidate: Vec<u8>) -> Value {
    let live_artifact = VerifierKeyRelationArtifact::to_json_vec(live).expect("live relation artifact");
    let rust_accepted = VerifierKeyRelationArtifact::validate_json(live, &candidate).is_ok();
    json!({
        "name": name,
        "mutation": mutation,
        "live_artifact_json": String::from_utf8(live_artifact).expect("live artifact JSON is UTF-8"),
        "candidate_artifact_json": String::from_utf8(candidate).expect("candidate artifact JSON is UTF-8"),
        "rust_accepted": rust_accepted,
    })
}

fn relation_artifact_lean_replay(cases: &[Value]) -> String {
    let rendered_cases = cases
        .iter()
        .map(|case| {
            format!(
                "  {{ live := {}, carried := {}, rustAccepted := {} }}",
                serde_json::to_string(
                    case["live_artifact_json"]
                        .as_str()
                        .expect("live artifact string")
                )
                .expect("encode Lean live string"),
                serde_json::to_string(
                    case["candidate_artifact_json"]
                        .as_str()
                        .expect("candidate artifact string")
                )
                .expect("encode Lean candidate string"),
                case["rust_accepted"]
                    .as_bool()
                    .expect("Rust decision is Boolean")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "import Nightstream.Assurance.RelationArtifactBinding\n\n\
namespace Nightstream.Assurance.RelationArtifactBinding.Generated\n\n\
open Nightstream.Assurance.RelationArtifactBinding\n\n\
structure Case where\n  live : String\n  carried : String\n  rustAccepted : Bool\n\n\
def model (bytes : String) : Artifact String Nat String String where\n  format := artifactFormat\n  schema := artifactSchema\n  matrixPayloadEncoding := \"complete-canonical-json\"\n  source := \"rust-origin\"\n  shape := {{ logicalRows := 0, assignmentFields := 0, paddedRows := 0, rowVariables := 0, \
publicStartField := 0, publicFields := 0, semanticMatrixCount := 0, jointMatrixCount := 0, \
polynomialDegree := 0 }}\n  \
structureDigest := \"bound-in-bytes\"\n  matrixDigest := \"bound-in-bytes\"\n  \
ajtaiPublicParametersDigest := \"bound-in-bytes\"\n  verifierKeyDigest := \"bound-in-bytes\"\n  \
matrices := [bytes]\n  polynomial := 0\n\n\
def cases : List Case := [\n{rendered_cases}\n]\n\n\
def agrees (case : Case) : Bool :=\n  \
ExactValidation (model case.live) (model case.carried) == case.rustAccepted\n\n\
/-- Fresh Rust decisions agree with complete canonical artifact equality. -/\n\
theorem rustOriginAllAgree : cases.all agrees = true := by\n  native_decide\n\n\
end Nightstream.Assurance.RelationArtifactBinding.Generated\n"
    )
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical repository root")
}

fn source_tree_hash(root: &Path) -> String {
    let output = Command::new("git")
        .current_dir(root)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            ".cargo",
            "crates",
            "protocol-contract",
            "formal/nightstream-lean",
        ])
        .output()
        .expect("list Rust and assurance inputs");
    assert!(output.status.success(), "git ls-files failed");
    let paths = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8(path.to_vec()).expect("repository paths are UTF-8"))
        .filter(|path| root.join(path).is_file())
        .collect::<BTreeSet<_>>();
    let mut hasher = Sha256::new();
    for path in paths {
        let bytes = fs::read(root.join(&path)).unwrap_or_else(|error| panic!("read source input {path}: {error}"));
        hash_framed(&mut hasher, path.as_bytes());
        hash_framed(&mut hasher, &bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_framed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(
        u64::try_from(bytes.len())
            .expect("hash frame length fits u64")
            .to_be_bytes(),
    );
    hasher.update(bytes);
}

fn command_text(root: &Path, program: &str, args: &[&str]) -> String {
    let output = Command::new(program)
        .current_dir(root)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("run {program}: {error}"));
    assert!(output.status.success(), "{program} {args:?} failed");
    String::from_utf8(output.stdout)
        .expect("command output is UTF-8")
        .trim()
        .to_owned()
}

fn hash_file(path: &Path) -> String {
    hash_bytes(&fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())))
}

fn hash_value(value: &Value) -> String {
    hash_bytes(canonical_json(value).as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("serialize canonical JSON string"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => canonical_object(values),
    }
}

fn canonical_object(values: &Map<String, Value>) -> String {
    let mut keys = values.keys().collect::<Vec<_>>();
    keys.sort_unstable();
    format!(
        "{{{}}}",
        keys.into_iter()
            .map(|key| format!(
                "{}:{}",
                serde_json::to_string(key).expect("serialize canonical JSON key"),
                canonical_json(&values[key])
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}
