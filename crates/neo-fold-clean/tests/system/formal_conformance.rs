#[path = "../support/mod.rs"]
mod support;

use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_clean::paper::construction2::{FoldProof, ProofState};
use neo_fold_clean::{Error, RunningInstance, Uncompressed};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: u32,
    sources: Vec<SourceHash>,
    theorem_anchors: Vec<TheoremAnchor>,
}

#[derive(Debug, Deserialize)]
struct SourceHash {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct TheoremAnchor {
    path: String,
    name: String,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn manifest() -> Manifest {
    let path = repo_root().join("formal/nightstream-lean/assurance/rust-conformance-manifest.json");
    serde_json::from_str(&fs::read_to_string(path).expect("read formal conformance manifest"))
        .expect("parse formal conformance manifest")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn one_batch_finished() -> (neo_fold_clean::Preprocessing, Uncompressed) {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 700)]]).expect("one-batch proof");
    let finished = neo_fold_clean::finish_uncompressed(&prep, audit).expect("terminal fold");
    (prep, finished)
}

fn final_running(proof: &Uncompressed) -> RunningInstance {
    match &proof.state.proof {
        ProofState::Active { running, latest } => {
            assert!(latest.instances.is_empty(), "fixture must be finalized");
            running
                .materialize()
                .expect("CPU fixture has materialized running state")
        }
        ProofState::Initial => panic!("fixture must be active"),
    }
}

#[test]
fn conformance_manifest_fails_closed_on_rust_or_lean_drift() {
    let root = repo_root();
    let manifest = manifest();
    assert_eq!(manifest.schema, 1, "unsupported conformance manifest schema");
    assert!(!manifest.sources.is_empty(), "manifest must hash mapped sources");
    assert!(
        !manifest.theorem_anchors.is_empty(),
        "manifest must name theorem-facing Lean declarations"
    );

    for source in manifest.sources {
        let path = root.join(&source.path);
        let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert_eq!(
            sha256_hex(&bytes),
            source.sha256,
            "formal conformance drift at {}; review RUST-REFINE/TERM-CE before updating the manifest",
            source.path
        );
    }

    for anchor in manifest.theorem_anchors {
        let path = root.join(&anchor.path);
        let source = fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            source.contains(&anchor.name),
            "missing theorem anchor {} in {}",
            anchor.name,
            anchor.path
        );
    }
}

#[test]
fn terminal_ce_native_success_and_each_authority_rejection_are_live() {
    let (prep, finished) = one_batch_finished();
    neo_fold_clean::verify_uncompressed(&prep, &finished).expect("honest terminal proof verifies");

    let running = final_running(&finished);
    neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &running)
        .expect("honest terminal CE authority verifies");

    let mut bad_commitment = running.clone();
    bad_commitment.claims[0].c.data[0] += F::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_commitment),
        Err(Error::FinalAccumulatorWitnessCommitmentMismatch { index: 0 })
    ));

    let mut bad_projection = running.clone();
    bad_projection.claims[0].X.as_mut_slice()[0] += F::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_projection),
        Err(Error::FinalAccumulatorPublicInputMismatch { index: 0 })
    ));

    let mut bad_norm = running.clone();
    bad_norm.witnesses[0].as_mut_slice()[0] = F::from_u64(prep.params.b() as u64);
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_norm),
        Err(Error::FinalAccumulatorLowNormViolation { index: 0, .. })
    ));

    let mut bad_point = running.clone();
    bad_point.claims[0].r.pop();
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_point),
        Err(Error::FinalAccumulatorEvaluationPointShapeMismatch { index: 0, .. })
    ));

    let mut bad_evaluation = running.clone();
    bad_evaluation.claims[0].y_ring[0][0] += K::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_evaluation),
        Err(Error::FinalAccumulatorCeRelationViolation { index: 0, .. })
    ));

    let mut bad_constant_term = running.clone();
    bad_constant_term.claims[0].ct[0] += K::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_constant_term),
        Err(Error::FinalAccumulatorCtMismatch { index: 0, .. })
    ));

    let mut bad_sidecar = running.clone();
    bad_sidecar.claims[0].u_offset = 1;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_sidecar),
        Err(Error::FinalAccumulatorUnsupportedSidecar {
            index: 0,
            field: "u_offset",
            ..
        })
    ));

    let mut disconnected_child = finished.clone();
    let ProofState::Active { running, .. } = &mut disconnected_child.state.proof else {
        unreachable!("finished fixture is active")
    };
    let running = running
        .as_materialized_mut()
        .expect("CPU fixture has materialized running state");
    running.claims[0].c.data[0] += F::ONE;
    assert!(
        neo_fold_clean::verify_uncompressed(&prep, &disconnected_child).is_err(),
        "recorded terminal child disconnected from verifier-derived NIFS output must reject"
    );
}

#[test]
fn lifecycle_replay_exercises_fprime_success_and_rejection_paths() {
    let prep = support::toy_preprocessing();
    let batches = vec![
        vec![support::toy_instance(&prep, 801)],
        vec![support::toy_instance(&prep, 802)],
    ];
    let audit = neo_fold_clean::prove(&prep, batches).expect("two-step F' audit");
    let finished = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("terminal fold");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finished).expect("honest lifecycle replay");

    let mut wrong_variant = finished.clone();
    wrong_variant.steps[1].fold = FoldProof::NoFold;
    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &wrong_variant).is_err(),
        "active state with NoFold proof must reject"
    );

    let mut empty_step = finished.clone();
    empty_step.public_batches[1].clear();
    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &empty_step).is_err(),
        "empty installed batch must reject"
    );

    let mut semantic_forgery = finished.clone();
    semantic_forgery.steps[0].semantic_state_digest[0] ^= 0x80;
    assert!(
        neo_fold_clean::verify_uncompressed_audit(&prep, &semantic_forgery).is_err(),
        "stateless semantic-state forgery must reject"
    );
}

#[test]
fn compact_decider_is_explicitly_fail_closed() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 901)]]).expect("one-step audit");

    assert!(matches!(
        neo_fold_clean::compress(&prep, audit),
        Err(Error::Decider(neo_fold_clean::paper::decider::Error::Unsupported))
    ));
}
