#[path = "formal_conformance/canonical_public_link.rs"]
mod canonical_public_link;
#[path = "formal_conformance/canonical_public_link_program_export.rs"]
mod canonical_public_link_program_export;
#[path = "formal_conformance/canonical_step_export.rs"]
mod canonical_step_export;
#[path = "formal_conformance/canonical_terminal_export.rs"]
mod canonical_terminal_export;
#[path = "formal_conformance/native_step_export.rs"]
mod native_step_export;
#[path = "formal_conformance/state_x_out_program_export.rs"]
mod state_x_out_program_export;
#[path = "../support/mod.rs"]
mod support;
#[path = "formal_conformance/terminal_link_program_export.rs"]
mod terminal_link_program_export;
#[path = "formal_conformance/terminal_native_guard_export.rs"]
mod terminal_native_guard_export;

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

fn record_artifact_drift(path: &Path, rendered: &str, drifted: &mut Vec<PathBuf>) {
    if !fs::read_to_string(path).is_ok_and(|committed| committed == rendered) {
        drifted.push(path.to_owned());
    }
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
            running.clone()
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
    bad_norm.witnesses[0].as_mut_slice()[0] = F::from_u64(prep.params().b() as u64);
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

    let mut bad_eval_k = running.clone();
    bad_eval_k.claims[0].eval_k[0] += K::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_eval_k),
        Err(Error::FinalAccumulatorCeRelationViolation { index: 0, .. })
    ));

    let mut bad_eval_a = running.clone();
    bad_eval_a.claims[0].eval_a[0][0] += K::ONE;
    assert!(matches!(
        neo_fold_clean::lifecycle::validate_final_witness_authority(&prep, &bad_eval_a),
        Err(Error::FinalAccumulatorCeRelationViolation { index: 0, .. })
    ));

    let mut disconnected_child = finished.clone();
    let ProofState::Active { running, .. } = &mut disconnected_child.state.proof else {
        unreachable!("finished fixture is active")
    };
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
fn native_verify_step_receipts_are_exact_and_deterministic() {
    let (json, lean) = native_step_export::checked_native_step_receipts();
    let json_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/system/formal_conformance/native_step_receipts.json");
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/NativeStep/Generated/Receipts.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&json_path, &json, &mut drifted);
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "native verify_step receipt drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn native_public_link_program_is_exact_and_deterministic() {
    let lean = canonical_public_link_program_export::checked_canonical_public_link_program();
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/NativeStep/Generated/PublicInputLinkProgram.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "native public-link program drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn native_state_x_out_programs_are_exact_and_deterministic() {
    let lean = state_x_out_program_export::checked_state_x_out_programs();
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/NativeStep/Generated/StateXOutProgram.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "native state_x_out programs drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn terminal_link_program_is_exact_and_deterministic() {
    let lean = terminal_link_program_export::checked_terminal_link_program();
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/TerminalLink/Generated/Program.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "terminal-link source program drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn terminal_native_guard_names_are_exact_and_deterministic() {
    let lean = terminal_native_guard_export::checked_terminal_native_guard_names();
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/\
         TerminalVerifierNativeGuards/Generated/Names.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "terminal native guard ledger drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn linked_canonical_step_cases_are_exact_and_deterministic() {
    let (json, lean) = canonical_step_export::checked_canonical_step_cases();
    let json_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/system/formal_conformance/canonical_step_cases.json");
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/OneSlot/Generated/StepCases.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&json_path, &json, &mut drifted);
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "linked canonical-step corpus drifted; frozen reference differs: {drifted:?}"
    );
}

#[test]
fn linked_canonical_terminal_cases_are_exact_and_deterministic() {
    let (json, lean) = canonical_terminal_export::checked_canonical_terminal_cases();
    let json_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/system/formal_conformance/canonical_terminal_cases.json");
    let lean_path = repo_root().join(
        "formal/nightstream-lean/Nightstream/Implementation/Rust/\
         CanonicalConformance/OneSlot/Generated/TerminalCases.lean",
    );
    let mut drifted = Vec::new();
    record_artifact_drift(&json_path, &json, &mut drifted);
    record_artifact_drift(&lean_path, &lean, &mut drifted);
    assert!(
        drifted.is_empty(),
        "linked canonical-terminal corpus drifted; frozen reference differs: {drifted:?}"
    );
}
