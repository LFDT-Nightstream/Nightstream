//! Runtime provenance envelope for the canonical one-slot Rust execution.
//!
//! The committed corpus remains a drift fixture. This module performs a fresh
//! Rust execution, binds the process and source identity, and emits a fresh
//! Lean replay program. The independent Python and Lean checks are run by the
//! formal validation wrapper; this producer does not assert its own evidence.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const EVIDENCE_SCHEMA: u64 = 3;

#[derive(Clone, Copy)]
enum Scope {
    Step,
    Terminal,
}

impl Scope {
    fn stem(self) -> &'static str {
        match self {
            Self::Step => "native-step",
            Self::Terminal => "native-terminal",
        }
    }

    fn model_id(self) -> &'static str {
        match self {
            Self::Step => "nightstream-lean-one-slot-step-v1",
            Self::Terminal => "nightstream-lean-one-slot-terminal-v1",
        }
    }

    fn predicate(self) -> &'static str {
        match self {
            Self::Step => "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.stepAgrees",
            Self::Terminal => "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.terminalAgrees",
        }
    }
}

pub struct EvidencePaths {
    pub evidence: PathBuf,
    pub corpus: PathBuf,
    pub lean_replay: PathBuf,
}

pub struct EvidenceSet {
    pub step: EvidencePaths,
    pub terminal: EvidencePaths,
}

pub fn emit() -> EvidenceSet {
    let root = repo_root();
    let output = root.join("formal/nightstream-lean/.lake/build/rust-origin");
    fs::create_dir_all(&output).expect("create Rust-origin evidence directory");
    EvidenceSet {
        step: emit_scope(&root, &output, Scope::Step),
        terminal: emit_scope(&root, &output, Scope::Terminal),
    }
}

fn emit_scope(root: &Path, output: &Path, scope: Scope) -> EvidencePaths {
    let (corpus_text, generated_lean) = match scope {
        Scope::Step => super::canonical_step_export::checked_canonical_step_cases(),
        Scope::Terminal => super::canonical_terminal_export::checked_canonical_terminal_cases(),
    };
    let corpus: Value = serde_json::from_str(&corpus_text).expect("parse fresh canonical-step corpus");
    let replay = runtime_lean_replay(scope, &generated_lean);
    let stem = scope.stem();

    let paths = EvidencePaths {
        evidence: output.join(format!("{stem}-evidence.json")),
        corpus: output.join(format!("{stem}-corpus.json")),
        lean_replay: output.join(format!("{stem}-replay.lean")),
    };
    fs::write(&paths.corpus, &corpus_text).expect("write fresh Rust-origin corpus");
    fs::write(&paths.lean_replay, &replay).expect("write fresh Rust-origin Lean replay");

    let evidence = evidence(scope, root, &corpus, corpus_text.trim_end(), &replay);
    fs::write(
        &paths.evidence,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&evidence).expect("serialize Rust-origin evidence")
        ),
    )
    .expect("write Rust-origin evidence envelope");
    paths
}

fn evidence(scope: Scope, root: &Path, corpus: &Value, corpus_text: &str, replay: &str) -> Value {
    let primary = case(corpus, "honest_recursive");
    let trace = trace(scope, primary);
    let decision = decision(scope, primary);
    let mutations = cases(corpus)
        .filter(|case| case["mutation"] != "none")
        .map(|case| mutation(scope, case))
        .collect::<Vec<_>>();
    assert!(
        !mutations.is_empty(),
        "Rust-origin evidence needs adversarial mutations"
    );

    let profile_json = canonical_json(&corpus["profile"]);
    let payload = json!({
        "corpus_json": corpus_text,
        "primary_case": "honest_recursive",
        "profile_json": profile_json,
        "replay_program": replay,
    });
    let input = json!({
        "encoding": "canonical-json-rfc8785",
        "payload": payload,
        "payload_hash": hash_value(&payload),
    });

    let binary = std::env::current_exe().expect("resolve Rust evidence producer binary");
    let binary_hash = hash_file(&binary);
    let revision = git(root, &["rev-parse", "HEAD"]);
    let dirty = !git(root, &["status", "--porcelain=v1", "--untracked-files=all"])
        .trim()
        .is_empty();
    let rustc = command_text(root, "rustc", &["-vV"]);
    let target = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .expect("rustc -vV includes host target")
        .to_owned();
    let command = std::env::args().collect::<Vec<_>>();
    let source_tree_hash = source_tree_hash(root);
    let event_log_hash = hash_value(&Value::Array(trace.clone()));
    let attested = json!({
        "input": input,
        "mutations": mutations,
        "rust_decision": decision,
        "trace": trace,
    });
    let attestation_hash = hash_value(&attested);
    let invocation_id =
        hash_bytes(format!("{revision}\n{source_tree_hash}\n{binary_hash}\n{event_log_hash}").as_bytes());

    let mut envelope = json!({
        "schema_version": EVIDENCE_SCHEMA,
        "contract_id": "nightstream-superneo-v1",
        "contract_hash": hash_file(&root.join("protocol-contract/superneo-v1.md")),
        "contract_rule": "NS-RUST-EVIDENCE-CONTENT",
        "profile_id": corpus["profile"]["name"],
        "profile_hash": hash_bytes(profile_json.as_bytes()),
        "origin": "rust-execution",
        "rust_revision": revision,
        "source_tree_hash": source_tree_hash,
        "features": active_features(),
        "producer": {
            "crate": "neo-fold-clean",
            "binary": binary.to_string_lossy(),
            "command": command,
            "binary_hash": binary_hash,
            "dirty": dirty,
            "cargo_lock_hash": hash_file(&root.join("Cargo.lock")),
            "rustc": rustc,
            "target": target,
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        },
        "run": {
            "runner_id": "local-cargo-test-process",
            "invocation_id": invocation_id,
            "exit_code": 0,
            "event_log_hash": event_log_hash,
            "attestation": {
                "format": "nightstream-local-process-v1",
                "identity": format!("{revision}:{binary_hash}"),
                "payload_hash": attestation_hash,
            },
        },
        "input": attested["input"].clone(),
        "rust_decision": attested["rust_decision"].clone(),
        "trace": attested["trace"].clone(),
        "semantic_target": {
            "model_id": scope.model_id(),
            "model_hash": hash_file(&root.join(
                "formal/nightstream-lean/Nightstream/Implementation/Rust/CanonicalConformance/OneSlot.lean"
            )),
            "predicate": scope.predicate(),
            "checker": format!(
                "formal/nightstream-lean/.lake/build/rust-origin/{}-replay.lean",
                scope.stem()
            ),
        },
        "mutations": attested["mutations"].clone(),
    });
    let content_hash = hash_value(&envelope);
    envelope
        .as_object_mut()
        .expect("evidence envelope is an object")
        .insert("content_hash".to_owned(), Value::String(content_hash));
    envelope
}

fn runtime_lean_replay(scope: Scope, generated: &str) -> String {
    let (namespace, predicate) = match scope {
        Scope::Step => (
            "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated",
            "stepAgrees",
        ),
        Scope::Terminal => (
            "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.Generated.Terminal",
            "terminalAgrees",
        ),
    };
    format!(
        "{generated}\n\nnamespace {namespace}\n\n\
open Nightstream.Implementation.Rust.CanonicalConformance.OneSlot\n\n\
/-- Fresh Rust-origin cases agree with the independent canonical checker. -/\n\
theorem rustOriginAllAgree : all.all {predicate} = true := by\n  native_decide\n\n\
end {namespace}\n"
    )
}

fn cases(corpus: &Value) -> impl Iterator<Item = &Value> {
    corpus["cases"]
        .as_array()
        .expect("canonical-step corpus has cases")
        .iter()
}

fn case<'a>(corpus: &'a Value, name: &str) -> &'a Value {
    cases(corpus)
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("canonical-step corpus omitted {name}"))
}

fn decision(scope: Scope, case: &Value) -> Value {
    let accepted = case["observed"]["rust_error"].is_null();
    if matches!(scope, Scope::Step) {
        assert_eq!(
            accepted,
            !case["observed"]["rust_output"].is_null(),
            "observed Rust result must contain exactly one success value or error"
        );
    }
    assert_eq!(
        case["mapped"]["rust_accepted"]
            .as_bool()
            .expect("mapped Rust decision is Boolean"),
        accepted,
        "mapped Rust decision must equal the observed execution result"
    );
    let result_code = if accepted {
        "ACCEPT".to_owned()
    } else {
        case["observed"]["rust_error"]
            .as_str()
            .expect("rejected Rust case has an error")
            .to_owned()
    };
    let first_reject_rule = if accepted {
        Value::Null
    } else {
        Value::String(rejection_rule(
            scope,
            case["mutation"]
                .as_str()
                .expect("mutation identifier is a string"),
        ))
    };
    let output_hash = if accepted {
        Value::String(hash_value(match scope {
            Scope::Step => &case["observed"]["rust_output"],
            Scope::Terminal => &Value::Null,
        }))
    } else {
        Value::Null
    };
    let mut value = json!({
        "accepted": accepted,
        "result_code": result_code,
        "first_reject_rule": first_reject_rule,
        "output_hash": output_hash,
    });
    let decision_hash = hash_value(&value);
    value
        .as_object_mut()
        .expect("decision is an object")
        .insert("decision_hash".to_owned(), Value::String(decision_hash));
    value
}

fn mutation(scope: Scope, case: &Value) -> Value {
    let mutation_id = case["name"]
        .as_str()
        .expect("mutation case name is a string");
    let field = case["mutation"]
        .as_str()
        .expect("mutation field is a string");
    json!({
        "mutation_id": mutation_id,
        "field": field,
        "operation": "replace-with-adversarial-value",
        "input_payload_hash": hash_value(&case["rust_input"]),
        "trace_hash": hash_value(&case["observed"]),
        "rust_decision": decision(scope, case),
    })
}

fn trace(scope: Scope, case: &Value) -> Vec<Value> {
    let input_hash = hash_value(&case["rust_input"]);
    let events = match scope {
        Scope::Step => case["observed"]["event_order"]
            .as_array()
            .expect("observed Rust event order is an array")
            .iter()
            .map(|event| event.as_str().expect("Rust event name is a string"))
            .collect(),
        Scope::Terminal => vec![
            "terminal_public_link",
            "terminal_running_relation",
            "terminal_fresh_relation",
        ],
    };
    events
        .into_iter()
        .enumerate()
        .map(|(seq, event)| {
            let (kind, rule, symbol) = event_contract(event);
            json!({
                "seq": seq,
                "phase": event,
                "kind": kind,
                "contract_rule": rule,
                "source_symbol": symbol,
                "input_hash": input_hash,
                "output_hash": hash_value(&json!({
                    "event": event,
                    "observed": case["observed"],
                    "seq": seq,
                })),
                "data": { "event_id": event },
            })
        })
        .collect()
}

fn event_contract(event: &str) -> (&'static str, &'static str, &'static str) {
    match event {
        "chunk_digest" => (
            "derive",
            "NS-TRANSCRIPT-ORDER",
            "neo_fold_clean::paper::digest::f_prime_chunk_public_digest",
        ),
        "dispatch" => (
            "shape-check",
            "SN-FOLD-TYPE",
            "neo_fold_clean::paper::construction2::verify_step_with_execution_receipt",
        ),
        "transcript_started" | "transcript_append" | "transcript_prefix" => (
            "absorb",
            "NS-TRANSCRIPT-FRAMING",
            "neo_fold_clean::paper::f_prime::native::f_prime_step_transcript",
        ),
        "nifs_verify" => ("relation-check", "SN-FOLD-PROOF", "neo_fold_clean::paper::nifs::verify"),
        "running_digest" => (
            "derive",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::RunningInstance::accumulator_digest",
        ),
        "state_advanced" => (
            "derive",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::advance_state",
        ),
        "verifier_digest_read" => (
            "derive",
            "NS-VERIFIER-KEY-DIGEST",
            "neo_fold_clean::paper::construction2::VerifierKey::digest",
        ),
        "pi_ccs_header_read" => (
            "derive",
            "NS-PICCS-VARIANT",
            "neo_fold_clean::lifecycle::Preprocessing::pi_ccs_header_bundle",
        ),
        "state_x_out_hash" => (
            "output",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::compute_x_out",
        ),
        "terminal_public_link" => (
            "relation-check",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::lifecycle::validate_required_f_prime_latest_link",
        ),
        "terminal_running_relation" => (
            "relation-check",
            "SN-REL-CE",
            "neo_fold_clean::lifecycle::validate_final_witness_authority",
        ),
        "terminal_fresh_relation" => (
            "relation-check",
            "SN-REL-CCS",
            "neo_fold_clean::lifecycle::validate_latest_witness_authority",
        ),
        other => panic!("unmapped Rust execution event {other}"),
    }
}

fn rejection_rule(scope: Scope, mutation: &str) -> String {
    match (scope, mutation) {
        (Scope::Step, "state.z_i[0] ^= 1")
        | (Scope::Step, "state.pc := 2")
        | (Scope::Step, "proof.x_out.bytes[0] ^= 1") => "NS-AUTH-DERIVED",
        (Scope::Step, "proof.fold := recursive") | (Scope::Step, "proof.fold := no_fold") => "SN-FOLD-TYPE",
        (Scope::Step, "state.latest[0].claim.x[1] += 1") => "NS-AUTH-CLAIM",
        (Scope::Step, "proof.nifs.pi_dec.children[0].commitment[0] += 1") => "SN-PIDEC-EQUATIONS",
        (Scope::Terminal, "state.z_i[0] ^= 1") => "NS-AUTH-DERIVED",
        (Scope::Terminal, "state.pc := 2") => "SN-FOLD-TYPE",
        (Scope::Terminal, "state.latest := prior honest latest") => "NS-AUTH-CLAIM",
        (Scope::Terminal, "state.running.witnesses[0][0] += 1") => "SN-REL-CE",
        (Scope::Terminal, "latest.private[m_in] toggled and consistently recommitted") => "SN-REL-CCS",
        (_, other) => panic!("unmapped Rust mutation {other}"),
    }
    .to_owned()
}

fn active_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "perf-timers") {
        features.push("perf-timers");
    }
    features
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
        .expect("list Rust build and assurance inputs");
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

fn git(root: &Path, args: &[&str]) -> String {
    command_text(root, "git", args)
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
