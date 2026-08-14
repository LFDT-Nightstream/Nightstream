#!/usr/bin/env python3
"""Independently check one runtime Rust verifier evidence envelope."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "contract_hash",
    "contract_rule",
    "profile_id",
    "profile_hash",
    "origin",
    "rust_revision",
    "source_tree_hash",
    "features",
    "producer",
    "run",
    "input",
    "rust_decision",
    "trace",
    "semantic_target",
    "mutations",
    "content_hash",
}
PRODUCER_KEYS = {
    "crate",
    "binary",
    "command",
    "binary_hash",
    "dirty",
    "cargo_lock_hash",
    "rustc",
    "target",
    "profile",
}
RUN_KEYS = {"runner_id", "invocation_id", "exit_code", "event_log_hash", "attestation"}
ATTESTATION_KEYS = {"format", "identity", "payload_hash"}
INPUT_KEYS = {"encoding", "payload", "payload_hash"}
DECISION_KEYS = {"accepted", "result_code", "first_reject_rule", "output_hash", "decision_hash"}
TRACE_KEYS = {
    "seq",
    "phase",
    "kind",
    "contract_rule",
    "source_symbol",
    "input_hash",
    "output_hash",
    "data",
}
MUTATION_KEYS = {
    "mutation_id",
    "field",
    "operation",
    "input_payload_hash",
    "trace_hash",
    "rust_decision",
}
SEMANTIC_TARGET_KEYS = {"model_id", "model_hash", "predicate", "checker"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_value(value: Any) -> str:
    return hash_bytes(canonical_json(value).encode())


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


def command(root: Path, *args: str) -> str:
    result = subprocess.run(args, cwd=root, check=True, capture_output=True)
    return result.stdout.decode().strip()


def tracked_source_paths(root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
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
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = {part.decode() for part in result.stdout.split(b"\0") if part}
    return sorted(relative for relative in paths if (root / relative).is_file())


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in tracked_source_paths(root):
        path = root / relative
        data = path.read_bytes()
        for frame in (relative.encode(), data):
            digest.update(len(frame).to_bytes(8, "big"))
            digest.update(frame)
    return digest.hexdigest()


def rejection_rule(scope: str, mutation: str) -> str:
    rules = {
        ("step", "state.z_i[0] ^= 1"): "NS-AUTH-DERIVED",
        ("step", "state.pc := 2"): "NS-AUTH-DERIVED",
        ("step", "proof.x_out.bytes[0] ^= 1"): "NS-AUTH-DERIVED",
        ("step", "proof.fold := recursive"): "SN-FOLD-TYPE",
        ("step", "proof.fold := no_fold"): "SN-FOLD-TYPE",
        ("step", "state.latest[0].claim.x[1] += 1"): "NS-AUTH-CLAIM",
        ("step", "proof.nifs.pi_dec.children[0].commitment[0] += 1"): "SN-PIDEC-EQUATIONS",
        ("terminal", "state.z_i[0] ^= 1"): "NS-AUTH-DERIVED",
        ("terminal", "state.pc := 2"): "SN-FOLD-TYPE",
        ("terminal", "state.latest := prior honest latest"): "NS-AUTH-CLAIM",
        ("terminal", "state.running.witnesses[0][0] += 1"): "SN-REL-CE",
        ("terminal", "latest.private[m_in] toggled and consistently recommitted"): "SN-REL-CCS",
    }
    if (scope, mutation) in rules:
        return rules[(scope, mutation)]
    raise ValueError(f"unmapped mutation {mutation!r}")


def live(value: int) -> bool:
    return value != 0


def expected_hash_input(
    mapped: dict[str, Any], current: int, running: int, iteration: int
) -> dict[str, Any]:
    return {
        "verifier_key": mapped["verifier_key"],
        "iteration": iteration,
        "z0": mapped["z0"],
        "current": current,
        "running": running,
        "pc": 1,
    }


def hash_receipt_output(receipt: Any, expected_input: dict[str, Any]) -> int:
    if receipt is None or receipt["input"] != expected_input:
        return 0
    return receipt["output"]


def semantic_step_accepted(case: dict[str, Any]) -> bool:
    """Replay the independent Lean one-slot checker without its Rust Boolean."""
    mapped = case["mapped"]
    receipt = mapped["step_receipt"]
    trace = mapped["trace"]
    claim = mapped["claim"]
    common_schema = (
        receipt["state"] == mapped["zi"]
        and receipt["witness"] == mapped["witness"]
        and live(receipt["output"])
        and live(claim["z_next"])
        and live(claim["running_next"])
        and live(claim["x"])
    )
    if not common_schema:
        return False

    if trace["branch"] == "base":
        if mapped["iteration"] != 0:
            return False
        next_hash = trace["next_hash"]
        if mapped["z0"] != mapped["zi"]:
            return False
        expected_next = expected_hash_input(
            mapped, receipt["output"], mapped["default_running"], mapped["iteration"] + 1
        )
        schema = (
            next_hash is not None
            and next_hash["input"] == expected_next
            and live(next_hash["output"])
        )
        if not schema:
            return False
        output = {
            "z_next": receipt["output"],
            "running_next": mapped["default_running"],
            "pc_next": 0,
            "x": hash_receipt_output(next_hash, expected_next),
        }
        return output == claim

    require(trace["branch"] == "recursive", "unknown mapped step branch")
    if mapped["iteration"] == 0:
        return False
    if mapped["prior_pc"] != 1:
        return False

    prior_hash = trace["prior_hash"]
    fresh_public = trace["fresh_public"]
    encode = trace["encode"]
    nifs = trace["nifs"]
    next_hash = trace["next_hash"]
    if prior_hash is None or fresh_public is None or encode is None:
        return False
    expected_prior = {
        "verifier_key": mapped["verifier_key"],
        "iteration": mapped["iteration"],
        "z0": mapped["z0"],
        "current": mapped["zi"],
        "running": mapped["running"],
        "pc": mapped["prior_pc"],
    }
    schema_prefix = (
        prior_hash["input"] == expected_prior
        and fresh_public["input"] == mapped["fresh"]
        and encode["input"] == prior_hash["output"]
        and live(prior_hash["output"])
        and live(fresh_public["output"])
        and live(encode["output"])
    )
    if not schema_prefix:
        return False
    if fresh_public["output"] != encode["output"]:
        return False
    if nifs is None:
        return False
    nifs_inputs_match = (
        nifs["key"] == mapped["verifier_key"]
        and nifs["running"] == mapped["running"]
        and nifs["fresh"] == mapped["fresh"]
        and nifs["proof"] == mapped["nifs_proof"]
    )
    if not nifs_inputs_match:
        return False
    folded = nifs["output"]
    if folded is None:
        return False
    expected_next = expected_hash_input(
        mapped, receipt["output"], folded, mapped["iteration"] + 1
    )
    schema_suffix = (
        next_hash is not None
        and next_hash["input"] == expected_next
        and live(folded)
        and live(next_hash["output"])
    )
    if not schema_suffix:
        return False
    output = {
        "z_next": receipt["output"],
        "running_next": folded,
        "pc_next": 0,
        "x": hash_receipt_output(next_hash, expected_next),
    }
    return output == claim


def semantic_terminal_accepted(case: dict[str, Any]) -> bool:
    """Replay the independent Lean one-slot terminal checker."""
    mapped = case["mapped"]
    trace = mapped["trace"]
    if trace["branch"] == "base":
        return mapped["iteration"] == 0 and mapped["zi"] == mapped["z0"]

    require(trace["branch"] == "recursive", "unknown mapped terminal branch")
    prior_hash = trace["prior_hash"]
    fresh_public = trace["fresh_public"]
    encode = trace["encode"]
    running = trace["running_relation"]
    fresh = trace["fresh_relation"]
    expected_prior = {
        "verifier_key": mapped["verifier_key"],
        "iteration": mapped["iteration"],
        "z0": mapped["z0"],
        "current": mapped["zi"],
        "running": mapped["running"],
        "pc": mapped["pc"],
    }
    schema = (
        mapped["iteration"] != 0
        and mapped["pc"] == 1
        and prior_hash["input"] == expected_prior
        and fresh_public["input"] == mapped["fresh"]
        and encode["input"] == prior_hash["output"]
        and fresh_public["output"] == encode["output"]
        and running["key"] == mapped["verifier_key"]
        and running["value"] == mapped["running"]
        and running["witness"] == mapped["running_witness"]
        and fresh["key"] == mapped["verifier_key"]
        and fresh["value"] == mapped["fresh"]
        and fresh["witness"] == mapped["fresh_witness"]
        and live(prior_hash["output"])
        and live(fresh_public["output"])
        and live(encode["output"])
    )
    return schema and running["accepted"] and fresh["accepted"]


def observed_rust_accepted(scope: str, case: dict[str, Any]) -> bool:
    observed = case["observed"]
    accepted = observed["rust_error"] is None
    if scope == "step":
        require(
            accepted == (observed["rust_output"] is not None),
            f"{case['name']} has an incomplete Rust result",
        )
    elif case["mapped"]["trace"]["branch"] == "recursive":
        checks = (
            observed["link_accepted"],
            observed["running_relation_accepted"],
            observed["fresh_relation_accepted"],
        )
        require(all(value is not None for value in checks), f"{case['name']} omits a Rust check")
        require(accepted == all(checks), f"{case['name']} Rust terminal result is inconsistent")
    return accepted


def expected_decision(scope: str, case: dict[str, Any]) -> dict[str, Any]:
    accepted = observed_rust_accepted(scope, case)
    value = {
        "accepted": accepted,
        "result_code": "ACCEPT" if accepted else case["observed"]["rust_error"],
        "first_reject_rule": None if accepted else rejection_rule(scope, case["mutation"]),
        "output_hash": hash_value(case["observed"].get("rust_output")) if accepted else None,
    }
    value["decision_hash"] = hash_value(value)
    return value


def event_contract(event: str) -> tuple[str, str, str]:
    entries = {
        "chunk_digest": (
            "derive",
            "NS-TRANSCRIPT-ORDER",
            "neo_fold_clean::paper::digest::f_prime_chunk_public_digest",
        ),
        "dispatch": (
            "shape-check",
            "SN-FOLD-TYPE",
            "neo_fold_clean::paper::construction2::verify_step_with_execution_receipt",
        ),
        "transcript_started": (
            "absorb",
            "NS-TRANSCRIPT-FRAMING",
            "neo_fold_clean::paper::f_prime::native::f_prime_step_transcript",
        ),
        "transcript_append": (
            "absorb",
            "NS-TRANSCRIPT-FRAMING",
            "neo_fold_clean::paper::f_prime::native::f_prime_step_transcript",
        ),
        "transcript_prefix": (
            "absorb",
            "NS-TRANSCRIPT-FRAMING",
            "neo_fold_clean::paper::f_prime::native::f_prime_step_transcript",
        ),
        "nifs_verify": (
            "relation-check",
            "SN-FOLD-PROOF",
            "neo_fold_clean::paper::nifs::verify",
        ),
        "running_digest": (
            "derive",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::RunningInstance::accumulator_digest",
        ),
        "state_advanced": (
            "derive",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::advance_state",
        ),
        "verifier_digest_read": (
            "derive",
            "NS-VERIFIER-KEY-DIGEST",
            "neo_fold_clean::paper::construction2::VerifierKey::digest",
        ),
        "pi_ccs_header_read": (
            "derive",
            "NS-PICCS-VARIANT",
            "neo_fold_clean::lifecycle::Preprocessing::pi_ccs_header_bundle",
        ),
        "state_x_out_hash": (
            "output",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::paper::construction2::compute_x_out",
        ),
        "terminal_public_link": (
            "relation-check",
            "NS-AUTH-DERIVED",
            "neo_fold_clean::lifecycle::validate_required_f_prime_latest_link",
        ),
        "terminal_running_relation": (
            "relation-check",
            "SN-REL-CE",
            "neo_fold_clean::lifecycle::validate_final_witness_authority",
        ),
        "terminal_fresh_relation": (
            "relation-check",
            "SN-REL-CCS",
            "neo_fold_clean::lifecycle::validate_latest_witness_authority",
        ),
    }
    require(event in entries, f"unmapped Rust event {event!r}")
    return entries[event]


def expected_trace(scope: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    input_hash = hash_value(case["rust_input"])
    result = []
    events = (
        case["observed"]["event_order"]
        if scope == "step"
        else ["terminal_public_link", "terminal_running_relation", "terminal_fresh_relation"]
    )
    for seq, event in enumerate(events):
        kind, rule, symbol = event_contract(event)
        result.append(
            {
                "seq": seq,
                "phase": event,
                "kind": kind,
                "contract_rule": rule,
                "source_symbol": symbol,
                "input_hash": input_hash,
                "output_hash": hash_value(
                    {"event": event, "observed": case["observed"], "seq": seq}
                ),
                "data": {"event_id": event},
            }
        )
    return result


def expected_mutation(scope: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "mutation_id": case["name"],
        "field": case["mutation"],
        "operation": "replace-with-adversarial-value",
        "input_payload_hash": hash_value(case["rust_input"]),
        "trace_hash": hash_value(case["observed"]),
        "rust_decision": expected_decision(scope, case),
    }


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} keys differ: {sorted(set(value) ^ expected)}")


def check_shape(evidence: dict[str, Any]) -> None:
    exact_keys(evidence, TOP_LEVEL_KEYS, "evidence")
    exact_keys(evidence["producer"], PRODUCER_KEYS, "producer")
    exact_keys(evidence["run"], RUN_KEYS, "run")
    exact_keys(evidence["run"]["attestation"], ATTESTATION_KEYS, "attestation")
    exact_keys(evidence["input"], INPUT_KEYS, "input")
    exact_keys(evidence["rust_decision"], DECISION_KEYS, "decision")
    exact_keys(evidence["semantic_target"], SEMANTIC_TARGET_KEYS, "semantic target")
    require(evidence["trace"], "evidence trace is empty")
    require(evidence["mutations"], "evidence mutations are empty")
    for index, event in enumerate(evidence["trace"]):
        exact_keys(event, TRACE_KEYS, f"trace[{index}]")
    for index, mutation in enumerate(evidence["mutations"]):
        exact_keys(mutation, MUTATION_KEYS, f"mutation[{index}]")
        exact_keys(mutation["rust_decision"], DECISION_KEYS, f"mutation[{index}].decision")


def check_hash_syntax(value: Any, path: str = "evidence") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("hash"):
                require(
                    (key == "output_hash" and child is None)
                    or (isinstance(child, str) and SHA256.fullmatch(child)),
                    f"{path}.{key} is not SHA-256",
                )
            check_hash_syntax(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_hash_syntax(child, f"{path}[{index}]")


def check(scope: str, root: Path, evidence_path: Path, corpus_path: Path, replay_path: Path) -> None:
    require(scope in {"step", "terminal"}, f"unknown Rust evidence scope {scope!r}")
    evidence = json.loads(evidence_path.read_text())
    corpus_text = corpus_path.read_text().rstrip("\n")
    corpus = json.loads(corpus_text)
    replay = replay_path.read_text()
    check_shape(evidence)
    check_hash_syntax(evidence)

    require(evidence["schema_version"] == 3, "unsupported Rust evidence schema")
    require(evidence["contract_id"] == "nightstream-superneo-v1", "wrong contract ID")
    require(evidence["origin"] == "rust-execution", "evidence is not Rust-origin")
    require(evidence["contract_rule"] == "NS-RUST-EVIDENCE-CONTENT", "wrong evidence rule")
    require(
        evidence["contract_hash"] == hash_file(root / "protocol-contract/superneo-v1.md"),
        "contract hash mismatch",
    )

    payload = evidence["input"]["payload"]
    require(evidence["input"]["encoding"] == "canonical-json-rfc8785", "wrong input encoding")
    require(evidence["input"]["payload_hash"] == hash_value(payload), "input payload hash mismatch")
    require(payload["corpus_json"] == corpus_text, "full Rust input corpus differs from envelope")
    require(payload["replay_program"] == replay, "fresh Lean replay differs from envelope")
    require(payload["profile_json"] == canonical_json(corpus["profile"]), "profile encoding mismatch")
    require(evidence["profile_id"] == corpus["profile"]["name"], "profile ID mismatch")
    require(evidence["profile_hash"] == hash_bytes(payload["profile_json"].encode()), "profile hash mismatch")
    expected_profile = {
        "step": "linked_bit_carrier_one_slot_stateless",
        "terminal": "linked_bit_carrier_one_slot_terminal",
    }[scope]
    require(evidence["profile_id"] == expected_profile, "wrong bounded profile")

    revision = command(root, "git", "rev-parse", "HEAD")
    dirty = bool(command(root, "git", "status", "--porcelain=v1", "--untracked-files=all"))
    require(evidence["rust_revision"] == revision, "Rust revision mismatch")
    require(evidence["source_tree_hash"] == source_tree_hash(root), "source-tree hash mismatch")
    producer = evidence["producer"]
    binary = Path(producer["binary"])
    require(binary.is_file(), "producer binary is absent")
    require(producer["binary_hash"] == hash_file(binary), "producer binary hash mismatch")
    require(producer["dirty"] == dirty, "dirty-state attestation mismatch")
    require(producer["cargo_lock_hash"] == hash_file(root / "Cargo.lock"), "Cargo.lock hash mismatch")
    rustc = command(root, "rustc", "-vV")
    host = next(line.removeprefix("host: ") for line in rustc.splitlines() if line.startswith("host: "))
    require(producer["rustc"] == rustc, "rustc identity mismatch")
    require(producer["target"] == host, "Rust target mismatch")
    require(producer["profile"] == "release", "Rust evidence was not produced in release mode")
    require(evidence["features"] == [], "unexpected Rust feature set")
    require(
        producer["command"]
        == [
            str(binary),
            "rust_origin_native_verifier_evidence_is_emitted_for_independent_checks",
            "--exact",
            "--nocapture",
        ],
        "producer command mismatch",
    )

    for case in corpus["cases"]:
        observed_accepted = observed_rust_accepted(scope, case)
        semantic_accepted = (
            semantic_step_accepted(case)
            if scope == "step"
            else semantic_terminal_accepted(case)
        )
        require(
            case["mapped"]["rust_accepted"] == observed_accepted,
            f"{case['name']} carried Rust Boolean differs from the observed result",
        )
        require(
            semantic_accepted == observed_accepted,
            f"{case['name']} Rust result differs from the independent one-slot checker",
        )
        if scope == "step" and observed_accepted:
            require(
                case["observed"]["rust_output"] == case["mapped"]["claim"],
                f"{case['name']} accepted Rust output differs from the canonical claim",
            )

    primary = next(case for case in corpus["cases"] if case["name"] == payload["primary_case"])
    expected_events = expected_trace(scope, primary)
    expected_decision_value = expected_decision(scope, primary)
    expected_mutations = [
        expected_mutation(scope, case)
        for case in corpus["cases"]
        if case["mutation"] != "none"
    ]
    require(evidence["trace"] == expected_events, "Rust event trace was not independently reproduced")
    require(evidence["rust_decision"] == expected_decision_value, "primary Rust decision mismatch")
    require(evidence["mutations"] == expected_mutations, "mutation decisions or traces mismatch")

    rule_index = json.loads((root / "protocol-contract/rule-index.json").read_text())
    rule_ids = {row["id"] for row in rule_index["rules"]}
    for event in evidence["trace"]:
        require(event["contract_rule"] in rule_ids, f"unknown event rule {event['contract_rule']}")
    for mutation in evidence["mutations"]:
        require(
            mutation["rust_decision"]["first_reject_rule"] in rule_ids,
            f"unknown mutation rule for {mutation['mutation_id']}",
        )

    semantic = evidence["semantic_target"]
    model = root / "formal/nightstream-lean/Nightstream/Implementation/Rust/CanonicalConformance/OneSlot.lean"
    model_id = {
        "step": "nightstream-lean-one-slot-step-v1",
        "terminal": "nightstream-lean-one-slot-terminal-v1",
    }[scope]
    predicate = {
        "step": "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.stepAgrees",
        "terminal": "Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.terminalAgrees",
    }[scope]
    require(semantic["model_id"] == model_id, "wrong semantic model ID")
    require(semantic["model_hash"] == hash_file(model), "semantic model hash mismatch")
    require(
        semantic["predicate"] == predicate,
        "wrong semantic predicate",
    )
    require(semantic["checker"] == replay_path.relative_to(root).as_posix(), "checker path mismatch")

    run = evidence["run"]
    require(run["exit_code"] == 0, "Rust producer did not report success")
    require(run["event_log_hash"] == hash_value(evidence["trace"]), "event-log hash mismatch")
    attested = {
        "input": evidence["input"],
        "mutations": evidence["mutations"],
        "rust_decision": evidence["rust_decision"],
        "trace": evidence["trace"],
    }
    attestation = run["attestation"]
    require(attestation["format"] == "nightstream-local-process-v1", "wrong attestation format")
    require(
        attestation["identity"] == f"{revision}:{producer['binary_hash']}",
        "attestation identity mismatch",
    )
    require(attestation["payload_hash"] == hash_value(attested), "attestation payload hash mismatch")
    invocation = hash_bytes(
        f"{revision}\n{evidence['source_tree_hash']}\n{producer['binary_hash']}\n{run['event_log_hash']}".encode()
    )
    require(run["invocation_id"] == invocation, "invocation ID mismatch")

    without_content_hash = dict(evidence)
    del without_content_hash["content_hash"]
    require(evidence["content_hash"] == hash_value(without_content_hash), "content hash mismatch")
    print(
        f"[rust-origin:{scope}] independently checked build identity, complete bounded corpus, "
        f"{len(evidence['trace'])} events, and {len(evidence['mutations'])} mutations"
    )


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("usage: check_rust_evidence.py SCOPE ROOT EVIDENCE CORPUS LEAN_REPLAY")
    check(sys.argv[1], *(Path(argument).resolve() for argument in sys.argv[2:]))


if __name__ == "__main__":
    main()
