#!/usr/bin/env python3
"""Independently check exact relation-artifact Rust-origin evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TOP_KEYS = {
    "schema_version",
    "contract_id",
    "contract_hash",
    "profile_id",
    "origin",
    "rust_revision",
    "source_tree_hash",
    "producer",
    "semantic_target",
    "authoritative_artifact_json",
    "cases",
    "content_hash",
}
PRODUCER_KEYS = {
    "crate",
    "binary",
    "binary_hash",
    "cargo_lock_hash",
    "command",
    "rustc",
}
TARGET_KEYS = {"model_id", "model_hash", "predicate", "checker", "replay_hash"}
CASE_KEYS = {
    "name",
    "mutation",
    "live_artifact_json",
    "candidate_artifact_json",
    "rust_accepted",
}
ARTIFACT_KEYS = {
    "format",
    "schema",
    "matrix_payload_encoding",
    "source",
    "params",
    "relation",
    "binding",
    "policy",
    "structure",
}
PARAM_KEYS = {
    "q",
    "eta",
    "ring_degree",
    "kappa",
    "row_domain_bound",
    "norm_base",
    "decomposition_exponent",
    "norm_bound",
    "expansion_factor",
    "extension_degree",
    "effective_lambda",
}
RELATION_KEYS = {
    "logical_rows",
    "assignment_fields",
    "padded_rows",
    "row_variables",
    "public_layout",
    "semantic_matrix_count",
    "joint_matrix_count",
    "polynomial_degree",
    "padded_identity",
    "padding_map",
}
BINDING_KEYS = {
    "structure_digest",
    "matrix_digest",
    "ajtai_public_parameters_digest",
    "verifier_key_digest",
}
POLICY_KEYS = {
    "stateful",
    "f_prime_recursive_link",
    "terminal_induction",
    "initial_semantic_state_digest",
}
STRUCTURE_KEYS = {"matrices", "f", "n", "m"}
EXPECTED_CASES = {
    "honest": "none",
    "logical_rows": "relation.logical_rows += 1",
    "binding_digest": "binding.structure_digest[0] += 1",
    "matrix_order": "structure.matrices[0..2] swapped",
    "source_kind": "source.kind replaced",
    "unknown_field": "unknown top-level field added",
    "noncanonical": "trailing newline added",
    "other_verifier_key": "live verifier key replaced",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == expected, f"{label} keys differ: {set(value) ^ expected}")
    return value


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=no_duplicate_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is invalid JSON: {error}") from error


def artifact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
        data = (root / relative).read_bytes()
        for frame in (relative.encode(), data):
            digest.update(len(frame).to_bytes(8, "big"))
            digest.update(frame)
    return digest.hexdigest()


def validate_live_artifact(text: str, label: str) -> dict[str, Any]:
    value = exact_keys(parse_json(text, label), ARTIFACT_KEYS, label)
    require(artifact_json(value) == text, f"{label} is not canonical Rust JSON")
    require(value["format"] == "nightstream/verifier-key-relation", f"{label} format differs")
    require(value["schema"] == 1, f"{label} schema differs")
    require(
        value["matrix_payload_encoding"] == "rust-ccs-structure-serde-json-v1",
        f"{label} matrix encoding differs",
    )
    source = exact_keys(value["source"], {"kind"}, f"{label}.source")
    require(source["kind"] == "verifier-owned-ccs", f"{label} source differs")
    exact_keys(value["params"], PARAM_KEYS, f"{label}.params")
    relation = exact_keys(value["relation"], RELATION_KEYS, f"{label}.relation")
    binding = exact_keys(value["binding"], BINDING_KEYS, f"{label}.binding")
    exact_keys(value["policy"], POLICY_KEYS, f"{label}.policy")
    structure = exact_keys(value["structure"], STRUCTURE_KEYS, f"{label}.structure")
    require(isinstance(structure["matrices"], list), f"{label} matrices must be a list")
    require(structure["matrices"], f"{label} matrix list must not be empty")
    require(relation["logical_rows"] == structure["n"], f"{label} row census differs")
    require(relation["assignment_fields"] == structure["m"], f"{label} column census differs")
    require(
        relation["semantic_matrix_count"] == len(structure["matrices"]),
        f"{label} matrix census differs",
    )
    require(
        relation["joint_matrix_count"] == relation["semantic_matrix_count"] + 1,
        f"{label} joint matrix census differs",
    )
    require(relation["padded_identity"] == "implicit-[I_m;0]", f"{label} identity rule differs")
    require(relation["padding_map"] == "logical-prefix-then-zero", f"{label} padding rule differs")
    public_layout = exact_keys(
        relation["public_layout"],
        {"assignment_layout", "kind", "start_field", "field_count"},
        f"{label}.relation.public_layout",
    )
    require(public_layout["assignment_layout"] == "z=x||w", f"{label} assignment layout differs")
    require(public_layout["kind"] == "x-is-assignment-prefix", f"{label} public layout differs")
    require(public_layout["start_field"] == 0, f"{label} public prefix does not start at zero")
    require(
        0 < public_layout["field_count"] <= relation["assignment_fields"],
        f"{label} public prefix width is invalid",
    )
    for key in ("structure_digest", "matrix_digest", "ajtai_public_parameters_digest"):
        require(
            isinstance(binding[key], list) and len(binding[key]) == 4,
            f"{label}.{key} must contain four field words",
        )
    require(
        isinstance(binding["verifier_key_digest"], list) and len(binding["verifier_key_digest"]) == 32,
        f"{label} verifier key digest must contain 32 bytes",
    )
    return value


def changed_copy(live: dict[str, Any], edit: Any) -> dict[str, Any]:
    expected = copy.deepcopy(live)
    edit(expected)
    return expected


def check_case(
    case: dict[str, Any],
    authoritative_text: str,
    authoritative: dict[str, Any],
) -> bool:
    name = case["name"]
    live_text = case["live_artifact_json"]
    candidate_text = case["candidate_artifact_json"]
    require(isinstance(live_text, str), f"{name} live artifact must be text")
    require(isinstance(candidate_text, str), f"{name} candidate artifact must be text")
    live = validate_live_artifact(live_text, f"case {name} live artifact")
    candidate = parse_json(candidate_text, f"case {name} candidate artifact")

    recomputed = candidate_text == live_text
    require(
        type(case["rust_accepted"]) is bool,
        f"{name} Rust decision must be Boolean",
    )
    require(
        case["rust_accepted"] == recomputed,
        f"{name} Rust decision differs from complete canonical artifact equality",
    )

    if name == "honest":
        require(live_text == authoritative_text, "honest live artifact differs from authority")
        require(candidate_text == authoritative_text, "honest candidate differs from authority")
    elif name == "logical_rows":
        expected = changed_copy(live, lambda value: value["relation"].__setitem__("logical_rows", 2))
        require(candidate == expected, "logical-row mutation changed another field")
    elif name == "binding_digest":
        expected = changed_copy(live, lambda value: value["binding"]["structure_digest"].__setitem__(0, 1))
        require(candidate == expected, "binding mutation changed another field")
    elif name == "matrix_order":
        expected = copy.deepcopy(live)
        expected["structure"]["matrices"][0], expected["structure"]["matrices"][1] = (
            expected["structure"]["matrices"][1],
            expected["structure"]["matrices"][0],
        )
        require(candidate == expected, "matrix-order mutation changed another field")
        require(candidate["structure"] != live["structure"], "matrix payload did not change")
    elif name == "source_kind":
        expected = changed_copy(live, lambda value: value["source"].__setitem__("kind", "unrecognized-source"))
        require(candidate == expected, "source mutation changed another field")
    elif name == "unknown_field":
        expected = changed_copy(live, lambda value: value.__setitem__("unrecognized", True))
        require(candidate == expected, "unknown-field mutation changed another field")
    elif name == "noncanonical":
        require(candidate == live, "noncanonical mutation changed decoded data")
        require(candidate_text == live_text + "\n", "noncanonical mutation is not the exact tested encoding")
    elif name == "other_verifier_key":
        require(candidate_text == authoritative_text, "other-key candidate differs from authority")
        require(live_text != authoritative_text, "other verifier key did not change its artifact")
        require(live["relation"] == authoritative["relation"], "other verifier key changed relation shape")
    else:
        raise ValueError(f"unknown evidence case {name!r}")
    return recomputed


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_relation_artifact_evidence.py REPO_ROOT EVIDENCE_JSON")
    root = Path(sys.argv[1]).resolve()
    evidence_path = Path(sys.argv[2]).resolve()
    evidence = exact_keys(
        parse_json(evidence_path.read_text(encoding="utf-8"), "evidence"),
        TOP_KEYS,
        "evidence",
    )

    content_hash = evidence.pop("content_hash")
    require(content_hash == hash_value(evidence), "evidence content hash differs")
    require(evidence["schema_version"] == 1, "evidence schema differs")
    require(evidence["contract_id"] == "nightstream-superneo-v1", "contract id differs")
    require(evidence["profile_id"] == "verifier-key-relation-artifact-v1", "profile id differs")
    require(evidence["origin"] == "rust-execution", "evidence origin differs")
    require(evidence["rust_revision"] == command(root, "git", "rev-parse", "HEAD"), "Rust revision differs")
    require(evidence["source_tree_hash"] == source_tree_hash(root), "source tree hash differs")
    require(
        evidence["contract_hash"] == hash_file(root / "protocol-contract/superneo-v1.md"),
        "contract hash differs",
    )

    producer = exact_keys(evidence["producer"], PRODUCER_KEYS, "producer")
    require(producer["crate"] == "neo-fold-clean", "producer crate differs")
    binary = Path(producer["binary"])
    require(binary.is_file(), "producer binary does not exist")
    require(producer["binary_hash"] == hash_file(binary), "producer binary hash differs")
    require(producer["cargo_lock_hash"] == hash_file(root / "Cargo.lock"), "Cargo.lock hash differs")
    require(producer["rustc"] == command(root, "rustc", "-vV"), "rustc identity differs")
    require(isinstance(producer["command"], list) and producer["command"], "producer command is empty")

    target = exact_keys(evidence["semantic_target"], TARGET_KEYS, "semantic target")
    model = root / "formal/nightstream-lean/Nightstream/Assurance/RelationArtifactBinding.lean"
    replay = root / target["checker"]
    require(target["model_id"] == "nightstream-relation-artifact-exact-binding-v1", "model id differs")
    require(target["model_hash"] == hash_file(model), "model hash differs")
    require(
        target["predicate"] == "Nightstream.Assurance.RelationArtifactBinding.ExactValidation",
        "model predicate differs",
    )
    require(replay.is_file(), "Lean replay does not exist")
    require(target["replay_hash"] == hash_file(replay), "Lean replay hash differs")

    authoritative_text = evidence["authoritative_artifact_json"]
    require(isinstance(authoritative_text, str), "authoritative artifact must be text")
    authoritative = validate_live_artifact(authoritative_text, "authoritative artifact")
    cases = evidence["cases"]
    require(isinstance(cases, list), "evidence cases must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for raw_case in cases:
        case = exact_keys(raw_case, CASE_KEYS, "case")
        name = case["name"]
        require(isinstance(name, str) and name not in by_name, f"duplicate case {name!r}")
        by_name[name] = case
    require(set(by_name) == set(EXPECTED_CASES), "evidence case census differs")

    accepted = 0
    for name, mutation in EXPECTED_CASES.items():
        case = by_name[name]
        require(case["mutation"] == mutation, f"{name} mutation label differs")
        accepted += int(check_case(case, authoritative_text, authoritative))
    require(accepted == 1, "exactly one relation artifact case must be accepted")
    print(f"[relation-artifact-evidence] checked {len(cases)} cases; exact binding passed")


if __name__ == "__main__":
    main()
