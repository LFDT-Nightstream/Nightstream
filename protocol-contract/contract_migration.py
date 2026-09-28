#!/usr/bin/env python3
"""Validate the sealed legacy-import record without freezing live semantics."""

from __future__ import annotations

import re
from typing import Any

from contract_model import _string_list, contract_path, digest_bytes, load_json, require

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and _SHA256.fullmatch(value),
        f"{label} is not a SHA-256 digest",
    )
    return value


def _unique_ids(items: Any, label: str) -> set[str]:
    require(isinstance(items, list), f"{label} is not a list")
    ids: list[str] = []
    for index, item in enumerate(items):
        require(isinstance(item, dict), f"{label}[{index}] is not an object")
        item_id = item.get("id")
        require(
            isinstance(item_id, str) and item_id,
            f"{label}[{index}] has no ID",
        )
        ids.append(item_id)
    require(len(ids) == len(set(ids)), f"{label} contains duplicate IDs")
    return set(ids)


def validate_migration_receipt(model: Any) -> None:
    """Validate the sealed receipt without freezing the live workflow state."""
    receipt_path = contract_path(model.bundle["authored"]["migration_receipt"])
    receipt = load_json(receipt_path)
    expected_fields = {
        "schema_version",
        "id",
        "contract_id",
        "baseline_path",
        "baseline_sha256",
        "verification_state",
        "review_owner",
        "verified_legacy_files",
        "verified_counts",
        "checks",
        "note",
    }
    require(set(receipt) == expected_fields, "unexpected migration-receipt fields")
    require(receipt["schema_version"] == 1, "unsupported migration-receipt schema")
    require(receipt["id"] == "MIG-LEGACY-V1-MODULAR", "unexpected migration receipt ID")
    require(receipt["contract_id"] == model.bundle["contract_id"], "migration receipt contract ID differs")
    require(receipt["verification_state"] == "verified", "lossless import is not verified")
    require(receipt["review_owner"] in model.policy["owner_roles"], "migration receipt has an invalid owner")
    require(isinstance(receipt["note"], str) and receipt["note"], "migration receipt has no limitation note")
    baseline_relative = model.bundle["authored"]["legacy_baseline"]
    require(receipt["baseline_path"] == baseline_relative, "migration receipt names the wrong baseline")
    baseline_path = contract_path(baseline_relative)
    _require_sha256(receipt["baseline_sha256"], "migration baseline digest")
    require(
        digest_bytes(baseline_path.read_bytes()) == receipt["baseline_sha256"],
        "migration baseline differs from the sealed receipt",
    )
    baseline = load_json(baseline_path)
    require(baseline.get("schema_version") == 1, "unsupported legacy baseline")
    require(baseline.get("contract_id") == model.bundle["contract_id"], "legacy baseline contract ID differs")
    require(receipt["verified_legacy_files"] == baseline["files"], "migration receipt file census differs")
    for name, digest in receipt["verified_legacy_files"].items():
        require(
            isinstance(name, str) and name and not name.startswith("/"),
            "invalid legacy file name",
        )
        _require_sha256(digest, f"legacy file digest {name}")
    expected_counts = {
        "normative_rules": len(baseline["normative_rules"]),
        "literal_paper_items": len(baseline["literal_paper_items"]),
        "decisions": len(baseline["decision_rows"]),
        "coverage_rows": len(baseline["coverage_rows"]),
        "assurance_claims": len(baseline["obligations"]),
        "assurance_rollups": len(baseline["legacy_assurance_ledger"]["claims"]),
        "assurance_artifacts": len(baseline["legacy_assurance_ledger"]["artifacts"]),
        "protocol_events": len(baseline["protocol_events"]),
        "protocol_challenges": len(baseline["protocol_challenges"]),
    }
    require(receipt["verified_counts"] == expected_counts, "migration receipt counts differ")
    expected_checks = {
        "normative-rule-blocks-byte-identical",
        "literal-paper-items-byte-identical",
        "decision-rows-lossless",
        "coverage-rows-byte-identical",
        "assurance-claims-lossless",
        "assurance-rollup-scopes-lossless",
        "protocol-events-lossless",
        "release-gates-lossless",
    }
    checks = _string_list(receipt["checks"], "migration receipt checks")
    require(set(checks) == expected_checks, "migration receipt check set differs")


def validate_lossless_import(model: Any) -> None:
    """Audit the sealed one-time import record, not the migrated live protocol."""
    validate_migration_receipt(model)
    baseline = load_json(contract_path(model.bundle["authored"]["legacy_baseline"]))
    expected_fields = {
        "schema_version",
        "contract_id",
        "purpose",
        "files",
        "obligations_sha256",
        "release_sha256",
        "protocol_states",
        "release",
        "legacy_assurance_ledger",
        "normative_rules",
        "literal_paper_items",
        "base_paper_items",
        "decision_rows",
        "coverage_rows",
        "obligations",
        "assurance_claims",
        "protocol_events",
        "protocol_challenges",
    }
    require(set(baseline) == expected_fields, "unexpected legacy-baseline fields")
    require(baseline.get("schema_version") == 1, "unsupported legacy baseline")
    require(
        isinstance(baseline["purpose"], str) and baseline["purpose"],
        "legacy baseline has no purpose",
    )

    _require_sha256(baseline["obligations_sha256"], "legacy obligations digest")
    _require_sha256(baseline["release_sha256"], "legacy release digest")
    require(
        baseline["files"].get("obligations.toml")
        == baseline["obligations_sha256"],
        "legacy obligations digest differs from the file census",
    )
    require(
        baseline["files"].get("release.toml") == baseline["release_sha256"],
        "legacy release digest differs from the file census",
    )

    for label in ("normative_rules", "literal_paper_items", "base_paper_items"):
        _unique_ids(baseline[label], f"legacy {label}")
        for item in baseline[label]:
            _require_sha256(
                item.get("sha256"), f"legacy {label} digest {item['id']}"
            )

    decision_ids: list[str] = []
    for index, row in enumerate(baseline["decision_rows"]):
        require(
            isinstance(row, list) and len(row) == 6,
            f"legacy decision row {index} has wrong width",
        )
        require(
            all(isinstance(value, str) for value in row),
            f"legacy decision row {index} is not textual",
        )
        require(row[0], f"legacy decision row {index} has no ID")
        decision_ids.append(row[0])
    require(
        len(decision_ids) == len(set(decision_ids)),
        "legacy decision rows contain duplicate IDs",
    )

    coverage_ids: list[str] = []
    for index, row in enumerate(baseline["coverage_rows"]):
        require(
            isinstance(row, dict) and row,
            f"legacy coverage row {index} is empty",
        )
        require(
            all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in row.items()
            ),
            f"legacy coverage row {index} is not a textual record",
        )
        rule_id = row.get("contract_id")
        require(
            isinstance(rule_id, str) and rule_id,
            f"legacy coverage row {index} has no rule ID",
        )
        coverage_ids.append(rule_id)
    require(
        len(coverage_ids) == len(set(coverage_ids)),
        "legacy coverage rows contain duplicate rule IDs",
    )

    obligation_ids = _unique_ids(baseline["obligations"], "legacy obligations")
    _unique_ids(baseline["protocol_events"], "legacy protocol events")
    _unique_ids(baseline["protocol_challenges"], "legacy protocol challenges")
    state_ids = _string_list(baseline["protocol_states"], "legacy protocol states")
    require(
        len(state_ids) == len(set(state_ids)),
        "legacy protocol states contain duplicates",
    )

    ledger = baseline["legacy_assurance_ledger"]
    require(isinstance(ledger, dict), "legacy assurance ledger is not an object")
    _unique_ids(ledger["claims"], "legacy assurance rollups")
    _unique_ids(ledger["artifacts"], "legacy assurance artifacts")
    require(
        _unique_ids(baseline["assurance_claims"], "legacy assurance claims")
        == {item["id"] for item in ledger["claims"]},
        "legacy assurance rollup IDs differ between recorded views",
    )

    release = baseline["release"]
    require(
        release.get("contract_id") == model.bundle["contract_id"],
        "legacy release contract ID differs",
    )
    require(
        release.get("root_obligation") in obligation_ids,
        "legacy release root is not an obligation",
    )
    gates = release.get("gate")
    require(isinstance(gates, list) and gates, "legacy release has no gates")
    gate_ids = [item.get("id") for item in gates if isinstance(item, dict)]
    require(
        len(gate_ids) == len(gates) and len(gate_ids) == len(set(gate_ids)),
        "legacy release gate IDs are invalid",
    )
