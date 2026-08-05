#!/usr/bin/env python3
"""Render deterministic views from the authored Nightstream contract model."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from contract_model import (
    CONTRACT_DIR,
    DECISION_ID,
    ERRATA_ID,
    EVIDENCE_TARGETS,
    MANIFEST_PATH,
    ContractModel,
    contract_path,
    coverage_rows,
    decision_impact,
    digest_bytes,
    expand_paper_references,
    is_ephemeral_contract_path,
    require,
    repository_path,
    unresolved_decision_ids,
)


def _render_coverage(model: ContractModel) -> bytes:
    fieldnames = [
        "contract_id", "source_ids", "decision_ids", "lean_model", "lean_assurance",
        "rust_implementation", "rust_conformance", "rust_origin_evidence",
        "rust_origin_assurance", "circuit_evidence", "circuit_assurance",
        "reduction_evidence", "reduction_assurance", "blocker_ids",
    ]
    lines = [",".join(fieldnames)]
    for row in coverage_rows(model):
        # Keep the historical review-friendly form: the stable ID is bare and
        # every list or evidence cell is quoted, including an empty cell.
        values = [row[fieldnames[0]]]
        values.extend(
            '"' + row[field].replace('"', '""') + '"'
            for field in fieldnames[1:]
        )
        lines.append(",".join(values))
    return ("\n".join(lines) + "\n").encode()


def _render_normative(model: ContractModel) -> bytes:
    lines = model.normative_text.splitlines()
    require(lines and lines[0].startswith("# "), "normative source has no title")
    replacements = [
        (rule_id, model.requirements[rule_id]["replaces"])
        for rule_id in model.rule_order
        if model.requirements[rule_id]["assembly"] == "replace"
    ]
    banner = [
        lines[0],
        "",
        "> Generated reading view. Edit `src/normative/` and",
        "> `src/requirements/`; `refresh_derived.py` rebuilds this file.",
    ]
    if replacements:
        banner.extend(["", "> Assembly overrides:"])
        for rule_id, targets in replacements:
            banner.append(
                "> - `" + rule_id + "` replaces "
                + ", ".join("`" + target + "`" for target in targets)
                + "."
            )
    banner.extend(["", *lines[1:]])
    return ("\n".join(banner).rstrip() + "\n").encode()


def _render_literal(model: ContractModel) -> bytes:
    lines = model.literal_text.splitlines()
    require(lines and lines[0].startswith("# "), "literal paper source has no title")
    banner = [
        lines[0],
        "",
        "> Generated reading view. Edit `src/paper/`;",
        "> `refresh_derived.py` rebuilds this file.",
        "",
        *lines[1:],
    ]
    return ("\n".join(banner).rstrip() + "\n").encode()


def _render_rule_index(
    model: ContractModel, coverage: bytes, normative_view: bytes
) -> bytes:
    coverage_map = {row["contract_id"]: row for row in csv.DictReader(io.StringIO(coverage.decode()))}
    entries = []
    for rule_id in model.rule_order:
        block = model.rules[rule_id]
        text = block["text"]
        requirement = model.requirements[rule_id]
        row = coverage_map[rule_id]
        paper_items = expand_paper_references(text)
        errata = sorted(set(ERRATA_ID.findall(text)))
        normative_decisions = sorted(set(DECISION_ID.findall(text)))
        states = {}
        evidence_locations = {}
        declaration_anchors = {}
        state_names = {"rust-origin": "rust_origin", "security": "reduction"}
        for target, (_, state_column) in EVIDENCE_TARGETS.items():
            key = state_names.get(target, target)
            states[key] = row[state_column]
            evidence_row = model.evidence[target][rule_id]
            evidence_locations[key] = evidence_row["locations"]
            declaration_anchors[key] = evidence_row["declaration_anchors"]
        entries.append({
            "id": rule_id,
            "title": block["title"],
            "section": block["section"],
            "kind": requirement["kind"],
            "assembly": requirement["assembly"],
            "replaces": requirement["replaces"],
            "depends_on": requirement["depends_on"],
            "normative_block_sha256": digest_bytes(block["bytes"]),
            "coverage_row_sha256": digest_bytes(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()),
            "authorities": {
                "reviewed_paper": bool(paper_items),
                "reviewed_errata": bool(errata),
                "nightstream_decision": bool(normative_decisions),
                "derived_conformance": requirement["derived_conformance"],
            },
            "source_ids": requirement["source_ids"],
            "paper_items": paper_items,
            "errata_ids": errata,
            "normative_decision_ids": normative_decisions,
            "coverage_decision_ids": requirement["decision_ids"],
            "blocker_ids": requirement["blocker_ids"],
            "review_flags": requirement["review_flags"],
            "states": states,
            "evidence_locations": evidence_locations,
            "declaration_anchors": declaration_anchors,
        })
    document = {
        "schema_version": 2,
        "contract_id": model.bundle["contract_id"],
        "generated_from": "src/bundle.json",
        "normative": {
            "path": "protocol-contract/superneo-v1.md",
            "sha256": digest_bytes(normative_view),
            "semantic_sha256": model.contract_hash,
        },
        "rules": entries,
    }
    lines = [
        "{",
        f'  "schema_version":{document["schema_version"]},',
        f'  "contract_id":{json.dumps(document["contract_id"])},',
        f'  "generated_from":{json.dumps(document["generated_from"])},',
        '  "normative":' + json.dumps(document["normative"], separators=(",", ":")) + ",",
        '  "rules":[',
    ]
    for index, entry in enumerate(entries):
        lines.append("    " + json.dumps(entry, separators=(",", ":")) + ("," if index + 1 < len(entries) else ""))
    lines.extend(["  ]", "}"])
    return ("\n".join(lines) + "\n").encode()


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _render_decisions(model: ContractModel) -> bytes:
    lines = [
        "# Nightstream decision and deviation ledger",
        "",
        "This file is generated from `src/decisions/decisions.jsonl` and the",
        "requirement graph. Do not edit this file.",
        "",
        "`Authorizes`, `directly blocks`, and `downstream impact` are separate",
        "derived relations. A decision does not author a paper rule only because",
        "that rule waits for the decision.",
        "",
        "| ID | Class | Decision | Reason | Authorizes | Directly blocks | Downstream impact | Owner | Selection | Integration |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for decision_id in model.decision_order:
        decision = model.decisions[decision_id]
        impact = decision_impact(model, decision_id)
        cells = [
            decision_id,
            decision["class"],
            decision["statement"],
            decision["reason"],
            ", ".join(impact["authorizes"]) or "—",
            ", ".join(impact["directly_blocks"]) or "—",
            ", ".join(impact["downstream_impact"]) or "—",
            decision["owner"],
            decision["selection_state"],
            decision["integration_state"],
        ]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    return ("\n".join(lines) + "\n").encode()


def _render_obligations(model: ContractModel) -> bytes:
    lines = [
        "# Generated from src/assurance/*.jsonl. Do not edit this view.",
        "schema_version = 2",
        f'contract_id = {json.dumps(model.bundle["contract_id"])}',
        f'root = {json.dumps(model.assurance_graph["root_claim"])}',
        "",
    ]
    for claim_id, claim in model.claims.items():
        state = model.claim_status[claim_id]
        lines.extend([
            "[[obligation]]",
            f'id = {json.dumps(claim_id)}',
            f'kind = {json.dumps(claim["kind"])}',
            f'applicability = {json.dumps(claim["applicability"])}',
            f'evidence_state = {json.dumps(claim["evidence_state"])}',
            f'dependency_state = {json.dumps(state["dependency_state"])}',
            f'blocker_state = {json.dumps(state["blocker_state"])}',
            f'closure_state = {json.dumps(state["closure_state"])}',
            f'display_status = {json.dumps(state["display_status"])}',
            f'freshness = {json.dumps(state["freshness"])}',
            f'depends_on = {_toml_array(claim["depends_on"])}',
            f'evidence = {_toml_array(claim["evidence"])}',
            f'blocker_ids = {_toml_array(claim["blocker_ids"])}',
            f'unresolved_blocker_ids = {_toml_array(state["unresolved_blocker_ids"])}',
            "",
        ])
    return ("\n".join(lines).rstrip() + "\n").encode()


def _render_release(model: ContractModel) -> bytes:
    release = model.release
    lines = [
        "# Generated from src/assurance/graph.json and leaf claim facts.",
        "schema_version = 2",
        f'contract_id = {json.dumps(model.bundle["contract_id"])}',
        f'root_claim = {json.dumps(release["root_claim"])}',
        f'eligible = {str(release["eligible"]).lower()}',
        f'implementation_ready = {str(release["implementation_ready"]).lower()}',
        f'implementation_ready_gate = {json.dumps(release["implementation_ready_gate"])}',
        f'production_claim_permitted = {str(release["production_claim_permitted"]).lower()}',
        f'highest_closed_gate = {json.dumps(release["highest_closed_gate"] or "none")}',
        f'next_gate = {json.dumps(release["next_gate"] or "none")}',
        "forbidden_claims = [",
    ]
    lines.extend(f"  {json.dumps(item)}," for item in model.assurance_graph["forbidden_claims"])
    lines.extend(["]", ""])
    for gate in model.gate_status:
        lines.extend([
            "[[gate]]",
            f'id = {json.dumps(gate["id"])}',
            f'closure_state = {json.dumps(gate["closure_state"])}',
            f'requires = {_toml_array(gate["requires"])}',
            "",
        ])
    return ("\n".join(lines).rstrip() + "\n").encode()


def _render_assurance_status(
    model: ContractModel, generated_bytes: dict[Path, bytes]
) -> bytes:
    claims = []
    for claim_id, claim in model.claims.items():
        claims.append({**claim, **model.claim_status[claim_id]})
    artifacts = []
    for source in model.artifacts.values():
        artifact = {
            key: value
            for key, value in source.items()
            if key != "migrated_from_path"
        }
        if artifact["availability"] == "packaged":
            path = repository_path(artifact["path"])
            data = generated_bytes[path] if path in generated_bytes else path.read_bytes()
            artifact["sha256"] = digest_bytes(data)
        artifacts.append(artifact)
    rollups = []
    for rollup_id, rollup in model.rollups.items():
        rollups.append(
            {
                **rollup,
                "scope": model.rollup_scopes[rollup_id],
                **model.rollup_status[rollup_id],
            }
        )
    document = {
        "schema_version": 2,
        "contract_id": model.bundle["contract_id"],
        "generated_from": ["src/assurance/graph.json", "src/assurance/*.jsonl"],
        "contract_hash": model.contract_hash,
        "profile_hash": model.profile_hash,
        "decisions": [
            {**model.decisions[item], **decision_impact(model, item)}
            for item in model.decision_order
        ],
        "issues": list(model.issues.values()),
        "reviews": list(model.reviews.values()),
        "artifacts": artifacts,
        "claims": claims,
        "rollups": rollups,
        "gates": model.gate_status,
        "release": model.release,
    }
    lines = [
        "{",
        f'  "schema_version":{document["schema_version"]},',
        f'  "contract_id":{json.dumps(document["contract_id"])},',
        '  "generated_from":' + json.dumps(document["generated_from"], separators=(",", ":")) + ",",
        f'  "contract_hash":{json.dumps(document["contract_hash"])},',
        f'  "profile_hash":{json.dumps(document["profile_hash"])},',
    ]
    list_names = ["decisions", "issues", "reviews", "artifacts", "claims", "rollups", "gates"]
    for name in list_names:
        lines.append(f'  "{name}":[')
        values = document[name]
        for index, value in enumerate(values):
            suffix = "," if index + 1 < len(values) else ""
            lines.append("    " + json.dumps(value, separators=(",", ":")) + suffix)
        lines.append("  ],")
    lines.append('  "release":' + json.dumps(document["release"], separators=(",", ":")))
    lines.append("}")
    return ("\n".join(lines) + "\n").encode()


def _render_protocol_events(model: ContractModel) -> bytes:
    all_blockers: set[str] = set()
    events = []
    for source in model.protocol["events"]:
        references = source["blocked_by"]
        blockers = unresolved_decision_ids(model.decisions, references)
        all_blockers.update(blockers)
        event = {
            "id": source["id"], "phase": source["phase"], "from_state": source["from_state"],
            "to_state": source["to_state"], "authority": source["authority"],
            "rule_ids": source["rule_ids"], "challenge_ids": source["challenge_ids"],
            "status": "profile-blocked" if blockers else "specified",
            "blocked_by": references,
            "unresolved_blocker_ids": blockers,
            "inputs": source["inputs"], "outputs": source["outputs"],
            "reject_conditions": source["reject_conditions"],
        }
        if "note" in source:
            event["note"] = source["note"]
        events.append(event)
    challenges = []
    for source in model.protocol["challenges"]:
        references = source["blocked_by"]
        blockers = unresolved_decision_ids(model.decisions, references)
        all_blockers.update(blockers)
        challenges.append({
            "id": source["id"], "role": source["role"], "sample_space": source["sample_space"],
            "sampled_in_event": source["sampled_in_event"],
            "transcript_tag": source["transcript_tag"],
            "decoder": source["decoder"],
            "after_events": source["after_events"],
            "status": "profile-blocked" if blockers else "specified",
            "blocked_by": references,
            "unresolved_blocker_ids": blockers,
            "rule_ids": source["rule_ids"],
        })
    blocked = bool(all_blockers)
    document = {
        "schema_version": 4,
        "contract_id": model.bundle["contract_id"],
        "contract_hash": model.contract_hash,
        "profile_id": model.bundle["profile_id"],
        "profile_hash": model.profile_hash,
        "status": "blocked" if blocked else "specified",
        "states": model.protocol["states"],
        "challenges": challenges,
        "events": events,
        "repetitions": model.protocol["repetitions"],
        "rejections": model.protocol["rejections"],
        "transcript_schedule": model.protocol["schedule"],
        "open_profile_dependencies": sorted(all_blockers),
    }
    return (json.dumps(document, indent=2) + "\n").encode()


def _render_sources(model: ContractModel) -> bytes:
    lock = model.source_lock
    registry = lock["registry"]
    semantic_state = model.gate_status[1]["closure_state"]
    status = (
        "mechanically locked; semantic normalization reviewed"
        if semantic_state == "closed"
        else "mechanically locked; semantic normalization open"
    )
    lines = [
        "# Contract source registry",
        "",
        f"Status: **{status}**.",
        "",
        "This file is generated from `src/sources/lock.toml`. Each source ID",
        "names one exact reviewed byte string. SHA-256 identifies artifacts",
        "only. It is not protocol authority.",
        "",
        f"Repository commit: `{registry['repository_commit']}`.",
        "",
        "| Source ID | Path | Reviewed SHA-256 | Base SHA-256 | Role |",
        "|---|---|---|---|---|",
    ]
    for source in lock["sources"]:
        base = source.get("base_sha256", "not-applicable")
        lines.append(
            f"| {source['id']} | `{source['path']}` | `{source['sha256']}` | "
            f"`{base}` | {source['role']} |"
        )
    derivation = lock["source_derivation"]
    lines.extend([
        "",
        "## Derivation rule",
        "",
        f"The checker reverse-applies `{derivation['patch_source_id']}` without fuzz",
        "or path substitution. It requires every reconstructed base hash. It then",
        "applies the patch forward and requires byte equality with the reviewed",
        "files. An unchanged file has equal base and reviewed hashes.",
        "",
        "The external v3 archive has SHA-256",
        f"`{registry['external_v3_archive_sha256']}`.",
        "It reconstructs the same base snapshot. It is not normative because this",
        "repository uses the later reviewed v4 patch and source bytes.",
        "",
        "Nightstream decisions are not paper sources. Their authority is in",
        "`src/decisions/decisions.jsonl`; `deviations.md` is a generated view.",
        "",
    ])
    return "\n".join(lines).encode()


def _render_profile(model: ContractModel) -> bytes:
    bundle = model.bundle
    release = model.release
    lines = [
        "# Generated from src/profile, src/sources, src/security, and assurance state.",
        "schema_version = 4",
        f'contract_id = {json.dumps(bundle["contract_id"])}',
        f'profile_id = {json.dumps(bundle["profile_id"])}',
        f'status = {json.dumps("release" if release["eligible"] else "candidate-draft")}',
        f'current_gate = {json.dumps(release["next_gate"] or "complete")}',
        f'profile_semantic_sha256 = {json.dumps(model.profile_hash)}',
        "",
        "[documents]",
    ]
    for key, value in bundle["documents"].items():
        lines.append(f"{key} = {json.dumps(value)}")
    lines.extend([
        "",
        "[review]",
        f'mechanical_source_lock = {json.dumps(model.gate_status[0]["closure_state"])}',
        f'literal_extraction = {json.dumps(model.gate_status[1]["closure_state"])}',
        f'candidate_decisions = {json.dumps(model.gate_status[2]["closure_state"])}',
        f'rust_provenance = {json.dumps(model.gate_status[4]["closure_state"])}',
        f'current_circuit = {json.dumps(model.gate_status[5]["closure_state"])}',
        f'end_to_end_security = {json.dumps(model.gate_status[6]["closure_state"])}',
        "",
    ])
    for relative in [bundle["authored"]["source_lock"], *bundle["authored"]["profile_inputs"], bundle["authored"]["security_planning"]]:
        lines.append(contract_path(relative).read_text().rstrip())
        lines.append("")
    decision_dependencies = sorted(
        {item for rule in model.requirements.values() for item in rule["blocker_ids"]}
    )
    open_blockers = unresolved_decision_ids(model.decisions, decision_dependencies)
    lines.extend([
        "[assurance]",
        f'allowed_tiers = {_toml_array([item for item in model.policy["evidence_levels"] if item != "none"])}',
        'workflow_states = ["open", "not-applicable"]',
        f'applicability_values = {_toml_array(model.policy["applicability_values"])}',
        "",
        "[conformance]",
        f'required_gate = {json.dumps(model.gate_status[-1]["id"])}',
        f'implementation_ready = {str(release["implementation_ready"]).lower()}',
        f'release_blocked = {str(not release["eligible"]).lower()}',
        f'decision_dependencies = {_toml_array(decision_dependencies)}',
        f'open_blockers = {_toml_array(open_blockers)}',
    ])
    return ("\n".join(lines).rstrip() + "\n").encode()


def _render_obligation_graph(model: ContractModel) -> bytes:
    lines = [
        "# Nightstream assurance claim graph",
        "",
        "This file is generated from `src/assurance/graph.json` and the leaf claim files.",
        "It does not own claim state.",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    for claim_id, claim in model.claims.items():
        label = f"{claim_id}\\n{model.claim_status[claim_id]['display_status']}"
        lines.append(f'  {claim_id.replace("-", "_")}["{label}"]')
        for dependency in claim["depends_on"]:
            lines.append(f'  {dependency.replace("-", "_")} --> {claim_id.replace("-", "_")}')
    lines.extend(["```", "", "## Release gates", "", "| Gate | Derived state | Required claims |", "|---|---|---|"])
    for gate in model.gate_status:
        lines.append(f"| `{gate['id']}` | {gate['closure_state']} | " + ", ".join(f"`{item}`" for item in gate["requires"]) + " |")
    lines.append("")
    return "\n".join(lines).encode()


def _render_requirement_graph(model: ContractModel) -> bytes:
    graph = {
        rule_id: model.requirements[rule_id]["depends_on"]
        for rule_id in model.rule_order
    }
    memo: dict[str, list[str]] = {}

    def longest(node: str) -> list[str]:
        if node in memo:
            return memo[node]
        dependencies = graph[node]
        prefix = max((longest(item) for item in dependencies), key=len, default=[])
        memo[node] = prefix + [node]
        return memo[node]

    critical = max((longest(item) for item in model.rule_order), key=len)
    lines = [
        "# Nightstream semantic requirement graph",
        "",
        "This file is generated from `src/requirements/*.jsonl`. The rule text",
        "stays in the ordered modules under `src/normative/`.",
        "",
        f"Requirements: **{len(graph)}**. Direct dependency edges: "
        f"**{sum(len(items) for items in graph.values())}**.",
        "",
        "Longest declared path:",
        "",
        "```text",
        " -> ".join(critical),
        "```",
        "",
        "## Assembly operations",
        "",
        "| Rule | Operation | Replaces |",
        "|---|---|---|",
    ]
    for rule_id in model.rule_order:
        item = model.requirements[rule_id]
        replacements = ", ".join(f"`{value}`" for value in item["replaces"]) or "—"
        lines.append(f"| `{rule_id}` | {item['assembly']} | {replacements} |")
    lines.extend([
        "",
        "## Direct dependencies and blockers",
        "",
        "| Rule | Kind | Depends on | Decision dependencies |",
        "|---|---|---|---|",
    ])
    for rule_id in model.rule_order:
        item = model.requirements[rule_id]
        dependencies = ", ".join(f"`{value}`" for value in item["depends_on"]) or "—"
        blockers = ", ".join(f"`{value}`" for value in item["blocker_ids"]) or "—"
        lines.append(f"| `{rule_id}` | {item['kind']} | {dependencies} | {blockers} |")
    lines.append("")
    return "\n".join(lines).encode()


def expected_outputs(model: ContractModel) -> dict[Path, bytes]:
    generated = model.bundle["generated"]
    coverage = _render_coverage(model)
    normative = _render_normative(model)
    outputs = {
        contract_path(generated["normative"]): normative,
        contract_path(generated["decisions"]): _render_decisions(model),
        contract_path(generated["literal_paper"]): _render_literal(model),
        contract_path(generated["sources"]): _render_sources(model),
        contract_path(generated["coverage"]): coverage,
        contract_path(generated["rule_index"]): _render_rule_index(model, coverage, normative),
        contract_path(generated["protocol_events"]): _render_protocol_events(model),
        contract_path(generated["profile"]): _render_profile(model),
        contract_path(generated["obligations"]): _render_obligations(model),
        contract_path(generated["release"]): _render_release(model),
        contract_path(generated["requirement_graph"]): _render_requirement_graph(model),
        CONTRACT_DIR / "obligation-graph.md": _render_obligation_graph(model),
    }
    outputs[contract_path(generated["assurance_status"])] = _render_assurance_status(
        model, outputs
    )
    return outputs


def expected_manifest(outputs: dict[Path, bytes]) -> bytes:
    rows = []
    for path in sorted(CONTRACT_DIR.rglob("*")):
        if (
            not path.is_file()
            or path == MANIFEST_PATH
            or is_ephemeral_contract_path(path)
        ):
            continue
        data = outputs.get(path, path.read_bytes())
        relative = path.relative_to(CONTRACT_DIR).as_posix()
        rows.append(f"{digest_bytes(data)}  ./{relative}\n")
    for path, data in outputs.items():
        if path.exists():
            continue
        relative = path.relative_to(CONTRACT_DIR).as_posix()
        rows.append(f"{digest_bytes(data)}  ./{relative}\n")
    rows.sort(key=lambda row: row.split("  ./", 1)[1])
    return "".join(rows).encode()


def refresh(model: ContractModel) -> list[str]:
    outputs = expected_outputs(model)
    changed = []
    for path, data in outputs.items():
        if not path.exists() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            changed.append(path.relative_to(CONTRACT_DIR).as_posix())
    manifest = expected_manifest(outputs)
    if not MANIFEST_PATH.exists() or MANIFEST_PATH.read_bytes() != manifest:
        MANIFEST_PATH.write_bytes(manifest)
        changed.append(MANIFEST_PATH.name)
    return changed


def check_generated(model: ContractModel) -> None:
    outputs = expected_outputs(model)
    stale = []
    for path, data in outputs.items():
        if not path.is_file() or path.read_bytes() != data:
            stale.append(path.relative_to(CONTRACT_DIR).as_posix())
    require(not stale, f"stale generated views: {stale}; run refresh_derived.py")
    require(MANIFEST_PATH.is_file(), "package manifest is missing")
    expected = expected_manifest(outputs)
    require(MANIFEST_PATH.read_bytes() == expected, "package manifest is stale")
    max_lines = model.bundle["limits"]["max_file_lines"]
    oversized = []
    for path in CONTRACT_DIR.rglob("*"):
        if path.is_file() and not is_ephemeral_contract_path(path):
            data = path.read_bytes()
            lines = data.count(b"\n") + (not data.endswith(b"\n"))
            if lines > max_lines:
                oversized.append(f"{path.relative_to(CONTRACT_DIR)} ({lines})")
    require(not oversized, f"contract files exceed {max_lines} lines: {oversized}")
