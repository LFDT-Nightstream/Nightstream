#!/usr/bin/env python3
"""Assurance claims, rollups, artifacts, and derived release state."""

from __future__ import annotations

import re
from typing import Any

from contract_model import (
    EVIDENCE_TARGETS,
    ContractModel,
    ModelError,
    _string_list,
    _unique,
    contract_path,
    decision_is_resolved,
    dependency_closure,
    find_cycle,
    framed_digest,
    is_packaged_path,
    load_json,
    load_jsonl,
    repository_path,
    require,
)


def _load_claims(
    bundle: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    graph = load_json(contract_path(bundle["authored"]["assurance"]["graph"]))
    claims: dict[str, dict[str, Any]] = {}
    for relative in graph["claim_files"]:
        for item in load_jsonl(contract_path(relative)):
            claim_id = item.get("id")
            require(isinstance(claim_id, str) and claim_id, f"claim without ID in {relative}")
            require(claim_id not in claims, f"duplicate assurance claim: {claim_id}")
            claims[claim_id] = item
    issues: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(contract_path(graph["issue_file"])):
        issue_id = item.get("id")
        require(isinstance(issue_id, str) and issue_id, "assurance issue without ID")
        require(issue_id not in issues, f"duplicate assurance issue: {issue_id}")
        issues[issue_id] = item
    artifacts: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(contract_path(graph["artifact_file"])):
        artifact_id = item.get("id")
        require(isinstance(artifact_id, str) and artifact_id, "assurance artifact without ID")
        require(artifact_id not in artifacts, f"duplicate assurance artifact: {artifact_id}")
        artifacts[artifact_id] = item
    rollups: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(contract_path(graph["rollup_file"])):
        rollup_id = item.get("id")
        require(isinstance(rollup_id, str) and rollup_id, "assurance rollup without ID")
        require(rollup_id not in rollups, f"duplicate assurance rollup: {rollup_id}")
        rollups[rollup_id] = item
    reviews: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(contract_path(graph["review_file"])):
        review_id = item.get("id")
        require(isinstance(review_id, str) and review_id, "assurance review without ID")
        require(review_id not in reviews, f"duplicate assurance review: {review_id}")
        reviews[review_id] = item
    return graph, claims, issues, artifacts, rollups, reviews


def claim_evidence_digest(
    claim: dict[str, Any], repository_mode: bool
) -> str | None:
    """Hash the exact evidence tree that a claim review examined."""
    files: list[tuple[str, bytes]] = []
    for relative in claim["evidence"]:
        if not repository_mode and not is_packaged_path(relative):
            return None
        path = repository_path(relative)
        if not path.exists():
            return None
        if path.is_file():
            files.append((relative, path.read_bytes()))
            continue
        if not path.is_dir():
            return None
        for child in sorted(path.rglob("*")):
            if child.is_file():
                name = f"{relative}/{child.relative_to(path).as_posix()}"
                files.append((name, child.read_bytes()))
    return framed_digest(files, b"nightstream-claim-evidence-v1\0")


def _derive_claim_status(
    claims: dict[str, dict[str, Any]],
    issues: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    contract_hash: str,
    profile_hash: str,
    repository_mode: bool,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    reviews_by_claim = {review["claim_id"]: review for review in reviews.values()}

    def evaluate(claim_id: str) -> dict[str, Any]:
        if claim_id in result:
            return result[claim_id]
        require(claim_id not in active, f"claim dependency cycle at {claim_id}")
        active.add(claim_id)
        claim = claims[claim_id]
        review = reviews_by_claim.get(claim_id)
        if claim["applicability"] == "not-applicable":
            freshness = "not-applicable"
        elif review is None:
            freshness = "missing"
        else:
            evidence_hash = claim_evidence_digest(claim, repository_mode)
            if evidence_hash is None:
                freshness = "unbound"
            elif (
                review["conclusion"] == "accepted"
                and review["contract_sha256"] == contract_hash
                and review["profile_sha256"] == profile_hash
                and review["evidence_sha256"] == evidence_hash
            ):
                freshness = "current"
            else:
                freshness = "stale"
        dependency_rows = [evaluate(item) for item in claim["depends_on"]]
        dependency_state = (
            "satisfied"
            if all(item["closure_state"] in {"closed", "not-applicable"} for item in dependency_rows)
            else "waiting"
        )
        unresolved = []
        for blocker in claim["blocker_ids"]:
            if blocker in issues and issues[blocker]["state"] != "resolved":
                unresolved.append(blocker)
            elif blocker in decisions:
                decision = decisions[blocker]
                if not decision_is_resolved(decision):
                    unresolved.append(blocker)
            elif blocker in claims and evaluate(blocker)["closure_state"] != "closed":
                unresolved.append(blocker)
        if claim["applicability"] == "not-applicable":
            closure = "not-applicable"
        elif (
            claim["evidence_state"] == "complete"
            and dependency_state == "satisfied"
            and not unresolved
            and freshness == "current"
        ):
            closure = "closed"
        else:
            closure = "open"
        if closure == "closed":
            display = "closed"
        elif closure == "not-applicable":
            display = "not-applicable"
        elif claim["evidence_state"] in {"partial", "complete"}:
            display = "partial"
        elif dependency_state == "waiting" or unresolved:
            display = "blocked"
        else:
            display = "open"
        row = {
            "dependency_state": dependency_state,
            "blocker_state": "blocked" if unresolved else "clear",
            "unresolved_blocker_ids": unresolved,
            "closure_state": closure,
            "display_status": display,
            "freshness": freshness,
        }
        result[claim_id] = row
        active.remove(claim_id)
        return row

    for claim_id in claims:
        evaluate(claim_id)
    return result


def _derive_release(
    graph: dict[str, Any], claim_status: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gates = []
    for gate in graph["gates"]:
        closed = all(claim_status[item]["closure_state"] == "closed" for item in gate["requires"])
        gates.append({"id": gate["id"], "requires": gate["requires"], "closure_state": "closed" if closed else "blocked"})
    contiguous = []
    for gate in gates:
        if gate["closure_state"] != "closed":
            break
        contiguous.append(gate["id"])
    next_gate = gates[len(contiguous)]["id"] if len(contiguous) < len(gates) else None
    root = graph["root_claim"]
    implementation_gate = graph["implementation_ready_gate"]
    implementation_ready = implementation_gate in contiguous
    eligible = claim_status[root]["closure_state"] == "closed" and len(contiguous) == len(gates)
    release = {
        "eligible": eligible,
        "root_claim": root,
        "implementation_ready": implementation_ready,
        "implementation_ready_gate": implementation_gate,
        "highest_closed_gate": contiguous[-1] if contiguous else None,
        "next_gate": next_gate,
        "production_claim_permitted": eligible,
    }
    return gates, release


def _expand_rollup_scopes(model: ContractModel) -> dict[str, dict[str, list[str]]]:
    source_ids = [item["id"] for item in model.source_lock["sources"]]
    result = {}
    for rollup_id, rollup in model.rollups.items():
        selector = rollup["scope_selector"]
        kind = selector["kind"]
        scope = {"source_ids": [], "decision_ids": [], "rule_ids": []}
        if kind == "all-sources":
            scope["source_ids"] = source_ids
        elif kind == "paper-rules-and-sources":
            scope["source_ids"] = source_ids
            scope["rule_ids"] = [item for item in model.rule_order if item.startswith("SN-")]
        elif kind == "decision-authority":
            scope["decision_ids"] = model.decision_order
            scope["rule_ids"] = [
                item for item in model.rule_order if model.requirements[item]["decision_ids"]
            ]
        elif kind == "evidence-target":
            target = selector["target"]
            scope["rule_ids"] = [
                item
                for item in model.rule_order
                if model.evidence[target][item]["applicability"] == "required"
            ]
        elif kind == "explicit":
            for key in scope:
                scope[key] = list(selector.get(key, []))
        elif kind == "all":
            scope["decision_ids"] = model.decision_order
            scope["rule_ids"] = model.rule_order
        else:
            raise ModelError(f"unknown rollup scope selector: {rollup_id}: {kind}")
        for key in scope:
            scope[key] = sorted(scope[key])
        result[rollup_id] = scope
    return result


def _derive_rollup_status(
    rollups: dict[str, dict[str, Any]], claim_status: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    active: set[str] = set()

    def evaluate(rollup_id: str) -> dict[str, Any]:
        if rollup_id in result:
            return result[rollup_id]
        require(rollup_id not in active, f"rollup dependency cycle at {rollup_id}")
        active.add(rollup_id)
        rollup = rollups[rollup_id]
        dependencies = [evaluate(item) for item in rollup["depends_on"]]
        dependency_state = (
            "satisfied"
            if all(item["closure_state"] == "closed" for item in dependencies)
            else "waiting"
        )
        leaves = [claim_status[item] for item in rollup["leaf_claim_ids"]]
        unresolved = [
            claim_id
            for claim_id in rollup["leaf_claim_ids"]
            if claim_status[claim_id]["closure_state"] not in {"closed", "not-applicable"}
        ]
        closure = "closed" if dependency_state == "satisfied" and not unresolved else "open"
        if closure == "closed":
            display = "closed"
        elif any(item["display_status"] == "partial" for item in leaves):
            display = "partial"
        else:
            display = "blocked"
        row = {
            "dependency_state": dependency_state,
            "closure_state": closure,
            "display_status": display,
            "unresolved_leaf_claim_ids": unresolved,
        }
        result[rollup_id] = row
        active.remove(rollup_id)
        return row

    for rollup_id in rollups:
        evaluate(rollup_id)
    return result


def validate_assurance(
    model: ContractModel,
    repository_mode: bool,
    rule_ids: set[str],
    decision_ids: set[str],
) -> tuple[set[str], set[str], set[str], set[str], list[str]]:
    policy = model.policy
    require(
        set(model.assurance_graph)
        == {
            "schema_version",
            "contract_id",
            "root_claim",
            "implementation_ready_gate",
            "claim_files",
            "issue_file",
            "review_file",
            "artifact_file",
            "rollup_file",
            "gate_order",
            "gates",
            "forbidden_claims",
        },
        "unexpected assurance-graph fields",
    )
    claim_ids = set(model.claims)
    issue_ids = set(model.issues)
    issue_labels = []
    issue_descriptions = []
    claim_graph: dict[str, list[str]] = {}
    decision_claim_coverage: list[str] = []
    decision_claim_ids: set[str] = set()
    for claim_id, claim in model.claims.items():
        require(
            set(claim)
            == {
                "id",
                "kind",
                "applicability",
                "evidence_state",
                "depends_on",
                "evidence",
                "blocker_ids",
            },
            f"unexpected claim fields: {claim_id}",
        )
        require(claim.get("kind") in policy["claim_kinds"], f"invalid claim kind: {claim_id}")
        require(claim.get("applicability") in policy["applicability_values"], f"invalid claim applicability: {claim_id}")
        require(claim.get("evidence_state") in policy["evidence_states"], f"invalid evidence state: {claim_id}")
        dependencies = _string_list(claim.get("depends_on"), f"claim dependencies for {claim_id}")
        blockers = _string_list(claim.get("blocker_ids"), f"claim blockers for {claim_id}")
        evidence = _string_list(claim.get("evidence"), f"claim evidence for {claim_id}")
        require(evidence, f"claim has no reviewable evidence set: {claim_id}")
        require(claim_id not in dependencies, f"self-dependent claim: {claim_id}")
        require(not (set(dependencies) - claim_ids), f"unknown claim dependencies: {claim_id}")
        known_blockers = decision_ids | issue_ids | claim_ids
        require(not (set(blockers) - known_blockers), f"unknown claim blockers: {claim_id}")
        if claim["kind"] == "decision":
            decision_claim_ids.add(claim_id)
            require(
                set(blockers) <= decision_ids,
                f"decision claim has a non-decision blocker: {claim_id}",
            )
            require(blockers, f"decision claim covers no decisions: {claim_id}")
            decision_claim_coverage.extend(blockers)
        for relative in evidence:
            path = repository_path(relative)
            if repository_mode or is_packaged_path(relative):
                require(path.exists(), f"missing claim evidence for {claim_id}: {relative}")
        claim_graph[claim_id] = dependencies
    cycle = find_cycle(claim_graph)
    require(cycle is None, f"claim dependency cycle: {' -> '.join(cycle or [])}")
    for claim_id, dependencies in claim_graph.items():
        for dependency in dependencies:
            reached: set[str] = set()
            pending = [item for item in dependencies if item != dependency]
            while pending:
                item = pending.pop()
                if item in reached:
                    continue
                reached.add(item)
                pending.extend(claim_graph[item])
            require(
                dependency not in reached,
                f"redundant transitive claim edge: {claim_id} -> {dependency}",
            )
    root = model.assurance_graph["root_claim"]
    require(root in claim_ids, "assurance root claim is unknown")
    require(dependency_closure(claim_graph, root) == claim_ids, "some claims are not reachable from the release root")
    _unique(decision_claim_coverage, "decision claim coverage")
    require(
        set(decision_claim_coverage) == decision_ids,
        "decision claims do not cover every canonical decision exactly once",
    )

    review_ids = set(model.reviews)
    reviewed_claims = []
    digest_pattern = r"[0-9a-f]{64}"
    for review_id, review in model.reviews.items():
        require(
            set(review)
            == {
                "id",
                "claim_id",
                "reviewer_role",
                "reviewer",
                "reviewed_at",
                "method",
                "conclusion",
                "contract_sha256",
                "profile_sha256",
                "evidence_sha256",
            },
            f"unexpected review fields: {review_id}",
        )
        require(review["claim_id"] in claim_ids, f"review has unknown claim: {review_id}")
        require(review["reviewer_role"] in policy["owner_roles"], f"review has invalid role: {review_id}")
        require(isinstance(review["reviewer"], str) and review["reviewer"], f"review has no reviewer: {review_id}")
        require(
            isinstance(review["reviewed_at"], str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", review["reviewed_at"]),
            f"review has invalid reviewed_at: {review_id}",
        )
        require(isinstance(review["method"], str) and review["method"], f"review has no method: {review_id}")
        require(review["conclusion"] in {"accepted", "rejected"}, f"review has invalid conclusion: {review_id}")
        for field in ("contract_sha256", "profile_sha256", "evidence_sha256"):
            require(
                isinstance(review[field], str) and re.fullmatch(digest_pattern, review[field]),
                f"review has invalid {field}: {review_id}",
            )
        reviewed_claims.append(review["claim_id"])
    _unique(reviewed_claims, "reviewed claim IDs")

    for issue_id, issue in model.issues.items():
        require(
            set(issue) == {"id", "description", "state", "owner", "legacy_label"},
            f"unexpected issue fields: {issue_id}",
        )
        require(issue.get("state") in policy["issue_states"], f"invalid issue state: {issue_id}")
        require(isinstance(issue.get("description"), str) and issue["description"], f"issue has no description: {issue_id}")
        require(issue.get("owner") in policy["owner_roles"], f"issue has an invalid owner role: {issue_id}")
        require(isinstance(issue.get("legacy_label"), str) and issue["legacy_label"], f"issue has no legacy label: {issue_id}")
        issue_labels.append(issue["legacy_label"])
        issue_descriptions.append(issue["description"].strip().casefold())
    _unique(issue_labels, "assurance issue legacy labels")
    _unique(issue_descriptions, "assurance issue descriptions")

    artifact_ids = set(model.artifacts)
    selected_artifact_paths = []
    for artifact_id, artifact in model.artifacts.items():
        allowed_artifact_fields = {
            "id",
            "kind",
            "path",
            "availability",
            "note",
            "migrated_from_path",
        }
        require(not (set(artifact) - allowed_artifact_fields), f"unexpected artifact fields: {artifact_id}")
        require(
            artifact.get("availability") in {"packaged", "repository-required", "unresolved"},
            f"invalid artifact availability: {artifact_id}",
        )
        require(artifact.get("kind") in policy["artifact_kinds"], f"artifact has an invalid kind: {artifact_id}")
        require("sha256" not in artifact, f"authored artifact contains a derived digest: {artifact_id}")
        if artifact["availability"] == "unresolved":
            require("path" not in artifact, f"unresolved artifact has a selected path: {artifact_id}")
        else:
            relative = artifact.get("path")
            require(isinstance(relative, str) and relative, f"artifact has no path: {artifact_id}")
            selected_artifact_paths.append(relative)
            path = repository_path(relative)
            if artifact["availability"] == "packaged" or repository_mode:
                require(path.exists(), f"missing assurance artifact: {artifact_id}: {relative}")
    _unique(selected_artifact_paths, "selected assurance artifact paths")

    rollup_ids = set(model.rollups)
    rollup_graph: dict[str, list[str]] = {}
    rollup_edges = []
    for rollup_id, rollup in model.rollups.items():
        require(
            set(rollup)
            == {
                "id",
                "edge",
                "scope_selector",
                "upstream",
                "downstream",
                "evidence",
                "depends_on",
                "leaf_claim_ids",
                "limitations",
            },
            f"unexpected rollup fields: {rollup_id}",
        )
        dependencies = _string_list(rollup.get("depends_on"), f"rollup dependencies for {rollup_id}")
        require(isinstance(rollup.get("edge"), str) and rollup["edge"], f"rollup has no edge: {rollup_id}")
        rollup_edges.append(rollup["edge"].strip().casefold())
        leaves = _string_list(rollup.get("leaf_claim_ids"), f"rollup leaves for {rollup_id}")
        upstream = _string_list(rollup.get("upstream"), f"rollup upstream for {rollup_id}")
        downstream = _string_list(rollup.get("downstream"), f"rollup downstream for {rollup_id}")
        require(not (set(dependencies) - rollup_ids), f"rollup has unknown dependencies: {rollup_id}")
        require(not (set(leaves) - claim_ids), f"rollup has unknown leaf claims: {rollup_id}")
        require(not (set(upstream + downstream) - artifact_ids), f"rollup has unknown artifacts: {rollup_id}")
        selector = rollup.get("scope_selector")
        require(isinstance(selector, dict), f"rollup has no scope selector: {rollup_id}")
        require(
            selector.get("kind") in policy["rollup_scope_selectors"],
            f"invalid rollup scope selector: {rollup_id}",
        )
        if selector["kind"] == "evidence-target":
            require(selector.get("target") in EVIDENCE_TARGETS, f"rollup has unknown evidence target: {rollup_id}")
        if selector["kind"] == "explicit":
            require(not (set(selector.get("rule_ids", [])) - rule_ids), f"rollup has unknown explicit rules: {rollup_id}")
            require(not (set(selector.get("decision_ids", [])) - decision_ids), f"rollup has unknown explicit decisions: {rollup_id}")
        evidence_rows = rollup.get("evidence")
        require(isinstance(evidence_rows, list) and evidence_rows, f"rollup has no evidence procedure: {rollup_id}")
        for evidence in evidence_rows:
            require(set(evidence) == {"kind", "artifact_id", "procedure"}, f"invalid rollup evidence fields: {rollup_id}")
            require(evidence["artifact_id"] in artifact_ids, f"rollup evidence has unknown artifact: {rollup_id}")
        limitations = _string_list(rollup.get("limitations"), f"rollup limitations for {rollup_id}")
        require(limitations, f"rollup has no limitation: {rollup_id}")
        rollup_graph[rollup_id] = dependencies
    _unique(rollup_edges, "assurance rollup edges")
    cycle = find_cycle(rollup_graph)
    require(cycle is None, f"assurance rollup cycle: {' -> '.join(cycle or [])}")

    gate_ids = [gate["id"] for gate in model.assurance_graph["gates"]]
    _unique(gate_ids, "gate IDs")
    require(gate_ids == model.assurance_graph["gate_order"], "gate order differs from gate declarations")
    require(
        model.assurance_graph["implementation_ready_gate"] in gate_ids,
        "implementation-ready gate is unknown",
    )
    for gate in model.assurance_graph["gates"]:
        require(not (set(gate["requires"]) - claim_ids), f"gate has unknown claims: {gate['id']}")
        if gate["id"] == "G1-DECISIONS":
            require(
                set(gate["requires"]) == decision_claim_ids,
                "G1 does not require every decision claim exactly once",
            )

    return claim_ids, issue_ids, artifact_ids, rollup_ids, review_ids, gate_ids


def validate_assurance_import(
    model: ContractModel, baseline: dict[str, Any]
) -> None:
    legacy_ledger = baseline["legacy_assurance_ledger"]
    require(
        legacy_ledger["required_edges"] == [item["edge"] for item in model.rollups.values()],
        "assurance edge import differs",
    )
    old_artifacts = {item["id"]: item for item in legacy_ledger["artifacts"]}
    require(set(old_artifacts) == set(model.artifacts), "assurance artifact ID import differs")
    for artifact_id, artifact in model.artifacts.items():
        old = old_artifacts[artifact_id]
        for key in ("kind", "availability"):
            require(artifact.get(key) == old.get(key), f"artifact {key} import differs: {artifact_id}")
        old_path = old.get("path")
        active_old_path = artifact.get("migrated_from_path", artifact.get("path"))
        require(active_old_path == old_path, f"artifact path import differs: {artifact_id}")
        require(artifact.get("note") == old.get("note"), f"artifact note import differs: {artifact_id}")

    old_rollups = {item["id"]: item for item in legacy_ledger["claims"]}
    require(set(old_rollups) == set(model.rollups), "assurance rollup ID import differs")
    for rollup_id, rollup in model.rollups.items():
        old = old_rollups[rollup_id]
        for key in ("edge", "upstream", "downstream", "depends_on", "limitations"):
            require(rollup[key] == old[key], f"rollup {key} import differs: {rollup_id}")
        old_evidence = [{key: value for key, value in item.items() if key != "status"} for item in old["evidence"]]
        require(rollup["evidence"] == old_evidence, f"rollup evidence import differs: {rollup_id}")
        require(model.rollup_scopes[rollup_id] == old["scope"], f"rollup scope import differs: {rollup_id}")
