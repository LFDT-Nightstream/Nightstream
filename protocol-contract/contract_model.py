#!/usr/bin/env python3
"""Load, validate, query, and render the Nightstream contract source model.

The files under ``protocol-contract/src`` are authored inputs. Large files at
the package root are generated compatibility or review views. This module has
no protocol authority. It only makes the declared data flow mechanical.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONTRACT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CONTRACT_DIR.parent
BUNDLE_PATH = CONTRACT_DIR / "src" / "bundle.json"
MANIFEST_PATH = CONTRACT_DIR / "MANIFEST.sha256"

RULE_HEADING = re.compile(r"^### ([A-Z][A-Z0-9-]+) — (.+)$", re.MULTILINE)
SECTION_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
PAPER_ITEM = re.compile(r"^`(PAPER-[A-Z0-9-]+)`", re.MULTILINE)
ERRATA_ID = re.compile(r"ERR-[A-Z0-9]+(?:-[A-Z0-9]+)*")
DECISION_ID = re.compile(r"NSD-[A-Z0-9]+(?:-[A-Z0-9]+)*")
NORMATIVE_KEYWORD = re.compile(r"\b(?:MUST(?: NOT)?|MAY)\b")

EVIDENCE_TARGETS = {
    "lean": ("lean_model", "lean_assurance"),
    "rust": ("rust_implementation", "rust_conformance"),
    "rust-origin": ("rust_origin_evidence", "rust_origin_assurance"),
    "circuit": ("circuit_evidence", "circuit_assurance"),
    "security": ("reduction_evidence", "reduction_assurance"),
}

TARGET_LEVELS = {
    "lean": {"none", "model-cited", "model-level"},
    "rust": {"none", "Rust-conformant"},
    "rust-origin": {"none", "artifact-checked", "Rust-conformant"},
    "circuit": {"none", "artifact-checked", "Rust-conformant"},
    "security": {"none", "model-cited", "model-level", "security-reduced"},
}


class ModelError(Exception):
    """A fail-closed contract-model error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelError(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def framed_digest(files: Iterable[tuple[str, bytes]], domain: bytes) -> str:
    """Hash an ordered file set with unambiguous path and length framing."""
    state = hashlib.sha256()
    state.update(domain)
    for name, data in files:
        encoded = name.encode("utf-8")
        state.update(len(encoded).to_bytes(8, "big"))
        state.update(encoded)
        state.update(len(data).to_bytes(8, "big"))
        state.update(data)
    return state.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_pairs)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except json.JSONDecodeError as error:
            raise ModelError(f"invalid JSONL at {path}:{number}: {error}") from error
        require(isinstance(value, dict), f"JSONL row is not an object: {path}:{number}")
        result.append(value)
    return result


def _parse_toml_value(value: str) -> Any:
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    return ast.literal_eval(value)


def load_toml(path: Path) -> dict[str, Any]:
    """Load the compact TOML subset used by this portable package."""
    result: dict[str, Any] = {}
    current = result
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[[") and line.endswith("]]" ):
            name = line[2:-2]
            target = result.setdefault(name, [])
            require(isinstance(target, list), f"TOML table type conflict: {path}:{name}")
            current = {}
            target.append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            current = result.setdefault(name, {})
            require(isinstance(current, dict), f"TOML table type conflict: {path}:{name}")
            continue
        require("=" in line, f"invalid TOML line in {path}: {line}")
        key, value = (part.strip() for part in line.split("=", 1))
        while value.count("[") > value.count("]"):
            require(index < len(lines), f"unterminated TOML array in {path}: {key}")
            value += "\n" + lines[index].strip()
            index += 1
        require(key not in current, f"duplicate TOML key in {path}: {key}")
        current[key] = _parse_toml_value(value)
    return result


def contract_path(relative: str) -> Path:
    require(isinstance(relative, str) and relative, "empty contract path")
    raw = Path(relative)
    require(not raw.is_absolute(), f"absolute contract path: {relative}")
    require(".." not in raw.parts, f"unsafe contract path: {relative}")
    resolved = (CONTRACT_DIR / raw).resolve(strict=False)
    root = CONTRACT_DIR.resolve()
    require(resolved == root or root in resolved.parents, f"contract path escapes package: {relative}")
    return CONTRACT_DIR / raw


def repository_path(relative: str) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    raw = Path(relative)
    require(not raw.is_absolute(), f"absolute repository path: {relative}")
    require(".." not in raw.parts, f"unsafe repository path: {relative}")
    resolved = (REPOSITORY_ROOT / raw).resolve(strict=False)
    root = REPOSITORY_ROOT.resolve()
    require(resolved == root or root in resolved.parents, f"repository path escapes root: {relative}")
    return REPOSITORY_ROOT / raw


def is_packaged_path(relative: str) -> bool:
    parts = Path(relative).parts
    return bool(parts) and parts[0] == "protocol-contract"


def is_ephemeral_contract_path(path: Path) -> bool:
    """Return true only for local cache files that have no package authority."""
    try:
        relative = path.relative_to(CONTRACT_DIR)
    except ValueError:
        relative = path
    cache_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    if any(part in cache_parts for part in relative.parts):
        return True
    return path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store"


def _unique(values: list[str], label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    require(not duplicates, f"duplicate {label}: {duplicates}")


def _string_list(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} is not a list")
    require(all(isinstance(item, str) and item for item in value), f"{label} has a non-string item")
    result = list(value)
    _unique(result, label)
    return result


def find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if state.get(node) == 1:
            start = stack.index(node)
            return stack[start:] + [node]
        if state.get(node) == 2:
            return None
        state[node] = 1
        stack.append(node)
        for dependency in graph[node]:
            cycle = visit(dependency)
            if cycle:
                return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def dependency_closure(graph: dict[str, list[str]], root: str) -> set[str]:
    result: set[str] = set()
    pending = [root]
    while pending:
        item = pending.pop()
        if item in result:
            continue
        result.add(item)
        pending.extend(graph[item])
    return result


def requirement_dependents(model: "ContractModel", roots: Iterable[str]) -> list[str]:
    """Return roots and all requirements that depend on them, in contract order."""
    reverse: dict[str, set[str]] = {rule_id: set() for rule_id in model.rule_order}
    for rule_id, requirement in model.requirements.items():
        for dependency in requirement["depends_on"]:
            reverse[dependency].add(rule_id)
    reached: set[str] = set()
    pending = list(roots)
    while pending:
        rule_id = pending.pop()
        if rule_id in reached:
            continue
        reached.add(rule_id)
        pending.extend(reverse[rule_id])
    return [rule_id for rule_id in model.rule_order if rule_id in reached]


def decision_impact(model: "ContractModel", decision_id: str) -> dict[str, list[str]]:
    """Derive decision authority, direct blockers, and downstream impact."""
    authorizes = [
        rule_id
        for rule_id in model.rule_order
        if decision_id in model.requirements[rule_id]["decision_ids"]
    ]
    directly_blocks = [
        rule_id
        for rule_id in model.rule_order
        if decision_id in model.requirements[rule_id]["blocker_ids"]
    ]
    downstream = requirement_dependents(model, authorizes + directly_blocks)
    return {
        "authorizes": authorizes,
        "directly_blocks": directly_blocks,
        "downstream_impact": downstream,
    }


def rule_blocks(text: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    matches = list(RULE_HEADING.finditer(text))
    sections = list(SECTION_HEADING.finditer(text))
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = "unsectioned"
        for candidate in sections:
            if candidate.start() < match.start():
                section = candidate.group(1)
            else:
                break
        block = (text[match.start():end].rstrip() + "\n").encode()
        rule_id = match.group(1)
        require(rule_id not in result, f"duplicate normative rule: {rule_id}")
        order.append(rule_id)
        result[rule_id] = {
            "title": match.group(2),
            "section": section,
            "bytes": block,
            "text": block.decode(),
        }
    require(order, "normative contract has no rule blocks")
    return order, result


def expand_paper_references(block: str) -> list[str]:
    result = set(re.findall(r"PAPER-[A-Z0-9-]+-\d{3}", block))
    pattern = re.compile(r"(PAPER-([A-Z0-9-]+)-)(\d{3})\s+through\s+PAPER-\2-(\d{3})")
    for match in pattern.finditer(block):
        first = int(match.group(3))
        last = int(match.group(4))
        require(first <= last, f"reversed paper range: {match.group(0)}")
        result.update(f"{match.group(1)}{number:03d}" for number in range(first, last + 1))
    return sorted(result)


def literal_paper_source_map(text: str) -> dict[str, set[str]]:
    """Map each literal paper item to the source files cited by its section."""
    sections = [
        match
        for match in SECTION_HEADING.finditer(text)
        if match.group(1).startswith("PAPER-")
    ]
    result: dict[str, set[str]] = {}
    for index, section in enumerate(sections):
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        block = text[section.start():end]
        sources = set(re.findall(r"SRC-PAPER(?:-[A-Z0-9]+)+", block))
        require(sources, f"literal paper section has no source: {section.group(1)}")
        for paper_id in PAPER_ITEM.findall(block):
            require(paper_id not in result, f"duplicate literal paper item: {paper_id}")
            result[paper_id] = sources
    require(result, "literal paper model has no indexed items")
    return result


def parse_decisions(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    normalized_text: dict[str, str] = {}
    expected = {
        "id",
        "class",
        "statement",
        "reason",
        "owner",
        "selection_state",
        "integration_state",
    }
    for item in load_jsonl(path):
        require(set(item) == expected, f"unexpected decision fields: {item.get('id')}")
        decision_id = item["id"]
        require(decision_id not in result, f"duplicate decision: {decision_id}")
        require(
            isinstance(decision_id, str) and DECISION_ID.fullmatch(decision_id),
            f"invalid decision ID: {decision_id}",
        )
        key = re.sub(r"\s+", " ", item["statement"]).strip().casefold()
        require(key not in normalized_text, f"duplicate decision text: {decision_id} and {normalized_text.get(key)}")
        normalized_text[key] = decision_id
        order.append(decision_id)
        result[decision_id] = item
    require(order, "decision ledger has no decision rows")
    return order, result


def decision_is_resolved(decision: dict[str, Any]) -> bool:
    """Return true only when selection and integration are both complete."""
    return (
        decision["selection_state"] in {"approved", "not-applicable"}
        and decision["integration_state"] in {"complete", "not-applicable"}
    )


def unresolved_decision_ids(
    decisions: dict[str, dict[str, Any]], decision_ids: Iterable[str]
) -> list[str]:
    """Derive active blockers while keeping stable decision references authored."""
    return sorted(
        decision_id
        for decision_id in decision_ids
        if not decision_is_resolved(decisions[decision_id])
    )


def protocol_open_decisions(model: "ContractModel") -> list[str]:
    references = {
        decision_id
        for item in model.protocol["events"] + model.protocol["challenges"]
        for decision_id in item["blocked_by"]
    }
    return unresolved_decision_ids(model.decisions, references)


@dataclass
class ContractModel:
    bundle: dict[str, Any]
    normative_text: str
    literal_text: str
    rule_order: list[str]
    rules: dict[str, dict[str, Any]]
    requirements: dict[str, dict[str, Any]]
    evidence: dict[str, dict[str, dict[str, Any]]]
    decision_order: list[str]
    decisions: dict[str, dict[str, Any]]
    claims: dict[str, dict[str, Any]]
    issues: dict[str, dict[str, Any]]
    artifacts: dict[str, dict[str, Any]]
    rollups: dict[str, dict[str, Any]]
    reviews: dict[str, dict[str, Any]]
    assurance_graph: dict[str, Any]
    claim_status: dict[str, dict[str, Any]]
    gate_status: list[dict[str, Any]]
    release: dict[str, Any]
    rollup_status: dict[str, dict[str, Any]]
    rollup_scopes: dict[str, dict[str, list[str]]]
    protocol: dict[str, Any]
    source_lock: dict[str, Any]
    policy: dict[str, Any]
    contract_hash: str
    profile_hash: str


def _assemble(paths: list[str], label: str) -> str:
    require(paths, f"{label} has no modules")
    chunks = []
    for relative in paths:
        path = contract_path(relative)
        require(path.is_file(), f"missing {label} module: {relative}")
        chunks.append(path.read_text())
    return "".join(chunks)


def _load_requirements(bundle: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for relative in bundle["authored"]["requirement_files"]:
        for item in load_jsonl(contract_path(relative)):
            rule_id = item.get("id")
            require(isinstance(rule_id, str) and rule_id, f"requirement without ID in {relative}")
            require(rule_id not in result, f"duplicate requirement metadata: {rule_id}")
            order.append(rule_id)
            result[rule_id] = item
    return order, result


def _load_evidence(bundle: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for target, relative in bundle["authored"]["evidence_files"].items():
        require(target in EVIDENCE_TARGETS, f"unknown evidence target: {target}")
        rows: dict[str, dict[str, Any]] = {}
        for item in load_jsonl(contract_path(relative)):
            rule_id = item.get("rule_id")
            require(isinstance(rule_id, str) and rule_id, f"evidence row without rule ID in {relative}")
            require(rule_id not in rows, f"duplicate {target} evidence row: {rule_id}")
            rows[rule_id] = item
        result[target] = rows
    require(set(result) == set(EVIDENCE_TARGETS), "evidence target set is incomplete")
    return result


def _load_protocol(bundle: dict[str, Any]) -> dict[str, Any]:
    paths = bundle["authored"]["protocol"]
    return {
        "machine": load_json(contract_path(paths["machine"])),
        "states": load_jsonl(contract_path(paths["states"])),
        "events": load_jsonl(contract_path(paths["events"])),
        "challenges": load_jsonl(contract_path(paths["challenges"])),
        "repetitions": load_jsonl(contract_path(paths["repetitions"])),
        "rejections": load_jsonl(contract_path(paths["rejections"])),
        "schedule": load_json(contract_path(paths["transcript_schedule"])),
    }


def load_model(repository_mode: bool = False) -> ContractModel:
    from contract_assurance import (
        _derive_claim_status,
        _derive_release,
        _derive_rollup_status,
        _expand_rollup_scopes,
        _load_claims,
    )
    from contract_migration import validate_migration_receipt
    bundle = load_json(BUNDLE_PATH)
    require(bundle.get("schema_version") == 1, "unsupported source-bundle schema")
    normative = _assemble(bundle["authored"]["normative_modules"], "normative")
    literal = _assemble(bundle["authored"]["literal_paper_modules"], "literal-paper")
    rule_order, rules = rule_blocks(normative)
    requirement_order, requirements = _load_requirements(bundle)
    require(requirement_order == rule_order, "requirement metadata order differs from normative rule order")
    evidence = _load_evidence(bundle)
    decision_order, decisions = parse_decisions(contract_path(bundle["authored"]["decision_file"]))
    assurance_graph, claims, issues, artifacts, rollups, reviews = _load_claims(bundle)
    protocol = _load_protocol(bundle)
    source_lock = load_toml(contract_path(bundle["authored"]["source_lock"]))
    policy = load_toml(contract_path(bundle["authored"]["assurance"]["policy"]))
    profile_files = [
        (relative, contract_path(relative).read_bytes())
        for relative in bundle["authored"].get(
            "profile_hash_inputs", bundle["authored"]["profile_inputs"]
        )
    ]
    profile_hash = framed_digest(profile_files, b"nightstream-profile-v1\0")
    contract_files = [
        (relative, contract_path(relative).read_bytes())
        for relative in (
            bundle["authored"]["normative_modules"]
            + bundle["authored"]["requirement_files"]
        )
    ]
    contract_hash = framed_digest(contract_files, b"nightstream-contract-v1\0")

    model = ContractModel(
        bundle=bundle,
        normative_text=normative,
        literal_text=literal,
        rule_order=rule_order,
        rules=rules,
        requirements=requirements,
        evidence=evidence,
        decision_order=decision_order,
        decisions=decisions,
        claims=claims,
        issues=issues,
        artifacts=artifacts,
        rollups=rollups,
        reviews=reviews,
        assurance_graph=assurance_graph,
        claim_status={},
        gate_status=[],
        release={},
        rollup_status={},
        rollup_scopes={},
        protocol=protocol,
        source_lock=source_lock,
        policy=policy,
        contract_hash=contract_hash,
        profile_hash=profile_hash,
    )
    validate_model(model, repository_mode=repository_mode)
    model.claim_status = _derive_claim_status(
        claims,
        issues,
        decisions,
        reviews,
        model.contract_hash,
        profile_hash,
        repository_mode,
    )
    model.gate_status, model.release = _derive_release(assurance_graph, model.claim_status)
    model.rollup_scopes = _expand_rollup_scopes(model)
    model.rollup_status = _derive_rollup_status(rollups, model.claim_status)
    validate_migration_receipt(model)
    return model


def validate_model(model: ContractModel, repository_mode: bool) -> None:
    rule_ids = set(model.rule_order)
    decision_ids = set(model.decision_order)
    policy = model.policy
    migration = model.bundle.get("migration", {})
    vocabulary_names = (
        "applicability_values",
        "evidence_states",
        "dependency_states",
        "assurance_levels",
        "evidence_levels",
        "issue_states",
        "decision_states",
        "integration_states",
        "requirement_kinds",
        "claim_kinds",
        "artifact_kinds",
        "decision_classes",
        "owner_roles",
        "protocol_authorities",
        "rollup_scope_selectors",
        "review_flags",
    )
    for vocabulary_name in vocabulary_names:
        _string_list(policy.get(vocabulary_name), f"policy vocabulary {vocabulary_name}")
    require(model.bundle.get("contract_id") == model.assurance_graph.get("contract_id"), "assurance graph contract ID differs")
    require(model.assurance_graph.get("schema_version") == 2, "unsupported assurance-graph schema")
    authored = model.bundle.get("authored", {})
    profile_inputs = _string_list(
        authored.get("profile_inputs"), "profile input paths"
    )
    profile_hash_inputs = _string_list(
        authored.get("profile_hash_inputs"), "profile hash input paths"
    )
    _unique(profile_inputs, "profile input paths")
    _unique(profile_hash_inputs, "profile hash input paths")
    protocol_inputs = authored.get("protocol")
    require(isinstance(protocol_inputs, dict), "protocol input map is not an object")
    required_profile_hash_inputs = (
        set(profile_inputs)
        | set(protocol_inputs.values())
        | {authored.get("security_planning")}
    )
    require(
        None not in required_profile_hash_inputs
        and required_profile_hash_inputs <= set(profile_hash_inputs),
        "profile hash omits a semantic profile input",
    )
    require(
        migration.get("lossless_import") in {"pending", "verified"},
        "invalid lossless-import state",
    )
    require(
        migration.get("requirement_granularity") in {"coarse-groups", "atomic"},
        "invalid requirement granularity",
    )
    require(
        migration.get("semantic_dependency_review") in {"pending-g0b", "reviewed"},
        "invalid semantic dependency review state",
    )

    registry = model.source_lock.get("registry", {})
    require(
        set(registry) == {"repository_commit", "external_v3_archive_sha256"},
        "unexpected source-registry metadata",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", registry["repository_commit"]) is not None, "invalid source repository commit")
    require(re.fullmatch(r"[0-9a-f]{64}", registry["external_v3_archive_sha256"]) is not None, "invalid external archive digest")
    sources = model.source_lock.get("sources", [])
    require(isinstance(sources, list) and sources, "source lock has no sources")
    source_ids = [item.get("id") for item in sources]
    source_paths = [item.get("path") for item in sources]
    _unique(source_ids, "source IDs")
    _unique(source_paths, "source paths")
    for source in sources:
        required = {"id", "path", "sha256", "role"}
        allowed = required | {"base_sha256"}
        require(required <= set(source) and not (set(source) - allowed), f"invalid source fields: {source.get('id')}")
        require(isinstance(source["role"], str) and source["role"], f"source has no role: {source['id']}")
        require(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, f"invalid source digest: {source['id']}")
        if "base_sha256" in source:
            require(re.fullmatch(r"[0-9a-f]{64}", source["base_sha256"]) is not None, f"invalid base source digest: {source['id']}")

    known_source_ids = set(source_ids)
    paper_source_map = literal_paper_source_map(model.literal_text)
    for paper_id, paper_sources in paper_source_map.items():
        require(
            paper_sources <= known_source_ids,
            f"literal paper item has unknown source IDs: {paper_id}",
        )

    require(set(model.requirements) == rule_ids, "requirement metadata and rule IDs differ")
    normalized_blocks: dict[str, str] = {}
    normalized_clauses: dict[str, str] = {}
    requirement_graph: dict[str, list[str]] = {}
    for rule_id in model.rule_order:
        item = model.requirements[rule_id]
        require(
            set(item)
            == {
                "id",
                "kind",
                "assembly",
                "replaces",
                "depends_on",
                "blocker_ids",
                "review_flags",
                "derived_conformance",
                "source_ids",
                "decision_ids",
            },
            f"unexpected requirement metadata fields: {rule_id}",
        )
        for field in ("kind", "assembly", "replaces", "depends_on", "blocker_ids", "review_flags", "source_ids", "decision_ids"):
            require(field in item, f"{rule_id} requirement metadata omits {field}")
        require(item["kind"] in policy["requirement_kinds"], f"invalid requirement kind: {rule_id}")
        require(item["assembly"] in {"adopt", "add", "replace", "remove"}, f"invalid assembly mode: {rule_id}")
        dependencies = _string_list(item["depends_on"], f"dependencies for {rule_id}")
        replacements = _string_list(item["replaces"], f"replacements for {rule_id}")
        blockers = _string_list(item["blocker_ids"], f"blockers for {rule_id}")
        review_flags = _string_list(item["review_flags"], f"review flags for {rule_id}")
        require(
            not (set(review_flags) - set(policy["review_flags"])),
            f"unknown review flags for {rule_id}",
        )
        sources = _string_list(item["source_ids"], f"sources for {rule_id}")
        authorities = _string_list(item["decision_ids"], f"decisions for {rule_id}")
        require(isinstance(item["derived_conformance"], bool), f"derived-conformance flag is not Boolean: {rule_id}")
        unknown_sources = set(sources) - known_source_ids - {"not-in-paper"}
        require(not unknown_sources, f"{rule_id} has unknown source IDs: {sorted(unknown_sources)}")
        require(
            not ("not-in-paper" in sources and len(sources) != 1),
            f"{rule_id} mixes not-in-paper with reviewed sources",
        )
        require(not (set(dependencies) - rule_ids), f"{rule_id} has unknown semantic dependencies")
        require(rule_id not in dependencies, f"self-dependent requirement: {rule_id}")
        require(not (set(replacements) - rule_ids), f"{rule_id} replaces an unknown rule")
        if item["assembly"] == "replace":
            require(replacements, f"replacement rule has no target: {rule_id}")
            require(set(replacements) <= set(dependencies), f"replacement target is not a dependency: {rule_id}")
        else:
            require(not replacements, f"non-replacement rule names replacement targets: {rule_id}")
        require(not (set(authorities) - decision_ids), f"{rule_id} has unknown decision authority")
        require(not (set(blockers) - decision_ids), f"{rule_id} has unknown decision blockers")
        if item["assembly"] == "adopt":
            require(sources and sources != ["not-in-paper"], f"adopted rule has no reviewed source: {rule_id}")
            require(rule_id.startswith("SN-"), f"adopted rule does not use an SN ID: {rule_id}")
        else:
            require(authorities, f"Nightstream rule has no decision authority: {rule_id}")
            require(rule_id.startswith("NS-"), f"Nightstream rule does not use an NS ID: {rule_id}")
        rule_text = model.rules[rule_id]["text"]
        rule_lines = len(rule_text.splitlines())
        require(
            rule_lines <= model.bundle["limits"]["max_rule_lines"],
            f"normative rule exceeds the line limit: {rule_id} ({rule_lines})",
        )
        keyword_count = len(NORMATIVE_KEYWORD.findall(rule_text))
        require(
            keyword_count <= model.bundle["limits"]["max_normative_keywords_per_rule"],
            f"normative rule has invalid atomicity count: {rule_id} ({keyword_count})",
        )
        if item["assembly"] == "adopt":
            require(rule_text.count("Source:") == 1, f"paper rule must have one Source citation: {rule_id}")
            require("Decision:" not in rule_text, f"paper rule mixes decision authority: {rule_id}")
        else:
            require(rule_text.count("Decision:") == 1, f"Nightstream rule must have one Decision citation: {rule_id}")
            require("Source:" not in rule_text, f"Nightstream rule mixes paper authority: {rule_id}")
        paper_references = expand_paper_references(rule_text)
        require(
            not (set(paper_references) - set(paper_source_map)),
            f"{rule_id} cites an unknown literal paper item",
        )
        if item["assembly"] == "adopt":
            require(paper_references, f"adopted rule has no literal paper-item citation: {rule_id}")
        required_paper_sources = {
            source_id
            for paper_id in paper_references
            for source_id in paper_source_map[paper_id]
        }
        require(
            required_paper_sources <= set(sources),
            f"{rule_id} omits a source file used by its literal paper citations",
        )
        normative_decisions = set(DECISION_ID.findall(rule_text))
        require(
            set(authorities) <= normative_decisions,
            f"{rule_id} has decision authority that its normative text does not name",
        )
        require(
            normative_decisions <= set(authorities) | set(blockers),
            f"{rule_id} has an unclassified normative decision reference: "
            f"{sorted(normative_decisions - set(authorities) - set(blockers))}",
        )
        for clause in re.split(r"(?<=[.!?])\s+", rule_text):
            if not NORMATIVE_KEYWORD.search(clause):
                continue
            clause_key = re.sub(r"\s+", " ", clause).strip().casefold()
            require(
                clause_key not in normalized_clauses,
                f"duplicate normative clause: {rule_id} and {normalized_clauses.get(clause_key)}",
            )
            normalized_clauses[clause_key] = rule_id
        block_key = re.sub(r"\s+", " ", rule_text).strip().casefold()
        require(block_key not in normalized_blocks, f"duplicate normative rule text: {rule_id} and {normalized_blocks.get(block_key)}")
        normalized_blocks[block_key] = rule_id
        requirement_graph[rule_id] = dependencies

    cycle = find_cycle(requirement_graph)
    require(cycle is None, f"requirement dependency cycle: {' -> '.join(cycle or [])}")
    for rule_id, dependencies in requirement_graph.items():
        for dependency in dependencies:
            reached: set[str] = set()
            pending = [item for item in dependencies if item != dependency]
            while pending:
                item = pending.pop()
                if item in reached:
                    continue
                reached.add(item)
                pending.extend(requirement_graph[item])
            require(
                dependency not in reached,
                f"redundant transitive requirement edge: {rule_id} -> {dependency}",
            )

    for decision_id, decision in model.decisions.items():
        require(isinstance(decision["owner"], str) and decision["owner"], f"decision has no owner: {decision_id}")
        require(decision["class"] in policy["decision_classes"], f"invalid decision class: {decision_id}")
        require(decision["owner"] in policy["owner_roles"], f"invalid decision owner role: {decision_id}")
        require(decision["selection_state"] in policy["decision_states"], f"invalid decision state: {decision_id}")
        require(decision["integration_state"] in policy["integration_states"], f"invalid integration state: {decision_id}")
        if decision["selection_state"] in {"open", "rejected"}:
            require(decision["integration_state"] != "complete", f"unselected decision is integrated: {decision_id}")
        if decision["selection_state"] == "not-applicable":
            require(
                decision["integration_state"] == "not-applicable",
                f"not-applicable decision has an integration state: {decision_id}",
            )
        if decision["integration_state"] == "not-applicable":
            require(
                decision["selection_state"] == "not-applicable",
                f"not-applicable integration has a selected decision: {decision_id}",
            )

    for target, rows in model.evidence.items():
        require(set(rows) == rule_ids, f"{target} evidence rule set differs")
        for rule_id, row in rows.items():
            require(set(row) == {
                "rule_id",
                "applicability",
                "level",
                "locations",
                "declaration_anchors",
            }, f"unexpected {target} evidence fields: {rule_id}")
            require(row.get("applicability") in policy["applicability_values"], f"invalid {target} applicability: {rule_id}")
            require(row.get("level") in TARGET_LEVELS[target], f"invalid {target} evidence level: {rule_id}")
            locations = _string_list(row.get("locations"), f"{target} locations for {rule_id}")
            anchors = row.get("declaration_anchors")
            require(isinstance(anchors, list), f"{target} declaration anchors are not a list: {rule_id}")
            normalized_anchors = []
            for anchor in anchors:
                require(
                    isinstance(anchor, dict) and set(anchor) == {"path", "declaration"},
                    f"invalid {target} declaration anchor: {rule_id}",
                )
                path_value = anchor["path"]
                declaration = anchor["declaration"]
                require(
                    isinstance(path_value, str) and path_value,
                    f"empty {target} declaration path: {rule_id}",
                )
                require(
                    isinstance(declaration, str) and declaration and not declaration[0].isdigit(),
                    f"invalid {target} declaration name: {rule_id}",
                )
                normalized_anchors.append((path_value, declaration))
            _unique(normalized_anchors, f"{target} declaration anchors for {rule_id}")
            if target not in {"lean", "rust"}:
                require(not anchors, f"{target} evidence cannot name source declarations: {rule_id}")
            if row["applicability"] == "not-applicable":
                require(row["level"] == "none", f"not-applicable {target} edge has evidence level: {rule_id}")
            elif row["level"] != "none":
                require(locations or anchors, f"{target} evidence level has no location: {rule_id}")
            if target == "lean" and row["level"] in {"model-cited", "model-level"}:
                require(anchors, f"{rule_id} claims {row['level']} without an exact Lean declaration")
            for relative in locations + [anchor["path"] for anchor in anchors]:
                path = repository_path(relative)
                if repository_mode or is_packaged_path(relative):
                    require(path.exists(), f"missing {target} evidence location for {rule_id}: {relative}")

    from contract_assurance import validate_assurance

    claim_ids, issue_ids, artifact_ids, rollup_ids, review_ids, gate_ids = validate_assurance(
        model, repository_mode, rule_ids, decision_ids
    )

    from contract_protocol import validate_protocol

    validate_protocol(model)
    global_groups = {
        "decision": decision_ids,
        "requirement": rule_ids,
        "claim": claim_ids,
        "issue": issue_ids,
        "gate": set(gate_ids),
        "assurance-rollup": rollup_ids,
        "assurance-review": review_ids,
        "artifact": artifact_ids,
        "protocol-state": {item["id"] for item in model.protocol["states"]},
        "protocol-event": {item["id"] for item in model.protocol["events"]},
        "challenge": {item["id"] for item in model.protocol["challenges"]},
        "repetition": {item["id"] for item in model.protocol["repetitions"]},
    }
    owners: dict[str, list[str]] = {}
    for group, values in global_groups.items():
        for value in values:
            owners.setdefault(value, []).append(group)
    collisions = {key: value for key, value in owners.items() if len(value) > 1}
    require(not collisions, f"global model ID collisions: {collisions}")

    max_lines = model.bundle["limits"]["max_authored_lines"]
    for relative in _authored_paths(model.bundle):
        path = contract_path(relative)
        require(path.is_file(), f"missing authored input: {relative}")
        lines = path.read_bytes().count(b"\n") + (not path.read_bytes().endswith(b"\n"))
        require(lines <= max_lines, f"authored file exceeds {max_lines} lines: {relative} ({lines})")


def _authored_paths(bundle: dict[str, Any]) -> list[str]:
    authored = bundle["authored"]
    result = []
    result.extend(authored["normative_modules"])
    result.extend(authored["literal_paper_modules"])
    result.extend(authored["requirement_files"])
    result.extend(authored["evidence_files"].values())
    result.extend(
        [
            authored["decision_file"],
            authored["legacy_baseline"],
            authored["migration_receipt"],
        ]
    )
    result.extend(authored["protocol"].values())
    result.extend([authored["source_lock"], authored["security_planning"]])
    result.extend(authored["profile_inputs"])
    result.extend(authored.get("profile_hash_inputs", []))
    result.extend(authored["assurance"].values())
    assurance_graph = load_json(contract_path(authored["assurance"]["graph"]))
    result.extend(assurance_graph["claim_files"])
    result.extend(
        [
            assurance_graph["issue_file"],
            assurance_graph["artifact_file"],
            assurance_graph["rollup_file"],
            assurance_graph["review_file"],
        ]
    )
    return sorted(set(result))


def coverage_rows(model: ContractModel) -> list[dict[str, str]]:
    rows = []
    for rule_id in model.rule_order:
        requirement = model.requirements[rule_id]
        row: dict[str, str] = {
            "contract_id": rule_id,
            "source_ids": ";".join(requirement["source_ids"]),
            "decision_ids": ";".join(requirement["decision_ids"]),
        }
        for target, (owner_column, state_column) in EVIDENCE_TARGETS.items():
            evidence = model.evidence[target][rule_id]
            anchors = [
                f"{anchor['path']}:{anchor['declaration']}"
                for anchor in evidence["declaration_anchors"]
            ]
            row[owner_column] = ";".join(anchors or evidence["locations"])
            if evidence["applicability"] == "not-applicable":
                row[state_column] = "not-applicable"
            elif evidence["level"] == "none":
                row[state_column] = "open"
            else:
                row[state_column] = evidence["level"]
        row["blocker_ids"] = ";".join(requirement["blocker_ids"])
        rows.append(row)
    return rows


def expected_outputs(model: ContractModel) -> dict[Path, bytes]:
    from contract_render import expected_outputs as render_outputs

    return render_outputs(model)


def refresh(model: ContractModel) -> list[str]:
    from contract_render import refresh as render_refresh

    return render_refresh(model)


def check_generated(model: ContractModel) -> None:
    from contract_render import check_generated as render_check

    render_check(model)


def query(model: ContractModel, identifier: str) -> dict[str, Any] | None:
    if identifier in model.requirements:
        unresolved = unresolved_decision_ids(
            model.decisions, model.requirements[identifier]["blocker_ids"]
        )
        return {
            "type": "requirement",
            "metadata": model.requirements[identifier],
            "title": model.rules[identifier]["title"],
            "evidence": {target: rows[identifier] for target, rows in model.evidence.items()},
            "direct_dependents": [
                rule_id
                for rule_id in model.rule_order
                if identifier in model.requirements[rule_id]["depends_on"]
            ],
            "downstream_dependents": [
                rule_id
                for rule_id in requirement_dependents(model, [identifier])
                if rule_id != identifier
            ],
            "unresolved_blocker_ids": unresolved,
        }
    if identifier in model.decisions:
        return {
            "type": "decision",
            **model.decisions[identifier],
            **decision_impact(model, identifier),
            "resolved": decision_is_resolved(model.decisions[identifier]),
        }
    if identifier in model.claims:
        return {"type": "claim", **model.claims[identifier], **model.claim_status[identifier]}
    if identifier in model.issues:
        return {"type": "issue", **model.issues[identifier]}
    if identifier in model.artifacts:
        return {"type": "artifact", **model.artifacts[identifier]}
    if identifier in model.rollups:
        return {
            "type": "assurance-rollup",
            **model.rollups[identifier],
            "scope": model.rollup_scopes[identifier],
            **model.rollup_status[identifier],
        }
    for gate in model.gate_status:
        if gate["id"] == identifier:
            return {"type": "gate", **gate}
    for collection, type_name in (
        (model.protocol["events"], "event"),
        (model.protocol["challenges"], "challenge"),
    ):
        for item in collection:
            if item["id"] == identifier:
                unresolved = unresolved_decision_ids(
                    model.decisions, item["blocked_by"]
                )
                return {
                    "type": type_name,
                    **item,
                    "status": "profile-blocked" if unresolved else "specified",
                    "unresolved_blocker_ids": unresolved,
                }
    for collection, type_name in (
        (model.protocol["states"], "state"),
        (model.protocol["repetitions"], "repetition"),
    ):
        for item in collection:
            if item["id"] == identifier:
                return {"type": type_name, **item}
    return None


def summary(model: ContractModel) -> dict[str, Any]:
    return {
        "requirements": len(model.requirements),
        "requirement_edges": sum(len(item["depends_on"]) for item in model.requirements.values()),
        "decisions": len(model.decisions),
        "claims": len(model.claims),
        "claim_edges": sum(len(item["depends_on"]) for item in model.claims.values()),
        "issues": len(model.issues),
        "artifacts": len(model.artifacts),
        "rollups": len(model.rollups),
        "reviews": len(model.reviews),
        "events": len(model.protocol["events"]),
        "challenges": len(model.protocol["challenges"]),
        "release_eligible": model.release["eligible"],
        "implementation_ready": model.release["implementation_ready"],
        "highest_closed_gate": model.release["highest_closed_gate"],
        "next_gate": model.release["next_gate"],
        "profile_hash": model.profile_hash,
    }
