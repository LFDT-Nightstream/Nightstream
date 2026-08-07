#!/usr/bin/env python3
"""Detailed source, package, binding, and arithmetic contract checks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from contract_model import is_ephemeral_contract_path, is_packaged_path
from contract_primitives import FieldDuplex, poseidon2_permute


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "protocol-contract"
CONFIG_PATH = CONTRACT_DIR / "superneo-v1.toml"
CONTRACT_PATH = CONTRACT_DIR / "superneo-v1.md"
COVERAGE_PATH = CONTRACT_DIR / "coverage.csv"
SOURCES_PATH = CONTRACT_DIR / "sources.md"
PACKAGE_MANIFEST_PATH = CONTRACT_DIR / "MANIFEST.sha256"

LEAN_KEYWORDS = ("theorem", "lemma", "def", "abbrev", "structure", "inductive", "class", "instance")
RUST_KEYWORDS = ("fn", "struct", "enum", "trait", "type", "const", "static", "union", "macro_rules!")

RULE_HEADING = re.compile(r"^### ([A-Z][A-Z0-9-]+) — ", re.MULTILINE)
ERRATA_ROW = re.compile(r"^\| (ERR-[A-Z0-9-]+) \|", re.MULTILINE)
SOURCE_ROW = re.compile(
    r"^\| (SRC-[A-Z0-9-]+) \| `([^`]+)` \| `([0-9a-f]{64})` \|",
    re.MULTILINE,
)
PAPER_ITEM = re.compile(r"^`(PAPER-[A-Z0-9-]+)`", re.MULTILINE)
BASE_ITEM = re.compile(r"^`(BASE-[A-Z0-9-]+)`", re.MULTILINE)
PATCH_HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


class ContractError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def split_ids(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def evidence_path(value: str) -> str:
    """Return the path part of a rendered path:declaration anchor."""
    for suffix in (".lean:", ".rs:"):
        if suffix in value:
            return value.split(suffix, 1)[0] + suffix[:-1]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_path(relative: str) -> Path:
    require(isinstance(relative, str) and relative, "empty repository path")
    raw = Path(relative)
    require(not raw.is_absolute(), f"absolute repository path: {relative}")
    require(".." not in raw.parts, f"unsafe repository path: {relative}")
    root = ROOT.resolve()
    resolved = (ROOT / raw).resolve(strict=False)
    require(
        resolved == root or root in resolved.parents,
        f"repository path escapes the root: {relative}",
    )
    return ROOT / raw


def parse_toml_value(value: str):
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    return ast.literal_eval(value)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Parse the small TOML subset used by this contract on Python 3.9."""
    config: dict = {}
    current = config
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("[[") and line.endswith("]]"):
            name = line[2:-2]
            target = config.setdefault(name, [])
            require(isinstance(target, list), f"TOML table type conflict: {name}")
            current = {}
            target.append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            current = config.setdefault(name, {})
            require(isinstance(current, dict), f"TOML table type conflict: {name}")
            continue
        require("=" in line, f"invalid TOML line: {line}")
        key, value = (part.strip() for part in line.split("=", 1))
        while value.count("[") > value.count("]"):
            require(index < len(lines), f"unterminated TOML array: {key}")
            value += "\n" + lines[index].strip()
            index += 1
        require(key not in current, f"duplicate TOML key in one table: {key}")
        current[key] = parse_toml_value(value)
    return config


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    hunks: list[tuple[int, int, int, int, list[str]]]


def parse_unified_patch(text: str) -> list[PatchFile]:
    lines = text.splitlines(keepends=True)
    files: list[PatchFile] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("diff --git "):
            index += 1
            continue
        index += 1
        while index < len(lines) and not lines[index].startswith("--- "):
            index += 1
        require(
            index + 1 < len(lines) and lines[index + 1].startswith("+++ "),
            "errata patch has a diff without old and new paths",
        )
        old_path = lines[index][4:].strip()
        new_path = lines[index + 1][4:].strip()
        index += 2
        hunks: list[tuple[int, int, int, int, list[str]]] = []
        while index < len(lines) and not lines[index].startswith("diff --git "):
            if not lines[index].startswith("@@ "):
                index += 1
                continue
            match = PATCH_HUNK.match(lines[index])
            require(match is not None, f"malformed patch hunk: {lines[index].rstrip()}")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            index += 1
            body: list[str] = []
            while index < len(lines):
                line = lines[index]
                if line.startswith("@@ ") or line.startswith("diff --git "):
                    break
                require(
                    line.startswith((" ", "+", "-", "\\")),
                    f"unexpected patch body line: {line.rstrip()}",
                )
                body.append(line)
                index += 1
            hunks.append((old_start, old_count, new_start, new_count, body))
        require(hunks, f"patch file has no hunks: {old_path}")
        files.append(PatchFile(old_path, new_path, hunks))
    require(files, "errata patch has no file diffs")
    return files


def apply_exact_patch_file(source_bytes: bytes, patch: PatchFile, reverse: bool) -> bytes:
    source = source_bytes.decode("utf-8").splitlines(keepends=True)
    output: list[str] = []
    source_index = 0
    target_index = 0

    for old_start, old_count, new_start, new_count, body in patch.hunks:
        if reverse:
            source_start, source_count = new_start, new_count
            target_start, target_count = old_start, old_count
        else:
            source_start, source_count = old_start, old_count
            target_start, target_count = new_start, new_count

        hunk_source_index = source_start - 1
        hunk_target_index = target_start - 1
        require(hunk_source_index >= source_index, f"overlapping hunks in {patch.old_path}")
        output.extend(source[source_index:hunk_source_index])
        target_index += hunk_source_index - source_index
        require(
            target_index == hunk_target_index,
            f"target hunk offset mismatch in {patch.new_path}",
        )

        cursor = hunk_source_index
        consumed = 0
        produced = 0
        for raw in body:
            if raw.startswith("\\"):
                continue
            prefix, content = raw[0], raw[1:]
            consumes = prefix == " " or (prefix == "+" if reverse else prefix == "-")
            produces = prefix == " " or (prefix == "-" if reverse else prefix == "+")
            if consumes:
                actual = source[cursor] if cursor < len(source) else None
                require(
                    actual == content,
                    f"patch context mismatch in {patch.old_path} at line {cursor + 1}",
                )
                cursor += 1
                consumed += 1
            if produces:
                output.append(content)
                produced += 1

        require(consumed == source_count, f"patch source count mismatch in {patch.old_path}")
        require(produced == target_count, f"patch target count mismatch in {patch.new_path}")
        source_index = cursor
        target_index = hunk_target_index + produced

    output.extend(source[source_index:])
    return "".join(output).encode("utf-8")


def strip_patch_prefix(path: str, prefix: str) -> str:
    require(path.startswith(prefix), f"patch path does not use locked prefix: {path}")
    name = path[len(prefix) :]
    require(name and "/" not in name and name not in {".", ".."}, f"unsafe patch path: {path}")
    return name


def check_source_derivation(config: dict) -> int:
    derivation = config["source_derivation"]
    sources = config["sources"]
    by_id = {source["id"]: source for source in sources}
    patch_sources = [by_id[source_id] for source_id in derivation["patch_source_ids"]]
    patch_text = "".join((ROOT / source["path"]).read_text() for source in patch_sources)
    patch_files = parse_unified_patch(patch_text)
    require(
        len(patch_files) == derivation["changed_file_count"],
        "errata changed-file count differs from the locked profile",
    )

    paper_directory = derivation["paper_directory"]
    paper_entries = {
        Path(source["path"]).name: source
        for source in sources
        if str(Path(source["path"]).parent) == paper_directory
        and Path(source["path"]).suffix == ".md"
    }
    actual_paper_files = {path.name for path in (ROOT / paper_directory).glob("*.md")}
    require(
        len(paper_entries) == derivation["paper_file_count"],
        "source registry does not lock the required paper file count",
    )
    require(set(paper_entries) == actual_paper_files, "source registry does not cover the paper tree exactly")

    changed: set[str] = set()
    for patch_file in patch_files:
        old_name = strip_patch_prefix(patch_file.old_path, derivation["old_prefix"])
        new_name = strip_patch_prefix(patch_file.new_path, derivation["new_prefix"])
        require(old_name == new_name, f"errata add, delete, or rename is unsupported: {old_name}")
        require(old_name in paper_entries, f"errata file is not source-locked: {old_name}")
        source = paper_entries[old_name]
        reviewed = (ROOT / source["path"]).read_bytes()
        base = apply_exact_patch_file(reviewed, patch_file, reverse=True)
        require(
            sha256_bytes(base) == source["base_sha256"],
            f"reverse errata result has wrong base hash: {old_name}",
        )
        require(
            apply_exact_patch_file(base, patch_file, reverse=False) == reviewed,
            f"forward errata result differs from reviewed bytes: {old_name}",
        )
        changed.add(old_name)

    for name, source in paper_entries.items():
        if name not in changed:
            require(
                source["base_sha256"] == source["sha256"],
                f"unchanged source has different base and reviewed hashes: {name}",
            )
    return len(paper_entries)


def check_documents(config: dict) -> None:
    for name, relative in config["documents"].items():
        path = repository_path(relative)
        require(path.is_file(), f"document {name} does not exist: {relative}")


def check_sources(config: dict) -> set[str]:
    source_text = SOURCES_PATH.read_text()
    registry_rows = SOURCE_ROW.findall(source_text)
    registry_ids = [row[0] for row in registry_rows]
    registry_duplicates = [item for item, count in Counter(registry_ids).items() if count > 1]
    require(not registry_duplicates, f"duplicate source registry IDs: {registry_duplicates}")
    registry = {source_id: (path, digest) for source_id, path, digest in registry_rows}
    source_ids: list[str] = []
    source_paths: list[str] = []
    for source in config["sources"]:
        source_id = source["id"]
        relative = source["path"]
        expected = source["sha256"]
        require(is_packaged_path(relative), f"source is outside the package: {source_id}")
        path = repository_path(relative)
        require(path.is_file(), f"source {source_id} does not exist: {relative}")
        actual = sha256(path)
        require(actual == expected, f"source hash mismatch for {source_id}: expected {expected}, got {actual}")
        require(source_id in registry, f"source registry does not name {source_id}")
        require(
            registry[source_id] == (relative, expected),
            f"source registry entry differs from config for {source_id}",
        )
        source_ids.append(source_id)
        source_paths.append(relative)

    duplicate_ids = [item for item, count in Counter(source_ids).items() if count > 1]
    duplicate_paths = [item for item, count in Counter(source_paths).items() if count > 1]
    require(not duplicate_ids, f"duplicate source IDs: {duplicate_ids}")
    require(not duplicate_paths, f"one source path has more than one ID: {duplicate_paths}")
    require(set(registry) == set(source_ids), "source registry and config use different source ID sets")
    return set(source_ids)


def check_cross_references(
    source_ids: set[str], decision_ids: set[str]
) -> tuple[set[str], set[str], set[str]]:
    errata_text = (CONTRACT_DIR / "reviewed-errata.md").read_text()
    errata_ids = set(ERRATA_ROW.findall(errata_text))
    require(errata_ids, "reviewed errata has no rows")

    literal_text = (CONTRACT_DIR / "literal-paper-model.md").read_text()
    paper_items = PAPER_ITEM.findall(literal_text)
    paper_duplicates = [item for item, count in Counter(paper_items).items() if count > 1]
    require(not paper_duplicates, f"duplicate literal paper IDs: {paper_duplicates}")
    require(paper_items, "literal paper model has no paper items")
    paper_ids = set(paper_items)

    base_text = (CONTRACT_DIR / "base-paper-model.md").read_text()
    base_items = BASE_ITEM.findall(base_text)
    base_duplicates = [item for item, count in Counter(base_items).items() if count > 1]
    require(not base_duplicates, f"duplicate base paper IDs: {base_duplicates}")
    require(base_items, "base paper model has no items")
    base_ids = set(base_items)

    markdown = "\n".join(path.read_text() for path in CONTRACT_DIR.glob("*.md"))
    # Source IDs use the SRC-PAPER and SRC-ERRATA namespaces. Assurance claim
    # IDs such as SRC-REVIEWED-LOCK are a different global ID family.
    used_sources = set(
        re.findall(r"SRC-(?:PAPER(?:-[A-Z0-9]+)*|ERRATA(?:-[A-Z0-9]+)*)", markdown)
    )
    used_decisions = set(re.findall(r"NSD-[A-Z0-9-]+", markdown))
    used_errata = set(re.findall(r"ERR-[A-Z0-9-]+", markdown))
    used_paper = set(re.findall(r"(?<!SRC-)PAPER-[A-Z0-9-]+-\d{3}", markdown))
    used_base = set(re.findall(r"BASE-[A-Z0-9-]+-\d{3}", markdown))

    require(not (used_sources - source_ids), f"unknown source references: {sorted(used_sources - source_ids)}")
    require(
        not (used_decisions - decision_ids),
        f"unknown decision references: {sorted(used_decisions - decision_ids)}",
    )
    require(not (used_errata - errata_ids), f"unknown errata references: {sorted(used_errata - errata_ids)}")
    require(not (used_paper - paper_ids), f"unknown paper references: {sorted(used_paper - paper_ids)}")
    require(not (used_base - base_ids), f"unknown base paper references: {sorted(used_base - base_ids)}")
    normative_text = CONTRACT_PATH.read_text()
    normative_errata = set(re.findall(r"ERR-[A-Z0-9-]+", normative_text))
    normative_decisions = set(re.findall(r"NSD-[A-Z0-9-]+", normative_text))
    require(
        errata_ids <= normative_errata,
        f"errata rows unused by the contract: "
        f"{sorted(errata_ids - normative_errata)}",
    )
    require(
        decision_ids <= normative_decisions,
        f"decision rows unused by the contract: {sorted(decision_ids - normative_decisions)}",
    )
    return base_ids, paper_ids, errata_ids


def check_evidence_schema(config: dict) -> None:
    relative = config["documents"]["rust_evidence_schema"]
    with (ROOT / relative).open() as source:
        schema = json.load(source)
    required = set(schema["required"])
    expected = {
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
    require(required == expected, "Rust evidence schema required-field set changed")
    require(schema["properties"]["schema_version"].get("const") == 3, "Rust evidence schema version is not 3")
    require(schema.get("additionalProperties") is False, "Rust evidence schema must reject unknown fields")
    require(schema["properties"]["origin"].get("const") == "rust-execution", "Rust evidence origin is not fixed")
    require(schema["properties"]["mutations"].get("minItems", 0) >= 1, "Rust evidence must include a mutation")
    semantic_properties = schema["properties"]["semantic_target"]["properties"]
    require("accepted" not in semantic_properties, "semantic result must be recomputed, not carried in Rust evidence")
    mutation_required = set(schema["properties"]["mutations"]["items"]["required"])
    require("rust_decision" in mutation_required, "mutations must carry observed Rust decisions")
    require("trace_hash" in mutation_required, "mutations must bind their Rust trace")
    require("expected_result_code" not in mutation_required, "mutations must not carry hand-authored expected results")
    run_required = set(schema["properties"]["run"]["required"])
    require("attestation" in run_required, "Rust evidence must include run provenance")
    producer_required = set(schema["properties"]["producer"]["required"])
    require(
        {"dirty", "cargo_lock_hash", "rustc", "target", "profile"} <= producer_required,
        "Rust evidence does not pin the full build identity",
    )
    decision_required = set(schema["$defs"]["rust_decision"]["required"])
    require("first_reject_rule" in decision_required, "Rust evidence lacks the first rejection rule")
    trace_required = set(schema["$defs"]["trace_event"]["required"])
    require(
        {"seq", "kind", "contract_rule", "source_symbol", "input_hash", "output_hash"}
        <= trace_required,
        "Rust trace event lacks a required semantic field",
    )


def check_auxiliary_schemas(config: dict) -> None:
    document_keys = (
        "transcript_profile_schema",
        "circuit_manifest_schema",
        "security_instantiation_schema",
        "rule_index_schema",
        "protocol_events_schema",
        "assurance_status_schema",
        "conformance_vector_schema",
    )
    schemas = []
    for key in document_keys:
        with (ROOT / config["documents"][key]).open() as source:
            schema = json.load(source)
        require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"{key} is not Draft 2020-12",
        )
        require(schema.get("additionalProperties") is False, f"{key} must reject unknown root fields")
        require(schema.get("required"), f"{key} has no required root fields")
        schemas.append(schema)

    ids = [schema.get("$id") for schema in schemas]
    require(all(ids) and len(ids) == len(set(ids)), "auxiliary schema IDs are missing or duplicate")

    transcript, circuit, security = schemas[:3]
    transcript_required = set(transcript["required"])
    require(
        {"encodings", "session", "events", "finalization", "security_model"}
        <= transcript_required,
        "transcript schema omits a protocol state-machine boundary",
    )
    challenge_required = set(transcript["$defs"]["challenge"]["required"])
    require(
        {"domain", "decoder", "distribution", "max_retries", "retry_counter_encoding", "on_exhaustion"}
        <= challenge_required,
        "transcript challenge schema is incomplete",
    )

    correspondence_required = set(circuit["properties"]["correspondence"]["required"])
    require(
        correspondence_required
        == {
            "public_encoding_injective",
            "verifier_key_digest_bound",
            "statement_digest_bound",
            "native_to_circuit_complete",
            "circuit_to_native_sound",
            "frontend_to_backend_sound",
        },
        "circuit schema does not require all correspondence directions",
    )
    require(circuit.get("allOf"), "circuit schema has no approved-profile restrictions")
    contract_properties = circuit["properties"]["contract"]["properties"]
    require(
        contract_properties.get("public_image_profile_id", {}).get("const")
        == "nightstream-statement-public-image-v1",
        "circuit schema does not bind the public-image profile",
    )
    approved_circuit = circuit["allOf"][0]["then"]["properties"]["circuit"]
    require(
        approved_circuit["properties"]["public_input_count"].get("const") == 9,
        "approved circuit schema does not fix nine public fields",
    )

    security_required = set(security["required"])
    require(
        {"model", "resource_limits", "terms", "assumptions", "total"} <= security_required,
        "security schema omits an accounting boundary",
    )
    require(security.get("allOf"), "security schema does not fail closed for release status")


def check_security_census(config: dict) -> None:
    paper = config["paper_goldilocks"]
    candidate = config["nightstream_candidate"]
    accounting = config["security_accounting"]
    selected = config["selected_joint_piccs"]
    sampler = config["sampler_accounting"]
    require(accounting["production_target_bits"] > 0, "security target is not positive")
    require(accounting["maximum_folds"] > 0, "maximum fold count is not positive")
    included_terms = accounting["included_terms"]
    open_terms = accounting["open_terms"]
    require(len(included_terms) == len(set(included_terms)), "duplicate included security term")
    require(len(open_terms) == len(set(open_terms)), "duplicate open security term")
    require(not (set(included_terms) & set(open_terms)), "a security term is both included and open")
    q = int(paper["q_decimal"])
    require(q == 2**64 - 2**32 + 1, "paper Goldilocks modulus is incorrect")
    require(int(paper["q_hex"], 16) == q, "paper Goldilocks decimal and hexadecimal moduli differ")
    require(
        paper["n_f_mod_phi_degree"] == paper["n_f_declared"] % paper["phi_degree"],
        "paper dimension remainder is incorrect",
    )
    require(
        paper["dimension_relation_satisfied"] == (paper["n_f_mod_phi_degree"] == 0),
        "paper dimension-conflict flag is incorrect",
    )
    require(paper["B"] == paper["b"] ** paper["k"], "paper protocol bound is not b^k")
    require(2 * paper["B"] < q, "paper protocol bound does not satisfy B<q/2")
    require(
        (paper["max_fresh_claims"] + paper["k"])
        * paper["expansion_T"]
        * (paper["b"] - 1)
        < paper["B"],
        "paper global PiRLC norm guard fails",
    )
    require(
        int(config["algebra_encoding"]["base_field_modulus"]) == q,
        "Nightstream base-field modulus differs from the paper profile",
    )
    require(
        candidate["public_field_width"]
        == candidate["public_ring_columns"] * paper["phi_degree"],
        "Nightstream public input width is not a complete ring width",
    )
    require(
        candidate["fresh_logical_width"] + candidate["fresh_zero_padding"]
        == candidate["public_field_width"],
        "Nightstream fresh public-input padding does not reach the public width",
    )
    require(
        candidate["logical_assignment_width"]
        == candidate["assignment_ring_columns"] * paper["phi_degree"],
        "Nightstream assignment width is not a complete ring width",
    )
    require(
        candidate["padded_rows"] == 2 ** candidate["row_variables"],
        "Nightstream padded row count does not match the row cube",
    )
    require(
        candidate["logical_rows"] <= candidate["padded_rows"]
        and candidate["logical_assignment_width"] <= candidate["padded_rows"],
        "Nightstream logical shape does not fit the row cube",
    )
    require(
        candidate["matrix_count"] == candidate["application_matrix_count"] + 1,
        "Nightstream matrix count does not include exactly one padded identity",
    )
    require(candidate["fresh_claims"] == 1, "Nightstream fresh source count differs from v1")
    require(candidate["running_claims"] == paper["k"], "Nightstream running count differs from paper k")
    require(
        candidate["source_claims"] == candidate["fresh_claims"] + candidate["running_claims"],
        "Nightstream source count differs from fresh plus running sources",
    )
    require(
        candidate["fresh_ccs_source_indices"] == list(range(candidate["fresh_claims"])),
        "Nightstream fresh CCS source indices differ",
    )
    require(
        candidate["norm_source_indices"] == list(range(candidate["source_claims"])),
        "Nightstream norm source indices differ",
    )
    require(
        candidate["carried_evaluation_source_indices"]
        == list(range(candidate["fresh_claims"], candidate["source_claims"]))
        and candidate["carried_local_to_global_offset"] == candidate["fresh_claims"],
        "Nightstream carried source mapping differs",
    )
    require(candidate["piccs_variant"] == "PaddedRowIdentity", "Nightstream PiCCS variant is not selected v1")
    require(candidate["norm_binding_closure"] == "padded-row-identity", "Nightstream norm binding is not closed")
    require(candidate["column_sumcheck"] == "absent", "selected profile still has a column SumCheck")
    require(candidate["extra_batch_challenges"] == "absent", "selected profile still has extra batching coins")

    extension_size = q ** paper["extension_degree"]
    challenge_size = len(paper["challenge_alphabet"]) ** paper["phi_degree"]
    require(
        paper["msis_infinity_bound"] == 8 * paper["expansion_T"] * paper["B"],
        "paper Goldilocks Module-SIS infinity bound is not 8*T*B",
    )

    fresh = candidate["fresh_claims"]
    running = candidate["running_claims"]
    matrices = candidate["matrix_count"]
    degree = paper["phi_degree"]
    row_variables = candidate["row_variables"]
    sumcheck_degree = max(candidate["polynomial_total_degree"] + 1, 2 * paper["b"], 2)
    sumcheck_terms = sumcheck_degree * row_variables
    d_sz_fresh = fresh - 1 + row_variables
    d_sz_norm = 2 * fresh + running - 1 + row_variables
    d_sz_carried = 2 * fresh + running + running * matrices * degree - 1
    d_sz = max(d_sz_fresh, d_sz_norm, d_sz_carried)
    n_field = sumcheck_terms + d_sz
    fork = fresh + running + 1
    numerator = n_field * challenge_size + fork * extension_size
    denominator = extension_size * challenge_size
    one_fold_bits = (denominator // numerator).bit_length() - 1
    lifetime_bits = (denominator // (numerator * accounting["maximum_folds"])).bit_length() - 1

    expected = {
        "fresh_claims": fresh,
        "running_claims": running,
        "source_claims": fresh + running,
        "row_variables": row_variables,
        "matrix_count": matrices,
        "ring_degree": degree,
        "polynomial_total_degree": candidate["polynomial_total_degree"],
        "sumcheck_degree": sumcheck_degree,
        "N_SC": sumcheck_terms,
        "D_SZ_fresh": d_sz_fresh,
        "D_SZ_norm": d_sz_norm,
        "D_SZ_carried": d_sz_carried,
        "D_SZ": d_sz,
        "N_field": n_field,
        "coordinate_fork_factor": fork,
        "one_fold_algebraic_bits_floor": one_fold_bits,
        "maximum_fold_algebraic_bits_floor": lifetime_bits,
    }
    for key, value in expected.items():
        require(selected[key] == value, f"selected_joint_piccs.{key} is {selected[key]}, expected {value}")
    require(candidate["sumcheck_degree"] == sumcheck_degree, "candidate SumCheck degree differs from the census")
    require(accounting["coordinate_fork_factor"] == fork, "security accounting has the wrong fork factor")

    sampler_profile = config["sampler_profile"]
    require(q % 5 == 1, "Goldilocks modulus does not have the selected sampler remainder")
    coefficient_count = sampler_profile["source_count"] * sampler_profile["ring_degree"]
    require(
        sampler_profile["coefficients_per_fold"] == coefficient_count,
        "sampler profile has the wrong coefficient count",
    )
    per_fold_numerator = coefficient_count
    maximum_numerator = coefficient_count * accounting["maximum_folds"]
    sampler_denominator = q ** sampler["per_fold_exhaustion_denominator_power"]
    sampler_expected = {
        "source_count": sampler_profile["source_count"],
        "coefficients_per_source": sampler_profile["ring_degree"],
        "coefficients_per_fold": coefficient_count,
        "maximum_attempts_per_coefficient": sampler_profile["maximum_attempts_per_coefficient"],
        "per_fold_exhaustion_numerator": per_fold_numerator,
        "per_fold_exhaustion_denominator_power": sampler_profile["maximum_attempts_per_coefficient"],
        "per_fold_exhaustion_bits_floor": (sampler_denominator // per_fold_numerator).bit_length() - 1,
        "maximum_fold_exhaustion_numerator": maximum_numerator,
        "maximum_fold_exhaustion_bits_floor": (sampler_denominator // maximum_numerator).bit_length() - 1,
    }
    for key, value in sampler_expected.items():
        require(sampler[key] == value, f"sampler_accounting.{key} is {sampler[key]}, expected {value}")

    maximum_fold_transcript_squeezes = (
        2
        + candidate["row_variables"]
        + coefficient_count * sampler_profile["maximum_attempts_per_coefficient"]
        + 1
    )
    maximum_public_image_squeezes = 1
    maximum_squeezes_per_fold = (
        maximum_fold_transcript_squeezes + maximum_public_image_squeezes
    )
    maximum_squeezes_per_proof = maximum_squeezes_per_fold * accounting["maximum_folds"]
    require(
        accounting["maximum_fold_transcript_squeezes_per_fold"]
        == maximum_fold_transcript_squeezes,
        "fold-transcript squeeze census differs",
    )
    require(
        accounting["maximum_public_image_squeezes_per_fold"]
        == maximum_public_image_squeezes,
        "public-image squeeze census differs",
    )
    require(accounting["maximum_protocol_squeezes_per_fold"] == maximum_squeezes_per_fold, "per-fold squeeze census differs")
    require(accounting["maximum_protocol_squeezes_per_proof"] == maximum_squeezes_per_proof, "per-proof squeeze census differs")
    require(
        accounting["maximum_verifier_key_squeezes_per_key"] == 1,
        "verifier-key squeeze census differs",
    )
    require(
        accounting["maximum_protocol_squeezes_per_key"]
        == maximum_squeezes_per_proof
        + accounting["maximum_verifier_key_squeezes_per_key"],
        "per-key squeeze census differs",
    )
    require(accounting["maximum_adaptive_oracle_queries"] >= accounting["maximum_protocol_squeezes_per_key"], "oracle-query limit excludes a maximum-size proof")

    policy = config["security_policy"]
    for key in (
        "production_target_bits",
        "maximum_folds_per_proof",
        "maximum_sessions_per_key",
        "maximum_proofs_per_key",
        "maximum_adaptive_oracle_queries",
    ):
        accounting_key = "maximum_folds" if key == "maximum_folds_per_proof" else key
        require(policy[key] == accounting[accounting_key], f"security policy and accounting differ: {key}")
    require(policy["model"] == accounting["model"], "security model and accounting differ")


def _determinant_mod(matrix: list[list[int]], modulus: int) -> int:
    """Return a square matrix determinant modulo a prime."""
    work = [[value % modulus for value in row] for row in matrix]
    result = 1
    for column in range(len(work)):
        pivot = next((row for row in range(column, len(work)) if work[row][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        pivot_value = work[column][column]
        result = result * pivot_value % modulus
        inverse = pow(pivot_value, modulus - 2, modulus)
        for row in range(column + 1, len(work)):
            factor = work[row][column] * inverse % modulus
            for item in range(column, len(work)):
                work[row][item] = (work[row][item] - factor * work[column][item]) % modulus
    return result % modulus


def _rotate_left_32(value: int, amount: int) -> int:
    value &= 0xFFFF_FFFF
    return ((value << amount) | (value >> (32 - amount))) & 0xFFFF_FFFF


def _chacha8_block(seed: list[int], block_counter: int) -> list[int]:
    """Return one standard ChaCha8 block for a zero 64-bit stream ID."""
    require(len(seed) == 32 and all(0 <= value <= 255 for value in seed), "invalid Ajtai ChaCha8 test seed")
    require(0 <= block_counter < 2**64, "Ajtai ChaCha8 block counter is outside u64")
    key = [
        int.from_bytes(bytes(seed[index : index + 4]), "little")
        for index in range(0, 32, 4)
    ]
    initial = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        *key,
        block_counter & 0xFFFF_FFFF,
        block_counter >> 32,
        0,
        0,
    ]
    state = list(initial)

    def quarter(a: int, b: int, c: int, d: int) -> None:
        state[a] = (state[a] + state[b]) & 0xFFFF_FFFF
        state[d] = _rotate_left_32(state[d] ^ state[a], 16)
        state[c] = (state[c] + state[d]) & 0xFFFF_FFFF
        state[b] = _rotate_left_32(state[b] ^ state[c], 12)
        state[a] = (state[a] + state[b]) & 0xFFFF_FFFF
        state[d] = _rotate_left_32(state[d] ^ state[a], 8)
        state[c] = (state[c] + state[d]) & 0xFFFF_FFFF
        state[b] = _rotate_left_32(state[b] ^ state[c], 7)

    for _ in range(4):
        quarter(0, 4, 8, 12)
        quarter(1, 5, 9, 13)
        quarter(2, 6, 10, 14)
        quarter(3, 7, 11, 15)
        quarter(0, 5, 10, 15)
        quarter(1, 6, 11, 12)
        quarter(2, 7, 8, 13)
        quarter(3, 4, 9, 14)
    return [
        (state[index] + initial[index]) & 0xFFFF_FFFF
        for index in range(16)
    ]


def _chacha8_words(seed: list[int], start: int, count: int) -> list[int]:
    require(start >= 0 and count >= 0, "negative Ajtai ChaCha8 word range")
    result = []
    block = start // 16
    offset = start % 16
    while len(result) < count:
        words = _chacha8_block(seed, block)
        result.extend(words[offset:])
        block += 1
        offset = 0
    return result[:count]


def _chacha8_bytes(seed: list[int], count: int) -> list[int]:
    words = _chacha8_words(seed, 0, (count + 3) // 4)
    data = b"".join(value.to_bytes(4, "little") for value in words)
    return list(data[:count])


def check_profile_consistency(config: dict) -> None:
    """Check exact encodings, section counts, tags, and Poseidon2 parameters."""
    paper = config["paper_goldilocks"]
    candidate = config["nightstream_candidate"]
    algebra = config["algebra_encoding"]
    commitment = config["commitment_profile"]
    setup = config["ajtai_setup_v1"]
    structure = config["structure_encoding_v1"]
    verifier_key_digest = config["verifier_key_digest_v1"]
    public_image = config["public_image_v1"]
    strong_set = config["strong_set_profile"]
    transcript = config["transcript_profile"]
    sections = config["container_sections"]
    tags = config["transcript_tags"]
    poseidon = config["poseidon2_goldilocks_v1"]
    duplex_test = config["duplex_test_v1"]
    sampler_profile = config["sampler_profile"]
    q = int(paper["q_decimal"])

    declared_literals = [
        (paper, "field", "Goldilocks"),
        (paper, "phi_nonzero_degrees", [0, 27, 54]),
        (candidate, "status", "approved"),
        (candidate, "profile_version", 1),
        (candidate, "source_index_map", "source-0=fresh-0;sources-1-through-14=running-0-through-13"),
        (candidate, "row_domain", "single-little-endian-boolean-cube-24"),
        (candidate, "domain_padding_map", "logical-prefix-then-zero"),
        (candidate, "identity_matrix", "M_0=[I_11437038;0]"),
        (candidate, "norm_terminal", "constant-term-of-y_ring-source-M_0"),
        (candidate, "public_carrier_authority", "x-only"),
        (candidate, "derived_carrier_evaluations", "verifier-recompute-or-discard"),
        (candidate, "split_algorithm", "centered-common-sign-binary-little-endian-14"),
        (candidate, "split_out_of_bound", "prover-assignment-or-verifier-public-input-error"),
        (candidate, "hash_family", "Poseidon2-only"),
        (candidate, "circuit_target", "PaddedRowIdentity-current-circuit-replacement"),
        (candidate, "decider_family", "Spartan-with-WHIR-profile-manifest-v1"),
        (algebra, "extension_polynomial", "U^2-7"),
        (algebra, "extension_basis", ["1", "U"]),
        (algebra, "ring_polynomial", "X^54+X^27+1"),
        (algebra, "ring_coefficient_order", "degree-0-through-degree-53"),
        (algebra, "public_input_order", "five-consecutive-ring-elements"),
        (algebra, "container_header", "magic[8]||version_u16_le||variant_u16_le||section_count_u32_le"),
        (algebra, "section_header", "section_id_u32_le||field_count_u32_le"),
        (algebra, "container_body", "for-each-section-in-declared-order:section-header||base-field-values"),
        (algebra, "section_payload", "field_count-consecutive-8-byte-canonical-base-fields"),
        (algebra, "unknown_section", "reject"),
        (algebra, "duplicate_section", "reject"),
        (algebra, "trailing_bytes", "reject"),
        (algebra, "noncanonical_field", "reject"),
        (commitment, "scheme", "Ajtai-Phi81-v1"),
        (commitment, "orientation", "left-matrix-vector"),
        (commitment, "commitment_equation", "commitment[row]=sum_column(A[row,column]*message[column])"),
        (commitment, "transposed_or_affine_variant", "reject"),
        (commitment, "setup_mode", "verifier-key-seeded-matrix"),
        (commitment, "setup_scope", "commitment-parameter-generation-only"),
        (commitment, "matrix_entries", "uniform-R_F-by-coefficient-rejection"),
        (commitment, "commitment_encoding", "18-ring-elements-in-row-order"),
        (commitment, "seed_and_dimensions_bound_by", "verifier-key-Poseidon2-digest"),
        (transcript, "state_field", "Goldilocks"),
        (transcript, "absorb", "additive-rate-lanes"),
        (transcript, "frame", "tag||payload-length||payload"),
        (transcript, "frame_tag_type", "canonical-base-field"),
        (transcript, "challenge_extension_codec", "two-consecutive-base-field-lanes-c0-c1"),
        (sampler_profile, "candidate_domain", "uniform-Goldilocks-field-element"),
        (sampler_profile, "acceptance_threshold", "q-1"),
        (sampler_profile, "accepted_condition", "candidate<q-1"),
        (sampler_profile, "digit_map", "candidate-mod-5 maps 0,1,2,3,4 to -2,-1,0,1,2 then iota_q"),
        (sampler_profile, "counter_order", "source-major-then-coefficient-then-attempt"),
        (sampler_profile, "exhaustion", "reject-proof"),
        (setup, "word_encoding", "sixteen-u32-little-endian-words-per-64-byte-block"),
        (setup, "next_u64_encoding", "low-u32-then-high-u32"),
        (setup, "column_to_chunk", "chunk=floor(column/chunk_size);local-column=column-mod-chunk-size"),
        (setup, "chunk_stream_order", "local-column-then-ring-coefficient"),
        (setup, "chunk_stream_counter_reset", "zero-for-each-chunk-seed"),
        (poseidon, "sbox", "x^7"),
        (poseidon, "external_matrix", "[[2*M4,M4],[M4,2*M4]]"),
        (poseidon, "internal_matrix", "all-ones-plus-diagonal"),
        (config["security_policy"], "proof_of_knowledge", True),
        (config["security_policy"], "quantum_security_claim", False),
        (config["security_policy"], "setup_trust", "uniform-Ajtai-matrix-or-stated-seeded-PRG-assumption"),
        (config["decider_profile"], "family", "Spartan-with-WHIR-profile-manifest-v1"),
        (config["decider_profile"], "public_image", "nightstream-statement-public-image-v1"),
        (config["decider_profile"], "unsupported_backend", "reject"),
        (config["decider_profile"], "backend_parameters", "verifier-key-owned-versioned-manifest"),
        (config["decider_profile"], "on_chain_parser", "same-canonical-container-rules"),
    ]
    for table, key, expected in declared_literals:
        require(table.get(key) == expected, f"declared profile field differs: {key}")

    require(int(algebra["base_field_modulus"]) == q, "encoding field modulus differs")
    require(algebra["base_field_bytes"] == 8, "base-field encoding is not eight bytes")
    require(algebra["base_field_byte_order"] == "little-endian", "base-field byte order differs")
    require(algebra["extension_degree"] == paper["extension_degree"], "extension degree differs")
    require(algebra["extension_coefficient_order"] == ["c0", "c1"], "extension coefficient order differs")
    require(pow(7, (q - 1) // 2, q) == q - 1, "U^2-7 is not irreducible over the base field")
    require(algebra["ring_element_base_fields"] == paper["phi_degree"], "ring encoding width differs")
    require(len(algebra["proof_container_magic"].encode()) == 8, "proof magic is not eight bytes")
    require(len(algebra["statement_container_magic"].encode()) == 8, "statement magic is not eight bytes")
    require(algebra["proof_container_magic"] != algebra["statement_container_magic"], "container magics collide")
    require(algebra["container_version"] == 1 and algebra["container_variant"] == 1, "container version or variant differs")

    commitment_fields = commitment["commitment_ring_elements"] * paper["phi_degree"]
    require(commitment["kappa"] == paper["kappa"], "commitment kappa differs from the paper profile")
    require(commitment["message_ring_columns"] == candidate["assignment_ring_columns"], "commitment message width differs")
    require(commitment["commitment_ring_elements"] == commitment["kappa"], "commitment output width differs")
    require(commitment["setup_expander"] == setup["id"], "commitment setup-expander ID differs")
    require(setup["rounds"] == 8 and setup["seed_bytes"] == 32, "Ajtai setup ChaCha8 parameters differ")
    require(setup["stream_id"] == 0 and setup["initial_block_counter"] == 0, "Ajtai setup stream position differs")
    require(setup["selected_output_rows"] == commitment["kappa"], "Ajtai setup row count differs")
    require(setup["selected_message_columns"] == commitment["message_ring_columns"], "Ajtai setup message width differs")
    require(setup["ring_coefficient_count"] == paper["phi_degree"], "Ajtai setup ring width differs")
    selected_chunk_size = max(1024, min(setup["selected_message_columns"], 32768))
    selected_chunk_count = (setup["selected_message_columns"] + selected_chunk_size - 1) // selected_chunk_size
    require(setup["selected_chunk_size"] == selected_chunk_size, "Ajtai selected chunk size differs")
    require(setup["selected_chunk_count"] == selected_chunk_count, "Ajtai selected chunk count differs")
    require(setup["matrix_order"] == "output-row-then-message-column-then-ring-coefficient", "Ajtai matrix order differs")
    require(setup["coefficient_batch"] == "read-54-consecutive-u64-values-before-any-fallback", "Ajtai coefficient batch differs")
    require(setup["coefficient_accept"] == f"x<{q}", "Ajtai coefficient acceptance differs")
    require(setup["coefficient_fallback"] == "replace-rejected-slots-in-index-order-with-next-accepted-u64", "Ajtai coefficient fallback differs")
    test_seed = setup["test_seed"]
    require(_chacha8_words(test_seed, 0, 64) == setup["test_first_64_u32"], "Ajtai ChaCha8 initial test vector differs")
    require(
        _chacha8_words(test_seed, setup["test_high_word_start"], 8)
        == setup["test_high_8_u32"],
        "Ajtai ChaCha8 random-access test vector differs",
    )
    test_chunk_size = max(1024, min(setup["test_setup_message_columns"], 32768))
    test_chunk_count = (setup["test_setup_message_columns"] + test_chunk_size - 1) // test_chunk_size
    row_seed_bytes = _chacha8_bytes(test_seed, setup["test_setup_rows"] * 32)
    row_seeds = [row_seed_bytes[index : index + 32] for index in range(0, len(row_seed_bytes), 32)]
    expected_chunk_seeds = [
        [
            _chacha8_bytes(row_seed, test_chunk_count * 32)[index : index + 32]
            for index in range(0, test_chunk_count * 32, 32)
        ]
        for row_seed in row_seeds
    ]
    require(setup["test_setup_chunk_size"] == test_chunk_size, "Ajtai setup chunk test size differs")
    require(setup["test_setup_chunk_seeds"] == expected_chunk_seeds, "Ajtai setup chunk-seed test vector differs")
    require(structure["id"] == "nightstream-sparse-structure-v1", "Structure encoding ID differs")
    require(structure["encoding_version"] == 1, "Structure encoding version differs")
    require(structure["identity_variant_code"] == 1, "Structure identity variant differs")
    require(
        structure["header_fields"]
        == [
            "encoding_version",
            "logical_rows",
            "padded_rows",
            "row_variables",
            "logical_assignment_width",
            "public_field_width",
            "application_matrix_count",
            "matrix_count",
            "polynomial_total_degree",
            "identity_variant_code",
        ],
        "Structure header layout differs",
    )
    require(structure["integer_encoding"] == "canonical-base-field", "Structure integer encoding differs")
    require(structure["identity_encoding"] == "M_0-is-implicit-from-identity-variant-code", "Structure identity encoding differs")
    require(structure["padding_encoding"] == "logical-prefix-only-zero-padding-is-implicit", "Structure padding encoding differs")
    require(structure["application_matrix_indices"] == "1-through-13", "Structure matrix indices differ")
    require(structure["application_matrix_layout"] == "matrix_index||nonzero_count||row_column_value_triples", "Structure matrix layout differs")
    require(structure["matrix_entry_order"] == "strict-row-major-then-column-major", "Structure matrix order differs")
    require(
        structure["matrix_entry_constraints"]
        == "row<logical_rows,column<logical_assignment_width,0<value<q,no-duplicates",
        "Structure matrix constraints differ",
    )
    require(structure["polynomial_variable_count"] == candidate["application_matrix_count"], "Structure polynomial arity differs")
    require(structure["polynomial_layout"] == "term_count||coefficient_and_13_exponents_per_term", "Structure polynomial layout differs")
    require(structure["polynomial_term_order"] == "strict-lexicographic-exponent-tuples", "Structure polynomial order differs")
    require(
        structure["polynomial_term_constraints"]
        == "0<coefficient<q,each-exponent<=8,1<=sum-exponents<=8,no-duplicates,maximum-sum-is-8",
        "Structure polynomial constraints differ",
    )
    require(structure["lift_rule"] == "prepend-zero-exponent-for-u_0", "Structure lift rule differs")
    require(
        verifier_key_digest["id"] == "nightstream-verifier-key-poseidon2-v1"
        and verifier_key_digest["algorithm"] == f"fresh-{transcript['id']}",
        "verifier-key digest algorithm differs",
    )
    require(
        verifier_key_digest["session_frame_tag"] == "session"
        and verifier_key_digest["session_frame_payload"]
        == ["contract_domain_tag", "profile_version_tag"],
        "verifier-key digest session frame differs",
    )
    require(
        verifier_key_digest["preimage_frame_tag"] == "verifier_key_preimage"
        and verifier_key_digest["preimage_layout"]
        == [
            "setup_variant_code",
            "kappa",
            "message_ring_columns",
            "setup_seed_byte_count",
            "setup_seed_bytes_as_field_lanes",
            "structure_stream_field_count",
            "structure_stream",
        ],
        "verifier-key digest preimage layout differs",
    )
    require(verifier_key_digest["setup_variant_code"] == 1, "verifier-key setup code differs")
    require(
        verifier_key_digest["setup_seed_byte_count"] == setup["seed_bytes"]
        and verifier_key_digest["setup_seed_lane_encoding"]
        == "one-u8-value-per-canonical-base-field",
        "verifier-key seed encoding differs",
    )
    require(verifier_key_digest["structure_encoding"] == structure["id"], "verifier-key Structure encoding differs")
    require(
        verifier_key_digest["digest_squeeze_tag"] == "verifier_key_digest"
        and verifier_key_digest["digest_base_fields"] == 4,
        "verifier-key digest output differs",
    )
    require(
        public_image["id"] == "nightstream-statement-public-image-v1"
        and public_image["algorithm"] == f"fresh-{transcript['id']}",
        "public-image digest algorithm differs",
    )
    require(
        public_image["session_frame_tag"] == "session"
        and public_image["session_frame_payload"]
        == ["contract_domain_tag", "profile_version_tag"],
        "public-image session frame differs",
    )
    require(
        public_image["verifier_key_frame_tag"] == "verifier_key"
        and public_image["verifier_key_digest_base_fields"]
        == verifier_key_digest["digest_base_fields"],
        "public-image verifier-key frame differs",
    )
    require(
        public_image["statement_frame_tag"] == "statement"
        and public_image["statement_field_stream"]
        == "canonical-statement-section-payloads-in-section-id-order"
        and public_image["statement_field_count"]
        == sections["statement_total_base_fields"],
        "public-image statement frame differs",
    )
    expected_public_fields = [
        "contract_domain_tag",
        "profile_version_tag",
        "container_variant",
        "fold_index",
        "fold_count",
        "statement_digest_0",
        "statement_digest_1",
        "statement_digest_2",
        "statement_digest_3",
    ]
    require(
        public_image["digest_squeeze_tag"] == "statement_digest"
        and public_image["digest_base_fields"] == 4
        and public_image["public_field_order"] == expected_public_fields
        and public_image["public_base_fields"] == len(expected_public_fields)
        and public_image["explicit_field_check"]
        == "match-selected-profile-and-decoded-statement",
        "public-image output layout differs",
    )
    require(
        commitment["msis_infinity_bound"]
        == 8 * strong_set["expansion_T"] * paper["B"],
        "commitment Module-SIS bound differs",
    )

    eta = strong_set["eta"]
    divisor = strong_set["theorem8_divisor"]
    order = next(number for number in range(1, eta + 1) if pow(q, number, eta) == 1)
    phi_eta = sum(1 for value in range(1, eta + 1) if math.gcd(value, eta) == 1)
    phi_divisor = sum(1 for value in range(1, divisor + 1) if math.gcd(value, divisor) == 1)
    require(eta == paper["eta"] and eta % divisor == 0, "strong-set divisor does not divide eta")
    require(q % divisor == 1, "field modulus is not one modulo the strong-set divisor")
    require(order == eta // divisor == strong_set["order_mod_eta"], "strong-set order condition differs")
    require(strong_set["tau"] == divisor, "selected odd-divisor tau differs")
    require(strong_set["phi_divisor"] == phi_divisor == 2, "selected divisor totient differs")
    require(strong_set["alphabet"] == paper["challenge_alphabet"], "strong-set alphabet differs")
    require(strong_set["maximum_member_norm"] == max(abs(value) for value in strong_set["alphabet"]), "strong-set member norm differs")
    require(strong_set["maximum_difference_norm"] == max(strong_set["alphabet"]) - min(strong_set["alphabet"]), "strong-set difference norm differs")
    require(strong_set["maximum_difference_norm"] ** 2 * strong_set["tau"] < q, "strong-set difference does not satisfy Theorem 8")
    require(strong_set["expansion_T"] == 2 * phi_eta * strong_set["maximum_member_norm"], "strong-set expansion factor differs")
    require(
        strong_set["challenge_set_cardinality"]
        == f"{len(strong_set['alphabet'])}^{paper['phi_degree']}",
        "strong-set cardinality differs",
    )
    guard_left = (candidate["fresh_claims"] + candidate["running_claims"]) * strong_set["expansion_T"] * (paper["b"] - 1)
    require(strong_set["norm_guard_left"] == guard_left, "selected norm-guard left side differs")
    require(strong_set["norm_guard_right"] == paper["B"] and guard_left < paper["B"], "selected norm guard fails")

    extension_fields = algebra["extension_degree"]
    point_fields = candidate["row_variables"] * extension_fields
    output_fields_per_claim = (
        candidate["matrix_count"] * paper["phi_degree"] * extension_fields
    )
    fresh_claim_fields = commitment_fields + candidate["public_field_width"]
    running_claim_fields = fresh_claim_fields + output_fields_per_claim
    expected_statement = [
        2,
        fresh_claim_fields,
        point_fields,
        candidate["running_claims"] * running_claim_fields,
    ]
    expected_proof = [
        candidate["row_variables"] * (candidate["sumcheck_degree"] + 1) * extension_fields,
        candidate["source_claims"] * output_fields_per_claim,
        paper["k"] * (commitment_fields + output_fields_per_claim),
    ]
    require(sections["statement_section_ids"] == sorted(set(sections["statement_section_ids"])), "statement section IDs are not unique and ordered")
    require(sections["proof_section_ids"] == sorted(set(sections["proof_section_ids"])), "proof section IDs are not unique and ordered")
    require(not (set(sections["statement_section_ids"]) & set(sections["proof_section_ids"])), "statement and proof section IDs collide")
    require(len(sections["statement_section_names"]) == len(expected_statement), "statement section names differ")
    require(len(sections["proof_section_names"]) == len(expected_proof), "proof section names differ")
    require(sections["statement_section_field_counts"] == expected_statement, "statement field census differs")
    require(sections["proof_section_field_counts"] == expected_proof, "proof field census differs")
    require(sections["statement_total_base_fields"] == sum(expected_statement), "statement total differs")
    require(sections["proof_total_base_fields"] == sum(expected_proof), "proof total differs")
    header_bytes = sections["container_header_bytes"]
    section_header_bytes = sections["section_header_bytes"]
    field_bytes = sections["field_bytes"]
    require(header_bytes == 16, "container header byte width differs")
    require(section_header_bytes == 8, "section header byte width differs")
    require(field_bytes == algebra["base_field_bytes"], "container field byte width differs")
    byte_censuses = (
        ("statement", sections["statement_section_ids"], sections["statement_total_base_fields"], sections["statement_total_bytes"], 318832),
        ("proof", sections["proof_section_ids"], sections["proof_total_base_fields"], sections["proof_total_bytes"], 463528),
    )
    for label, section_ids, total_fields, declared_bytes, expected_bytes in byte_censuses:
        computed_bytes = header_bytes + len(section_ids) * section_header_bytes + total_fields * field_bytes
        require(declared_bytes == computed_bytes == expected_bytes, f"{label} byte census differs")
    require(
        sections["length_arithmetic"] == "checked-before-payload-allocation",
        "container length-check policy differs",
    )
    require(sections["lifecycle_fields"] == ["fold_index", "fold_count"], "lifecycle field order differs")
    require(sections["lifecycle_condition"] == "0<=fold_index<fold_count<=64", "lifecycle condition differs")
    require(sections["fresh_claim_order"] == "commitment-then-270-field-x", "fresh claim order differs")
    require(sections["shared_row_point_order"] == "coordinate-0-through-23-each-as-c0-then-c1", "row point order differs")
    require(sections["running_claim_order"] == "source-0-through-13-each-as-commitment-then-270-field-x-then-14-y_ring-values", "running claim order differs")
    require(sections["piccs_round_order"] == "round-0-through-23-each-as-degree-0-through-9-each-c0-then-c1", "PiCCS round order differs")
    require(sections["piccs_output_order"] == "source-0-through-14-then-matrix-0-through-13-each-R_K-in-ring-order", "PiCCS output order differs")
    require(sections["pidec_child_order"] == "child-0-through-13-each-as-commitment-then-14-y_ring-values; child-x-is-verifier-derived", "PiDEC child order differs")

    require(transcript["width"] == transcript["rate"] + transcript["capacity"], "sponge width differs from rate plus capacity")
    require(
        transcript["absorb_cursor_initial"] == 0
        and transcript["absorb_cursor_after_permutation"] == 0,
        "transcript absorb cursor rule differs",
    )
    require(transcript["last_rate_lane"] == transcript["rate"] - 1, "last rate lane differs")
    require(transcript["first_capacity_lane"] == transcript["rate"], "first capacity lane differs")
    require(
        transcript["squeeze_request_frame"]
        == "challenge_frame||2||squeeze_tag||requested_base_field_count"
        and transcript["squeeze_block_order"] == "rate-lanes-0-through-3",
        "transcript squeeze framing differs",
    )
    domain_tag = transcript["contract_domain_tag"]
    require(
        set(tags)
        == {
            "session",
            "verifier_key",
            "statement",
            "verifier_key_preimage",
            "verifier_key_digest",
            "statement_digest",
            "piccs_input",
            "piccs_alpha",
            "piccs_gamma",
            "sumcheck_round",
            "piccs_output",
            "pirlc_candidate",
            "pirlc_output",
            "pidec_output",
            "fold_finalize",
            "challenge_frame",
        },
        "transcript tag family differs from v1",
    )
    tag_values = list(tags.values())
    require(len(tag_values) == len(set(tag_values)), "duplicate transcript tag value")
    require(domain_tag not in tag_values, "contract-domain tag collides with an event tag")
    require(all(isinstance(value, int) and 0 < value < q for value in [domain_tag, *tag_values]), "invalid transcript tag value")
    require(
        [
            transcript["absorb_padding_cursor_value"],
            transcript["absorb_padding_last_rate_value"],
            transcript["ratchet_capacity_value"],
            transcript["continuation_capacity_value"],
        ]
        == [1, 2, 3, 4],
        "transcript direction constants differ",
    )

    require(int(poseidon["field_modulus"]) == q, "Poseidon2 field differs")
    require(poseidon["width"] == transcript["width"], "Poseidon2 width differs from transcript")
    require(poseidon["rate"] == transcript["rate"], "Poseidon2 rate differs from transcript")
    require(poseidon["capacity"] == transcript["capacity"], "Poseidon2 capacity differs from transcript")
    require(poseidon["state_orientation"] == "column-vector", "Poseidon2 state orientation differs")
    require(
        poseidon["linear_application"]
        == "new_state[row]=sum_column(matrix[row,column]*old_state[column])"
        and poseidon["constant_order"] == "round-major-then-lane-major",
        "Poseidon2 operation order differs",
    )
    initial_constants = poseidon["initial_round_constants"]
    partial_constants = poseidon["partial_round_constants"]
    terminal_constants = poseidon["terminal_round_constants"]
    require(len(initial_constants) == poseidon["initial_full_rounds"] * poseidon["width"], "initial Poseidon2 constant count differs")
    require(len(partial_constants) == poseidon["partial_rounds"], "partial Poseidon2 constant count differs")
    require(len(terminal_constants) == poseidon["terminal_full_rounds"] * poseidon["width"], "terminal Poseidon2 constant count differs")
    require(all(0 <= value < q for value in initial_constants + partial_constants + terminal_constants), "Poseidon2 round constant is outside the field")

    m4 = [[int(value) for value in row.split(",")] for row in poseidon["external_m4_rows"]]
    require(len(m4) == 4 and all(len(row) == 4 for row in m4), "Poseidon2 M4 shape differs")
    external = []
    for row in range(8):
        external.append([])
        for column in range(8):
            block = m4[row % 4][column % 4]
            external[row].append(block * (2 if row // 4 == column // 4 else 1))
    diagonal = poseidon["internal_diagonal"]
    require(len(diagonal) == poseidon["width"], "Poseidon2 internal diagonal width differs")
    internal = [
        [1 + (diagonal[row] if row == column else 0) for column in range(8)]
        for row in range(8)
    ]
    require(_determinant_mod(external, q) != 0, "Poseidon2 external matrix is singular")
    require(_determinant_mod(internal, q) != 0, "Poseidon2 internal matrix is singular")
    require(
        poseidon2_permute(poseidon["test_zero_input"], poseidon)
        == poseidon["test_zero_output"],
        "Poseidon2 zero test vector differs",
    )
    require(
        poseidon2_permute(poseidon["test_sequence_input"], poseidon)
        == poseidon["test_sequence_output"],
        "Poseidon2 sequence test vector differs",
    )
    duplex = FieldDuplex(poseidon, transcript)
    duplex.frame(tags[duplex_test["first_frame_tag"]], duplex_test["first_frame_payload"])
    first_output = duplex.tagged_squeeze(
        tags["challenge_frame"],
        tags[duplex_test["first_squeeze_tag"]],
        duplex_test["first_squeeze_base_fields"],
    )
    require(first_output == duplex_test["first_squeeze_output"], "field-duplex first test vector differs")
    duplex.frame(tags[duplex_test["second_frame_tag"]], duplex_test["second_frame_payload"])
    second_output = duplex.tagged_squeeze(
        tags["challenge_frame"],
        tags[duplex_test["second_squeeze_tag"]],
        duplex_test["second_squeeze_base_fields"],
    )
    require(second_output == duplex_test["second_squeeze_output"], "field-duplex ratchet test vector differs")


def check_coverage(
    config: dict,
    source_ids: set[str],
    decision_ids: set[str],
    repository_mode: bool,
) -> tuple[set[str], dict[str, str]]:
    contract_text = CONTRACT_PATH.read_text()
    contract_matches = list(RULE_HEADING.finditer(contract_text))
    contract_ids = [match.group(1) for match in contract_matches]
    duplicates = [item for item, count in Counter(contract_ids).items() if count > 1]
    require(not duplicates, f"duplicate normative rule headings: {duplicates}")
    require(contract_ids, "normative contract has no rule headings")
    for index, match in enumerate(contract_matches):
        rule = match.group(1)
        end = contract_matches[index + 1].start() if index + 1 < len(contract_matches) else len(contract_text)
        block = contract_text[match.end() : end]
        if rule.startswith("SN-"):
            require("Source:" in block, f"paper rule {rule} has no source citation")
        else:
            require("Decision:" in block, f"Nightstream rule {rule} has no decision citation")

    with COVERAGE_PATH.open(newline="") as source:
        rows = list(csv.DictReader(source))
    require(rows, "coverage map has no rows")

    coverage_ids = [row["contract_id"] for row in rows]
    duplicate_coverage = [item for item, count in Counter(coverage_ids).items() if count > 1]
    require(not duplicate_coverage, f"duplicate coverage rows: {duplicate_coverage}")

    missing = sorted(set(contract_ids) - set(coverage_ids))
    extra = sorted(set(coverage_ids) - set(contract_ids))
    require(not missing, f"normative rules missing from coverage: {missing}")
    require(not extra, f"coverage rows without normative rules: {extra}")

    state_columns = (
        "lean_assurance",
        "rust_conformance",
        "rust_origin_assurance",
        "circuit_assurance",
        "reduction_assurance",
    )
    evidence_columns = (
        "lean_model",
        "rust_implementation",
        "rust_origin_evidence",
        "circuit_evidence",
        "reduction_evidence",
    )

    for row in rows:
        rule = row["contract_id"]
        row_sources = split_ids(row["source_ids"])
        row_decisions = split_ids(row["decision_ids"])
        blockers = split_ids(row["blocker_ids"])

        require(row_sources, f"{rule} has no source classification")
        unknown_sources = [item for item in row_sources if item != "not-in-paper" and item not in source_ids]
        require(not unknown_sources, f"{rule} uses unknown sources: {unknown_sources}")
        if "not-in-paper" in row_sources:
            require(len(row_sources) == 1, f"{rule} mixes not-in-paper with paper sources")
            require(row_decisions, f"{rule} is not in the paper and has no decision ID")

        unknown_decisions = [item for item in row_decisions if item not in decision_ids]
        unknown_blockers = [item for item in blockers if item not in decision_ids]
        require(not unknown_decisions, f"{rule} uses unknown decisions: {unknown_decisions}")
        require(not unknown_blockers, f"{rule} uses unknown blockers: {unknown_blockers}")

        for column in evidence_columns:
            for relative in split_ids(row[column]):
                relative_path = evidence_path(relative)
                path = repository_path(relative_path)
                if repository_mode or is_packaged_path(relative_path):
                    require(path.exists(), f"{rule} has missing {column} path: {relative_path}")

        require(
            row["lean_assurance"] in {"open", "not-applicable", "model-cited", "model-level"},
            f"{rule} has an invalid Lean semantic tier",
        )
        require(
            row["rust_conformance"] in {"open", "not-applicable", "Rust-conformant"},
            f"{rule} has an invalid Rust conformance tier",
        )
        require(
            row["rust_origin_assurance"] in {"open", "not-applicable", "artifact-checked", "Rust-conformant"},
            f"{rule} has an invalid Rust-origin tier",
        )
        require(
            row["circuit_assurance"] in {"open", "not-applicable", "artifact-checked", "Rust-conformant"},
            f"{rule} has an invalid circuit tier",
        )
        require(
            row["reduction_assurance"]
            in {"open", "not-applicable", "model-cited", "model-level", "security-reduced"},
            f"{rule} has an invalid reduction tier",
        )

        if row["rust_conformance"] == "Rust-conformant":
            require(row["rust_origin_evidence"], f"{rule} claims Rust conformance without Rust-origin evidence")
        if row["reduction_assurance"] == "security-reduced":
            open_upstream = [
                row["lean_assurance"],
                row["rust_conformance"],
                row["circuit_assurance"],
            ]
            require("open" not in open_upstream, f"{rule} is security-reduced with an open upstream edge")

    listed_dependencies = set(config["conformance"]["decision_dependencies"])
    unknown_listed = sorted(listed_dependencies - decision_ids)
    require(not unknown_listed, f"configuration uses unknown decision dependencies: {unknown_listed}")
    coverage_blockers = {item for row in rows for item in split_ids(row["blocker_ids"])}
    require(
        listed_dependencies == coverage_blockers,
        "configured decision dependencies differ from the coverage map",
    )
    open_blockers = set(config["conformance"]["open_blockers"])
    require(not (open_blockers - listed_dependencies), "an open blocker is not a decision dependency")
    require(
        isinstance(config["conformance"]["release_blocked"], bool),
        "generated release-blocked flag is not Boolean",
    )
    return set(contract_ids), {row["contract_id"]: row["lean_assurance"] for row in rows}


def declaration_exists(path: Path, name: str) -> bool:
    """Require one top-level declaration with this exact name."""
    keywords = LEAN_KEYWORDS if path.suffix == ".lean" else RUST_KEYWORDS
    escaped = re.escape(name)
    if path.suffix == ".lean":
        pattern = re.compile(rf"^(?:{'|'.join(keywords)})\s+{escaped}(?![\w.'])", re.MULTILINE)
    else:
        pattern = re.compile(
            rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+|async\s+)*"
            rf"(?:{'|'.join(keywords)})\s+{escaped}(?![\w])",
            re.MULTILINE,
        )
    return pattern.search(path.read_text(errors="replace")) is not None


def check_bindings(
    model,
    repository_mode: bool,
) -> tuple[int, list[str], int]:
    """Check exact declarations stored on the canonical evidence edges."""
    declaration_count = 0
    external_anchor_count = 0
    unbound: list[str] = []
    for rule in model.rule_order:
        entries = []
        for target in ("lean", "rust"):
            for anchor in model.evidence[target][rule]["declaration_anchors"]:
                entries.append((target, anchor["path"], anchor["declaration"]))
        if not entries:
            unbound.append(rule)
        for target, relative, name in entries:
            path = repository_path(relative)
            require(
                path.suffix == (".lean" if target == "lean" else ".rs"),
                f"{rule} {target} anchor has the wrong file type: {relative}",
            )
            if repository_mode:
                require(path.is_file(), f"{rule} declaration path does not exist: {relative}")
                require(
                    declaration_exists(path, name),
                    f"{rule} anchor names a declaration that {relative} does not define: {name}",
                )
                declaration_count += 1
            else:
                external_anchor_count += 1

    return declaration_count, unbound, external_anchor_count


def check_package_manifest() -> int:
    require(PACKAGE_MANIFEST_PATH.is_file(), "package manifest is missing")
    expected: dict[str, str] = {}
    for number, line in enumerate(PACKAGE_MANIFEST_PATH.read_text().splitlines(), 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", line)
        require(match is not None, f"malformed package manifest row {number}")
        digest, name = match.groups()
        require(name not in expected, f"duplicate package manifest path: {name}")
        require(not name.startswith("/") and ".." not in Path(name).parts, f"unsafe manifest path: {name}")
        expected[name] = digest

    actual_paths = {
        str(path.relative_to(CONTRACT_DIR))
        for path in CONTRACT_DIR.rglob("*")
        if path.is_file()
        and path != PACKAGE_MANIFEST_PATH
        and not is_ephemeral_contract_path(path)
    }
    require(set(expected) == actual_paths, "package manifest does not cover the contract tree exactly")
    for name, expected_digest in expected.items():
        path = CONTRACT_DIR / name
        require(not path.is_symlink(), f"package manifest path is a symlink: {name}")
        require(sha256(path) == expected_digest, f"package manifest hash mismatch: {name}")
    return len(expected)


def check_global_id_uniqueness(groups: dict[str, set[str]]) -> None:
    owners: dict[str, list[str]] = {}
    for group, ids in groups.items():
        for item in ids:
            owners.setdefault(item, []).append(group)
    collisions = {item: groups for item, groups in owners.items() if len(groups) > 1}
    require(not collisions, f"global semantic ID collisions: {collisions}")
