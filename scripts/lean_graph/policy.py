"""Validate reviewed expectations. Candidate metadata cannot remove a gate."""

from __future__ import annotations

import hashlib
import hmac
import re
from graphlib import TopologicalSorter, CycleError
from pathlib import Path

from .snapshot import EvidenceError, digest, encoded, read_json, safe_relative


# These ceilings come from the root and F-prime AGENTS.md files.
CAPS = {"lean": 1500, "rust": 300, "python": 300, "static": 300}
STATUSES = ("Compiler-closed", "Conformance-closed", "Production-closed")


def gate_order(policy, selected):
    """Order only the selected gates and their complete prerequisite set."""
    graph, pending = {}, list(selected)
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        if name not in policy["gates"]:
            raise EvidenceError(f"unknown gate: {name}")
        gate = policy["gates"][name]
        graph[name] = list(gate.get("requires", []))
        if gate.get("declaration_freshness"):
            graph[name].append(gate["declaration_freshness"]["gate"])
        pending.extend(graph[name])
    return list(TopologicalSorter(graph).static_order())


def gate_scope(policy, selected):
    sources, inputs = set(), set()
    for name in gate_order(policy, selected):
        gate = policy["gates"][name]
        sources.update(gate["sources"])
        inputs.update(gate.get("inputs", []))
        if gate.get("identity_bound"):
            inputs.update(policy["identity_inputs"])
    return {"sources": sorted(sources), "inputs": sorted(inputs)}


def checker_key():
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + [root.parent / "fprime_stage1_review_manifest.py"]
    paths.append(root / "ExportMetadata.lean")
    lean = root.parents[1] / "formal/nightstream-fprime"
    paths += [lean / "scripts/validate.sh", lean / "scripts/check-boundaries.sh", lean / "tests/AxiomAudit.lean",
              lean / "tests/EvidenceMetadata.lean", lean / "tests/EvidenceAcceptance.lean"]
    return digest({str(path.relative_to(root.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths})


def verify_checker_sources(snapshot):
    """Candidate edits cannot substitute the trusted Lean checker commands."""
    root = Path(__file__).resolve().parents[2]
    for relative in ("scripts/lean_graph/ExportMetadata.lean",
                     "formal/nightstream-fprime/scripts/validate.sh",
                     "formal/nightstream-fprime/scripts/check-boundaries.sh",
                     "formal/nightstream-fprime/tests/AxiomAudit.lean",
                     "formal/nightstream-fprime/tests/EvidenceMetadata.lean",
                     "formal/nightstream-fprime/tests/EvidenceAcceptance.lean"):
        candidate = Path(snapshot) / "source" / relative
        if candidate.exists() and candidate.read_bytes() != (root / relative).read_bytes():
            raise EvidenceError(f"candidate changed the approved checker: {relative}")


def validate(policy):
    if policy.get("schema") != 1:
        raise EvidenceError("unsupported obligation-map schema")
    for collection in ("sources", "inputs", "gates", "obligations", "reviews"):
        if not isinstance(policy.get(collection), dict):
            raise EvidenceError(f"missing obligation-map object: {collection}")
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)
               for name in policy[collection]):
            raise EvidenceError(f"invalid identifier in {collection}")
    for group in policy["sources"].values():
        for path in group["roots"]:
            safe_relative(path)
    if not set(policy["identity_inputs"]) <= policy["inputs"].keys():
        raise EvidenceError("unknown identity input")
    if policy.get("graph_gate") and policy["graph_gate"] not in policy["gates"]:
        raise EvidenceError("unknown declaration graph gate")
    for name, review in policy["reviews"].items():
        if not isinstance(review.get("scope"), str) or not review["scope"].strip():
            raise EvidenceError(f"review needs its exact scope: {name}")
    graph = {}
    for name, gate in policy["gates"].items():
        if not gate["sources"] or not set(gate["sources"]) <= policy["sources"].keys():
            raise EvidenceError(f"invalid source dependencies: {name}")
        if not set(gate.get("inputs", [])) <= policy["inputs"].keys():
            raise EvidenceError(f"unknown gate input: {name}")
        graph[name] = gate.get("requires", [])
        if not set(graph[name]) <= policy["gates"].keys():
            raise EvidenceError(f"unknown required gate: {name}")
        precise = gate.get("declaration_freshness")
        if precise:
            if (precise.get("source") not in gate["sources"] or
                    precise.get("use") not in ("meaning", "proof") or
                    not isinstance(precise.get("roots"), list) or not precise["roots"] or
                    any(not isinstance(root, str) or not root for root in precise["roots"])):
                raise EvidenceError(f"invalid declaration freshness: {name}")
            exporter = policy["gates"].get(precise.get("gate"))
            if (not exporter or exporter.get("declaration_freshness") or
                    precise["source"] not in exporter["sources"]):
                raise EvidenceError(f"invalid freshness exporter: {name}")
            graph[name] = [*graph[name], precise["gate"]]
        if not gate["commands"]:
            raise EvidenceError(f"gate has no executable check: {name}")
        for command in gate["commands"]:
            kind = command["kind"]
            if kind not in CAPS or not command["argv"]:
                raise EvidenceError(f"invalid command: {name}")
            safe_relative(command["cwd"])
            cap = command.get("cap_seconds", CAPS[kind])
            if type(cap) not in (int, float) or not 0 < cap <= CAPS[kind]:
                raise EvidenceError(f"invalid command cap: {name}")
            argv = command["argv"]
            if kind == "lean" and argv[:2] != ["bash", "scripts/validate.sh"]:
                raise EvidenceError("Lean commands must use scripts/validate.sh")
            if kind == "rust" and (argv[0] != "cargo" or "--release" not in argv):
                raise EvidenceError("Rust gates must use Cargo in release mode")
            if not any(command.get("completion", {}).get(key)
                       for key in ("patterns", "tests", "closures", "equal_files")):
                raise EvidenceError(f"command has no completion check: {name}")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise EvidenceError("gate dependency cycle") from error
    for name, obligation in policy["obligations"].items():
        if not obligation.get("owner") or not obligation.get("gap"):
            raise EvidenceError(f"obligation needs its owner reference and closing connection: {name}")
        if obligation["status"] not in STATUSES:
            raise EvidenceError(f"invalid assurance status: {name}")
        if not set(obligation["gates"]) <= policy["gates"].keys():
            raise EvidenceError(f"unknown obligation gate: {name}")
        if not set(obligation["reviews"]) <= policy["reviews"].keys():
            raise EvidenceError(f"unknown required review: {name}")
        if "decomposition" in obligation["reviews"] and not obligation.get("target_required"):
            raise EvidenceError(f"decomposition review needs an exact target criterion: {name}")
    return policy


class Authority:
    """Read a separately provisioned checker policy and authentication key.

    Filesystem separation alone is not isolation. The operator must prevent
    candidate processes from reading this key or changing the approved policy.
    The CLI does not provision that boundary or write review decisions.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.policy = validate(read_json(self.directory / "policy.json"))
        approval = read_json(self.directory / "approval.json")
        if (approval.get("outcome") != "pass" or not approval.get("reviewer")
                or approval.get("policy") != digest(self.policy)
                or approval.get("checker") != checker_key()):
            raise EvidenceError("approved policy or checker does not match")
        self.approval = approval
        self.key = (self.directory / "record.key").read_bytes()
        if not self.key:
            raise EvidenceError("empty checker authentication key")

    def sign(self, record):
        return {"record": record, "authentication": hmac.new(
            self.key, encoded(record), hashlib.sha256).hexdigest()}

    def read(self, path):
        envelope = read_json(path)
        expected = self.sign(envelope.get("record"))["authentication"]
        authentication = envelope.get("authentication")
        if not isinstance(authentication, str) or not hmac.compare_digest(authentication, expected):
            raise EvidenceError(f"untrusted or changed record: {path}")
        return envelope["record"]


def load_policy(path=None, authority=None):
    if authority:
        return authority.policy
    return validate(read_json(path or Path(__file__).with_name("obligations.json")))
