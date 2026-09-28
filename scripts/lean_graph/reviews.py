"""Bind independent decomposition assessments to exact targets and source cuts.

The CLI prepares requests and imports decisions. It does not make a semantic
judgment or sign a review on behalf of the controlled checker.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

from .policy import checker_key
from .snapshot import EvidenceError, digest, read_json, write_json


ASSESSMENTS = ("substantiveness", "premises", "argument", "correspondence", "parent_use")


def text_field(value, name):
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"review needs nonempty {name}")
    return value


def validate_proposal(proposal):
    if not isinstance(proposal, dict):
        raise EvidenceError("the proof proposal must be an object")
    for name in ("author", "statement", "argument", "parent_use"):
        text_field(proposal.get(name), name)
    for name in ("premises", "dependencies"):
        values = proposal.get(name)
        if not isinstance(values, list):
            raise EvidenceError(f"the proof proposal needs a {name} list")
        for value in values:
            text_field(value, name)
    return proposal


def request_path(store, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise EvidenceError("invalid review request identifier")
    return Path(store) / "review-requests" / (identifier + ".json")


def read_request(store, identifier):
    request = read_json(request_path(store, identifier))
    if digest(request) != identifier:
        raise EvidenceError("review request content changed")
    validate_proposal(request["proposal"])
    return request


def binding(request, identifier):
    return {**{name: request[name] for name in
               ("obligation", "target", "scope", "snapshot", "policy", "checker")},
            "review": "decomposition", "request": identifier}


def create_request(obligation, proposal, policy, manifest, store):
    selected = policy["obligations"][obligation]
    if not selected.get("target"):
        raise EvidenceError("register the exact Lean target before requesting decomposition review")
    if "decomposition" not in selected["reviews"]:
        raise EvidenceError("this criterion has no registered decomposition review")
    request = {"schema": 1, "obligation": obligation, "target": selected["target"],
               "owner": selected["owner"], "gap": selected["gap"],
               "scope": policy["reviews"]["decomposition"]["scope"],
               "snapshot": digest(manifest), "policy": digest(policy), "checker": checker_key(),
               "proposal": validate_proposal(proposal)}
    identifier = digest(request)
    path = request_path(store, identifier)
    if path.exists() and read_json(path) != request:
        raise EvidenceError("review request content changed")
    if not path.exists():
        write_json(path, request)
    template = {**binding(request, identifier), "reviewer": "", "reviewed_at": "",
                "outcome": "pending", "assessments": {
                    name: {"outcome": "pending", "reason": ""} for name in ASSESSMENTS}}
    return {"request": identifier, "path": str(path), "content": request,
            "response_template": template,
            "next_action": "Give the request, captured source, and response template to an independent reviewer."}


def reviewed_at(record):
    try:
        result = datetime.datetime.fromisoformat(record["reviewed_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError("review needs an ISO timestamp with an offset") from error
    if result.tzinfo is None:
        raise EvidenceError("review needs an ISO timestamp with an offset")
    return result


def validate_decision(record, request, identifier):
    if not isinstance(record, dict):
        raise EvidenceError("the review decision must be an object")
    if any(record.get(name) != value for name, value in binding(request, identifier).items()):
        raise EvidenceError("review does not match the exact request, target, policy, and snapshot")
    reviewer = text_field(record.get("reviewer"), "reviewer")
    if reviewer.strip().casefold() == request["proposal"]["author"].strip().casefold():
        raise EvidenceError("the proposal author cannot review the same proposal")
    reviewed_at(record)
    assessments = record.get("assessments")
    if not isinstance(assessments, dict) or set(assessments) != set(ASSESSMENTS):
        raise EvidenceError("review must assess every registered decomposition question")
    for name, assessment in assessments.items():
        if not isinstance(assessment, dict) or assessment.get("outcome") not in ("pass", "fail"):
            raise EvidenceError(f"review assessment is unfinished: {name}")
        text_field(assessment.get("reason"), name + " reason")
    outcome = "pass" if all(item["outcome"] == "pass" for item in assessments.values()) else "fail"
    if record.get("outcome") != outcome:
        raise EvidenceError("review outcome disagrees with its assessments")
    return record


def record_review(identifier, response, store, authority=None):
    request = read_request(store, identifier)
    if authority:
        # Only the review process may authenticate a decision. Never sign an
        # untrusted response merely because this importer can access the key.
        record = authority.read(response)
        envelope = read_json(response)
    else:
        envelope = read_json(response)
        record = envelope.get("record", envelope)
        envelope = {"record": record}
    validate_decision(record, request, identifier)
    directory = Path(authority.directory if authority else store) / "reviews"
    path = directory / (digest(record) + ".json")
    write_json(path, envelope)
    return {"record": str(path), "outcome": record["outcome"],
            "checker": "approved" if authority else "diagnostic",
            "accepted_closure": "requires current proof gates and all other reviews"}


def decomposition_results(policy, manifest, store, authority=None):
    selected = {name: {"state": "missing", "accepted": False, "record": None,
                       "request": None, "reviewer": None, "reasons": []}
                for name, item in policy["obligations"].items() if "decomposition" in item["reviews"]}
    current, old = {name: [] for name in selected}, set()
    directory = Path(authority.directory if authority else store) / "reviews"
    for path in sorted(directory.glob("*.json")):
        try:
            envelope = read_json(path) if not authority else None
            record = authority.read(path) if authority else envelope.get("record", {})
            name = record.get("obligation")
            if record.get("review") != "decomposition" or name not in selected:
                continue
            identifier = record.get("request")
            request = read_request(store, identifier)
            validate_decision(record, request, identifier)
            expected = {"snapshot": digest(manifest), "policy": digest(policy), "checker": checker_key(),
                        "target": policy["obligations"][name]["target"],
                        "scope": policy["reviews"]["decomposition"]["scope"]}
            if any(record[key] != value for key, value in expected.items()):
                old.add(name)
                continue
            current[name].append((reviewed_at(record), record, path))
        except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
            for result in selected.values():
                result["reasons"].append(f"Rejected review {path}: {error}")
    for name, values in current.items():
        if not values:
            selected[name]["state"] = "stale" if name in old else "missing"
            continue
        latest = max(item[0] for item in values)
        # A later rejection revokes an earlier approval for the same target
        # and cut. Conflicting decisions at the same timestamp fail closed.
        _, record, path = max((item for item in values if item[0] == latest),
                              key=lambda item: item[1]["outcome"] == "fail")
        selected[name].update(state="passed" if record["outcome"] == "pass" else "failed",
            accepted=bool(authority and record["outcome"] == "pass"), record=str(path),
            request=str(request_path(store, record["request"])), reviewer=record["reviewer"],
            assessments=record["assessments"], checker="approved" if authority else "diagnostic")
    return selected


def require_review(obligation, policy, manifest, store, authority):
    if authority and "decomposition" in policy["obligations"][obligation]["reviews"]:
        result = decomposition_results(policy, manifest, store, authority)[obligation]
        if not result["accepted"]:
            raise EvidenceError("decomposition review is " + result["state"] + ": " + obligation)
