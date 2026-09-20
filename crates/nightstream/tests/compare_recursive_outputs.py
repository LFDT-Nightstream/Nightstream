#!/usr/bin/env python3
"""Compare a fresh staged fold with its recorded complete reference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
GOLDILOCKS_MODULUS = 18446744069414584321
ENVELOPE_FIELDS = {
    "schema", "iteration", "z0", "current", "child_witness_count",
    "running_claims", "running_parent",
}
# These old record fields describe file locations and the evidence scope.
REFERENCE_METADATA = {
    "child_witness_directory", "fresh_claim_file", "fresh_witness_file", "scope",
}


def canonical_field(value):
    # Goldilocks serde stores a u64 representative, which can exceed the modulus.
    # Only its typed {"value": integer} wrapper receives field normalization.
    if set(value) == {"value"} and type(value["value"]) is int:
        if not 0 <= value["value"] < 2**64:
            raise ValueError("Goldilocks serde value is not a u64")
        value["value"] %= GOLDILOCKS_MODULUS
    return value


def load(path):
    with path.open() as source:
        return json.load(source, object_hook=canonical_field)


def equal(actual, expected, label):
    """Require all JSON values and types; report the first different field."""
    if type(actual) is not type(expected):
        raise ValueError(f"{label}: different JSON types")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: different object fields")
        for key in actual:
            equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label}: different array lengths")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label}: different values")


def file_record(actual, reference, scope, mode):
    return {
        "actual": str(actual), "reference": str(reference),
        "actual_bytes": actual.stat().st_size,
        "reference_bytes": reference.stat().st_size,
        "scope": scope, "comparison": mode,
    }


def compare_json(actual, reference, scope):
    left, right = actual.read_bytes(), reference.read_bytes()
    # The staged writer adds a newline; several old writers did not.
    if left.rstrip(b"\r\n") == right.rstrip(b"\r\n"):
        mode = "complete bytes, excluding final CR/LF"
    else:
        equal(json.loads(left, object_hook=canonical_field),
              json.loads(right, object_hook=canonical_field), scope)
        mode = "complete typed JSON, with canonical Goldilocks representatives"
    return file_record(actual, reference, scope, mode)


def compare_envelope(run, fold, reference, parent):
    actual_path = run / f"step-{fold + 1}" / "envelope.json"
    reference_path = reference / "envelope.json"
    actual, expected = load(actual_path), load(reference_path)
    equal(set(actual), ENVELOPE_FIELDS | {"package_identity"}, "new envelope fields")
    equal(set(expected), ENVELOPE_FIELDS | REFERENCE_METADATA, "reference envelope fields")
    equal({key: actual[key] for key in ENVELOPE_FIELDS},
          {key: expected[key] for key in ENVELOPE_FIELDS}, "semantic envelope")
    equal(actual["iteration"], fold + 1, "successor iteration")
    equal(actual["child_witness_count"], 16, "successor child count")
    equal(len(actual["running_claims"]), 16, "successor claim count")
    source = load(run / f"step-{fold}" / "envelope.json")
    equal(actual["package_identity"], source["package_identity"], "source package identity")
    equal(actual["package_identity"], parent["package_identity"], "fold package identity")
    record = file_record(actual_path, reference_path, sorted(ENVELOPE_FIELDS),
                         "complete typed semantic fields; canonical Goldilocks representatives")
    record["reference_metadata_excluded"] = sorted(REFERENCE_METADATA)
    record["new_package_identity_checked_against"] = [
        str(run / f"step-{fold}" / "envelope.json"), str(run / f"fold-{fold}" / "nifs.json"),
    ]
    return record, actual


def compare_later(run, reference, nifs, envelope):
    directory = reference / "nonzero-nifs"
    native_path = directory / "actual_result.json"
    lean_path = directory / "nightstream-native-nonzero-nifs-1.lean.json"
    native, lean = load(native_path), load(lean_path)
    equal(set(native), {"schema", "structural_identifier", "package_identity", "pi_ccs_input",
                        "pi_ccs_phase", "pi_rlc_parent", "children", "outgoing_state", "absorbed"},
          "recorded native result fields")
    equal(native["schema"], 1, "recorded native schema")
    equal(len(lean), 10, "complete Lean result length")
    for label, value in [("schema", lean[0]), ("C acceptance", lean[5][0]),
                         ("R acceptance", lean[7][0]), ("D acceptance", lean[9][0]),
                         ("running output", lean[9][16][0])]:
        equal(value, 1, f"Lean {label}")
    mappings = {
        "package_identity": lean[6][6], "pi_ccs_input": lean[1], "pi_ccs_phase": lean[5],
        "pi_rlc_parent": [lean[7][3], lean[7][4], lean[7][5], lean[7][6], lean[7][7], 1],
        "children": lean[9][16][1], "outgoing_state": lean[7][9], "absorbed": 0,
    }
    for key, expected in mappings.items():
        equal(native[key], expected, f"native/Lean {key}")
    equal(native["outgoing_state"], lean[9][14], "native/Lean D outgoing state")
    equal(native["pi_ccs_input"], load(directory / "pi_ccs_input.json"), "saved C input")
    equal(native["children"], load(directory / "children.json"), "saved children")
    parent = nifs["parent"]
    equal(parent["structural_identifier"], native["structural_identifier"], "structural identifier")
    equal(parent["package_identity"], native["package_identity"], "reference package identity")
    equal([word["value"] for word in parent["transcript_state"]], native["outgoing_state"],
          "generated outgoing transcript")
    equal(parent["transcript_absorbed"], native["absorbed"], "generated transcript cursor")

    proof = run / "fold-2" / "proof.native"
    expected_proof = directory / "proof.bin"
    equal(proof.read_bytes(), expected_proof.read_bytes(), "every later proof byte")
    caller_path = reference / "nonzero-successor" / "nightstream-native-nonzero-step-1.json"
    caller = load(caller_path)
    equal(caller[0], 1, "later caller schema")
    equal(envelope["current"], caller[4][0], "Lean application output")
    fresh = load(run / "step-3" / "fresh-claim.json")
    equal([word["value"] for word in fresh["x"]], caller[4][2], "Lean fresh public input")
    return {
        "proof": file_record(proof, expected_proof, "every canonical proof byte", "exact bytes"),
        "native_result": {"path": str(native_path), "bytes": native_path.stat().st_size},
        "lean_result": {"path": str(lean_path), "bytes": lean_path.stat().st_size},
        "native_lean_fields": list(mappings) + ["D outgoing state"],
        "generated_fields": ["structural_identifier", "package_identity", "transcript_state",
                             "transcript_absorbed", "application output", "fresh public input"],
        "caller": {"path": str(caller_path), "bytes": caller_path.stat().st_size,
                   "compared_fields": ["[0] schema", "[4][0] output", "[4][2] public input"],
                   "private_words": len(caller[2]), "public_words": len(caller[3]),
                   "complete_caller_word_equality_checked": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    run, reference = args.directory.resolve(), args.reference.resolve()
    receipt_path = run / f"comparison-fold-{args.fold}.json"
    if receipt_path.exists():
        parser.error(f"comparison receipt already exists: {receipt_path}")
    material = reference / ("material" if args.fold == 1 else "output/material")
    reference_envelope = reference if args.fold == 1 else reference / "output/envelope"
    files = []
    for child in range(16):
        name = f"digit-{child}.json"
        files.append(compare_json(run / f"fold-{args.fold}" / name, material / name,
                                  f"complete child {child} matrix, including the full carrier"))
    for name in ("fresh-witness.json", "fresh-claim.json"):
        files.append(compare_json(run / f"step-{args.fold + 1}" / name,
                                  reference_envelope / name, f"complete {name}"))
    nifs = load(run / f"fold-{args.fold}" / "nifs.json")
    equal(nifs["parent"], load(run / f"fold-{args.fold}" / "parent.json"), "saved NIFS parent")
    envelope_record, envelope = compare_envelope(run, args.fold, reference_envelope, nifs["parent"])
    files.append(envelope_record)
    later = compare_later(run, reference, nifs, envelope) if args.fold == 2 else None
    producer_sources = []
    for path in sorted((run / "logs").glob("*.json")):
        record = load(path)
        request = record.get("request", {})
        if request.get("step") == args.fold or request.get("phase") == "base":
            producer_sources.append({"record": str(path), "source_commit": record["source_commit"],
                                     "source_changes": record["source_changes"], "outcome": record["outcome"]})
    receipt = {
        "schema": 1, "outcome": "passed", "fold": args.fold,
        "run_directory": str(run), "reference_directory": str(reference),
        "comparison_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "comparison_source_changes": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "command": [sys.executable, str(Path(__file__).resolve()), "--directory", str(run),
                    "--fold", str(args.fold), "--reference", str(reference)],
        "producer_sources": producer_sources, "files": files, "later_fold": later,
        "scope": "Complete child and fresh assignments, fresh claim and semantic envelope equality. "
                 "Fold 2 also checks proof bytes, transcript and applicable Lean result fields. "
                 "No direct complete caller-word comparison or new proof claim.",
    }
    with receipt_path.open("x") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
    print(f"complete recursive output comparison passed: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
