#!/usr/bin/env python3
"""Check the complete Lean 1→2 output connection to the retained 2→3 input."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "formal/nightstream-fprime"
sys.path[:0] = [str(ROOT / "crates/nightstream/tests"), str(FORMAL / "scripts")]
from check_lean_fold import compare_files
from compare_recursive_outputs import ENVELOPE_FIELDS, REFERENCE_METADATA, equal, load
import project_replay_sources as projection
import replay_recursive_loop as replay


def state_and_parent(first_root, second_root):
    first = first_root / "step-1-to-2"
    second = second_root / "step-2-to-3"
    original = second_root / "original-sources"
    envelope = load(original / "envelope.json")
    caller, second_caller = load(first / "caller.json"), load(second / "caller.json")
    nifs, children = load(first / "nifs-result.json"), load(first / "children.json")
    original_request = load(first_root / "original-sources/next-message-input.json")
    next_request = load(first / "next-message-input.json")
    second_request = load(original / "next-message-input.json")
    equal(set(envelope), ENVELOPE_FIELDS | REFERENCE_METADATA, "complete retained envelope fields")
    equal([envelope["schema"], envelope["child_witness_count"], envelope["iteration"]], [1, 16, 2], "retained envelope shape")
    equal([len(caller), len(nifs), len(children), len(caller[2]), len(caller[3]), len(caller[4])],
          [5, 10, 5, 177326, 278, 7], "selected complete first-fold outputs")
    equal([caller[0], nifs[0], nifs[5][0], nifs[7][0], nifs[9][0], nifs[9][16][0]],
          [1, 1, 1, 1, 1, 1], "first C/R/D acceptance")
    equal(children, nifs[9][16][1], "first complete children")
    equal(caller[1], list(map(int, replay.CONTEXT)), "selected verifier context")
    equal(caller[1], second_caller[1], "first/second verifier context")
    equal(nifs[6][6], list(map(int, replay.PACKAGE)), "selected package identity")
    private, result = caller[2], caller[4]
    equal([private[28], private[30:34], private[35:39], private[-4:]], original_request,
          "first requested original state and message")
    equal(original_request[0], 1, "first iteration")
    offset = 49393
    equal(private[offset + 35:offset + 39], result[0], "first returned application state")
    derived = [private[offset + 28], private[offset + 30:offset + 34], result[0], original_request[3]]
    equal(derived, next_request, "first caller-derived next request")
    equal(next_request, second_request, "complete retained second request")
    equal([envelope["iteration"], envelope["z0"], envelope["current"]], next_request[:3], "complete successor state")
    equal(caller[4][3], nifs[5][6], "caller/C point")
    equal(caller[4][3], nifs[7][5], "caller/R point")
    equal(caller[4][3], children[0], "caller/children point")
    equal(caller[4][4], nifs[5][14], "caller/C outgoing state")
    equal(caller[4][5], nifs[7][9], "caller/R outgoing state")
    equal(caller[4][5], nifs[9][14], "caller/D outgoing state")
    equal(caller[4][6], nifs[7][4], "caller/R public values")
    parent = envelope["running_parent"]
    claim_fields = {"X", "adv", "c", "eval_a", "eval_k", "fold_digest", "m_in", "r"}
    equal(set(parent), claim_fields, "complete parent fields")
    equal([parent["m_in"], parent["adv"]], [270, None], "parent public shape")
    parent_values = [projection.commitment(parent["c"]), projection.public_matrix(parent["X"]),
                     projection.vector(parent["r"], 28, projection.extension), projection.padded_evaluation(parent["eval_k"]),
                     projection.vector(parent["eval_a"], 14, projection.padded_evaluation), 1]
    equal(parent_values, [nifs[7][3], nifs[7][4], nifs[7][5], nifs[7][6], nifs[7][7], 1], "complete carried R parent")
    words = projection.vector(result[1], 4, projection.field)
    frame = list(b"".join(word.to_bytes(8, "little") for word in words))
    equal(len(envelope["running_claims"]), 16, "all carried children")
    for index, claim in enumerate([parent, *envelope["running_claims"]]):
        equal(set(claim), claim_fields, f"carried claim {index} fields")
        equal([claim["m_in"], claim["adv"]], [270, None], f"carried claim {index} public shape")
        equal(claim["fold_digest"], frame, f"carried claim {index} successor frame")
    public = load(second / "sources/public.json")
    ccs = load(second / "ccs-input.json")
    equal([ccs[1], ccs[2], ccs[6]], public, "exact public input consumed by retained C")
    equal(projection.fresh_claim(original / "fresh-claim.json")[1], result[2], "successor fresh public input")
    return {"state_words": 4, "message_words": 4, "parent_fields": "commitment, public matrix, point, K/A evaluations, output flag",
            "carried_frames": 17, "frame_bytes_each": 32, "frame_source": "first caller successor state digest",
            "context": "first and completed second caller equal the selected verifier context",
            "retained_C_public_fields": [1, 2, 6], "reference_metadata_excluded": sorted(REFERENCE_METADATA)}


def check(first_root, second_root, directory):
    first, second = first_root / "step-1-to-2", second_root / "step-2-to-3"
    comparisons = []
    for mode, retained in (("first-original", first), ("feedback", second), ("original", second)):
        for name in ("public.json", "sources.jsonl"):
            actual, expected = directory / mode / name, retained / "sources" / name
            compare_files(actual, expected, "complete first-successor/retained-input projection")
            comparisons.append({"actual": str(actual), "retained": str(expected),
                                "exact_bytes": actual.stat().st_size})
    for package in (second_root / "original-package.json",
                    FORMAL / "artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"):
        compare_files(first_root / "original-package.json", package, "complete selected package")
    return {"outcome": "passed", "iterations": [1, 2, 3], "sources": 17,
            "blocks_per_source": projection.BLOCKS, "projected_bytes": comparisons,
            "state_and_parent": state_and_parent(first_root, second_root),
            "scope": "Complete first Lean successor witnesses and public values equal the original inputs "
                     "and source projection consumed by the second fold. Parent, state digest frames, "
                     "context, package, message and retained C public input are checked separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    result = directory / "result.json"
    if result.exists():
        raise ValueError("input-connection result already exists")
    record = check(args.first_root.resolve(strict=True), args.second_root.resolve(strict=True), directory)
    replay.write_new(result, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
