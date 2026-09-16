#!/usr/bin/env python3
"""Compare independently produced Lean values with the saved Rust round zero.

This command reads comparison targets only. It does not construct prover data.
"""

import json
import sys
from pathlib import Path


def compare(public, lean, proof, trace):
    if not (isinstance(lean, list) and len(lean) == 10 and lean[0] == 1):
        raise ValueError("expected Lean first-round schema 1 with ten fields")
    if not (isinstance(proof, list) and len(proof) == 7 and proof[0] == 2):
        raise ValueError("expected Rust PiCCS input schema 2 with seven fields")
    if not (isinstance(trace, list) and len(trace) == 15 and trace[0] == 1):
        raise ValueError("expected Rust PiCCS phase schema 1 with fifteen fields")
    if public != [proof[1], proof[2], proof[6]]:
        raise ValueError("Rust target uses different original public inputs")
    if trace[12] != proof[4] or trace[13] != proof[5]:
        raise ValueError("Rust proof and phase have different final evaluations")
    if len(lean[4]) != 10 or len(proof[3][0]) != 10:
        raise ValueError("round zero must retain all ten K coefficients")
    fields = [
        ("alpha", lean[1], trace[1]),
        ("gamma", lean[2], trace[2]),
        ("pre-round state", lean[3], trace[3]),
        *[(f"Q[0] coefficient {i}", lean[4][i], proof[3][0][i]) for i in range(10)],
        ("round-zero challenge", lean[5], trace[4][0]),
        ("post-challenge state", lean[6], trace[5][0]),
        ("initial claim", lean[7], trace[7]),
        ("endpoint sum", lean[8], trace[7]),
        ("next claim", lean[9], trace[8][0]),
    ]
    for name, actual, target in fields:
        if actual != target:
            raise ValueError(f"{name} differs: Lean={actual}, Rust={target}")
    return [name for name, _, _ in fields]


def main():
    if len(sys.argv) != 5:
        raise ValueError("usage: compare-piccs-first-round.py <original-public> <lean-round> <rust-input> <rust-phase>")
    values = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
    fields = compare(*values)
    print(json.dumps({"event": "piccs_first_round_comparison_passed", "matched_fields": fields}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
