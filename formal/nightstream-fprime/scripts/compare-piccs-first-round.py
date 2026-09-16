#!/usr/bin/env python3
"""Compare independently produced Lean values with the corresponding saved Rust round.

This command reads comparison targets only. It does not construct prover data.
"""

import json
import sys
from pathlib import Path


def compare(public, lean, proof, trace, round_index=0):
    if not 0 <= round_index < 28:
        raise ValueError("round index must be within the selected 28-round protocol")
    if not (isinstance(lean, list) and len(lean) == 10 and lean[0] == 1):
        raise ValueError("expected Lean round schema 1 with ten fields")
    if not (isinstance(proof, list) and len(proof) == 7 and proof[0] == 2):
        raise ValueError("expected Rust PiCCS input schema 2 with seven fields")
    if not (isinstance(trace, list) and len(trace) == 15 and trace[0] == 1):
        raise ValueError("expected Rust PiCCS phase schema 1 with fifteen fields")
    if public != [proof[1], proof[2], proof[6]]:
        raise ValueError("Rust target uses different original public inputs")
    if trace[12] != proof[4] or trace[13] != proof[5]:
        raise ValueError("Rust proof and phase have different final evaluations")
    if len(lean[4]) != 10 or len(proof[3][round_index]) != 10:
        raise ValueError("each round must retain all ten K coefficients")
    initial = trace[7] if round_index == 0 else trace[8][round_index - 1]
    before = trace[3] if round_index == 0 else trace[5][round_index - 1]
    fields = [
        ("alpha", lean[1], trace[1]),
        ("gamma", lean[2], trace[2]),
        ("pre-round state", lean[3], before),
        *[(f"Q[{round_index}] coefficient {i}", lean[4][i], proof[3][round_index][i]) for i in range(10)],
        ("round challenge", lean[5], trace[4][round_index]),
        ("post-challenge state", lean[6], trace[5][round_index]),
        ("initial claim", lean[7], initial),
        ("endpoint sum", lean[8], initial),
        ("next claim", lean[9], trace[8][round_index]),
    ]
    for name, actual, target in fields:
        if actual != target:
            raise ValueError(f"{name} differs: Lean={actual}, Rust={target}")
    return [name for name, _, _ in fields]


def main():
    if len(sys.argv) not in (5, 6):
        raise ValueError("usage: compare-piccs-first-round.py <original-public> <lean-round> <rust-input> <rust-phase> [round-index]")
    values = [json.loads(Path(path).read_text()) for path in sys.argv[1:5]]
    round_index = int(sys.argv[5]) if len(sys.argv) == 6 else 0
    fields = compare(*values, round_index)
    event = "piccs_first_round_comparison_passed" if round_index == 0 else "piccs_round_comparison_passed"
    print(json.dumps({"event": event, "round": round_index, "matched_fields": fields}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
