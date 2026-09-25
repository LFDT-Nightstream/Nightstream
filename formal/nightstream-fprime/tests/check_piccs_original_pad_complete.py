#!/usr/bin/env python3
"""Compare the complete independent original-source Pad family only."""

import copy
import json
from pathlib import Path
import runpy
import sys

from check_piccs_original_evaluations import decode
from check_piccs_binary_fold import P, require

compare_round = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]


def compare_pad(actual, proof, phase):
    require(actual[4] == phase[4], "Pad point differs from Rust target")
    for values in (proof[4], phase[12]):
        decode("pad", json.dumps([1, 4685394, 0, 4685394, phase[4], values]).encode())
        require(actual[5] == values, "complete original Pad values differ")


def main():
    require(len(sys.argv) == 6,
            "expected original public, Lean Q27, complete Lean Pad, Rust input, Rust phase")
    public, final_round = [json.loads(Path(path).read_bytes()) for path in sys.argv[1:3]]
    actual = decode("pad", Path(sys.argv[3]).read_bytes())
    require(actual[:4] == [1, 4685394, 0, 4685394], "Pad range is not complete")
    proof, phase = [json.loads(Path(path).read_bytes()) for path in sys.argv[4:6]]
    compare_round(public, final_round, proof, phase, 27)
    compare_pad(actual, proof, phase)
    changed, changed_phase = copy.deepcopy(proof), copy.deepcopy(phase)
    changed[4][16][53][1] = (changed[4][16][53][1] + 1) % P
    changed_phase[12] = copy.deepcopy(changed[4])
    compare_round(public, final_round, changed, changed_phase, 27)
    try:
        compare_pad(actual, changed, changed_phase)
    except ValueError as error:
        require(str(error) == "complete original Pad values differ", "unrelated mutation rejection")
    else:
        raise ValueError("changed final Pad target was accepted")
    print(json.dumps({"event": "piccs_original_pad_complete_comparison_passed",
                      "sources": 17, "K_coefficients": 918, "field_words": 1836,
                      "point_field_words": 56, "changed_target": "rejected",
                      "scope": "complete Pad family only; matrices and full proof encoding remain separate"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError) as error:
        sys.exit(str(error))
