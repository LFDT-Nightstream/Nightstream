#!/usr/bin/env python3
"""Compare all individual Lean PiCCS evaluations with Rust comparison targets.

This covers the point and every Pad/matrix coefficient for all 17 sources,
including the nonconstant coefficients of fresh source zero. Full proof
encoding and the post-output transcript are outside this check.
"""

import copy
import json
from pathlib import Path
import runpy
import sys

from check_piccs_binary_fold import P, require

compare_round = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]


def array(value, length, name):
    require(type(value) is list and len(value) == length,
            f"{name}: expected array length {length}")


def tensor(value, dimensions, name):
    if not dimensions:
        require(type(value) is int and 0 <= value < P,
                f"{name}: noncanonical field word")
        return
    array(value, dimensions[0], name)
    for index, entry in enumerate(value):
        tensor(entry, dimensions[1:], f"{name}[{index}]")


def schema(value, length, version, name):
    array(value, length, name)
    require(type(value[0]) is int and value[0] == version,
            f"{name}: wrong schema")


def validate(merged, proof, phase):
    schema(merged, 4, 1, "Lean evaluations")
    tensor(merged[1], (28, 2), "Lean point")
    tensor(merged[2], (17, 54, 2), "Lean Pad")
    tensor(merged[3], (17, 14, 54, 2), "Lean matrix")
    schema(proof, 7, 2, "Rust PiCCS input")
    schema(phase, 15, 1, "Rust PiCCS phase")
    tensor(phase[4], (28, 2), "Rust point")
    tensor(proof[4], (17, 54, 2), "Rust proof Pad")
    tensor(phase[12], (17, 54, 2), "Rust phase Pad")
    tensor(proof[5], (17, 14, 54, 2), "Rust proof matrix")
    tensor(phase[13], (17, 14, 54, 2), "Rust phase matrix")


def compare_evaluations(merged, proof, phase):
    validate(merged, proof, phase)
    require(merged[1] == phase[4], "complete evaluation point differs")
    require(proof[4] == phase[12], "Rust proof and phase Pad targets differ")
    require(proof[5] == phase[13], "Rust proof and phase matrix targets differ")
    for source in range(17):
        for lane in range(54):
            for component in range(2):
                if merged[2][source][lane][component] != proof[4][source][lane][component]:
                    raise ValueError(f"Pad[{source}][{lane}][{component}] differs")
        for port in range(14):
            for lane in range(54):
                for component in range(2):
                    if merged[3][source][port][lane][component] != proof[5][source][port][lane][component]:
                        raise ValueError(f"matrix[{source}][{port}][{lane}][{component}] differs")


def main():
    require(len(sys.argv) == 6,
            "expected original public, Lean Q27, merged evaluations, Rust input, Rust phase")
    public, final_round, merged, proof, phase = [
        json.loads(Path(path).read_bytes()) for path in sys.argv[1:]
    ]
    compare_round(public, final_round, proof, phase, 27)
    compare_evaluations(merged, proof, phase)
    for family in ("pad", "matrix"):
        changed, changed_phase = copy.deepcopy(proof), copy.deepcopy(phase)
        if family == "pad":
            coefficient = changed[4][16][53]
            coefficient[1] = (coefficient[1] + 1) % P
            changed_phase[12] = copy.deepcopy(changed[4])
            expected = "Pad[16][53][1] differs"
        else:
            coefficient = changed[5][16][13][53]
            coefficient[1] = (coefficient[1] + 1) % P
            changed_phase[13] = copy.deepcopy(changed[5])
            expected = "matrix[16][13][53][1] differs"
        # Both Rust target copies change consistently. The round comparison
        # must still pass, and only the actual field comparison must reject.
        compare_round(public, final_round, changed, changed_phase, 27)
        try:
            compare_evaluations(merged, changed, changed_phase)
        except ValueError as error:
            require(str(error) == expected, f"{family}: unrelated rejection: {error}")
        else:
            raise ValueError(f"changed final {family} target was accepted")
    print(json.dumps({
        "event": "piccs_original_complete_comparison_passed",
        "sources": 17,
        "Pad_K_values": 17 * 54,
        "matrix_K_values": 17 * 14 * 54,
        "compared_K_values": 13770,
        "compared_field_words": 27540,
        "point_K_values": 28,
        "fresh_source_nonconstant_K_values": (54 - 1) * (1 + 14),
        "pad_target_mutation": "rejected",
        "matrix_target_mutation": "rejected",
        "scope": "complete individual evaluation families; proof encoding and post-output transcript excluded",
    }))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
