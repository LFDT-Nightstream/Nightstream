#!/usr/bin/env python3
"""Compare the terminal fields provided by complete Lean prefixes.

PiCCSInputCheck.outputMessage selects coefficient zero of every Eval_K and
coefficient zero of source-zero Eval_A for the fresh matrix values. This
check does not cover the other individual evaluation coefficients.
"""

import copy
import json
from pathlib import Path
import runpy
import sys

from check_piccs_binary_fold import FIELD, P, read_prefix, require

compare_round = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]


def values(directory, kind, width, point):
    manifest, chunks = read_prefix(directory)
    require(manifest == [1, 28, kind, width, 1, point, [[0, 1]]],
            f"{directory}: not the complete selected final prefix")
    require(len(chunks) == 1, f"{directory}: expected one final row")
    data = chunks[0].path.read_bytes()
    require(len(data) == width * FIELD.size, f"{directory}: wrong final byte count")
    result = [list(pair) for pair in FIELD.iter_unpack(data)]
    require(all(0 <= word < P for pair in result for word in pair),
            f"{directory}: noncanonical final field word")
    return result


def compare_terminal(norm, fresh, proof):
    require(len(proof[4]) == 17 and len(proof[5]) == 17, "wrong output source count")
    for source in range(17):
        require(len(proof[4][source]) == 54, "wrong Pad coefficient count")
        if norm[source] != proof[4][source][0]:
            raise ValueError(f"sourceAssignment[{source}] differs")
    require(len(proof[5][0]) == 14, "wrong fresh matrix count")
    for matrix in range(14):
        require(len(proof[5][0][matrix]) == 54, "wrong fresh matrix coefficient count")
        if fresh[matrix] != proof[5][0][matrix][0]:
            raise ValueError(f"freshMatrixImage[{matrix}] differs")


def main():
    require(len(sys.argv) == 7,
            "expected original public, Lean Q27, final fresh prefix, final norm prefixes, Rust input, Rust phase")
    public, final_round = [json.loads(Path(path).read_bytes()) for path in sys.argv[1:3]]
    fresh_dir, norm_dir = map(Path, sys.argv[3:5])
    proof, phase = [json.loads(Path(path).read_bytes()) for path in sys.argv[5:7]]
    compare_round(public, final_round, proof, phase, 27)
    point = phase[4]  # Comparison target only; no producer runs in this script.
    fresh = values(fresh_dir, 2, 14, point)
    require({path.name for path in norm_dir.iterdir()} ==
            {f"source-{source}" for source in range(17)}, "missing or extra final norm source")
    norm = [values(norm_dir / f"source-{source}", 3 + source, 1, point)[0]
            for source in range(17)]
    compare_terminal(norm, fresh, proof)
    for name in ("norm", "fresh"):
        changed, changed_phase = copy.deepcopy(proof), copy.deepcopy(phase)
        if name == "norm":
            changed[4][16][0][0] = (changed[4][16][0][0] + 1) % P
            changed_phase[12] = copy.deepcopy(changed[4])
            expected = "sourceAssignment[16] differs"
        else:
            changed[5][0][13][0][1] = (changed[5][0][13][0][1] + 1) % P
            changed_phase[13] = copy.deepcopy(changed[5])
            expected = "freshMatrixImage[13] differs"
        # A mutually consistent proof/phase mutation must still fail the actual
        # field comparison. It must not reject only for mismatched target copies.
        compare_round(public, final_round, changed, changed_phase, 27)
        try:
            compare_terminal(norm, fresh, changed)
        except ValueError as error:
            require(str(error) == expected, f"{name}: unrelated rejection: {error}")
        else:
            raise ValueError(f"changed {name} terminal target was accepted")
    print(json.dumps({"event": "piccs_terminal_prefix_comparison_passed",
                      "norm_sources": 17, "fresh_matrices": 14,
                      "compared_K_values": 31, "compared_field_words": 62,
                      "canonical_field_bytes": 496,
                      "norm_target_mutation": "rejected",
                      "fresh_target_mutation": "rejected",
                      "scope": "coefficient-zero fields only; full individual evaluations remain open"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
