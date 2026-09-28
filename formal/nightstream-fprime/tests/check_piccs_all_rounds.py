#!/usr/bin/env python3
"""Compare all saved independent Lean rounds and reject changed Rust targets.

Generation provenance and final individual evaluations are separate evidence.
"""

import copy
import json
from pathlib import Path
import runpy
import sys

from check_piccs_prefix_fold import extension, read_round, require, sequence

compare = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]


def main():
    require(len(sys.argv) == 5, "expected original public, complete Lean rounds, Rust input and Rust phase")
    public = json.loads(Path(sys.argv[1]).read_bytes())
    directory = Path(sys.argv[2])
    proof, phase = [json.loads(Path(path).read_bytes()) for path in sys.argv[3:5]]
    names = {f"round-{index}.json" for index in range(28)}
    require({path.name for path in directory.iterdir()} == names,
            "the Lean directory must contain exactly all 28 round files")
    sequence(proof, 7, "Rust input")
    for row in sequence(proof[3], 28, "Rust round messages"):
        for value in sequence(row, 10, "Rust round coefficients"):
            extension(value, "Rust round coefficient")
    previous = None
    for index in range(28):
        path = directory / f"round-{index}.json"
        require(path.is_file(), f"{path}: not a round file")
        lean = read_round(path)
        if previous is not None:
            require(previous[1:3] == lean[1:3] and previous[6] == lean[3] and previous[9] == lean[7],
                    f"Lean round {index} does not continue the prior saved round")
        compare(public, lean, proof, phase, index)
        changed = copy.deepcopy(proof)
        changed[3][index][9][0] = (changed[3][index][9][0] + 1) % 18446744069414584321
        try:
            compare(public, lean, changed, phase, index)
        except ValueError as error:
            require(str(error).startswith(f"Q[{index}] coefficient 9 differs:"),
                    f"round {index}: target mutation failed for an unrelated reason")
        else:
            raise ValueError(f"round {index}: changed target was accepted")
        previous = lean
    print(json.dumps({"event": "piccs_all_rounds_comparison_passed", "rounds": 28,
                      "matched_K_coefficients": 280, "matched_field_words": 560,
                      "matched_transcript_transitions": 28, "changed_targets_rejected": 28,
                      "scope": "complete round messages and causal saved traces; final evaluations remain separate"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
        sys.exit(str(error))
