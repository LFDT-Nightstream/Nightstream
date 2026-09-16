#!/usr/bin/env python3
"""Compare complete independent C/R/D and recursive caller bytes."""

import copy
import json
from pathlib import Path
import sys

P = 18446744069414584321


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_bytes(actual, expected, name):
    require(actual == expected, f"{name}: complete bytes differ")


def main():
    require(len(sys.argv) == 7,
            "expected independent children/result/caller then reference children/result/caller")
    raw = [Path(path).read_bytes() for path in sys.argv[1:]]
    names = ("children", "C/R/D result", "recursive caller")
    for index, name in enumerate(names):
        equal_bytes(raw[index], raw[index + 3], name)
    children, result, caller = map(json.loads, raw[:3])
    require(len(children) == 5 and len(result) == 10 and len(caller) == 5,
            "wrong complete result schema")
    require(result[0] == caller[0] == 1 and
            result[5][0] == result[7][0] == result[9][0] == result[9][16][0] == 1,
            "complete C/R/D was not accepted")
    require(result[9][16][1] == children, "complete result has different children")
    require(len(caller[2]) == 177326 and len(caller[3]) == 278,
            "wrong selected caller width")
    require(caller[4][3] == result[5][6] and caller[4][4] == result[5][14] and
            caller[4][5] == result[7][9], "caller point or transcript differs")
    changed = copy.deepcopy(children)
    changed[4][15][13][53][1] = (changed[4][15][13][53][1] + 1) % P
    try:
        equal_bytes(raw[0], json.dumps(changed, separators=(",", ":")).encode() + b"\n",
                    "children")
    except ValueError as error:
        require(str(error) == "children: complete bytes differ", "unrelated target rejection")
    else:
        raise ValueError("changed final child target was accepted")
    print(json.dumps({"event": "independent_nifs_caller_bytes_match",
                      "children_bytes": len(raw[0]), "complete_result_bytes": len(raw[1]),
                      "caller_bytes": len(raw[2]), "caller_private_words": len(caller[2]),
                      "caller_public_words": len(caller[3]), "changed_target": "rejected",
                      "scope": "complete C/R/D and caller bytes; native proof wire is checked "
                               "separately, full Lean fresh-assignment execution remains open"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError) as error:
        sys.exit(str(error))
