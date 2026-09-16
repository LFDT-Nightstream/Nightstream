#!/usr/bin/env python3
"""Compare complete original-source range bytes with the existing scalar kernels."""

import copy
import hashlib
import json
from pathlib import Path
import sys

from check_piccs_binary_fold import P, require


def decode(kind, data):
    value = json.loads(data)
    require(isinstance(value, list) and len(value) == 6, "wrong range field count")
    version, rows, first, finish, point, sources = value
    require(version == 1 and rows == (6377559 if kind == "matrix" else 4685394), "wrong selected evaluation schema or domain")
    require(isinstance(first, int) and isinstance(finish, int) and 0 <= first < finish <= rows,
            "invalid evaluation range")
    require(len(point) == 28 and len(sources) == 17, "wrong point or source count")
    words = list(point)
    for source in sources:
        if kind == "matrix":
            require(len(source) == 14, "wrong matrix count")
            rings = source
        else:
            rings = [source]
        for lanes in rings:
            require(len(lanes) == 54, "wrong ring coefficient count")
            words.extend(lanes)
    for coefficient in words:
        require(isinstance(coefficient, list) and len(coefficient) == 2 and
                all(type(word) is int and 0 <= word < P for word in coefficient),
                "noncanonical extension coefficient")
    return value


def compare(kind, reference, actual):
    left, right = decode(kind, reference), decode(kind, actual)
    require(left[:5] == right[:5], "range metadata differs")
    require(left[5] == right[5], "evaluation coefficients differ")
    require(reference == actual, "evaluation output bytes differ")
    return left


def main():
    require(len(sys.argv) > 2 and len(sys.argv) % 2 == 0 and
            sys.argv[1] in ("matrix", "pad"), "expected matrix|pad and reference/result path pairs")
    kind = sys.argv[1]
    coefficient_count = 17 * 54 * (14 if kind == "matrix" else 1)
    results = []
    for offset in range(2, len(sys.argv), 2):
        reference_path, actual_path = map(Path, sys.argv[offset:offset + 2])
        reference, actual = reference_path.read_bytes(), actual_path.read_bytes()
        value = compare(kind, reference, actual)
        changed = copy.deepcopy(value)
        coefficient = changed[5][16][13][53] if kind == "matrix" else changed[5][16][53]
        coefficient[1] = (coefficient[1] + 1) % P
        changed_bytes = (json.dumps(changed, separators=(",", ":")) + "\n").encode()
        try:
            compare(kind, reference, changed_bytes)
        except ValueError as error:
            require(str(error) == "evaluation coefficients differ", "unrelated mutant rejection")
        else:
            raise ValueError("changed final evaluation coefficient was accepted")
        results.append({"reference": str(reference_path), "result": str(actual_path),
                        "first": value[2], "end": value[3], "sources": 17,
                        "K_coefficients": coefficient_count, "field_words": coefficient_count * 2,
                        "matched_bytes": len(actual),
                        "sha256": hashlib.sha256(actual).hexdigest(),
                        "changed_coefficient": "rejected"})
    print(json.dumps({"event": "original_evaluation_ranges_match", "family": kind, "ranges": results}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError) as error:
        sys.exit(str(error))
