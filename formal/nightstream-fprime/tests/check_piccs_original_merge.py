#!/usr/bin/env python3
"""Check range assembly and rejection using synthetic values, not prover evidence."""

import copy
import json
from pathlib import Path
import subprocess
import sys

from check_piccs_binary_fold import P, require

PAD_BLOCKS = 4685394
MATRIX_ROWS = 6377559


def main():
    require(len(sys.argv) == 32, "expected executable, public, new directory and 28 Lean rounds")
    executable, public, directory = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    rounds = [Path(path).resolve() for path in sys.argv[4:]]
    point = [json.loads(path.read_bytes())[5] for path in rounds]
    directory.mkdir()
    formal = Path(__file__).resolve().parents[1]
    validator = formal / "scripts" / "validate.sh"
    zero_pad = [[[0, 0] for _ in range(54)] for _ in range(17)]
    zero_matrix = [[[[0, 0] for _ in range(54)] for _ in range(14)] for _ in range(17)]

    def write(name, value):
        path = directory / (name + ".json")
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        return path

    def part(domain, first, finish, values):
        return [1, domain, first, finish, copy.deepcopy(point), copy.deepcopy(values)]

    pads = [part(PAD_BLOCKS, 0, 1, zero_pad), part(PAD_BLOCKS, 1, PAD_BLOCKS, zero_pad)]
    matrices = [part(MATRIX_ROWS, 0, 1, zero_matrix),
                part(MATRIX_ROWS, 1, MATRIX_ROWS, zero_matrix)]
    pads[0][5][0][1] = [P - 1, 7]
    pads[1][5][0][1] = [3, 9]
    pads[0][5][16][53] = [4, P - 2]
    pads[1][5][16][53] = [6, 7]
    matrices[0][5][0][0][1] = [P - 1, 8]
    matrices[1][5][0][0][1] = [4, 11]
    matrices[0][5][16][13][53] = [5, P - 3]
    matrices[1][5][16][13][53] = [12, 10]
    pad_paths = [write(f"pad-{index}", value) for index, value in enumerate(pads)]
    matrix_paths = [write(f"matrix-{index}", value) for index, value in enumerate(matrices)]

    def run(name, selected_pads, selected_matrices, error=None, existing=False):
        output = directory / (name + "-out.json")
        if existing:
            output.write_bytes(b"preserve existing output\n")
        command = ["bash", str(validator), "lean-executable", str(executable),
                   "merge-original", str(public), str(output),
                   *map(str, selected_pads), "--", *map(str, selected_matrices),
                   "--", *map(str, rounds)]
        result = subprocess.run(command, cwd=formal, capture_output=True, timeout=300)
        (directory / (name + ".log")).write_bytes(result.stdout + result.stderr)
        if error is None:
            require(result.returncode == 0, f"{name}: merge failed")
            return json.loads(output.read_bytes())
        require(result.returncode != 0 and error.encode() in result.stdout + result.stderr,
                f"{name}: missing expected rejection: {error}")
        if existing:
            require(output.read_bytes() == b"preserve existing output\n", "existing output changed")
        else:
            require(not output.exists(), f"{name}: rejected merge wrote output")
        return None

    result = run("valid", pad_paths, matrix_paths)
    expected_pad, expected_matrix = copy.deepcopy(zero_pad), copy.deepcopy(zero_matrix)
    for source in range(17):
        for lane in range(54):
            expected_pad[source][lane] = [
                (pads[0][5][source][lane][part] + pads[1][5][source][lane][part]) % P
                for part in range(2)]
        for matrix in range(14):
            for lane in range(54):
                expected_matrix[source][matrix][lane] = [
                    (matrices[0][5][source][matrix][lane][part] +
                     matrices[1][5][source][matrix][lane][part]) % P
                    for part in range(2)]
    require(result == [1, point, expected_pad, expected_matrix],
            "synthetic merge changed a source, port, lane, point or field sum")

    invalid = []
    def reject(name, family, index, change, message):
        values = copy.deepcopy(pads if family == "pad" else matrices)
        change(values[index])
        paths = list(pad_paths if family == "pad" else matrix_paths)
        paths[index] = write(name, values[index])
        run(name, paths if family == "pad" else pad_paths,
            paths if family == "matrix" else matrix_paths, message)
        invalid.append(name)

    reject("gap", "pad", 1, lambda value: value.__setitem__(2, 2), "gap, overlap")
    reject("overlap", "matrix", 1, lambda value: value.__setitem__(2, 0), "gap, overlap")
    reject("wrong-domain", "matrix", 0, lambda value: value.__setitem__(1, MATRIX_ROWS + 1),
           "wrong original evaluation range schema or domain")
    reject("wrong-point", "pad", 0, lambda value: value[4][0].__setitem__(0, (value[4][0][0] + 1) % P),
           "range point differs")
    reject("source-width", "pad", 0, lambda value: value[5].pop(), "expected array length 17")
    reject("matrix-width", "matrix", 0, lambda value: value[5][16].pop(), "expected array length 14")
    reject("noncanonical", "matrix", 0, lambda value: value[5][16][13][53].__setitem__(1, P),
           "noncanonical Goldilocks word")
    run("incomplete", pad_paths, matrix_paths[:1], "ranges are incomplete")
    invalid.append("incomplete")
    run("existing", pad_paths, matrix_paths, "output already exists", existing=True)
    invalid.append("existing")
    print(json.dumps({"event": "original_merge_checks_passed",
                      "synthetic_field_words_checked": 27540, "rejections": invalid,
                      "scope": "assembly and rejection only; synthetic values are not prover evidence"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
