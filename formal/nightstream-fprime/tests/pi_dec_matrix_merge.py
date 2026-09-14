"""Synthetic codec/addition checks. These are not matrix producer conformance."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import time


# Existing selected package sizes: Poseidon2HashChainV1Setup.messageColumns_eq,
# PerApplicationCanonicalPackage row count, and the fixed protocol dimensions.
BLOCKS = 4_685_394
ROWS = 6_377_559
POINT_COORDINATES = 28
CHILDREN = 16
MATRICES = 14
LANES = 54
MODULUS = 2**64 - 2**32 + 1
SCOPE = "synthetic_codec_addition"


def array(value, count, where):
    assert isinstance(value, list) and len(value) == count, \
        f"{where}: expected {count} entries"
    return value


def field(value, where):
    assert type(value) is int and 0 <= value < MODULUS, \
        f"{where}: noncanonical field word"


def point_shape(point, where):
    for coordinate, pair in enumerate(array(point, POINT_COORDINATES, where)):
        for component, word in enumerate(array(pair, 2, f"{where}[{coordinate}]")):
            field(word, f"{where}[{coordinate}][{component}]")


def ring_shape(ring, where):
    for lane, pair in enumerate(array(ring, LANES, where)):
        for component, word in enumerate(array(pair, 2, f"{where}[{lane}]")):
            field(word, f"{where}[{lane}][{component}]")


def complete_pad(path):
    pad = json.loads(path.read_text())
    array(pad, 6, "supplied Pad")
    assert pad[:4] == [1, BLOCKS, 0, BLOCKS], "expected complete selected Lean Pad"
    point_shape(pad[4], "supplied Pad point")
    for child, ring in enumerate(array(pad[5], CHILDREN, "supplied Pad children")):
        ring_shape(ring, f"supplied Pad child {child}")
    return pad


def synthetic_values(part):
    values = []
    for child in range(CHILDREN):
        matrices = []
        for matrix in range(MATRICES):
            ring = []
            for lane in range(LANES):
                base = ((child * MATRICES + matrix) * LANES + lane) * 2
                ring.append([(part + 1) * (base + component) + part + 1
                             for component in range(2)])
            matrices.append(ring)
        values.append(matrices)
    # Check both exact wrap to zero and a sum strictly beyond the modulus.
    values[-1][-1][-1] = [MODULUS - 1, MODULUS - 2] if part == 0 else [1, 6]
    return values


def check_sum(output, pad, ranges):
    merged = json.loads(output.read_text())
    array(merged, 5, "merged family")
    assert merged[:2] == [1, BLOCKS], "wrong complete family header"
    point_shape(merged[2], "merged point")
    assert merged[2] == pad[4], "the merger changed the complete Pad point"
    for child, ring in enumerate(array(merged[3], CHILDREN, "merged Pad")):
        ring_shape(ring, f"merged Pad child {child}")
    assert merged[3] == pad[5], "the merger changed a Pad coefficient"
    for child, matrices in enumerate(array(merged[4], CHILDREN, "merged children")):
        for matrix, ring in enumerate(array(matrices, MATRICES, f"child {child} matrices")):
            ring_shape(ring, f"child {child} matrix {matrix}")
            for lane in range(LANES):
                for component in range(2):
                    expected = sum(part[5][child][matrix][lane][component]
                                   for part in ranges) % MODULUS
                    actual = ring[lane][component]
                    assert actual == expected, \
                        (f"sum mismatch: child={child} matrix={matrix} lane={lane} "
                         f"component={component} expected={expected} actual={actual}")
    assert merged[4][-1][-1][-1] == [0, 4], "modulus-wrap fixture failed"


def write_json(path, value):
    with path.open("x") as output:
        json.dump(value, output, separators=(",", ":"))
        output.write("\n")


def check_case(binary, ccs, directory, name, pad, ranges, deadline, error=None):
    pad_path = directory / f"{name}.pad.json"
    write_json(pad_path, pad)
    parts = []
    for index, value in enumerate(ranges):
        path = directory / f"{name}.{index}.part.json"
        write_json(path, value)
        parts.append(str(path))
    output = directory / f"{name}.output.json"
    command = [str(binary), str(ccs), str(pad_path), str(output), *parts]
    # AGENTS.md caps the complete Python test at 300 seconds. The caller must
    # also apply that outer cap. Every child uses only the remaining allowance.
    remaining = deadline - time.monotonic()
    assert remaining > 0, "the 300-second test cap was reached"
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=min(300, remaining))
    text = result.stdout + result.stderr
    with (directory / f"{name}.log").open("x") as log:
        log.write(text)
    if error is None:
        assert result.returncode == 0, f"{name}: valid merge failed\n{text}"
        assert output.is_file(), f"{name}: missing complete output"
        check_sum(output, pad, ranges)
        disposition = "accepted"
    else:
        assert result.returncode != 0, f"{name}: malformed input accepted"
        assert error in text, f"{name}: wrong rejection\n{text}"
        assert not output.exists(), f"{name}: output written before rejection"
        disposition = "rejected"
    print(f"pidec_matrix_merge_case={name} {disposition} scope={SCOPE}", flush=True)


def main():
    deadline = time.monotonic() + 300
    if len(sys.argv) != 5:
        raise SystemExit("usage: pi_dec_matrix_merge.py <merge-binary> <valid-C-input> "
                         "<complete-Lean-Pad> <new-directory>")
    binary = Path(sys.argv[1]).resolve(strict=True)
    ccs = Path(sys.argv[2]).resolve(strict=True)
    pad = complete_pad(Path(sys.argv[3]).resolve(strict=True))
    directory = Path(sys.argv[4]).resolve()
    directory.mkdir()
    # Synthetic partial values exercise only decoding, coverage and addition.
    # They do not purport to be evaluations of the two represented row ranges.
    valid = [[1, ROWS, 0, 1, copy.deepcopy(pad[4]), synthetic_values(0)],
             [1, ROWS, 1, ROWS, copy.deepcopy(pad[4]), synthetic_values(1)]]
    check_case(binary, ccs, directory, "two_part_all_words", pad, valid, deadline)

    coverage_error = "matrix ranges do not cover the complete selected row domain"
    endpoint_error = "matrix ranges have a gap, overlap or invalid endpoint"
    check_case(binary, ccs, directory, "missing_all", pad, [], deadline, coverage_error)
    check_case(binary, ccs, directory, "missing_tail", pad, valid[:1], deadline, coverage_error)

    for name, first in [("gap", 2), ("overlap", 0)]:
        changed = copy.deepcopy(valid)
        changed[1][2] = first
        check_case(binary, ccs, directory, name, pad, changed, deadline, endpoint_error)

    changed = copy.deepcopy(valid)
    changed[1][1] -= 1
    check_case(binary, ccs, directory, "wrong_selected_rows", pad, changed, deadline,
               "expected a selected Lean matrix range")

    changed = copy.deepcopy(valid)
    changed[1][4][-1][1] = (changed[1][4][-1][1] + 1) % MODULUS
    check_case(binary, ccs, directory, "wrong_matrix_point", pad, changed, deadline,
               "matrix range point differs from accepted C execution")

    changed_pad = copy.deepcopy(pad)
    changed_pad[4][-1][1] = (changed_pad[4][-1][1] + 1) % MODULUS
    check_case(binary, ccs, directory, "wrong_pad_point", changed_pad, valid, deadline,
               "Pad point differs from accepted C execution")

    canonical_error = "noncanonical matrix field coefficient"
    for name, path in [("noncanonical_matrix_word", [5, -1, -1, -1, 1]),
                       ("noncanonical_matrix_point", [4, -1, 1])]:
        changed = copy.deepcopy(valid)
        target = changed[1]
        for index in path[:-1]:
            target = target[index]
        target[path[-1]] = MODULUS
        check_case(binary, ccs, directory, name, pad, changed, deadline, canonical_error)

    for name, path in [("noncanonical_pad_word", [5, -1, -1, 1]),
                       ("noncanonical_pad_point", [4, -1, 1])]:
        changed_pad = copy.deepcopy(pad)
        target = changed_pad
        for index in path[:-1]:
            target = target[index]
        target[path[-1]] = MODULUS
        check_case(binary, ccs, directory, name, changed_pad, valid, deadline, canonical_error)

    for name, path, count in [
            ("matrix_child_count", [5], CHILDREN),
            ("matrix_count", [5, -1], MATRICES),
            ("matrix_lane_count", [5, -1, -1], LANES),
            ("matrix_pair_length", [5, -1, -1, -1], 2),
            ("matrix_point_count", [4], POINT_COORDINATES),
            ("matrix_point_pair_length", [4, -1], 2)]:
        changed = copy.deepcopy(valid)
        target = changed[1]
        for index in path:
            target = target[index]
        target.pop()
        check_case(binary, ccs, directory, name, pad, changed, deadline,
                   f"expected {count} entries")

    for name, path, count in [
            ("pad_child_count", [5], CHILDREN),
            ("pad_lane_count", [5, -1], LANES),
            ("pad_pair_length", [5, -1, -1], 2),
            ("pad_point_count", [4], POINT_COORDINATES),
            ("pad_point_pair_length", [4, -1], 2)]:
        changed_pad = copy.deepcopy(pad)
        target = changed_pad
        for index in path:
            target = target[index]
        target.pop()
        check_case(binary, ccs, directory, name, changed_pad, valid, deadline,
                   f"expected {count} entries")

    print(f"pidec_matrix_merge_checks=passed scope={SCOPE} "
          f"matrix_words={CHILDREN * MATRICES * LANES * 2} "
          f"pad_words={CHILDREN * LANES * 2} "
          f"point_words={POINT_COORDINATES * 2} producer_conformance=not_tested", flush=True)


if __name__ == "__main__":
    main()
