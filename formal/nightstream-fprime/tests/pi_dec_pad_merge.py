"""Check Pad merge decoding and coverage; no production evaluation is asserted."""

import copy
import json
from pathlib import Path
import subprocess
import sys


# Selected Poseidon2HashChainV1Setup message columns and protocol dimensions.
BLOCKS = 4_685_394
POINT_COORDINATES = 28
CHILDREN = 16
LANES = 54
MODULUS = 2**64 - 2**32 + 1


def zero_range(start, end):
    point = [[index, index + 1] for index in range(POINT_COORDINATES)]
    values = [[[0, 0] for _ in range(LANES)] for _ in range(CHILDREN)]
    return [1, BLOCKS, start, end, point, values]


def check_case(binary, directory, name, ranges, error=None):
    paths = []
    for index, value in enumerate(ranges):
        path = directory / f"{name}.{index}.input.json"
        with path.open("x") as source:
            json.dump(value, source, separators=(",", ":"))
            source.write("\n")
        paths.append(str(path))
    output = directory / f"{name}.output.json"
    command = [str(binary), "merge-pad", str(output), *paths]
    # The caller must also cap this complete Python test at 300 seconds.
    # The requested 300-second test cap also applies to each child process.
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    text = result.stdout + result.stderr
    with (directory / f"{name}.log").open("x") as log:
        log.write(text)
    if error is None:
        assert result.returncode == 0, f"{name}: valid merge failed\n{text}"
        assert output.is_file(), f"{name}: no complete output"
        assert json.loads(output.read_text()) == zero_range(0, BLOCKS), \
            f"{name}: incorrect complete zero sum, point or coverage"
        print(f"pidec_pad_merge_boundary={name} accepted", flush=True)
    else:
        assert result.returncode != 0, f"{name}: malformed ranges accepted"
        assert error in text, f"{name}: wrong rejection\n{text}"
        assert not output.exists(), f"{name}: output created before rejection"
        print(f"pidec_pad_merge_boundary={name} rejected", flush=True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: pi_dec_pad_merge.py <replayPiDECEvaluation-binary> <new-results-directory>")
    binary = Path(sys.argv[1]).resolve(strict=True)
    directory = Path(sys.argv[2]).resolve()
    directory.mkdir()
    # Two nonempty ranges cover the exact selected carrier. Only zero sums
    # are supplied: this exercises decoding and merge boundaries cheaply.
    valid = [zero_range(0, 1), zero_range(1, BLOCKS)]
    check_case(binary, directory, "valid_zero_two_ranges", valid)

    coverage_error = "Pad ranges do not cover the complete carrier"
    endpoint_error = "Pad ranges have a gap, overlap or invalid endpoint"
    check_case(binary, directory, "missing_all", [], coverage_error)
    check_case(binary, directory, "missing_tail", [valid[0]], coverage_error)

    gap = copy.deepcopy(valid)
    gap[1][2] = 2
    check_case(binary, directory, "gap", gap, endpoint_error)

    overlap = copy.deepcopy(valid)
    overlap[1][2] = 0
    check_case(binary, directory, "overlap", overlap, endpoint_error)

    point = copy.deepcopy(valid)
    point[1][4][-1][1] += 1
    check_case(binary, directory, "last_point_component", point, "Pad ranges have different points")

    coefficient = copy.deepcopy(valid)
    coefficient[1][5][-1][-1][1] = MODULUS
    check_case(binary, directory, "noncanonical_pad_limb", coefficient,
               "noncanonical Pad field coefficient")

    point_coefficient = copy.deepcopy(valid)
    point_coefficient[1][4][-1][1] = MODULUS
    check_case(binary, directory, "noncanonical_point_limb", point_coefficient,
               "noncanonical Pad field coefficient")

    children = copy.deepcopy(valid)
    children[1][5].pop()
    check_case(binary, directory, "child_count", children, "expected 16 entries")

    lanes = copy.deepcopy(valid)
    lanes[1][5][-1].pop()
    check_case(binary, directory, "lane_count", lanes, "expected 54 entries")

    pair = copy.deepcopy(valid)
    pair[1][5][-1][-1].pop()
    check_case(binary, directory, "pad_pair_length", pair, "expected 2 entries")

    point_shape = copy.deepcopy(valid)
    point_shape[1][4].pop()
    check_case(binary, directory, "point_count", point_shape, "expected 28 entries")

    point_pair = copy.deepcopy(valid)
    point_pair[1][4][-1].pop()
    check_case(binary, directory, "point_pair_length", point_pair, "expected 2 entries")

    block_count = copy.deepcopy(valid)
    block_count[1][1] -= 1
    check_case(binary, directory, "selected_block_count", block_count,
               "expected a selected Lean Pad range")
    print("pidec_pad_merge_boundaries=passed scope=decoder_and_coverage", flush=True)


if __name__ == "__main__":
    main()
