"""Check parent range decoding and zero-input arithmetic, not a producer trace."""

import json
from pathlib import Path
import subprocess
import sys


# The selected setup and field; these are existing production dimensions.
BLOCKS = 4_685_394
MODULUS = 2**64 - 2**32 + 1
LANES = 54


def stream(first=0, last=BLOCKS, records=(), terminator=True, extra=""):
    lines = [json.dumps([1, BLOCKS, first, last])]
    lines.extend(json.dumps(record) for record in records)
    if terminator:
        lines.append("[]")
    return "\n".join(lines) + "\n" + extra


def check_case(binary, ccs, directory, name, ranges, error=None, first=0, last=94):
    paths = []
    for index, text in enumerate(ranges):
        path = directory / f"{name}.{index}.jsonl"
        with path.open("x") as output:
            output.write(text)
        paths.append(str(path))
    output = directory / f"{name}.output.json"
    result = subprocess.run(
        [str(binary), str(ccs), str(output), "0", str(first), str(last), *paths],
        capture_output=True, text=True, timeout=300,
    )
    text = result.stdout + result.stderr
    with (directory / f"{name}.log").open("x") as log:
        log.write(text)
    if error is not None:
        assert result.returncode != 0, f"{name}: invalid parent accepted"
        assert error in text, f"{name}: wrong rejection\n{text}"
        assert not output.exists(), f"{name}: result written before rejection"
        print(f"pidec_parent_boundary={name} rejected", flush=True)
        return
    assert result.returncode == 0, f"{name}: valid zero parent rejected\n{text}"
    values = json.loads(output.read_text())
    assert values[:4] == [1, 6_377_559, 0, 94], "selected first invocation"
    assert len(values[4]) == 28, "complete C-derived point"
    assert len(values[5]) == 16, "all children"
    for matrices in values[5]:
        assert len(matrices) == 14, "all matrices"
        for lanes in matrices:
            assert lanes == [[0, 0] for _ in range(LANES)], "zero matrix action"
    print(f"pidec_parent_boundary={name} accepted", flush=True)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: pi_dec_parent_input.py <replayPiDECMatrix> <valid-C-input> <new-results-directory>")
    binary = Path(sys.argv[1]).resolve(strict=True)
    ccs = Path(sys.argv[2]).resolve(strict=True)
    directory = Path(sys.argv[3]).resolve()
    directory.mkdir()
    zero = [0] * LANES
    check_case(binary, ccs, directory, "complete_zero_ranges",
               [stream(0, 1), stream(1, BLOCKS)])
    check_case(binary, ccs, directory, "missing_tail", [stream(0, BLOCKS - 1)],
               "parent ranges do not cover the complete carrier")
    check_case(binary, ccs, directory, "gap", [stream(1, BLOCKS)],
               "parent ranges have a gap, overlap or wrong length")
    check_case(binary, ccs, directory, "overlap", [stream(0, 1), stream(0, BLOCKS)],
               "parent ranges have a gap, overlap or wrong length")
    check_case(binary, ccs, directory, "duplicate", [stream(records=[(0, zero), (0, zero)])],
               "duplicate or out-of-range parent block")
    check_case(binary, ccs, directory, "past_tail", [stream(records=[(BLOCKS, zero)])],
               "duplicate or out-of-range parent block")
    check_case(binary, ccs, directory, "noncanonical", [stream(records=[(0, [MODULUS] + zero[1:])])],
               "noncanonical parent coefficient")
    check_case(binary, ccs, directory, "strict_bound", [stream(records=[(0, [2**16] + zero[1:])])],
               "parent exceeds the strict B bound")
    check_case(binary, ccs, directory, "short_block", [stream(records=[(0, zero[:-1])])],
               "expected 54 parent coefficients")
    check_case(binary, ccs, directory, "missing_terminator", [stream(terminator=False)],
               "missing parent terminator")
    check_case(binary, ccs, directory, "after_terminator", [stream(extra="[]\n")],
               "extra data after parent terminator")
    check_case(binary, ccs, directory, "empty_invocation_range", [stream()],
               "invalid selected matrix row range", last=0)
    check_case(binary, ccs, directory, "reversed_invocation_range", [stream()],
               "invalid selected matrix row range", first=1, last=0)
    check_case(binary, ccs, directory, "past_invocation_range", [stream()],
               "invalid selected matrix row range", last=6_377_559)
    check_case(binary, ccs, directory, "incomplete_poseidon_invocation", [stream()],
               "Poseidon range must contain complete 94-row invocations", last=1)
    print("pidec_parent_boundaries=passed scope=decoder_coverage_and_zero_action", flush=True)


if __name__ == "__main__":
    main()
