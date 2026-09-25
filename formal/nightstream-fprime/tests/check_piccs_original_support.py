#!/usr/bin/env python3
"""Check full-mask source selection, exact outputs and input rejection."""

import json
from pathlib import Path
import subprocess
import sys

from check_piccs_binary_fold import require
from check_piccs_original_evaluations import compare


def main():
    require(len(sys.argv) == 32, "expected executable, public, new directory and 28 Lean rounds")
    executable, public = map(lambda value: Path(value).resolve(), sys.argv[1:3])
    directory = Path(sys.argv[3]).resolve()
    rounds = [str(Path(value).resolve()) for value in sys.argv[4:]]
    directory.mkdir()
    formal = Path(__file__).resolve().parents[1]
    validator = formal / "scripts" / "validate.sh"
    header = [1, 54, 17, 4685394]
    cases = {
        "zero": [],
        "last-source": [[0, [[16, 2 ** 54 - 1, 0]]]],
        "last-source-tail": [[4685393, [[16, 2 ** 53, 2 ** 52]]]],
    }

    def capture(name, records, terminated=True, extra=""):
        path = directory / (name + ".jsonl")
        rows = [header, *records] + ([[]] if terminated else [])
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                               for row in rows) + extra)
        return path

    def run(name, source, reference=False, error=None):
        outputs = [directory / (name + f"-{block}.json") for block in (0, 10, 5)]
        groups = [str(outputs[0]), "0", "0", "94",
                  str(outputs[1]), "10", "0", "34",
                  str(outputs[2]), "5", "0", "1330"]
        command = ["bash", str(validator), "lean-executable", str(executable),
                   "original-matrix-reference" if reference else "original-matrix",
                   str(public), str(source), *groups, "--", *rounds]
        result = subprocess.run(command, cwd=formal, capture_output=True, timeout=300)
        log = result.stdout + result.stderr
        (directory / (name + ".log")).write_bytes(log)
        if error is not None:
            require(result.returncode != 0 and error.encode() in log,
                    f"{name}: missing expected rejection")
            require(all(not path.exists() for path in outputs),
                    f"{name}: rejected input wrote output")
            return None
        require(result.returncode == 0, f"{name}: execution failed")
        events = [json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith(b'{"') and b'"original_source_support"' in line]
        require(len(events) == 1, f"{name}: missing full-mask support result")
        return outputs, events[0]["zero_sources"]

    matched = []
    for name, records in cases.items():
        source = capture(name, records)
        reference, _ = run(name + "-reference", source, reference=True)
        actual, zero_sources = run(name + "-supported", source)
        require(zero_sources == [True] * 16 + [name == "zero"],
                f"{name}: support omitted or changed the last source")
        values = [compare("matrix", left.read_bytes(), right.read_bytes())
                  for left, right in zip(reference, actual)]
        if name == "zero":
            require(all(word == 0 for value in values for source in value[5]
                        for matrix in source for lane in matrix for word in lane),
                    "zero input produced a nonzero result")
        if name == "last-source":
            require(any(word != 0 for value in values for matrix in value[5][16]
                        for lane in matrix for word in lane),
                    "last-source fixture did not exercise matrix arithmetic")
        matched.append(name)

    invalid = [
        ("overlap-tail", capture("overlap-tail", [[4685393, [[16, 1, 1]]]]),
         "invalid signed-unit source masks"),
        ("missing-terminator", capture("missing-terminator", [], terminated=False),
         "missing source terminator"),
        ("extra-after-zero", capture("extra-after-zero", [], extra="[]\n"),
         "extra data after source terminator"),
    ]
    for name, source, error in invalid:
        run(name, source, error=error)
    print(json.dumps({"event": "original_source_support_checks_passed",
                      "byte_matched_cases": matched, "ranges_per_case": 3,
                      "field_words_compared": len(matched) * 3 * 17 * 14 * 54 * 2,
                      "rejections": [name for name, _, _ in invalid],
                      "scope": "synthetic source-selection regression; not production proof evidence"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
