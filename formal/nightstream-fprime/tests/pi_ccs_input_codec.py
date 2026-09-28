"""Check the executable PiCCS input boundary with caller-owned external data.

The positive input and expected result must already have valid-opening
evidence. These tests check encoding and rejection, not opening validity.
All Lean commands use validate.sh. All sinks must be outside the repository.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time


def numeric_json(value):
    return json.dumps(value, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_package", type=Path)
    parser.add_argument("positive_input", type=Path)
    parser.add_argument("positive_result", type=Path)
    parser.add_argument("external_output", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    output = args.external_output.resolve()
    assert not output.is_relative_to(repository), "test sinks must be external"
    output.mkdir(exist_ok=False)
    positive = json.loads(args.positive_input.read_text())
    expected = json.loads(args.positive_result.read_text())
    assert positive[0] == 2 and len(positive) == 7
    assert expected[1] == positive and expected[5][0] == 1
    records = []

    def check(label, text, accepted_encoding):
        input_path = output / f"{label}-input.json"
        result_path = output / f"{label}-result.json"
        input_path.write_text(text)
        command = [
            "bash", "scripts/validate.sh", "pi-ccs-input-check",
            str(input_path), str(result_path),
        ]
        started = time.monotonic()
        # The root test policy caps every non-Lean test driver at 300 s.
        # The enclosing invocation also needs a process-tree supervisor.
        with (output / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=args.validation_package, stdout=log,
                stderr=subprocess.STDOUT, timeout=300, check=False,
            )
        record = {
            "case": label, "command": command, "exit": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
        records.append(record)
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        assert result.returncode == (0 if accepted_encoding else 2), record
        if accepted_encoding:
            assert json.loads(result_path.read_text()) == expected, label
        else:
            assert not result_path.exists(), "decoder failure must not emit a phase result"
        print(json.dumps(record), flush=True)

    check("canonical-no-newline", numeric_json(positive), True)
    check("canonical-newline", numeric_json(positive) + "\n", True)
    check("noncanonical-whitespace", numeric_json(positive) + " \n", False)
    changed = copy.deepcopy(positive)
    changed[0] = 1
    check("old-schema", numeric_json(changed), False)
    check("missing-running", numeric_json(positive[:6]), False)
    check("extra-root-field", numeric_json(positive + [0]), False)

    # Each path names one distinct required vector dimension.
    dimensions = {
        "running-fields": [6],
        "point": [6, 0],
        "point-limbs": [6, 0, 0],
        "commitment-sources": [6, 1],
        "commitment-words": [6, 1, 0],
        "public-sources": [6, 2],
        "public-words": [6, 2, 0],
        "eval-k-sources": [6, 3],
        "eval-k-coefficients": [6, 3, 0],
        "eval-k-limbs": [6, 3, 0, 0],
        "eval-a-sources": [6, 4],
        "eval-a-matrices": [6, 4, 0],
        "eval-a-coefficients": [6, 4, 0, 0],
        "eval-a-limbs": [6, 4, 0, 0, 0],
    }
    for label, path in dimensions.items():
        changed = copy.deepcopy(positive)
        vector = changed
        for index in path:
            vector = vector[index]
        vector.pop()
        check(f"short-{label}", numeric_json(changed), False)

    for label, word in [
        ("field-modulus", 0xFFFFFFFF00000001),
        ("negative-word", -1),
        ("string-word", "0"),
        ("float-spelling", 0.0),
    ]:
        changed = copy.deepcopy(positive)
        changed[6][4][15][13][53][1] = word
        check(label, numeric_json(changed), False)
    print(f"pi_ccs_input_codec_cases_passed={len(records)}", flush=True)


if __name__ == "__main__":
    main()
