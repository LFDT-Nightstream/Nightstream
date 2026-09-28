"""Reject malformed inputs before Lean emits a complete commitment sum."""

import copy
import json
from pathlib import Path
import subprocess
import sys


def rejected(directory, name, value, error):
    source = directory / f"{name}.input.json"
    output = directory / f"{name}.output.json"
    source.write_text(json.dumps(value) + "\n")
    command = ["bash", "scripts/validate.sh", "pi-dec-commitment-merge",
               str(output), str(source)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    text = result.stdout + result.stderr
    (directory / f"{name}.log").write_text(text)
    assert result.returncode != 0, f"{name}: malformed range accepted"
    assert error in text, f"{name}: wrong rejection"
    assert not output.exists(), f"{name}: incomplete output published"
    print(f"pidec_commitment_merge_boundary={name} rejected", flush=True)


if __name__ == "__main__":
    original = json.loads(Path(sys.argv[1]).read_text())
    assert original[2] == 0 and original[3] < original[1], "use the first partial range"
    directory = Path(sys.argv[2]).resolve()
    directory.mkdir()
    rejected(directory, "incomplete", original, "do not cover the complete carrier")
    gap = copy.deepcopy(original)
    gap[2] = 1
    rejected(directory, "gap", gap, "gap, overlap or invalid endpoint")
    shape = copy.deepcopy(original)
    shape[4].pop()
    rejected(directory, "row_shape", shape, "expected 22 entries")
    noncanonical = copy.deepcopy(original)
    noncanonical[4][0][0][0] = 2**64 - 2**32 + 1
    rejected(directory, "noncanonical", noncanonical, "noncanonical commitment coefficient")
    print("pidec_commitment_merge_boundaries=passed", flush=True)
