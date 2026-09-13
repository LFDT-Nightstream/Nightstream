"""Exercise the compiled PiDEC replay at the strict signed-digit boundary."""

import json
from pathlib import Path
import subprocess
import sys


def run_case(directory, name, words, error=None):
    source = directory / f"{name}.input.jsonl"
    output = directory / f"{name}.output.jsonl"
    blocks = 3 if error is None else 1
    rows = [[1, blocks, 0, blocks], [0, words + [0] * (54 - len(words))]]
    if error is None:
        rows.append([1, [0] * 54])  # Explicit and omitted zero blocks agree.
    rows.append([])
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    command = ["bash", "scripts/validate.sh", "pi-dec-witness-replay",
               str(source), str(output), "0", str(blocks)]
    # The native test cap is 300 s, also below the Lean invocation cap.
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (directory / f"{name}.log").write_text(result.stdout + result.stderr)
    if error is not None:
        assert result.returncode != 0, f"{name}: invalid parent accepted"
        assert error in result.stdout + result.stderr, f"{name}: wrong rejection"
        print(f"pidec_boundary={name} rejected", flush=True)
        return
    assert result.returncode == 0, f"{name}: {result.stderr}"
    expected = [[1, 54, 16, blocks, 0, blocks],
                [0, [[0, 10, 20]] + [[child, 8, 16] for child in range(1, 16)]], []]
    assert [json.loads(line) for line in output.read_text().splitlines()] == expected
    print("pidec_boundary=signed_edges all_children_and_zero_blocks=matched", flush=True)


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir()  # Each validation uses a fresh result directory.
    modulus = 2**64 - 2**32 + 1
    bound = 2**16
    run_case(directory, "signed_edges", [0, 1, modulus - 1, bound - 1, modulus - bound + 1])
    run_case(directory, "positive_bound", [bound], "parent norm exceeds the strict B bound")
    run_case(directory, "negative_bound", [modulus - bound], "parent norm exceeds the strict B bound")
    run_case(directory, "noncanonical", [modulus], "noncanonical parent coefficient")
    print("pidec_replay_boundaries=passed", flush=True)
