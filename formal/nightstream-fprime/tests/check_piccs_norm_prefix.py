"""Reject changed norm bytes, malformed streams, and changed causal traces."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

# The coordinator runs this test under guard.py's 300-second native cap.
# Each child also uses validate.sh and a 300-second ceiling (below the Lean cap).
tail_only = len(sys.argv) == 12 and sys.argv[-1] == "--tail-only"
if len(sys.argv) != 11 and not tail_only:
    sys.exit("expected executable, public, sources, Q0, Q1, codes, first, end, chunk, new-results")
binary, public, sources, q0, q1, codes, first, end, chunk, results = sys.argv[1:11]
codes = Path(codes).resolve()
results = Path(results).resolve()
results.mkdir(exist_ok=False)
binary = str(Path(binary).resolve())
validate = Path(__file__).resolve().parents[1] / "scripts" / "validate.sh"


def run(name, directory=codes, zero=q0, one=q1, mode="check-norm-codes", expected=None):
    command = ["bash", str(validate), "lean-executable", binary, mode,
               public, sources, str(zero), str(one), str(directory), first, end, chunk]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=300, check=False)
    (results / f"{name}.log").write_text(result.stdout)
    if expected is None:
        if result.returncode or '"event":"norm_codes_byte_match"' not in result.stdout:
            raise AssertionError(f"{name}: unchanged norm comparison failed")
    elif not result.returncode or expected not in result.stdout:
        raise AssertionError(f"{name}: expected rejection containing {expected!r}")
    print(f"norm_prefix_case={name} result={'passed' if expected is None else 'rejected'}",
          flush=True)


def copy_codes(name):
    directory = results / name
    shutil.copytree(codes, directory)
    return directory


if not tail_only:
    run("unchanged")

directory = copy_codes("changed_last_source_tail")
path = directory / "source-16.bin"
data = bytearray(path.read_bytes())
data[-1] = (data[-1] + 1) % 81
path.write_bytes(data)
run("changed_last_source_tail", directory, expected="norm field bytes differ: source=16")
if tail_only:
    print("piccs_norm_prefix_full_tail=rejected", flush=True)
    sys.exit(0)

directory = copy_codes("invalid_code")
path = directory / "source-0.bin"
data = bytearray(path.read_bytes())
data[0] = 81
path.write_bytes(data)
run("invalid_code", directory, expected="invalid norm code")

directory = copy_codes("truncated_source")
path = directory / "source-0.bin"
path.write_bytes(path.read_bytes()[:-1])
run("truncated_source", directory, expected="truncated norm source")

directory = copy_codes("extra_byte")
with (directory / "source-0.bin").open("ab") as handle:
    handle.write(b"\x28")
run("extra_byte", directory, expected="extra bytes in norm source")

directory = copy_codes("missing_source")
(directory / "source-16.bin").unlink()
run("missing_source", directory, expected="missing or extra files")

directory = copy_codes("wrong_extent")
path = directory / "manifest.json"
manifest = json.loads(path.read_text())
manifest[5] += 1
path.write_text(json.dumps(manifest))
run("wrong_extent", directory, expected="norm code manifest differs")

for name, trace_path, field in [
    ("changed_round_zero_challenge", q0, 5),
    ("changed_round_one_state", q1, 6),
    ("changed_round_one_claim", q1, 9),
]:
    value = json.loads(Path(trace_path).read_text())
    value[field][0] = (value[field][0] + 1) % 18446744069414584321
    changed = results / f"{name}.json"
    changed.write_text(json.dumps(value))
    run(name, zero=changed if trace_path == q0 else q0,
        one=changed if trace_path == q1 else q1,
        expected="challenge, state or claim differs")

run("existing_output", mode="norm-codes", expected="output directory already exists")
print("piccs_norm_prefix_mutations=passed", flush=True)
