"""Reject one changed Rust coefficient against the actual Lean round result."""

import copy
import json
from pathlib import Path
import runpy
import sys

compare = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]

if len(sys.argv) != 5:
    sys.exit("expected original public input, Lean round, Rust input and Rust phase")
public, lean, proof, trace = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
compare(public, lean, proof, trace)
changed = copy.deepcopy(proof)
index = len(changed[3][0]) - 1
changed[3][0][index][0] = (changed[3][0][index][0] + 1) % 18446744069414584321
try:
    compare(public, lean, changed, trace)
except ValueError as error:
    if f"Q[0] coefficient {index} differs:" not in str(error):
        raise
else:
    raise AssertionError("changed Rust coefficient was accepted")
print(f"piccs_first_round_mutation=rejected coefficient={index} component=0")
