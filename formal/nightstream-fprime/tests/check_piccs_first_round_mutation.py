"""Reject one changed Rust coefficient against the actual Lean round result."""

import copy
import json
from pathlib import Path
import runpy
import sys

compare = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "compare-piccs-first-round.py")
)["compare"]

if len(sys.argv) not in (5, 6):
    sys.exit("expected original public input, Lean round, Rust input, Rust phase and optional round index")
public, lean, proof, trace = [json.loads(Path(path).read_text()) for path in sys.argv[1:5]]
round_index = int(sys.argv[5]) if len(sys.argv) == 6 else 0
compare(public, lean, proof, trace, round_index)
changed = copy.deepcopy(proof)
index = len(changed[3][round_index]) - 1
changed[3][round_index][index][0] = (changed[3][round_index][index][0] + 1) % 18446744069414584321
try:
    compare(public, lean, changed, trace, round_index)
except ValueError as error:
    if f"Q[{round_index}] coefficient {index} differs:" not in str(error):
        raise
else:
    raise AssertionError("changed Rust coefficient was accepted")
if round_index == 0:
    print(f"piccs_first_round_mutation=rejected coefficient={index} component=0")
else:
    print(f"piccs_round_mutation=rejected round={round_index} coefficient={index} component=0")
