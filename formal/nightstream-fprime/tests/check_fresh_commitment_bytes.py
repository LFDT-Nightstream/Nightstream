"""Compare the complete fresh commitment and serialize its Lean-derived claim."""
import argparse
import copy
import json
from pathlib import Path

P = 18446744069414584321

def require(condition, message):
    if not condition:
        raise ValueError(message)

def compare(actual, expected):
    require(actual == expected, "complete fresh claim bytes differ")

def canonical(words):
    return all(type(word) is int and 0 <= word < P for word in words)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commitment")
    parser.add_argument("caller")
    parser.add_argument("native_claim")
    parser.add_argument("claim_output")
    args = parser.parse_args()
    output = Path(args.claim_output)
    require(not output.exists(), "claim output already exists")
    commitment = json.loads(Path(args.commitment).read_text())
    require(len(commitment) == 5 and commitment[:4] == [1, 4685394, 0, 4685394],
            "expected the complete selected Lean commitment")
    rows = commitment[4]
    require(len(rows) == 22 and all(len(row) == 54 and canonical(row) for row in rows),
            "invalid complete commitment coefficients")
    caller = json.loads(Path(args.caller).read_text())
    require(len(caller) == 5 and caller[0] == 1 and len(caller[4]) == 7,
            "expected the checked recursive caller")
    public = caller[4][2]
    require(len(public) == 270 and canonical(public), "invalid derived fresh public input")
    claim = {
        "c": {"d": 54, "kappa": 22, "data": [{"value": word} for row in rows for word in row]},
        "x": [{"value": word} for word in public],
        "m_in": 270, "adv": None,
    }
    actual = json.dumps(claim, separators=(",", ":")).encode()
    expected = Path(args.native_claim).read_bytes()
    compare(actual, expected)
    changed = copy.deepcopy(claim)
    changed["c"]["data"][-1]["value"] = (changed["c"]["data"][-1]["value"] + 1) % P
    try:
        compare(actual, json.dumps(changed, separators=(",", ":")).encode())
    except ValueError:
        pass
    else:
        raise ValueError("changed final commitment target was accepted")
    with output.open("xb") as stream:
        stream.write(actual)
    print(json.dumps({"status": "passed", "commitment_coefficients": 1188,
                      "public_coefficients": 270, "complete_claim_bytes": len(actual),
                      "changed_target_rejected": True}))

if __name__ == "__main__":
    main()
