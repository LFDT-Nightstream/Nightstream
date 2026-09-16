#!/usr/bin/env python3
"""Check finish-original rejection. Run from the Nightstream repository root.

Arguments: executable public evaluations new-output-directory 28-Lean-round-paths.
This check does not run a successful finish or claim source-bridge coverage.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys

P = 18446744069414584321


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(len(sys.argv) == 33,
            "expected executable, public, evaluations, new directory and 28 Lean rounds")
    executable, public, evaluations = [Path(value).resolve() for value in sys.argv[1:4]]
    directory = Path(sys.argv[4]).resolve()
    rounds = [Path(value).resolve() for value in sys.argv[5:]]
    formal = Path.cwd() / "formal" / "nightstream-fprime"
    validator = formal / "scripts" / "validate.sh"
    require(validator.is_file(), "run this check from the Nightstream repository root")
    require(executable.is_file() and public.is_file(), "missing executable or original public input")
    original = json.loads(evaluations.read_bytes())
    saved_rounds = [json.loads(path.read_bytes()) for path in rounds]
    require(len(original) == 4 and original[0] == 1, "expected complete original evaluations")
    require(all(len(value) == 10 for value in saved_rounds), "expected complete Lean round traces")
    require(original[1] == [value[5] for value in saved_rounds],
            "baseline evaluation point differs from the supplied round challenges")
    directory.mkdir()
    rejected = []

    def run(name, error, change=None, selected_rounds=None, changed_round=None,
            duplicate=False, existing=False):
        case = directory / name
        case.mkdir()
        selected_evaluations = evaluations
        if change is not None:
            value = copy.deepcopy(original)
            change(value)
            selected_evaluations = case / "evaluations.json"
            selected_evaluations.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        selected_rounds = list(rounds if selected_rounds is None else selected_rounds)
        if changed_round is not None:
            index, value = changed_round
            path = case / f"round-{index}.json"
            path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
            selected_rounds[index] = path
        outputs = [case / "input.json", case / "phase.json", case / "proof-words.json"]
        if duplicate:
            outputs[2] = outputs[1]
        else:
            require(len(set(outputs)) == 3, f"{name}: test outputs are not distinct")
        sentinel = b"preserve existing final PiCCS output\n"
        if existing:
            outputs[2].write_bytes(sentinel)
        before = {path: path.read_bytes() if path.exists() else None for path in outputs}
        command = ["bash", str(validator), "lean-executable", str(executable),
                   "finish-original", str(public), str(selected_evaluations),
                   *map(str, outputs), *map(str, selected_rounds)]
        # Match the existing rejection-test cap. The caller also applies the
        # project test guard; no nested graph guard is started here.
        result = subprocess.run(command, cwd=formal, capture_output=True, timeout=300)
        log = result.stdout + result.stderr
        (case / "run.log").write_bytes(log)
        # Check the files before the error text so partial output is always reported.
        for path, content in before.items():
            if content is None:
                require(not path.exists(), f"{name}: rejected input wrote {path.name}")
            else:
                require(path.is_file() and path.read_bytes() == content,
                        f"{name}: existing output changed")
        require(result.returncode != 0, f"{name}: invalid input was accepted")
        require(error.encode() in log, f"{name}: missing expected rejection: {error}")
        require(b'"event":"independent_piccs_complete"' not in result.stdout,
                f"{name}: rejected input reported completion")
        rejected.append({"case": name, "reason": error})

    run("missing-round", "final PiCCS replay requires all 28 Lean rounds",
        selected_rounds=rounds[:-1])
    changed = copy.deepcopy(saved_rounds[-1])
    changed[3][0] = (changed[3][0] + 1) % P
    run("changed-causal-round", "Lean round 27 differs from the causal public transcript",
        changed_round=(27, changed))
    run("duplicate-outputs", "duplicate final PiCCS output path", duplicate=True)
    # An existing third slot must reject before either earlier output is written.
    run("existing-last-output", "output already exists", existing=True)
    run("schema", "unexpected complete original evaluation schema",
        change=lambda value: value.__setitem__(0, 2))
    run("missing-family", "expected the complete original Pad and matrix families",
        change=lambda value: value.pop())
    run("wrong-point", "original evaluation point differs from the Lean transcript",
        change=lambda value: value[1][27].__setitem__(1, (value[1][27][1] + 1) % P))
    run("point-width", "expected array length 28", change=lambda value: value[1].pop())
    run("pad-sources", "expected array length 17", change=lambda value: value[2].pop())
    run("pad-lanes", "expected array length 54", change=lambda value: value[2][16].pop())
    run("pad-extension", "expected array length 2", change=lambda value: value[2][16][53].pop())
    run("matrix-sources", "expected array length 17", change=lambda value: value[3].pop())
    run("matrix-ports", "expected array length 14", change=lambda value: value[3][16].pop())
    run("matrix-lanes", "expected array length 54", change=lambda value: value[3][16][13].pop())
    run("matrix-extension", "expected array length 2",
        change=lambda value: value[3][16][13][53].pop())
    run("pad-noncanonical", "noncanonical Goldilocks word",
        change=lambda value: value[2][16][53].__setitem__(1, P))
    run("matrix-noncanonical", "noncanonical Goldilocks word",
        change=lambda value: value[3][16][13][53].__setitem__(1, P))
    run("matrix-word-type", "Natural number expected",
        change=lambda value: value[3][16][13][53].__setitem__(1, "not-a-field"))
    # These remain well-formed and canonical, so the independent final check
    # must reject the changed values rather than a decoder rejecting the shape.
    run("changed-pad", "independent final PiCCS check rejected",
        change=lambda value: value[2][16][53].__setitem__(1, (value[2][16][53][1] + 1) % P))
    run("changed-matrix", "independent final PiCCS check rejected",
        change=lambda value: value[3][16][13][53].__setitem__(1,
                                                         (value[3][16][13][53][1] + 1) % P))
    print(json.dumps({"event": "piccs_finish_rejection_checks_passed", "rejections": rejected,
                      "scope": "finish-original input and output rejection only; "
                               "successful production parity and source bridge are separate"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
