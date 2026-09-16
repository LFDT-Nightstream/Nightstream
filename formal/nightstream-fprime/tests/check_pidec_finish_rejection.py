#!/usr/bin/env python3
"""Check from-replay rejection. Run from the Nightstream repository root.

Arguments: executable package[4] C-input commitments evaluations new-directory.
The successful production run and original-source provenance are separate.
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
    require(len(sys.argv) == 10,
            "expected executable, package[4], C input, commitments, evaluations and new directory")
    executable = Path(sys.argv[1]).resolve()
    package = sys.argv[2:6]
    inputs = [Path(value).resolve() for value in sys.argv[6:9]]
    directory = Path(sys.argv[9]).resolve()
    formal = Path.cwd() / "formal" / "nightstream-fprime"
    validator = formal / "scripts" / "validate.sh"
    require(validator.is_file(), "run this check from the Nightstream repository root")
    require(executable.is_file(), "missing executable")
    raw = [path.read_bytes() for path in inputs]
    original = [json.loads(value) for value in raw]
    ccs, commitments, evaluations = original
    require(len(ccs) == 7 and ccs[0] == 2 and len(ccs[3]) == 28,
            "expected a complete baseline C input")
    require(len(commitments) == 5 and commitments[:4] == [1, 4685394, 0, 4685394],
            "expected complete selected baseline commitments")
    require(len(evaluations) == 5 and evaluations[:2] == [1, 4685394],
            "expected complete selected baseline evaluations")
    require(len(commitments[4]) == 22 and len(commitments[4][21]) == 16 and
            len(commitments[4][21][15]) == 54, "wrong baseline commitment shape")
    require(len(evaluations[2]) == 28 and len(evaluations[3]) == 16 and
            len(evaluations[3][15]) == 54 and len(evaluations[4]) == 16 and
            len(evaluations[4][15]) == 14 and len(evaluations[4][15][13]) == 54,
            "wrong baseline evaluation shape")
    directory.mkdir()
    rejected = []

    def run(name, error, family=None, change=None, duplicate=False, existing=False, alias=False):
        case = directory / name
        case.mkdir()
        selected = list(inputs)
        if change is not None:
            value = copy.deepcopy(original[family])
            change(value)
            selected[family] = case / ("ccs.json", "commitments.json", "evaluations.json")[family]
            selected[family].write_text(json.dumps(value, separators=(",", ":")) + "\n")
        outputs = [case / "children.json", case / "complete.json"]
        if duplicate:
            outputs[1] = outputs[0]
        if existing:
            outputs[1].write_bytes(b"preserve existing complete C/R/D output\n")
        before = {path: path.read_bytes() if path.exists() else None for path in outputs}
        output_arguments = list(map(str, outputs))
        if alias:
            output_arguments[-1] = str(case) + "/./" + outputs[0].name
        command = ["bash", str(validator), "lean-executable", str(executable),
                   "from-replay", *package, *map(str, selected), *output_arguments]
        # Project rejection-test cap. The caller supplies the outer graph guard;
        # this child uses validate.sh directly and starts no nested graph guard.
        result = subprocess.run(command, cwd=formal, capture_output=True, timeout=300)
        log = result.stdout + result.stderr
        (case / "run.log").write_bytes(log)
        for path, content in before.items():
            if content is None:
                require(not path.exists(), f"{name}: rejected input wrote {path.name}")
            else:
                require(path.is_file() and path.read_bytes() == content,
                        f"{name}: existing output changed")
        require(result.returncode != 0, f"{name}: invalid input was accepted")
        require(error.encode() in log, f"{name}: missing expected rejection: {error}")
        require(b"independent_pidec_replay_complete" not in result.stdout,
                f"{name}: rejected input reported completion")
        rejected.append({"case": name, "reason": error})

    run("duplicate-outputs", "duplicate PiDEC replay output path", duplicate=True)
    run("aliased-outputs", "duplicate PiDEC replay output path", alias=True)
    # The second output already exists; the first must remain absent.
    run("existing-last-output", "output already exists", existing=True)
    run("commitment-fields", "expected five complete commitment fields", 1,
        lambda value: value.pop())
    for name, field, changed in [("schema", 0, 2), ("domain", 1, commitments[1] + 1),
                                 ("start", 2, 1), ("end", 3, commitments[3] - 1)]:
        run("commitment-" + name, "expected a complete selected Lean commitment result", 1,
            lambda value, field=field, changed=changed: value.__setitem__(field, changed))
    run("commitment-rows", "expected array length 22", 1, lambda value: value[4].pop())
    run("commitment-children", "expected array length 16", 1,
        lambda value: value[4][21].pop())
    run("commitment-lanes", "expected array length 54", 1,
        lambda value: value[4][21][15].pop())
    run("commitment-noncanonical-tail", "noncanonical Goldilocks word", 1,
        lambda value: value[4][21][15].__setitem__(53, P))
    run("evaluation-fields", "expected five complete evaluation fields", 2,
        lambda value: value.pop())
    run("evaluation-schema", "expected a complete selected Lean evaluation result", 2,
        lambda value: value.__setitem__(0, 2))
    run("evaluation-domain", "expected a complete selected Lean evaluation result", 2,
        lambda value: value.__setitem__(1, value[1] + 1))
    run("point-width", "expected array length 28", 2, lambda value: value[2].pop())
    run("point-extension", "expected array length 2", 2, lambda value: value[2][27].pop())
    run("pad-children", "expected array length 16", 2, lambda value: value[3].pop())
    run("pad-lanes", "expected array length 54", 2, lambda value: value[3][15].pop())
    run("pad-extension", "expected array length 2", 2, lambda value: value[3][15][53].pop())
    run("matrix-children", "expected array length 16", 2, lambda value: value[4].pop())
    run("matrix-ports", "expected array length 14", 2, lambda value: value[4][15].pop())
    run("matrix-lanes", "expected array length 54", 2, lambda value: value[4][15][13].pop())
    run("matrix-extension", "expected array length 2", 2,
        lambda value: value[4][15][13][53].pop())
    run("pad-noncanonical-tail", "noncanonical Goldilocks word", 2,
        lambda value: value[3][15][53].__setitem__(1, P))
    run("matrix-noncanonical-tail", "noncanonical Goldilocks word", 2,
        lambda value: value[4][15][13][53].__setitem__(1, P))
    run("wrong-derived-point", "Lean evaluation point differs from the derived parent", 2,
        lambda value: value[2][27].__setitem__(1, (value[2][27][1] + 1) % P))
    # These keep complete canonical framing. Only the existing D recomposition
    # check can reject the changed final commitment or matrix coefficient.
    run("changed-last-commitment", "independent PiDEC replay rejected", 1,
        lambda value: value[4][21][15].__setitem__(53, (value[4][21][15][53] + 1) % P))
    run("changed-last-matrix", "independent PiDEC replay rejected", 2,
        lambda value: value[4][15][13][53].__setitem__(1,
                                                   (value[4][15][13][53][1] + 1) % P))
    # Changing the last round's constant term changes its endpoint sum by two,
    # while all prior challenges and the incoming final-round claim stay fixed.
    run("changed-C-round", "PiCCS/PiRLC rejected or returned no parent", 0,
        lambda value: value[3][27][0].__setitem__(0, (value[3][27][0][0] + 1) % P))
    for path, content in zip(inputs, raw):
        require(path.read_bytes() == content, f"original input changed: {path}")
    print(json.dumps({"event": "pidec_finish_rejection_checks_passed", "rejections": rejected,
                      "scope": "from-replay rejection only; successful production parity and "
                               "original-source provenance are separate"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, TypeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
