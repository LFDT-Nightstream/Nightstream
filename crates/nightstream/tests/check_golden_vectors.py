#!/usr/bin/env python3
"""Replay the retained two-fold vector with the current Lean and Rust verifiers.

The archive contains proof inputs and expected interface values, not private
witness matrices. Acceptance comes from the verifiers and exact comparisons.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

from check_lean_fold import Check, compare_files, require


FOLD_INPUTS = ("pi_ccs_input.json", "children.json", "actual_result.json", "proof.native", "caller-inputs.json")
SOURCE_INPUTS = ("envelope.json", "fresh-claim.json")


def extract_vectors(archive, directory):
    names = {f"native/fold-{step}/{name}" for step in (1, 2) for name in FOLD_INPUTS}
    names |= {f"native/step-{step}/{name}" for step in (1, 2) for name in SOURCE_INPUTS}
    names |= {f"expected/step-{step}-{kind}.json" for step in (1, 2) for kind in ("nifs", "caller")}
    names.add("native/step-3/envelope.json")
    with zipfile.ZipFile(archive) as source:
        require(len(source.namelist()) == len(names) and set(source.namelist()) == names,
                "golden vector must contain exactly the two-fold interface files")
        for name in sorted(names):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as input_file, path.open("xb") as output:
                shutil.copyfileobj(input_file, output)


def compare_expected(output, expected, step):
    for kind in ("nifs", "caller"):
        name = f"step-{step}-{kind}.json"
        compare_files(output / name, expected / name, f"retained Lean {kind} vector for fold {step}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path,
                        default=Path(__file__).parent / "fixtures/golden-wide-v1.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-checker", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    native_checker = args.native_checker.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    result = {"outcome": "failed", "folds": [],
              "scope": "Retained proof inputs replayed by current Lean and Rust; exact Lean interface values. "
                       "No private witness generation or terminal-opening check."}
    try:
        vectors = output / "vectors"
        extract_vectors(args.vectors, vectors)
        for step in (1, 2):
            checked = output / f"lean-step-{step}"
            checked.mkdir()
            checker = Check(vectors / "native", step, checked, native_checker)
            fold = checker.run()
            compare_expected(checked, vectors / "expected", step)
            for original, snapshot in checker.originals.items():
                compare_files(original, snapshot, "golden input changed during checking")
            result["folds"].append({"step": step, "outcome": "passed", **fold,
                                    "commands": checker.records})
        result["outcome"] = "passed"
    except Exception as error:
        result["error"] = str(error)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"outcome": result["outcome"], "result": str(output / "result.json"),
                      "error": result.get("error")}), flush=True)
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
