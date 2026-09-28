#!/usr/bin/env python3
"""Run explicit golden checks through the existing bounded coordinators."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "crates/nightstream/tests"
sys.path[:0] = [str(TESTS), str(ROOT / "scripts")]
from check_lean_fold import compare_caller, compare_files  # noqa: E402
from compare_recursive_outputs import equal, load  # noqa: E402


def run(command):
    command = list(map(str, command))
    print("golden conformance: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def bounded(kind, command):
    # Each single command takes the shared lock and the existing project cap.
    # Coordinators are run separately; their children own their caps and locks.
    return [sys.executable, "-B", ROOT / "scripts/lean_graph/guard.py",
            "--kind", kind, "--cwd", ROOT, "--", *command]


def build(directory):
    target = "nightstream"
    arguments = ["test", "-p", "nightstream", "--lib", "--no-run"]
    toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    command = bounded("rust", ["cargo", f"+{toolchain}", *arguments, "--locked", "--release", "--message-format=json"])
    log = directory / f"build-{target}.jsonl"
    with log.open("x") as output:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(), file=sys.stderr)
        raise ValueError(f"current {target} build failed")
    artifacts = []
    for line in log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        item = json.loads(line)
        if (item.get("reason") == "compiler-artifact" and item["target"]["name"] == target
                and item.get("executable") and item["profile"]["test"]):
            artifacts.append(Path(item["executable"]))
    if len(artifacts) != 1 or not artifacts[0].is_file():
        raise ValueError(f"build must return exactly one current executable: {target}")
    return artifacts[0]


def cpu_handoff(directory):
    """Compare transported CPU bytes with the inputs actually checked by Lean."""
    equal(load(directory / "cpu/conformance.json")["outcome"], "passed", "native CPU result")
    for step in (1, 2):
        checked = directory / f"lean-step-{step}"
        equal(load(checked / "result.json")["outcome"], "passed", "fresh Lean result")
        for name in ("proof.native", "pi_ccs_input.json", "children.json", "actual_result.json", "caller-inputs.json"):
            compare_files(directory / f"cpu/fold-{step}" / name,
                          checked / f"inputs/fold-{step}" / name, "CPU/Lean input handoff")
        for name in ("envelope.json", "fresh-claim.json"):
            compare_files(directory / f"cpu/step-{step}" / name,
                          checked / f"inputs/step-{step}" / name, "CPU/Lean source handoff")
        compare_files(directory / f"cpu/fold-{step}/proof.native", checked / f"step-{step}-lean-proof.native",
                      "CPU/fresh Lean proof handoff")
        compare_files(directory / f"cpu/fold-{step}/physical.bin", checked / "lean-physical.bin",
                      "CPU/fresh Lean physical handoff")
        caller = load(checked / f"step-{step}-caller.json")
        compare_caller(load(directory / f"cpu/fold-{step}/caller-inputs.json"), caller,
                       load(directory / f"cpu/fold-{step}/actual_result.json"),
                       load(checked / "inputs/base.json")[1])


def execute(archives, directory):
    directory.mkdir(parents=True, exist_ok=False)
    references = directory / "references" if archives is not None else None
    if references is not None:
        run(bounded("python", [sys.executable, "-B", TESTS / "restore_golden_inputs.py",
                               "--archives", archives, "--directory", references]))
    binary = build(directory)
    native = directory / "cpu"
    command = [sys.executable, "-B", TESTS / "run_golden_conformance.py", "--binary", binary,
               "--directory", native]
    if references is not None:
        command += ["--references", references]
    run(command)
    for step in (1, 2):
        run([sys.executable, "-B", TESTS / "check_lean_fold.py", "--directory", native,
             "--step", step, "--output", directory / f"lean-step-{step}", "--native-checker", binary])
    cpu_handoff(directory)
    receipt = {"outcome": "passed", "engine": "optimized",
               "scope": "Current CPU folds 1–2 and state-3 terminal checks; fresh Lean verifier and caller "
                        "and physical-witness comparisons. This run checks native proofs with Lean; it does not generate proofs in Lean."}
    (directory / "cpu-result.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, help="optional native references for the same package")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    execute(args.archives.resolve() if args.archives else None, args.directory.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(f"golden conformance failed: {error}")
