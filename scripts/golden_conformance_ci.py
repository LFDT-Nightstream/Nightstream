#!/usr/bin/env python3
"""Run explicit golden checks through the existing bounded coordinators."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "crates/nightstream/tests"
FORMAL = ROOT / "formal/nightstream-fprime"
sys.path[:0] = [str(TESTS), str(ROOT / "scripts")]
from check_lean_fold import compare_caller, compare_files  # noqa: E402
from compare_recursive_outputs import compare_envelope, compare_json, equal, load  # noqa: E402


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
        caller = load(checked / f"step-{step}-caller.json")
        compare_caller(load(directory / f"cpu/fold-{step}/caller-inputs.json"), caller,
                       load(directory / f"cpu/fold-{step}/actual_result.json"),
                       load(checked / "inputs/base.json")[1])


def independent_expectations(directory, references, cpu_reference):
    replay = directory / "independent-loop"
    replay.mkdir()
    original = references / "original-sources/inputs/original-sources"
    cpu = cpu_reference / "cpu"
    # Bind the retained original witnesses to the current CPU chain before
    # either independent producer can consume them.
    for name in ["fresh-witness.json", "fresh-claim.json"] + [f"digit-{child}.json" for child in range(16)]:
        compare_json(cpu / "step-2" / name, original / name, "independent original source")
    compare_envelope(cpu, 1, original, load(cpu / "fold-1/nifs.json")["parent"])
    shutil.copytree(original, replay / "original-sources")
    package = ROOT / "crates/nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
    shutil.copyfile(package, replay / "original-package.json")
    first = directory / "independent-first"
    first.mkdir()
    shutil.copytree(cpu / "step-1", first / "original-sources")
    initial = load(cpu / "step-1/envelope.json")
    message = load(TESTS / "fixtures/stage1_recursive_states/nonzero-running.json")[3]
    request = [initial["iteration"], initial["z0"], initial["current"], message]
    (first / "original-sources/next-message-input.json").write_text(json.dumps(request) + "\n")
    shutil.copyfile(package, first / "original-package.json")
    run([sys.executable, "-B", FORMAL / "scripts/replay_recursive_loop.py", first, "1", "first-fold"])
    for phase in ("build", "prepare", "native", "ccs", "reductions", "successor", "terminal"):
        run([sys.executable, "-B", FORMAL / "scripts/replay_recursive_loop.py", replay, "2", phase])
    run([sys.executable, "-B", ROOT / "scripts/check_selected_replay.py",
         "--first-root", first, "--second-root", replay, "--cpu-reference", cpu_reference,
         "--directory", directory / "selected-comparison"])
    return "passed for the independent 1→2→3 chain and its complete input connection"


def execute(mode, archives, directory, cpu_reference=None):
    if mode == "independent" and cpu_reference is None:
        raise ValueError("independent requires the CPU handoff artifact")
    if mode == "cpu" and cpu_reference is not None:
        raise ValueError("CPU generation cannot use another CPU result")
    if mode == "independent" and archives is None:
        raise ValueError("independent requires its source archives")
    directory.mkdir(parents=True, exist_ok=False)
    references = directory / "references" if archives is not None else None
    if references is not None:
        run(bounded("python", [sys.executable, "-B", TESTS / "restore_golden_inputs.py",
                               "--archives", archives, "--directory", references]))
    if mode == "independent":
        cpu_handoff(cpu_reference)
        scope = independent_expectations(directory, references, cpu_reference)
        result = {"outcome": "passed", "scope": scope}
        (directory / "independent-result.json").write_text(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        return
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
                        "comparisons. Physical witnesses are not compared. Independent generation is separate."}
    (directory / "cpu-result.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "independent"))
    parser.add_argument("--archives", type=Path, help="optional native references; required for independent generation")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    execute(args.mode, args.archives.resolve() if args.archives else None, args.directory.resolve(),
            args.cpu_reference.resolve() if args.cpu_reference else None)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(f"golden conformance failed: {error}")
