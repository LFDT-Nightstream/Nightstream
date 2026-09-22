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


def build(directory, engine, checker=False):
    target = "generate_pi_ccs_fixture" if checker else "nightstream"
    arguments = (["build", "-p", "neo-fold-clean", "--bin", target] if checker else
                 ["test", "-p", "nightstream", "--lib", "--no-run"])
    if not checker and engine == "metal":
        arguments += ["--features", "metal"]
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
                and item.get("executable") and (checker or item["profile"]["test"])):
            artifacts.append(Path(item["executable"]))
    if len(artifacts) != 1 or not artifacts[0].is_file():
        raise ValueError(f"build must return exactly one current executable: {target}")
    return artifacts[0]


def cpu_handoff(directory):
    """Compare transported CPU bytes with the inputs actually checked by Lean."""
    equal(load(directory / "cpu/conformance.json")["outcome"], "passed", "native CPU result")
    for step in (1, 2, 3):
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
                       load(directory / f"cpu/fold-{step}/actual_result.json"), caller[1])


def independent_expectations(directory, references, checker, cpu_reference):
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
    run([sys.executable, "-B", FORMAL / "scripts/replay_recursive_loop.py", replay, "2", "all"])
    for step in (1, 2, 3):
        independent = (first if step == 1 else replay) / f"step-{step}-to-{step + 1}"
        checked = cpu_reference / f"lean-step-{step}"
        # Compare every field coefficient with this CPU run. Valid Goldilocks
        # representatives can have different JSON bytes.
        run(bounded("static", [checker, "compare-pirlc-replay", cpu / f"fold-{step}/parent-witness.json",
                              independent / "parent-0.jsonl", independent / "parent-1.jsonl"]))
        # check-owned-nifs independently encodes every Lean proof field and
        # compares it with the same CPU snapshot used by the fresh verifier.
        run(bounded("static", [checker, "check-owned-nifs", package,
                              checked / f"inputs/fold-{step}", independent / "nifs-result.json"]))
        compare_json(checked / f"step-{step}-caller.json", independent / "caller.json", "independent caller")
        compare_files(cpu / f"fold-{step}/physical.bin", independent / "physical.bin", "independent physical witness")
        for name in ("fresh-witness.json", "fresh-claim.json"):
            compare_json(cpu / f"step-{step + 1}" / name, independent / name, "independent fresh result")
        for child in range(16):
            compare_json(cpu / f"fold-{step}/digit-{child}.json",
                         independent / f"native-material/digit-{child}.json", "independent checked child witness")
    cpu_handoff(cpu_reference)
    return "passed for the exact first fold and independent iterations 2→3→4"


def execute(mode, archives, directory, cpu_reference=None):
    if mode == "metal" and sys.platform != "darwin":
        raise ValueError("selected Metal check needs a macOS host with an available Metal device")
    if mode in ("metal", "independent") and cpu_reference is None:
        raise ValueError(f"{mode} requires the CPU handoff artifact")
    if mode == "cpu" and cpu_reference is not None:
        raise ValueError("CPU generation cannot use another CPU result")
    directory.mkdir(parents=True, exist_ok=False)
    references = directory / "references"
    run(bounded("python", [sys.executable, "-B", TESTS / "restore_golden_inputs.py",
                           "--archives", archives, "--directory", references]))
    if cpu_reference is not None:
        cpu_handoff(cpu_reference)
    if mode == "independent":
        checker = build(directory, "optimized", checker=True)
        scope = independent_expectations(directory, references, checker, cpu_reference)
        result = {"outcome": "passed", "scope": scope}
        (directory / "independent-result.json").write_text(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        return
    engine = "optimized" if mode == "cpu" else "metal"
    binary = build(directory, engine)
    native = directory / ("cpu" if engine == "optimized" else "metal")
    command = [sys.executable, "-B", TESTS / "run_golden_conformance.py", "--binary", binary,
               "--directory", native, "--references", references, "--engine", engine]
    if cpu_reference is not None:
        command += ["--cpu-reference", cpu_reference / "cpu"]
    run(command)
    if engine == "optimized":
        checker = build(directory, engine, checker=True)
        for step in (1, 2, 3):
            run([sys.executable, "-B", TESTS / "check_lean_fold.py", "--directory", native,
                 "--step", step, "--output", directory / f"lean-step-{step}", "--native-checker", checker])
        cpu_handoff(directory)
    else:
        # The comparison coordinator consumes the same transported CPU files.
        # Recheck custody after the engine comparison so a changed handoff fails.
        cpu_handoff(cpu_reference)
    receipt = {"outcome": "passed", "engine": engine,
               "scope": "Current native folds 1–3 and terminal checks; fresh Lean verifier/caller/physical checks "
                        "on CPU; CPU/Metal equality when Metal is selected. Independent generation is a separate job."}
    (directory / ("cpu-result.json" if engine == "optimized" else "metal-result.json")).write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "metal", "independent"))
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    execute(args.mode, args.archives.resolve(), args.directory.resolve(),
            args.cpu_reference.resolve() if args.cpu_reference else None)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(f"golden conformance failed: {error}")
