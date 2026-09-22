#!/usr/bin/env python3
"""Compare retained independent Lean folds with a fresh CPU 1→2→3 run.

Every comparison uses complete values or bytes. Source and input digests
record custody of earlier computations; they are not protocol authority.
This command checks retained generation, and does not claim a new Lean run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "formal/nightstream-fprime"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "crates/nightstream/tests"), str(FORMAL / "scripts")]
from check_lean_fold import compare_files
from compare_recursive_outputs import compare_envelope, compare_json, equal, load
from golden_conformance_ci import bounded, build, cpu_handoff
from replay_recursive_loop import identity, producer_sources, write_new
from run_golden_conformance import source_identity


def audit_inputs(first, second, cpu_reference):
    current = producer_sources()
    prefix = "formal/nightstream-fprime/"
    # Test harness path normalization does not change a Lean producer. All
    # Lean definitions, producer scripts, toolchain and package remain pinned.
    def producer(name):
        return name.startswith(prefix) and not Path(name).name.startswith("test_")

    selected = {name: digest for name, digest in current.items() if producer(name)}
    audits = []
    for step, root in ((1, first), (2, second)):
        pin = load(root / "producer-sources.json")
        saved = {name: digest for name, digest in pin["files"].items() if producer(name)}
        equal(set(saved), set(selected), "complete Lean producer source set")
        renamed = []
        for name, digest in selected.items():
            if digest == saved[name]:
                continue
            # The rebase renamed only the native comparison crate in these
            # two scripts. No other source transition is accepted here.
            if name not in {prefix + "scripts/replay_recursive_loop.py", prefix + "scripts/check-boundaries.sh"}:
                raise ValueError(f"retained Lean producer changed: {name}")
            previous = (ROOT / name).read_bytes().replace(b"neo-fold-legacy", b"neo-fold-clean")
            equal(hashlib.sha256(previous).hexdigest(), saved[name], "comparison crate rename only")
            renamed.append(name)
        for name, path in (("original-sources", root / "original-sources"),
                           ("original-package", root / "original-package.json")):
            equal(identity(path), pin["files"][name], "retained input custody")
        package = ROOT / "crates/nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"
        compare_files(root / "original-package.json", package, "current selected package")
        cpu = cpu_reference / "cpu"
        source = root / "original-sources"
        files = []
        for name in ["fresh-witness.json", "fresh-claim.json"] + [f"digit-{child}.json" for child in range(16)]:
            files.append(compare_json(cpu / f"step-{step}" / name, source / name, "complete original input"))
        if step == 1:
            files.append(compare_json(cpu / "step-1/envelope.json", source / "envelope.json", "initial envelope"))
        else:
            record, _ = compare_envelope(cpu, 1, source, load(cpu / "fold-1/nifs.json")["parent"])
            files.append(record)
        envelope = load(cpu / f"step-{step}/envelope.json")
        message = load(ROOT / "crates/nightstream/tests/fixtures/stage1_recursive_states/nonzero-running.json")[3]
        equal(load(source / "next-message-input.json"),
              [step, envelope["z0"], envelope["current"], message], "complete requested application state")
        audits.append({"iteration": step, "retained_source_commit": pin["commit"],
                       "unchanged_Lean_source_files": len(selected) - len(renamed),
                       "native_comparison_crate_rename_only": renamed, "original_input_comparisons": files})
    return audits


def compare_outputs(first, second, cpu_reference):
    comparisons = []
    for step, root in ((1, first), (2, second)):
        generated = root / f"step-{step}-to-{step + 1}"
        checked, cpu = cpu_reference / f"lean-step-{step}", cpu_reference / "cpu"
        comparisons.append(compare_json(checked / f"step-{step}-caller.json", generated / "caller.json",
                                        "complete independent caller"))
        compare_files(cpu / f"fold-{step}/physical.bin", generated / "physical.bin", "independent physical witness")
        for name in ("fresh-witness.json", "fresh-claim.json"):
            comparisons.append(compare_json(cpu / f"step-{step + 1}" / name, generated / name,
                                            "independent fresh result"))
        for child in range(16):
            comparisons.append(compare_json(cpu / f"fold-{step}/digit-{child}.json",
                                            generated / f"native-material/digit-{child}.json",
                                            "independent checked child witness"))
    cpu_handoff(cpu_reference)
    return comparisons


def execute(first, second, cpu_reference, directory):
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema": 1, "outcome": "running", "iterations": [1, 2, 3],
              "source": source_identity(), "first_root": str(first), "second_root": str(second),
              "cpu_reference": str(cpu_reference), "commands": [],
              "scope": "Current CPU comparison with retained independent Lean generation, complete input "
                       "connection, and fresh state-3 terminal checks. No third fold, CI or Metal execution."}
    arguments = ["--first-root", first, "--second-root", second,
                 "--cpu-reference", cpu_reference, "--directory", directory]

    def phase(name, kind, command):
        log = directory / f"{name}.log"
        entry = {"name": name, "command": list(map(str, command)), "log": str(log), "outcome": "failed"}
        record["commands"].append(entry)
        started = time.monotonic()
        print(f"selected replay check: {name}", flush=True)
        with log.open("x") as output:
            result = subprocess.run(list(map(str, bounded(kind, command))), cwd=ROOT,
                                    stdout=output, stderr=subprocess.STDOUT)
        entry.update(exit=result.returncode, elapsed_seconds=time.monotonic() - started,
                     outcome="passed" if result.returncode == 0 else "failed")
        write_new(directory / f"{name}.json", entry)
        if result.returncode:
            raise ValueError(f"{name} failed; see {log}")

    try:
        phase("input-audit", "python", [sys.executable, "-B", __file__, *arguments, "--phase", "inputs"])
        checker = build(directory, checker=True)
        package = first / "original-package.json"
        for step, root in ((1, first), (2, second)):
            generated = root / f"step-{step}-to-{step + 1}"
            phase(f"parent-{step}", "static", [checker, "compare-pirlc-replay",
                  cpu_reference / f"cpu/fold-{step}/parent-witness.json",
                  generated / "parent-0.jsonl", generated / "parent-1.jsonl"])
            phase(f"digits-{step}", "static", [checker, "compare-pidec-replay", cpu_reference / f"cpu/fold-{step}",
                  generated / "digits-0.jsonl", generated / "digits-1.jsonl"])
            phase(f"proof-{step}", "static", [checker, "check-owned-nifs", package,
                  cpu_reference / f"lean-step-{step}/inputs/fold-{step}", generated / "nifs-result.json"])
        phase("complete-outputs", "python", [sys.executable, "-B", __file__, *arguments, "--phase", "outputs"])
        bridge = directory / "input-connection"
        bridge.mkdir()
        projector = [sys.executable, "-B", FORMAL / "scripts/project_replay_sources.py"]
        for name, root in (("first-original", first), ("original", second)):
            phase(name, "python", [*projector, "original", root / "original-sources", bridge / name])
        generated = first / "step-1-to-2"
        phase("feedback", "python", [*projector, "feedback", generated / "fresh-witness.json",
              generated / "fresh-claim.json", generated / "children.json", bridge / "feedback",
              generated / "digits-0.jsonl", generated / "digits-1.jsonl"])
        phase("input-connection", "python", [sys.executable, "-B", ROOT / "scripts/bridge_first_second.py",
              "--first-root", first, "--second-root", second, "--directory", bridge])
        generated = second / "step-2-to-3"
        native, material, caller = generated / "native-successor", generated / "native-material", generated / "caller.json"
        phase("terminal-accepted", "static", [checker, "check-later-terminal", "accepted", native, material,
              caller, directory / "terminal-accepted-result.json"])
        for case in ("ce-evaluation", "ce-matrix-evaluation", "fresh-private"):
            changed = directory / f"terminal-{case}-input"
            phase(f"terminal-prepare-{case}", "static", [checker, "prepare-terminal-mutation", case,
                  native, material, caller, changed])
            phase(f"terminal-check-{case}", "static", [checker, "check-prepared-terminal-mutation", case,
                  changed, material, caller, directory / f"terminal-{case}-result.json"])
        # Check source/input custody again after execution so a changed input
        # or coordinator cannot receive a successful completion record.
        phase("final-input-audit", "python", [sys.executable, "-B", __file__, *arguments, "--phase", "inputs"])
        equal(source_identity(), record["source"], "source changed during selected comparison")
        record["outcome"] = "passed"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        record.update(outcome="failed", error=str(error))
    write_new(directory / "result.json", record)
    if record["outcome"] != "passed":
        raise ValueError(record["error"])
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first-root", "second-root", "cpu-reference", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("inputs", "outputs"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in (args.first_root, args.second_root, args.cpu_reference)]
    if args.phase:
        result = (audit_inputs if args.phase == "inputs" else compare_outputs)(*roots)
        print(json.dumps(result), flush=True)
    else:
        execute(*roots, args.directory.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        sys.exit(f"selected replay comparison failed: {error}")
