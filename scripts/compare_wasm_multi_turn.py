"""Compare completed multi-turn test corpora under the same wall-clock limit."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def build():
    targets = ("wasm_entry_inputs", "wasm_multi_turn")
    root = Path(__file__).resolve().parents[1]
    command = ["cargo", "test", "--release", "--locked", "-p", "neo-wasm",
               "--no-run", "--message-format=json-render-diagnostics"]
    for target in targets:
        command.extend(["--test", target])
    result = subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, text=True)
    binaries = {}
    for line in result.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("executable") and msg["target"]["name"] in targets:
            binaries[msg["target"]["name"]] = msg["executable"]
    return binaries


def measure(commands, seconds):
    start = time.monotonic()
    total_passed = 0
    transitions = Counter()
    for command in commands:
        remaining = seconds - (time.monotonic() - start)
        try:
            result = subprocess.run(
                [*command, "--test-threads=1", "--nocapture"], timeout=max(0, remaining),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except subprocess.TimeoutExpired:
            raise SystemExit(f"Corpus exceeded {seconds:g} s. Comparison incomplete.") from None
        if result.returncode:
            raise SystemExit(result.stdout)
        passed = re.search(r"test result: ok\. (\d+) passed; 0 failed;", result.stdout)
        if not passed or not int(passed[1]):
            raise SystemExit(f"Missing completed tests:\n{result.stdout}")
        total_passed += int(passed[1])
        transitions.update(re.findall(r"\bturn-transition (value|void|trap) (\d+) (value|void) (same|different)\b",
                                      result.stdout))
    elapsed = time.monotonic() - start
    if elapsed > seconds:
        raise SystemExit(f"Corpus exceeded {seconds:g} s. Comparison incomplete.")
    if not transitions:
        raise SystemExit("Corpus produced no observed transitions")
    return total_passed, transitions, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60, help="wall-clock cap per corpus, excluding build")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if not (0 < args.seconds <= 300 and args.trials > 0):
        parser.error("require 0 < seconds <= 300 and trials > 0")
    binaries = build()
    multi = binaries["wasm_multi_turn"]
    corpora = {
        "existing": [[multi, "--skip", "stitching::"], [binaries["wasm_entry_inputs"]]],
        "pilot": [[multi, "stitching::"]],
    }
    samples = {name: [] for name in corpora}
    print(f"Each corpus has {args.seconds:g} s per trial.", flush=True)
    for trial in range(args.trials):
        for name in (tuple(corpora) if trial % 2 == 0 else tuple(reversed(corpora))):
            passed, counts, elapsed = measure(corpora[name], args.seconds)
            boundaries = sum(counts.values())
            print(f"trial {trial + 1} {name}: {passed} passed, {boundaries} boundaries, {elapsed:.3f} s", flush=True)
            samples[name].append((passed, counts, elapsed))
    for name, runs in samples.items():
        if any(run[:2] != runs[0][:2] for run in runs):
            raise SystemExit(f"{name} changed test count or transition counts between trials")
        times = [elapsed for _, _, elapsed in runs]
        print(f"{name}: {statistics.median(times):.3f} s median, {min(times):.3f}..{max(times):.3f} s")
    before, after = (samples[name][0][1] for name in corpora)
    print(f"{len(before)} existing classes, {len(after)} pilot classes, "
          f"{len(after.keys() - before.keys())} new classes, {len(before.keys() | after.keys())} combined")
    print("previous  next params  next result  export     existing  pilot")
    for key in sorted(before.keys() | after.keys()):
        previous, params, following, export = key
        print(f"{previous:8}  {params:11}  {following:11}  {export:9}  {before[key]:8}  {after[key]:5}")


if __name__ == "__main__":
    main()
