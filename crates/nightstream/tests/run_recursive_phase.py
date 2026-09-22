#!/usr/bin/env python3
"""Run one fresh recursive replay phase under the repository's native-test cap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lean_graph.guard import build_lock  # noqa: E402
from lean_graph.policy import CAPS  # noqa: E402

TEST = "lifecycle::tests::staged::run_phase"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--child", type=int)
    parser.add_argument("--engine", choices=("optimized", "metal"))
    parser.add_argument("--cpu-reference", type=Path,
                        help="CPU run directory whose source files and PiCCS proof must match")
    parser.add_argument("--reference-proof", type=Path,
                        help="saved canonical CPU proof for the complete prove phase")
    args = parser.parse_args()
    if args.cpu_reference is not None and args.phase != "ccs":
        parser.error("--cpu-reference is a PiCCS comparison input")
    if args.phase == "ccs" and args.engine == "metal" and args.cpu_reference is None:
        parser.error("Metal PiCCS acceptance requires --cpu-reference")
    if args.reference_proof is not None and args.phase != "prove":
        parser.error("--reference-proof is a complete-fold comparison input")
    if args.phase == "prove" and (args.engine is None or args.reference_proof is None):
        parser.error("prove requires --engine and --reference-proof")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    request = {"phase": args.phase, "directory": str(directory)}
    if args.cpu_reference is not None:
        request["cpu_reference"] = str(args.cpu_reference.resolve())
    if args.reference_proof is not None:
        request["reference_proof"] = str(args.reference_proof.resolve())
    parts = [args.phase]
    if args.engine is not None:
        if args.phase not in ("base", "sources", "ccs", "child", "prove", "successor", "terminal", "reject"):
            parser.error("--engine selects evaluation for base, sources, ccs, child, prove, successor, terminal, and reject phases")
        request["engine"] = args.engine
    for name in ("step", "child"):
        value = getattr(args, name)
        if value is not None:
            request[name] = value
            parts.append(f"{name}-{value}")
    if any(not part or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in part) for part in parts):
        parser.error("phase and indices must form an ordinary phase filename")
    name = "-".join(parts)
    log_path = logs / f"{name}.log"
    record_path = logs / f"{name}.json"
    if record_path.exists():
        parser.error(f"phase record already exists: {record_path}")
    command = [str(args.binary.resolve()), TEST, "--ignored", "--exact", "--nocapture"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    record = {"schema": 1, "source_commit": commit, "source_changes": changes,
              "command": command, "request": request, "cap_seconds": CAPS["rust"]}

    with build_lock(), log_path.open("xb") as output:
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        # Keep the inherited process group. An outer timeout must also stop
        # this test; the Rust phase itself starts no child processes.
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                   stdout=output, stderr=subprocess.STDOUT)
        previous = {}

        def interrupted(signum, _frame):
            process.kill()
            raise InterruptedError(f"signal {signum}")

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, interrupted)
            remaining = max(0.0, CAPS["rust"] - (time.monotonic() - started))
            try:
                process.communicate(json.dumps(request).encode(), timeout=remaining)
                code = process.returncode
                outcome = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                code, outcome = 124, "timed-out"
            except InterruptedError:
                process.kill()
                process.communicate()
                code, outcome = 130, "interrupted"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        record.update(elapsed_seconds=time.monotonic() - started, exit=code, outcome=outcome,
                      user_seconds=after.ru_utime - before.ru_utime,
                      system_seconds=after.ru_stime - before.ru_stime,
                      maximum_resident_bytes=after.ru_maxrss * (1 if sys.platform == "darwin" else 1024))

    # An unmatched libtest filter exits successfully without running a test.
    if code == 0 and "test result: ok. 1 passed; 0 failed; 0 ignored;" not in log_path.read_text():
        record.update(exit=1, outcome="missing-test-result")
        code = 1
    with record_path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(json.dumps(record), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
