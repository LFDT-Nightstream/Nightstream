#!/usr/bin/env python3
"""Run an ordinary development command under the shared build lock and cap."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "lean_graph"

from .policy import CAPS
from .snapshot import EvidenceError

_LOCK_PATH = Path("/tmp/lean-graph-build.lock")


@contextmanager
def build_lock(store=None):
    # TMPDIR can differ between agent clients. All clients must use this path.
    descriptor = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            owner = handle.read().strip()
            raise EvidenceError("another command holds the shared build lock" +
                                (": " + owner if owner else "")) from error
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid()}, handle)
            handle.flush()
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_build_processes():
    active = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    found = [line.strip() for line in active if len(line.split(None, 1)) == 2
             and Path(line.split(None, 1)[1]).name in ("lake", "lean", "cargo", "rustc")]
    if found:
        raise EvidenceError("an unmanaged Lean or Rust process is active: " + "; ".join(found))


def run(command, kind, cwd):
    if not command or kind not in CAPS:
        raise EvidenceError("a command and its kind are required")
    if kind == "lean" and command[:2] != ["bash", "scripts/validate.sh"]:
        raise EvidenceError("Lean commands must use scripts/validate.sh")
    if kind == "rust" and (command[0] != "cargo" or "--release" not in command):
        raise EvidenceError("Rust commands must use Cargo in release mode")
    cap = CAPS[kind]
    environment = dict(os.environ, RUSTC_WRAPPER="")
    if kind == "lean":
        environment["LEAN_TIMEOUT_SECONDS"] = str(cap)
    with build_lock():
        check_build_processes()
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True)
        handlers = {}

        def interrupted(signum, _frame):
            raise InterruptedError(f"signal {signum}")

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, interrupted)
            try:
                code = process.wait(timeout=cap)
                outcome = "passed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                code, outcome = 124, "timed-out"
            except InterruptedError:
                code, outcome = 130, "interrupted"
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        return {"outcome": outcome, "exit": code, "cap_seconds": cap,
                "elapsed_seconds": time.monotonic() - started,
                "command": command, "cwd": str(cwd), "evidence": "development diagnostic only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=CAPS, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = run(command, args.kind, args.cwd.resolve())
        print(json.dumps(result), file=sys.stderr)
        return result["exit"]
    except (EvidenceError, OSError, subprocess.SubprocessError) as error:
        print(f"lean-graph guard: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
