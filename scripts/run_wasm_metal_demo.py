#!/usr/bin/env python3
"""Build the WASM/Nebula demo, then enforce the repository's 300-second test cap."""

import os
from pathlib import Path
import signal
import subprocess
import sys

repository = Path(__file__).resolve().parent.parent
options = [
    "-p", "neo-wasm", "--release", "--test", "wasm_nebula_metal_demo",
    "--features", "audit-html,perf-timers",
]
subprocess.run(["cargo", "build", *options], cwd=repository, check=True)
process = subprocess.Popen(
    ["cargo", "test", *options, "--", "--ignored", "--exact",
     "wasm_nebula_metal_import_export_demo", "--nocapture"],
    cwd=repository,
    start_new_session=True,
)
try:
    sys.exit(process.wait(timeout=300))
except (subprocess.TimeoutExpired, KeyboardInterrupt):
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    print("Demo stopped: interrupted or reached the 300-second test cap.", file=sys.stderr)
    sys.exit(124)
