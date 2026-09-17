#!/usr/bin/env python3
"""Check canonical-row input rejection using the complete selected carrier."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

P = 18446744069414584321
CARRIER = 253011276


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("carrier", type=Path)
    parser.add_argument("caller", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    carrier, caller_path = args.carrier.resolve(), args.caller.resolve()
    directory = args.directory.resolve()
    directory.mkdir()
    formal = Path.cwd() / "formal/nightstream-fprime"
    caller = json.loads(caller_path.read_text())
    rejected = []

    def run(name, source, caller_source, reason, existing=False):
        output = directory / (name + "-result.json")
        retained = b"retained output\n"
        if existing:
            output.write_bytes(retained)
        result = subprocess.run(
            ["bash", "scripts/validate.sh", "lean-executable",
             ".lake/build/bin/checkFreshRows", str(source), str(caller_source), str(output)],
            cwd=formal, capture_output=True, timeout=300)
        log = result.stdout + result.stderr
        (directory / (name + ".log")).write_bytes(log)
        if result.returncode == 0 or reason.encode() not in log:
            raise ValueError(name + ": did not reject with the required reason")
        if existing:
            if output.read_bytes() != retained:
                raise ValueError(name + ": changed existing output")
        elif output.exists():
            raise ValueError(name + ": rejected input wrote a result")
        rejected.append(name)

    run("existing-output", carrier, caller_path, "output already exists", True)
    empty = directory / "empty.bin"
    empty.write_bytes(b"")
    run("truncated", empty, caller_path, "wrong complete byte count")
    for name, index, code, reason in [
            ("noncanonical", 0, 2, "noncanonical signed-unit byte"),
            ("tail", CARRIER - 1, 1, "fresh carrier has nonzero tail")]:
        source = directory / (name + ".bin")
        with source.open("wb") as stream:
            stream.truncate(CARRIER)
            stream.seek(index)
            stream.write(bytes([code]))
        run(name, source, caller_path, reason)
    for name, change, reason in [
            ("public-width", lambda value: value[4][2].pop(), "wrong width"),
            ("public-noncanonical", lambda value: value[4][2].__setitem__(0, P),
             "noncanonical caller fresh public field"),
            ("changed-public", lambda value: value[4][2].__setitem__(0, (value[4][2][0] + 1) % P),
             "fresh public input differs from caller")]:
        value = copy.deepcopy(caller)
        change(value)
        source = directory / (name + ".json")
        source.write_text(json.dumps(value, separators=(",", ":")))
        run(name, carrier, source, reason)
    changed = directory / "changed-private.bin"
    shutil.copyfile(carrier, changed)
    with changed.open("r+b") as stream:
        stream.seek(270)  # First coordinate after the complete public prefix.
        old = stream.read(1)
        stream.seek(270)
        stream.write(b"\x01" if old == b"\x00" else b"\x00")
    run("changed-private", changed, caller_path, "canonical scalar production row check failed")
    print(json.dumps({"status": "passed", "rejected": rejected}))


if __name__ == "__main__":
    main()
