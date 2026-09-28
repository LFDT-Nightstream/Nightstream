#!/usr/bin/env python3
"""Reject malformed fresh binary prefixes and changed causal round claims.

This gate checks framing and rejection behavior, not source provenance.
The complete positive computation is separate recorded evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys


P = 18446744069414584321


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(len(sys.argv) == 7, "expected executable, public, Q0, Q1, full fresh prefix, new results")
    binary, public, q0, q1, prefix, results = [Path(value).resolve() for value in sys.argv[1:]]
    package = Path(__file__).resolve().parents[1]
    validate = package / "scripts" / "validate.sh"
    require(not results.is_relative_to(prefix), "results must be outside the original prefix")
    manifest = json.loads((prefix / "manifest.json").read_text())
    require(len(manifest) == 7 and manifest[:4] == [1, 2, 2, 14], "expected a fresh second prefix")
    ranges = manifest[6]
    require(len(ranges) >= 2, "the overlap test needs adjacent source files")
    filenames = [f"{first}-{finish}.bin" for first, finish in ranges]
    require({path.name for path in prefix.iterdir()} == {"manifest.json", *filenames},
            "original prefix has missing or extra files")
    require(all((prefix / name).is_file() for name in filenames), "original payload is not a file")
    first_file, final_file = filenames[0], filenames[-1]
    final_first, final_finish = ranges[-1]
    require(final_finish - final_first > 1 and final_finish - 1 >= 2,
            "final mutation must be in a non-first row outside arithmetic pair range 0..1")
    row_bytes = manifest[3] * 16
    require((prefix / final_file).stat().st_size == (final_finish - final_first) * row_bytes,
            "original final payload has a wrong byte count")
    results.mkdir(exist_ok=False)

    def clone(name):
        directory = results / name
        directory.mkdir(exist_ok=False)
        shutil.copyfile(prefix / "manifest.json", directory / "manifest.json")
        for filename in filenames:
            os.link(prefix / filename, directory / filename)
        return directory

    def detached(directory, filename):
        # Never write through a shared hardlink. Remove it before making a copy.
        target = directory / filename
        target.unlink()
        shutil.copyfile(prefix / filename, target)
        require(not os.path.samefile(prefix / filename, target), "payload copy still shares the source inode")
        return target

    def alter_manifest(directory, change):
        path = directory / "manifest.json"
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value) + "\n")

    def reject(name, directory=prefix, one=q1, expected="", existing=False):
        output = results / f"{name}.json"
        marker = b"existing output must remain unchanged\n"
        if existing:
            output.write_bytes(marker)
        command = ["timeout", "--signal=KILL", "300", "bash", str(validate),
                   "lean-executable", str(binary), "fresh-prefix",
                   str(public), str(directory), str(output), "0", "1", str(q0), str(one)]
        # Both deadlines use the project's 300-second native cap. The command
        # timeout survives an outer test stop; the subprocess timeout also kills
        # the complete child group when this test remains active.
        with subprocess.Popen(command, cwd=package, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, start_new_session=True) as child:
            try:
                log, _ = child.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                log, _ = child.communicate()
                (results / f"{name}.log").write_text(log)
                raise AssertionError(f"{name}: child exceeded the 300-second native cap")
        (results / f"{name}.log").write_text(log)
        require(child.returncode != 0 and expected in log,
                f"{name}: expected rejection containing {expected!r}")
        if existing:
            require(output.read_bytes() == marker, f"{name}: existing output was changed")
        else:
            require(not output.exists(), f"{name}: failed input produced an output")
        print(f"binary_prefix_case={name} result=rejected", flush=True)

    directory = clone("noncanonical_first_row")
    with detached(directory, first_file).open("r+b") as stream:
        stream.write(struct.pack("<Q", P))
    reject("noncanonical_first_row", directory, expected="noncanonical Goldilocks word in prefix")

    directory = clone("truncated_payload")
    path = detached(directory, first_file)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    reject("truncated_payload", directory, expected="wrong type or byte count")

    directory = clone("extra_payload_byte")
    with detached(directory, first_file).open("ab") as stream:
        stream.write(b"\x00")
    reject("extra_payload_byte", directory, expected="wrong type or byte count")

    directory = clone("missing_payload")
    (directory / first_file).unlink()
    # The filesystem metadata error names the missing manifest-owned path.
    reject("missing_payload", directory, expected=first_file)

    directory = clone("extra_payload_file")
    (directory / "unexpected.bin").write_bytes(b"")
    reject("extra_payload_file", directory, expected="missing or extra files")

    directory = clone("coverage_gap")
    alter_manifest(directory, lambda value: value[6][0].__setitem__(0, 1))
    reject("coverage_gap", directory, expected="gap, overlap or invalid extent")

    directory = clone("coverage_overlap")
    alter_manifest(directory, lambda value: value[6][1].__setitem__(0, value[6][1][0] - 1))
    reject("coverage_overlap", directory, expected="gap, overlap or invalid extent")

    directory = clone("wrong_profile")
    alter_manifest(directory, lambda value: value.__setitem__(3, value[3] + 1))
    reject("wrong_profile", directory, expected="profile differs from the selected input")

    directory = clone("changed_manifest_challenge")
    alter_manifest(directory, lambda value: value[5][1].__setitem__(0, (value[5][1][0] + 1) % P))
    reject("changed_manifest_challenge", directory, expected="challenges differ from the Lean transcript")

    changed_round = json.loads(q1.read_text())
    changed_round[9][0] = (changed_round[9][0] + 1) % P
    changed_path = results / "changed-q1-claim.json"
    changed_path.write_text(json.dumps(changed_round) + "\n")
    reject("changed_causal_q1_claim", one=changed_path, expected="challenge, state or claim differs")

    reject("preexisting_output", expected="output already exists", existing=True)

    directory = clone("noncanonical_final_nonfirst_row")
    path = detached(directory, final_file)
    with path.open("r+b") as stream:
        stream.seek(-8, os.SEEK_END)
        stream.write(struct.pack("<Q", P))
    reject("noncanonical_final_nonfirst_row", directory,
           expected="noncanonical Goldilocks word in prefix")
    print("piccs_binary_prefix_mutations=passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, IndexError, TypeError) as error:
        sys.exit(str(error))
