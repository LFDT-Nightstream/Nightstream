#!/usr/bin/env python3
"""Reject malformed depth-17 norm inputs without changing original files.

Source provenance and the complete positive run are separate evidence.
The coordinator runs this script under the 300-second native test cap.
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
# The requested retained fixture has 17 sources, depth 17, and 1931 values each.
SOURCES = 17
DEPTH = 17
ROWS = 1931


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(len(sys.argv) == 5 + DEPTH,
            "expected executable, public, norm directory, new results, Q0 through Q16")
    binary, public, prefix, results = [Path(value).resolve() for value in sys.argv[1:5]]
    rounds = [Path(value).resolve() for value in sys.argv[5:]]
    package = Path(__file__).resolve().parents[1]
    validate = package / "scripts" / "validate.sh"
    require(not results.is_relative_to(prefix), "results must be outside the original norm prefix")
    source_names = [f"source-{source}" for source in range(SOURCES)]
    require({entry.name for entry in prefix.iterdir()} == set(source_names),
            "original norm prefix has missing or extra source directories")
    challenges = [json.loads(path.read_text())[5] for path in rounds]
    manifests = []
    for source, name in enumerate(source_names):
        directory = prefix / name
        require(directory.is_dir(), f"{directory}: not a source directory")
        manifest = json.loads((directory / "manifest.json").read_text())
        require(len(manifest) == 7 and manifest[:6] == [1, DEPTH, 3 + source, 1, ROWS, challenges],
                f"{directory}: expected the retained complete depth-17 source")
        filenames = {f"{first}-{finish}.bin" for first, finish in manifest[6]}
        require({entry.name for entry in directory.iterdir()} == {"manifest.json", *filenames},
                f"{directory}: missing or extra payload files")
        require(all((directory / name).is_file() for name in filenames),
                f"{directory}: payload is not a file")
        manifests.append(manifest)
    final_first, final_finish = manifests[-1][6][-1]
    require(final_finish - final_first > 1 and final_finish - 1 >= 2,
            "final mutation needs a non-first row outside arithmetic pair range 0..1")
    final_relative = Path(source_names[-1]) / f"{final_first}-{final_finish}.bin"
    require((prefix / final_relative).stat().st_size == (final_finish - final_first) * 16,
            "original final payload has a wrong byte count")
    results.mkdir(exist_ok=False)

    def clone(name):
        directory = results / name
        directory.mkdir(exist_ok=False)
        for source_name in source_names:
            target = directory / source_name
            target.mkdir(exist_ok=False)
            for path in (prefix / source_name).iterdir():
                if path.name == "manifest.json":
                    shutil.copyfile(path, target / path.name)
                else:
                    os.link(path, target / path.name)
        return directory

    def detach(directory, relative):
        target = directory / relative
        target.unlink()
        shutil.copyfile(prefix / relative, target)
        require(not os.path.samefile(prefix / relative, target), "payload still shares the source inode")
        return target

    def reject(name, directory=prefix, saved_rounds=rounds, expected="", existing=False):
        output = results / f"{name}.json"
        marker = b"existing output must remain unchanged\n"
        if existing:
            output.write_bytes(marker)
        command = ["timeout", "--signal=KILL", "300", "bash", str(validate),
                   "lean-executable", str(binary), "norm-prefix", str(public), str(directory),
                   str(output), "0", "1", *map(str, saved_rounds)]
        # Both deadlines use the project native cap. The command timeout also
        # remains active if the outer test process stops.
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
            require(not output.exists(), f"{name}: rejected input produced an output")
        print(f"norm_field_input_case={name} result=rejected", flush=True)

    directory = clone("missing_source_directory")
    shutil.rmtree(directory / source_names[-1])
    reject("missing_source_directory", directory, expected="missing or extra source directories")

    directory = clone("extra_source_directory")
    (directory / f"source-{SOURCES}").mkdir()
    reject("extra_source_directory", directory, expected="missing or extra source directories")

    directory = clone("swapped_source_identities")
    temporary = directory / "swapping"
    (directory / "source-0").rename(temporary)
    (directory / "source-1").rename(directory / "source-0")
    temporary.rename(directory / "source-1")
    reject("swapped_source_identities", directory, expected="profile differs from the selected input")

    directory = clone("wrong_depth")
    path = directory / "source-0" / "manifest.json"
    value = json.loads(path.read_text())
    value[1] -= 1
    path.write_text(json.dumps(value) + "\n")
    reject("wrong_depth", directory, expected="profile differs from the selected input")

    directory = clone("changed_manifest_challenge")
    path = directory / "source-0" / "manifest.json"
    value = json.loads(path.read_text())
    value[5][-1][0] = (value[5][-1][0] + 1) % P
    path.write_text(json.dumps(value) + "\n")
    reject("changed_manifest_challenge", directory, expected="challenges differ from the Lean transcript")

    directory = clone("noncanonical_final_source_tail")
    with detach(directory, final_relative).open("r+b") as stream:
        stream.seek(-8, os.SEEK_END)
        stream.write(struct.pack("<Q", P))
    reject("noncanonical_final_source_tail", directory, expected="noncanonical Goldilocks word in prefix")

    value = json.loads(rounds[-1].read_text())
    value[9][0] = (value[9][0] + 1) % P
    changed_round = results / "changed-q16-claim.json"
    changed_round.write_text(json.dumps(value) + "\n")
    reject("changed_final_round_claim", saved_rounds=[*rounds[:-1], changed_round],
           expected=f"Lean round {DEPTH - 1} challenge, state or claim differs")

    reject("existing_output", expected="output already exists", existing=True)
    print("piccs_norm_field_inputs=passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, IndexError, TypeError) as error:
        sys.exit(str(error))
