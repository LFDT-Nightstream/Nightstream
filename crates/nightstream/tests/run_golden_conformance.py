#!/usr/bin/env python3
"""Run the selected fresh native folds, then compare their complete outputs.

Each native phase and comparison has the repository's 300-second process cap.
The coordinator runs phases in order; its total time includes all phases.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from lean_graph.policy import CAPS  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def file_identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "changes": git("status", "--porcelain").decode().splitlines(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "untracked_files": [file_identity(ROOT / name) for name in untracked if name],
    }


def phase(binary, directory, name, commands, **arguments):
    request = {"phase": name, "directory": str(directory), **arguments}
    parts = [name] + [f"{key}-{arguments[key]}" for key in ("step", "child") if key in arguments]
    receipt = directory / "logs" / ("-".join(parts) + ".json")
    require(not receipt.exists(), f"phase receipt already exists: {receipt}")
    command = ["timeout", "--signal=KILL", str(CAPS["rust"]), sys.executable, "-B",
               str(TESTS / "run_recursive_phase.py"), "--binary", str(binary),
               "--directory", str(directory), "--phase", name]
    for key, value in arguments.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    commands.append({"command": command, "receipt": str(receipt)})
    print(f"native conformance phase: {' '.join(parts)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    result = read(receipt)
    require(result.get("request") == request, f"wrong phase request: {receipt}")
    require(result.get("outcome") == "passed" and result.get("exit") == 0
            and result.get("process_exit") == 0, f"phase did not pass: {receipt}")
    require(result.get("cap_seconds") == CAPS["rust"], f"wrong phase cap: {receipt}")


def compare_fold(directory, fold, reference, commands, engine=None, receipt=None):
    receipt = receipt or directory / f"comparison-fold-{fold}.json"
    require(not receipt.exists(), f"comparison receipt already exists: {receipt}")
    command = ["timeout", "--signal=KILL", str(CAPS["python"]), sys.executable, "-B",
               str(TESTS / "compare_recursive_outputs.py"), "--directory", str(directory),
               "--fold", str(fold), "--receipt", str(receipt)]
    if reference is not None:
        command += ["--reference", str(reference)]
    if engine is not None:
        command += ["--engine-directory", str(engine)]
    commands.append({"command": command, "receipt": str(receipt)})
    subprocess.run(command, cwd=ROOT, check=True)
    result = read(receipt)
    require(result.get("outcome") == "passed" and result.get("fold") == fold
            and result.get("run_directory") == str(directory)
            and result.get("reference_directory") == (str(reference) if reference else None),
            f"comparison did not pass: {receipt}")
    if engine is not None:
        compared = result.get("later_fold", {}).get("engine_comparisons", [])
        require([entry.get("engine_directory") for entry in compared] == [str(engine)],
                f"missing engine comparison: {receipt}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--engine", choices=("optimized", "metal"), required=True)
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    if (args.engine == "metal") != (args.cpu_reference is not None):
        parser.error("Metal requires --cpu-reference; optimized does not use it")
    binary, directory, references = args.binary.resolve(), args.directory.resolve(), args.references.resolve()
    cpu = args.cpu_reference.resolve() if args.cpu_reference else None
    require(binary.is_file(), f"missing test binary: {binary}")
    for name in ("reference-first", "reference-later"):
        require((references / name).is_dir(), f"missing restored reference: {references / name}")
    if cpu is not None:
        for step in (1, 2):
            require((cpu / f"fold-{step}/proof.native").is_file(), f"missing CPU proof for fold {step}")
    require(not directory.exists(), f"run directory already exists: {directory}")
    # Capture custody before creating any run outputs. These hashes identify
    # files; they do not prove build correctness or protocol semantics.
    record = {"schema": 1, "outcome": "running", "engine": args.engine,
              "binary": file_identity(binary), "source": source_identity(), "commands": [],
              "references": str(references), "cpu_reference": str(cpu) if cpu else None,
              "identity_scope": "file custody only; not proof of binary/source correspondence",
              "scope": "Fresh selected native two-fold execution, terminal acceptance and mutation rejection, "
                       "with archive output comparisons. No fresh Lean execution or universal correctness claim."}
    directory.mkdir(parents=True)
    try:
        phase(binary, directory, "base", record["commands"], engine=args.engine)
        for step in (1, 2):
            phase(binary, directory, "sources", record["commands"], step=step, engine=args.engine)
            if args.engine == "optimized":
                phase(binary, directory, "ccs", record["commands"], step=step, engine=args.engine)
                phase(binary, directory, "rlc", record["commands"], step=step)
                phase(binary, directory, "split", record["commands"], step=step)
                phase(binary, directory, "openings", record["commands"], step=step)
                phase(binary, directory, "nifs", record["commands"], step=step)
            else:
                phase(binary, directory, "prove", record["commands"], step=step, engine=args.engine,
                      reference_proof=str(cpu / f"fold-{step}/proof.native"))
            phase(binary, directory, "successor", record["commands"], step=step, engine=args.engine)
        phase(binary, directory, "terminal", record["commands"], step=3, engine=args.engine)
        phase(binary, directory, "mutation", record["commands"], step=3)
        phase(binary, directory, "reject", record["commands"], step=3, engine=args.engine)
        if cpu is None:
            for fold, name in ((1, "reference-first"), (2, "reference-later")):
                compare_fold(directory, fold, references / name, record["commands"])
        else:
            compare_fold(cpu, 2, references / "reference-later", record["commands"],
                         engine=directory, receipt=directory / "comparison-lean-cpu-engine.json")
        record["outcome"] = "passed"
        print(f"native golden conformance passed: {directory}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        record.update(outcome="failed", error=str(error))
        print(f"native golden conformance failed: {error}", file=sys.stderr, flush=True)
    finally:
        with (directory / "conformance.json").open("x") as output:
            json.dump(record, output, indent=2)
            output.write("\n")
    return 0 if record["outcome"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        sys.exit(str(error))
