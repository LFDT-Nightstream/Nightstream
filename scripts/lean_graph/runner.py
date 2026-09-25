"""Execute approved gates with project caps and a shared process lock."""

from __future__ import annotations

import datetime
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import builds
from .guard import build_lock, check_build_processes
from .policy import CAPS, checker_key, verify_checker_sources
from .snapshot import (EvidenceError, copy_file, dependency_keys, digest, entries, file_entry,
                       read_json, safe_relative, signature, verify, write_json)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def measured(timings, name):
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.monotonic() - started


def expand(value, work):
    def replace(match):
        kind, name, index = match.groups()
        directory = "inputs" if kind in ("input", "value") else kind
        path = Path(work) / directory / safe_relative(name)
        if kind == "output":
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        if kind == "value":
            result = read_json(path)
            if index is not None:
                result = result[int(index)]
            if not isinstance(result, (str, int)) or isinstance(result, bool):
                raise EvidenceError(f"command value is not a scalar: {name}")
            return str(result)
        if not path.exists():
            raise EvidenceError(f"command input is missing: {name}")
        return str(path)
    return re.sub(r"\{(input|source|value|output):([^:{}]+)(?::([0-9]+))?\}", replace, value)


def completion(text, expected, work=None):
    for pattern in expected.get("patterns", []):
        if re.search(pattern, text, re.MULTILINE) is None:
            raise EvidenceError(f"required completion was not observed: {pattern}")
    for name in expected.get("tests", []):
        if re.search(r"^test " + re.escape(name) + r" \.\.\. ok$", text, re.MULTILINE) is None:
            raise EvidenceError(f"required test did not pass: {name}")
    if expected.get("tests") and re.search(r"test result: ok\. [1-9][0-9]* passed; 0 failed;", text) is None:
        raise EvidenceError("the selected test process did not report completed tests")
    for target, witness in expected.get("closures", {}).items():
        found = []
        for line in text.splitlines():
            if line.startswith("LEAN_GRAPH_CLOSED "):
                found.append(json.loads(line.removeprefix("LEAN_GRAPH_CLOSED ")))
        if {"target": target, "witness": witness} not in found:
            raise EvidenceError(f"Lean did not check the required closure: {target}")
    for left, right in expected.get("equal_files", []):
        if file_entry(expand(left, work)) != file_entry(expand(right, work)):
            raise EvidenceError("generated output differs from the selected input")
    return expected


def runtime_inputs(command, work, environment):
    executable = shutil.which(command["argv"][0], path=environment.get("PATH"))
    paths = [Path(executable).resolve()] if executable else []
    home = Path(environment.get("HOME", str(Path.home())))
    if command["kind"] == "lean":
        project = Path(work) / "source" / command["cwd"]
        selected = (project / "lean-toolchain").read_text().strip()
        folder = selected.replace("/", "--").replace(":", "---")
        runtime = Path(environment.get("ELAN_HOME", home / ".elan")) / "toolchains" / folder
        paths.extend(runtime / "bin" / name for name in ("lean", "lake"))
    elif command["kind"] == "rust":
        selected = (Path(work) / "source/rust-toolchain.toml").read_text()
        match = re.search(r'^channel\s*=\s*"([^"\n]+)"\s*$', selected, re.MULTILINE)
        if not match:
            raise EvidenceError("cannot identify the pinned Rust toolchain")
        root = Path(environment.get("RUSTUP_HOME", home / ".rustup")) / "toolchains"
        runtimes = [path for path in root.glob(match[1] + "*") if (path / "bin/rustc").is_file()]
        if len(runtimes) != 1:
            raise EvidenceError("the pinned Rust toolchain must resolve to one installed runtime")
        paths.extend(runtimes[0] / "bin" / name for name in ("cargo", "rustc"))
    return {str(path.resolve()): file_entry(path.resolve()) for path in paths}


def command_environment(command):
    # Record the effective environment without forwarding credentials or build injection.
    inherited = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT",
                 "ELAN_HOME", "RUSTUP_HOME", "CARGO_HOME")
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment["RUSTC_WRAPPER"] = ""
    cap = command.get("cap_seconds", CAPS[command["kind"]])
    if command["kind"] == "lean":
        environment["LEAN_TIMEOUT_SECONDS"] = str(cap)
    return environment


def build_context(command, work, dependencies, policy):
    if command["kind"] != "lean" or command["argv"][:3] != ["bash", "scripts/validate.sh", "build"]:
        return None
    environment = command_environment(command)
    return {"dependencies": dependencies, "command": command, "settings": environment,
            "runtime": runtime_inputs(command, work, environment),
            "threads": os.sysconf("SC_NPROCESSORS_ONLN"),
            "policy": digest(policy), "checker": checker_key()}


def execute(command, work, log_path):
    argv = [expand(value, work) for value in command["argv"]]
    cwd = Path(work) / "source" / safe_relative(command["cwd"])
    environment = command_environment(command)
    cap = command.get("cap_seconds", CAPS[command["kind"]])
    runtime = runtime_inputs(command, work, environment)
    started, clock = now(), time.monotonic()
    outcome, reason, process = "failed", None, None
    stdin_data = None
    if "stdin_json" in command:
        def substitute(value):
            if isinstance(value, str):
                if re.fullmatch(r"\{value:[^:{}]+(?::[0-9]+)?\}", value):
                    return json.loads(expand(value, work))
                return expand(value, work)
            if isinstance(value, list):
                return [substitute(item) for item in value]
            if isinstance(value, dict):
                return {key: substitute(item) for key, item in value.items()}
            return value
        stdin_data = json.dumps(substitute(command["stdin_json"])).encode()
    handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}")
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, interrupted)
        with Path(log_path).open("wb") as output:
            process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                                       start_new_session=True)
            try:
                process.communicate(stdin_data, timeout=cap)
                if process.returncode == 0:
                    completion(Path(log_path).read_text(errors="replace"), command["completion"], work)
                    outcome = "pass"
                else:
                    reason = f"exit {process.returncode}"
            except subprocess.TimeoutExpired:
                outcome, reason = "timed-out", f"project command cap: {cap} seconds"
            except InterruptedError as error:
                outcome, reason = "interrupted", str(error)
            finally:
                # Also remove descendants of an exited parent before releasing the lock.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    except (OSError, EvidenceError, ValueError) as error:
        reason = str(error)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    elapsed = time.monotonic() - clock
    if elapsed > cap:
        outcome, reason = "timed-out", f"project command cap: {cap} seconds"
    if any(file_entry(Path(path)) != expected for path, expected in runtime.items()):
        outcome, reason = "failed", "checker runtime changed during execution"
    return {"command": argv, "cwd": str(cwd), "settings": environment, "runtime": runtime,
            "started": started, "finished": now(), "elapsed_seconds": elapsed,
            "cap_seconds": cap, "exit": process.returncode if process else None,
            "outcome": outcome, "reason": reason,
            "completion": command["completion"] if outcome == "pass" else None}


def run_gate(name, policy, manifest, snapshot, store, authority=None):
    gate = policy["gates"][name]
    dependencies = dependency_keys(manifest, gate, policy)
    missing = [key for key, value in dependencies.items() if value is None]
    if missing:
        raise EvidenceError("missing required dependencies: " + ", ".join(missing))
    started, clock, timings = now(), time.monotonic(), {}
    with build_lock(store):
        with measured(timings, "preflight"):
            if any(command["kind"] in ("lean", "rust") for command in gate["commands"]):
                check_build_processes()
        with measured(timings, "snapshot_verification"):
            verify(snapshot, manifest)
        if authority:
            verify_checker_sources(snapshot)
        run_dir = Path(store) / "runs" / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        record = {"schema": 1, "gate": name, "gate_key": digest(gate),
                  "policy": digest(policy), "checker": checker_key(),
                  "snapshot": digest(manifest), "dependencies": dependencies,
                  "started": started, "outcome": "incomplete", "commands": [], "artifacts": {},
                  "timings_seconds": timings}
        if gate.get("declaration_freshness"):
            from .records import declaration_keys, declaration_records, read_runs
            runs, _ = read_runs(store, authority)
            precise = declaration_keys(gate["declaration_freshness"],
                                       declaration_records(policy, manifest, runs))
            if precise:
                record["declarations"] = precise
        # A killed runner leaves this non-passing record, never an implicit success.
        write_json(run_dir / "result.json", authority.sign(record) if authority else {"record": record})
        try:
            with ExitStack() as staging:
                temporary = staging.enter_context(tempfile.TemporaryDirectory(prefix="lean-graph-check-"))
                retained_builds = []
                work = Path(temporary).resolve()
                with measured(timings, "preparation"):
                    shutil.copytree(snapshot / "source", work / "source", symlinks=True,
                                    copy_function=copy_file)
                    if (snapshot / "inputs").exists():
                        shutil.copytree(snapshot / "inputs", work / "inputs", symlinks=True,
                                        copy_function=copy_file)
                    seed = work / "inputs/library_seed"
                    if seed.exists() and any(command["kind"] == "lean" for command in gate["commands"]):
                        # Lake may update only this disposable dependency copy.
                        installed = work / "source/formal/nightstream-fprime/.lake/packages"
                        if installed.exists():
                            shutil.rmtree(installed)
                        shutil.copytree(seed, installed, symlinks=True, copy_function=copy_file)
                    before = {name: signature(work / name) for name in entries(manifest)}
                for index, command in enumerate(gate["commands"]):
                    log_name = f"command-{index}.log"
                    with measured(timings, "cache_restore"):
                        context = build_context(command, work, dependencies, policy)
                        cache_state = builds.restore(context, work, store, authority) if context else None
                    with measured(timings, "commands"):
                        result = execute(command, work, run_dir / log_name)
                    if context:
                        result["build_cache"] = {"key": digest(context), "state": cache_state}
                        if result["outcome"] == "pass":
                            if (result["runtime"] != context["runtime"] or
                                    result["settings"] != context["settings"]):
                                raise EvidenceError("build context changed during execution")
                            retained = staging.enter_context(tempfile.TemporaryDirectory(
                                prefix="pending-", dir=builds.cache_root(store, authority)))
                            with measured(timings, "cache_capture"):
                                built = builds.stage(context, work, retained)
                            if built:
                                retained_builds.append((retained, built))
                    record["commands"].append(result)
                    record["artifacts"][log_name] = file_entry(run_dir / log_name)
                    if command["kind"] == "lean":
                        from .metadata import from_log
                        with measured(timings, "metadata"):
                            metadata = from_log(run_dir / log_name,
                                                work / "source" / command["cwd"], manifest)
                        if metadata:
                            if any(not item["complete"] for item in metadata):
                                result.update(outcome="incomplete", completion=None,
                                              reason="declaration metadata has incomplete provenance or dependencies")
                            metadata_name = f"metadata-{index}.json"
                            write_json(run_dir / metadata_name, metadata)
                            record["artifacts"][metadata_name] = file_entry(run_dir / metadata_name)
                            # Retain the exact raw export in compressed form. The graph
                            # record keeps fingerprints and edges, not repeated term bodies.
                            compressed = run_dir / (log_name + ".gz")
                            with (run_dir / log_name).open("rb") as raw, gzip.open(compressed, "wb") as output:
                                shutil.copyfileobj(raw, output)
                            (run_dir / log_name).unlink()
                            del record["artifacts"][log_name]
                            record["artifacts"][compressed.name] = file_entry(compressed)
                    if result["outcome"] != "pass":
                        record["outcome"] = result["outcome"]
                        break
                else:
                    record["outcome"] = "pass"
                with measured(timings, "input_validation"):
                    # Byte and inode/time checks also detect edits restored before exit.
                    for name, expected in entries(manifest).items():
                        actual = file_entry(work / name)
                        if (actual["sha256"] != expected["sha256"] or
                                signature(work / name) != before[name]):
                            raise EvidenceError(f"checked input changed: {name}")
                    from .snapshot import inspect
                    input_paths = {name: str(work / "inputs" / name)
                                   for name, value in manifest["inputs"].items() if value is not None}
                    current, _ = inspect(work / "source", policy, input_paths, {
                        "sources": list(manifest["sources"]), "inputs": list(manifest["inputs"])})
                    def content(values):
                        return {name: (value["sha256"], value["bytes"])
                                for name, value in entries(values).items()}
                    if content(current) != content(manifest):
                        raise EvidenceError("checked source or input file set changed")
                for output in sorted((work / "output").glob("*")):
                    if output.is_file():
                        name = "output-" + output.name
                        shutil.copyfile(output, run_dir / name)
                        record["artifacts"][name] = file_entry(run_dir / name)
                with measured(timings, "snapshot_verification"):
                    verify(snapshot, manifest)
                if record["checker"] != checker_key():
                    raise EvidenceError("checker changed during execution")
                if record["outcome"] == "pass":
                    with measured(timings, "cache_publish"):
                        for retained, built in retained_builds:
                            builds.publish(retained, built, store, authority)
        except (EvidenceError, OSError, InterruptedError, KeyboardInterrupt) as error:
            record["outcome"], record["reason"] = "failed", str(error)
        record["finished"] = now()
        record["elapsed_seconds"] = time.monotonic() - clock
        timings["other"] = max(0.0, record["elapsed_seconds"] - sum(timings.values()))
        write_json(run_dir / "result.json", authority.sign(record) if authority else {"record": record})
        return record
