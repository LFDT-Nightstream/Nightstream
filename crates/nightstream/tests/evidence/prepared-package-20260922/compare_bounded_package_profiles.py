#!/usr/bin/env python3
"""Compare only the completed bounded-package CPU/Metal captures and their exports.

Run after both recorders and watchdogs have ended and TOC/thermal XML exists.
This script starts no workload or export. It reads the frozen image UUID with
dwarfdump and creates one new comparison file only after every check passes.
"""
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "bounded-package-benchmark-binary"
PACKAGE = ROOT / "bounded-poseidon2.nsc"
SCOPE = "prepared_package_load_prove_verify"
UUID = "7DCAD994-817D-33E5-95BB-666FA9543762"
CAP = 1800  # Root AGENTS.md: recording, target execution and finalization.
LABELS = {"metal": "bounded-package-metal-lifecycle-profile-retry", "cpu": "bounded-package-cpu-lifecycle-profile"}
ENGINES = {"metal": "metal", "cpu": "optimized"}
OUTPUT = ROOT / "bounded-package-matched-time-profile-comparison.json"
SET_ATTRIBUTES = {"codes", "subsystem", "category", "dynamic-tracing-enabled-subsystems"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def seconds(value, location):
    result = Decimal(str(value))
    require(result.is_finite() and result > 0, f"Invalid elapsed time: {location}")
    return result


def attributes(element):
    return {key: sorted(shlex.split(value)) if key in SET_ATTRIBUTES else value
            for key, value in element.attrib.items()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_toc(path):
    result = {"tables": [], "instrument_settings": [], "processes": []}
    stack = []
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            continue
        tags = [item.tag for item in stack]
        if tags == ["trace-toc", "run"]:
            require(element.get("number") == "1" and "run_number" not in result, "TOC must have exactly one run")
            result["run_number"] = 1
        elif tags[-2:] == ["target", "device"]:
            require("device" not in result, "Multiple target devices")
            result["device"] = dict(element.attrib)
        elif tags[-2:] == ["target", "process"]:
            require("target_process" not in result, "Multiple target processes")
            result["target_process"] = dict(element.attrib)
        elif tags[-2:] == ["processes", "process"]:
            result["processes"].append(dict(element.attrib))
        elif len(tags) > 1 and tags[-2] == "summary" and len(element) == 0:
            if element.text and element.text.strip():
                result[element.tag] = element.text.strip()
        if element.tag == "table":
            result["tables"].append(attributes(element))
        settings_tag = next((tag for tag in tags if tag in {
            "intruments-recording-settings", "instruments-recording-settings"}), None)
        if settings_tag:
            start = tags.index(settings_tag)
            result["instrument_settings"].append({
                "path": [{"tag": item.tag, "attributes": dict(item.attrib)} for item in stack[start:]],
                "text": (element.text or "").strip(),
            })
        if len(stack) > 1:
            stack[-2].remove(element)
        stack.pop()
        element.clear()
    require("run_number" in result and "device" in result and "target_process" in result, "Incomplete TOC")
    result["tables"].sort(key=canonical)
    result["instrument_settings"].sort(key=canonical)
    require(any(table.get("schema") == "time-profile" for table in result["tables"]), "Time Profiler data absent")
    require(any(table.get("schema") == "time-sample" for table in result["tables"]), "Time-sample settings absent")
    return result


def read_thermal(path, duration):
    definitions, rows = {}, []
    stack = []
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            continue
        if element.tag == "row":
            for child in element:
                if child.get("id"):
                    definitions[(child.tag, child.get("id"))] = child.text

            def value(child):
                require(child is not None, "Incomplete thermal row")
                return child.text if child.text is not None else definitions[(child.tag, child.get("ref"))]

            times = element.findall("start-time")
            require(len(times) == 2, "Thermal interval needs start and end")
            row = {"start_ns": int(value(times[0])), "end_ns": int(value(times[1])),
                   "duration_ns": int(value(element.find("duration"))),
                   "state": value(element.find("thermal-state")),
                   "induced": bool(int(value(element.find("boolean"))))}
            require(row["start_ns"] == (rows[-1]["end_ns"] if rows else 0), "Gap or overlap in thermal coverage")
            require(row["duration_ns"] >= 0 and row["end_ns"] - row["start_ns"] == row["duration_ns"],
                    "Invalid thermal interval duration")
            rows.append(row)
            stack[-2].remove(element)
            element.clear()
        stack.pop()
    require(rows, "Thermal export has no intervals")
    # Compare at the precision printed by the TOC, rather than invent a tolerance.
    quantum = Decimal(1).scaleb(duration.as_tuple().exponent)
    require((Decimal(rows[-1]["end_ns"]) / Decimal(10**9)).quantize(quantum) == duration,
            "Thermal intervals do not cover the complete recorded duration")
    return rows


def option(command, name):
    require(command.count(name) == 1, f"Expected one {name} option")
    return command[command.index(name) + 1]


def read_events(label, run, engine):
    events = []
    for line in (ROOT / (label + "-benchmark.log")).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    require(events == run["events"], f"Benchmark log and receipt disagree: {label}")
    require(len(events) >= 2 and events[0].get("event") == "benchmark_started"
            and events[-1].get("event") == "benchmark_finished", f"Incomplete lifecycle: {label}")
    start, finish = events[0], events[-1]
    require(start.get("schema") == finish.get("schema") == 2, "Expected bounded-package schema2 events")
    require(start.get("timing_scope") == finish.get("timing_scope") == SCOPE, "Unexpected native timing scope")
    require(start.get("package") == str(PACKAGE), "Native event uses a different prepared package")
    require(start.get("benchmark") == "poseidon2_hash_chain_v1", "Unexpected benchmark")
    require(start["engine"] == engine and start["steps"] == finish["steps"] == 3, "Unexpected engine or lifecycle size")
    require(start["profile"] == {"b": 2, "k_rho": 16, "B": 65536}, "Unexpected frozen production profile")
    require(finish["verified"] is True, f"Native verification failed: {label}")
    phases, active = [], None
    for event in events[1:-1]:
        key = (event.get("phase"), event.get("step"))
        if event.get("event") == "phase_started":
            require(active is None, "Overlapping lifecycle phases")
            active = key
        elif event.get("event") == "phase_finished":
            require(active == key, "Unmatched lifecycle phase completion")
            seconds(event["seconds"], f"{label}:{key}")
            phases.append(event)
            active = None
        else:
            raise ValueError(f"Unexpected lifecycle event: {event}")
    require(active is None, "Unfinished lifecycle phase")
    expected = [("load", None), ("prove", 1), ("extend", 2), ("extend", 3), ("verify", 3)]
    require([(p["phase"], p.get("step")) for p in phases] == expected, "Incomplete or reordered lifecycle phases")
    total = seconds(finish["seconds"], label)
    require(sum(Decimal(str(p["seconds"])) for p in phases) <= total, "Phase times exceed native lifecycle time")
    return start, finish, phases


def memory_scope(run):
    fields, samples = run["sample_fields"], run["samples"]
    require(samples and len(set(fields)) == len(fields), "Missing or ambiguous memory samples")
    positions = {name: fields.index(name) for name in ["elapsed_seconds", "kernel_peak_resident_bytes", "lifetime_peak_footprint_bytes"]}
    require(all(len(sample) == len(fields) for sample in samples), "Malformed memory sample")
    times = [sample[positions["elapsed_seconds"]] for sample in samples]
    require(all(a <= b for a, b in zip(times, times[1:])), "Memory sample clock moved backwards")
    rss = max(sample[positions["kernel_peak_resident_bytes"]] for sample in samples)
    footprint = max(sample[positions["lifetime_peak_footprint_bytes"]] or 0 for sample in samples)
    require(rss == run["observed_kernel_peak_resident_bytes"], "RSS peak disagrees with live samples")
    require(footprint == run["peak_physical_footprint_bytes"], "Footprint peak disagrees with live samples")
    return {"observed_kernel_lifetime_peak_resident_bytes": rss,
            "observed_lifetime_peak_physical_footprint_bytes": footprint,
            "final_kernel_peak_after_exit": run["final_kernel_peak_after_exit"],
            "guard_bytes": run["memory_guard_bytes"], "scope": run["rss_scope"],
            "sample_fields": fields, "sample_count": len(samples),
            "first_sample_elapsed_seconds": times[0], "last_sample_elapsed_seconds": times[-1],
            "full_run_receipt": str(ROOT / (run["state"]["label"] + ".json"))}


def read_capture(role):
    label, engine = LABELS[role], ENGINES[role]
    path = ROOT / (label + ".json")
    run = load_json(path)
    state = load_json(ROOT / (label + "-state.json"))
    deadline = load_json(ROOT / (label + "-deadline.json"))
    require(run["state"] == state and state["label"] == label and state["engine"] == engine, "Capture state mismatch")
    require(run["outcome"] == "passed" and run["exit"] == 0 and run["error"] is None and run["verified"] is True,
            f"Capture did not pass: {label}")
    require(run["timing_scope"] == SCOPE and state["package"] == str(PACKAGE), "Wrong saved-package timing scope")
    for key, expected in {
        "one_time_compile_included": False, "prepared_package_load_included": True,
        "terminal_verification_included": True, "preparation_time_subtracted": False,
        "directly_comparable_to_historical_compile_inclusive_results": False,
    }.items():
        require(run[key] is expected, f"Unexpected scope flag: {key}")
    require(run["package_unchanged"] is True and run["package_before"] == run["package_after"],
            "Prepared package metadata changed during the run")
    require(run["package_before"]["path"] == str(PACKAGE), "Wrong package path in file metadata")
    require(state["recorder_finished"] is True and deadline["outcome"] == "processes-ended"
            and deadline["stopped_pids"] == [], f"Capture or watchdog is incomplete: {label}")
    require(run["time_cap_seconds"] == deadline["cap_seconds"] == CAP, "Incorrect profiling cap")
    require("timeout --signal=KILL 1800 " in state["controller_command"], "Outer cap is not recorded")
    for key in ["controller_pid", "target_pid", "recorder_pid", "deadline_unix_seconds"]:
        require(state[key] == deadline[key], f"Watchdog state differs: {key}")
    command = run["command"]
    require(command[:3] == ["xcrun", "xctrace", "record"], "Unexpected recorder")
    require(option(command, "--template") == "Time Profiler" and option(command, "--time-limit") == "1800s",
            "Unexpected recorder settings")
    require(option(command, "--output") == state["trace_path"] == str(ROOT / (label + ".trace")), "Trace path mismatch")
    require(option(command, "--target-stdout") == str(ROOT / (label + "-benchmark.log")), "Benchmark log path mismatch")
    launch = command.index("--launch")
    require(command[launch + 1:] == ["--", str(BINARY), "run", "--package", str(PACKAGE), "--engine", engine, "--steps", "3"], "Unexpected launched executable or arguments")
    require(state["target_command"] == f"{BINARY} run --package {PACKAGE} --engine {engine} --steps 3", "Native command differs")
    normalized_command = list(command)
    for name in ["--output", "--target-stdout", "--engine"]:
        normalized_command[normalized_command.index(name) + 1] = f"<{name}>"
    toc = read_toc(ROOT / (label + "-toc.xml"))
    target = toc["target_process"]
    require(target["type"] == "launched" and target["name"] == BINARY.name
            and int(target["pid"]) == state["target_pid"], "TOC target mismatch")
    require(target["return-exit-status"] == "0" and target["termination-reason"] == "exit(0)"
            and toc["end-reason"] == "Target app exited", "Native target did not exit normally")
    target_arguments = ["run", "--package", str(PACKAGE), "--engine", engine, "--steps", "3"]
    # Launch argv and the native package event are checked independently. Accept
    # exact TOC argument text with shell quoting or with the spaces left literal.
    require(shlex.split(target["arguments"]) == target_arguments
            or target["arguments"] == " ".join(target_arguments), "TOC target arguments differ")
    require(any(p.get("pid") == target["pid"] and p.get("path") == str(BINARY) for p in toc["processes"]),
            "TOC process path differs from the frozen executable")
    start_time, end_time = (dt.datetime.fromisoformat(toc[key]) for key in ["start-date", "end-date"])
    require(start_time.tzinfo is not None and end_time.tzinfo is not None and start_time < end_time, "Invalid capture timestamps")
    deadline_time = Decimal(str(state["deadline_unix_seconds"]))
    controller_start = dt.datetime.fromtimestamp(float(deadline_time - CAP), tz=start_time.tzinfo)
    require(controller_start.strftime("%a %b %d %H:%M:%S %Y") == state["controller_start"], "Deadline differs from the recorded controller start")
    require(controller_start <= start_time and Decimal(str(end_time.timestamp())) <= deadline_time, "Trace lies outside its deadline")
    receipt_time = Decimal(path.stat().st_mtime_ns) / Decimal(10**9)
    require(Decimal(str(end_time.timestamp())) <= receipt_time <= deadline_time, "Final receipt lies outside the capture deadline")
    duration = seconds(toc["duration"], "TOC duration")
    start, finish, phases = read_events(label, run, engine)
    require(seconds(finish["seconds"], "native duration") <= duration <= seconds(run["elapsed_monotonic_seconds"], "controller duration") <= CAP,
            "Native, trace or controller duration is inconsistent")
    require(seconds(run["elapsed_wall_seconds"], "controller wall duration") <= CAP, "Controller wall time exceeds cap")
    return {"label": label, "run": run, "toc": toc, "deadline": deadline,
            "thermal": read_thermal(ROOT / (label + "-thermal.xml"), duration),
            "start": start, "finish": finish, "phases": phases, "memory": memory_scope(run),
            "start_time": start_time, "end_time": end_time, "receipt_unix_seconds": receipt_time,
            "recording_command": normalized_command}


def main():
    require(re.fullmatch(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}", UUID),
            "Root must fill the frozen bounded-package executable UUID after building")
    require(not OUTPUT.exists(), f"Refuse to overwrite {OUTPUT}")
    captures = {role: read_capture(role) for role in LABELS}
    metal, cpu = captures["metal"], captures["cpu"]
    require(metal["end_time"] <= cpu["start_time"], "Capture intervals overlap or are out of order")
    require(metal["receipt_unix_seconds"] <= Decimal(str(cpu["start_time"].timestamp())),
            "CPU recording began before the Metal completion receipt")
    for key in ["device", "instruments-version", "template-name", "recording-mode", "time-limit", "tables", "instrument_settings"]:
        require(cpu["toc"][key] == metal["toc"][key], f"Recording/device mismatch: {key}")
    require(cpu["toc"]["template-name"] == "Time Profiler" and cpu["toc"]["recording-mode"] == "Deferred", "Unexpected profiling method")
    require(cpu["recording_command"] == metal["recording_command"], "Recorder commands differ")
    require(cpu["run"]["power_control"] == metal["run"]["power_control"], "Power controls differ")
    require(cpu["run"]["memory_guard_bytes"] == metal["run"]["memory_guard_bytes"], "Memory guards differ")
    require(cpu["run"]["package_before"] == metal["run"]["package_before"], "Saved package file metadata differs")
    package_stat = PACKAGE.stat()
    package_current = {"path": str(PACKAGE), "device": package_stat.st_dev, "inode": package_stat.st_ino,
                       "bytes": package_stat.st_size, "mtime_ns": package_stat.st_mtime_ns,
                       "ctime_ns": package_stat.st_ctime_ns}
    require(package_current == metal["run"]["package_before"], "Saved package metadata changed after captures")
    require(Decimal(package_stat.st_mtime_ns) / Decimal(10**9)
            <= Decimal(str(metal["start_time"].timestamp())), "Saved package changed after recording began")
    require({k: v for k, v in cpu["start"].items() if k != "engine"}
            == {k: v for k, v in metal["start"].items() if k != "engine"}, "Complete start records differ")
    require({k: v for k, v in cpu["finish"].items() if k != "seconds"}
            == {k: v for k, v in metal["finish"].items() if k != "seconds"}, "Complete finish records differ")
    image = load_json(ROOT / "bounded-package-benchmark-image.json")
    require(Path(image["image"]) == BINARY and re.search(r"UUID:\s*" + re.escape(UUID) + r"\s+\(arm64\)", image["uuid"]),
            "Frozen image receipt has the wrong path or UUID")
    before = BINARY.stat()
    require(Decimal(before.st_mtime_ns) / Decimal(10**9) <= Decimal(str(metal["start_time"].timestamp())), "Saved image changed after captures began")
    uuid_command = ["xcrun", "dwarfdump", "--uuid", str(BINARY)]
    uuid_output = subprocess.run(uuid_command, capture_output=True, text=True, check=True).stdout
    require((UUID, "arm64") in {(u.upper(), a) for u, a in re.findall(r"UUID: ([0-9a-fA-F-]+) \(([^)]+)\)", uuid_output)},
            "Saved executable UUID differs")
    after = BINARY.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Saved image changed during inspection")
    nominal = all(row["state"] == "Nominal" and not row["induced"] for capture in captures.values() for row in capture["thermal"])
    result = {
        "schema": 2, "scope": "Matched bounded-package load, three-step proof lifecycle and terminal verification under Time Profiler; not an uninstrumented speed ratio.",
        "timing_scope": SCOPE, "one_time_compile_included": False,
        "prepared_package_load_included": True, "terminal_verification_included": True,
        "preparation_time_subtracted": False,
        "directly_comparable_to_historical_compile_inclusive_results": False,
        "historical_baseline_scope": "Schema1 measured reference read, frontend construction and preparation inside the lifecycle total.",
        "same_saved_package": str(PACKAGE), "package_file_metadata": package_current,
        "package_verification_scope": "Same caller-selected path and unchanged device/inode/size/mtime/ctime before and after both captures; no content authentication or whole-artifact digest replay.",
        "cpu_seconds": cpu["finish"]["seconds"], "metal_seconds": metal["finish"]["seconds"],
        "cpu_over_metal": cpu["finish"]["seconds"] / metal["finish"]["seconds"],
        "both_verified": True, "same_start_fields_except_engine": True, "same_finish_fields_except_seconds": True,
        "same_saved_executable": str(BINARY), "saved_native_uuid": UUID,
        "image_verification": {"command": uuid_command, "output": uuid_output, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                               "scope": "Frozen image receipt, current saved-image UUID, and both recorded launch paths; TOC does not itself expose an image UUID."},
        "source_patch": image["source_patch"], "new_source_archive": image["new_source_archive"],
        "start_records": {role: c["start"] for role, c in captures.items()},
        "finish_records": {role: c["finish"] for role, c in captures.items()},
        "phases": [{"phase": c["phase"], "step": c.get("step"), "cpu_seconds": c["seconds"], "metal_seconds": m["seconds"],
                    "cpu_over_metal": c["seconds"] / m["seconds"]} for c, m in zip(cpu["phases"], metal["phases"])],
        "captures": {role: c["toc"] for role, c in captures.items()},
        "serial_captures": True, "metal_finalized_before_cpu_recording": True,
        "same_recording_schemas_settings_and_device": True, "normalized_recording_command": cpu["recording_command"],
        "thermal_intervals": {role: c["thermal"] for role, c in captures.items()}, "both_nominal_noninduced": nominal,
        "controller_elapsed_monotonic_seconds": {role: c["run"]["elapsed_monotonic_seconds"] for role, c in captures.items()},
        "controller_elapsed_wall_seconds": {role: c["run"]["elapsed_wall_seconds"] for role, c in captures.items()},
        "capture_cap_seconds": CAP, "capture_cap_authority": "Root AGENTS.md 30-minute Instruments cap, including target execution and trace finalization.",
        "deadline_checks": {role: {"watchdog": c["deadline"], "completion_receipt_unix_seconds": str(c["receipt_unix_seconds"]),
                                  "within_cap": True} for role, c in captures.items()},
        "memory": {role: c["memory"] for role, c in captures.items()},
        "power_control": {role: c["run"]["power_control"] for role, c in captures.items()},
        "limits": ["One matched instrumented pair. No ratio across images, against raw Metal timing, or against historical compile-inclusive results is calculated. No preparation time is subtracted.",
                   "Complete benchmark start/finish equality does not compare intermediate serialized proofs.",
                   "Live-observed RSS and footprint peaks have separate scopes; a null final post-exit peak leaves the last unsampled interval unproved.",
                   "Thermal records do not establish equal CPU/GPU clock frequencies or GPU limiter state.",
                   "This comparison reads TOC and thermal exports only and makes no new CPU or GPU sample-attribution claim."],
    }
    if not nominal:
        result["limits"].append("At least one thermal interval was not Nominal and non-induced; inspect the full thermal records before interpreting the ratio.")
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ["cpu_seconds", "metal_seconds", "cpu_over_metal", "both_verified", "both_nominal_noninduced"]}, indent=2))
    print("Saved", OUTPUT)


if __name__ == "__main__":
    require(len(sys.argv) == 1, "Usage: python3 compare_bounded_package_profiles.py")
    main()
