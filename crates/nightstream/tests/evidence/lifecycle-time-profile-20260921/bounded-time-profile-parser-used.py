#!/usr/bin/env python3
"""Stream one saved xctrace Time Profiler export; retain no timestamp table."""
import collections
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE_UUID = "D66506D5-8D84-3D01-8EEB-E21332765BD7"


def ident(element):
    return element.get("ref") or element.get("id")


def main(prefix, target_pid):
    assert Path(prefix).name == prefix, "The label must be one private file-name component."
    SOURCE = ROOT / f"{prefix}-samples.xml"
    OUTPUT = ROOT / f"{prefix}-data.json"
    started = time.monotonic()
    source_stat = SOURCE.stat()
    pids, processes, weights, binaries, frames, backtraces = {}, {}, {}, {}, {}, {}
    process_rows = collections.Counter()
    process_weights = collections.Counter()
    target_backtraces = {}
    weight_distribution = collections.Counter()
    total_rows = target_rows = target_weight = sentinel_rows = sentinel_weight = 0
    other_missing_stack_rows = other_missing_stack_weight = 0
    time_reference_rows = 0
    minimum_time = maximum_time = None
    stack = []
    for event, element in ET.iterparse(SOURCE, events=("start", "end")):
        if event == "start":
            stack.append(element)
            continue
        if element.tag == "row":
            total_rows += 1
            # Definitions can be nested in a thread or a new backtrace. Register
            # their dependencies before resolving the row's direct references.
            for item in element.iter("pid"):
                if item.get("id"):
                    pids[item.get("id")] = int(item.text)
            for item in element.iter("process"):
                if item.get("id"):
                    processes[item.get("id")] = {
                        "pid": pids[ident(item.find("pid"))], "label": item.get("fmt")
                    }
            for item in element.iter("weight"):
                if item.get("id"):
                    weights[item.get("id")] = int(item.text)
            for item in element.iter("binary"):
                if item.get("id"):
                    binaries[item.get("id")] = dict(item.attrib)
            for item in element.iter("frame"):
                if item.get("id"):
                    binary = item.find("binary")
                    frames[item.get("id")] = {
                        "name": item.get("name"), "address": item.get("addr"),
                        "binary_id": ident(binary) if binary is not None else None,
                    }
            for item in element.iter("tagged-backtrace"):
                if item.get("id"):
                    backtraces[item.get("id")] = tuple(ident(frame) for frame in item.findall("frame"))

            process_id = ident(element.find("process"))
            pid = processes[process_id]["pid"]
            weight = weights[ident(element.find("weight"))]
            process_rows[pid] += 1
            process_weights[pid] += weight
            if pid == target_pid:
                target_rows += 1
                target_weight += weight
                weight_distribution[weight] += 1
                sample_time = element.find("sample-time")
                if sample_time.text:
                    value = int(sample_time.text)
                    minimum_time = value if minimum_time is None else min(minimum_time, value)
                    maximum_time = value if maximum_time is None else max(maximum_time, value)
                else:
                    time_reference_rows += 1
                trace = element.find("tagged-backtrace")
                if trace is not None:
                    trace_id = ident(trace)
                    entry = target_backtraces.setdefault(trace_id, [0, 0])
                    entry[0] += 1
                    entry[1] += weight
                elif element.find("sentinel") is not None:
                    sentinel_rows += 1
                    sentinel_weight += weight
                else:
                    other_missing_stack_rows += 1
                    other_missing_stack_weight += weight
            # Removing each row also removes unique sample-time descriptors.
            stack[-2].remove(element)
            element.clear()
        stack.pop()

    selected_frames = set()
    traces_out = {}
    empty_stack_rows = empty_stack_weight = 0
    for trace_id, (count, weight) in target_backtraces.items():
        members = backtraces[trace_id]
        selected_frames.update(members)
        traces_out[trace_id] = {"frame_ids": members, "sample_count": count, "weight_ns": weight}
        if not members:
            empty_stack_rows += count
            empty_stack_weight += weight
    selected_binaries = {frames[key]["binary_id"] for key in selected_frames} - {None}
    selected_processes = {key: value for key, value in processes.items() if value["pid"] == target_pid}
    trace_samples = sum(value["sample_count"] for value in traces_out.values())
    trace_weight = sum(value["weight_ns"] for value in traces_out.values())
    assert target_rows == trace_samples + sentinel_rows + other_missing_stack_rows
    assert target_weight == trace_weight + sentinel_weight + other_missing_stack_weight
    assert SOURCE.stat().st_size == source_stat.st_size
    assert SOURCE.stat().st_mtime_ns == source_stat.st_mtime_ns
    result = {
        "schema": 1, "source": str(SOURCE), "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns, "target_pid": target_pid,
        "expected_native_uuid": NATIVE_UUID,
        "native_binary_ids": [key for key in sorted(selected_binaries, key=int)
                              if binaries[key].get("UUID", "").upper() == NATIVE_UUID],
        "scope": "Summed Time Profiler sample weights across target threads; not wall time.",
        "total_xml_rows": total_rows, "target_sample_count": target_rows,
        "target_weight_ns": target_weight, "backtrace_sample_count": trace_samples,
        "backtrace_weight_ns": trace_weight, "sentinel_sample_count": sentinel_rows,
        "sentinel_weight_ns": sentinel_weight,
        "other_missing_stack_sample_count": other_missing_stack_rows,
        "other_missing_stack_weight_ns": other_missing_stack_weight,
        "empty_backtrace_sample_count": empty_stack_rows,
        "empty_backtrace_weight_ns": empty_stack_weight,
        "first_inline_sample_time_ns": minimum_time, "last_inline_sample_time_ns": maximum_time,
        "sample_time_reference_rows": time_reference_rows,
        "weight_ns_distribution": dict(weight_distribution),
        "all_processes": [{"pid": pid, "sample_count": count, "weight_ns": process_weights[pid]}
                          for pid, count in sorted(process_rows.items())],
        "processes": selected_processes,
        "binaries": {key: binaries[key] for key in sorted(selected_binaries, key=int)},
        "frames": {key: frames[key] for key in sorted(selected_frames, key=int)},
        "backtraces": traces_out,
        "retained_descriptor_counts": {"processes": len(selected_processes),
            "weights": len(weights), "binaries": len(selected_binaries),
            "frames": len(selected_frames), "backtraces": len(traces_out), "sample_times": 0},
        "parse_seconds": time.monotonic() - started,
    }
    with OUTPUT.open("w") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"frames", "backtraces", "binaries"}}, indent=2))
    print("Saved", OUTPUT)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: parse_bounded_cpu_profile.py LABEL TARGET_PID")
    main(sys.argv[1], int(sys.argv[2]))
