#!/usr/bin/env python3
"""Aggregate saved CPU samples using only the exact original native image."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_UUID = "D66506D5-8D84-3D01-8EEB-E21332765BD7"
BINARY = ROOT / "application-replay-benchmark-binary"
HEX = re.compile(r"^0x[0-9a-fA-F]+$")


def identified(name):
    return bool(name) and not HEX.fullmatch(name) and not name.startswith("<")


def main(prefix, target_pid):
    assert Path(prefix).name == prefix, "The label must be one private file-name component."
    started = time.monotonic()
    data_path = ROOT / f"{prefix}-data.json"
    data = json.loads(data_path.read_text())
    assert data["target_pid"] == target_pid
    native_ids = {key for key, value in data["binaries"].items()
                  if value.get("UUID", "").upper() == EXPECTED_UUID}
    assert native_ids, "The export has no native image with the expected UUID."
    before = BINARY.stat()
    uuid_command = ["xcrun", "dwarfdump", "--uuid", str(BINARY)]
    uuid_result = subprocess.run(uuid_command, capture_output=True, text=True, check=True, timeout=300)
    binary_uuids = {(value.upper(), arch) for value, arch in
                    re.findall(r"UUID: ([0-9a-fA-F-]+) \(([^)]+)\)", uuid_result.stdout)}
    mapping, atos_runs = {}, []
    for binary_id in sorted(native_ids, key=int):
        native = data["binaries"][binary_id]
        assert (EXPECTED_UUID, native["arch"]) in binary_uuids
        addresses = sorted({frame["address"] for frame in data["frames"].values()
                            if frame["binary_id"] == binary_id}, key=lambda address: int(address, 16))
        address_file = ROOT / f"{prefix}-atos-{binary_id}-addresses.txt"
        output_file = ROOT / f"{prefix}-atos-{binary_id}-output.txt"
        address_file.write_text("".join(address + "\n" for address in addresses))
        command = ["atos", "-o", str(BINARY), "-arch", native["arch"],
                   "-l", native["load-addr"], "-f", str(address_file)]
        atos_started = time.monotonic()
        with output_file.open("w") as output:
            subprocess.run(command, stdout=output, check=True, timeout=300)
        lines = output_file.read_text().splitlines()
        assert len(lines) == len(addresses), (len(lines), len(addresses))
        mapping.update({(binary_id, address): line for address, line in zip(addresses, lines)})
        atos_runs.append({"binary_id": binary_id, "command": command, "address_file": str(address_file),
                          "output_file": str(output_file), "address_count": len(addresses),
                          "seconds": time.monotonic() - atos_started})
    assert BINARY.stat().st_size == before.st_size
    assert BINARY.stat().st_mtime_ns == before.st_mtime_ns

    locations, functions, binary_stats = {}, {}, {}
    location_ids = {}
    exported_native_named_frames = resolved_native_frames = 0
    for frame_id, frame in data["frames"].items():
        binary_id = frame["binary_id"]
        descriptor = data["binaries"].get(binary_id, {})
        # Alias descriptors for the same loaded image have one aggregation key.
        image_key = json.dumps([descriptor["UUID"], descriptor.get("arch"), descriptor.get("load-addr")]) \
            if descriptor.get("UUID") else json.dumps(["binary_id", binary_id])
        name = frame["name"]
        symbol = name if identified(name) else None
        resolution = "exported_symbol" if symbol is not None else "unknown"
        atos_line = None
        if binary_id in native_ids:
            exported_native_named_frames += identified(name)
            atos_line = mapping[(binary_id, frame["address"])]
            match = re.match(r"^(.*?) \(in [^)]+\)", atos_line)
            candidate = match[1] if match else None
            if candidate and identified(candidate):
                symbol = candidate
                resolution = "atos_exact_original_uuid_and_recorded_load_address"
                resolved_native_frames += 1
        function_key = json.dumps([image_key, symbol if symbol is not None else frame["address"]])
        location_key = (image_key, name, frame["address"])
        location_id = location_ids.setdefault(location_key, frame_id)
        locations.setdefault(location_id, {
            "frame_ids": [], "exported_name": name, "address": frame["address"],
            "image_key": image_key, "binary": descriptor.get("name"),
            "function": symbol, "function_key": function_key,
            "resolution": resolution, "atos_output": atos_line,
            "self_sample_count": 0, "self_weight_ns": 0,
            "inclusive_sample_count": 0, "inclusive_weight_ns": 0,
        })["frame_ids"].append(frame_id)
        frame.update({"location_id": location_id, "function_key": function_key, "image_key": image_key})
        functions.setdefault(function_key, {
            "function_key": function_key, "function": symbol,
            "unknown_address": frame["address"] if symbol is None else None,
            "image_key": image_key, "binary": descriptor.get("name"),
            "resolution": resolution, "frame_ids": [],
            "self_sample_count": 0, "self_weight_ns": 0,
            "inclusive_sample_count": 0, "inclusive_weight_ns": 0,
        })["frame_ids"].append(frame_id)
        binary_stats.setdefault(image_key, {
            "image_key": image_key, "binary": descriptor.get("name"),
            "self_sample_count": 0, "self_weight_ns": 0,
            "inclusive_sample_count": 0, "inclusive_weight_ns": 0,
        })

    def add(record, mode, count, weight):
        record[f"{mode}_sample_count"] += count
        record[f"{mode}_weight_ns"] += weight

    unknown_top_samples = unknown_top_weight = 0
    unannotated_top_samples = unannotated_top_weight = 0
    for trace in data["backtraces"].values():
        members = trace["frame_ids"]
        count, weight = trace["sample_count"], trace["weight_ns"]
        if not members:
            continue
        top = data["frames"][members[0]]
        add(locations[top["location_id"]], "self", count, weight)
        add(functions[top["function_key"]], "self", count, weight)
        add(binary_stats[top["image_key"]], "self", count, weight)
        if functions[top["function_key"]]["function"] is None:
            unknown_top_samples += count
            unknown_top_weight += weight
        if top["binary_id"] is None:
            unannotated_top_samples += count
            unannotated_top_weight += weight
        for location_id in {data["frames"][frame]["location_id"] for frame in members}:
            add(locations[location_id], "inclusive", count, weight)
        for function_key in {data["frames"][frame]["function_key"] for frame in members}:
            add(functions[function_key], "inclusive", count, weight)
        for image_key in {data["frames"][frame]["image_key"] for frame in members}:
            add(binary_stats[image_key], "inclusive", count, weight)

    nonempty_samples = data["backtrace_sample_count"] - data["empty_backtrace_sample_count"]
    nonempty_weight = data["backtrace_weight_ns"] - data["empty_backtrace_weight_ns"]
    assert sum(item["self_sample_count"] for item in locations.values()) == nonempty_samples
    assert sum(item["self_weight_ns"] for item in functions.values()) == nonempty_weight
    assert all(item["inclusive_weight_ns"] <= nonempty_weight for item in functions.values())
    metadata = {key: value for key, value in data.items() if key not in {"frames", "backtraces"}}
    metadata.update({
        "raw_data_file": str(data_path),
        "aggregation": {
            "self": "The first frame in each exported backtrace receives the complete sample count and weight.",
            "inclusive_frame": "Each distinct loaded-image/name/PC location receives a stack's weight once.",
            "inclusive_function": "Each distinct resolved loaded-image/function receives a stack's weight once; recursion is deduplicated after exact-image symbolication.",
            "unknowns": "Unknown symbols remain identified by recorded image and PC. Unannotated frames are never assigned to the native image.",
            "time": "Weights sum sampled CPU work across threads, not wall time. Inclusive totals overlap and must not be added as elapsed time.",
        },
        "symbolication": {
            "native_binary_ids": sorted(native_ids, key=int), "expected_uuid": EXPECTED_UUID,
            "exact_original_binary": str(BINARY), "binary_bytes": before.st_size,
            "binary_mtime_ns": before.st_mtime_ns,
            "uuid_command": uuid_command, "uuid_output": uuid_result.stdout,
            "atos_runs": atos_runs, "native_address_count": len(mapping),
            "native_frames_with_exported_symbols": exported_native_named_frames,
            "native_frames_resolved_by_atos": resolved_native_frames,
            "unknown_top_sample_count": unknown_top_samples, "unknown_top_weight_ns": unknown_top_weight,
            "unannotated_top_sample_count": unannotated_top_samples, "unannotated_top_weight_ns": unannotated_top_weight,
            "scope": "Only the exact original binary with matching recorded UUID is used. There is no rebuilt-image or function-byte-matching fallback.",
        },
        "frame_locations": sorted(locations.values(), key=lambda value: (-value["self_weight_ns"], -value["inclusive_weight_ns"], value["address"])),
        "functions_by_self": sorted(functions.values(), key=lambda value: (-value["self_weight_ns"], -value["inclusive_weight_ns"], value["function_key"])),
        "function_keys_by_inclusive": [value["function_key"] for value in sorted(functions.values(), key=lambda value: (-value["inclusive_weight_ns"], -value["self_weight_ns"], value["function_key"]))],
        "binary_totals": sorted(binary_stats.values(), key=lambda value: -value["self_weight_ns"]),
        "summary_seconds": time.monotonic() - started,
    })
    summary = ROOT / f"{prefix}-summary.json"
    with summary.open("w") as stream:
        json.dump(metadata, stream, separators=(",", ":"))
        stream.write("\n")
    receipt_path = ROOT / f"{prefix}-attribution.json"
    receipt = {
        "source": data["source"], "source_bytes": data["source_bytes"], "target_pid": target_pid,
        "total_sample_count": data["target_sample_count"], "total_weight_ns": data["target_weight_ns"],
        "backtrace_sample_count": data["backtrace_sample_count"], "backtrace_weight_ns": data["backtrace_weight_ns"],
        "sentinel_sample_count": data["sentinel_sample_count"], "sentinel_weight_ns": data["sentinel_weight_ns"],
        "symbolication": metadata["symbolication"],
        "recorded_native_images": {key: data["binaries"][key] for key in sorted(native_ids, key=int)},
        "aggregation": metadata["aggregation"],
        "complete_data_file": str(data_path), "complete_summary_file": str(summary),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Saved attribution receipt", receipt_path)
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in {"frame_locations", "functions_by_self", "function_keys_by_inclusive", "binaries"}}, indent=2))
    print("Saved", summary)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: summarize_bounded_cpu_profile.py LABEL TARGET_PID")
    main(sys.argv[1], int(sys.argv[2]))
