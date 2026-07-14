"""Offline stage decomposition for raw Nsight Systems traces."""

from collections import Counter, defaultdict
import json
from pathlib import Path

try:
    from .levers import build_levers
    from .parse import build_phases, load_timeline, select_input_bundle
    from .taxonomy import KNOWN_STAGES, TAXONOMY_VERSION, ancestors, ordered_stage_ids, source_for
    from .validate import validate_against_gpuprof
except ImportError:  # Direct `python scripts/gpuscope/cli.py ...` execution.
    from levers import build_levers
    from parse import build_phases, load_timeline, select_input_bundle
    from taxonomy import KNOWN_STAGES, TAXONOMY_VERSION, ancestors, ordered_stage_ids, source_for
    from validate import validate_against_gpuprof

SCHEMA_VERSION = 1
EPSILON_MS = 0.001

RECONCILE_FIELDS = (
    "wall_ms",
    "gpu_busy_ms",
    "sync_wait_ms",
    "api_ms",
    "host_gap_ms",
    "transfer_wait_ms",
    "unattributed_ms",
    "kernel_busy_ms",
    "memset_busy_ms",
)

DIAGNOSTIC_FIELDS = (
    "idle_overlap_clamp_ms",
)

WINDOW_FIELDS = (
    "window_wall_ms",
    "window_gpu_busy_ms",
    "window_sync_wait_ms",
    "window_api_ms",
    "window_host_gap_ms",
    "window_transfer_wait_ms",
    "window_unattributed_ms",
    "window_count",
)

ATTRIBUTION_FIELDS = (
    "kernel_attributed_ms",
    "busy_outside_window_ms",
    "launches",
    "syncs",
    "sync_total_ms",
    "api_total_ms",
    "api_calls",
    "h2d_mb",
    "h2d_ms",
    "h2d_copies",
    "d2h_mb",
    "d2h_ms",
    "d2h_copies",
    "dtod_mb",
    "dtod_ms",
    "dtod_copies",
    "memset_mb",
    "memset_ms",
    "memset_count",
)


def analyze_bundle(bundle_dir, validate_gpuprof=False, gpuprof_json=None):
    requested = Path(bundle_dir).resolve()
    oracle_path = None
    oracle_artifacts = {}
    if validate_gpuprof or gpuprof_json:
        oracle_path = Path(gpuprof_json).resolve() if gpuprof_json else requested / "gpuprof.json"
        oracle_artifacts = read_gpuprof_artifacts(oracle_path)

    bundle = select_input_bundle(
        bundle_dir,
        preferred_sqlite=oracle_artifacts.get("nsys_sqlite"),
        preferred_stdout=oracle_artifacts.get("stdout"),
        preferred_stderr=oracle_artifacts.get("stderr"),
        preferred_selection="gpuprof oracle artifacts.nsys_sqlite" if oracle_artifacts.get("nsys_sqlite") else None,
    )
    timeline = load_timeline(bundle.sqlite_path)
    all_phases, phase_counts = build_phases(bundle.stderr_path, timeline.get("session_start"), timeline["nvtx"])
    phases = [phase for phase in all_phases if phase["chain"] == "gpu"]
    if not phases:
        raise RuntimeError(f"no GPU-chain phases found in {bundle.sqlite_path}")

    own = defaultdict(new_stage)
    terminal_range = terminal_fold_range(phases)
    intervals = build_interval_sets(timeline)
    add_window_wall(own, phases, terminal_range, intervals)
    add_exclusive_decomposition(own, phases, timeline, terminal_range)
    add_enqueue_attribution(own, phases, timeline, terminal_range)
    records = build_stage_records(own)
    lever_report = build_levers(records, timeline)
    max_error = max((abs(row["reconciliation_error_ms"]) for row in records), default=0.0)
    idle_overlap_clamp_ms = sum(row["idle_overlap_clamp_ms"] for row in records if row["stage_id"].count(".") == 0)

    out = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "source": {
            "requested_dir": str(bundle.requested_dir),
            "run_dir": str(bundle.run_dir),
            "sqlite_path": str(bundle.sqlite_path),
            "stdout_path": str(bundle.stdout_path) if bundle.stdout_path else None,
            "stderr_path": str(bundle.stderr_path) if bundle.stderr_path else None,
            "selection": bundle.selection,
        },
        "counts": {
            "nvtx_ranges": len(phases),
            "kernels": len(timeline["kernels"]),
            "memcpys": len(timeline["memcpys"]),
            "memsets": len(timeline["memsets"]),
            "syncs": len(timeline["syncs"]),
            "api_calls": len(timeline["api_calls"]),
            "cpu_phases": phase_counts["cpu_phases"],
            "gpu_phases": phase_counts["gpu_phases"],
            "stderr_stamped_lines": phase_counts["stderr_stamped_lines"],
            "stderr_timer_lines": phase_counts["stderr_timer_lines"],
            "stderr_stamp_coverage": phase_counts["stderr_stamp_coverage"],
            "stages": len(records),
            "synthetic_terminal_fold": terminal_range is not None,
        },
        "reconciliation": {
            "epsilon_ms": EPSILON_MS,
            "host_gap_semantics": "device-idle time not explained by sync/API; inferred host time, not directly observed",
            "idle_overlap_clamp_ms": round(idle_overlap_clamp_ms, 6),
            "max_abs_error_ms": round(max_error, 6),
            "ok": max_error <= EPSILON_MS and idle_overlap_clamp_ms <= EPSILON_MS,
        },
        "unknown_nvtx": unknown_nvtx(phases),
        "levers": {
            "path": str(bundle.analysis_dir / "levers.json"),
            "count": len(lever_report["levers"]),
            "top": lever_report["levers"][0] if lever_report["levers"] else None,
        },
        "stages": records,
    }

    validation = None
    if oracle_path is not None:
        validation = validate_against_gpuprof(out, all_phases, oracle_path)
        out["gpuprof_validation"] = {
            "path": str(bundle.analysis_dir / "gpuprof_diff.json"),
            "oracle": str(oracle_path),
            "ok": validation["ok"],
            "nvtx_mismatches": len(validation["nvtx_ranges"]["mismatches"]),
            "stderr_mismatches": len(validation["stderr_phases"]["mismatches"]),
            "stage_mismatches": len(validation["stage_attribution"]["mismatches"]),
        }

    output_path = bundle.analysis_dir / "stages.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if validation is not None:
        validation_path = bundle.analysis_dir / "gpuprof_diff.json"
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    levers_path = bundle.analysis_dir / "levers.json"
    levers_path.write_text(json.dumps(lever_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, out


def read_gpuprof_artifacts(path):
    oracle = json.loads(Path(path).read_text(encoding="utf-8"))
    artifacts = oracle.get("artifacts") or {}
    return {
        key: artifacts.get(key)
        for key in ("nsys_sqlite", "stdout", "stderr")
        if artifacts.get(key)
    }


def new_stage():
    row = {field: 0.0 for field in RECONCILE_FIELDS + DIAGNOSTIC_FIELDS + WINDOW_FIELDS + ATTRIBUTION_FIELDS}
    row["kernels"] = defaultdict(float)
    row["cuda_api"] = defaultdict(float)
    row["instances"] = []
    row["window_instances"] = []
    return row


def add_window_wall(own, phases, terminal_range, intervals):
    for phase in phases:
        add_window_to_row(own[phase["label"]], phase["start"], phase["end"], intervals)
    if terminal_range is not None:
        add_window_to_row(own["finalize.terminal_fold"], terminal_range[0], terminal_range[1], intervals)


def add_window_to_row(row, start, end, intervals):
    values = decompose_window(start, end, intervals)
    row["window_wall_ms"] += ns_to_ms(end - start)
    row["window_gpu_busy_ms"] += values["gpu_busy_ms"]
    row["window_sync_wait_ms"] += values["sync_wait_ms"]
    row["window_api_ms"] += values["api_ms"]
    row["window_host_gap_ms"] += values["host_gap_ms"]
    row["window_transfer_wait_ms"] += values["transfer_wait_ms"]
    row["window_unattributed_ms"] += values["unattributed_ms"]
    row["window_count"] += 1
    row["window_instances"].append(
        {
            "start_ms": rounded(ns_to_ms(start)),
            "end_ms": rounded(ns_to_ms(end)),
            "wall_ms": rounded(ns_to_ms(end - start)),
        }
    )


def add_exclusive_decomposition(own, phases, timeline, terminal_range):
    intervals = build_interval_sets(timeline)
    for segment in exclusive_segments(phases):
        start, end = segment["start"], segment["end"]
        values = decompose_window(start, end, intervals)
        add_decomposition_to_row(own[segment["label"]], start, end, values)
        if range_contains_window(terminal_range, start, end):
            add_decomposition_to_row(own["finalize.terminal_fold"], start, end, values)


def decompose_window(start, end, intervals):
    wall_ns = end - start
    kernel_ns = union_overlap(start, end, intervals["kernel"])
    memset_ns = union_overlap(start, end, intervals["memset"])
    gpu_busy_ns = union_overlap(start, end, intervals["device_busy"])
    transfer_ns = 0.0
    for gap_start, gap_end in complement_intervals(start, end, intervals["device_busy"]):
        transfer_ns += union_overlap(gap_start, gap_end, intervals["copy"])

    sync_ns = 0.0
    api_ns = 0.0
    idle_ns = 0.0
    for gap_start, gap_end in complement_intervals(start, end, intervals["device"]):
        idle_ns += gap_end - gap_start
        sync_here = union_overlap(gap_start, gap_end, intervals["sync_api"])
        sync_ns += sync_here
        for rest_start, rest_end in complement_intervals(gap_start, gap_end, intervals["sync_api"]):
            api_ns += union_overlap(rest_start, rest_end, intervals["other_api"])

    host_raw_ns = idle_ns - sync_ns - api_ns
    host_ns = max(0.0, host_raw_ns)
    clamp_ns = max(0.0, -host_raw_ns)
    sum_ns = gpu_busy_ns + transfer_ns + sync_ns + api_ns + host_ns
    return {
        "wall_ms": ns_to_ms(wall_ns),
        "gpu_busy_ms": ns_to_ms(gpu_busy_ns),
        "kernel_busy_ms": ns_to_ms(kernel_ns),
        "memset_busy_ms": ns_to_ms(memset_ns),
        "transfer_wait_ms": ns_to_ms(transfer_ns),
        "sync_wait_ms": ns_to_ms(sync_ns),
        "api_ms": ns_to_ms(api_ns),
        "host_gap_ms": ns_to_ms(host_ns),
        "unattributed_ms": ns_to_ms(wall_ns - sum_ns),
        "idle_overlap_clamp_ms": ns_to_ms(clamp_ns),
    }


def add_decomposition_to_row(row, start, end, values):
    for field, value in values.items():
        row[field] += value
    append_instance(row["instances"], start, end)


def add_enqueue_attribution(own, phases, timeline, terminal_range):
    api_by_corr = {
        row["correlation_id"]: row
        for row in timeline["api_calls"]
        if row.get("correlation_id") is not None
    }

    def owner_for(activity):
        api = api_by_corr.get(activity.get("correlation_id"))
        if api is not None:
            t = midpoint(api)
        else:
            t = midpoint(activity)
        return phase_label_at(phases, t), t

    for kernel in timeline["kernels"]:
        label, t = owner_for(kernel)
        if label is None:
            continue
        add_kernel_attribution(own[label], phases, label, kernel)
        if range_contains_point(terminal_range, t):
            add_kernel_attribution(own["finalize.terminal_fold"], phases, label, kernel)

    for copy in timeline["memcpys"]:
        label, t = owner_for(copy)
        if label is None:
            continue
        add_copy_attribution(own[label], copy)
        if range_contains_point(terminal_range, t):
            add_copy_attribution(own["finalize.terminal_fold"], copy)

    for memset in timeline["memsets"]:
        label, t = owner_for(memset)
        if label is None:
            continue
        add_memset_attribution(own[label], memset)
        if range_contains_point(terminal_range, t):
            add_memset_attribution(own["finalize.terminal_fold"], memset)

    for sync in timeline["syncs"]:
        t = midpoint(sync)
        label = phase_label_at(phases, t)
        if label is None:
            continue
        add_sync_attribution(own[label], sync)
        if range_contains_point(terminal_range, t):
            add_sync_attribution(own["finalize.terminal_fold"], sync)

    for api in timeline["api_calls"]:
        t = midpoint(api)
        label = phase_label_at(phases, t)
        if label is None:
            continue
        add_api_attribution(own[label], api)
        if range_contains_point(terminal_range, t):
            add_api_attribution(own["finalize.terminal_fold"], api)


def add_kernel_attribution(row, phases, owner_label, kernel):
    elapsed = elapsed_ms(kernel)
    row["launches"] += 1
    row["kernel_attributed_ms"] += elapsed
    row["kernels"][kernel["name"]] += elapsed
    inside = interval_overlap_with_phase(phases, owner_label, kernel["start"], kernel["end"])
    row["busy_outside_window_ms"] += max(0.0, elapsed - ns_to_ms(inside))


def add_copy_attribution(row, copy):
    # Nsight's ENUM_CUDA_MEMCPY_OPER uses 8 for ordinary device-to-device
    # copies. Older traces used 3, so accept both spellings here.
    prefix = {
        1: "h2d",
        2: "d2h",
        3: "dtod",
        8: "dtod",
        10: "dtod",
        11: "h2d",
        12: "d2h",
        13: "dtod",
    }.get(copy["copy_kind"], "copy")
    if prefix == "copy":
        return
    row[f"{prefix}_mb"] += copy["bytes"] / 1e6
    row[f"{prefix}_ms"] += elapsed_ms(copy)
    row[f"{prefix}_copies"] += 1


def add_memset_attribution(row, memset):
    row["memset_mb"] += memset["bytes"] / 1e6
    row["memset_ms"] += elapsed_ms(memset)
    row["memset_count"] += 1


def add_sync_attribution(row, sync):
    row["syncs"] += 1
    row["sync_total_ms"] += elapsed_ms(sync)


def add_api_attribution(row, api):
    elapsed = elapsed_ms(api)
    row["api_total_ms"] += elapsed
    row["api_calls"] += 1
    row["cuda_api"][api["name"]] += elapsed


def build_stage_records(own):
    rollup = defaultdict(new_stage)
    for stage_id, row in own.items():
        for ancestor in ancestors(stage_id):
            dst = rollup[ancestor]
            for field in RECONCILE_FIELDS + DIAGNOSTIC_FIELDS + WINDOW_FIELDS + ATTRIBUTION_FIELDS:
                dst[field] += row[field]
            merge_named_ms(dst["kernels"], row["kernels"])
            merge_named_ms(dst["cuda_api"], row["cuda_api"])
        rollup[stage_id]["instances"].extend(row["instances"])
        rollup[stage_id]["window_instances"].extend(row["window_instances"])

    records = []
    for stage_id in ordered_stage_ids(rollup):
        row = rollup[stage_id]
        bucket_sum = (
            row["gpu_busy_ms"]
            + row["sync_wait_ms"]
            + row["api_ms"]
            + row["host_gap_ms"]
            + row["transfer_wait_ms"]
            + row["unattributed_ms"]
        )
        record = {
            "stage_id": stage_id,
            "chain": "gpu",
            "source": source_for(stage_id),
            "wall_ms": rounded(row["wall_ms"]),
            "window_wall_ms": rounded(row["window_wall_ms"]),
            "window_gpu_busy_ms": rounded(row["window_gpu_busy_ms"]),
            "window_sync_wait_ms": rounded(row["window_sync_wait_ms"]),
            "window_api_ms": rounded(row["window_api_ms"]),
            "window_host_gap_ms": rounded(row["window_host_gap_ms"]),
            "window_transfer_wait_ms": rounded(row["window_transfer_wait_ms"]),
            "window_unattributed_ms": rounded(row["window_unattributed_ms"]),
            "window_count": int(row["window_count"]),
            "gpu_busy_ms": rounded(row["gpu_busy_ms"]),
            "kernel_busy_ms": rounded(row["kernel_busy_ms"]),
            "memset_busy_ms": rounded(row["memset_busy_ms"]),
            "sync_wait_ms": rounded(row["sync_wait_ms"]),
            "api_ms": rounded(row["api_ms"]),
            "host_gap_ms": rounded(row["host_gap_ms"]),
            "transfer_wait_ms": rounded(row["transfer_wait_ms"]),
            "unattributed_ms": rounded(row["unattributed_ms"]),
            "idle_overlap_clamp_ms": rounded(row["idle_overlap_clamp_ms"]),
            "reconciliation_error_ms": rounded(row["wall_ms"] - bucket_sum),
            "kernel_attributed_ms": rounded(row["kernel_attributed_ms"]),
            "busy_outside_window_ms": rounded(row["busy_outside_window_ms"]),
            "launches": int(row["launches"]),
            "syncs": int(row["syncs"]),
            "sync_total_ms": rounded(row["sync_total_ms"]),
            "api_total_ms": rounded(row["api_total_ms"]),
            "api_calls": int(row["api_calls"]),
            "h2d_mb": rounded(row["h2d_mb"]),
            "h2d_ms": rounded(row["h2d_ms"]),
            "h2d_copies": int(row["h2d_copies"]),
            "d2h_mb": rounded(row["d2h_mb"]),
            "d2h_ms": rounded(row["d2h_ms"]),
            "d2h_copies": int(row["d2h_copies"]),
            "dtod_mb": rounded(row["dtod_mb"]),
            "dtod_ms": rounded(row["dtod_ms"]),
            "dtod_copies": int(row["dtod_copies"]),
            "memset_mb": rounded(row["memset_mb"]),
            "memset_ms": rounded(row["memset_ms"]),
            "memset_count": int(row["memset_count"]),
            "top_kernels": top_names(row["kernels"], 8),
            "kernels": rounded_dict(row["kernels"]),
            "top_cuda_api": top_names(row["cuda_api"], 12),
            "cuda_api": rounded_dict(row["cuda_api"]),
            "instances": row["instances"],
            "window_instances": row["window_instances"],
        }
        records.append(record)
    return records


def build_interval_sets(timeline):
    kernels = [(row["start"], row["end"]) for row in timeline["kernels"]]
    copies = [(row["start"], row["end"]) for row in timeline["memcpys"]]
    memsets = [(row["start"], row["end"]) for row in timeline["memsets"]]
    api = [(row["start"], row["end"], api_bucket(row["name"])) for row in timeline["api_calls"]]
    return {
        "kernel": interval_union(kernels),
        "copy": interval_union(copies),
        "memset": interval_union(memsets),
        "device_busy": interval_union(kernels + memsets),
        "device": interval_union(kernels + copies + memsets),
        "sync_api": interval_union((start, end) for start, end, bucket in api if bucket == "sync"),
        "other_api": interval_union((start, end) for start, end, bucket in api if bucket != "sync"),
    }


def exclusive_segments(phases):
    bounds = sorted({point for phase in phases for point in (phase["start"], phase["end"])})
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        label = phase_label_at(phases, (start + end) // 2)
        if label is None:
            continue
        if segments and segments[-1]["label"] == label and segments[-1]["end"] == start:
            segments[-1]["end"] = end
        else:
            segments.append({"label": label, "start": start, "end": end})
    return segments


def phase_label_at(phases, t):
    active = [phase for phase in phases if phase["start"] <= t < phase["end"]]
    if not active:
        return None
    return min(active, key=lambda phase: (phase["duration"], -phase["start"], phase["label"]))["label"]


def terminal_fold_range(phases):
    ingests = sorted(phase["start"] for phase in phases if phase["label"] == "fold.ingest")
    if len(ingests) < 2:
        return None
    start = ingests[-1]
    ends = [phase["end"] for phase in phases if phase["start"] >= start]
    if not ends:
        return None
    return start, max(ends)


def range_contains_point(bounds, t):
    return bounds is not None and bounds[0] <= t < bounds[1]


def range_contains_window(bounds, start, end):
    return bounds is not None and bounds[0] <= start and end <= bounds[1]


def interval_overlap_with_phase(phases, label, start, end):
    total = 0.0
    for phase in phases:
        if phase["label"] == label:
            total += overlap(start, end, phase["start"], phase["end"])
    return total


def append_instance(instances, start, end):
    item = {
        "start_ms": rounded(ns_to_ms(start)),
        "end_ms": rounded(ns_to_ms(end)),
        "wall_ms": rounded(ns_to_ms(end - start)),
    }
    if instances and instances[-1]["end_ms"] == item["start_ms"]:
        instances[-1]["end_ms"] = item["end_ms"]
        instances[-1]["wall_ms"] = rounded(instances[-1]["wall_ms"] + item["wall_ms"])
    else:
        instances.append(item)


def unknown_nvtx(phases):
    counts = Counter(phase["label"] for phase in phases if phase["label"] not in KNOWN_STAGES)
    return [{"stage_id": stage_id, "count": count} for stage_id, count in sorted(counts.items())]


def merge_named_ms(dst, src):
    for name, elapsed in src.items():
        dst[name] += elapsed


def top_names(values, limit):
    return [name for name, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def rounded_dict(values):
    return {name: rounded(value) for name, value in sorted(values.items()) if abs(value) >= 0.000001}


def rounded(value):
    return round(float(value), 6)


def elapsed_ms(row):
    return ns_to_ms(row["end"] - row["start"])


def midpoint(row):
    return (row["start"] + row["end"]) // 2


def ns_to_ms(value):
    return value / 1e6


def api_bucket(name):
    if "LaunchKernel" in name:
        return "launch"
    if "MemcpyHtoD" in name:
        return "memcpy_h2d"
    if "MemcpyDtoH" in name:
        return "memcpy_d2h"
    if "MemcpyDtoD" in name:
        return "memcpy_d2d"
    if "Memset" in name:
        return "memset"
    if "StreamSynchronize" in name or "EventSynchronize" in name:
        return "sync"
    if "MemAlloc" in name:
        return "memalloc"
    if "MemFree" in name:
        return "memfree"
    if "ModuleLoad" in name:
        return "module_load"
    return "other"


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def interval_union(intervals):
    merged = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def union_overlap(window_start, window_end, union):
    total = 0.0
    for start, end in union:
        if start >= window_end:
            break
        total += overlap(window_start, window_end, start, end)
    return total


def complement_intervals(window_start, window_end, union):
    gaps = []
    cursor = window_start
    for start, end in union:
        if end <= window_start:
            continue
        if start >= window_end:
            break
        if start > cursor:
            gaps.append((cursor, min(start, window_end)))
        cursor = max(cursor, end)
        if cursor >= window_end:
            break
    if cursor < window_end:
        gaps.append((cursor, window_end))
    return gaps
