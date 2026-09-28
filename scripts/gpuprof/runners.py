"""External tool runners for gpuprof: ncu, perf, compute-sanitizer, metadata."""

import csv
import io
import os
import re
import shutil
from collections import defaultdict

from taxonomy import NCU_NAME_MAP
from util import run_text, safe_slug, short_kernel_name, tool_path, write_text

VERSION_COMMANDS = {
    "nvcc": ["--version"],
    "ncu": ["--version"],
    "nsys": ["--version"],
    "compute-sanitizer": ["--version"],
    "nvidia-smi": [],
    "rustc": ["--version"],
    "cargo": ["--version"],
}

OSS_TOOL_COMMANDS = {
    "nvitop": ["--version"],
    "nvtop": ["--version"],
    "gpustat": ["--version"],
    "dcgmi": ["--version"],
}


def collect_metadata(repo_root):
    tools = {}
    for name, args in {**VERSION_COMMANDS, **OSS_TOOL_COMMANDS}.items():
        path = tool_path(name)
        entry = {"path": path, "available": bool(path)}
        if path:
            entry["version"] = run_text([path, *args], cwd=repo_root)
        tools[name] = entry

    gpu_query = None
    smi = tool_path("nvidia-smi")
    if smi:
        gpu_query = run_text([
            smi,
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ], cwd=repo_root)

    return {
        "repo_root": repo_root,
        "git": {
            "rev_parse_head": run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "status_short": run_text(["git", "status", "--short"], cwd=repo_root),
        },
        "tools": tools,
        "gpu_query": gpu_query,
    }


def kernel_names_from_trace(trace, limit):
    totals = defaultdict(float)
    for phase in trace:
        for name, elapsed_ms in phase.get("nsys", {}).get("kernels", {}).items():
            totals[name] += elapsed_ms
    return [name for name, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:limit]]


def parse_ncu_text(text):
    parsed = {"status": "raw", "metrics": {}}
    if "ERR_NVGPUCTRPERM" in text:
        parsed["status"] = "blocked"
        parsed["blocked_reason"] = "NVIDIA performance counters restricted by driver permissions"
        return parsed
    patterns = [
        ("sm_throughput_pct", r"(?:SM|Compute \(SM\)) Throughput\\s+%\\s+([\\d.]+)"),
        ("dram_throughput_pct", r"DRAM Throughput\\s+%\\s+([\\d.]+)"),
        ("l2_throughput_pct", r"(?:L2|L1/TEX) Throughput\\s+%\\s+([\\d.]+)"),
        ("achieved_occupancy_pct", r"(?:Achieved Occupancy|Occupancy)\\s+%\\s+([\\d.]+)"),
        ("registers_per_thread", r"Registers Per Thread\\s+register/thread\\s+([\\d.]+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, text)
        if m:
            parsed["metrics"][key] = float(m.group(1))
    if parsed["metrics"]:
        parsed["status"] = "parsed_text"
    return parsed


def parse_ncu_csv(text):
    result = {"metrics": {}, "launches": [], "selected_launch": None}
    try:
        raw_rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return result
    if not raw_rows:
        return result

    header = raw_rows[0]
    if "Metric Name" in header or "Metric" in header:
        rows = list(csv.DictReader(io.StringIO(text)))
        for row in rows:
            name = row.get("Metric Name") or row.get("Metric") or row.get("Name")
            value = row.get("Metric Value") or row.get("Value")
            add_ncu_metric(result["metrics"], name, value)
        return result

    # Nsight Compute 2025 raw CSV is one wide row per profiled launch:
    # header, units, then values. Preserve every launch and select the
    # longest profiled launch for the top-level metrics; first-launch metrics
    # are often tiny warmup/control kernels and are misleading for tuning.
    for values in raw_rows[2:] if len(raw_rows) > 2 else raw_rows[1:]:
        launch = {"meta": ncu_launch_meta(header, values), "metrics": {}}
        for name, value in zip(header, values):
            add_ncu_metric(launch["metrics"], name, value)
        if launch["metrics"]:
            result["launches"].append(launch)

    if result["launches"]:
        selected = max(
            result["launches"],
            key=lambda row: row["metrics"].get("kernel_duration_us", 0.0),
        )
        result["selected_launch"] = selected
        result["metrics"].update(selected["metrics"])
        durations = [
            row["metrics"]["kernel_duration_us"]
            for row in result["launches"]
            if "kernel_duration_us" in row["metrics"]
        ]
        result["metrics"]["profiled_launches"] = float(len(result["launches"]))
        if durations:
            result["metrics"]["max_kernel_duration_us"] = max(durations)
            result["metrics"]["total_kernel_duration_us"] = sum(durations)
    return result


def ncu_launch_meta(header, values):
    row = dict(zip(header, values))
    return {
        "id": row.get("ID"),
        "kernel": row.get("Kernel Name"),
        "context": row.get("Context"),
        "stream": row.get("Stream"),
        "block_size": row.get("Block Size"),
        "grid_size": row.get("Grid Size"),
    }


def add_ncu_metric(metrics, name, value):
    if not name or value in (None, "", "nan", "no data"):
        return
    key = NCU_NAME_MAP.get(name)
    if not key:
        return
    try:
        metrics[key] = float(str(value).replace(",", ""))
    except ValueError:
        metrics[key] = value


def run_ncu_profiles(args, trace, artifact_dir):
    if args.ncu_top <= 0:
        return []
    ncu = tool_path("ncu")
    out = []
    kernels = kernel_names_from_trace(trace, args.ncu_top)
    ncu_dir = os.path.join(artifact_dir, "ncu")
    os.makedirs(ncu_dir, exist_ok=True)
    if not ncu:
        return [{
            "ok": False,
            "error": "ncu not found on PATH or in known CUDA install paths",
            "requested_top": args.ncu_top,
            "kernels": kernels,
        }]

    for idx, kernel in enumerate(kernels, 1):
        slug = safe_slug(short_kernel_name(kernel))
        base = os.path.join(ncu_dir, f"{idx:02d}-{slug}")
        stdout_path = base + ".stdout.txt"
        stderr_path = base + ".stderr.txt"
        export_base = base
        cmd = [
            ncu,
            "--set", args.ncu_set,
            "--target-processes", "all",
            "--kernel-name", f"regex:{re.escape(short_kernel_name(kernel))}",
            "--launch-skip", str(args.ncu_launch_skip),
            "--launch-count", str(args.ncu_launch_count),
            "--force-overwrite",
            "--export", export_base,
            args.binary,
            args.gate,
        ]
        result = run_text(cmd)
        write_text(stdout_path, result["stdout"])
        write_text(stderr_path, result["stderr"])
        parsed = parse_ncu_text(result["stdout"] + "\n" + result["stderr"])
        csv_path = base + ".raw.csv"
        if result["ok"]:
            imported = run_text([ncu, "--import", export_base + ".ncu-rep", "--page", "raw", "--csv"])
            write_text(csv_path, imported["stdout"] + imported["stderr"])
            csv_result = parse_ncu_csv(imported["stdout"])
            if csv_result["metrics"]:
                parsed["status"] = "parsed_csv"
                parsed["metrics"].update(csv_result["metrics"])
                parsed["launches"] = csv_result["launches"]
                parsed["selected_launch"] = csv_result["selected_launch"]
        out.append({
            "kernel": kernel,
            "kernel_regex": short_kernel_name(kernel),
            "cmd": result["cmd"],
            "exit_code": result["exit_code"],
            "ok": result["ok"],
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "report_path": export_base + ".ncu-rep",
            "raw_csv_path": csv_path if result["ok"] else None,
            "parsed": parsed,
        })
    return out


def run_cpu_profiles(args, artifact_dir):
    if args.cpu_profile == "none":
        return []
    out_dir = os.path.join(artifact_dir, "cpu")
    os.makedirs(out_dir, exist_ok=True)
    perf = shutil.which("perf")
    if not perf:
        return [{"tool": "perf", "ok": False, "error": "perf not found on PATH"}]
    data_path = os.path.join(out_dir, "perf.data")
    stdout_path = os.path.join(out_dir, "perf-record.stdout.txt")
    stderr_path = os.path.join(out_dir, "perf-record.stderr.txt")
    cmd = [
        perf, "record", "-F", str(args.cpu_perf_freq), "-g", "--call-graph", "dwarf",
        "-o", data_path, args.binary, args.gate,
    ]
    result = run_text(cmd)
    write_text(stdout_path, result["stdout"])
    write_text(stderr_path, result["stderr"])
    entry = {
        "tool": "perf",
        "cmd": result["cmd"],
        "exit_code": result["exit_code"],
        "ok": result["ok"],
        "data_path": data_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }
    if result["ok"]:
        script_path = os.path.join(out_dir, "perf-script.txt")
        report_path = os.path.join(out_dir, "perf-report.txt")
        script = run_text([perf, "script", "-i", data_path])
        report = run_text([perf, "report", "--stdio", "-i", data_path])
        write_text(script_path, script["stdout"] + script["stderr"])
        write_text(report_path, report["stdout"] + report["stderr"])
        entry["script_path"] = script_path
        entry["report_path"] = report_path
    return [entry]


def expand_sanitizers(values):
    expanded = []
    for value in values:
        if value == "all":
            expanded.extend(["memcheck", "racecheck", "initcheck", "synccheck"])
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def run_sanitizers(args, artifact_dir):
    tools = expand_sanitizers(args.sanitize)
    if not tools:
        return []
    sanitizer = tool_path("compute-sanitizer")
    out = []
    san_dir = os.path.join(artifact_dir, "sanitizer")
    os.makedirs(san_dir, exist_ok=True)
    if not sanitizer:
        return [{
            "ok": False,
            "error": "compute-sanitizer not found on PATH or in known CUDA install paths",
            "requested_tools": tools,
        }]

    for tool in tools:
        log_path = os.path.join(san_dir, f"{safe_slug(tool)}.log")
        stdout_path = os.path.join(san_dir, f"{safe_slug(tool)}.stdout.txt")
        stderr_path = os.path.join(san_dir, f"{safe_slug(tool)}.stderr.txt")
        cmd = [
            sanitizer,
            "--tool", tool,
            "--target-processes", "all",
            "--error-exitcode", "99",
            "--log-file", log_path,
            args.binary,
            args.gate,
        ]
        result = run_text(cmd)
        write_text(stdout_path, result["stdout"])
        write_text(stderr_path, result["stderr"])
        out.append({
            "tool": tool,
            "cmd": result["cmd"],
            "exit_code": result["exit_code"],
            "ok": result["ok"],
            "log_path": log_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        })
    return out


def run_external_tools(args, artifact_dir):
    out = []
    ext_dir = os.path.join(artifact_dir, "external")
    os.makedirs(ext_dir, exist_ok=True)
    for spec in args.external_tool:
        if "=" not in spec:
            out.append({"ok": False, "spec": spec, "error": "expected NAME=COMMAND"})
            continue
        name, command = spec.split("=", 1)
        name = safe_slug(name)
        command = command.format(binary=args.binary, gate=args.gate, artifact_dir=artifact_dir)
        result = run_text(command, shell=True)
        stdout_path = os.path.join(ext_dir, f"{name}.stdout.txt")
        stderr_path = os.path.join(ext_dir, f"{name}.stderr.txt")
        write_text(stdout_path, result["stdout"])
        write_text(stderr_path, result["stderr"])
        out.append({
            "name": name,
            "cmd": result["cmd"],
            "exit_code": result["exit_code"],
            "ok": result["ok"],
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        })
    return out
