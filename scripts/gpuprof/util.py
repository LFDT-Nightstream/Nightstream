"""Shared helpers for gpuprof: subprocess, paths, formatting, interval math."""

import json
import os
import re
import shlex
import shutil
import subprocess

CUDA_BIN_DIRS = [
    os.environ.get("CUDA_HOME", "") and os.path.join(os.environ["CUDA_HOME"], "bin"),
    os.environ.get("CUDA_PATH", "") and os.path.join(os.environ["CUDA_PATH"], "bin"),
    "/usr/local/cuda-13.0/bin",
    "/usr/local/cuda/bin",
]


def tool_path(name):
    found = shutil.which(name)
    if found:
        return found
    for directory in CUDA_BIN_DIRS:
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_text(cmd, cwd=None, shell=False):
    try:
        proc = subprocess.run(cmd, cwd=cwd, shell=shell, text=True, capture_output=True)
        rendered = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)
        return {
            "cmd": rendered,
            "exit_code": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        rendered = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)
        return {"cmd": rendered, "exit_code": None, "ok": False, "stdout": "", "stderr": repr(exc)}


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    json.dump(value, open(path, "w"), indent=1)


def safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value[:96] or "item"


def short_kernel_name(name):
    name = name.split("(")[0]
    name = name.rsplit("::", 1)[-1]
    return name[:34]


def copy_kind_name(kind):
    return {1: "H2D", 2: "D2H", 3: "D2D", 8: "D2D", 10: "P2P", 11: "UVM H2D", 12: "UVM D2H", 13: "UVM D2D"}.get(
        kind, f"copyKind={kind}"
    )


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
    """Merge (start, end) intervals into a sorted disjoint union."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def union_overlap(window_start, window_end, union):
    """Total overlap of [window_start, window_end] with a disjoint union."""
    total = 0.0
    for s, e in union:
        if s >= window_end:
            break
        total += overlap(window_start, window_end, s, e)
    return total


def complement_intervals(window_start, window_end, union):
    """Gaps of [window_start, window_end] not covered by a disjoint union."""
    gaps, cursor = [], window_start
    for s, e in union:
        if e <= window_start:
            continue
        if s >= window_end:
            break
        if s > cursor:
            gaps.append((cursor, min(s, window_end)))
        cursor = max(cursor, e)
        if cursor >= window_end:
            break
    if cursor < window_end:
        gaps.append((cursor, window_end))
    return gaps


def fmt_ms(value):
    return f"{value:.1f}" if value >= 0.05 else "."


def fmt_pct(value):
    return f"{value:.0f}%" if value >= 0.5 else "."


def fmt_count(value):
    return str(int(value)) if value >= 0.5 else "."


def fmt_mb_count(mb, count):
    return "." if mb < 0.5 and count < 0.5 else f"{mb:.0f}/{int(count)}"
