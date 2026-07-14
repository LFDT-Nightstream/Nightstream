#!/usr/bin/env python3
"""Phase-attributed GPU profiling for neo-prover-cuda parity gates."""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

from analysis import (
    build_boundary_scorecard,
    build_levers,
    build_superneo_context,
    check_residency,
    lint_kernels,
    print_boundary_scorecard,
    print_lint,
    print_levers,
    print_residency,
    print_superneo_context,
    structural_causes,
)
from runners import collect_metadata, run_cpu_profiles, run_external_tools, run_ncu_profiles, run_sanitizers
from taxonomy import NUMERIC_COLS, SOURCE_MAP, STAGE_ORDER
from trace_export import chrome_trace
from util import (
    api_bucket,
    complement_intervals,
    copy_kind_name,
    fmt_count,
    fmt_mb_count,
    fmt_ms,
    fmt_pct,
    interval_union,
    overlap,
    short_kernel_name,
    tool_path,
    union_overlap,
    write_json,
)

DUR_RE = re.compile(r"([\d][\d_,.]*)\s*(ms|µs|us|ns|s)\b")
LINE_PATTERNS = [
    (re.compile(r"^\[neo-prover-cuda\]\s+(\S+)"), "cuda"),
    (re.compile(r"^\[nifs-prove\]\s+(.+?)\s{2,}"), "nifs"),
    (re.compile(r"^\[r1cs-chain\]\s+(.+?)\s{2,}"), "chain"),
    (re.compile(r"^\[r1cs-compile\]\s+(.+?)\s{2,}"), "compile"),
    (re.compile(r"^\[r1cs-encode\]\s+(.+?)\s{2,}"), "encode"),
    (re.compile(r"^\[f-prime\]\s+(.+?)\s{2,}"), "fprime"),
    (re.compile(r"^optimized_prove:\s+(.+?)\s{2,}"), "ccs"),
    (re.compile(r"^RowStreamState::build:\s+(.+?)\s{2,}"), "oracle"),
    (re.compile(r"^NcOracle::new:\s+(.+?)\s{2,}"), "nc-oracle"),
    (re.compile(r"^OptimizedOracle::(\S+?):\s+(.+?)\s{2,}"), "oracle"),
    (re.compile(r"^OptimizedStructureCache::build:\s+(.+?)\s{2,}"), "setup"),
]
ALIASES = {
    "oracle:1. chi tables": "fold.superneo.pi_ccs.oracle.Q",
    "ccs:1. bind/header": "fold.superneo.pi_ccs.bind",
    "ccs:2. sample challenges": "fold.superneo.pi_ccs.challenge_alpha_gamma",
    "ccs:3. oracle build": "fold.superneo.pi_ccs.oracle",
    "ccs:4. FE sumcheck": "fold.superneo.pi_ccs.sumcheck.fe",
    "ccs:5. NC sumcheck": "fold.superneo.pi_ccs.sumcheck.nc",
    "ccs:6. output": "fold.superneo.pi_ccs.output.claims",
}

UNIT_NS = {"s": 1e9, "ms": 1e6, "µs": 1e3, "us": 1e3, "ns": 1.0}

def parse_phase_line(line):
    """Return (label, elapsed_ns) for a recognized timer line, else None."""
    m = DUR_RE.search(line)
    if not m:
        return None
    elapsed_ns = float(m.group(1).replace(",", "").replace("_", "")) * UNIT_NS[m.group(2)]
    for pat, family in LINE_PATTERNS:
        pm = pat.match(line)
        if pm:
            label = " ".join(g for g in pm.groups() if g).strip()
            label = re.sub(r"\s+", " ", label)
            full = f"{family}:{label}"
            return ALIASES.get(full, full), family, elapsed_ns
    return None

def capture(binary, gate, workdir):
    nsys = tool_path("nsys")
    if not nsys:
        sys.exit("error: nsys not found on PATH or in known CUDA install paths")
    rep = os.path.join(workdir, "gpuprof")
    err_path = os.path.join(workdir, "stderr.txt")
    out_path = os.path.join(workdir, "stdout.txt")
    cmd = [
        nsys, "profile", "--trace=cuda,osrt,nvtx", "--force-overwrite=true",
        "-o", rep, binary, gate,
    ]
    with open(out_path, "w") as out, open(err_path, "w") as err:
        subprocess.run(cmd, stdout=out, stderr=err, check=True)
    sqlite_path = rep + ".sqlite"
    subprocess.run(
        [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
         "-o", sqlite_path, rep + ".nsys-rep"],
        check=True, capture_output=True,
    )
    return sqlite_path, out_path, err_path, rep + ".nsys-rep"


def table_exists(db, name):
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_timeline(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    def q(table, sql):
        return db.execute(sql).fetchall() if table_exists(db, table) else []

    kernels = [{
        "start": r[0], "end": r[1], "name": r[2], "stream_id": r[3],
        "correlation_id": r[4], "registers_per_thread": r[5],
        "grid": [r[6], r[7], r[8]], "block": [r[9], r[10], r[11]],
        "static_shared_memory": r[12], "dynamic_shared_memory": r[13],
        "local_memory_per_thread": r[14],
    } for r in q(
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "SELECT k.start, k.end, s.value, k.streamId, k.correlationId, "
        "k.registersPerThread, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, "
        "k.blockZ, k.staticSharedMemory, k.dynamicSharedMemory, "
        "k.localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id"
    )]
    memcpys = [{
        "start": r[0], "end": r[1], "bytes": r[2], "copy_kind": r[3],
        "stream_id": r[4], "correlation_id": r[5], "src_kind": r[6],
        "dst_kind": r[7], "copy_count": r[8],
    } for r in q(
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "SELECT start, end, bytes, copyKind, streamId, correlationId, "
        "srcKind, dstKind, copyCount FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    )]
    memsets = [{
        "start": r[0], "end": r[1], "bytes": r[2], "stream_id": r[3],
        "correlation_id": r[4], "mem_kind": r[5],
    } for r in q(
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "SELECT start, end, bytes, streamId, correlationId, memKind "
        "FROM CUPTI_ACTIVITY_KIND_MEMSET"
    )]
    syncs = [{
        "start": r[0], "end": r[1], "stream_id": r[2], "correlation_id": r[3],
        "sync_type": r[4], "sync_label": r[5],
    } for r in q(
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "SELECT s.start, s.end, s.streamId, s.correlationId, s.syncType, "
        "coalesce(e.label, e.name, '') FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s "
        "LEFT JOIN ENUM_CUPTI_SYNC_TYPE e ON s.syncType = e.id"
    )]
    api_calls = [{
        "start": r[0], "end": r[1], "name": r[2], "correlation_id": r[3],
        "thread_id": r[4], "return_value": r[5],
    } for r in q(
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "SELECT r.start, r.end, s.value, r.correlationId, r.globalTid, "
        "r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id"
    )]
    nvtx = [{
        "start": r[0], "end": r[1], "event_type": r[2], "text": r[3],
        "range_id": r[4], "thread_id": r[5],
    } for r in q(
        "NVTX_EVENTS",
        "SELECT start, end, eventType, text, rangeId, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND end > start"
    )]
    session_start = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    db.close()
    return {
        "kernels": kernels,
        "memcpys": memcpys,
        "memsets": memsets,
        "syncs": syncs,
        "api_calls": api_calls,
        "nvtx": nvtx,
        "session_start": session_start,
    }


TS_RE = re.compile(r"@(\d+)\s*$")


def align_phases(err_path, session_start_epoch_ns, nvtx_ranges=None):
    """Build phase windows, preferring NVTX ranges over timer-line stamps.

    NVTX ranges are recorded by nsys in its own clock and land in the same
    sqlite export, so they need no cross-clock alignment; when present they
    replace the cuda-family stamp phases entirely. Engine-family stamps (both
    chains) always come from the `@<epoch_ns>` suffixes: the nsys timeline
    zero is the session start (TARGET_INFO_SESSION_START_TIME in UTC epoch
    ns), so `line_epoch - session_start` places each phase end directly on
    the CUDA timeline; the start is end minus the reported elapsed.
    """
    phases, stamped, total = [], 0, 0
    for line in open(err_path, encoding="utf-8", errors="replace"):
        line = line.rstrip()
        tsm = TS_RE.search(line)
        parsed = parse_phase_line(line)
        if parsed:
            total += 1
        if not (tsm and parsed):
            continue
        stamped += 1
        label, family, elapsed_ns = parsed
        end = int(tsm.group(1)) - session_start_epoch_ns
        phases.append({
            "label": label,
            "family": family,
            "start": end - elapsed_ns,
            "end": end,
            "wall_ms": elapsed_ns / 1e6,
        })
    coverage = stamped / max(total, 1)

    nvtx_phases = [{
        "label": r["text"],
        "family": "cuda",
        "start": r["start"],
        "end": r["end"],
        "wall_ms": (r["end"] - r["start"]) / 1e6,
        "source": "nvtx",
    } for r in (nvtx_ranges or []) if r.get("text")]
    if nvtx_phases:
        phases = [p for p in phases if p["family"] != "cuda"] + nvtx_phases

    # Chain tagging: [neo-prover-cuda] lines only exist on the GPU chain;
    # engine lines fire on both chains, split by where the GPU chain's
    # first adapter stage begins.
    adapter_starts = [p["start"] for p in phases if p["family"] == "cuda"]
    boundary = min(adapter_starts) if adapter_starts else 0
    for p in phases:
        if p["family"] == "cuda":
            p["chain"] = "gpu"
        else:
            p["chain"] = "gpu" if p["end"] >= boundary else "cpu"

    # Mark the last fold as the terminal (finalize) fold.
    ingest_starts = sorted(p["start"] for p in phases if p["label"] == "fold.ingest")
    if len(ingest_starts) >= 2:
        last = ingest_starts[-1]
        fold_end = max(p["end"] for p in phases if p["start"] >= last)
        phases.append({"label": "finalize.terminal_fold", "start": last,
                       "end": fold_end, "wall_ms": (fold_end - last) / 1e6,
                       "chain": "gpu", "synthetic": True})
    return phases, coverage


def find_phase_index(phases, order, t):
    for i in order:
        p = phases[i]
        if p["start"] <= t <= p["end"]:
            return i
    return None


def attribute(phases, kernels, memcpys, memsets, syncs, api_calls):
    """Attribute CUDA activity to phase windows.

    Device work (kernels/memcpys/memsets) is attributed by ENQUEUE time when
    its launching API row is known via correlation id — the phase that issued
    the work owns it even when it executes later under async command streams —
    falling back to the execution midpoint. Sync and API rows attribute by
    their own midpoint. Idle is decomposed per phase window on the execution
    timeline into sync-wait / other-API / host-gap components.
    """
    # Innermost-first: sort candidate windows by duration so the smallest
    # containing window claims each activity's attribution time.
    order = sorted(range(len(phases)), key=lambda i: phases[i]["end"] - phases[i]["start"])
    api_by_corr = {a["correlation_id"]: a for a in api_calls if a.get("correlation_id")}

    def enqueue_time(activity):
        api = api_by_corr.get(activity.get("correlation_id"))
        if api is not None:
            return (api["start"] + api["end"]) / 2
        return (activity["start"] + activity["end"]) / 2

    stats = defaultdict(lambda: defaultdict(float))
    kernel_union = sorted((k["start"], k["end"]) for k in kernels)

    for k in kernels:
        s, e, name = k["start"], k["end"], k["name"]
        i = find_phase_index(phases, order, enqueue_time(k))
        if i is not None:
            elapsed_ms = (e - s) / 1e6
            st = stats[i]
            st["gpu_ms"] += elapsed_ms
            st["launches"] += 1
            st.setdefault("kernels", defaultdict(float))[name] += elapsed_ms
    for c in memcpys:
        s, e, b, kind = c["start"], c["end"], c["bytes"], c["copy_kind"]
        i = find_phase_index(phases, order, enqueue_time(c))
        if i is not None:
            st = stats[i]
            key = "h2d" if kind == 1 else "d2h" if kind == 2 else "dtod"
            st[f"{key}_mb"] += b / 1e6
            st[f"{key}_ms"] += (e - s) / 1e6
            st[f"{key}_copies"] += 1
    for m in memsets:
        s, e, b = m["start"], m["end"], m["bytes"]
        i = find_phase_index(phases, order, enqueue_time(m))
        if i is not None:
            stats[i]["memset_mb"] += b / 1e6
            stats[i]["memset_ms"] += (e - s) / 1e6
            stats[i]["memset_count"] += 1
    for s0 in syncs:
        s, e = s0["start"], s0["end"]
        i = find_phase_index(phases, order, (s + e) / 2)
        if i is None:
            continue
        busy = 0.0
        for ks, ke in kernel_union:
            if ks > e:
                break
            busy += overlap(s, e, ks, ke)
        stats[i]["sync_ms"] += (e - s) / 1e6
        stats[i]["sync_idle_ms"] += max(0.0, (e - s) - busy) / 1e6
        stats[i]["syncs"] += 1
    for api in api_calls:
        s, e, name = api["start"], api["end"], api["name"]
        i = find_phase_index(phases, order, (s + e) / 2)
        if i is None:
            continue
        elapsed_ms = (e - s) / 1e6
        st = stats[i]
        bucket = api_bucket(name)
        st["api_ms"] += elapsed_ms
        st["api_calls"] += 1
        st[f"api_{bucket}_ms"] += elapsed_ms
        st[f"api_{bucket}_calls"] += 1
        st.setdefault("cuda_api", defaultdict(float))[name] += elapsed_ms

    # Idle decomposition: for each phase window, the time not covered by any
    # device activity (execution timeline), split by what the host was doing
    # in the gap — blocked in a sync call, in other CUDA API calls (launch
    # latency, allocs), or pure host think-time.
    device_union = interval_union(
        [(k["start"], k["end"]) for k in kernels]
        + [(c["start"], c["end"]) for c in memcpys]
        + [(m["start"], m["end"]) for m in memsets]
    )
    sync_union = interval_union(
        [(a["start"], a["end"]) for a in api_calls if api_bucket(a["name"]) == "sync"]
    )
    other_api_union = interval_union(
        [(a["start"], a["end"]) for a in api_calls if api_bucket(a["name"]) != "sync"]
    )
    for i, p in enumerate(phases):
        if p.get("synthetic"):
            continue
        idle = sync_wait = api_time = 0.0
        for gs, ge in complement_intervals(p["start"], p["end"], device_union):
            idle += ge - gs
            sync_wait += union_overlap(gs, ge, sync_union)
            api_time += union_overlap(gs, ge, other_api_union)
        st = stats[i]
        st["idle_ms"] += idle / 1e6
        st["idle_sync_ms"] += sync_wait / 1e6
        st["idle_api_ms"] += api_time / 1e6
        st["idle_host_ms"] += max(0.0, idle - sync_wait - api_time) / 1e6
    return stats


ONLINE_RE = re.compile(
    r"online prove cpu=([\d.]+)ms gpu=([\d.]+)ms \(([\d.]+)x\)"
)
APPEND_RE = re.compile(r"append\s+(\d+): cpu ([\d.]+)ms\s+gpu ([\d.]+)ms")
FINISH_RE = re.compile(r"finish\s+: cpu ([\d.]+)ms\s+gpu ([\d.]+)ms")
SYNTH_RE = re.compile(r"synth (\d+)ms\s+setup (\d+)ms\s+cuda-init (\d+)ms")


def top_kernels(kernels, limit=2):
    if not kernels:
        return "."
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:limit]
    return ", ".join(f"{short_kernel_name(k)}:{v:.0f}" for k, v in top)


def top_items(items, limit):
    if not items:
        return []
    return [
        {"name": name, "ms": elapsed_ms}
        for name, elapsed_ms in sorted(items.items(), key=lambda kv: -kv[1])[:limit]
    ]


def top_kernel_items(kernels, limit=8):
    return top_items(kernels, limit)


def top_api_items(calls, limit=12):
    return top_items(calls, limit)


def merge_kernel_ms(dst, src):
    kernels = src.get("kernels")
    if not kernels:
        return
    bucket = dst.setdefault("kernels", defaultdict(float))
    for name, elapsed_ms in kernels.items():
        bucket[name] += elapsed_ms


def merge_api_ms(dst, src):
    calls = src.get("cuda_api")
    if not calls:
        return
    bucket = dst.setdefault("cuda_api", defaultdict(float))
    for name, elapsed_ms in calls.items():
        bucket[name] += elapsed_ms


def build_tree(phases, stats):
    agg = defaultdict(lambda: defaultdict(float))
    for i, p in enumerate(phases):
        short = p["label"].split(":", 1)[-1]
        a = agg[short]
        chain = p.get("chain", "gpu")
        a[f"n_{chain}"] += 1
        a[f"wall_{chain}"] += p["wall_ms"]
        if chain == "gpu" and not p.get("synthetic"):
            st = stats.get(i, {})
            for c in NUMERIC_COLS:
                a[c] += st.get(c, 0.0)
            merge_kernel_ms(a, st)
            merge_api_ms(a, st)

    tree = defaultdict(lambda: defaultdict(float))
    for label, a in agg.items():
        parts = label.split(".")
        for depth in range(1, len(parts) + 1):
            node = ".".join(parts[:depth])
            t = tree[node]
            for c, v in a.items():
                if c in ("kernels", "cuda_api"):
                    continue
                t[c] += v
            merge_kernel_ms(t, a)
            merge_api_ms(t, a)
        tree[label]["_leaf"] = 1.0

    return tree


def ordered_nodes(tree, include_empty=False):
    if include_empty:
        ordered = list(STAGE_ORDER)
    else:
        ordered = [s for s in STAGE_ORDER if s in tree]
    ordered += sorted(set(tree) - set(ordered))
    return ordered


def parse_run_summary(summary_lines):
    out = {"raw": summary_lines, "appends": []}
    for line in summary_lines:
        m = ONLINE_RE.search(line)
        if m:
            out["online"] = {
                "cpu_ms": float(m.group(1)),
                "cuda_ms": float(m.group(2)),
                "speedup": float(m.group(3)),
                "proof": "byte-identical",
            }
            continue
        m = APPEND_RE.search(line)
        if m:
            out["appends"].append({
                "index": int(m.group(1)),
                "cpu_ms": float(m.group(2)),
                "cuda_ms": float(m.group(3)),
            })
            continue
        m = FINISH_RE.search(line)
        if m:
            out["finish"] = {"cpu_ms": float(m.group(1)), "cuda_ms": float(m.group(2))}
            continue
        m = SYNTH_RE.search(line)
        if m:
            out["setup"] = {
                "synth_ms": int(m.group(1)),
                "setup_ms": int(m.group(2)),
                "cuda_init_ms": int(m.group(3)),
            }
    return out


def print_summary(summary_lines, parsed):
    print("SUMMARY")
    print("-------")
    online = parsed.get("online")
    if online:
        cpu_ms = online["cpu_ms"]
        gpu_ms = online["cuda_ms"]
        speedup = online["speedup"]
        hdr = f"{'path':<8}{'online ms':>12}{'speedup':>10}  proof"
        print(hdr)
        print("-" * len(hdr))
        print(f"{'CPU':<8}{cpu_ms:>12.1f}{'1.00x':>10}  canonical")
        print(f"{'CUDA':<8}{gpu_ms:>12.1f}{speedup:>9.2f}x  byte-identical")
    else:
        print("online prove summary not found")
    for line in summary_lines:
        print(f"  {line}")
    print()


def print_cpu_table(tree, parsed):
    online = parsed.get("online")
    total = online["cpu_ms"] if online else 0.0
    rows = [n for n in ordered_nodes(tree) if tree[n].get("wall_cpu", 0.0) >= 0.05]
    if not rows:
        print("CPU REFERENCE BREAKDOWN")
        print("-----------------------")
        print("no CPU-chain phase timers were stamped")
        print()
        return

    print("CPU REFERENCE BREAKDOWN")
    print("-----------------------")
    hdr = f"{'stage':<46}{'n':>4}{'wall ms':>10}{'% online':>10}"
    print(hdr)
    print("-" * len(hdr))
    for node in rows:
        t = tree[node]
        depth = node.count(".")
        name = ("  " * depth) + node.split(".")[-1]
        wall = t.get("wall_cpu", 0.0)
        pct = f"{100.0 * wall / total:.1f}%" if total else "."
        print(f"{name:<46}{int(t.get('n_cpu', 0)):>4}{wall:>10.1f}{pct:>10}")
    print()


def print_cuda_table(tree):
    print("CUDA PATH DIAGNOSTICS (NSYS)")
    print("----------------------------")
    hdr = (
        f"{'stage':<42}{'n':>4}{'wall':>8}{'busy':>8}{'util':>7}{'idle':>8}"
        f"{'idle s/a/h':>13}{'launch':>8}{'avgKus':>8}{'H2D MB/n':>11}{'D2H MB/n':>11}"
        f"{'sync idle/n':>13}{'memset MB/n':>13}{'api ms/n':>11}  top kernels(ms)"
    )
    print(hdr)
    print("-" * len(hdr))
    for node in ordered_nodes(tree, include_empty=True):
        t = tree.get(node, {})
        depth = node.count(".")
        name = ("  " * depth) + node.split(".")[-1]
        wall = t.get("wall_gpu", 0.0)
        busy = t.get("gpu_ms", 0.0)
        idle = max(0.0, wall - busy)
        launches = t.get("launches", 0.0)
        util = 100.0 * busy / wall if wall > 0.0 else 0.0
        avg_kernel_us = 1000.0 * busy / launches if launches else 0.0
        sync = "."
        if t.get("sync_idle_ms", 0.0) >= 0.05 or t.get("syncs", 0.0) >= 0.5:
            sync = f"{t.get('sync_idle_ms', 0.0):.1f}/{int(t.get('syncs', 0.0))}"
        memset = fmt_mb_count(t.get("memset_mb", 0.0), t.get("memset_count", 0.0))
        api = "."
        if t.get("api_ms", 0.0) >= 0.05 or t.get("api_calls", 0.0) >= 0.5:
            api = f"{t.get('api_ms', 0.0):.1f}/{int(t.get('api_calls', 0.0))}"
        split = "."
        if t.get("idle_ms", 0.0) >= 0.05:
            split = (
                f"{t.get('idle_sync_ms', 0.0):.0f}/{t.get('idle_api_ms', 0.0):.0f}"
                f"/{t.get('idle_host_ms', 0.0):.0f}"
            )
        print(
            f"{name:<42}{int(t.get('n_gpu', 0)):>4}"
            f"{fmt_ms(wall):>8}{fmt_ms(busy):>8}{fmt_pct(util):>7}{fmt_ms(idle):>8}"
            f"{split:>13}{fmt_count(launches):>8}{fmt_ms(avg_kernel_us):>8}"
            f"{fmt_mb_count(t.get('h2d_mb', 0.0), t.get('h2d_copies', 0.0)):>11}"
            f"{fmt_mb_count(t.get('d2h_mb', 0.0), t.get('d2h_copies', 0.0)):>11}"
            f"{sync:>13}{memset:>13}{api:>11}  {top_kernels(t.get('kernels'))}"
        )
    print("-" * len(hdr))
    print("wall/busy/idle are GPU-chain stage wall, CUDA kernel time, and non-kernel wall.")
    print("idle s/a/h splits measured idle into sync-wait / other CUDA API / host think-time (ms).")
    print("Device work is attributed to the phase that ENQUEUED it (correlation id), so async")
    print("work counts where it was issued; busy can exceed wall for fire-and-forget stages.")
    print("H2D/D2H and memset columns are MB/count. sync idle/n is idle sync wait ms/count.")
    print("api ms/n is CUDA runtime/driver API wall time and call count from nsys.")
    print("Parent rows are subtree totals; use leaf rows for exclusive bottlenecks.")
    print()


def plain_tree(tree):
    out = {}
    for label, t in tree.items():
        row = {}
        for key, value in t.items():
            if key == "_leaf":
                continue
            if key in ("kernels", "cuda_api"):
                row[key] = dict(value)
            else:
                row[key] = value
        out[label] = row
    return out


def plain_stats(stats):
    row = {}
    for key, value in stats.items():
        if key in ("kernels", "cuda_api"):
            row[key] = dict(value)
        else:
            row[key] = value
    if "kernels" in row:
        row["top_kernels"] = top_kernel_items(row["kernels"])
    if "cuda_api" in row:
        row["top_cuda_api"] = top_api_items(row["cuda_api"])
    return row


def phase_trace(phases, stats):
    trace = []
    for i, p in enumerate(phases):
        trace.append({
            "index": i,
            "stage": p["label"].split(":", 1)[-1],
            "raw_label": p["label"],
            "family": p.get("family"),
            "chain": p.get("chain", "gpu"),
            "synthetic": bool(p.get("synthetic")),
            "start_ms": p["start"] / 1e6,
            "end_ms": p["end"] / 1e6,
            "wall_ms": p["wall_ms"],
            "nsys": plain_stats(stats.get(i, {})),
        })
    return trace


def stage_for_time(phases, t):
    order = sorted(range(len(phases)), key=lambda i: phases[i]["end"] - phases[i]["start"])
    idx = find_phase_index(phases, order, t)
    if idx is None:
        return None
    return phases[idx]["label"].split(":", 1)[-1]


def nvtx_stage_for_time(nvtx, t):
    matches = [r for r in nvtx if r["start"] <= t <= r["end"]]
    if not matches:
        return None
    return min(matches, key=lambda r: r["end"] - r["start"])["text"]


def kernel_enqueue_attribution(timeline):
    api_by_corr = {a["correlation_id"]: a for a in timeline["api_calls"]}
    out = []
    for k in timeline["kernels"]:
        api = api_by_corr.get(k.get("correlation_id"))
        if not api:
            continue
        out.append({
            "kernel": k["name"],
            "kernel_start_ms": k["start"] / 1e6,
            "kernel_ms": (k["end"] - k["start"]) / 1e6,
            "correlation_id": k.get("correlation_id"),
            "api_name": api["name"],
            "api_start_ms": api["start"] / 1e6,
            "api_ms": (api["end"] - api["start"]) / 1e6,
            "nvtx_stage": nvtx_stage_for_time(timeline["nvtx"], (api["start"] + api["end"]) / 2),
        })
    return out


def summarize_range(phases, timeline, start, end):
    p = [{"label": "range", "start": start, "end": end, "wall_ms": (end - start) / 1e6}]
    s = attribute(
        p,
        timeline["kernels"],
        timeline["memcpys"],
        timeline["memsets"],
        timeline["syncs"],
        timeline["api_calls"],
    )
    return plain_stats(s.get(0, {}))


def build_fold_segments(phases, timeline, parsed):
    ingests = sorted((
        p for p in phases
        if p.get("chain") == "gpu" and p["label"].split(":", 1)[-1] == "fold.ingest"
    ), key=lambda p: p["start"])
    if not ingests:
        return []
    max_end = max(p["end"] for p in phases if p.get("chain") == "gpu")
    appends = {row["index"]: row for row in parsed.get("appends", [])}
    segments = []
    for i, ingest in enumerate(ingests):
        start = ingest["start"]
        end = ingests[i + 1]["start"] if i + 1 < len(ingests) else max_end
        child_phases = [
            {
                "stage": p["label"].split(":", 1)[-1],
                "wall_ms": p["wall_ms"],
                "start_ms": p["start"] / 1e6,
                "end_ms": p["end"] / 1e6,
            }
            for p in phases
            if p.get("chain") == "gpu" and start <= (p["start"] + p["end"]) / 2 <= end
        ]
        segments.append({
            "index": i,
            "label": f"append_{i}",
            "start_ms": start / 1e6,
            "end_ms": end / 1e6,
            "wall_ms": (end - start) / 1e6,
            "summary_wall_ms": appends.get(i, {}).get("cuda_ms"),
            "nsys": summarize_range(phases, timeline, start, end),
            "phases": child_phases,
        })
    if "finish" in parsed:
        segments.append({
            "index": len(segments),
            "label": "finish_summary",
            "summary_wall_ms": parsed["finish"]["cuda_ms"],
            "note": "summary-only unless a distinct finish phase is stamped",
        })
    return segments


def activity_rows(timeline):
    rows = []
    for k in timeline["kernels"]:
        rows.append({
            "kind": "kernel", "name": k["name"], "start": k["start"], "end": k["end"],
            "stream_id": k.get("stream_id"), "correlation_id": k.get("correlation_id"),
        })
    for c in timeline["memcpys"]:
        rows.append({
            "kind": "memcpy", "name": copy_kind_name(c["copy_kind"]),
            "start": c["start"], "end": c["end"], "bytes": c["bytes"],
            "stream_id": c.get("stream_id"), "correlation_id": c.get("correlation_id"),
        })
    for m in timeline["memsets"]:
        rows.append({
            "kind": "memset", "name": "memset", "start": m["start"], "end": m["end"],
            "bytes": m["bytes"], "stream_id": m.get("stream_id"),
            "correlation_id": m.get("correlation_id"),
        })
    return sorted(rows, key=lambda r: (r["start"], r["end"]))


def critical_path_inputs(phases, timeline, limit=40):
    activities = activity_rows(timeline)
    sync_waits = []
    for s in timeline["syncs"]:
        overlaps = []
        for a in activities:
            ov = overlap(s["start"], s["end"], a["start"], a["end"])
            if ov > 0:
                overlaps.append({
                    "kind": a["kind"], "name": a["name"], "overlap_ms": ov / 1e6,
                    "correlation_id": a.get("correlation_id"),
                })
        busy = sum(o["overlap_ms"] for o in overlaps)
        elapsed = (s["end"] - s["start"]) / 1e6
        sync_waits.append({
            "stage": stage_for_time(phases, (s["start"] + s["end"]) / 2),
            "start_ms": s["start"] / 1e6,
            "end_ms": s["end"] / 1e6,
            "elapsed_ms": elapsed,
            "busy_overlap_ms": busy,
            "idle_ms": max(0.0, elapsed - busy),
            "sync_type": s.get("sync_type"),
            "sync_label": s.get("sync_label"),
            "correlation_id": s.get("correlation_id"),
            "overlapped_activity": sorted(overlaps, key=lambda x: -x["overlap_ms"])[:8],
        })

    idle_gaps = []
    last = None
    for a in activities:
        if last and a["start"] > last["end"]:
            gap_ms = (a["start"] - last["end"]) / 1e6
            if gap_ms >= 0.05:
                idle_gaps.append({
                    "start_ms": last["end"] / 1e6,
                    "end_ms": a["start"] / 1e6,
                    "gap_ms": gap_ms,
                    "stage": stage_for_time(phases, (last["end"] + a["start"]) / 2),
                    "previous": {"kind": last["kind"], "name": last["name"]},
                    "next": {"kind": a["kind"], "name": a["name"]},
                })
        if last is None or a["end"] > last["end"]:
            last = a

    api = [{
        "stage": stage_for_time(phases, (c["start"] + c["end"]) / 2),
        "name": c["name"],
        "bucket": api_bucket(c["name"]),
        "elapsed_ms": (c["end"] - c["start"]) / 1e6,
        "start_ms": c["start"] / 1e6,
        "end_ms": c["end"] / 1e6,
        "correlation_id": c.get("correlation_id"),
        "return_value": c.get("return_value"),
    } for c in timeline["api_calls"]]

    return {
        "sync_waits": sorted(sync_waits, key=lambda x: -x["elapsed_ms"])[:limit],
        "gpu_idle_gaps": sorted(idle_gaps, key=lambda x: -x["gap_ms"])[:limit],
        "long_cuda_api_calls": sorted(api, key=lambda x: -x["elapsed_ms"])[:limit],
    }


def build_residency_ledger(stages):
    def row(name, stage, producer, consumer, resident):
        s = stages.get(stage, {})
        return {
            "buffer": name,
            "stage": stage,
            "producer": producer,
            "consumer": consumer,
            "intended_residency": resident,
            "measured_h2d_mb": s.get("h2d_mb", 0.0),
            "measured_h2d_copies": s.get("h2d_copies", 0.0),
            "measured_d2h_mb": s.get("d2h_mb", 0.0),
            "measured_d2h_copies": s.get("d2h_copies", 0.0),
            "measured_dtod_mb": s.get("dtod_mb", 0.0),
            "measured_dtod_copies": s.get("dtod_copies", 0.0),
        }
    return [
        row("bar/row matrices", "session.structure", "host setup", "ccs/dec", "one-time H2D"),
        row("fresh CCS planes", "fold.ingest.fresh", "host witness", "commit/ccs/rlc/dec", "H2D per fresh fold"),
        row("running child planes", "fold.ingest.running", "previous fold", "next fold", "device-retained/D2D"),
        row("oracle row/eval tables", "fold.superneo.pi_ccs.oracle", "Pi_CCS", "FE/NC sumcheck", "device scratch"),
        row("mixed witness z", "fold.superneo.pi_rlc.mix_witness", "Pi_RLC", "Pi_DEC", "device resident"),
        row("DEC child openings", "fold.superneo.pi_dec.open_children", "Pi_DEC", "emit/proof", "device until emit"),
        row("DEC emitted planes", "fold.superneo.pi_dec.emit.planes", "Pi_DEC", "host proof/public material", "D2H observed here"),
        row("final proof/public", "finalize.proof_export", "finalize", "host caller", "final D2H only"),
    ]


def render(phases, stats, coverage, summary_lines, tree=None, parsed=None, repeat_onlines=None):
    tree = tree if tree is not None else build_tree(phases, stats)
    parsed = parsed if parsed is not None else parse_run_summary(summary_lines)
    print_summary(summary_lines, parsed)
    if repeat_onlines and len(repeat_onlines) > 1:
        vals = ", ".join(f"{v:.1f}" for v in repeat_onlines)
        spread = (max(repeat_onlines) - min(repeat_onlines)) / max(min(repeat_onlines), 1e-9) * 100
        print(f"repeat: {len(repeat_onlines)} runs, online cuda ms = [{vals}] "
              f"(spread {spread:.1f}%); tables show per-stage medians")
        print()
    print_cpu_table(tree, parsed)
    print_cuda_table(tree)
    nvtx_count = sum(1 for p in phases if p.get("source") == "nvtx")
    if nvtx_count:
        print(f"phases: {nvtx_count} from NVTX ranges (profiler clock), "
              f"{len(phases) - nvtx_count} from timer stamps; "
              f"coverage: {coverage:.0%} of timer lines stamped")
    else:
        print(f"coverage: {coverage:.0%} of timer lines stamped "
              "(no NVTX ranges found — binary predates NVTX support?)")
    print("json trace fields: summary, stages, phase_trace")
    stages = plain_tree(tree)
    return {
        "summary": parsed,
        "coverage": coverage,
        "stages": stages,
        "phases": stages,
        "phase_trace": phase_trace(phases, stats),
    }


SKIP_MEDIAN_KEYS = {"kernels", "cuda_api", "_leaf"}


def median_tree(trees):
    out = {}
    for label in set().union(*trees):
        rows = [t.get(label, {}) for t in trees]
        keys = set().union(*rows) - SKIP_MEDIAN_KEYS
        row = {k: median(r.get(k, 0.0) for r in rows) for k in keys}
        last = rows[-1]
        for k in ("kernels", "cuda_api"):
            if k in last:
                row[k] = last[k]
        out[label] = row
    return out


def median_parsed(parsed_runs):
    out = dict(parsed_runs[-1])
    onlines = [p["online"] for p in parsed_runs if p.get("online")]
    if onlines:
        out["online"] = {
            "cpu_ms": median(o["cpu_ms"] for o in onlines),
            "cuda_ms": median(o["cuda_ms"] for o in onlines),
            "speedup": median(o["speedup"] for o in onlines),
            "proof": "byte-identical",
            "runs": len(onlines),
        }
    return out


def run_once(args, workdir):
    sqlite_path, out_path, err_path, rep = capture(args.binary, args.gate, workdir)
    timeline = load_timeline(sqlite_path)
    phases, coverage = align_phases(err_path, timeline["session_start"], timeline["nvtx"])
    if not phases:
        sys.exit(
            f"error: gate ran but emitted no stamped timer lines ({err_path}).\n"
            f"{args.binary} was built without the `perf-timers` feature. Rebuild:\n"
            "  cd crates/neo-prover-cuda && "
            "cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers"
        )
    stats = attribute(
        phases, timeline["kernels"], timeline["memcpys"], timeline["memsets"],
        timeline["syncs"], timeline["api_calls"],
    )
    summary = [l for l in open(out_path).read().splitlines() if "OK:" in l]
    summary += [l.strip() for l in open(err_path).read().splitlines()
                if re.match(r"\s+(append|finish|synth)", l)]
    return {
        "sqlite_path": sqlite_path, "out_path": out_path, "err_path": err_path,
        "rep": rep, "timeline": timeline, "phases": phases, "coverage": coverage,
        "stats": stats, "summary": summary, "parsed": parse_run_summary(summary),
        "tree": build_tree(phases, stats),
    }


def cmd_run(args):
    repo_root = os.getcwd()
    workdir = args.artifacts or tempfile.mkdtemp(prefix="gpuprof-")
    os.makedirs(workdir, exist_ok=True)
    if not args.trace_json and args.artifacts:
        args.trace_json = os.path.join(workdir, "trace.json")
    if not args.metadata_json and args.artifacts:
        args.metadata_json = os.path.join(workdir, "metadata.json")
    repeat = max(1, args.repeat)
    runs = []
    for ridx in range(repeat):
        rdir = workdir if repeat == 1 else os.path.join(workdir, f"run-{ridx + 1:02d}")
        os.makedirs(rdir, exist_ok=True)
        runs.append(run_once(args, rdir))
    last = runs[-1]
    timeline = last["timeline"]
    phases, stats = last["phases"], last["stats"]
    kernels, memcpys = timeline["kernels"], timeline["memcpys"]
    memsets, syncs, api_calls = timeline["memsets"], timeline["syncs"], timeline["api_calls"]
    if repeat > 1:
        tree = median_tree([r["tree"] for r in runs])
        parsed = median_parsed([r["parsed"] for r in runs])
        repeat_onlines = [r["parsed"]["online"]["cuda_ms"] for r in runs if r["parsed"].get("online")]
    else:
        tree, parsed, repeat_onlines = last["tree"], last["parsed"], None
    report = render(
        phases, stats, last["coverage"], last["summary"],
        tree=tree, parsed=parsed, repeat_onlines=repeat_onlines,
    )
    report["runs"] = [{"online": r["parsed"].get("online")} for r in runs]

    online_cuda_ms = (parsed.get("online") or {}).get("cuda_ms")
    report["lint"] = lint_kernels(kernels)
    report["levers"] = build_levers(report["stages"], online_cuda_ms, memcpys)
    residency_results, residency_failures = check_residency(report["stages"])
    report["residency_check"] = residency_results
    report["boundary_scorecard"] = build_boundary_scorecard(report["stages"])
    report["superneo_context"] = build_superneo_context(report["stages"], report["lint"])
    print_lint(report["lint"])
    print_levers(report["levers"], online_cuda_ms)
    print_residency(residency_results, residency_failures)
    print_boundary_scorecard(report["boundary_scorecard"])
    print_superneo_context(report["superneo_context"])

    metadata = collect_metadata(repo_root)
    trace = chrome_trace(phases, kernels, memcpys, memsets, syncs, api_calls, timeline["nvtx"])
    report["fold_segments"] = build_fold_segments(phases, timeline, report["summary"])
    report["critical_path_inputs"] = critical_path_inputs(phases, timeline)
    report["memory_residency_ledger"] = build_residency_ledger(report["stages"])
    report["source_map"] = SOURCE_MAP
    report["nvtx_ranges"] = [{
        "stage": r["text"], "start_ms": r["start"] / 1e6, "end_ms": r["end"] / 1e6,
        "wall_ms": (r["end"] - r["start"]) / 1e6, "range_id": r.get("range_id"),
        "event_type": r.get("event_type"), "thread_id": r.get("thread_id"),
    } for r in timeline["nvtx"]]
    report["kernel_enqueue_attribution"] = kernel_enqueue_attribution(timeline)
    report["cuda_api_trace"] = [{
        "stage": stage_for_time(phases, (c["start"] + c["end"]) / 2),
        "nvtx_stage": nvtx_stage_for_time(timeline["nvtx"], (c["start"] + c["end"]) / 2),
        "name": c["name"],
        "bucket": api_bucket(c["name"]),
        "start_ms": c["start"] / 1e6,
        "end_ms": c["end"] / 1e6,
        "elapsed_ms": (c["end"] - c["start"]) / 1e6,
        "correlation_id": c.get("correlation_id"),
        "thread_id": c.get("thread_id"),
        "return_value": c.get("return_value"),
    } for c in api_calls]
    report["artifacts"] = {
        "directory": workdir,
        "nsys_report": last["rep"],
        "nsys_sqlite": last["sqlite_path"],
        "stdout": last["out_path"],
        "stderr": last["err_path"],
    }
    analysis_dir = os.path.join(workdir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    protocol_dag_path = os.path.join(analysis_dir, "superneo_dag.json")
    parallelism_path = os.path.join(analysis_dir, "parallelism_scorecard.json")
    overlap_path = os.path.join(analysis_dir, "overlap_candidates.json")
    write_json(
        protocol_dag_path,
        {
            "schema_version": report["superneo_context"]["schema_version"],
            "rule": report["superneo_context"]["rule"],
            "nodes": report["superneo_context"]["nodes"],
            "edges": report["superneo_context"]["edges"],
        },
    )
    write_json(
        parallelism_path,
        {
            "schema_version": report["superneo_context"]["schema_version"],
            "nodes": report["superneo_context"]["nodes"],
            "migration_queue": report["superneo_context"]["migration_queue"],
        },
    )
    write_json(
        overlap_path,
        {
            "schema_version": report["superneo_context"]["schema_version"],
            "overlap_candidates": report["superneo_context"]["overlap_candidates"],
            "blocked_parallelism": report["superneo_context"]["blocked_parallelism"],
        },
    )
    report["artifacts"]["superneo_dag_json"] = protocol_dag_path
    report["artifacts"]["parallelism_scorecard_json"] = parallelism_path
    report["artifacts"]["overlap_candidates_json"] = overlap_path
    if args.trace_json:
        write_json(args.trace_json, trace)
        report["artifacts"]["trace_json"] = args.trace_json
    if args.metadata_json:
        write_json(args.metadata_json, metadata)
        report["artifacts"]["metadata_json"] = args.metadata_json
    report["metadata"] = metadata
    report["ncu"] = run_ncu_profiles(args, report["phase_trace"], workdir)
    report["cpu_profile"] = run_cpu_profiles(args, workdir)
    report["sanitizer"] = run_sanitizers(args, workdir)
    report["external_tools"] = run_external_tools(args, workdir)
    if args.json:
        write_json(args.json, report)
        print(f"json: {args.json}")
    if args.keep_rep:
        print(f"nsys-rep: {last['rep']}")
    append_history(args.gate, report, args.json)
    print(f"artifacts: {workdir}")
    if args.assert_residency and residency_failures:
        sys.exit("error: residency budgets violated:\n  " + "\n  ".join(residency_failures))


def stage_wall(row):
    return row.get("wall_gpu", row.get("wall_ms", 0.0))


def cmd_diff(args):
    a_report = json.load(open(args.a))
    b_report = json.load(open(args.b))
    a = report_stages(a_report)
    b = report_stages(b_report)
    old_online = report_online(a_report)
    new_online = report_online(b_report)
    if old_online or new_online:
        print("SUMMARY DELTA")
        print("-------------")
        for key in ("cpu_ms", "cuda_ms", "speedup"):
            av = old_online.get(key, 0.0)
            bv = new_online.get(key, 0.0)
            print(f"{key:<10}{av:>10.3f} -> {bv:>10.3f}  delta {bv - av:+.3f}")
        print()

    hdr = (
        f"{'phase':<42}{'wall Δ':>9}{'gpu Δ':>9}{'H2D Δ':>9}{'D2H Δ':>9}"
        f"{'launch Δ':>10}{'syncidle Δ':>12}{'api Δ':>9}{'memset Δ':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label in sorted(set(a) | set(b), key=lambda l: -stage_wall(b.get(l, {}))):
        av, bv = a.get(label, {}), b.get(label, {})
        dw = stage_wall(bv) - stage_wall(av)
        dg = bv.get("gpu_ms", 0) - av.get("gpu_ms", 0)
        dh = bv.get("h2d_mb", 0) - av.get("h2d_mb", 0)
        dd = bv.get("d2h_mb", 0) - av.get("d2h_mb", 0)
        dl = bv.get("launches", 0) - av.get("launches", 0)
        di = bv.get("sync_idle_ms", 0) - av.get("sync_idle_ms", 0)
        da = bv.get("api_ms", 0) - av.get("api_ms", 0)
        dm = bv.get("memset_mb", 0) - av.get("memset_mb", 0)
        if max(abs(dw), abs(dg), abs(dh), abs(dd), abs(dl), abs(di), abs(da), abs(dm)) > 0.5:
            print(
                f"{label:<42}{dw:>+9.1f}{dg:>+9.1f}{dh:>+9.1f}{dd:>+9.1f}"
                f"{dl:>+10.0f}{di:>+12.1f}{da:>+9.1f}{dm:>+10.1f}"
            )

    changed = sorted(
        (label for label in set(a) | set(b)
         if abs(stage_wall(b.get(label, {})) - stage_wall(a.get(label, {}))) > 2.0),
        key=lambda label: -abs(stage_wall(b.get(label, {})) - stage_wall(a.get(label, {}))),
    )
    causes_shown = 0
    for label in changed:
        causes = structural_causes(a.get(label, {}), b.get(label, {}))
        if not causes:
            continue
        if causes_shown == 0:
            print()
            print("STRUCTURAL CAUSES (stages with |wall Δ| > 2ms)")
            print("----------------------------------------------")
        dw = stage_wall(b.get(label, {})) - stage_wall(a.get(label, {}))
        print(f"{label}  wall {dw:+.1f}ms")
        for cause in causes:
            print(f"    {cause}")
        causes_shown += 1
        if causes_shown >= 10:
            break


HISTORY_PATH = os.path.join("benchmark-results", "gpuprof-history.jsonl")


def append_history(gate, report, json_path):
    """One line per run: the campaign trend log. Never breaks the run."""
    try:
        summary = report.get("summary")
        online = summary.get("online") if isinstance(summary, dict) else None
        head = (
            report.get("metadata", {}).get("git", {}).get("rev_parse_head", {})
            .get("stdout", "").strip() or None
        )
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gate": gate,
            "online": online,
            "git_head": head,
            "json": json_path,
            "stages": {
                label: round(row.get("wall_gpu", 0.0), 1)
                for label, row in report.get("stages", {}).items()
                if row.get("wall_gpu", 0.0) >= 5.0
            },
        }
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"history: appended to {HISTORY_PATH}")
    except Exception as exc:  # trend log must never fail a run
        print(f"history: skipped ({exc})")


def cmd_trend(args):
    if not os.path.exists(HISTORY_PATH):
        sys.exit(f"error: no history at {HISTORY_PATH} — run some gates first")
    entries = []
    for line in open(HISTORY_PATH):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("gate") == args.gate:
            entries.append(e)
    if not entries:
        sys.exit(f"error: no history entries for gate `{args.gate}`")
    entries = entries[-args.limit:]

    stage_hdr = f"{args.stage.split('.')[-1]:>12}" if args.stage else ""
    hdr = f"{'ts':<21}{'cuda ms':>10}{'Δ':>9}{'speedup':>9}{stage_hdr}  git"
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for e in entries:
        online = e.get("online") or {}
        cuda = online.get("cuda_ms")
        delta = f"{cuda - prev:+.1f}" if (cuda is not None and prev is not None) else "."
        speedup = f"{online.get('speedup'):.2f}x" if online.get("speedup") else "."
        stage_col = ""
        if args.stage:
            wall = e.get("stages", {}).get(args.stage)
            stage_col = f"{wall:>12.1f}" if wall is not None else f"{'.':>12}"
        head = (e.get("git_head") or "")[:9]
        cuda_txt = f"{cuda:.1f}" if cuda is not None else "."
        print(f"{e['ts']:<21}{cuda_txt:>10}{delta:>9}{speedup:>9}{stage_col}  {head}")
        if cuda is not None:
            prev = cuda
    print("-" * len(hdr))
    onlines = [e["online"]["cuda_ms"] for e in entries if e.get("online")]
    if len(onlines) >= 2:
        total = onlines[-1] - onlines[0]
        print(f"net over {len(onlines)} runs: {total:+.1f}ms "
              f"({onlines[0]:.1f} -> {onlines[-1]:.1f})")


def cmd_metadata(args):
    metadata = collect_metadata(os.getcwd())
    if args.json:
        write_json(args.json, metadata)
        print(f"metadata: {args.json}")
    else:
        print(json.dumps(metadata, indent=1))


def report_online(report):
    summary = report.get("summary")
    return (summary.get("online") or {}) if isinstance(summary, dict) else {}


def report_stages(report):
    return report.get("stages") or report.get("phases") or {}


def cmd_check(args):
    """Regression gate: exit 1 if the candidate run regresses past tolerance."""
    base = json.load(open(args.baseline))
    cand = json.load(open(args.candidate))
    tol, floor = args.tolerance, args.abs_floor_ms
    failures, checked = [], 0

    b_on = report_online(base)
    c_on = report_online(cand)
    if b_on and c_on:
        checked += 1
        limit = b_on["cuda_ms"] * (1 + tol) + floor
        regressed = c_on["cuda_ms"] > limit
        if regressed:
            failures.append("online.cuda_ms")
        print(f"[{'FAIL' if regressed else 'ok':>4}] online cuda_ms "
              f"{b_on['cuda_ms']:.1f} -> {c_on['cuda_ms']:.1f} (limit {limit:.1f})")

    b_stages = report_stages(base)
    c_stages = report_stages(cand)
    for label in sorted(set(b_stages) & set(c_stages)):
        bw = b_stages[label].get("wall_gpu", 0.0)
        cw = c_stages[label].get("wall_gpu", 0.0)
        if bw < args.min_stage_ms:
            continue
        checked += 1
        limit = bw * (1 + tol) + floor
        if cw > limit:
            failures.append(label)
            print(f"[FAIL] {label} wall_gpu {bw:.1f} -> {cw:.1f} (limit {limit:.1f})")
            for cause in structural_causes(b_stages[label], c_stages[label]):
                print(f"       {cause}")

    verdict = "REGRESSED" if failures else "clean"
    print(f"check: {checked} comparisons, {len(failures)} regressions — {verdict}")
    sys.exit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser(prog="gpuprof")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("gate")
    r.add_argument("--binary", default="./target/release/parity")
    r.add_argument("--json")
    r.add_argument("--trace-json")
    r.add_argument("--metadata-json")
    r.add_argument("--keep-rep", action="store_true")
    r.add_argument("--repeat", type=int, default=1,
                   help="Capture N runs; tables report per-stage medians")
    r.add_argument("--assert-residency", action="store_true",
                   help="Exit 1 if measured transfers violate the architecture's residency budgets")
    r.add_argument("--artifacts", help="Directory for nsys/ncu/sanitizer/external artifacts")
    r.add_argument("--ncu-top", type=int, default=0, help="Run Nsight Compute for the top N kernels from nsys")
    r.add_argument("--ncu-set", default="full", help="Nsight Compute section set")
    r.add_argument("--ncu-launch-skip", type=int, default=0, help="Matching launches to skip before NCU profiling")
    r.add_argument("--ncu-launch-count", type=int, default=1, help="Matching launches to profile per NCU kernel")
    r.add_argument(
        "--cpu-profile",
        choices=["none", "perf"],
        default="none",
        help="Collect a CPU call-stack profile by rerunning the gate under perf",
    )
    r.add_argument("--cpu-perf-freq", type=int, default=99, help="perf sampling frequency")
    r.add_argument(
        "--sanitize",
        action="append",
        default=[],
        choices=["memcheck", "racecheck", "initcheck", "synccheck", "all"],
        help="Run compute-sanitizer tool after nsys; can be repeated",
    )
    r.add_argument(
        "--external-tool",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="Run an installed external/OSS tool; placeholders: {binary}, {gate}, {artifact_dir}",
    )
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(fn=cmd_diff)
    m = sub.add_parser("metadata")
    m.add_argument("--json")
    m.set_defaults(fn=cmd_metadata)
    c = sub.add_parser("check", help="Regression gate between two run JSONs (exit 1 on regression)")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--tolerance", type=float, default=0.05,
                   help="Allowed relative regression (default 5%%)")
    c.add_argument("--abs-floor-ms", type=float, default=2.0,
                   help="Absolute slack added to every limit")
    c.add_argument("--min-stage-ms", type=float, default=10.0,
                   help="Ignore stages below this baseline wall")
    c.set_defaults(fn=cmd_check)
    t = sub.add_parser("trend", help="Show the campaign trajectory for a gate from the history log")
    t.add_argument("gate")
    t.add_argument("--limit", type=int, default=20, help="Show the most recent N entries")
    t.add_argument("--stage", help="Also track one stage's wall_gpu (full dotted label)")
    t.set_defaults(fn=cmd_trend)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
