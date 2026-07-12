"""Differential validation against legacy gpuprof reports."""

import json
from collections import defaultdict
from pathlib import Path

STAGE_FIELD_MAP = {
    "window_wall_ms": "wall_gpu",
    "kernel_attributed_ms": "gpu_ms",
    "launches": "launches",
    "h2d_mb": "h2d_mb",
    "h2d_ms": "h2d_ms",
    "h2d_copies": "h2d_copies",
    "d2h_mb": "d2h_mb",
    "d2h_ms": "d2h_ms",
    "d2h_copies": "d2h_copies",
    "dtod_mb": "dtod_mb",
    "dtod_ms": "dtod_ms",
    "dtod_copies": "dtod_copies",
    "memset_mb": "memset_mb",
    "memset_ms": "memset_ms",
    "memset_count": "memset_count",
    "sync_total_ms": "sync_ms",
    "syncs": "syncs",
    "api_total_ms": "api_ms",
    "api_calls": "api_calls",
}

SKIP_STAGE_REASONS = {
    "finalize": "legacy gpuprof synthetic finalize rows carry wall only, not attributed CUDA work",
    "finalize.terminal_fold": "legacy gpuprof synthetic terminal fold carries wall only, not attributed CUDA work",
}


def validate_against_gpuprof(report, phases, gpuprof_json_path):
    oracle_path = Path(gpuprof_json_path)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    nvtx = compare_nvtx_ranges(phases, oracle)
    stderr = compare_stderr_phases(phases, oracle)
    stages = compare_stage_attribution(report, oracle)
    return {
        "schema_version": 1,
        "oracle": str(oracle_path),
        "ok": nvtx["ok"] and stderr["ok"] and stages["ok"],
        "intentional_deltas": intentional_deltas(),
        "nvtx_ranges": nvtx,
        "stderr_phases": stderr,
        "stage_attribution": stages,
    }


def compare_nvtx_ranges(phases, oracle):
    ours = summarize_phases(phases, source="nvtx")
    theirs = summarize_gpuprof_nvtx(oracle)
    return compare_summaries(ours, theirs, tolerance_ms=0.001)


def compare_stderr_phases(phases, oracle):
    ours = summarize_phases(phases, source="stderr", chain="gpu")
    theirs = summarize_gpuprof_stderr_phases(oracle)
    return compare_summaries(ours, theirs, tolerance_ms=0.02)


def compare_summaries(ours, theirs, tolerance_ms):
    labels = sorted(set(ours) | set(theirs))
    mismatches = []
    for label in labels:
        a = ours.get(label, {"count": 0, "wall_ms": 0.0})
        b = theirs.get(label, {"count": 0, "wall_ms": 0.0})
        count_delta = a["count"] - b["count"]
        wall_delta = a["wall_ms"] - b["wall_ms"]
        if count_delta or abs(wall_delta) > tolerance_ms:
            mismatches.append(
                {
                    "stage_id": label,
                    "gpuscope_count": a["count"],
                    "gpuprof_count": b["count"],
                    "count_delta": count_delta,
                    "gpuscope_wall_ms": rounded(a["wall_ms"]),
                    "gpuprof_wall_ms": rounded(b["wall_ms"]),
                    "wall_delta_ms": rounded(wall_delta),
                }
            )
    return {
        "ok": not mismatches,
        "compared_labels": len(labels),
        "mismatches": mismatches,
    }


def compare_stage_attribution(report, oracle):
    runs = oracle.get("runs") or []
    if len(runs) > 1:
        return {
            "ok": True,
            "skipped": True,
            "reason": "legacy gpuprof stage rollups are medianized across repeats; raw NVTX ranges remain the comparable oracle",
            "shared_stages": 0,
            "compared_fields": 0,
            "mismatches": [],
            "intentional_deltas": [],
        }

    ours = {row["stage_id"]: row for row in report.get("stages", [])}
    theirs = oracle.get("stages") or oracle.get("phases") or {}
    shared = sorted(set(ours) & set(theirs))
    mismatches = []
    intentional_deltas = []
    compared_fields = 0
    for stage_id in shared:
        reason = SKIP_STAGE_REASONS.get(stage_id)
        if reason:
            intentional_deltas.append({"stage_id": stage_id, "reason": reason})
            continue
        for our_field, gpuprof_field in STAGE_FIELD_MAP.items():
            if gpuprof_field not in theirs[stage_id] and zeroish(ours[stage_id].get(our_field, 0.0)):
                continue
            compared_fields += 1
            a = float(ours[stage_id].get(our_field, 0.0))
            b = float(theirs[stage_id].get(gpuprof_field, 0.0))
            tolerance = tolerance_for(our_field)
            if abs(a - b) > tolerance:
                mismatches.append(
                    {
                        "stage_id": stage_id,
                        "gpuscope_field": our_field,
                        "gpuprof_field": gpuprof_field,
                        "gpuscope_value": rounded(a),
                        "gpuprof_value": rounded(b),
                        "delta": rounded(a - b),
                        "tolerance": tolerance,
                    }
                )
    return {
        "ok": not mismatches,
        "shared_stages": len(shared),
        "compared_fields": compared_fields,
        "mismatches": mismatches,
        "intentional_deltas": intentional_deltas,
    }


def summarize_phases(phases, source=None, chain=None):
    summary = defaultdict(lambda: {"count": 0, "wall_ms": 0.0})
    for phase in phases:
        if source is not None and phase.get("source") != source:
            continue
        if chain is not None and phase.get("chain") != chain:
            continue
        row = summary[phase["label"]]
        row["count"] += 1
        row["wall_ms"] += (phase["end"] - phase["start"]) / 1e6
    return summary


def summarize_gpuprof_nvtx(oracle):
    summary = defaultdict(lambda: {"count": 0, "wall_ms": 0.0})
    for row in oracle.get("nvtx_ranges", []):
        stage_id = row.get("stage")
        if not stage_id:
            continue
        out = summary[stage_id]
        out["count"] += 1
        out["wall_ms"] += float(row.get("wall_ms", 0.0))
    return summary


def summarize_gpuprof_stderr_phases(oracle):
    summary = defaultdict(lambda: {"count": 0, "wall_ms": 0.0})
    for row in oracle.get("phase_trace", []):
        if row.get("synthetic") or row.get("chain") != "gpu" or row.get("family") == "cuda":
            continue
        stage_id = row.get("stage")
        if not stage_id:
            continue
        out = summary[stage_id]
        out["count"] += 1
        out["wall_ms"] += float(row.get("wall_ms", 0.0))
    return summary


def intentional_deltas():
    return [
        {
            "stage_id": "finalize.terminal_fold",
            "fields": [
                "kernel_attributed_ms",
                "launches",
                "h2d_mb",
                "d2h_mb",
                "dtod_mb",
                "api_total_ms",
            ],
            "reason": "legacy gpuprof synthetic terminal fold carries wall only; gpuscope also attributes last-fold work to make the row actionable",
        }
    ]


def tolerance_for(field):
    if field.endswith("_copies") or field in {"launches", "syncs", "api_calls", "memset_count"}:
        return 0.5
    if field.endswith("_mb"):
        return 0.001
    return 0.01


def zeroish(value):
    return abs(float(value or 0.0)) <= 0.000001


def rounded(value):
    return round(float(value), 6)
