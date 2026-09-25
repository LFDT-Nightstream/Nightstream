"""Chrome/Perfetto trace-event export for gpuprof timelines."""

from util import api_bucket, copy_kind_name, short_kernel_name


def chrome_trace(phases, kernels, memcpys, memsets, syncs, api_calls, nvtx):
    """Return Chrome/Perfetto trace events for protocol phases and CUDA work."""

    def complete(name, cat, start_ns, end_ns, pid, tid, args=None):
        dur_us = max(0.0, (end_ns - start_ns) / 1e3)
        return {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": start_ns / 1e3,
            "dur": dur_us,
            "pid": pid,
            "tid": tid,
            "args": args or {},
        }

    events = [
        {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "protocol phases"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "CPU chain phases"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "CUDA chain phases"}},
        {"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "CUDA activity"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 10, "args": {"name": "kernels"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 20, "args": {"name": "memcpy"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 30, "args": {"name": "memset"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 40, "args": {"name": "sync"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 50, "args": {"name": "CUDA API"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 60, "args": {"name": "NVTX ranges"}},
    ]

    for phase in phases:
        label = phase["label"].split(":", 1)[-1]
        chain = phase.get("chain", "gpu")
        events.append(complete(
            label,
            f"phase.{chain}",
            phase["start"],
            phase["end"],
            1,
            1 if chain == "cpu" else 2,
            {
                "family": phase.get("family"),
                "source": phase.get("source", "stamp"),
                "synthetic": bool(phase.get("synthetic")),
                "wall_ms": phase["wall_ms"],
            },
        ))

    for k in kernels:
        start, end, name = k["start"], k["end"], k["name"]
        events.append(complete(
            short_kernel_name(name),
            "cuda.kernel",
            start,
            end,
            2,
            10,
            {
                "full_name": name,
                "elapsed_ms": (end - start) / 1e6,
                "stream_id": k.get("stream_id"),
                "correlation_id": k.get("correlation_id"),
                "registers_per_thread": k.get("registers_per_thread"),
                "grid": k.get("grid"),
                "block": k.get("block"),
            },
        ))
    for c in memcpys:
        start, end, byte_count, kind = c["start"], c["end"], c["bytes"], c["copy_kind"]
        direction = copy_kind_name(kind)
        events.append(complete(
            direction,
            "cuda.memcpy",
            start,
            end,
            2,
            20,
            {
                "bytes": byte_count,
                "mb": byte_count / 1e6,
                "kind": kind,
                "stream_id": c.get("stream_id"),
                "correlation_id": c.get("correlation_id"),
            },
        ))
    for m in memsets:
        start, end, byte_count = m["start"], m["end"], m["bytes"]
        events.append(complete(
            "memset",
            "cuda.memset",
            start,
            end,
            2,
            30,
            {
                "bytes": byte_count,
                "mb": byte_count / 1e6,
                "stream_id": m.get("stream_id"),
                "correlation_id": m.get("correlation_id"),
            },
        ))
    for s in syncs:
        start, end = s["start"], s["end"]
        events.append(complete(
            s.get("sync_label") or "sync",
            "cuda.sync",
            start,
            end,
            2,
            40,
            {
                "elapsed_ms": (end - start) / 1e6,
                "stream_id": s.get("stream_id"),
                "correlation_id": s.get("correlation_id"),
                "sync_type": s.get("sync_type"),
            },
        ))
    for api in api_calls:
        start, end, name = api["start"], api["end"], api["name"]
        events.append(complete(
            name,
            "cuda.api",
            start,
            end,
            2,
            50,
            {
                "elapsed_ms": (end - start) / 1e6,
                "bucket": api_bucket(name),
                "correlation_id": api.get("correlation_id"),
                "thread_id": api.get("thread_id"),
                "return_value": api.get("return_value"),
            },
        ))
    for r in nvtx:
        events.append(complete(
            r["text"], "nvtx.range", r["start"], r["end"], 2, 60,
            {"range_id": r.get("range_id"), "event_type": r.get("event_type"), "thread_id": r.get("thread_id")},
        ))

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "format": "Chrome trace event JSON",
            "open_with": "chrome://tracing or https://ui.perfetto.dev",
        },
    }
