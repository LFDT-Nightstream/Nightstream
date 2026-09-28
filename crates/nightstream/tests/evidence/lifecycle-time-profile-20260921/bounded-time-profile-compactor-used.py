#!/usr/bin/env python3
"""Keep every function total without duplicating frame or backtrace records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def compact(label, target_pid):
    source = ROOT / f"{label}-summary.json"
    summary = json.loads(source.read_text())
    assert summary["target_pid"] == target_pid
    source_functions = summary["functions_by_self"]
    image_keys = sorted({function["image_key"] for function in source_functions})
    image_indexes = {key: index for index, key in enumerate(image_keys)}
    images = {}
    for binary_id, descriptor in summary["binaries"].items():
        key = json.dumps([descriptor["UUID"], descriptor.get("arch"), descriptor.get("load-addr")]) \
            if descriptor.get("UUID") else json.dumps(["binary_id", binary_id])
        if key in image_indexes:
            images.setdefault(image_indexes[key], {"name": descriptor.get("name"),
                "uuid": descriptor.get("UUID"), "arch": descriptor.get("arch"),
                "load_address": descriptor.get("load-addr"), "path": descriptor.get("path")})
    functions = []
    for function in source_functions:
        index = image_indexes[function["image_key"]]
        images.setdefault(index, {"name": None, "unknown_image": True})
        record = {"image": index}
        if function["function"] is None:
            record["unknown_pc"] = function["unknown_address"]
        else:
            record["symbol"] = function["function"]
        record.update({key: function[key] for key in (
            "self_sample_count", "self_weight_ns", "inclusive_sample_count", "inclusive_weight_ns")})
        functions.append(record)
    nonempty_count = summary["backtrace_sample_count"] - summary["empty_backtrace_sample_count"]
    nonempty_weight = summary["backtrace_weight_ns"] - summary["empty_backtrace_weight_ns"]
    symbols = summary["symbolication"]
    named_count = nonempty_count - symbols["unknown_top_sample_count"]
    named_weight = nonempty_weight - symbols["unknown_top_weight_ns"]
    assert len(functions) == len(source_functions)
    assert sum(function["self_sample_count"] for function in functions) == nonempty_count
    assert sum(function["self_weight_ns"] for function in functions) == nonempty_weight
    return {
        "target_pid": target_pid, "source_bytes": summary["source_bytes"],
        "source_xml": summary["source"], "full_summary": str(source),
        "attribution_receipt": str(ROOT / f"{label}-attribution.json"),
        "completion_receipts": [str(ROOT / f"{label}-{stage}-completion.json") for stage in ("parse", "aggregate")],
        "coverage": {
            "all_target_samples": {"sample_count": summary["target_sample_count"], "weight_ns": summary["target_weight_ns"]},
            "backtrace_samples": {"sample_count": summary["backtrace_sample_count"], "weight_ns": summary["backtrace_weight_ns"]},
            "sentinel_samples": {"sample_count": summary["sentinel_sample_count"], "weight_ns": summary["sentinel_weight_ns"]},
            "empty_backtrace_samples": {"sample_count": summary["empty_backtrace_sample_count"], "weight_ns": summary["empty_backtrace_weight_ns"]},
            "other_missing_stack_samples": {"sample_count": summary["other_missing_stack_sample_count"], "weight_ns": summary["other_missing_stack_weight_ns"]},
            "unknown_top_frame_samples": {"sample_count": symbols["unknown_top_sample_count"], "weight_ns": symbols["unknown_top_weight_ns"]},
            "unannotated_top_frame_samples": {"sample_count": symbols["unannotated_top_sample_count"], "weight_ns": symbols["unannotated_top_weight_ns"]},
        },
        "denominators": {
            "nonempty_backtrace": {"sample_count": nonempty_count, "weight_ns": nonempty_weight,
                "definition": "All target samples with at least one frame; excludes sentinel, empty, and other missing stacks."},
            "named_top_frame": {"sample_count": named_count, "weight_ns": named_weight,
                "definition": "Nonempty backtrace samples whose top frame has an identified symbol; excludes unknown top frames."},
        },
        "symbol_attribution": {
            "exact_original_uuid": symbols["expected_uuid"], "exact_original_binary": symbols["exact_original_binary"],
            "recorded_native_binary_ids": symbols["native_binary_ids"],
            "native_address_count": symbols["native_address_count"],
            "native_frames_resolved_by_atos": symbols["native_frames_resolved_by_atos"],
            "method": "Recorded image UUID and load address, verified against the saved original binary. No rebuild or function-matching fallback."},
        "images": images, "function_count": len(functions),
        "zero_self_with_inclusive_count": sum(function["self_sample_count"] == 0 and function["inclusive_sample_count"] > 0 for function in functions),
        "functions": functions,
    }


def main():
    result = {"schema": 1,
        "time_scope": "All weights are sampled CPU nanoseconds summed across threads. They are not wall time or GPU execution time.",
        "inclusive_scope": "Each resolved loaded-image/function receives a backtrace's weight once; recursion is deduplicated. Unknown functions remain separated by image and recorded PC. Inclusive function weights overlap.",
        "denominator_scope": "State the selected denominator for any percentage. Sentinel weight is not part of either function denominator; unknown top-frame weight is not part of the named-top-frame denominator.",
        "script_versions": str(ROOT / "bounded-time-profile-script-versions.json"),
        "engines": {"optimized": compact("bounded-cpu-time-profile", 80683),
                    "metal": compact("bounded-metal-time-profile", 83259)}}
    output = ROOT / "function-totals.json"
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
        "engines": {engine: {key: value[key] for key in ("target_pid", "coverage", "denominators", "symbol_attribution", "function_count", "zero_self_with_inclusive_count")}
                    for engine, value in result["engines"].items()}}, indent=2))


if __name__ == "__main__":
    main()
