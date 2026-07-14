"""Command-line entry point for gpuscope."""

import argparse
import sys

try:
    from .analyze import EPSILON_MS, analyze_bundle
except ImportError:  # Direct `python scripts/gpuscope/cli.py ...` execution.
    from analyze import EPSILON_MS, analyze_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpuscope")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="derive analysis artifacts from a raw bundle")
    analyze.add_argument("bundle_dir", help="gpuscope bundle or old gpuprof bundle/run directory")
    analyze.add_argument(
        "--validate-gpuprof",
        action="store_true",
        help="also compare raw facts against a legacy gpuprof.json oracle",
    )
    analyze.add_argument("--gpuprof-json", help="legacy gpuprof.json path; defaults to <bundle>/gpuprof.json")

    args = parser.parse_args(argv)
    if args.command == "analyze":
        output_path, report = analyze_bundle(
            args.bundle_dir,
            validate_gpuprof=args.validate_gpuprof,
            gpuprof_json=args.gpuprof_json,
        )
        stages = report["counts"]["stages"]
        reconciliation = report["reconciliation"]
        max_error = reconciliation["max_abs_error_ms"]
        clamp = reconciliation["idle_overlap_clamp_ms"]
        unknown = len(report["unknown_nvtx"])
        print(f"wrote {output_path}")
        print(
            f"stages={stages} max_reconciliation_error_ms={max_error:.6f} "
            f"idle_overlap_clamp_ms={clamp:.6f} unknown_labels={unknown}"
        )
        top_lever = (report.get("levers") or {}).get("top")
        if top_lever:
            print(
                "top_lever="
                f"{top_lever['stage_id']} "
                f"recoverable_ms={top_lever['recoverable_ms']:.3f} "
                f"fix_class={top_lever['fix_class']}"
            )
        validation = report.get("gpuprof_validation")
        if validation:
            print(
                "gpuprof_validation="
                f"{'ok' if validation['ok'] else 'FAIL'} "
                f"nvtx_mismatches={validation['nvtx_mismatches']} "
                f"stderr_mismatches={validation['stderr_mismatches']} "
                f"stage_mismatches={validation['stage_mismatches']}"
            )
            if not validation["ok"]:
                return 3
        return 0 if reconciliation["ok"] else 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
