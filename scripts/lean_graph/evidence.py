#!/usr/bin/env python3
"""Inspect Lean dependencies and validation evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "lean_graph"

from .checkpoint import checkpoint
from .policy import Authority, gate_scope, load_policy
from .records import (declaration_records, explain, gate_results, markdown,
                      read_runs, report)
from . import queries
from .runner import run_gate
from .reviews import create_request, record_review, require_review
from .snapshot import (EvidenceError, capture, digest, inspect, read_json,
                       verify, write_json)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(prog="lean-graph", description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--store", type=Path, required=True,
                        help="evidence directory outside the candidate source")
    parser.add_argument("--policy", type=Path, help="draft obligation map (diagnostics only)")
    parser.add_argument("--authority", type=Path, help="separately provisioned approved checker")
    parser.add_argument("--inputs", type=Path, help="JSON object mapping input names to file/directory paths")
    parser.add_argument("--snapshot", help="select an existing frozen snapshot instead of current source")
    parser.add_argument("--json", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run").add_argument("gate")
    commands.add_parser("status")
    commands.add_parser("stale")
    commands.add_parser("checkpoint").add_argument("obligation")
    commands.add_parser("explain").add_argument("obligation")
    review = commands.add_parser("review-request")
    review.add_argument("obligation")
    review.add_argument("--proposal", type=Path, required=True)
    decision = commands.add_parser("record-review")
    decision.add_argument("request")
    decision.add_argument("response", type=Path)
    for name in ("requires", "used-by", "path"):
        query = commands.add_parser(name)
        query.add_argument("declaration")
        if name == "path":
            query.add_argument("parent")
    active = commands.add_parser("active")
    active.add_argument("obligation")
    active.add_argument("--evidence", required=True, help="the check intended to close the active criterion")
    args = parser.parse_args(argv)
    try:
        source, store = args.source.resolve(), args.store.resolve()
        if store == source or source in store.parents:
            raise EvidenceError("the evidence store must be outside candidate source")
        authority = Authority(args.authority) if args.authority else None
        if authority and (authority.directory == source or source in authority.directory.parents):
            raise EvidenceError("the approved checker must be outside candidate source")
        if authority and args.policy:
            raise EvidenceError("candidate policy cannot replace the approved checker policy")
        policy = load_policy(args.policy, authority)
        if args.command == "record-review":
            print(json.dumps(record_review(args.request, args.response, store, authority), indent=2))
            return 0
        if hasattr(args, "obligation") and args.obligation not in policy["obligations"]:
            raise EvidenceError("unknown owner criterion")
        if args.command == "review-request" and not policy["obligations"][args.obligation].get("target"):
            raise EvidenceError("register the exact Lean target before requesting decomposition review")
        query_command = args.command in ("requires", "used-by", "path")
        if args.command == "run":
            selected = [args.gate]
        elif args.command in ("checkpoint", "explain", "review-request"):
            selected = policy["obligations"][args.obligation]["gates"]
            if args.command == "checkpoint" and not selected:
                raise EvidenceError("no closing gate is registered: " + policy["obligations"][args.obligation]["gap"])
        elif query_command:
            selected = [policy["graph_gate"]] if policy.get("graph_gate") else []
            if not selected:
                raise EvidenceError("no declaration graph gate is registered")
        else:
            selected = list(policy["gates"])
        scope = gate_scope(policy, selected)
        inputs = read_json(args.inputs) if args.inputs else {}
        if not isinstance(inputs, dict) or not set(inputs) <= policy["inputs"].keys():
            raise EvidenceError("unknown input name or invalid inputs object")
        if "library_seed" in policy["inputs"]:
            # Third-party source, Git metadata, and compiled modules are one explicit
            # toolchain input. Accepted runs use the operator's protected copy.
            expected = (authority.directory / "libraries" if authority else
                        source / "formal/nightstream-fprime/.lake/packages")
            selected = Path(inputs.get("library_seed", expected)).resolve()
            if authority and selected != expected.resolve():
                raise EvidenceError("accepted Lean runs must use the checker-owned library seed")
            if "library_seed" in scope["inputs"] or "libraries" in scope["sources"]:
                inputs["library_seed"] = str(selected)
        if args.command == "active":
            state = {"obligation": args.obligation, "evidence": args.evidence}
            write_json(store / "active.json", state)
            print(json.dumps(state) if args.json else f"Active criterion: {args.obligation}. {args.evidence}")
            return 0
        state = read_json(store / "active.json") if (store / "active.json").exists() else None
        if args.command == "run" and state is None:
            raise EvidenceError("select an active owner criterion before running a gate")
        if args.command == "checkpoint":
            state = {"obligation": args.obligation, "evidence": "checkpoint " + args.obligation}
            # Criterion selection belongs to this invocation. Other tasks can
            # use this store without overwriting one shared active pointer.
        snapshot_started = time.monotonic()
        if args.snapshot:
            if not args.snapshot.isalnum():
                raise EvidenceError("invalid snapshot identifier")
            snapshot = store / "snapshots" / args.snapshot
            manifest = read_json(snapshot / "manifest.json")
            if digest(manifest) != args.snapshot:
                raise EvidenceError("snapshot manifest identity does not match")
            verify(snapshot, manifest)
        elif args.command in ("run", "checkpoint", "review-request"):
            _, manifest, snapshot = capture(source, policy, inputs, store, scope)
        else:
            manifest, _ = inspect(source, policy, inputs, scope)
        snapshot_seconds = time.monotonic() - snapshot_started
        if args.command == "review-request":
            print(json.dumps(create_request(args.obligation, read_json(args.proposal),
                                           policy, manifest, store), indent=2))
            return 0
        invocation = [sys.executable, str(Path(__file__).resolve()), "--source", str(source), "--store", str(store)]
        for option in ("policy", "authority", "inputs", "snapshot"):
            value = getattr(args, option)
            if value:
                invocation += ["--" + option, str(value.resolve() if isinstance(value, Path) else value)]
        if args.command == "run":
            if state["obligation"] not in policy["obligations"]:
                raise EvidenceError("active criterion is not registered in this policy")
            require_review(state["obligation"], policy, manifest, store, authority)
            runs, _ = read_runs(store, authority)
            current, _ = gate_results(policy, manifest, runs)
            missing = [gate for gate in policy["gates"][args.gate].get("requires", [])
                       if not current[gate]["pass" if authority else "completed"]]
            if missing:
                raise EvidenceError("required gates remain open: " + ", ".join(missing))
            result = run_gate(args.gate, policy, manifest, snapshot, store, authority)
            result["invocation_timing"] = {"snapshot_seconds": snapshot_seconds,
                                            "total_seconds": time.monotonic() - started}
            print(json.dumps(result, indent=2) if args.json else
                  f"{args.gate}: execution {'passed' if result['outcome'] == 'pass' else result['outcome']}; "
                  f"freshness current; checker {'approved' if authority else 'diagnostic'}; "
                  f"snapshot {result['snapshot']}.")
            return 0 if result["outcome"] == "pass" else 1
        checked = None
        if args.command == "checkpoint":
            checked = checkpoint(args.obligation, policy, manifest, snapshot, store, authority)
        result = report(policy, manifest, store, authority, state, invocation)
        result["invocation_timing"] = {"snapshot_seconds": snapshot_seconds,
                                        "total_seconds": time.monotonic() - started}
        if query_command:
            runs, _ = read_runs(store, authority)
            exports = declaration_records(policy, manifest, runs)
            if not exports:
                import shlex
                raise EvidenceError("no current complete declaration export; run: " +
                                    shlex.join([*invocation, "run", policy["graph_gate"]]))
            answer = queries.query(exports, args.command, args.declaration, getattr(args, "parent", None),
                                   policy, result, snapshot / "source" if args.snapshot else source)
            print(json.dumps(answer, indent=2) if args.json else queries.markdown(answer), end="\n")
        elif args.command in ("checkpoint", "explain"):
            answer = {"checkpoint": checked, "status": result} if checked else result
            print(json.dumps(answer, indent=2) if args.json else explain(result, args.obligation), end="\n")
            if checked and checked["execution"] != "passed":
                return 1
        elif args.command == "stale":
            print(json.dumps({"snapshot": result["snapshot"], "stale": result["stale"],
                              "rejected": result["rejected"]}, indent=2))
        else:
            print(json.dumps(result, indent=2) if args.json else markdown(result), end="\n")
        return 0
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"lean-graph error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
