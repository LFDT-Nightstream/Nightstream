#!/usr/bin/env python3
"""Public command-line validator for the Nightstream protocol contract."""

from __future__ import annotations

import csv
import json
import sys

from contract_checks import (
    ContractError,
    check_auxiliary_schemas,
    check_bindings,
    check_coverage,
    check_cross_references,
    check_documents,
    check_evidence_schema,
    check_global_id_uniqueness,
    check_package_manifest,
    check_profile_consistency,
    check_security_census,
    check_source_derivation,
    check_sources,
    load_config,
    require,
)
from contract_model import (
    ModelError,
    check_generated,
    load_model,
    protocol_open_decisions,
    query as query_model,
    summary as model_summary,
    unresolved_decision_ids,
)
from contract_migration import validate_lossless_import
from contract_protocol import check_protocol_profile_alignment, schedule_step_ids


def main() -> int:
    arguments = list(sys.argv[1:])
    query_id = None
    if "--query" in arguments:
        index = arguments.index("--query")
        if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
            print("--query needs one ID", file=sys.stderr)
            return 2
        query_id = arguments[index + 1]
        del arguments[index : index + 2]
    allowed = {
        "--release",
        "--implementation-ready",
        "--bind",
        "--package-only",
        "--repository",
        "--verify-import",
    }
    if any(argument not in allowed for argument in arguments) or len(set(arguments)) != len(arguments):
        print(
            "usage: check_contract.py [--package-only | --repository] "
            "[--release | --implementation-ready] [--bind] "
            "[--verify-import] [--query ID]",
            file=sys.stderr,
        )
        return 2
    if "--package-only" in arguments and "--repository" in arguments:
        print("choose one of --package-only or --repository", file=sys.stderr)
        return 2
    if "--release" in arguments and "--implementation-ready" in arguments:
        print("choose one of --release or --implementation-ready", file=sys.stderr)
        return 2
    release_requested = "--release" in arguments
    implementation_requested = "--implementation-ready" in arguments
    bind_requested = "--bind" in arguments
    verify_import_requested = "--verify-import" in arguments
    repository_mode = "--package-only" not in arguments

    try:
        model = load_model(repository_mode=repository_mode)
        if verify_import_requested:
            validate_lossless_import(model)
        check_generated(model)
        config = load_config()
        require(config["schema_version"] == 4, "unsupported generated profile schema")
        require(config["profile_id"], "candidate profile has no ID")
        require(
            config["profile_semantic_sha256"] == model.profile_hash,
            "generated profile identity differs from semantic profile inputs",
        )
        require(
            config["conformance"]["release_blocked"] == (not model.release["eligible"]),
            "generated profile release state differs from the claim graph",
        )
        require(
            config["conformance"]["implementation_ready"]
            == model.release["implementation_ready"],
            "generated profile implementation state differs from the claim graph",
        )
        require(
            config["current_gate"] == (model.release["next_gate"] or "complete"),
            "generated profile current gate differs from the claim graph",
        )
        requirement_decisions = {
            decision_id
            for requirement in model.requirements.values()
            for decision_id in requirement["blocker_ids"]
        }
        require(
            config["conformance"]["open_blockers"]
            == unresolved_decision_ids(model.decisions, requirement_decisions),
            "generated profile open blockers are not derived from decision state",
        )
        check_documents(config)
        package_file_count = check_package_manifest()
        source_ids = check_sources(config)
        paper_file_count = check_source_derivation(config)
        decision_ids = set(model.decisions)
        base_ids, paper_ids, errata_ids = check_cross_references(source_ids, decision_ids)
        check_evidence_schema(config)
        check_auxiliary_schemas(config)
        check_profile_consistency(config)
        check_security_census(config)
        check_protocol_profile_alignment(model, config)
        rule_ids, _coverage_tiers = check_coverage(
            config, source_ids, decision_ids, repository_mode
        )
        declaration_count, unbound_rules, external_anchor_count = check_bindings(
            model, repository_mode
        )
        obligation_ids = set(model.claims)
        gate_ids = {item["id"] for item in model.gate_status}
        issue_ids = set(model.issues)
        obligation_count = len(obligation_ids)
        check_global_id_uniqueness(
            {
                "source": source_ids,
                "base-paper": base_ids,
                "paper": paper_ids,
                "errata": errata_ids,
                "decision": decision_ids,
                "rule": rule_ids,
                "obligation": obligation_ids,
                "issue": issue_ids,
                "gate": gate_ids,
                "assurance-rollup": set(model.rollups),
                "artifact": set(model.artifacts),
                "protocol-state": {
                    item["id"] for item in model.protocol["states"]
                },
                "protocol-event": {
                    item["id"] for item in model.protocol["events"]
                },
                "challenge": {
                    item["id"] for item in model.protocol["challenges"]
                },
                "repetition": {
                    item["id"] for item in model.protocol["repetitions"]
                },
                "rejection": {
                    item["id"] for item in model.protocol["rejections"]
                },
                "transcript-step": schedule_step_ids(model.protocol["schedule"]),
            }
        )
    except (
        ContractError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        SyntaxError,
        csv.Error,
        json.JSONDecodeError,
        ModelError,
    ) as error:
        print(f"contract check: FAIL: {error}", file=sys.stderr)
        return 1

    if implementation_requested:
        blockers = []
        if not model.release["implementation_ready"]:
            required_claims = set()
            for gate in model.assurance_graph["gates"]:
                required_claims.update(gate["requires"])
                if gate["id"] == model.release["implementation_ready_gate"]:
                    break
            blockers.extend(
                sorted(
                    claim_id
                    for claim_id in required_claims
                    if model.claim_status[claim_id]["closure_state"]
                    not in {"closed", "not-applicable"}
                )
            )
        protocol_blockers = protocol_open_decisions(model)
        if protocol_blockers:
            blockers.extend(protocol_blockers)
        if blockers:
            print(
                "contract check: IMPLEMENTATION BLOCKED: " + ", ".join(blockers),
                file=sys.stderr,
            )
            return 2
        print("contract implementation-ready check: PASS")
        return 0

    if release_requested:
        blockers = []
        if not repository_mode:
            blockers.append("package-only mode cannot validate repository evidence")
        if unbound_rules:
            blockers.append(
                f"{len(unbound_rules)} unbound rules: {', '.join(sorted(unbound_rules))}"
            )
        if not model.release["eligible"]:
            nonclosed = sorted(
                claim_id
                for claim_id, state in model.claim_status.items()
                if state["closure_state"] not in {"closed", "not-applicable"}
            )
            blockers.append("non-closed claims: " + ", ".join(nonclosed))
        protocol_blockers = protocol_open_decisions(model)
        if protocol_blockers:
            blockers.append("protocol profile blockers: " + ", ".join(protocol_blockers))
        if blockers:
            for blocker in blockers:
                print(f"contract check: RELEASE BLOCKED: {blocker}", file=sys.stderr)
            return 2
        print("contract release check: PASS")
        return 0

    label = "contract repository check" if repository_mode else "contract package check"
    source_summary = model_summary(model)
    print(f"{label}: PASS")
    print(f"  package files: {package_file_count}")
    print(f"  pinned sources: {len(source_ids)} ({paper_file_count} base/reviewed paper files)")
    print(f"  errata rows: {len(errata_ids)}")
    print(f"  decision rows: {len(decision_ids)}")
    print(
        f"  normative requirements: {len(rule_ids)} "
        f"({source_summary['requirement_edges']} semantic dependency edges)"
    )
    flagged = sum(bool(item["review_flags"]) for item in model.requirements.values())
    print(f"  requirements with review flags: {flagged}")
    print(
        "  protocol events/challenges/repetitions: "
        f"{len(model.protocol['events'])}/{len(model.protocol['challenges'])}/"
        f"{len(model.protocol['repetitions'])}"
    )
    closed = sum(item["closure_state"] == "closed" for item in model.claim_status.values())
    partial = sum(item["display_status"] == "partial" for item in model.claim_status.values())
    print(
        f"  assurance claims: {len(model.claims)} "
        f"({closed} closed; {partial} partial; {source_summary['claim_edges']} dependency edges)"
    )
    print(
        f"  assurance rollups/artifacts: {source_summary['rollups']}/"
        f"{source_summary['artifacts']}"
    )
    print(f"  open tracked issues: {sum(item['state'] != 'resolved' for item in model.issues.values())}")
    if repository_mode:
        print(f"  checked declarations: {declaration_count}")
    else:
        print(
            f"  repository declaration anchors: {external_anchor_count} NOT CHECKED"
        )
        print("  repository evidence paths: NOT CHECKED")
    print(f"  unbound rules: {len(unbound_rules)} of {len(rule_ids)}")
    print(f"  obligations: {obligation_count}")
    if verify_import_requested:
        print("  sealed legacy import record: VERIFIED")
    release_label = "ELIGIBLE" if model.release["eligible"] else f"BLOCKED at {model.release['next_gate']}"
    print(f"  release: {release_label}")
    if bind_requested:
        for rule in sorted(unbound_rules):
            print(f"  unbound: {rule}")
    if query_id is not None:
        result = query_model(model, query_id)
        if result is None:
            print(f"contract query: unknown ID: {query_id}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
