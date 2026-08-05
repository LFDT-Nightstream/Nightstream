#!/usr/bin/env python3
"""Validate the authored Nightstream protocol state machine."""

from __future__ import annotations

import re
from typing import Any

from contract_model import _string_list, require


def schedule_step_ids(schedule: dict[str, Any]) -> set[str]:
    """Return all static transcript-schedule step IDs."""
    result: set[str] = set()

    def visit(steps: list[dict[str, Any]]) -> None:
        for step in steps:
            step_id = step.get("id")
            if isinstance(step_id, str):
                result.add(step_id)
            if step.get("kind") == "repeat" and isinstance(step.get("body"), list):
                visit(step["body"])

    steps = schedule.get("steps")
    if isinstance(steps, list):
        visit(steps)
    return result


def check_protocol_profile_alignment(model: Any, config: dict[str, Any]) -> None:
    """Bind protocol loops, frames, and lifecycle to the selected profile."""
    challenge_ids = {item["id"] for item in model.protocol["challenges"]}
    require(
        challenge_ids
        == {
            "CH-PICCS-ALPHA",
            "CH-PICCS-GAMMA",
            "CH-SUMCHECK-ROUND",
            "CH-PIRLC-CANDIDATE",
        },
        "protocol challenge family differs from PaddedRowIdentity",
    )
    repetitions = {item["id"]: item for item in model.protocol["repetitions"]}
    require(
        set(repetitions)
        == {
            "REP-PICCS-ROUNDS",
            "REP-RHO-SOURCES",
            "REP-RHO-COEFFICIENTS",
            "REP-RHO-ATTEMPTS",
            "REP-FOLD-STEPS",
        },
        "protocol repetition family differs from PaddedRowIdentity",
    )
    candidate = config["nightstream_candidate"]
    sampler = config["sampler_profile"]
    accounting = config["security_accounting"]
    expected_bounds = {
        "REP-PICCS-ROUNDS": (candidate["row_variables"], candidate["row_variables"]),
        "REP-RHO-SOURCES": (candidate["source_claims"], candidate["source_claims"]),
        "REP-RHO-COEFFICIENTS": (sampler["ring_degree"], sampler["ring_degree"]),
        "REP-RHO-ATTEMPTS": (1, sampler["maximum_attempts_per_coefficient"]),
        "REP-FOLD-STEPS": (1, accounting["maximum_folds"]),
    }
    for repetition_id, (minimum, maximum) in expected_bounds.items():
        row = repetitions[repetition_id]
        require(
            (row["minimum"], row["maximum"]) == (minimum, maximum),
            f"protocol repetition differs from profile: {repetition_id}",
        )

    event_order = [item["id"] for item in model.protocol["events"]]
    require(
        event_order
        == [
            "EV-DECODE",
            "EV-BIND-PROFILE",
            "EV-PICCS-ABSORB",
            "EV-PICCS-COINS",
            "EV-PICCS-SUMCHECK",
            "EV-PICCS-OUTPUT",
            "EV-PICCS-CHECK",
            "EV-PIRLC-START",
            "EV-PIRLC-SAMPLE",
            "EV-PIRLC-FINALIZE",
            "EV-PIDEC",
            "EV-FOLD-FINALIZE",
        ],
        "protocol event order differs from the selected v1 schedule",
    )

    schedule = model.protocol["schedule"]
    lifecycle = config["lifecycle_profile"]
    require(
        set(lifecycle)
        == {
            "id",
            "sequence_input",
            "minimum_folds",
            "maximum_folds",
            "index_order",
            "fold_count_consistency",
            "verifier_key_consistency",
            "transcript_initialization",
            "state_link",
            "initial_running_state",
            "terminal_running_state",
            "fold_transcript_digest_base_fields",
            "fold_transcript_digest_role",
            "fold_transcript_digest_next_fold_input",
        },
        "unexpected lifecycle profile fields",
    )
    require(
        schedule["lifecycle_profile_id"] == lifecycle["id"]
        and lifecycle["id"] == "nightstream-bounded-fold-sequence-v1",
        "transcript schedule lifecycle profile differs",
    )
    require(
        (lifecycle["minimum_folds"], lifecycle["maximum_folds"])
        == (1, accounting["maximum_folds"]),
        "lifecycle fold bounds differ",
    )
    require(
        lifecycle["sequence_input"]
        == "ordered-array-of-one-fold-statement-proof-container-pairs"
        and lifecycle["index_order"] == "0-through-fold_count-minus-1"
        and lifecycle["fold_count_consistency"] == "same-fold_count-in-every-statement"
        and lifecycle["verifier_key_consistency"]
        == "one-selected-verifier-key-for-all-folds",
        "lifecycle sequence rule differs",
    )
    require(
        lifecycle["transcript_initialization"] == "fresh-zero-state-duplex-per-fold"
        and lifecycle["state_link"]
        == "ordered-children-of-fold-j-equal-ordered-running-claims-of-fold-j-plus-1-as-canonical-typed-CE-claims",
        "lifecycle transcript or state link differs",
    )
    require(
        lifecycle["initial_running_state"] == "first-fold-statement-input"
        and lifecycle["terminal_running_state"] == "last-fold-ordered-children",
        "lifecycle boundary state differs",
    )
    require(
        lifecycle["fold_transcript_digest_base_fields"] == 4
        and lifecycle["fold_transcript_digest_role"] == "verifier-derived-receipt-only"
        and lifecycle["fold_transcript_digest_next_fold_input"] is False,
        "fold transcript digest authority differs",
    )

    steps: dict[str, dict[str, Any]] = {}
    top_order = [step["id"] for step in schedule["steps"]]

    def collect(items: list[dict[str, Any]]) -> None:
        for step in items:
            steps[step["id"]] = step
            if step["kind"] == "repeat":
                collect(step["body"])

    collect(schedule["steps"])
    require(
        top_order
        == [
            "FR-SESSION",
            "FR-VERIFIER-KEY",
            "FR-STATEMENT",
            "FR-PICCS-INPUT",
            "SQ-PICCS-ALPHA",
            "SQ-PICCS-GAMMA",
            "LOOP-PICCS-ROUNDS",
            "FR-PICCS-OUTPUT",
            "LOOP-RHO-SOURCES",
            "FR-PIRLC-OUTPUT",
            "FR-PIDEC-OUTPUT",
            "FR-FOLD-FINALIZE",
            "SQ-FOLD-DIGEST",
        ],
        "one-fold transcript top-level order differs from v1",
    )
    require(
        [step["id"] for step in steps["LOOP-PICCS-ROUNDS"]["body"]]
        == ["FR-PICCS-ROUND", "SQ-PICCS-ROUND"],
        "PiCCS round transcript body differs from v1",
    )
    require(
        [step["id"] for step in steps["LOOP-RHO-SOURCES"]["body"]]
        == ["LOOP-RHO-COEFFICIENTS"]
        and [step["id"] for step in steps["LOOP-RHO-COEFFICIENTS"]["body"]]
        == ["LOOP-RHO-ATTEMPTS"]
        and [step["id"] for step in steps["LOOP-RHO-ATTEMPTS"]["body"]]
        == ["FR-PIRLC-CANDIDATE", "SQ-PIRLC-CANDIDATE"],
        "PiRLC sampler transcript nesting differs from v1",
    )

    sections = config["container_sections"]
    paper = config["paper_goldilocks"]
    algebra = config["algebra_encoding"]
    commitment = config["commitment_profile"]
    tags = config["transcript_tags"]
    extension_fields = algebra["extension_degree"]
    commitment_fields = commitment["commitment_ring_elements"] * paper["phi_degree"]
    output_fields = candidate["matrix_count"] * paper["phi_degree"] * extension_fields
    complete_claim_fields = commitment_fields + candidate["public_field_width"] + output_fields
    expected_frames = {
        "FR-SESSION": ("EV-PICCS-ABSORB", "session", 2),
        "FR-VERIFIER-KEY": ("EV-PICCS-ABSORB", "verifier_key", 4),
        "FR-STATEMENT": (
            "EV-PICCS-ABSORB",
            "statement",
            sections["statement_total_base_fields"],
        ),
        "FR-PICCS-INPUT": ("EV-PICCS-ABSORB", "piccs_input", 0),
        "FR-PICCS-ROUND": (
            "EV-PICCS-SUMCHECK",
            "sumcheck_round",
            1 + (candidate["sumcheck_degree"] + 1) * extension_fields,
        ),
        "FR-PICCS-OUTPUT": (
            "EV-PICCS-OUTPUT",
            "piccs_output",
            candidate["source_claims"] * output_fields,
        ),
        "FR-PIRLC-CANDIDATE": ("EV-PIRLC-SAMPLE", "pirlc_candidate", 3),
        "FR-PIRLC-OUTPUT": ("EV-PIRLC-FINALIZE", "pirlc_output", complete_claim_fields),
        "FR-PIDEC-OUTPUT": (
            "EV-PIDEC",
            "pidec_output",
            candidate["running_claims"] * complete_claim_fields,
        ),
        "FR-FOLD-FINALIZE": ("EV-FOLD-FINALIZE", "fold_finalize", 2),
    }
    for step_id, (event_id, tag, field_count) in expected_frames.items():
        step = steps[step_id]
        require(
            step["kind"] == "frame"
            and step["event_id"] == event_id
            and step["tag"] == tag
            and step["base_field_count"] == field_count,
            f"transcript frame differs from the selected profile: {step_id}",
        )

    expected_squeezes = {
        "SQ-PICCS-ALPHA": ("EV-PICCS-COINS", "piccs_alpha", candidate["row_variables"] * extension_fields, "CH-PICCS-ALPHA"),
        "SQ-PICCS-GAMMA": ("EV-PICCS-COINS", "piccs_gamma", extension_fields, "CH-PICCS-GAMMA"),
        "SQ-PICCS-ROUND": ("EV-PICCS-SUMCHECK", "sumcheck_round", extension_fields, "CH-SUMCHECK-ROUND"),
        "SQ-PIRLC-CANDIDATE": ("EV-PIRLC-SAMPLE", "pirlc_candidate", 1, "CH-PIRLC-CANDIDATE"),
        "SQ-FOLD-DIGEST": ("EV-FOLD-FINALIZE", "fold_finalize", 4, None),
    }
    for step_id, (event_id, tag, field_count, challenge_id) in expected_squeezes.items():
        step = steps[step_id]
        require(
            step["kind"] == "squeeze"
            and step["event_id"] == event_id
            and step["tag"] == tag
            and step["requested_base_fields"] == field_count
            and step["challenge_id"] == challenge_id,
            f"transcript squeeze differs from the selected profile: {step_id}",
        )
    used_tags = {
        step["tag"]
        for step in steps.values()
        if step["kind"] in {"frame", "squeeze"}
    }
    require(used_tags <= set(tags), "transcript schedule uses an undefined numeric tag")


def validate_protocol(model: Any) -> None:
    protocol = model.protocol
    machine = protocol["machine"]
    require(
        set(machine)
        == {"schema_version", "contract_id", "profile_id", "initial_state", "reject_state"},
        "unexpected protocol-machine fields",
    )
    require(machine.get("schema_version") == 2, "unsupported protocol-machine schema")
    require(machine["contract_id"] == model.bundle["contract_id"], "protocol-machine contract ID differs")
    require(machine["profile_id"] == model.bundle["profile_id"], "protocol-machine profile ID differs")
    states = {item["id"]: item for item in protocol["states"]}
    require(len(states) == len(protocol["states"]), "duplicate protocol states")
    require(machine["initial_state"] in states, "unknown initial protocol state")
    require(machine["reject_state"] in states, "unknown reject protocol state")
    require(states[machine["reject_state"]]["terminal"], "reject state is not terminal")
    events = {item["id"]: item for item in protocol["events"]}
    challenges = {item["id"]: item for item in protocol["challenges"]}
    repetitions = {item["id"]: item for item in protocol["repetitions"]}
    rejections = {item["id"]: item for item in protocol["rejections"]}
    require(len(events) == len(protocol["events"]), "duplicate protocol events")
    require(len(challenges) == len(protocol["challenges"]), "duplicate challenges")
    require(len(repetitions) == len(protocol["repetitions"]), "duplicate repetition IDs")
    require(len(rejections) == len(protocol["rejections"]), "duplicate rejection IDs")
    decision_ids = set(model.decisions)
    rule_ids = set(model.rules)

    for state_id, state in states.items():
        require(not (set(state) - {"id", "terminal", "description"}), f"unexpected state fields: {state_id}")
        require({"id", "terminal"} <= set(state), f"state omits a required field: {state_id}")
        require(isinstance(state["terminal"], bool), f"state terminal flag is not Boolean: {state_id}")
        if "description" in state:
            require(isinstance(state["description"], str) and state["description"], f"state has an empty description: {state_id}")

    used_rejections: set[str] = set()
    for event_id, event in events.items():
        required = {
            "id",
            "phase",
            "from_state",
            "to_state",
            "authority",
            "rule_ids",
            "challenge_ids",
            "blocked_by",
            "inputs",
            "outputs",
            "reject_conditions",
        }
        require(required <= set(event), f"event omits a required field: {event_id}")
        require(not (set(event) - (required | {"note"})), f"unexpected event fields: {event_id}")
        require(isinstance(event["phase"], str) and event["phase"], f"event has no phase: {event_id}")
        require(event["authority"] in model.policy["protocol_authorities"], f"event has invalid authority: {event_id}")
        require(event["from_state"] in states and event["to_state"] in states, f"event has unknown state: {event_id}")
        require(not states[event["from_state"]]["terminal"], f"event starts from a terminal state: {event_id}")
        event_rules = _string_list(event["rule_ids"], f"event rules for {event_id}")
        event_challenges = _string_list(event["challenge_ids"], f"event challenges for {event_id}")
        event_blockers = _string_list(event["blocked_by"], f"event blockers for {event_id}")
        _string_list(event["inputs"], f"event inputs for {event_id}")
        _string_list(event["outputs"], f"event outputs for {event_id}")
        require(not (set(event_rules) - rule_ids), f"event has unknown rules: {event_id}")
        require(not (set(event_challenges) - set(challenges)), f"event has unknown challenges: {event_id}")
        require(not (set(event_blockers) - decision_ids), f"event has unknown decision blockers: {event_id}")
        reject_conditions = _string_list(event["reject_conditions"], f"reject conditions for {event_id}")
        require(not (set(reject_conditions) - set(rejections)), f"event has an unknown rejection code: {event_id}")
        used_rejections.update(reject_conditions)
        if "note" in event:
            require(isinstance(event["note"], str) and event["note"], f"event has an empty note: {event_id}")

    for rejection_id, rejection in rejections.items():
        require(
            set(rejection) == {"id", "description", "rule_ids"},
            f"unexpected rejection fields: {rejection_id}",
        )
        require(re.fullmatch(r"REJECT-[A-Z0-9-]+", rejection_id) is not None, f"invalid rejection ID: {rejection_id}")
        require(isinstance(rejection["description"], str) and rejection["description"], f"rejection has no description: {rejection_id}")
        rejection_rules = _string_list(rejection["rule_ids"], f"rejection rules for {rejection_id}")
        require(rejection_rules, f"rejection has no normative rule: {rejection_id}")
        require(not (set(rejection_rules) - rule_ids), f"rejection has unknown rules: {rejection_id}")
    require(used_rejections == set(rejections), "rejection registry and event use differ")

    for challenge_id, challenge in challenges.items():
        require(
            set(challenge)
            == {
                "id",
                "role",
                "sample_space",
                "sampled_in_event",
                "transcript_tag",
                "decoder",
                "after_events",
                "blocked_by",
                "rule_ids",
            },
            f"unexpected challenge fields: {challenge_id}",
        )
        require(isinstance(challenge["role"], str) and challenge["role"], f"challenge has no role: {challenge_id}")
        require(isinstance(challenge["sample_space"], str) and challenge["sample_space"], f"challenge has no sample space: {challenge_id}")
        require(
            challenge["sampled_in_event"] in events,
            f"challenge has an unknown sampling event: {challenge_id}",
        )
        require(
            isinstance(challenge["transcript_tag"], str) and challenge["transcript_tag"],
            f"challenge has no transcript tag: {challenge_id}",
        )
        require(
            isinstance(challenge["decoder"], str) and challenge["decoder"],
            f"challenge has no decoder: {challenge_id}",
        )
        after_events = _string_list(challenge["after_events"], f"challenge predecessor events for {challenge_id}")
        challenge_rules = _string_list(challenge["rule_ids"], f"challenge rules for {challenge_id}")
        challenge_blockers = _string_list(challenge["blocked_by"], f"challenge blockers for {challenge_id}")
        require(not (set(after_events) - set(events)), f"challenge has unknown predecessor event: {challenge_id}")
        require(not (set(challenge_rules) - rule_ids), f"challenge has unknown rules: {challenge_id}")
        require(not (set(challenge_blockers) - decision_ids), f"challenge has unknown decision blockers: {challenge_id}")

    repetition_graph: dict[str, list[str]] = {}
    for repetition_id, repetition in repetitions.items():
        require(
            set(repetition)
            == {
                "id",
                "event_id",
                "kind",
                "parent_id",
                "unit",
                "index_variable",
                "minimum",
                "maximum",
                "index_order",
                "on_exhaustion",
            },
            f"unexpected repetition fields: {repetition_id}",
        )
        require(repetition["event_id"] in events, f"repetition has unknown event: {repetition_id}")
        require(
            repetition["kind"] in {"exact", "bounded-retry", "bounded-lifecycle"},
            f"repetition has invalid kind: {repetition_id}",
        )
        for field in ("unit", "index_variable", "index_order", "on_exhaustion"):
            require(
                isinstance(repetition[field], str) and repetition[field],
                f"repetition has no {field}: {repetition_id}",
            )
        minimum = repetition["minimum"]
        maximum = repetition["maximum"]
        require(
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and 0 <= minimum <= maximum,
            f"repetition has invalid numeric bounds: {repetition_id}",
        )
        if repetition["kind"] == "exact":
            require(minimum == maximum, f"exact repetition has unequal bounds: {repetition_id}")
        else:
            require(minimum < maximum, f"bounded repetition has no range: {repetition_id}")
        parent = repetition["parent_id"]
        require(
            parent is None or parent in repetitions,
            f"repetition has unknown parent: {repetition_id}",
        )
        require(parent != repetition_id, f"self-parent repetition: {repetition_id}")
        repetition_graph[repetition_id] = [parent] if parent is not None else []

    from contract_model import find_cycle

    repetition_cycle = find_cycle(repetition_graph)
    require(
        repetition_cycle is None,
        f"repetition dependency cycle: {' -> '.join(repetition_cycle or [])}",
    )

    schedule = protocol["schedule"]
    require(
        set(schedule)
        == {
            "schema_version",
            "scope",
            "lifecycle_repetition_id",
            "lifecycle_profile_id",
            "steps",
        },
        "unexpected transcript-schedule fields",
    )
    require(schedule["schema_version"] == 1, "unsupported transcript-schedule schema")
    require(schedule["scope"] == "one-fold", "transcript schedule does not have one-fold scope")
    require(
        schedule["lifecycle_profile_id"] == "nightstream-bounded-fold-sequence-v1",
        "transcript schedule has an unknown lifecycle profile",
    )
    lifecycle_id = schedule["lifecycle_repetition_id"]
    require(lifecycle_id in repetitions, "transcript schedule has an unknown lifecycle repetition")
    require(
        repetitions[lifecycle_id]["kind"] == "bounded-lifecycle"
        and repetitions[lifecycle_id]["parent_id"] is None,
        "transcript schedule lifecycle repetition is not a top-level lifecycle bound",
    )
    steps = schedule["steps"]
    require(isinstance(steps, list) and steps, "transcript schedule has no steps")
    event_positions = {event_id: index for index, event_id in enumerate(events)}
    static_step_ids: set[str] = set()
    scheduled_challenges: list[str] = []
    scheduled_repetitions: set[str] = set()
    scheduled_event_positions: list[int] = []
    final_squeezes: list[str] = []

    def validate_steps(items: list[dict[str, Any]], parent_id: str | None) -> None:
        require(isinstance(items, list) and items, "transcript repeat has an empty body")
        for step in items:
            require(isinstance(step, dict), "transcript schedule step is not an object")
            step_id = step.get("id")
            kind = step.get("kind")
            require(
                isinstance(step_id, str)
                and re.fullmatch(r"[A-Z][A-Z0-9-]+", step_id) is not None,
                "transcript schedule step has an invalid ID",
            )
            require(step_id not in static_step_ids, f"duplicate transcript schedule step: {step_id}")
            static_step_ids.add(step_id)
            require(kind in {"frame", "squeeze", "repeat"}, f"invalid transcript schedule step kind: {step_id}")
            if kind == "frame":
                require(
                    set(step)
                    == {"id", "kind", "event_id", "tag", "payload_layout", "base_field_count"},
                    f"unexpected transcript frame fields: {step_id}",
                )
                event_id = step["event_id"]
                require(event_id in events, f"transcript frame has an unknown event: {step_id}")
                require(isinstance(step["tag"], str) and step["tag"], f"transcript frame has no tag: {step_id}")
                payload = _string_list(step["payload_layout"], f"transcript frame payload for {step_id}")
                count = step["base_field_count"]
                require(
                    isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    f"transcript frame has an invalid field count: {step_id}",
                )
                require(bool(payload) == (count > 0), f"transcript frame payload/count mismatch: {step_id}")
            elif kind == "squeeze":
                require(
                    set(step)
                    == {"id", "kind", "event_id", "tag", "requested_base_fields", "challenge_id"},
                    f"unexpected transcript squeeze fields: {step_id}",
                )
                event_id = step["event_id"]
                require(event_id in events, f"transcript squeeze has an unknown event: {step_id}")
                require(isinstance(step["tag"], str) and step["tag"], f"transcript squeeze has no tag: {step_id}")
                count = step["requested_base_fields"]
                require(
                    isinstance(count, int) and not isinstance(count, bool) and count > 0,
                    f"transcript squeeze has an invalid field count: {step_id}",
                )
                challenge_id = step["challenge_id"]
                if challenge_id is None:
                    final_squeezes.append(step_id)
                else:
                    require(challenge_id in challenges, f"transcript squeeze has an unknown challenge: {step_id}")
                    challenge = challenges[challenge_id]
                    require(
                        challenge["sampled_in_event"] == event_id,
                        f"transcript squeeze uses the wrong event: {step_id}",
                    )
                    require(
                        challenge["transcript_tag"] == step["tag"],
                        f"transcript squeeze uses the wrong challenge tag: {step_id}",
                    )
                    scheduled_challenges.append(challenge_id)
            else:
                require(
                    set(step) == {"id", "kind", "repetition_id", "body"},
                    f"unexpected transcript repeat fields: {step_id}",
                )
                repetition_id = step["repetition_id"]
                require(repetition_id in repetitions, f"transcript repeat is unknown: {step_id}")
                require(repetition_id != lifecycle_id, "one-fold transcript embeds the lifecycle repetition")
                require(repetition_id not in scheduled_repetitions, f"duplicate transcript repeat: {repetition_id}")
                repetition = repetitions[repetition_id]
                require(
                    repetition["parent_id"] == parent_id,
                    f"transcript repeat nesting differs from its parent: {repetition_id}",
                )
                scheduled_repetitions.add(repetition_id)
                validate_steps(step["body"], repetition_id)
                continue

            if parent_id is not None:
                require(
                    events[event_id]["id"] == repetitions[parent_id]["event_id"],
                    f"transcript step event differs from its repetition: {step_id}",
                )
            scheduled_event_positions.append(event_positions[event_id])

    validate_steps(steps, None)
    require(
        scheduled_repetitions == set(repetitions) - {lifecycle_id},
        "transcript schedule does not cover every local repetition",
    )
    require(
        len(scheduled_challenges) == len(set(scheduled_challenges))
        and set(scheduled_challenges) == set(challenges),
        "transcript schedule does not use each challenge family exactly once",
    )
    require(final_squeezes == ["SQ-FOLD-DIGEST"], "transcript schedule has an invalid final squeeze")
    require(steps[-1].get("id") == "SQ-FOLD-DIGEST", "transcript final squeeze is not last")
    require(
        scheduled_event_positions == sorted(scheduled_event_positions),
        "transcript schedule event order differs from the verifier state machine",
    )

    challenge_users = {
        challenge_id: [event for event in events.values() if challenge_id in event["challenge_ids"]]
        for challenge_id in challenges
    }
    for challenge_id, users in challenge_users.items():
        require(users, f"challenge is never used by an event: {challenge_id}")
        require(
            any(event["id"] == challenges[challenge_id]["sampled_in_event"] for event in users),
            f"challenge sampling event does not use the challenge: {challenge_id}",
        )

    state_edges: dict[str, set[str]] = {state_id: set() for state_id in states}
    for event in events.values():
        state_edges[event["from_state"]].add(event["to_state"])

    def state_reaches(start: str, target: str) -> bool:
        pending = [start]
        seen = set()
        while pending:
            state_id = pending.pop()
            if state_id == target:
                return True
            if state_id in seen:
                continue
            seen.add(state_id)
            pending.extend(state_edges[state_id])
        return False

    for challenge_id, challenge in challenges.items():
        require(challenge["after_events"], f"challenge has no transcript predecessor: {challenge_id}")
        for predecessor_id in challenge["after_events"]:
            predecessor_state = events[predecessor_id]["to_state"]
            for user in challenge_users[challenge_id]:
                require(
                    state_reaches(predecessor_state, user["from_state"]),
                    f"challenge is used before its transcript predecessor: {challenge_id}",
                )

    reached = {machine["initial_state"]}
    changed = True
    while changed:
        changed = False
        for event in events.values():
            if event["from_state"] in reached and event["to_state"] not in reached:
                reached.add(event["to_state"])
                changed = True
    unreachable = set(states) - reached - {machine["reject_state"]}
    require(not unreachable, f"unreachable protocol states: {sorted(unreachable)}")
