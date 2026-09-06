"""Derive applicability and closure from authenticated runs and scoped reviews."""

from __future__ import annotations

from pathlib import Path
from graphlib import TopologicalSorter

from .policy import STATUSES, checker_key, gate_order
from .snapshot import EvidenceError, dependency_keys, digest, file_entry, read_json
from .reviews import decomposition_results
from .drift import describe


def changes(record, manifest, gate, policy, declarations=None):
    expected = dependency_keys(manifest, gate, policy)
    actual = record.get("dependencies", {})
    reasons = [name for name in sorted(set(expected) | set(actual))
               if expected.get(name) is None or expected.get(name) != actual.get(name)]
    if record.get("gate_key") != digest(gate):
        reasons.append("gate definition")
    if record.get("policy") != digest(policy):
        reasons.append("approved obligation map")
    if record.get("checker") != checker_key():
        reasons.append("checker implementation")
    for command in record.get("commands", []):
        for path, runtime_expected in command.get("runtime", {}).items():
            try:
                if file_entry(Path(path)) == runtime_expected:
                    continue
            except (EvidenceError, OSError):
                pass
            reasons.append("checker runtime: " + path)
    precise = gate.get("declaration_freshness")
    if precise and declarations:
        selected = declaration_keys(precise, declarations)
        previous = record.get("declarations", {})
        group = "source:" + precise["source"]
        if (group in reasons and expected.get(group) and actual.get(group) and
                selected and previous.get("keys") == selected["keys"] and
                previous.get("source") == precise["source"] and
                previous.get("use") == precise["use"]):
            reasons.remove(group)
    return reasons


def declaration_records(policy, manifest, runs):
    """Only a complete export checked against current source can narrow freshness."""
    selected = {}
    for record in sorted(runs, key=lambda item: item.get("finished", "")):
        gate = policy["gates"].get(record.get("gate"))
        if not gate or record.get("outcome") != "pass" or changes(record, manifest, gate, policy):
            continue
        documents = []
        for name in record.get("artifacts", {}):
            if name.startswith("metadata-") and name.endswith(".json"):
                documents.extend(read_json(Path(record["path"]).parent / name))
        if documents and all(item.get("complete") for item in documents):
            selected[record["gate"]] = {"graphs": {item["root"]: item for item in documents},
                                        "record": record["path"], "trusted": record["trusted"]}
    return selected


def declaration_keys(precise, declarations):
    exported = declarations.get(precise["gate"])
    if not exported:
        return None
    keys = {}
    for root in precise["roots"]:
        graph = exported["graphs"].get(root)
        if not graph or not graph.get("complete") or not graph.get(precise["use"] + "_key"):
            return None
        keys[root] = graph[precise["use"] + "_key"]
    return {"source": precise["source"], "use": precise["use"], "keys": keys,
            "metadata_record": exported["record"]}


def read_runs(store, authority):
    runs, rejected = [], []
    for path in sorted((Path(store) / "runs").glob("*/result.json")):
        try:
            if authority is None:
                envelope = read_json(path)
                record = envelope.get("record", {})
                record = {**record, "trusted": False}
            else:
                record = {**authority.read(path), "trusted": True}
            for relative, expected in record.get("artifacts", {}).items():
                artifact = path.parent / relative
                if artifact.resolve().parent != path.parent.resolve() or file_entry(artifact) != expected:
                    raise EvidenceError(f"result artifact changed: {relative}")
            record["path"] = str(path)
            runs.append(record)
        except (EvidenceError, OSError, KeyError, TypeError) as error:
            rejected.append(f"{path}: {error}")
    return runs, rejected


def gate_results(policy, manifest, runs, declarations=None, store=None):
    declarations = declarations if declarations is not None else declaration_records(policy, manifest, runs)
    results, stale = {}, []
    for name, gate in policy["gates"].items():
        matches, previous = [], []
        missing = [key for key, value in dependency_keys(manifest, gate, policy).items() if value is None]
        for record in runs:
            if record.get("gate") != name:
                continue
            previous.append(record)
            drift = changes(record, manifest, gate, policy, declarations)
            if drift:
                precise = gate.get("declaration_freshness")
                keys = declaration_keys(precise, declarations) if precise else None
                stale.append({"gate": name, "snapshot": record.get("snapshot"), "changes": drift,
                              "details": describe(record, manifest, drift, store, precise, keys)})
            else:
                matches.append(record)
        # A failed attempt does not turn into a pass or erase an existing matching proof.
        passing = [record for record in matches if record.get("outcome") == "pass"]
        latest = max(passing or matches,
                     key=lambda record: record.get("finished", record.get("started", "")), default=None)
        shown = latest or max(previous, key=lambda record: record.get("finished", record.get("started", "")),
                              default=None)
        results[name] = {"pass": bool(latest and latest.get("trusted")
                                    and latest.get("outcome") == "pass"),
                         "completed": bool(latest and latest.get("outcome") == "pass"),
                         "record": latest, "shown": shown, "missing_inputs": missing,
                         "freshness": "not-captured" if missing else (
                             "current" if latest else "stale" if previous else "missing"),
                         "basis": "declarations" if latest and changes(latest, manifest, gate, policy) else "sources"}
    graph = {name: gate.get("requires", []) for name, gate in policy["gates"].items()}
    for name in TopologicalSorter(graph).static_order():
        results[name]["prerequisites"] = [required for required in graph[name]
                                          if not results[required]["completed"]]
        results[name]["pass"] = results[name]["pass"] and all(
            results[required]["pass"] for required in graph[name])
        results[name]["completed"] = results[name]["completed"] and all(
            results[required]["completed"] for required in graph[name])
    return results, stale


def review_results(policy, manifest, authority):
    result = {name: False for name in policy["reviews"]}
    if authority is None:
        return result
    for path in sorted((authority.directory / "reviews").glob("*.json")):
        try:
            record = authority.read(path)
        except EvidenceError:
            continue
        name = record.get("review")
        if name not in result or name == "decomposition":
            continue
        expected = policy["reviews"][name]
        # Reviews name the exact source manifest, not merely a Git commit.
        valid = (record.get("snapshot") == digest(manifest)
                 and record.get("scope") == expected["scope"]
                 and record.get("policy") == digest(policy)
                 and bool(record.get("reviewer")) and record.get("outcome") == "pass")
        result[name] = result[name] or valid
    return result


def report(policy, manifest, store, authority, active=None, invocation=None):
    runs, rejected = read_runs(store, authority)
    gates, stale = gate_results(policy, manifest, runs, store=store)
    reviews = review_results(policy, manifest, authority)
    decompositions = decomposition_results(policy, manifest, store, authority)
    obligations = []
    for name, obligation in policy["obligations"].items():
        missing = []
        if not authority:
            missing.append("approved checker authority")
        missing.extend(obligation.get("open_requirements", []))
        if obligation.get("target_required") and not obligation.get("target"):
            missing.append("exact Lean target and checked closure registration")
        if not obligation["gates"]:
            missing.append("implemented closing gate")
        for gate in obligation["gates"]:
            for required in [gate, *policy["gates"][gate].get("requires", [])]:
                if not gates[required]["pass"]:
                    missing.append(("approved check " if gates[required]["completed"] else "execution of ") + required)
        review_states = {review: ("approved" if reviews[review] else "missing")
                         for review in obligation["reviews"]}
        if name in decompositions:
            decision = decompositions[name]
            review_states["decomposition"] = ("approved" if decision["accepted"] else
                "diagnostic " + decision["state"] if decision.get("checker") == "diagnostic" else decision["state"])
        missing.extend("review " + review for review, state in review_states.items() if state != "approved")
        obligations.append({"id": name, "status": obligation["status"],
                            "phase": obligation.get("phase", "tracked scope"),
                            "closed": not missing, "missing": sorted(set(missing)),
                            "gap": obligation["gap"], "target": obligation.get("target"),
                            "argument": obligation.get("argument"),
                            "gates": gate_order(policy, obligation["gates"]),
                            "reviews": review_states,
                            "decomposition": decompositions.get(name),
                            "next_command": [*(invocation or []), "checkpoint", name]
                                            if obligation["gates"] and any(
                                                not gates[gate]["completed" if not authority else "pass"]
                                                for gate in obligation["gates"]) else None})
    phase_statuses = {}
    for phase in sorted({item["phase"] for item in obligations}):
        phase_statuses[phase], prefix = {}, True
        for status in STATUSES:
            selected = [item for item in obligations if item["phase"] == phase and item["status"] == status]
            prefix = prefix and bool(selected) and all(item["closed"] for item in selected)
            phase_statuses[phase][status] = prefix
    statuses = {status: bool(phase_statuses) and all(value[status] for value in phase_statuses.values())
                for status in STATUSES}
    return {"snapshot": digest(manifest), "active": active, "statuses": statuses,
            "phase_statuses": phase_statuses,
            "obligations": obligations, "stale": stale, "rejected": rejected,
            "gates": {name: {"accepted": result["pass"],
                             "execution": ("passed" if result["shown"].get("outcome") == "pass" else
                                           result["shown"].get("outcome")) if result["shown"] else "not-run",
                             "freshness": result["freshness"], "freshness_basis": result["basis"],
                             "checker": ("approved" if result["shown"].get("trusted") else "diagnostic")
                                        if result["shown"] else "none",
                             "prerequisites": result["prerequisites"],
                             "missing_inputs": result["missing_inputs"],
                             "elapsed_seconds": result["shown"].get("elapsed_seconds") if result["shown"] else None,
                             "timings_seconds": result["shown"].get("timings_seconds") if result["shown"] else None,
                             "record": result["shown"].get("path") if result["shown"] else None}
                      for name, result in gates.items()}}


def markdown(result):
    import shlex
    lines = [f"Snapshot: `{result['snapshot']}`", "",
             "Active criterion: " + (result["active"]["obligation"] if result["active"] else "none"), "",
             "| Phase | Compiler-closed | Conformance-closed | Production-closed |",
             "| --- | --- | --- | --- |"]
    lines.extend("| " + phase + " | " + " | ".join("closed" if value[status] else "open"
                 for status in STATUSES) + " |" for phase, value in result["phase_statuses"].items())
    lines += ["", "| Gate | Execution | Freshness | Checker | Prerequisites |",
              "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {name} | {gate['execution']} | {gate['freshness']} | {gate['checker']} | " +
                 (", ".join(gate["missing_inputs"] + gate["prerequisites"]) or "ready") + " |"
                 for name, gate in result["gates"].items())
    lines += ["", "| Phase / criterion | Status | Remaining connection |", "| --- | --- | --- |"]
    for item in result["obligations"]:
        detail = "Checked against the approved target and gates." if item["closed"] else (
            item["gap"] + " Missing: " + ", ".join(item["missing"]) + ".")
        lines.append(f"| {item['phase']} / {item['id']} | {'closed' if item['closed'] else 'open'} | {detail.replace('|', '/')} |")
    for item in result["obligations"]:
        if item["next_command"]:
            lines += ["", f"Next check for {item['id']}: `{shlex.join(item['next_command'])}`"]
    if result["stale"]:
        lines += ["", "Stale for this selection:", ""]
        for item in result["stale"]:
            lines.append(f"- {item['gate']}: {', '.join(item['changes'])}.")
            for changed in item.get("details", {}).get("files", []):
                lines.append(f"  {changed['change']}: `{changed['path'] or changed['dependency']}`.")
            for changed in item.get("details", {}).get("declarations", []):
                lines.append(f"  {changed['key']} key {changed['change']}: `{changed['name']}`.")
    if result["rejected"]:
        lines += ["", "Rejected records:", ""] + ["- " + value for value in result["rejected"]]
    return "\n".join(lines) + "\n"


def explain(result, name):
    import shlex
    item = next(item for item in result["obligations"] if item["id"] == name)
    lines = [f"Criterion: {name}", "", f"Required connection: {item['gap']}",
             f"Exact target: {item['target'] or 'not registered'}"]
    if item["argument"]:
        lines += ["", "Argument: " + item["argument"]]
    if item.get("decomposition"):
        decision = item["decomposition"]
        lines += ["", "Decomposition review: " + item["reviews"]["decomposition"] + "."]
        if decision.get("request"):
            lines.append(f"Request: [{item['target']}]({decision['request']}).")
        if decision.get("record"):
            lines.append(f"Decision: [{decision['reviewer']}]({decision['record']}).")
        for name, assessment in decision.get("assessments", {}).items():
            lines.append(f"- {name}: {assessment['outcome']}. {assessment['reason']}")
        lines.extend(decision["reasons"])
    lines += ["", "Checks:", ""]
    for gate in item["gates"]:
        value = result["gates"][gate]
        lines.append(f"- {gate}: execution {value['execution']}; freshness {value['freshness']}; "
                     f"checker {value['checker']}.")
        if value["missing_inputs"]:
            lines.append("  Supply: " + ", ".join(value["missing_inputs"]) + ".")
        if value["record"]:
            lines.append(f"  Evidence: [{gate}]({value['record']}).")
        if value.get("elapsed_seconds") is not None:
            lines.append(f"  Gate total: {value['elapsed_seconds']:.3f} seconds.")
            if value.get("timings_seconds"):
                lines.append("  Components: " + ", ".join(
                    f"{name}={seconds:.3f}s" for name, seconds in value["timings_seconds"].items()) + ".")
    lines += ["", "Reviews:", ""]
    lines += [f"- {review}: {state}." for review, state in item["reviews"].items()] or ["None registered."]
    lines += ["", f"Accepted closure: {'closed' if item['closed'] else 'open'}."]
    if item["missing"]:
        lines += ["Remaining: " + ", ".join(item["missing"]) + "."]
    if item["next_command"]:
        lines += ["", "Next check:", "", "```sh", shlex.join(item["next_command"]), "```"]
    elif not item["closed"]:
        lines += ["No executable check can supply the missing review or registration."]
    return "\n".join(lines) + "\n"
