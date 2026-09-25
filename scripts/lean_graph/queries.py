"""Query checked declaration graphs without treating graph edges as proofs."""

from collections import deque
from pathlib import Path

from .snapshot import EvidenceError


def index(exports):
    nodes, reverse = {}, {}
    for exported in exports.values():
        for graph in exported["graphs"].values():
            for name, node in graph["proof"].items():
                if name in nodes and any(nodes[name][key] != node[key]
                                         for key in ("meaning_fingerprint", "proof_fingerprint")):
                    raise EvidenceError(f"conflicting declaration metadata: {name}")
                nodes[name] = node
    for name, node in nodes.items():
        if not node["local"]:
            continue
        for kind in ("meaning", "proof"):
            for child in node[kind + "_dependencies"]:
                reverse.setdefault(child, set()).add((name, kind))
    return nodes, reverse


def query(exports, operation, name, parent, policy, status, source):
    nodes, reverse = index(exports)
    if name not in nodes:
        raise EvidenceError(f"declaration is outside the checked exported graphs: {name}")
    node = nodes[name]
    result = {"declaration": name, "kind": node["kind"], "statement": node.get("statement"),
              "proposition": node.get("proposition"), "type_expression": node.get("type_expression"),
              "scope": "The selected exported roots; external libraries stop at their pinned provenance.",
              "metadata_records": sorted({item["record"] for item in exports.values()})}
    if isinstance(node.get("origin"), str) and node["origin"].startswith("source/"):
        result["source"] = str(Path(source) / node["origin"].removeprefix("source/"))
    if operation == "requires":
        result["meaning_dependencies"] = sorted(set(node["meaning_dependencies"]))
        result["proof_dependencies"] = sorted(set(node["proof_dependencies"]))
    elif operation == "used-by":
        result["dependents"] = [{"declaration": name, "edge": kind}
                                for name, kind in sorted(reverse.get(name, set()))]
    elif operation == "path":
        if parent not in nodes:
            raise EvidenceError(f"parent is outside the checked exported graphs: {parent}")
        previous, pending = {name: None}, deque([name])
        while pending and parent not in previous:
            current = pending.popleft()
            for next_name, kind in sorted(reverse.get(current, set())):
                if next_name not in previous:
                    previous[next_name] = (current, kind)
                    pending.append(next_name)
        path = []
        current = parent
        if current in previous:
            while previous[current] is not None:
                child, kind = previous[current]
                path.append({"dependency": child, "consumer": current, "edge": kind})
                current = child
        result["parent"], result["path"] = parent, list(reversed(path))
        result["connected"] = parent in previous
    else:
        raise EvidenceError(f"unknown graph query: {operation}")
    ancestors, pending = set(), [name]
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(parent for parent, _ in reverse.get(current, set()))
    witnesses = {}
    for gate in policy["gates"].values():
        for command in gate["commands"]:
            witnesses.update(command["completion"].get("closures", {}))
    result["obligations"] = []
    for obligation in status["obligations"]:
        target = obligation["target"]
        if target == name or witnesses.get(target) in ancestors:
            result["obligations"].append({"id": obligation["id"], "target": target,
                "closed": obligation["closed"], "missing": obligation["missing"],
                "evidence": {gate: status["gates"][gate] for gate in obligation["gates"]}})
    return result


def markdown(result):
    lines = [f"Declaration: `{result['declaration']}`", "", result["scope"], ""]
    if result.get("source"):
        lines += [f"Source: [{result['declaration']}]({result['source']})", ""]
    if result.get("statement"):
        lines += ["Elaborated type:", "", "```lean", result["statement"], "```"]
    if result.get("proposition"):
        lines += ["", "Target proposition, including its premises:", "", "```lean", result["proposition"], "```"]
    for key, label in (("meaning_dependencies", "Direct meaning dependencies"),
                       ("proof_dependencies", "Direct proof dependencies")):
        if key in result:
            lines += ["", label + ":", ""]
            lines += [f"- `{name}`" for name in result[key]] or ["None."]
    if "dependents" in result:
        lines += ["", "Direct dependents:", ""]
        lines += [f"- `{item['declaration']}` ({item['edge']})" for item in result["dependents"]] or ["None in this export."]
    if "path" in result:
        lines += ["", "Dependency path:", ""]
        lines += [f"- `{item['dependency']}` → `{item['consumer']}` ({item['edge']})"
                  for item in result["path"]]
        if not result["connected"]:
            lines.append("No path in this export. This does not decide mathematical composition.")
        elif not result["path"]:
            lines.append("The selected declaration is the parent.")
    lines += ["", "Linked evidence:", ""]
    for item in result["obligations"]:
        lines.append(f"- {item['id']}: accepted closure {'closed' if item['closed'] else 'open'}.")
        for name, gate in item["evidence"].items():
            if gate["record"]:
                lines.append(f"  [{name}]({gate['record']}): {gate['execution']}; {gate['freshness']}; {gate['checker']}.")
    lines += [f"- [Declaration export]({path})" for path in result["metadata_records"]]
    return "\n".join(lines) + "\n"
