"""Validate exported Lean origins and compute declaration keys.

Registered freshness rules can use current, complete exports. These keys do
not replace target review or mandatory relation-identity checks.
"""

from pathlib import Path

from .snapshot import EvidenceError, digest, entries, file_entry, read_json


DISPLAY_FIELDS = ("statement", "proposition", "type_expression")


def analyze(document, lean_root, manifest):
    lean_root = Path(lean_root).resolve()
    known = entries(manifest)
    nodes, failures = {}, []
    if document.get("schema") != 1 or not document.get("complete"):
        failures.append("exporter metadata is incomplete")
    packages = {item["name"]: item for item in read_json(lean_root / "lake-manifest.json")["packages"]}
    runtime = Path(document.get("runtime", "")).resolve()
    toolchain = (lean_root / "lean-toolchain").read_text().strip()
    checked = {}
    package_sources = {}
    for package_name in packages:
        prefix = "source/formal/nightstream-fprime/.lake/packages/" + package_name + "/"
        source = {key: value for key, value in known.items() if key.startswith(prefix)}
        package_sources[package_name] = digest(source) if source else None
    def checked_entry(path):
        if path not in checked:
            checked[path] = file_entry(path)
        return checked[path]
    for node in document.get("nodes", []):
        name = node.get("name")
        if not name or name in nodes or node.get("missing"):
            failures.append(f"missing or repeated declaration: {name}")
            continue
        if node.get("kind") not in {"definition", "theorem", "opaque", "inductive",
                                    "constructor", "recursor", "kernel-assumption", "quotient"}:
            failures.append(f"{name}: unsupported declaration form")
        path = Path(node["origin"]).resolve()
        normalized = {key: value for key, value in node.items()
                      if key != "origin" and key not in DISPLAY_FIELDS}
        if node["local"]:
            try:
                relative = path.relative_to(lean_root).as_posix()
                key = "source/formal/nightstream-fprime/" + relative
                if key not in known or checked_entry(path) != known[key]:
                    raise EvidenceError("local source is outside the captured manifest")
                normalized["origin"] = key
            except (ValueError, EvidenceError, OSError) as error:
                failures.append(f"{name}: {error}")
        else:
            provenance = None
            for package_name, package in packages.items():
                root = lean_root / ".lake/packages" / package_name
                if root in path.parents:
                    if package_sources[package_name] and package.get("rev") and package.get("url"):
                        provenance = {"package": package, "sources": package_sources[package_name],
                                      "module": node["module"]}
                    break
            if provenance is None and runtime / "lib/lean" in path.parents and toolchain:
                provenance = {"toolchain": toolchain, "module": node["module"]}
            if provenance is None:
                failures.append(f"{name}: unknown external module provenance")
            else:
                try:
                    provenance["compiled"] = checked_entry(path)
                except (EvidenceError, OSError) as error:
                    failures.append(f"{name}: {error}")
                normalized["origin"] = provenance
        meaning = {key: value for key, value in normalized.items()
                   if key not in ("proof", "proof_dependencies")}
        nodes[name] = {key: normalized[key] for key in
                       ("name", "kind", "module", "origin", "local",
                        "meaning_dependencies", "proof_dependencies") if key in normalized}
        nodes[name]["meaning_fingerprint"] = digest(meaning)
        nodes[name]["proof_fingerprint"] = digest(normalized.get("proof"))
        nodes[name].update({key: node[key] for key in DISPLAY_FIELDS if key in node})
    root = document.get("root")

    def closure(proofs):
        found, pending = {}, [root]
        while pending:
            name = pending.pop()
            if name in found:
                continue
            if name not in nodes:
                failures.append(f"missing dependency node: {name}")
                continue
            value = dict(nodes[name])
            if not proofs:
                value.pop("proof_fingerprint", None)
                value.pop("proof_dependencies", None)
            found[name] = value
            if value["local"]:
                pending.extend(value["meaning_dependencies"])
                if proofs:
                    pending.extend(value["proof_dependencies"])
        return found

    meaning, proof = closure(False), closure(True)
    def fingerprint(values):
        return digest({name: {key: value for key, value in node.items() if key not in DISPLAY_FIELDS}
                       for name, node in values.items()})
    return {"root": root, "complete": not failures, "failures": sorted(set(failures)),
            "meaning_key": fingerprint(meaning) if not failures else None,
            "proof_key": fingerprint(proof) if not failures else None,
            "meaning": meaning, "proof": proof}


def from_log(log_path, lean_root, manifest):
    import json
    result = []
    with Path(log_path).open() as handle:
        for line in handle:
            if line.startswith("LEAN_GRAPH_METADATA "):
                result.append(analyze(json.loads(line.removeprefix("LEAN_GRAPH_METADATA ")),
                                      lean_root, manifest))
    return result
