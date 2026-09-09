"""Explain file and registered-declaration changes without changing freshness."""

from pathlib import Path
import re

from .snapshot import EvidenceError, digest, read_json


def describe(record, manifest, reasons, store, precise=None, current_keys=None):
    result = {"files": [], "declarations": []}
    identifier = record.get("snapshot", "")
    if store and isinstance(identifier, str) and re.fullmatch(r"[0-9a-f]{64}", identifier):
        try:
            old = read_json(Path(store) / "snapshots" / identifier / "manifest.json")
            if digest(old) != identifier:
                raise EvidenceError("the earlier snapshot manifest changed")
            for reason in reasons:
                kind, _, name = reason.partition(":")
                if kind not in ("source", "input"):
                    continue
                collection = "sources" if kind == "source" else "inputs"
                before = old[collection].get(name)
                after = manifest[collection].get(name)
                if before is None or after is None:
                    result["files"].append({"dependency": reason, "path": None, "change": "not-captured"})
                    continue
                for path in sorted(set(before) | set(after)):
                    if before.get(path) != after.get(path):
                        result["files"].append({"dependency": reason, "path": path,
                            "change": "added" if path not in before else "removed" if path not in after else "changed"})
        except (EvidenceError, OSError, KeyError, TypeError) as error:
            result["unavailable"] = str(error)
    if precise and current_keys:
        previous = record.get("declarations", {}).get("keys", {})
        for name, key in current_keys["keys"].items():
            if previous.get(name) != key:
                result["declarations"].append({"name": name, "key": precise["use"],
                                                "change": "changed" if name in previous else "not-recorded"})
    return result
