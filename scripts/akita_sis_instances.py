#!/usr/bin/env python3
"""The Module-SIS matrix views of the Akita schedule files (`SECURITY_MODEL.md`, trust item 7).

    python3 scripts/akita_sis_instances.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = {name: ROOT / f"crates/ns-compress/artifacts/{name}.aks" for name in ("key20", "key30")}
Q = 2**64 - 59
# `Spec.Compression.Akita.Key20.instances`
LEAN_KEY20 = [(32, 1, 128, "inf", 1), (64, 1, 1024, "inf", 1), (64, 2, 77, "inf", 63), (64, 3, 616, "inf", 63),
              (64, 8, 256, "l2", 2**46), (64, 8, 512, "l2", 2**45), (128, 1, 1936, "inf", 7), (128, 1, 1092, "inf", 31),
              (128, 4, 512, "inf", 126852), (128, 4, 256, "l2", 2**44), (512, 2, 56, "inf", 53477172),
              (1024, 1, 224, "inf", 32505732), (1024, 1, 56, "inf", 53477172)]


def inner(mx):
    route = mx["security_route"]
    if "Linf" in route:
        r = route["Linf"]
        return (r["ring_dimension"], mx["output_rank"], mx["input_width"], "inf", r["coeff_linf_bound"])
    k = route["L2"]["table_key"]
    return (k["ring_dimension"], mx["output_rank"], mx["input_width"], "l2", k["collision_l2_sq"])


def keyed(mx):
    k = mx["sis_table_key"]
    return (k["ring_dimension"], mx["output_rank"], mx["input_width"], "inf", k["coeff_linf_bound"])


def level_views(tag, params):
    out = []
    compressed = params["payload_mode"] == "Compressed"
    for g, e in enumerate(params["groups"]["entries"]):
        p = e["profile"]
        out.append((f"{tag} g{g} A", inner(p["inner"]["matrix"])))
        b = keyed(p["outer"]["matrix"])
        out.append((f"{tag} g{g} B", b))
        # The two compression maps of the pinned Akita code (`akita-params`, `compression.rs`).
        if compressed:
            out.append((f"{tag} g{g} F1", (64, 1, p["outer_slice_count"] * b[1] * b[0], "inf", 1)))
            out.append((f"{tag} g{g} F2", (32, 1, 128, "inf", 1)))
    d = keyed(params["open_matrix"])
    out.append((f"{tag} D", d))
    if compressed:
        out.append((f"{tag} H1", (64, 1, d[1] * d[0], "inf", 1)))
        out.append((f"{tag} H2", (32, 1, 128, "inf", 1)))
    return out


def views(path):
    out = []
    for r, row in enumerate(json.loads(path.read_text(encoding="utf-8"))["rows"]):
        s = row["schedule"]
        out += level_views(f"r{r} L0", s["root"]["params"])
        for i, f in enumerate(s["recursive_folds"], 1):
            out += level_views(f"r{r} L{i}", f["params"])
        out.append((f"r{r} T A_term", inner(s["terminal"]["inner"]["matrix"])))
    return out


def maximal(keys):
    keys = set(keys)
    return sorted(a for a in keys if not any(
        b != a and b[:2] == a[:2] and b[3] == a[3] and b[2] >= a[2] and b[4] >= a[4] for b in keys))


def instances():
    uses, top = {}, set()
    for name, path in SCHEDULES.items():
        found = views(path)
        for use, key in found:
            uses.setdefault(key, []).append(f"{name} {use}")
        top.update(maximal(key for _, key in found))
    return [(key, uses[key], key in top) for key in sorted(uses)]


def main():
    for name, path in SCHEDULES.items():
        found = views(path)
        top = maximal(key for _, key in found)
        print(f"{name}: {len(found)} views, {len({key for _, key in found})} distinct, {len(top)} maximal")
        if name == "key20":
            assert top == sorted(LEAN_KEY20)
    rows = instances()
    print(f"both: {len(rows)} distinct, {sum(key[3] == 'inf' for key, _, _ in rows)} in ℓ∞, "
          f"{sum(m for _, _, m in rows)} maximal")


if __name__ == "__main__":
    main()
