"""Module-SIS estimates of the main key and of each matrix view of the Akita schedule files, by the
method of SuperNeo v1.2 Appendix B.6 (`decisions/field-2p64-59-ring-x64.md`; `SECURITY_MODEL.md`,
the `MSIS` entry and trust item 7). It writes `scripts/akita_sis_estimates.csv`.

Run with malb/lattice-estimator commit 3e48ef421ec256afddb3e7d2249a77eab6e9ba12 on PYTHONPATH:

    sage -python scripts/estimate_akita_sis.sage
"""

import csv
from pathlib import Path

from sage.all import sqrt, log, oo

from estimator import SIS
from estimator.reduction import ADPS16

from akita_sis_instances import Q, instances

QUANTUM = ADPS16(mode="quantum")
TABLE = Path(__file__).resolve().parent / "akita_sis_estimates.csv"
MAIN_KEY = (64, 18, 2**22, "inf", 8 * 128 * (2**14 + 64 * 128))
COLUMNS = ["views", "d", "n", "m", "norm", "bound", "maximal", "quantum", "beta", "classical", "default",
           "rank_up_n", "rank_up_quantum", "base_down_bound", "base_down_m", "base_down_quantum"]


def length(d, m, norm, bound):
    return bound * sqrt(m * d) if norm == "inf" else sqrt(bound)


def run(d, n, m, norm, bound):
    p = SIS.Parameters(n=n * d, q=Q, m=m * d, length_bound=length(d, m, norm, bound), norm=2)
    out = {}
    for name, c in (
        ("quantum", SIS.estimate(p, red_cost_model=QUANTUM, quiet=True)["lattice"]),
        ("classical", SIS.estimate.rough(p, quiet=True)["lattice"]),
        ("default", SIS.estimate(p, quiet=True)["lattice"]),
    ):
        out[name] = {"bits": float("inf") if c["rop"] == oo else float(log(c["rop"], 2)), "beta": int(c["beta"]), "d": int(c["d"])}
    return out


def fix(d, n, m, norm, bound):
    out = {"ranks": []}
    k = n
    while True:
        k += 1
        x = run(d, k, m, norm, bound)
        out["ranks"].append({"n": k, **x})
        if x["quantum"]["bits"] >= 128:
            break
    k = (bound + 1).bit_length() - 1
    out["bases"] = []
    if bound + 1 == 2**k:
        for j in range(k - 1, 0, -1):
            w = -(-m * -(-64 // j) // -(-64 // k))
            x = run(d, n, w, norm, 2**j - 1)
            out["bases"].append({"log_basis": j, "bound": 2**j - 1, "m": w, **x})
            if x["quantum"]["bits"] >= 128:
                break
    return out


def bits(value):
    return f"{value:.3f}"


def row(uses, key, top, x, low):
    d, n, m, norm, bound = key
    out = {"views": "; ".join(uses), "d": d, "n": n, "m": m, "norm": norm, "bound": bound,
           "maximal": top, "quantum": bits(x["quantum"]["bits"]),
           "beta": x["quantum"]["beta"], "classical": bits(x["classical"]["bits"]),
           "default": bits(x["default"]["bits"])}
    if low:
        up = low["ranks"][0]
        out.update(rank_up_n=up["n"], rank_up_quantum=bits(up["quantum"]["bits"]))
        if low["bases"]:
            base = low["bases"][-1]
            out.update(base_down_bound=base["bound"], base_down_m=base["m"],
                       base_down_quantum=bits(base["quantum"]["bits"]))
    return out


def main():
    rows = [row(["main key"], MAIN_KEY, "", run(*MAIN_KEY), None)]
    for key, uses, top in instances():
        x = run(*key)
        rows.append(row(uses, key, "yes" if top else "no", x, fix(*key) if x["quantum"]["bits"] < 128 else None))
    with TABLE.open("w", newline="", encoding="utf-8") as out:
        table = csv.DictWriter(out, COLUMNS, lineterminator="\n")
        table.writeheader()
        table.writerows(rows)
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
