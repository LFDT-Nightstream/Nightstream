#!/usr/bin/env python3
"""Check the seven PiDEC source intervals used by Wide/PiDECSource.lean.

These arithmetic controls do not prove circuit or witness equivalence. Lean
owns the starts, slot meanings, bounds, and source transport proofs.
"""

import json

from wide_layout_controls import solve

# (old start, new start, slot count), in PiDECDirectPlan.Location order.
# Exact offsets are proved by PiDECSource.source_offsets.
REGIONS = (
    (19_795_693, 19_587_528, 1188),
    (19_801_201, 19_593_036, 270),
    (19_803_199, 19_595_034, 108),
    (19_827_499, 19_619_334, 1512),
    (28_421_542, 27_496_062, 49248),
    (28_470_790, 27_545_310, 270),
    (28_471_060, 27_545_580, 17820),
)


def select(suffix, position):
    result = str(REGIONS[-1][position])
    for kind in reversed(range(len(REGIONS) - 1)):
        result = f"(ite (= kind{suffix} {kind}) {REGIONS[kind][position]} {result})"
    return result


def variables(suffix):
    return f"""
(declare-const kind{suffix} Int)
(declare-const slot{suffix} Int)
(assert (and (<= 0 kind{suffix}) (< kind{suffix} 7)))
(assert (and (<= 0 slot{suffix}) (< slot{suffix} {select(suffix, 2)})))
(define-fun old{suffix} () Int (+ {select(suffix, 0)} slot{suffix}))
(define-fun new{suffix} () Int (+ {select(suffix, 1)} slot{suffix}))
(define-fun shift{suffix} () Int (ite (< kind{suffix} 4) 208165 925480))
"""


def main():
    checks = {
        "one_global_shift_corrupts_parent_fields": solve(
            variables("") + "(assert (distinct (- old 925480) new))", "sat",
            ("kind", "slot", "old", "new")),
        "separate_offsets_recover_every_source": solve(
            variables("") + "(assert (distinct (- old shift) new))", "unsat"),
        "source_intervals_do_not_alias": solve(
            variables("a") + variables("b") +
            "(assert (= newa newb))\n(assert (distinct olda oldb))", "unsat"),
        "all_sources_precede_running_transition": solve(
            variables("") + "(assert (or (< new 0) (>= new 27563400)))", "unsat"),
    }
    print(json.dumps({"scope": "seven PiDEC source intervals", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
