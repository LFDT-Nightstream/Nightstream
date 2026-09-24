#!/usr/bin/env python3
"""Check PiDEC source shifts and the retained reference lane/cell order.

These arithmetic controls do not prove circuit or witness equivalence. Lean
owns the starts, slot meanings, bounds, and source transport proofs.
"""

import json
import shutil
import subprocess


def solve(body, expected, values=()):
    solver = shutil.which("cvc5")
    if solver is None:
        raise RuntimeError("cvc5 is required")
    query = "(set-logic QF_LIA)\n(set-option :produce-models true)\n"
    query += body + "\n(check-sat)\n"
    if values:
        query += "(get-value (" + " ".join(values) + "))\n"
    result = subprocess.run([solver, "--lang=smt2"], input=query, text=True,
                            capture_output=True, check=True)
    output = result.stdout.strip()
    if output.splitlines()[0] != expected:
        raise AssertionError(output + result.stderr)
    return output

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


def slot_variables():
    return """
(declare-const lane Int)
(declare-const cell Int)
(declare-const digit Int)
(declare-const start Int)
(assert (and (<= 0 lane) (< lane 54) (<= 0 cell) (< cell 2)))
(assert (and (<= 0 digit) (< digit 41) (<= 0 start)))
(define-fun reference () Int (+ (* 41 (+ (* 2 lane) cell)) digit))
(define-fun retained () Int (+ start reference))
(define-fun transposed () Int (+ start (* 41 (+ (* 54 cell) lane)) digit))
"""


def main():
    checks = {
        "ring_major_order_changes_reference_cells": solve(
            slot_variables() + "(assert (distinct retained transposed))", "sat",
            ("lane", "cell", "digit", "retained", "transposed")),
        "shift_recovers_reference_lane_cell_and_digit": solve(
            slot_variables() + """
(define-fun recovered () Int (- retained start))
(assert (or (distinct (div (div recovered 41) 2) lane)
            (distinct (mod (div recovered 41) 2) cell)
            (distinct (mod recovered 41) digit)))
""", "unsat"),
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
    print(json.dumps({"scope": "PiDEC source intervals and reference slot order", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
