#!/usr/bin/env python3
"""Check the extension-cell coordinate move; Lean remains the proof authority.

Run with: timeout --signal=KILL 300 python3 -B wide_layout_controls.py
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


def variables(suffix):
    return f"""
(declare-const lane{suffix} Int)
(declare-const cell{suffix} Int)
(declare-const digit{suffix} Int)
(assert (and (<= 0 lane{suffix}) (< lane{suffix} 54)))
(assert (and (<= 0 cell{suffix}) (< cell{suffix} 2)))
(assert (and (<= 0 digit{suffix}) (< digit{suffix} 41)))
(define-fun old{suffix} () Int (+ (* 41 (+ (* 2 lane{suffix}) cell{suffix})) digit{suffix}))
(define-fun new{suffix} () Int (+ (* 41 (+ (* 54 cell{suffix}) lane{suffix})) digit{suffix}))
"""


def main():
    checks = {
        "plain_shift_changes_cell_lane_meaning": solve(
            variables("") + "(assert (distinct old new))", "sat",
            ("lane", "cell", "digit", "old", "new")),
        "permutation_has_no_collisions": solve(
            variables("a") + variables("b") +
            "(assert (= newa newb))\n(assert (distinct olda oldb))", "unsat"),
        "permutation_stays_in_its_field_block": solve(
            variables("") + "(assert (or (< new 0) (>= new 4428)))", "unsat"),
        "inverse_recovers_all_indices": solve(
            variables("") + """
(define-fun recoveredCell () Int (div (div new 41) 54))
(define-fun recoveredLane () Int (mod (div new 41) 54))
(define-fun recoveredDigit () Int (mod new 41))
(assert (or (distinct recoveredCell cell) (distinct recoveredLane lane)
            (distinct recoveredDigit digit)))
""", "unsat"),
    }
    print(json.dumps({"scope": "two-cell ring-coordinate relocation", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
