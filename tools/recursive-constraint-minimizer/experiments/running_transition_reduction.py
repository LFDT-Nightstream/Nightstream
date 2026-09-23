#!/usr/bin/env python3
"""Untrusted Goldilocks controls for a shared running-transition flag.

Run with `timeout --signal=KILL 300 python3 ...`, per AGENTS.md.
The selected gate stays quadratic; Lean owns the actual circuit proof.
"""

import json
from pathlib import Path
import shutil
import subprocess

P = 18446744069414584321
VARIABLES = ("t", "inv", "g", "d", "r", "o", "initial", "current")


def value(x):
    return f"(as ff{x % P} F)"


def add(*args):
    return f"(ff.add {' '.join(args)})"


def mul(*args):
    return f"(ff.mul {' '.join(args)})"


def sub(left, right):
    return add(left, mul(value(-1), right))


def eq(left, right):
    return f"(= {left} {right})"


def conjunction(terms):
    return f"(and {' '.join(terms)})"


def main():
    solver = shutil.which("cvc5")
    assert solver, "cvc5 is required"
    old_flag = mul("t", "inv")
    old = [
        eq(mul("t", sub(value(1), old_flag)), value(0)),
        eq(add(mul(sub(value(1), old_flag), "d"), mul(old_flag, "r")), "o"),
        eq(mul(sub(value(1), old_flag), sub("initial", "current")), value(0)),
    ]
    new = [
        eq("g", old_flag),
        eq(mul("t", sub(value(1), "g")), value(0)),
        eq(mul("g", sub("r", "d")), sub("o", "d")),
        eq(mul(sub(value(1), "g"), sub("initial", "current")), value(0)),
    ]
    cases = [
        ("projection_preserves_original", new + [f"(not {conjunction(old)})"], "unsat", None),
        ("constructive_flag_extension", old + [new[0], f"(not {conjunction(new)})"], "unsat", None),
        ("flag_is_boolean", new[:2] + [f"(not {eq('g', value(0))})", f"(not {eq('g', value(1))})"], "unsat", None),
        ("missing_flag_link", new[1:] + [f"(not {conjunction(old)})"], "sat",
         dict(zip(VARIABLES, [0, 0, 1, 0, 1, 1, 0, 1]))),
        ("missing_iteration_binding", [new[0]] + new[2:] + [f"(not {conjunction(old)})"], "sat",
         dict(zip(VARIABLES, [1, 0, 0, 0, 1, 0, 0, 0]))),
    ]
    results = {}
    for name, assertions, expected, witness in cases:
        lines = ["(set-logic QF_FF)", f"(define-sort F () (_ FiniteField {P}))"]
        lines += [f"(declare-const {var} F)" for var in VARIABLES]
        if witness:
            assertions += [eq(var, value(x)) for var, x in witness.items()]
        lines += [f"(assert {assertion})" for assertion in assertions]
        query = "\n".join(lines + ["(check-sat)"]) + "\n"
        run = subprocess.run([solver, "--lang=smt2", "--ff-solver=gb"],
                             input=query, text=True, capture_output=True, check=True)
        status = run.stdout.strip()
        assert status == expected, (name, run.stdout, run.stderr)
        if witness:
            t, inv, g, d, r, o, initial, current = [witness[var] for var in VARIABLES]
            f = t * inv % P
            original = [t * (1 - f) % P == 0,
                        ((1 - f) * d + f * r - o) % P == 0,
                        (1 - f) * (initial - current) % P == 0]
            proposed = [(g - f) % P == 0, t * (1 - g) % P == 0,
                        (g * (r - d) - (o - d)) % P == 0,
                        (1 - g) * (initial - current) % P == 0]
            omitted = 0 if name == "missing_flag_link" else 1
            assert not all(original)
            assert all(holds for index, holds in enumerate(proposed) if index != omitted)
        results[name] = {"status": status, "smt2": query, "counterexample": witness}
        print(f"{name}: {status}", flush=True)
    savings = 296137 * 41 - 1
    logical = 184359519 - savings
    committed = ((logical + 53) // 54) * 54
    report = {
        "scope": "Local equations only; no selected layout, sparse-matrix, or production integration claim.",
        "solver": subprocess.check_output([solver, "--version"], text=True).splitlines()[0],
        "field_modulus": P,
        "unchanged_parameters": {"b": 2, "k_rho": 16, "B": 65536},
        "candidate": "Replace lowering scratch by one Boolean recursive flag; retain the inverse field.",
        "witness_projection": "Drop g; preserve all original logical inputs, outputs and inverse.",
        "constructive_extension": "Set g=t*inv; the original binding forces g in {0,1}.",
        "conditional_geometry": {
            "removed_field_slots": 296137, "added_unit_coordinates": 1,
            "logical_coordinate_saving": savings, "logical_coordinates": logical,
            "committed_coordinates": committed,
            "committed_coordinate_saving": 184359564 - committed,
            "old_transition_rows": 345495, "new_transition_rows": 49359,
            "logical_rows": 4703127 - (345495 - 49359),
            "matrix_nonzeros": "See running-transition-next-candidate.md for analytical counts; no generated candidate count.",
        },
        "required_lean_obligations": [
            "Equivalence to the actual selected running-transition constraints.",
            "Low-norm flag encoding and constructive old/new witness maps.",
            "Scratch support and reconstruction for the complete selected assignment.",
            "Unchanged fixed-polynomial ports and exact matrix forms.",
        ],
        "runs": results,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
