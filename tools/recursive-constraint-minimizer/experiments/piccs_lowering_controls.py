#!/usr/bin/env python3
"""Finite-field controls for the compact degree-nine SumCheck evaluator.

Run with the project non-Lean test cap: timeout --signal=KILL 300.
Solver results are candidate evidence; Lean proves the witness mappings.
"""

import json
from pathlib import Path
import shutil
import subprocess


P = 18446744069414584321


def scalar(value):
    return f"(as ff{value % P} F)"


def add(a, b):
    return tuple(f"(ff.add {x} {y})" for x, y in zip(a, b))


def mul(a, b):
    return (
        f"(ff.add (ff.mul {a[0]} {b[0]}) (ff.mul {scalar(7)} {a[1]} {b[1]}))",
        f"(ff.add (ff.mul {a[0]} {b[1]}) (ff.mul {a[1]} {b[0]}))",
    )


def define(lines, name, value):
    for i, term in enumerate(value):
        lines.append(f"(define-fun {name}_{i} () F {term})")
    return (f"{name}_0", f"{name}_1")


def setup():
    lines = ["(set-logic QF_FF)", f"(define-sort F () (_ FiniteField {P}))"]
    coefficients = [(f"c{i}_0", f"c{i}_1") for i in range(10)]
    for value in [*coefficients, ("r0", "r1")]:
        for component in value:
            lines.append(f"(declare-const {component} F)")
    return lines, coefficients


def evaluate(lines, name, point, coefficients):
    result = (scalar(0), scalar(0))
    for i, coefficient in reversed(list(enumerate(coefficients))):
        result = define(lines, f"{name}_{i}", add(coefficient, mul(point, result)))
    return result


def different(a, b):
    return f"(or (distinct {a[0]} {b[0]}) (distinct {a[1]} {b[1]}))"


def solve(solver, lines, expected):
    query = "\n".join([*lines, "(check-sat)"]) + "\n"
    result = subprocess.run(
        [solver, "--lang=smt2", "--ff-solver=gb"], input=query,
        capture_output=True, text=True, check=True,
    )
    actual = result.stdout.strip()
    if actual != expected:
        raise AssertionError(f"expected {expected}, received {actual}: {result.stderr}")
    return actual


def gamma_controls(solver):
    arguments = [(0, 0), (1, 0), (2, 2), (3, 3), (4, 4), (5, 2),
                 (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
                 (12, 11), (13, 13), (14, 14), (15, 13)]
    exponents = [1]
    for index, (left, right) in enumerate(arguments):
        assert left <= index and right <= index
        exponents.append(exponents[left] + exponents[right])
    assert exponents[11] == 864 and exponents[16] == 12960
    lines = ["(set-logic QF_FF)", f"(define-sort F () (_ FiniteField {P}))"]
    values = [(f"g{i}_0", f"g{i}_1") for i in range(17)]
    for value in values:
        for component in value:
            lines.append(f"(declare-const {component} F)")
    for i, value in enumerate(values):
        for component, name in enumerate(value):
            forced = int(i == 16 and component == 0)
            lines.append(f"(assert (= {name} {scalar(forced)}))")
    equations = []
    for i, (left, right) in enumerate(arguments):
        equations.append([
            f"(assert (= {actual} {expected}))"
            for actual, expected in zip(values[i + 1], mul(values[left], values[right]))
        ])
    return {
        "gamma_zero_input_rejects_false_output": solve(
            solver, lines + [e for pair in equations for e in pair], "unsat"),
        "missing_gamma_link_counterexample": solve(
            solver, lines + [e for pair in equations[:-1] for e in pair], "sat"),
    }, exponents


def main():
    solver = shutil.which("cvc5")
    if solver is None:
        raise RuntimeError("cvc5 is required")
    checks = {}
    lines, coefficients = setup()
    at_zero = evaluate(lines, "zero", (scalar(0), scalar(0)), coefficients)
    at_one = evaluate(lines, "one", (scalar(1), scalar(0)), coefficients)
    simplified = coefficients[0]
    for coefficient in coefficients:
        simplified = add(simplified, coefficient)
    lines.append(f"(assert {different(add(at_zero, at_one), simplified)})")
    checks["boolean_sum_equivalence"] = solve(solver, lines, "unsat")

    lines, coefficients = setup()
    old = evaluate(lines, "old", ("r0", "r1"), coefficients)
    current = coefficients[-1]
    constraints = []
    for i in reversed(range(9)):
        product = (f"h{i}_0", f"h{i}_1")
        for component in product:
            lines.append(f"(declare-const {component} F)")
        expected = mul(("r0", "r1"), current)
        constraints.append([
            f"(assert (= {actual} {value}))" for actual, value in zip(product, expected)
        ])
        current = add(coefficients[i], product)
    full = lines + [c for pair in constraints for c in pair]
    full.append(f"(assert {different(old, current)})")
    checks["materialized_horner_equivalence"] = solve(solver, full, "unsat")

    # Omit the last product link, then replay an explicit false assignment.
    bad = lines + [c for pair in constraints[:-1] for c in pair]
    for coefficient in coefficients:
        for component in coefficient:
            bad.append(f"(assert (= {component} {scalar(0)}))")
    for component in ("r0", "r1"):
        bad.append(f"(assert (= {component} {scalar(0)}))")
    for i in range(9):
        for component in range(2):
            value = int(i == 0 and component == 0)
            bad.append(f"(assert (= h{i}_{component} {scalar(value)}))")
    bad.append(f"(assert {different(old, current)})")
    checks["missing_horner_link_counterexample"] = solve(solver, bad, "sat")
    assert (0 + 1) % P != 0
    gamma_checks, gamma_exponents = gamma_controls(solver)
    checks.update(gamma_checks)

    result = {
        "field_modulus": P,
        "extension": "F[u]/(u^2 - 7)",
        "degree": 9,
        "checks": checks,
        "gamma_exponents": gamma_exponents,
        "counterexample": "all inputs zero; unlinked final product (1,0); true output (0,0)",
        "scope": "Local algebra and a replayed omission attack, not production integration.",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
